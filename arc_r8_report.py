"""CPU-only final-adapter measurements for completed ARC runs. Never trains a model."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re

TARGETS = {'q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'}
BASE_EXPERIMENTS = 'table2-arc-resume,arc-lora-isr-v1,arc-isr-arcc-lambda-0p1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def names(value):
    result=[s.strip() for s in value.split(',') if s.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError('Provide unique comma-separated experiment names')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',s) for s in result):
        raise ValueError('Experiment names may contain letters, digits, underscores and hyphens')
    return result


def factor_metrics(A, B, S=None, gamma=1.):
    """Exact nonzero spectrum through thin QR; never allocate the dense update."""
    import numpy as np
    A=np.asarray(A,dtype=np.float64);B=np.asarray(B,dtype=np.float64)
    if A.ndim!=2 or B.ndim!=2 or A.shape[0]!=B.shape[1]:
        raise ValueError('Invalid A/B shapes')
    if not math.isfinite(gamma) or gamma<=0 or not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError('Nonfinite factors or invalid scaling')
    r=A.shape[0]
    if S is not None:
        S=np.asarray(S,dtype=np.float64)
        if S.shape!=(r,r) or not np.isfinite(S).all():raise ValueError('Invalid S')
    rb=np.linalg.qr(B,mode='reduced')[1]
    ra=np.linalg.qr(A.T,mode='reduced')[1]
    core=rb@ra.T if S is None else rb@S@ra.T
    singular=np.linalg.svd(core,compute_uv=False)*gamma
    squares=singular**2;top=float(squares.max());energy=float(squares.sum())
    sr=energy/top if top>0 else None
    if sr is not None and not 1-1e-8<=sr<=r+1e-8:raise ValueError('Invalid stable rank')
    return {'stable_rank':sr,'frobenius':math.sqrt(energy),'spectral':math.sqrt(top),
            'orthogonality_A':float(np.square(A@A.T-np.eye(r)).sum()),
            'orthogonality_B':float(np.square(B.T@B-np.eye(r)).sum())}


def parse_factors(tensors):
    grouped={}
    for key,value in tensors.items():
        match=re.fullmatch(r'(.+)\.lora_([ABS])(?:\.default)?\.(weight|parametrizations\.weight\.original)',key)
        if not match:continue
        module,kind,storage=match.groups()
        if module.rsplit('.',1)[-1] not in TARGETS:continue
        if kind!='S' and storage!='weight':raise ValueError('Unexpected factor parametrization: '+key)
        item=grouped.setdefault(module,{})
        if kind in item:raise ValueError('Duplicate adapter factor: '+key)
        item[kind]=value
    if not grouped:raise ValueError('No selected LoRA A/B tensors found')
    for module,item in grouped.items():
        if not {'A','B'}<=set(item):raise ValueError('Missing A/B: '+module)
    return grouped


def load_tensors(folder):
    path=folder/'adapter_model.safetensors'
    if path.is_file():
        from safetensors.numpy import load_file
        return load_file(str(path)),path
    path=folder/'adapter_model.bin'
    if path.is_file():
        try:import torch
        except ImportError as error:
            raise ValueError('Legacy adapter .bin needs PyTorch. Use the offline command in an environment with torch; this CPU Modal image reads .safetensors.') from error
        return {k:v.float().numpy() for k,v in torch.load(path,map_location='cpu',weights_only=True).items()},path
    raise FileNotFoundError('Final adapter file missing; checkpoints are not substituted for final weights')


def mean(values):
    return math.fsum(values)/len(values) if values else None


def measure_adapter(folder,cfg,expected=160):
    import numpy as np
    adapter=read_json(folder/'adapter_config.json')
    if adapter.get('rank_pattern') or adapter.get('alpha_pattern') or adapter.get('use_rslora'):
        raise ValueError('Reporter expects fixed-rank standard alpha/r scaling')
    tensors,path=load_tensors(folder)
    grouped=parse_factors(tensors)
    if len(grouped)!=expected:raise ValueError(f'Expected {expected} target modules, found {len(grouped)}')
    if expected==160:
        actual=set()
        for module in grouped:
            match=re.search(r'\.layers\.(\d+)\.',module)
            if match:actual.add((int(match.group(1)),module.rsplit('.',1)[-1]))
        wanted={(layer,target) for layer in range(32) for target in TARGETS}
        if actual!=wanted:raise ValueError('Module set must be all five targets in each of 32 Llama blocks')
    count=sum(int(np.asarray(v).size) for k,v in tensors.items() if 'lora_' in k)
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata={'module_count':len(grouped),'adapter_parameter_count':count,
              'adapter_sha256':checksum,'adapter_file':str(path),'sr_status':'measured',
              'sr_mean':None,'sr_std_modules':None,'sr_over_rank':None,'sr_valid_modules':0,
              'sr_zero_modules':0,'sr_mean_15':None,'frobenius_mean':None,'spectral_mean':None,
              'orthogonality_A_mean':None,'orthogonality_B_mean':None}
    if cfg['adapter']=='dora' or adapter.get('use_dora'):
        metadata['sr_status']='not_applicable_dora_full_update_requires_base_weights'
        return metadata,[]
    r=int(adapter['r'])
    if r!=int(cfg['rank']):raise ValueError('Adapter/config rank mismatch')
    gamma=float(adapter['lora_alpha'])/r
    detail=[]
    for module,item in sorted(grouped.items()):
        if np.asarray(item['A']).shape[0]!=r:raise ValueError('Factor rank mismatch: '+module)
        S=item.get('S')
        if cfg['adapter']=='polar':
            if S is None:raise ValueError('PoLAR needs S; refusing to report BA as the full update')
            mode=cfg.get('parameterize_S',adapter.get('parameterize_S','identity'))
            if mode=='diagonal':S=np.diag(np.diag(S))
            elif mode=='removal':S=np.eye(r)
            elif mode!='identity':raise ValueError('Unknown PoLAR S parametrization: '+str(mode))
        elif S is not None:raise ValueError('Unexpected S in standard LoRA')
        match=re.search(r'\.layers\.(\d+)\.',module)
        layer=int(match.group(1)) if match else None
        detail.append({'module':module,'layer':layer,'target':module.rsplit('.',1)[-1],
                       **factor_metrics(item['A'],item['B'],S,gamma)})
    stable=[x['stable_rank'] for x in detail if x['stable_rank'] is not None]
    metadata.update(sr_mean=mean(stable),sr_std_modules=float(np.std(stable)) if stable else None,
        sr_over_rank=mean(stable)/r if stable else None,sr_valid_modules=len(stable),
        sr_zero_modules=len(detail)-len(stable),
        sr_mean_15=mean([x['stable_rank'] for x in detail if x['layer'] in {5,15,25} and x['stable_rank'] is not None]),
        frobenius_mean=mean([x['frobenius'] for x in detail]),spectral_mean=mean([x['spectral'] for x in detail]),
        orthogonality_A_mean=mean([x['orthogonality_A'] for x in detail]),
        orthogonality_B_mean=mean([x['orthogonality_B'] for x in detail]))
    return metadata,detail


def last_ce(folder,step):
    candidates=[]
    for state in folder.glob('checkpoint-*/trainer_state.json'):
        try:
            data=read_json(state)
            checkpoint_step=int(data['global_step']) if 'global_step' in data else int(state.parent.name.split('-')[-1])
            if checkpoint_step>step:continue
            for item in reversed(data.get('log_history',[])):
                if 'loss' in item and int(item.get('step',-1))<=step:
                    candidates.append((checkpoint_step,int(item['step']),float(item['loss'])))
                    break
        except (OSError,ValueError,KeyError):continue
    return max(candidates,key=lambda x:x[:2])[2] if candidates else None


def collect(root,experiments,rank=8,task='arc_challenge',expected=160):
    root=Path(root);rows=[];details=[];warnings=[];signatures={};seen={};configs={}
    common=['model_id','model_revision','dataset_id','dataset_revision','repo_commit','seed','epochs',
            'cutoff_len','dtype','effective_batch','micro_batch','target_modules','lora_alpha','lora_dropout',
            'scheduler','warmup_ratio','base_weights','normalization']
    for experiment in experiments:
        exp=root/experiment
        if not exp.is_dir():warnings.append('Experiment not found: '+experiment);continue
        found=0
        for manifest in sorted((exp/'runs').glob('*/run_config.json')):
            cfg=read_json(manifest)
            if int(cfg['rank'])!=rank or cfg['task']!=task or cfg.get('eval_split','test')!='test':continue
            if cfg['adapter'] not in {'lora','polar','dora','lora_isr'}:continue
            found+=1;folder=manifest.parent
            lam=cfg.get('sr_config',{}).get('lambda_max') if cfg['adapter']=='lora_isr' else None
            label={'lora':'LoRA','dora':'DoRA','polar':'PoLAR','lora_isr':'LoRA-ISR'}[cfg['adapter']]
            if lam is not None:label+=f' (lambda={lam:g})'
            row={'experiment':experiment,'run_id':cfg['run_id'],'method':label,'adapter':cfg['adapter'],
                 'task':task,'rank':rank,'seed':cfg.get('seed',0),'isr_lambda':lam,
                 'learning_rate':cfg.get('learning_rate'),'epochs_config':cfg.get('epochs'),
                 'global_step':None,'epoch_actual':None,'last_logged_ce':None,
                 'acc_norm_pct':None,'acc_norm_stderr_pct':None,'acc_pct':None,
                 'delta_vs_lora_pp':None,'sr_mean':None,'sr_std_modules':None,'sr_over_rank':None,
                 'sr_valid_modules':None,'sr_zero_modules':None,'sr_mean_15':None,
                 'frobenius_mean':None,'spectral_mean':None,'orthogonality_A_mean':None,
                 'orthogonality_B_mean':None,'module_count':None,'adapter_parameter_count':None,
                 'status':'pending_train','sr_status':'not_measured','comparability':'not_checked',
                 'wandb_url':None,'source_path':str(folder),'notes':''}
            # Preserve every run and its configuration. Do not choose a best seed/lambda.
            key=(cfg['run_id'],json.dumps(cfg,sort_keys=True))
            if key in seen:
                warnings.append('Same run ID/config appears in multiple experiments; all records retained: '+str(folder))
            seen[key]=str(folder)
            configs[f'{experiment}/{cfg["run_id"]}']=cfg
            signatures[(experiment,cfg['run_id'])]={k:cfg.get(k) for k in common}
            complete=folder/'train_complete.json';result=folder/'eval_results.json'
            if complete.is_file():
                tc=read_json(complete);row.update(global_step=tc.get('global_step'),epoch_actual=tc.get('epoch'),status='pending_eval')
                row['last_logged_ce']=last_ce(folder,int(tc.get('global_step',0)))
            identities=list((folder/'wandb_targets').glob('*/identity.json'))
            if identities:row['wandb_url']=read_json(max(identities,key=lambda p:p.stat().st_mtime)).get('url')
            if complete.is_file() and result.is_file():
                data=read_json(result)
                if data.get('config',{}).get('limit') is not None or data.get('isr_eval_split','test')!='test':
                    row.update(status='invalid_eval',notes='Full test evaluation required')
                else:
                    scores=data['results'][task]
                    for column,metric in [('acc_norm_pct','acc_norm,none'),('acc_pct','acc,none'),('acc_norm_stderr_pct','acc_norm_stderr,none')]:
                        value=scores.get(metric)
                        if isinstance(value,(int,float)) and math.isfinite(value):row[column]=100*value
                    if row['acc_norm_pct'] is None or row['acc_pct'] is None:raise ValueError('Missing accuracy: '+str(folder))
                    row['status']='complete'
                    try:
                        print('Measuring final adapter:',experiment,cfg['run_id'],flush=True)
                        measurements,per_module=measure_adapter(folder,cfg,expected)
                        row.update(measurements)
                        for item in per_module:details.append({'experiment':experiment,'run_id':cfg['run_id'],**item})
                    except (ValueError,OSError,ImportError) as error:
                        row.update(sr_status='measurement_error',notes=str(error))
            rows.append(row)
        if not found:warnings.append('No matching full-run configs: '+experiment)
    baselines=[r for r in rows if r['adapter']=='lora' and r['status']=='complete']
    for row in rows:
        candidates=[r for r in baselines if r['seed']==row['seed']]
        if len(candidates)!=1:
            row['comparability']='missing_or_ambiguous_lora_baseline';continue
        base=candidates[0]
        a=signatures[(row['experiment'],row['run_id'])];b=signatures[(base['experiment'],base['run_id'])]
        missing=[k for k in common if a[k] is None or b[k] is None]
        different=[k for k in common if a[k]!=b[k]]
        if missing or different:
            row['comparability']='check_config: '+','.join(sorted(set(missing+different)))
        elif row['status']=='complete':
            row['comparability']='matched_recorded_settings'
            row['delta_vs_lora_pp']=row['acc_norm_pct']-base['acc_norm_pct']
        if row['global_step'] is not None and base['global_step'] is not None and row['global_step']!=base['global_step']:
            row['notes']+=' Optimizer update counts differ.'
    rows.sort(key=lambda r:({'lora':0,'dora':1,'polar':2,'lora_isr':3}[r['adapter']],r['isr_lambda'] or 0,r['seed'],r['experiment']))
    return {'generated_utc':datetime.now(timezone.utc).isoformat(),'rank':rank,'task':task,
            'expected_modules':expected,'rows':rows,'modules':details,'warnings':warnings,
            'run_configs':configs,
            'measurement':'Final saved adapters, float64 thin QR plus SVD of rank-by-rank core; stable rank mean over nonzero updates.'}


def fmt(value,digits=3):
    if value is None:return '—'
    return f'{value:.{digits}f}' if isinstance(value,(int,float)) else str(value).replace('|','/')


def markdown(payload):
    lines=[f"# {payload['task']} · rank {payload['rank']}",'',
        '| Method | Seed | Acc norm (%) | Delta vs LoRA (pp) | SR mean | SR / r | Params (M) | Updates | Status |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in payload['rows']:
        acc=fmt(row['acc_norm_pct'])
        if row['acc_norm_stderr_pct'] is not None:acc+=' ± '+fmt(row['acc_norm_stderr_pct'])+' SE'
        params=row.get('adapter_parameter_count')
        status=row['status']
        if row['sr_status']=='measurement_error':status+='; SR unavailable'
        if row['comparability']!='matched_recorded_settings':status+='; config check'
        values=[row['method'],str(row['seed']),acc,fmt(row['delta_vs_lora_pp']),fmt(row['sr_mean']),
                fmt(row['sr_over_rank']),fmt(params/1e6 if params is not None else None),fmt(row['global_step'],0),status]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['','Accuracy ± SE is uncertainty over evaluation samples, not variability across training seeds.',
        f"Stable rank is measured on the same {payload['expected_modules']} target modules from final adapters; zero updates are excluded and counted separately.",
        'SR std in the CSV describes variation across modules, not across seeds.',
        'LoRA/ISR update = gamma BA; PoLAR update = gamma BSA. DoRA SR is left unavailable because BA is not the complete DoRA weight change.',
        'Frobenius/spectral norms are means of per-module norms. Orthogonality defects depend on factor scale and do not replace stable rank.',
        'Recorded-config matching does not verify unrecorded code or environment changes. A single seed is an exploratory result.']
    for warning in payload['warnings']:lines.append('- '+warning)
    for row in payload['rows']:
        if row['notes']:lines.append('- '+row['method']+': '+row['notes'])
    return '\n'.join(lines)+'\n'


def write_outputs(payload,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('summary.csv',payload['rows']),('per_module.csv',payload['modules'])]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            if rows:
                fields=list(dict.fromkeys(k for row in rows for k in row))
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (out/'report.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (out/'report.md').write_text(markdown(payload),encoding='utf-8')
    return str(out.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Local backup parent containing experiment folders')
    p.add_argument('--experiments',default=BASE_EXPERIMENTS)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--task',choices=['arc_easy','arc_challenge'],default='arc_challenge')
    p.add_argument('--output',default='reports/arc_challenge_r8')
    args=p.parse_args()
    result=collect(args.root,names(args.experiments),args.rank,args.task)
    if not result['rows']:raise SystemExit('No matching runs. Check --root and --experiments.')
    print('Saved:',write_outputs(result,args.output));print(markdown(result))


if __name__=='__main__':main()
