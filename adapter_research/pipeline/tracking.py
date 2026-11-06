"""Read-only diagnostics and W&B lifecycle; never changes gradients or RNG."""
import hashlib, json, os, re, time
from pathlib import Path
import torch
from transformers import TrainerCallback
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.polar.layer import PoLARLinear


def tracking_target(folder):
    if not enabled():
        return None
    entity=os.environ.get('WANDB_ENTITY','').strip()
    project=os.environ.get('WANDB_PROJECT','polar-table2-arc').strip()
    if not entity:
        raise ValueError('Set WANDB_ENTITY in the polar-credentials Modal Secret')
    namespace=hashlib.sha256((entity+'|'+project).encode()).hexdigest()[:16]
    target=Path(folder)/'wandb_targets'/namespace
    target.mkdir(parents=True,exist_ok=True)
    return target


def enabled():
    return os.environ.get('POLAR_USE_WANDB','0')=='1'


def start_tracking(output_dir, phase='train'):
    if not enabled():
        return None
    import wandb
    folder=Path(output_dir)
    cfg=json.loads((folder/'run_config.json').read_text())
    identity=tracking_target(folder)/'identity.json'
    project=os.environ['WANDB_PROJECT']
    entity=os.environ.get('WANDB_ENTITY') or None
    if identity.exists():
        previous=json.loads(identity.read_text())
        assert previous['project']==project, 'W&B project changed; restore the previous project.'
        if entity:
            assert previous['entity']==entity, 'W&B entity changed; use the original workspace.'
        run_id=previous['id']; entity=previous['entity']
    else:
        run_id=wandb.util.generate_id()
        # Persist before contacting server; reconnect even if init is interrupted.
        identity.write_text(json.dumps({'id':run_id,'project':project,'entity':entity},indent=2))
    smoke='_smoke' in folder.parts
    tags=[cfg['adapter'],f"rank-{cfg['rank']}",cfg['task'],'smoke' if smoke else 'reproduction']
    name=('smoke_' if smoke else '')+f"{cfg['adapter']}_r{cfg['rank']}_{cfg['task']}"
    if cfg.get('method_label'):
        name += ' | '+cfg['method_label']
    tags.append(cfg.get('pipeline_version','legacy'))
    if cfg.get('experiment'):tags.append(cfg['experiment'])
    run=wandb.init(project=project,entity=entity,id=run_id,resume='allow',
        name=name,group=cfg['task'],tags=tags,job_type='reproduction',
        config=cfg,dir=str(tracking_target(folder)),save_code=False,settings=wandb.Settings(disable_git=True))
    resolved={'id':run.id,'entity':run.entity,'project':run.project,'url':run.url}
    identity.write_text(json.dumps(resolved,indent=2))
    run.define_metric('train/global_step')
    for pattern in ['diagnostics/*','stable_rank/*','orthogonality/*','gpu/*','train/sr/*','train/projection/*']:
        run.define_metric(pattern,step_metric='train/global_step')
    run.summary['phase']=phase
    run.summary['is_smoke']=smoke
    run.config.update({'monitoring':{
        'diagnostics':os.environ.get('POLAR_LOG_DIAGNOSTICS')=='1',
        'every_steps':int(os.environ.get('POLAR_DIAGNOSTIC_EVERY','20')),
        'layer_indices':json.loads(os.environ.get('POLAR_DIAGNOSTIC_LAYERS','[5,15,25]')),
        'stable_rank_algorithm':'thin_QR_then_SVD_of_r_by_r_core',
        'dora_stable_rank':'not_logged: BA alone is not the complete DoRA weight update'}},allow_val_change=True)
    print('W&B run:',run.url,flush=True)
    return run


def finish_tracking(phase, **summary):
    if not enabled():
        return
    import wandb
    if wandb.run is not None:
        wandb.run.summary.update({'phase':phase,**summary})
        wandb.finish()


@torch.no_grad()
def stable_rank_factors(A,B,S=None):
    # Delta W = B S A, scale alpha/r cancels in stable rank.
    # QR(B)=QB RB and QR(A.T)=QA RA. Nonzero singular values are
    # exactly those of RB S RA.T; never construct d_out x d_in delta W.
    _, rb=torch.linalg.qr(B.float(),mode='reduced')
    _, ra=torch.linalg.qr(A.float().t(),mode='reduced')
    core=rb @ ra.t() if S is None else rb @ S.float() @ ra.t()
    squares=torch.linalg.svdvals(core).square()
    top=squares.max()
    if top.item()==0:
        return None  # stable rank of zero update is undefined, not 0 or 1
    return (squares.sum()/top).item()


class FactorDiagnostics(TrainerCallback):
    def __init__(self):
        self.every=int(os.environ.get('POLAR_DIAGNOSTIC_EVERY','20'))
        self.layers=set(json.loads(os.environ.get('POLAR_DIAGNOSTIC_LAYERS','[5,15,25]')))
        self.last_step=None

    @torch.no_grad()
    def measure(self,state,model):
        if not state.is_world_process_zero or self.last_step==state.global_step:
            return
        import wandb
        if wandb.run is None:
            return
        started=time.perf_counter()
        metrics={'train/global_step':state.global_step}
        stable=[]; orth_a=[]; orth_b=[]; zeros=0; count=0
        for name,module in model.named_modules():
            if not isinstance(module,(LoraLinear,PoLARLinear)):
                continue
            match=re.search(r'layers\.(\d+)\.(.+)',name)
            if not match or int(match.group(1)) not in self.layers:
                continue
            key=f'layer{match.group(1)}.{match.group(2)}'
            adapter=module.active_adapters[0]
            A=module.lora_A[adapter].weight.float()
            B=module.lora_B[adapter].weight.float()
            if not torch.isfinite(A).all() or not torch.isfinite(B).all():
                raise FloatingPointError(f'Nonfinite adapter weights in {name}')
            ea=(A@A.t()-torch.eye(A.shape[0],device=A.device)).square().sum().item()
            eb=(B.t()@B-torch.eye(B.shape[1],device=B.device)).square().sum().item()
            orth_a.append(ea);orth_b.append(eb);count+=1
            metrics[f'orthogonality/{key}/A']=ea
            metrics[f'orthogonality/{key}/B']=eb
            # DoRA includes magnitude and normalization; BA is not its full update.
            if isinstance(module,PoLARLinear):
                S=module.lora_S[adapter].weight
                sr=stable_rank_factors(A,B,S)
            elif not module.use_dora[adapter]:
                sr=stable_rank_factors(A,B)
            else:
                continue
            if sr is None:
                zeros+=1
            else:
                metrics[f'stable_rank/{key}']=sr;stable.append(sr)
        if stable:
            metrics['diagnostics/stable_rank_mean']=sum(stable)/len(stable)
        if orth_a:
            metrics['diagnostics/orthogonality_A_mean']=sum(orth_a)/len(orth_a)
            metrics['diagnostics/orthogonality_B_mean']=sum(orth_b)/len(orth_b)
        metrics['diagnostics/zero_updates_skipped']=zeros
        metrics['diagnostics/modules_monitored']=count
        if torch.cuda.is_available():
            metrics['gpu/allocated_GB']=torch.cuda.memory_allocated()/1024**3
            metrics['gpu/reserved_GB']=torch.cuda.memory_reserved()/1024**3
            metrics['gpu/peak_allocated_GB']=torch.cuda.max_memory_allocated()/1024**3
        metrics['diagnostics/measurement_seconds']=time.perf_counter()-started
        wandb.log(metrics)
        self.last_step=state.global_step

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.measure(state,model)
    def on_step_end(self,args,state,control,model=None,**kwargs):
        if state.global_step%self.every==0:
            self.measure(state,model)
    def on_train_end(self,args,state,control,model=None,**kwargs):
        self.measure(state,model)


def monitor_callbacks():
    return [FactorDiagnostics()] if enabled() and os.environ.get('POLAR_LOG_DIAGNOSTICS')=='1' else []


def publish_evaluation(folder,result):
    if not enabled():
        return
    folder=Path(folder)
    run=start_tracking(folder,'eval')
    cfg=json.loads((folder/'run_config.json').read_text())
    assert result.get('isr_eval_split','test')==cfg.get('eval_split','test'), 'Evaluation split mismatch'
    task=cfg['task'];scores=result['results'][task]
    step=json.loads((folder/'train_complete.json').read_text())['global_step']
    data={'train/global_step':step,'eval/acc_norm_pct':100*scores['acc_norm,none'],
          'eval/acc_pct':100*scores['acc,none'],'eval/is_smoke':'_smoke' in folder.parts,
          'eval/split':cfg.get('eval_split','test')}
    stderr=scores.get('acc_norm_stderr,none')
    if isinstance(stderr,(int,float)):
        data['eval/acc_norm_stderr_pct']=100*stderr
    references={
        (4,'lora'):(81.90,57.68),(4,'dora'):(81.69,57.85),(4,'polar'):(81.61,56.31),
        (8,'lora'):(82.24,58.11),(8,'dora'):(82.24,58.19),(8,'polar'):(81.36,56.91),
        (16,'lora'):(82.58,57.85),(16,'dora'):(82.62,57.68),(16,'polar'):(81.82,57.51),
        (32,'lora'):(82.37,56.83),(32,'dora'):(82.28,57.17),(32,'polar'):(81.99,56.14)}
    if not data['eval/is_smoke']:
        assert result['config'].get('limit') is None, 'Limited eval cannot be published as full test.'
        values=references.get((cfg['rank'],cfg['adapter'])) if data['eval/split']=='test' else None
        if values is not None:
            ref=values[0 if task=='arc_easy' else 1]
            data.update({'eval/paper_acc_norm_pct':ref,'eval/delta_pp':data['eval/acc_norm_pct']-ref})
    run.log(data)
    run.summary.update({k:v for k,v in data.items() if k!='train/global_step'})
    finish_tracking('complete',eval_complete=True)
    # Marker only after successful W&B sync; no secrets in this file.
    (tracking_target(folder)/'eval_synced.json').write_text(json.dumps({'run_id':run.id,'project':run.project,'entity':run.entity}))
