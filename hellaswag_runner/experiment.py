"""Pure configuration and result handling; safe to import without GPU packages."""
import hashlib
import json
from pathlib import Path

REPO_COMMIT = '7c9eeabe54cf94e2051632829019c86c4198b3df'
MODEL_ID = 'meta-llama/Llama-2-7b-hf'
DATASET_ID = 'Rowan/hellaswag'
DATASET_REVISION = '218ec52e09a7e7462a5400043bb9a69a41d06b76'
VOLUME_RESULTS = 'polar-table2-results'
VOLUME_CACHE = 'polar-table2-hf-cache'
REFERENCE = {
    (4,'lora'):82.36,(4,'dora'):82.39,(4,'polar'):81.23,
    (8,'lora'):82.68,(8,'dora'):82.32,(8,'polar'):82.14,
    (16,'lora'):82.57,(16,'dora'):82.39,(16,'polar'):82.68,
    (32,'lora'):82.51,(32,'dora'):82.50,(32,'polar'):83.12,
}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def make_plan(lock, ranks=(8,), adapters=('polar',), tasks=('hellaswag',),
              micro_batch=1, batch=128, dtype='fp16', save_steps=10):
    ranks = list(ranks); adapters = list(adapters); tasks = list(tasks)
    if not ranks or not set(ranks) <= {4,8,16,32} or len(set(ranks)) != len(ranks):
        raise ValueError('Ranks must be a unique subset of 4,8,16,32')
    if not adapters or not set(adapters) <= {'lora','dora','polar'} or len(set(adapters)) != len(adapters):
        raise ValueError('Adapters must be a unique subset of lora,dora,polar')
    if tasks != ['hellaswag']:
        raise ValueError('This entrypoint supports only --tasks hellaswag; use modal_app.py for ARC.')
    if lock.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ValueError('This experiment lock is not for HellaSwag; use a new --experiment name.')
    if micro_batch < 1 or batch % micro_batch or save_steps < 1:
        raise ValueError('Positive micro_batch must divide batch; save_steps must be positive')
    if dtype not in ('fp16','bf16'):
        raise ValueError('dtype must be fp16 or bf16')
    plan=[]
    for rank in ranks:
        for adapter in adapters:
            for task in tasks:
                cfg={**lock,'rank':rank,'adapter':adapter,'task':task,'seed':0,'epochs':5,
                     'learning_rate':.004 if adapter=='polar' else .0004,
                     'regularization_lambda':.005,'lora_alpha':32,'lora_dropout':.05,
                     'target_modules':['q_proj','k_proj','v_proj','up_proj','down_proj'],
                     'parameterize_S':'identity','gradient_type':'landing','cutoff_len':512,
                     'micro_batch':micro_batch,'effective_batch':batch,'dtype':dtype,
                     'gradient_checkpointing':True,'warmup_ratio':.03,'scheduler':'cosine',
                     'base_weights':'fp32_after_prepare','normalization':'repo_character_length_acc_norm',
                     'train_split':'train','eval_split':'validation',
                     'task_data_pipeline':'hellaswag_rowan_v1'}
                digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
                cfg['run_id']=f'{adapter}_r{rank}_{task}_{digest}'
                cfg['save_steps']=save_steps # cadence does not change the numerical run identity
                plan.append(cfg)
    return plan


def ready_checkpoint(folder, require_scaler=True):
    """Never resume a checkpoint that was interrupted before on_save finished."""
    candidates=[]
    for path in Path(folder).glob('checkpoint-*'):
        if not path.is_dir() or not path.name[11:].isdigit():
            continue
        required=['checkpoint_ready.json','trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json']
        if require_scaler:
            required.append('scaler.pt')
        if not all((path/name).is_file() for name in required):
            continue
        if not any((path/name).is_file() for name in ['adapter_model.safetensors','adapter_model.bin']):
            continue
        try:
            state=json.loads((path/'trainer_state.json').read_text())
            marker=json.loads((path/'checkpoint_ready.json').read_text())
            step=int(path.name[11:])
            if state['global_step']==step and marker['global_step']==step:
                candidates.append((step,path))
        except (OSError,ValueError,KeyError):
            continue
    return str(max(candidates)[1]) if candidates else None


def collect_rows(root):
    rows=[]
    for manifest in sorted((Path(root)/'runs').glob('*/run_config.json')):
        cfg=json.loads(manifest.read_text()); folder=manifest.parent
        if cfg['task'] != 'hellaswag':
            raise ValueError(f'Non-HellaSwag run found in this experiment: {manifest}')
        ref=REFERENCE[(cfg['rank'],cfg['adapter'])]
        row={'run_id':cfg['run_id'],'rank':cfg['rank'],'adapter':cfg['adapter'],'task':cfg['task'],
             'paper_pct':ref,'reproduced_pct':None,'delta_pp':None,'acc_pct':None,
             'stderr_pct':None,'status':'pending','wandb_url':None,
             'eval_split':'validation'}
        result=folder/'eval_results.json'
        if result.exists():
            r=json.loads(result.read_text())
            if r['config'].get('limit') is not None:
                raise ValueError(f'Limited evaluation found in full runs: {folder}')
            s=r['results'][cfg['task']]; score=100*s['acc_norm,none']
            se=s.get('acc_norm_stderr,none')
            row.update(reproduced_pct=score,delta_pp=score-ref,acc_pct=100*s['acc,none'],
                       stderr_pct=100*se if isinstance(se,(int,float)) else None,status='done')
        identities=list((folder/'wandb_targets').glob('*/identity.json'))
        if identities:
            row['wandb_url']=json.loads(max(identities,key=lambda p:p.stat().st_mtime).read_text()).get('url')
        rows.append(row)
    return rows
