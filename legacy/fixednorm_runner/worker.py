"""Remote execution: original training subprocess, then separate evaluation."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from fixednorm_runner.experiment import MODEL_ID, DATASET_ID, REPO_COMMIT, atomic_json

REPO=Path('/opt/polar')
RESULTS=Path('/results')


def runtime_env(lock=None,use_wandb=True,diagnostics=True,diagnostic_every=20):
    env=os.environ.copy()
    env.update(PYTHONPATH='/opt/polar:/opt/runner',HF_HOME='/hf_cache',MPLBACKEND='Agg',
               TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               WANDB_MODE='online' if use_wandb else 'disabled',
               WANDB_PROJECT=env.get('WANDB_PROJECT','polar-table2-arc'),
               WANDB_LOG_MODEL='false',WANDB_WATCH='false',
               POLAR_USE_WANDB=str(int(use_wandb)),POLAR_LOG_DIAGNOSTICS=str(int(diagnostics)),
               POLAR_DIAGNOSTIC_EVERY=str(diagnostic_every),POLAR_DIAGNOSTIC_LAYERS='[5,15,25]')
    for key in ['WANDB_RUN_ID','WANDB_RESUME','WANDB_DISABLED']:
        env.pop(key,None)
    if lock:
        env['POLAR_MODEL_REVISION']=lock['model_revision']
    return env


def pin_dataset(lock):
    path=REPO/'lm-evaluation-harness/lm_eval/tasks/arc/arc_easy.yaml'
    original=subprocess.check_output(['git','show',f'{REPO_COMMIT}:lm-evaluation-harness/lm_eval/tasks/arc/arc_easy.yaml'],cwd=REPO,text=True)
    path.write_text(original+'\ndataset_kwargs:\n  revision: '+lock['dataset_revision']+'\n')


def bootstrap(experiment,use_wandb=True,prefetch=True,baseline_experiment=''):
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    root=RESULTS/experiment; root.mkdir(parents=True,exist_ok=True)
    if not os.environ.get('HF_TOKEN'):
        raise ValueError('Secret polar-credentials must contain HF_TOKEN')
    if use_wandb:
        if not all(os.environ.get(k) for k in ['WANDB_API_KEY','WANDB_ENTITY']):
            raise ValueError('Set WANDB_API_KEY and WANDB_ENTITY in polar-credentials')
        import wandb
        assert wandb.login(key=os.environ['WANDB_API_KEY'],verify=True)
        # Validate workspace write permission before allocating any GPU.
        check=wandb.init(project=os.environ.get('WANDB_PROJECT','polar-table2-arc'),
            entity=os.environ['WANDB_ENTITY'],name='modal_connection_check',tags=['connection-check'],
            save_code=False,settings=wandb.Settings(disable_git=True))
        print('W&B workspace check:',check.url,flush=True)
        check.finish()
    api=HfApi()
    path=root/'source_lock.json'
    code_hash=(REPO/'modal_source_sha256.txt').read_text().strip()
    if path.exists():
        lock=json.loads(path.read_text())
        assert lock['repo_commit']==REPO_COMMIT and lock['source_sha256']==code_hash, 'Source changed: use a new --experiment name.'
        if baseline_experiment:
            baseline=json.loads((RESULTS/baseline_experiment/'source_lock.json').read_text())
            for key in ['repo_commit','model_id','model_revision','dataset_id','dataset_revision']:
                assert lock[key]==baseline[key], f'Baseline provenance changed: {key}; use a new experiment.'
    else:
        inherited=None
        if baseline_experiment:
            inherited=json.loads((RESULTS/baseline_experiment/'source_lock.json').read_text())
            assert inherited['repo_commit']==REPO_COMMIT and inherited['model_id']==MODEL_ID and inherited['dataset_id']==DATASET_ID, 'Baseline provenance does not match ARC/Llama-2-7B'
        lock={'repo_commit':REPO_COMMIT,'source_sha256':code_hash,'model_id':MODEL_ID,
              'model_revision':inherited['model_revision'] if inherited else api.model_info(MODEL_ID).sha,
              'dataset_id':DATASET_ID,'dataset_revision':inherited['dataset_revision'] if inherited else api.dataset_info(DATASET_ID).sha}
        if baseline_experiment:
            atomic_json(root/'baseline_provenance.json',{'baseline_experiment':baseline_experiment,'source_lock':inherited})
        atomic_json(path,lock)
    hf_hub_download(MODEL_ID,'config.json',revision=lock['model_revision'])
    pin_dataset(lock)
    from lm_eval.tasks import TaskManager,get_task_dict
    tasks=get_task_dict(['arc_easy','arc_challenge'],TaskManager())
    data_manifest={}
    for name,task in tasks.items():
        train=task.training_docs();test=task.test_docs();doc=train[0]
        choices=task.doc_to_choice(doc);gold=task.doc_to_target(doc)
        assert isinstance(gold,int) and 0<=gold<len(choices)
        data_manifest[name]={'train_n':len(train),'test_n':len(test),
            'validation_n':len(task.validation_docs()),
            'train_fingerprint':train._fingerprint,'test_fingerprint':test._fingerprint,
            'num_fewshot':task.config.num_fewshot,'target_delimiter':task.config.target_delimiter}
        print(name,'train:',len(train),'test:',len(test),flush=True)
    atomic_json(root/'dataset_manifest.json',data_manifest)
    if prefetch:
        names=[item.rfilename for item in api.model_info(MODEL_ID,revision=lock['model_revision']).siblings]
        weight='*.safetensors' if any(s.endswith('.safetensors') for s in names) else '*.bin'
        snapshot_download(MODEL_ID,revision=lock['model_revision'],
            allow_patterns=[weight,'*.json','*.model','tokenizer*','special_tokens_map.json'])
    (root/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for name in ['modal_changes.diff','modal_source_sha256.txt']:
        (root/name).write_text((REPO/name).read_text())
    print('Sources and cache ready:',root,flush=True)
    return lock


def logged_process(command,log_path,env,commit):
    """Commit checkpoints after on_save; never require Modal SDK in child process."""
    stop=threading.Event()
    marker=Path(log_path).parent/'commit_request.json'
    last=None
    def watcher():
        nonlocal last
        while not stop.wait(5):
            try:
                current=marker.read_text() if marker.exists() else None
                if current and current!=last:
                    commit();last=current
            except Exception as error:
                # A final commit is required too; transient commit failures are visible.
                print('Volume checkpoint commit warning:',type(error).__name__,flush=True)
    thread=threading.Thread(target=watcher,daemon=True);thread.start()
    secrets=[env.get(k) for k in ['HF_TOKEN','WANDB_API_KEY'] if env.get(k)]
    process=None
    try:
        with open(log_path,'a',buffering=1) as log:
            print('Launching:', ' '.join(map(str,command)),flush=True)
            process=subprocess.Popen(list(map(str,command)),cwd=REPO,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                for secret in secrets:
                    line=line.replace(secret,'[TOKEN_HIDDEN]')
                print(line,end='',flush=True);log.write(line)
            code=process.wait()
        if code:
            raise RuntimeError(f'Child process exited {code}; inspect {log_path}')
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    finally:
        stop.set();thread.join(timeout=10)
        commit()


def run_one(cfg,experiment,smoke,use_wandb,diagnostics,diagnostic_every,eval_batch,commit):
    root=RESULTS/experiment
    folder=root/('_smoke' if smoke else 'runs')/cfg['run_id']
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'run_config.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        # Checkpoint frequency may change without invalidating trained parameters.
        assert {k:v for k,v in old.items() if k!='save_steps'}=={k:v for k,v in cfg.items() if k!='save_steps'},'Run config mismatch'
    else:
        atomic_json(manifest,cfg)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    atomic_json(folder/f'attempt_{time.time_ns()}.json',{'gpu':hardware,'smoke':smoke})
    pin_dataset(cfg)
    env=runtime_env(cfg,use_wandb,diagnostics,diagnostic_every)
    env.update(POLAR_SAVE_STEPS=str(cfg['save_steps']),POLAR_MAX_STEPS='2' if smoke else '-1')
    # Explicit fresh per-process settings; no stale Colab MPLBACKEND or credentials.
    result_path=folder/'eval_results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        assert 'acc_norm,none' in result['results'][cfg['task']]
        assert result.get('isr_eval_split','test')==cfg.get('eval_split','test'), 'Evaluation split mismatch'
        if not smoke:
            assert result['config'].get('limit') is None
        if use_wandb:
            sync="import json,sys; from pathlib import Path; from modal_tracking import publish_evaluation,tracking_target; p=Path(sys.argv[1]); target=tracking_target(p); publish_evaluation(p,json.loads((p/'eval_results.json').read_text())) if not (target/'eval_synced.json').exists() else None"
            logged_process([sys.executable,'-c',sync,folder],folder/'wandb_sync.log',env,commit)
        print('SKIP completed:',folder,flush=True)
        return {'run_id':cfg['run_id'],'status':'already_complete','path':str(folder)}
    if not (folder/'train_complete.json').exists():
        params={'base_model':MODEL_ID,'data_path':cfg['task'],'output_dir':str(folder),
            'adapter_name':cfg.get('adapter_backend',cfg['adapter']),'seed':cfg['seed'],'batch_size':cfg['effective_batch'],
            'micro_batch_size':cfg['micro_batch'],'num_epochs':cfg['epochs'],
            'learning_rate':cfg['learning_rate'],'cutoff_len':cfg['cutoff_len'],'val_set_size':0,
            'dtype':cfg['dtype'],'lora_r':cfg['rank'],'lora_alpha':32,'lora_dropout':.05,
            'lora_target_modules':json.dumps(cfg['target_modules']),
            'parameterize_S':'identity','gradient_type':'landing','regularization_lambda':.005,
            'init_lora_weights':'default','save_strategy':'steps','lr_scheduler_type':'cosine',
            'do_eval':False,'add_stable_rank_callback':False,'activate_profiling':False,
            'wandb_project':os.environ.get('WANDB_PROJECT','polar-table2-arc') if use_wandb else '',
            'wandb_run_name':cfg['run_id']}
        command=[sys.executable,REPO/'finetune_modal.py']+[f'--{k}={v}' for k,v in params.items()]
        logged_process(command,folder/'train.log',env,commit)
        assert (folder/'train_complete.json').exists()
    command=[sys.executable,'-m','fixednorm_runner.evaluate','--adapter',folder,'--task',cfg['task'],'--batch',eval_batch]
    command+=['--split',cfg.get('eval_split','test')]
    if smoke:command+=['--limit','16']
    logged_process(command,folder/'eval.log',env,commit)
    return {'run_id':cfg['run_id'],'status':'complete','path':str(folder)}
