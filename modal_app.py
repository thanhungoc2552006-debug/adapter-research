"""Modal entrypoint. See README_VI.md for credentials and commands."""
import csv
import io
import json
import re
from pathlib import Path
import modal
from polar_runner.experiment import REPO_COMMIT,VOLUME_RESULTS,VOLUME_CACHE,make_plan

PACKAGE=Path(__file__).parent/'polar_runner'
image=(modal.Image.debian_slim(python_version='3.10')
    .apt_install('git','libgomp1')
    .run_commands(
        'git clone https://github.com/kcc-lion/polar.git /opt/polar',
        f'git -C /opt/polar checkout --detach {REPO_COMMIT}',
        "python -c \"from pathlib import Path; p=Path('/opt/polar/requirements.txt'); s=p.read_text(); assert 'wandb==0.19.9' in s; p.write_text(s.replace('wandb==0.19.9','wandb==0.22.3'))\"",
        'cd /opt/polar && python -m pip install -r requirements.txt',
        'python -m pip check')
    .env({'PYTHONPATH':'/opt/polar:/opt/runner','HF_HOME':'/hf_cache','MPLBACKEND':'Agg',
          'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
          'WANDB_LOG_MODEL':'false','WANDB_WATCH':'false'})
    .add_local_dir(PACKAGE,'/opt/runner/polar_runner',copy=True)
    .run_commands('python -m polar_runner.prepare'))

app=modal.App('polar-table2-arc-modal')
results=modal.Volume.from_name(VOLUME_RESULTS,create_if_missing=True)
cache=modal.Volume.from_name(VOLUME_CACHE,create_if_missing=True)
credentials=modal.Secret.from_name('polar-credentials',required_keys=['HF_TOKEN'])
mounts={'/results':results,'/hf_cache':cache}


@app.function(image=image,volumes=mounts,secrets=[credentials],cpu=4,memory=16384,
              timeout=3*3600,max_containers=1,scaledown_window=2)
def setup(experiment: str,use_wandb: bool=True):
    import os
    from polar_runner.worker import bootstrap,runtime_env
    results.reload();cache.reload()
    os.environ.update(runtime_env(use_wandb=use_wandb))
    try:
        return bootstrap(experiment,use_wandb=use_wandb,prefetch=True)
    finally:
        results.commit();cache.commit()


@app.function(image=image,volumes=mounts,secrets=[credentials],gpu='A100-80GB',
              cpu=8,memory=65536,timeout=23*3600,max_containers=1,
              scaledown_window=2,retries=0)
def train_job(cfg: dict,experiment: str,smoke: bool=True,use_wandb: bool=True,
              diagnostics: bool=True,diagnostic_every: int=20,eval_batch: int=4):
    import os
    from polar_runner.worker import run_one
    results.reload();cache.reload()
    try:
        return run_one(cfg,experiment,smoke,use_wandb,diagnostics,diagnostic_every,
                       eval_batch,commit=results.commit)
    finally:
        results.commit();cache.commit()


@app.function(image=image,volumes={'/results':results},cpu=1,memory=2048,
              timeout=600,scaledown_window=2)
def report(experiment: str):
    from polar_runner.experiment import collect_rows
    results.reload()
    rows=collect_rows(Path('/results')/experiment)
    if not rows:
        return 'No full runs found (smoke results are excluded).\n'
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=out.getvalue()
    root=Path('/results')/experiment
    (root/'table2_arc_comparison.csv').write_text(text)
    results.commit()
    return text


@app.local_entrypoint()
def main(stage: str='smoke',ranks: str='4',adapters: str='polar',tasks: str='arc_easy',
         experiment: str='table2-arc-v1',gpu: str='A100-80GB',micro_batch: int=1,
         effective_batch: int=128,dtype: str='fp16',save_steps: int=10,
         use_wandb: bool=True,diagnostics: bool=True,diagnostic_every: int=20,eval_batch: int=4):
    """Stages: plan (local only), smoke, train, report. GPU jobs are queued, one at a time."""
    if stage not in {'plan','smoke','train','report'}:
        raise ValueError('stage must be plan,smoke,train,report')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',experiment):
        raise ValueError('Use a simple experiment name: letters, digits, underscores, hyphens')
    if diagnostic_every<1 or eval_batch<1:
        raise ValueError('diagnostic_every and eval_batch must be positive')
    selected_ranks=[int(s.strip()) for s in ranks.split(',')]
    selected_adapters=[s.strip() for s in adapters.split(',')]
    selected_tasks=[s.strip() for s in tasks.split(',')]
    # Validate before network/GPU work.
    preview=make_plan({},selected_ranks,selected_adapters,selected_tasks,micro_batch,effective_batch,dtype,save_steps)
    if stage=='report':
        text=report.remote(experiment)
        Path(f'{experiment}_comparison.csv').write_text(text)
        print(text)
        return
    print('Configurations:',len(preview),'| GPU:',gpu,'| max concurrent GPUs: 1')
    for cfg in preview:
        print(cfg['rank'],cfg['adapter'],cfg['task'],'lr=',cfg['learning_rate'])
    if stage=='plan':
        return
    lock=setup.remote(experiment,use_wandb)
    plan=make_plan(lock,selected_ranks,selected_adapters,selected_tasks,micro_batch,effective_batch,dtype,save_steps)
    worker=train_job.with_options(gpu=gpu)
    submissions=[]
    # Spawn every durable input before waiting. max_containers=1 enforces sequential GPU use.
    for cfg in plan:
        call=worker.spawn(cfg,experiment,stage=='smoke',use_wandb,diagnostics,diagnostic_every,eval_batch)
        submissions.append((call,cfg['run_id']))
    receipt={'experiment':experiment,'stage':stage,
             'calls':[{'call_id':c.object_id,'run_id':r} for c,r in submissions]}
    Path(f'{experiment}_{stage}_calls.json').write_text(json.dumps(receipt,indent=2))
    print('Submitted all jobs. Keep call IDs; --detach lets submitted work survive terminal disconnect.')
    errors=[]
    for call,run_id in submissions:
        try:print(call.get())
        except Exception as error:
            errors.append(run_id);print('FAILED',run_id,type(error).__name__,str(error))
    text=report.remote(experiment)
    Path(f'{experiment}_comparison.csv').write_text(text)
    print(text)
    if errors:
        raise RuntimeError(f'{len(errors)} jobs failed; inspect Modal logs, fix the cause, then rerun the same command to resume.')
