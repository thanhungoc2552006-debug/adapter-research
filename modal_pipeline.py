"""One Modal entrypoint for ARC training and baseline comparisons."""
import json
from datetime import datetime, timezone
from pathlib import Path
import modal
from adapter_research.config import load
from adapter_research.common import REPO_COMMIT, VOLUME_RESULTS, VOLUME_CACHE, atomic_json
from adapter_research.planning import make_plan

PACKAGE = Path(__file__).parent / 'adapter_research'
image = (modal.Image.debian_slim(python_version='3.10')
    .apt_install('git', 'libgomp1')
    .run_commands(
        'git clone https://github.com/kcc-lion/polar.git /opt/polar',
        f'git -C /opt/polar checkout --detach {REPO_COMMIT}',
        "python -c \"from pathlib import Path; p=Path('/opt/polar/requirements.txt'); s=p.read_text(); assert 'wandb==0.19.9' in s; p.write_text(s.replace('wandb==0.19.9','wandb==0.22.3'))\"",
        'cd /opt/polar && python -m pip install -r requirements.txt',
        'python -m pip check')
    .env({'PYTHONPATH': '/opt/polar:/opt/runner', 'HF_HOME': '/hf_cache', 'MPLBACKEND': 'Agg',
          'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
          'WANDB_LOG_MODEL': 'false', 'WANDB_WATCH': 'false'})
    .add_local_dir(PACKAGE, '/opt/runner/adapter_research', copy=True, ignore=['**/__pycache__/**', '**/*.pyc'])
    .run_commands('python -m adapter_research.pipeline.prepare'))
report_image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.4', 'safetensors==0.5.3', 'wandb==0.22.3')
    .env({'PYTHONPATH': '/opt/runner', 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'})
    .add_local_dir(PACKAGE, '/opt/runner/adapter_research', copy=True, ignore=['**/__pycache__/**', '**/*.pyc']))

app = modal.App('adapter-research-arc')
results = modal.Volume.from_name(VOLUME_RESULTS, create_if_missing=True)
cache = modal.Volume.from_name(VOLUME_CACHE, create_if_missing=True)
credentials = modal.Secret.from_name('polar-credentials')
mounts = {'/results': results, '/hf_cache': cache}


@app.function(image=image, volumes=mounts, secrets=[credentials], cpu=4, memory=16384,
              timeout=3*3600, max_containers=1, scaledown_window=2)
def setup(config: dict, smoke: bool = False):
    import os
    from adapter_research.pipeline.worker import bootstrap, runtime_env, inspect_runs
    results.reload(); cache.reload()
    runtime = config['runtime']
    os.environ.update(runtime_env(use_wandb=runtime['use_wandb']))
    try:
        lock = bootstrap(config['experiment'], use_wandb=runtime['use_wandb'],
                         prefetch=True, baseline_experiment=config['baseline_experiment'])
        plan = make_plan(lock, config)
        return {'plan': plan, 'statuses': inspect_runs(plan, config['experiment'], smoke, runtime['use_wandb'])}
    finally:
        results.commit(); cache.commit()


@app.function(image=image, volumes=mounts, secrets=[credentials], gpu='H100',
              cpu=8, memory=65536, timeout=23*3600, max_containers=1,
              scaledown_window=2, retries=0)
def train_job(cfg: dict, config: dict, smoke: bool = False):
    from adapter_research.pipeline.worker import run_one
    results.reload(); cache.reload()
    runtime = config['runtime']
    try:
        return run_one(cfg, config['experiment'], smoke, runtime['use_wandb'],
                       runtime['diagnostics'], runtime['diagnostic_every'],
                       runtime['eval_batch'], commit=results.commit)
    finally:
        results.commit(); cache.commit()


@app.function(image=report_image, volumes={'/results': results}, secrets=[credentials],
              cpu=4, memory=8192, timeout=3600, scaledown_window=2)
def comparison(config: dict):
    import os
    from adapter_research.reporting.service import build_reports
    from adapter_research.reporting.collect import write_outputs
    from adapter_research.reporting.wandb import publish
    results.reload()
    payloads = build_reports('/results', config)
    if config['runtime']['use_wandb']:
        if not all(os.environ.get(key) for key in ['WANDB_API_KEY', 'WANDB_ENTITY']):
            raise ValueError('polar-credentials needs WANDB_API_KEY and WANDB_ENTITY for publishing')
        import wandb
        wandb.login(key=os.environ['WANDB_API_KEY'], verify=True, relogin=True)
    for payload in payloads:
        if payload['rows'] and config['runtime']['use_wandb']:
            payload['wandb_receipt'] = publish(payload, entity=os.environ['WANDB_ENTITY'],
                project=os.environ.get('WANDB_PROJECT', 'polar-table2-arc'),
                label=config['comparison']['label'])
        path = (Path('/results/_reports') / payload['task'] /
                f"r{payload['rank']}" / config['comparison']['label'])
        write_outputs(payload, path)
    results.commit()
    return payloads


def save_comparison(payloads, config):
    from adapter_research.reporting.service import persist_local
    from adapter_research.reporting.collect import markdown
    paths = persist_local(payloads, config)
    for payload, path in zip(payloads, paths):
        print(markdown(payload))
        print('Report:', path)
        if payload.get('wandb_receipt'):
            print('W&B comparison:', payload['wandb_receipt']['url'])


@app.local_entrypoint()
def main(stage: str = 'plan', config: str = 'configs/arc_fixednorm_r8.json'):
    """plan, setup, smoke, train, report. Configuration is one JSON file."""
    if stage not in {'plan', 'setup', 'smoke', 'train', 'report'}:
        raise ValueError('stage must be plan, setup, smoke, train or report')
    cfg = load(config)
    if stage == 'report':
        print('CPU comparison only: reading saved adapters; no training jobs submitted.')
        save_comparison(comparison.remote(cfg), cfg)
        return
    preview = make_plan({}, cfg)
    print('Experiment:', cfg['experiment'], '| GPU:', cfg['runtime']['gpu'], '| configurations:', len(preview))
    for row in preview:
        print(row['method_label'], row['task'], 'r='+str(row['rank']), 'seed='+str(row['seed']))
    if stage == 'plan':
        return
    prepared = setup.remote(cfg, stage == 'smoke')
    if stage == 'setup':
        print('CPU setup ready. Existing Secret, Volumes and model cache reused. No GPU jobs submitted.')
        return
    pending = [row for row, status in zip(prepared['plan'], prepared['statuses']) if status != 'complete']
    print('Completed configurations skipped on CPU:', len(prepared['plan']) - len(pending))
    worker = train_job.with_options(gpu=cfg['runtime']['gpu'])
    calls = [(worker.spawn(row, cfg, stage == 'smoke'), row['run_id']) for row in pending]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt = Path('run_receipts') / cfg['experiment'] / f'{stage}_{stamp}.json'
    atomic_json(receipt, {'experiment': cfg['experiment'], 'stage': stage,
        'skipped_complete': len(prepared['plan']) - len(pending),
        'calls': [{'call_id': call.object_id, 'run_id': run_id} for call, run_id in calls]})
    print('Submitted:', len(calls), '| receipt:', receipt)
    errors = []
    for call, run_id in calls:
        try:
            print(call.get())
        except Exception as error:
            errors.append(run_id)
            print('FAILED', run_id, type(error).__name__, str(error))
    if stage == 'train' and cfg['comparison']['after_train']:
        save_comparison(comparison.remote(cfg), cfg)
    if errors:
        raise RuntimeError(f'{len(errors)} jobs failed. Inspect logs, fix the cause and rerun the same config to resume.')
    print('Finished', stage)
