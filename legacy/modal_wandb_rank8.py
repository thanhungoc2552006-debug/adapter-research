"""Upload a local report.json as W&B tables/charts on Modal CPU; no training."""
import json
from pathlib import Path
import modal

image=(modal.Image.debian_slim(python_version='3.11').pip_install('wandb==0.22.3')
    .env({'PYTHONPATH':'/opt/report','WANDB_LOG_MODEL':'false','WANDB_WATCH':'false'})
    .add_local_file(Path(__file__).with_name('wandb_rank8_table.py'),'/opt/report/wandb_rank8_table.py',copy=True))
app=modal.App('adapter-wandb-rank8-table')
credentials=modal.Secret.from_name('polar-credentials',required_keys=['WANDB_API_KEY','WANDB_ENTITY'])


@app.function(image=image,secrets=[credentials],cpu=1,memory=2048,timeout=900,scaledown_window=2)
def upload_table(report:dict,label:str='main'):
    import os
    import wandb
    from wandb_rank8_table import prepare_tables,prepare_norm_tables,publish
    prepare_tables(report)  # validate before contacting W&B
    prepare_norm_tables(report)
    wandb.login(key=os.environ['WANDB_API_KEY'],verify=True,relogin=True)
    return publish(report,entity=os.environ['WANDB_ENTITY'],project=os.environ.get('WANDB_PROJECT','polar-table2-arc'),label=label)


@app.local_entrypoint()
def main(report:str='reports/arc_challenge_r8/report.json',label:str='main'):
    from wandb_rank8_table import prepare_tables,prepare_norm_tables,export_norm_csv
    path=Path(report)
    if not path.is_file():
        raise FileNotFoundError(f'Report not found: {path.resolve()}. Run modal_arc_r8_report.py first or provide --report with the correct path.')
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    prepare_tables(data)
    prepare_norm_tables(data)
    count=export_norm_csv(data,path.parent/'per_layer_norms.csv')
    print('Per-layer norm rows:',count,'| CSV:',(path.parent/'per_layer_norms.csv').resolve())
    print('Uploading',len(data['rows']),'rows for',data['task'],'rank',data['rank'])
    for warning in data.get('warnings',[]):print('Source warning:',warning)
    result=upload_table.remote(data,label)
    receipt=path.parent/'wandb_comparison.json'
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('View comparison table:',result['url'])
    print('Open Charts, section comparison, panel summary.')
    print('Per-module norms: layer_norms/per_module. Per-layer curves: sections frobenius and spectral.')
    print('Saved link:',receipt.resolve())
