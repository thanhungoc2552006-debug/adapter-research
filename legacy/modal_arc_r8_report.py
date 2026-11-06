"""Read the existing results Volume and measure final adapters on CPU only."""
from pathlib import Path
import modal
from arc_r8_report import BASE_EXPERIMENTS,names,markdown,write_outputs,discovery_text

image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.4','safetensors==0.5.3')
    .env({'PYTHONPATH':'/opt/report','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    .add_local_file(Path(__file__).with_name('arc_r8_report.py'),'/opt/report/arc_r8_report.py',copy=True))
app=modal.App('adapter-arc-r8-report')
volume=modal.Volume.from_name('polar-table2-results',create_if_missing=False)


@app.function(image=image,volumes={'/results':volume},cpu=4,memory=8192,timeout=3600,scaledown_window=2)
def collect_report(experiments:list[str],rank:int=8,task:str='arc_challenge',discover_isr_lambda:float=-1.):
    from arc_r8_report import collect,collect_with_discovery
    volume.reload()
    if discover_isr_lambda>=0:
        return collect_with_discovery('/results',experiments,discover_isr_lambda,rank,task)
    return collect('/results',experiments,rank,task)


@app.local_entrypoint()
def main(experiments:str=BASE_EXPERIMENTS,rank:int=8,task:str='arc_challenge',output:str='reports/arc_challenge_r8',discover_isr_lambda:float=-1.):
    if rank not in {4,8,16,32}:raise ValueError('rank must be 4,8,16,32')
    if task not in {'arc_easy','arc_challenge'}:raise ValueError('Unsupported ARC task')
    payload=collect_report.remote(names(experiments),rank,task,discover_isr_lambda)
    if 'discovery' in payload:
        print(discovery_text(payload['discovery']))
        if not payload['discovery']['matches']:
            print('Requested run not found. Existing local report files were not overwritten. Check the Modal workspace or restore the run backup.')
            raise SystemExit(2)
    if not payload['rows']:raise ValueError('No matching runs. Check experiment names and current Modal workspace.')
    print('Saved locally:',write_outputs(payload,output))
    print(markdown(payload))
