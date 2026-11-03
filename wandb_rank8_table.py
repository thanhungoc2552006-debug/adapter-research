"""Publish the existing CPU report as a W&B comparison table, without training."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile


def finite(value,low=None,high=None):
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric metric, got '+repr(value))
    if low is not None and value<low:raise ValueError('Metric below valid range')
    if high is not None and value>high:raise ValueError('Metric above valid range')
    return value


def rounded(value,digits=3):
    return round(value,digits) if value is not None else None


def prepare_tables(report):
    rank=int(report['rank']);task=report['task']
    if rank!=8:raise ValueError('This comparison is for rank 8')
    if task not in {'arc_challenge','arc_easy'}:raise ValueError('Expected an ARC task')
    if report.get('expected_modules')!=160:raise ValueError('Use a report measured on the common 160-module pool')
    rows=report.get('rows',[])
    if not rows:raise ValueError('No rows in report.json; run the CPU reporter first')
    labels=[]
    for row in rows:
        if int(row['rank'])!=rank or row['task']!=task:raise ValueError('Mixed task/rank in report')
        method={'lora':'LoRA','dora':'DoRA','polar':'PoLAR','lora_isr':'ISR'}[row['adapter']]
        if row['adapter']=='lora_isr':
            lam=finite(row.get('isr_lambda'),low=0)
            if lam is None:raise ValueError('Missing ISR lambda in report')
            method+=f' (lambda={lam:g})'
        labels.append(method)
    counts=Counter(labels)
    summary=[];details=[];accuracy=[];stable=[];delta=[];notes=[]
    used=set()
    for index,(row,method) in enumerate(zip(rows,labels)):
        if counts[method]>1:method+=f" / seed {row['seed']} / {row['run_id'][-6:]}"
        if method in used:method+=f' / row {index+1}'
        used.add(method)
        complete=row['status']=='complete'
        acc=finite(row.get('acc_norm_pct'),0,100) if complete else None
        se=finite(row.get('acc_norm_stderr_pct'),0) if complete else None
        acc_raw=finite(row.get('acc_pct'),0,100) if complete else None
        sr=finite(row.get('sr_mean'),1-1e-6,rank+1e-6) if row.get('sr_status')=='measured' and complete else None
        if sr is not None and row.get('module_count')!=160:
            raise ValueError('Stable rank was not measured on 160 modules')
        # BA is not the full DoRA update; its full-update SR remains unavailable.
        if row['adapter']=='dora':sr=None
        diff=finite(row.get('delta_vs_lora_pp')) if complete and row.get('comparability')=='matched_recorded_settings' else None
        params=finite(row.get('adapter_parameter_count'),0)
        updates=finite(row.get('global_step'),0)
        visible=method if complete else method+' ['+row['status']+']'
        summary.append([visible,rounded(acc),rounded(diff),rounded(sr),rounded(sr/rank if sr is not None else None),
                        rounded(params/1e6 if params is not None else None),updates,row['seed']])
        details.append([method,row['experiment'],row['run_id'],row['status'],row.get('comparability'),
            row.get('isr_lambda'),row.get('learning_rate'),acc,acc_raw,se,sr,
            row.get('sr_std_modules') if sr is not None else None,row.get('sr_valid_modules'),row.get('sr_zero_modules'),
            row.get('frobenius_mean') if sr is not None else None,row.get('spectral_mean') if sr is not None else None,
            row.get('orthogonality_A_mean'),row.get('orthogonality_B_mean'),row.get('last_logged_ce'),
            row.get('epoch_actual'),row.get('wandb_url'),row.get('notes','')])
        if acc is not None:accuracy.append([method,acc])
        if sr is not None:stable.append([method,sr])
        if diff is not None:delta.append([method,diff])
    for warning in report.get('warnings',[]):notes.append(['Source warning',str(warning)])
    notes.extend([
        ['Measurement','Final saved adapters; same 160 target modules; exact thin QR + rank-core SVD.'],
        ['DoRA','Full-update stable rank is unavailable from adapter factors alone; it is not plotted as zero.'],
        ['Uncertainty','SE (pp) is test-sample standard error. SR std describes variation across modules, not training seeds.'],
        ['Display','Summary rounds to 3 decimals. Full precision and run configs are retained in details and the report artifact.'],
        ['Interpretation','Higher stable rank does not by itself establish higher task accuracy.'],
        ['Snapshot','Source report generated at '+str(report.get('generated_utc','unknown'))],
    ])
    return {'summary_columns':['Method','Acc norm (%)','Delta vs LoRA (pp)','Stable rank mean','SR/r','Params (M)','Updates','Seed'],
        'summary':summary,'details_columns':['Method','Experiment','Run ID','Status','Comparability','ISR lambda','Learning rate',
            'Acc norm (%)','Acc (%)','SE (pp)','Stable rank mean','SR std across modules','SR valid modules','Zero modules',
            'Frobenius mean','Spectral mean','Orthogonality A','Orthogonality B','Last logged CE','Actual epoch','Source W&B run','Notes'],
        'details':details,'accuracy':accuracy,'stable_rank':stable,'delta':delta,'notes':notes}


def report_run_id(entity,project,task,rank,label):
    scope=json.dumps([entity,project,task,rank,label],ensure_ascii=False)
    return 'cmp'+hashlib.sha256(scope.encode()).hexdigest()[:20]


def publish(report,entity,project='polar-table2-arc',label='main',mode='online',directory=None):
    """The same label updates one dedicated report run; never resumes a training run."""
    tables=prepare_tables(report)
    if not entity or not project:raise ValueError('W&B entity and project are required')
    import wandb
    rank=report['rank'];task=report['task']
    identity=report_run_id(entity,project,task,rank,label)
    run=wandb.init(entity=entity,project=project,id=identity,resume='allow',mode=mode,
        name=f'comparison_{task}_r{rank}'+('' if label=='main' else '_'+label),
        job_type='comparison',group='comparison',tags=['results-table','rank-8',task],
        config={'task':task,'rank':rank,'report_label':label,'source_generated_utc':report.get('generated_utc'),
                'source_experiments':sorted({r['experiment'] for r in report['rows']}),
                'source_warnings':report.get('warnings',[]),'measurement':report.get('measurement','')},
        allow_val_change=True,dir=directory,save_code=False,settings=wandb.Settings(disable_git=True))
    try:
        logged={
            'comparison/summary':wandb.Table(columns=tables['summary_columns'],data=tables['summary']),
            'comparison/details':wandb.Table(columns=tables['details_columns'],data=tables['details']),
            'comparison/notes':wandb.Table(columns=['Type','Note'],data=tables['notes']),
        }
        for name,title in [('accuracy','Accuracy: acc_norm (%)'),('stable_rank','Final stable rank mean'),('delta','Accuracy change versus LoRA (pp)')]:
            if tables[name]:
                chart_table=wandb.Table(columns=['Method','Value'],data=tables[name])
                logged['comparison/'+name]=wandb.plot.bar(chart_table,'Method','Value',title=title)
        run.log(logged)
        run.summary['comparison/row_count']=len(tables['summary'])
        run.summary['comparison/source_warning_count']=len(report.get('warnings',[]))
        run.summary['comparison/source_generated_utc']=report.get('generated_utc','unknown')
        with tempfile.TemporaryDirectory(prefix='rank8-report-') as tmp:
            path=Path(tmp)/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            artifact=wandb.Artifact(f'{task}-rank-{rank}-comparison',type='evaluation-report',
                description='Final-adapter comparison, source evaluation metrics and complete measurement/config provenance.')
            artifact.add_file(str(path),name='report.json')
            run.log_artifact(artifact,aliases=['latest'])
            # finish waits for table/artifact upload before the temporary source is removed.
            result={'id':run.id,'url':run.url,'entity':run.entity,'project':run.project,'rows':len(tables['summary']),
                    'table_key':'comparison/summary','source_warnings':report.get('warnings',[])}
            run.finish()
        return result
    except BaseException:
        run.finish(exit_code=1)
        raise
