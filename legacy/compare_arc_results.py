"""Join new ISR scores to previously completed LoRA/PoLAR scores; standard library only."""
import argparse
import csv
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def score(row):
    value=float(row['reproduced_pct'])
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('Invalid accuracy: '+str(row.get('run_id')))
    return value


def compare(baseline_rows, new_rows, baseline_seed=0):
    references={}
    for row in baseline_rows:
        if (row.get('status')!='done' or row.get('adapter') not in {'lora','polar'}
                or row.get('eval_split','test')!='test'):
            continue
        seed=int(row.get('seed') or baseline_seed)
        key=(row['adapter'],int(row['rank']),row['task'],seed)
        if key in references:
            raise ValueError('Ambiguous duplicate baseline; supply one configuration per adapter/rank/task/seed: '+str(key))
        references[key]=(score(row),row['run_id'])
    rows=[]
    for row in new_rows:
        if row.get('status')!='done' or row.get('adapter')!='lora_isr':
            continue
        if row.get('eval_split','test')!='test':
            continue  # Never compare validation ISR to test baselines.
        rank=int(row['rank']);seed=int(row.get('seed') or 0);accuracy=score(row)
        out={'run_id':row['run_id'],'task':row['task'],'rank':rank,'seed':seed,
             'sr_lambda_max':row.get('sr_lambda_max',''),'isr_acc_norm_pct':accuracy}
        for method in ['lora','polar']:
            ref=references.get((method,rank,row['task'],seed))
            out[method+'_acc_norm_pct']=ref[0] if ref else None
            out['isr_minus_'+method+'_pp']=accuracy-ref[0] if ref else None
            out[method+'_run_id']=ref[1] if ref else None
        out['baseline_status']='matched' if all(out[m+'_run_id'] for m in ['lora','polar']) else 'missing_baseline'
        rows.append(out)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,help='Old comparison CSV from the original ARC runner')
    parser.add_argument('--new',required=True,help='New ISR comparison CSV')
    parser.add_argument('--output',default='arc_isr_vs_baselines.csv')
    parser.add_argument('--baseline-seed',type=int,default=0,help='Seed used by legacy CSVs without a seed column')
    args=parser.parse_args()
    rows=compare(read_rows(args.baseline),read_rows(args.new),args.baseline_seed)
    if not rows:raise SystemExit('No completed ISR test results found. Run --stage report after training/evaluation.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved:',path.resolve())
    for row in rows:
        print(row['task'],'r='+str(row['rank']),'ISR='+format(row['isr_acc_norm_pct'],'.4f'),row['baseline_status'])
    print('CSV joins match task/rank/seed/test split. Confirm other training settings and source revisions using run_config.json and source_lock.json.')


if __name__=='__main__':main()
