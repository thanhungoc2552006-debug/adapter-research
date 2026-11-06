import argparse
import json
import os
from pathlib import Path
import lm_eval
from lm_eval.utils import handle_non_serializable,make_table
from modal_tracking import publish_evaluation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True);p.add_argument('--task',required=True)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args()
    result=lm_eval.simple_evaluate(model='hf',
        model_args={'pretrained':'meta-llama/Llama-2-7b-hf','peft':args.adapter,
                    'revision':os.environ['POLAR_MODEL_REVISION'],'dtype':'float16','trust_remote_code':False},
        tasks=[args.task],batch_size=args.batch,device='cuda:0',num_fewshot=None,
        random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0,
        log_samples=False,write_out=True,limit=args.limit,bootstrap_iters=100000)
    assert 'acc_norm,none' in result['results'][args.task]
    path=Path(args.adapter)/'eval_results.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,default=handle_non_serializable));tmp.replace(path)
    print(make_table(result),flush=True)
    publish_evaluation(args.adapter,result)

if __name__=='__main__':
    main()
