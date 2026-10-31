import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from polar_runner.experiment import make_plan,ready_checkpoint,collect_rows,atomic_json

class PipelineTests(unittest.TestCase):
    def test_24_independent_runs(self):
        lock={'repo_commit':'abc','model_revision':'m1','dataset_revision':'d1','source_sha256':'s1'}
        plan=make_plan(lock,[4,8,16,32],['lora','dora','polar'],['arc_easy','arc_challenge'])
        self.assertEqual(len(plan),24)
        self.assertEqual(len({p['run_id'] for p in plan}),24)
        for cfg in plan:
            self.assertEqual(cfg['epochs'],5)
            self.assertEqual(cfg['learning_rate'],.004 if cfg['adapter']=='polar' else .0004)
        # Selecting fewer runs or changing checkpoint cadence must not invalidate an existing run.
        one=make_plan(lock,[4],['lora'],['arc_easy'],save_steps=2)[0]
        self.assertEqual(plan[0]['run_id'],one['run_id'])
        # A different model/data revision or numerical config must never reuse those checkpoints.
        changed=make_plan({**lock,'dataset_revision':'d2'},[4],['lora'],['arc_easy'])[0]
        self.assertNotEqual(one['run_id'],changed['run_id'])

    def test_validation(self):
        for kwargs in [{'ranks':[0]},{'ranks':[4,4]},{'adapters':['galore']},{'micro_batch':6},{'dtype':'int8'}]:
            with self.assertRaises(ValueError):make_plan({},**kwargs)

    def checkpoint(self,root,step,ready=True,scaler=True):
        p=Path(root)/f'checkpoint-{step}';p.mkdir()
        for name in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors']:
            (p/name).touch()
        if scaler:(p/'scaler.pt').touch()
        atomic_json(p/'trainer_state.json',{'global_step':step})
        if ready:atomic_json(p/'checkpoint_ready.json',{'global_step':step})
        return p

    def test_incomplete_checkpoint_not_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            older=self.checkpoint(d,10)
            self.checkpoint(d,20,ready=False)
            self.assertEqual(ready_checkpoint(d),str(older))
            newer=self.checkpoint(d,30,scaler=False)
            self.assertEqual(ready_checkpoint(d),str(older))
            self.assertEqual(ready_checkpoint(d,require_scaler=False),str(newer))

    def test_report_excludes_smoke_and_uses_correct_reference(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=make_plan({},[16],['polar'],['arc_easy'])[0]
            folder=Path(d)/'runs'/cfg['run_id'];folder.mkdir(parents=True)
            atomic_json(folder/'run_config.json',cfg)
            atomic_json(folder/'eval_results.json',{'config':{'limit':None},'results':{'arc_easy':{'acc_norm,none':.8182,'acc,none':.8}}})
            smoke=Path(d)/'_smoke'/'x';smoke.mkdir(parents=True)
            atomic_json(smoke/'run_config.json',cfg)
            rows=collect_rows(d)
            self.assertEqual(len(rows),1)
            self.assertAlmostEqual(rows[0]['delta_pp'],0)
            self.assertEqual(rows[0]['paper_pct'],81.82)
            r=json.loads((folder/'eval_results.json').read_text());r['config']['limit']=16
            atomic_json(folder/'eval_results.json',r)
            with self.assertRaises(ValueError):collect_rows(d)

    def test_wandb_account_namespace_preserves_checkpoint(self):
        source=(Path(__file__).parents[1]/'polar_runner/tracking.py').read_text()
        tree=ast.parse(source)
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='tracking_target']
        import hashlib
        ns={'enabled':lambda:True,'os':os,'Path':Path,'hashlib':hashlib}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'tracking','exec'),ns)
        before=dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as d:
                checkpoint=self.checkpoint(d,10)
                os.environ.update(WANDB_ENTITY='account1',WANDB_PROJECT='project')
                target1=ns['tracking_target'](d)
                os.environ['WANDB_ENTITY']='account2'
                target2=ns['tracking_target'](d)
                self.assertNotEqual(target1,target2)
                self.assertTrue((checkpoint/'optimizer.pt').exists())
        finally:
            os.environ.clear();os.environ.update(before)

if __name__=='__main__':unittest.main()
