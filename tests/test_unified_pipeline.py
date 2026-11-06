import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from adapter_research.common import atomic_json, ready_checkpoint, MODEL_ID, DATASET_ID, REPO_COMMIT
from adapter_research.config import normalize
from adapter_research.planning import make_plan, common_fingerprint, method_fingerprint
from adapter_research.registry import methods, method

ROOT = Path(__file__).parents[1]
LOCK = {'model_id': MODEL_ID, 'model_revision': 'model-revision', 'dataset_id': DATASET_ID,
        'dataset_revision': 'dataset-revision', 'repo_commit': REPO_COMMIT, 'source_sha256': 'common-image'}


class ConfigAndPluginTests(unittest.TestCase):
    def test_pure_plan_config_changes_and_runtime_not_numerical_identity(self):
        config = normalize({})
        a = make_plan(LOCK, config)[0]
        config['runtime']['gpu'] = 'A100-80GB'
        config['comparison']['label'] = 'other'
        config['training']['save_steps'] = 20
        self.assertEqual(a['run_id'], make_plan(LOCK, config)[0]['run_id'])
        config['methods'][0]['parameters']['frobenius_target'] = .5
        self.assertNotEqual(a['run_id'], make_plan(LOCK, config)[0]['run_id'])
        self.assertEqual(a['adapter_backend'], 'lora')
        self.assertEqual(a['lora_alpha'], 32)
        self.assertEqual(a['effective_batch'], 128)
        self.assertEqual(a['learning_rate'], .0004)
        self.assertEqual(a['method_label'], 'ISR fixed F (lambda=0.1) (rho=1)')

    def test_invalid_settings_and_legacy_aliases(self):
        self.assertEqual(method('lora_isr_fixednorm').id, 'isr_fixednorm')
        for value in [{'tasks':['hellaswag']}, {'runtime':{'gpu':'H100:2'}},
                      {'methods':[{'name':'isr','parameters':{'lamda':.1}}]},
                      {'training':{'micro_batch':3}}, {'runtime':{'use_wandb':'false'}},
                      {'experiment':'../bad'}, {'methods':[{'name':'isr_fixednorm','parameters':{'frobenius_target':1e-9}}]}]:
            with self.assertRaises((ValueError,TypeError)):normalize(value)

    def test_plugin_scaffold_discovery_fingerprints_and_no_torch_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shutil.copytree(ROOT/'adapter_research',root/'adapter_research',ignore=shutil.ignore_patterns('__pycache__'))
            code = '''
import json,sys
from pathlib import Path
from adapter_research.scaffold import create
from adapter_research.planning import common_fingerprint,method_fingerprint,make_plan
from adapter_research.config import normalize
before=(common_fingerprint(),method_fingerprint('isr'))
create(Path('.'),'custom_aux')
from adapter_research.registry import methods
assert 'custom_aux' in methods()[0]
assert before==(common_fingerprint(),method_fingerprint('isr'))
config=normalize({'methods':[{'name':'custom_aux'}]})
assert make_plan({},config)[0]['adapter']=='custom_aux'
assert 'torch' not in sys.modules
print('Plugin discovery and planning need no torch; existing source hashes unchanged')
'''
            process=subprocess.run([sys.executable,'-c',code],cwd=root,env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)

    def test_ready_checkpoint_requires_controller_state_and_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint-3';path.mkdir()
            for name in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors','scaler.pt']:(path/name).touch()
            atomic_json(path/'trainer_state.json',{'global_step':3})
            atomic_json(path/'checkpoint_ready.json',{'global_step':3})
            self.assertIsNone(ready_checkpoint(temporary,require_method=True))
            (path/'method_state.pt').touch()
            self.assertEqual(ready_checkpoint(temporary,require_method=True),str(path))
            atomic_json(path/'checkpoint_ready.json',{'global_step':2})
            self.assertIsNone(ready_checkpoint(temporary,require_method=True))

    def test_pinned_upstream_patch_and_reporting_independent_source_hash(self):
        from adapter_research.pipeline.prepare import prepare
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);(path/'callbacks').mkdir()
            shutil.copy(ROOT/'legacy/finetune_commonsense.py',path)
            shutil.copy(ROOT/'legacy/callbacks/landing_callback.py',path/'callbacks')
            prepare(path)
            result=(path/'finetune_modal.py').read_text()
            self.assertIn('from adapter_research.pipeline.trainer import make_trainer',result)
            self.assertIn('trainer = make_trainer(',result)
            self.assertIn('resume_from_checkpoint=checkpoint',result)
            self.assertIn('logging_steps=1',result)
            compile(result,'patched','exec')

    def test_completed_runs_skip_on_cpu_invalid_eval_refused(self):
        from adapter_research.pipeline import worker
        cfg=make_plan(LOCK,normalize({}))[0]
        with tempfile.TemporaryDirectory() as temporary,patch.object(worker,'RESULTS',Path(temporary)):
            root=Path(temporary)/'experiment/runs'/cfg['run_id'];root.mkdir(parents=True)
            atomic_json(root/'run_config.json',cfg)
            self.assertEqual(worker.inspect_runs([cfg],'experiment',use_wandb=False),['pending'])
            atomic_json(root/'train_complete.json',{'global_step':40})
            (root/'adapter_model.safetensors').touch()
            payload={'config':{'limit':None},'results':{'arc_challenge':{'acc_norm,none':.5}}}
            atomic_json(root/'eval_results.json',payload)
            self.assertEqual(worker.inspect_runs([cfg],'experiment',use_wandb=False),['complete'])
            payload['config']['limit']=16
            atomic_json(root/'eval_results.json',payload)
            with self.assertRaises(ValueError):worker.inspect_runs([cfg],'experiment',use_wandb=False)


if __name__=='__main__':unittest.main()
