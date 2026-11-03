"""CPU tests for task selection, provenance, validation split, reports and W&B."""
import ast
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hellaswag_runner.experiment import DATASET_ID, REFERENCE, atomic_json, collect_rows, make_plan, ready_checkpoint
from hellaswag_runner import worker

YAML = '''task: hellaswag
dataset_path: hellaswag
training_split: train
validation_split: validation
test_split: null
process_docs: !function utils.process_docs
doc_to_target: "{{label}}"
dataset_kwargs:
  trust_remote_code: true
'''


class HellaSwagTests(unittest.TestCase):
    def test_single_selected_configuration(self):
        cfg = make_plan({})[0]
        self.assertEqual((cfg['task'], cfg['rank'], cfg['adapter']), ('hellaswag', 8, 'polar'))
        self.assertEqual(cfg['eval_split'], 'validation')
        self.assertEqual(cfg['epochs'], 5)
        self.assertEqual(cfg['learning_rate'], .004)
        self.assertEqual(REFERENCE[8, 'polar'], 82.14)

    def test_unique_configs_and_resume_identity(self):
        lock = {'dataset_id': DATASET_ID, 'dataset_revision': 'data1', 'model_revision': 'model1'}
        all_runs = make_plan(lock, [4, 8, 16, 32], ['lora', 'dora', 'polar'])
        self.assertEqual(len(all_runs), 12)
        self.assertEqual(len({p['run_id'] for p in all_runs}), 12)
        one = make_plan(lock, [8], ['polar'], save_steps=2)[0]
        same = next(p for p in all_runs if p['rank'] == 8 and p['adapter'] == 'polar')
        self.assertEqual(one['run_id'], same['run_id'])
        changed = make_plan({**lock, 'dataset_revision': 'data2'})[0]
        self.assertNotEqual(changed['run_id'], one['run_id'])

    def test_invalid_or_arc_inputs_rejected(self):
        for args in [{'tasks': ['arc_easy']}, {'tasks': []}, {'ranks': [4, 4]},
                     {'ranks': [0]}, {'adapters': ['new']}, {'micro_batch': 6}, {'dtype': 'int8'}]:
            with self.assertRaises(ValueError):
                make_plan({}, **args)
        with self.assertRaises(ValueError):
            make_plan({'dataset_id': 'allenai/ai2_arc'})

    def test_dataset_yaml_pin_is_idempotent_and_preserves_task(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            target = repo / 'lm-evaluation-harness/lm_eval/tasks/hellaswag/hellaswag.yaml'
            target.parent.mkdir(parents=True)
            with patch.object(worker, 'REPO', repo), patch.object(worker.subprocess, 'check_output', return_value=YAML):
                lock = {'dataset_id': DATASET_ID, 'dataset_revision': 'a' * 40}
                worker.pin_dataset(lock)
                first = target.read_text()
                worker.pin_dataset(lock)
                self.assertEqual(first, target.read_text())
                self.assertIn('dataset_path: Rowan/hellaswag', first)
                self.assertIn('validation_split: validation', first)
                self.assertIn('test_split: null', first)
                self.assertIn('process_docs: !function utils.process_docs', first)
                self.assertEqual(first.count('dataset_kwargs:'), 1)
                self.assertNotIn('trust_remote_code', first)
                with self.assertRaises(ValueError):
                    worker.pin_dataset({'dataset_id': 'wrong', 'dataset_revision': 'x'})

    def add_result(self, root, limit=None):
        cfg = make_plan({})[0]
        folder = Path(root) / 'runs' / cfg['run_id']
        folder.mkdir(parents=True)
        atomic_json(folder / 'run_config.json', cfg)
        atomic_json(folder / 'eval_results.json', {'config': {'limit': limit},
            'results': {'hellaswag': {'acc_norm,none': .8214, 'acc,none': .65}}})
        return folder

    def test_report_reference_smoke_exclusion_and_pending(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.add_result(root)
            smoke = Path(root) / '_smoke' / 'unused'
            smoke.mkdir(parents=True)
            atomic_json(smoke / 'run_config.json', make_plan({})[0])
            rows = collect_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['eval_split'], 'validation')
            self.assertEqual(rows[0]['status'], 'done')
            self.assertAlmostEqual(rows[0]['delta_pp'], 0)
            (folder / 'eval_results.json').unlink()
            row = collect_rows(root)[0]
            self.assertEqual(row['status'], 'pending')
            self.assertIsNone(row['reproduced_pct'])

    def test_smoke_result_cannot_be_reported_as_full(self):
        with tempfile.TemporaryDirectory() as root:
            self.add_result(root, limit=16)
            with self.assertRaises(ValueError):
                collect_rows(root)

    def test_incomplete_checkpoint_not_resumed(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / 'checkpoint-10'
            checkpoint.mkdir()
            for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'scaler.pt',
                         'adapter_config.json', 'adapter_model.safetensors']:
                (checkpoint / name).touch()
            atomic_json(checkpoint / 'trainer_state.json', {'global_step': 10})
            self.assertIsNone(ready_checkpoint(root))
            atomic_json(checkpoint / 'checkpoint_ready.json', {'global_step': 10})
            self.assertEqual(ready_checkpoint(root), str(checkpoint))

    def test_wandb_hellaswag_reference_and_validation_label(self):
        source = (Path(__file__).parents[1] / 'hellaswag_runner/tracking.py').read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'publish_evaluation')
        class Run:
            id = 'offline-test'
            project = 'test'
            entity = 'test'
            summary = {}
            def log(self, data):
                self.data = data
        run = Run()
        namespace = {'Path': Path, 'json': json, 'enabled': lambda: True,
                     'start_tracking': lambda *a: run, 'finish_tracking': lambda *a, **kw: None,
                     'tracking_target': lambda p: p}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'tracking', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as root:
            folder = self.add_result(root)
            atomic_json(folder / 'train_complete.json', {'global_step': 100})
            result = json.loads((folder / 'eval_results.json').read_text())
            namespace['publish_evaluation'](folder, result)
            self.assertEqual(run.data['eval/paper_acc_norm_pct'], 82.14)
            self.assertEqual(run.data['eval/split'], 'validation')
            self.assertAlmostEqual(run.data['eval/delta_pp'], 0)
            self.assertTrue((folder / 'eval_synced.json').exists())

    def test_prepare_against_pinned_source_when_available(self):
        original = Path(os.environ.get('POLAR_UPSTREAM_DIR', str(Path(__file__).parents[1])))
        if not (original / 'finetune_commonsense.py').exists():
            self.skipTest('Set POLAR_UPSTREAM_DIR to a checkout of the pinned upstream commit')
        from hellaswag_runner.prepare import prepare
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            (repo / 'callbacks').mkdir()
            shutil.copy(original / 'finetune_commonsense.py', repo)
            shutil.copy(original / 'callbacks/landing_callback.py', repo / 'callbacks')
            prepare(repo)
            generated = (repo / 'finetune_modal.py').read_text()
            self.assertIn('resume_from_checkpoint=checkpoint', generated)
            self.assertIn('DurableCheckpointCallback()', generated)
            self.assertIn('from hellaswag_runner.experiment import REFERENCE', (repo / 'modal_tracking.py').read_text())
            compile(generated, 'finetune_modal.py', 'exec')


if __name__ == '__main__':
    unittest.main()
