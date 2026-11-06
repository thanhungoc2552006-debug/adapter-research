import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import modal_pipeline as entry


class OrchestrationTests(unittest.TestCase):
    def run_entry(self, stage, statuses):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=root/'config.json';config.write_text(json.dumps({'runtime':{'use_wandb':False}}))
            setup=SimpleNamespace(remote=Mock(return_value={'plan':[{'run_id':f'run-{i}'} for i in range(len(statuses))],'statuses':statuses}))
            calls=[]
            def spawn(*args):
                call=SimpleNamespace(object_id='fc-fixture',get=Mock(return_value={'status':'complete'}));calls.append(call);return call
            worker=SimpleNamespace(spawn=Mock(side_effect=spawn))
            train=SimpleNamespace(with_options=Mock(return_value=worker))
            report=SimpleNamespace(remote=Mock(return_value=[]))
            cwd=os.getcwd()
            try:
                os.chdir(root)
                with patch.object(entry,'setup',setup),patch.object(entry,'train_job',train),patch.object(entry,'comparison',report),patch.object(entry,'save_comparison') as save:
                    entry.main(stage=stage,config=str(config))
                    if stage=='report':
                        setup.remote.assert_not_called();train.with_options.assert_not_called()
                    if stage=='train':self.assertEqual(worker.spawn.call_count,statuses.count('pending'))
                    report.remote.assert_called_once();save.assert_called_once()
            finally:os.chdir(cwd)
            return calls

    def test_report_stage_never_calls_setup_or_gpu(self):self.run_entry('report',[])
    def test_completed_train_does_not_spawn_gpu_and_updates_report(self):self.run_entry('train',['complete'])
    def test_only_pending_inputs_spawn_and_auto_comparison_runs(self):self.run_entry('train',['complete','pending'])


if __name__=='__main__':unittest.main()
