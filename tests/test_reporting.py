import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file

from adapter_research.common import atomic_json
from adapter_research.config import normalize
from adapter_research.planning import make_plan
from adapter_research.registry import methods, method
from adapter_research.reporting.collect import collect, factor_metrics
from adapter_research.reporting.service import experiment_names, build_reports, persist_local
from adapter_research.reporting.wandb import prepare_tables, prepare_norm_tables, publish, report_run_id
from test_unified_pipeline import LOCK


def add_run(root, experiment, name='lora', legacy=False, seed=0, complete=True, rank=8, smoke=False, limited=False):
    config=normalize({'experiment':'fixture-v2','methods':[name],'seeds':[seed],'ranks':[rank]})
    cfg=make_plan(LOCK,config)[0]
    if legacy:
        aliases={'isr':'lora_isr','isr_fixednorm':'lora_isr_fixednorm'}
        cfg['adapter']=aliases.get(name,name)
        cfg['method']=cfg['adapter']
        cfg['sr_config']=cfg.pop('method_parameters')
        cfg.pop('pipeline_version')
    folder=Path(root)/experiment/('_smoke' if smoke else 'runs')/cfg['run_id'];folder.mkdir(parents=True)
    atomic_json(folder/'run_config.json',cfg)
    if not complete:return cfg,folder
    atomic_json(folder/'train_complete.json',{'global_step':40,'epoch':4.45755})
    atomic_json(folder/'eval_results.json',{'config':{'limit':16 if limited else None},
        'results':{'arc_challenge':{'acc_norm,none':.5,'acc,none':.4,'acc_norm_stderr,none':.01}}})
    atomic_json(folder/'adapter_config.json',{'r':rank,'lora_alpha':32,'use_dora':name=='dora'})
    rng=np.random.default_rng(99);tensors={};gamma=32/rank
    for layer in range(32):
        for target in cfg['target_modules']:
            A=rng.normal(size=(rank,12)).astype(np.float32)
            B=rng.normal(size=(10,rank)).astype(np.float32)
            if name=='isr_fixednorm':B/=np.linalg.norm(gamma*B.astype(float)@A.astype(float))
            kind='mlp' if target in {'up_proj','down_proj'} else 'self_attn'
            key=f'base_model.model.model.layers.{layer}.{kind}.{target}'
            tensors[key+'.lora_A.weight']=A;tensors[key+'.lora_B.weight']=B
            if name=='polar':tensors[key+'.lora_S.parametrizations.weight.original']=np.diag(np.arange(1,rank+1,dtype=np.float32))
    save_file(tensors,str(folder/'adapter_model.safetensors'))
    return cfg,folder


class ReportingTests(unittest.TestCase):
    def test_exact_norm_dense_reference(self):
        rng=np.random.default_rng(42)
        A=rng.normal(size=(8,12));B=rng.normal(size=(10,8));S=rng.normal(size=(8,8))
        for scale in [None,S]:
            W=4*B@A if scale is None else 4*B@scale@A
            metrics=factor_metrics(A,B,scale,gamma=4)
            self.assertAlmostEqual(metrics['frobenius'],np.linalg.norm(W),places=10)
            self.assertAlmostEqual(metrics['spectral'],np.linalg.svd(W,compute_uv=False)[0],places=10)

    def test_legacy_and_new_methods_same_comparison_160_pool_fixednorm(self):
        with tempfile.TemporaryDirectory() as root:
            add_run(root,'table2-arc-resume','lora',legacy=True)
            add_run(root,'table2-arc-resume','dora',legacy=True)
            add_run(root,'table2-arc-resume','polar',legacy=True)
            add_run(root,'arc-lora-isr-v1','isr',legacy=True)
            add_run(root,'new-fixed','isr_fixednorm')
            # Different lambda experiment names are found from manifests, not guessed names.
            cfg,folder=add_run(root,'different-name','isr',legacy=True,complete=False)
            add_run(root,'smoke-only','isr_fixednorm',smoke=True)
            names=experiment_names(root,[],True)
            self.assertIn('different-name',names);self.assertNotIn('smoke-only',names)
            payload=collect(root,names)
            self.assertEqual(len(payload['rows']),6)
            self.assertEqual(len(payload['modules']),640)
            fixed=next(row for row in payload['rows'] if row['method_id']=='isr_fixednorm')
            self.assertAlmostEqual(fixed['frobenius_mean'],1,places=6)
            self.assertEqual(fixed['delta_vs_lora_pp'],0)
            dora=next(row for row in payload['rows'] if row['method_id']=='dora')
            self.assertIsNone(dora['sr_mean']);self.assertNotEqual(dora['sr_status'],'measured')
            norms=prepare_norm_tables(payload)
            self.assertEqual(len(norms['data']),640)
            self.assertTrue(any('rho=1' in row[0] for row in norms['data']))
            self.assertEqual(len(prepare_tables(payload)['summary']),6)

    def test_seed_rank_split_limits_and_config_matching(self):
        with tempfile.TemporaryDirectory() as root:
            add_run(root,'table2-arc-resume')
            add_run(root,'another-baseline')
            cfg,folder=add_run(root,'new-fixed','isr_fixednorm')
            add_run(root,'seed1','isr',seed=1)
            add_run(root,'rank4','isr',rank=4)
            add_run(root,'limited','isr',limited=True)
            payload=collect(root,experiment_names(root),seeds=[0])
            fixed=next(row for row in payload['rows'] if row['experiment']=='new-fixed')
            self.assertEqual(fixed['comparability'],'matched_recorded_settings')
            self.assertFalse(any(row['experiment'] in {'seed1','rank4'} for row in payload['rows']))
            self.assertEqual(next(row for row in payload['rows'] if row['experiment']=='limited')['status'],'invalid_eval')
            cfg['dataset_revision']='changed';atomic_json(folder/'run_config.json',cfg)
            payload=collect(root,['table2-arc-resume','new-fixed'])
            fixed=next(row for row in payload['rows'] if row['experiment']=='new-fixed')
            self.assertIsNone(fixed['delta_vs_lora_pp'])
            self.assertIn('dataset_revision',fixed['comparability'])

    def test_generic_plugin_reaches_tables_without_reporting_edits(self):
        found,aliases=methods();new=replace(method('isr'),id='custom_aux',label='Custom auxiliary',aliases=())
        with tempfile.TemporaryDirectory() as root:
            cfg,folder=add_run(root,'new-method','isr')
            cfg['method']='custom_aux';cfg['adapter']='custom_aux'
            atomic_json(folder/'run_config.json',cfg)
            with patch('adapter_research.registry.methods',return_value=({**found,'custom_aux':new},aliases)):
                payload=collect(root,['new-method'])
            self.assertEqual(len(payload['rows']),1)
            self.assertTrue(prepare_tables(payload)['summary'][0][0].startswith('Custom auxiliary'))
            self.assertEqual(len(prepare_norm_tables(payload)['data']),160)

    def test_cpu_export_actual_sdk_offline_and_report_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            add_run(root,'table2-arc-resume')
            add_run(root,'fixed','isr_fixednorm')
            config=normalize({'experiment':'fixed','comparison':{'experiments':['table2-arc-resume'],'output_dir':str(root/'local-reports')}})
            payloads=build_reports(root,config)
            paths=persist_local(payloads,config)
            self.assertTrue((Path(paths[0])/'per_layer_norms.csv').is_file())
            result=publish(payloads[0],entity='test-local',project='pipeline-offline-test',mode='offline',directory=str(root))
            self.assertEqual(result['norm_rows'],320)
            self.assertEqual(result['rows'],2)
            self.assertEqual(result['id'],report_run_id('test-local','pipeline-offline-test','arc_challenge',8,'main'))
            self.assertTrue(list(root.glob('wandb/offline-run-*/run-*.wandb')))


if __name__=='__main__':unittest.main()
