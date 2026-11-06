import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments

from isr_runner.sr_config import SRConfig
from isr_runner.regularizer import (SRController, exact_factor_spectrum,
    fixed_vector_penalty, gram_frobenius_squared, power_iteration, select_lora_factors)
from isr_runner.trainer import StableRankTrainer
from isr_runner.experiment import atomic_json, collect_rows, make_plan, ready_checkpoint


class ToyAdapter(nn.Module):
    def __init__(self, dropout=0.0, zero_B=True):
        super().__init__()
        self.base_layer = nn.Linear(6, 4, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(6, 3, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(3, 4, bias=False)})
        if zero_B:
            nn.init.zeros_(self.lora_B['default'].weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = {'default': 2.0}
        self.active_adapters = ['default']
        self.use_dora = {'default': False}

    def forward(self, x):
        return self.base_layer(x) + 2 * self.lora_B['default'](self.lora_A['default'](self.dropout(x)))


class ToyModel(nn.Module):
    def __init__(self, dropout=0.0, zero_B=True):
        super().__init__()
        self.q_proj = ToyAdapter(dropout, zero_B)

    def forward(self, input_ids, labels=None):
        logits = self.q_proj(input_ids)
        return {'loss': F.cross_entropy(logits, labels), 'logits': logits}


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n=32):
        generator = torch.Generator().manual_seed(91)
        self.x = torch.randn(n, 6, generator=generator)
        self.y = torch.randint(4, (n,), generator=generator)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return {'input_ids': self.x[i], 'labels': self.y[i]}


def arguments(folder, accumulation=2, max_steps=6, save_steps=1):
    return TrainingArguments(output_dir=str(folder), use_cpu=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=accumulation,
        max_steps=max_steps, learning_rate=.01, warmup_ratio=0.,
        save_strategy='steps', save_steps=save_steps, save_total_limit=2,
        logging_steps=1, report_to=[], disable_tqdm=True, seed=11,
        dataloader_pin_memory=False, remove_unused_columns=False)


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_gram_and_fixed_vector_gradient(self):
        torch.manual_seed(7)
        A = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        B = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        v = F.normalize(torch.randn(5, dtype=torch.float64), dim=0)
        gamma = 4.
        W = gamma * (B @ A)
        self.assertTrue(torch.allclose(gram_frobenius_squared(A, B, gamma), W.square().sum()))
        self.assertTrue(torch.autograd.gradcheck(lambda a,b: fixed_vector_penalty(a,b,v,gamma)[0], (A,B)))
        penalty, s, d = fixed_vector_penalty(A, B, v, gamma)
        ga, gb = torch.autograd.grad(penalty, (A,B))
        q = W.detach() @ v
        G = 2 * torch.outer(q,v) / d.detach() - 2 * s.detach() * W.detach() / d.detach().square()
        self.assertTrue(torch.allclose(ga, gamma * B.detach().T @ G, atol=1e-10))
        self.assertTrue(torch.allclose(gb, gamma * G @ A.detach().T, atol=1e-10))
        self.assertLess(abs((G * W.detach()).sum().item()), 1e-10)

    def test_scale_invariance_and_rank_one(self):
        A = torch.randn(3,6,dtype=torch.float64)
        B = torch.randn(7,3,dtype=torch.float64)
        v = F.normalize(torch.randn(6,dtype=torch.float64),dim=0)
        expected = fixed_vector_penalty(A,B,v)[0]
        for scale in [.1,2.,-3.]:
            self.assertTrue(torch.allclose(expected, fixed_vector_penalty(A*scale,B,v)[0]))
        a = torch.randn(1,6,dtype=torch.float64,requires_grad=True)
        b = torch.randn(7,1,dtype=torch.float64,requires_grad=True)
        right = a.detach().flatten(); right = right / right.norm()
        value = fixed_vector_penalty(a,b,right)[0]
        grads = torch.autograd.grad(value,(a,b))
        self.assertAlmostEqual(value.item(),1.,places=12)
        self.assertLess(max(g.abs().max().item() for g in grads),1e-10)

    def test_power_iteration_and_qr_spectrum(self):
        A = torch.eye(3,dtype=torch.float64)
        B = torch.diag(torch.tensor([4.,2.,1.],dtype=torch.float64))
        v = power_iteration(A,B,F.normalize(torch.ones(3,dtype=torch.float64),dim=0),30)
        loss,s,d = fixed_vector_penalty(A,B,v)
        exact_d,exact_top = exact_factor_spectrum(A,B)
        self.assertAlmostEqual(s.item(),16.,places=10)
        self.assertAlmostEqual(d.item(),21.,places=10)
        self.assertAlmostEqual(exact_top.item(),16.,places=10)
        self.assertTrue(torch.allclose(d,exact_d))
        balanced = fixed_vector_penalty(A,A,v)[0]
        self.assertAlmostEqual(balanced.item(),1/3,places=12)

    def test_zero_warmup_and_rng_isolation(self):
        model = ToyModel(zero_B=False)
        controller = SRController(model,SRConfig(expected_modules=1,warmup_ratio=.5),7)
        controller.configure(10)
        before = torch.random.get_rng_state().clone()
        self.assertIsNone(controller.weighted_loss())
        self.assertEqual(controller.vectors,{})
        self.assertEqual(controller.generators,{})
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        controller.successful_updates = 5
        self.assertIsNotNone(controller.weighted_loss())
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        zero = SRController(ToyModel(),SRConfig(expected_modules=1,warmup_ratio=0),7)
        zero.configure(10)
        self.assertIsNone(zero.weighted_loss())
        self.assertEqual(zero.metrics['sr/skipped_small_norm'],1)

    def test_fixed_M_and_degenerate_vector(self):
        model = ToyModel(zero_B=False)
        model.k_proj = ToyAdapter(zero_B=True)
        controller = SRController(model,SRConfig(expected_modules=2,warmup_ratio=0,ramp_ratio=0,lambda_max=1),0)
        controller.configure(10)
        value = controller.weighted_loss()
        factor = next(f for f in controller.factors if f.name=='q_proj')
        expected = fixed_vector_penalty(factor.A,factor.B,controller.vectors['q_proj'],factor.gamma)[0]/2
        self.assertTrue(torch.allclose(value,expected))
        self.assertEqual(controller.metrics['sr/modules_active'],1)
        A = torch.tensor([[1.,0.],[0.,0.]])
        B = torch.eye(2)
        self.assertIsNone(power_iteration(A,B,torch.tensor([0.,1.]),5))
        controller.vectors['q_proj'] = torch.zeros(6)
        self.assertIsNotNone(controller.weighted_loss())
        self.assertGreater(controller.metrics['sr/pi_restarts'],0)

    def test_counter_and_state_roundtrip(self):
        model = ToyModel(zero_B=False)
        config = SRConfig(expected_modules=1,warmup_ratio=0)
        one = SRController(model,config,3); one.configure(10)
        one.weighted_loss(); one.after_optimizer(skipped=True)
        self.assertEqual(one.successful_updates,0)
        state = one.state_dict()
        two = SRController(copy.deepcopy(model),config,3)
        two.load_state_dict(state); two.configure(10)
        self.assertTrue(torch.allclose(one.weighted_loss(),two.weighted_loss()))
        for name in one.vectors:
            self.assertTrue(torch.equal(one.vectors[name],two.vectors[name]))
        with self.assertRaises(ValueError):
            SRController(copy.deepcopy(model),SRConfig(expected_modules=1,lambda_max=.9),3).load_state_dict(state)
        with self.assertRaises(ValueError):two.configure(20)

    def test_bad_parameters_and_dora_rejected(self):
        model = ToyModel(zero_B=False)
        model.q_proj.use_dora['default'] = True
        with self.assertRaises(ValueError):select_lora_factors(model,1)
        model.q_proj.use_dora['default'] = False
        with torch.no_grad():model.q_proj.lora_A['default'].weight[0,0] = float('nan')
        controller = SRController(model,SRConfig(expected_modules=1,warmup_ratio=0),0)
        controller.configure(10)
        with self.assertRaises(FloatingPointError):controller.weighted_loss()


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_auxiliary_once_and_accumulation_coefficient(self):
        torch.manual_seed(29)
        original = ToyModel(zero_B=False)
        batch = {'input_ids':torch.randn(2,6),'labels':torch.tensor([0,2])}
        reference = None
        for accumulation in [1,2,4,128]:
            with tempfile.TemporaryDirectory() as root:
                model = copy.deepcopy(original)
                trainer = StableRankTrainer(model=model,args=arguments(root,accumulation),
                    sr_config=SRConfig(expected_modules=1,warmup_ratio=0,ramp_ratio=0),sr_seed=5)
                trainer.sr.configure(10)
                for index in range(accumulation):
                    trainer.accelerator.gradient_state._set_sync_gradients(index==accumulation-1)
                    trainer.training_step(model,batch)
                gradients = [p.grad.clone() for p in model.parameters() if p.requires_grad]
                self.assertEqual(trainer.sr.aux_backward_calls,1)
                if reference is None:reference=gradients
                for actual,expected in zip(gradients,reference):
                    self.assertTrue(torch.allclose(actual,expected,rtol=2e-5,atol=2e-6))

    def test_lambda_zero_matches_baseline_with_dropout(self):
        torch.manual_seed(41)
        original = ToyModel(dropout=.15)
        dataset = ToyDataset()
        with tempfile.TemporaryDirectory() as root:
            baseline_model = copy.deepcopy(original)
            baseline = Trainer(model=baseline_model,args=arguments(Path(root)/'baseline'),train_dataset=dataset)
            baseline.train()
            new_model = copy.deepcopy(original)
            new = StableRankTrainer(model=new_model,args=arguments(Path(root)/'isr'),train_dataset=dataset,
                sr_config=SRConfig(expected_modules=1,lambda_max=0))
            new.train()
            for name,value in baseline_model.state_dict().items():
                self.assertTrue(torch.equal(value,new_model.state_dict()[name]),name)
            self.assertEqual(new.sr.aux_backward_calls,0)
            self.assertEqual(new.sr.generators,{})

    def test_real_trainer_resume_matches_uninterrupted(self):
        class StopAfterThree(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                if state.global_step==3:control.should_training_stop=True
        torch.manual_seed(53)
        original = ToyModel(dropout=.1)
        dataset = ToyDataset()
        config = SRConfig(expected_modules=1,lambda_max=.03,warmup_ratio=.2,ramp_ratio=.3,audit_every=2)
        with tempfile.TemporaryDirectory() as root:
            full_model = copy.deepcopy(original)
            full = StableRankTrainer(model=full_model,args=arguments(Path(root)/'full'),train_dataset=dataset,sr_config=config)
            full.train()
            part = StableRankTrainer(model=copy.deepcopy(original),args=arguments(Path(root)/'resume'),
                train_dataset=dataset,sr_config=config,callbacks=[StopAfterThree()])
            part.train()
            checkpoint = Path(root)/'resume/checkpoint-3'
            self.assertTrue((checkpoint/'sr_state.pt').is_file())
            resumed_model = copy.deepcopy(original)
            resumed = StableRankTrainer(model=resumed_model,args=arguments(Path(root)/'resume'),train_dataset=dataset,sr_config=config)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            for name,value in full_model.state_dict().items():
                self.assertTrue(torch.allclose(value,resumed_model.state_dict()[name],rtol=1e-6,atol=1e-7),name)
            self.assertEqual(full.sr.successful_updates,resumed.sr.successful_updates)
            self.assertEqual(full.sr.aux_backward_calls,resumed.sr.aux_backward_calls)
            for name,value in full.sr.vectors.items():
                self.assertTrue(torch.allclose(value,resumed.sr.vectors[name],atol=1e-7))

    def test_partial_last_accumulation_has_one_auxiliary(self):
        with tempfile.TemporaryDirectory() as root:
            args=arguments(root,3,max_steps=2);args.per_device_train_batch_size=1
            trainer=StableRankTrainer(model=ToyModel(zero_B=False),args=args,
                train_dataset=ToyDataset(n=5),sr_config=SRConfig(expected_modules=1,warmup_ratio=0))
            trainer.train()  # five micro-batches: one group of three, one group of two
            self.assertEqual(trainer.state.epoch,1.)
            self.assertEqual(trainer.sr.attempted_updates,2)
            self.assertEqual(trainer.sr.aux_backward_calls,2)

    def test_cpu_grad_scaler_before_unscale_clip_and_overflow(self):
        torch.manual_seed(91)
        original=ToyModel(zero_B=False)
        batch={'input_ids':torch.randn(2,6),'labels':torch.tensor([1,3])}
        config=SRConfig(expected_modules=1,warmup_ratio=0,ramp_ratio=0)
        reference=copy.deepcopy(original)
        controller=SRController(reference,config,7);controller.configure(4)
        (reference(**batch)['loss']+controller.weighted_loss()).backward()
        params=[p for p in reference.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params,.1)
        expected=[p.grad.clone() for p in params]
        with tempfile.TemporaryDirectory() as root:
            model=copy.deepcopy(original)
            trainer=StableRankTrainer(model=model,args=arguments(root,2),sr_config=config,sr_seed=7)
            trainer.sr.configure(4)
            scaler=torch.amp.GradScaler('cpu',init_scale=128.)
            trainer.accelerator.scaler=scaler
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
            for i in range(2):
                trainer.accelerator.gradient_state._set_sync_gradients(i==1)
                trainer.training_step(model,batch)
            scaler.unscale_(optimizer)
            params=[p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params,.1)
            for actual,wanted in zip(params,expected):
                self.assertTrue(torch.allclose(actual.grad,wanted,atol=2e-6,rtol=2e-5))
            scaler.step(optimizer);scaler.update()
            optimizer.zero_grad()
            before=[p.detach().clone() for p in params]
            scaler.scale(model(**batch)['loss']).backward()
            params[0].grad.fill_(float('inf'))
            old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<old_scale
            trainer.sr.after_optimizer(skipped)
            self.assertTrue(skipped)
            self.assertEqual(trainer.sr.successful_updates,0)
            for actual,wanted in zip(params,before):self.assertTrue(torch.equal(actual,wanted))

    def test_real_upstream_peft_tiny_llama_save_reload(self):
        try:
            import peft
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import LlamaConfig,LlamaForCausalLM
        except ImportError:self.skipTest('Install editable PEFT from the pinned PoLAR source')
        self.assertEqual(peft.__version__,'0.14.1.dev0')
        torch.manual_seed(83)
        base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,use_cache=False))
        original_base=copy.deepcopy(base)
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=32,lora_dropout=.05,
            target_modules=['q_proj','k_proj','v_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        initial=copy.deepcopy(model)
        dataset=[{'input_ids':torch.tensor([1,4,7,9,2]),'labels':torch.tensor([1,4,7,9,-100])} for _ in range(8)]
        with tempfile.TemporaryDirectory() as root:
            config=SRConfig(expected_modules=10,warmup_ratio=0,ramp_ratio=0)
            trainer=StableRankTrainer(model=model,args=arguments(root,2,max_steps=3),train_dataset=dataset,
                sr_config=config)
            trainer.train()
            self.assertEqual(trainer.sr.successful_updates,3)
            self.assertEqual(trainer.sr.aux_backward_calls,2)  # first update starts at B=0
            model.save_pretrained(Path(root)/'adapter')
            restored=PeftModel.from_pretrained(original_base,Path(root)/'adapter')
            model.eval();restored.eval()
            with torch.no_grad():
                x=dataset[0]['input_ids'][None]
                self.assertTrue(torch.allclose(model(x).logits,restored(x).logits,atol=1e-6))
            class StopAtTwo(TrainerCallback):
                def on_step_end(self,args,state,control,**kwargs):
                    if state.global_step==2:control.should_training_stop=True
            partial=StableRankTrainer(model=copy.deepcopy(initial),args=arguments(Path(root)/'resume',2,max_steps=3),
                train_dataset=dataset,sr_config=config,callbacks=[StopAtTwo()])
            partial.train()
            resumed=StableRankTrainer(model=copy.deepcopy(initial),args=arguments(Path(root)/'resume',2,max_steps=3),
                train_dataset=dataset,sr_config=config)
            resumed.train(resume_from_checkpoint=str(Path(root)/'resume/checkpoint-2'))
            for name,value in model.state_dict().items():
                self.assertTrue(torch.allclose(value,resumed.model.state_dict()[name],atol=1e-7,rtol=1e-6),name)
            self.assertEqual(trainer.sr.aux_backward_calls,resumed.sr.aux_backward_calls)


class PipelineTests(unittest.TestCase):
    def test_comparison_matches_rank_seed_split_and_keeps_variants(self):
        from compare_arc_results import compare
        baseline=[{'adapter':method,'rank':'8','task':'arc_easy','reproduced_pct':str(value),
                   'run_id':method,'status':'done'} for method,value in [('lora',82.),('polar',81.)]]
        new=[{'adapter':'lora_isr','rank':'8','task':'arc_easy','seed':'0','reproduced_pct':'83.',
              'run_id':'isr1','status':'done','sr_lambda_max':'.01','eval_split':'test'}]
        rows=compare(baseline,new)
        self.assertEqual(rows[0]['isr_minus_lora_pp'],1.)
        self.assertEqual(rows[0]['isr_minus_polar_pp'],2.)
        self.assertEqual(len(compare(baseline,new+[dict(new[0],run_id='isr2',sr_lambda_max='.1')])),2)
        self.assertEqual(compare(baseline,[dict(new[0],eval_split='validation')]),[])
        self.assertEqual(compare(baseline,[dict(new[0],seed='1')])[0]['baseline_status'],'missing_baseline')
        with self.assertRaises(ValueError):compare(baseline+baseline[:1],new)

    def test_report_refuses_validation_test_mix(self):
        cfg=make_plan({},eval_split='validation')[0]
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'runs'/cfg['run_id'];folder.mkdir(parents=True)
            atomic_json(folder/'run_config.json',cfg)
            atomic_json(folder/'eval_results.json',{'config':{'limit':None},'isr_eval_split':'test'})
            with self.assertRaises(ValueError):collect_rows(root)

    def test_plans_and_missing_paper_reference(self):
        plan = make_plan({})
        self.assertEqual(len(plan),2)
        self.assertEqual({p['task'] for p in plan},{'arc_easy','arc_challenge'})
        self.assertTrue(all(p['adapter_backend']=='lora' and p['learning_rate']==.0004 for p in plan))
        changed = make_plan({},sr_config=SRConfig(lambda_max=.1))
        self.assertNotEqual(plan[0]['run_id'],changed[0]['run_id'])
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'runs'/plan[0]['run_id']; folder.mkdir(parents=True)
            atomic_json(folder/'run_config.json',plan[0])
            atomic_json(folder/'eval_results.json',{'config':{'limit':None},'results':{'arc_easy':{'acc_norm,none':.8,'acc,none':.7}}})
            row = collect_rows(root)[0]
            self.assertEqual(row['status'],'done')
            self.assertIsNone(row['paper_pct'])
            self.assertIsNone(row['delta_pp'])

    def test_ready_checkpoint_requires_ISR_state(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'checkpoint-10'; folder.mkdir()
            for name in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors','scaler.pt']:
                (folder/name).touch()
            atomic_json(folder/'trainer_state.json',{'global_step':10})
            atomic_json(folder/'checkpoint_ready.json',{'global_step':10})
            self.assertIsNone(ready_checkpoint(root,require_sr=True))
            (folder/'sr_state.pt').touch()
            self.assertEqual(ready_checkpoint(root,require_sr=True),str(folder))

    def test_checked_upstream_patch_when_source_available(self):
        upstream = Path(os.environ.get('POLAR_UPSTREAM_DIR',str(Path(__file__).parents[1])))
        if not (upstream/'finetune_commonsense.py').is_file():self.skipTest('Set POLAR_UPSTREAM_DIR for pinned patch validation')
        from isr_runner.prepare import prepare
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root);(repo/'callbacks').mkdir()
            shutil.copy(upstream/'finetune_commonsense.py',repo)
            shutil.copy(upstream/'callbacks/landing_callback.py',repo/'callbacks')
            prepare(repo)
            source = (repo/'finetune_modal.py').read_text()
            self.assertIn('trainer = make_trainer(',source)
            self.assertIn('trainer.train(resume_from_checkpoint=checkpoint)',source)
            compile(source,'finetune_modal.py','exec')


if __name__=='__main__':unittest.main()
