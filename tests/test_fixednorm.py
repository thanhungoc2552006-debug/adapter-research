import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import TrainerCallback
from adapter_research.methods.fixednorm.projection import project_factors, core_spectrum
from adapter_research.methods.isr.controller import Factors, select_lora_factors
from adapter_research.methods.fixednorm.controller import FixedNormController as SRController
from adapter_research.methods.isr.config import SRConfig
from adapter_research.pipeline.trainer import ResearchTrainer
from adapter_research.planning import make_plan
from adapter_research.config import normalize
from test_isr_training import ToyModel, ToyDataset, arguments


class StableRankTrainer(ResearchTrainer):
    def __init__(self, *args, sr_config=SRConfig(), sr_seed=0, **kwargs):
        super().__init__(*args, controller_factory=lambda model,parameters,seed: SRController(model,SRConfig(**parameters),seed),
                         method_parameters=sr_config.to_dict(),sr_seed=sr_seed,**kwargs)
    @property
    def sr(self):return self.controller


class FixedNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projection_changes_only_B_preserves_direction_SR_and_moments(self):
        torch.manual_seed(72)
        model = ToyModel(zero_B=False)
        factors = select_lora_factors(model, 1)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(torch.randn(4, 6), torch.tensor([0,1,2,3]))['loss'].backward()
        optimizer.step()
        A, B = factors[0].A, factors[0].B
        before_A = A.detach().clone()
        before_D = (2*B@A).detach().double()
        before_state = copy.deepcopy(optimizer.state_dict())
        before_grad = B.grad.clone()
        initial_sr = core_spectrum(A,B,2)[2]
        metrics = project_factors(factors, .75)
        after_D = (2*B.double()@A.double()).detach()
        self.assertAlmostEqual(after_D.norm().item(), .75, places=6)
        self.assertTrue(torch.equal(before_A,A))
        self.assertTrue(torch.equal(before_grad,B.grad))
        self.assertTrue(torch.allclose(after_D/.75, before_D/before_D.norm(), atol=1e-6))
        self.assertAlmostEqual(initial_sr, core_spectrum(A,B,2)[2], places=6)
        self.assertAlmostEqual(metrics['projection/spectral_post_mean'], torch.linalg.svdvals(after_D)[0].item(), places=8)
        for key,state in before_state['state'].items():
            for name,tensor in state.items():
                self.assertTrue(torch.equal(tensor,optimizer.state_dict()['state'][key][name]))

    def test_zero_small_norm_and_nonfinite_preflight(self):
        zero = ToyModel()
        metrics = project_factors(select_lora_factors(zero,1),1)
        self.assertEqual(metrics['projection/skipped_small_norm'],1)
        self.assertEqual(metrics['projection/modules_projected'],0)
        self.assertEqual(metrics['projection/frobenius_post_max'],0)
        small = Factors('small',torch.eye(2),torch.eye(2)*1e-9,1.)
        original=small.B.clone()
        self.assertEqual(project_factors([small],1)['projection/skipped_small_norm'],1)
        self.assertTrue(torch.equal(original,small.B))
        good=Factors('good',torch.eye(2),torch.eye(2),4.)
        bad=Factors('bad',torch.eye(2),torch.full((2,2),float('nan')),4.)
        original=good.B.clone()
        with self.assertRaises(FloatingPointError):project_factors([good,bad],1)
        self.assertTrue(torch.equal(good.B,original))

    def test_amp_skip_no_projection_then_warmup_projects(self):
        model=ToyModel(zero_B=False)
        config=SRConfig(expected_modules=1,warmup_ratio=.5,frobenius_target=.7)
        controller=SRController(model,config);controller.configure(10)
        self.assertIsNone(controller.weighted_loss())
        before=copy.deepcopy(model.state_dict())
        controller.after_optimizer(skipped=True)
        self.assertEqual(controller.successful_updates,0)
        for name,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[name]))
        controller.after_optimizer(skipped=False)
        self.assertEqual(controller.successful_updates,1)
        self.assertAlmostEqual(core_spectrum(controller.factors[0].A,controller.factors[0].B,2)[0],.7,places=6)
        self.assertEqual(controller.aux_backward_calls,0)

    def test_norm_target_and_run_identity_validation(self):
        for rho in [-1,float('nan'),float('inf'),1e-9]:
            with self.assertRaises(ValueError):SRConfig(frobenius_target=rho)
        a=make_plan({},normalize({"methods":[{"name":"isr_fixednorm","parameters":{"frobenius_target":1}}]}))[0]
        b=make_plan({},normalize({"methods":[{"name":"isr_fixednorm","parameters":{"frobenius_target":2}}]}))[0]
        self.assertNotEqual(a['run_id'],b['run_id'])
        model=ToyModel(zero_B=False);one=SRController(model,SRConfig(expected_modules=1))
        with self.assertRaises(ValueError):
            SRController(copy.deepcopy(model),SRConfig(expected_modules=1,frobenius_target=2)).load_state_dict(one.state_dict())

    def test_trainer_checkpoints_are_projected_and_resume_matches(self):
        class StopAtThree(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                if state.global_step==3:control.should_training_stop=True
        class CheckPostStep(TrainerCallback):
            def on_step_end(self,args,state,control,model=None,**kwargs):
                A=model.q_proj.lora_A['default'].weight;B=model.q_proj.lora_B['default'].weight
                if abs(core_spectrum(A,B,2)[0]-.8)>2e-6:raise AssertionError('Projection must precede step_end')
        torch.manual_seed(53)
        original=ToyModel(dropout=.1)
        config=SRConfig(expected_modules=1,lambda_max=.1,warmup_ratio=.2,ramp_ratio=.3,frobenius_target=.8,audit_every=2)
        with tempfile.TemporaryDirectory() as root:
            dataset=ToyDataset()
            full=StableRankTrainer(model=copy.deepcopy(original),args=arguments(Path(root)/'full'),train_dataset=dataset,sr_config=config,callbacks=[CheckPostStep()])
            full.train()
            partial=StableRankTrainer(model=copy.deepcopy(original),args=arguments(Path(root)/'resume'),train_dataset=dataset,sr_config=config,callbacks=[CheckPostStep(),StopAtThree()])
            partial.train()
            checkpoint=Path(root)/'resume/checkpoint-3'
            from safetensors.torch import load_file
            saved=load_file(str(checkpoint/'model.safetensors'))
            self.assertAlmostEqual(core_spectrum(saved['q_proj.lora_A.default.weight'],saved['q_proj.lora_B.default.weight'],2)[0],.8,places=6)
            resumed=StableRankTrainer(model=copy.deepcopy(original),args=arguments(Path(root)/'resume'),train_dataset=dataset,sr_config=config,callbacks=[CheckPostStep()])
            resumed.train(resume_from_checkpoint=str(checkpoint))
            for name,wanted in full.model.state_dict().items():
                self.assertTrue(torch.allclose(wanted,resumed.model.state_dict()[name],atol=1e-7,rtol=1e-6),name)
            self.assertEqual(full.sr.successful_updates,resumed.sr.successful_updates)
            self.assertEqual(full.sr.aux_backward_calls,resumed.sr.aux_backward_calls)
            metrics=[json.loads(s) for s in (Path(root)/'full/sr_metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(metrics),6)
            self.assertTrue(all(abs(x['projection/frobenius_post_mean']-.8)<2e-6 for x in metrics))
            self.assertTrue(all(x['projection/modules_projected']==1 for x in metrics))

    def test_real_peft_tiny_llama_fixednorm_save_reload(self):
        from peft import LoraConfig,PeftModel,get_peft_model
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(112)
        base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,use_cache=False))
        frozen=copy.deepcopy(base)
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        data=[{'input_ids':torch.tensor([1,4,7,9,2]),'labels':torch.tensor([1,4,7,9,-100])} for _ in range(8)]
        with tempfile.TemporaryDirectory() as root:
            trainer=StableRankTrainer(model=model,args=arguments(root,max_steps=3),train_dataset=data,sr_config=SRConfig(expected_modules=10,frobenius_target=.5,warmup_ratio=0,lambda_max=.1))
            trainer.train()
            for factor in trainer.sr.factors:self.assertAlmostEqual(core_spectrum(factor.A,factor.B,factor.gamma)[0],.5,places=6)
            for name,value in frozen.state_dict().items():
                # Adapter injection renames base layer tensors; frozen weights stay unchanged.
                parts=name.rsplit('.',1);target=parts[0]
                if target.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','up_proj','down_proj'}:
                    name=target+'.base_layer.'+parts[1]
                self.assertTrue(torch.equal(value,model.base_model.model.state_dict()[name]))
            model.save_pretrained(Path(root)/'adapter')
            restored=PeftModel.from_pretrained(frozen,Path(root)/'adapter')
            model.eval();restored.eval()
            with torch.no_grad():
                x=data[0]['input_ids'][None]
                self.assertTrue(torch.allclose(model(x).logits,restored(x).logits,atol=1e-6))


if __name__=='__main__':unittest.main()
