"""Refactor must preserve the old method's actual gradients and updates."""
import copy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'legacy'))
from fixednorm_runner.regularizer import SRController as LegacyFixed
from fixednorm_runner.sr_config import SRConfig as LegacyConfig
from adapter_research.methods.fixednorm.controller import FixedNormController
from adapter_research.methods.isr.config import SRConfig
from test_isr_training import ToyModel


class MethodEquivalenceTests(unittest.TestCase):
    def test_old_and_new_fixednorm_loss_gradients_and_adamw_updates_match(self):
        torch.set_num_threads(1);torch.manual_seed(144)
        old_model=ToyModel(zero_B=True);new_model=copy.deepcopy(old_model)
        settings={'lambda_max':.1,'warmup_ratio':.2,'ramp_ratio':.2,'expected_modules':1,'frobenius_target':1.,'audit_every':2}
        old=LegacyFixed(old_model,LegacyConfig(**settings),seed=7)
        new=FixedNormController(new_model,SRConfig(**settings),seed=7)
        old.configure(5);new.configure(5)
        old_opt=torch.optim.AdamW([p for p in old_model.parameters() if p.requires_grad],lr=.01)
        new_opt=torch.optim.AdamW([p for p in new_model.parameters() if p.requires_grad],lr=.01)
        x=torch.randn(4,6);y=torch.tensor([0,1,2,3])
        for step in range(5):
            for model,controller,optimizer in [(old_model,old,old_opt),(new_model,new,new_opt)]:
                optimizer.zero_grad()
                loss=model(x,y)['loss'];aux=controller.weighted_loss()
                if aux is not None:loss=loss+aux
                loss.backward()
            for old_p,new_p in zip(old_model.parameters(),new_model.parameters()):
                if old_p.requires_grad:self.assertTrue(torch.equal(old_p.grad,new_p.grad))
            for model,controller,optimizer in [(old_model,old,old_opt),(new_model,new,new_opt)]:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();controller.after_optimizer(False)
            for name,wanted in old_model.state_dict().items():self.assertTrue(torch.equal(wanted,new_model.state_dict()[name]),name)
            self.assertEqual(old.successful_updates,new.successful_updates)


if __name__=='__main__':unittest.main()
