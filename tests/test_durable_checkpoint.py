import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer
from peft import LoraConfig, get_peft_model

from adapter_research.common import atomic_json, ready_checkpoint
from adapter_research.pipeline.checkpointing import DurableCheckpointCallback
from adapter_research.pipeline.trainer import ResearchTrainer
from test_isr_training import arguments


class DurableCheckpointTests(unittest.TestCase):
    def test_real_adapter_marker_and_controller_identity(self):
        torch.set_num_threads(1)
        torch.manual_seed(321)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, use_cache=False))
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=32,
            target_modules=['q_proj','k_proj','v_proj','up_proj','down_proj'],
            task_type='CAUSAL_LM'))
        data = [{'input_ids': torch.tensor([1,4,7,9,2]),
                 'labels': torch.tensor([1,4,7,9,-100])} for _ in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            for controlled in [False, True]:
                folder = Path(temporary) / str(controlled)
                cfg = {'run_id':'fixture', 'method':'isr' if controlled else 'lora',
                    'adapter':'isr' if controlled else 'lora', 'seed':0,
                    'common_source_sha256':'shared', 'method_source_sha256':'method',
                    'requires_method_state':controlled,
                    'method_parameters':{'expected_modules':10, 'warmup_ratio':0}}
                atomic_json(folder/'run_config.json',cfg)
                kwargs = dict(model=copy.deepcopy(model), args=arguments(folder,max_steps=1),
                    train_dataset=data, callbacks=[DurableCheckpointCallback()])
                trainer = ResearchTrainer(run_config=cfg,**kwargs) if controlled else Trainer(**kwargs)
                trainer.train()
                checkpoint = folder/'checkpoint-1'
                self.assertEqual(ready_checkpoint(folder,require_scaler=False,
                    require_method=controlled),str(checkpoint))
                if controlled:
                    restored = ResearchTrainer(run_config=cfg, model=copy.deepcopy(model),
                        args=arguments(folder,max_steps=1),train_dataset=data)
                    restored._load_from_checkpoint(str(checkpoint))
                    self.assertEqual(restored.controller.successful_updates,1)
                    saved=torch.load(checkpoint/'method_state.pt',weights_only=True)
                    saved['identity']['method_source_sha256']='foreign'
                    torch.save(saved,checkpoint/'method_state.pt')
                    with self.assertRaisesRegex(ValueError,'different method/run/source'):
                        restored._load_from_checkpoint(str(checkpoint))
                marker=json.loads((checkpoint/'checkpoint_ready.json').read_text())
                marker['identity']['method_source_sha256']='foreign'
                atomic_json(checkpoint/'checkpoint_ready.json',marker)
                self.assertIsNone(ready_checkpoint(folder,require_scaler=False,
                    require_method=controlled))


if __name__=='__main__':unittest.main()
