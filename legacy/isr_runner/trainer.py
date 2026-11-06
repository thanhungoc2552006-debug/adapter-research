"""Pinned single-process Trainer integration: auxiliary before AMP unscale/clipping."""
import json
import os
from pathlib import Path
import time
import torch
import transformers
import accelerate
from transformers import Trainer, TrainerCallback
from .regularizer import SRController
from .sr_config import SRConfig


class _SRLifecycle(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.sr.configure(state.max_steps)

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer.sr.after_optimizer(self.trainer.accelerator.optimizer_step_was_skipped)

    def on_step_end(self, args, state, control, **kwargs):
        metrics = dict(self.trainer.sr.metrics)
        if metrics:
            self.trainer.log(metrics)
            path = Path(args.output_dir) / 'sr_metrics.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps({'global_step': state.global_step, **metrics}) + '\n')


class StableRankTrainer(Trainer):
    def __init__(self, *args, sr_config=SRConfig(), sr_seed=0, **kwargs):
        if transformers.__version__ != '4.50.3' or accelerate.__version__ != '1.6.0':
            raise RuntimeError('ISR Trainer is validated for transformers==4.50.3 and accelerate==1.6.0')
        super().__init__(*args, **kwargs)
        if (self.accelerator.num_processes != 1 or self.args.n_gpu > 1 or self.use_apex
                or self.is_deepspeed_enabled or self.is_fsdp_enabled):
            raise ValueError('ISR v1 supports one process / one GPU with native AMP only')
        self.sr = SRController(self.model, sr_config, sr_seed)
        allowed = {id(parameter) for factors in self.sr.factors for parameter in (factors.A, factors.B)}
        if any(parameter.requires_grad and id(parameter) not in allowed for parameter in self.model.parameters()):
            raise ValueError('ISR expects frozen pretrained weights and only selected LoRA A/B trainable')
        self._sr_last_attempt = None
        self.add_callback(_SRLifecycle(self))
        if self.args.should_save:
            path = Path(self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / 'sr_modules.json').write_text(json.dumps({f.name: {
                'A': list(f.A.shape), 'B': list(f.B.shape), 'gamma': f.gamma}
                for f in self.sr.factors}, indent=2))

    def training_step(self, model, inputs, num_items_in_batch=None):
        ce_loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            attempt = self.state.global_step
            if self._sr_last_attempt == attempt:
                raise RuntimeError('Duplicate auxiliary boundary for one optimizer update')
            self._sr_last_attempt = attempt
            started = time.perf_counter()
            auxiliary = self.sr.weighted_loss()
            if auxiliary is not None:
                # Accelerate.backward divides by its OWN GAS. This auxiliary occurs once/update.
                gas = self.accelerator.gradient_accumulation_steps
                self.accelerator.backward(auxiliary * gas)
                self.sr.aux_backward_calls += 1
            self.sr.metrics['sr/backward_wall_seconds'] = time.perf_counter() - started
            self.sr.metrics['sr/accelerate_gas'] = self.accelerator.gradient_accumulation_steps
            self.sr.metrics['sr/trainer_gas'] = self.args.gradient_accumulation_steps
        return ce_loss  # preserve Trainer's CE-only loss reporting

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            folder = Path(self._get_output_dir(trial)) / f'checkpoint-{self.state.global_step}'
            target = folder / 'sr_state.pt'
            temporary = folder / 'sr_state.pt.tmp'
            torch.save(self.sr.state_dict(), temporary)
            os.replace(temporary, target)
            # Trainer invokes on_save / durable-ready marker only AFTER this returns.

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / 'sr_state.pt'
        if not path.is_file():
            raise ValueError('ISR checkpoint is missing sr_state.pt; exact resume refused')
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.sr.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self._sr_last_attempt = None


def make_trainer(*args, **kwargs):
    """Original trainer for baselines; ISR trainer only for the new method."""
    folder = Path(kwargs['args'].output_dir)
    cfg = json.loads((folder / 'run_config.json').read_text())
    if cfg['adapter'] != 'lora_isr':
        return Trainer(*args, **kwargs)
    return StableRankTrainer(*args, sr_config=SRConfig(**cfg['sr_config']), sr_seed=cfg['seed'], **kwargs)
