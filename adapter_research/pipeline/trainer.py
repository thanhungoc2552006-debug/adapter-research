"""Pinned single-process Trainer integration: auxiliary before AMP unscale/clipping."""
import json
import os
from pathlib import Path
import time
import torch
import transformers
import accelerate
from transformers import Trainer, TrainerCallback
from ..registry import build_controller


class _MethodLifecycle(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.controller.configure(state.max_steps)

    def on_optimizer_step(self, args, state, control, **kwargs):
        # HF 4.50.3 invokes this after optimizer.step, before step_end/save/eval.
        self.trainer.controller.after_optimizer(self.trainer.accelerator.optimizer_step_was_skipped)

    def on_step_end(self, args, state, control, **kwargs):
        metrics = dict(self.trainer.controller.metrics)
        if metrics:
            self.trainer.log(metrics)
            path = Path(args.output_dir) / 'sr_metrics.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps({'global_step': state.global_step, **metrics}) + '\n')


class ResearchTrainer(Trainer):
    def __init__(self, *args, run_config=None, controller_factory=None, method_parameters=None, sr_seed=0, **kwargs):
        if transformers.__version__ != '4.50.3' or accelerate.__version__ != '1.6.0':
            raise RuntimeError('ISR Trainer is validated for transformers==4.50.3 and accelerate==1.6.0')
        super().__init__(*args, **kwargs)
        if (self.accelerator.num_processes != 1 or self.args.n_gpu > 1 or self.use_apex
                or self.is_deepspeed_enabled or self.is_fsdp_enabled):
            raise ValueError('ISR v1 supports one process / one GPU with native AMP only')
        self.controller = (controller_factory(self.model, method_parameters or {}, sr_seed)
                           if controller_factory is not None else build_controller(self.model, run_config))
        if self.controller is None:
            raise ValueError('ResearchTrainer requires a method controller')
        allowed = {id(parameter) for factors in self.controller.factors for parameter in (factors.A, factors.B)}
        if any(parameter.requires_grad and id(parameter) not in allowed for parameter in self.model.parameters()):
            raise ValueError('ISR expects frozen pretrained weights and only selected LoRA A/B trainable')
        identity_config = run_config or {}
        self._method_identity = {key: identity_config.get(key) for key in
            ['run_id','method','common_source_sha256','method_source_sha256']}
        self._sr_last_attempt = None
        self.add_callback(_MethodLifecycle(self))
        if self.args.should_save:
            path = Path(self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / 'method_modules.json').write_text(json.dumps({f.name: {
                'A': list(f.A.shape), 'B': list(f.B.shape), 'gamma': f.gamma}
                for f in self.controller.factors}, indent=2))

    def training_step(self, model, inputs, num_items_in_batch=None):
        ce_loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            attempt = self.state.global_step
            if self._sr_last_attempt == attempt:
                raise RuntimeError('Duplicate auxiliary boundary for one optimizer update')
            self._sr_last_attempt = attempt
            started = time.perf_counter()
            auxiliary = self.controller.weighted_loss()
            if auxiliary is not None:
                # Accelerate.backward divides by its OWN GAS. This auxiliary occurs once/update.
                gas = self.accelerator.gradient_accumulation_steps
                self.accelerator.backward(auxiliary * gas)
                self.controller.aux_backward_calls += 1
            self.controller.metrics['sr/backward_wall_seconds'] = time.perf_counter() - started
            self.controller.metrics['sr/accelerate_gas'] = self.accelerator.gradient_accumulation_steps
            self.controller.metrics['sr/trainer_gas'] = self.args.gradient_accumulation_steps
        return ce_loss  # preserve Trainer's CE-only loss reporting

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            folder = Path(self._get_output_dir(trial)) / f'checkpoint-{self.state.global_step}'
            target = folder / 'method_state.pt'
            temporary = folder / 'method_state.pt.tmp'
            torch.save({'format':'arc-method-state-v2', 'identity':self._method_identity,
                        'controller':self.controller.state_dict()}, temporary)
            os.replace(temporary, target)
            # Trainer invokes on_save / durable-ready marker only AFTER this returns.

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / 'method_state.pt'
        if not path.is_file():
            raise ValueError('ISR checkpoint is missing method_state.pt; exact resume refused')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') != 'arc-method-state-v2' or saved.get('identity') != self._method_identity:
            raise ValueError('Checkpoint belongs to a different method/run/source version')
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.controller.load_state_dict(saved['controller'])
        self._sr_last_attempt = None


def make_trainer(*args, **kwargs):
    """Original trainer for baselines; ISR trainer only for the new method."""
    folder = Path(kwargs['args'].output_dir)
    cfg = json.loads((folder / 'run_config.json').read_text())
    if not cfg.get('requires_method_state', False):
        return Trainer(*args, **kwargs)
    return ResearchTrainer(*args, run_config=cfg, **kwargs)
