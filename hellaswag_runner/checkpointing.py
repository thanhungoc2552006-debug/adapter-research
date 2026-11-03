import os
from pathlib import Path
from transformers import TrainerCallback
from hellaswag_runner.experiment import atomic_json, ready_checkpoint


def find_ready_checkpoint(folder, dtype):
    checkpoint=ready_checkpoint(folder,require_scaler=dtype=='fp16')
    present=list(Path(folder).glob('checkpoint-*'))
    if present and checkpoint is None:
        print('No fully saved checkpoint found; restarting this run from base model.',flush=True)
    if checkpoint:
        print('Resuming durable checkpoint:',checkpoint,flush=True)
    return checkpoint


class DurableCheckpointCallback(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if state.is_world_process_zero:
            folder=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            atomic_json(folder/'checkpoint_ready.json',{'global_step':state.global_step})
            # Parent process watches this marker and explicitly commits Modal Volume.
            atomic_json(Path(args.output_dir)/'commit_request.json',{'global_step':state.global_step})
