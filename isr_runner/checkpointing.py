import os
import json
from pathlib import Path
from transformers import TrainerCallback
from isr_runner.experiment import atomic_json, ready_checkpoint


def find_ready_checkpoint(folder, dtype):
    config=Path(folder)/'run_config.json'
    require_sr=config.is_file() and json.loads(config.read_text())['adapter']=='lora_isr'
    checkpoint=ready_checkpoint(folder,require_scaler=dtype=='fp16',require_sr=require_sr)
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
            config=Path(args.output_dir)/'run_config.json'
            if config.is_file() and json.loads(config.read_text())['adapter']=='lora_isr':
                if not (folder/'sr_state.pt').is_file():
                    raise RuntimeError('ISR checkpoint cannot be marked ready without sr_state.pt')
            atomic_json(folder/'checkpoint_ready.json',{'global_step':state.global_step})
            # Parent process watches this marker and explicitly commits Modal Volume.
            atomic_json(Path(args.output_dir)/'commit_request.json',{'global_step':state.global_step})
