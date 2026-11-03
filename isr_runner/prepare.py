"""Build-time, checked patches against the pinned upstream commit."""
import difflib
import hashlib
import json
from pathlib import Path


def prepare(repo=Path('/opt/polar')):
    source=(repo/'finetune_commonsense.py').read_text()
    patched=source
    def replace(old,new):
        nonlocal patched
        if patched.count(old)!=1:
            raise ValueError('Unexpected upstream source: '+old[:100])
        patched=patched.replace(old,new,1)
    replace('import transformers\n',
            'import transformers\nfrom modal_tracking import start_tracking, finish_tracking, monitor_callbacks\nfrom modal_checkpointing import DurableCheckpointCallback, find_ready_checkpoint\nfrom isr_runner.trainer import make_trainer\n')
    replace('            load_in_8bit=False,',
            "            load_in_8bit=False,\n            revision=os.environ['POLAR_MODEL_REVISION'],")
    replace('        gradient_checkpointing = False\n        tokenizer = LlamaTokenizer.from_pretrained(base_model)',
            "        gradient_checkpointing = True\n        tokenizer = LlamaTokenizer.from_pretrained(base_model, revision=os.environ['POLAR_MODEL_REVISION'])")
    replace('        model = prepare_model_for_kbit_training(model)',
            '        model = prepare_model_for_kbit_training(model)\n        model.gradient_checkpointing_enable()\n        model.enable_input_require_grads()')
    replace('    callbacks = []',"    start_tracking(output_dir, 'train')\n    callbacks = monitor_callbacks() + [DurableCheckpointCallback()]")
    replace('    trainer = transformers.Trainer(', '    trainer = make_trainer(')
    replace('            save_steps=200,',"            save_steps=int(os.environ.get('POLAR_SAVE_STEPS','10')),")
    replace('            save_total_limit=1,','            save_total_limit=2,')
    replace('            num_train_epochs=num_epochs,',"            num_train_epochs=num_epochs,\n            max_steps=int(os.environ.get('POLAR_MAX_STEPS','-1')),")
    replace('            logging_steps=0.02,','            logging_steps=1,\n            logging_first_step=True,')
    replace('            report_to="wandb" if use_wandb else None,',
            '            report_to=["wandb"] if os.environ.get("POLAR_USE_WANDB")=="1" else [],')
    replace('        model = torch.compile(model)','        pass  # no whole-model compile')
    replace('        trainer.train()\n\n    model.save_pretrained(output_dir)',
            "        checkpoint = find_ready_checkpoint(output_dir, dtype)\n        trainer.train(resume_from_checkpoint=checkpoint)\n\n    model.save_pretrained(output_dir)\n    with open(os.path.join(output_dir, 'train_complete.json.tmp'), 'w') as f:\n        json.dump({'global_step':trainer.state.global_step, 'epoch':trainer.state.epoch}, f)\n    os.replace(os.path.join(output_dir,'train_complete.json.tmp'),os.path.join(output_dir,'train_complete.json'))\n    finish_tracking('train_complete',train_complete=True,train_global_step=trainer.state.global_step)")
    patched=patched.replace('.shuffle().map(tokenize)','.shuffle(seed=seed).map(tokenize)')
    callback=repo/'callbacks/landing_callback.py'
    original=callback.read_text()
    if original.count('    @torch.compile\n')!=1:
        raise ValueError('Unexpected landing callback')
    modified=original.replace('    @torch.compile\n','')
    callback.write_text(modified)
    (repo/'finetune_modal.py').write_text(patched)
    package=Path(__file__).parent
    (repo/'modal_tracking.py').write_text((package/'tracking.py').read_text())
    (repo/'modal_checkpointing.py').write_text((package/'checkpointing.py').read_text())
    compile(patched,'finetune_modal.py','exec')
    diff=''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='finetune_commonsense.py',tofile='finetune_modal.py'))
    diff+=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='landing_callback.py',tofile='landing_callback_modal.py'))
    (repo/'modal_changes.diff').write_text(diff)
    # Hash the source used for training, including observers for full traceability.
    files=['finetune_modal.py','callbacks/landing_callback.py','modal_tracking.py','modal_checkpointing.py']
    content=''.join((repo/file).read_text() for file in files)
    content+=''.join(file.name+'\n'+file.read_text() for file in sorted(package.glob('*.py')))
    (repo/'modal_source_sha256.txt').write_text(hashlib.sha256(content.encode()).hexdigest())

if __name__=='__main__':
    prepare()
