"""Run identities and source fingerprints independent of filesystem location."""
import hashlib
import importlib.util
import json
from pathlib import Path
from .common import PIPELINE_VERSION, TARGET_MODULES
from .registry import method, label

PACKAGE = Path(__file__).parent


def fingerprint_files(paths):
    content = []
    for path in sorted(set(paths)):
        content.append(path.relative_to(PACKAGE).as_posix() + '\n' + path.read_text(encoding='utf-8'))
    return hashlib.sha256('\n'.join(content).encode()).hexdigest()


def common_fingerprint():
    return fingerprint_files(list((PACKAGE / 'pipeline').glob('*.py')) +
                             [PACKAGE / name for name in ['common.py', 'registry.py', 'planning.py', 'config.py']])


def method_fingerprint(name):
    spec = method(name)
    paths = []
    for name in spec.source_modules:
        info = importlib.util.find_spec(name)
        if info is None or not info.origin:
            raise ValueError('Missing method source module: ' + name)
        path = Path(info.origin)
        if info.submodule_search_locations:
            paths.extend(path.parent.glob('*.py'))
        else:
            paths.append(path)
    if not paths:
        raise ValueError('Method must declare source_modules for reproducibility')
    return fingerprint_files(paths)


def make_plan(lock, config):
    train = config['training']
    plan = []
    shared_hash = common_fingerprint()
    for rank in config['ranks']:
        for entry in config['methods']:
            spec = method(entry['name'])
            source_hash = method_fingerprint(spec.id)
            for task in config['tasks']:
                for seed in config['seeds']:
                    cfg = {**lock, 'pipeline_version': PIPELINE_VERSION,
                        'experiment': config['experiment'], 'rank': rank, 'adapter': spec.id, 'method': spec.id, 'adapter_backend': spec.backend,
                        'method_label': label(spec.id, entry['parameters']),
                        'method_parameters': entry['parameters'], 'requires_method_state': bool(spec.controller_factory),
                        'factorization': spec.factorization, 'common_source_sha256': shared_hash,
                        'method_source_sha256': source_hash, 'task': task, 'seed': seed,
                        'epochs': train['epochs'], 'learning_rate': entry['learning_rate'],
                        'regularization_lambda': .005, 'lora_alpha': train['lora_alpha'],
                        'lora_dropout': train['lora_dropout'], 'target_modules': TARGET_MODULES,
                        'parameterize_S': 'identity', 'gradient_type': 'landing', 'cutoff_len': train['cutoff_len'],
                        'micro_batch': train['micro_batch'], 'effective_batch': train['effective_batch'],
                        'dtype': train['dtype'], 'gradient_checkpointing': True, 'warmup_ratio': .03,
                        'scheduler': 'cosine', 'base_weights': 'fp32_after_prepare',
                        'normalization': 'repo_character_length_acc_norm', 'eval_split': train['eval_split']}
                    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
                    cfg['run_id'] = f'{spec.id}_r{rank}_{task}_{digest}'
                    cfg['save_steps'] = train['save_steps']
                    plan.append(cfg)
    return plan
