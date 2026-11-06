"""Shared constants, paths and durable checkpoint selection; no ML imports."""
import json
from pathlib import Path
import re

REPO_COMMIT = '7c9eeabe54cf94e2051632829019c86c4198b3df'
MODEL_ID = 'meta-llama/Llama-2-7b-hf'
DATASET_ID = 'allenai/ai2_arc'
VOLUME_RESULTS = 'polar-table2-results'
VOLUME_CACHE = 'polar-table2-hf-cache'
PIPELINE_VERSION = 'arc-unified-v2'
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']


def simple_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Use a simple name of up to 64 letters, digits, underscores or hyphens')
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def ready_checkpoint(folder, require_scaler=True, require_method=False):
    candidates = []
    config_path = Path(folder) / 'run_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    expected_identity = {key: config.get(key) for key in
        ['run_id','method','common_source_sha256','method_source_sha256']} if config else None
    for path in Path(folder).glob('checkpoint-*'):
        if not path.is_dir() or not path.name[11:].isdigit():
            continue
        required = ['checkpoint_ready.json', 'trainer_state.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'adapter_config.json']
        if require_scaler:
            required.append('scaler.pt')
        if require_method:
            required.append('method_state.pt')
        if not all((path / name).is_file() for name in required):
            continue
        if not any((path / name).is_file() for name in ['adapter_model.safetensors', 'adapter_model.bin']):
            continue
        try:
            state = json.loads((path / 'trainer_state.json').read_text())
            marker = json.loads((path / 'checkpoint_ready.json').read_text())
            step = int(path.name[11:])
            if expected_identity is not None and marker.get('identity') != expected_identity:
                continue
            if state['global_step'] == step and marker['global_step'] == step:
                candidates.append((step, path))
        except (OSError, ValueError, KeyError):
            continue
    return str(max(candidates)[1]) if candidates else None
