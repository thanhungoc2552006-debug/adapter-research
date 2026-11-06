"""Validated experiment configuration. JSON is sufficient on the local machine."""
from copy import deepcopy
import json
import importlib
import math
from pathlib import Path
from .common import simple_name
from .registry import method

DEFAULTS = {
    'experiment': 'arc-research-v2', 'baseline_experiment': 'table2-arc-resume',
    'tasks': ['arc_challenge'], 'ranks': [8], 'seeds': [0],
    'methods': [{'name': 'isr_fixednorm', 'parameters': {'lambda_max': .1, 'frobenius_target': 1.}}],
    'training': {'epochs': 5, 'micro_batch': 1, 'effective_batch': 128,
                 'dtype': 'fp16', 'save_steps': 10, 'eval_split': 'test',
                 'lora_alpha': 32, 'lora_dropout': .05, 'cutoff_len': 512},
    'runtime': {'gpu': 'H100', 'eval_batch': 4, 'diagnostics': True,
                'diagnostic_every': 20, 'use_wandb': True},
    'comparison': {'experiments': ['table2-arc-resume', 'arc-lora-isr-v1',
                                  'arcc-isr-fixednorm-rho1-lam0p1-v1'],
                   'discover': True, 'baseline_experiment': 'table2-arc-resume',
                   'label': 'main', 'after_train': True, 'output_dir': 'reports'},
}


def merge(base, update):
    result = deepcopy(base)
    unknown = set(update) - set(base)
    if unknown:
        raise ValueError('Unknown config fields: ' + ', '.join(sorted(unknown)))
    for key, value in update.items():
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError('Expected an object for ' + key)
            result[key] = merge(base[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')


def normalize(value):
    cfg = merge(DEFAULTS, value)
    simple_name(cfg['experiment'])
    if cfg['baseline_experiment']:
        simple_name(cfg['baseline_experiment'])
    if cfg['experiment'] == cfg['baseline_experiment']:
        raise ValueError('Use a new v2 experiment; old baselines remain read-only inputs')
    for field, choices in [('tasks', {'arc_easy', 'arc_challenge'}), ('ranks', {4, 8, 16, 32})]:
        values = cfg[field]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or not set(values) <= choices:
            raise ValueError('Invalid unique ' + field)
    if not isinstance(cfg['seeds'], list) or not cfg['seeds'] or any(type(x) is not int or x < 0 for x in cfg['seeds']) or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('seeds must be unique nonnegative integers')
    entries = []
    for item in cfg['methods']:
        if isinstance(item, str):
            item = {'name': item}
        if not isinstance(item, dict) or set(item) - {'name', 'parameters', 'learning_rate'}:
            raise ValueError('Method entries accept name, parameters, learning_rate')
        spec = method(item['name'])
        parameters = deepcopy(spec.defaults)
        supplied = item.get('parameters', {})
        if not isinstance(supplied, dict) or set(supplied) - set(parameters):
            raise ValueError('Unknown method parameters for ' + spec.id)
        parameters.update(supplied)
        if spec.validate_parameters:
            module, validator = spec.validate_parameters.split(':', 1)
            getattr(importlib.import_module(module), validator)(**parameters)
        if spec.id == 'isr' and parameters.get('frobenius_target', 0) != 0:
            raise ValueError('Use isr_fixednorm to enable rescaling')
        lr = item.get('learning_rate', spec.learning_rate)
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning_rate must be positive and finite')
        entry = {'name': spec.id, 'parameters': parameters, 'learning_rate': lr}
        if entry in entries:
            raise ValueError('Duplicate method configuration')
        entries.append(entry)
    if not entries:
        raise ValueError('Select at least one method')
    cfg['methods'] = entries
    train, runtime, comparison = cfg['training'], cfg['runtime'], cfg['comparison']
    for key in ['epochs', 'micro_batch', 'effective_batch', 'save_steps', 'lora_alpha', 'cutoff_len']:
        positive_integer(train[key], key)
    if train['effective_batch'] % train['micro_batch']:
        raise ValueError('micro_batch must divide effective_batch')
    if train['dtype'] not in {'fp16', 'bf16'} or train['eval_split'] not in {'test', 'validation'}:
        raise ValueError('Unsupported dtype or evaluation split')
    if not isinstance(train['lora_dropout'], (int, float)) or not 0 <= train['lora_dropout'] < 1:
        raise ValueError('Invalid lora_dropout')
    if runtime['gpu'] not in {'H100', 'H100:1', 'A100-80GB', 'A100-80GB:1'}:
        raise ValueError('This pipeline supports one H100 or one A100-80GB')
    for key in ['eval_batch', 'diagnostic_every']:
        positive_integer(runtime[key], key)
    for group, key in [(runtime, 'use_wandb'), (runtime, 'diagnostics'), (comparison, 'discover'), (comparison, 'after_train')]:
        if type(group[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    simple_name(comparison['label'])
    if comparison['baseline_experiment']:
        simple_name(comparison['baseline_experiment'])
    if not isinstance(comparison['experiments'], list):
        raise ValueError('comparison.experiments must be a list')
    for name in comparison['experiments']:
        simple_name(name)
    if not isinstance(comparison['output_dir'], str) or not comparison['output_dir']:
        raise ValueError('comparison.output_dir must be a path')
    if train['eval_split']!='test' and comparison['after_train']:
        raise ValueError('Validation runs require comparison.after_train=false; test tables are built only from full test evaluations')
    return cfg


def load(path):
    return normalize(json.loads(Path(path).read_text(encoding='utf-8-sig')))
