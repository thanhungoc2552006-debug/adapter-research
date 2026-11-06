"""Lightweight method discovery. Method metadata must not import torch."""
from dataclasses import dataclass, field
import importlib
import pkgutil
import re


@dataclass(frozen=True)
class MethodSpec:
    id: str
    label: str
    backend: str = 'lora'
    learning_rate: float = .0004
    defaults: dict = field(default_factory=dict)
    aliases: tuple = ()
    controller_factory: str = ''
    source_modules: tuple = ()
    validate_parameters: str = ''
    factorization: str = 'BA'  # BA, BSA, or unavailable
    label_parameters: tuple = ()  # ((config key, display key), ...)


def methods():
    from . import methods as package
    found, aliases = {}, {}
    for entry in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if entry.name.startswith('_'):
            continue
        module = importlib.import_module(package.__name__ + '.' + entry.name)
        spec = getattr(module, 'SPEC', None)
        if spec is None:
            continue
        if not isinstance(spec, MethodSpec) or not re.fullmatch(r'[a-z][a-z0-9_]*', spec.id):
            raise ValueError('Invalid method metadata: ' + module.__name__)
        if spec.backend not in {'lora', 'dora', 'polar'} or spec.factorization not in {'BA', 'BSA', 'unavailable'}:
            raise ValueError('Unsupported adapter backend or factorization: ' + spec.id)
        if spec.id in found or spec.id in aliases:
            raise ValueError('Duplicate method: ' + spec.id)
        found[spec.id] = spec
        for alias in spec.aliases:
            if alias in found or alias in aliases:
                raise ValueError('Duplicate method alias: ' + alias)
            aliases[alias] = spec.id
    if set(found) & set(aliases):
        raise ValueError('Method IDs collide with aliases')
    return found, aliases


def method(name):
    found, aliases = methods()
    name = aliases.get(name, name)
    if name not in found:
        raise ValueError('Unknown method ' + repr(name) + '; available: ' + ', '.join(found))
    return found[name]


def label(name, parameters=None):
    spec = method(name)
    parameters = parameters or {}
    result = spec.label
    for key, display in spec.label_parameters:
        if key in parameters:
            value = parameters[key]
            result += f' ({display}={value:g})' if isinstance(value, (int, float)) else f' ({display}={value})'
    return result


def from_config(cfg):
    return method(cfg.get('method', cfg['adapter']))


def parameters_from_config(cfg):
    return cfg.get('method_parameters', cfg.get('sr_config', {}))


def build_controller(model, cfg):
    spec = from_config(cfg)
    if not spec.controller_factory:
        return None
    module, name = spec.controller_factory.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(model, parameters_from_config(cfg), cfg['seed'])
