"""Create one method plugin and one config without copying pipeline code."""
import json
from pathlib import Path
import re
from .registry import method, methods


def create(root, name, base='isr'):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('Method name must be a lowercase Python identifier')
    if base != 'isr':
        raise ValueError('The starter template currently targets LoRA auxiliary methods based on isr')
    method(base)
    found, aliases = methods()
    if name in found or name in aliases:
        raise ValueError('Method ID/alias already exists: '+name)
    folder = Path(root) / 'adapter_research/methods' / name
    config = Path(root) / 'configs' / f'arc_{name}_r8.json'
    if folder.exists() or config.exists():
        raise FileExistsError('Method or config already exists: ' + name)
    folder.mkdir(parents=True)
    label = name.replace('_', ' ').title()
    (folder / '__init__.py').write_text(
        'from dataclasses import replace\nfrom ..isr import SPEC as BASE\n\n'
        f'SPEC = replace(BASE, id={name!r}, label={label!r}, aliases=(),\n'
        f'    controller_factory="adapter_research.methods.{name}.controller:create_controller",\n'
        f'    source_modules=BASE.source_modules + (__name__, "adapter_research.methods.{name}.controller"))\n',
        encoding='utf-8')
    (folder / 'controller.py').write_text(
        '"""Starter only: same mathematics as ISR until weighted_loss is changed."""\n'
        'from ..isr.controller import SRController\nfrom ..isr.config import SRConfig\n\n\n'
        'class Controller(SRController):\n'
        '    def weighted_loss(self):\n'
        '        # Return one weighted scalar with graph, or None during warm-up.\n'
        '        # Called once per optimizer update before unscale and clipping.\n'
        '        return super().weighted_loss()\n\n\n'
        'def create_controller(model, parameters, seed=0):\n'
        '    return Controller(model, SRConfig(**parameters), seed)\n', encoding='utf-8')
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({'experiment': 'arc-' + name.replace('_', '-') + '-v2',
        'methods': [{'name': name, 'parameters': {'lambda_max': .01}}]}, indent=2) + '\n', encoding='utf-8')
    return folder, config
