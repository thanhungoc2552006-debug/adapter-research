"""One CPU comparison path for legacy and new experiments; never trains."""
import json
from pathlib import Path
from .collect import collect, write_outputs
from ..registry import from_config


def experiment_names(root, explicit=(), discover=True):
    root = Path(root).resolve()
    names = list(dict.fromkeys(explicit))
    if discover:
        for path in sorted(root.rglob('run_config.json')):
            relative = path.relative_to(root)
            if '_smoke' in relative.parts or path.parent.parent.name != 'runs':
                continue
            try:
                cfg = json.loads(path.read_text(encoding='utf-8'))
                if cfg.get('task') not in {'arc_easy', 'arc_challenge'}:
                    continue
                from_config(cfg)
            except (OSError, ValueError, KeyError):
                continue
            name = path.parent.parent.parent.relative_to(root).as_posix()
            if name not in names:
                names.append(name)
    return names


def build_reports(root, config):
    comparison = config['comparison']
    explicit = comparison['experiments'] + [config['experiment']]
    names = experiment_names(root, explicit, comparison['discover'])
    return [collect(root, names, rank, task, baseline_experiment=comparison['baseline_experiment'],
                    seeds=config['seeds'])
            for rank in config['ranks'] for task in config['tasks']]


def persist_local(payloads, config):
    from .wandb import export_norm_csv
    result = []
    for payload in payloads:
        name = f"comparison_{payload['task']}_r{payload['rank']}"
        folder = Path(config['comparison']['output_dir']) / name
        write_outputs(payload, folder)
        if payload['rows']:
            export_norm_csv(payload, folder / 'per_layer_norms.csv')
        if payload.get('wandb_receipt'):
            (folder / 'wandb_comparison.json').write_text(json.dumps(payload['wandb_receipt'], indent=2), encoding='utf-8')
        result.append(str(folder.resolve()))
    return result
