"""Local entrypoint. Planning and method discovery need no ML packages."""
import argparse
import json
from pathlib import Path
from .config import load
from .registry import methods
from .planning import make_plan


def main():
    parser = argparse.ArgumentParser(prog='python -m adapter_research')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='Validate config without Modal, network or GPU')
    p.add_argument('--config', default='configs/arc_fixednorm_r8.json')
    sub.add_parser('methods', help='List discovered methods')
    p = sub.add_parser('new-method', help='Create a LoRA auxiliary method starter and config')
    p.add_argument('--name', required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p = sub.add_parser('report-local', help='Measure an existing local results backup, CPU only')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', default='configs/arc_report_existing.json')
    args = parser.parse_args()
    if args.command == 'methods':
        for spec in methods()[0].values():
            print(spec.id, '|', spec.label, '| backend:', spec.backend)
    elif args.command == 'new-method':
        from .scaffold import create
        folder, config = create(args.root, args.name)
        print('Method:', folder)
        print('Config:', config)
        print('Starter uses ISR unchanged. Implement your loss in controller.py before treating it as a new method.')
    elif args.command == 'report-local':
        from .reporting.service import build_reports, persist_local
        from .reporting.collect import markdown
        config = load(args.config)
        payloads = build_reports(args.root, config)
        for payload, path in zip(payloads, persist_local(payloads, config)):
            print(markdown(payload))
            print('Report:', path)
    else:
        config = load(args.config)
        plan = make_plan({}, config)
        print('Experiment:', config['experiment'], '| GPU:', config['runtime']['gpu'])
        for row in plan:
            print(row['method_label'], '|', row['task'], '| rank', row['rank'], '| seed', row['seed'])
        print('Configurations:', len(plan), '| no GPU jobs submitted')
        print('Final run IDs include resolved model/dataset revisions from CPU setup.')


if __name__ == '__main__':
    main()
