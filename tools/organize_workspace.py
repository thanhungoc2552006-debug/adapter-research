"""Organize known generated files from the supplied workspace-tree.txt.

Default: print a plan. --apply moves files without overwriting destinations.
--undo MANIFEST restores verified files. Training source is never moved/edited.
Python 3.10+, standard library only. No network or Modal calls.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid


SUMMARIES = (
    'arc-isr-arcc-l003_comparison.csv',
    'arc-lora-isr-v1_comparison.csv',
    'arcc-isr-fixednorm-rho1-lam0p1-v1_comparison.csv',
)
RECEIPTS = (
    'arc-isr-arcc-l003_train_calls.json',
    'arc-lora-isr-v1_smoke_calls.json',
    'arc-lora-isr-v1_train_calls.json',
    'arcc-isr-fixednorm-rho1-lam0p1-v1_smoke_calls.json',
    'arcc-isr-fixednorm-rho1-lam0p1-v1_train_calls.json',
    'hellaswag-polar-r8-v1_train_calls.json',
)
MAPPING = {
    **{name: 'reports/run_summaries/' + name for name in SUMMARIES},
    **{name: 'run_receipts/' + name for name in RECEIPTS},
    'workspace-tree.txt': 'docs/inventory/workspace-tree.txt',
}


def inside(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path leaves project: ' + relative)
    return path


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_root(root):
    if not (root / 'modal_app.py').is_file() or not (root / 'fixednorm_runner').is_dir():
        raise ValueError('Expected adapter-research root with modal_app.py and fixednorm_runner/. Pass --root PATH.')


def plan(root):
    rows = []
    for source, destination in MAPPING.items():
        src, dst = inside(root, source), inside(root, destination)
        if not src.exists():
            print('SKIP (not at root):', source)
            continue
        if src.is_symlink() or not src.is_file():
            raise ValueError('Expected a regular file: ' + source)
        if dst.exists():
            raise FileExistsError('Destination exists; nothing moved: ' + str(dst))
        rows.append({'source': source, 'destination': destination,
                     'sha256': digest(src), 'state': 'planned'})
    return rows


def write_manifest(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def apply(root, rows):
    if not rows:
        print('Nothing to move.')
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    folder = inside(root, 'archives/workspace_cleanup/' + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = folder / 'manifest.json'
    data = {'schema': 'adapter-workspace-cleanup-v1', 'created_utc': stamp,
            'root_at_apply': str(root), 'status': 'in_progress', 'moves': rows}
    write_manifest(manifest, data)
    try:
        for row in rows:
            src, dst = inside(root, row['source']), inside(root, row['destination'])
            if dst.exists() or digest(src) != row['sha256']:
                raise ValueError('File changed or destination appeared after planning: ' + row['source'])
            dst.parent.mkdir(parents=True, exist_ok=True)
            # No destination overwrite; do not use replace() for user files.
            if dst.exists():
                raise FileExistsError(str(dst))
            src.rename(dst)
            row['state'] = 'moved'
            write_manifest(manifest, data)
        data['status'] = 'complete'
        write_manifest(manifest, data)
    except BaseException:
        print('Move interrupted. Recovery manifest:', manifest)
        raise
    print('Moved files:', len(rows))
    print('Recovery manifest:', manifest)
    relative = manifest.relative_to(root).as_posix()
    print('Undo: python tools/organize_workspace.py --undo "' + relative + '" --apply')
    return manifest


def undo(root, path, do_apply=False):
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError('Recovery manifest must be inside this project')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'adapter-workspace-cleanup-v1':
        raise ValueError('Unsupported recovery manifest')
    actions = []
    seen = set()
    # Validate every file before restoring any file, including after interrupted apply/undo.
    for row in data['moves']:
        source, destination = row['source'], row['destination']
        if source in seen or MAPPING.get(source) != destination:
            raise ValueError('Unexpected or duplicate move in recovery manifest')
        seen.add(source)
        src, dst = inside(root, source), inside(root, destination)
        if src.exists() and dst.exists():
            raise FileExistsError('Both paths exist; refusing overwrite: ' + source)
        current = dst if dst.exists() else src
        if current.is_symlink() or not current.is_file() or digest(current) != row['sha256']:
            raise ValueError('Missing or changed file; refusing undo: ' + source)
        if dst.exists():
            actions.append((row, src, dst))
    for row, src, dst in actions:
        print('RESTORE:', row['destination'], '->', row['source'])
    if not do_apply:
        print('Preview only. Add --apply to restore these files.')
        return
    for row, src, dst in reversed(actions):
        if src.exists() or digest(dst) != row['sha256']:
            raise ValueError('File changed after undo preflight: ' + row['source'])
        dst.rename(src)
        row['state'] = 'restored'
        write_manifest(path, data)
    data['status'] = 'undone'
    write_manifest(path, data)
    print('Restored files:', len(actions), '(empty folders and recovery history retained)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply', action='store_true', help='Apply the printed moves; default is preview only')
    parser.add_argument('--undo', type=Path, help='Preview or apply restoration using a saved manifest')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    validate_root(root)
    if args.undo:
        undo(root, args.undo, args.apply)
        return
    rows = plan(root)
    for row in rows:
        print('MOVE:', row['source'], '->', row['destination'])
    print('Files in plan:', len(rows))
    if args.apply:
        apply(root, rows)
    else:
        print('Preview only. No files or directories created. Add --apply to execute.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(1)
