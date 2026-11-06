"""Install this extracted bundle into an existing repo. Default: preview only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

OLD_PATHS = [
    'polar_runner','isr_runner','fixednorm_runner','hellaswag_runner','tests',
    'modal_app.py','modal_lora_isr.py','modal_lora_isr_fixednorm.py','modal_hellaswag.py',
    'modal_arc_r8_report.py','modal_wandb_rank8.py','arc_r8_report.py',
    'wandb_rank8_table.py','compare_arc_results.py',
    'README_VI.md','README_LORA_ISR_ARC.md','README_FIXEDNORM_VI.md',
    'VALIDATION_LORA_ISR.md','VALIDATION_FIXEDNORM.md','validation/hellaswag_data_audit.json',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    path=root/relative
    if not path.resolve().is_relative_to(root):raise ValueError('Path leaves repo: '+relative)
    return path


def install(bundle, root, apply=False):
    bundle=Path(bundle).resolve();root=Path(root).resolve()
    if bundle==root:raise ValueError('Extract bundle outside the target repo; source and destination must differ')
    if not root.is_dir():raise ValueError('Target repo directory does not exist')
    manifest=json.loads((bundle/'INSTALL_MANIFEST.json').read_text(encoding='utf-8'))
    files=manifest['files']
    for name,checksum in files.items():
        source=inside(bundle,name)
        if not source.is_file() or sha(source)!=checksum:raise ValueError('Bundle changed or incomplete: '+name)
    archive=[name for name in OLD_PATHS if inside(root,name).exists()]
    # Existing legacy/package copies are preserved as a whole before an upgrade.
    for name in ['adapter_research','legacy']:
        if (root/name).exists():archive.append(name)
    archive=sorted(set(archive),key=lambda name:(len(Path(name).parts),name))
    archive=[name for name in archive if not any(name!=parent and Path(parent) in Path(name).parents for parent in archive)]
    writes=[]
    for name,checksum in files.items():
        target=inside(root,name)
        if name=='.gitignore' and target.exists():continue  # preserve user's ignore rules
        under_archive=any(name==parent or Path(parent) in Path(name).parents for parent in archive)
        if target.is_file() and sha(target)==checksum and not under_archive:continue
        if target.exists() and not under_archive:archive.append(name)
        writes.append(name)
    print('Archive old paths:',len(archive))
    for name in archive:print('  ARCHIVE',name)
    print('Install files:',len(writes))
    if not apply:
        print('Preview only. Add --apply to install. No files changed.')
        return None
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    history=inside(root,'archives/unified-migration/'+stamp)
    original=history/'original';original.mkdir(parents=True,exist_ok=False)
    record={'status':'in_progress','archived':[],'installed':[],'bundle_files':files}
    record_path=history/'migration.json'
    def save():record_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    save()
    try:
        for name in archive:
            target=inside(root,name);backup=inside(original,name)
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(target),str(backup));record['archived'].append(name);save()
        for name in writes:
            target=inside(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError('Destination appeared during install: '+name)
            with target.open('xb') as stream:stream.write(inside(bundle,name).read_bytes())
            if sha(target)!=files[name]:raise ValueError('Installed checksum mismatch: '+name)
            record['installed'].append(name);save()
        record['status']='complete';save()
    except BaseException:
        record['status']='interrupted';save()
        print('Installation interrupted. Originals retained at:',original)
        raise
    print('Installed. Old files retained at:',original)
    print('Git metadata, reports, receipts and Modal Volumes were not moved.')
    return history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Existing adapter-research directory')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    install(Path(__file__).resolve().parents[1],args.root,args.apply)


if __name__=='__main__':main()
