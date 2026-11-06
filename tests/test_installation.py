import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('installer',Path(__file__).parents[1]/'tools/install_into_repo.py')
installer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(installer)


def snapshot(root):
    return {path.relative_to(root).as_posix():path.read_bytes() for path in root.rglob('*') if path.is_file()}


class InstallationTests(unittest.TestCase):
    def fixture(self,path):
        bundle=path/'bundle';repo=path/'repo';bundle.mkdir();repo.mkdir()
        files={'modal_pipeline.py':b'new entry','adapter_research/__init__.py':b'new package',
               'tests/test_new.py':b'new test','README.md':b'new guide','.gitignore':b'new ignore'}
        for name,data in files.items():
            target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (bundle/'INSTALL_MANIFEST.json').write_text(json.dumps({'files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}))
        originals={'modal_app.py':b'old entry','fixednorm_runner/projection.py':b'old math',
                   'tests/test_hellaswag.py':b'old HS test','README.md':b'old guide',
                   '.gitignore':b'user ignore','.git/config':b'user git metadata',
                   'reports/arcc/report.json':b'user results','run_receipts/id.json':b'user receipt'}
        for name,data in originals.items():
            target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        return bundle,repo,originals

    def test_preview_and_install_preserve_originals_git_results_and_ignore(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle,repo,originals=self.fixture(Path(temporary))
            before=snapshot(repo);installer.install(bundle,repo);self.assertEqual(snapshot(repo),before)
            history=installer.install(bundle,repo,apply=True)
            self.assertFalse((repo/'modal_app.py').exists())
            self.assertFalse((repo/'fixednorm_runner').exists())
            self.assertTrue((repo/'adapter_research/__init__.py').is_file())
            self.assertFalse((repo/'tests/test_hellaswag.py').exists())
            for name in ['.git/config','reports/arcc/report.json','run_receipts/id.json','.gitignore']:
                self.assertEqual((repo/name).read_bytes(),originals[name])
            for name in ['modal_app.py','fixednorm_runner/projection.py','tests/test_hellaswag.py','README.md']:
                self.assertEqual((history/'original'/name).read_bytes(),originals[name])

    def test_damaged_bundle_refused_before_repo_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle,repo,_=self.fixture(Path(temporary));before=snapshot(repo)
            (bundle/'modal_pipeline.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):installer.install(bundle,repo,apply=True)
            self.assertEqual(snapshot(repo),before)


if __name__=='__main__':unittest.main()
