"""Ensure corrupted evidence fails closed without GPU dependencies."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repro', ROOT / 'scripts/verify_reproduction_5090.py')
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class ReproductionIntegrityTests(unittest.TestCase):
    def test_tracked_evidence(self):
        self.assertFalse(repro.validate()['gpu_rerun'])

    def test_corrupt_evidence_is_rejected(self):
        mutations = [
            ('detector-replay.json', lambda d: d.update(checkpoint_sha256='0' * 64), 'Checkpoint identity'),
            ('taxonomy-summary.json', lambda d: d['bucket_counts'].update(SUCCESS=483), 'Taxonomy GT total'),
            ('a9-replay.json', lambda d: d['recorded_metric_comparison'].update(per_instance_exact_claim=True), 'raw-pose identity'),
            ('summary.json', lambda d: d['performance'].update(decision='ACCEPTED'), 'Rejected variant'),
        ]
        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(ROOT / 'reports/reproduction_5090', root / 'reports/reproduction_5090')
                required = ['release/v1.1.0/results.json', 'protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json']
                lock = json.loads((root / 'reports/reproduction_5090/a9-replay.json').read_text())['run_lock']
                for relative in set(required) | set(lock['implementation']['files_sha256']):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, target)
                target = root / 'reports/reproduction_5090' / name
                data = json.loads(target.read_text())
                mutate(data)
                target.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, message):
                    repro.validate(root)

    def test_wrong_external_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.pt'
            path.write_bytes(b'not-the-frozen-model')
            with self.assertRaisesRegex(ValueError, 'External checkpoint SHA-256'):
                repro.validate(checkpoint=path)


if __name__ == '__main__':
    unittest.main()
