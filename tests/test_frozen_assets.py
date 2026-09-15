"""CPU-only boundary checks; fixtures are not model-validation evidence."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_assets", ROOT / "scripts/verify_frozen_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class FrozenAssetTests(unittest.TestCase):
    def test_missing_and_corrupted_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            expected = {"size_bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
            self.assertEqual(assets.inspect_file(path, expected)["status"], "MISSING_OR_NOT_FILE")
            path.write_bytes(b"short")
            self.assertEqual(assets.inspect_file(path, expected)["status"], "SIZE_MISMATCH")
            path.write_bytes(b"evil")
            self.assertEqual(assets.inspect_file(path, expected)["status"], "SHA256_MISMATCH")
            path.write_bytes(b"good")
            self.assertEqual(assets.inspect_file(path, expected)["status"], "VERIFIED")

    def test_catalog_matches_frozen_protocol(self):
        catalog = json.loads(assets.CATALOG.read_text())
        protocol = json.loads((ROOT / "protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json").read_text())
        self.assertEqual(catalog["detector"]["sha256"], protocol["upstream_detector"]["checkpoint_sha256"])
        self.assertEqual(catalog["detector"]["dataset_manifest_sha256"], protocol["dataset"]["a9_dataset_manifest_sha256"])
        self.assertEqual(assets.sha256_file(ROOT / catalog["detector"]["protocol_path"]), catalog["detector"]["protocol_sha256"])
        for role in ("refiner", "scorer"):
            self.assertEqual(catalog["foundationpose"]["assets"][role + "_checkpoint"], protocol["foundationpose"][role + "_checkpoint"])
            self.assertEqual(catalog["foundationpose"]["assets"][role + "_config"]["sha256"], protocol["foundationpose"][role + "_config_sha256"])
        self.assertIsNone(catalog["detector"]["download_url"])

    def test_no_assets_blocks_without_loading_and_receipt_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            with patch.object(assets, "load_detector") as load, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(assets.main(["--receipt", str(receipt)]), 3)
                load.assert_not_called()
            before = receipt.read_bytes()
            result = json.loads(before)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertFalse(result["full_pipeline_reproduced"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                assets.main(["--receipt", str(receipt)])
            self.assertEqual(receipt.read_bytes(), before)

    def test_wrong_detector_never_reaches_model_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(assets, "load_detector") as load, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(assets.main(["--detector-checkpoint", str(Path(directory) / "absent.pt"),
                                             "--dataset-manifest", str(Path(directory) / "unopened.json"),
                                             "--load-detector"]), 3)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
