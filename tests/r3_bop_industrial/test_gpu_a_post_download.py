from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from r3_bop_industrial.gpu_a_post_download import (
    StageError,
    extract_normalized_dataset,
    validate_public_dataset,
    verify_archive,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class GpuAPostDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_verify_archive_binds_size_and_sha256(self) -> None:
        archive = self.root / "xyzibd_val.zip.part"
        payload = b"fixed archive bytes"
        archive.write_bytes(payload)
        contract = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

        audit = verify_archive(archive, contract)

        self.assertIs(audit["verified"], True)
        self.assertEqual(audit["bytes"], len(payload))
        self.assertEqual(audit["sha256"], contract["sha256"])

    def test_verify_archive_rejects_wrong_hash(self) -> None:
        archive = self.root / "xyzibd_val.zip.part"
        archive.write_bytes(b"bytes")
        with self.assertRaisesRegex(StageError, "SHA-256 mismatch"):
            verify_archive(archive, {"bytes": 5, "sha256": "0" * 64})

    def test_public_validation_never_enumerates_val(self) -> None:
        root = self.root / "xyzibd"
        (root / "val").mkdir(parents=True)
        _write_json(
            root / "test_targets_bop19.json",
            [{"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 1}],
        )
        _write_json(
            root / "test_targets_bop24.json", [{"scene_id": 1, "im_id": 2}]
        )
        _write_json(root / "models_eval" / "models_info.json", {"1": {}})

        audit = validate_public_dataset(root)

        self.assertEqual(audit["val"], {"exists": True, "enumerated": False})
        self.assertEqual(audit["evaluator_only_label_paths_accessed"], [])
        self.assertEqual(audit["test_targets_bop19.json"]["row_count"], 1)
        self.assertEqual(audit["test_targets_bop24.json"]["row_count"], 1)

    def test_public_validation_rejects_target_schema_drift(self) -> None:
        root = self.root / "xyzibd"
        (root / "val").mkdir(parents=True)
        _write_json(
            root / "test_targets_bop19.json",
            [
                {
                    "scene_id": 1,
                    "im_id": 2,
                    "obj_id": 3,
                    "inst_count": 1,
                    "gt_id": 0,
                }
            ],
        )
        _write_json(
            root / "test_targets_bop24.json", [{"scene_id": 1, "im_id": 2}]
        )
        _write_json(root / "models_eval" / "models_info.json", {"1": {}})

        with self.assertRaisesRegex(StageError, "fields differ"):
            validate_public_dataset(root)

    @unittest.skipUnless(shutil.which("unzip"), "unzip is required")
    def test_extract_normalizes_only_documented_archive_roots(self) -> None:
        archives = self.root / "archives"
        archives.mkdir()
        base = archives / "xyzibd_base.zip"
        models = archives / "xyzibd_models.zip"
        val = archives / "xyzibd_val.zip"
        with zipfile.ZipFile(base, "w") as handle:
            handle.writestr(
                "xyzibd/test_targets_bop19.json",
                json.dumps(
                    [{"scene_id": 1, "im_id": 2, "obj_id": 3, "inst_count": 1}]
                ),
            )
            handle.writestr(
                "xyzibd/test_targets_bop24.json",
                json.dumps([{"scene_id": 1, "im_id": 2}]),
            )
        with zipfile.ZipFile(models, "w") as handle:
            handle.writestr("models/obj_000001.ply", "synthetic")
            handle.writestr("models_eval/models_info.json", json.dumps({"1": {}}))
        with zipfile.ZipFile(val, "w") as handle:
            handle.writestr("xyzibd_val/val/synthetic_public_file.txt", "synthetic")

        dataset_root, extraction = extract_normalized_dataset(
            {
                "xyzibd_base.zip": base,
                "xyzibd_models.zip": models,
                "xyzibd_val.zip": val,
            },
            self.root / "data",
        )

        self.assertEqual(dataset_root.name, "xyzibd")
        self.assertIs(extraction["reused"], False)
        self.assertEqual(extraction["public"]["val"]["enumerated"], False)
        self.assertTrue((dataset_root / "models" / "obj_000001.ply").is_file())


if __name__ == "__main__":
    unittest.main()
