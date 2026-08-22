from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from r3_bop_industrial_supported_v3_readiness.cli import parse_args
from r3_bop_industrial_supported_v3_readiness.core import (
    ContractError,
    audit_frozen_predictions,
    build_coco_generation_command,
    derive_validation_targets,
    freeze_readiness_lock,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_r3_bop_industrial_xyzibd_label_readiness_v3_protocol.json"


class ReadinessContractTests(unittest.TestCase):
    def test_protocol_is_readiness_only_and_v2_is_immutable(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["scoring"]["evaluator_invocations_allowed"], 0)
        self.assertFalse(protocol["scoring"]["authorization_receipt_allowed"])
        self.assertFalse(protocol["source_v2"]["rerun_permitted"])
        self.assertTrue(protocol["source_v2"]["immutable"])

    def test_cli_has_no_evaluate_or_authorize_command(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["evaluate"])
        with self.assertRaises(SystemExit):
            parse_args(["authorize"])

    def test_frozen_prediction_audit_hashes_without_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {"a.json": b"not-json", "b.csv": b"opaque"}
            hashes = {}
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            protocol = {"frozen_prediction_sha256": hashes}
            receipt = audit_frozen_predictions(root, protocol)
            self.assertTrue(receipt["ready"])
            self.assertFalse(receipt["prediction_files_parsed"])
            self.assertEqual(receipt["prediction_selection_label_access_count"], 0)

    def test_target_derivation_matches_bop_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "val" / "000000"
            (scene / "mask_xyz").mkdir(parents=True)
            (scene / "mask_visib_xyz").mkdir()
            gt = {"0": [{"obj_id": 2}, {"obj_id": 2}, {"obj_id": 3}]}
            info = {"0": [{"visib_fract": 0.8}, {"visib_fract": 0.05}, {"visib_fract": 0.9}]}
            camera = {"0": {"cam_K": [1] * 9}}
            for name, value in (
                ("scene_gt_xyz.json", gt),
                ("scene_gt_info_xyz.json", info),
                ("scene_camera_xyz.json", camera),
            ):
                (scene / name).write_text(json.dumps(value), encoding="utf-8")
            for index in range(3):
                (scene / "mask_xyz" / f"000000_{index:06d}.png").write_bytes(b"x")
                (scene / "mask_visib_xyz" / f"000000_{index:06d}.png").write_bytes(b"x")
            bop19, bop24, summary = derive_validation_targets(
                root, [0], images_per_scene=1, min_visibility=0.1
            )
            self.assertEqual(bop24, [{"scene_id": 0, "im_id": 0}])
            self.assertEqual(
                bop19,
                [
                    {"scene_id": 0, "im_id": 0, "obj_id": 2, "inst_count": 1},
                    {"scene_id": 0, "im_id": 0, "obj_id": 3, "inst_count": 1},
                ],
            )
            self.assertEqual(summary["mask_full_count"], 3)

    def test_coco_generation_command_is_not_a_scorer(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        command = build_coco_generation_command(
            python_executable="python",
            toolkit_root=Path("/toolkit"),
            datasets_root=Path("/evaluator/datasets"),
            protocol=protocol,
        )
        joined = " ".join(command)
        self.assertIn("calc_gt_coco.py", joined)
        self.assertIn("--dataset_split=val", joined)
        self.assertIn("--use_all_gt", joined)
        self.assertNotIn("eval_bop", joined)

    def test_blocked_receipt_cannot_be_frozen(self) -> None:
        with self.assertRaises(ContractError):
            freeze_readiness_lock({"status": "blocked"})


if __name__ == "__main__":
    unittest.main()
