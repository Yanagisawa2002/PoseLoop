from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from r3_bop_industrial import core as source_core
from r3_bop_industrial_supported.core import (
    ContractError,
    build_supported_commands,
    canonical_sha256,
    create_final_prescore_receipt,
    create_label_access_authorization,
    execute_official_evaluation,
    freeze_input_lock,
    load_protocol,
    validate_label_access_authorization,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "protocols"
    / "poseloop_r3_bop_industrial_xyzibd_supported_protocol.json"
)
SOURCE_PROTOCOL_PATH = (
    REPO_ROOT / "protocols" / "poseloop_r3_bop_industrial_protocol.json"
)


class SupportedR3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(PROTOCOL_PATH)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_support_matrix_keeps_bop19_unavailable(self) -> None:
        bop19 = self.protocol["support_matrix"]["official_bop19_average_recall"]
        self.assertEqual(bop19["status"], "unavailable")
        self.assertTrue(bop19["must_not_invoke"])
        self.assertEqual(bop19["required_error_types"], ["vsd", "mssd", "mspd"])
        self.assertEqual(bop19["missing_xyzibd_definition"], "vsd")
        self.assertNotIn("vsd", self.protocol["toolkit"]["xyzibd_supported_error_types"])

    def test_supported_command_list_has_four_commands_and_no_bop19(self) -> None:
        source_protocol = source_core.load_protocol(SOURCE_PROTOCOL_PATH)
        commands = build_supported_commands(
            self.root / "inputs",
            self.root / "toolkit",
            self.root / "artifacts" / "r3_bop_industrial_supported" / "official_once",
            source_protocol,
            self.protocol,
            "python3",
        )
        self.assertEqual(len(commands), 4)
        self.assertEqual(
            [command["command_id"] for command in commands],
            self.protocol["evaluation"]["command_order"],
        )
        self.assertFalse(
            any(
                "eval_bop19_pose.py" in argument
                for command in commands
                for argument in command["argv"]
            )
        )
        self.assertEqual(commands[0]["expected_score_key"], "bop24_mAP")
        self.assertEqual(commands[2]["expected_score_key"], "AP")

    def _fake_preflight(self, *, fail_first: bool = False) -> tuple[dict, Path, Path, Path]:
        output_root = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported"
            / "official_once"
        )
        input_root = self.root / "inputs"
        input_root.mkdir()
        helper = self.root / "fake_official.py"
        helper.write_text(
            """
import json
import pathlib
import sys

if sys.argv[1] == "fail":
    raise SystemExit(7)
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(json.loads(sys.argv[2])), encoding="utf-8")
""".lstrip(),
            encoding="utf-8",
        )
        definitions = [
            (
                "official_bop24_pose_map:single_view",
                "official_bop24_pose_map",
                "single_view",
                output_root / "eval" / "single" / "scores_bop24.json",
                {"bop24_mAP": 0.20, "bop24_mAP_mssd": 0.18, "bop24_mAP_mspd": 0.22},
                "bop24_mAP",
            ),
            (
                "official_bop24_pose_map:multi_view",
                "official_bop24_pose_map",
                "multi_view",
                output_root / "eval" / "multi" / "scores_bop24.json",
                {"bop24_mAP": 0.35, "bop24_mAP_mssd": 0.30, "bop24_mAP_mspd": 0.40},
                "bop24_mAP",
            ),
            (
                "official_bop22_bbox_ap:shared_predicted_input",
                "official_bop22_bbox_ap",
                "shared_predicted_input",
                output_root / "eval" / "coco" / "scores_bop22_coco_bbox.json",
                {"AP": 0.40, "AP50": 0.60},
                "AP",
            ),
            (
                "official_bop22_segm_ap:shared_predicted_input",
                "official_bop22_segm_ap",
                "shared_predicted_input",
                output_root / "eval" / "coco" / "scores_bop22_coco_segm.json",
                {"AP": 0.30, "AP50": 0.50},
                "AP",
            ),
        ]
        commands = []
        for index, (command_id, metric, variant, score_path, payload, score_key) in enumerate(
            definitions
        ):
            argv = (
                [sys.executable, str(helper), "fail"]
                if fail_first and index == 0
                else [sys.executable, str(helper), str(score_path), json.dumps(payload)]
            )
            commands.append(
                {
                    "command_id": command_id,
                    "metric": metric,
                    "variant": variant,
                    "argv": argv,
                    "expected_score": str(score_path),
                    "expected_score_key": score_key,
                }
            )
        preflight = {
            "status": "ready",
            "protocol_id": self.protocol["protocol_id"],
            "fingerprint_sha256": "a" * 64,
            "official_commands": commands,
            "official_command_manifest_sha256": canonical_sha256(commands),
            "source_audit": {"prediction_bundle": {"root": str(input_root)}},
            "official_evaluation_executed": False,
            "orchestrator_source_audit": {"manifest_sha256": "c" * 64},
            "support_audit": {"official_bop19_entrypoint_command_count": 0},
            "unavailable_metrics": {
                "official_bop19_average_recall": {
                    "status": "unavailable",
                }
            },
            "label_access_audit": {
                "prediction_and_selection_label_access_count": 0,
                "evaluator_only_label_paths_accessed": [],
                "official_evaluator_label_access_started": False,
            },
        }
        lock = freeze_input_lock(preflight)
        pre_score = (
            self.root / "artifacts" / "r3_bop_industrial_supported" / "pre_score"
        )
        lock_path = pre_score / "input-lock.json"
        write_json_atomic(lock_path, lock)
        authorization = create_label_access_authorization(
            input_lock_path=lock_path,
            approved_by="unit-test-user",
            authorization_reference="unit-test-authorization",
        )
        authorization_path = pre_score / "authorization.json"
        write_json_atomic(authorization_path, authorization)
        dataset_root = self.root / "datasets" / "xyzibd"
        dataset_root.mkdir(parents=True)
        return preflight, lock_path, authorization_path, dataset_root

    def test_final_prescore_is_read_only_and_hash_bound(self) -> None:
        preflight, lock_path, authorization_path, _ = self._fake_preflight()
        output_root = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported"
            / "official_once"
        )
        receipt = create_final_prescore_receipt(
            current_preflight=preflight,
            input_lock_path=lock_path,
            authorization_path=authorization_path,
            output_root=output_root,
        )
        self.assertEqual(receipt["status"], "ready")
        self.assertTrue(receipt["official_output_root_absent"])
        self.assertFalse(receipt["official_evaluation_executed"])
        self.assertEqual(receipt["official_bop19_entrypoint_command_count"], 0)

    def test_one_evaluate_call_saves_scores_delta_and_sealed_receipt(self) -> None:
        preflight, lock_path, authorization_path, dataset_root = self._fake_preflight()
        output_root = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported"
            / "official_once"
        )
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=lock_path,
            authorization_path=authorization_path,
            dataset_root=dataset_root,
            output_root=output_root,
            repo_root=self.root,
            protocol=self.protocol,
        )
        self.assertEqual(result["status"], "complete")
        pose = result["metrics"]["official_bop24_pose_map"]
        self.assertAlmostEqual(pose["single_view"], 0.20)
        self.assertAlmostEqual(pose["multi_view"], 0.35)
        self.assertAlmostEqual(pose["multi_minus_single"], 0.15)
        self.assertEqual(
            result["metrics"]["official_bop19_average_recall"]["status"],
            "unavailable",
        )
        sealed = json.loads((output_root / "sealed_receipt.json").read_text())
        self.assertEqual(sealed["status"], "complete")
        self.assertEqual(sealed["all_exit_codes"], [0, 0, 0, 0])
        self.assertFalse(sealed["rerun_permitted"])
        with self.assertRaisesRegex(ContractError, "rerun is permanently refused"):
            execute_official_evaluation(
                current_preflight=preflight,
                input_lock_path=lock_path,
                authorization_path=authorization_path,
                dataset_root=dataset_root,
                output_root=output_root,
                repo_root=self.root,
                protocol=self.protocol,
            )

    def test_failure_is_sealed_and_cannot_be_retried(self) -> None:
        preflight, lock_path, authorization_path, dataset_root = self._fake_preflight(
            fail_first=True
        )
        output_root = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported"
            / "official_once"
        )
        with self.assertRaisesRegex(RuntimeError, "command 1/4"):
            execute_official_evaluation(
                current_preflight=preflight,
                input_lock_path=lock_path,
                authorization_path=authorization_path,
                dataset_root=dataset_root,
                output_root=output_root,
                repo_root=self.root,
                protocol=self.protocol,
            )
        sealed = json.loads((output_root / "sealed_receipt.json").read_text())
        self.assertEqual(sealed["status"], "failed_after_evaluate_started")
        self.assertEqual(sealed["executions"][0]["exit_code"], 7)
        self.assertFalse(sealed["rerun_permitted"])
        with self.assertRaisesRegex(ContractError, "rerun is permanently refused"):
            execute_official_evaluation(
                current_preflight=preflight,
                input_lock_path=lock_path,
                authorization_path=authorization_path,
                dataset_root=dataset_root,
                output_root=output_root,
                repo_root=self.root,
                protocol=self.protocol,
            )

    def test_authorization_is_bound_to_lock_and_command_manifest(self) -> None:
        preflight, lock_path, authorization_path, _ = self._fake_preflight()
        lock = json.loads(lock_path.read_text())
        validate_label_access_authorization(authorization_path, lock_path, lock)
        authorization = json.loads(authorization_path.read_text())
        authorization["official_command_manifest_sha256"] = "b" * 64
        write_json_atomic(authorization_path, authorization)
        with self.assertRaisesRegex(ContractError, "command manifest"):
            validate_label_access_authorization(authorization_path, lock_path, lock)
        self.assertEqual(
            preflight["official_command_manifest_sha256"],
            canonical_sha256(preflight["official_commands"]),
        )

    def test_imported_gpu_a_failure_evidence_is_non_rerunnable(self) -> None:
        evidence_root = (
            REPO_ROOT
            / "artifacts"
            / "r3_bop_industrial_supported"
            / "remote_gpu_a"
        )
        archive = evidence_root / "r3_xyzibd_supported_once_evidence_6d9756e.tar.gz"
        self.assertEqual(
            source_core.sha256_file(archive),
            "2e2d63b589004406fdabf4988ff9ec353dbc44c37e77a4e31ff1bf381f0a51ce",
        )
        run_root = (
            evidence_root
            / "extracted"
            / "workspace"
            / "artifacts"
            / "r3_bop_industrial_supported"
        )
        sealed = json.loads(
            (run_root / "official_once" / "sealed_receipt.json").read_text()
        )
        self.assertEqual(sealed["status"], "failed_after_evaluate_started")
        self.assertEqual(sealed["evaluate_invocation_count"], 1)
        self.assertEqual(sealed["executions"][0]["exit_code"], 1)
        self.assertFalse(sealed["rerun_permitted"])
        self.assertEqual(
            list((run_root / "official_once").rglob("scores*.json")), []
        )
        raw_log = (
            run_root
            / "official_once"
            / "logs"
            / "01_official_bop24_pose_map_single_view.log"
        ).read_text()
        self.assertIn("ModuleNotFoundError: No module named 'bop_toolkit_lib'", raw_log)
        final_prescore = json.loads(
            (run_root / "pre_score" / "final-prescore.json").read_text()
        )
        self.assertEqual(final_prescore["status"], "ready")
        self.assertEqual(final_prescore["official_bop19_entrypoint_command_count"], 0)
        self.assertEqual(
            final_prescore["label_access_audit"][
                "prediction_and_selection_label_access_count"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
