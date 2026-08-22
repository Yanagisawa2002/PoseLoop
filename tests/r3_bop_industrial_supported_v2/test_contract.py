from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from r3_bop_industrial import core as source_core
from r3_bop_industrial_supported import core as v1_core
from r3_bop_industrial_supported_v2.core import (
    ContractError,
    _SMOKE_PROGRAM,
    _resolve_executable,
    canonical_sha256,
    create_evaluator_authorization,
    create_final_prescore_receipt,
    execute_official_evaluation,
    freeze_input_lock,
    load_protocol,
    validate_evaluator_authorization,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "protocols"
    / "poseloop_r3_bop_industrial_xyzibd_supported_v2_protocol.json"
)
SOURCE_PROTOCOL_PATH = (
    REPO_ROOT / "protocols" / "poseloop_r3_bop_industrial_protocol.json"
)


class SupportedV2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(PROTOCOL_PATH)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_protocol_keeps_bop19_unavailable_and_freezes_four_commands(self) -> None:
        bop19 = self.protocol["support_matrix"]["official_bop19_average_recall"]
        self.assertEqual(bop19["status"], "unavailable")
        self.assertTrue(bop19["must_not_invoke"])
        self.assertEqual(bop19["required_error_types"], ["vsd", "mssd", "mspd"])
        self.assertEqual(bop19["missing_xyzibd_definition"], "vsd")
        self.assertTrue(self.protocol["evaluation"]["continue_after_command_failure"])
        self.assertEqual(self.protocol["evaluation"]["maximum_invocations_per_command"], 1)

        commands = v1_core.build_supported_commands(
            self.root / "inputs",
            self.root / "toolkit",
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported_v2"
            / "official_once",
            source_core.load_protocol(SOURCE_PROTOCOL_PATH),
            self.protocol,
            sys.executable,
        )
        self.assertEqual([item["command_id"] for item in commands], self.protocol["evaluation"]["command_order"])
        self.assertFalse(
            any(
                "eval_bop19_pose.py" in argument
                for command in commands
                for argument in command["argv"]
            )
        )

    def test_venv_launcher_path_is_not_dereferenced(self) -> None:
        expected = Path(sys.executable).absolute()
        with mock.patch.object(
            Path,
            "resolve",
            side_effect=AssertionError("launcher symlink must not be resolved"),
        ):
            self.assertEqual(_resolve_executable(sys.executable), expected)

        target = self.root / "base-python"
        target.write_text("placeholder", encoding="utf-8")
        launcher = self.root / "venv-python"
        try:
            launcher.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.assertEqual(_resolve_executable(str(launcher)), launcher.absolute())

    def test_renderer_smoke_uses_pinned_toolkit_rendering_namespace(self) -> None:
        self.assertIn("from bop_toolkit_lib.rendering import renderer", _SMOKE_PROGRAM)
        self.assertNotIn("from bop_toolkit_lib import renderer", _SMOKE_PROGRAM)
        self.assertIn('distribution.read_text("direct_url.json")', _SMOKE_PROGRAM)
        self.assertIn('metadata.distributions(name="bop_toolkit_lib")', _SMOKE_PROGRAM)
        self.assertIn('"editable_match_count": len(editable_matches)', _SMOKE_PROGRAM)
        self.assertIn('"venv_active": sys.prefix != sys.base_prefix', _SMOKE_PROGRAM)
        self.assertNotIn("pathlib.Path(sys.executable).resolve()", _SMOKE_PROGRAM)

    def _frozen_bundle(self, *, fail_first: bool) -> tuple[dict, Path, Path, Path, Path, Path]:
        output_root = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported_v2"
            / "official_once"
        )
        input_root = self.root / "inputs"
        input_root.mkdir()
        toolkit_root = self.root / "toolkit"
        toolkit_root.mkdir()
        dataset_root = self.root / "datasets" / "xyzibd"
        dataset_root.mkdir(parents=True)
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
                {"bop24_mAP": 0.20},
                "bop24_mAP",
            ),
            (
                "official_bop24_pose_map:multi_view",
                "official_bop24_pose_map",
                "multi_view",
                output_root / "eval" / "multi" / "scores_bop24.json",
                {"bop24_mAP": 0.35},
                "bop24_mAP",
            ),
            (
                "official_bop22_bbox_ap:shared_predicted_input",
                "official_bop22_bbox_ap",
                "shared_predicted_input",
                output_root / "eval" / "coco" / "scores_bop22_coco_bbox.json",
                {"AP": 0.40},
                "AP",
            ),
            (
                "official_bop22_segm_ap:shared_predicted_input",
                "official_bop22_segm_ap",
                "shared_predicted_input",
                output_root / "eval" / "coco" / "scores_bop22_coco_segm.json",
                {"AP": 0.30},
                "AP",
            ),
        ]
        commands = []
        for index, (command_id, metric, variant, score_path, payload, key) in enumerate(definitions):
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
                    "expected_score_key": key,
                }
            )
        preflight = {
            "status": "ready",
            "protocol_id": self.protocol["protocol_id"],
            "protocol": {"sha256": "1" * 64},
            "implementation_commit": "2" * 40,
            "fingerprint": {"environment_smoke_receipt_sha256": "3" * 64},
            "fingerprint_sha256": "4" * 64,
            "official_commands": commands,
            "official_command_manifest_sha256": canonical_sha256(commands),
            "source_audit": {"prediction_bundle": {"root": str(input_root)}},
            "environment_smoke_audit": {"sha256": "3" * 64},
            "official_evaluation_executed": False,
            "orchestrator_source_audit": {"manifest_sha256": "5" * 64},
            "support_audit": {"official_bop19_entrypoint_command_count": 0},
            "score_file_audit": {"count": 0},
            "official_process_audit": {"count": 0},
            "unavailable_metrics": {
                "official_bop19_average_recall": {"status": "unavailable"}
            },
            "label_access_audit": {
                "prediction_and_selection_label_access_count": 0,
                "evaluator_only_label_paths_accessed": [],
                "official_evaluator_label_access_started": False,
            },
        }
        lock = freeze_input_lock(preflight)
        pre_score = (
            self.root
            / "artifacts"
            / "r3_bop_industrial_supported_v2"
            / "pre_score"
        )
        lock_path = pre_score / "input-lock.json"
        write_json_atomic(lock_path, lock)
        authorization = create_evaluator_authorization(
            input_lock_path=lock_path,
            protocol=self.protocol,
            approved_by="unit-test-user",
            authorization_reference=self.protocol["authorization"]["reference"],
        )
        authorization_path = pre_score / "evaluator-authorization.json"
        write_json_atomic(authorization_path, authorization)
        return preflight, lock_path, authorization_path, dataset_root, toolkit_root, output_root

    def test_final_prescore_is_hash_bound_and_read_only(self) -> None:
        preflight, lock_path, authorization_path, _, _, output_root = self._frozen_bundle(fail_first=False)
        receipt = create_final_prescore_receipt(
            current_preflight=preflight,
            input_lock_path=lock_path,
            authorization_path=authorization_path,
            output_root=output_root,
            protocol=self.protocol,
        )
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(receipt["score_file_count"], 0)
        self.assertEqual(receipt["official_process_count"], 0)
        self.assertFalse(receipt["official_evaluation_executed"])

    def test_failure_does_not_skip_remaining_fixed_commands_and_seals(self) -> None:
        preflight, lock_path, authorization_path, dataset_root, toolkit_root, output_root = self._frozen_bundle(fail_first=True)
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=lock_path,
            authorization_path=authorization_path,
            dataset_root=dataset_root,
            toolkit_root=toolkit_root,
            output_root=output_root,
            repo_root=self.root,
            protocol=self.protocol,
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual([item["exit_code"] for item in result["executions"]], [7, 0, 0, 0])
        self.assertEqual([item["attempt_count"] for item in result["executions"]], [1, 1, 1, 1])
        self.assertEqual(result["metrics"]["official_bop24_pose_map"]["multi_view"], 0.35)
        self.assertEqual(result["metrics"]["official_bop22_bbox_ap"]["shared_predicted_input"], 0.40)
        sealed = json.loads((output_root / "sealed_receipt.json").read_text())
        self.assertEqual(sealed["attempted_command_count"], 4)
        self.assertEqual(sealed["all_exit_codes"], [7, 0, 0, 0])
        self.assertFalse(sealed["rerun_permitted"])
        with self.assertRaisesRegex(ContractError, "rerun is permanently refused"):
            execute_official_evaluation(
                current_preflight=preflight,
                input_lock_path=lock_path,
                authorization_path=authorization_path,
                dataset_root=dataset_root,
                toolkit_root=toolkit_root,
                output_root=output_root,
                repo_root=self.root,
                protocol=self.protocol,
            )

    def test_complete_run_saves_single_multi_delta_and_four_exit_codes(self) -> None:
        preflight, lock_path, authorization_path, dataset_root, toolkit_root, output_root = self._frozen_bundle(fail_first=False)
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=lock_path,
            authorization_path=authorization_path,
            dataset_root=dataset_root,
            toolkit_root=toolkit_root,
            output_root=output_root,
            repo_root=self.root,
            protocol=self.protocol,
        )
        self.assertEqual(result["status"], "complete")
        pose = result["metrics"]["official_bop24_pose_map"]
        self.assertAlmostEqual(pose["multi_minus_single"], 0.15)
        self.assertEqual(result["metrics"]["official_bop19_average_recall"]["status"], "unavailable")
        sealed = json.loads((output_root / "sealed_receipt.json").read_text())
        self.assertEqual(sealed["all_exit_codes"], [0, 0, 0, 0])
        self.assertEqual(set(sealed["command_invocation_counts"].values()), {1})

    def test_authorization_binds_lock_protocol_commit_environment_and_commands(self) -> None:
        preflight, lock_path, authorization_path, _, _, _ = self._frozen_bundle(fail_first=False)
        lock = json.loads(lock_path.read_text())
        validate_evaluator_authorization(
            authorization_path, lock_path, lock, self.protocol
        )
        value = json.loads(authorization_path.read_text())
        value["environment_smoke_receipt_sha256"] = "f" * 64
        write_json_atomic(authorization_path, value)
        with self.assertRaisesRegex(ContractError, "environment_smoke_receipt_sha256"):
            validate_evaluator_authorization(
                authorization_path, lock_path, lock, self.protocol
            )
        self.assertEqual(
            preflight["official_command_manifest_sha256"],
            canonical_sha256(preflight["official_commands"]),
        )

    def test_imported_gpu_a_v2_failure_evidence_is_sealed_and_non_rerunnable(self) -> None:
        evidence_root = (
            REPO_ROOT
            / "artifacts"
            / "r3_bop_industrial_supported_v2"
            / "remote_gpu_a"
        )
        archive = evidence_root / "r3_xyzibd_supported_v2_once_1278f69.tar.gz"
        self.assertEqual(
            source_core.sha256_file(archive),
            "f50e19c01c239c2fd5c9095a55f901a5dbfad3623f5663d84e25ebc90fa8b79a",
        )
        run_root = (
            evidence_root
            / "extracted"
            / "workspace"
            / "artifacts"
            / "r3_bop_industrial_supported_v2"
        )
        pre_score = run_root / "pre_score"
        official = run_root / "official_once"
        final_prescore = json.loads((pre_score / "final-prescore.json").read_text())
        self.assertEqual(final_prescore["status"], "ready")
        self.assertEqual(final_prescore["implementation_commit"], "1278f690d9714e213ff5dd7ab23a0ae67c16c67e")
        self.assertEqual(final_prescore["score_file_count"], 0)
        self.assertEqual(final_prescore["official_process_count"], 0)
        self.assertEqual(final_prescore["official_bop19_entrypoint_command_count"], 0)
        self.assertEqual(
            final_prescore["label_access_audit"]["prediction_and_selection_label_access_count"],
            0,
        )

        smoke = json.loads((pre_score / "environment-smoke.json").read_text())
        self.assertEqual(smoke["status"], "ready")
        self.assertEqual(smoke["official_evaluator_invocation_count"], 0)
        self.assertEqual(smoke["child_payload"]["editable_match_count"], 1)
        self.assertEqual(smoke["child_payload"]["renderer"]["status"], "ready")

        sealed = json.loads((official / "sealed_receipt.json").read_text())
        self.assertEqual(sealed["result_status"], "failed")
        self.assertEqual(sealed["evaluate_invocation_count"], 1)
        self.assertEqual(sealed["attempted_command_count"], 4)
        self.assertEqual(sealed["all_exit_codes"], [1, 1, 1, 1])
        self.assertEqual(set(sealed["command_invocation_counts"].values()), {1})
        self.assertTrue(sealed["continued_after_failures"])
        self.assertFalse(sealed["rerun_permitted"])

        result = json.loads((official / "official_scores.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["metrics"]["official_bop19_average_recall"]["status"],
            "unavailable",
        )
        self.assertIsNone(
            result["metrics"]["official_bop24_pose_map"]["multi_minus_single"]
        )
        self.assertEqual(list((official / "eval").rglob("scores*.json")), [])
        single_log = (
            official / "logs" / "01_official_bop24_pose_map_single_view.log"
        ).read_text()
        bbox_log = (
            official / "logs" / "03_official_bop22_bbox_ap_shared_predicted_input.log"
        ).read_text()
        self.assertIn("scene_gt_xyz.json", single_log)
        self.assertIn("scene_gt_coco_xyz.json", bbox_log)


if __name__ == "__main__":
    unittest.main()
