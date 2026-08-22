from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r3_bop_industrial_supported_v3 import core as v3_core
from r3_bop_industrial_supported_v3.core import (
    ContractError,
    build_official_commands,
    create_evaluator_authorization,
    freeze_input_lock,
    load_protocol,
)
from r3_bop_industrial_supported_v3_readiness.core import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_r3_bop_industrial_xyzibd_supported_v3_protocol.json"


class SupportedV3ContractTests(unittest.TestCase):
    def test_protocol_keeps_bop19_unavailable_and_v2_immutable(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["support_matrix"]["official_bop19_average_recall"]["status"], "unavailable")
        self.assertTrue(protocol["support_matrix"]["official_bop19_average_recall"]["must_not_invoke"])
        self.assertFalse(protocol["immutable_v2_failure"]["rerun_permitted"])
        self.assertEqual(protocol["immutable_v2_failure"]["all_four_exit_codes"], [1, 1, 1, 1])

    def test_commands_use_validation_targets_and_never_construct_bop19(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        commands = build_official_commands(
            protocol=protocol,
            python_executable="python",
            toolkit_root=Path("/toolkit"),
            input_root=Path("/inputs"),
            output_root=Path("/official_once"),
        )
        self.assertEqual([item["command_id"] for item in commands], protocol["evaluation"]["command_order"])
        joined = "\n".join(" ".join(item["argv"]) for item in commands)
        self.assertEqual(len(commands), 4)
        self.assertNotIn("eval_bop19_pose.py", joined)
        self.assertEqual(joined.count("--targets_filename=val_targets_bop24.json"), 2)
        self.assertEqual(joined.count("--targets_filename=val_targets_bop19.json"), 2)

    def test_blocked_preflight_cannot_freeze(self) -> None:
        with self.assertRaises(ContractError):
            freeze_input_lock({"status": "blocked"})

    def test_authorization_binds_lock_commit_protocol_and_commands(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = {
                "schema_version": "poseloop.r3.xyzibd-supported.input-lock.v3",
                "protocol_id": protocol["protocol_id"],
                "fingerprint_sha256": "1" * 64,
                "repository": {"head": "2" * 40},
                "protocol": {"sha256": "3" * 64},
                "label_readiness": {"sha256": "4" * 64},
                "command_manifest_sha256": "5" * 64,
            }
            lock_path = root / "input-lock.json"
            write_json_atomic(lock_path, lock)
            receipt = create_evaluator_authorization(
                input_lock_path=lock_path,
                protocol=protocol,
                approved_by="user",
                authorization_reference=protocol["authorization"]["reference"],
            )
            self.assertEqual(receipt["input_lock_sha256"], sha256_file(lock_path))
            self.assertEqual(receipt["implementation_commit"], "2" * 40)
            self.assertEqual(receipt["authorized_command_count"], 4)
            self.assertFalse(receipt["rerun_after_score_or_error"])

    def test_wrong_authorization_reference_is_rejected(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            write_json_atomic(path, {"schema_version": "poseloop.r3.xyzibd-supported.input-lock.v3", "protocol_id": protocol["protocol_id"]})
            with self.assertRaises(ContractError):
                create_evaluator_authorization(
                    input_lock_path=path,
                    protocol=protocol,
                    approved_by="user",
                    authorization_reference="stale-authorization",
                )

    def test_command_failure_does_not_skip_remaining_commands_and_seals(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "official_once"
            commands = build_official_commands(
                protocol=protocol,
                python_executable="python",
                toolkit_root=root / "toolkit",
                input_root=root / "inputs",
                output_root=output,
            )
            preflight = {
                "status": "ready",
                "fingerprint_sha256": "1" * 64,
                "repository": {"head": "2" * 40},
                "label_readiness": {"sha256": "3" * 64},
                "command_manifest_sha256": "4" * 64,
                "official_commands": commands,
            }
            lock = {
                "schema_version": "poseloop.r3.xyzibd-supported.input-lock.v3",
                "protocol_id": protocol["protocol_id"],
                "state": "frozen_before_unique_supported_v3_evaluation",
                "fingerprint_sha256": preflight["fingerprint_sha256"],
                "repository": preflight["repository"],
                "protocol": {"sha256": "5" * 64},
                "label_readiness": preflight["label_readiness"],
                "command_manifest_sha256": preflight["command_manifest_sha256"],
            }
            lock_path = root / "input-lock.json"
            authorization_path = root / "authorization.json"
            write_json_atomic(lock_path, lock)
            write_json_atomic(
                authorization_path,
                create_evaluator_authorization(
                    input_lock_path=lock_path,
                    protocol=protocol,
                    approved_by="user",
                    authorization_reference=protocol["authorization"]["reference"],
                ),
            )
            returns = [
                v3_core.subprocess.CompletedProcess([], code, f"stdout-{index}", f"stderr-{index}")
                for index, code in enumerate((1, 0, 1, 0), start=1)
            ]
            with patch.object(v3_core.subprocess, "run", side_effect=returns) as mocked:
                result = v3_core.execute_official_evaluation(
                    current_preflight=preflight,
                    input_lock_path=lock_path,
                    authorization_path=authorization_path,
                    dataset_root=root / "datasets" / "xyzibd",
                    toolkit_root=root / "toolkit",
                    output_root=output,
                    protocol=protocol,
                )
            self.assertEqual(mocked.call_count, 4)
            self.assertEqual([item["exit_code"] for item in result["executions"]], [1, 0, 1, 0])
            self.assertEqual(result["attempted_command_count"], 4)
            sealed = json.loads((output / "sealed_receipt.json").read_text())
            self.assertEqual(sealed["all_exit_codes"], [1, 0, 1, 0])
            self.assertFalse(sealed["rerun_permitted"])


if __name__ == "__main__":
    unittest.main()
