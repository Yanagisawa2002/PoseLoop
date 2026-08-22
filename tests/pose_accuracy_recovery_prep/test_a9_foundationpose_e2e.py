from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_accuracy_recovery_prep.a9_foundationpose_e2e import runtime


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT / "protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json"
)


def _mutated_protocol(tmp_path: Path, *keys: str, value: object) -> Path:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cursor = payload
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_contract_accepts_the_frozen_development_protocol() -> None:
    protocol = runtime.load_protocol(PROTOCOL)
    assert protocol["dataset_role"] == "ALREADY_CONSUMED_REAL_DEVELOPMENT"
    assert protocol["scene9_replay_permitted"] is False
    assert protocol["failure_policy"]["oracle_diagnostic_run_limit"] == 1
    assert protocol["failure_policy"]["rescue_run_limit"] == 1


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("dataset", "scenes"), [9, 10, 25, 30, 40]),
        (("upstream_detector", "operating_score_threshold"), 0.2),
        (("foundationpose", "candidate_count"), 128),
        (("foundationpose", "iterations"), 3),
        (("failure_policy", "rescue_run_limit"), 2),
        (("scene9_replay_permitted",), True),
    ],
)
def test_contract_rejects_policy_relaxation(
    tmp_path: Path, keys: tuple[str, ...], value: object
) -> None:
    path = _mutated_protocol(tmp_path, *keys, value=value)
    with pytest.raises(runtime.ContractError):
        runtime.load_protocol(path)


def test_implementation_identity_closes_over_memory_bounded_runtime() -> None:
    identity = runtime._implementation_identity(PROTOCOL)
    assert identity["files_sha256"]["scripts/run_r1_sealed_inference.py"] == (
        runtime.EXPECTED_MEMORY_PATCH_SHA256
    )
    assert identity["files_sha256"]["scripts/run_xyzibd_batch.py"] == (
        runtime.EXPECTED_RUN_XYZIBD_SHA256
    )
    assert len(identity["identity_sha256"]) == 64


def test_repository_head_is_a_full_commit_identity() -> None:
    head = runtime._git_head(ROOT)
    assert len(head) == 40
    assert all(character in "0123456789abcdef" for character in head)


class _CudaLikeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.cpu_called = False

    def detach(self) -> _CudaLikeTensor:
        return self

    def cpu(self) -> _CudaLikeTensor:
        self.cpu_called = True
        return self

    def numpy(self) -> np.ndarray:
        assert self.cpu_called
        return self.value


def test_pose_serialization_moves_cuda_like_tensor_to_cpu() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 0.75
    tensor = _CudaLikeTensor(pose)
    assert np.asarray(runtime._pose_list(tensor)) == pytest.approx(pose)
    assert tensor.cpu_called


def test_pose_serialization_rejects_nonpositive_depth() -> None:
    with pytest.raises(runtime.ContractError, match=r"invalid SE\(3\)"):
        runtime._pose_list(np.eye(4, dtype=np.float64))


def test_standard_ap_is_deterministic_under_score_ties() -> None:
    labels = [False, True, True]
    scores = [0.8, 0.8, 0.5]
    assert runtime._standard_ap(
        labels, scores, total_gt=2, keys=["b", "a", "c"]
    ) == pytest.approx(5.0 / 6.0)


def test_existing_results_rejects_foreign_run_lock(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"record_type": "metadata", "run_lock_sha256": "b" * 64}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runtime.ContractError, match="metadata differs"):
        runtime._load_existing_results(path, "a" * 64, {"item-1"})


def test_package_rejects_no_go_result(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze"
    primary = tmp_path / "primary"
    evaluation = tmp_path / "evaluation"
    freeze.mkdir()
    primary.mkdir()
    evaluation.mkdir()
    (evaluation / "evaluation-result.json").write_text(
        json.dumps({"status": "NO_GO_ORACLE_DIAGNOSIS_PERMITTED_ONCE"}),
        encoding="utf-8",
    )
    with pytest.raises(runtime.ContractError, match="Only a passing"):
        runtime.package_evidence(
            protocol_path=PROTOCOL,
            freeze_root=freeze,
            primary_root=primary,
            evaluation_root=evaluation,
            archive_path=tmp_path / "evidence.tar.gz",
        )
