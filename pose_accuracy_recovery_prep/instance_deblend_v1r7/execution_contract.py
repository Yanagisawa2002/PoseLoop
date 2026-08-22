"""Immutable synthetic-only execution contract for A-R7.3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)


SCHEMA = "poseloop.pose-accuracy-recovery.instance-deblend-execution-protocol.v1r7.3"
PROTOCOL_ID = (
    "poseloop.pose-accuracy-recovery.development.instance-deblend.synthetic.v1r7.3"
)
ROLE = "DEVELOPMENT_ONLY_SYNTHETIC_OCCLUSION_GATE_V1R7_3"
STRATA = [
    "clean_single_instance",
    "touching_pair_different_cad",
    "touching_pair_same_cad",
    "depth_ordered_occlusion_pair",
    "same_cad_pile_three_plus",
    "container_and_frame_boundary_distractor",
]
BOUNDARY_ZERO = {
    "post_freeze_replay_rgb_open_count": 0,
    "post_freeze_replay_depth_open_count": 0,
    "post_freeze_replay_prediction_open_count": 0,
    "gt_pose_open_count": 0,
    "gt_mask_open_count": 0,
    "evaluator_open_count": 0,
    "foundationpose_run_count": 0,
    "official_scorer_run_count": 0,
    "downstream_export_count": 0,
}


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def validate_execution_protocol(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    protocol = _exact(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "role",
            "predecessor",
            "supersedes",
            "sam2_freeze",
            "a_r5_asset_freeze",
            "synthetic_development",
            "gates",
            "access_boundary",
            "authorization",
            "implementation_files",
            "protocol_lock_sha256",
        },
        "A-R7 execution protocol",
    )
    if (
        protocol["schema_version"] != SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["role"] != ROLE
    ):
        raise ContractError("A-R7 execution protocol identity changed")
    if protocol["predecessor"] != {
        "prep_commit": "6617ac717372bbc5c06f25534d3f3c1dc9a2cca5",
        "prep_tree": "9daf9e47fd79b4c0580e6ae145db8664f363abe4",
        "prep_protocol_lock_sha256": (
            "4fbae675e2f1ce7e096cb81680a7f1e982ae6fbeaa095de2baf4390ed6ac5504"
        ),
        "a_r6_policy_mutation_permitted": False,
    }:
        raise ContractError("A-R7 execution predecessor changed")
    if protocol["supersedes"] != {
        "protocol_id": (
            "poseloop.pose-accuracy-recovery.development."
            "instance-deblend.synthetic.v1r7.2"
        ),
        "protocol_lock_sha256": (
            "a222b8bb33f7bf08b516db3a9f55f0b48e721e375cabe901d823007ff7b8f71a"
        ),
        "old_protocol_mutation_permitted": False,
        "semantic_change": (
            "NEAREST_PROMPT_DEPTH_OWNERSHIP_PARTITIONS_THE_SAM2_UNION_AND_"
            "A_R6_FAILED_PROMPT_PARENT_ENVELOPE"
        ),
    }:
        raise ContractError("A-R7.3 superseded protocol identity changed")
    if protocol["sam2_freeze"] != {
        "repository": "https://github.com/facebookresearch/sam2",
        "commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "tree": "64becbca23f880e0056449377496da248a74da43",
        "source_archive_sha256": (
            "a9b182a4e502160a22226a32f1db4c31ea6a93000ca0f4502ddebdb7682d9209"
        ),
        "source_manifest_sha256": (
            "d7ffdda54476ef486083e4cd3e7ba89a39c4fabc030315f362da39dabbdd1d4f"
        ),
        "model_config_relative_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "model_config_bytes": 3798,
        "model_config_sha256": (
            "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
        ),
        "checkpoint_bytes": 898083611,
        "checkpoint_sha256": (
            "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        ),
    }:
        raise ContractError("A-R7 SAM2 freeze changed")
    if protocol["a_r5_asset_freeze"] != {
        "runtime_request_bytes": 11033,
        "runtime_request_sha256": (
            "2d1220987eee7cd32e7feb917363a4080116fa38853fc857d027e8d70ed30469"
        ),
        "template_manifest_bytes": 82509,
        "template_manifest_sha256": (
            "6cc2c958401a209ac3b69f3c93c730dc9c7d2b111a122d911bac9cdfc395b031"
        ),
        "adapter_config_bytes": 1927,
        "adapter_config_sha256": (
            "b1ee1d3638b3ce9f1659118c3b7f243fea8c35bfcd3640a4fc62f58c48207898"
        ),
        "catalog_object_ids": [1, 2, 4, 5, 6],
    }:
        raise ContractError("A-R7 A-R5 asset freeze changed")
    if protocol["synthetic_development"] != {
        "required_strata": STRATA,
        "development_first_seed": 4096,
        "rows_per_stratum": 8,
        "development_row_count": 48,
        "train_seed_interval": [0, 47],
        "development_seed_interval": [4096, 4143],
        "real_replay_path_argument_permitted": False,
        "result_driven_stratum_replacement_permitted": False,
    }:
        raise ContractError("A-R7 synthetic development freeze changed")
    if protocol["gates"] != {
        "matching_iou_thresholds": [0.5, 0.75],
        "occlusion_strict_recall75_improvement": True,
        "positive_baseline_merge_strict_reduction": True,
        "zero_baseline_merge_action": (
            "REQUIRE_ZERO_CANDIDATE_MERGE_WITH_STRICT_RECALL_GAIN"
        ),
        "clean_singleton_recall75_regression_permitted": False,
        "unmatched_prediction_increase_permitted": False,
        "split_count_increase_permitted": False,
        "every_stratum_required": True,
        "failure_action": "NO_GO_NO_SCENE9_REPLAY",
    }:
        raise ContractError("A-R7 synthetic gates changed")
    if protocol["access_boundary"] != BOUNDARY_ZERO:
        raise ContractError("A-R7 synthetic boundary is not zero")
    if protocol["authorization"] != {
        "sam2_asset_freeze_permitted": True,
        "train_smoke_permitted": True,
        "development_gate_permitted": True,
        "scene9_replay_permitted_before_gate_pass": False,
        "foundationpose_permitted": False,
        "official_scorer_permitted": False,
        "downstream_export_permitted": False,
    }:
        raise ContractError("A-R7 execution authorization changed")
    files = protocol["implementation_files"]
    if not isinstance(files, list) or not files:
        raise ContractError("A-R7 execution implementation inventory is empty")
    observed: set[str] = set()
    for index, raw in enumerate(files):
        row = _exact(
            raw,
            {"relative_path", "bytes", "sha256"},
            f"implementation_files[{index}]",
        )
        relative = row["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in observed
        ):
            raise ContractError("A-R7 execution implementation path changed")
        observed.add(relative)
        path = (repository_root / Path(*relative.split("/"))).resolve()
        try:
            path.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ContractError(
                "A-R7 execution implementation escapes repository"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ContractError(f"A-R7 execution implementation changed: {relative}")
    required = {
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/adapter.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/adapter.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/contracts.py",
        "pose_accuracy_recovery_prep/instance_selection_v1r6.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/adapter.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/contracts.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/core.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/execution_contract.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/synthetic_runtime.py",
    }
    if observed != required:
        raise ContractError("A-R7 execution implementation closure changed")
    expected_lock = protocol["protocol_lock_sha256"]
    if not isinstance(expected_lock, str) or len(expected_lock) != 64:
        raise ContractError("A-R7 execution protocol lock is invalid")
    actual_lock = canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_lock_sha256"}
    )
    if actual_lock != expected_lock:
        raise ContractError("A-R7 execution protocol lock changed")
    return dict(protocol)


__all__ = [
    "BOUNDARY_ZERO",
    "PROTOCOL_ID",
    "ROLE",
    "SCHEMA",
    "validate_execution_protocol",
]
