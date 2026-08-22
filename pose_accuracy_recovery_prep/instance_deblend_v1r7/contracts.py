"""Fail-closed A-R7 protocol contract.

A-R7 is a new proposal family, not a revision of A-R6 thresholds.  The old ten
frames are replay-only and cannot be used to choose prompts, thresholds, model
weights, checkpoints, or stopping criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)


PROTOCOL_SCHEMA = "poseloop.pose-accuracy-recovery.instance-deblend-protocol.v1r7"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-deblend.v1r7"
PROTOCOL_ROLE = "PREP_ONLY_NEW_INSTANCE_BACKEND_SYNTHETIC_CALIBRATION"

A_R6_COMMIT = "b9e6e4b6e0172135142ace74718b3d65a635509e"
A_R6_TREE = "49350f81eca5c9055b32595103171c81cf6b3c6a"
A_R6_PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-selection.v1r6"
A_R6_PROTOCOL_LOCK = "3761a567e138e7f5c28235f04f3cac82e7936c9610f0f2450e4b6bc7cec1f319"
A_R6_POLICY_SHA256 = "bc5bdbf433715a43ff594b7cab12184a5f3e09df25e524a6a58052d6fac0701c"
A_R6_OUTPUT_LOCK = "75780180f823c5a77bb19ae023b63632bdd2142cc0bbad09b67e709065949d13"
A_R6_OUTPUT_SHA256 = "eab0901fd4b277fa261d52b872a92b0564d80aa1ef1b1e53c478665857189704"
A_R6_RECEIPT_LOCK = "f3de0f1ef7c83f7c8f6a06c3982379a72d3cb411e84a7b3de718d55a0a9b9ef6"
A_R6_RECEIPT_SHA256 = "745892890367e0e6cdcd268eb1fc197c7962210135b7c9a9e1cc1da9224d96e0"

REPLAY_FRAME_IDS = [
    f"scene-{scene:06d}-image-{image:06d}"
    for scene in (0, 3, 9, 12, 15)
    for image in (0, 1)
]
SYNTHETIC_STRATA = [
    "clean_single_instance",
    "touching_pair_different_cad",
    "touching_pair_same_cad",
    "depth_ordered_occlusion_pair",
    "same_cad_pile_three_plus",
    "container_and_frame_boundary_distractor",
]
FORMAL_BLOCKERS = [
    "SAM2_SOURCE_TREE_AND_ARCHIVE_UNFROZEN",
    "SAM2_1_HIERA_LARGE_CHECKPOINT_BYTES_UNFROZEN",
    "SAM2_RUNTIME_AND_CAD_SCORER_UNINTEGRATED",
    "SYNTHETIC_CAD_OCCLUSION_BENCHMARK_UNBUILT",
]


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be SHA-256")
    return value


def _relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a POSIX relative path")
    path = Path(*value.split("/"))
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} escapes the repository")
    return path


def validate_protocol(
    protocol: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate the immutable generic PREP boundary.

    Passing this check does not authorize model download, inference, replay, or
    downstream export.  Those operations require a successor protocol after
    every blocker in ``FORMAL_BLOCKERS`` has been replaced by exact identities.
    """

    protocol = _exact(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "role",
            "predecessor_freeze",
            "new_method",
            "synthetic_development",
            "gates",
            "access_boundary",
            "readiness",
            "wrapper_files",
            "protocol_lock_sha256",
        },
        "A-R7 protocol",
    )
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["role"] != PROTOCOL_ROLE
    ):
        raise ContractError("A-R7 protocol identity changed")

    predecessor = _exact(
        protocol["predecessor_freeze"],
        {
            "commit",
            "tree",
            "protocol_id",
            "protocol_lock_sha256",
            "selection_policy_sha256",
            "selection_policy_mutation_permitted",
            "output_lock_sha256",
            "output_sha256",
            "receipt_lock_sha256",
            "receipt_sha256",
            "replay_frame_ids",
            "historical_diagnostic_exposure",
            "post_freeze_replay_role",
            "post_freeze_replay_may_influence_configuration",
        },
        "A-R7 predecessor freeze",
    )
    expected_predecessor = {
        "commit": A_R6_COMMIT,
        "tree": A_R6_TREE,
        "protocol_id": A_R6_PROTOCOL_ID,
        "protocol_lock_sha256": A_R6_PROTOCOL_LOCK,
        "selection_policy_sha256": A_R6_POLICY_SHA256,
        "selection_policy_mutation_permitted": False,
        "output_lock_sha256": A_R6_OUTPUT_LOCK,
        "output_sha256": A_R6_OUTPUT_SHA256,
        "receipt_lock_sha256": A_R6_RECEIPT_LOCK,
        "receipt_sha256": A_R6_RECEIPT_SHA256,
        "replay_frame_ids": REPLAY_FRAME_IDS,
        "historical_diagnostic_exposure": {
            "a_r6_output_bundle_opened": True,
            "human_content_review_completed": True,
            "identified_failure_frame_id": "scene-000009-image-000000",
            "identified_failure_taxonomy": "MERGED_ADJACENT_OCCLUDED_INSTANCES",
            "raw_rgb_or_depth_opened_for_a_r7_configuration": False,
            "numeric_threshold_or_weight_change_permitted": False,
        },
        "post_freeze_replay_role": "ONE_SHOT_POST_COMMIT_CONTENT_REPLAY_ONLY",
        "post_freeze_replay_may_influence_configuration": False,
    }
    if predecessor != expected_predecessor:
        raise ContractError("A-R7 changed the frozen A-R6/replay boundary")

    method = _exact(
        protocol["new_method"],
        {
            "proposal_family",
            "sam2_repository",
            "observed_main_commit",
            "source_tree_sha1",
            "source_archive_sha256",
            "model_config",
            "checkpoint_url",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "prompt_sources",
            "prompt_rule",
            "negative_prompt_exclusion_required",
            "visible_masks_must_be_mutually_exclusive",
            "a_r6_policy_reused_byte_exact",
            "threshold_override_permitted",
        },
        "A-R7 new method",
    )
    if method != {
        "proposal_family": "SAM2_1_RGBD_NEGATIVE_PROMPT_INSTANCE_DEBLEND",
        "sam2_repository": "https://github.com/facebookresearch/sam2",
        "observed_main_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "source_tree_sha1": None,
        "source_archive_sha256": None,
        "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint_url": (
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_large.pt"
        ),
        "checkpoint_bytes": None,
        "checkpoint_sha256": None,
        "prompt_sources": [
            "RGB_IMAGE_EMBEDDING",
            "RAW_SENSOR_DEPTH_DISCONTINUITY_SEEDS",
            "RIVAL_SEEDS_AS_NEGATIVE_PROMPTS",
        ],
        "prompt_rule": "SYNTHETIC_CALIBRATION_ONLY_UNTIL_FROZEN",
        "negative_prompt_exclusion_required": True,
        "visible_masks_must_be_mutually_exclusive": True,
        "a_r6_policy_reused_byte_exact": True,
        "threshold_override_permitted": False,
    }:
        raise ContractError("A-R7 method or no-retuning rule changed")

    synthetic = _exact(
        protocol["synthetic_development"],
        {
            "label_source",
            "required_strata",
            "training_seed_interval",
            "development_seed_interval",
            "seed_intervals_disjoint",
            "real_replay_frames_permitted",
            "gt_sensor_or_evaluator_assets_permitted",
            "result_driven_stratum_replacement_permitted",
        },
        "A-R7 synthetic development",
    )
    if synthetic != {
        "label_source": "CAD_RENDERER_VISIBLE_INSTANCE_IDS_ONLY",
        "required_strata": SYNTHETIC_STRATA,
        "training_seed_interval": [0, 4095],
        "development_seed_interval": [4096, 5119],
        "seed_intervals_disjoint": True,
        "real_replay_frames_permitted": False,
        "gt_sensor_or_evaluator_assets_permitted": False,
        "result_driven_stratum_replacement_permitted": False,
    }:
        raise ContractError("A-R7 synthetic-only calibration contract changed")

    gates = _exact(
        protocol["gates"],
        {
            "matching_iou_thresholds",
            "occlusion_strata_require_strict_recall75_improvement",
            "occlusion_strata_require_strict_merge_reduction",
            "clean_singleton_recall75_may_decrease",
            "unmatched_prediction_count_may_increase",
            "split_count_may_increase",
            "every_required_stratum_must_pass",
            "scene9_pass_is_a_development_gate",
            "failure_action",
        },
        "A-R7 gates",
    )
    if gates != {
        "matching_iou_thresholds": [0.5, 0.75],
        "occlusion_strata_require_strict_recall75_improvement": True,
        "occlusion_strata_require_strict_merge_reduction": True,
        "clean_singleton_recall75_may_decrease": False,
        "unmatched_prediction_count_may_increase": False,
        "split_count_may_increase": False,
        "every_required_stratum_must_pass": True,
        "scene9_pass_is_a_development_gate": False,
        "failure_action": "NO_GO_NO_THRESHOLD_TUNING_ON_REPLAY",
    }:
        raise ContractError("A-R7 scientific gates changed")

    boundary = _exact(
        protocol["access_boundary"],
        {
            "post_freeze_replay_rgb_open_count",
            "post_freeze_replay_depth_open_count",
            "post_freeze_replay_prediction_open_count",
            "gt_pose_open_count",
            "gt_mask_open_count",
            "evaluator_open_count",
            "foundationpose_run_count",
            "official_scorer_run_count",
            "downstream_export_count",
        },
        "A-R7 access boundary",
    )
    if boundary != {key: 0 for key in boundary}:
        raise ContractError("A-R7 PREP access boundary is not zero")

    readiness = _exact(
        protocol["readiness"],
        {
            "prep_contract_ready",
            "formal_workload_permitted",
            "scene9_replay_permitted",
            "blockers",
        },
        "A-R7 readiness",
    )
    if readiness != {
        "prep_contract_ready": True,
        "formal_workload_permitted": False,
        "scene9_replay_permitted": False,
        "blockers": FORMAL_BLOCKERS,
    }:
        raise ContractError("A-R7 PREP readiness was overstated")

    wrappers = protocol["wrapper_files"]
    if not isinstance(wrappers, list) or not wrappers:
        raise ContractError("A-R7 wrapper inventory is empty")
    seen: set[str] = set()
    for index, raw in enumerate(wrappers):
        item = _exact(
            raw,
            {"relative_path", "bytes", "sha256"},
            f"wrapper_files[{index}]",
        )
        relative = _relative(item["relative_path"], f"wrapper_files[{index}]")
        name = relative.as_posix()
        if name in seen:
            raise ContractError("A-R7 wrapper inventory contains duplicates")
        seen.add(name)
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ContractError("A-R7 wrapper bytes must be positive")
        _sha(item["sha256"], "A-R7 wrapper SHA")
        if repository_root is not None:
            path = (repository_root / relative).resolve()
            try:
                path.relative_to(repository_root.resolve())
            except ValueError as exc:
                raise ContractError("A-R7 wrapper escapes repository") from exc
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ContractError(f"A-R7 wrapper changed: {name}")
    expected_wrappers = {
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/__init__.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/__main__.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/adapter.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/cli.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/contracts.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/core.py",
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/instance_selection_v1r6.py",
    }
    if seen != expected_wrappers:
        raise ContractError("A-R7 wrapper inventory is not the exact PREP closure")

    expected_lock = _sha(protocol["protocol_lock_sha256"], "A-R7 protocol lock")
    observed_lock = canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_lock_sha256"}
    )
    if observed_lock != expected_lock:
        raise ContractError("A-R7 protocol lock changed")
    return dict(protocol)


__all__ = [
    "FORMAL_BLOCKERS",
    "PROTOCOL_ID",
    "PROTOCOL_ROLE",
    "PROTOCOL_SCHEMA",
    "REPLAY_FRAME_IDS",
    "SYNTHETIC_STRATA",
    "validate_protocol",
]
