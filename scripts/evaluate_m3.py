#!/usr/bin/env python3
"""Evaluate the frozen PoseLoop-AB M3 active-view budget policies."""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_m1 import (
    MAX_SYMMETRY_DISCRETIZATION_STEP,
    aggregate,
    finite_or_none,
    load_official_models,
    toolkit_commit,
)
from evaluate_m2 import (
    GT_TRANSFORM_MAX_MSPD_PX,
    GT_TRANSFORM_MAX_NORMALIZED_MSSD,
    choose_medoid,
    prepare_groups,
    result_from_selected,
    save_figure_atomic,
)
from fit_m3_policy import (
    CV_FOLDS,
    CV_SEED,
    K1_FEATURE_NAMES,
    K3_FEATURE_NAMES,
    apply_frozen_logistic,
    extract_prefix_features,
    pose_is_usable,
)
from m1_common import (
    ALLOWED_STATUSES,
    BATCH_SCHEMA_VERSION,
    GT_ROTATION_ATOL,
    MODALITY,
    canonical_sha256,
    load_jsonl,
    raw_pose_errors,
    read_json,
    sha256_file,
    stable_sample_id,
    write_json_atomic,
    write_jsonl_atomic,
    assert_pose,
)


SCHEMA_VERSION = 1
EXPECTED_OBJECT_COUNT = 15
MINIMUM_HOLDOUT_GROUPS = 100
VIEWS_PER_GROUP = 5
ACTIVE_POLICY_NAMES = ("fast", "balanced", "conservative")
FIXED_METHODS = (
    "fixed_k1_target",
    "fixed_k3_medoid",
    "fixed_k5_medoid",
    "fixed_k3_max_mask",
    "fixed_k5_max_mask",
)
VISIBILITY_BINS = ("low", "mid", "high")
BOOTSTRAP_SEED = 20260730
RANDOM_ASSIGNMENT_SEED = 20260730
LOW_N_THRESHOLD = 5

REQUIRED_CAVEATS = (
    "Known object IDs and ground-truth visible masks are used.",
    "Ground truth is used for physical-instance association and final evaluation.",
    "Confidence features contain neither ground truth nor future-view information.",
    "All confidence models and thresholds were frozen on M2 before holdout evaluation.",
    "The holdout is physical-instance-disjoint from M2 but not scene-disjoint.",
    "Holdout object counts are unequal; macro-object metrics are primary.",
    "Per-object estimates with fewer than five targets are low-n descriptive results.",
    "Additional views use the fixed M2 geometry-based camera-diversity order.",
    "M3 adapts the view budget; it does not rank or learn next-best views.",
    "These oracle-mask subset diagnostics are not official BOP detection AP.",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "artifacts" / "m3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")
        ),
    )
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=repo_root / "third_party" / "bop_toolkit",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=artifact_root / "holdout_groups.jsonl",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=artifact_root / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--holdout-summary",
        type=Path,
        default=artifact_root / "holdout_summary.json",
    )
    parser.add_argument(
        "--leakage-audit",
        type=Path,
        default=artifact_root / "leakage_audit.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=artifact_root / "view_predictions.jsonl",
    )
    parser.add_argument(
        "--frozen-policy",
        type=Path,
        default=artifact_root / "frozen_policy.json",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=artifact_root / "feature_audit.json",
    )
    parser.add_argument(
        "--fold-audit",
        type=Path,
        default=artifact_root / "development_fold_audit.json",
    )
    parser.add_argument(
        "--development-confidence",
        type=Path,
        default=artifact_root / "development_confidence_metrics.json",
    )
    parser.add_argument(
        "--development-oof",
        type=Path,
        default=artifact_root / "development_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--m2-groups",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--m2-metrics",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "metrics.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=artifact_root / "metrics.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=artifact_root / "evaluation_summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "m3_active_budget.md",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "reports",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Object-stratified physical-instance bootstrap replicates.",
    )
    return parser.parse_args()


def _artifact_path(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    root = (repo_root / "artifacts" / "m3").resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"M3 raw artifact must remain under {root}: {resolved}")
    return resolved


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _percentile(values: Iterable[Any]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "sample_count": len(finite),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate_candidate_manifest(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("M3 candidate manifest is empty")
    indexed: dict[str, dict[str, Any]] = {}
    identities: set[tuple[int, int, int, int]] = set()
    object_ids: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        identity = (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        if (
            not sample_id
            or sample_id in indexed
            or identity in identities
            or sample_id != stable_sample_id(*identity)
        ):
            raise ValueError(f"Invalid/duplicate M3 candidate at row {row_number}")
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Invalid M3 candidate schema: {sample_id}")
        if row.get("sensor_modality") != MODALITY:
            raise ValueError(f"Non-RealSense M3 candidate: {sample_id}")
        if not str(row.get("physical_instance_id", "")):
            raise ValueError(f"Missing physical instance: {sample_id}")
        rank = int(row.get("m3_acquisition_rank", -1))
        role = str(row.get("m3_role", ""))
        if rank not in range(VIEWS_PER_GROUP):
            raise ValueError(f"Invalid rank for {sample_id}")
        if role != ("target" if rank == 0 else "additional"):
            raise ValueError(f"Role/rank mismatch for {sample_id}")
        if int(row.get("m3_group_reference_count", -1)) != 1:
            raise ValueError(f"Candidate is not uniquely grouped: {sample_id}")
        if not str(row.get("m3_group_id", "")):
            raise ValueError(f"Candidate lacks an M3 group: {sample_id}")
        if int(row.get("selection_index", -1)) != row_number - 1:
            raise ValueError(f"Non-contiguous selection index: {sample_id}")
        if int(row["visible_mask_pixel_count"]) <= 0:
            raise ValueError(f"Empty M3 candidate mask: {sample_id}")
        if not 0.10 <= _finite(row["visible_fraction"], "visibility") <= 1.01:
            raise ValueError(f"Invalid visibility: {sample_id}")
        depth_ratio = _finite(row["valid_depth_ratio_inside_mask"], "depth-valid ratio")
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"Invalid depth-valid ratio: {sample_id}")
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            f"M3 candidate GT {sample_id}",
            GT_ROTATION_ATOL,
        )
        camera = np.asarray(row["camera_intrinsics_row_major"], dtype=np.float64)
        if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
            raise ValueError(f"Invalid camera intrinsics: {sample_id}")
        indexed[sample_id] = row
        identities.add(identity)
        object_ids.add(identity[3])
    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError(f"Expected 15 objects, found {sorted(object_ids)}")
    return indexed


def validate_groups(
    rows: list[dict[str, Any]],
    candidate_index: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) < MINIMUM_HOLDOUT_GROUPS:
        raise ValueError(f"Only {len(rows)} holdout physical instances")
    group_ids: set[str] = set()
    physical_ids: set[str] = set()
    candidate_ids: set[str] = set()
    object_ids: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for group in rows:
        group_id = str(group.get("group_id", ""))
        target_id = str(group.get("target_sample_id", ""))
        physical_id = str(group.get("physical_instance_id", ""))
        if (
            not group_id
            or group_id != target_id
            or group_id in group_ids
            or not physical_id
            or physical_id in physical_ids
        ):
            raise ValueError(f"Invalid/duplicate M3 group: {group_id!r}")
        if int(group.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Invalid M3 group schema: {group_id}")
        views = group.get("views")
        if not isinstance(views, list) or len(views) != VIEWS_PER_GROUP:
            raise ValueError(f"M3 group must have five views: {group_id}")
        views = sorted(views, key=lambda row: int(row["acquisition_rank"]))
        if [int(view["acquisition_rank"]) for view in views] != list(
            range(VIEWS_PER_GROUP)
        ):
            raise ValueError(f"Non-consecutive ranks: {group_id}")
        if str(views[0]["sample_id"]) != target_id:
            raise ValueError(f"Rank zero is not target: {group_id}")
        object_id = int(group["object_id"])
        scene_id = int(group["scene_id"])
        for rank, view in enumerate(views):
            sample_id = str(view.get("sample_id", ""))
            if (
                not sample_id
                or sample_id in candidate_ids
                or sample_id not in candidate_index
            ):
                raise ValueError(f"Invalid/reused grouped sample: {sample_id}")
            candidate = candidate_index[sample_id]
            if view.get("prediction_source") != "m3":
                raise ValueError(f"Non-M3 prediction source: {sample_id}")
            if str(view.get("physical_instance_id")) != physical_id:
                raise ValueError(f"Physical-instance mismatch: {sample_id}")
            if str(candidate.get("physical_instance_id")) != physical_id:
                raise ValueError(f"Manifest physical-instance mismatch: {sample_id}")
            if str(candidate.get("m3_group_id")) != group_id:
                raise ValueError(f"Manifest group mismatch: {sample_id}")
            if int(candidate.get("m3_acquisition_rank", -1)) != rank:
                raise ValueError(f"Manifest rank mismatch: {sample_id}")
            for field in ("scene_id", "image_id", "gt_instance_index", "object_id"):
                if int(view[field]) != int(candidate[field]):
                    raise ValueError(f"View/manifest {field} mismatch: {sample_id}")
            if int(view["scene_id"]) != scene_id or int(view["object_id"]) != object_id:
                raise ValueError(f"Cross-scene/object group: {group_id}")
            if not np.allclose(
                np.asarray(view["gt_model_to_camera_pose_m"], dtype=np.float64),
                np.asarray(candidate["gt_model_to_camera_pose_m"], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"View/manifest GT mismatch: {sample_id}")
            assert_pose(
                np.asarray(view["camera_world_to_camera_pose_m"], dtype=np.float64),
                f"M3 grouped camera pose {sample_id}",
                GT_ROTATION_ATOL,
            )
            candidate_ids.add(sample_id)
        normalized_group = copy.deepcopy(group)
        normalized_group["views"] = views
        normalized.append(normalized_group)
        group_ids.add(group_id)
        physical_ids.add(physical_id)
        object_ids.add(object_id)
    if candidate_ids != set(candidate_index):
        raise ValueError("M3 groups do not exactly cover candidate manifest")
    if len(candidate_index) != VIEWS_PER_GROUP * len(rows):
        raise ValueError("M3 candidate universe is not five rows per group")
    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError("M3 group object coverage is incomplete")
    return normalized


def validate_holdout_evidence(
    paths: Mapping[str, Path],
    groups: Sequence[dict[str, Any]],
    candidate_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = read_json(paths["holdout_summary"])
    leakage = read_json(paths["leakage_audit"])
    if summary.get("status") != "pass" or leakage.get("status") != "pass":
        raise ValueError("M3 holdout construction evidence is not a pass")
    gates = summary.get("hard_gates", {})
    if not gates or not all(bool(gate.get("passed")) for gate in gates.values()):
        raise ValueError("M3 holdout hard-gate evidence is incomplete")
    if int(summary.get("holdout_group_count", -1)) != len(groups):
        raise ValueError("M3 holdout summary group count mismatch")
    if int(summary.get("deduplicated_candidate_count", -1)) != len(candidate_index):
        raise ValueError("M3 holdout summary candidate count mismatch")
    outputs = summary.get("outputs", {})
    if outputs.get("groups_sha256") != sha256_file(paths["groups"]):
        raise ValueError("M3 group hash differs from holdout summary")
    if outputs.get("candidate_manifest_sha256") != sha256_file(
        paths["candidate_manifest"]
    ):
        raise ValueError("M3 manifest hash differs from holdout summary")
    source_paths = {
        "m1_manifest": paths["groups"].parents[1] / "m1" / "manifest.jsonl",
        "m2_groups": paths["m2_groups"],
        "build_m2_groups_source": SCRIPT_DIR / "build_m2_groups.py",
        "m2_common_source": SCRIPT_DIR / "m2_common.py",
        "build_m3_holdout_source": SCRIPT_DIR / "build_m3_holdout.py",
    }
    source_hashes = summary.get("source_sha256", {})
    for label, source_path in source_paths.items():
        if not source_path.is_file() or source_hashes.get(label) != sha256_file(
            source_path
        ):
            raise ValueError(f"M3 holdout source provenance mismatch: {label}")

    development_ids = set(leakage.get("development_physical_instance_ids", []))
    holdout_ids = set(leakage.get("holdout_physical_instance_ids", []))
    grouped_ids = {str(group["physical_instance_id"]) for group in groups}
    intersection = development_ids & holdout_ids
    if (
        holdout_ids != grouped_ids
        or intersection
        or int(leakage.get("development_holdout_intersection_count", -1)) != 0
        or leakage.get("development_holdout_intersection") != []
    ):
        raise ValueError("Physical-instance leakage audit failed")

    m2_groups = load_jsonl(paths["m2_groups"])
    current_development_ids = {
        str(group["oracle_association"]["track_id"]) for group in m2_groups
    }
    if current_development_ids != development_ids:
        raise ValueError("Saved leakage audit differs from current M2 tracks")
    expected_counts = Counter(int(group["object_id"]) for group in groups)
    saved_counts = {
        int(key): int(value)
        for key, value in summary["holdout_group_count_by_object"].items()
    }
    if expected_counts != Counter(saved_counts):
        raise ValueError("Holdout object counts mismatch")
    if (
        len(expected_counts) != EXPECTED_OBJECT_COUNT
        or min(expected_counts.values()) < 1
    ):
        raise ValueError("Holdout does not represent every object")
    return {
        "summary": summary,
        "leakage_status": "pass",
        "development_physical_instance_count": len(development_ids),
        "holdout_physical_instance_count": len(holdout_ids),
        "intersection_count": 0,
        "group_count_by_object": {
            str(key): int(value) for key, value in sorted(expected_counts.items())
        },
    }


def _validate_feature_audit(audit: Mapping[str, Any]) -> None:
    if (
        not audit.get("passed")
        or audit.get("policy_feature_inputs_use_ground_truth") is not False
        or audit.get("prefix_contract", {}).get("future_view_access") is not False
        or audit.get("feature_name_forbidden_token_hits") != {}
    ):
        raise ValueError("M3 feature audit does not prove GT/future-free inputs")
    copy_for_hash = dict(audit)
    claimed = str(copy_for_hash.pop("audit_sha256", ""))
    if claimed != canonical_sha256(copy_for_hash):
        raise ValueError("M3 feature-audit configuration hash mismatch")
    if tuple(audit.get("k1_feature_names", [])) != K1_FEATURE_NAMES:
        raise ValueError("k=1 audited feature list differs from evaluator")
    if tuple(audit.get("k3_feature_names", [])) != K3_FEATURE_NAMES:
        raise ValueError("k=3 audited feature list differs from evaluator")


def validate_frozen_policy(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(paths["frozen_policy"])
    if (
        frozen.get("freeze_status") != "frozen_before_holdout_pose_result_evaluation"
        or frozen.get("holdout_artifacts_read") is not False
    ):
        raise ValueError("Policy is not declared as an M2-only pre-holdout freeze")
    claimed_hash = str(frozen.get("configuration_sha256", ""))
    hash_payload = dict(frozen)
    hash_payload.pop("configuration_sha256", None)
    hash_payload.pop("frozen_utc", None)
    if not claimed_hash or canonical_sha256(hash_payload) != claimed_hash:
        raise ValueError("Frozen policy configuration hash mismatch")

    standalone_audit = read_json(paths["feature_audit"])
    _validate_feature_audit(standalone_audit)
    if frozen.get("feature_audit") != standalone_audit:
        raise ValueError("Frozen and standalone feature audits differ")
    for label, source_path in (
        ("groups", paths["m2_groups"]),
        ("metrics", paths["m2_metrics"]),
    ):
        declared = frozen.get("input_provenance", {}).get(label, {})
        if declared.get("sha256") != sha256_file(source_path):
            raise ValueError(f"Frozen M2 input provenance mismatch: {label}")

    fold = read_json(paths["fold_audit"])
    folds = fold.get("folds", [])
    if (
        int(fold.get("n_splits", -1)) != CV_FOLDS
        or int(fold.get("random_state", -1)) != CV_SEED
        or not fold.get("all_fold_group_intersections_empty")
        or not fold.get("each_row_tested_exactly_once")
        or len(folds) != CV_FOLDS
        or any(
            int(item.get("physical_instance_intersection_count", -1)) != 0
            or item.get("physical_instance_intersection") != []
            for item in folds
        )
    ):
        raise ValueError("Grouped development-fold leakage audit failed")
    if int(fold.get("physical_instance_count", -1)) != int(
        frozen["development_contract"]["physical_instance_count"]
    ):
        raise ValueError("Frozen/fold physical-instance counts differ")

    confidence = read_json(paths["development_confidence"])
    if confidence != frozen.get("oof_confidence_metrics"):
        raise ValueError("Frozen and standalone OOF confidence metrics differ")
    oof_rows = load_jsonl(paths["development_oof"])
    if len(oof_rows) != int(frozen["development_contract"]["row_count"]):
        raise ValueError("OOF row count differs from frozen development contract")
    if any(row.get("record_type") != "development_oof_prediction" for row in oof_rows):
        raise ValueError("Unexpected development OOF record")
    if len({str(row["group_id"]) for row in oof_rows}) != len(oof_rows):
        raise ValueError("Duplicate development OOF group")
    track_to_folds: dict[str, set[int]] = defaultdict(set)
    for row in oof_rows:
        track_id = str(row["physical_instance_track_id"])
        fold_index = int(row["fold"])
        if fold_index not in range(CV_FOLDS):
            raise ValueError(f"Invalid OOF fold assignment: {fold_index}")
        track_to_folds[track_id].add(fold_index)
    leaked_tracks = {
        track_id: sorted(values)
        for track_id, values in track_to_folds.items()
        if len(values) != 1
    }
    derived_assignment = {
        track_id: next(iter(values)) for track_id, values in track_to_folds.items()
    }
    if (
        leaked_tracks
        or len(derived_assignment) != int(fold["physical_instance_count"])
        or derived_assignment != fold.get("track_to_test_fold")
    ):
        raise ValueError(
            "OOF rows do not reproduce the zero-leakage grouped fold assignment"
        )
    m2_group_index = {
        str(group["group_id"]): group for group in load_jsonl(paths["m2_groups"])
    }
    if set(m2_group_index) != {str(row["group_id"]) for row in oof_rows}:
        raise ValueError("Development OOF rows do not exactly cover M2 targets")
    for row in oof_rows:
        group = m2_group_index[str(row["group_id"])]
        if int(row["object_id"]) != int(group["object_id"]) or str(
            row["physical_instance_track_id"]
        ) != str(group["oracle_association"]["track_id"]):
            raise ValueError("Development OOF identity differs from M2")

    models = frozen.get("confidence_models", {})
    if tuple(models.get("k1", {}).get("feature_names", [])) != K1_FEATURE_NAMES:
        raise ValueError("Frozen k=1 feature order mismatch")
    if tuple(models.get("k3", {}).get("feature_names", [])) != K3_FEATURE_NAMES:
        raise ValueError("Frozen k=3 feature order mismatch")
    points = frozen.get("policy", {}).get("operating_points", [])
    if [point.get("name") for point in points] != list(ACTIVE_POLICY_NAMES):
        raise ValueError("Frozen operating points are missing or reordered")
    for point in points:
        for field in ("t1", "t3"):
            value = _finite(point[field], f"{point['name']} {field}")
            if value not in frozen["policy"]["threshold_grid"]:
                raise ValueError(f"Frozen threshold is outside grid: {value}")
    return frozen, {
        "configuration_sha256": claimed_hash,
        "frozen_utc": frozen.get("frozen_utc"),
        "fold_audit": fold,
        "confidence_metrics": confidence,
        "oof_rows": oof_rows,
    }


def load_predictions(
    path: Path,
    candidate_index: Mapping[str, dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    groups_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("M3 predictions must start with metadata")
    metadata_rows = [row for row in rows if row.get("record_type") == "metadata"]
    prediction_rows = [row for row in rows if row.get("record_type") == "prediction"]
    if len(metadata_rows) != 1 or len(prediction_rows) != len(rows) - 1:
        raise ValueError("M3 prediction stream has unexpected record types")
    metadata = metadata_rows[0]
    if (
        int(metadata.get("schema_version", -1)) != SCHEMA_VERSION
        or metadata.get("candidate_manifest_sha256") != sha256_file(candidate_path)
        or metadata.get("groups_sha256") != sha256_file(groups_path)
        or int(metadata.get("candidate_universe_sample_count", -1))
        != len(candidate_index)
        or int(metadata.get("holdout_group_count", -1)) != len(groups)
    ):
        raise ValueError("M3 prediction metadata provenance/count mismatch")
    candidate_ids = [str(row["sample_id"]) for row in candidate_index.values()]
    group_ids = [str(group["group_id"]) for group in groups]
    if metadata.get("candidate_universe_sample_ids_sha256") != canonical_sha256(
        candidate_ids
    ):
        raise ValueError("M3 prediction candidate-order hash mismatch")
    if metadata.get("holdout_group_ids_sha256") != canonical_sha256(group_ids):
        raise ValueError("M3 prediction group-order hash mismatch")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or metadata.get(
        "batch_fingerprint"
    ) != canonical_sha256(config):
        raise ValueError("M3 prediction batch fingerprint mismatch")
    if (
        config.get("candidate_manifest_sha256") != sha256_file(candidate_path)
        or config.get("groups_sha256") != sha256_file(groups_path)
        or config.get("gt_pose_passed_to_foundationpose") is not False
    ):
        raise ValueError("M3 prediction immutable config is inconsistent")
    for source_name, declared_hash in config.get("poseloop_source_sha256", {}).items():
        source_path = SCRIPT_DIR / source_name
        if not source_path.is_file() or sha256_file(source_path) != declared_hash:
            raise ValueError(f"M3 inference source provenance mismatch: {source_name}")
    foundation = metadata.get("foundationpose", {})
    foundation_root = Path(str(foundation.get("root", "")))
    declared_commit = str(config.get("foundationpose_commit_sha", ""))
    if not foundation_root.is_dir() or foundation.get("commit_sha") != declared_commit:
        raise ValueError("FoundationPose metadata/root mismatch")
    completed = subprocess.run(
        ["git", "-C", str(foundation_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != declared_commit:
        raise ValueError("FoundationPose checkout has drifted since M3 inference")
    checkpoint_hashes = config.get("checkpoint_sha256", {})
    for label, checkpoint in metadata.get("checkpoints", {}).get("models", {}).items():
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        expected_hash = checkpoint_hashes.get(label)
        if (
            not checkpoint_path.is_file()
            or checkpoint.get("sha256") != expected_hash
            or sha256_file(checkpoint_path) != expected_hash
        ):
            raise ValueError(f"FoundationPose checkpoint drift: {label}")

    predictions: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for row_number, row in enumerate(prediction_rows, start=2):
        sample_id = str(row.get("sample_id", ""))
        if (
            not sample_id
            or sample_id in predictions
            or sample_id not in candidate_index
        ):
            raise ValueError(f"Invalid/duplicate M3 prediction at row {row_number}")
        candidate = candidate_index[sample_id]
        if int(row.get("schema_version", -1)) != BATCH_SCHEMA_VERSION:
            raise ValueError(f"Prediction schema mismatch: {sample_id}")
        status = str(row.get("status", ""))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid prediction status: {sample_id}/{status}")
        for field in (
            "scene_id",
            "image_id",
            "gt_instance_index",
            "object_id",
            "sensor_modality",
        ):
            if row.get(field) != candidate.get(field):
                raise ValueError(f"Prediction identity mismatch {field}: {sample_id}")
        if not np.allclose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            np.asarray(candidate["gt_model_to_camera_pose_m"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Prediction GT audit pose mismatch: {sample_id}")
        for field in ("visible_fraction", "valid_depth_ratio_inside_mask"):
            if not math.isclose(
                float(row[field]),
                float(candidate[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Prediction {field} mismatch: {sample_id}")
        if int(row["visible_mask_pixel_count"]) != int(
            candidate["visible_mask_pixel_count"]
        ):
            raise ValueError(f"Prediction mask count mismatch: {sample_id}")
        if row.get("visibility_bin") != candidate.get("visibility_bin"):
            raise ValueError(f"Prediction visibility-bin mismatch: {sample_id}")
        runtime = finite_or_none(row.get("registration_seconds"))
        if row.get("registration_seconds") is not None and (
            runtime is None or runtime < 0.0
        ):
            raise ValueError(f"Invalid registration runtime: {sample_id}")
        if status == "success":
            pose = np.asarray(
                row.get("predicted_model_to_camera_pose_m"), dtype=np.float64
            )
            assert_pose(pose, f"M3 predicted pose {sample_id}")
            expected_translation, expected_rotation = raw_pose_errors(
                pose,
                np.asarray(candidate["gt_model_to_camera_pose_m"], dtype=np.float64),
            )
            if not math.isclose(
                float(row["translation_error_mm"]),
                expected_translation,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ) or not math.isclose(
                float(row["raw_rotation_error_degrees"]),
                expected_rotation,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Stored diagnostic error mismatch: {sample_id}")
            if runtime is None:
                raise ValueError(f"Successful prediction lacks runtime: {sample_id}")
        else:
            if "predicted_model_to_camera_pose_m" in row:
                raise ValueError(f"Failed prediction has a pose: {sample_id}")
            error = row.get("error")
            if not isinstance(error, Mapping) or not error.get("type"):
                raise ValueError(f"Failed prediction lacks explicit error: {sample_id}")
        predictions[sample_id] = row
        status_counts[status] += 1
    if set(predictions) != set(candidate_index):
        raise ValueError("Not every holdout candidate has exactly one result")

    pilot = metadata.get("initial_pilot_selection")
    if metadata.get("initial_invocation") != "pilot" or not isinstance(pilot, Mapping):
        raise ValueError("M3 output was not initialized by the required pilot")
    pilot_group_ids = [str(value) for value in pilot.get("group_ids", [])]
    group_index = {str(group["group_id"]): group for group in groups}
    if (
        len(pilot_group_ids) != EXPECTED_OBJECT_COUNT
        or len(set(pilot_group_ids)) != EXPECTED_OBJECT_COUNT
        or any(group_id not in group_index for group_id in pilot_group_ids)
        or len(
            {int(group_index[group_id]["object_id"]) for group_id in pilot_group_ids}
        )
        != EXPECTED_OBJECT_COUNT
        or int(pilot.get("candidate_sample_count", -1))
        != EXPECTED_OBJECT_COUNT * VIEWS_PER_GROUP
    ):
        raise ValueError("Saved pilot does not cover one complete group per object")
    expected_pilot_group_ids = [
        min(
            str(group["group_id"])
            for group in groups
            if int(group["object_id"]) == object_id
        )
        for object_id in sorted({int(group["object_id"]) for group in groups})
    ]
    if pilot_group_ids != expected_pilot_group_ids or pilot.get(
        "group_ids_sha256"
    ) != canonical_sha256(pilot_group_ids):
        raise ValueError("Saved pilot differs from deterministic all-object selection")
    pilot_ids = {
        str(view["sample_id"])
        for group_id in pilot_group_ids
        for view in group_index[group_id]["views"]
    }
    if not pilot_ids.issubset(predictions):
        raise ValueError("Pilot rows are absent from the final prediction stream")
    ordered_pilot_ids = [
        sample_id for sample_id in candidate_ids if sample_id in pilot_ids
    ]
    if pilot.get("candidate_sample_ids_sha256") != canonical_sha256(ordered_pilot_ids):
        raise ValueError("Saved pilot candidate-order hash mismatch")
    return (
        metadata,
        predictions,
        {
            "attempted_count": len(predictions),
            "expected_count": len(candidate_index),
            "status_counts": dict(sorted(status_counts.items())),
            "pilot_group_count": len(pilot_group_ids),
            "pilot_candidate_count": len(pilot_ids),
            "pilot_object_count": EXPECTED_OBJECT_COUNT,
            "all_rows_attempted": True,
        },
    )


def validate_freeze_precedes_predictions(
    frozen: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_time = _parse_utc(frozen.get("frozen_utc"), "frozen_utc")
    prediction_time = _parse_utc(metadata.get("created_utc"), "prediction created_utc")
    if not frozen_time < prediction_time:
        raise ValueError(
            "Frozen policy does not predate holdout prediction stream: "
            f"{frozen_time.isoformat()} >= {prediction_time.isoformat()}"
        )
    return {
        "passed": True,
        "frozen_utc": frozen_time.isoformat(),
        "prediction_stream_created_utc": prediction_time.isoformat(),
        "freeze_precedes_prediction_stream": True,
    }


def fixed_results(
    prepared_groups: Sequence[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for group_number, prepared in enumerate(prepared_groups, start=1):
        group = prepared["group"]
        group_id = str(group["group_id"])
        physical_id = str(group["physical_instance_id"])
        views = prepared["views"]
        per_group: dict[str, dict[str, Any]] = {}

        target = result_from_selected(
            prepared, views[:1], views[0], "fixed_k1_target", 1
        )
        started = time.perf_counter_ns()
        medoid3, scores3 = choose_medoid(views[:3], prepared["object_data"])
        medoid3_seconds = (time.perf_counter_ns() - started) / 1e9
        medoid3_result = result_from_selected(
            prepared,
            views[:3],
            medoid3,
            "fixed_k3_medoid",
            3,
            scores3,
        )
        started = time.perf_counter_ns()
        medoid5, scores5 = choose_medoid(views[:5], prepared["object_data"])
        medoid5_seconds = (time.perf_counter_ns() - started) / 1e9
        medoid5_result = result_from_selected(
            prepared,
            views[:5],
            medoid5,
            "fixed_k5_medoid",
            5,
            scores5,
        )
        started = time.perf_counter_ns()
        max3 = max(
            views[:3],
            key=lambda view: (
                int(view["view"]["visible_mask_pixel_count"]),
                -int(view["view"]["acquisition_rank"]),
            ),
        )
        max3_seconds = (time.perf_counter_ns() - started) / 1e9
        max3_result = result_from_selected(
            prepared, views[:3], max3, "fixed_k3_max_mask", 3
        )
        started = time.perf_counter_ns()
        max5 = max(
            views[:5],
            key=lambda view: (
                int(view["view"]["visible_mask_pixel_count"]),
                -int(view["view"]["acquisition_rank"]),
            ),
        )
        max5_seconds = (time.perf_counter_ns() - started) / 1e9
        max5_result = result_from_selected(
            prepared, views[:5], max5, "fixed_k5_max_mask", 5
        )
        selection_seconds = {
            "fixed_k1_target": 0.0,
            "fixed_k3_medoid": medoid3_seconds,
            "fixed_k5_medoid": medoid5_seconds,
            "fixed_k3_max_mask": max3_seconds,
            "fixed_k5_max_mask": max5_seconds,
        }
        for result in (
            target,
            medoid3_result,
            medoid5_result,
            max3_result,
            max5_result,
        ):
            result.update(
                {
                    "physical_instance_id": physical_id,
                    "policy_probability_k1": None,
                    "policy_probability_k3": None,
                    "confidence_decision_overhead_seconds": 0.0,
                    "selection_overhead_seconds": selection_seconds[
                        str(result["method"])
                    ],
                    "policy_overhead_seconds": selection_seconds[str(result["method"])],
                }
            )
            per_group[str(result["method"])] = result
            rows.append(result)
        indexed[group_id] = per_group
        if group_number % 25 == 0 or group_number == len(prepared_groups):
            print(
                f"fixed holdout methods: {group_number}/{len(prepared_groups)}",
                flush=True,
            )
    return indexed, rows


def _policy_row(
    source: Mapping[str, Any],
    method: str,
    probability_k1: float | None,
    probability_k3: float | None,
    overhead_seconds: float,
) -> dict[str, Any]:
    row = copy.deepcopy(dict(source))
    row["method"] = method
    row["policy_probability_k1"] = probability_k1
    row["policy_probability_k3"] = probability_k3
    row["confidence_decision_overhead_seconds"] = float(overhead_seconds)
    row["selection_overhead_seconds"] = 0.0
    row["policy_overhead_seconds"] = float(overhead_seconds)
    return row


def _prefix_only_feature_group(
    group: Mapping[str, Any],
    budget: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    """Build the only group view that a confidence decision may inspect."""
    raw_views = group["views"]
    if not isinstance(raw_views, list) or len(raw_views) < budget:
        raise ValueError(f"Group lacks the requested acquired prefix k={budget}")
    prefix_views = raw_views[:budget]
    prefix_group = {"views": prefix_views}
    ranks = [int(view["acquisition_rank"]) for view in prefix_views]
    if set(prefix_group) != {"views"}:
        raise AssertionError("Confidence feature input exposed non-view group fields")
    if len(prefix_views) != budget or ranks != list(range(budget)):
        raise AssertionError(
            f"Confidence feature input is not the exact acquired prefix k={budget}"
        )
    if any(prefix_views[index] is not raw_views[index] for index in range(budget)):
        raise AssertionError("Confidence feature input changed prefix record identity")
    return prefix_group, ranks


def active_results(
    groups: Sequence[dict[str, Any]],
    fixed: Mapping[str, Mapping[str, dict[str, Any]]],
    frozen: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    models = frozen["confidence_models"]
    points = frozen["policy"]["operating_points"]
    outputs: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for point in points:
        name = str(point["name"])
        method = f"active_{name}"
        selected_rows: list[dict[str, Any]] = []
        stop_counts: Counter[int] = Counter()
        prefix_access_counts: Counter[str] = Counter()
        prefix_input_sizes: dict[str, set[int]] = defaultdict(set)
        prefix_input_ranks: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        for group in groups:
            group_id = str(group["group_id"])
            started = time.perf_counter_ns()
            k1 = fixed[group_id]["fixed_k1_target"]
            prefix_group1, ranks1 = _prefix_only_feature_group(group, 1)
            features1 = extract_prefix_features(prefix_group1, k1, 1)
            p1 = apply_frozen_logistic(
                features1,
                models["k1"],
                pose_usable=pose_is_usable(k1),
            )
            prefix_access_counts["k1"] += 1
            prefix_input_sizes["k1"].add(len(prefix_group1["views"]))
            prefix_input_ranks["k1"].add(tuple(ranks1))
            p3: float | None = None
            if p1 >= float(point["t1"]):
                budget = 1
                source = k1
            else:
                k3 = fixed[group_id]["fixed_k3_medoid"]
                prefix_group3, ranks3 = _prefix_only_feature_group(group, 3)
                features3 = extract_prefix_features(prefix_group3, k3, 3)
                p3 = apply_frozen_logistic(
                    features3,
                    models["k3"],
                    pose_usable=pose_is_usable(k3),
                )
                prefix_access_counts["k3"] += 1
                prefix_input_sizes["k3"].add(len(prefix_group3["views"]))
                prefix_input_ranks["k3"].add(tuple(ranks3))
                if p3 >= float(point["t3"]):
                    budget = 3
                    source = k3
                else:
                    budget = 5
                    source = fixed[group_id]["fixed_k5_medoid"]
            overhead = (time.perf_counter_ns() - started) / 1e9
            medoid_overhead = 0.0
            if budget >= 3:
                medoid_overhead += float(
                    fixed[group_id]["fixed_k3_medoid"]["selection_overhead_seconds"]
                )
            if budget == 5:
                medoid_overhead += float(
                    fixed[group_id]["fixed_k5_medoid"]["selection_overhead_seconds"]
                )
            row = _policy_row(source, method, p1, p3, overhead)
            row["selection_overhead_seconds"] = medoid_overhead
            row["policy_overhead_seconds"] = overhead + medoid_overhead
            row["frozen_threshold_t1"] = float(point["t1"])
            row["frozen_threshold_t3"] = float(point["t3"])
            row["requested_view_budget"] = budget
            row["acquired_view_count"] = budget
            selected_rows.append(row)
            stop_counts[budget] += 1
        expected_prefix_access = {
            "k1": len(groups),
            "k3": int(stop_counts[3] + stop_counts[5]),
        }
        observed_prefix_access = {
            stage: int(prefix_access_counts[stage]) for stage in ("k1", "k3")
        }
        if observed_prefix_access != expected_prefix_access:
            raise AssertionError(
                "Confidence feature access count is inconsistent with stopping"
            )
        expected_sizes = {"k1": {1}}
        expected_ranks = {"k1": {(0,)}}
        if expected_prefix_access["k3"]:
            expected_sizes["k3"] = {3}
            expected_ranks["k3"] = {(0, 1, 2)}
        if dict(prefix_input_sizes) != expected_sizes:
            raise AssertionError(
                "Confidence extractor received a non-prefix view count"
            )
        if dict(prefix_input_ranks) != expected_ranks:
            raise AssertionError(
                "Confidence extractor received future acquisition ranks"
            )
        outputs[method] = selected_rows
        audit[name] = {
            "t1": float(point["t1"]),
            "t3": float(point["t3"]),
            "row_count": len(selected_rows),
            "prefix_feature_access_counts": dict(sorted(prefix_access_counts.items())),
            "prefix_input_view_counts": {
                stage: sorted(values)
                for stage, values in sorted(prefix_input_sizes.items())
            },
            "prefix_input_acquisition_ranks": {
                stage: [list(ranks) for ranks in sorted(values)]
                for stage, values in sorted(prefix_input_ranks.items())
            },
            "stopping_counts": {
                f"k{budget}": int(stop_counts[budget]) for budget in (1, 3, 5)
            },
            "extractor_input_group_keys": ["views"],
            "exact_prefix_slice_identity_assertion_passed": True,
            "prefix_only_runtime_assertions_passed": True,
            "future_view_record_references_passed_to_extractor": 0,
            "future_view_field_dereference_count": 0,
            "future_view_feature_access": False,
            "numpy_frozen_model_only": True,
        }
    return outputs, audit


def random_budget_results(
    groups: Sequence[dict[str, Any]],
    fixed: Mapping[str, Mapping[str, dict[str, Any]]],
    active: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    fixed_name = {
        1: "fixed_k1_target",
        3: "fixed_k3_medoid",
        5: "fixed_k5_medoid",
    }
    ordered_groups = sorted(groups, key=lambda row: str(row["group_id"]))
    for point_index, name in enumerate(ACTIVE_POLICY_NAMES):
        active_method = f"active_{name}"
        method = f"random_{name}"
        counts = Counter(
            int(row["acquired_view_count"]) for row in active[active_method]
        )
        budget_vector = np.asarray(
            [budget for budget in (1, 3, 5) for _ in range(counts[budget])],
            dtype=np.int64,
        )
        seed = RANDOM_ASSIGNMENT_SEED + point_index
        shuffled = np.random.default_rng(seed).permutation(budget_vector)
        rows: list[dict[str, Any]] = []
        for group, budget_value in zip(ordered_groups, shuffled, strict=True):
            budget = int(budget_value)
            source = fixed[str(group["group_id"])][fixed_name[budget]]
            row = _policy_row(source, method, None, None, 0.0)
            row["random_assignment_seed"] = seed
            row["requested_view_budget"] = budget
            row["acquired_view_count"] = budget
            rows.append(row)
        observed = Counter(int(row["acquired_view_count"]) for row in rows)
        if observed != counts:
            raise AssertionError("Matched random assignment changed stopping counts")
        outputs[method] = rows
        audit[name] = {
            "seed": seed,
            "active_stopping_counts": {
                f"k{budget}": int(counts[budget]) for budget in (1, 3, 5)
            },
            "random_stopping_counts": {
                f"k{budget}": int(observed[budget]) for budget in (1, 3, 5)
            },
            "counts_match_exactly": True,
        }
    return outputs, audit


def oracle_minimum_budget_results(
    groups: Sequence[dict[str, Any]],
    fixed: Mapping[str, Mapping[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = {
        1: "fixed_k1_target",
        3: "fixed_k3_medoid",
        5: "fixed_k5_medoid",
    }
    for group in groups:
        per_group = fixed[str(group["group_id"])]
        joint_success = {
            budget: bool(per_group[names[budget]]["diagnostic_success"]["joint"])
            for budget in (1, 3, 5)
        }
        budget = next(
            (
                candidate_budget
                for candidate_budget in (1, 3, 5)
                if joint_success[candidate_budget]
            ),
            5,
        )
        row = _policy_row(
            per_group[names[budget]],
            "oracle_minimum_budget",
            None,
            None,
            0.0,
        )
        row["requested_view_budget"] = budget
        row["acquired_view_count"] = budget
        row["oracle_uses_gt_joint_diagnostic_for_stopping"] = True
        row["oracle_joint_success_by_budget"] = {
            f"k{candidate_budget}": joint_success[candidate_budget]
            for candidate_budget in (1, 3, 5)
        }
        row["oracle_fell_back_to_k5_without_joint_success"] = not any(
            joint_success.values()
        )
        row["oracle_objective"] = (
            "choose the first k=1,3,5 medoid whose final GT joint diagnostic "
            "succeeds, falling back to k=5 if none succeeds"
        )
        rows.append(row)
    return rows


def _aggregate_macro_micro(
    rows: Sequence[dict[str, Any]],
    *,
    require_all_objects: bool = True,
) -> dict[str, Any]:
    micro = aggregate(rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["object_id"])].append(row)
    if require_all_objects and len(grouped) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Policy result does not cover all 15 objects")
    per_object = []
    for object_id in sorted(grouped):
        value = aggregate(grouped[object_id])
        value.update(
            {
                "object_id": object_id,
                "physical_instance_count": len(grouped[object_id]),
                "low_n_descriptive": len(grouped[object_id]) < LOW_N_THRESHOLD,
            }
        )
        per_object.append(value)
    macro = {
        "object_count": len(per_object),
        "ar_mssd": float(np.mean([row["ar_mssd"] for row in per_object])),
        "ar_mspd": float(np.mean([row["ar_mspd"] for row in per_object])),
        "oracle_mask_subset_ar_mssd_mspd": float(
            np.mean([row["oracle_mask_subset_ar_mssd_mspd"] for row in per_object])
        ),
        "finite_pose_rate": float(
            np.mean([row["finite_pose_rate"] for row in per_object])
        ),
    }
    return {"macro_object": macro, "micro": micro, "by_object": per_object}


def _subset_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "represented_object_count": 0,
            "macro_object": None,
            "micro": aggregate([]),
        }
    value = _aggregate_macro_micro(rows, require_all_objects=False)
    return {
        "sample_count": len(rows),
        "represented_object_count": len({int(row["object_id"]) for row in rows}),
        "macro_object": value["macro_object"],
        "micro": value["micro"],
    }


def rescue_harm(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base_index = {str(row["group_id"]): row for row in baseline}
    candidate_index = {str(row["group_id"]): row for row in candidate}
    if set(base_index) != set(candidate_index):
        raise ValueError("Rescue/harm policies do not share the same targets")
    output: dict[str, Any] = {}
    for flag in ("joint", "mssd_0.10d", "mspd_10r"):
        base_success = {
            key: bool(row["diagnostic_success"][flag])
            for key, row in base_index.items()
        }
        candidate_success = {
            key: bool(candidate_index[key]["diagnostic_success"][flag])
            for key in base_index
        }
        failures = sum(not value for value in base_success.values())
        successes = sum(base_success.values())
        rescued = sum(
            not base_success[key] and candidate_success[key] for key in base_index
        )
        harmed = sum(
            base_success[key] and not candidate_success[key] for key in base_index
        )
        output[flag] = {
            "baseline_failure_count": failures,
            "baseline_success_count": successes,
            "rescued_count": rescued,
            "harmed_count": harmed,
            "rescue_rate": rescued / failures if failures else None,
            "harm_rate": harmed / successes if successes else None,
        }
    return output


def summarize_policy(
    method: str,
    rows: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_result = _aggregate_macro_micro(rows)
    views = [int(row["acquired_view_count"]) for row in rows]
    latency = [row.get("sequential_runtime_seconds") for row in rows]
    overhead = [row.get("policy_overhead_seconds") for row in rows]
    decision_overhead = [
        row.get("confidence_decision_overhead_seconds") for row in rows
    ]
    selection_overhead = [row.get("selection_overhead_seconds") for row in rows]
    stop_counts = Counter(views)
    visibility = {
        label: _subset_summary(
            [row for row in rows if row["target_visibility_bin"] == label]
        )
        for label in VISIBILITY_BINS
    }
    object15 = next(
        row for row in aggregate_result["by_object"] if int(row["object_id"]) == 15
    )
    return {
        "method": method,
        **aggregate_result,
        "views": _percentile(views),
        "registration_latency_seconds": _percentile(latency),
        "policy_overhead_seconds": _percentile(overhead),
        "confidence_decision_overhead_seconds": _percentile(decision_overhead),
        "selection_overhead_seconds": _percentile(selection_overhead),
        "stopping_counts": {
            f"k{budget}": int(stop_counts[budget]) for budget in (1, 3, 5)
        },
        "stopping_percentages": {
            f"k{budget}": 100.0 * stop_counts[budget] / len(rows)
            for budget in (1, 3, 5)
        },
        "rescue_harm_vs_fixed_k1": rescue_harm(baseline, rows),
        "by_target_visibility_bin": visibility,
        "object_15": object15,
    }


def _linear_mixture_at_budget(
    fixed_summaries: Mapping[int, Mapping[str, Any]],
    mean_views: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for low, high in ((1, 3), (3, 5), (1, 5)):
        if low - 1e-12 <= mean_views <= high + 1e-12:
            high_weight = (mean_views - low) / (high - low)
            low_weight = 1.0 - high_weight
            metrics = {}
            for field in (
                "ar_mssd",
                "ar_mspd",
                "oracle_mask_subset_ar_mssd_mspd",
            ):
                metrics[field] = (
                    low_weight * fixed_summaries[low]["macro_object"][field]
                    + high_weight * fixed_summaries[high]["macro_object"][field]
                )
            candidates.append(
                {
                    "lower_budget": low,
                    "upper_budget": high,
                    "lower_method": str(fixed_summaries[low]["method"]),
                    "upper_method": str(fixed_summaries[high]["method"]),
                    "lower_weight": float(low_weight),
                    "upper_weight": float(high_weight),
                    **metrics,
                }
            )
    if not candidates:
        raise ValueError(f"Mean view count is outside [1,5]: {mean_views}")
    return min(
        candidates,
        key=lambda row: (
            -row["oracle_mask_subset_ar_mssd_mspd"],
            row["upper_budget"] - row["lower_budget"],
            row["lower_budget"],
        ),
    )


def add_policy_comparisons(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixed1 = summaries["fixed_k1_target"]
    fixed3 = summaries["fixed_k3_medoid"]
    fixed5 = summaries["fixed_k5_medoid"]
    fixed3_frontier = max(
        (fixed3, summaries["fixed_k3_max_mask"]),
        key=lambda row: (
            row["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            row["method"],
        ),
    )
    fixed5_frontier = max(
        (fixed5, summaries["fixed_k5_max_mask"]),
        key=lambda row: (
            row["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            row["method"],
        ),
    )
    fixed_by_budget = {1: fixed1, 3: fixed3_frontier, 5: fixed5_frontier}
    baseline_score = fixed1["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
    fixed5_score = fixed5["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
    available_gain = fixed5_score - baseline_score
    active_comparisons: dict[str, Any] = {}
    for name in ACTIVE_POLICY_NAMES:
        active = summaries[f"active_{name}"]
        random_summary = summaries[f"random_{name}"]
        score = active["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
        frontier = _linear_mixture_at_budget(
            fixed_by_budget, float(active["views"]["mean"])
        )
        active_comparisons[name] = {
            "active_method": f"active_{name}",
            "matched_random_method": f"random_{name}",
            "gain_over_fixed_k1_combined": score - baseline_score,
            "active_minus_fixed_k3_combined": (
                score - fixed3["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
            ),
            "active_minus_fixed_k5_combined": score - fixed5_score,
            "fixed_k5_gain_retained_fraction": (
                (score - baseline_score) / available_gain
                if available_gain > 1e-15
                else None
            ),
            "registration_latency_reduction_vs_fixed_k5_fraction": (
                1.0
                - float(active["registration_latency_seconds"]["mean"])
                / float(fixed5["registration_latency_seconds"]["mean"])
            ),
            "active_minus_matched_random_combined": (
                score
                - random_summary["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
            ),
            "matched_random_stopping_counts_equal": (
                active["stopping_counts"] == random_summary["stopping_counts"]
            ),
            "linear_fixed_budget_mixture_frontier": frontier,
            "active_minus_mixture_frontier_combined": (
                score - frontier["oracle_mask_subset_ar_mssd_mspd"]
            ),
        }
        if not active_comparisons[name]["matched_random_stopping_counts_equal"]:
            raise AssertionError("Matched-random stopping proportions differ")
    return active_comparisons


def bootstrap_confidence_intervals(
    result_sets: Mapping[str, Sequence[dict[str, Any]]],
    replicates: int,
) -> dict[str, Any]:
    if replicates < 200:
        raise ValueError("Use at least 200 bootstrap replicates")
    methods = list(result_sets)
    group_orders = [
        [str(row["group_id"]) for row in result_sets[method]] for method in methods
    ]
    if any(set(order) != set(group_orders[0]) for order in group_orders[1:]):
        raise ValueError("Bootstrap methods do not cover identical instances")
    canonical_ids = sorted(group_orders[0])
    indexed = {
        method: {str(row["group_id"]): row for row in rows}
        for method, rows in result_sets.items()
    }
    object_ids = sorted(
        {int(indexed[methods[0]][group_id]["object_id"]) for group_id in canonical_ids}
    )
    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Bootstrap does not span all objects")
    ids_by_object = {
        object_id: [
            group_id
            for group_id in canonical_ids
            if int(indexed[methods[0]][group_id]["object_id"]) == object_id
        ]
        for object_id in object_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {
        object_id: rng.integers(
            0,
            len(ids),
            size=(replicates, len(ids)),
            endpoint=False,
        )
        for object_id, ids in ids_by_object.items()
    }

    samples: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        metric_samples = {
            "ar_mssd": np.zeros(replicates, dtype=np.float64),
            "ar_mspd": np.zeros(replicates, dtype=np.float64),
            "combined": np.zeros(replicates, dtype=np.float64),
            "mean_views": np.zeros(replicates, dtype=np.float64),
            "mean_registration_seconds": np.zeros(replicates, dtype=np.float64),
        }
        for object_id in object_ids:
            ids = ids_by_object[object_id]
            rows = [indexed[method][group_id] for group_id in ids]
            arrays = {
                "ar_mssd": np.asarray(
                    [row["sample_ar_mssd"] for row in rows], dtype=np.float64
                ),
                "ar_mspd": np.asarray(
                    [row["sample_ar_mspd"] for row in rows], dtype=np.float64
                ),
                "mean_views": np.asarray(
                    [row["acquired_view_count"] for row in rows], dtype=np.float64
                ),
                "mean_registration_seconds": np.asarray(
                    [
                        (
                            float(row["sequential_runtime_seconds"])
                            if row.get("sequential_runtime_seconds") is not None
                            else np.nan
                        )
                        for row in rows
                    ],
                    dtype=np.float64,
                ),
            }
            arrays["combined"] = (arrays["ar_mssd"] + arrays["ar_mspd"]) / 2.0
            indices = draws[object_id]
            for field, values in arrays.items():
                sampled = values[indices]
                if field == "mean_registration_seconds":
                    per_object = np.nanmean(sampled, axis=1)
                else:
                    per_object = np.mean(sampled, axis=1)
                metric_samples[field] += per_object / EXPECTED_OBJECT_COUNT
        samples[method] = metric_samples

    def interval(values: np.ndarray) -> dict[str, float] | None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return {
            "lower_2.5_percent": float(np.percentile(finite, 2.5)),
            "median": float(np.percentile(finite, 50.0)),
            "upper_97.5_percent": float(np.percentile(finite, 97.5)),
        }

    method_intervals = {
        method: {field: interval(values) for field, values in fields.items()}
        for method, fields in samples.items()
    }
    paired_active_gains = {
        name: {
            "combined_gain_vs_fixed_k1": interval(
                samples[f"active_{name}"]["combined"]
                - samples["fixed_k1_target"]["combined"]
            ),
            "combined_gain_vs_matched_random": interval(
                samples[f"active_{name}"]["combined"]
                - samples[f"random_{name}"]["combined"]
            ),
        }
        for name in ACTIVE_POLICY_NAMES
    }
    return {
        "method": (
            "Object-stratified physical-instance bootstrap: resample holdout "
            "physical instances with replacement within each object, compute "
            "per-object metrics, then average all 15 objects equally."
        ),
        "seed": BOOTSTRAP_SEED,
        "replicate_count": replicates,
        "confidence_level": 0.95,
        "method_intervals": method_intervals,
        "paired_active_intervals": paired_active_gains,
    }


def _format_percent(value: Any, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+0.2%}" if signed else f"{float(value):0.2%}"


def _format_number(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _format_pp(value: Any, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    prefix = "+" if signed and float(value) >= 0.0 else ""
    return f"{prefix}{100.0 * float(value):.2f} pp"


def _ci_text(interval: Mapping[str, Any] | None, percent: bool = True) -> str:
    if interval is None:
        return "N/A"
    lower = interval["lower_2.5_percent"]
    upper = interval["upper_97.5_percent"]
    return (
        f"[{_format_percent(lower)}, {_format_percent(upper)}]"
        if percent
        else f"[{_format_number(lower)}, {_format_number(upper)}]"
    )


def _ci_pp_text(interval: Mapping[str, Any] | None) -> str:
    if interval is None:
        return "N/A"
    lower = 100.0 * float(interval["lower_2.5_percent"])
    upper = 100.0 * float(interval["upper_97.5_percent"])
    return f"[{lower:+.2f}, {upper:+.2f}] pp"


def markdown_report(summary: Mapping[str, Any]) -> str:
    summaries = {row["method"]: row for row in summary["policy_summaries"]}
    comparisons = summary["active_policy_comparisons"]
    bootstrap = summary["bootstrap"]
    confidence = summary["development"]["confidence_metrics"]
    frozen_points = summary["development"]["frozen_operating_points"]
    prefix_audit = summary["active_prefix_audit"]
    caveats = "\n".join(f"- **{item}**" for item in summary["caveats"])

    overall_rows = []
    display_methods = [
        "fixed_k1_target",
        "fixed_k3_medoid",
        "fixed_k5_medoid",
        "fixed_k3_max_mask",
        "fixed_k5_max_mask",
        "active_fast",
        "active_balanced",
        "active_conservative",
        "random_fast",
        "random_balanced",
        "random_conservative",
        "oracle_minimum_budget",
    ]
    for method in display_methods:
        row = summaries[method]
        macro = row["macro_object"]
        micro = row["micro"]
        overall_rows.append(
            f"| `{method}` | {_format_percent(macro['ar_mssd'])} | "
            f"{_format_percent(macro['ar_mspd'])} | "
            f"{_format_percent(macro['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_percent(micro['ar_mssd'])} | "
            f"{_format_percent(micro['ar_mspd'])} | "
            f"{_format_percent(micro['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_number(row['views']['mean'], 2)} / "
            f"{_format_number(row['views']['p50'], 2)} / "
            f"{_format_number(row['views']['p95'], 2)} | "
            f"{_format_number(row['registration_latency_seconds']['mean'])} / "
            f"{_format_number(row['registration_latency_seconds']['p50'])} / "
            f"{_format_number(row['registration_latency_seconds']['p95'])} |"
        )

    frozen_rows = [
        f"| {point['name']} | {point['t1']:.2f} | {point['t3']:.2f} | "
        f"{point['mean_view_cap']:.1f} | "
        f"{_format_percent(point['macro_object_combined'])} | "
        f"{point['mean_acquired_view_count']:.3f} |"
        for point in frozen_points
    ]
    active_rows = []
    stop_rows = []
    rescue_rows = []
    bootstrap_rows = [
        f"| `{method}` | "
        f"{_ci_text(bootstrap['method_intervals'][method]['combined'])} | "
        f"N/A | N/A |"
        for method in (
            "fixed_k1_target",
            "fixed_k3_medoid",
            "fixed_k5_medoid",
        )
    ]
    for name in ACTIVE_POLICY_NAMES:
        method = f"active_{name}"
        row = summaries[method]
        comp = comparisons[name]
        frontier = comp["linear_fixed_budget_mixture_frontier"]
        active_rows.append(
            f"| {name} | {_format_percent(row['macro_object']['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_number(row['views']['mean'], 3)} | "
            f"{_format_number(row['registration_latency_seconds']['mean'])} | "
            f"{_format_number(1000.0 * row['policy_overhead_seconds']['mean'], 3)} ms | "
            f"{_format_pp(comp['active_minus_matched_random_combined'])} | "
            f"{_format_percent(frontier['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_pp(comp['active_minus_mixture_frontier_combined'])} |"
        )
        stop = row["stopping_percentages"]
        stop_rows.append(
            f"| {name} | {stop['k1']:.2f}% | {stop['k3']:.2f}% | "
            f"{stop['k5']:.2f}% |"
        )
        joint = row["rescue_harm_vs_fixed_k1"]["joint"]
        rescue_rows.append(
            f"| {name} | {joint['rescued_count']}/{joint['baseline_failure_count']} "
            f"({_format_percent(joint['rescue_rate'])}) | "
            f"{joint['harmed_count']}/{joint['baseline_success_count']} "
            f"({_format_percent(joint['harm_rate'])}) |"
        )
        method_ci = bootstrap["method_intervals"][method]["combined"]
        gain_ci = bootstrap["paired_active_intervals"][name][
            "combined_gain_vs_fixed_k1"
        ]
        random_ci = bootstrap["paired_active_intervals"][name][
            "combined_gain_vs_matched_random"
        ]
        bootstrap_rows.append(
            f"| `active_{name}` | {_ci_text(method_ci)} | "
            f"{_ci_pp_text(gain_ci)} | {_ci_pp_text(random_ci)} |"
        )

    visibility_rows = []
    for method in display_methods:
        row = summaries[method]
        for label in VISIBILITY_BINS:
            value = row["by_target_visibility_bin"][label]
            macro = value["macro_object"]
            score = None if macro is None else macro["oracle_mask_subset_ar_mssd_mspd"]
            visibility_rows.append(
                f"| `{method}` | {label} | {value['sample_count']} | "
                f"{value['represented_object_count']} | {_format_percent(score)} |"
            )

    tradeoff_rows = []
    for name in ACTIVE_POLICY_NAMES:
        comp = comparisons[name]
        tradeoff_rows.append(
            f"| {name} | "
            f"{_format_pp(comp['gain_over_fixed_k1_combined'])} | "
            f"{_format_pp(comp['active_minus_fixed_k3_combined'])} | "
            f"{_format_pp(comp['active_minus_fixed_k5_combined'])} | "
            f"{_format_percent(comp['fixed_k5_gain_retained_fraction'])} | "
            f"{_format_percent(comp['registration_latency_reduction_vs_fixed_k5_fraction'])} |"
        )

    object_rows = []
    fixed1_objects = {
        int(row["object_id"]): row for row in summaries["fixed_k1_target"]["by_object"]
    }
    fixed5_objects = {
        int(row["object_id"]): row for row in summaries["fixed_k5_medoid"]["by_object"]
    }
    active_objects = {
        name: {
            int(row["object_id"]): row
            for row in summaries[f"active_{name}"]["by_object"]
        }
        for name in ACTIVE_POLICY_NAMES
    }
    for object_id in sorted(fixed1_objects):
        base = fixed1_objects[object_id]
        low_n = "yes" if base["low_n_descriptive"] else "no"
        object_rows.append(
            f"| {object_id} | {base['sample_count']} | {low_n} | "
            f"{_format_percent(base['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_percent(active_objects['fast'][object_id]['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_percent(active_objects['balanced'][object_id]['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_percent(active_objects['conservative'][object_id]['oracle_mask_subset_ar_mssd_mspd'])} | "
            f"{_format_percent(fixed5_objects[object_id]['oracle_mask_subset_ar_mssd_mspd'])} |"
        )
    per_object_detail_rows = []
    for method in display_methods:
        for row in summaries[method]["by_object"]:
            per_object_detail_rows.append(
                f"| `{method}` | {row['object_id']} | {row['sample_count']} | "
                f"{'yes' if row['low_n_descriptive'] else 'no'} | "
                f"{_format_percent(row['ar_mssd'])} | "
                f"{_format_percent(row['ar_mspd'])} | "
                f"{_format_percent(row['oracle_mask_subset_ar_mssd_mspd'])} |"
            )

    object15 = {
        method: next(
            row for row in summaries[method]["by_object"] if int(row["object_id"]) == 15
        )
        for method in (
            "fixed_k1_target",
            "active_fast",
            "active_balanced",
            "active_conservative",
            "fixed_k5_medoid",
        )
    }
    object15_text = (
        f"Object 15 has n={object15['fixed_k1_target']['sample_count']}: "
        f"fixed k=1 {_format_percent(object15['fixed_k1_target']['oracle_mask_subset_ar_mssd_mspd'])}, "
        f"fast {_format_percent(object15['active_fast']['oracle_mask_subset_ar_mssd_mspd'])}, "
        f"balanced {_format_percent(object15['active_balanced']['oracle_mask_subset_ar_mssd_mspd'])}, "
        f"conservative {_format_percent(object15['active_conservative']['oracle_mask_subset_ar_mssd_mspd'])}, "
        f"and fixed k=5 {_format_percent(object15['fixed_k5_medoid']['oracle_mask_subset_ar_mssd_mspd'])}."
    )

    fixed1 = summaries["fixed_k1_target"]["macro_object"][
        "oracle_mask_subset_ar_mssd_mspd"
    ]
    fixed5 = summaries["fixed_k5_medoid"]["macro_object"][
        "oracle_mask_subset_ar_mssd_mspd"
    ]
    fast = summaries["active_fast"]
    conservative = summaries["active_conservative"]
    answer = (
        f"Fixed k=5 improves macro-object combined score only "
        f"{_format_pp(fixed5 - fixed1)} over fixed k=1 "
        f"({_format_percent(fixed1)} to {_format_percent(fixed5)}). "
        f"Fast retains {_format_percent(comparisons['fast']['fixed_k5_gain_retained_fraction'])} "
        f"of that gain at {fast['views']['mean']:.2f} views and "
        f"{_format_percent(comparisons['fast']['registration_latency_reduction_vs_fixed_k5_fraction'])} "
        f"less registration latency; conservative retains "
        f"{_format_percent(comparisons['conservative']['fixed_k5_gain_retained_fraction'])} "
        f"at {conservative['views']['mean']:.2f} views. "
        f"However, fast beats the fixed-mixture frontier by only "
        f"{_format_pp(comparisons['fast']['active_minus_mixture_frontier_combined'])} "
        f"but trails matched random by "
        f"{_format_pp(-comparisons['fast']['active_minus_matched_random_combined'], signed=False)}; "
        f"balanced trails those controls by "
        f"{_format_pp(-comparisons['balanced']['active_minus_mixture_frontier_combined'], signed=False)} "
        f"and {_format_pp(-comparisons['balanced']['active_minus_matched_random_combined'], signed=False)}, "
        f"and conservative is "
        f"{_format_pp(comparisons['conservative']['active_minus_mixture_frontier_combined'])} "
        f"versus the mixture but trails random by "
        f"{_format_pp(-comparisons['conservative']['active_minus_matched_random_combined'], signed=False)}. "
        f"Thus the frozen confidence models reduce budget while retaining "
        f"near-k=5 accuracy at fast/conservative, but do not demonstrate a "
        f"reliable allocation advantage over budget-matched controls on this holdout."
    )
    freeze = summary["freeze_audit"]
    holdout = summary["holdout"]
    transform = summary["cross_view_gt_transform_check"]
    return f"""# PoseLoop M3 confidence-triggered active view budgeting

## Answer first

{answer}

## Scope and hard caveats

{caveats}

The unequal holdout contains **{holdout['holdout_physical_instance_count']}**
physical instances across all 15 objects, with development/holdout intersection
exactly `{holdout['intersection_count']}`. Its per-object counts are
`{holdout['group_count_by_object']}`. Target visibility-bin counts are
`{holdout['summary']['target_visibility_bin_counts']}`.

## Development confidence and immutable freeze

The grouped five-fold development split has zero physical-instance overlap in
every fold. M2-only out-of-fold diagnostics are:

The audited k=1 feature list is
`{summary['development']['feature_audit']['k1_feature_names']}`; the k=3 list is
`{summary['development']['feature_audit']['k3_feature_names']}`. The audit
records zero GT feature inputs and zero future-view access.

| Stage | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| k=1 | {confidence['k1']['auroc']:.4f} | {confidence['k1']['auprc']:.4f} | {confidence['k1']['brier_score']:.4f} | {confidence['k1']['ece']:.4f} |
| k=3 | {confidence['k3']['auroc']:.4f} | {confidence['k3']['auprc']:.4f} | {confidence['k3']['brier_score']:.4f} | {confidence['k3']['ece']:.4f} |

| Operating point | t1 | t3 | Development mean-view cap | Development macro combined | Development mean views |
|---|---:|---:|---:|---:|---:|
{chr(10).join(frozen_rows)}

Configuration hash: `{freeze['configuration_sha256']}`. It was frozen at
`{freeze['chronology']['frozen_utc']}`, before the prediction stream was created
at `{freeze['chronology']['prediction_stream_created_utc']}`.

![Development confidence calibration](m3_confidence_calibration.png)

## Holdout fixed and active results

Macro-object scores are primary. Micro AR_MSSD, AR_MSPD, and combined are shown
only as secondary unequal-count summaries.

| Method | Macro AR_MSSD | Macro AR_MSPD | Macro combined | Micro AR_MSSD | Micro AR_MSPD | Micro combined | Views mean / p50 / p95 | Registration seconds mean / p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_rows)}

![Accuracy versus acquired views](m3_accuracy_vs_views.png)

![Accuracy versus registration latency](m3_accuracy_vs_latency.png)

`oracle_minimum_budget` uses the final GT joint diagnostic to stop at the first
successful medoid budget in k=1/k=3/k=5 order, falling back to k=5 when none
succeeds. It is therefore a diagnostic upper bound on perfect success-aware
stopping, not a deployable policy.

## Budget controls

Matched random assigns exactly the active policy's k=1/k=3/k=5 counts using a
fixed seed. The linear frontier is the best convex interpolation of the
stronger medoid or max-mask fixed selector at k=3 and k=5 (plus target-only
k=1) at the same mean view count.
Active policy overhead measures NumPy feature/model decisions plus the actually
timed symmetry-aware medoid selections required by the reached prefix; it
excludes registration, I/O, and report/evaluation aggregation.

| Policy | Macro combined | Mean views | Mean registration seconds | Mean policy overhead | Gain vs matched random (pp) | Mixture frontier | Gain vs frontier (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(active_rows)}

| Policy | Gain vs fixed k=1 (pp) | Active minus fixed k=3 (pp) | Active minus fixed k=5 (pp) | Fixed-k5 gain retained | Latency reduction vs fixed k=5 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tradeoff_rows)}

## Stopping, rescue, and harm

| Policy | Stop k=1 | Stop k=3 | Stop k=5 |
|---|---:|---:|---:|
{chr(10).join(stop_rows)}

Joint rescue/harm uses the inclusive `<= 0.10d` and `<= 10r` diagnostic
thresholds and fixed k=1 as baseline.

| Policy | Joint rescue | Joint harm |
|---|---:|---:|
{chr(10).join(rescue_rows)}

## Visibility analysis

Target visibility is used only for post-hoc analysis. A missing stratum is
reported as `N/A`; no result is imputed.

| Policy | Visibility | n | Represented objects | Macro combined |
|---|---|---:|---:|---:|
{chr(10).join(visibility_rows)}

## Per-object results

Rows with `n < {LOW_N_THRESHOLD}` are explicitly low-n descriptive estimates.
Object 15 is included without special treatment.

| Object | n | Low-n | Fixed k=1 | Fast | Balanced | Conservative | Fixed k=5 |
|---:|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(object_rows)}

{object15_text}

![Per-object active-budget results](m3_per_object.png)

Complete per-object metric results:

| Method | Object | n | Low-n | AR_MSSD | AR_MSPD | Combined |
|---|---:|---:|---|---:|---:|---:|
{chr(10).join(per_object_detail_rows)}

## Object-stratified physical-instance bootstrap

The bootstrap resamples physical instances within each object and then
macro-averages the 15 object metrics. Intervals are percentile 95% intervals
from `{bootstrap['replicate_count']}` deterministic replicates.

| Method | Macro combined 95% CI | Gain vs fixed k=1 95% CI (pp) | Gain vs matched random 95% CI (pp) |
|---|---:|---:|---:|
{chr(10).join(bootstrap_rows)}

## Provenance, calibration, and timing

All `{summary['inference']['attempted_count']}` required holdout view rows were
attempted after the all-object pilot
(`{summary['inference']['pilot_candidate_count']}` rows across
`{summary['inference']['pilot_object_count']}` objects). Failure rows remain in
the denominator. Recorded statuses are
`{summary['inference']['status_counts']}`. Registration latency is the sum of
recorded per-view
FoundationPose registration times in fixed acquisition order; warm-up, I/O,
association, transforms, and policy overhead are excluded.

The imported feature-extractor source has SHA-256
`{freeze['feature_extractor_source']['sha256']}` and its recorded modification
time predates the policy freeze. Runtime assertions passed only rank `[0]` to
the k=1 extractor and ranks `[0, 1, 2]` to the k=3 extractor, with
`{prefix_audit['fast']['future_view_record_references_passed_to_extractor']}`
future-view records passed to either extractor.

Reapplying the calibrated cross-view transform to grouped GT poses gives maximum
normalized MSSD `{transform['normalized_mssd']['max']:.8f}` and maximum MSPD
`{transform['mspd_px']['max']:.8f}` px over
`{transform['sample_count']}` views, passing the unchanged M2 gate.

The official symmetry-aware evaluator uses BOP Toolkit commit
`{summary['bop_toolkit']['commit_sha']}`, `models_eval`, and
`max_sym_disc_step = {summary['bop_toolkit']['max_sym_disc_step']}`.
"""


def generate_figures(summary: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "fixed": "#0072B2",
        "fast": "#E69F00",
        "balanced": "#009E73",
        "conservative": "#D55E00",
        "random": "#999999",
        "oracle": "#CC79A7",
    }
    summaries = {row["method"]: row for row in summary["policy_summaries"]}
    comparisons = summary["active_policy_comparisons"]

    fixed_budgets = np.asarray([1.0, 3.0, 5.0])
    fixed_scores = np.asarray(
        [
            summaries["fixed_k1_target"]["macro_object"][
                "oracle_mask_subset_ar_mssd_mspd"
            ],
            summaries["fixed_k3_medoid"]["macro_object"][
                "oracle_mask_subset_ar_mssd_mspd"
            ],
            summaries["fixed_k5_medoid"]["macro_object"][
                "oracle_mask_subset_ar_mssd_mspd"
            ],
        ]
    )
    fixed_max_mask_scores = np.asarray(
        [
            fixed_scores[0],
            summaries["fixed_k3_max_mask"]["macro_object"][
                "oracle_mask_subset_ar_mssd_mspd"
            ],
            summaries["fixed_k5_max_mask"]["macro_object"][
                "oracle_mask_subset_ar_mssd_mspd"
            ],
        ]
    )
    figure, axis = plt.subplots(figsize=(8.8, 5.3))
    axis.plot(
        fixed_budgets,
        fixed_scores,
        marker="o",
        linewidth=2.0,
        color=colors["fixed"],
        label="fixed medoid budget",
    )
    axis.plot(
        fixed_budgets,
        fixed_max_mask_scores,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        color="#56B4E9",
        label="fixed max-mask budget",
    )
    for name, marker in zip(ACTIVE_POLICY_NAMES, ("^", "s", "D"), strict=True):
        active = summaries[f"active_{name}"]
        random = summaries[f"random_{name}"]
        axis.scatter(
            active["views"]["mean"],
            active["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            marker=marker,
            s=75,
            color=colors[name],
            label=f"active {name}",
            zorder=3,
        )
        axis.scatter(
            random["views"]["mean"],
            random["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            marker="x",
            s=50,
            color=colors[name],
            alpha=0.8,
            label="matched random" if name == "fast" else None,
        )
        frontier = comparisons[name]["linear_fixed_budget_mixture_frontier"]
        axis.scatter(
            active["views"]["mean"],
            frontier["oracle_mask_subset_ar_mssd_mspd"],
            marker="_",
            s=110,
            linewidths=2,
            color="black",
            zorder=4,
            label="fixed-mixture frontier" if name == "fast" else None,
        )
    axis.set(
        xlabel="Mean acquired views per target",
        ylabel="Macro-object combined score",
        title="M3 accuracy versus acquired views",
        xlim=(0.85, 5.15),
    )
    axis.legend(
        frameon=False,
        ncol=2,
        markerscale=0.7,
        handletextpad=0.9,
        columnspacing=1.3,
    )
    save_figure_atomic(figure, output_dir / "m3_accuracy_vs_views.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.4, 5.3))
    fixed_names = ("fixed_k1_target", "fixed_k3_medoid", "fixed_k5_medoid")
    fixed_max_names = (
        "fixed_k1_target",
        "fixed_k3_max_mask",
        "fixed_k5_max_mask",
    )
    axis.plot(
        [
            summaries[name]["registration_latency_seconds"]["mean"]
            for name in fixed_names
        ],
        [
            summaries[name]["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
            for name in fixed_names
        ],
        marker="o",
        linewidth=2,
        color=colors["fixed"],
        label="fixed medoid budget",
    )
    axis.plot(
        [
            summaries[name]["registration_latency_seconds"]["mean"]
            for name in fixed_max_names
        ],
        [
            summaries[name]["macro_object"]["oracle_mask_subset_ar_mssd_mspd"]
            for name in fixed_max_names
        ],
        marker="s",
        linewidth=1.5,
        linestyle="--",
        color="#56B4E9",
        label="fixed max-mask budget",
    )
    for name, marker in zip(ACTIVE_POLICY_NAMES, ("^", "s", "D"), strict=True):
        active = summaries[f"active_{name}"]
        random = summaries[f"random_{name}"]
        axis.scatter(
            active["registration_latency_seconds"]["mean"],
            active["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            marker=marker,
            s=75,
            color=colors[name],
            label=f"active {name}",
            zorder=3,
        )
        axis.scatter(
            random["registration_latency_seconds"]["mean"],
            random["macro_object"]["oracle_mask_subset_ar_mssd_mspd"],
            marker="x",
            s=50,
            color=colors[name],
            alpha=0.8,
            label="matched random" if name == "fast" else None,
        )
    axis.set(
        xlabel="Mean registration latency per target (s)",
        ylabel="Macro-object combined score",
        title="M3 accuracy versus measured registration latency",
    )
    axis.legend(
        frameon=False,
        ncol=2,
        markerscale=0.7,
        handletextpad=0.9,
        columnspacing=1.3,
    )
    save_figure_atomic(figure, output_dir / "m3_accuracy_vs_latency.png")
    plt.close(figure)

    confidence = summary["development"]["confidence_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharex=True, sharey=True)
    for axis, stage, color in zip(
        axes,
        ("k1", "k3"),
        (colors["fixed"], colors["balanced"]),
        strict=True,
    ):
        bins = [
            row
            for row in confidence[stage]["calibration_bins"]
            if int(row["count"]) > 0
        ]
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1,
            color="#666666",
            label="perfect calibration",
        )
        axis.plot(
            [row["mean_probability"] for row in bins],
            [row["empirical_positive_rate"] for row in bins],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{stage} OOF",
        )
        for row in bins:
            axis.annotate(
                f"n={row['count']}",
                (
                    row["mean_probability"],
                    row["empirical_positive_rate"],
                ),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Empirical joint success rate" if stage == "k1" else None,
            title=f"{stage} grouped OOF calibration",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(frameon=False, loc="upper left")
    save_figure_atomic(figure, output_dir / "m3_confidence_calibration.png")
    plt.close(figure)

    object_ids = [
        int(row["object_id"]) for row in summaries["fixed_k1_target"]["by_object"]
    ]
    method_names = (
        "fixed_k1_target",
        "active_balanced",
        "fixed_k5_medoid",
    )
    labels = ("fixed k=1", "active balanced", "fixed k=5")
    method_colors = (colors["fixed"], colors["balanced"], colors["oracle"])
    x = np.arange(len(object_ids))
    width = 0.26
    figure, axis = plt.subplots(figsize=(11.2, 5.3))
    for index, (method, label, color) in enumerate(
        zip(method_names, labels, method_colors, strict=True)
    ):
        values = [
            row["oracle_mask_subset_ar_mssd_mspd"]
            for row in summaries[method]["by_object"]
        ]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
    low_n_ids = {
        int(row["object_id"])
        for row in summaries["fixed_k1_target"]["by_object"]
        if row["low_n_descriptive"]
    }
    for index, object_id in enumerate(object_ids):
        if object_id in low_n_ids:
            axis.text(index, 1.015, "*", ha="center", va="bottom", fontsize=13)
    axis.set(
        xlabel="XYZ-IBD object ID",
        ylabel="Combined score",
        title="M3 per-object results (* low-n descriptive)",
        xticks=x,
        xticklabels=[str(value) for value in object_ids],
        ylim=(0, 1.08),
    )
    axis.legend(frameon=False)
    save_figure_atomic(figure, output_dir / "m3_per_object.png")
    plt.close(figure)

    expected = {
        "m3_accuracy_vs_views.png",
        "m3_accuracy_vs_latency.png",
        "m3_confidence_calibration.png",
        "m3_per_object.png",
    }
    actual = {path.name for path in output_dir.glob("m3_*.png")}
    if actual != expected:
        raise RuntimeError(
            f"Expected exactly four M3 PNGs; expected={expected}, actual={actual}"
        )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    raw_paths = {
        "groups": _artifact_path(args.groups, repo_root),
        "candidate_manifest": _artifact_path(args.candidate_manifest, repo_root),
        "holdout_summary": _artifact_path(args.holdout_summary, repo_root),
        "leakage_audit": _artifact_path(args.leakage_audit, repo_root),
        "predictions": _artifact_path(args.predictions, repo_root),
        "frozen_policy": _artifact_path(args.frozen_policy, repo_root),
        "feature_audit": _artifact_path(args.feature_audit, repo_root),
        "fold_audit": _artifact_path(args.fold_audit, repo_root),
        "development_confidence": _artifact_path(
            args.development_confidence, repo_root
        ),
        "development_oof": _artifact_path(args.development_oof, repo_root),
        "metrics_output": _artifact_path(args.metrics_output, repo_root),
        "summary_output": _artifact_path(args.summary_output, repo_root),
        "m2_groups": args.m2_groups.resolve(),
        "m2_metrics": args.m2_metrics.resolve(),
    }
    required_inputs = {
        key: value
        for key, value in raw_paths.items()
        if key not in {"metrics_output", "summary_output"}
    }
    missing = [str(path) for path in required_inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing M3 evaluation inputs: {missing}")
    if args.bootstrap_replicates < 200:
        raise ValueError("--bootstrap-replicates must be at least 200")

    candidate_rows = load_jsonl(raw_paths["candidate_manifest"])
    candidate_index = validate_candidate_manifest(candidate_rows)
    groups = validate_groups(load_jsonl(raw_paths["groups"]), candidate_index)
    holdout = validate_holdout_evidence(raw_paths, groups, candidate_index)
    frozen, development = validate_frozen_policy(raw_paths)
    prediction_metadata, predictions, inference = load_predictions(
        raw_paths["predictions"],
        candidate_index,
        groups,
        raw_paths["groups"],
        raw_paths["candidate_manifest"],
    )
    chronology = validate_freeze_precedes_predictions(frozen, prediction_metadata)
    fit_policy_source = (SCRIPT_DIR / "fit_m3_policy.py").resolve()
    fit_policy_modified_utc = datetime.fromtimestamp(
        fit_policy_source.stat().st_mtime,
        timezone.utc,
    )
    frozen_utc = _parse_utc(frozen.get("frozen_utc"), "frozen_utc")
    if not fit_policy_modified_utc < frozen_utc:
        raise ValueError(
            "Confidence feature extractor source does not predate the frozen policy"
        )
    feature_extractor_source = {
        "path": str(fit_policy_source),
        "sha256": sha256_file(fit_policy_source),
        "last_modified_utc": fit_policy_modified_utc.isoformat(),
        "last_modified_precedes_freeze": True,
        "runtime_feature_names_match_frozen_configuration": True,
    }

    # Reuse the exact M2 transformation/evaluation implementation. The copy
    # changes only the internal dispatch label expected by that implementation.
    m2_compatible_groups = copy.deepcopy(groups)
    for group in m2_compatible_groups:
        for view in group["views"]:
            view["prediction_source"] = "m2"
    model_params, model_info = load_official_models(args.dataset_root.resolve())
    toolkit_sha = toolkit_commit(args.toolkit_root.resolve())
    prepared, transform_audit = prepare_groups(
        m2_compatible_groups,
        candidate_index,
        {},
        predictions,
        model_params,
        model_info,
        GT_TRANSFORM_MAX_NORMALIZED_MSSD,
        GT_TRANSFORM_MAX_MSPD_PX,
    )
    # Restore M3 group metadata required by policy feature extraction/reporting.
    original_by_id = {str(group["group_id"]): group for group in groups}
    for item in prepared:
        item["group"] = original_by_id[str(item["group"]["group_id"])]

    fixed_index, fixed_rows = fixed_results(prepared)
    active_sets, active_audit = active_results(groups, fixed_index, frozen)
    random_sets, random_audit = random_budget_results(groups, fixed_index, active_sets)
    oracle_rows = oracle_minimum_budget_results(groups, fixed_index)

    result_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed_rows:
        result_sets[str(row["method"])].append(row)
    result_sets.update(active_sets)
    result_sets.update(random_sets)
    result_sets["oracle_minimum_budget"] = oracle_rows
    expected_methods = (
        set(FIXED_METHODS)
        | {f"active_{name}" for name in ACTIVE_POLICY_NAMES}
        | {f"random_{name}" for name in ACTIVE_POLICY_NAMES}
        | {"oracle_minimum_budget"}
    )
    if set(result_sets) != expected_methods:
        raise AssertionError("M3 evaluated-method set is incomplete")
    baseline = result_sets["fixed_k1_target"]
    summaries = {
        method: summarize_policy(method, rows, baseline)
        for method, rows in sorted(result_sets.items())
    }
    comparisons = add_policy_comparisons(summaries)
    bootstrap = bootstrap_confidence_intervals(result_sets, args.bootstrap_replicates)

    metric_rows = [
        row
        for method in sorted(result_sets)
        for row in sorted(result_sets[method], key=lambda item: str(item["group_id"]))
    ]
    write_jsonl_atomic(raw_paths["metrics_output"], metric_rows)
    policy_summaries = [summaries[method] for method in sorted(summaries)]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PoseLoop-AB confidence-triggered active view budgeting",
        "primary_aggregation": "macro_object",
        "holdout": holdout,
        "development": {
            "source": "M2 only",
            "confidence_metrics": development["confidence_metrics"],
            "fold_audit": development["fold_audit"],
            "frozen_operating_points": frozen["policy"]["operating_points"],
            "feature_audit": frozen["feature_audit"],
        },
        "freeze_audit": {
            "configuration_sha256": development["configuration_sha256"],
            "chronology": chronology,
            "feature_extractor_source": feature_extractor_source,
        },
        "inference": inference,
        "active_prefix_audit": active_audit,
        "matched_random_audit": random_audit,
        "cross_view_gt_transform_check": transform_audit,
        "bop_toolkit": {
            "commit_sha": toolkit_sha,
            "model_type": "models_eval",
            "max_sym_disc_step": MAX_SYMMETRY_DISCRETIZATION_STEP,
        },
        "policy_summaries": policy_summaries,
        "active_policy_comparisons": comparisons,
        "bootstrap": bootstrap,
        "caveats": list(REQUIRED_CAVEATS),
        "provenance": {
            "evaluate_m3_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "fit_m3_policy_source": feature_extractor_source,
            **{
                key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for key, path in required_inputs.items()
            },
        },
        "outputs": {
            "metrics": str(raw_paths["metrics_output"]),
            "metrics_sha256": sha256_file(raw_paths["metrics_output"]),
            "summary": str(raw_paths["summary_output"]),
            "report": str(args.report.resolve()),
            "figures": [
                str((args.figures_dir / name).resolve())
                for name in (
                    "m3_accuracy_vs_views.png",
                    "m3_accuracy_vs_latency.png",
                    "m3_confidence_calibration.png",
                    "m3_per_object.png",
                )
            ],
        },
    }
    write_json_atomic(raw_paths["summary_output"], summary)
    write_text_atomic(args.report.resolve(), markdown_report(summary))
    generate_figures(summary, args.figures_dir.resolve())

    print(
        "M3 evaluation complete: "
        f"holdout={len(groups)} methods={len(result_sets)} "
        f"metric_rows={len(metric_rows)} bootstrap={args.bootstrap_replicates}",
        flush=True,
    )
    for name in ACTIVE_POLICY_NAMES:
        row = summaries[f"active_{name}"]
        print(
            f"{name}: macro_combined="
            f"{row['macro_object']['oracle_mask_subset_ar_mssd_mspd']:.6f} "
            f"mean_views={row['views']['mean']:.6f} "
            f"mean_seconds={row['registration_latency_seconds']['mean']:.6f}",
            flush=True,
        )
    print(f"saved: {raw_paths['summary_output']}", flush=True)
    print(f"report: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
