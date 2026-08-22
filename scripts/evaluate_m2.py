#!/usr/bin/env python3
"""Evaluate the PoseLoop M2 calibrated multi-view rescue diagnostic."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from bop_toolkit_lib import pose_error

# M1 scripts are executable modules rather than an installed Python package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_m1 import (
    MAX_SYMMETRY_DISCRETIZATION_STEP,
    MSPD_THRESHOLD_MULTIPLIERS,
    MSSD_THRESHOLD_FRACTIONS,
    aggregate,
    finite_or_none,
    load_object_evaluation_data,
    load_official_models,
    official_errors,
    percentile_summary,
    toolkit_commit,
    validate_manifest,
)
from m1_common import (
    ALLOWED_STATUSES,
    BATCH_SCHEMA_VERSION,
    GT_ROTATION_ATOL,
    assert_pose,
    load_jsonl,
    raw_pose_errors,
    read_json,
    sha256_file,
    stable_sample_id,
    write_json_atomic,
    write_jsonl_atomic,
)
from m2_common import (
    GT_AUDIT_MAX_MSPD_PX,
    GT_AUDIT_MAX_NORMALIZED_MSSD,
)


SCHEMA_VERSION = 1
VIEW_BUDGETS = (1, 2, 3, 5)
METHODS = (
    "target_only",
    "max_mask_area",
    "symmetry_aware_medoid",
    "oracle_max_visibility",
    "oracle_any_view",
)
VISIBILITY_BINS = ("low", "mid", "high")
M1_EXPECTED_COMBINED = 0.6686666666666666
M1_REPRODUCTION_ATOL = 1e-12
GT_TRANSFORM_MAX_NORMALIZED_MSSD = GT_AUDIT_MAX_NORMALIZED_MSSD
GT_TRANSFORM_MAX_MSPD_PX = GT_AUDIT_MAX_MSPD_PX
DIAGNOSTIC_MSSD_THRESHOLD = 0.10
DIAGNOSTIC_MSPD_MULTIPLIER = 10

CAVEATS = (
    "Known object IDs and ground-truth visible masks are used.",
    "Ground truth is used for oracle cross-view physical-instance association.",
    "The target set is exactly the same 300 RealSense target views used in M1.",
    "This oracle-mask subset is not a full BOP detection submission.",
    "oracle_max_visibility and oracle_any_view are non-deployable diagnostic "
    "upper bounds.",
    "These scores are not directly comparable to BOP Industrial detection AP.",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
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
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--groups-summary",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups_summary.json",
    )
    parser.add_argument(
        "--extrinsics-audit",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "extrinsics_audit.json",
    )
    parser.add_argument(
        "--m1-manifest",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--m1-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "predictions.jsonl",
    )
    parser.add_argument(
        "--m1-summary",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "summary.json",
    )
    parser.add_argument(
        "--view-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "view_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "metrics.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "m2_multiview.md",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "reports",
    )
    parser.add_argument(
        "--max-gt-transform-normalized-mssd",
        type=float,
        default=GT_TRANSFORM_MAX_NORMALIZED_MSSD,
    )
    parser.add_argument(
        "--max-gt-transform-mspd-px",
        type=float,
        default=GT_TRANSFORM_MAX_MSPD_PX,
    )
    return parser.parse_args()


def load_m1_predictions(
    path: Path,
    manifest_index: dict[str, dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Use M1's strict prediction contract without importing its CLI entrypoint."""
    from evaluate_m1 import load_predictions

    return load_predictions(path, manifest_index, manifest_sha256)


def load_view_predictions(
    path: Path,
    candidate_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("M2 view predictions must start with metadata")
    metadata_rows = [row for row in rows if row.get("record_type") == "metadata"]
    prediction_rows = [row for row in rows if row.get("record_type") == "prediction"]
    unexpected = [
        row.get("record_type")
        for row in rows
        if row.get("record_type") not in {"metadata", "prediction"}
    ]
    if len(metadata_rows) != 1:
        raise ValueError(
            "M2 view predictions must contain exactly one metadata row, "
            f"found {len(metadata_rows)}"
        )
    if unexpected:
        raise ValueError(f"Unexpected M2 prediction record types: {unexpected}")
    metadata = metadata_rows[0]
    if (
        int(metadata.get("schema_version", -1)) != SCHEMA_VERSION
        or not metadata.get("batch_fingerprint")
    ):
        raise ValueError("Invalid M2 prediction metadata schema/fingerprint")

    predictions: dict[str, dict[str, Any]] = {}
    for line_index, row in enumerate(prediction_rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Missing sample ID in M2 prediction row {line_index}")
        if sample_id in predictions:
            raise ValueError(f"Duplicate M2 prediction sample ID: {sample_id}")
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid M2 status for {sample_id}: {status}")
        if int(row.get("schema_version", -1)) != BATCH_SCHEMA_VERSION:
            raise ValueError(f"Invalid M2 prediction schema for {sample_id}")
        if sample_id not in candidate_index:
            raise ValueError(f"M2 prediction is absent from candidate manifest: {sample_id}")
        candidate = candidate_index[sample_id]
        for field in (
            "scene_id",
            "image_id",
            "gt_instance_index",
            "object_id",
            "sensor_modality",
        ):
            if row.get(field) != candidate.get(field):
                raise ValueError(
                    f"M2 prediction/candidate identity mismatch for "
                    f"{sample_id}: {field}"
                )
        stored_gt = np.asarray(
            row.get("gt_model_to_camera_pose_m"),
            dtype=np.float64,
        )
        candidate_gt = np.asarray(
            candidate["gt_model_to_camera_pose_m"],
            dtype=np.float64,
        )
        if stored_gt.shape != (4, 4) or not np.allclose(
            stored_gt,
            candidate_gt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"M2 prediction/candidate GT mismatch for {sample_id}")
        exact_fields = ("visibility_bin", "visible_mask_pixel_count")
        for field in exact_fields:
            if row.get(field) != candidate.get(field):
                raise ValueError(
                    f"M2 prediction/candidate field mismatch for "
                    f"{sample_id}: {field}"
                )
        numeric_fields = (
            "visible_fraction",
            "valid_depth_ratio_inside_mask",
        )
        for field in numeric_fields:
            if not math.isclose(
                float(row.get(field)),
                float(candidate.get(field)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"M2 prediction/candidate numeric mismatch for "
                    f"{sample_id}: {field}"
                )
        if status == "success":
            if "predicted_model_to_camera_pose_m" not in row:
                raise ValueError(f"Successful M2 prediction has no pose: {sample_id}")
            assert_pose(
                np.asarray(
                    row["predicted_model_to_camera_pose_m"], dtype=np.float64
                ),
                f"M2 prediction {sample_id}",
            )
            predicted_pose = np.asarray(
                row["predicted_model_to_camera_pose_m"],
                dtype=np.float64,
            )
            expected_translation_mm, expected_rotation_degrees = raw_pose_errors(
                predicted_pose,
                candidate_gt,
            )
            if not math.isclose(
                float(row.get("translation_error_mm")),
                expected_translation_mm,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ) or not math.isclose(
                float(row.get("raw_rotation_error_degrees")),
                expected_rotation_degrees,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"M2 stored diagnostic errors mismatch for {sample_id}"
                )
        elif "predicted_model_to_camera_pose_m" in row:
            raise ValueError(
                f"Failed M2 prediction unexpectedly has a pose: {sample_id}"
            )
        runtime = finite_or_none(row.get("registration_seconds"))
        if row.get("registration_seconds") is not None and (
            runtime is None or runtime < 0
        ):
            raise ValueError(f"Invalid M2 registration runtime for {sample_id}")
        predictions[sample_id] = row

    required_sample_ids = set(candidate_index)
    missing = required_sample_ids - set(predictions)
    extra = set(predictions) - required_sample_ids
    if missing or extra:
        raise ValueError(
            "M2 candidate predictions must attempt every deduplicated M2-source "
            f"row exactly once; missing={len(missing)}, extra={len(extra)}"
        )
    return metadata, predictions


def load_candidate_manifest(
    path: Path,
    required_sample_ids: set[str],
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        expected = stable_sample_id(
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        if not sample_id or sample_id != expected or sample_id in indexed:
            raise ValueError(
                f"Invalid/duplicate M2 candidate identity at row {row_number}"
            )
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            f"M2 candidate GT {sample_id}",
            GT_ROTATION_ATOL,
        )
        indexed[sample_id] = row
    if set(indexed) != required_sample_ids:
        raise ValueError(
            "Candidate manifest identities do not exactly match M2 group sources"
        )
    return indexed


def validate_m2_provenance(
    metadata: dict[str, Any],
    required_paths: dict[str, Path],
    candidate_index: dict[str, dict[str, Any]],
) -> None:
    expected_hash_fields = {
        "candidate_manifest_sha256": "candidate_manifest",
        "groups_sha256": "groups",
        "m1_manifest_sha256": "m1_manifest",
        "m1_predictions_sha256": "m1_predictions",
    }
    for metadata_field, path_key in expected_hash_fields.items():
        expected_hash = sha256_file(required_paths[path_key])
        if metadata.get(metadata_field) != expected_hash:
            raise ValueError(
                f"M2 prediction provenance mismatch: {metadata_field}"
            )
    if int(metadata.get("candidate_universe_sample_count", -1)) != len(
        candidate_index
    ):
        raise ValueError("M2 prediction candidate-universe count mismatch")

    groups_summary = read_json(required_paths["groups_summary"])
    if (
        int(groups_summary.get("group_count", -1)) != 300
        or int(groups_summary.get("deduplicated_missing_candidate_count", -1))
        != len(candidate_index)
    ):
        raise ValueError("M2 groups summary count contract failed")
    extrinsics_audit = read_json(required_paths["extrinsics_audit"])
    if (
        extrinsics_audit.get("status") != "pass"
        or int(extrinsics_audit.get("target_count", -1)) != 300
        or int(
            extrinsics_audit.get("official_bop_gt_transform_residual", {}).get(
                "outlier_count",
                -1,
            )
        )
        != 0
    ):
        raise ValueError("Saved calibrated-extrinsics audit is not a clean pass")


def validate_groups(
    rows: list[dict[str, Any]],
    manifest_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    if len(rows) != 300:
        raise ValueError(f"M2 requires exactly 300 target groups, found {len(rows)}")

    groups: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    target_ids: set[str] = set()
    m2_source_ids: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        group_id = str(row.get("group_id", ""))
        target_id = str(row.get("target_sample_id", ""))
        if not group_id or group_id in group_ids:
            raise ValueError(f"Missing or duplicate group_id at row {row_index}")
        if target_id not in manifest_index or target_id in target_ids:
            raise ValueError(f"Unknown or duplicate target sample ID: {target_id}")
        group_ids.add(group_id)
        target_ids.add(target_id)

        manifest_row = manifest_index[target_id]
        object_id = int(row["object_id"])
        scene_id = int(row["scene_id"])
        if object_id != int(manifest_row["object_id"]):
            raise ValueError(f"Target object mismatch in group {group_id}")
        if scene_id != int(manifest_row["scene_id"]):
            raise ValueError(f"Target scene mismatch in group {group_id}")
        if row["target_visibility_bin"] != manifest_row["visibility_bin"]:
            raise ValueError(f"Target visibility-bin mismatch in group {group_id}")

        raw_views = row.get("views")
        if not isinstance(raw_views, list) or not 1 <= len(raw_views) <= 5:
            raise ValueError(f"Group {group_id} must contain 1 to 5 views")
        ranks = [int(view["acquisition_rank"]) for view in raw_views]
        if sorted(ranks) != list(range(len(raw_views))):
            raise ValueError(f"Non-consecutive acquisition ranks in group {group_id}")
        views = sorted(raw_views, key=lambda view: int(view["acquisition_rank"]))
        if str(views[0]["sample_id"]) != target_id:
            raise ValueError(f"Rank-zero view is not the target in group {group_id}")
        if views[0].get("prediction_source") != "m1":
            raise ValueError(f"Rank-zero target must use M1 prediction in {group_id}")

        seen_samples: set[str] = set()
        for view in views:
            sample_id = str(view.get("sample_id", ""))
            if not sample_id or sample_id in seen_samples:
                raise ValueError(f"Missing or duplicate view in group {group_id}")
            seen_samples.add(sample_id)
            if int(view["object_id"]) != object_id:
                raise ValueError(f"View object mismatch in group {group_id}")
            if int(view["scene_id"]) != scene_id:
                raise ValueError(f"Cross-scene view in group {group_id}")
            if int(view["visible_mask_pixel_count"]) <= 0:
                raise ValueError(f"Empty visible mask in group {group_id}")
            visible_fraction = float(view["visible_fraction"])
            if not 0.10 <= visible_fraction <= 1.01:
                raise ValueError(f"Invalid view visibility in group {group_id}")
            source = view.get("prediction_source")
            if source not in {"m1", "m2"}:
                raise ValueError(f"Unknown prediction source in group {group_id}")
            if source == "m1" and sample_id not in manifest_index:
                raise ValueError(f"Unknown M1-source view {sample_id}")
            if source == "m2":
                m2_source_ids.add(sample_id)
            assert_pose(
                np.asarray(
                    view["camera_world_to_camera_pose_m"], dtype=np.float64
                ),
                f"world-to-camera pose {sample_id}",
                GT_ROTATION_ATOL,
            )
            assert_pose(
                np.asarray(view["gt_model_to_camera_pose_m"], dtype=np.float64),
                f"view GT pose {sample_id}",
                GT_ROTATION_ATOL,
            )

        target_gt = np.asarray(
            views[0]["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        manifest_gt = np.asarray(
            manifest_row["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        if not np.allclose(target_gt, manifest_gt, rtol=0.0, atol=1e-8):
            raise ValueError(f"Target GT differs from M1 manifest in group {group_id}")

        normalized = dict(row)
        normalized["views"] = views
        groups.append(normalized)

    if target_ids != set(manifest_index):
        raise ValueError(
            "M2 group targets do not exactly cover the M1 manifest; "
            f"missing={len(set(manifest_index) - target_ids)}, "
            f"extra={len(target_ids - set(manifest_index))}"
        )
    return groups, m2_source_ids


def prediction_for_view(
    view: dict[str, Any],
    m1_predictions: dict[str, dict[str, Any]],
    m2_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = view["prediction_source"]
    sample_id = str(view["sample_id"])
    prediction = (
        m1_predictions[sample_id] if source == "m1" else m2_predictions[sample_id]
    )
    for field in ("scene_id", "image_id", "gt_instance_index", "object_id"):
        if field in prediction and int(prediction[field]) != int(view[field]):
            raise ValueError(
                f"Prediction/view {field} mismatch for {sample_id}: "
                f"{prediction[field]} != {view[field]}"
            )
    return prediction


def transform_to_target(
    model_to_camera_pose_m: np.ndarray,
    camera_world_to_camera_pose_m: np.ndarray,
    target_world_to_camera_pose_m: np.ndarray,
) -> np.ndarray:
    model_to_camera_pose_m = np.asarray(
        model_to_camera_pose_m, dtype=np.float64
    )
    camera_world_to_camera_pose_m = np.asarray(
        camera_world_to_camera_pose_m, dtype=np.float64
    )
    target_world_to_camera_pose_m = np.asarray(
        target_world_to_camera_pose_m, dtype=np.float64
    )
    assert_pose(model_to_camera_pose_m, "model-to-camera pose")
    assert_pose(
        camera_world_to_camera_pose_m,
        "candidate world-to-camera pose",
        GT_ROTATION_ATOL,
    )
    assert_pose(
        target_world_to_camera_pose_m,
        "target world-to-camera pose",
        GT_ROTATION_ATOL,
    )
    model_to_world = (
        np.linalg.inv(camera_world_to_camera_pose_m) @ model_to_camera_pose_m
    )
    result = target_world_to_camera_pose_m @ model_to_world
    assert_pose(result, "transformed model-to-target-camera pose")
    return result


def symmetric_normalized_mssd(
    first_pose_m: np.ndarray,
    second_pose_m: np.ndarray,
    object_data: dict[str, Any],
) -> float:
    first_pose_m = np.asarray(first_pose_m, dtype=np.float64)
    second_pose_m = np.asarray(second_pose_m, dtype=np.float64)
    first_translation_mm = first_pose_m[:3, 3:4] * 1000.0
    second_translation_mm = second_pose_m[:3, 3:4] * 1000.0
    forward_mm = float(
        pose_error.mssd(
            first_pose_m[:3, :3],
            first_translation_mm,
            second_pose_m[:3, :3],
            second_translation_mm,
            object_data["points_mm"],
            object_data["symmetries"],
        )
    )
    reverse_mm = float(
        pose_error.mssd(
            second_pose_m[:3, :3],
            second_translation_mm,
            first_pose_m[:3, :3],
            first_translation_mm,
            object_data["points_mm"],
            object_data["symmetries"],
        )
    )
    result = (forward_mm + reverse_mm) / (2.0 * object_data["diameter_mm"])
    if not math.isfinite(result):
        raise ValueError("Pairwise official MSSD returned a non-finite value")
    return result


def summarize_values(values: Iterable[Any]) -> dict[str, Any]:
    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    base = percentile_summary(finite_values)
    base["mean"] = (
        float(np.mean(finite_values)) if finite_values else None
    )
    base["min"] = min(finite_values) if finite_values else None
    base["max"] = max(finite_values) if finite_values else None
    return base


def prepare_groups(
    groups: list[dict[str, Any]],
    manifest_index: dict[str, dict[str, Any]],
    m1_predictions: dict[str, dict[str, Any]],
    m2_predictions: dict[str, dict[str, Any]],
    model_params: dict[str, Any],
    model_info: dict[int, Any],
    max_gt_normalized_mssd: float,
    max_gt_mspd_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_cache: dict[int, dict[str, Any]] = {}
    prepared_groups: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, start=1):
        target_id = str(group["target_sample_id"])
        target_manifest = manifest_index[target_id]
        object_id = int(group["object_id"])
        if object_id not in object_cache:
            object_cache[object_id] = load_object_evaluation_data(
                object_id, model_params, model_info
            )
        object_data = object_cache[object_id]
        camera_matrix = np.asarray(
            target_manifest["camera_intrinsics_row_major"], dtype=np.float64
        )
        image_width = int(target_manifest["image_width"])
        mspd_scale_r = image_width / 640.0
        target_gt = np.asarray(
            target_manifest["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        target_world_to_camera = np.asarray(
            group["views"][0]["camera_world_to_camera_pose_m"],
            dtype=np.float64,
        )
        prepared_views: list[dict[str, Any]] = []
        for view in group["views"]:
            prediction = prediction_for_view(
                view, m1_predictions, m2_predictions
            )
            transformed_gt = transform_to_target(
                np.asarray(view["gt_model_to_camera_pose_m"], dtype=np.float64),
                np.asarray(
                    view["camera_world_to_camera_pose_m"], dtype=np.float64
                ),
                target_world_to_camera,
            )
            gt_mssd_mm, gt_mspd_px = official_errors(
                transformed_gt,
                target_gt,
                camera_matrix,
                object_data,
            )
            gt_normalized_mssd = gt_mssd_mm / object_data["diameter_mm"]
            residual_rows.append(
                {
                    "group_id": group["group_id"],
                    "sample_id": view["sample_id"],
                    "acquisition_rank": int(view["acquisition_rank"]),
                    "normalized_mssd": gt_normalized_mssd,
                    "mspd_px": gt_mspd_px,
                }
            )

            status = str(prediction["status"])
            transformed_prediction: np.ndarray | None = None
            normalized_mssd: float | None = None
            mspd_px: float | None = None
            if status == "success":
                transformed_prediction = transform_to_target(
                    np.asarray(
                        prediction["predicted_model_to_camera_pose_m"],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        view["camera_world_to_camera_pose_m"], dtype=np.float64
                    ),
                    target_world_to_camera,
                )
                if int(view["acquisition_rank"]) == 0:
                    original_prediction = np.asarray(
                        prediction["predicted_model_to_camera_pose_m"],
                        dtype=np.float64,
                    )
                    if not np.allclose(
                        transformed_prediction,
                        original_prediction,
                        rtol=0.0,
                        atol=1e-8,
                    ):
                        raise RuntimeError(
                            f"Rank-zero transform is not identity for {target_id}"
                        )
                mssd_mm, mspd_px = official_errors(
                    transformed_prediction,
                    target_gt,
                    camera_matrix,
                    object_data,
                )
                normalized_mssd = mssd_mm / object_data["diameter_mm"]

            prepared_views.append(
                {
                    "view": view,
                    "prediction": prediction,
                    "status": status,
                    "finite_pose": transformed_prediction is not None,
                    "transformed_pose_m": transformed_prediction,
                    "normalized_mssd": normalized_mssd,
                    "mspd_px": mspd_px,
                    "registration_seconds": finite_or_none(
                        prediction.get("registration_seconds")
                    ),
                }
            )

        prepared_groups.append(
            {
                "group": group,
                "target_manifest": target_manifest,
                "target_gt_pose_m": target_gt,
                "camera_matrix": camera_matrix,
                "mspd_scale_r": mspd_scale_r,
                "object_data": object_data,
                "views": prepared_views,
            }
        )
        if group_index % 25 == 0 or group_index == len(groups):
            print(
                f"transform and official view metrics: "
                f"{group_index}/{len(groups)}",
                flush=True,
            )

    residual_summary = {
        "sample_count": len(residual_rows),
        "normalized_mssd": summarize_values(
            row["normalized_mssd"] for row in residual_rows
        ),
        "mspd_px": summarize_values(row["mspd_px"] for row in residual_rows),
        "acceptance_limits": {
            "max_normalized_mssd": max_gt_normalized_mssd,
            "max_mspd_px": max_gt_mspd_px,
        },
    }
    observed_mssd = residual_summary["normalized_mssd"]["max"]
    observed_mspd = residual_summary["mspd_px"]["max"]
    if (
        observed_mssd is None
        or observed_mspd is None
        or observed_mssd > max_gt_normalized_mssd
        or observed_mspd > max_gt_mspd_px
    ):
        raise RuntimeError(
            "Cross-view GT transformation residual exceeds the fixed M2 "
            "evaluation gate: "
            f"max normalized MSSD={observed_mssd}, "
            f"max MSPD={observed_mspd}px"
        )
    residual_summary["passed"] = True
    return prepared_groups, residual_summary


def choose_medoid(
    acquired: list[dict[str, Any]],
    object_data: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    successful = [view for view in acquired if view["finite_pose"]]
    if not successful:
        return None, {}
    if len(successful) == 1:
        sample_id = str(successful[0]["view"]["sample_id"])
        return successful[0], {sample_id: 0.0}

    scores: dict[str, float] = {}
    for candidate in successful:
        disagreements = [
            symmetric_normalized_mssd(
                candidate["transformed_pose_m"],
                other["transformed_pose_m"],
                object_data,
            )
            for other in successful
            if other is not candidate
        ]
        scores[str(candidate["view"]["sample_id"])] = float(
            np.mean(disagreements)
        )
    selected = min(
        successful,
        key=lambda candidate: (
            scores[str(candidate["view"]["sample_id"])],
            int(candidate["view"]["acquisition_rank"]),
            str(candidate["view"]["sample_id"]),
        ),
    )
    return selected, scores


def diagnostic_flags(view: dict[str, Any], scale_r: float) -> dict[str, bool]:
    mssd_success = bool(
        view["finite_pose"]
        and view["normalized_mssd"] <= DIAGNOSTIC_MSSD_THRESHOLD
    )
    mspd_success = bool(
        view["finite_pose"]
        and view["mspd_px"] <= DIAGNOSTIC_MSPD_MULTIPLIER * scale_r
    )
    return {
        "mssd_0.10d": mssd_success,
        "mspd_10r": mspd_success,
        "joint": mssd_success and mspd_success,
    }


def latency_for_acquired(acquired: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [view["registration_seconds"] for view in acquired]
    known = [runtime for runtime in runtimes if runtime is not None]
    return {
        "acquired_view_count": len(acquired),
        "timed_view_count": len(known),
        "all_attempts_timed": len(known) == len(acquired),
        "acquired_registration_seconds": runtimes,
        "known_runtime_sum_seconds": float(sum(known)),
        "sequential_runtime_seconds": (
            float(sum(known)) if len(known) == len(acquired) else None
        ),
    }


def result_from_selected(
    prepared: dict[str, Any],
    acquired: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    method: str,
    budget: int,
    selection_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    group = prepared["group"]
    scale_r = float(prepared["mspd_scale_r"])
    finite_pose = selected is not None and bool(selected["finite_pose"])
    normalized_mssd = selected["normalized_mssd"] if finite_pose else None
    mspd_px = selected["mspd_px"] if finite_pose else None
    flags = (
        diagnostic_flags(selected, scale_r)
        if finite_pose
        else {"mssd_0.10d": False, "mspd_10r": False, "joint": False}
    )
    latency = latency_for_acquired(acquired)
    row = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "method_result",
        "group_id": str(group["group_id"]),
        "target_sample_id": str(group["target_sample_id"]),
        "scene_id": int(group["scene_id"]),
        "object_id": int(group["object_id"]),
        "target_visibility_bin": str(group["target_visibility_bin"]),
        "target_visible_fraction": float(
            prepared["target_manifest"]["visible_fraction"]
        ),
        "requested_view_budget": int(budget),
        "acquired_view_count": len(acquired),
        "usable_view_count": sum(view["finite_pose"] for view in acquired),
        "method": method,
        "status": (
            str(selected["status"]) if selected is not None else "no_usable_prediction"
        ),
        "finite_pose": finite_pose,
        "selected_sample_id": (
            str(selected["view"]["sample_id"]) if selected is not None else None
        ),
        "selected_acquisition_rank": (
            int(selected["view"]["acquisition_rank"])
            if selected is not None
            else None
        ),
        "predicted_model_to_target_camera_pose_m": (
            np.asarray(selected["transformed_pose_m"], dtype=np.float64).tolist()
            if finite_pose
            else None
        ),
        "normalized_mssd": normalized_mssd,
        "mspd_px": mspd_px,
        "mspd_scale_r": scale_r,
        "sample_ar_mssd": (
            float(
                np.mean(
                    [
                        normalized_mssd < threshold
                        for threshold in MSSD_THRESHOLD_FRACTIONS
                    ]
                )
            )
            if finite_pose
            else 0.0
        ),
        "sample_ar_mspd": (
            float(
                np.mean(
                    [
                        mspd_px < multiplier * scale_r
                        for multiplier in MSPD_THRESHOLD_MULTIPLIERS
                    ]
                )
            )
            if finite_pose
            else 0.0
        ),
        "diagnostic_success": flags,
        "selection_scores_normalized_symmetric_mssd": selection_scores,
        **latency,
    }
    return row


def result_oracle_any(
    prepared: dict[str, Any],
    acquired: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    successful = [view for view in acquired if view["finite_pose"]]
    scale_r = float(prepared["mspd_scale_r"])
    latency = latency_for_acquired(acquired)
    normalized_mssd = (
        min(float(view["normalized_mssd"]) for view in successful)
        if successful
        else None
    )
    mspd_px = (
        min(float(view["mspd_px"]) for view in successful)
        if successful
        else None
    )
    per_view_flags = [diagnostic_flags(view, scale_r) for view in successful]
    flags = {
        key: any(view_flags[key] for view_flags in per_view_flags)
        for key in ("mssd_0.10d", "mspd_10r", "joint")
    }
    group = prepared["group"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "method_result",
        "group_id": str(group["group_id"]),
        "target_sample_id": str(group["target_sample_id"]),
        "scene_id": int(group["scene_id"]),
        "object_id": int(group["object_id"]),
        "target_visibility_bin": str(group["target_visibility_bin"]),
        "target_visible_fraction": float(
            prepared["target_manifest"]["visible_fraction"]
        ),
        "requested_view_budget": int(budget),
        "acquired_view_count": len(acquired),
        "usable_view_count": len(successful),
        "method": "oracle_any_view",
        "status": (
            "upper_bound_success" if successful else "no_usable_prediction"
        ),
        "finite_pose": bool(successful),
        "selected_sample_id": None,
        "selected_acquisition_rank": None,
        "predicted_model_to_target_camera_pose_m": None,
        "normalized_mssd": normalized_mssd,
        "mspd_px": mspd_px,
        "mspd_scale_r": scale_r,
        "sample_ar_mssd": (
            float(
                np.mean(
                    [
                        normalized_mssd < threshold
                        for threshold in MSSD_THRESHOLD_FRACTIONS
                    ]
                )
            )
            if successful
            else 0.0
        ),
        "sample_ar_mspd": (
            float(
                np.mean(
                    [
                        mspd_px < multiplier * scale_r
                        for multiplier in MSPD_THRESHOLD_MULTIPLIERS
                    ]
                )
            )
            if successful
            else 0.0
        ),
        "diagnostic_success": flags,
        "selection_scores_normalized_symmetric_mssd": None,
        **latency,
    }


def evaluate_methods(prepared_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for group_index, prepared in enumerate(prepared_groups, start=1):
        views = prepared["views"]
        for budget in VIEW_BUDGETS:
            acquired = views[: min(budget, len(views))]
            target = acquired[0]
            results.append(
                result_from_selected(
                    prepared, acquired, target, "target_only", budget
                )
            )

            max_area = max(
                acquired,
                key=lambda view: (
                    int(view["view"]["visible_mask_pixel_count"]),
                    -int(view["view"]["acquisition_rank"]),
                ),
            )
            results.append(
                result_from_selected(
                    prepared, acquired, max_area, "max_mask_area", budget
                )
            )

            medoid, medoid_scores = choose_medoid(
                acquired, prepared["object_data"]
            )
            results.append(
                result_from_selected(
                    prepared,
                    acquired,
                    medoid,
                    "symmetry_aware_medoid",
                    budget,
                    medoid_scores,
                )
            )

            max_visibility = max(
                acquired,
                key=lambda view: (
                    float(view["view"]["visible_fraction"]),
                    -int(view["view"]["acquisition_rank"]),
                ),
            )
            results.append(
                result_from_selected(
                    prepared,
                    acquired,
                    max_visibility,
                    "oracle_max_visibility",
                    budget,
                )
            )
            results.append(result_oracle_any(prepared, acquired, budget))
        if group_index % 25 == 0 or group_index == len(prepared_groups):
            print(
                f"multi-view methods: {group_index}/{len(prepared_groups)}",
                flush=True,
            )
    return results


def result_subset(
    rows: list[dict[str, Any]], method: str, budget: int
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and int(row["requested_view_budget"]) == budget
    ]
    if len(selected) != 300:
        raise RuntimeError(
            f"Expected 300 rows for {method} k={budget}, found {len(selected)}"
        )
    return selected


def aggregate_result_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return aggregate(rows)


def rescue_harm(
    baseline_by_target: dict[str, dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for criterion in ("mssd_0.10d", "mspd_10r", "joint"):
        baseline_failures = 0
        baseline_successes = 0
        rescued = 0
        harmed = 0
        for row in method_rows:
            target_id = str(row["target_sample_id"])
            baseline_success = bool(
                baseline_by_target[target_id]["diagnostic_success"][criterion]
            )
            method_success = bool(row["diagnostic_success"][criterion])
            if baseline_success:
                baseline_successes += 1
                harmed += not method_success
            else:
                baseline_failures += 1
                rescued += method_success
        output[criterion] = {
            "baseline_failure_count": baseline_failures,
            "rescued_count": rescued,
            "rescue_rate": (
                rescued / baseline_failures if baseline_failures else None
            ),
            "baseline_success_count": baseline_successes,
            "harmed_count": harmed,
            "harm_rate": harmed / baseline_successes if baseline_successes else None,
            "method_success_count": baseline_successes - harmed + rescued,
            "net_success_rate_change": (rescued - harmed) / len(method_rows),
        }
        if (
            output[criterion]["method_success_count"] - baseline_successes
            != rescued - harmed
        ):
            raise RuntimeError("Rescue/harm accounting identity failed")
    return output


def summarize_latency(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_rows = {
        budget: result_subset(results, "target_only", budget)
        for budget in VIEW_BUDGETS
    }
    summaries: list[dict[str, Any]] = []
    previous_rows: dict[str, dict[str, Any]] | None = None
    for budget in VIEW_BUDGETS:
        rows = target_rows[budget]
        complete = [
            row["sequential_runtime_seconds"]
            for row in rows
            if row["sequential_runtime_seconds"] is not None
        ]
        acquired = [int(row["acquired_view_count"]) for row in rows]
        usable = [int(row["usable_view_count"]) for row in rows]
        acquired_distribution = Counter(acquired)
        usable_distribution = Counter(usable)
        entry = {
            "requested_view_budget": budget,
            "target_count": len(rows),
            "mean_acquired_view_count": float(np.mean(acquired)),
            "acquired_view_count_distribution": {
                str(key): value
                for key, value in sorted(acquired_distribution.items())
            },
            "mean_usable_view_count": float(np.mean(usable)),
            "usable_view_count_distribution": {
                str(key): value
                for key, value in sorted(usable_distribution.items())
            },
            "complete_latency_target_count": len(complete),
            "sequential_inference_seconds_per_target": summarize_values(complete),
            "marginal_seconds_per_additional_view": None,
            "added_view_runtime_seconds": summarize_values([]),
            "timed_added_view_count": 0,
            "additional_view_count_from_previous_budget": 0,
        }
        if previous_rows is not None:
            additional_count = 0
            additional_seconds = 0.0
            added_view_runtimes: list[float] = []
            complete_delta = True
            for row in rows:
                target_id = str(row["target_sample_id"])
                previous = previous_rows[target_id]
                added = int(row["acquired_view_count"]) - int(
                    previous["acquired_view_count"]
                )
                if added < 0:
                    raise RuntimeError("Acquired view count decreased with budget")
                additional_count += added
                current_runtime = row["sequential_runtime_seconds"]
                previous_runtime = previous["sequential_runtime_seconds"]
                current_attempts = row["acquired_registration_seconds"]
                previous_attempts = previous["acquired_registration_seconds"]
                if len(current_attempts) - len(previous_attempts) != added:
                    raise RuntimeError("Stored acquired-runtime list is inconsistent")
                added_view_runtimes.extend(
                    float(value)
                    for value in current_attempts[len(previous_attempts) :]
                    if value is not None
                )
                if current_runtime is None or previous_runtime is None:
                    complete_delta = False
                else:
                    additional_seconds += current_runtime - previous_runtime
            entry["additional_view_count_from_previous_budget"] = additional_count
            entry["timed_added_view_count"] = len(added_view_runtimes)
            entry["added_view_runtime_seconds"] = summarize_values(
                added_view_runtimes
            )
            if complete_delta and additional_count:
                entry["marginal_seconds_per_additional_view"] = (
                    additional_seconds / additional_count
                )
        summaries.append(entry)
        previous_rows = {
            str(row["target_sample_id"]): row for row in rows
        }
    return summaries


def check_m1_reproduction(
    results: list[dict[str, Any]],
    m1_summary: dict[str, Any],
) -> dict[str, Any]:
    rows = result_subset(results, "target_only", 1)
    reproduced = aggregate_result_rows(rows)
    expected = m1_summary["overall"]
    fields = (
        "ar_mssd",
        "ar_mspd",
        "oracle_mask_subset_ar_mssd_mspd",
    )
    differences = {
        field: float(reproduced[field]) - float(expected[field]) for field in fields
    }
    if not math.isclose(
        float(expected["oracle_mask_subset_ar_mssd_mspd"]),
        M1_EXPECTED_COMBINED,
        rel_tol=0.0,
        abs_tol=M1_REPRODUCTION_ATOL,
    ):
        raise RuntimeError(
            "Stored M1 summary does not contain the expected 0.6686666667 score"
        )
    if any(
        not math.isclose(
            float(reproduced[field]),
            float(expected[field]),
            rel_tol=0.0,
            abs_tol=M1_REPRODUCTION_ATOL,
        )
        for field in fields
    ):
        raise RuntimeError(
            "M2 target_only k=1 does not reproduce M1 exactly: "
            f"differences={differences}"
        )
    baseline_by_target = {
        str(row["target_sample_id"]): row for row in rows
    }
    for method in METHODS:
        method_rows = result_subset(results, method, 1)
        method_aggregate = aggregate_result_rows(method_rows)
        for field in fields:
            if not math.isclose(
                float(method_aggregate[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=M1_REPRODUCTION_ATOL,
            ):
                raise RuntimeError(
                    f"M2 {method} k=1 does not reproduce M1 field {field}"
                )
        for row in method_rows:
            baseline = baseline_by_target[str(row["target_sample_id"])]
            for field in ("sample_ar_mssd", "sample_ar_mspd"):
                if not math.isclose(
                    float(row[field]),
                    float(baseline[field]),
                    rel_tol=0.0,
                    abs_tol=M1_REPRODUCTION_ATOL,
                ):
                    raise RuntimeError(
                        f"M2 {method} k=1 per-target mismatch: "
                        f"{row['target_sample_id']} {field}"
                    )
    return {
        "passed": True,
        "all_required_methods_k1_match": True,
        "absolute_tolerance": M1_REPRODUCTION_ATOL,
        "expected": {field: float(expected[field]) for field in fields},
        "reproduced": {field: float(reproduced[field]) for field in fields},
        "differences": differences,
    }


def build_summary(
    results: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    gt_transform_audit: dict[str, Any],
    m1_reproduction: dict[str, Any],
    toolkit_sha: str,
    input_paths: dict[str, Path],
    m1_prediction_metadata: dict[str, Any],
    m2_prediction_metadata: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = result_subset(results, "target_only", 1)
    baseline_by_target = {
        str(row["target_sample_id"]): row for row in baseline_rows
    }
    object_ids = sorted({int(row["object_id"]) for row in baseline_rows})
    if len(object_ids) != 15 or 15 not in object_ids:
        raise RuntimeError(
            "M2 expected the same 15-object M1 subset including object 15, "
            f"found {object_ids}"
        )
    result_summaries: list[dict[str, Any]] = []
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            rows = result_subset(results, method, budget)
            overall = aggregate_result_rows(rows)
            by_visibility = []
            for visibility_bin in VISIBILITY_BINS:
                entry = {"visibility_bin": visibility_bin}
                entry.update(
                    aggregate_result_rows(
                        row
                        for row in rows
                        if row["target_visibility_bin"] == visibility_bin
                    )
                )
                by_visibility.append(entry)
            by_object = []
            for object_id in object_ids:
                entry = {"object_id": object_id}
                entry.update(
                    aggregate_result_rows(
                        row for row in rows if int(row["object_id"]) == object_id
                    )
                )
                by_object.append(entry)
            object_15 = next(
                entry for entry in by_object if int(entry["object_id"]) == 15
            )
            status_counts = Counter(str(row["status"]) for row in rows)
            result_summaries.append(
                {
                    "method": method,
                    "requested_view_budget": budget,
                    "overall": overall,
                    "by_target_visibility_bin": by_visibility,
                    "by_object": by_object,
                    "object_15": object_15,
                    "rescue_harm_at_0.10d_10r": rescue_harm(
                        baseline_by_target, rows
                    ),
                    "status_counts": dict(sorted(status_counts.items())),
                    "mean_acquired_view_count": float(
                        np.mean([row["acquired_view_count"] for row in rows])
                    ),
                    "mean_usable_view_count": float(
                        np.mean([row["usable_view_count"] for row in rows])
                    ),
                    "acquired_view_count_distribution": {
                        str(key): value
                        for key, value in sorted(
                            Counter(
                                int(row["acquired_view_count"]) for row in rows
                            ).items()
                        )
                    },
                    "usable_view_count_distribution": {
                        str(key): value
                        for key, value in sorted(
                            Counter(
                                int(row["usable_view_count"]) for row in rows
                            ).items()
                        )
                    },
                    "sequential_inference_seconds_per_target": summarize_values(
                        row["sequential_runtime_seconds"] for row in rows
                    ),
                }
            )

    indexed = {
        (entry["method"], int(entry["requested_view_budget"])): entry
        for entry in result_summaries
    }
    for budget in VIEW_BUDGETS:
        target_rows = {
            str(row["target_sample_id"]): row
            for row in result_subset(results, "target_only", budget)
        }
        for target_id, baseline in baseline_by_target.items():
            row = target_rows[target_id]
            if (
                row["selected_sample_id"] != baseline["selected_sample_id"]
                or not math.isclose(
                    float(row["sample_ar_mssd"]),
                    float(baseline["sample_ar_mssd"]),
                    rel_tol=0.0,
                    abs_tol=M1_REPRODUCTION_ATOL,
                )
                or not math.isclose(
                    float(row["sample_ar_mspd"]),
                    float(baseline["sample_ar_mspd"]),
                    rel_tol=0.0,
                    abs_tol=M1_REPRODUCTION_ATOL,
                )
            ):
                raise RuntimeError(
                    f"target_only changed with view budget for {target_id}"
                )
        oracle_diagnostic = indexed[
            ("oracle_any_view", budget)
        ]["rescue_harm_at_0.10d_10r"]
        if any(
            int(oracle_diagnostic[criterion]["harmed_count"]) != 0
            for criterion in ("mssd_0.10d", "mspd_10r", "joint")
        ):
            raise RuntimeError("oracle_any_view harmed a target despite including it")

    target_k2 = {
        str(row["target_sample_id"]): row
        for row in result_subset(results, "target_only", 2)
    }
    for medoid_row in result_subset(results, "symmetry_aware_medoid", 2):
        target_row = target_k2[str(medoid_row["target_sample_id"])]
        if (
            medoid_row["selected_sample_id"] != target_row["selected_sample_id"]
            or not math.isclose(
                float(medoid_row["sample_ar_mssd"]),
                float(target_row["sample_ar_mssd"]),
                rel_tol=0.0,
                abs_tol=M1_REPRODUCTION_ATOL,
            )
            or not math.isclose(
                float(medoid_row["sample_ar_mspd"]),
                float(target_row["sample_ar_mspd"]),
                rel_tol=0.0,
                abs_tol=M1_REPRODUCTION_ATOL,
            )
        ):
            raise RuntimeError(
                "Two-view medoid tie-break did not preserve the target-first pose"
            )

    oracle_gaps = []
    for budget in VIEW_BUDGETS:
        target_overall = indexed[("target_only", budget)]["overall"]
        medoid_overall = indexed[("symmetry_aware_medoid", budget)]["overall"]
        oracle_overall = indexed[("oracle_any_view", budget)]["overall"]
        visibility_overall = indexed[("oracle_max_visibility", budget)]["overall"]
        entry: dict[str, Any] = {"requested_view_budget": budget}
        metric_fields = {
            "ar_mssd": "ar_mssd",
            "ar_mspd": "ar_mspd",
            "combined": "oracle_mask_subset_ar_mssd_mspd",
        }
        for label, field in metric_fields.items():
            target_score = float(target_overall[field])
            medoid_score = float(medoid_overall[field])
            oracle_score = float(oracle_overall[field])
            visibility_score = float(visibility_overall[field])
            entry[f"target_only_{label}"] = target_score
            entry[f"symmetry_aware_medoid_{label}"] = medoid_score
            entry[f"oracle_max_visibility_{label}"] = visibility_score
            entry[f"oracle_any_view_{label}"] = oracle_score
            entry[f"oracle_headroom_vs_target_only_{label}"] = (
                oracle_score - target_score
            )
            entry[f"medoid_to_oracle_any_gap_{label}"] = (
                oracle_score - medoid_score
            )
            if entry[f"medoid_to_oracle_any_gap_{label}"] < -1e-12:
                raise RuntimeError(
                    f"Oracle-any upper bound fell below medoid for {label}, k={budget}"
                )
            entry[f"oracle_visibility_gain_vs_target_only_{label}"] = (
                visibility_score - target_score
            )
        # Backwards-readable aliases used by the compact report.
        entry["oracle_headroom_vs_target_only"] = entry[
            "oracle_headroom_vs_target_only_combined"
        ]
        entry["medoid_to_oracle_any_gap"] = entry[
            "medoid_to_oracle_any_gap_combined"
        ]
        entry["oracle_visibility_gain_vs_target_only"] = entry[
            "oracle_visibility_gain_vs_target_only_combined"
        ]
        oracle_gaps.append(entry)

    available_counts = [len(group["views"]) for group in groups]
    additional_counts = [count - 1 for count in available_counts]
    input_metadata: dict[str, Any] = {}
    for name, path in input_paths.items():
        input_metadata[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    input_metadata["m1_predictions"]["metadata"] = m1_prediction_metadata
    input_metadata["view_predictions"]["metadata"] = m2_prediction_metadata

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PoseLoop-MV calibrated symmetry-aware multi-view rescue",
        "metric_scope": {
            "combined_metric_name": "oracle_mask_subset_ar_mssd_mspd",
            "description": (
                "Arithmetic mean of subset AR_MSSD and AR_MSPD over the same "
                "300 M1 targets."
            ),
            "failure_denominator": (
                "All 300 target groups; failed and non-finite selected poses are "
                "misses at every threshold."
            ),
            "mssd_threshold_fractions_of_diameter": list(
                MSSD_THRESHOLD_FRACTIONS
            ),
            "mspd_threshold_pixel_multipliers_of_r": list(
                MSPD_THRESHOLD_MULTIPLIERS
            ),
            "mspd_r_definition": "target image_width / 640",
            "threshold_comparison": "strictly less than (exact M1 semantics)",
            "diagnostic_thresholds": {
                "normalized_mssd": DIAGNOSTIC_MSSD_THRESHOLD,
                "mspd_multiplier_of_r": DIAGNOSTIC_MSPD_MULTIPLIER,
                "comparison": "inclusive less than or equal to",
                "primary_rescue_harm_definition": (
                    "joint success requires one selected pose, or one "
                    "oracle-any constituent pose, to pass both thresholds"
                ),
            },
        },
        "method_contract": {
            "view_budgets": list(VIEW_BUDGETS),
            "acquisition_order": (
                "target first, then the fixed camera-pose-diversity order stored "
                "in groups.jsonl"
            ),
            "target_only": "Always returns the original M1 target prediction.",
            "max_mask_area": (
                "Selects the acquired view with the largest GT visible-mask "
                "pixel count; ties use acquisition rank."
            ),
            "symmetry_aware_medoid": (
                "Among finite acquired predictions, minimizes mean pairwise "
                "bidirectionally symmetrized official MSSD normalized by "
                "object diameter; ties use acquisition rank."
            ),
            "oracle_max_visibility": (
                "Selects the acquired view with highest GT visible fraction; "
                "non-deployable."
            ),
            "oracle_any_view": (
                "Threshold-specific any-view upper bound; normalized MSSD and "
                "MSPD minima may come from different views and do not define a "
                "single fused pose."
            ),
            "fixed_medoid_parameters": {
                "pairwise_metric": "mean of forward and reverse official MSSD",
                "normalization": "official object diameter",
                "symmetry_discretization_step": (
                    MAX_SYMMETRY_DISCRETIZATION_STEP
                ),
                "gt_used_for_selection": False,
            },
        },
        "bop_toolkit": {
            "commit_sha": toolkit_sha,
            "environment": "poseloop-bop",
            "python_executable": sys.executable,
            "imported_pose_error_module": str(Path(pose_error.__file__).resolve()),
            "model_type": "models_eval",
            "max_sym_disc_step": MAX_SYMMETRY_DISCRETIZATION_STEP,
        },
        "inputs": input_metadata,
        "group_counts": {
            "target_group_count": len(groups),
            "available_total_view_count_distribution": {
                str(key): value
                for key, value in sorted(Counter(available_counts).items())
            },
            "available_additional_view_count_distribution": {
                str(key): value
                for key, value in sorted(Counter(additional_counts).items())
            },
        },
        "cross_view_gt_transform_check": gt_transform_audit,
        "m1_target_only_k1_reproduction": m1_reproduction,
        "results": result_summaries,
        "oracle_headroom_and_medoid_gap": oracle_gaps,
        "latency_by_view_budget": summarize_latency(results),
        "caveats": list(CAVEATS),
    }


def result_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (entry["method"], int(entry["requested_view_budget"])): entry
        for entry in summary["results"]
    }


def format_percent(value: Any, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{100.0 * float(value):.2f}%"


def format_number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def format_points(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):+.2f} pp"


def conclusion_text(summary: dict[str, Any]) -> str:
    indexed = result_index(summary)
    target = indexed[
        ("target_only", 5)
    ]["overall"]["oracle_mask_subset_ar_mssd_mspd"]
    max_mask = indexed[
        ("max_mask_area", 5)
    ]["overall"]["oracle_mask_subset_ar_mssd_mspd"]
    medoid = indexed[
        ("symmetry_aware_medoid", 5)
    ]["overall"]["oracle_mask_subset_ar_mssd_mspd"]
    any_oracle = indexed[
        ("oracle_any_view", 5)
    ]["overall"]["oracle_mask_subset_ar_mssd_mspd"]
    return (
        "At k=5, the fixed max-mask selector raises the combined diagnostic "
        f"score from {format_percent(target)} to {format_percent(max_mask)} "
        f"({format_points(max_mask - target)}), the main positive M2 result. "
        "It selects among acquired poses using ground-truth visible-mask area, "
        "so this is an oracle-mask diagnostic rather than a deployable policy "
        "or independent holdout result. The fixed symmetry-aware medoid also "
        f"improves over target-only ({format_points(medoid - target)}), while "
        "the threshold-specific any-view upper bound shows "
        f"{format_points(any_oracle - target)} of recoverable headroom."
    )


def markdown_report(summary: dict[str, Any]) -> str:
    indexed = result_index(summary)
    conclusion = conclusion_text(summary)
    caveats = "\n".join(f"- **{item}**" for item in summary["caveats"])

    overall_rows = []
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            entry = indexed[(method, budget)]
            overall = entry["overall"]
            overall_rows.append(
                "| {k} | `{method}` | {finite} | {mssd} | {mspd} | "
                "{combined} | {usable:.2f} |".format(
                    k=budget,
                    method=method,
                    finite=format_percent(overall["finite_pose_rate"]),
                    mssd=format_percent(overall["ar_mssd"]),
                    mspd=format_percent(overall["ar_mspd"]),
                    combined=format_percent(
                        overall["oracle_mask_subset_ar_mssd_mspd"]
                    ),
                    usable=entry["mean_usable_view_count"],
                )
            )

    visibility_rows = []
    baseline_visibility = {
        row["visibility_bin"]: row
        for row in indexed[
            ("target_only", 1)
        ]["by_target_visibility_bin"]
    }
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            for row in indexed[(method, budget)]["by_target_visibility_bin"]:
                label = row["visibility_bin"]
                score = row["oracle_mask_subset_ar_mssd_mspd"]
                baseline = baseline_visibility[label][
                    "oracle_mask_subset_ar_mssd_mspd"
                ]
                visibility_rows.append(
                    f"| {budget} | `{method}` | {label} | "
                    f"{row['sample_count']} | {format_percent(score)} | "
                    f"{format_percent(score - baseline, signed=True)} |"
                )

    object_rows = []
    baseline_objects = {
        int(row["object_id"]): row
        for row in indexed[("target_only", 1)]["by_object"]
    }
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            for row in indexed[(method, budget)]["by_object"]:
                object_id = int(row["object_id"])
                score = row["oracle_mask_subset_ar_mssd_mspd"]
                baseline = baseline_objects[object_id][
                    "oracle_mask_subset_ar_mssd_mspd"
                ]
                object_rows.append(
                    f"| {budget} | `{method}` | {object_id} | "
                    f"{format_percent(row['ar_mssd'])} | "
                    f"{format_percent(row['ar_mspd'])} | "
                    f"{format_percent(score)} | "
                    f"{format_percent(score - baseline, signed=True)} |"
                )

    object15_rows = []
    baseline15 = baseline_objects[15]["oracle_mask_subset_ar_mssd_mspd"]
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            row = indexed[(method, budget)]["object_15"]
            score = row["oracle_mask_subset_ar_mssd_mspd"]
            object15_rows.append(
                f"| {budget} | `{method}` | {format_percent(row['ar_mssd'])} | "
                f"{format_percent(row['ar_mspd'])} | {format_percent(score)} | "
                f"{format_percent(score - baseline15, signed=True)} |"
            )

    rescue_rows = []
    for budget in VIEW_BUDGETS:
        for method in METHODS:
            diagnostic = indexed[
                (method, budget)
            ]["rescue_harm_at_0.10d_10r"]
            joint = diagnostic["joint"]
            mssd = diagnostic["mssd_0.10d"]
            mspd = diagnostic["mspd_10r"]
            rescue_rows.append(
                f"| {budget} | `{method}` | "
                f"{joint['rescued_count']}/{joint['baseline_failure_count']} "
                f"({format_percent(joint['rescue_rate'])}) | "
                f"{joint['harmed_count']}/{joint['baseline_success_count']} "
                f"({format_percent(joint['harm_rate'])}) | "
                f"{format_percent(mssd['rescue_rate'])} / "
                f"{format_percent(mssd['harm_rate'])} | "
                f"{format_percent(mspd['rescue_rate'])} / "
                f"{format_percent(mspd['harm_rate'])} |"
            )

    oracle_rows = [
        "| {k} | {h_mssd} | {h_mspd} | {h_combined} | "
        "{g_mssd} | {g_mspd} | {g_combined} | {visibility} |".format(
            k=row["requested_view_budget"],
            h_mssd=format_percent(
                row["oracle_headroom_vs_target_only_ar_mssd"], signed=True
            ),
            h_mspd=format_percent(
                row["oracle_headroom_vs_target_only_ar_mspd"], signed=True
            ),
            h_combined=format_percent(
                row["oracle_headroom_vs_target_only_combined"], signed=True
            ),
            g_mssd=format_percent(row["medoid_to_oracle_any_gap_ar_mssd"]),
            g_mspd=format_percent(row["medoid_to_oracle_any_gap_ar_mspd"]),
            g_combined=format_percent(
                row["medoid_to_oracle_any_gap_combined"]
            ),
            visibility=format_percent(
                row["oracle_visibility_gain_vs_target_only_combined"],
                signed=True,
            ),
        )
        for row in summary["oracle_headroom_and_medoid_gap"]
    ]

    latency_rows = []
    for row in summary["latency_by_view_budget"]:
        sequential = row["sequential_inference_seconds_per_target"]
        added = row["added_view_runtime_seconds"]
        latency_rows.append(
            f"| {row['requested_view_budget']} | "
            f"{row['mean_acquired_view_count']:.2f} | "
            f"{row['mean_usable_view_count']:.2f} | "
            f"{format_number(sequential['mean'])} | "
            f"{format_number(sequential['median'])} | "
            f"{format_number(sequential['p95'])} | "
            f"{format_number(row['marginal_seconds_per_additional_view'])} | "
            f"{format_number(added['median'])} / "
            f"{format_number(added['p95'])} | "
            f"{row['timed_added_view_count']}/"
            f"{row['additional_view_count_from_previous_budget']} | "
            f"{row['complete_latency_target_count']}/300 |"
        )

    reproduction = summary["m1_target_only_k1_reproduction"]
    residual = summary["cross_view_gt_transform_check"]
    group_counts = summary["group_counts"]
    toolkit = summary["bop_toolkit"]
    return f"""# PoseLoop M2 calibrated multi-view rescue diagnostic

## Answer first

{conclusion}

The k=1 target-only combined score is
{format_percent(reproduction['reproduced']['oracle_mask_subset_ar_mssd_mspd'])}
({reproduction['reproduced']['oracle_mask_subset_ar_mssd_mspd']:.10f}), exactly
reproducing M1 within `{reproduction['absolute_tolerance']}`.

## Scope and hard caveats

{caveats}

## Overall results

| k | Method | Finite/upper-bound usable | AR_MSSD | AR_MSPD | Combined | Mean usable views |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(overall_rows)}

![Combined diagnostic score by view budget](m2_budget_curve.png)

`oracle_any_view` is threshold-specific: its MSSD and MSPD successes may come
from different views, so it is not a selected or fused pose.

## Target visibility

Gain is relative to the matching target-visibility stratum at target-only
k=1.

| k | Method | Target visibility | n | Combined | Gain |
|---:|---|---|---:|---:|---:|
{chr(10).join(visibility_rows)}

![k=5 gain by target visibility](m2_gain_by_visibility.png)

## Per-object results

Gain is relative to the same object's target-only k=1 result.

| k | Method | Object | AR_MSSD | AR_MSPD | Combined | Gain |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(object_rows)}

![k=5 gain by object](m2_per_object_gain.png)

### Object 15

| k | Method | AR_MSSD | AR_MSPD | Combined | Gain |
|---:|---|---:|---:|---:|---:|
{chr(10).join(object15_rows)}

## Rescue and harm at 0.10d / 10r

These rescue/harm diagnostics use the requested inclusive `<=` comparison at
0.10d and 10r; the AR curves above preserve M1's strict `<` comparisons.
The primary joint result requires the selected pose to pass both thresholds.
For `oracle_any_view`, one constituent view must pass both; separate MSSD and
MSPD upper bounds may still come from different views.

| k | Method | Joint rescue | Joint harm | MSSD rescue / harm | MSPD rescue / harm |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rescue_rows)}

## Oracle headroom

| k | Headroom MSSD | Headroom MSPD | Headroom combined | Medoid gap MSSD | Medoid gap MSPD | Medoid gap combined | Visibility-oracle combined gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(oracle_rows)}

![Oracle headroom and medoid gap](m2_oracle_gap.png)

## View availability and sequential cost

All 300 target groups are retained. Total-view availability distribution:
`{group_counts['available_total_view_count_distribution']}`; additional-view
distribution: `{group_counts['available_additional_view_count_distribution']}`.
Failed predictions remain in score denominators. Latency sums the empirically
recorded single-view registration times in acquisition order; rows with an
untimed attempt are excluded from complete-latency summaries.

| k | Mean acquired | Mean usable | Mean seconds/target | p50 | p95 | Marginal mean seconds/additional view | Added-view p50 / p95 | Added timing | Complete target timing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(latency_rows)}

## Calibration and transformation checks

Candidate poses use `T_w_m = inverse(T_c_w) @ T_c_m`, followed by
`T_target_c_m = T_target_c_w @ T_w_m`. Reapplying that path to the grouped GT
poses produced max normalized MSSD
{format_number(residual['normalized_mssd']['max'], 8)} and max MSPD
{format_number(residual['mspd_px']['max'], 8)} px over
{residual['sample_count']} grouped views, passing the fixed
{residual['acceptance_limits']['max_normalized_mssd']}d /
{residual['acceptance_limits']['max_mspd_px']} px evaluation gate.

The official symmetry-aware evaluator uses BOP Toolkit commit
`{toolkit['commit_sha']}`, `models_eval`, and
`max_sym_disc_step = {toolkit['max_sym_disc_step']}`. Camera-diversity
acquisition order is fixed in `groups.jsonl`; no final-set GT pose error or GT
visibility was used to order views. The medoid uses no target GT for selection.
"""


def save_figure_atomic(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(
        temporary,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        format="png",
    )
    os.replace(temporary, path)


def generate_figures(summary: dict[str, Any], output_dir: Path) -> None:
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
    indexed = result_index(summary)
    colors = {
        "target_only": "#0072B2",
        "max_mask_area": "#E69F00",
        "symmetry_aware_medoid": "#009E73",
        "oracle_max_visibility": "#D55E00",
        "oracle_any_view": "#CC79A7",
    }

    budget_metrics = (
        ("ar_mssd", "AR_MSSD"),
        ("ar_mspd", "AR_MSPD"),
        ("oracle_mask_subset_ar_mssd_mspd", "Combined"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for axis, (field, label) in zip(axes, budget_metrics, strict=True):
        for method in METHODS:
            values = [
                indexed[(method, budget)]["overall"][field]
                for budget in VIEW_BUDGETS
            ]
            axis.plot(
                VIEW_BUDGETS,
                values,
                marker="o",
                linewidth=2,
                label=method,
                color=colors[method],
                linestyle="--" if method.startswith("oracle_") else "-",
            )
        axis.set(
            xlabel="Acquired-view budget k",
            ylabel="Score" if field == "ar_mssd" else None,
            title=label,
            xticks=VIEW_BUDGETS,
            ylim=(0, 1),
        )
    axes[-1].legend(fontsize=7, loc="best", frameon=False)
    save_figure_atomic(figure, output_dir / "m2_budget_curve.png")
    plt.close(figure)

    baseline_visibility = {
        row["visibility_bin"]: row["oracle_mask_subset_ar_mssd_mspd"]
        for row in indexed[
            ("target_only", 1)
        ]["by_target_visibility_bin"]
    }
    plotted_methods = METHODS[1:]
    x = np.arange(len(VISIBILITY_BINS))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    for method_index, method in enumerate(plotted_methods):
        entries = {
            row["visibility_bin"]: row["oracle_mask_subset_ar_mssd_mspd"]
            for row in indexed[(method, 5)]["by_target_visibility_bin"]
        }
        gains = [
            entries[label] - baseline_visibility[label]
            for label in VISIBILITY_BINS
        ]
        axis.bar(
            x + (method_index - 1.5) * width,
            gains,
            width,
            label=method,
            color=colors[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Target-view visibility bin",
        ylabel="Combined-score gain vs target-only",
        title="M2 k=5 gain by target visibility",
        xticks=x,
        xticklabels=VISIBILITY_BINS,
    )
    axis.legend(fontsize=8, frameon=False)
    save_figure_atomic(figure, output_dir / "m2_gain_by_visibility.png")
    plt.close(figure)

    baseline_objects = {
        int(row["object_id"]): row["oracle_mask_subset_ar_mssd_mspd"]
        for row in indexed[("target_only", 1)]["by_object"]
    }
    object_ids = sorted(baseline_objects)
    x = np.arange(len(object_ids))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.0, 5.2))
    for offset, method in (
        (-width / 2, "symmetry_aware_medoid"),
        (width / 2, "oracle_any_view"),
    ):
        entries = {
            int(row["object_id"]): row["oracle_mask_subset_ar_mssd_mspd"]
            for row in indexed[(method, 5)]["by_object"]
        }
        gains = [
            entries[object_id] - baseline_objects[object_id]
            for object_id in object_ids
        ]
        axis.bar(
            x + offset,
            gains,
            width,
            label=method,
            color=colors[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="XYZ-IBD object ID",
        ylabel="Combined-score gain vs target-only",
        title="M2 k=5 per-object gain",
        xticks=x,
        xticklabels=[str(value) for value in object_ids],
    )
    axis.legend(frameon=False)
    if 15 in object_ids:
        object15_index = object_ids.index(15)
        axis.axvspan(
            object15_index - 0.5,
            object15_index + 0.5,
            color="#f2cf5b",
            alpha=0.16,
            zorder=0,
        )
    save_figure_atomic(figure, output_dir / "m2_per_object_gain.png")
    plt.close(figure)

    gaps = summary["oracle_headroom_and_medoid_gap"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    for axis, (label, title) in zip(
        axes,
        (("ar_mssd", "AR_MSSD"), ("ar_mspd", "AR_MSPD"), ("combined", "Combined")),
        strict=True,
    ):
        axis.plot(
            VIEW_BUDGETS,
            [
                row[f"oracle_headroom_vs_target_only_{label}"]
                for row in gaps
            ],
            marker="o",
            linewidth=2,
            label="oracle_any - target",
            color=colors["oracle_any_view"],
        )
        axis.plot(
            VIEW_BUDGETS,
            [row[f"medoid_to_oracle_any_gap_{label}"] for row in gaps],
            marker="s",
            linewidth=2,
            label="oracle_any - medoid",
            color=colors["symmetry_aware_medoid"],
        )
        if label == "combined":
            axis.plot(
                VIEW_BUDGETS,
                [
                    row["oracle_visibility_gain_vs_target_only_combined"]
                    for row in gaps
                ],
                marker="^",
                linewidth=2,
                label="visibility oracle - target",
                color=colors["oracle_max_visibility"],
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel="Acquired-view budget k",
            ylabel="Absolute score difference" if label == "ar_mssd" else None,
            title=title,
            xticks=VIEW_BUDGETS,
        )
    axes[-1].legend(fontsize=7, frameon=False)
    save_figure_atomic(figure, output_dir / "m2_oracle_gap.png")
    plt.close(figure)


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
    required_paths = {
        "groups": args.groups.resolve(),
        "candidate_manifest": args.candidate_manifest.resolve(),
        "groups_summary": args.groups_summary.resolve(),
        "extrinsics_audit": args.extrinsics_audit.resolve(),
        "m1_manifest": args.m1_manifest.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
        "m1_summary": args.m1_summary.resolve(),
        "view_predictions": args.view_predictions.resolve(),
    }
    missing_paths = [
        str(path) for path in required_paths.values() if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing M2 evaluation inputs: {missing_paths}")

    toolkit_sha = toolkit_commit(args.toolkit_root.resolve())
    model_params, model_info = load_official_models(args.dataset_root.resolve())
    manifest_rows = load_jsonl(required_paths["m1_manifest"])
    manifest_index = validate_manifest(manifest_rows)
    manifest_sha256 = sha256_file(required_paths["m1_manifest"])
    m1_prediction_metadata, m1_predictions = load_m1_predictions(
        required_paths["m1_predictions"], manifest_index, manifest_sha256
    )
    groups, m2_source_ids = validate_groups(
        load_jsonl(required_paths["groups"]), manifest_index
    )
    candidate_index = load_candidate_manifest(
        required_paths["candidate_manifest"],
        m2_source_ids,
    )
    m2_prediction_metadata, m2_predictions = load_view_predictions(
        required_paths["view_predictions"],
        candidate_index,
    )
    validate_m2_provenance(
        m2_prediction_metadata,
        required_paths,
        candidate_index,
    )

    prepared_groups, gt_transform_audit = prepare_groups(
        groups,
        manifest_index,
        m1_predictions,
        m2_predictions,
        model_params,
        model_info,
        args.max_gt_transform_normalized_mssd,
        args.max_gt_transform_mspd_px,
    )
    results = evaluate_methods(prepared_groups)
    m1_summary = read_json(required_paths["m1_summary"])
    m1_reproduction = check_m1_reproduction(results, m1_summary)
    print(
        "M1 reproduction passed: target_only k=1 combined="
        f"{m1_reproduction['reproduced']['oracle_mask_subset_ar_mssd_mspd']:.10f}",
        flush=True,
    )

    summary = build_summary(
        results,
        groups,
        gt_transform_audit,
        m1_reproduction,
        toolkit_sha,
        required_paths,
        m1_prediction_metadata,
        m2_prediction_metadata,
    )
    write_jsonl_atomic(args.metrics.resolve(), results)
    write_json_atomic(args.summary.resolve(), summary)
    generate_figures(summary, args.figures_dir.resolve())
    write_text_atomic(args.report.resolve(), markdown_report(summary))

    print(conclusion_text(summary), flush=True)
    print(f"saved: {args.metrics.resolve()}", flush=True)
    print(f"saved: {args.summary.resolve()}", flush=True)
    print(f"saved: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
