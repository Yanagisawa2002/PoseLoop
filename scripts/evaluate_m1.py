#!/usr/bin/env python3
"""Evaluate the PoseLoop M1 subset with the official BOP Toolkit metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from bop_toolkit_lib import dataset_params, inout, misc, pose_error

from m1_common import (
    ALLOWED_STATUSES,
    GT_ROTATION_ATOL,
    assert_pose,
    load_jsonl,
    raw_pose_errors,
    read_json,
    sha256_file,
    visibility_bin,
    write_json_atomic,
)


BOP_TOOLKIT_COMMIT = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
MAX_SYMMETRY_DISCRETIZATION_STEP = 0.01
GT_ZERO_TOLERANCE = 1e-6
MSSD_THRESHOLD_FRACTIONS = tuple(float(value) for value in np.arange(0.05, 0.51, 0.05))
MSPD_THRESHOLD_MULTIPLIERS = tuple(range(5, 51, 5))
DEPTH_GROUPS = (
    ("[0.00,0.90)", 0.00, 0.90, False),
    ("[0.90,0.99)", 0.90, 0.99, False),
    ("[0.99,1.00]", 0.99, 1.00, True),
)
CAVEATS = (
    "Ground-truth visible masks and known object IDs are used.",
    "Only a deterministic 300-instance RealSense validation subset is evaluated.",
    "oracle_mask_subset_ar_mssd_mspd is not official BOP AP, leaderboard AR, or full-dataset performance.",
    "Raw rotation error is not symmetry-aware and is diagnostic only.",
    "These results are diagnostic evidence for choosing the next PoseLoop research module.",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=repo_root / "third_party" / "bop_toolkit",
    )
    parser.add_argument(
        "--m0-result",
        type=Path,
        default=repo_root / "artifacts" / "smoke" / "result.json",
    )
    parser.add_argument(
        "--m0-sanity-output",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "m0_symmetry_check.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "predictions.jsonl",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "metrics.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "summary.json",
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "summary.md",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "m1_baseline.md",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "reports",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run GT-vs-GT and the saved M0 prediction check, then stop.",
    )
    return parser.parse_args()


def toolkit_commit(toolkit_root: Path) -> str:
    if not toolkit_root.is_dir():
        raise FileNotFoundError(f"BOP Toolkit checkout not found: {toolkit_root}")
    completed = subprocess.run(
        ["git", "-C", str(toolkit_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != BOP_TOOLKIT_COMMIT:
        raise RuntimeError(
            f"BOP Toolkit must be pinned at {BOP_TOOLKIT_COMMIT}, found {commit}"
        )
    imported_source = Path(pose_error.__file__).resolve()
    expected_source_root = (toolkit_root / "bop_toolkit_lib").resolve()
    if not imported_source.is_relative_to(expected_source_root):
        raise RuntimeError(
            "Imported bop_toolkit_lib does not come from the pinned checkout: "
            f"{imported_source}"
        )
    tracked_diff = subprocess.run(
        [
            "git",
            "-C",
            str(toolkit_root),
            "diff",
            "--quiet",
            "--ignore-cr-at-eol",
            "HEAD",
            "--",
            "bop_toolkit_lib",
        ],
        capture_output=True,
        text=True,
    )
    if tracked_diff.returncode not in (0, 1):
        raise RuntimeError(
            f"Could not verify BOP Toolkit metric source: {tracked_diff.stderr.strip()}"
        )
    if tracked_diff.returncode == 1:
        raise RuntimeError("Pinned BOP Toolkit metric source has tracked modifications")
    return commit


def load_official_models(dataset_root: Path) -> tuple[dict[str, Any], dict[int, Any]]:
    if dataset_root.name != "xyzibd":
        raise ValueError(f"Expected XYZ-IBD dataset root, got: {dataset_root}")
    params = dataset_params.get_model_params(
        str(dataset_root.parent),
        dataset_root.name,
        model_type="eval",
    )
    info_path = Path(params["models_info_path"])
    if info_path.parent.name != "models_eval":
        raise RuntimeError(f"Official evaluation models were not selected: {info_path}")
    model_info = inout.load_json(str(info_path), keys_to_int=True)
    return params, model_info


def load_object_evaluation_data(
    object_id: int,
    model_params: dict[str, Any],
    model_info: dict[int, Any],
) -> dict[str, Any]:
    if object_id not in model_info:
        raise KeyError(f"Object {object_id} is absent from models_eval/models_info.json")
    model_path = Path(model_params["model_tpath"].format(obj_id=object_id))
    points_mm = np.asarray(inout.load_ply(str(model_path))["pts"], dtype=np.float64)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3 or not np.isfinite(points_mm).all():
        raise ValueError(f"Invalid official evaluation model points: {model_path}")
    diameter_mm = float(model_info[object_id]["diameter"])
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError(f"Invalid diameter for object {object_id}: {diameter_mm}")
    symmetries = misc.get_symmetry_transformations(
        model_info[object_id],
        MAX_SYMMETRY_DISCRETIZATION_STEP,
    )
    if not symmetries:
        raise ValueError(f"No identity symmetry returned for object {object_id}")
    return {
        "model_path": str(model_path),
        "points_mm": points_mm,
        "diameter_mm": diameter_mm,
        "symmetries": symmetries,
    }


def official_errors(
    predicted_pose_m: np.ndarray,
    gt_pose_m: np.ndarray,
    camera_matrix: np.ndarray,
    object_data: dict[str, Any],
) -> tuple[float, float]:
    predicted_pose_m = np.asarray(predicted_pose_m, dtype=np.float64)
    gt_pose_m = np.asarray(gt_pose_m, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    assert_pose(predicted_pose_m, "predicted pose")
    assert_pose(gt_pose_m, "GT pose", rotation_atol=GT_ROTATION_ATOL)
    if not np.isfinite(camera_matrix).all():
        raise ValueError("Camera matrix is non-finite")

    predicted_translation_mm = predicted_pose_m[:3, 3:4] * 1000.0
    gt_translation_mm = gt_pose_m[:3, 3:4] * 1000.0
    mssd_mm = float(
        pose_error.mssd(
            predicted_pose_m[:3, :3],
            predicted_translation_mm,
            gt_pose_m[:3, :3],
            gt_translation_mm,
            object_data["points_mm"],
            object_data["symmetries"],
        )
    )
    mspd_px = float(
        pose_error.mspd(
            predicted_pose_m[:3, :3],
            predicted_translation_mm,
            gt_pose_m[:3, :3],
            gt_translation_mm,
            camera_matrix,
            object_data["points_mm"],
            object_data["symmetries"],
        )
    )
    if not math.isfinite(mssd_mm) or not math.isfinite(mspd_px):
        raise ValueError("Official evaluator returned a non-finite metric")
    return mssd_mm, mspd_px


def run_sanity_checks(
    m0_result_path: Path,
    output_path: Path,
    toolkit_sha: str,
    model_params: dict[str, Any],
    model_info: dict[int, Any],
) -> dict[str, Any]:
    m0_result = read_json(m0_result_path)
    object_id = int(m0_result["selection"]["object_id"])
    object_data = load_object_evaluation_data(object_id, model_params, model_info)
    gt_pose_m = np.asarray(m0_result["gt_model_to_camera_pose_m"], dtype=np.float64)
    predicted_pose_m = np.asarray(
        m0_result["predicted_model_to_camera_pose_m"], dtype=np.float64
    )
    camera_matrix = np.asarray(
        m0_result["inputs"]["camera_intrinsics_row_major"], dtype=np.float64
    )

    gt_gt_mssd_mm, gt_gt_mspd_px = official_errors(
        gt_pose_m,
        gt_pose_m,
        camera_matrix,
        object_data,
    )
    if (
        gt_gt_mssd_mm > GT_ZERO_TOLERANCE
        or gt_gt_mspd_px > GT_ZERO_TOLERANCE
    ):
        raise RuntimeError(
            "Evaluator sanity failure: GT-vs-GT was not approximately zero "
            f"(MSSD={gt_gt_mssd_mm}, MSPD={gt_gt_mspd_px})"
        )

    mssd_mm, mspd_px = official_errors(
        predicted_pose_m,
        gt_pose_m,
        camera_matrix,
        object_data,
    )
    translation_error_mm, raw_rotation_error_degrees = raw_pose_errors(
        predicted_pose_m,
        gt_pose_m,
    )
    sanity = {
        "schema_version": 1,
        "bop_toolkit": {
            "commit_sha": toolkit_sha,
            "max_sym_disc_step": MAX_SYMMETRY_DISCRETIZATION_STEP,
        },
        "evaluation_assets": {
            "model_path": object_data["model_path"],
            "models_type": "models_eval",
            "object_diameter_mm": object_data["diameter_mm"],
            "model_point_count": int(object_data["points_mm"].shape[0]),
            "symmetry_transform_count": len(object_data["symmetries"]),
        },
        "gt_vs_gt": {
            "mssd_mm": gt_gt_mssd_mm,
            "mspd_px": gt_gt_mspd_px,
            "tolerance": GT_ZERO_TOLERANCE,
            "passed": True,
        },
        "m0_prediction": {
            "source": str(m0_result_path.resolve()),
            "object_id": object_id,
            "translation_error_mm": translation_error_mm,
            "raw_rotation_error_degrees_non_symmetry_aware": (
                raw_rotation_error_degrees
            ),
            "mssd_mm": mssd_mm,
            "normalized_mssd": mssd_mm / object_data["diameter_mm"],
            "mspd_px": mspd_px,
        },
        "interpretation": (
            "MSSD and MSPD are official symmetry-aware BOP Toolkit pose errors. "
            "The raw rotation error is not symmetry-aware."
        ),
    }
    write_json_atomic(output_path, sanity)
    return sanity


def validate_manifest(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != 300:
        raise ValueError(f"M1 evaluation requires exactly 300 manifest rows, got {len(rows)}")
    indexed: dict[str, dict[str, Any]] = {}
    object_counts: Counter[int] = Counter()
    visibility_counts: Counter[str] = Counter()
    object_frames: set[tuple[int, int, int]] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate sample ID at manifest row {index}")
        object_id = int(row["object_id"])
        object_counts[object_id] += 1
        if row.get("sensor_modality") != "realsense":
            raise ValueError(f"Non-RealSense row in M1 manifest: {sample_id}")
        visible_fraction = float(row["visible_fraction"])
        if not 0.10 <= visible_fraction <= 1.01:
            raise ValueError(f"Invalid visibility for {sample_id}")
        expected_bin = visibility_bin(visible_fraction)
        if row.get("visibility_bin") != expected_bin:
            raise ValueError(
                f"Visibility-bin mismatch for {sample_id}: "
                f"{row.get('visibility_bin')} != {expected_bin}"
            )
        visibility_counts[expected_bin] += 1
        frame_key = (object_id, int(row["scene_id"]), int(row["image_id"]))
        if frame_key in object_frames:
            raise ValueError(f"Repeated object/image frame in manifest: {frame_key}")
        object_frames.add(frame_key)
        if int(row["visible_mask_pixel_count"]) <= 0:
            raise ValueError(f"Empty visible mask declared for {sample_id}")
        depth_ratio = float(row["valid_depth_ratio_inside_mask"])
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"Invalid valid-depth ratio for {sample_id}")
        image_width = int(row["image_width"])
        if image_width <= 0:
            raise ValueError(f"Invalid image width for {sample_id}: {image_width}")
        camera_matrix = np.asarray(
            row["camera_intrinsics_row_major"], dtype=np.float64
        )
        if (
            camera_matrix.shape != (3, 3)
            or not np.isfinite(camera_matrix).all()
            or camera_matrix[0, 0] <= 0
            or camera_matrix[1, 1] <= 0
            or not np.allclose(camera_matrix[2], [0, 0, 1], atol=1e-8)
        ):
            raise ValueError(f"Invalid camera intrinsics for {sample_id}")
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"]),
            f"GT {sample_id}",
            GT_ROTATION_ATOL,
        )
        indexed[sample_id] = row
    if len(object_counts) != 15 or set(object_counts.values()) != {20}:
        raise ValueError(f"Expected 15 objects x 20 rows, got {dict(object_counts)}")
    if any(visibility_counts[label] == 0 for label in ("low", "mid", "high")):
        raise ValueError(
            f"Manifest lacks global low/mid/high visibility coverage: "
            f"{dict(visibility_counts)}"
        )
    return indexed


def load_predictions(
    path: Path,
    manifest_index: dict[str, dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("Predictions JSONL must start with exactly one metadata row")
    metadata = rows[0]
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Prediction metadata manifest hash does not match the manifest")
    predictions: dict[str, dict[str, Any]] = {}
    for line_index, row in enumerate(rows[1:], start=2):
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected record type at predictions line {line_index}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in manifest_index:
            raise ValueError(f"Unknown prediction sample ID: {sample_id}")
        if sample_id in predictions:
            raise ValueError(f"Duplicate prediction sample ID: {sample_id}")
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid prediction status for {sample_id}: {status}")
        if status == "success":
            if "predicted_model_to_camera_pose_m" not in row:
                raise ValueError(f"Successful prediction has no pose: {sample_id}")
            assert_pose(
                np.asarray(row["predicted_model_to_camera_pose_m"]),
                f"prediction {sample_id}",
            )
        runtime = finite_or_none(row.get("registration_seconds"))
        if row.get("registration_seconds") is not None and (
            runtime is None or runtime < 0
        ):
            raise ValueError(f"Invalid registration runtime for {sample_id}")
        predictions[sample_id] = row
    missing = set(manifest_index) - set(predictions)
    extra = set(predictions) - set(manifest_index)
    if missing or extra:
        raise ValueError(
            f"All 300 rows must be attempted exactly once; missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    return metadata, predictions


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_rows(
    manifest_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    model_params: dict[str, Any],
    model_info: dict[int, Any],
) -> list[dict[str, Any]]:
    object_cache: dict[int, dict[str, Any]] = {}
    metrics: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows, start=1):
        sample_id = manifest_row["sample_id"]
        prediction = predictions[sample_id]
        object_id = int(manifest_row["object_id"])
        image_width = int(manifest_row["image_width"])
        scale_r = image_width / 640.0
        status = prediction["status"]
        metric: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": int(manifest_row["scene_id"]),
            "image_id": int(manifest_row["image_id"]),
            "gt_instance_index": int(manifest_row["gt_instance_index"]),
            "object_id": object_id,
            "visibility_bin": manifest_row["visibility_bin"],
            "visible_fraction": float(manifest_row["visible_fraction"]),
            "visible_mask_pixel_count": int(manifest_row["visible_mask_pixel_count"]),
            "valid_depth_ratio_inside_mask": float(
                manifest_row["valid_depth_ratio_inside_mask"]
            ),
            "image_width": image_width,
            "mspd_scale_r": scale_r,
            "status": status,
            "finite_pose": status == "success",
            "registration_seconds": finite_or_none(
                prediction.get("registration_seconds")
            ),
            "cuda_peak_memory_bytes": prediction.get("cuda_peak_memory_bytes"),
            "foundationpose_top_score": finite_or_none(
                prediction.get("foundationpose_top_score")
            ),
            "foundationpose_top_score_margin": finite_or_none(
                prediction.get("foundationpose_top_score_margin")
            ),
            "translation_error_mm": finite_or_none(
                prediction.get("translation_error_mm")
            ),
            "raw_rotation_error_degrees": finite_or_none(
                prediction.get("raw_rotation_error_degrees")
            ),
            "diameter_mm": None,
            "mssd_mm": None,
            "normalized_mssd": None,
            "mspd_px": None,
            "sample_ar_mssd": 0.0,
            "sample_ar_mspd": 0.0,
        }
        if status == "success":
            if object_id not in object_cache:
                object_cache[object_id] = load_object_evaluation_data(
                    object_id,
                    model_params,
                    model_info,
                )
            object_data = object_cache[object_id]
            predicted_pose = np.asarray(
                prediction["predicted_model_to_camera_pose_m"],
                dtype=np.float64,
            )
            gt_pose = np.asarray(
                manifest_row["gt_model_to_camera_pose_m"],
                dtype=np.float64,
            )
            expected_translation, expected_rotation = raw_pose_errors(
                predicted_pose,
                gt_pose,
            )
            if (
                metric["translation_error_mm"] is None
                or not np.isclose(
                    metric["translation_error_mm"],
                    expected_translation,
                    rtol=1e-6,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"Stored translation error mismatch for {sample_id}")
            if (
                metric["raw_rotation_error_degrees"] is None
                or not np.isclose(
                    metric["raw_rotation_error_degrees"],
                    expected_rotation,
                    rtol=1e-6,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"Stored raw rotation error mismatch for {sample_id}")
            mssd_mm, mspd_px = official_errors(
                predicted_pose,
                gt_pose,
                np.asarray(manifest_row["camera_intrinsics_row_major"]),
                object_data,
            )
            normalized_mssd = mssd_mm / object_data["diameter_mm"]
            metric.update(
                {
                    "diameter_mm": object_data["diameter_mm"],
                    "mssd_mm": mssd_mm,
                    "normalized_mssd": normalized_mssd,
                    "mspd_px": mspd_px,
                    "sample_ar_mssd": float(
                        np.mean(
                            [
                                normalized_mssd < threshold
                                for threshold in MSSD_THRESHOLD_FRACTIONS
                            ]
                        )
                    ),
                    "sample_ar_mspd": float(
                        np.mean(
                            [
                                mspd_px < multiplier * scale_r
                                for multiplier in MSPD_THRESHOLD_MULTIPLIERS
                            ]
                        )
                    ),
                }
            )
        metrics.append(metric)
        if index % 25 == 0 or index == len(manifest_rows):
            print(f"official BOP metrics: {index}/{len(manifest_rows)}", flush=True)
    return metrics


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    count = len(selected)
    if count == 0:
        return {
            "sample_count": 0,
            "finite_pose_count": 0,
            "finite_pose_rate": None,
            "ar_mssd": None,
            "ar_mspd": None,
            "oracle_mask_subset_ar_mssd_mspd": None,
            "mssd_threshold_recalls": {},
            "mspd_threshold_recalls": {},
        }
    mssd_recalls = {
        f"{threshold:.2f}d": float(
            np.mean(
                [
                    row["finite_pose"]
                    and row["normalized_mssd"] < threshold
                    for row in selected
                ]
            )
        )
        for threshold in MSSD_THRESHOLD_FRACTIONS
    }
    mspd_recalls = {
        f"{multiplier:d}r": float(
            np.mean(
                [
                    row["finite_pose"]
                    and row["mspd_px"] < multiplier * row["mspd_scale_r"]
                    for row in selected
                ]
            )
        )
        for multiplier in MSPD_THRESHOLD_MULTIPLIERS
    }
    ar_mssd = float(np.mean(list(mssd_recalls.values())))
    ar_mspd = float(np.mean(list(mspd_recalls.values())))
    finite_count = sum(bool(row["finite_pose"]) for row in selected)
    return {
        "sample_count": count,
        "finite_pose_count": finite_count,
        "finite_pose_rate": finite_count / count,
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "oracle_mask_subset_ar_mssd_mspd": (ar_mssd + ar_mspd) / 2.0,
        "mssd_threshold_recalls": mssd_recalls,
        "mspd_threshold_recalls": mspd_recalls,
    }


def percentile_summary(values: Iterable[Any]) -> dict[str, Any]:
    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite_values:
        return {"sample_count": 0, "median": None, "p95": None}
    return {
        "sample_count": len(finite_values),
        "median": float(np.median(finite_values)),
        "p95": float(np.percentile(finite_values, 95)),
    }


def correlation(
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str = "normalized_mssd",
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    pairs = [
        (finite_or_none(row.get(x_field)), finite_or_none(row.get(y_field)))
        for row in rows
        if row["finite_pose"]
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return {"sample_count": len(pairs), "rho": None, "p_value": None}
    xs, ys = zip(*pairs)
    result = spearmanr(xs, ys)
    rho = finite_or_none(result.statistic)
    p_value = finite_or_none(result.pvalue)
    return {"sample_count": len(pairs), "rho": rho, "p_value": p_value}


def depth_group(value: float) -> str:
    for label, lower, upper, inclusive_upper in DEPTH_GROUPS:
        if value >= lower and (value <= upper if inclusive_upper else value < upper):
            return label
    raise ValueError(f"Depth-validity value outside [0, 1]: {value}")


def build_summary(
    metrics: list[dict[str, Any]],
    toolkit_sha: str,
    manifest_path: Path,
    predictions_path: Path,
    prediction_metadata: dict[str, Any],
    sanity: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in metrics)
    overall = aggregate(metrics)
    by_object = []
    for object_id in sorted({int(row["object_id"]) for row in metrics}):
        entry = {"object_id": object_id}
        entry.update(aggregate(row for row in metrics if row["object_id"] == object_id))
        by_object.append(entry)
    by_visibility = []
    for label in ("low", "mid", "high"):
        entry = {"visibility_bin": label}
        entry.update(
            aggregate(row for row in metrics if row["visibility_bin"] == label)
        )
        by_visibility.append(entry)
    by_depth = []
    for label, *_ in DEPTH_GROUPS:
        entry = {"valid_depth_ratio_range": label}
        entry.update(
            aggregate(
                row
                for row in metrics
                if depth_group(row["valid_depth_ratio_inside_mask"]) == label
            )
        )
        by_depth.append(entry)

    runtimes = percentile_summary(row["registration_seconds"] for row in metrics)
    translations = percentile_summary(row["translation_error_mm"] for row in metrics)
    rotations = percentile_summary(
        row["raw_rotation_error_degrees"] for row in metrics
    )
    successful_count = overall["finite_pose_count"]
    return {
        "schema_version": 1,
        "metric_scope": {
            "combined_metric_name": "oracle_mask_subset_ar_mssd_mspd",
            "description": (
                "Arithmetic mean of subset AR_MSSD and AR_MSPD over one "
                "deterministic oracle-mask sample per manifest row."
            ),
            "failure_denominator": (
                "All 300 manifest rows; failures and non-finite poses are misses "
                "at every threshold."
            ),
            "mssd_threshold_fractions_of_diameter": list(MSSD_THRESHOLD_FRACTIONS),
            "mspd_threshold_pixel_multipliers_of_r": list(
                MSPD_THRESHOLD_MULTIPLIERS
            ),
            "mspd_r_definition": "image_width / 640",
            "threshold_comparison": "strictly less than",
        },
        "bop_toolkit": {
            "commit_sha": toolkit_sha,
            "environment": "poseloop-bop",
            "python_executable": sys.executable,
            "imported_pose_error_module": str(Path(pose_error.__file__).resolve()),
            "model_type": "models_eval",
            "max_sym_disc_step": MAX_SYMMETRY_DISCRETIZATION_STEP,
        },
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "prediction_metadata": prediction_metadata,
        },
        "counts": {
            "manifest_rows": len(metrics),
            "finite_pose_count": successful_count,
            "finite_pose_rate": successful_count / len(metrics),
            "inference_failure_count": len(metrics) - successful_count,
            "inference_failure_rate": (len(metrics) - successful_count) / len(metrics),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "overall": overall,
        "runtime_seconds": runtimes,
        "translation_error_mm_successful_poses": translations,
        "raw_rotation_error_degrees_successful_poses": rotations,
        "by_object": by_object,
        "by_visibility_bin": by_visibility,
        "by_valid_depth_ratio": by_depth,
        "spearman_successful_poses": {
            "visibility_vs_normalized_mssd": correlation(
                metrics, "visible_fraction"
            ),
            "valid_depth_ratio_vs_normalized_mssd": correlation(
                metrics, "valid_depth_ratio_inside_mask"
            ),
            "foundationpose_top_score_vs_normalized_mssd": correlation(
                metrics, "foundationpose_top_score"
            ),
            "foundationpose_top_score_margin_vs_normalized_mssd": correlation(
                metrics, "foundationpose_top_score_margin"
            ),
        },
        "m0_symmetry_check": sanity,
        "caveats": list(CAVEATS),
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def format_number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def markdown_report(summary: dict[str, Any], figure_prefix: str) -> str:
    overall = summary["overall"]
    counts = summary["counts"]
    m0 = summary["m0_symmetry_check"]["m0_prediction"]
    status_text = ", ".join(
        f"`{key}` {value}" for key, value in counts["status_counts"].items()
    )
    object_rows = "\n".join(
        "| {object_id} | {sample_count} | {finite} | {mssd} | {mspd} | {combined} |".format(
            object_id=row["object_id"],
            sample_count=row["sample_count"],
            finite=format_percent(row["finite_pose_rate"]),
            mssd=format_percent(row["ar_mssd"]),
            mspd=format_percent(row["ar_mspd"]),
            combined=format_percent(row["oracle_mask_subset_ar_mssd_mspd"]),
        )
        for row in summary["by_object"]
    )
    visibility_rows = "\n".join(
        "| {label} | {sample_count} | {finite} | {mssd} | {mspd} | {combined} |".format(
            label=row["visibility_bin"],
            sample_count=row["sample_count"],
            finite=format_percent(row["finite_pose_rate"]),
            mssd=format_percent(row["ar_mssd"]),
            mspd=format_percent(row["ar_mspd"]),
            combined=format_percent(row["oracle_mask_subset_ar_mssd_mspd"]),
        )
        for row in summary["by_visibility_bin"]
    )
    depth_rows = "\n".join(
        "| {label} | {sample_count} | {finite} | {mssd} | {mspd} | {combined} |".format(
            label=row["valid_depth_ratio_range"],
            sample_count=row["sample_count"],
            finite=format_percent(row["finite_pose_rate"]),
            mssd=format_percent(row["ar_mssd"]),
            mspd=format_percent(row["ar_mspd"]),
            combined=format_percent(row["oracle_mask_subset_ar_mssd_mspd"]),
        )
        for row in summary["by_valid_depth_ratio"]
    )
    correlation_rows = "\n".join(
        f"| {name} | {value['sample_count']} | {format_number(value['rho'])} | "
        f"{format_number(value['p_value'])} |"
        for name, value in summary["spearman_successful_poses"].items()
    )
    runtime = summary["runtime_seconds"]
    translation = summary["translation_error_mm_successful_poses"]
    rotation = summary["raw_rotation_error_degrees_successful_poses"]
    caveats = "\n".join(f"- **{text}**" for text in summary["caveats"])
    return f"""# PoseLoop M1 symmetry-aware single-view baseline

## Scope and caveats

{caveats}

## Overall diagnostic result

| Attempted | Finite poses | AR_MSSD | AR_MSPD | oracle_mask_subset_ar_mssd_mspd |
|---:|---:|---:|---:|---:|
| {counts['manifest_rows']} | {counts['finite_pose_count']} ({format_percent(counts['finite_pose_rate'])}) | {format_percent(overall['ar_mssd'])} | {format_percent(overall['ar_mspd'])} | {format_percent(overall['oracle_mask_subset_ar_mssd_mspd'])} |

Failures stay in the denominator and are misses at every threshold: {counts['inference_failure_count']} / {counts['manifest_rows']} ({format_percent(counts['inference_failure_rate'])}). Statuses: {status_text}.

Runtime over {runtime['sample_count']} timed attempts: p50 {format_number(runtime['median'])} s; p95 {format_number(runtime['p95'])} s. Successful-pose translation error: median {format_number(translation['median'])} mm, p95 {format_number(translation['p95'])} mm. Raw non-symmetry-aware rotation error: median {format_number(rotation['median'])} degrees, p95 {format_number(rotation['p95'])} degrees.

## M0 evaluator sanity

GT-vs-GT: MSSD {summary['m0_symmetry_check']['gt_vs_gt']['mssd_mm']:.9f} mm and MSPD {summary['m0_symmetry_check']['gt_vs_gt']['mspd_px']:.9f} px. The M0 smoke prediction has raw rotation error {m0['raw_rotation_error_degrees_non_symmetry_aware']:.3f} degrees, symmetry-aware MSSD {m0['mssd_mm']:.3f} mm ({m0['normalized_mssd']:.4f} diameter), and symmetry-aware MSPD {m0['mspd_px']:.3f} px.

## Per-object result

| Object | n | Finite pose rate | AR_MSSD | AR_MSPD | Combined |
|---:|---:|---:|---:|---:|---:|
{object_rows}

![Per-object AR]({figure_prefix}m1_per_object_ar.png)

## Visibility and valid depth

| Visibility | n | Finite pose rate | AR_MSSD | AR_MSPD | Combined |
|---|---:|---:|---:|---:|---:|
{visibility_rows}

![AR by visibility]({figure_prefix}m1_ar_vs_visibility.png)

| Valid-depth ratio | n | Finite pose rate | AR_MSSD | AR_MSPD | Combined |
|---|---:|---:|---:|---:|---:|
{depth_rows}

## Runtime distribution

![Registration latency distribution]({figure_prefix}m1_latency_distribution.png)

## Spearman diagnostics

Correlations use successful finite poses only; missing or constant scores are reported as unavailable. FoundationPose score fields are raw upstream scores, not calibrated confidence.

| Pair | n | rho | p-value |
|---|---:|---:|---:|
{correlation_rows}

## Reproducibility boundary

Official MSSD/MSPD calls use BOP Toolkit commit `{summary['bop_toolkit']['commit_sha']}` in the separate `{summary['bop_toolkit']['environment']}` environment, `models_eval`, and `max_sym_disc_step = {summary['bop_toolkit']['max_sym_disc_step']}`. Thresholds are MSSD 0.05 through 0.50 object diameters and MSPD 5r through 50r pixels, with `r = image_width / 640`.
"""


def save_figure_atomic(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=160, bbox_inches="tight", format="png")
    os.replace(temporary, path)


def generate_figures(summary: dict[str, Any], metrics: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    objects = summary["by_object"]
    labels = [str(row["object_id"]) for row in objects]
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(x - width / 2, [row["ar_mssd"] for row in objects], width, label="AR_MSSD")
    axis.bar(x + width / 2, [row["ar_mspd"] for row in objects], width, label="AR_MSPD")
    axis.set(
        xlabel="XYZ-IBD object ID",
        ylabel="Average recall",
        title="M1 oracle-mask subset AR by object",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axis.legend()
    save_figure_atomic(figure, output / "m1_per_object_ar.png")
    plt.close(figure)

    visibility = summary["by_visibility_bin"]
    labels = [row["visibility_bin"] for row in visibility]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(
        x - width / 2,
        [row["ar_mssd"] for row in visibility],
        width,
        label="AR_MSSD",
    )
    axis.bar(
        x + width / 2,
        [row["ar_mspd"] for row in visibility],
        width,
        label="AR_MSPD",
    )
    for index, row in enumerate(visibility):
        axis.text(index, 0.02, f"n={row['sample_count']}", ha="center", va="bottom")
    axis.set(
        xlabel="Visible-fraction bin",
        ylabel="Average recall",
        title="M1 oracle-mask subset AR by visibility",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axis.legend()
    save_figure_atomic(figure, output / "m1_ar_vs_visibility.png")
    plt.close(figure)

    runtimes = [
        row["registration_seconds"]
        for row in metrics
        if row["registration_seconds"] is not None
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    if runtimes:
        bins = min(30, max(8, int(math.sqrt(len(runtimes)))))
        axis.hist(runtimes, bins=bins, color="#4c78a8", edgecolor="white")
        axis.axvline(
            summary["runtime_seconds"]["median"],
            color="#e45756",
            linestyle="--",
            label=f"p50 {summary['runtime_seconds']['median']:.2f}s",
        )
        axis.axvline(
            summary["runtime_seconds"]["p95"],
            color="#f2cf5b",
            linestyle=":",
            label=f"p95 {summary['runtime_seconds']['p95']:.2f}s",
        )
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No finite runtime values", ha="center", va="center")
    axis.set(
        xlabel="FoundationPose registration time (seconds)",
        ylabel="Attempt count",
        title="M1 registration latency distribution",
    )
    save_figure_atomic(figure, output / "m1_latency_distribution.png")
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
    toolkit_sha = toolkit_commit(args.toolkit_root.resolve())
    model_params, model_info = load_official_models(args.dataset_root.resolve())
    sanity = run_sanity_checks(
        args.m0_result.resolve(),
        args.m0_sanity_output.resolve(),
        toolkit_sha,
        model_params,
        model_info,
    )
    gt_gt = sanity["gt_vs_gt"]
    m0 = sanity["m0_prediction"]
    print(
        "sanity passed: "
        f"GT-vs-GT MSSD={gt_gt['mssd_mm']:.9f} mm, "
        f"MSPD={gt_gt['mspd_px']:.9f} px",
        flush=True,
    )
    print(
        "M0 official symmetry-aware errors: "
        f"MSSD={m0['mssd_mm']:.6f} mm "
        f"({m0['normalized_mssd']:.6f}d), MSPD={m0['mspd_px']:.6f} px",
        flush=True,
    )
    if args.sanity_only:
        print(f"saved: {args.m0_sanity_output.resolve()}", flush=True)
        return

    manifest_path = args.manifest.resolve()
    predictions_path = args.predictions.resolve()
    manifest_rows = load_jsonl(manifest_path)
    manifest_index = validate_manifest(manifest_rows)
    manifest_sha256 = sha256_file(manifest_path)
    prediction_metadata, predictions = load_predictions(
        predictions_path,
        manifest_index,
        manifest_sha256,
    )
    metrics = evaluate_rows(
        manifest_rows,
        predictions,
        model_params,
        model_info,
    )
    summary = build_summary(
        metrics,
        toolkit_sha,
        manifest_path,
        predictions_path,
        prediction_metadata,
        sanity,
    )

    write_csv_atomic(args.metrics_csv.resolve(), metrics)
    write_json_atomic(args.summary_json.resolve(), summary)
    generate_figures(summary, metrics, args.figures_dir.resolve())
    write_text_atomic(
        args.summary_markdown.resolve(),
        markdown_report(summary, "../../reports/"),
    )
    write_text_atomic(
        args.report.resolve(),
        markdown_report(summary, ""),
    )
    print(
        "overall: "
        f"AR_MSSD={summary['overall']['ar_mssd']:.6f}, "
        f"AR_MSPD={summary['overall']['ar_mspd']:.6f}, "
        "oracle_mask_subset_ar_mssd_mspd="
        f"{summary['overall']['oracle_mask_subset_ar_mssd_mspd']:.6f}",
        flush=True,
    )
    print(f"saved: {args.summary_json.resolve()}", flush=True)
    print(f"saved: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
