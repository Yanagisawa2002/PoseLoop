#!/usr/bin/env python3
"""Run the PoseLoop M1 FoundationPose oracle-mask batch in one process."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh

from m1_common import (
    ALLOWED_STATUSES,
    BATCH_SCHEMA_VERSION,
    FOUNDATIONPOSE_ITERATIONS,
    GT_ROTATION_ATOL,
    INFERENCE_SEED,
    MODALITY,
    append_jsonl_durable,
    assert_pose,
    canonical_sha256,
    load_jsonl,
    raw_pose_errors,
    sha256_file,
    write_jsonl_atomic,
)
from run_xyzibd_smoke import (
    FOUNDATIONPOSE_COMMIT,
    git_commit,
    package_version,
    verify_checkpoints,
    verify_dataset_structure,
)


PILOT_SAMPLES_PER_OBJECT = 2
SCORE_SEMANTICS = "raw_upstream_score_not_calibrated_confidence"


class InvalidSampleInput(ValueError):
    """A single manifest sample is unusable without changing the protocol."""


class FreshRegistrationResultMissing(RuntimeError):
    """FoundationPose returned before producing fresh pose hypotheses and scores."""


class SystemicEnvironmentFailure(RuntimeError):
    """The shared inference process is no longer safe for additional samples."""


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "POSELOOP_XYZIBD_ROOT",
                Path(os.environ.get("POSELOOP_DATA_ROOT", Path.home() / "datasets"))
                / "xyzibd",
            )
        ),
    )
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to pilot_predictions.jsonl in pilot mode, predictions.jsonl otherwise.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Defaults to pilot_batch.log in pilot mode, batch.log otherwise.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a fixed robust 2-instance-per-object (30-row) pilot.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume",
        action="store_true",
        help="Validate the existing metadata/results and attempt only missing IDs.",
    )
    action.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and log with a new batch.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path, append: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        log_path,
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def validate_manifest(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != 300:
        raise ValueError(f"Expected exactly 300 manifest rows, got {len(rows)}")
    indexed: dict[str, dict[str, Any]] = {}
    object_counts: Counter[int] = Counter()
    frames: set[tuple[int, int, int]] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate sample ID at manifest row {index}")
        if row.get("sensor_modality") != MODALITY:
            raise ValueError(f"M1 supports only RealSense: {sample_id}")
        object_id = int(row["object_id"])
        frame = (object_id, int(row["scene_id"]), int(row["image_id"]))
        if frame in frames:
            raise ValueError(f"Repeated object/image frame: {frame}")
        frames.add(frame)
        object_counts[object_id] += 1
        visible_fraction = float(row["visible_fraction"])
        if not 0.10 <= visible_fraction <= 1.01:
            raise ValueError(f"Invalid visible fraction for {sample_id}")
        mask_pixels = int(row["visible_mask_pixel_count"])
        if mask_pixels <= 0:
            raise ValueError(f"Manifest declares an empty mask for {sample_id}")
        depth_ratio = float(row["valid_depth_ratio_inside_mask"])
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"Invalid depth ratio for {sample_id}")
        camera_matrix = np.asarray(
            row["camera_intrinsics_row_major"], dtype=np.float64
        )
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError(f"Invalid camera matrix for {sample_id}")
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            f"manifest GT {sample_id}",
            GT_ROTATION_ATOL,
        )
        indexed[sample_id] = row
    if len(object_counts) != 15 or set(object_counts.values()) != {20}:
        raise ValueError(f"Expected 15 objects x 20 rows, got {dict(object_counts)}")
    return indexed


def select_pilot_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    object_ids = sorted({int(row["object_id"]) for row in manifest_rows})
    for object_id in object_ids:
        object_rows = [
            row for row in manifest_rows if int(row["object_id"]) == object_id
        ]
        robust_first = sorted(
            object_rows,
            key=lambda row: (
                -float(row["valid_depth_ratio_inside_mask"]),
                -float(row["visible_fraction"]),
                -int(row["visible_mask_pixel_count"]),
                str(row["sample_id"]),
            ),
        )
        selected.extend(robust_first[:PILOT_SAMPLES_PER_OBJECT])
    selected.sort(
        key=lambda row: (
            int(row["object_id"]),
            str(row["sample_id"]),
        )
    )
    if len(selected) != 30:
        raise ValueError(f"Expected a 30-row pilot, got {len(selected)}")
    if len({int(row["object_id"]) for row in selected}) != 15:
        raise ValueError("Pilot does not cover all 15 objects")
    return selected


def choose_warmup_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot choose a warm-up sample from an empty batch")
    return min(
        rows,
        key=lambda row: (
            -float(row["valid_depth_ratio_inside_mask"]),
            -int(row["visible_mask_pixel_count"]),
            -float(row["visible_fraction"]),
            str(row["sample_id"]),
        ),
    )


def abbreviated_error(exc: BaseException, limit: int = 500) -> dict[str, str]:
    message = " ".join(str(exc).split())
    return {
        "type": type(exc).__name__,
        "message": message[:limit] if message else type(exc).__name__,
    }


def load_input(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    rgb_path = Path(row["rgb_path"])
    depth_path = Path(row["depth_path"])
    mask_path = Path(row["mask_path"])
    missing = [
        str(path) for path in (rgb_path, depth_path, mask_path) if not path.is_file()
    ]
    if missing:
        raise InvalidSampleInput(f"Missing RGB/depth/mask files: {missing}")

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None or mask_image is None:
        raise InvalidSampleInput(f"OpenCV could not read inputs for {sample_id}")
    if mask_image.ndim == 3:
        mask_image = mask_image[..., 0]
    if (
        bgr.ndim != 3
        or bgr.shape[2] != 3
        or raw_depth.ndim != 2
        or mask_image.ndim != 2
    ):
        raise InvalidSampleInput(f"Unexpected RGB/depth/mask dimensions for {sample_id}")
    if bgr.shape[:2] != raw_depth.shape or raw_depth.shape != mask_image.shape:
        raise InvalidSampleInput(f"RGB/depth/mask shape mismatch for {sample_id}")
    if (
        int(row["image_height"]) != raw_depth.shape[0]
        or int(row["image_width"]) != raw_depth.shape[1]
    ):
        raise InvalidSampleInput(f"Image dimensions changed since manifest: {sample_id}")

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = np.ascontiguousarray(mask_image > 0)
    mask_pixels = int(mask.sum())
    if mask_pixels <= 0:
        raise InvalidSampleInput(f"Visible mask is empty: {sample_id}")
    if mask_pixels != int(row["visible_mask_pixel_count"]):
        raise InvalidSampleInput(f"Visible mask changed since manifest: {sample_id}")

    depth_scale = float(row["raw_depth_scale"])
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise InvalidSampleInput(f"Invalid raw depth_scale for {sample_id}")
    depth_m = np.ascontiguousarray(
        raw_depth.astype(np.float32) * depth_scale * 0.001,
        dtype=np.float32,
    )
    valid_depth = mask & np.isfinite(depth_m) & (depth_m > 0)
    valid_depth_pixels = int(valid_depth.sum())
    valid_depth_ratio = valid_depth_pixels / mask_pixels
    if valid_depth_pixels != int(row["valid_depth_pixel_count"]) or not np.isclose(
        valid_depth_ratio,
        float(row["valid_depth_ratio_inside_mask"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise InvalidSampleInput(f"Depth validity changed since manifest: {sample_id}")
    if valid_depth_pixels < 4:
        raise InvalidSampleInput(f"Fewer than four valid masked depth pixels: {sample_id}")

    median_depth_m = float(np.median(depth_m[valid_depth]))
    if not 0.05 < median_depth_m < 5.0:
        raise InvalidSampleInput(
            f"Implausible masked depth in metres for {sample_id}: {median_depth_m}"
        )
    camera_matrix = np.ascontiguousarray(
        np.asarray(row["camera_intrinsics_row_major"], dtype=np.float64)
    )
    height, width = raw_depth.shape
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
        or not np.allclose(camera_matrix[2], [0, 0, 1], atol=1e-8)
        or not 0 <= camera_matrix[0, 2] < width
        or not 0 <= camera_matrix[1, 2] < height
    ):
        raise InvalidSampleInput(f"Invalid camera intrinsics for {sample_id}")
    gt_pose_m = np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64)
    assert_pose(gt_pose_m, f"GT pose {sample_id}", GT_ROTATION_ATOL)
    depth_to_gt_ratio = float(gt_pose_m[2, 3]) / median_depth_m
    if not 0.1 < depth_to_gt_ratio < 10.0:
        raise InvalidSampleInput(
            f"GT/depth unit consistency failed for {sample_id}: {depth_to_gt_ratio}"
        )
    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "mask": mask,
        "camera_matrix": camera_matrix,
        "gt_pose_m": gt_pose_m,
        "median_valid_masked_depth_m": median_depth_m,
    }


def load_mesh(
    row: dict[str, Any],
    data_root: Path,
    models_info: dict[str, Any],
) -> trimesh.Trimesh:
    object_id = int(row["object_id"])
    expected_path = (data_root / "models" / f"obj_{object_id:06d}.ply").resolve()
    declared_path = Path(row["model_path"]).resolve()
    if declared_path != expected_path:
        raise SystemicEnvironmentFailure(
            f"Manifest model path does not match the M0 asset root: {declared_path}"
        )
    mesh = trimesh.load(expected_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise SystemicEnvironmentFailure(f"Invalid CAD mesh: {expected_path}")
    object_info = models_info[str(object_id)]
    raw_extents_mm = np.asarray(mesh.extents, dtype=np.float64)
    expected_extents_mm = np.asarray(
        [object_info["size_x"], object_info["size_y"], object_info["size_z"]],
        dtype=np.float64,
    )
    ratios = raw_extents_mm / expected_extents_mm
    if (
        not np.isfinite(raw_extents_mm).all()
        or not np.all(raw_extents_mm > 1.0)
        or not np.all((ratios > 0.5) & (ratios < 2.0))
    ):
        raise SystemicEnvironmentFailure(
            f"Object {object_id} mesh is not in expected millimetre units"
        )
    mesh.apply_scale(0.001)
    if not 0.001 < float(np.max(mesh.extents)) < 2.0:
        raise SystemicEnvironmentFailure(
            f"Object {object_id} scaled mesh has implausible metre extents"
        )
    _ = mesh.vertex_normals
    return mesh


def clear_per_sample_estimator_state(estimator: Any) -> None:
    estimator.pose_last = None
    estimator.poses = None
    estimator.scores = None
    estimator.best_id = None


def validate_fresh_registration(
    estimator: Any,
    predicted_pose: Any,
) -> tuple[np.ndarray, int, float, float | None]:
    if estimator.poses is None or estimator.scores is None:
        raise FreshRegistrationResultMissing(
            "FoundationPose returned before producing fresh hypotheses/scores"
        )
    poses = np.asarray(estimator.poses.detach().cpu(), dtype=np.float64)
    scores = np.asarray(estimator.scores.detach().cpu(), dtype=np.float64).reshape(-1)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or scores.ndim != 1
        or len(poses) == 0
        or len(scores) != len(poses)
    ):
        raise RuntimeError(
            f"Unexpected fresh hypothesis shapes: poses={poses.shape}, scores={scores.shape}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(scores).all():
        raise FloatingPointError("FoundationPose hypotheses or scores are non-finite")
    pose = np.asarray(predicted_pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise FloatingPointError("FoundationPose returned a non-finite pose")
    assert_pose(pose, "FoundationPose predicted pose")
    if pose[2, 3] <= 0:
        raise RuntimeError("FoundationPose predicted a pose behind the camera")
    top_score = float(scores[0])
    score_margin = float(scores[0] - scores[1]) if len(scores) >= 2 else None
    return pose, int(len(poses)), top_score, score_margin


def probe_cuda_health() -> None:
    try:
        torch.cuda.empty_cache()
        probe = torch.ones(8, device="cuda")
        value = float((probe * 2).sum().item())
        torch.cuda.synchronize()
    except Exception as exc:
        raise SystemicEnvironmentFailure(
            f"CUDA health probe failed after a sample error: {exc}"
        ) from exc
    if value != 16.0:
        raise SystemicEnvironmentFailure(f"CUDA health probe returned {value}")


def base_prediction_row(row: dict[str, Any], status: str) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unknown status: {status}")
    return {
        "record_type": "prediction",
        "schema_version": BATCH_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "scene_id": int(row["scene_id"]),
        "image_id": int(row["image_id"]),
        "gt_instance_index": int(row["gt_instance_index"]),
        "object_id": int(row["object_id"]),
        "sensor_modality": row["sensor_modality"],
        "status": status,
        "gt_model_to_camera_pose_m": row["gt_model_to_camera_pose_m"],
        "visible_fraction": float(row["visible_fraction"]),
        "visibility_bin": row["visibility_bin"],
        "visible_mask_pixel_count": int(row["visible_mask_pixel_count"]),
        "valid_depth_ratio_inside_mask": float(
            row["valid_depth_ratio_inside_mask"]
        ),
    }


def attempt_sample(
    row: dict[str, Any],
    estimator: Any,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    try:
        inputs = load_input(row)
    except Exception as exc:
        logging.exception("Invalid input for %s", sample_id)
        result = base_prediction_row(row, "invalid_input")
        result["error"] = abbreviated_error(exc)
        return result

    clear_per_sample_estimator_state(estimator)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        predicted_pose = estimator.register(
            K=inputs["camera_matrix"],
            rgb=inputs["rgb"],
            depth=inputs["depth_m"],
            ob_mask=inputs["mask"],
            ob_id=int(row["object_id"]),
            iteration=FOUNDATIONPOSE_ITERATIONS,
        )
        torch.cuda.synchronize()
        registration_seconds = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated())
        (
            predicted_pose_m,
            hypothesis_count,
            top_score,
            top_score_margin,
        ) = validate_fresh_registration(estimator, predicted_pose)
    except FreshRegistrationResultMissing as exc:
        registration_seconds = time.perf_counter() - started
        logging.exception("No fresh registration result for %s", sample_id)
        probe_cuda_health()
        result = base_prediction_row(row, "invalid_input")
        result.update(
            {
                "registration_seconds": registration_seconds,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "error": abbreviated_error(exc),
            }
        )
        return result
    except FloatingPointError as exc:
        registration_seconds = time.perf_counter() - started
        logging.exception("Non-finite registration result for %s", sample_id)
        probe_cuda_health()
        result = base_prediction_row(row, "nonfinite_pose")
        result.update(
            {
                "registration_seconds": registration_seconds,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "error": abbreviated_error(exc),
            }
        )
        return result
    except Exception as exc:
        registration_seconds = time.perf_counter() - started
        logging.exception("FoundationPose inference failed for %s", sample_id)
        probe_cuda_health()
        result = base_prediction_row(row, "inference_error")
        result.update(
            {
                "registration_seconds": registration_seconds,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "error": abbreviated_error(exc),
            }
        )
        return result

    translation_error_mm, raw_rotation_error_degrees = raw_pose_errors(
        predicted_pose_m,
        inputs["gt_pose_m"],
    )
    result = base_prediction_row(row, "success")
    result.update(
        {
            "predicted_model_to_camera_pose_m": predicted_pose_m.tolist(),
            "translation_error_mm": translation_error_mm,
            "raw_rotation_error_degrees": raw_rotation_error_degrees,
            "registration_seconds": registration_seconds,
            "cuda_peak_memory_bytes": peak_memory,
            "pose_hypothesis_count": hypothesis_count,
            "foundationpose_top_score": top_score,
            "foundationpose_top_score_margin": top_score_margin,
            "foundationpose_score_semantics": SCORE_SEMANTICS,
        }
    )
    return result


def run_warmup(row: dict[str, Any], estimator: Any) -> None:
    inputs = load_input(row)
    clear_per_sample_estimator_state(estimator)
    logging.info("Starting unmeasured warm-up with sample %s", row["sample_id"])
    predicted_pose = estimator.register(
        K=inputs["camera_matrix"],
        rgb=inputs["rgb"],
        depth=inputs["depth_m"],
        ob_mask=inputs["mask"],
        ob_id=int(row["object_id"]),
        iteration=FOUNDATIONPOSE_ITERATIONS,
    )
    torch.cuda.synchronize()
    validate_fresh_registration(estimator, predicted_pose)
    clear_per_sample_estimator_state(estimator)
    logging.info("Unmeasured warm-up completed successfully")


def expected_metadata(
    args: argparse.Namespace,
    manifest_path: Path,
    effective_rows: list[dict[str, Any]],
    foundationpose_root: Path,
    foundationpose_commit: str,
    checkpoints: dict[str, Any],
    warmup_row: dict[str, Any],
) -> dict[str, Any]:
    mode = "pilot" if args.pilot else "full"
    selected_sample_ids = [str(row["sample_id"]) for row in effective_rows]
    script_dir = Path(__file__).resolve().parent
    runtime_contract = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "opencv": cv2.__version__,
        "trimesh": trimesh.__version__,
        "nvdiffrast": package_version("nvdiffrast"),
        "pytorch3d": package_version("pytorch3d"),
    }
    config = {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "mode": mode,
        "poseloop_source_sha256": {
            "run_xyzibd_batch.py": sha256_file(Path(__file__).resolve()),
            "m1_common.py": sha256_file(script_dir / "m1_common.py"),
            "run_xyzibd_smoke.py": sha256_file(script_dir / "run_xyzibd_smoke.py"),
        },
        "runtime_contract": runtime_contract,
        "manifest_sha256": sha256_file(manifest_path),
        "selected_sample_ids_sha256": canonical_sha256(selected_sample_ids),
        "selected_sample_count": len(selected_sample_ids),
        "foundationpose_commit_sha": foundationpose_commit,
        "checkpoint_sha256": {
            label: value["sha256"]
            for label, value in checkpoints["models"].items()
        },
        "registration_iteration_count": FOUNDATIONPOSE_ITERATIONS,
        "inference_seed": INFERENCE_SEED,
        "sensor_modality": MODALITY,
        "mesh_input_unit": "millimetres",
        "mesh_scale_to_metres": 0.001,
        "raw_depth_to_millimetres": "raw_depth * depth_scale",
        "depth_scale_to_metres": 0.001,
        "inference_symmetry_tfs": None,
        "warmup_registration_count": 1,
        "per_sample_retry_count": 0,
    }
    return {
        "record_type": "metadata",
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_fingerprint": canonical_sha256(config),
        "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": config["manifest_sha256"],
        "selected_sample_count": len(selected_sample_ids),
        "selected_sample_ids_sha256": config["selected_sample_ids_sha256"],
        "pilot_selection": (
            "two rows per object sorted by depth validity, visibility, mask area, "
            "and stable sample ID"
            if args.pilot
            else None
        ),
        "warmup": {
            "sample_id": warmup_row["sample_id"],
            "included_in_runtime_statistics": False,
            "performed_once_before_new_timed_attempts_per_process": True,
        },
        "foundationpose": {
            "root": str(foundationpose_root),
            "commit_sha": foundationpose_commit,
            "per_process_predictor_instantiation_count": {
                "ScorePredictor": 1,
                "PoseRefinePredictor": 1,
            },
            "per_process_raster_context_instantiation_count": 1,
            "per_process_estimator_instantiation_count": 1,
            "score_semantics": SCORE_SEMANTICS,
        },
        "checkpoints": checkpoints,
        "runtime_environment": {
            **runtime_contract,
            "cuda_device_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
        },
        "result_contract": {
            "allowed_statuses": sorted(ALLOWED_STATUSES),
            "failures_are_completed_rows": True,
            "failed_rows_are_not_retried": True,
            "resume_may_add_a_new_process_session": True,
            "shared_model_and_warmup_counts_are_per_process": True,
            "official_evaluation_failure_denominator": (
                "Failures are misses at every threshold."
            ),
        },
    }


def prepare_output(
    output_path: Path,
    metadata: dict[str, Any],
    effective_index: dict[str, dict[str, Any]],
    resume: bool,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    if output_path.exists() and overwrite:
        write_jsonl_atomic(output_path, [metadata])
        return {}
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists; use --resume or --overwrite: {output_path}"
        )
    if not output_path.exists():
        write_jsonl_atomic(output_path, [metadata])
        return {}

    existing = load_jsonl(output_path)
    if not existing or existing[0].get("record_type") != "metadata":
        raise ValueError("Existing output does not start with metadata")
    existing_metadata = existing[0]
    if existing_metadata.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise ValueError("Existing output metadata schema version does not match")
    if existing_metadata.get("batch_fingerprint") != metadata["batch_fingerprint"]:
        raise ValueError(
            "Existing batch fingerprint does not match the manifest/configuration"
        )
    if existing_metadata.get("manifest_sha256") != metadata["manifest_sha256"]:
        raise ValueError("Existing output manifest hash does not match")
    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(existing[1:], start=2):
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected record at output line {line_number}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in effective_index:
            raise ValueError(f"Existing output contains an unknown sample: {sample_id}")
        if sample_id in completed:
            raise ValueError(f"Existing output contains duplicate sample: {sample_id}")
        if row.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status in existing output for {sample_id}")
        if row.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError(f"Invalid schema version in existing row for {sample_id}")
        manifest_row = effective_index[sample_id]
        identity_fields = (
            "scene_id",
            "image_id",
            "gt_instance_index",
            "object_id",
            "sensor_modality",
        )
        for field in identity_fields:
            if row.get(field) != manifest_row.get(field):
                raise ValueError(
                    f"Existing row identity mismatch for {sample_id}: {field}"
                )
        stored_gt = np.asarray(
            row.get("gt_model_to_camera_pose_m"), dtype=np.float64
        )
        manifest_gt = np.asarray(
            manifest_row["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        if stored_gt.shape != (4, 4) or not np.allclose(
            stored_gt,
            manifest_gt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Existing row GT mismatch for {sample_id}")
        runtime = row.get("registration_seconds")
        if runtime is not None and (
            not np.isfinite(float(runtime)) or float(runtime) < 0
        ):
            raise ValueError(f"Invalid existing runtime for {sample_id}")
        if row["status"] == "success":
            predicted_pose = np.asarray(
                row.get("predicted_model_to_camera_pose_m"), dtype=np.float64
            )
            assert_pose(predicted_pose, f"existing prediction {sample_id}")
            expected_translation, expected_rotation = raw_pose_errors(
                predicted_pose,
                manifest_gt,
            )
            if not np.isclose(
                float(row.get("translation_error_mm")),
                expected_translation,
                rtol=1e-6,
                atol=1e-6,
            ) or not np.isclose(
                float(row.get("raw_rotation_error_degrees")),
                expected_rotation,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(f"Existing diagnostic errors mismatch for {sample_id}")
            scores = (
                row.get("foundationpose_top_score"),
                row.get("foundationpose_top_score_margin"),
            )
            if any(value is not None and not np.isfinite(float(value)) for value in scores):
                raise ValueError(f"Invalid existing FoundationPose score for {sample_id}")
        elif "predicted_model_to_camera_pose_m" in row:
            raise ValueError(f"Failed existing row unexpectedly has a pose: {sample_id}")
        completed[sample_id] = row
    return completed


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "artifacts" / "m1"
    output_path = (
        args.output
        if args.output is not None
        else artifact_dir
        / ("pilot_predictions.jsonl" if args.pilot else "predictions.jsonl")
    ).resolve()
    log_path = (
        args.log
        if args.log is not None
        else artifact_dir / ("pilot_batch.log" if args.pilot else "batch.log")
    ).resolve()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    foundationpose_root = args.foundationpose_root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_jsonl(manifest_path)
    validate_manifest(manifest_rows)
    effective_rows = (
        select_pilot_rows(manifest_rows) if args.pilot else list(manifest_rows)
    )
    effective_index = {row["sample_id"]: row for row in effective_rows}
    warmup_row = choose_warmup_row(effective_rows)

    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose checkout not found: {foundationpose_root}")
    dataset = verify_dataset_structure(data_root)
    foundationpose_commit = git_commit(foundationpose_root)
    if foundationpose_commit != FOUNDATIONPOSE_COMMIT:
        raise SystemicEnvironmentFailure(
            f"FoundationPose commit mismatch: {foundationpose_commit}"
        )
    checkpoints = verify_checkpoints(foundationpose_root)
    if not torch.cuda.is_available():
        raise SystemicEnvironmentFailure("PyTorch cannot access CUDA")
    torch.cuda.set_device(0)

    metadata = expected_metadata(
        args,
        manifest_path,
        effective_rows,
        foundationpose_root,
        foundationpose_commit,
        checkpoints,
        warmup_row,
    )
    completed = prepare_output(
        output_path,
        metadata,
        effective_index,
        args.resume,
        args.overwrite,
    )
    pending_rows = [
        row for row in effective_rows if row["sample_id"] not in completed
    ]
    configure_logging(log_path, append=args.resume and log_path.exists())
    logging.info(
        "PoseLoop M1 %s batch start: selected=%d completed=%d pending=%d",
        metadata["mode"],
        len(effective_rows),
        len(completed),
        len(pending_rows),
    )
    logging.info("Manifest: %s sha256=%s", manifest_path, sha256_file(manifest_path))
    logging.info("Output: %s", output_path)
    logging.info("Dataset: %s", dataset)
    if not pending_rows:
        counts = Counter(row["status"] for row in completed.values())
        print(
            f"nothing pending: {len(completed)}/{len(effective_rows)} completed; "
            f"statuses={dict(sorted(counts.items()))}",
            flush=True,
        )
        return

    os.chdir(foundationpose_root)
    sys.path.insert(0, str(foundationpose_root))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import set_seed

    # FoundationPose's Utils module reloads Python logging at import time.
    # Restore our durable file handler after that upstream side effect.
    configure_logging(log_path, append=True)
    logging.info("FoundationPose imports complete; durable batch logging restored")

    models_info_path = data_root / "models" / "models_info.json"
    from m1_common import read_json

    models_info = read_json(models_info_path)
    mesh_cache: dict[int, trimesh.Trimesh] = {}

    def object_mesh(row: dict[str, Any]) -> trimesh.Trimesh:
        object_id = int(row["object_id"])
        if object_id not in mesh_cache:
            mesh_cache[object_id] = load_mesh(row, data_root, models_info)
        return mesh_cache[object_id]

    warmup_mesh = object_mesh(warmup_row)
    try:
        set_seed(INFERENCE_SEED)
        logging.info("Instantiating shared ScorePredictor")
        scorer = ScorePredictor()
        logging.info("Instantiating shared PoseRefinePredictor")
        refiner = PoseRefinePredictor()
        logging.info("Instantiating shared CUDA rasterization context")
        glctx = dr.RasterizeCudaContext(device=0)
        logging.info("Instantiating one shared FoundationPose estimator")
        estimator = FoundationPose(
            model_pts=warmup_mesh.vertices.copy(),
            model_normals=warmup_mesh.vertex_normals.copy(),
            symmetry_tfs=None,
            mesh=warmup_mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=str(artifact_dir / "foundationpose_debug"),
        )
        current_object_id = int(warmup_row["object_id"])
        run_warmup(warmup_row, estimator)
    except Exception as exc:
        logging.exception("Shared predictor/context/estimator warm-up failed")
        raise SystemicEnvironmentFailure(
            f"Predictor/context/estimator warm-up failed: {exc}"
        ) from exc

    newly_attempted = 0
    total_selected = len(effective_rows)
    for row in effective_rows:
        sample_id = row["sample_id"]
        if sample_id in completed:
            continue
        object_id = int(row["object_id"])
        if object_id != current_object_id:
            mesh = object_mesh(row)
            try:
                set_seed(INFERENCE_SEED)
                estimator.reset_object(
                    model_pts=mesh.vertices.copy(),
                    model_normals=mesh.vertex_normals.copy(),
                    symmetry_tfs=None,
                    mesh=mesh,
                )
                clear_per_sample_estimator_state(estimator)
                current_object_id = object_id
            except Exception as exc:
                logging.exception("Object reset failed for object %d", object_id)
                raise SystemicEnvironmentFailure(
                    f"FoundationPose object reset failed for object {object_id}: {exc}"
                ) from exc

        result = attempt_sample(row, estimator)
        append_jsonl_durable(output_path, result)
        completed[sample_id] = result
        newly_attempted += 1
        count = len(completed)
        print(
            f"[{count:03d}/{total_selected:03d}] object={object_id:02d} "
            f"sample={sample_id} status={result['status']} "
            f"time={result.get('registration_seconds', float('nan')):.3f}s",
            flush=True,
        )
        logging.info(
            "Completed %s status=%s registration_seconds=%s",
            sample_id,
            result["status"],
            result.get("registration_seconds"),
        )

    if len(completed) != total_selected:
        raise RuntimeError(
            f"Batch exited with unattempted rows: {len(completed)}/{total_selected}"
        )
    counts = Counter(row["status"] for row in completed.values())
    logging.info(
        "PoseLoop M1 %s batch complete: newly_attempted=%d statuses=%s",
        metadata["mode"],
        newly_attempted,
        dict(sorted(counts.items())),
    )
    print(
        f"complete: {len(completed)}/{total_selected} attempted; "
        f"statuses={dict(sorted(counts.items()))}",
        flush=True,
    )
    print(f"saved: {output_path}", flush=True)
    print(f"log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
