#!/usr/bin/env python3
"""Fit and freeze the PoseLoop M4 pre-acquisition utility regressor.

This script is deliberately development-only. It reads M1/M2 artifacts, builds
four target-candidate rows for every M2 target, evaluates their post-acquisition
two-view max-mask outcomes, runs physical-instance-grouped out-of-fold
diagnostics, and freezes one compact regressor. It never opens an M3 artifact.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m1_common import (  # noqa: E402
    canonical_sha256,
    load_jsonl,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
CV_FOLDS = 5
CV_SEED = 20260730
EXPECTED_TARGET_COUNT = 300
EXPECTED_CANDIDATES_PER_TARGET = 4
EXPECTED_DEVELOPMENT_ROW_COUNT = EXPECTED_TARGET_COUNT * EXPECTED_CANDIDATES_PER_TARGET
EXPECTED_OBJECT_COUNT = 15
IMAGE_AREA_EPSILON = 1e-12
MAX_GT_TRANSFORM_NORMALIZED_MSSD = 5e-4
MAX_GT_TRANSFORM_MSPD_PX = 0.01
NEW_SURFACE_FEATURE_NAME = "cad_new_surface_fraction"

FEATURE_NAMES = (
    "target_pose_usable",
    "target_mask_area_fraction",
    "target_bbox_width_fraction",
    "target_bbox_height_fraction",
    "target_bbox_area_fraction",
    "target_mask_bbox_fill_fraction",
    "target_bbox_aspect_log",
    "target_mask_centroid_x_normalized",
    "target_mask_centroid_y_normalized",
    "target_valid_depth_ratio",
    "target_depth_median_over_diameter",
    "target_depth_iqr_over_diameter",
    "target_depth_std_over_diameter",
    "target_predicted_depth_over_diameter",
    "target_predicted_translation_norm_over_diameter",
    "target_predicted_lateral_over_depth",
    "target_foundationpose_top_score",
    "target_foundationpose_top_score_margin",
    "cad_extent_x_over_diameter",
    "cad_extent_y_over_diameter",
    "cad_extent_z_over_diameter",
    "cad_min_over_max_extent",
    "cad_surface_area_over_diameter_sq",
    "cad_volume_over_diameter_cubed",
    "relative_view_angle_rad",
    "relative_azimuth_change_sin",
    "relative_azimuth_change_cos",
    "relative_elevation_change_rad",
    "camera_baseline_over_target_distance",
    "camera_baseline_over_diameter",
    "candidate_distance_over_target_distance",
    "candidate_distance_delta_over_diameter",
    "target_view_direction_object_x",
    "target_view_direction_object_y",
    "target_view_direction_object_z",
    "candidate_view_direction_object_x",
    "candidate_view_direction_object_y",
    "candidate_view_direction_object_z",
    "candidate_center_delta_object_x_over_diameter",
    "candidate_center_delta_object_y_over_diameter",
    "candidate_center_delta_object_z_over_diameter",
    "cad_target_visible_area_fraction",
    "cad_candidate_front_facing_area_fraction",
    "cad_visible_area_ratio",
    NEW_SURFACE_FEATURE_NAME,
    "cad_target_projected_size_proxy",
    "cad_candidate_projected_size_proxy",
    "cad_projected_size_ratio",
)

FORBIDDEN_FEATURE_TOKENS = (
    "candidate_rgb",
    "candidate_depth",
    "candidate_mask",
    "candidate_prediction",
    "candidate_score",
    "gt_",
    "ground_truth",
    "visible_fraction",
    "visibility_bin",
    "sample_id",
    "scene_id",
    "object_id",
    "group_id",
    "physical_instance",
    "future",
    "outcome",
    "utility",
    "success",
    "error",
)

MODEL_PARAMETERS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": CV_SEED,
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "artifacts" / "m4"
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
        "--m2-groups",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--m2-candidate-manifest",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--m2-groups-summary",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups_summary.json",
    )
    parser.add_argument(
        "--m2-extrinsics-audit",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "extrinsics_audit.json",
    )
    parser.add_argument(
        "--m2-view-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "view_predictions.jsonl",
    )
    parser.add_argument(
        "--m2-metrics",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "metrics.jsonl",
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
        "--features-out",
        type=Path,
        default=artifact_root / "development_features.jsonl",
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=artifact_root / "development_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--fold-audit-out",
        type=Path,
        default=artifact_root / "development_fold_audit.json",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=artifact_root / "development_diagnostics.json",
    )
    parser.add_argument(
        "--feature-audit-out",
        type=Path,
        default=artifact_root / "feature_audit.json",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=artifact_root / "voi_regressor.joblib",
    )
    parser.add_argument(
        "--frozen-out",
        type=Path,
        default=artifact_root / "frozen_voi.json",
    )
    return parser.parse_args()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError(f"{label} is degenerate")
    return vector / norm


def _camera_center_world(world_to_camera_m: np.ndarray) -> np.ndarray:
    transform = np.asarray(world_to_camera_m, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Camera extrinsic must be a finite 4x4 matrix")
    return np.linalg.inv(transform)[:3, 3]


def _resolve_dataset_path(dataset_root: Path, stored_path: Any) -> Path:
    raw = str(stored_path)
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()
    posix = PurePosixPath(raw.replace("\\", "/"))
    parts = list(posix.parts)
    try:
        dataset_index = parts.index(dataset_root.name)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Cannot relocate dataset path under {dataset_root}: {raw}"
        ) from exc
    relocated = dataset_root.joinpath(*parts[dataset_index + 1 :])
    if not relocated.is_file():
        raise FileNotFoundError(f"Relocated dataset file is missing: {relocated}")
    return relocated.resolve()


def load_cad_geometry(dataset_root: Path, object_id: int) -> dict[str, Any]:
    """Load deterministic face-normal/face-area CAD geometry in metres."""
    import trimesh

    dataset_root = Path(dataset_root).resolve()
    models_info_path = dataset_root / "models" / "models_info.json"
    model_path = dataset_root / "models" / f"obj_{int(object_id):06d}.ply"
    models_info = read_json(models_info_path)
    info = models_info.get(str(int(object_id)))
    if not isinstance(info, Mapping):
        raise KeyError(f"Object {object_id} is absent from {models_info_path}")
    mesh = trimesh.load(model_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"CAD file is not a triangle mesh: {model_path}")
    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas_m2 = np.asarray(mesh.area_faces, dtype=np.float64) * 1e-6
    if (
        vertices_mm.ndim != 2
        or vertices_mm.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or normals.shape != (len(faces), 3)
        or face_areas_m2.shape != (len(faces),)
        or len(faces) == 0
        or not np.all(np.isfinite(vertices_mm))
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(face_areas_m2))
        or np.any(face_areas_m2 <= 0.0)
    ):
        raise ValueError(f"Invalid CAD face geometry: {model_path}")
    diameter_m = _finite(info["diameter"], "CAD diameter mm") * 0.001
    extents_m = (np.max(vertices_mm, axis=0) - np.min(vertices_mm, axis=0)) * 0.001
    surface_area_m2 = float(np.sum(face_areas_m2))
    volume_m3 = abs(_finite(mesh.volume, "CAD volume mm3")) * 1e-9
    if diameter_m <= 0.0 or np.any(extents_m <= 0.0) or surface_area_m2 <= 0.0:
        raise ValueError(f"Invalid CAD dimensions for object {object_id}")
    return {
        "object_id": int(object_id),
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "models_info_path": models_info_path,
        "models_info_sha256": sha256_file(models_info_path),
        "diameter_m": diameter_m,
        "extents_m": extents_m,
        "surface_area_m2": surface_area_m2,
        "volume_m3": volume_m3,
        "face_normals_object": normals,
        "face_areas_m2": face_areas_m2,
    }


def _target_observation_features(
    target_manifest: Mapping[str, Any],
    dataset_root: Path,
    diameter_m: float,
) -> dict[str, float]:
    import cv2

    mask_path = _resolve_dataset_path(dataset_root, target_manifest["mask_path"])
    depth_path = _resolve_dataset_path(dataset_root, target_manifest["depth_path"])
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None or depth_raw is None:
        raise ValueError(
            f"Failed to decode target mask/depth: {mask_path}, {depth_path}"
        )
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[..., 0]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    mask = np.asarray(mask_raw) > 0
    depth = np.asarray(depth_raw, dtype=np.float64)
    if mask.shape != depth.shape or not np.any(mask):
        raise ValueError(f"Invalid target mask/depth shapes for {mask_path}")
    height, width = mask.shape
    if (
        int(target_manifest["image_width"]) != width
        or int(target_manifest["image_height"]) != height
    ):
        raise ValueError("Target observation dimensions differ from the manifest")
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels != int(target_manifest["visible_mask_pixel_count"]):
        raise ValueError("Decoded target mask count differs from the manifest")
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        raise ValueError("Target mask contains no valid depth")
    valid_ratio = valid_pixels / mask_pixels
    expected_valid_ratio = _finite(
        target_manifest["valid_depth_ratio_inside_mask"],
        "target manifest valid-depth ratio",
    )
    if not math.isclose(valid_ratio, expected_valid_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Decoded target valid-depth ratio differs from the manifest")

    ys, xs = np.nonzero(mask)
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    bbox_area = bbox_width * bbox_height
    depth_scale_mm = _finite(target_manifest["raw_depth_scale"], "raw depth scale")
    depths_m = depth[valid] * depth_scale_mm * 0.001
    q25, median, q75 = np.percentile(depths_m, [25.0, 50.0, 75.0])
    return {
        "target_mask_area_fraction": mask_pixels / (width * height),
        "target_bbox_width_fraction": bbox_width / width,
        "target_bbox_height_fraction": bbox_height / height,
        "target_bbox_area_fraction": bbox_area / (width * height),
        "target_mask_bbox_fill_fraction": mask_pixels / bbox_area,
        "target_bbox_aspect_log": float(math.log(bbox_width / bbox_height)),
        "target_mask_centroid_x_normalized": (float(np.mean(xs)) - (width - 1) / 2.0)
        / max(width / 2.0, 1.0),
        "target_mask_centroid_y_normalized": (float(np.mean(ys)) - (height - 1) / 2.0)
        / max(height / 2.0, 1.0),
        "target_valid_depth_ratio": valid_ratio,
        "target_depth_median_over_diameter": float(median / diameter_m),
        "target_depth_iqr_over_diameter": float((q75 - q25) / diameter_m),
        "target_depth_std_over_diameter": float(np.std(depths_m) / diameter_m),
    }


def extract_target_context(
    group: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    target_prediction: Mapping[str, Any],
    cad_geometry: Mapping[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Build reusable target-only context before any candidate is acquired."""
    views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
    if [int(row["acquisition_rank"]) for row in views] != list(range(5)):
        raise ValueError("M4 requires target rank 0 plus exactly four candidates")
    target_view = views[0]
    diameter_m = _finite(cad_geometry["diameter_m"], "CAD diameter")
    world_to_target_camera = np.asarray(
        target_view["camera_world_to_camera_pose_m"], dtype=np.float64
    )
    target_camera_center_world = _camera_center_world(world_to_target_camera)

    raw_pose = target_prediction.get("predicted_model_to_camera_pose_m")
    pose = np.asarray(raw_pose, dtype=np.float64)
    pose_usable = bool(
        target_prediction.get("status") == "success"
        and pose.shape == (4, 4)
        and np.all(np.isfinite(pose))
    )
    observation = _target_observation_features(
        target_manifest, Path(dataset_root), diameter_m
    )
    if pose_usable:
        model_to_world = np.linalg.inv(world_to_target_camera) @ pose
        object_center_world = model_to_world[:3, 3]
        object_rotation_world = model_to_world[:3, :3]
        translation = pose[:3, 3]
    else:
        camera_to_world = np.linalg.inv(world_to_target_camera)
        fallback_depth = observation["target_depth_median_over_diameter"] * diameter_m
        fallback_camera = np.asarray([0.0, 0.0, fallback_depth, 1.0])
        object_center_world = (camera_to_world @ fallback_camera)[:3]
        object_rotation_world = camera_to_world[:3, :3]
        translation = np.asarray([0.0, 0.0, fallback_depth])

    target_direction_world = _unit(
        target_camera_center_world - object_center_world,
        "target object-to-camera direction",
    )
    target_direction_object = _unit(
        object_rotation_world.T @ target_direction_world,
        "target viewing direction in object coordinates",
    )
    target_distance = float(
        np.linalg.norm(target_camera_center_world - object_center_world)
    )
    if target_distance <= 1e-9:
        raise ValueError("Predicted object centre coincides with target camera")
    predicted_depth = float(translation[2])
    predicted_lateral = float(np.linalg.norm(translation[:2]))
    static_features = {
        "target_pose_usable": float(pose_usable),
        **observation,
        "target_predicted_depth_over_diameter": predicted_depth / diameter_m,
        "target_predicted_translation_norm_over_diameter": float(
            np.linalg.norm(translation) / diameter_m
        ),
        "target_predicted_lateral_over_depth": predicted_lateral
        / max(abs(predicted_depth), 1e-6),
        "target_foundationpose_top_score": _finite(
            target_prediction.get("foundationpose_top_score", 0.0),
            "target FoundationPose top score",
        ),
        "target_foundationpose_top_score_margin": _finite(
            target_prediction.get("foundationpose_top_score_margin", 0.0),
            "target FoundationPose score margin",
        ),
    }
    camera_matrix = np.asarray(
        target_manifest["camera_intrinsics_row_major"], dtype=np.float64
    ).reshape(3, 3)
    return {
        "static_features": static_features,
        "object_center_world": object_center_world,
        "object_rotation_world": object_rotation_world,
        "target_camera_center_world": target_camera_center_world,
        "target_view_direction_object": target_direction_object,
        "target_distance_m": target_distance,
        "camera_matrix": camera_matrix,
        "image_width": int(target_manifest["image_width"]),
        "image_height": int(target_manifest["image_height"]),
    }


def _azimuth_elevation(direction: np.ndarray) -> tuple[float, float]:
    direction = _unit(direction, "spherical direction")
    azimuth = float(math.atan2(direction[1], direction[0]))
    elevation = float(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
    return azimuth, elevation


def extract_candidate_features(
    group: Mapping[str, Any],
    candidate_slot: int,
    target_context: Mapping[str, Any],
    cad_geometry: Mapping[str, Any],
) -> dict[str, float]:
    """Extract one audited feature row without candidate observations or GT."""
    slot = int(candidate_slot)
    if slot not in range(1, EXPECTED_CANDIDATES_PER_TARGET + 1):
        raise ValueError(f"Candidate slot must be 1..4, got {candidate_slot}")
    views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
    if [int(row["acquisition_rank"]) for row in views] != list(range(5)):
        raise ValueError("M4 requires exactly four explicit candidate views")
    candidate = views[slot]
    candidate_camera_center_world = _camera_center_world(
        np.asarray(candidate["camera_world_to_camera_pose_m"], dtype=np.float64)
    )
    object_center_world = np.asarray(
        target_context["object_center_world"], dtype=np.float64
    )
    object_rotation_world = np.asarray(
        target_context["object_rotation_world"], dtype=np.float64
    )
    target_camera_center_world = np.asarray(
        target_context["target_camera_center_world"], dtype=np.float64
    )
    target_direction_object = _unit(
        np.asarray(target_context["target_view_direction_object"], dtype=np.float64),
        "target object direction",
    )
    candidate_direction_world = _unit(
        candidate_camera_center_world - object_center_world,
        "candidate object-to-camera direction",
    )
    candidate_direction_object = _unit(
        object_rotation_world.T @ candidate_direction_world,
        "candidate viewing direction in object coordinates",
    )
    target_distance = _finite(target_context["target_distance_m"], "target distance")
    candidate_distance = float(
        np.linalg.norm(candidate_camera_center_world - object_center_world)
    )
    baseline_world = candidate_camera_center_world - target_camera_center_world
    baseline = float(np.linalg.norm(baseline_world))
    diameter_m = _finite(cad_geometry["diameter_m"], "CAD diameter")
    baseline_object = object_rotation_world.T @ baseline_world

    dot = float(
        np.clip(np.dot(target_direction_object, candidate_direction_object), -1.0, 1.0)
    )
    relative_angle = float(math.acos(dot))
    target_azimuth, target_elevation = _azimuth_elevation(target_direction_object)
    candidate_azimuth, candidate_elevation = _azimuth_elevation(
        candidate_direction_object
    )
    azimuth_delta = float(
        math.atan2(
            math.sin(candidate_azimuth - target_azimuth),
            math.cos(candidate_azimuth - target_azimuth),
        )
    )
    elevation_delta = candidate_elevation - target_elevation

    normals = np.asarray(cad_geometry["face_normals_object"], dtype=np.float64)
    areas = np.asarray(cad_geometry["face_areas_m2"], dtype=np.float64)
    surface_area = _finite(cad_geometry["surface_area_m2"], "CAD surface area")
    target_cosines = normals @ target_direction_object
    candidate_cosines = normals @ candidate_direction_object
    target_projected_area = float(np.sum(areas * np.clip(target_cosines, 0.0, None)))
    candidate_projected_area = float(
        np.sum(areas * np.clip(candidate_cosines, 0.0, None))
    )
    new_surface_area = float(
        np.sum(areas[(candidate_cosines > 0.0) & (target_cosines <= 0.0)])
    )
    camera_matrix = np.asarray(target_context["camera_matrix"], dtype=np.float64)
    focal_product = float(camera_matrix[0, 0] * camera_matrix[1, 1])
    image_area = int(target_context["image_width"]) * int(
        target_context["image_height"]
    )
    target_size_proxy = (
        (target_projected_area / max(target_distance**2, IMAGE_AREA_EPSILON))
        * focal_product
        / image_area
    )
    candidate_size_proxy = (
        (candidate_projected_area / max(candidate_distance**2, IMAGE_AREA_EPSILON))
        * focal_product
        / image_area
    )

    extents = np.asarray(cad_geometry["extents_m"], dtype=np.float64)
    features = {
        **dict(target_context["static_features"]),
        "cad_extent_x_over_diameter": float(extents[0] / diameter_m),
        "cad_extent_y_over_diameter": float(extents[1] / diameter_m),
        "cad_extent_z_over_diameter": float(extents[2] / diameter_m),
        "cad_min_over_max_extent": float(np.min(extents) / np.max(extents)),
        "cad_surface_area_over_diameter_sq": surface_area / (diameter_m**2),
        "cad_volume_over_diameter_cubed": _finite(
            cad_geometry["volume_m3"], "CAD volume"
        )
        / (diameter_m**3),
        "relative_view_angle_rad": relative_angle,
        "relative_azimuth_change_sin": float(math.sin(azimuth_delta)),
        "relative_azimuth_change_cos": float(math.cos(azimuth_delta)),
        "relative_elevation_change_rad": elevation_delta,
        "camera_baseline_over_target_distance": baseline / target_distance,
        "camera_baseline_over_diameter": baseline / diameter_m,
        "candidate_distance_over_target_distance": candidate_distance / target_distance,
        "candidate_distance_delta_over_diameter": (candidate_distance - target_distance)
        / diameter_m,
        "target_view_direction_object_x": float(target_direction_object[0]),
        "target_view_direction_object_y": float(target_direction_object[1]),
        "target_view_direction_object_z": float(target_direction_object[2]),
        "candidate_view_direction_object_x": float(candidate_direction_object[0]),
        "candidate_view_direction_object_y": float(candidate_direction_object[1]),
        "candidate_view_direction_object_z": float(candidate_direction_object[2]),
        "candidate_center_delta_object_x_over_diameter": float(
            baseline_object[0] / diameter_m
        ),
        "candidate_center_delta_object_y_over_diameter": float(
            baseline_object[1] / diameter_m
        ),
        "candidate_center_delta_object_z_over_diameter": float(
            baseline_object[2] / diameter_m
        ),
        "cad_target_visible_area_fraction": target_projected_area / surface_area,
        "cad_candidate_front_facing_area_fraction": candidate_projected_area
        / surface_area,
        "cad_visible_area_ratio": candidate_projected_area
        / max(target_projected_area, IMAGE_AREA_EPSILON),
        NEW_SURFACE_FEATURE_NAME: new_surface_area / surface_area,
        "cad_target_projected_size_proxy": target_size_proxy,
        "cad_candidate_projected_size_proxy": candidate_size_proxy,
        "cad_projected_size_ratio": candidate_size_proxy
        / max(target_size_proxy, IMAGE_AREA_EPSILON),
    }
    if set(features) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(features))
        extra = sorted(set(features) - set(FEATURE_NAMES))
        raise RuntimeError(
            f"M4 feature contract mismatch: missing={missing}, extra={extra}"
        )
    ordered = {
        name: _finite(features[name], f"feature {name}") for name in FEATURE_NAMES
    }
    return ordered


def _ordered_feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("Feature mapping does not exactly match FEATURE_NAMES")
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError("Feature vector is non-finite or mis-shaped")
    return vector


def load_frozen_regressor(frozen: Mapping[str, Any], frozen_path: Path) -> Any:
    """Load and hash-check the frozen joblib regressor."""
    import joblib

    model_contract = frozen["model"]
    if model_contract["serialization"] != "joblib":
        raise ValueError("Unsupported frozen M4 model serialization")
    model_path = Path(frozen_path).resolve().parent / str(model_contract["artifact"])
    if sha256_file(model_path) != str(model_contract["sha256"]):
        raise ValueError("Frozen M4 model artifact hash mismatch")
    return joblib.load(model_path)


def predict_frozen_utility(
    features: Mapping[str, Any],
    frozen: Mapping[str, Any],
    regressor: Any,
) -> float:
    """Apply the frozen regressor in its declared exact feature order."""
    if tuple(frozen["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Frozen M4 feature order differs from this implementation")
    prediction = float(
        regressor.predict(_ordered_feature_vector(features).reshape(1, -1))[0]
    )
    return float(np.clip(prediction, 0.0, 1.0))


def _feature_audit() -> dict[str, Any]:
    token_hits = {
        name: [token for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()]
        for name in FEATURE_NAMES
    }
    token_hits = {name: hits for name, hits in token_hits.items() if hits}
    if token_hits:
        raise RuntimeError(f"Forbidden feature-name audit failed: {token_hits}")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "new_surface_feature_name": NEW_SURFACE_FEATURE_NAME,
        "pre_acquisition_only": True,
        "target_observation_scope": (
            "The already acquired target mask, target depth, target FoundationPose "
            "prediction, target intrinsics, and calibrated target extrinsic."
        ),
        "candidate_geometry_source_fields": [
            "acquisition_rank",
            "camera_world_to_camera_pose_m",
        ],
        "cad_source_fields": [
            "mesh vertices",
            "mesh face normals",
            "mesh face areas",
            "models_info diameter",
        ],
        "candidate_observation_files_opened": False,
        "candidate_prediction_rows_accessed_by_feature_extractor": False,
        "group_viewing_direction_world_accessed": False,
        "uses_group_viewing_direction_world": False,
        "candidate_gt_pose_accessed_by_feature_extractor": False,
        "candidate_observation_or_outcome_feature_count": 0,
        "ground_truth_use": (
            "M2 GT-derived max-mask pair outcomes are labels/evaluation evidence "
            "only and are structurally separate from the features mapping."
        ),
        "known_target_mask_scope": (
            "The target visible mask is treated as the currently observed mask, "
            "consistent with the existing PoseLoop oracle-mask diagnostic scope."
        ),
        "explicitly_forbidden_feature_sources": [
            "candidate RGB",
            "candidate depth",
            "candidate mask or visible-mask area",
            "candidate FoundationPose prediction or score",
            "candidate GT visibility or pose",
            "candidate pose error, utility, or success",
            "group.viewing_direction_world because its object centre is GT-derived",
            "sample, group, scene, physical-instance, or object identity",
            "all M3 artifacts",
        ],
        "identity_metadata_use": (
            "Development identities are retained only for joins, physical-instance "
            "fold grouping, and macro-object diagnostics; none enters features."
        ),
        "feature_name_forbidden_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "feature_name_forbidden_token_hits": token_hits,
        "forbidden_feature_name_token_hits": token_hits,
        "machine_readable_forbidden_sources": {
            "candidate_rgb": False,
            "candidate_depth": False,
            "candidate_mask": False,
            "candidate_gt": False,
            "candidate_outcome": False,
            "viewing_direction_world": False,
        },
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def _validate_paths(
    repo_root: Path,
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> None:
    m1_root = (repo_root / "artifacts" / "m1").resolve()
    m2_root = (repo_root / "artifacts" / "m2").resolve()
    m4_root = (repo_root / "artifacts" / "m4").resolve()
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing M4 development input {label}: {path}")
        expected_root = m1_root if label.startswith("m1_") else m2_root
        if not path.is_relative_to(expected_root):
            raise ValueError(
                f"M4 fitter input {label} must remain under {expected_root}: {path}"
            )
    for label, path in output_paths.items():
        if not path.is_relative_to(m4_root):
            raise ValueError(f"M4 output {label} must remain under {m4_root}: {path}")
    if len(set(output_paths.values())) != len(output_paths):
        raise ValueError("M4 output paths must be unique")


def _load_prepared_m2(
    args: argparse.Namespace,
    input_paths: Mapping[str, Path],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    toolkit_root = args.toolkit_root.resolve()
    if str(toolkit_root) not in sys.path:
        sys.path.insert(0, str(toolkit_root))
    from evaluate_m1 import (
        load_official_models,
        toolkit_commit,
        validate_manifest,
    )
    from evaluate_m2 import (
        load_candidate_manifest,
        load_m1_predictions,
        load_view_predictions,
        prepare_groups,
        validate_groups,
        validate_m2_provenance,
    )

    toolkit_sha = toolkit_commit(toolkit_root)
    model_params, model_info = load_official_models(args.dataset_root.resolve())
    manifest_rows = load_jsonl(input_paths["m1_manifest"])
    manifest_index = validate_manifest(manifest_rows)
    manifest_sha256 = sha256_file(input_paths["m1_manifest"])
    _, m1_predictions = load_m1_predictions(
        input_paths["m1_predictions"], manifest_index, manifest_sha256
    )
    groups, m2_source_ids = validate_groups(
        load_jsonl(input_paths["m2_groups"]), manifest_index
    )
    candidate_index = load_candidate_manifest(
        input_paths["m2_candidate_manifest"], m2_source_ids
    )
    m2_metadata, m2_predictions = load_view_predictions(
        input_paths["m2_view_predictions"], candidate_index
    )
    required_paths = {
        "groups": input_paths["m2_groups"],
        "candidate_manifest": input_paths["m2_candidate_manifest"],
        "groups_summary": input_paths["m2_groups_summary"],
        "extrinsics_audit": input_paths["m2_extrinsics_audit"],
        "m1_manifest": input_paths["m1_manifest"],
        "m1_predictions": input_paths["m1_predictions"],
    }
    validate_m2_provenance(m2_metadata, required_paths, candidate_index)
    prepared, transform_audit = prepare_groups(
        groups,
        manifest_index,
        m1_predictions,
        m2_predictions,
        model_params,
        model_info,
        MAX_GT_TRANSFORM_NORMALIZED_MSSD,
        MAX_GT_TRANSFORM_MSPD_PX,
    )
    return (
        prepared,
        manifest_index,
        {
            "toolkit_commit_sha": toolkit_sha,
            "gt_transform_audit": transform_audit,
        },
    )


def _pair_outcome(prepared: Mapping[str, Any], candidate_slot: int) -> dict[str, Any]:
    from evaluate_m2 import result_from_selected

    target = prepared["views"][0]
    candidate = prepared["views"][candidate_slot]
    acquired = [target, candidate]
    selected = max(
        acquired,
        key=lambda view: (
            int(view["view"]["visible_mask_pixel_count"]),
            -int(view["view"]["acquisition_rank"]),
        ),
    )
    result = result_from_selected(
        prepared,
        acquired,
        selected,
        "m4_candidate_utility",
        2,
    )
    sample_ar_mssd = _finite(result["sample_ar_mssd"], "sample AR_MSSD")
    sample_ar_mspd = _finite(result["sample_ar_mspd"], "sample AR_MSPD")
    utility = 0.5 * (sample_ar_mssd + sample_ar_mspd)
    return {
        "actual_utility": utility,
        "joint_success": bool(result["diagnostic_success"]["joint"]),
        "sample_ar_mssd": sample_ar_mssd,
        "sample_ar_mspd": sample_ar_mspd,
        "outcome_evidence": {
            "post_acquisition_rule": "max_mask",
            "post_acquisition_rule_description": (
                "max visible-mask pixel count over target and acquired candidate; "
                "ties prefer the target/lower acquisition rank"
            ),
            "target_visible_mask_pixel_count": int(
                target["view"]["visible_mask_pixel_count"]
            ),
            "candidate_visible_mask_pixel_count": int(
                candidate["view"]["visible_mask_pixel_count"]
            ),
            "selected_sample_id": result["selected_sample_id"],
            "selected_acquisition_rank": result["selected_acquisition_rank"],
            "status": result["status"],
            "finite_pose": bool(result["finite_pose"]),
            "normalized_mssd": result["normalized_mssd"],
            "mspd_px": result["mspd_px"],
            "mspd_scale_r": result["mspd_scale_r"],
        },
    }


def _build_development_rows(
    prepared_groups: Sequence[dict[str, Any]],
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if len(prepared_groups) != EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"M4 development needs {EXPECTED_TARGET_COUNT} targets, "
            f"got {len(prepared_groups)}"
        )
    cad_cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    track_objects: dict[str, int] = {}
    for group_index, prepared in enumerate(prepared_groups, start=1):
        group = prepared["group"]
        group_id = str(group["group_id"])
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate M2 group ID: {group_id}")
        seen_group_ids.add(group_id)
        object_id = int(group["object_id"])
        track_id = str(group["oracle_association"]["track_id"])
        if not track_id:
            raise ValueError(f"Missing M2 physical-instance track: {group_id}")
        if track_id in track_objects and track_objects[track_id] != object_id:
            raise ValueError(f"Physical-instance track crosses objects: {track_id}")
        track_objects[track_id] = object_id
        if object_id not in cad_cache:
            cad_cache[object_id] = load_cad_geometry(dataset_root, object_id)
        cad = cad_cache[object_id]
        target_prediction = prepared["views"][0]["prediction"]
        target_context = extract_target_context(
            group,
            prepared["target_manifest"],
            target_prediction,
            cad,
            dataset_root,
        )
        views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
        for candidate_slot in range(1, EXPECTED_CANDIDATES_PER_TARGET + 1):
            outcome = _pair_outcome(prepared, candidate_slot)
            rows.append(
                {
                    "record_type": "m4_development_candidate",
                    "schema_version": SCHEMA_VERSION,
                    "group_id": group_id,
                    "target_sample_id": str(group["target_sample_id"]),
                    "candidate_sample_id": str(views[candidate_slot]["sample_id"]),
                    "object_id": object_id,
                    "physical_instance_id": track_id,
                    "candidate_slot": candidate_slot,
                    "features": extract_candidate_features(
                        group, candidate_slot, target_context, cad
                    ),
                    **outcome,
                }
            )
        if group_index % 25 == 0 or group_index == len(prepared_groups):
            print(
                f"M4 development features: {group_index}/{len(prepared_groups)}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["group_id"], row["candidate_slot"]))
    if len(rows) != EXPECTED_DEVELOPMENT_ROW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DEVELOPMENT_ROW_COUNT} development rows, "
            f"got {len(rows)}"
        )
    slot_counts = Counter(int(row["candidate_slot"]) for row in rows)
    if slot_counts != Counter({slot: EXPECTED_TARGET_COUNT for slot in range(1, 5)}):
        raise RuntimeError(f"Development candidate-slot counts failed: {slot_counts}")
    if len(set(int(row["object_id"]) for row in rows)) != EXPECTED_OBJECT_COUNT:
        raise RuntimeError("M4 development rows do not cover all 15 objects")
    return rows, cad_cache


def _fit_model(features: np.ndarray, labels: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    model.fit(features, labels)
    return model


def _make_folds(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    from sklearn.model_selection import StratifiedGroupKFold

    objects = np.asarray([int(row["object_id"]) for row in rows], dtype=np.int64)
    groups = np.asarray(
        [str(row["physical_instance_id"]) for row in rows], dtype=object
    )
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_SEED,
    )
    folds = list(
        splitter.split(
            np.zeros((len(rows), 1), dtype=np.float64),
            objects,
            groups,
        )
    )
    seen_test_indices: list[int] = []
    track_to_test_fold: dict[str, int] = {}
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_tracks = set(str(value) for value in groups[train_indices])
        test_tracks = set(str(value) for value in groups[test_indices])
        overlap = sorted(train_tracks & test_tracks)
        if overlap:
            raise RuntimeError(
                f"Physical-instance leakage in fold {fold_index}: {overlap}"
            )
        for track_id in test_tracks:
            if track_id in track_to_test_fold:
                raise RuntimeError(f"Track assigned to two test folds: {track_id}")
            track_to_test_fold[track_id] = fold_index
        seen_test_indices.extend(int(value) for value in test_indices)
        fold_rows.append(
            {
                "fold": fold_index,
                "train_candidate_row_count": int(len(train_indices)),
                "test_candidate_row_count": int(len(test_indices)),
                "train_target_count": len(
                    set(str(rows[index]["group_id"]) for index in train_indices)
                ),
                "test_target_count": len(
                    set(str(rows[index]["group_id"]) for index in test_indices)
                ),
                "train_physical_instance_count": len(train_tracks),
                "test_physical_instance_count": len(test_tracks),
                "physical_instance_intersection": overlap,
                "physical_instance_intersection_count": len(overlap),
                "test_object_candidate_row_counts": dict(
                    sorted(
                        Counter(
                            int(rows[index]["object_id"]) for index in test_indices
                        ).items()
                    )
                ),
            }
        )
    if sorted(seen_test_indices) != list(range(len(rows))):
        raise RuntimeError("Every development row must appear in one OOF fold")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": CV_FOLDS,
        "shuffle": True,
        "random_state": CV_SEED,
        "group_field": "physical_instance_id",
        "stratification_metadata": "object_id, audit only and never a model feature",
        "candidate_row_count": len(rows),
        "target_count": len(set(str(row["group_id"]) for row in rows)),
        "physical_instance_count": len(set(groups)),
        "all_fold_group_intersections_empty": True,
        "each_candidate_row_tested_exactly_once": True,
        "each_row_tested_exactly_once": True,
        "track_to_test_fold": dict(sorted(track_to_test_fold.items())),
        "folds": fold_rows,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return folds, audit


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman_if_defined(
    actual: Sequence[float], predicted: Sequence[float]
) -> float | None:
    actual_ranks = _rankdata(actual)
    predicted_ranks = _rankdata(predicted)
    if np.std(actual_ranks) <= 0.0 or np.std(predicted_ranks) <= 0.0:
        return None
    return float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1])


def _pose_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_object: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[int(row["object_id"])].append(row)
    per_object = []
    for object_id, object_rows in sorted(by_object.items()):
        ar_mssd = float(np.mean([row["sample_ar_mssd"] for row in object_rows]))
        ar_mspd = float(np.mean([row["sample_ar_mspd"] for row in object_rows]))
        per_object.append(
            {
                "object_id": object_id,
                "target_count": len(object_rows),
                "ar_mssd": ar_mssd,
                "ar_mspd": ar_mspd,
                "combined": 0.5 * (ar_mssd + ar_mspd),
                "joint_success_rate": float(
                    np.mean([bool(row["joint_success"]) for row in object_rows])
                ),
            }
        )
    micro_ar_mssd = float(np.mean([row["sample_ar_mssd"] for row in rows]))
    micro_ar_mspd = float(np.mean([row["sample_ar_mspd"] for row in rows]))
    return {
        "micro": {
            "ar_mssd": micro_ar_mssd,
            "ar_mspd": micro_ar_mspd,
            "combined": 0.5 * (micro_ar_mssd + micro_ar_mspd),
            "joint_success_rate": float(
                np.mean([bool(row["joint_success"]) for row in rows])
            ),
        },
        "macro_object": {
            "ar_mssd": float(np.mean([row["ar_mssd"] for row in per_object])),
            "ar_mspd": float(np.mean([row["ar_mspd"] for row in per_object])),
            "combined": float(np.mean([row["combined"] for row in per_object])),
            "joint_success_rate": float(
                np.mean([row["joint_success_rate"] for row in per_object])
            ),
        },
        "by_object": per_object,
    }


def _decorate_oof_and_diagnostics(
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fold_audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if predictions.shape != (len(rows),) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("OOF prediction vector is incomplete or non-finite")
    predictions = np.clip(predictions, 0.0, 1.0)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[str(row["group_id"])].append(index)
    selected_indices: list[int] = []
    regrets: list[float] = []
    oracle_hits: list[bool] = []
    correlations: list[float] = []
    group_diagnostics: dict[str, dict[str, Any]] = {}
    for group_id, indices in sorted(by_group.items()):
        if len(indices) != EXPECTED_CANDIDATES_PER_TARGET:
            raise RuntimeError(f"Target does not have four OOF candidates: {group_id}")
        selected_index = min(
            indices,
            key=lambda index: (
                -float(predictions[index]),
                int(rows[index]["candidate_slot"]),
            ),
        )
        actual_values = [float(rows[index]["actual_utility"]) for index in indices]
        predicted_values = [float(predictions[index]) for index in indices]
        oracle_utility = max(actual_values)
        selected_utility = float(rows[selected_index]["actual_utility"])
        correlation = _spearman_if_defined(actual_values, predicted_values)
        if correlation is not None:
            correlations.append(correlation)
        selected_indices.append(selected_index)
        regrets.append(oracle_utility - selected_utility)
        oracle_hits.append(
            math.isclose(selected_utility, oracle_utility, rel_tol=0.0, abs_tol=1e-12)
        )
        group_diagnostics[group_id] = {
            "selected_index": selected_index,
            "oracle_utility": oracle_utility,
            "rank_correlation": correlation,
        }

    oof_rows = []
    selected_index_set = set(selected_indices)
    for index, row in enumerate(rows):
        group_info = group_diagnostics[str(row["group_id"])]
        oof_rows.append(
            {
                "record_type": "m4_development_oof_prediction",
                "schema_version": SCHEMA_VERSION,
                "group_id": row["group_id"],
                "target_sample_id": row["target_sample_id"],
                "candidate_sample_id": row["candidate_sample_id"],
                "object_id": row["object_id"],
                "physical_instance_id": row["physical_instance_id"],
                "candidate_slot": row["candidate_slot"],
                "fold": int(
                    fold_audit["track_to_test_fold"][row["physical_instance_id"]]
                ),
                "actual_utility": row["actual_utility"],
                "joint_success": row["joint_success"],
                "sample_ar_mssd": row["sample_ar_mssd"],
                "sample_ar_mspd": row["sample_ar_mspd"],
                "predicted_utility": float(predictions[index]),
                "selected_by_learned_voi_oof": index in selected_index_set,
                "oracle_best_actual_utility": group_info["oracle_utility"],
                "is_oracle_best_candidate": math.isclose(
                    float(row["actual_utility"]),
                    float(group_info["oracle_utility"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "target_rank_correlation": group_info["rank_correlation"],
            }
        )

    labels = np.asarray([float(row["actual_utility"]) for row in rows])
    selected_rows = [rows[index] for index in selected_indices]
    absolute_errors = np.abs(predictions - labels)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": (
            "Five-fold physical-instance-grouped out-of-fold development "
            "predictions; candidate selection is within each four-row target."
        ),
        "utility_regression": {
            "candidate_row_count": len(rows),
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        },
        "per_target_rank_correlation": {
            "target_count": len(by_group),
            "defined_count": len(correlations),
            "undefined_count": len(by_group) - len(correlations),
            "mean_where_defined": (
                float(np.mean(correlations)) if correlations else None
            ),
            "median_where_defined": (
                float(np.median(correlations)) if correlations else None
            ),
        },
        "ranking": {
            "mean_regret_to_best_of_four": float(np.mean(regrets)),
            "median_regret_to_best_of_four": float(np.median(regrets)),
            "oracle_best_candidate_hit_rate": float(np.mean(oracle_hits)),
            "oracle_best_candidate_hit_count": int(np.sum(oracle_hits)),
            "target_count": len(by_group),
        },
        "selected_pose_metrics": _pose_aggregate(selected_rows),
    }
    diagnostics.update(
        {
            "utility_mae": diagnostics["utility_regression"]["mae"],
            "utility_rmse": diagnostics["utility_regression"]["rmse"],
            "mean_regret": diagnostics["ranking"]["mean_regret_to_best_of_four"],
            "oracle_best_candidate_hit_rate_tie_aware": diagnostics["ranking"][
                "oracle_best_candidate_hit_rate"
            ],
        }
    )
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return oof_rows, diagnostics


def _best_static_slot(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    slot_scores = []
    for slot in range(1, EXPECTED_CANDIDATES_PER_TARGET + 1):
        slot_rows = [row for row in rows if int(row["candidate_slot"]) == slot]
        if len(slot_rows) != EXPECTED_TARGET_COUNT:
            raise RuntimeError(f"Static candidate slot {slot} is incomplete")
        aggregate = _pose_aggregate(slot_rows)
        slot_scores.append(
            {
                "candidate_slot": slot,
                "target_count": len(slot_rows),
                "macro_object": aggregate["macro_object"],
                "micro": aggregate["micro"],
            }
        )
    selected = min(
        slot_scores,
        key=lambda row: (
            -float(row["macro_object"]["combined"]),
            int(row["candidate_slot"]),
        ),
    )
    return {
        "candidate_slot": int(selected["candidate_slot"]),
        "selection_rule": (
            "Highest M2 development macro-object combined score; ties prefer "
            "the lower existing camera-diversity slot."
        ),
        "development_source": "M2 only",
        "slot_scores": {
            str(row["candidate_slot"]): float(row["macro_object"]["combined"])
            for row in slot_scores
        },
        "slot_details": slot_scores,
    }


def _clean_model_parameters(model: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in model.get_params(deep=False).items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def _dump_joblib_atomic(path: Path, model: Any) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_paths = {
        "m2_groups": args.m2_groups.resolve(),
        "m2_candidate_manifest": args.m2_candidate_manifest.resolve(),
        "m2_groups_summary": args.m2_groups_summary.resolve(),
        "m2_extrinsics_audit": args.m2_extrinsics_audit.resolve(),
        "m2_view_predictions": args.m2_view_predictions.resolve(),
        "m2_metrics": args.m2_metrics.resolve(),
        "m1_manifest": args.m1_manifest.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
    }
    output_paths = {
        "development_features": args.features_out.resolve(),
        "development_oof_predictions": args.oof_out.resolve(),
        "development_fold_audit": args.fold_audit_out.resolve(),
        "development_diagnostics": args.diagnostics_out.resolve(),
        "feature_audit": args.feature_audit_out.resolve(),
        "model": args.model_out.resolve(),
        "frozen": args.frozen_out.resolve(),
    }
    _validate_paths(repo_root, input_paths, output_paths)
    if not args.dataset_root.resolve().is_dir():
        raise FileNotFoundError(f"XYZ-IBD dataset root is missing: {args.dataset_root}")

    prepared_groups, _, preparation_audit = _load_prepared_m2(args, input_paths)
    rows, cad_cache = _build_development_rows(
        prepared_groups, args.dataset_root.resolve()
    )
    features = np.asarray(
        [_ordered_feature_vector(row["features"]) for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray(
        [float(row["actual_utility"]) for row in rows], dtype=np.float64
    )
    folds, fold_audit = _make_folds(rows)
    oof_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    per_fold_model_iterations = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        model = _fit_model(features[train_indices], labels[train_indices])
        oof_predictions[test_indices] = model.predict(features[test_indices])
        per_fold_model_iterations.append(
            {
                "fold": fold_index,
                "n_iter": int(model.n_iter_),
            }
        )
    oof_rows, diagnostics = _decorate_oof_and_diagnostics(
        rows, oof_predictions, fold_audit
    )
    diagnostics["per_fold_model_iterations"] = per_fold_model_iterations
    diagnostics["diagnostics_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in diagnostics.items()
            if key != "diagnostics_sha256"
        }
    )
    final_model = _fit_model(features, labels)
    best_static_slot = _best_static_slot(rows)
    feature_audit = _feature_audit()

    training_payload = [
        {
            "group_id": row["group_id"],
            "physical_instance_id": row["physical_instance_id"],
            "candidate_slot": row["candidate_slot"],
            "features": row["features"],
            "actual_utility": row["actual_utility"],
            "joint_success": row["joint_success"],
            "sample_ar_mssd": row["sample_ar_mssd"],
            "sample_ar_mspd": row["sample_ar_mspd"],
        }
        for row in rows
    ]
    training_sha256 = canonical_sha256(training_payload)

    write_jsonl_atomic(output_paths["development_features"], rows)
    write_jsonl_atomic(output_paths["development_oof_predictions"], oof_rows)
    write_json_atomic(output_paths["development_fold_audit"], fold_audit)
    write_json_atomic(output_paths["development_diagnostics"], diagnostics)
    write_json_atomic(output_paths["feature_audit"], feature_audit)
    _dump_joblib_atomic(output_paths["model"], final_model)

    import sklearn

    input_provenance = {
        label: {
            "path": str(path),
            "artifact": str(path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(path),
        }
        for label, path in sorted(input_paths.items())
    }
    cad_provenance = {
        str(object_id): {
            "model_artifact": str(
                cad["model_path"].relative_to(args.dataset_root.resolve())
            ).replace("\\", "/"),
            "model_sha256": cad["model_sha256"],
        }
        for object_id, cad in sorted(cad_cache.items())
    }
    artifact_provenance = {
        label: {
            "artifact": path.name,
            "sha256": sha256_file(path),
        }
        for label, path in output_paths.items()
        if label != "frozen"
    }
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PoseLoop-VOI one-step pre-acquisition view ranking",
        "freeze_status": "frozen_before_m3_holdout_candidate_outcome_evaluation",
        "holdout_artifacts_read": False,
        "holdout_predictions_metrics_outcomes_read": False,
        "development_source": "M2 targets and M1/M2 predictions only",
        "feature_names": list(FEATURE_NAMES),
        "new_surface_feature_name": NEW_SURFACE_FEATURE_NAME,
        "input_provenance": input_provenance,
        "cad_provenance": {
            "dataset": "xyzibd",
            "models_info_artifact": "models/models_info.json",
            "models_info_sha256": next(iter(cad_cache.values()))["models_info_sha256"],
            "objects": cad_provenance,
        },
        "development_contract": {
            "target_count": EXPECTED_TARGET_COUNT,
            "candidate_rows_per_target": EXPECTED_CANDIDATES_PER_TARGET,
            "candidate_row_count": len(rows),
            "object_count": len(set(int(row["object_id"]) for row in rows)),
            "physical_instance_count": len(
                set(str(row["physical_instance_id"]) for row in rows)
            ),
            "physical_instance_field": "group.oracle_association.track_id",
            "training_data_sha256": training_sha256,
        },
        "training_data": {
            "row_count": len(rows),
            "target_count": EXPECTED_TARGET_COUNT,
            "physical_instance_count": len(
                set(str(row["physical_instance_id"]) for row in rows)
            ),
            "object_count": len(set(int(row["object_id"]) for row in rows)),
            "sha256": sha256_file(output_paths["development_features"]),
        },
        "label_contract": {
            "post_acquisition_pose_selection": (
                "Common two-view max-mask result over target and exactly one "
                "candidate, transformed to the target camera frame."
            ),
            "combined_utility": (
                "Mean of the ten AR_MSSD and ten AR_MSPD threshold indicators, "
                "equivalently 0.5 * (sample_ar_mssd + sample_ar_mspd)."
            ),
            "joint_success": "normalized MSSD <= 0.10 and MSPD <= 10r pixels",
            "features_include_labels_or_outcomes": False,
        },
        "feature_audit": {
            "artifact": output_paths["feature_audit"].name,
            "sha256": sha256_file(output_paths["feature_audit"]),
            "audit_sha256": feature_audit["audit_sha256"],
            "passed": True,
        },
        "cross_validation": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": CV_FOLDS,
            "shuffle": True,
            "random_state": CV_SEED,
            "group_field": "physical_instance_id",
            "all_fold_group_intersections_empty": True,
            "fold_audit_artifact": output_paths["development_fold_audit"].name,
            "fold_audit_sha256": sha256_file(output_paths["development_fold_audit"]),
        },
        "oof_diagnostics": {
            "artifact": output_paths["development_diagnostics"].name,
            "sha256": sha256_file(output_paths["development_diagnostics"]),
            "summary": diagnostics,
        },
        "best_static_slot": best_static_slot,
        "model": {
            "serialization": "joblib",
            "type": "HistGradientBoostingRegressor",
            "artifact": output_paths["model"].name,
            "sha256": sha256_file(output_paths["model"]),
            "sklearn_version": sklearn.__version__,
            "parameters": _clean_model_parameters(final_model),
            "n_iter": int(final_model.n_iter_),
        },
        "artifacts": artifact_provenance,
        "source_contract": {
            "fit_script_sha256": sha256_file(Path(__file__).resolve()),
            "bop_toolkit_commit_sha": preparation_audit["toolkit_commit_sha"],
            "m2_gt_transform_audit": preparation_audit["gt_transform_audit"],
        },
        "holdout_gate": (
            "The evaluator must load and validate this frozen receipt before it "
            "first opens M3, then validate 230 targets, 920 candidate rows, four "
            "candidate slots per target, and zero M2/M3 physical-instance overlap."
        ),
    }
    frozen["configuration_sha256"] = canonical_sha256(frozen)
    frozen["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_paths["frozen"], frozen)

    utility = diagnostics["utility_regression"]
    ranking = diagnostics["ranking"]
    selected = diagnostics["selected_pose_metrics"]
    print(
        "M4 development freeze complete: "
        f"targets={EXPECTED_TARGET_COUNT}, candidate_rows={len(rows)}, "
        f"tracks={frozen['development_contract']['physical_instance_count']}",
        flush=True,
    )
    print(
        f"OOF utility: MAE={utility['mae']:.6f}, RMSE={utility['rmse']:.6f}, "
        f"regret={ranking['mean_regret_to_best_of_four']:.6f}, "
        f"oracle_hit={ranking['oracle_best_candidate_hit_rate']:.6f}",
        flush=True,
    )
    print(
        "OOF selected pose: "
        f"macro={selected['macro_object']['combined']:.6f}, "
        f"micro={selected['micro']['combined']:.6f}",
        flush=True,
    )
    print(
        f"best_static_slot={best_static_slot['candidate_slot']}, "
        f"model_sha256={frozen['model']['sha256']}, "
        f"configuration_sha256={frozen['configuration_sha256']}",
        flush=True,
    )
    print(f"frozen model: {output_paths['frozen']}", flush=True)


if __name__ == "__main__":
    main()
