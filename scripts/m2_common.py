#!/usr/bin/env python3
"""Shared calibrated-view helpers for the PoseLoop M2 diagnostic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from m1_common import assert_pose


M2_SCHEMA_VERSION = 1
M2_MAX_ADDITIONAL_VIEWS = 4
M2_MIN_VISIBLE_FRACTION = 0.10
ASSOCIATION_MAX_DISTANCE_MM = 0.05
ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM = 5.0
GT_AUDIT_MAX_MSSD_MM = 0.01
GT_AUDIT_MAX_MSPD_PX = 0.01
GT_AUDIT_MAX_NORMALIZED_MSSD = 5e-4

CAMERA_ROTATION_KEYS = ("cam_R_w2c", "R_w2c")
CAMERA_TRANSLATION_KEYS = ("cam_t_w2c", "t_w2c")


def resolve_camera_field(
    entry: dict[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, list[float]]:
    """Resolve a canonical BOP field or the XYZ-IBD field alias."""
    present = [key for key in keys if key in entry]
    if not present:
        raise KeyError(f"Missing {label}; tried {keys}")
    reference = np.asarray(entry[present[0]], dtype=np.float64)
    for key in present[1:]:
        value = np.asarray(entry[key], dtype=np.float64)
        if reference.shape != value.shape or not np.allclose(
            reference,
            value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Conflicting aliases for {label}: {present}")
    return present[0], reference.reshape(-1).tolist()


def camera_world_to_camera_pose_m(
    entry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, str]]:
    """Return ^camera T_world with the BOP millimetre translation in metres."""
    rotation_key, rotation_values = resolve_camera_field(
        entry,
        CAMERA_ROTATION_KEYS,
        label="world-to-camera rotation",
    )
    translation_key, translation_values = resolve_camera_field(
        entry,
        CAMERA_TRANSLATION_KEYS,
        label="world-to-camera translation",
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_values, dtype=np.float64) * 0.001
    assert_pose(transform, "official world-to-camera pose", rotation_atol=5e-3)
    return transform, {
        "rotation_key": rotation_key,
        "translation_key": translation_key,
    }


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    """Apply the literal homogeneous inverse required by the BOP convention."""
    transform = np.asarray(transform, dtype=np.float64)
    assert_pose(transform, "transform to invert", rotation_atol=5e-3)
    inverse = np.linalg.inv(transform)
    assert_pose(inverse, "inverted transform", rotation_atol=5e-3)
    return inverse


def model_to_world_pose_m(
    camera_world_to_camera_m: np.ndarray,
    model_to_camera_m: np.ndarray,
) -> np.ndarray:
    """Apply T_w_m = inverse(T_c_w) @ T_c_m."""
    pose = rigid_inverse(camera_world_to_camera_m) @ np.asarray(
        model_to_camera_m,
        dtype=np.float64,
    )
    assert_pose(pose, "model-to-world pose", rotation_atol=1e-2)
    return pose


def transform_model_pose_to_target(
    target_world_to_camera_m: np.ndarray,
    source_world_to_camera_m: np.ndarray,
    source_model_to_camera_m: np.ndarray,
) -> np.ndarray:
    """Express a source-camera model pose in the target camera frame."""
    pose = (
        np.asarray(target_world_to_camera_m, dtype=np.float64)
        @ rigid_inverse(source_world_to_camera_m)
        @ np.asarray(source_model_to_camera_m, dtype=np.float64)
    )
    assert_pose(pose, "target-camera model pose", rotation_atol=1e-2)
    return pose


def camera_center_world_m(camera_world_to_camera_m: np.ndarray) -> np.ndarray:
    """Return the camera origin expressed in the shared world frame."""
    return rigid_inverse(camera_world_to_camera_m)[:3, 3]


def unit_viewing_direction_world(
    camera_world_to_camera_m: np.ndarray,
    object_center_world_m: np.ndarray,
) -> np.ndarray:
    """Return the object-to-camera unit direction in the world frame."""
    direction = camera_center_world_m(camera_world_to_camera_m) - np.asarray(
        object_center_world_m,
        dtype=np.float64,
    ).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Camera centre and object centre are coincident")
    return direction / norm


def angular_distance_radians(first: np.ndarray, second: np.ndarray) -> float:
    """Stable angular separation between two unit vectors."""
    dot = float(
        np.dot(
            np.asarray(first, dtype=np.float64).reshape(3),
            np.asarray(second, dtype=np.float64).reshape(3),
        )
    )
    return float(math.acos(float(np.clip(dot, -1.0, 1.0))))


def greedy_camera_center_diversity(
    target_center_world_m: np.ndarray,
    candidates: Iterable[dict[str, Any]],
    *,
    count: int = M2_MAX_ADDITIONAL_VIEWS,
) -> list[dict[str, Any]]:
    """Greedy farthest-point acquisition using official camera centres only."""
    remaining = [dict(candidate) for candidate in candidates]
    selected: list[dict[str, Any]] = []
    selected_centers = [
        np.asarray(target_center_world_m, dtype=np.float64).reshape(3)
    ]
    while remaining and len(selected) < count:
        scored: list[tuple[float, int, int, str, dict[str, Any]]] = []
        for candidate in remaining:
            center = np.asarray(
                candidate["camera_center_world_m"],
                dtype=np.float64,
            ).reshape(3)
            minimum_distance = min(
                float(np.linalg.norm(center - acquired))
                for acquired in selected_centers
            )
            scored.append(
                (
                    minimum_distance,
                    int(candidate["image_id"]),
                    int(candidate["gt_instance_index"]),
                    str(candidate["sample_id"]),
                    candidate,
                )
            )
        best_score = max(item[0] for item in scored)
        ties = [
            item
            for item in scored
            if math.isclose(item[0], best_score, rel_tol=0.0, abs_tol=1e-12)
        ]
        chosen = min(ties, key=lambda item: (item[1], item[2], item[3]))
        candidate = chosen[4]
        candidate["diversity_score_camera_center_distance_m"] = float(chosen[0])
        selected.append(candidate)
        selected_centers.append(
            np.asarray(
                candidate["camera_center_world_m"],
                dtype=np.float64,
            ).reshape(3)
        )
        remaining = [
            item
            for item in remaining
            if item["sample_id"] != candidate["sample_id"]
        ]
    return selected


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Compact finite distribution summary suitable for audit JSON."""
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def ensure_raw_artifact_path(path: Path, repo_root: Path) -> Path:
    """Keep M2 raw outputs under the ignored artifact subtree."""
    resolved = path.resolve()
    artifact_root = (repo_root / "artifacts" / "m2").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(f"Raw M2 artifact must stay under {artifact_root}: {resolved}")
    return resolved
