"""Small, dependency-light SE(3) helpers for evaluator-only diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import ContractError


def as_pose(value: Any, *, label: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ContractError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ContractError(f"{label} has an invalid homogeneous last row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ContractError(f"{label} rotation is not a legal SO(3) matrix")
    return pose


def axis_angle_rotation(axis: str, degrees: float) -> np.ndarray:
    if axis not in {"x", "y", "z"}:
        raise ContractError(f"Unsupported rotation axis: {axis}")
    radians = math.radians(float(degrees))
    sine, cosine = math.sin(radians), math.cos(radians)
    if axis == "x":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "y":
        return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def perturb_pose(
    gt_pose: np.ndarray,
    *,
    rotation_axis: str,
    rotation_degrees: float,
    translation_axis: str,
    translation_mm: float,
) -> np.ndarray:
    gt = as_pose(gt_pose, label="gt_pose")
    output = gt.copy()
    output[:3, :3] = axis_angle_rotation(rotation_axis, rotation_degrees) @ gt[:3, :3]
    translation = np.zeros(3, dtype=np.float64)
    translation[{"x": 0, "y": 1, "z": 2}[translation_axis]] = (
        float(translation_mm) / 1000.0
    )
    output[:3, 3] = gt[:3, 3] + translation
    return as_pose(output, label="perturbed_pose")


def interpolate_perturbation(
    gt_pose: np.ndarray, metadata: Mapping[str, Any], scale: float
) -> np.ndarray:
    if not 0.0 <= scale <= 1.0:
        raise ContractError("Refiner interpolation scale must be in [0, 1]")
    return perturb_pose(
        gt_pose,
        rotation_axis=str(metadata["rotation_axis"]),
        rotation_degrees=float(metadata["rotation_degrees"]) * scale,
        translation_axis=str(metadata["translation_axis"]),
        translation_mm=float(metadata["translation_mm"]) * scale,
    )


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ContractError("CAD points must be a finite Nx3 array")
    matrix = as_pose(pose)
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


def add_s_error_mm(
    points: np.ndarray, prediction: np.ndarray, gt_pose: np.ndarray, *, symmetric: bool
) -> float:
    predicted = transform_points(points, prediction)
    ground_truth = transform_points(points, gt_pose)
    if symmetric:
        distances = np.linalg.norm(
            predicted[:, None, :] - ground_truth[None, :, :], axis=2
        )
        error_m = float(np.mean(np.min(distances, axis=1)))
    else:
        error_m = float(np.mean(np.linalg.norm(predicted - ground_truth, axis=1)))
    return error_m * 1000.0


def rotation_error_degrees(prediction: np.ndarray, gt_pose: np.ndarray) -> float:
    predicted = as_pose(prediction)
    gt = as_pose(gt_pose)
    relative = predicted[:3, :3] @ gt[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def translation_error_mm(prediction: np.ndarray, gt_pose: np.ndarray) -> float:
    predicted = as_pose(prediction)
    gt = as_pose(gt_pose)
    return float(np.linalg.norm(predicted[:3, 3] - gt[:3, 3]) * 1000.0)


def pose_errors(
    points: np.ndarray,
    prediction: np.ndarray,
    gt_pose: np.ndarray,
    *,
    symmetric: bool,
) -> dict[str, float]:
    return {
        "add_s_mm": add_s_error_mm(points, prediction, gt_pose, symmetric=symmetric),
        "rotation_error_degrees": rotation_error_degrees(prediction, gt_pose),
        "translation_error_mm": translation_error_mm(prediction, gt_pose),
    }


def load_pose_asset(data_root: Path, asset: Mapping[str, Any]) -> np.ndarray:
    value = json.loads((data_root / str(asset["path"])).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("model_to_camera_pose_m")
    return as_pose(value, label=str(asset["path"]))


def load_cad_asset(
    data_root: Path, asset: Mapping[str, Any]
) -> tuple[np.ndarray, float]:
    value = json.loads((data_root / str(asset["path"])).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("CAD fixture must be a JSON object")
    points = np.asarray(value.get("vertices_m"), dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or not np.all(np.isfinite(points))
    ):
        raise ContractError("CAD fixture requires at least four finite vertices_m")
    diameter = value.get("diameter_m")
    if (
        not isinstance(diameter, (int, float))
        or not math.isfinite(diameter)
        or diameter <= 0
    ):
        raise ContractError("CAD fixture requires a positive diameter_m")
    return points, float(diameter)
