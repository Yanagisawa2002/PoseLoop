#!/usr/bin/env python3
"""Deterministic metrics for the PoseLoop M5-G0 synthetic mechanism gate.

This module deliberately keeps evaluator-only metadata out of estimator code.  Its
public entry point, :func:`compute_trajectory_metrics`, consumes completed arrays
and half-open corruption intervals.  The normalized pose error used throughout is

    sqrt((translation_metres / 0.01) ** 2 + (rotation_degrees / 5.0) ** 2).

The MSSD helper is a diagnostic on synthetic model points.  It is BOP-compatible
when the vendored implementation can be imported, but it is not an official BOP
benchmark result (and the exact local fallback carries the same restriction).
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _load_core() -> Any | None:
    """Import the sibling core in both script and package execution modes."""

    package = __package__
    if package:
        try:
            return importlib.import_module(f"{package}.m5_g0_core")
        except ModuleNotFoundError as exc:
            if exc.name != f"{package}.m5_g0_core":
                raise
    try:
        return importlib.import_module("m5_g0_core")
    except ModuleNotFoundError as exc:
        if exc.name != "m5_g0_core":
            raise
        return None


_CORE = _load_core()

TRANSLATION_NORMALIZER_M = 0.01
ROTATION_NORMALIZER_DEG = 5.0
NOMINAL_NORMALIZED_ERROR_THRESHOLD = 1.0
CATASTROPHIC_RAW_STEP_DEG = 90.0
CATASTROPHIC_QUOTIENT_STEP_DEG = 10.0
AXIAL_GAUGE_SPIN_DEG = 45.0
MONOTONIC_TOLERANCE = 1e-12
_EPS = 1e-12


@dataclass(frozen=True)
class FrameInterval:
    """A validated half-open frame interval ``[start, end)``."""

    start: int
    end: int
    label: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


def normalized_pose_error(
    translation_error_m: float | np.ndarray,
    rotation_error_deg: float | np.ndarray,
) -> float | np.ndarray:
    """Return the frozen dimensionless M5-G0 pose error."""

    translation = np.asarray(translation_error_m, dtype=np.float64)
    rotation = np.asarray(rotation_error_deg, dtype=np.float64)
    value = np.sqrt(
        np.square(translation / TRANSLATION_NORMALIZER_M)
        + np.square(rotation / ROTATION_NORMALIZER_DEG)
    )
    if value.ndim == 0:
        return float(value)
    return value


def _pose_series(
    values: Sequence[Any] | np.ndarray,
    label: str,
    *,
    expected_length: int | None = None,
    allow_missing: bool = False,
) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype != object:
        result = np.asarray(values, dtype=np.float64)
        if result.shape == (4, 4):
            result = result[None, :, :]
        if result.ndim != 3 or result.shape[1:] != (4, 4):
            raise ValueError(f"{label} must have shape (N, 4, 4)")
    else:
        rows = list(values)
        result = np.full((len(rows), 4, 4), np.nan, dtype=np.float64)
        for index, value in enumerate(rows):
            if value is None and allow_missing:
                continue
            pose = np.asarray(value, dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"{label}[{index}] must have shape (4, 4)")
            result[index] = pose
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(
            f"{label} length {len(result)} does not match expected {expected_length}"
        )
    return result


def _timestamps(
    values: Sequence[float] | np.ndarray, expected_length: int
) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(timestamps) != expected_length:
        raise ValueError(
            f"timestamps length {len(timestamps)} does not match {expected_length} poses"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return timestamps


def _finite_pose(pose: np.ndarray) -> bool:
    return pose.shape == (4, 4) and bool(np.isfinite(pose).all())


def _rotation_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(_rotation_angle_rad(first, second))


def _enum_text(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    aliases = {
        "NONE": "ASYMMETRIC",
        "IDENTITY": "ASYMMETRIC",
        "ASYM": "ASYMMETRIC",
        "SO2": "CONTINUOUS_AXIAL",
        "SO(2)": "CONTINUOUS_AXIAL",
        "CONTINUOUS": "CONTINUOUS_AXIAL",
        "AXIAL": "CONTINUOUS_AXIAL",
    }
    return aliases.get(text, text)


def symmetry_class_name(symmetry: Any) -> str:
    """Return a stable class name from a core ``SymmetrySpec`` or a string."""

    if symmetry is None:
        return "ASYMMETRIC"
    if isinstance(symmetry, str) or hasattr(symmetry, "value"):
        text = _enum_text(symmetry)
    elif isinstance(symmetry, Mapping):
        value = next(
            (
                symmetry[key]
                for key in ("symmetry_class", "kind", "name", "class_name")
                if key in symmetry
            ),
            "ASYMMETRIC",
        )
        text = _enum_text(value)
    else:
        value = next(
            (
                getattr(symmetry, key)
                for key in ("symmetry_class", "kind", "name", "class_name")
                if hasattr(symmetry, key)
            ),
            symmetry.__class__.__name__,
        )
        text = _enum_text(value)
    if text not in {"ASYMMETRIC", "C2", "C4", "CONTINUOUS_AXIAL"}:
        raise ValueError(f"Unsupported M5-G0 symmetry class: {text}")
    return text


def _axis_object(symmetry: Any) -> np.ndarray:
    value: Any = None
    keys = ("axis_object", "continuous_axis", "axis", "symmetry_axis")
    if isinstance(symmetry, Mapping):
        value = next((symmetry[key] for key in keys if key in symmetry), None)
    elif symmetry is not None:
        value = next(
            (getattr(symmetry, key) for key in keys if hasattr(symmetry, key)), None
        )
    if value is None:
        value = (0.0, 0.0, 1.0)
    axis = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or norm <= _EPS:
        raise ValueError("continuous symmetry axis must be a finite non-zero vector")
    return axis / norm


def _z_rotation_transform(angle_rad: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    transform[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return transform


def _as_transform(value: Any) -> np.ndarray:
    if isinstance(value, Mapping) and "R" in value:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(value["R"], dtype=np.float64).reshape(3, 3)
        if "t" in value:
            transform[:3, 3] = np.asarray(value["t"], dtype=np.float64).reshape(3)
    else:
        transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("finite symmetry transforms must be finite 4x4 matrices")
    return transform


def finite_symmetry_transforms(symmetry: Any) -> tuple[np.ndarray, ...]:
    """Extract explicit right-acting object-frame transforms from a spec."""

    name = symmetry_class_name(symmetry)
    if name == "CONTINUOUS_AXIAL":
        raise ValueError("continuous axial symmetry has no finite transform list")
    raw: Any = None
    keys = (
        "transforms",
        "finite_transforms",
        "object_transforms",
        "symmetry_transforms",
    )
    if isinstance(symmetry, Mapping):
        raw = next((symmetry[key] for key in keys if key in symmetry), None)
    elif symmetry is not None:
        raw = next(
            (getattr(symmetry, key) for key in keys if hasattr(symmetry, key)), None
        )
    if callable(raw):
        raw = raw()
    if raw is not None:
        transforms = tuple(_as_transform(value) for value in raw)
        if not transforms:
            raise ValueError("finite symmetry transform list cannot be empty")
        return transforms
    count = {"ASYMMETRIC": 1, "C2": 2, "C4": 4}[name]
    return tuple(
        _z_rotation_transform(2.0 * math.pi * index / count) for index in range(count)
    )


def quotient_pose_error(
    output_pose: np.ndarray,
    gt_pose: np.ndarray,
    symmetry: Any,
) -> tuple[float, float]:
    """Return translation metres and symmetry-aware rotation degrees.

    Finite equivalence follows the frozen right-action convention ``T_gt @ S``.
    For continuous axial symmetry, the observable rotational error is the angle
    between the two transformed object-frame symmetry axes.
    """

    output = np.asarray(output_pose, dtype=np.float64)
    gt = np.asarray(gt_pose, dtype=np.float64)
    if output.shape != (4, 4) or gt.shape != (4, 4):
        raise ValueError("quotient_pose_error expects two 4x4 poses")
    if not _finite_pose(output) or not _finite_pose(gt):
        return math.nan, math.nan
    name = symmetry_class_name(symmetry)
    if name == "CONTINUOUS_AXIAL":
        translation = float(np.linalg.norm(output[:3, 3] - gt[:3, 3]))
        axis = _axis_object(symmetry)
        first = output[:3, :3] @ axis
        second = gt[:3, :3] @ axis
        cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
        return translation, math.degrees(math.acos(cosine))

    candidates: list[tuple[float, float, float, int]] = []
    for index, transform in enumerate(finite_symmetry_transforms(symmetry)):
        equivalent = gt @ transform
        translation = float(np.linalg.norm(output[:3, 3] - equivalent[:3, 3]))
        rotation = _rotation_angle_deg(output[:3, :3], equivalent[:3, :3])
        score = float(normalized_pose_error(translation, rotation))
        candidates.append((score, translation, rotation, index))
    _, translation, rotation, _ = min(candidates)
    return translation, rotation


def quotient_rotation_error_degrees(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
    symmetry: Any,
) -> float:
    """Return the declared-symmetry geodesic between two rotations."""

    first = np.asarray(first_rotation, dtype=np.float64).reshape(3, 3)
    second = np.asarray(second_rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return math.nan
    if symmetry_class_name(symmetry) == "CONTINUOUS_AXIAL":
        axis = _axis_object(symmetry)
        cosine = float(np.clip(np.dot(first @ axis, second @ axis), -1.0, 1.0))
        return math.degrees(math.acos(cosine))
    return min(
        _rotation_angle_deg(first, second @ transform[:3, :3])
        for transform in finite_symmetry_transforms(symmetry)
    )


def _finite_summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(len(array)),
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(array) - len(finite)),
    }
    if len(finite) == 0:
        result.update(
            {"mean": None, "rmse": None, "median": None, "p95": None, "max": None}
        )
        return result
    result.update(
        {
            "mean": float(np.mean(finite)),
            "rmse": float(np.sqrt(np.mean(np.square(finite)))),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }
    )
    return result


def compute_per_frame_pose_errors(
    gt_poses: Sequence[Any] | np.ndarray,
    output_poses: Sequence[Any] | np.ndarray,
    symmetry: Any,
) -> dict[str, Any]:
    """Compute exact per-frame quotient errors and trajectory summaries."""

    gt = _pose_series(gt_poses, "gt_poses")
    output = _pose_series(
        output_poses, "output_poses", expected_length=len(gt), allow_missing=True
    )
    valid = np.isfinite(output).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    translation = np.full(len(gt), np.nan, dtype=np.float64)
    rotation = np.full(len(gt), np.nan, dtype=np.float64)
    translation[valid] = np.linalg.norm(output[valid, :3, 3] - gt[valid, :3, 3], axis=1)
    name = symmetry_class_name(symmetry)
    if name == "CONTINUOUS_AXIAL":
        axis = _axis_object(symmetry)
        output_axes = output[valid, :3, :3] @ axis
        gt_axes = gt[valid, :3, :3] @ axis
        cosine = np.clip(np.sum(output_axes * gt_axes, axis=1), -1.0, 1.0)
        rotation[valid] = np.degrees(np.arccos(cosine))
    else:
        output_rotations = output[valid, :3, :3]
        gt_rotations = gt[valid, :3, :3]
        candidates = []
        for transform in finite_symmetry_transforms(symmetry):
            equivalent = gt_rotations @ transform[:3, :3]
            relative = np.swapaxes(output_rotations, 1, 2) @ equivalent
            cosine = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
            candidates.append(np.degrees(np.arccos(cosine)))
        rotation[valid] = np.min(np.stack(candidates, axis=1), axis=1)
    normalized = np.asarray(
        normalized_pose_error(translation, rotation), dtype=np.float64
    )
    return {
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "normalized_pose_error": normalized,
        "translation": _finite_summary(translation),
        "rotation": _finite_summary(rotation),
        "normalized_pose": _finite_summary(normalized),
        "quotient_pose_rmse": _finite_summary(normalized)["rmse"],
        "median_quotient_pose_error": _finite_summary(normalized)["median"],
        "p95_quotient_pose_error": _finite_summary(normalized)["p95"],
    }


per_frame_pose_errors = compute_per_frame_pose_errors


def _rotation_between_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine > 1e-10:
        axis = cross / sine
        return Rotation.from_rotvec(axis * math.atan2(sine, cosine)).as_matrix()
    if cosine > 0.0:
        return np.eye(3, dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(first)))]
    axis = np.cross(first, basis)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * math.pi).as_matrix()


def continuous_axial_spin_degrees(
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    symmetry: Any = "CONTINUOUS_AXIAL",
) -> float:
    """Extract signed swing-twist axial gauge spin between two outputs."""

    previous = np.asarray(previous_rotation, dtype=np.float64).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        return math.nan
    axis_object = _axis_object(symmetry)
    old_axis = previous @ axis_object
    new_axis = current @ axis_object
    swing = _rotation_between_vectors(old_axis, new_axis)
    residual = (swing @ previous).T @ current
    quaternion = Rotation.from_matrix(residual).as_quat()
    projected = float(np.dot(quaternion[:3], axis_object))
    angle = 2.0 * math.atan2(projected, float(quaternion[3]))
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(angle)


def compute_representation_stability(
    timestamps: Sequence[float] | np.ndarray,
    output_poses: Sequence[Any] | np.ndarray,
    symmetry: Any,
) -> dict[str, Any]:
    """Compute exact finite jumps or continuous axial gauge jumps."""

    output = _pose_series(output_poses, "output_poses", allow_missing=True)
    times = _timestamps(timestamps, len(output))
    count = max(0, len(output) - 1)
    raw_steps = np.full(count, np.nan, dtype=np.float64)
    quotient_steps = np.full(count, np.nan, dtype=np.float64)
    axial_spin = np.full(count, np.nan, dtype=np.float64)
    axial_velocity = np.full(count, np.nan, dtype=np.float64)
    name = symmetry_class_name(symmetry)
    previous_rotations = output[:-1, :3, :3]
    current_rotations = output[1:, :3, :3]
    valid = np.isfinite(previous_rotations).all(axis=(1, 2)) & np.isfinite(
        current_rotations
    ).all(axis=(1, 2))
    relative = np.swapaxes(previous_rotations[valid], 1, 2) @ current_rotations[valid]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    raw_steps[valid] = np.degrees(np.arccos(cosine))
    if name == "CONTINUOUS_AXIAL":
        axis = _axis_object(symmetry)
        old_axes = previous_rotations[valid] @ axis
        new_axes = current_rotations[valid] @ axis
        axis_cosine = np.clip(np.sum(old_axes * new_axes, axis=1), -1.0, 1.0)
        quotient_steps[valid] = np.degrees(np.arccos(axis_cosine))
        valid_indices = np.flatnonzero(valid)
        for compact_index, original_index in enumerate(valid_indices):
            axial_spin[original_index] = continuous_axial_spin_degrees(
                previous_rotations[original_index],
                current_rotations[original_index],
                symmetry,
            )
            axial_velocity[original_index] = math.radians(
                axial_spin[original_index]
            ) / (times[original_index + 1] - times[original_index])
    else:
        candidate_steps = []
        for transform in finite_symmetry_transforms(symmetry):
            aligned = current_rotations[valid] @ transform[:3, :3]
            candidate_relative = np.swapaxes(previous_rotations[valid], 1, 2) @ aligned
            candidate_cosine = np.clip(
                (np.trace(candidate_relative, axis1=1, axis2=2) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
            candidate_steps.append(np.degrees(np.arccos(candidate_cosine)))
        quotient_steps[valid] = np.min(np.stack(candidate_steps, axis=1), axis=1)
    finite_jump = (
        np.isfinite(raw_steps)
        & np.isfinite(quotient_steps)
        & (raw_steps > CATASTROPHIC_RAW_STEP_DEG)
        & (quotient_steps < CATASTROPHIC_QUOTIENT_STEP_DEG)
    )
    gauge_jump = (
        np.isfinite(axial_spin)
        & np.isfinite(quotient_steps)
        & (np.abs(axial_spin) > AXIAL_GAUGE_SPIN_DEG)
        & (quotient_steps < CATASTROPHIC_QUOTIENT_STEP_DEG)
    )
    velocity_summary = _finite_summary(axial_velocity)
    return {
        "raw_rotation_step_deg": raw_steps,
        "quotient_rotation_step_deg": quotient_steps,
        "axial_spin_deg": axial_spin,
        "axial_angular_velocity_rad_s": axial_velocity,
        "catastrophic_jump_mask": finite_jump,
        "axial_gauge_jump_mask": gauge_jump,
        "catastrophic_representation_jump_count": int(np.count_nonzero(finite_jump)),
        "axial_gauge_jump_count": int(np.count_nonzero(gauge_jump)),
        "p95_raw_rotation_step_deg": _finite_summary(raw_steps)["p95"],
        "p95_quotient_rotation_step_deg": _finite_summary(quotient_steps)["p95"],
        "unobservable_axial_angular_velocity_rms_rad_s": velocity_summary["rmse"],
        "unobservable_axial_angular_velocity_rms_deg_s": (
            None
            if velocity_summary["rmse"] is None
            else math.degrees(velocity_summary["rmse"])
        ),
    }


representation_stability_metrics = compute_representation_stability


def _mask_intervals(mask: np.ndarray, label: str | None = None) -> list[FrameInterval]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        FrameInterval(int(start), int(end), label) for start, end in zip(starts, ends)
    ]


def _one_interval(value: Any, frame_count: int, label: str | None) -> FrameInterval:
    if isinstance(value, FrameInterval):
        interval = value
    elif isinstance(value, Mapping):
        start = next(
            (value[key] for key in ("start", "start_frame", "begin") if key in value),
            None,
        )
        if start is None:
            raise ValueError(f"interval has no start: {value}")
        if "end_exclusive" in value:
            end = value["end_exclusive"]
        elif "end_frame_exclusive" in value:
            end = value["end_frame_exclusive"]
        elif "end_inclusive" in value:
            end = int(value["end_inclusive"]) + 1
        elif "end_frame_inclusive" in value:
            end = int(value["end_frame_inclusive"]) + 1
        else:
            end = next(
                (value[key] for key in ("end", "end_frame", "stop") if key in value),
                None,
            )
        if end is None:
            raise ValueError(f"interval has no end: {value}")
        interval = FrameInterval(
            int(start),
            int(end),
            str(value.get("label", label))
            if value.get("label", label) is not None
            else None,
        )
    else:
        pair = list(value)
        if len(pair) != 2:
            raise ValueError(f"interval must contain start and exclusive end: {value}")
        interval = FrameInterval(int(pair[0]), int(pair[1]), label)
    if not 0 <= interval.start < interval.end <= frame_count:
        raise ValueError(
            f"invalid half-open interval [{interval.start}, {interval.end}) for {frame_count} frames"
        )
    return interval


def coerce_intervals(
    values: Any,
    frame_count: int,
    *,
    label: str | None = None,
) -> list[FrameInterval]:
    """Normalize masks, mappings, or pairs to disjoint half-open intervals."""

    if values is None:
        return []
    if isinstance(values, np.ndarray) and values.dtype == bool:
        if values.size != frame_count:
            raise ValueError("boolean interval mask has the wrong frame count")
        intervals = _mask_intervals(values, label)
    elif isinstance(values, Mapping) or isinstance(values, FrameInterval):
        intervals = [_one_interval(values, frame_count, label)]
    else:
        sequence = list(values)
        if len(sequence) == frame_count and all(
            isinstance(value, (bool, np.bool_)) for value in sequence
        ):
            intervals = _mask_intervals(np.asarray(sequence, dtype=bool), label)
        elif len(sequence) == 2 and all(
            isinstance(value, (int, np.integer)) for value in sequence
        ):
            intervals = [_one_interval(sequence, frame_count, label)]
        else:
            intervals = [_one_interval(value, frame_count, label) for value in sequence]
    intervals.sort(
        key=lambda interval: (interval.start, interval.end, interval.label or "")
    )
    for previous, current in zip(intervals, intervals[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"overlapping evaluator intervals: {previous} and {current}"
            )
    return intervals


def _metadata_intervals(
    metadata: Any, kind: str, frame_count: int
) -> list[FrameInterval]:
    if metadata is None:
        return []
    aliases = {
        "dropout": ("dropout_intervals", "dropouts", "dropout"),
        "outlier": ("outlier_intervals", "outliers", "outlier_bursts", "outlier"),
    }[kind]
    if isinstance(metadata, Mapping):
        for key in aliases:
            if key in metadata:
                return coerce_intervals(metadata[key], frame_count, label=kind)
        return []
    records = list(metadata)
    if len(records) != frame_count:
        raise ValueError("per-frame evaluator metadata has the wrong length")
    mask = np.zeros(frame_count, dtype=bool)
    flag_aliases = {
        "dropout": ("is_dropout", "dropout", "measurement_missing", "missing"),
        "outlier": ("is_outlier", "outlier", "outlier_frame"),
    }[kind]
    for index, record in enumerate(records):
        if isinstance(record, Mapping):
            mask[index] = any(bool(record.get(key, False)) for key in flag_aliases)
        else:
            mask[index] = any(bool(getattr(record, key, False)) for key in flag_aliases)
    return coerce_intervals(mask, frame_count, label=kind)


def _boolean_series(values: Any, frame_count: int, label: str) -> np.ndarray:
    if values is None:
        return np.zeros(frame_count, dtype=bool)
    result = np.asarray(values, dtype=bool).reshape(-1)
    if len(result) != frame_count:
        raise ValueError(f"{label} length does not match frame count")
    return result


def _uncertainty_scalar(values: Any, frame_count: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] != frame_count:
        raise ValueError("uncertainty length does not match frame count")
    if array.ndim == 1:
        return array
    if array.ndim == 3 and array.shape[1] == array.shape[2]:
        return np.trace(array, axis1=1, axis2=2)
    if array.ndim == 2:
        return np.mean(array, axis=1)
    raise ValueError(
        "uncertainty must be scalar, a diagonal vector, or covariance per frame"
    )


def _first_finite_at_or_after(values: np.ndarray, start: int) -> int | None:
    indices = np.flatnonzero(np.isfinite(values[start:]))
    return None if len(indices) == 0 else int(start + indices[0])


def _first_within(values: np.ndarray, start: int, threshold: float) -> int | None:
    mask = np.isfinite(values[start:]) & (values[start:] <= threshold)
    indices = np.flatnonzero(mask)
    return None if len(indices) == 0 else int(start + indices[0])


def compute_dropout_metrics(
    output_pose_error: Sequence[float] | np.ndarray,
    measurement_pose_error: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    uncertainty: Any,
    intervals: Any,
    *,
    nominal_error_threshold: float = NOMINAL_NORMALIZED_ERROR_THRESHOLD,
    monotonic_tolerance: float = MONOTONIC_TOLERANCE,
) -> dict[str, Any]:
    """Summarize deterministic dropout intervals (exclusive-end convention)."""

    output_error = np.asarray(output_pose_error, dtype=np.float64).reshape(-1)
    measurement_error = np.asarray(measurement_pose_error, dtype=np.float64).reshape(-1)
    if len(measurement_error) != len(output_error):
        raise ValueError("measurement and output pose-error lengths differ")
    accepted_array = _boolean_series(accepted, len(output_error), "accepted")
    uncertainty_scalar = _uncertainty_scalar(uncertainty, len(output_error))
    dropout_intervals = coerce_intervals(intervals, len(output_error), label="dropout")
    rows: list[dict[str, Any]] = []
    for interval in dropout_intervals:
        return_index = _first_finite_at_or_after(measurement_error, interval.end)
        accepted_indices = np.flatnonzero(accepted_array[interval.end :])
        first_accepted = (
            None
            if len(accepted_indices) == 0
            else int(interval.end + accepted_indices[0])
        )
        recovery = _first_within(output_error, interval.end, nominal_error_threshold)
        stable_length = 0
        for value in output_error[interval.start : interval.end]:
            if not np.isfinite(value) or value > nominal_error_threshold:
                break
            stable_length += 1
        row: dict[str, Any] = {
            "start_frame": interval.start,
            "end_frame_exclusive": interval.end,
            "length_frames": interval.length,
            "final_missing_frame": interval.end - 1,
            "final_missing_pose_error": (
                float(output_error[interval.end - 1])
                if np.isfinite(output_error[interval.end - 1])
                else None
            ),
            "first_return_frame": return_index,
            "first_return_measurement_error": (
                None if return_index is None else float(measurement_error[return_index])
            ),
            "first_accepted_post_dropout_frame": first_accepted,
            "first_accepted_post_dropout_output_error": (
                None
                if first_accepted is None
                or not np.isfinite(output_error[first_accepted])
                else float(output_error[first_accepted])
            ),
            "recovery_frame": recovery,
            "frames_to_return_within_nominal": (
                None if recovery is None else int(recovery - interval.end)
            ),
            "stable_dropout_prefix_length_frames": int(stable_length),
            "recovery_failure": recovery is None,
            "first_accepted_failure": first_accepted is None,
        }
        if uncertainty_scalar is None:
            row.update(
                {
                    "uncertainty_before_dropout": None,
                    "uncertainty_at_first_missing": None,
                    "uncertainty_at_final_missing": None,
                    "uncertainty_growth": None,
                    "uncertainty_growth_ratio": None,
                    "uncertainty_monotonic": None,
                }
            )
        else:
            base_index = max(0, interval.start - 1)
            trace = uncertainty_scalar[base_index : interval.end]
            finite_trace = bool(np.isfinite(trace).all())
            before = float(uncertainty_scalar[base_index]) if finite_trace else None
            first = float(uncertainty_scalar[interval.start]) if finite_trace else None
            final = (
                float(uncertainty_scalar[interval.end - 1]) if finite_trace else None
            )
            monotonic = finite_trace and bool(
                np.all(np.diff(trace) >= -abs(monotonic_tolerance))
            )
            row.update(
                {
                    "uncertainty_before_dropout": before,
                    "uncertainty_at_first_missing": first,
                    "uncertainty_at_final_missing": final,
                    "uncertainty_growth": None if before is None else final - before,
                    "uncertainty_growth_ratio": (
                        None
                        if before is None or abs(before) <= _EPS
                        else final / before
                    ),
                    "uncertainty_monotonic": monotonic,
                }
            )
        rows.append(row)
    monotonic_values = [
        row["uncertainty_monotonic"]
        for row in rows
        if row["uncertainty_monotonic"] is not None
    ]
    return {
        "interval_count": len(rows),
        "per_interval": rows,
        "final_missing_pose_error": _finite_summary(
            [row["final_missing_pose_error"] for row in rows]
        ),
        "first_return_measurement_error": _finite_summary(
            [row["first_return_measurement_error"] for row in rows]
        ),
        "first_accepted_post_dropout_output_error": _finite_summary(
            [row["first_accepted_post_dropout_output_error"] for row in rows]
        ),
        "frames_to_return_within_nominal": _finite_summary(
            [row["frames_to_return_within_nominal"] for row in rows]
        ),
        "maximum_stable_dropout_length_frames": (
            0
            if not rows
            else max(row["stable_dropout_prefix_length_frames"] for row in rows)
        ),
        "uncertainty_monotonic_all_intervals": (
            None if not monotonic_values else all(monotonic_values)
        ),
        "uncertainty_monotonic_failure_count": sum(
            value is False for value in monotonic_values
        ),
        "recovery_failure_count": sum(bool(row["recovery_failure"]) for row in rows),
        "first_accepted_failure_count": sum(
            bool(row["first_accepted_failure"]) for row in rows
        ),
    }


dropout_metrics = compute_dropout_metrics


def compute_outlier_metrics(
    output_pose_error: Sequence[float] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    intervals: Any,
    *,
    nominal_error_threshold: float = NOMINAL_NORMALIZED_ERROR_THRESHOLD,
) -> dict[str, Any]:
    """Summarize exact accepted/rejected counts and post-burst contamination."""

    output_error = np.asarray(output_pose_error, dtype=np.float64).reshape(-1)
    accepted_array = _boolean_series(accepted, len(output_error), "accepted")
    outlier_intervals = coerce_intervals(intervals, len(output_error), label="outlier")
    rows: list[dict[str, Any]] = []
    for interval in outlier_intervals:
        recovery = _first_within(output_error, interval.end, nominal_error_threshold)
        stop = len(output_error) if recovery is None else recovery + 1
        window = output_error[interval.start : stop]
        finite_window = window[np.isfinite(window)]
        rows.append(
            {
                "start_frame": interval.start,
                "end_frame_exclusive": interval.end,
                "length_frames": interval.length,
                "outliers_accepted": int(
                    np.count_nonzero(accepted_array[interval.start : interval.end])
                ),
                "outliers_rejected": int(
                    np.count_nonzero(~accepted_array[interval.start : interval.end])
                ),
                "maximum_error_after_outlier_onset": (
                    None if len(finite_window) == 0 else float(np.max(finite_window))
                ),
                "recovery_frame": recovery,
                "frames_until_recovery_from_onset": (
                    None if recovery is None else int(recovery - interval.start)
                ),
                "frames_until_recovery_after_burst": (
                    None if recovery is None else int(recovery - interval.end)
                ),
                "contamination_duration_frames": (
                    len(output_error) - interval.end
                    if recovery is None
                    else int(recovery - interval.end)
                ),
                "recovery_failure": recovery is None,
            }
        )
    return {
        "interval_count": len(rows),
        "per_interval": rows,
        "outliers_accepted": sum(row["outliers_accepted"] for row in rows),
        "outliers_rejected": sum(row["outliers_rejected"] for row in rows),
        "maximum_error_after_outlier_onset": (
            None
            if not any(
                row["maximum_error_after_outlier_onset"] is not None for row in rows
            )
            else max(
                row["maximum_error_after_outlier_onset"]
                for row in rows
                if row["maximum_error_after_outlier_onset"] is not None
            )
        ),
        "frames_until_recovery_after_burst": _finite_summary(
            [row["frames_until_recovery_after_burst"] for row in rows]
        ),
        "contamination_duration_frames": sum(
            row["contamination_duration_frames"] for row in rows
        ),
        "recovery_failure_count": sum(bool(row["recovery_failure"]) for row in rows),
    }


outlier_metrics = compute_outlier_metrics


def observable_velocity_series(
    timestamps: Sequence[float] | np.ndarray,
    poses: Sequence[Any] | np.ndarray,
    symmetry: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame-aligned world translation and observable angular velocity."""

    pose_array = _pose_series(poses, "poses", allow_missing=True)
    times = _timestamps(timestamps, len(pose_array))
    translation = np.full((len(pose_array), 3), np.nan, dtype=np.float64)
    angular = np.full((len(pose_array), 3), np.nan, dtype=np.float64)
    name = symmetry_class_name(symmetry)
    previous = pose_array[:-1]
    current = pose_array[1:]
    valid = np.isfinite(previous).all(axis=(1, 2)) & np.isfinite(current).all(
        axis=(1, 2)
    )
    dt = np.diff(times)
    valid_dt = dt[valid]
    translation[1:][valid] = (
        current[valid, :3, 3] - previous[valid, :3, 3]
    ) / valid_dt[:, None]
    previous_rotation = previous[valid, :3, :3]
    current_rotation = current[valid, :3, :3]
    if name == "CONTINUOUS_AXIAL":
        axis_object = _axis_object(symmetry)
        old_axis = previous_rotation @ axis_object
        new_axis = current_rotation @ axis_object
        cross = np.cross(old_axis, new_axis)
        sine = np.linalg.norm(cross, axis=1)
        cosine = np.clip(np.sum(old_axis * new_axis, axis=1), -1.0, 1.0)
        angle = np.arctan2(sine, cosine)
        rotvec = np.zeros_like(cross)
        ordinary = sine > 1e-10
        rotvec[ordinary] = (
            cross[ordinary] / sine[ordinary, None] * angle[ordinary, None]
        )
        angular[1:][valid] = rotvec / valid_dt[:, None]
    else:
        candidates = []
        relatives = []
        for transform in finite_symmetry_transforms(symmetry):
            aligned = current_rotation @ transform[:3, :3]
            relative = np.swapaxes(previous_rotation, 1, 2) @ aligned
            cosine = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
            candidates.append(np.arccos(cosine))
            relatives.append(relative)
        choice = np.argmin(np.stack(candidates, axis=1), axis=1)
        relative_stack = np.stack(relatives, axis=1)
        selected_relative = relative_stack[np.arange(len(choice)), choice]
        body_rotvec = Rotation.from_matrix(selected_relative).as_rotvec()
        world_rotvec = np.einsum("nij,nj->ni", previous_rotation, body_rotvec)
        angular[1:][valid] = world_rotvec / valid_dt[:, None]
    return translation, angular


def _direction_change_details(
    reference_velocity: Sequence[float] | np.ndarray,
    estimated_velocity: Sequence[float] | np.ndarray,
    reversal_frame: int,
    *,
    pre_window: int = 10,
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    reference = np.asarray(reference_velocity, dtype=np.float64)
    estimate = np.asarray(estimated_velocity, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference[:, None]
    if estimate.ndim == 1:
        estimate = estimate[:, None]
    if reference.shape != estimate.shape or reference.ndim != 2:
        raise ValueError(
            "direction-change inputs must have matching (N,) or (N,D) shape"
        )
    if not 0 <= int(reversal_frame) < len(reference):
        raise ValueError("reversal_frame is outside the velocity trace")
    reversal_frame = int(reversal_frame)
    start = max(0, reversal_frame - int(pre_window))
    pre = reference[start:reversal_frame]
    valid = np.isfinite(pre).all(axis=1) & (np.linalg.norm(pre, axis=1) > epsilon)
    direction: np.ndarray | None = None
    if np.any(valid):
        mean = np.mean(pre[valid], axis=0)
        if np.linalg.norm(mean) > epsilon:
            direction = mean / np.linalg.norm(mean)
        else:
            direction = pre[np.flatnonzero(valid)[-1]]
            direction = direction / np.linalg.norm(direction)
    if direction is None:
        return {
            "lag_frames": None,
            "reference_change_frame": None,
            "estimated_change_frame": None,
            "failure": True,
        }
    reference_projection = reference @ direction
    estimate_projection = estimate @ direction
    reference_crossings = np.flatnonzero(
        np.isfinite(reference_projection[reversal_frame:])
        & (reference_projection[reversal_frame:] < -abs(epsilon))
    )
    if len(reference_crossings) == 0:
        return {
            "lag_frames": None,
            "reference_change_frame": None,
            "estimated_change_frame": None,
            "failure": True,
        }
    reference_change = int(reversal_frame + reference_crossings[0])
    estimate_crossings = np.flatnonzero(
        np.isfinite(estimate_projection[reversal_frame:])
        & (estimate_projection[reversal_frame:] < -abs(epsilon))
    )
    estimate_change = (
        None
        if len(estimate_crossings) == 0
        else int(reversal_frame + estimate_crossings[0])
    )
    return {
        "lag_frames": (
            None
            if estimate_change is None
            else max(0, estimate_change - reference_change)
        ),
        "reference_change_frame": reference_change,
        "estimated_change_frame": estimate_change,
        "failure": estimate_change is None,
    }


def direction_change_lag(
    reference_velocity: Sequence[float] | np.ndarray,
    estimated_velocity: Sequence[float] | np.ndarray,
    reversal_frame: int,
    *,
    pre_window: int = 10,
    epsilon: float = 1e-9,
) -> int | None:
    """Return exact non-negative frame lag for a declared direction reversal."""

    return _direction_change_details(
        reference_velocity,
        estimated_velocity,
        reversal_frame,
        pre_window=pre_window,
        epsilon=epsilon,
    )["lag_frames"]


def _rms_vector_norm(values: np.ndarray) -> float | None:
    valid = np.isfinite(values).all(axis=1)
    if not np.any(valid):
        return None
    return float(np.sqrt(np.mean(np.sum(np.square(values[valid]), axis=1))))


def _peak_vector_norm(values: np.ndarray) -> float | None:
    valid = np.isfinite(values).all(axis=1)
    if not np.any(valid):
        return None
    return float(np.max(np.linalg.norm(values[valid], axis=1)))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= _EPS:
        return None
    return float(numerator / denominator)


def compute_dynamic_guardrails(
    timestamps: Sequence[float] | np.ndarray,
    gt_poses: Sequence[Any] | np.ndarray,
    output_poses: Sequence[Any] | np.ndarray,
    symmetry: Any,
    *,
    reversal_frame: int | None = None,
    interval: FrameInterval | Sequence[int] | Mapping[str, Any] | None = None,
    accuracy_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute observable velocity, reversal-lag, attenuation, and smoothing metrics."""

    gt = _pose_series(gt_poses, "gt_poses")
    output = _pose_series(
        output_poses, "output_poses", expected_length=len(gt), allow_missing=True
    )
    times = _timestamps(timestamps, len(gt))
    gt_translation_velocity, gt_angular_velocity = observable_velocity_series(
        times, gt, symmetry
    )
    output_translation_velocity, output_angular_velocity = observable_velocity_series(
        times, output, symmetry
    )
    if interval is None:
        selected = FrameInterval(0, len(gt), "dynamic")
    else:
        selected = coerce_intervals(interval, len(gt), label="dynamic")[0]
    frame_slice = slice(selected.start, selected.end)
    gt_tv = gt_translation_velocity[frame_slice]
    out_tv = output_translation_velocity[frame_slice]
    gt_av = gt_angular_velocity[frame_slice]
    out_av = output_angular_velocity[frame_slice]
    translation_peak_gt = _peak_vector_norm(gt_tv)
    translation_peak_output = _peak_vector_norm(out_tv)
    rotation_peak_gt = _peak_vector_norm(gt_av)
    rotation_peak_output = _peak_vector_norm(out_av)
    translation_peak_ratio = _ratio(translation_peak_output, translation_peak_gt)
    rotation_peak_ratio = _ratio(rotation_peak_output, rotation_peak_gt)

    gt_speed = np.sqrt(
        np.square(np.linalg.norm(gt_tv, axis=1) / TRANSLATION_NORMALIZER_M)
        + np.square(np.degrees(np.linalg.norm(gt_av, axis=1)) / ROTATION_NORMALIZER_DEG)
    )
    output_speed = np.sqrt(
        np.square(np.linalg.norm(out_tv, axis=1) / TRANSLATION_NORMALIZER_M)
        + np.square(
            np.degrees(np.linalg.norm(out_av, axis=1)) / ROTATION_NORMALIZER_DEG
        )
    )
    gt_speed_rms = _finite_summary(gt_speed)["rmse"]
    output_speed_rms = _finite_summary(output_speed)["rmse"]
    oversmoothing_ratio = _ratio(output_speed_rms, gt_speed_rms)

    velocity_pair_valid = np.isfinite(gt_tv).all(axis=1) & np.isfinite(out_tv).all(
        axis=1
    )
    angular_pair_valid = np.isfinite(gt_av).all(axis=1) & np.isfinite(out_av).all(
        axis=1
    )
    translation_velocity_rmse = (
        None
        if not np.any(velocity_pair_valid)
        else float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            out_tv[velocity_pair_valid] - gt_tv[velocity_pair_valid]
                        ),
                        axis=1,
                    )
                )
            )
        )
    )
    angular_velocity_rmse = (
        None
        if not np.any(angular_pair_valid)
        else math.degrees(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.square(
                                out_av[angular_pair_valid] - gt_av[angular_pair_valid]
                            ),
                            axis=1,
                        )
                    )
                )
            )
        )
    )
    accuracy = (
        compute_per_frame_pose_errors(gt, output, symmetry)
        if accuracy_metrics is None
        else accuracy_metrics
    )
    normalized_dynamic_error = accuracy["normalized_pose_error"][frame_slice]
    result: dict[str, Any] = {
        "interval_start_frame": selected.start,
        "interval_end_frame_exclusive": selected.end,
        "translation_peak_velocity_gt_m_s": translation_peak_gt,
        "translation_peak_velocity_output_m_s": translation_peak_output,
        "translation_peak_velocity_ratio": translation_peak_ratio,
        "translation_peak_velocity_attenuation": (
            None
            if translation_peak_ratio is None
            else max(0.0, 1.0 - translation_peak_ratio)
        ),
        "observable_rotation_peak_velocity_gt_deg_s": (
            None if rotation_peak_gt is None else math.degrees(rotation_peak_gt)
        ),
        "observable_rotation_peak_velocity_output_deg_s": (
            None if rotation_peak_output is None else math.degrees(rotation_peak_output)
        ),
        "observable_rotation_peak_velocity_ratio": rotation_peak_ratio,
        "observable_rotation_peak_velocity_attenuation": (
            None if rotation_peak_ratio is None else max(0.0, 1.0 - rotation_peak_ratio)
        ),
        "translation_velocity_rmse_m_s": translation_velocity_rmse,
        "observable_rotation_velocity_rmse_deg_s": angular_velocity_rmse,
        "dynamic_trajectory_rmse": _finite_summary(normalized_dynamic_error)["rmse"],
        "oversmoothing_ratio": oversmoothing_ratio,
        "oversmoothing_deficit": (
            None if oversmoothing_ratio is None else max(0.0, 1.0 - oversmoothing_ratio)
        ),
        "gt_observable_speed_rms": gt_speed_rms,
        "output_observable_speed_rms": output_speed_rms,
        "gt_translation_velocity_m_s": gt_translation_velocity,
        "output_translation_velocity_m_s": output_translation_velocity,
        "gt_observable_angular_velocity_rad_s": gt_angular_velocity,
        "output_observable_angular_velocity_rad_s": output_angular_velocity,
    }
    if reversal_frame is None:
        result.update(
            {
                "translation_direction_change_lag_frames": None,
                "observable_rotation_direction_change_lag_frames": None,
                "translation_direction_change": None,
                "observable_rotation_direction_change": None,
            }
        )
    else:
        translation_change = _direction_change_details(
            gt_translation_velocity, output_translation_velocity, reversal_frame
        )
        rotation_change = _direction_change_details(
            gt_angular_velocity, output_angular_velocity, reversal_frame
        )
        result.update(
            {
                "translation_direction_change_lag_frames": translation_change[
                    "lag_frames"
                ],
                "observable_rotation_direction_change_lag_frames": rotation_change[
                    "lag_frames"
                ],
                "translation_direction_change": translation_change,
                "observable_rotation_direction_change": rotation_change,
            }
        )
    return result


dynamic_guardrail_metrics = compute_dynamic_guardrails


def compute_asymmetric_guardrails(
    accuracy: Mapping[str, Any],
    dynamic: Mapping[str, Any],
    output_poses: Sequence[Any] | np.ndarray,
) -> dict[str, Any]:
    """Return ordinary-pose guardrails for an ASYMMETRIC trajectory."""

    output = _pose_series(output_poses, "output_poses", allow_missing=True)
    nonfinite_frame_count = int(sum(not _finite_pose(pose) for pose in output))
    return {
        "translation_rmse_m": accuracy["translation"]["rmse"],
        "rotation_rmse_deg": accuracy["rotation"]["rmse"],
        "translation_direction_change_lag_frames": dynamic.get(
            "translation_direction_change_lag_frames"
        ),
        "observable_rotation_direction_change_lag_frames": dynamic.get(
            "observable_rotation_direction_change_lag_frames"
        ),
        "numerical_failure_frame_count": nonfinite_frame_count,
    }


asymmetric_guardrail_metrics = compute_asymmetric_guardrails


def compute_runtime_metrics(
    runtimes_s: Sequence[float] | np.ndarray | None,
    output_poses: Sequence[Any] | np.ndarray,
    *,
    residuals: Any = None,
    uncertainty: Any = None,
    initialization_failures: Sequence[bool] | np.ndarray | int | None = None,
    recovery_failures: Sequence[bool] | np.ndarray | int | None = None,
    peak_memory_bytes: int | float | None = None,
) -> dict[str, Any]:
    """Return finite CPU timing and explicit numerical/failure counts."""

    output = _pose_series(output_poses, "output_poses", allow_missing=True)
    raw_output = np.asarray(output, dtype=np.float64)
    nonfinite_values = ~np.isfinite(raw_output)
    if runtimes_s is None:
        runtime = np.full(len(output), np.nan, dtype=np.float64)
    else:
        runtime = np.asarray(runtimes_s, dtype=np.float64).reshape(-1)
        if len(runtime) != len(output):
            raise ValueError("runtime length does not match output poses")
    finite_runtime = runtime[np.isfinite(runtime) & (runtime >= 0.0)]
    residual_nonfinite = 0
    if residuals is not None:
        residual_array = np.asarray(residuals, dtype=np.float64)
        if residual_array.shape[0] != len(output):
            raise ValueError("residual length does not match output poses")
        residual_nonfinite = int(np.count_nonzero(~np.isfinite(residual_array)))
    if uncertainty is None:
        uncertainty_array = np.asarray([], dtype=np.float64)
    else:
        uncertainty_array = np.asarray(uncertainty, dtype=np.float64)
        if uncertainty_array.shape[0] != len(output):
            raise ValueError("uncertainty length does not match output poses")
    uncertainty_nonfinite = ~np.isfinite(uncertainty_array)

    def failure_count(values: Sequence[bool] | np.ndarray | int | None) -> int:
        if values is None:
            return 0
        if isinstance(values, (int, np.integer)):
            return int(values)
        return int(np.count_nonzero(np.asarray(values, dtype=bool)))

    return {
        "frame_count": len(output),
        "finite_runtime_count": int(len(finite_runtime)),
        "nonfinite_runtime_count": int(len(runtime) - len(finite_runtime)),
        "mean_estimator_time_s": (
            None if len(finite_runtime) == 0 else float(np.mean(finite_runtime))
        ),
        "p95_estimator_time_s": (
            None
            if len(finite_runtime) == 0
            else float(np.percentile(finite_runtime, 95))
        ),
        "maximum_estimator_time_s": (
            None if len(finite_runtime) == 0 else float(np.max(finite_runtime))
        ),
        "peak_memory_bytes": None
        if peak_memory_bytes is None
        else int(peak_memory_bytes),
        "nan_output_value_count": int(np.count_nonzero(np.isnan(raw_output))),
        "inf_output_value_count": int(np.count_nonzero(np.isinf(raw_output))),
        "nonfinite_output_value_count": int(np.count_nonzero(nonfinite_values)),
        "nonfinite_output_frame_count": int(
            np.count_nonzero(np.any(nonfinite_values.reshape(len(output), -1), axis=1))
        ),
        "nonfinite_residual_value_count": residual_nonfinite,
        "nan_uncertainty_value_count": int(
            np.count_nonzero(np.isnan(uncertainty_array))
        ),
        "inf_uncertainty_value_count": int(
            np.count_nonzero(np.isinf(uncertainty_array))
        ),
        "nonfinite_uncertainty_value_count": int(
            np.count_nonzero(uncertainty_nonfinite)
        ),
        "initialization_failure_count": failure_count(initialization_failures),
        "recovery_failure_count": failure_count(recovery_failures),
    }


runtime_metrics = compute_runtime_metrics


def _bop_pose_error_module() -> tuple[Any | None, str | None]:
    vendor_root = Path(__file__).resolve().parents[1] / "third_party" / "bop_toolkit"
    package_root = vendor_root / "bop_toolkit_lib"
    if not package_root.is_dir():
        return None, "vendored BOP Toolkit is absent"
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    try:
        module = importlib.import_module("bop_toolkit_lib.pose_error")
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(package_root.resolve()):
            return None, f"bop_toolkit_lib resolved outside vendored checkout: {source}"
        return module, None
    except Exception as exc:  # The exact local implementation remains available.
        return (
            None,
            f"vendored BOP pose_error import failed: {type(exc).__name__}: {exc}",
        )


def _bop_symmetry_dicts(symmetry: Any) -> list[dict[str, np.ndarray]]:
    return [
        {
            "R": transform[:3, :3],
            "t": transform[:3, 3:4],
        }
        for transform in finite_symmetry_transforms(symmetry)
    ]


def _local_mssd(
    output_pose: np.ndarray,
    gt_pose: np.ndarray,
    model_points: np.ndarray,
    symmetries: Sequence[Mapping[str, np.ndarray]],
) -> float:
    estimated = model_points @ output_pose[:3, :3].T + output_pose[:3, 3]
    errors: list[float] = []
    for symmetry in symmetries:
        rotation = gt_pose[:3, :3] @ np.asarray(symmetry["R"], dtype=np.float64)
        translation = (
            gt_pose[:3, :3] @ np.asarray(symmetry["t"], dtype=np.float64).reshape(3)
            + gt_pose[:3, 3]
        )
        transformed = model_points @ rotation.T + translation
        errors.append(float(np.max(np.linalg.norm(estimated - transformed, axis=1))))
    return min(errors)


def bop_normalized_mssd_diagnostic(
    gt_poses: Sequence[Any] | np.ndarray,
    output_poses: Sequence[Any] | np.ndarray,
    symmetry: Any,
    model_points: Sequence[Sequence[float]] | np.ndarray,
    object_diameter: float,
    *,
    prefer_vendored: bool = True,
) -> dict[str, Any]:
    """Compute synthetic normalized MSSD without making an official claim."""

    name = symmetry_class_name(symmetry)
    gt = _pose_series(gt_poses, "gt_poses")
    output = _pose_series(
        output_poses, "output_poses", expected_length=len(gt), allow_missing=True
    )
    if name == "CONTINUOUS_AXIAL":
        return {
            "applicable": False,
            "reason": "continuous axial symmetry is not discretized for this diagnostic",
            "official_benchmark_claim": False,
            "implementation": None,
            "per_frame_normalized_mssd": [None] * len(gt),
        }
    points = np.asarray(model_points, dtype=np.float64)
    diameter = float(object_diameter)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("model_points must be a non-empty finite (P,3) array")
    if not np.isfinite(points).all() or not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(
            "model points and object diameter must be finite; diameter must be positive"
        )
    symmetries = _bop_symmetry_dicts(symmetry)
    module, import_note = (
        _bop_pose_error_module()
        if prefer_vendored
        else (None, "vendored implementation disabled")
    )
    implementation = (
        "vendored_bop_pose_error" if module is not None else "local_exact_mssd_fallback"
    )
    values = np.full(len(gt), np.nan, dtype=np.float64)
    for index, (output_pose, gt_pose) in enumerate(zip(output, gt)):
        if not _finite_pose(output_pose) or not _finite_pose(gt_pose):
            continue
        if module is not None:
            distance = float(
                module.mssd(
                    output_pose[:3, :3],
                    output_pose[:3, 3:4],
                    gt_pose[:3, :3],
                    gt_pose[:3, 3:4],
                    points,
                    symmetries,
                )
            )
        else:
            distance = _local_mssd(output_pose, gt_pose, points, symmetries)
        values[index] = distance / diameter
    return {
        "applicable": True,
        "symmetry_class": name,
        "official_benchmark_claim": False,
        "claim_boundary": (
            "Synthetic, model-point normalized MSSD diagnostic only; not BOP AP, "
            "leaderboard AR, or full-dataset performance."
        ),
        "implementation": implementation,
        "vendored_import_note": import_note,
        "per_frame_normalized_mssd": values,
        "summary": _finite_summary(values),
    }


normalized_mssd_diagnostic = bop_normalized_mssd_diagnostic


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return value.astype(bool).tolist()
        return (
            [
                None if not np.isfinite(item) else float(item)
                for item in value.reshape(-1)
            ]
            if value.ndim == 1
            else _plain(value.tolist())
        )
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _interval_mask(intervals: Sequence[FrameInterval], frame_count: int) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for interval in intervals:
        mask[interval.start : interval.end] = True
    return mask


def _metadata_reversal_frame(metadata: Any) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in ("reversal_frame", "direction_reversal_frame", "reversal_index"):
        if key in metadata and metadata[key] is not None:
            return int(metadata[key])
    return None


def compute_trajectory_metrics(
    timestamps: Sequence[float] | np.ndarray,
    gt_poses: Sequence[Any] | np.ndarray,
    measurement_poses: Sequence[Any] | np.ndarray,
    output_poses: Sequence[Any] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    residuals: Any,
    uncertainty: Any,
    runtimes_s: Sequence[float] | np.ndarray | None,
    symmetry: Any,
    *,
    evaluator_intervals: Any = None,
    dropout_intervals: Any = None,
    outlier_intervals: Any = None,
    reversal_frame: int | None = None,
    nominal_error_threshold: float = NOMINAL_NORMALIZED_ERROR_THRESHOLD,
    model_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    object_diameter: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_frame_metrics: bool = True,
    initialization_failures: Sequence[bool] | np.ndarray | int | None = None,
    recovery_failures: Sequence[bool] | np.ndarray | int | None = None,
    peak_memory_bytes: int | float | None = None,
) -> dict[str, Any]:
    """Evaluate one completed trajectory without exposing truth to estimators."""

    gt = _pose_series(gt_poses, "gt_poses")
    measurement = _pose_series(
        measurement_poses,
        "measurement_poses",
        expected_length=len(gt),
        allow_missing=True,
    )
    output = _pose_series(
        output_poses, "output_poses", expected_length=len(gt), allow_missing=True
    )
    times = _timestamps(timestamps, len(gt))
    accepted_array = _boolean_series(accepted, len(gt), "accepted")
    output_accuracy = compute_per_frame_pose_errors(gt, output, symmetry)
    measurement_accuracy = compute_per_frame_pose_errors(gt, measurement, symmetry)
    stability = compute_representation_stability(times, output, symmetry)
    dropouts = (
        coerce_intervals(dropout_intervals, len(gt), label="dropout")
        if dropout_intervals is not None
        else _metadata_intervals(evaluator_intervals, "dropout", len(gt))
    )
    outliers = (
        coerce_intervals(outlier_intervals, len(gt), label="outlier")
        if outlier_intervals is not None
        else _metadata_intervals(evaluator_intervals, "outlier", len(gt))
    )
    if reversal_frame is None:
        reversal_frame = _metadata_reversal_frame(evaluator_intervals)
    dropout_result = compute_dropout_metrics(
        output_accuracy["normalized_pose_error"],
        measurement_accuracy["normalized_pose_error"],
        accepted_array,
        uncertainty,
        dropouts,
        nominal_error_threshold=nominal_error_threshold,
    )
    outlier_result = compute_outlier_metrics(
        output_accuracy["normalized_pose_error"],
        accepted_array,
        outliers,
        nominal_error_threshold=nominal_error_threshold,
    )
    if reversal_frame is None:
        dynamic = {
            "translation_direction_change_lag_frames": None,
            "observable_rotation_direction_change_lag_frames": None,
            "dynamic_trajectory_rmse": output_accuracy["quotient_pose_rmse"],
            "translation_peak_velocity_ratio": None,
            "observable_rotation_peak_velocity_ratio": None,
            "oversmoothing_ratio": None,
        }
    else:
        dynamic = compute_dynamic_guardrails(
            times,
            gt,
            output,
            symmetry,
            reversal_frame=reversal_frame,
            accuracy_metrics=output_accuracy,
        )
    runtime = compute_runtime_metrics(
        runtimes_s,
        output,
        residuals=residuals,
        uncertainty=uncertainty,
        initialization_failures=initialization_failures,
        recovery_failures=recovery_failures,
        peak_memory_bytes=peak_memory_bytes,
    )
    accuracy_scalars = {
        key: value
        for key, value in output_accuracy.items()
        if key
        not in {"translation_error_m", "rotation_error_deg", "normalized_pose_error"}
    }
    result: dict[str, Any] = {
        "metric_schema_version": 1,
        "metadata": dict(metadata or {}),
        "symmetry_class": symmetry_class_name(symmetry),
        "frame_count": len(gt),
        "accuracy": accuracy_scalars,
        "representation_stability": {
            key: value
            for key, value in stability.items()
            if not isinstance(value, np.ndarray)
        },
        "dropout": dropout_result,
        "outlier": outlier_result,
        "dynamic_guardrails": {
            key: value
            for key, value in dynamic.items()
            if not isinstance(value, np.ndarray)
        },
        "runtime_and_finite_counts": runtime,
        "metric_definitions": {
            "normalized_pose_error": "sqrt((translation_m/0.01)^2 + (rotation_deg/5)^2)",
            "interval_convention": "half-open [start_frame,end_frame_exclusive)",
            "nominal_normalized_error_threshold": float(nominal_error_threshold),
            "catastrophic_jump": "raw_step_deg > 90 and quotient_step_deg < 10",
            "continuous_gauge_jump": "abs(axial_spin_deg) > 45 and observable_step_deg < 10",
            "oversmoothing_ratio": "output observable-speed RMS / GT observable-speed RMS; 0 is frozen, 1 is matched",
        },
    }
    if symmetry_class_name(symmetry) == "ASYMMETRIC":
        result["asymmetric_guardrails"] = compute_asymmetric_guardrails(
            output_accuracy, dynamic, output
        )
    else:
        result["asymmetric_guardrails"] = None
    if model_points is not None or object_diameter is not None:
        if model_points is None or object_diameter is None:
            raise ValueError(
                "model_points and object_diameter must be supplied together"
            )
        result["normalized_mssd_diagnostic"] = bop_normalized_mssd_diagnostic(
            gt, output, symmetry, model_points, object_diameter
        )
    else:
        result["normalized_mssd_diagnostic"] = None

    if include_frame_metrics:
        uncertainty_scalar = _uncertainty_scalar(uncertainty, len(gt))
        runtime_array = (
            np.full(len(gt), np.nan, dtype=np.float64)
            if runtimes_s is None
            else np.asarray(runtimes_s, dtype=np.float64).reshape(-1)
        )
        residual_array = (
            None if residuals is None else np.asarray(residuals, dtype=np.float64)
        )
        dropout_mask = _interval_mask(dropouts, len(gt))
        outlier_mask = _interval_mask(outliers, len(gt))
        frames: list[dict[str, Any]] = []
        for index in range(len(gt)):
            residual_value: Any = None
            if residual_array is not None:
                current = residual_array[index]
                residual_value = (
                    float(current)
                    if np.ndim(current) == 0
                    else np.asarray(current).tolist()
                )
            frames.append(
                {
                    "frame_index": index,
                    "timestamp_s": float(times[index]),
                    "translation_error_m": output_accuracy["translation_error_m"][
                        index
                    ],
                    "rotation_error_deg": output_accuracy["rotation_error_deg"][index],
                    "normalized_pose_error": output_accuracy["normalized_pose_error"][
                        index
                    ],
                    "measurement_normalized_pose_error": measurement_accuracy[
                        "normalized_pose_error"
                    ][index],
                    "raw_rotation_step_deg": None
                    if index == 0
                    else stability["raw_rotation_step_deg"][index - 1],
                    "quotient_rotation_step_deg": None
                    if index == 0
                    else stability["quotient_rotation_step_deg"][index - 1],
                    "axial_spin_deg": None
                    if index == 0
                    else stability["axial_spin_deg"][index - 1],
                    "axial_angular_velocity_rad_s": None
                    if index == 0
                    else stability["axial_angular_velocity_rad_s"][index - 1],
                    "catastrophic_jump": False
                    if index == 0
                    else stability["catastrophic_jump_mask"][index - 1],
                    "axial_gauge_jump": False
                    if index == 0
                    else stability["axial_gauge_jump_mask"][index - 1],
                    "accepted": accepted_array[index],
                    "residual": residual_value,
                    "uncertainty_scalar": None
                    if uncertainty_scalar is None
                    else uncertainty_scalar[index],
                    "runtime_s": runtime_array[index],
                    "measurement_missing_evaluator_only": dropout_mask[index],
                    "outlier_evaluator_only": outlier_mask[index],
                }
            )
        result["frame_metrics"] = frames
    return _plain(result)


evaluate_trajectory = compute_trajectory_metrics


def _flatten_scalars(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output is None:
        output = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in {"frame_metrics", "per_interval", "metric_definitions"}:
                continue
            _flatten_scalars(item, prefix=path, output=output)
    elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
        output[prefix] = _plain(value)
    return output


def make_trajectory_row(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten scalar trajectory metrics into one deterministic result row."""

    row = dict(metadata or {})
    flattened = _flatten_scalars(metrics)
    for key, value in flattened.items():
        output_key = key.replace(".", "__")
        if output_key in row:
            raise ValueError(
                f"trajectory metadata collides with metric field: {output_key}"
            )
        row[output_key] = value
    return row


trajectory_row = make_trajectory_row


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    alternate = key.replace(".", "__")
    return row.get(alternate)


def _is_count_metric(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "count",
            "jumps",
            "outliers_accepted",
            "outliers_rejected",
            "failure",
            "contamination_duration",
        )
    )


def aggregate_trajectory_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Aggregate scalar rows deterministically by condition metadata."""

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if "accuracy" in row or "runtime_and_finite_counts" in row:
            normalized_rows.append(make_trajectory_row(row))
        else:
            normalized_rows.append(dict(row))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in normalized_rows:
        key = tuple(_row_value(row, field) for field in group_by)
        groups.setdefault(key, []).append(row)
    results: list[dict[str, Any]] = []
    for group_key in sorted(
        groups, key=lambda value: tuple(str(item) for item in value)
    ):
        group_rows = groups[group_key]
        fields = sorted(set().union(*(row.keys() for row in group_rows)))
        metrics: dict[str, Any] = {}
        for field in fields:
            if field in group_by:
                continue
            numeric = [
                float(row[field])
                for row in group_rows
                if field in row
                and not isinstance(row[field], (bool, np.bool_))
                and isinstance(row[field], (int, float, np.integer, np.floating))
                and math.isfinite(float(row[field]))
            ]
            if not numeric:
                continue
            summary = _finite_summary(numeric)
            if _is_count_metric(field):
                summary["sum"] = float(np.sum(numeric))
            metrics[field] = summary
        results.append(
            {
                "group": dict(zip(group_by, group_key)),
                "trajectory_count": len(group_rows),
                "metrics": metrics,
            }
        )
    return results


def aggregate_asymmetric_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = ("method",),
) -> list[dict[str, Any]]:
    """Aggregate only ASYMMETRIC trajectory rows for formal guardrails."""

    selected = []
    for row in rows:
        symmetry = _row_value(row, "symmetry_class")
        if symmetry is None:
            symmetry = _row_value(row, "metadata.symmetry_class")
        if symmetry is not None and _enum_text(symmetry) == "ASYMMETRIC":
            selected.append(row)
    return aggregate_trajectory_rows(selected, group_by=group_by)


__all__ = [
    "AXIAL_GAUGE_SPIN_DEG",
    "CATASTROPHIC_QUOTIENT_STEP_DEG",
    "CATASTROPHIC_RAW_STEP_DEG",
    "FrameInterval",
    "NOMINAL_NORMALIZED_ERROR_THRESHOLD",
    "ROTATION_NORMALIZER_DEG",
    "TRANSLATION_NORMALIZER_M",
    "aggregate_asymmetric_rows",
    "aggregate_trajectory_rows",
    "asymmetric_guardrail_metrics",
    "bop_normalized_mssd_diagnostic",
    "coerce_intervals",
    "compute_asymmetric_guardrails",
    "compute_dropout_metrics",
    "compute_dynamic_guardrails",
    "compute_outlier_metrics",
    "compute_per_frame_pose_errors",
    "compute_representation_stability",
    "compute_runtime_metrics",
    "compute_trajectory_metrics",
    "continuous_axial_spin_degrees",
    "direction_change_lag",
    "dropout_metrics",
    "dynamic_guardrail_metrics",
    "evaluate_trajectory",
    "finite_symmetry_transforms",
    "make_trajectory_row",
    "normalized_mssd_diagnostic",
    "normalized_pose_error",
    "observable_velocity_series",
    "outlier_metrics",
    "per_frame_pose_errors",
    "quotient_pose_error",
    "quotient_rotation_error_degrees",
    "representation_stability_metrics",
    "runtime_metrics",
    "symmetry_class_name",
    "trajectory_row",
]
