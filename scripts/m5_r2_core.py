#!/usr/bin/env python3
"""M5-R2 coordinate adapters around the byte-frozen M5 temporal kernels."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import m5_g0_core as legacy


def normalized_axis(axis_object: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_object, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Continuous symmetry axis must be finite and non-zero")
    return axis / norm


def axial_rotation_transform(axis_object: np.ndarray, angle_rad: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(
        normalized_axis(axis_object) * float(angle_rad)
    ).as_matrix()
    return transform


def canonical_object_to_original_object(axis_object: np.ndarray) -> np.ndarray:
    """Return a right-handed basis whose canonical z axis is the original axis."""

    axis = normalized_axis(axis_object)
    if np.array_equal(axis, np.array([0.0, 0.0, 1.0])):
        return np.eye(4, dtype=np.float64)
    seed = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
    first = seed - axis * float(np.dot(axis, seed))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack([first, second, axis])
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12):
        raise RuntimeError("M5-R2 continuous canonical basis is not orthonormal")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-12):
        raise RuntimeError("M5-R2 continuous canonical basis is not right-handed")
    return transform


def canonicalization_for_symmetry(
    symmetry: legacy.SymmetrySpec,
) -> Tuple[np.ndarray, legacy.SymmetrySpec]:
    """Adapt arbitrary continuous axes to the frozen z-axis filter convention."""

    if not symmetry.continuous:
        return np.eye(4, dtype=np.float64), symmetry
    basis = canonical_object_to_original_object(symmetry.axis_object)
    return basis, legacy.symmetry_spec(legacy.SymmetryClass.CONTINUOUS_AXIAL)


def to_canonical_pose(
    original_pose: np.ndarray, canonical_to_original: np.ndarray
) -> np.ndarray:
    return np.asarray(original_pose, dtype=np.float64).reshape(4, 4) @ np.asarray(
        canonical_to_original, dtype=np.float64
    ).reshape(4, 4)


def from_canonical_pose(
    canonical_pose: np.ndarray, canonical_to_original: np.ndarray
) -> np.ndarray:
    return np.asarray(canonical_pose, dtype=np.float64).reshape(4, 4) @ legacy.rigid_inverse(
        np.asarray(canonical_to_original, dtype=np.float64).reshape(4, 4)
    )


__all__ = [
    "axial_rotation_transform",
    "canonical_object_to_original_object",
    "canonicalization_for_symmetry",
    "from_canonical_pose",
    "normalized_axis",
    "to_canonical_pose",
]
