#!/usr/bin/env python3
"""Small shared contracts for the PoseLoop M1 scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MANIFEST_SCHEMA_VERSION = 1
BATCH_SCHEMA_VERSION = 1
SELECTION_SEED = 20260730
INFERENCE_SEED = 0
FOUNDATIONPOSE_ITERATIONS = 5
MODALITY = "realsense"
GT_ROTATION_ATOL = 1e-4
ALLOWED_STATUSES = {
    "success",
    "invalid_input",
    "inference_error",
    "nonfinite_pose",
}
VISIBILITY_BIN_ORDER = ("low", "mid", "high")
VISIBILITY_TARGETS = {"low": 6, "mid": 7, "high": 7}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def visibility_bin(visible_fraction: float) -> str:
    if 0.10 <= visible_fraction < 0.40:
        return "low"
    if 0.40 <= visible_fraction < 0.70:
        return "mid"
    if 0.70 <= visible_fraction <= 1.01:
        return "high"
    raise ValueError(f"Visibility is outside M1 range: {visible_fraction}")


def seeded_key(*parts: Any, seed: int = SELECTION_SEED) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_sample_id(
    scene_id: int, image_id: int, gt_index: int, object_id: int
) -> str:
    return (
        f"xyzibd-val-rs-s{scene_id:06d}-i{image_id:06d}-"
        f"g{gt_index:06d}-o{object_id:06d}"
    )


def bop_pose_m(gt_entry: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        gt_entry["cam_R_m2c"], dtype=np.float64
    ).reshape(3, 3)
    transform[:3, 3] = (
        np.asarray(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3) * 0.001
    )
    assert_pose(transform, "BOP GT pose", rotation_atol=GT_ROTATION_ATOL)
    return transform


def assert_pose(transform: np.ndarray, label: str, rotation_atol: float = 5e-3) -> None:
    transform = np.asarray(transform)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} is not a finite 4x4 matrix")
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=rotation_atol)
    rotation = transform[:3, :3]
    np.testing.assert_allclose(
        rotation.T @ rotation, np.eye(3), atol=rotation_atol
    )
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=rotation_atol):
        raise ValueError(f"{label} rotation determinant is not +1")


def raw_pose_errors(
    predicted_pose_m: np.ndarray, gt_pose_m: np.ndarray
) -> tuple[float, float]:
    predicted_pose_m = np.asarray(predicted_pose_m, dtype=np.float64)
    gt_pose_m = np.asarray(gt_pose_m, dtype=np.float64)
    translation_error_mm = float(
        1000.0
        * np.linalg.norm(predicted_pose_m[:3, 3] - gt_pose_m[:3, 3])
    )
    cosine = float(
        np.clip(
            (
                np.trace(
                    predicted_pose_m[:3, :3] @ gt_pose_m[:3, :3].T
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )
    rotation_error_degrees = float(np.degrees(np.arccos(cosine)))
    return translation_error_mm, rotation_error_degrees
