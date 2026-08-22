#!/usr/bin/env python3
"""Audit RU-APC target-pose and visible-mask source encoding without selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/cgliu/datasets/ruapc")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r6_protocol.json",
    )
    return parser.parse_args()


def rotation_diagnostics(entry: dict[str, Any]) -> dict[str, float]:
    rotation = np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("RU-APC source rotation contains a non-finite value")
    residual = rotation.T @ rotation - np.eye(3, dtype=np.float64)
    left, singular_values, right_t = np.linalg.svd(rotation)
    projected = left @ right_t
    if np.linalg.det(projected) < 0:
        left = left.copy()
        left[:, -1] *= -1.0
        projected = left @ right_t
    return {
        "orthogonality_max_abs": float(np.max(np.abs(residual))),
        "orthogonality_inf": float(np.linalg.norm(residual, ord=np.inf)),
        "determinant": float(np.linalg.det(rotation)),
        "singular_value_min": float(np.min(singular_values)),
        "singular_value_max": float(np.max(singular_values)),
        "projection_max_abs": float(np.max(np.abs(projected - rotation))),
        "projection_fro": float(np.linalg.norm(projected - rotation, ord="fro")),
    }


def summarize_rotation_diagnostics(
    rows: list[dict[str, float]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("RU-APC rotation source audit is empty")

    def maximum(name: str) -> float:
        return max(float(row[name]) for row in rows)

    def minimum(name: str) -> float:
        return min(float(row[name]) for row in rows)

    return {
        "orthogonality_max_abs_max": maximum("orthogonality_max_abs"),
        "orthogonality_inf_max": maximum("orthogonality_inf"),
        "determinant_min": minimum("determinant"),
        "determinant_max": maximum("determinant"),
        "singular_value_min": minimum("singular_value_min"),
        "singular_value_max": maximum("singular_value_max"),
        "projection_max_abs_max": maximum("projection_max_abs"),
        "projection_fro_max": maximum("projection_fro"),
        "shared_strict_parser_failure_count": sum(
            row["orthogonality_max_abs"] > 1e-4
            or abs(row["determinant"] - 1.0) > 1e-4
            for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "poseloop-m5-r6-v1":
        raise ValueError("Unexpected M5-R6 protocol")
    scene_object_map = {
        int(scene_id): int(object_id)
        for scene_id, object_id in protocol["source"]["scene_object_map"].items()
    }

    rotation_rows: list[dict[str, float]] = []
    mask_deltas: list[int] = []
    mismatch_examples: list[dict[str, Any]] = []
    by_scene: dict[str, dict[str, Any]] = {}
    for scene_id, target_object_id in sorted(scene_object_map.items()):
        scene_dir = dataset_root / "test" / f"{scene_id:06d}"
        ground_truth = json.loads(
            (scene_dir / "scene_gt.json").read_text(encoding="utf-8")
        )
        information = json.loads(
            (scene_dir / "scene_gt_info.json").read_text(encoding="utf-8")
        )
        if set(ground_truth) != set(information):
            raise ValueError(f"RU-APC GT/info frame keys differ: scene={scene_id}")
        scene_deltas: list[int] = []
        scene_rows = 0
        for key in sorted(ground_truth, key=int):
            gt_entries = ground_truth[key]
            info_entries = information[key]
            if len(gt_entries) != len(info_entries):
                raise ValueError(
                    f"RU-APC GT/info length mismatch: scene={scene_id} image={key}"
                )
            indices = [
                index
                for index, entry in enumerate(gt_entries)
                if int(entry["obj_id"]) == target_object_id
            ]
            if len(indices) > 1:
                raise ValueError(
                    f"RU-APC duplicate target instance: scene={scene_id} image={key}"
                )
            if not indices:
                continue
            gt_index = indices[0]
            rotation_rows.append(rotation_diagnostics(gt_entries[gt_index]))
            mask_path = (
                scene_dir / "mask_visib" / f"{int(key):06d}_{gt_index:06d}.png"
            )
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                raise ValueError(f"Cannot decode RU-APC visible mask: {mask_path}")
            if mask_image.ndim == 3:
                mask_image = mask_image[..., 0]
            actual = int(np.count_nonzero(np.asarray(mask_image) > 0))
            declared = int(info_entries[gt_index]["px_count_visib"])
            delta = actual - declared
            mask_deltas.append(delta)
            scene_deltas.append(delta)
            scene_rows += 1
            if delta and len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "path": str(mask_path),
                        "actual": actual,
                        "declared": declared,
                        "delta": delta,
                    }
                )
        by_scene[str(scene_id)] = {
            "object_id": target_object_id,
            "target_entry_count": scene_rows,
            "mask_delta_max_abs": max(map(abs, scene_deltas), default=0),
            "mask_nonzero_delta_count": sum(delta != 0 for delta in scene_deltas),
        }

    if not rotation_rows or not mask_deltas:
        raise RuntimeError("RU-APC source audit found no target entries")

    delta_counts = Counter(mask_deltas)
    result = {
        "protocol_id": protocol["protocol_id"],
        "audit_scope": "source encoding only; no window selection or pose error",
        "target_entry_count": len(rotation_rows),
        "rotation": summarize_rotation_diagnostics(rotation_rows),
        "visible_mask_metadata": {
            "exact_match_count": int(delta_counts.get(0, 0)),
            "nonzero_delta_count": sum(delta != 0 for delta in mask_deltas),
            "maximum_absolute_delta_pixels": max(map(abs, mask_deltas)),
            "signed_delta_counts": {
                str(delta): int(count) for delta, count in sorted(delta_counts.items())
            },
            "availability_will_use_actual_png_count": True,
            "mismatch_examples_first_20": mismatch_examples,
        },
        "by_scene": by_scene,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
