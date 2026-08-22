#!/usr/bin/env python3
"""Build the deterministic PoseLoop M1 XYZ-IBD RealSense manifest."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from m1_common import (
    MANIFEST_SCHEMA_VERSION,
    MODALITY,
    SELECTION_SEED,
    VISIBILITY_BIN_ORDER,
    VISIBILITY_TARGETS,
    bop_pose_m,
    read_json,
    seeded_key,
    sha256_file,
    stable_sample_id,
    visibility_bin,
    write_json_atomic,
    write_jsonl_atomic,
)


SAMPLES_PER_OBJECT = 20


@dataclass(frozen=True)
class Candidate:
    scene_id: int
    image_id: int
    gt_instance_index: int
    object_id: int
    visible_fraction: float
    declared_visible_mask_pixels: int
    visibility_bin: str
    scene_dir: Path
    gt_entry: dict[str, Any]
    camera_entry: dict[str, Any]

    @property
    def frame_key(self) -> tuple[int, int]:
        return self.scene_id, self.image_id

    @property
    def deterministic_key(self) -> tuple[str, int, int, int]:
        return (
            seeded_key(
                self.object_id,
                self.visibility_bin,
                self.scene_id,
                self.image_id,
                self.gt_instance_index,
            ),
            self.scene_id,
            self.image_id,
            self.gt_instance_index,
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    asset_base = Path(os.environ.get("POSELOOP_DATA_ROOT", Path.home() / "datasets"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=asset_base / "xyzibd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest_summary.json",
    )
    return parser.parse_args()


def numeric_scene_dirs(validation_root: Path) -> list[Path]:
    if not validation_root.is_dir():
        raise FileNotFoundError(f"XYZ-IBD validation directory not found: {validation_root}")
    scene_dirs = sorted(
        (
            path
            for path in validation_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    if not scene_dirs:
        raise ValueError(f"No numeric validation scenes found in {validation_root}")
    return scene_dirs


def discover_candidates(data_root: Path) -> tuple[list[Candidate], dict[str, Any]]:
    validation_root = data_root / "val"
    candidates: list[Candidate] = []
    scanned_scene_ids: list[int] = []
    scanned_frame_count = 0
    metadata_instance_count = 0
    below_visibility_count = 0
    empty_declared_mask_count = 0

    for scene_dir in numeric_scene_dirs(validation_root):
        gt_path = scene_dir / "scene_gt_realsense.json"
        info_path = scene_dir / "scene_gt_info_realsense.json"
        camera_path = scene_dir / "scene_camera_realsense.json"
        if not (gt_path.is_file() and info_path.is_file() and camera_path.is_file()):
            continue
        scene_id = int(scene_dir.name)
        scanned_scene_ids.append(scene_id)
        scene_gt = read_json(gt_path)
        scene_info = read_json(info_path)
        scene_camera = read_json(camera_path)
        if set(scene_gt) != set(scene_info):
            raise ValueError(f"GT/info frame mismatch in scene {scene_id:06d}")

        for image_key in sorted(scene_gt, key=int):
            scanned_frame_count += 1
            if image_key not in scene_camera:
                raise ValueError(
                    f"Missing RealSense camera entry in scene {scene_id:06d}, "
                    f"image {int(image_key):06d}"
                )
            gt_entries = scene_gt[image_key]
            info_entries = scene_info[image_key]
            if len(gt_entries) != len(info_entries):
                raise ValueError(
                    f"GT/info instance mismatch in scene {scene_id:06d}, "
                    f"image {int(image_key):06d}"
                )
            for gt_index, (gt_entry, info_entry) in enumerate(
                zip(gt_entries, info_entries, strict=True)
            ):
                metadata_instance_count += 1
                visible_fraction = float(info_entry["visib_fract"])
                declared_pixels = int(info_entry["px_count_visib"])
                if visible_fraction < 0.10:
                    below_visibility_count += 1
                    continue
                if declared_pixels <= 0:
                    empty_declared_mask_count += 1
                    continue
                candidates.append(
                    Candidate(
                        scene_id=scene_id,
                        image_id=int(image_key),
                        gt_instance_index=gt_index,
                        object_id=int(gt_entry["obj_id"]),
                        visible_fraction=visible_fraction,
                        declared_visible_mask_pixels=declared_pixels,
                        visibility_bin=visibility_bin(visible_fraction),
                        scene_dir=scene_dir,
                        gt_entry=gt_entry,
                        camera_entry=scene_camera[image_key],
                    )
                )

    if not candidates:
        raise ValueError("No RealSense candidates satisfy visibility >= 0.10")
    discovery = {
        "scene_ids": scanned_scene_ids,
        "scene_count": len(scanned_scene_ids),
        "frame_count_from_gt": scanned_frame_count,
        "metadata_instance_count": metadata_instance_count,
        "eligible_metadata_instance_count": len(candidates),
        "below_minimum_visibility_count": below_visibility_count,
        "empty_declared_mask_count": empty_declared_mask_count,
    }
    return candidates, discovery


def canonicalize_per_frame_and_bin(candidates: Iterable[Candidate]) -> list[Candidate]:
    grouped: dict[tuple[int, str, int, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                candidate.object_id,
                candidate.visibility_bin,
                candidate.scene_id,
                candidate.image_id,
            )
        ].append(candidate)
    canonical = [
        min(frame_candidates, key=lambda item: item.deterministic_key)
        for frame_candidates in grouped.values()
    ]
    return sorted(
        canonical,
        key=lambda item: (
            item.object_id,
            VISIBILITY_BIN_ORDER.index(item.visibility_bin),
            item.scene_id,
            item.image_id,
            item.gt_instance_index,
        ),
    )


def dispersion_distance(
    candidate: Candidate,
    selected: list[Candidate],
) -> int:
    same_scene_image_ids = [
        existing.image_id
        for existing in selected
        if existing.scene_id == candidate.scene_id
    ]
    if not same_scene_image_ids:
        return 1_000_000_000
    return min(abs(candidate.image_id - image_id) for image_id in same_scene_image_ids)


def choose_one(
    available: list[Candidate],
    selected: list[Candidate],
    used_frames: set[tuple[int, int]],
) -> Candidate | None:
    eligible = [
        candidate for candidate in available if candidate.frame_key not in used_frames
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            -dispersion_distance(item, selected),
            *item.deterministic_key,
        ),
    )


def select_object_candidates(
    object_id: int,
    candidates: list[Candidate],
) -> list[Candidate]:
    by_bin = {
        label: [
            candidate
            for candidate in candidates
            if candidate.object_id == object_id and candidate.visibility_bin == label
        ]
        for label in VISIBILITY_BIN_ORDER
    }
    selected: list[Candidate] = []
    used_frames: set[tuple[int, int]] = set()
    selected_counts: Counter[str] = Counter()

    for label in VISIBILITY_BIN_ORDER:
        for _ in range(VISIBILITY_TARGETS[label]):
            candidate = choose_one(by_bin[label], selected, used_frames)
            if candidate is None:
                break
            selected.append(candidate)
            used_frames.add(candidate.frame_key)
            selected_counts[label] += 1

    while len(selected) < SAMPLES_PER_OBJECT:
        choices: list[tuple[int, int, Candidate]] = []
        for bin_index, label in enumerate(VISIBILITY_BIN_ORDER):
            candidate = choose_one(by_bin[label], selected, used_frames)
            if candidate is not None:
                choices.append((selected_counts[label], bin_index, candidate))
        if not choices:
            break
        _, _, candidate = min(choices, key=lambda item: (item[0], item[1]))
        selected.append(candidate)
        used_frames.add(candidate.frame_key)
        selected_counts[candidate.visibility_bin] += 1

    if len(selected) != SAMPLES_PER_OBJECT:
        raise ValueError(
            f"Object {object_id} has only {len(selected)} selectable distinct frames"
        )
    return selected


def selected_to_manifest_row(
    candidate: Candidate,
    data_root: Path,
    selection_index: int,
) -> dict[str, Any]:
    stem = f"{candidate.image_id:06d}"
    rgb_path = candidate.scene_dir / "rgb_realsense" / f"{stem}.png"
    depth_path = candidate.scene_dir / "depth_realsense" / f"{stem}.png"
    mask_path = (
        candidate.scene_dir
        / "mask_visib_realsense"
        / f"{stem}_{candidate.gt_instance_index:06d}.png"
    )
    model_path = data_root / "models" / f"obj_{candidate.object_id:06d}.ply"
    required = (rgb_path, depth_path, mask_path, model_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected instance references missing files: {missing}")

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or raw_depth is None or mask_image is None:
        raise RuntimeError(
            f"OpenCV failed to read selected RGB/depth/mask for {candidate}"
        )
    if mask_image.ndim == 3:
        mask_image = mask_image[..., 0]
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image: {rgb_path}")
    if raw_depth.ndim != 2 or mask_image.ndim != 2:
        raise ValueError(f"Expected 2D depth and mask: {depth_path}, {mask_path}")
    if rgb_bgr.shape[:2] != raw_depth.shape or raw_depth.shape != mask_image.shape:
        raise ValueError(
            f"RGB/depth/mask shape mismatch for scene {candidate.scene_id}, "
            f"image {candidate.image_id}"
        )

    visible_mask = mask_image > 0
    actual_mask_pixels = int(visible_mask.sum())
    if actual_mask_pixels <= 0:
        raise ValueError(f"Selected visible mask is empty: {mask_path}")
    if actual_mask_pixels != candidate.declared_visible_mask_pixels:
        raise ValueError(
            f"Visible-mask pixel mismatch for {mask_path}: "
            f"{actual_mask_pixels} != {candidate.declared_visible_mask_pixels}"
        )

    depth_scale = float(candidate.camera_entry["depth_scale"])
    if not math_is_finite_positive(depth_scale):
        raise ValueError(f"Invalid BOP depth_scale for {candidate}: {depth_scale}")
    physical_depth_mm = raw_depth.astype(np.float64) * depth_scale
    valid_depth_mask = (
        visible_mask & np.isfinite(physical_depth_mm) & (physical_depth_mm > 0)
    )
    valid_depth_pixels = int(valid_depth_mask.sum())
    valid_depth_ratio = valid_depth_pixels / actual_mask_pixels
    camera_matrix = np.asarray(
        candidate.camera_entry["cam_K"],
        dtype=np.float64,
    ).reshape(3, 3)
    if not np.isfinite(camera_matrix).all():
        raise ValueError(f"Non-finite camera matrix for {candidate}")
    gt_pose_m = bop_pose_m(candidate.gt_entry)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_index": selection_index,
        "selection_seed": SELECTION_SEED,
        "sample_id": stable_sample_id(
            candidate.scene_id,
            candidate.image_id,
            candidate.gt_instance_index,
            candidate.object_id,
        ),
        "dataset": "xyzibd",
        "dataset_split": "val",
        "scene_id": candidate.scene_id,
        "image_id": candidate.image_id,
        "gt_instance_index": candidate.gt_instance_index,
        "object_id": candidate.object_id,
        "sensor_modality": MODALITY,
        "visible_fraction": candidate.visible_fraction,
        "visibility_bin": candidate.visibility_bin,
        "visible_mask_pixel_count": actual_mask_pixels,
        "valid_depth_pixel_count": valid_depth_pixels,
        "valid_depth_ratio_inside_mask": valid_depth_ratio,
        "image_height": int(rgb_bgr.shape[0]),
        "image_width": int(rgb_bgr.shape[1]),
        "rgb_path": str(rgb_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "model_path": str(model_path.resolve()),
        "camera_intrinsics_row_major": camera_matrix.tolist(),
        "raw_depth_scale": depth_scale,
        "gt_model_to_camera_pose_m": gt_pose_m.tolist(),
    }


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def validate_selection(candidates: list[Candidate]) -> None:
    if len(candidates) != 300:
        raise ValueError(f"Expected 300 selected instances, got {len(candidates)}")
    object_counts = Counter(candidate.object_id for candidate in candidates)
    if len(object_counts) != 15 or set(object_counts.values()) != {20}:
        raise ValueError(f"Expected 15 objects x 20 instances, got {dict(object_counts)}")
    sample_ids = [
        stable_sample_id(
            candidate.scene_id,
            candidate.image_id,
            candidate.gt_instance_index,
            candidate.object_id,
        )
        for candidate in candidates
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Stable sample IDs are not unique")
    object_frames = [
        (candidate.object_id, candidate.scene_id, candidate.image_id)
        for candidate in candidates
    ]
    if len(object_frames) != len(set(object_frames)):
        raise ValueError("More than one GT instance per object/image was selected")


def build_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    discovery: dict[str, Any],
    canonical_candidate_count: int,
) -> dict[str, Any]:
    object_ids = sorted({int(row["object_id"]) for row in rows})
    by_object = []
    for object_id in object_ids:
        object_rows = [row for row in rows if int(row["object_id"]) == object_id]
        counts = Counter(row["visibility_bin"] for row in object_rows)
        by_object.append(
            {
                "object_id": object_id,
                "sample_count": len(object_rows),
                "visibility_bins": {
                    label: counts[label] for label in VISIBILITY_BIN_ORDER
                },
                "distinct_frame_count": len(
                    {(row["scene_id"], row["image_id"]) for row in object_rows}
                ),
            }
        )
    visibility_counts = Counter(row["visibility_bin"] for row in rows)
    depth_ratios = np.asarray(
        [row["valid_depth_ratio_inside_mask"] for row in rows],
        dtype=np.float64,
    )
    return {
        "schema_version": 1,
        "manifest": str(output_path.resolve()),
        "manifest_sha256": sha256_file(output_path),
        "selection": {
            "seed": SELECTION_SEED,
            "sensor_modality": MODALITY,
            "minimum_visible_fraction": 0.10,
            "target_visibility_bins_per_object": VISIBILITY_TARGETS,
            "samples_per_object": SAMPLES_PER_OBJECT,
            "depth_filtering": "none",
            "instance_tie_break": "minimum seeded SHA-256 per object/bin/frame",
            "frame_distribution": "greedy maximin image-ID spacing with stable ties",
            "deficit_fill": "least-populated available visibility bin with stable ties",
        },
        "discovery": {
            **discovery,
            "canonical_object_bin_frame_candidate_count": canonical_candidate_count,
        },
        "sample_count": len(rows),
        "object_count": len(object_ids),
        "object_ids": object_ids,
        "visibility_bins": {
            label: visibility_counts[label] for label in VISIBILITY_BIN_ORDER
        },
        "by_object": by_object,
        "valid_depth_ratio_inside_mask": {
            "minimum": float(depth_ratios.min()),
            "median": float(np.median(depth_ratios)),
            "maximum": float(depth_ratios.max()),
            "below_0_90_count": int(np.sum(depth_ratios < 0.90)),
            "below_0_50_count": int(np.sum(depth_ratios < 0.50)),
        },
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_path = args.output.resolve()
    summary_path = args.summary_output.resolve()
    candidates, discovery = discover_candidates(data_root)
    canonical = canonicalize_per_frame_and_bin(candidates)
    object_ids = sorted({candidate.object_id for candidate in canonical})
    if len(object_ids) != 15:
        raise ValueError(f"Expected 15 discovered objects, got {object_ids}")

    selected: list[Candidate] = []
    for object_id in object_ids:
        selected.extend(select_object_candidates(object_id, canonical))
    selected.sort(
        key=lambda item: (
            item.object_id,
            VISIBILITY_BIN_ORDER.index(item.visibility_bin),
            item.scene_id,
            item.image_id,
            item.gt_instance_index,
        )
    )
    validate_selection(selected)

    rows = [
        selected_to_manifest_row(candidate, data_root, selection_index=index)
        for index, candidate in enumerate(selected)
    ]
    write_jsonl_atomic(output_path, rows)
    summary = build_summary(rows, output_path, discovery, len(canonical))
    write_json_atomic(summary_path, summary)

    print(
        "object  low  mid  high  total",
        flush=True,
    )
    for row in summary["by_object"]:
        bins = row["visibility_bins"]
        print(
            f"{row['object_id']:>6}  {bins['low']:>3}  {bins['mid']:>3}  "
            f"{bins['high']:>4}  {row['sample_count']:>5}",
            flush=True,
        )
    totals = summary["visibility_bins"]
    print(
        f" total  {totals['low']:>3}  {totals['mid']:>3}  "
        f"{totals['high']:>4}  {summary['sample_count']:>5}",
        flush=True,
    )
    print(
        "valid-depth ratio (recorded, never filtered): "
        f"min={summary['valid_depth_ratio_inside_mask']['minimum']:.6f}, "
        f"median={summary['valid_depth_ratio_inside_mask']['median']:.6f}, "
        f"max={summary['valid_depth_ratio_inside_mask']['maximum']:.6f}",
        flush=True,
    )
    print(f"saved: {output_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
