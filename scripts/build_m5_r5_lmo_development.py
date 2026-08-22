#!/usr/bin/env python3
"""Build the frozen LM-O development replay for PoseLoop M5-R5."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from m1_common import (
    GT_ROTATION_ATOL,
    assert_pose,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "poseloop-m5-r5-v1"
RECORD_PREFIX = "lmo-test"
LMO_VISIBLE_MASK_COUNT_TOLERANCE_PIXELS = 2
LMO_RAW_ROTATION_ORTHOGONALITY_INF_MAX = 0.02
LMO_RAW_ROTATION_SINGULAR_VALUE_MIN = 0.99
LMO_RAW_ROTATION_SINGULAR_VALUE_MAX = 1.01
LMO_ROTATION_PROJECTION_MAX_ABS = 0.01
LMO_ROTATION_PROJECTION_FROBENIUS_MAX = 0.02


@dataclass(frozen=True, slots=True)
class FrameObservation:
    scene_id: int
    image_id: int
    gt_instance_index: int
    object_id: int
    gt_pose_m: np.ndarray
    camera_entry: Mapping[str, Any]
    visible_fraction: float
    mask_pixels: int
    valid_depth_pixels: int
    valid_depth_ratio: float
    mask_area_fraction: float
    median_depth_m: float
    available: bool
    rgb_path: Path
    depth_path: Path
    mask_path: Path

    @property
    def sample_id(self) -> str:
        return (
            f"{RECORD_PREFIX}-s{self.scene_id:06d}-i{self.image_id:06d}-"
            f"o{self.object_id:06d}"
        )

    @property
    def track_id(self) -> str:
        return f"{RECORD_PREFIX}-s{self.scene_id:06d}-o{self.object_id:06d}"


def lmo_bop_pose_m(
    gt_entry: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert legacy LM-O labels to a rigid pose with bounded, audited projection.

    The archived LM-O labels contain object-dependent scale/shear in fields named
    ``cam_R_m2c``.  A pose consumer requires an SO(3) rotation, so this adapter
    maps every bounded, orientation-preserving matrix to its closest proper
    rotation in Frobenius norm.  It deliberately does not relax the shared BOP
    parser used by other stages.
    """

    raw_rotation = np.asarray(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    translation_m = (
        np.asarray(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3) * 0.001
    )
    if not np.isfinite(raw_rotation).all() or not np.isfinite(translation_m).all():
        raise ValueError("LM-O GT pose contains a non-finite value")

    raw_orthogonality_inf = float(
        np.max(np.abs(raw_rotation.T @ raw_rotation - np.eye(3)))
    )
    raw_determinant = float(np.linalg.det(raw_rotation))
    left, singular_values, right_t = np.linalg.svd(raw_rotation)
    singular_min = float(np.min(singular_values))
    singular_max = float(np.max(singular_values))
    if raw_determinant <= 0:
        raise ValueError("LM-O raw rotation is not orientation preserving")
    if raw_orthogonality_inf > LMO_RAW_ROTATION_ORTHOGONALITY_INF_MAX:
        raise ValueError(
            "LM-O raw rotation exceeds the frozen orthogonality adapter bound: "
            f"{raw_orthogonality_inf}"
        )
    if (
        singular_min < LMO_RAW_ROTATION_SINGULAR_VALUE_MIN
        or singular_max > LMO_RAW_ROTATION_SINGULAR_VALUE_MAX
    ):
        raise ValueError(
            "LM-O raw rotation exceeds the frozen singular-value adapter bounds: "
            f"min={singular_min} max={singular_max}"
        )

    determinant_correction = np.eye(3, dtype=np.float64)
    determinant_correction[-1, -1] = float(np.linalg.det(left @ right_t))
    projected_rotation = left @ determinant_correction @ right_t
    correction = projected_rotation - raw_rotation
    projection_max_abs = float(np.max(np.abs(correction)))
    projection_frobenius = float(np.linalg.norm(correction, ord="fro"))
    if projection_max_abs > LMO_ROTATION_PROJECTION_MAX_ABS:
        raise ValueError(
            "LM-O rotation projection exceeds the frozen elementwise bound: "
            f"{projection_max_abs}"
        )
    if projection_frobenius > LMO_ROTATION_PROJECTION_FROBENIUS_MAX:
        raise ValueError(
            "LM-O rotation projection exceeds the frozen Frobenius bound: "
            f"{projection_frobenius}"
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = projected_rotation
    transform[:3, 3] = translation_m
    assert_pose(transform, "projected LM-O GT pose", rotation_atol=1e-10)
    return transform, {
        "raw_orthogonality_inf": raw_orthogonality_inf,
        "raw_determinant": raw_determinant,
        "raw_abs_determinant_minus_one": abs(raw_determinant - 1.0),
        "raw_singular_value_min": singular_min,
        "raw_singular_value_max": singular_max,
        "projection_max_abs": projection_max_abs,
        "projection_frobenius": projection_frobenius,
    }


def summarize_lmo_rotation_adapter(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("LM-O rotation adapter audit is empty")

    def maximum(name: str) -> float:
        return float(max(float(row[name]) for row in rows))

    def minimum(name: str) -> float:
        return float(min(float(row[name]) for row in rows))

    return {
        "method": "closest_proper_rotation_svd",
        "entry_count": len(rows),
        "projected_entry_count": len(rows),
        "legacy_rotation_tolerance_exceedance_count": sum(
            float(row["raw_orthogonality_inf"]) > GT_ROTATION_ATOL for row in rows
        ),
        "raw_orthogonality_inf_max": maximum("raw_orthogonality_inf"),
        "raw_abs_determinant_minus_one_max": maximum(
            "raw_abs_determinant_minus_one"
        ),
        "raw_determinant_min": minimum("raw_determinant"),
        "raw_determinant_max": maximum("raw_determinant"),
        "raw_singular_value_min": minimum("raw_singular_value_min"),
        "raw_singular_value_max": maximum("raw_singular_value_max"),
        "projection_max_abs_max": maximum("projection_max_abs"),
        "projection_frobenius_max": maximum("projection_frobenius"),
        "bounds": {
            "raw_orthogonality_inf_max": LMO_RAW_ROTATION_ORTHOGONALITY_INF_MAX,
            "raw_singular_value_min": LMO_RAW_ROTATION_SINGULAR_VALUE_MIN,
            "raw_singular_value_max": LMO_RAW_ROTATION_SINGULAR_VALUE_MAX,
            "projection_max_abs": LMO_ROTATION_PROJECTION_MAX_ABS,
            "projection_frobenius_max": LMO_ROTATION_PROJECTION_FROBENIUS_MAX,
        },
        "selection_uses_adapter_magnitude": False,
        "translation_modified": False,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_LMO_ROOT", "/home/cgliu/datasets/lmo")),
    )
    parser.add_argument(
        "--archive-root", type=Path, default=Path("/home/cgliu/datasets/archives/lmo")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r5_protocol.json",
    )
    parser.add_argument("--expected-protocol-id", default=PROTOCOL_ID)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "m5_r5",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r5_input_audit.md",
    )
    return parser.parse_args()


def select_best_window(
    availability: Sequence[bool],
    *,
    window_length: int,
    minimum_available: int,
    minimum_missing: int,
) -> tuple[int, int] | None:
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    candidates: list[tuple[int, int, int]] = []
    for start in range(0, len(availability) - window_length + 1):
        stop = start + window_length
        window = availability[start:stop]
        available = sum(bool(value) for value in window)
        missing = window_length - available
        if (
            bool(window[0])
            and available >= minimum_available
            and missing >= minimum_missing
        ):
            candidates.append((-missing, start, stop))
    if not candidates:
        return None
    _, start, stop = min(candidates)
    return start, stop


def missing_gap_frame_counts(availability: Sequence[bool]) -> dict[str, int]:
    counts = {"short_gap_1_2": 0, "medium_gap_3_7": 0, "long_gap_8_plus": 0}
    index = 0
    while index < len(availability):
        if availability[index]:
            index += 1
            continue
        start = index
        while index < len(availability) and not availability[index]:
            index += 1
        length = index - start
        if length <= 2:
            counts["short_gap_1_2"] += length
        elif length <= 7:
            counts["medium_gap_3_7"] += length
        else:
            counts["long_gap_8_plus"] += length
    return counts


def validate_visible_mask_count(actual: int, declared: int, path: Path) -> int:
    delta = actual - declared
    if abs(delta) > LMO_VISIBLE_MASK_COUNT_TOLERANCE_PIXELS:
        raise ValueError(
            "LM-O visible mask count mismatch beyond two-pixel archive tolerance: "
            f"{path} actual={actual} declared={declared}"
        )
    return delta


def summarize_lmo_visible_mask_metadata(deltas: Sequence[int]) -> dict[str, Any]:
    if not deltas:
        raise ValueError("LM-O visible-mask metadata audit is empty")
    counts = Counter(int(delta) for delta in deltas)
    return {
        "entry_count": len(deltas),
        "exact_match_count": int(counts.get(0, 0)),
        "nonzero_delta_count": sum(int(delta) != 0 for delta in deltas),
        "maximum_absolute_delta_pixels": max(abs(int(delta)) for delta in deltas),
        "signed_delta_counts": {
            str(delta): int(count) for delta, count in sorted(counts.items())
        },
        "allowed_absolute_delta_pixels": LMO_VISIBLE_MASK_COUNT_TOLERANCE_PIXELS,
        "availability_uses_actual_png_count": True,
    }


def _frame_support(
    scene_dir: Path,
    image_id: int,
    gt_index: int,
    info_entry: Mapping[str, Any],
    camera_entry: Mapping[str, Any],
    *,
    minimum_mask_fraction: float,
    minimum_depth_ratio: float,
    minimum_valid_depth_pixels: int,
    expected_height: int,
    expected_width: int,
    depth_cache: dict[int, np.ndarray],
) -> dict[str, Any]:
    rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
    depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
    mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{gt_index:06d}.png"
    for path in (rgb_path, depth_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if image_id not in depth_cache:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise ValueError(f"Cannot decode LM-O depth: {depth_path}")
        if depth.shape != (expected_height, expected_width):
            raise ValueError(f"Unexpected LM-O frame shape: {depth.shape}")
        depth_cache[image_id] = np.asarray(depth)
    raw_depth = depth_cache[image_id]
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"Cannot decode LM-O mask: {mask_path}")
    if mask_image.ndim == 3:
        mask_image = mask_image[..., 0]
    mask = np.asarray(mask_image) > 0
    if mask.shape != raw_depth.shape:
        raise ValueError(f"LM-O mask/depth mismatch: {mask_path}")
    mask_pixels = int(np.count_nonzero(mask))
    mask_metadata_delta_pixels = validate_visible_mask_count(
        mask_pixels, int(info_entry["px_count_visib"]), mask_path
    )
    depth_scale = float(camera_entry["depth_scale"])
    depth_mm = raw_depth.astype(np.float64) * depth_scale
    valid = mask & np.isfinite(depth_mm) & (depth_mm > 0)
    valid_depth_pixels = int(np.count_nonzero(valid))
    valid_depth_ratio = valid_depth_pixels / max(mask_pixels, 1)
    mask_area_fraction = mask_pixels / float(expected_height * expected_width)
    available = bool(
        mask_area_fraction >= minimum_mask_fraction
        and valid_depth_ratio >= minimum_depth_ratio
        and valid_depth_pixels >= minimum_valid_depth_pixels
    )
    median_depth_m = (
        float(np.median(depth_mm[valid])) * 0.001 if valid_depth_pixels else 1.0
    )
    return {
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "mask_path": mask_path,
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": valid_depth_pixels,
        "valid_depth_ratio": valid_depth_ratio,
        "mask_area_fraction": mask_area_fraction,
        "median_depth_m": median_depth_m,
        "available": available,
        "mask_metadata_delta_pixels": mask_metadata_delta_pixels,
    }


def load_object_rows(
    dataset_root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[int, list[FrameObservation]], dict[str, Any]]:
    split = protocol["splits"]["development"]
    selection = protocol["selection"]
    scene_id = int(split["scene_id"])
    scene_dir = dataset_root / str(split["directory"]) / f"{scene_id:06d}"
    camera = json.loads((scene_dir / "scene_camera.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    information = json.loads(
        (scene_dir / "scene_gt_info.json").read_text(encoding="utf-8")
    )
    if set(camera) != set(ground_truth) or set(camera) != set(information):
        raise ValueError("LM-O camera/GT/info frame keys differ")
    expected_ids = set(map(int, protocol["source"]["development_object_ids"]))
    tracks: dict[int, list[FrameObservation]] = {object_id: [] for object_id in expected_ids}
    depth_cache: dict[int, np.ndarray] = {}
    rotation_adapter_rows: list[dict[str, float]] = []
    mask_metadata_deltas: list[int] = []
    for key in sorted(camera, key=int):
        image_id = int(key)
        gt_entries = ground_truth[key]
        info_entries = information[key]
        if len(gt_entries) != len(info_entries):
            raise ValueError(f"LM-O GT/info length mismatch: image={image_id}")
        object_ids = [int(entry["obj_id"]) for entry in gt_entries]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"LM-O duplicate object instance: image={image_id}")
        if not set(object_ids).issubset(expected_ids):
            raise ValueError(f"Unexpected LM-O object ID: image={image_id}")
        for gt_index, (gt_entry, info_entry) in enumerate(
            zip(gt_entries, info_entries, strict=True)
        ):
            object_id = int(gt_entry["obj_id"])
            gt_pose_m, rotation_adapter_row = lmo_bop_pose_m(gt_entry)
            rotation_adapter_rows.append(rotation_adapter_row)
            support = _frame_support(
                scene_dir,
                image_id,
                gt_index,
                info_entry,
                camera[key],
                minimum_mask_fraction=float(selection["minimum_mask_area_fraction"]),
                minimum_depth_ratio=float(selection["minimum_valid_depth_ratio"]),
                minimum_valid_depth_pixels=int(
                    selection["minimum_valid_depth_pixels"]
                ),
                expected_height=int(selection["image_height"]),
                expected_width=int(selection["image_width"]),
                depth_cache=depth_cache,
            )
            mask_metadata_deltas.append(int(support["mask_metadata_delta_pixels"]))
            tracks[object_id].append(
                FrameObservation(
                    scene_id=scene_id,
                    image_id=image_id,
                    gt_instance_index=gt_index,
                    object_id=object_id,
                    gt_pose_m=gt_pose_m,
                    camera_entry=camera[key],
                    visible_fraction=float(info_entry["visib_fract"]),
                    mask_pixels=int(support["mask_pixels"]),
                    valid_depth_pixels=int(support["valid_depth_pixels"]),
                    valid_depth_ratio=float(support["valid_depth_ratio"]),
                    mask_area_fraction=float(support["mask_area_fraction"]),
                    median_depth_m=float(support["median_depth_m"]),
                    available=bool(support["available"]),
                    rgb_path=Path(support["rgb_path"]),
                    depth_path=Path(support["depth_path"]),
                    mask_path=Path(support["mask_path"]),
                )
            )
    if any(not rows for rows in tracks.values()):
        raise ValueError("LM-O development object has no GT-present frames")
    source_adapter_audit = {
        "gt_rotation": summarize_lmo_rotation_adapter(rotation_adapter_rows),
        "visible_mask_metadata": summarize_lmo_visible_mask_metadata(
            mask_metadata_deltas
        ),
    }
    return tracks, source_adapter_audit


def _consecutive_segments(rows: Sequence[FrameObservation]) -> list[list[FrameObservation]]:
    ordered = sorted(rows, key=lambda row: row.image_id)
    segments: list[list[FrameObservation]] = []
    for row in ordered:
        if not segments or row.image_id != segments[-1][-1].image_id + 1:
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def select_object_window(
    rows: Sequence[FrameObservation], selection: Mapping[str, Any]
) -> list[FrameObservation] | None:
    length = int(selection["window_length_frames"])
    eligible = selection["eligible_window"]
    candidates: list[tuple[int, int, list[FrameObservation]]] = []
    for segment in _consecutive_segments(rows):
        window = select_best_window(
            [row.available for row in segment],
            window_length=length,
            minimum_available=int(eligible["available_frame_count_min"]),
            minimum_missing=int(eligible["natural_missing_frame_count_min"]),
        )
        if window is None:
            continue
        start, stop = window
        selected = segment[start:stop]
        missing = sum(not row.available for row in selected)
        candidates.append((-missing, selected[0].image_id, selected))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _inference_row(observation: FrameObservation, dataset_root: Path) -> dict[str, Any]:
    camera_matrix = np.asarray(observation.camera_entry["cam_K"], dtype=np.float64).reshape(3, 3)
    unit_check_pose = np.eye(4, dtype=np.float64)
    unit_check_pose[2, 3] = observation.median_depth_m
    return {
        "record_type": "m5_r5_development_inference_sample",
        "schema_version": SCHEMA_VERSION,
        "sample_id": observation.sample_id,
        "dataset": "lmo",
        "dataset_split": "test",
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": "linemod_rgbd",
        "visible_mask_pixel_count": observation.mask_pixels,
        "valid_depth_pixel_count": observation.valid_depth_pixels,
        "input_mask_area_fraction": observation.mask_area_fraction,
        "valid_depth_ratio_inside_mask": observation.valid_depth_ratio,
        "image_height": 480,
        "image_width": 640,
        "rgb_path": str(observation.rgb_path.resolve()),
        "depth_path": str(observation.depth_path.resolve()),
        "mask_path": str(observation.mask_path.resolve()),
        "model_path": str(
            (dataset_root / "models" / f"obj_{observation.object_id:06d}.ply").resolve()
        ),
        "camera_intrinsics_row_major": camera_matrix.tolist(),
        "raw_depth_scale": float(observation.camera_entry["depth_scale"]),
        "input_unit_check_pose_m": unit_check_pose.tolist(),
        "input_available": True,
        "evaluator_label_read": False,
    }


def render_report(summary: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    adapter = summary["source_adapter_audit"]["gt_rotation"]
    mask_audit = summary["source_adapter_audit"]["visible_mask_metadata"]
    lines = [
        f"# PoseLoop {summary['stage_id']} LM-O development input audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This bundle uses only GT-present LM-O frames. Natural missing frames are caused by the frozen visible-mask/depth support rule; frames without an evaluable GT pose are excluded rather than counted as dropout. No artificial deletion was used, and no YCB-V sealed archive was downloaded or read.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Selected tracks/objects | {summary['track_count']} |",
        f"| Replay frames | {summary['replay_frame_count']} |",
        f"| Inference frames | {summary['inference_sample_count']} |",
        f"| Natural missing frames | {summary['natural_missing_frame_count']} |",
    ]
    for name, count in summary["missing_gap_frame_counts"].items():
        lines.append(f"| {name} frames | {count} |")
    lines.extend(
        [
            "",
            "## Legacy LM-O rigid-pose adapter",
            "",
            "The initially frozen builder stopped before window selection because legacy LM-O `cam_R_m2c` values are not all valid SO(3) rotations. The source-only adapter was added after that stop and before any inference or pose-error outcome. It projects every bounded, orientation-preserving matrix to the closest proper rotation by SVD; translation, selection, candidates, metrics, and gates are unchanged.",
            "",
            "| Adapter audit | Value |",
            "|---|---:|",
            f"| Archived GT entries checked/projected | {adapter['entry_count']} |",
            f"| Entries beyond the shared parser tolerance | {adapter['legacy_rotation_tolerance_exceedance_count']} |",
            f"| Maximum raw orthogonality error | {adapter['raw_orthogonality_inf_max']:.12g} |",
            f"| Raw determinant range | {adapter['raw_determinant_min']:.12g} to {adapter['raw_determinant_max']:.12g} |",
            f"| Raw singular-value range | {adapter['raw_singular_value_min']:.12g} to {adapter['raw_singular_value_max']:.12g} |",
            f"| Maximum elementwise projection correction | {adapter['projection_max_abs_max']:.12g} |",
            f"| Maximum Frobenius projection correction | {adapter['projection_frobenius_max']:.12g} |",
            f"| Visible-mask metadata exact matches | {mask_audit['exact_match_count']} / {mask_audit['entry_count']} |",
            f"| Visible-mask metadata nonzero deltas | {mask_audit['nonzero_delta_count']} |",
            f"| Maximum visible-mask metadata delta | {mask_audit['maximum_absolute_delta_pixels']} pixels |",
            "",
            "Availability is always computed from the decoded PNG mask; archived `px_count_visib` is only a bounded source-integrity cross-check.",
            "",
            "| Object | Window start | Available | Missing |",
            "|---:|---:|---:|---:|",
        ]
    )
    for object_id, row in sorted(summary["by_object"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {object_id} | {row['source_start_image_id']} | "
            f"{row['available_frame_count']} | {row['natural_missing_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "Source archives:",
            "",
            f"- `lmo_base.zip`: `{source['base']['sha256']}`",
            f"- `lmo_models.zip`: `{source['models']['sha256']}`",
            f"- `lmo_test_all.zip`: `{source['development']['sha256']}`",
            "",
            "This audit is not a method outcome.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    archive_root = args.archive_root.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if output_root.exists():
        raise FileExistsError(f"M5-R5 development output already exists: {output_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != args.expected_protocol_id:
        raise ValueError("Unexpected M5-R5 protocol")
    protocol_id = str(protocol["protocol_id"])
    stage_id = str(protocol.get("stage_id", "M5-R5"))

    source_receipt: dict[str, dict[str, Any]] = {}
    for key in ("base", "models", "development"):
        expected = protocol["source"]["archives"][key]
        path = archive_root / str(expected["filename"])
        if not path.is_file() or path.stat().st_size != int(expected["size_bytes"]):
            raise FileNotFoundError(f"Missing or wrong-size LM-O archive: {path}")
        digest = sha256_file(path)
        if digest != str(expected["sha256"]):
            raise RuntimeError(f"LM-O archive hash mismatch: {path}")
        source_receipt[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    required = [
        dataset_root / "models" / "models_info.json",
        dataset_root / "models_eval" / "models_info.json",
        dataset_root / "test" / "000002",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete LM-O development dataset: {missing}")

    rows_by_object, source_adapter_audit = load_object_rows(dataset_root, protocol)
    selection = protocol["selection"]
    selected: dict[int, list[FrameObservation]] = {}
    for object_id, rows in rows_by_object.items():
        window = select_object_window(rows, selection)
        if window is not None:
            selected[object_id] = window
    gate = protocol["evaluation"]["development_gate"]
    if len(selected) < int(gate["represented_object_count_min"]):
        raise RuntimeError(f"M5-R5 selected too few development objects: {len(selected)}")

    inference_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    by_object: dict[str, dict[str, int]] = {}
    aggregate_gap_counts = Counter()
    for object_id, rows in sorted(selected.items()):
        frames: list[dict[str, Any]] = []
        availability = [row.available for row in rows]
        aggregate_gap_counts.update(missing_gap_frame_counts(availability))
        totals = {
            "track_count": 1,
            "replay_frame_count": len(rows),
            "available_frame_count": sum(availability),
            "natural_missing_frame_count": sum(not value for value in availability),
            "source_start_image_id": rows[0].image_id,
        }
        by_object[str(object_id)] = totals
        for replay_index, observation in enumerate(rows):
            if observation.available:
                inference_rows.append(_inference_row(observation, dataset_root))
            evaluator_rows.append(
                {
                    "record_type": "m5_r5_development_evaluator_frame",
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": observation.sample_id,
                    "track_id": observation.track_id,
                    "scene_id": observation.scene_id,
                    "image_id": observation.image_id,
                    "object_id": observation.object_id,
                    "replay_frame_index": replay_index,
                    "gt_model_to_camera_pose_m": observation.gt_pose_m.tolist(),
                    "natural_input_missing": not observation.available,
                }
            )
            frames.append(
                {
                    "sample_id": observation.sample_id,
                    "replay_frame_index": replay_index,
                    "source_frame_ordinal": observation.image_id,
                    "timestamp_s": replay_index / 30.0,
                    "input_available": observation.available,
                    "input_mask_area_fraction": (
                        observation.mask_area_fraction if observation.available else None
                    ),
                    "valid_depth_ratio_inside_mask": (
                        observation.valid_depth_ratio if observation.available else None
                    ),
                }
            )
        track_rows.append(
            {
                "record_type": "m5_r5_development_track",
                "schema_version": SCHEMA_VERSION,
                "track_id": rows[0].track_id,
                "scene_id": rows[0].scene_id,
                "object_id": object_id,
                "frame_count": len(frames),
                "frames": frames,
            }
        )

    inference_rows.sort(key=lambda row: (int(row["object_id"]), str(row["sample_id"])))
    evaluator_rows.sort(key=lambda row: (str(row["track_id"]), int(row["replay_frame_index"])))
    track_rows.sort(key=lambda row: str(row["track_id"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": f"{stage_id} LM-O development input freeze",
        "stage_id": stage_id,
        "status": "DEVELOPMENT_BUNDLE_READY",
        "protocol_id": protocol_id,
        "track_count": len(track_rows),
        "object_count": len(by_object),
        "replay_frame_count": sum(int(row["frame_count"]) for row in track_rows),
        "inference_sample_count": len(inference_rows),
        "natural_missing_frame_count": sum(
            int(row["natural_missing_frame_count"]) for row in by_object.values()
        ),
        "missing_gap_frame_counts": dict(sorted(aggregate_gap_counts.items())),
        "by_object": by_object,
        "selection_reads_pose_or_error_outcome": False,
        "artificial_frame_deletion": False,
        "gt_absent_frames_counted_as_missing": False,
        "sealed_archive_or_label_read": False,
        "source_adapter_audit": source_adapter_audit,
    }
    if summary["track_count"] < int(gate["track_count_min"]):
        raise RuntimeError("M5-R5 selected too few development tracks")
    if summary["natural_missing_frame_count"] < int(
        gate["natural_missing_frame_count_min"]
    ):
        raise RuntimeError("M5-R5 selected too few natural missing frames")
    for name, minimum in protocol["selection"]["missing_gap_frame_minimums"].items():
        if int(summary["missing_gap_frame_counts"].get(name, 0)) < int(minimum):
            raise RuntimeError(f"M5-R5 missing-gap stratum is too small: {name}")

    output_root.mkdir(parents=True, exist_ok=False)
    inference_path = output_root / "inference_manifest.jsonl"
    evaluator_path = output_root / "evaluator_labels.jsonl"
    tracks_path = output_root / "track_manifest.jsonl"
    summary_path = output_root / "selection_summary.json"
    write_jsonl_atomic(inference_path, inference_rows)
    write_jsonl_atomic(evaluator_path, evaluator_rows)
    write_jsonl_atomic(tracks_path, track_rows)
    write_json_atomic(summary_path, summary)

    code_paths = {
        "builder": Path(__file__).resolve(),
        "core": repo_root / "scripts" / "m5_r5_core.py",
        "inference": repo_root / "scripts" / "run_m5_r5_inference.py",
        "evaluator": repo_root / "scripts" / "evaluate_m5_r5_development.py",
        "legacy_core": repo_root / "scripts" / "m5_g0_core.py",
        "foundationpose_runner": repo_root / "scripts" / "run_r1_sealed_inference.py",
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"M5-R5 freeze code missing: {path}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "stage": f"{stage_id} LM-O development bundle",
        "stage_id": stage_id,
        "status": "development_labels_opened_bundle_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_labels_opened": True,
        "sealed_archive_or_label_read": False,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "source_archives": source_receipt,
        "dataset": {
            "root": str(dataset_root),
            "models_info": {
                "path": str(dataset_root / "models_eval" / "models_info.json"),
                "sha256": sha256_file(dataset_root / "models_eval" / "models_info.json"),
            },
        },
        "code": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in code_paths.items()
        },
        "files": {
            "inference_manifest": {
                "path": str(inference_path),
                "sha256": sha256_file(inference_path),
            },
            "evaluator_labels": {
                "path": str(evaluator_path),
                "sha256": sha256_file(evaluator_path),
            },
            "track_manifest": {
                "path": str(tracks_path),
                "sha256": sha256_file(tracks_path),
            },
            "selection_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
        "selection_summary": summary,
    }
    write_json_atomic(output_root / "contract.json", contract)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, source_receipt), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_root / "contract.json")
    print(report_path)


if __name__ == "__main__":
    main()
