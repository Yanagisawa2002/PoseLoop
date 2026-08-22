#!/usr/bin/env python3
"""Build the one-open YCB-V sealed replay for PoseLoop M5-R6A-S1."""

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

from build_m5_r5_lmo_development import (
    _consecutive_segments,
    lmo_bop_pose_m,
    missing_gap_frame_counts,
    select_best_window,
    summarize_lmo_rotation_adapter,
    summarize_lmo_visible_mask_metadata,
)
from m1_common import (
    bop_pose_m,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "poseloop-m5-r6a-ycbv-sealed-v1"
STAGE_ID = "M5-R6A-S1"
RECORD_PREFIX = "ycbv-test"
SENSOR_MODALITY = "ycbv_rgbd"
VISIBLE_MASK_COUNT_TOLERANCE_PIXELS = 2


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
    mask_metadata_delta_pixels: int
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
            f"g{self.gt_instance_index:06d}-o{self.object_id:06d}"
        )

    @property
    def track_id(self) -> str:
        return f"{RECORD_PREFIX}-s{self.scene_id:06d}-o{self.object_id:06d}"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_YCBV_ROOT", "/home/cgliu/datasets/ycbv")),
    )
    parser.add_argument(
        "--archive-root", type=Path, default=Path("/home/cgliu/datasets/archives/ycbv")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root
        / "protocols"
        / "poseloop_m5_r6a_ycbv_sealed_protocol.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "m5_r6a_ycbv_sealed",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r6a_ycbv_input_audit.md",
    )
    return parser.parse_args()


def validate_visible_mask_count(actual: int, declared: int, path: Path) -> int:
    delta = actual - declared
    if abs(delta) > VISIBLE_MASK_COUNT_TOLERANCE_PIXELS:
        raise ValueError(
            "YCB-V visible-mask PNG/metadata mismatch beyond frozen two-pixel "
            f"tolerance: {path} actual={actual} declared={declared}"
        )
    return delta


def sealed_bop_pose_m(entry: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    projected_pose, audit = lmo_bop_pose_m(entry)
    try:
        strict_pose = bop_pose_m(dict(entry))
    except (AssertionError, ValueError):
        result = projected_pose
        adapter_applied = True
    else:
        result = strict_pose
        adapter_applied = False
    return result, {**audit, "conditional_projection_applied": adapter_applied}


def _frame_support(
    scene_dir: Path,
    image_id: int,
    gt_index: int,
    info_entry: Mapping[str, Any],
    camera_entry: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
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
            raise ValueError(f"Cannot decode YCB-V depth: {depth_path}")
        expected_shape = (
            int(selection["image_height"]),
            int(selection["image_width"]),
        )
        if depth.shape != expected_shape:
            raise ValueError(f"Unexpected YCB-V frame shape: {depth.shape}")
        depth_cache[image_id] = np.asarray(depth)
    raw_depth = depth_cache[image_id]
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"Cannot decode YCB-V visible mask: {mask_path}")
    if mask_image.ndim == 3:
        mask_image = mask_image[..., 0]
    mask = np.asarray(mask_image) > 0
    if mask.shape != raw_depth.shape:
        raise ValueError(f"YCB-V mask/depth shape mismatch: {mask_path}")
    mask_pixels = int(np.count_nonzero(mask))
    mask_delta = validate_visible_mask_count(
        mask_pixels, int(info_entry["px_count_visib"]), mask_path
    )
    depth_mm = raw_depth.astype(np.float64) * float(camera_entry["depth_scale"])
    valid = mask & np.isfinite(depth_mm) & (depth_mm > 0)
    valid_depth_pixels = int(np.count_nonzero(valid))
    valid_depth_ratio = valid_depth_pixels / max(mask_pixels, 1)
    mask_area_fraction = mask_pixels / float(raw_depth.shape[0] * raw_depth.shape[1])
    available = bool(
        mask_area_fraction >= float(selection["minimum_mask_area_fraction"])
        and valid_depth_ratio >= float(selection["minimum_valid_depth_ratio"])
        and valid_depth_pixels >= int(selection["minimum_valid_depth_pixels"])
    )
    median_depth_m = (
        float(np.median(depth_mm[valid])) * 0.001 if valid_depth_pixels else 1.0
    )
    return {
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "mask_path": mask_path,
        "mask_pixels": mask_pixels,
        "mask_metadata_delta_pixels": mask_delta,
        "valid_depth_pixels": valid_depth_pixels,
        "valid_depth_ratio": valid_depth_ratio,
        "mask_area_fraction": mask_area_fraction,
        "median_depth_m": median_depth_m,
        "available": available,
    }


def discover_scene_directories(dataset_root: Path) -> list[Path]:
    test_root = dataset_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    scenes = sorted(
        (path for path in test_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not scenes:
        raise ValueError("YCB-V sealed archive contains no numeric test scenes")
    return scenes


def load_target_rows(
    dataset_root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[int, list[FrameObservation]], dict[str, Any], list[int]]:
    object_ids = {int(value) for value in protocol["source"]["object_ids"]}
    tracks = {object_id: [] for object_id in sorted(object_ids)}
    rotation_rows: list[dict[str, Any]] = []
    mask_deltas: list[int] = []
    scene_directories = discover_scene_directories(dataset_root)
    for scene_dir in scene_directories:
        scene_id = int(scene_dir.name)
        camera = json.loads((scene_dir / "scene_camera.json").read_text(encoding="utf-8"))
        ground_truth = json.loads((scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
        information = json.loads(
            (scene_dir / "scene_gt_info.json").read_text(encoding="utf-8")
        )
        if set(camera) != set(ground_truth) or set(camera) != set(information):
            raise ValueError(f"YCB-V camera/GT/info keys differ: scene={scene_id}")
        depth_cache: dict[int, np.ndarray] = {}
        for key in sorted(camera, key=int):
            image_id = int(key)
            gt_entries = ground_truth[key]
            info_entries = information[key]
            if len(gt_entries) != len(info_entries):
                raise ValueError(
                    f"YCB-V GT/info length mismatch: scene={scene_id} image={image_id}"
                )
            target_counts = Counter(
                int(entry["obj_id"])
                for entry in gt_entries
                if int(entry["obj_id"]) in object_ids
            )
            duplicates = [object_id for object_id, count in target_counts.items() if count > 1]
            if duplicates:
                raise ValueError(
                    f"YCB-V duplicate target instances: scene={scene_id} "
                    f"image={image_id} objects={duplicates}"
                )
            for gt_index, gt_entry in enumerate(gt_entries):
                object_id = int(gt_entry["obj_id"])
                if object_id not in object_ids:
                    continue
                pose, rotation_audit = sealed_bop_pose_m(gt_entry)
                support = _frame_support(
                    scene_dir,
                    image_id,
                    gt_index,
                    info_entries[gt_index],
                    camera[key],
                    selection=protocol["selection"],
                    depth_cache=depth_cache,
                )
                tracks[object_id].append(
                    FrameObservation(
                        scene_id=scene_id,
                        image_id=image_id,
                        gt_instance_index=gt_index,
                        object_id=object_id,
                        gt_pose_m=pose,
                        camera_entry=camera[key],
                        visible_fraction=float(info_entries[gt_index]["visib_fract"]),
                        mask_pixels=int(support["mask_pixels"]),
                        mask_metadata_delta_pixels=int(
                            support["mask_metadata_delta_pixels"]
                        ),
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
                rotation_rows.append(rotation_audit)
                mask_deltas.append(int(support["mask_metadata_delta_pixels"]))
    rotation_summary = summarize_lmo_rotation_adapter(rotation_rows)
    rotation_summary["conditional_projection_count"] = sum(
        bool(row["conditional_projection_applied"]) for row in rotation_rows
    )
    return (
        tracks,
        {
            "gt_rotation": rotation_summary,
            "visible_mask_metadata": summarize_lmo_visible_mask_metadata(mask_deltas),
        },
        [int(path.name) for path in scene_directories],
    )


def select_object_window(
    rows: Sequence[FrameObservation], selection: Mapping[str, Any]
) -> list[FrameObservation] | None:
    length = int(selection["window_length_frames"])
    eligible = selection["eligible_window"]
    by_scene: dict[int, list[FrameObservation]] = {}
    for row in rows:
        by_scene.setdefault(row.scene_id, []).append(row)
    candidates: list[tuple[int, int, int, list[FrameObservation]]] = []
    for scene_id, scene_rows in sorted(by_scene.items()):
        for segment in _consecutive_segments(scene_rows):
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
            candidates.append(
                (
                    -sum(not row.available for row in selected),
                    scene_id,
                    selected[0].image_id,
                    selected,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _inference_row(observation: FrameObservation, dataset_root: Path) -> dict[str, Any]:
    camera_matrix = np.asarray(
        observation.camera_entry["cam_K"], dtype=np.float64
    ).reshape(3, 3)
    unit_check_pose = np.eye(4, dtype=np.float64)
    unit_check_pose[2, 3] = observation.median_depth_m
    return {
        "record_type": "m5_r5_development_inference_sample",
        "schema_version": SCHEMA_VERSION,
        "sample_id": observation.sample_id,
        "dataset": "ycbv",
        "dataset_split": "test",
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": SENSOR_MODALITY,
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
    rotation = summary["source_adapter_audit"]["gt_rotation"]
    mask = summary["source_adapter_audit"]["visible_mask_metadata"]
    lines = [
        "# PoseLoop M5-R6A-S1 YCB-V sealed input audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "The deterministic sealed builder selected tracks without reading any FoundationPose prediction or pose-error outcome. This first label open is final under the frozen protocol.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Discovered test scenes | {len(summary['scene_ids'])} |",
        f"| Selected tracks/objects | {summary['track_count']} |",
        f"| Replay frames | {summary['replay_frame_count']} |",
        f"| Inference frames | {summary['inference_sample_count']} |",
        f"| Natural missing frames | {summary['natural_missing_frame_count']} |",
    ]
    for name, count in summary["missing_gap_frame_counts"].items():
        lines.append(f"| {name} frames | {count} |")
    lines.extend(["", "| Object | Scene | Start | Available | Missing |", "|---:|---:|---:|---:|---:|"])
    for object_id, row in sorted(summary["by_object"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {object_id} | {row['scene_id']} | {row['source_start_image_id']} | "
            f"{row['available_frame_count']} | {row['natural_missing_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "Source integrity:",
            "",
            f"- sealed archive SHA-256: `{source['sealed']['sha256']}`;",
            f"- conditional SO(3) projections: {rotation['conditional_projection_count']};",
            f"- maximum visible-mask metadata delta: {mask['maximum_absolute_delta_pixels']} pixels;",
            "- availability uses decoded PNG masks;",
            "- no artificial frame deletion or candidate reselection was used.",
            "",
            "This is an input audit, not the sealed method outcome.",
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
        raise FileExistsError(f"YCB-V sealed output already exists: {output_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected YCB-V sealed protocol")

    development = protocol["development_freeze"]
    development_result_path = repo_root / str(development["result_path"])
    if sha256_file(development_result_path) != str(development["result_sha256"]):
        raise RuntimeError("M5-R6A development result hash mismatch")
    development_result = json.loads(development_result_path.read_text(encoding="utf-8"))
    if (
        development_result.get("status") != "PASS_M5_R6A_DEVELOPMENT"
        or development_result.get("modal_candidate_id") != "measurement_first"
        or development_result.get("fold_winner_counts") != {"measurement_first": 14}
        or bool(development_result.get("sealed_archive_or_label_read"))
    ):
        raise RuntimeError("M5-R6A development result does not authorize sealed open")

    source_receipt: dict[str, dict[str, Any]] = {}
    for key in ("base", "models", "sealed"):
        expected = protocol["source"]["archives"][key]
        path = archive_root / str(expected["filename"])
        if not path.is_file() or path.stat().st_size != int(expected["size_bytes"]):
            raise FileNotFoundError(f"Missing or wrong-size YCB-V archive: {path}")
        digest = sha256_file(path)
        if digest != str(expected["sha256"]):
            raise RuntimeError(f"YCB-V archive hash mismatch: {path}")
        source_receipt[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    for path in (
        dataset_root / "models" / "models_info.json",
        dataset_root / "models_eval" / "models_info.json",
        dataset_root / "test",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Incomplete YCB-V sealed dataset: {path}")

    rows_by_object, source_adapter_audit, scene_ids = load_target_rows(
        dataset_root, protocol
    )
    selection = protocol["selection"]
    selected = {
        object_id: window
        for object_id, rows in rows_by_object.items()
        if (window := select_object_window(rows, selection)) is not None
    }
    gate = protocol["evaluation"]["sealed_gate"]
    if len(selected) < int(gate["represented_object_count_min"]):
        raise RuntimeError(
            f"YCB-V sealed input has too few eligible objects: {len(selected)}"
        )

    inference_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    by_object: dict[str, dict[str, int]] = {}
    gap_counts = Counter()
    for object_id, rows in sorted(selected.items()):
        availability = [row.available for row in rows]
        gap_counts.update(missing_gap_frame_counts(availability))
        by_object[str(object_id)] = {
            "scene_id": rows[0].scene_id,
            "source_start_image_id": rows[0].image_id,
            "replay_frame_count": len(rows),
            "available_frame_count": sum(availability),
            "natural_missing_frame_count": sum(not value for value in availability),
        }
        frames: list[dict[str, Any]] = []
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
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "SEALED_BUNDLE_READY",
        "dataset": "YCB-V",
        "sensor_modality": SENSOR_MODALITY,
        "scene_ids": scene_ids,
        "track_count": len(track_rows),
        "object_count": len(by_object),
        "missing_represented_object_count": sum(
            row["natural_missing_frame_count"] > 0 for row in by_object.values()
        ),
        "replay_frame_count": sum(int(row["frame_count"]) for row in track_rows),
        "inference_sample_count": len(inference_rows),
        "natural_missing_frame_count": sum(
            int(row["natural_missing_frame_count"]) for row in by_object.values()
        ),
        "missing_gap_frame_counts": dict(sorted(gap_counts.items())),
        "by_object": by_object,
        "source_adapter_audit": source_adapter_audit,
        "selection_reads_method_prediction_or_pose_error": False,
        "artificial_frame_deletion": False,
        "sealed_archive_or_label_read": True,
        "sealed_candidate_id": "measurement_first",
    }
    if summary["track_count"] < int(gate["track_count_min"]):
        raise RuntimeError("YCB-V sealed input has too few tracks")
    if summary["missing_represented_object_count"] < int(
        gate["missing_represented_object_count_min"]
    ):
        raise RuntimeError("YCB-V sealed input has too few missing-represented objects")
    if summary["natural_missing_frame_count"] < int(
        gate["natural_missing_frame_count_min"]
    ):
        raise RuntimeError("YCB-V sealed input has too few natural missing frames")
    for name, minimum in selection["missing_gap_frame_minimums"].items():
        if int(summary["missing_gap_frame_counts"].get(name, 0)) < int(minimum):
            raise RuntimeError(f"YCB-V sealed missing-gap stratum too small: {name}")

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
        "core": repo_root / "scripts" / "m5_r6_core.py",
        "inference": repo_root / "scripts" / "run_m5_r6a_ycbv_inference.py",
        "inference_shared": repo_root / "scripts" / "run_m5_r5_inference.py",
        "evaluator": repo_root / "scripts" / "evaluate_m5_r6a_ycbv_sealed.py",
        "development_evaluation_core": repo_root
        / "scripts"
        / "evaluate_m5_r5_development.py",
        "legacy_core": repo_root / "scripts" / "m5_g0_core.py",
        "metric_core": repo_root / "scripts" / "m5_g0_metrics.py",
        "bootstrap_core": repo_root / "scripts" / "evaluate_m5_r2_once.py",
        "io_common": repo_root / "scripts" / "m1_common.py",
        "source_adapter": repo_root / "scripts" / "build_m5_r5_lmo_development.py",
        "foundationpose_runner": repo_root / "scripts" / "run_r1_sealed_inference.py",
        "foundationpose_batch": repo_root / "scripts" / "run_xyzibd_batch.py",
        "foundationpose_smoke": repo_root / "scripts" / "run_xyzibd_smoke.py",
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"YCB-V sealed freeze code missing: {path}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "sealed_labels_opened_bundle_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_archive_or_label_read": True,
        "sealed_labels_opened": True,
        "evaluation_invocation_count": 0,
        "candidate_id": "measurement_first",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "development_result": {
            "path": str(development_result_path),
            "sha256": sha256_file(development_result_path),
        },
        "source_archives": source_receipt,
        "dataset": {
            "root": str(dataset_root),
            "models_info": {
                "path": str(dataset_root / "models_eval" / "models_info.json"),
                "sha256": sha256_file(dataset_root / "models_eval" / "models_info.json"),
            },
        },
        "code": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in code_paths.items()
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
