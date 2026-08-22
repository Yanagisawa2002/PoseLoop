#!/usr/bin/env python3
"""Build the frozen HB Primesense development replay for PoseLoop M5-R4A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from m1_common import (
    bop_pose_m,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "poseloop-m5-r4a-v1"
RECORD_PREFIX = "hb-val-ps"
VISIBLE_MASK_COUNT_TOLERANCE_PIXELS = 1


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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_HB_ROOT", "/home/cgliu/datasets/hb")),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/cgliu/datasets/archives/hb"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r4a_protocol.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "m5_r4a",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r4a_input_audit.md",
    )
    return parser.parse_args()


def stable_track_key(namespace: str, track_id: str) -> str:
    return hashlib.sha256(f"{namespace}{track_id}".encode("utf-8")).hexdigest()


def select_best_window(
    availability: list[bool], *, window_length: int, minimum_available: int, minimum_missing: int
) -> tuple[int, int] | None:
    """Select the earliest maximally missing full window starting available."""

    if window_length < 1 or len(availability) < window_length:
        return None
    candidates: list[tuple[int, int, int]] = []
    for start in range(0, len(availability) - window_length + 1):
        if not availability[start]:
            continue
        window = availability[start : start + window_length]
        available_count = int(sum(window))
        missing_count = window_length - available_count
        if available_count < minimum_available or missing_count < minimum_missing:
            continue
        candidates.append((-missing_count, start, available_count))
    if not candidates:
        return None
    _negative_missing, start, _available = min(candidates)
    return start, start + window_length


def select_tracks(
    candidates: list[dict[str, Any]],
    *,
    namespace: str,
    track_cap: int,
    per_object_cap: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (stable_track_key(namespace, str(row["track_id"])), str(row["track_id"])),
    )
    selected: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for row in ordered:
        object_id = int(row["object_id"])
        if counts[object_id] >= per_object_cap:
            continue
        selected.append(row)
        counts[object_id] += 1
        if len(selected) >= track_cap:
            break
    return sorted(selected, key=lambda row: str(row["track_id"]))


def validate_visible_mask_count(actual: int, declared: int, path: Path) -> None:
    """Validate HB's stored mask against its metadata without changing the mask."""
    if abs(actual - declared) > VISIBLE_MASK_COUNT_TOLERANCE_PIXELS:
        raise ValueError(
            "HB visible mask count mismatch beyond one-pixel archive tolerance: "
            f"{path} actual={actual} declared={declared}"
        )


def difficulty_stratum(
    natural_missing_frame_count: int, strata: Mapping[str, Mapping[str, Any]]
) -> str:
    matches = []
    for name, specification in strata.items():
        lower = int(specification["natural_missing_frame_count_min"])
        upper_raw = specification.get("natural_missing_frame_count_max")
        upper = int(upper_raw) if upper_raw is not None else None
        if natural_missing_frame_count >= lower and (
            upper is None or natural_missing_frame_count <= upper
        ):
            matches.append(str(name))
    if len(matches) != 1:
        raise ValueError(
            "Natural missing-frame count must map to exactly one difficulty stratum: "
            f"count={natural_missing_frame_count} matches={matches}"
        )
    return matches[0]


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
            raise ValueError(f"Cannot decode HB depth: {depth_path}")
        if depth.shape != (expected_height, expected_width):
            raise ValueError(f"Unexpected HB frame shape: {depth.shape}")
        depth_cache[image_id] = np.asarray(depth)
    raw_depth = depth_cache[image_id]
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"Cannot decode HB mask: {mask_path}")
    if mask_image.ndim == 3:
        mask_image = mask_image[..., 0]
    mask = np.asarray(mask_image) > 0
    if mask.shape != raw_depth.shape:
        raise ValueError(f"HB mask/depth mismatch: {mask_path}")
    mask_pixels = int(np.count_nonzero(mask))
    validate_visible_mask_count(
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
    }


def load_development_tracks(
    dataset_root: Path, protocol: Mapping[str, Any]
) -> dict[str, list[FrameObservation]]:
    split = protocol["splits"]["development"]
    selection = protocol["selection"]
    split_root = dataset_root / str(split["directory"])
    tracks: dict[str, list[FrameObservation]] = {}
    for scene_id in [int(value) for value in split["scene_ids"]]:
        scene_dir = split_root / f"{scene_id:06d}"
        gt = read_json(scene_dir / "scene_gt.json")
        info = read_json(scene_dir / "scene_gt_info.json")
        camera = read_json(scene_dir / "scene_camera.json")
        keys = sorted(set(gt) & set(info) & set(camera), key=int)
        if len(keys) != 340:
            raise ValueError(f"HB scene {scene_id} expected 340 frames, found {len(keys)}")
        first_object_ids: set[int] | None = None
        depth_cache: dict[int, np.ndarray] = {}
        for key in keys:
            image_id = int(key)
            gt_entries = list(gt[key])
            info_entries = list(info[key])
            if len(gt_entries) != len(info_entries):
                raise ValueError(f"HB GT/info length mismatch: scene={scene_id} frame={image_id}")
            object_ids = [int(entry["obj_id"]) for entry in gt_entries]
            if len(object_ids) != len(set(object_ids)):
                raise ValueError(
                    f"HB single-instance contract violated: scene={scene_id} frame={image_id}"
                )
            current = set(object_ids)
            if first_object_ids is None:
                first_object_ids = current
            elif current != first_object_ids:
                raise ValueError(f"HB object set changed inside scene {scene_id}")
            for gt_index, (gt_entry, info_entry) in enumerate(
                zip(gt_entries, info_entries, strict=True)
            ):
                object_id = int(gt_entry["obj_id"])
                support = _frame_support(
                    scene_dir,
                    image_id,
                    gt_index,
                    info_entry,
                    camera[key],
                    minimum_mask_fraction=float(
                        selection["minimum_mask_area_fraction"]
                    ),
                    minimum_depth_ratio=float(selection["minimum_valid_depth_ratio"]),
                    minimum_valid_depth_pixels=int(
                        selection["minimum_valid_depth_pixels"]
                    ),
                    expected_height=int(selection["image_height"]),
                    expected_width=int(selection["image_width"]),
                    depth_cache=depth_cache,
                )
                observation = FrameObservation(
                    scene_id=scene_id,
                    image_id=image_id,
                    gt_instance_index=gt_index,
                    object_id=object_id,
                    gt_pose_m=bop_pose_m(gt_entry),
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
                tracks.setdefault(observation.track_id, []).append(observation)
    for track_id, rows in tracks.items():
        rows.sort(key=lambda row: row.image_id)
        if len(rows) != 340 or len({row.object_id for row in rows}) != 1:
            raise ValueError(f"Invalid HB track: {track_id}")
    return tracks


def _inference_row(observation: FrameObservation, dataset_root: Path) -> dict[str, Any]:
    camera_matrix = np.asarray(observation.camera_entry["cam_K"], dtype=np.float64).reshape(3, 3)
    unit_check_pose = np.eye(4, dtype=np.float64)
    unit_check_pose[2, 3] = observation.median_depth_m
    return {
        "record_type": "m5_r4_development_inference_sample",
        "schema_version": SCHEMA_VERSION,
        "sample_id": observation.sample_id,
        "dataset": "hb",
        "dataset_split": "val_primesense",
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": "primesense",
        "visible_mask_pixel_count": observation.mask_pixels,
        "input_mask_area_fraction": observation.mask_area_fraction,
        "valid_depth_pixel_count": observation.valid_depth_pixels,
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
    by_object = summary["by_object"]
    lines = [
        "# PoseLoop M5-R4A HB Primesense development input audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This development bundle was selected from new HomebrewedDB Primesense scenes 1-8 using only frozen mask/depth availability rules. The earlier M5-R4 builder opened development labels only for an input-feasibility scan and produced no predictions or pose-error results; M5-R4A was frozen before inference. No Kinect 2 sealed data was downloaded or read by this builder.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Selected tracks | {summary['track_count']} |",
        f"| Represented objects | {summary['object_count']} |",
        f"| Replay frames | {summary['replay_frame_count']} |",
        f"| FoundationPose inference frames | {summary['inference_sample_count']} |",
        f"| Natural missing frames | {summary['natural_missing_frame_count']} |",
        f"| Sparse-dropout tracks | {summary['difficulty_stratum_track_counts'].get('sparse_dropout', 0)} |",
        f"| Stress-dropout tracks | {summary['difficulty_stratum_track_counts'].get('stress_dropout', 0)} |",
        "",
        "| Object | Tracks | Replay | Available | Missing |",
        "|---:|---:|---:|---:|---:|",
    ]
    for object_id in sorted(by_object, key=int):
        row = by_object[object_id]
        lines.append(
            f"| {object_id} | {row['track_count']} | {row['replay_frame_count']} | "
            f"{row['available_frame_count']} | {row['natural_missing_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "Source archives:",
            "",
            f"- `hb_base.zip`: `{source['base']['sha256']}`",
            f"- `hb_models.zip`: `{source['models']['sha256']}`",
            f"- `hb_val_primesense.zip`: `{source['development']['sha256']}`",
            "",
            "This audit is not an outcome and cannot justify advancement by itself.",
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
        raise FileExistsError(f"M5-R4 development output already exists: {output_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected M5-R4 protocol")
    archives = protocol["source"]["archives"]
    source_receipt: dict[str, Any] = {}
    for key in ("base", "models", "development"):
        expected = archives[key]
        path = archive_root / str(expected["filename"])
        if not path.is_file() or path.stat().st_size != int(expected["size_bytes"]):
            raise FileNotFoundError(f"Missing or wrong-size HB source archive: {path}")
        digest = sha256_file(path)
        if digest != str(expected["sha256"]):
            raise RuntimeError(f"HB source archive hash mismatch: {path}")
        source_receipt[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    required = [
        dataset_root / "models" / "models_info.json",
        dataset_root / "models_eval" / "models_info.json",
        dataset_root / "val_primesense",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete HB development dataset: {missing}")

    tracks = load_development_tracks(dataset_root, protocol)
    selection = protocol["selection"]
    eligible: list[dict[str, Any]] = []
    for track_id, rows in tracks.items():
        window = select_best_window(
            [row.available for row in rows],
            window_length=int(selection["window_length_frames"]),
            minimum_available=int(selection["eligible_window"]["available_frame_count_min"]),
            minimum_missing=int(selection["eligible_window"]["natural_missing_frame_count_min"]),
        )
        if window is None:
            continue
        start, stop = window
        selected_rows = rows[start:stop]
        natural_missing_frame_count = sum(not row.available for row in selected_rows)
        eligible.append(
            {
                "track_id": track_id,
                "object_id": rows[0].object_id,
                "scene_id": rows[0].scene_id,
                "start_index": start,
                "stop_index_exclusive": stop,
                "available_frame_count": sum(row.available for row in selected_rows),
                "natural_missing_frame_count": natural_missing_frame_count,
                "difficulty_stratum": difficulty_stratum(
                    natural_missing_frame_count, selection["difficulty_strata"]
                ),
            }
        )
    selected = select_tracks(
        eligible,
        namespace="poseloop-m5-r4a-development:",
        track_cap=int(selection["track_cap"]),
        per_object_cap=int(selection["track_cap_per_object"]),
    )
    gate = protocol["evaluation"]["development_gate"]
    selected_object_count = len({int(row["object_id"]) for row in selected})
    selected_strata = Counter(str(row["difficulty_stratum"]) for row in selected)
    if len(selected) < int(gate["track_count_min"]):
        raise RuntimeError(f"M5-R4 selected too few development tracks: {len(selected)}")
    if selected_object_count < int(gate["represented_object_count_min"]):
        raise RuntimeError(
            f"M5-R4 selected too few development objects: {selected_object_count}"
        )
    for stratum_name, specification in selection["difficulty_strata"].items():
        observed = int(selected_strata[str(stratum_name)])
        required = int(specification["selected_track_count_min"])
        if observed < required:
            raise RuntimeError(
                f"M5-R4A selected too few {stratum_name} tracks: "
                f"observed={observed} required={required}"
            )

    inference_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    by_object: dict[str, dict[str, int]] = {}
    for choice in selected:
        rows = tracks[str(choice["track_id"])][
            int(choice["start_index"]) : int(choice["stop_index_exclusive"])
        ]
        object_key = str(int(choice["object_id"]))
        totals = by_object.setdefault(
            object_key,
            {
                "track_count": 0,
                "replay_frame_count": 0,
                "available_frame_count": 0,
                "natural_missing_frame_count": 0,
            },
        )
        totals["track_count"] += 1
        frames: list[dict[str, Any]] = []
        for replay_index, observation in enumerate(rows):
            if observation.available:
                inference_rows.append(_inference_row(observation, dataset_root))
            evaluator_rows.append(
                {
                    "record_type": "m5_r4_development_evaluator_frame",
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
            totals["replay_frame_count"] += 1
            if observation.available:
                totals["available_frame_count"] += 1
            else:
                totals["natural_missing_frame_count"] += 1
        track_rows.append(
            {
                "record_type": "m5_r4_development_track",
                "schema_version": SCHEMA_VERSION,
                "track_id": observation.track_id,
                "scene_id": observation.scene_id,
                "object_id": observation.object_id,
                "frame_count": len(frames),
                "frames": frames,
            }
        )

    inference_rows.sort(key=lambda row: (int(row["object_id"]), str(row["sample_id"])))
    evaluator_rows.sort(key=lambda row: (str(row["track_id"]), int(row["replay_frame_index"])))
    track_rows.sort(key=lambda row: str(row["track_id"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R4A HB Primesense development input freeze",
        "status": "DEVELOPMENT_BUNDLE_READY",
        "protocol_id": PROTOCOL_ID,
        "track_count": len(track_rows),
        "object_count": len(by_object),
        "replay_frame_count": sum(int(row["frame_count"]) for row in track_rows),
        "inference_sample_count": len(inference_rows),
        "natural_missing_frame_count": sum(
            int(row["natural_missing_frame_count"]) for row in by_object.values()
        ),
        "by_object": dict(sorted(by_object.items(), key=lambda item: int(item[0]))),
        "eligible_track_count_before_cap": len(eligible),
        "difficulty_stratum_track_counts": dict(sorted(selected_strata.items())),
        "selection_reads_pose_or_error_outcome": False,
        "artificial_frame_deletion": False,
        "sealed_archive_or_label_read": False,
    }
    if summary["natural_missing_frame_count"] < int(
        gate["natural_missing_frame_count_min"]
    ):
        raise RuntimeError("M5-R4 selected development bundle has too few missing frames")

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
        "core": repo_root / "scripts" / "m5_r4_core.py",
        "inference": repo_root / "scripts" / "run_m5_r4_inference.py",
        "evaluator": repo_root / "scripts" / "evaluate_m5_r4_development.py",
        "legacy_core": repo_root / "scripts" / "m5_g0_core.py",
        "foundationpose_runner": repo_root / "scripts" / "run_r1_sealed_inference.py",
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"M5-R4 freeze code missing: {path}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "M5-R4A HB Primesense development bundle",
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
