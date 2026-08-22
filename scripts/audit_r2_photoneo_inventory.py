#!/usr/bin/env python3
"""Inventory untouched Photoneo tracks for an R2 calibration/sealed split."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_m2_groups import associate_scene_tracks
from build_r1_sealed_photoneo import _input_record, load_scene_photoneo
from m1_common import load_jsonl, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
MIN_VIEWS_PER_TRACK = 5


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--r1-groups",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "sealed_photoneo" / "groups_inference.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "photoneo_inventory.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    r1_groups_path = args.r1_groups.resolve()
    output_path = args.output.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not r1_groups_path.is_file():
        raise FileNotFoundError(r1_groups_path)

    r1_groups = load_jsonl(r1_groups_path)
    r1_tracks = {
        str(group["oracle_association"]["track_id"])
        for group in r1_groups
    }
    r1_sample_ids = {
        str(view["sample_id"])
        for group in r1_groups
        for view in group["views"]
    }

    available_by_track: dict[str, list[str]] = defaultdict(list)
    object_by_track: dict[str, int] = {}
    all_sample_ids: set[str] = set()
    scene_ids: list[int] = []
    for scene_id in range(0, 75, 5):
        scene_ids.append(scene_id)
        # Photoneo frames are 2064x1544. Keep only one scene's decoded depth
        # frames resident so this label-free inventory cannot grow to several GB.
        frame_cache: dict[tuple[int, int], tuple[Any, tuple[int, int]]] = {}
        scene = load_scene_photoneo(dataset_root, scene_id)
        _, tracks, _ = associate_scene_tracks(scene)
        for observations in tracks.values():
            for observation in observations:
                inference, _ = _input_record(observation, dataset_root, frame_cache)
                sample_id = str(inference["sample_id"])
                if sample_id in all_sample_ids:
                    raise RuntimeError(f"Duplicate Photoneo sample ID: {sample_id}")
                all_sample_ids.add(sample_id)
                track_id = str(observation.track_id)
                object_id = int(observation.object_id)
                previous = object_by_track.setdefault(track_id, object_id)
                if previous != object_id:
                    raise RuntimeError(f"Track changes object identity: {track_id}")
                if bool(inference["input_available"]):
                    available_by_track[track_id].append(sample_id)
        print(f"inventory scene={scene_id}: samples={len(all_sample_ids)}", flush=True)

    eligible_tracks = {
        track_id
        for track_id, sample_ids in available_by_track.items()
        if len(sample_ids) >= MIN_VIEWS_PER_TRACK
    }
    untouched_tracks = eligible_tracks - r1_tracks
    if r1_sample_ids - all_sample_ids:
        raise RuntimeError("R1 inference groups contain unknown Photoneo sample IDs")
    if any(set(available_by_track[track_id]) & r1_sample_ids for track_id in untouched_tracks):
        raise RuntimeError("Track-disjoint inventory unexpectedly overlaps R1 sample IDs")

    object_ids = sorted(set(object_by_track.values()))
    per_object: dict[str, Any] = {}
    for object_id in object_ids:
        object_tracks = {
            track_id
            for track_id in eligible_tracks
            if object_by_track[track_id] == object_id
        }
        object_r1_tracks = object_tracks & r1_tracks
        object_untouched = object_tracks - r1_tracks
        per_object[str(object_id)] = {
            "eligible_tracks_total": len(object_tracks),
            "r1_calibration_tracks": len(object_r1_tracks),
            "untouched_tracks": len(object_untouched),
            "untouched_available_observations": sum(
                len(available_by_track[track_id]) for track_id in object_untouched
            ),
            "max_one_target_per_untouched_track": len(object_untouched),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "PoseLoop R2 Photoneo label-free inventory",
        "status": "complete",
        "scene_ids": scene_ids,
        "selection_fields": [
            "input availability",
            "object ID",
            "oracle association track ID for split isolation only",
        ],
        "prediction_or_pose_error_read": False,
        "r1_evaluator_artifact_read": False,
        "minimum_available_views_per_track": MIN_VIEWS_PER_TRACK,
        "counts": {
            "all_samples": len(all_sample_ids),
            "eligible_tracks": len(eligible_tracks),
            "r1_calibration_target_groups": len(r1_groups),
            "r1_calibration_tracks": len(r1_tracks & eligible_tracks),
            "r1_calibration_unique_group_samples": len(r1_sample_ids),
            "untouched_tracks": len(untouched_tracks),
            "untouched_available_observations": sum(
                len(available_by_track[track_id]) for track_id in untouched_tracks
            ),
        },
        "per_object": per_object,
        "provenance": {
            "dataset_root": str(dataset_root),
            "r1_groups": {
                "path": str(r1_groups_path),
                "sha256": sha256_file(r1_groups_path),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, result)
    print(
        "R2 Photoneo inventory: "
        f"eligible_tracks={len(eligible_tracks)}, r1_tracks={len(r1_tracks & eligible_tracks)}, "
        f"untouched_tracks={len(untouched_tracks)}"
    )
    for object_id in sorted(per_object, key=int):
        row = per_object[object_id]
        print(
            f"object={object_id}: untouched_tracks={row['untouched_tracks']}, "
            f"untouched_observations={row['untouched_available_observations']}"
        )
    print(output_path)


if __name__ == "__main__":
    main()
