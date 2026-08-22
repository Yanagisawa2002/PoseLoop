#!/usr/bin/env python3
"""Audit whether local real sequences satisfy the frozen M5-R1 replay gate."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m1_common import sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
MASK_AREA_FRACTION_MIN = 0.001
VALID_DEPTH_RATIO_MIN = 0.5


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "artifacts" / "r1" / "m5_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r1_protocol.json"
    )
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "m5_r1_real_replay_input_audit.md",
    )
    return parser.parse_args()


def choose_median_support_track(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("Cannot select from an empty object track set")
    median = float(np.median([int(row["available_observation_count"]) for row in rows]))
    return min(
        rows,
        key=lambda row: (
            abs(int(row["available_observation_count"]) - median),
            str(row["track_id"]),
        ),
    )


def input_gate(
    selected: Sequence[Mapping[str, Any]], phase2: Mapping[str, Any]
) -> dict[str, Any]:
    sequences_with_missing = sum(
        int(row["observed_missing_frame_count"]) > 0 for row in selected
    )
    total_missing = sum(int(row["observed_missing_frame_count"]) for row in selected)
    sequences_with_five_outputs = sum(
        int(row["available_observation_count"]) >= 5 for row in selected
    )
    conditions = {
        "minimum_sequence_count": len(selected)
        >= int(phase2["minimum_sequence_count"]),
        "every_sequence_has_at_least_five_ordered_outputs": sequences_with_five_outputs
        == len(selected),
        "minimum_sequences_with_observed_missing_frames": sequences_with_missing
        >= int(phase2["minimum_sequences_with_observed_missing_frames"]),
        "minimum_total_observed_missing_frames": total_missing
        >= int(phase2["minimum_total_observed_missing_frames"]),
        "ground_truth_not_used_for_observation_availability": True,
        "frame_clock_mapping_frozen_before_pose_inference": True,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "observed": {
            "sequence_count": len(selected),
            "sequences_with_at_least_five_ordered_outputs": sequences_with_five_outputs,
            "sequences_with_observed_missing_frames": sequences_with_missing,
            "total_observed_missing_frames": total_missing,
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    gate = result["input_gate"]
    return "\n".join(
        [
            "# PoseLoop M5-R1 real replay input audit",
            "",
            f"**{result['status']}**",
            "",
            "The local XYZ-IBD RealSense data provide ordered recorded frames but no "
            "hardware timestamps, so the frozen protocol maps common-frame ordinal to "
            "a nominal 20 Hz clock. Observation availability uses only input mask area "
            "and valid-depth support; predictor outcomes and pose errors are not read.",
            "",
            "| Input-gate quantity | Observed |",
            "| --- | ---: |",
            f"| Selected object sequences | {gate['observed']['sequence_count']} |",
            f"| Sequences with at least five observations | {gate['observed']['sequences_with_at_least_five_ordered_outputs']} |",
            f"| Sequences with observed missing frames | {gate['observed']['sequences_with_observed_missing_frames']} |",
            f"| Total observed missing frames | {gate['observed']['total_observed_missing_frames']} |",
            "",
            "FoundationPose replay was not launched because the input gate is evaluated "
            "before pose inference. Artificially deleting valid frames would create a "
            "synthetic-dropout replay and would not satisfy the frozen real-dropout claim.",
            "",
        ]
    )


def main() -> None:
    from build_m2_groups import (  # noqa: PLC0415
        associate_scene_tracks,
        load_scene_observations,
        observation_to_manifest_row,
    )

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    phase2 = protocol["stages"]["M5-R1"]["phase_2_protocol"]
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r1").resolve()):
        raise ValueError("M5-R1 audit outputs must stay under artifacts/r1")
    report_path = args.report.resolve()
    if not report_path.is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("M5-R1 audit report must stay under reports/r1")

    frame_cache: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]] = {}
    track_rows: list[dict[str, Any]] = []
    selection_index = 0
    for scene_id in range(0, 75, 5):
        observations, _ = load_scene_observations(dataset_root, scene_id)
        _, tracks, _ = associate_scene_tracks(observations)
        for track_id, track in sorted(tracks.items()):
            frame_rows = []
            for ordinal, observation in enumerate(
                sorted(track, key=lambda item: item.image_id)
            ):
                manifest = observation_to_manifest_row(
                    observation, dataset_root, selection_index, frame_cache
                )
                selection_index += 1
                image_pixels = int(manifest["image_height"]) * int(manifest["image_width"])
                mask_area_fraction = int(manifest["visible_mask_pixel_count"]) / image_pixels
                available = bool(
                    mask_area_fraction >= MASK_AREA_FRACTION_MIN
                    and float(manifest["valid_depth_ratio_inside_mask"])
                    >= VALID_DEPTH_RATIO_MIN
                )
                frame_rows.append(
                    {
                        "frame_ordinal": ordinal,
                        "timestamp_s": ordinal / 20.0,
                        "sample_id": manifest["sample_id"],
                        "image_id": observation.image_id,
                        "mask_area_fraction": mask_area_fraction,
                        "valid_depth_ratio_inside_mask": manifest[
                            "valid_depth_ratio_inside_mask"
                        ],
                        "observation_available": available,
                    }
                )
            available_count = sum(row["observation_available"] for row in frame_rows)
            track_rows.append(
                {
                    "record_type": "m5_r1_real_track_input_audit",
                    "schema_version": SCHEMA_VERSION,
                    "track_id": track_id,
                    "scene_id": scene_id,
                    "object_id": track[0].object_id,
                    "frame_count": len(frame_rows),
                    "available_observation_count": available_count,
                    "observed_missing_frame_count": len(frame_rows) - available_count,
                    "frames": frame_rows,
                }
            )
    by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in track_rows:
        by_object[int(row["object_id"])].append(row)
    selected = [
        dict(choose_median_support_track(by_object[object_id]))
        for object_id in sorted(by_object)
    ]
    gate = input_gate(selected, phase2)
    status = "READY_FOR_FOUNDATIONPOSE_REAL_REPLAY" if gate["passed"] else "BLOCKED_REAL_REPLAY_INPUT_GATE"
    output_root.mkdir(parents=True, exist_ok=True)
    tracks_path = output_root / "real_replay_track_input_audit.jsonl"
    selected_path = output_root / "real_replay_selected_sequences.jsonl"
    write_jsonl_atomic(tracks_path, track_rows)
    write_jsonl_atomic(selected_path, selected)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R1 phase 2 input audit",
        "status": status,
        "input_gate": gate,
        "availability_contract": {
            "mask_area_fraction_min": MASK_AREA_FRACTION_MIN,
            "valid_depth_ratio_min": VALID_DEPTH_RATIO_MIN,
            "clock": "sorted common-frame ordinal / 20 Hz",
            "hardware_timestamps_available": False,
            "predictor_outcomes_read": False,
            "evaluator_pose_errors_read": False,
        },
        "track_count": len(track_rows),
        "object_count": len(by_object),
        "foundationpose_replay_launched": False,
        "reason_not_launched": (
            None if gate["passed"] else "Frozen pre-inference real-dropout input gate did not pass."
        ),
        "input_provenance": {
            "dataset_root": str(dataset_root),
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        },
        "output_provenance": {
            "all_tracks": {"path": str(tracks_path), "sha256": sha256_file(tracks_path)},
            "selected_sequences": {"path": str(selected_path), "sha256": sha256_file(selected_path)},
        },
    }
    result_path = output_root / "real_replay_input_audit.json"
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(
        f"{status}: sequences={gate['observed']['sequence_count']}, "
        f"with_missing={gate['observed']['sequences_with_observed_missing_frames']}, "
        f"missing_frames={gate['observed']['total_observed_missing_frames']}"
    )
    print(result_path)


if __name__ == "__main__":
    main()
