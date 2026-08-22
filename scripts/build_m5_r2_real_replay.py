#!/usr/bin/env python3
"""Freeze the label-blind M5-R2 observed-dropout real replay bundle."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from build_m2_groups import (
    Observation,
    associate_scene_tracks,
    load_scene_observations,
    observation_to_manifest_row,
)
from m1_common import sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
MASK_AREA_FRACTION_MIN = 0.001
VALID_DEPTH_RATIO_MIN = 0.5
MIN_VALID_DEPTH_PIXELS = 4
CONTROL_HZ = 20.0
INFERENCE_ONLY_FIELDS = {
    "gt_model_to_camera_pose_m",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r2_protocol.json",
    )
    parser.add_argument(
        "--frozen-configs",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m5_r1" / "frozen_phase1_configurations.json",
    )
    parser.add_argument(
        "--predecessor-audit",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m5_r1" / "real_replay_input_audit.json",
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r2_input_freeze.md",
    )
    return parser.parse_args()


def observation_is_available(row: Mapping[str, Any]) -> bool:
    image_pixels = int(row["image_height"]) * int(row["image_width"])
    mask_area = int(row["visible_mask_pixel_count"]) / image_pixels
    return bool(
        mask_area >= MASK_AREA_FRACTION_MIN
        and float(row["valid_depth_ratio_inside_mask"]) >= VALID_DEPTH_RATIO_MIN
        and int(row["valid_depth_pixel_count"]) >= MIN_VALID_DEPTH_PIXELS
    )


def first_available_index(rows: Sequence[Mapping[str, Any]]) -> int | None:
    return next(
        (index for index, row in enumerate(rows) if bool(row["observation_available"])),
        None,
    )


def track_is_selected(rows: Sequence[Mapping[str, Any]]) -> bool:
    start = first_available_index(rows)
    if start is None:
        return False
    replay = rows[start:]
    available = sum(bool(row["observation_available"]) for row in replay)
    missing = len(replay) - available
    return available >= 5 and missing >= 1


def _input_unit_check_pose(
    row: Mapping[str, Any], raw_depth: np.ndarray
) -> list[list[float]]:
    mask_raw = np.asarray(imageio.imread(Path(str(row["mask_path"]))))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[..., 0]
    mask = mask_raw > 0
    depth_m = np.asarray(raw_depth, dtype=np.float64) * float(row["raw_depth_scale"]) * 0.001
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    if int(np.count_nonzero(valid)) != int(row["valid_depth_pixel_count"]):
        raise ValueError(f"Depth validity changed for {row['sample_id']}")
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = float(np.median(depth_m[valid]))
    return pose.tolist()


def _inference_row(
    generated: Mapping[str, Any], raw_depth: np.ndarray
) -> dict[str, Any]:
    row = {
        key: value for key, value in generated.items() if key not in INFERENCE_ONLY_FIELDS
    }
    row.update(
        {
            "record_type": "m5_r2_inference_sample",
            "schema_version": SCHEMA_VERSION,
            "input_mask_area_fraction": int(generated["visible_mask_pixel_count"])
            / (int(generated["image_height"]) * int(generated["image_width"])),
            "input_unit_check_pose_m": _input_unit_check_pose(generated, raw_depth),
            "input_available": True,
            "evaluator_label_read": False,
        }
    )
    forbidden = INFERENCE_ONLY_FIELDS & set(row)
    if forbidden:
        raise RuntimeError(f"Evaluator fields leaked into inference row: {sorted(forbidden)}")
    return row


def validate_expected_counts(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for field in (
        "track_count",
        "object_count",
        "replay_frame_count",
        "inference_sample_count",
        "natural_missing_frame_count",
    ):
        if int(observed[field]) != int(expected[field]):
            raise RuntimeError(
                f"M5-R2 audited {field}={observed[field]}, expected {expected[field]}"
            )
    if [int(value) for value in observed["object_ids"]] != [
        int(value) for value in expected["object_ids"]
    ]:
        raise RuntimeError("M5-R2 audited object IDs differ from the frozen protocol")


def render_report(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    by_object = summary["by_object"]
    lines = [
        "# PoseLoop M5-R2 input freeze",
        "",
        "**FROZEN_LABELS_UNOPENED**",
        "",
        "The old M5-R1 result remains blocked. M5-R2 uses every audited physical "
        "track that contains a natural missing observation after causal initialization; "
        "selection reads mask/depth availability but no FoundationPose or temporal-method outcome.",
        "",
        "| Quantity | Frozen value |",
        "| --- | ---: |",
        f"| Physical tracks | {summary['track_count']} |",
        f"| Represented objects | {summary['object_count']} |",
        f"| Replay frames | {summary['replay_frame_count']} |",
        f"| FoundationPose inference inputs | {summary['inference_sample_count']} |",
        f"| Natural missing replay frames | {summary['natural_missing_frame_count']} |",
        f"| Initial unavailable frames excluded | {summary['preinitialization_missing_frame_count']} |",
        "",
        "| Object | Tracks | Replay frames | Natural missing |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for object_id in sorted(by_object, key=int):
        row = by_object[object_id]
        lines.append(
            f"| {int(object_id):02d} | {row['track_count']} | "
            f"{row['replay_frame_count']} | {row['natural_missing_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "The replay clock is frame ordinal at nominal 20 Hz; the source has no "
            "hardware timestamps. Objects are static in the dataset world frame while "
            "the recorded camera moves. Oracle association is used only to construct "
            "the non-deployable sequences.",
            "",
            f"Contract: `{contract['protocol_id']}`; evaluator invocation count is 0.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    configs_path = args.frozen_configs.resolve()
    predecessor_path = args.predecessor_audit.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    for path in (protocol_path, configs_path, predecessor_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not output_root.is_relative_to((repo_root / "artifacts" / "r2").resolve()):
        raise ValueError("M5-R2 artifacts must stay under artifacts/r2")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M5-R2 report must stay under reports/r2")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("M5-R2 output directory already exists and is non-empty")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "poseloop-m5-r2-v1":
        raise ValueError("Unexpected M5-R2 protocol")
    frozen_configs = json.loads(configs_path.read_text(encoding="utf-8"))
    if frozen_configs.get("status") != "frozen_for_real_replay":
        raise RuntimeError("M5-R1 phase-1 configurations are not frozen")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("status") != "BLOCKED_REAL_REPLAY_INPUT_GATE":
        raise RuntimeError("M5-R1 predecessor is not the frozen blocked result")

    selected_tracks: list[dict[str, Any]] = []
    inference_by_id: dict[str, dict[str, Any]] = {}
    evaluator_labels: list[dict[str, Any]] = []
    preinitialization_missing = 0
    selection_index = 0
    for scene_id in protocol["source"]["scene_ids"]:
        observations, _ = load_scene_observations(dataset_root, int(scene_id))
        _, tracks, _ = associate_scene_tracks(observations)
        frame_cache: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]] = {}
        for track_id, track in sorted(tracks.items()):
            source_rows: list[dict[str, Any]] = []
            for source_ordinal, observation in enumerate(
                sorted(track, key=lambda item: item.image_id)
            ):
                generated = observation_to_manifest_row(
                    observation, dataset_root, selection_index, frame_cache
                )
                selection_index += 1
                source_rows.append(
                    {
                        "source_frame_ordinal": source_ordinal,
                        "observation": observation,
                        "generated": generated,
                        "observation_available": observation_is_available(generated),
                    }
                )
            if not track_is_selected(source_rows):
                continue
            start = first_available_index(source_rows)
            assert start is not None
            preinitialization_missing += start
            replay_source = source_rows[start:]
            track_frames: list[dict[str, Any]] = []
            for replay_index, source in enumerate(replay_source):
                observation: Observation = source["observation"]
                generated = source["generated"]
                available = bool(source["observation_available"])
                frame_row = {
                    "record_type": "m5_r2_track_frame",
                    "schema_version": SCHEMA_VERSION,
                    "track_id": track_id,
                    "scene_id": int(observation.scene_id),
                    "object_id": int(observation.object_id),
                    "replay_frame_index": replay_index,
                    "source_frame_ordinal": int(source["source_frame_ordinal"]),
                    "timestamp_s": (
                        int(source["source_frame_ordinal"])
                        - int(replay_source[0]["source_frame_ordinal"])
                    )
                    / CONTROL_HZ,
                    "sample_id": str(generated["sample_id"]),
                    "image_id": int(observation.image_id),
                    "gt_instance_index": int(observation.gt_instance_index),
                    "observation_available": available,
                    "input_mask_area_fraction": int(generated["visible_mask_pixel_count"])
                    / (int(generated["image_height"]) * int(generated["image_width"])),
                    "valid_depth_ratio_inside_mask": float(
                        generated["valid_depth_ratio_inside_mask"]
                    ),
                    "camera_world_to_camera_pose_m": observation.camera_pose_m.tolist(),
                }
                track_frames.append(frame_row)
                if available:
                    sample_id = str(generated["sample_id"])
                    if sample_id in inference_by_id:
                        raise RuntimeError(f"Duplicate M5-R2 inference sample: {sample_id}")
                    raw_depth = frame_cache[
                        (int(observation.scene_id), int(observation.image_id))
                    ][1]
                    inference_by_id[sample_id] = _inference_row(generated, raw_depth)
                evaluator_labels.append(
                    {
                        "record_type": "m5_r2_evaluator_frame",
                        "schema_version": SCHEMA_VERSION,
                        "track_id": track_id,
                        "scene_id": int(observation.scene_id),
                        "object_id": int(observation.object_id),
                        "replay_frame_index": replay_index,
                        "source_frame_ordinal": int(source["source_frame_ordinal"]),
                        "sample_id": str(generated["sample_id"]),
                        "observation_available": available,
                        "gt_model_to_world_pose_m": observation.model_to_world_pose_m.tolist(),
                        "gt_model_to_camera_pose_m": observation.model_to_camera_pose_m.tolist(),
                        "visible_fraction": float(observation.info_entry["visib_fract"]),
                    }
                )
            selected_tracks.append(
                {
                    "record_type": "m5_r2_track",
                    "schema_version": SCHEMA_VERSION,
                    "track_id": track_id,
                    "scene_id": int(track[0].scene_id),
                    "object_id": int(track[0].object_id),
                    "source_frame_count": len(source_rows),
                    "excluded_preinitialization_frame_count": start,
                    "replay_frame_count": len(track_frames),
                    "available_observation_count": sum(
                        bool(row["observation_available"]) for row in track_frames
                    ),
                    "natural_missing_frame_count": sum(
                        not bool(row["observation_available"]) for row in track_frames
                    ),
                    "frames": track_frames,
                }
            )
        print(
            f"M5-R2 scan scene={int(scene_id):02d}: selected_tracks={len(selected_tracks)}",
            flush=True,
        )

    selected_tracks.sort(key=lambda row: str(row["track_id"]))
    evaluator_labels.sort(
        key=lambda row: (str(row["track_id"]), int(row["replay_frame_index"]))
    )
    inference_rows = [inference_by_id[key] for key in sorted(inference_by_id)]
    by_object: dict[str, dict[str, int]] = defaultdict(
        lambda: {"track_count": 0, "replay_frame_count": 0, "natural_missing_frame_count": 0}
    )
    for track in selected_tracks:
        item = by_object[str(track["object_id"])]
        item["track_count"] += 1
        item["replay_frame_count"] += int(track["replay_frame_count"])
        item["natural_missing_frame_count"] += int(track["natural_missing_frame_count"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R2 input freeze",
        "status": "FROZEN_LABELS_UNOPENED",
        "track_count": len(selected_tracks),
        "object_count": len(by_object),
        "object_ids": sorted(int(key) for key in by_object),
        "replay_frame_count": len(evaluator_labels),
        "inference_sample_count": len(inference_rows),
        "natural_missing_frame_count": sum(
            int(track["natural_missing_frame_count"]) for track in selected_tracks
        ),
        "preinitialization_missing_frame_count": preinitialization_missing,
        "by_object": dict(sorted(by_object.items(), key=lambda item: int(item[0]))),
        "selection_reads_prediction_or_pose_error_outcome": False,
        "artificial_frame_deletion": False,
        "track_association": "oracle GT world-centre association; non-deployable",
    }
    validate_expected_counts(summary, protocol["selection"]["expected_after_audit"])

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "track_manifest": output_root / "track_manifest.jsonl",
        "inference_manifest": output_root / "inference_manifest.jsonl",
        "evaluator_labels": output_root / "evaluator_labels.jsonl",
        "selection_summary": output_root / "selection_summary.json",
    }
    write_jsonl_atomic(paths["track_manifest"], selected_tracks)
    write_jsonl_atomic(paths["inference_manifest"], inference_rows)
    write_jsonl_atomic(paths["evaluator_labels"], evaluator_labels)
    write_json_atomic(paths["selection_summary"], summary)
    code_paths = {
        "builder": Path(__file__).resolve(),
        "inference": repo_root / "scripts" / "run_m5_r2_inference.py",
        "evaluator": repo_root / "scripts" / "evaluate_m5_r2_once.py",
        "m5_core": repo_root / "scripts" / "m5_g0_core.py",
        "m5_r1_core": repo_root / "scripts" / "m5_r1_core.py",
        "m5_r2_core": repo_root / "scripts" / "m5_r2_core.py",
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    models_info_path = dataset_root / "models_eval" / "models_info.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M5-R2 observed-dropout real replay",
        "status": "development_labels_unopened",
        "labels_opened": False,
        "evaluation_invocation_count": 0,
        "claim_scope": protocol["claim_boundary"]["allowed"],
        "selection_summary": summary,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "frozen_phase1_configurations": {
                "path": str(configs_path),
                "sha256": sha256_file(configs_path),
            },
            "predecessor_audit": {
                "path": str(predecessor_path),
                "sha256": sha256_file(predecessor_path),
                "status": predecessor["status"],
            },
            "models_eval_info": {
                "path": str(models_info_path),
                "sha256": sha256_file(models_info_path),
            },
        },
        "code": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in code_paths.items()
        },
        "configuration_ids": {
            method: value["selected_config_id"]
            for method, value in frozen_configs["selected"].items()
        },
        "selection_uses_prediction_or_pose_error_outcome": False,
        "artificial_frame_deletion": False,
    }
    contract_path = output_root / "contract.json"
    write_json_atomic(contract_path, contract)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, contract), encoding="utf-8")
    print(
        "M5-R2 frozen: "
        f"tracks={summary['track_count']}, frames={summary['replay_frame_count']}, "
        f"inference={summary['inference_sample_count']}, missing={summary['natural_missing_frame_count']}"
    )
    print(contract_path)


if __name__ == "__main__":
    main()
