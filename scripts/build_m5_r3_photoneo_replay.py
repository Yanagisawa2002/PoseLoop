#!/usr/bin/env python3
"""Freeze the track-disjoint M5-R3 Photoneo replay before inference."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_m2_groups import Observation, associate_scene_tracks
from build_m5_r2_real_replay import first_available_index, track_is_selected
from build_r1_sealed_photoneo import _input_record, load_scene_photoneo
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
CONTROL_HZ = 20.0
FORBIDDEN_INFERENCE_FIELDS = {
    "gt_model_to_camera_pose_m",
    "gt_model_to_world_pose_m",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r3_protocol.json",
    )
    parser.add_argument(
        "--development-result",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r3_development.json",
    )
    parser.add_argument(
        "--frozen-configs",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m5_r1" / "frozen_phase1_configurations.json",
    )
    parser.add_argument(
        "--r1-groups",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "sealed_photoneo" / "groups_inference.jsonl",
    )
    parser.add_argument(
        "--r2-groups",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "sealed_photoneo" / "groups_inference.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r3_input_freeze.md",
    )
    return parser.parse_args()


def consumed_track_ids(groups: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for group in groups:
        association = group.get("oracle_association")
        if not isinstance(association, Mapping) or not association.get("track_id"):
            raise ValueError("Prior Photoneo group lacks an oracle track boundary")
        result.add(str(association["track_id"]))
    return result


def inference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(row.get("input_available")):
        raise ValueError("M5-R3 inference rows must be naturally available")
    forbidden = FORBIDDEN_INFERENCE_FIELDS & set(row)
    if forbidden:
        raise RuntimeError(f"Evaluator fields leaked into M5-R3 input: {sorted(forbidden)}")
    result = dict(row)
    result.update(
        {
            "record_type": "m5_r3_inference_sample",
            "schema_version": SCHEMA_VERSION,
            "evaluator_label_read": False,
        }
    )
    return result


def validate_expected_counts(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for field in (
        "track_count",
        "object_count",
        "replay_frame_count",
        "inference_sample_count",
        "natural_missing_frame_count",
        "preinitialization_missing_frame_count",
    ):
        if int(observed[field]) != int(expected[field]):
            raise RuntimeError(
                f"M5-R3 audited {field}={observed[field]}, expected {expected[field]}"
            )
    if [int(value) for value in observed["object_ids"]] != [
        int(value) for value in expected["object_ids"]
    ]:
        raise RuntimeError("M5-R3 object IDs differ from the availability-only audit")
    observed_tracks = {
        key: int(value["track_count"]) for key, value in observed["by_object"].items()
    }
    expected_tracks = {
        str(key): int(value) for key, value in expected["track_counts_by_object"].items()
    }
    if observed_tracks != expected_tracks:
        raise RuntimeError("M5-R3 per-object track counts differ from the audit")


def render_report(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    lines = [
        "# PoseLoop M5-R3 Photoneo input freeze",
        "",
        "**SEALED_LABELS_UNOPENED**",
        "",
        "The static all-history quotient medoid was frozen using only the consumed "
        "M5-R2 RealSense replay. These Photoneo tracks have zero overlap with the "
        "union of every R1 and R2 Photoneo group track.",
        "",
        "| Quantity | Frozen value |",
        "| --- | ---: |",
        f"| Prior Photoneo tracks excluded | {summary['prior_photoneo_track_count']} |",
        f"| Sealed physical tracks | {summary['track_count']} |",
        f"| Represented objects | {summary['object_count']} |",
        f"| Replay frames | {summary['replay_frame_count']} |",
        f"| FoundationPose inference inputs | {summary['inference_sample_count']} |",
        f"| Natural missing frames | {summary['natural_missing_frame_count']} |",
        f"| Pre-initialization missing excluded | {summary['preinitialization_missing_frame_count']} |",
        "",
        "| Object | Tracks | Replay frames | Natural missing |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for object_id in sorted(summary["by_object"], key=int):
        row = summary["by_object"][object_id]
        lines.append(
            f"| {int(object_id):02d} | {row['track_count']} | "
            f"{row['replay_frame_count']} | {row['natural_missing_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "Selection used only natural input availability, frame order, object ID, "
            "and oracle association for split isolation. It did not read predictions, "
            "temporal outputs, or pose-error outcomes from the retained tracks.",
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
    development_path = args.development_result.resolve()
    configs_path = args.frozen_configs.resolve()
    r1_groups_path = args.r1_groups.resolve()
    r2_groups_path = args.r2_groups.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    for path in (
        protocol_path,
        development_path,
        configs_path,
        r1_groups_path,
        r2_groups_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not output_root.is_relative_to((repo_root / "artifacts" / "r2").resolve()):
        raise ValueError("M5-R3 artifacts must stay under artifacts/r2")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("M5-R3 output directory already exists and is non-empty")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M5-R3 report must stay under reports/r2")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "poseloop-m5-r3-v1":
        raise ValueError("Unexpected M5-R3 protocol")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if (
        development.get("status") != "FROZEN_M5_R3_CANDIDATE"
        or development.get("selected_candidate", {}).get("candidate_id") != "all_history"
        or development.get("selected_candidate", {}).get("history_limit") is not None
    ):
        raise RuntimeError("M5-R3 development candidate is not frozen")
    frozen_development = protocol["development"]["frozen_result"]
    if sha256_file(development_path) != str(frozen_development["sha256"]):
        raise RuntimeError("M5-R3 development result changed after protocol freeze")
    frozen_configs = json.loads(configs_path.read_text(encoding="utf-8"))
    if frozen_configs.get("status") != "frozen_for_real_replay":
        raise RuntimeError("M5-R1 nearest baseline is not frozen")

    source_boundary = protocol["source"]["prior_photoneo_boundaries"]
    if sha256_file(r1_groups_path) != source_boundary["r1_groups"]["sha256"]:
        raise RuntimeError("R1 Photoneo group boundary changed")
    if sha256_file(r2_groups_path) != source_boundary["r2_groups"]["sha256"]:
        raise RuntimeError("R2 Photoneo group boundary changed")
    r1_tracks = consumed_track_ids(load_jsonl(r1_groups_path))
    r2_tracks = consumed_track_ids(load_jsonl(r2_groups_path))
    consumed_tracks = r1_tracks | r2_tracks
    if r1_tracks & r2_tracks:
        raise RuntimeError("R1 and R2 Photoneo track boundaries unexpectedly overlap")
    if len(consumed_tracks) != int(source_boundary["expected_unique_consumed_track_count"]):
        raise RuntimeError("Prior Photoneo consumed-track count changed")

    selected_tracks: list[dict[str, Any]] = []
    inference_by_id: dict[str, dict[str, Any]] = {}
    evaluator_labels: list[dict[str, Any]] = []
    preinitialization_missing = 0
    for scene_id in protocol["source"]["scene_ids"]:
        scene = load_scene_photoneo(dataset_root, int(scene_id))
        _, tracks, _ = associate_scene_tracks(scene)
        frame_cache: dict[tuple[int, int], tuple[Any, tuple[int, int]]] = {}
        for track_id, track in sorted(tracks.items()):
            if str(track_id) in consumed_tracks:
                continue
            source_rows: list[dict[str, Any]] = []
            for source_ordinal, observation in enumerate(
                sorted(track, key=lambda item: item.image_id)
            ):
                source_input, _ = _input_record(observation, dataset_root, frame_cache)
                source_rows.append(
                    {
                        "source_frame_ordinal": source_ordinal,
                        "observation": observation,
                        "inference": source_input,
                        "observation_available": bool(source_input["input_available"]),
                    }
                )
            if not track_is_selected(source_rows):
                continue
            start = first_available_index(source_rows)
            assert start is not None
            preinitialization_missing += start
            replay_source = source_rows[start:]
            frames: list[dict[str, Any]] = []
            for replay_index, source in enumerate(replay_source):
                observation: Observation = source["observation"]
                source_input = source["inference"]
                available = bool(source["observation_available"])
                frame = {
                    "record_type": "m5_r3_track_frame",
                    "schema_version": SCHEMA_VERSION,
                    "track_id": str(track_id),
                    "scene_id": int(observation.scene_id),
                    "object_id": int(observation.object_id),
                    "replay_frame_index": replay_index,
                    "source_frame_ordinal": int(source["source_frame_ordinal"]),
                    "timestamp_s": (
                        int(source["source_frame_ordinal"])
                        - int(replay_source[0]["source_frame_ordinal"])
                    )
                    / CONTROL_HZ,
                    "sample_id": str(source_input["sample_id"]),
                    "image_id": int(observation.image_id),
                    "gt_instance_index": int(observation.gt_instance_index),
                    "observation_available": available,
                    "input_mask_area_fraction": float(source_input["input_mask_area_fraction"]),
                    "valid_depth_ratio_inside_mask": float(
                        source_input["valid_depth_ratio_inside_mask"]
                    ),
                    "camera_world_to_camera_pose_m": observation.camera_pose_m.tolist(),
                }
                frames.append(frame)
                if available:
                    sample_id = str(source_input["sample_id"])
                    if sample_id in inference_by_id:
                        raise RuntimeError(f"Duplicate M5-R3 inference sample: {sample_id}")
                    inference_by_id[sample_id] = inference_row(source_input)
                evaluator_labels.append(
                    {
                        "record_type": "m5_r3_evaluator_frame",
                        "schema_version": SCHEMA_VERSION,
                        "track_id": str(track_id),
                        "scene_id": int(observation.scene_id),
                        "object_id": int(observation.object_id),
                        "replay_frame_index": replay_index,
                        "source_frame_ordinal": int(source["source_frame_ordinal"]),
                        "sample_id": str(source_input["sample_id"]),
                        "observation_available": available,
                        "gt_model_to_world_pose_m": observation.model_to_world_pose_m.tolist(),
                        "gt_model_to_camera_pose_m": observation.model_to_camera_pose_m.tolist(),
                        "visible_fraction": float(observation.info_entry["visib_fract"]),
                    }
                )
            selected_tracks.append(
                {
                    "record_type": "m5_r3_track",
                    "schema_version": SCHEMA_VERSION,
                    "track_id": str(track_id),
                    "scene_id": int(track[0].scene_id),
                    "object_id": int(track[0].object_id),
                    "source_frame_count": len(source_rows),
                    "excluded_preinitialization_frame_count": start,
                    "replay_frame_count": len(frames),
                    "available_observation_count": sum(
                        bool(row["observation_available"]) for row in frames
                    ),
                    "natural_missing_frame_count": sum(
                        not bool(row["observation_available"]) for row in frames
                    ),
                    "frames": frames,
                }
            )
        print(
            f"M5-R3 scan scene={int(scene_id):02d}: selected_tracks={len(selected_tracks)}",
            flush=True,
        )

    selected_tracks.sort(key=lambda row: str(row["track_id"]))
    if {str(row["track_id"]) for row in selected_tracks} & consumed_tracks:
        raise RuntimeError("M5-R3 selected a previously consumed Photoneo track")
    evaluator_labels.sort(
        key=lambda row: (str(row["track_id"]), int(row["replay_frame_index"]))
    )
    inference_rows = [inference_by_id[key] for key in sorted(inference_by_id)]
    by_object: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "track_count": 0,
            "replay_frame_count": 0,
            "natural_missing_frame_count": 0,
        }
    )
    for track in selected_tracks:
        item = by_object[str(track["object_id"])]
        item["track_count"] += 1
        item["replay_frame_count"] += int(track["replay_frame_count"])
        item["natural_missing_frame_count"] += int(track["natural_missing_frame_count"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R3 Photoneo input freeze",
        "status": "SEALED_LABELS_UNOPENED",
        "prior_photoneo_track_count": len(consumed_tracks),
        "prior_track_overlap": 0,
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
    validate_expected_counts(
        summary, protocol["selection"]["expected_after_availability_only_audit"]
    )

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
        "inference": repo_root / "scripts" / "run_m5_r3_inference.py",
        "evaluator": repo_root / "scripts" / "evaluate_m5_r3_once.py",
        "m5_r2_builder_helpers": repo_root / "scripts" / "build_m5_r2_real_replay.py",
        "m5_r2_evaluator_helpers": repo_root / "scripts" / "evaluate_m5_r2_once.py",
        "photoneo_input_builder": repo_root / "scripts" / "build_r1_sealed_photoneo.py",
        "photoneo_inference_helpers": repo_root / "scripts" / "run_r1_sealed_inference.py",
        "foundationpose_batch_helpers": repo_root / "scripts" / "run_xyzibd_batch.py",
        "foundationpose_environment_helpers": repo_root / "scripts" / "run_xyzibd_smoke.py",
        "m5_core": repo_root / "scripts" / "m5_g0_core.py",
        "m5_metrics": repo_root / "scripts" / "m5_g0_metrics.py",
        "m5_r1_core": repo_root / "scripts" / "m5_r1_core.py",
        "m5_r2_core": repo_root / "scripts" / "m5_r2_core.py",
        "m5_r3_core": repo_root / "scripts" / "m5_r3_core.py",
    }
    for path in code_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    models_info_path = dataset_root / "models_eval" / "models_info.json"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M5-R3 track-disjoint Photoneo sealed replay",
        "status": "sealed_labels_unopened",
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
            "development_result": {
                "path": str(development_path),
                "sha256": sha256_file(development_path),
            },
            "frozen_phase1_configurations": {
                "path": str(configs_path),
                "sha256": sha256_file(configs_path),
            },
            "r1_groups": {"path": str(r1_groups_path), "sha256": sha256_file(r1_groups_path)},
            "r2_groups": {"path": str(r2_groups_path), "sha256": sha256_file(r2_groups_path)},
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
            "NEAREST_REPRESENTATIVE_CT": frozen_configs["selected"]
            ["NEAREST_REPRESENTATIVE_CT"]["selected_config_id"],
            "STATIC_QUOTIENT_ALL_HISTORY_MEDOID": "all_history",
        },
        "prior_photoneo_track_count": len(consumed_tracks),
        "prior_track_overlap": 0,
        "selection_uses_prediction_or_pose_error_outcome": False,
        "artificial_frame_deletion": False,
    }
    contract_path = output_root / "contract.json"
    write_json_atomic(contract_path, contract)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, contract), encoding="utf-8")
    print(
        "M5-R3 sealed: "
        f"tracks={summary['track_count']}, frames={summary['replay_frame_count']}, "
        f"inference={summary['inference_sample_count']}, "
        f"missing={summary['natural_missing_frame_count']}, prior_overlap=0"
    )
    print(contract_path)


if __name__ == "__main__":
    main()
