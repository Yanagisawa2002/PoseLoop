#!/usr/bin/env python3
"""Build the R2 Photoneo package on tracks disjoint from R1 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_m2_groups import Observation, associate_scene_tracks
from build_r1_sealed_photoneo import _input_record, load_scene_photoneo
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from m2_common import camera_center_world_m, greedy_camera_center_diversity


SCHEMA_VERSION = 1
TARGETS_PER_OBJECT = 10
MIN_VIEWS_PER_TRACK = 5
SENSOR_MODALITY = "photoneo"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    r2 = repo_root / "artifacts" / "r2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r2_protocol.json"
    )
    parser.add_argument(
        "--inventory", type=Path, default=r2 / "photoneo_inventory.json"
    )
    parser.add_argument(
        "--r1-groups", type=Path, default=r1 / "sealed_photoneo" / "groups_inference.jsonl"
    )
    parser.add_argument(
        "--m3-scheduler", type=Path, default=r2 / "m3_r2" / "frozen_scheduler.json"
    )
    parser.add_argument(
        "--m3-model", type=Path, default=r1 / "m3_r1" / "frozen_policy.joblib"
    )
    parser.add_argument(
        "--m4-contract", type=Path, default=r1 / "m4_r1" / "frozen_ranker.json"
    )
    parser.add_argument(
        "--m4-model", type=Path, default=r1 / "m4_r1" / "frozen_ranker.joblib"
    )
    parser.add_argument(
        "--m6-contract", type=Path, default=r2 / "m6_r2" / "frozen_model_contract.json"
    )
    parser.add_argument(
        "--m6-model", type=Path, default=r2 / "m6_r2" / "frozen_risk_model.joblib"
    )
    parser.add_argument(
        "--output-root", type=Path, default=r2 / "sealed_photoneo"
    )
    return parser.parse_args()


def stable_hash(prefix: str, value: str) -> str:
    return hashlib.sha256(f"{prefix}{value}".encode("utf-8")).hexdigest()


def round_robin_targets(
    sample_ids_by_track: Mapping[str, Sequence[str]], count: int
) -> list[str]:
    if count < 1:
        raise ValueError("Target count must be positive")
    tracks = sorted(
        sample_ids_by_track,
        key=lambda track_id: (stable_hash("r2-track|", track_id), track_id),
    )
    if not tracks:
        raise ValueError("No tracks are available for target selection")
    ordered = {
        track_id: sorted(
            (str(sample_id) for sample_id in sample_ids_by_track[track_id]),
            key=lambda sample_id: (stable_hash("r2-sample|", sample_id), sample_id),
        )
        for track_id in tracks
    }
    selected: list[str] = []
    depth = 0
    while len(selected) < count:
        added = 0
        for track_id in tracks:
            if depth < len(ordered[track_id]):
                selected.append(ordered[track_id][depth])
                added += 1
                if len(selected) == count:
                    break
        if added == 0:
            raise RuntimeError("The untouched tracks contain too few target observations")
        depth += 1
    if len(selected) != len(set(selected)):
        raise RuntimeError("Round-robin selection produced duplicate target IDs")
    return selected


def validate_components(paths: Mapping[str, Path], protocol: Mapping[str, Any]) -> None:
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    scheduler = json.loads(paths["m3_scheduler"].read_text(encoding="utf-8"))
    if scheduler.get("status") != "frozen_after_development_gate_pass":
        raise RuntimeError("M3-R2 scheduler is not frozen")
    if sha256_file(paths["m3_model"]) != str(
        protocol["M3-R2"]["marginal_value_model"]["sha256"]
    ):
        raise RuntimeError("M3 marginal-value model hash changed")
    m4 = json.loads(paths["m4_contract"].read_text(encoding="utf-8"))
    if m4.get("status") != "frozen_after_development_gate_pass":
        raise RuntimeError("M4 ranker is not frozen")
    if sha256_file(paths["m4_model"]) != str(m4["model"]["sha256"]):
        raise RuntimeError("M4 model hash changed")
    m6 = json.loads(paths["m6_contract"].read_text(encoding="utf-8"))
    if (
        m6.get("status") != "frozen_for_sealed_validation"
        or bool(m6.get("future_or_unacquired_view_feature_used"))
    ):
        raise RuntimeError("M6-R2 deployable model is not frozen")
    if sha256_file(paths["m6_model"]) != str(m6["model"]["sha256"]):
        raise RuntimeError("M6-R2 model hash changed")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    inventory_path = args.inventory.resolve()
    r1_groups_path = args.r1_groups.resolve()
    component_paths = {
        "m3_scheduler": args.m3_scheduler.resolve(),
        "m3_model": args.m3_model.resolve(),
        "m4_contract": args.m4_contract.resolve(),
        "m4_model": args.m4_model.resolve(),
        "m6_contract": args.m6_contract.resolve(),
        "m6_model": args.m6_model.resolve(),
    }
    for path in (protocol_path, inventory_path, r1_groups_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r2").resolve()):
        raise ValueError("R2 sealed outputs must stay under artifacts/r2")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("R2 sealed output directory already exists and is non-empty")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "poseloop-r2-v1":
        raise ValueError("Unexpected R2 protocol")
    sealed_spec = protocol["splits"]["photoneo_sealed"]
    if sha256_file(inventory_path) != str(sealed_spec["inventory"]["sha256"]):
        raise RuntimeError("R2 inventory hash changed")
    validate_components(component_paths, protocol)

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

    inference_by_id: dict[str, dict[str, Any]] = {}
    evaluator_by_id: dict[str, dict[str, Any]] = {}
    observation_by_id: dict[str, Observation] = {}
    available_by_track: dict[str, list[str]] = defaultdict(list)
    object_by_track: dict[str, int] = {}
    for scene_id in range(0, 75, 5):
        frame_cache: dict[tuple[int, int], tuple[Any, tuple[int, int]]] = {}
        scene = load_scene_photoneo(dataset_root, scene_id)
        _, tracks, _ = associate_scene_tracks(scene)
        for observations in tracks.values():
            for observation in observations:
                inference, evaluator = _input_record(observation, dataset_root, frame_cache)
                sample_id = str(inference["sample_id"])
                if sample_id in inference_by_id:
                    raise RuntimeError(f"Duplicate Photoneo sample ID: {sample_id}")
                inference_by_id[sample_id] = inference
                evaluator_by_id[sample_id] = evaluator
                observation_by_id[sample_id] = observation
                track_id = str(observation.track_id)
                object_by_track.setdefault(track_id, int(observation.object_id))
                if bool(inference["input_available"]):
                    available_by_track[track_id].append(sample_id)
        print(f"R2 package scan scene={scene_id}: samples={len(inference_by_id)}", flush=True)

    eligible_tracks = {
        track_id
        for track_id, sample_ids in available_by_track.items()
        if len(sample_ids) >= MIN_VIEWS_PER_TRACK
    }
    untouched_tracks = eligible_tracks - r1_tracks
    by_object_track: dict[int, dict[str, list[str]]] = defaultdict(dict)
    for track_id in untouched_tracks:
        by_object_track[object_by_track[track_id]][track_id] = available_by_track[track_id]
    target_ids: list[str] = []
    target_tracks_by_object: dict[str, int] = {}
    for object_id in sorted(by_object_track):
        selected = round_robin_targets(by_object_track[object_id], TARGETS_PER_OBJECT)
        target_ids.extend(selected)
        target_tracks_by_object[str(object_id)] = len(
            {str(observation_by_id[sample_id].track_id) for sample_id in selected}
        )
    if len(target_ids) != int(sealed_spec["target_count"]):
        raise RuntimeError("R2 selected target count differs from protocol")

    groups_inference = []
    groups_evaluator = []
    universe_ids: set[str] = set()
    selected_tracks: set[str] = set()
    for target_id in sorted(target_ids):
        target = observation_by_id[target_id]
        track_id = str(target.track_id)
        selected_tracks.add(track_id)
        candidates = []
        for sample_id in available_by_track[track_id]:
            if sample_id == target_id:
                continue
            observation = observation_by_id[sample_id]
            candidates.append(
                {
                    "sample_id": sample_id,
                    "image_id": observation.image_id,
                    "gt_instance_index": observation.gt_instance_index,
                    "camera_center_world_m": camera_center_world_m(
                        observation.camera_pose_m
                    ).tolist(),
                }
            )
        selected_candidates = greedy_camera_center_diversity(
            camera_center_world_m(target.camera_pose_m), candidates, count=4
        )
        if len(selected_candidates) != 4:
            raise RuntimeError(f"R2 group lacks four candidates: {target_id}")
        ordered_ids = [target_id] + [str(row["sample_id"]) for row in selected_candidates]
        views_inference = []
        views_evaluator = []
        for rank, sample_id in enumerate(ordered_ids):
            observation = observation_by_id[sample_id]
            inference = inference_by_id[sample_id]
            evaluator = evaluator_by_id[sample_id]
            base = {
                "acquisition_rank": rank,
                "sample_id": sample_id,
                "scene_id": observation.scene_id,
                "image_id": observation.image_id,
                "gt_instance_index": observation.gt_instance_index,
                "object_id": observation.object_id,
                "prediction_source": "m1" if rank == 0 else "m2",
                "visible_mask_pixel_count": inference["visible_mask_pixel_count"],
                "camera_world_to_camera_pose_m": observation.camera_pose_m.tolist(),
            }
            views_inference.append(base)
            views_evaluator.append(
                {
                    **base,
                    "visible_fraction": evaluator["visible_fraction"],
                    "gt_model_to_camera_pose_m": evaluator["gt_model_to_camera_pose_m"],
                }
            )
            universe_ids.add(sample_id)
        group_base = {
            "group_id": target_id,
            "target_sample_id": target_id,
            "scene_id": target.scene_id,
            "object_id": target.object_id,
            "sensor_modality": SENSOR_MODALITY,
            "oracle_association": {"track_id": track_id, "deployable": False},
        }
        groups_inference.append({**group_base, "views": views_inference})
        target_evaluator = evaluator_by_id[target_id]
        groups_evaluator.append(
            {
                **group_base,
                "target_visibility_bin": target_evaluator["visibility_bin"],
                "target_visible_fraction": target_evaluator["visible_fraction"],
                "views": views_evaluator,
            }
        )

    if selected_tracks & r1_tracks or universe_ids & r1_sample_ids:
        raise RuntimeError("R2 sealed package overlaps R1 calibration tracks or samples")
    inference_manifest = [inference_by_id[sample_id] for sample_id in sorted(universe_ids)]
    target_manifest = [inference_by_id[sample_id] for sample_id in sorted(target_ids)]
    evaluator_labels = [evaluator_by_id[sample_id] for sample_id in sorted(universe_ids)]
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "inference_manifest": output_root / "inference_manifest.jsonl",
        "target_manifest": output_root / "target_manifest.jsonl",
        "groups_inference": output_root / "groups_inference.jsonl",
        "evaluator_labels": output_root / "evaluator_labels.jsonl",
        "groups_evaluator": output_root / "groups_evaluator.jsonl",
    }
    payloads = {
        "inference_manifest": inference_manifest,
        "target_manifest": target_manifest,
        "groups_inference": groups_inference,
        "evaluator_labels": evaluator_labels,
        "groups_evaluator": groups_evaluator,
    }
    for name, path in paths.items():
        write_jsonl_atomic(path, payloads[name])
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "R2 track-disjoint Photoneo sealed validation",
        "status": "sealed_unopened",
        "sensor_modality": SENSOR_MODALITY,
        "target_count": len(target_ids),
        "inference_sample_count": len(inference_manifest),
        "object_count": len(by_object_track),
        "sealed_physical_track_count": len(selected_tracks),
        "target_track_counts_by_object": target_tracks_by_object,
        "calibration_track_overlap": 0,
        "calibration_sample_id_overlap": 0,
        "claim_scope": protocol["claim_boundary"]["allowed"],
        "single_open_gates": protocol["single_open_gates"],
        "evaluator_transform_audit": protocol["evaluator_transform_audit"],
        "evaluation_invocation_count": 0,
        "labels_opened": False,
        "selection_uses_prediction_or_pose_error_outcome": False,
        "legacy_compatible_inference_record_schema": "r1_sealed_*",
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "components": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in component_paths.items()
        },
        "calibration_boundary": {
            "r1_groups": {"path": str(r1_groups_path), "sha256": sha256_file(r1_groups_path)},
            "track_count": len(r1_tracks),
            "unique_group_sample_count": len(r1_sample_ids),
        },
        "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
    }
    write_json_atomic(output_root / "sealed_contract.json", contract)
    print(
        f"R2 Photoneo sealed: targets={len(target_ids)}, samples={len(inference_manifest)}, "
        f"tracks={len(selected_tracks)}, calibration_overlap=0"
    )
    print(output_root / "sealed_contract.json")


if __name__ == "__main__":
    main()
