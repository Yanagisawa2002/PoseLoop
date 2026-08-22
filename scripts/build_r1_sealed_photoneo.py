#!/usr/bin/env python3
"""Build and seal the untouched Photoneo manifest before R1 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from build_m2_groups import Observation, associate_scene_tracks
from m1_common import (
    bop_pose_m,
    read_json,
    sha256_file,
    visibility_bin,
    write_json_atomic,
    write_jsonl_atomic,
)
from m2_common import (
    camera_center_world_m,
    camera_world_to_camera_pose_m,
    greedy_camera_center_diversity,
    model_to_world_pose_m,
)


SCHEMA_VERSION = 1
SENSOR_MODALITY = "photoneo"
TARGETS_PER_OBJECT = 10
MIN_MASK_AREA_FRACTION = 0.001
MIN_VALID_DEPTH_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "sealed_photoneo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r1_protocol.json"
    )
    parser.add_argument("--output-root", type=Path, default=root)
    return parser.parse_args()


def sealed_sample_id(observation: Observation) -> str:
    return (
        f"xyzibd-val-pn-s{observation.scene_id:06d}-i{observation.image_id:06d}-"
        f"g{observation.gt_instance_index:06d}-o{observation.object_id:06d}"
    )


def load_scene_photoneo(
    dataset_root: Path, scene_id: int
) -> dict[int, list[Observation]]:
    scene = dataset_root / "val" / f"{scene_id:06d}"
    gt = read_json(scene / "scene_gt_photoneo.json")
    info = read_json(scene / "scene_gt_info_photoneo.json")
    camera = read_json(scene / "scene_camera_photoneo.json")
    keys = sorted(set(gt) & set(info) & set(camera), key=int)
    observations: dict[int, list[Observation]] = {}
    for key in keys:
        image_id = int(key)
        camera_pose, _ = camera_world_to_camera_pose_m(camera[key])
        frame = []
        for gt_index, (gt_entry, info_entry) in enumerate(
            zip(gt[key], info[key], strict=True)
        ):
            model_camera = bop_pose_m(gt_entry)
            frame.append(
                Observation(
                    scene_id=scene_id,
                    image_id=image_id,
                    gt_instance_index=gt_index,
                    object_id=int(gt_entry["obj_id"]),
                    gt_entry=gt_entry,
                    info_entry=info_entry,
                    camera_entry=camera[key],
                    camera_pose_m=camera_pose,
                    model_to_camera_pose_m=model_camera,
                    model_to_world_pose_m=model_to_world_pose_m(
                        camera_pose, model_camera
                    ),
                )
            )
        observations[image_id] = frame
    return observations


def _input_record(
    observation: Observation,
    dataset_root: Path,
    frame_cache: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = dataset_root / "val" / f"{observation.scene_id:06d}"
    gray_path = scene / "gray_photoneo" / f"{observation.image_id:06d}.png"
    depth_path = scene / "depth_photoneo" / f"{observation.image_id:06d}.png"
    mask_path = (
        scene
        / "mask_visib_photoneo"
        / f"{observation.image_id:06d}_{observation.gt_instance_index:06d}.png"
    )
    cache_key = observation.scene_id, observation.image_id
    if cache_key not in frame_cache:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        gray = cv2.imread(str(gray_path), cv2.IMREAD_UNCHANGED)
        if depth is None or gray is None or depth.ndim != 2:
            raise ValueError(f"Cannot decode Photoneo frame: {depth_path}")
        frame_cache[cache_key] = (np.asarray(depth), depth.shape)
    raw_depth, shape = frame_cache[cache_key]
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise ValueError(f"Cannot decode Photoneo mask: {mask_path}")
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[..., 0]
    mask = np.asarray(mask_raw) > 0
    if mask.shape != shape:
        raise ValueError(f"Photoneo mask/depth mismatch: {mask_path}")
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels != int(observation.info_entry["px_count_visib"]):
        raise ValueError(f"Photoneo mask count mismatch: {mask_path}")
    depth_scale = float(observation.camera_entry["depth_scale"])
    depth_mm = raw_depth.astype(float) * depth_scale
    valid = mask & np.isfinite(depth_mm) & (depth_mm > 0)
    valid_pixels = int(np.count_nonzero(valid))
    valid_ratio = valid_pixels / max(mask_pixels, 1)
    height, width = shape
    mask_area_fraction = mask_pixels / (height * width)
    available = bool(
        mask_area_fraction >= MIN_MASK_AREA_FRACTION
        and valid_ratio >= MIN_VALID_DEPTH_RATIO
        and valid_pixels >= 4
    )
    median_depth_m = (
        float(np.median(depth_mm[valid])) * 0.001 if valid_pixels else 1.0
    )
    unit_check_pose = np.eye(4)
    unit_check_pose[2, 3] = median_depth_m
    camera_matrix = np.asarray(
        observation.camera_entry["cam_K"], dtype=float
    ).reshape(3, 3)
    sample_id = sealed_sample_id(observation)
    inference = {
        "record_type": "r1_sealed_inference_sample",
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset": "xyzibd",
        "dataset_split": "val",
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": SENSOR_MODALITY,
        "visible_mask_pixel_count": mask_pixels,
        "input_mask_area_fraction": mask_area_fraction,
        "valid_depth_pixel_count": valid_pixels,
        "valid_depth_ratio_inside_mask": valid_ratio,
        "image_height": height,
        "image_width": width,
        "rgb_path": str(gray_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "model_path": str(
            (dataset_root / "models" / f"obj_{observation.object_id:06d}.ply").resolve()
        ),
        "camera_intrinsics_row_major": camera_matrix.tolist(),
        "raw_depth_scale": depth_scale,
        "input_unit_check_pose_m": unit_check_pose.tolist(),
        "input_available": available,
    }
    evaluator = {
        "record_type": "r1_sealed_evaluator_sample",
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": SENSOR_MODALITY,
        "physical_instance_id": observation.track_id,
        "visible_fraction": float(observation.info_entry["visib_fract"]),
        "visibility_bin": visibility_bin(
            float(observation.info_entry["visib_fract"])
        ),
        "gt_model_to_camera_pose_m": observation.model_to_camera_pose_m.tolist(),
    }
    return inference, evaluator


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sealed = protocol["splits"]["sealed"]["r1_execution"]
    if sealed["sensor_modality"].lower() != SENSOR_MODALITY:
        raise ValueError("Protocol sealed sensor is not Photoneo")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r1").resolve()):
        raise ValueError("Sealed artifacts must stay under artifacts/r1")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("Sealed Photoneo directory already exists and is non-empty")
    output_root.mkdir(parents=True, exist_ok=True)

    observations: list[Observation] = []
    for scene_id in range(0, 75, 5):
        scene = load_scene_photoneo(dataset_root, scene_id)
        _, tracks, _ = associate_scene_tracks(scene)
        observations.extend(
            observation for track in tracks.values() for observation in track
        )
    frame_cache: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]] = {}
    inference_by_id: dict[str, dict[str, Any]] = {}
    evaluator_by_id: dict[str, dict[str, Any]] = {}
    observation_by_id: dict[str, Observation] = {}
    by_track: dict[str, list[str]] = defaultdict(list)
    for observation in observations:
        inference, evaluator = _input_record(observation, dataset_root, frame_cache)
        sample_id = inference["sample_id"]
        if sample_id in inference_by_id:
            raise ValueError(f"Duplicate sealed Photoneo sample: {sample_id}")
        inference_by_id[sample_id] = inference
        evaluator_by_id[sample_id] = evaluator
        observation_by_id[sample_id] = observation
        if inference["input_available"]:
            by_track[observation.track_id].append(sample_id)

    eligible = [
        sample_id
        for sample_id, row in inference_by_id.items()
        if row["input_available"] and len(by_track[evaluator_by_id[sample_id]["physical_instance_id"]]) >= 5
    ]
    by_object: dict[int, list[str]] = defaultdict(list)
    for sample_id in eligible:
        by_object[int(inference_by_id[sample_id]["object_id"])].append(sample_id)
    target_ids = []
    for object_id in sorted(by_object):
        selected = sorted(
            by_object[object_id], key=lambda sample_id: (_stable_hash(sample_id), sample_id)
        )[:TARGETS_PER_OBJECT]
        if len(selected) != TARGETS_PER_OBJECT:
            raise RuntimeError(f"Object {object_id} has too few sealed targets")
        target_ids.extend(selected)
    if len(target_ids) != int(sealed["target_count"]):
        raise RuntimeError("Sealed target count differs from protocol")

    groups_inference = []
    groups_evaluator = []
    universe_ids: set[str] = set()
    for target_id in sorted(target_ids):
        target_observation = observation_by_id[target_id]
        track_id = evaluator_by_id[target_id]["physical_instance_id"]
        candidates = []
        for sample_id in by_track[track_id]:
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
            camera_center_world_m(target_observation.camera_pose_m), candidates, count=4
        )
        if len(selected_candidates) != 4:
            raise RuntimeError(f"Sealed group lacks four candidates: {target_id}")
        ordered_ids = [target_id] + [row["sample_id"] for row in selected_candidates]
        views_inference = []
        views_evaluator = []
        for rank, sample_id in enumerate(ordered_ids):
            observation = observation_by_id[sample_id]
            inference = inference_by_id[sample_id]
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
                    "visible_fraction": evaluator_by_id[sample_id]["visible_fraction"],
                    "gt_model_to_camera_pose_m": evaluator_by_id[sample_id][
                        "gt_model_to_camera_pose_m"
                    ],
                }
            )
            universe_ids.add(sample_id)
        group_base = {
            "group_id": target_id,
            "target_sample_id": target_id,
            "scene_id": target_observation.scene_id,
            "object_id": target_observation.object_id,
            "sensor_modality": SENSOR_MODALITY,
            "oracle_association": {"track_id": track_id, "deployable": False},
        }
        groups_inference.append({**group_base, "views": views_inference})
        groups_evaluator.append(
            {
                **group_base,
                "target_visibility_bin": evaluator_by_id[target_id]["visibility_bin"],
                "target_visible_fraction": evaluator_by_id[target_id]["visible_fraction"],
                "views": views_evaluator,
            }
        )

    inference_manifest = [inference_by_id[sample_id] for sample_id in sorted(universe_ids)]
    evaluator_labels = [evaluator_by_id[sample_id] for sample_id in sorted(universe_ids)]
    target_manifest = [inference_by_id[sample_id] for sample_id in sorted(target_ids)]
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
    development_ids = set()
    for path in (
        repo_root / "artifacts" / "m1" / "predictions.jsonl",
        repo_root / "artifacts" / "m2" / "view_predictions.jsonl",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("record_type") == "prediction":
                development_ids.add(str(row["sample_id"]))
    overlap = sorted(universe_ids & development_ids)
    if overlap:
        raise RuntimeError("Sealed sample IDs overlap development IDs")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "R1 single sealed validation",
        "status": "sealed_unopened",
        "sensor_modality": SENSOR_MODALITY,
        "target_count": len(target_ids),
        "inference_sample_count": len(inference_manifest),
        "object_count": len(by_object),
        "sample_id_overlap_with_realsense_development": 0,
        "physical_instance_overlap_not_excluded": True,
        "claim_scope": sealed["claim_scope"],
        "single_open_gates": sealed["single_open_gates"],
        "evaluation_invocation_count": 0,
        "labels_opened": False,
        "selection_uses_prediction_or_pose_error_outcome": False,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
    }
    write_json_atomic(output_root / "sealed_contract.json", contract)
    print(
        f"Photoneo sealed: targets={len(target_ids)}, samples={len(inference_manifest)}, "
        f"objects={len(by_object)}, dev_id_overlap=0"
    )
    print(output_root / "sealed_contract.json")


if __name__ == "__main__":
    main()
