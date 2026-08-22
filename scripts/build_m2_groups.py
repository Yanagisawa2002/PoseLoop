#!/usr/bin/env python3
"""Audit XYZ-IBD extrinsics and build calibrated PoseLoop M2 view groups."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_m1 import (
    BOP_TOOLKIT_COMMIT,
    load_object_evaluation_data,
    load_official_models,
    official_errors,
    toolkit_commit,
)
from m1_common import (
    MANIFEST_SCHEMA_VERSION,
    MODALITY,
    SELECTION_SEED,
    bop_pose_m,
    load_jsonl,
    read_json,
    sha256_file,
    stable_sample_id,
    visibility_bin,
    write_json_atomic,
    write_jsonl_atomic,
)
from m2_common import (
    ASSOCIATION_MAX_DISTANCE_MM,
    ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM,
    GT_AUDIT_MAX_NORMALIZED_MSSD,
    GT_AUDIT_MAX_MSPD_PX,
    GT_AUDIT_MAX_MSSD_MM,
    M2_MAX_ADDITIONAL_VIEWS,
    M2_MIN_VISIBLE_FRACTION,
    M2_SCHEMA_VERSION,
    camera_center_world_m,
    camera_world_to_camera_pose_m,
    ensure_raw_artifact_path,
    greedy_camera_center_diversity,
    model_to_world_pose_m,
    percentile_summary,
    transform_model_pose_to_target,
    unit_viewing_direction_world,
)


@dataclass(frozen=True)
class Observation:
    scene_id: int
    image_id: int
    gt_instance_index: int
    object_id: int
    gt_entry: dict[str, Any]
    info_entry: dict[str, Any]
    camera_entry: dict[str, Any]
    camera_pose_m: np.ndarray
    model_to_camera_pose_m: np.ndarray
    model_to_world_pose_m: np.ndarray
    track_id: str = ""

    @property
    def sample_id(self) -> str:
        return stable_sample_id(
            self.scene_id,
            self.image_id,
            self.gt_instance_index,
            self.object_id,
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")
        ),
    )
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=repo_root / "third_party" / "bop_toolkit",
    )
    parser.add_argument(
        "--m1-manifest",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--m1-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "predictions.jsonl",
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--groups-summary-output",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups_summary.json",
    )
    parser.add_argument(
        "--candidate-manifest-output",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--extrinsics-audit-output",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "extrinsics_audit.json",
    )
    return parser.parse_args()


def validate_m1_inputs(
    manifest_path: Path,
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_rows = load_jsonl(manifest_path)
    if len(manifest_rows) != 300:
        raise ValueError(f"Expected 300 M1 targets, got {len(manifest_rows)}")
    manifest_index: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        expected = stable_sample_id(
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        if sample_id != expected or sample_id in manifest_index:
            raise ValueError(f"Invalid or duplicate M1 sample identity: {sample_id}")
        manifest_index[sample_id] = row

    prediction_records = load_jsonl(predictions_path)
    if not prediction_records or prediction_records[0].get("record_type") != "metadata":
        raise ValueError("M1 predictions do not start with metadata")
    prediction_index: dict[str, dict[str, Any]] = {}
    for row in prediction_records[1:]:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in manifest_index or sample_id in prediction_index:
            raise ValueError(f"Unexpected or duplicate M1 prediction: {sample_id}")
        if row.get("status") not in {
            "success",
            "invalid_input",
            "inference_error",
            "nonfinite_pose",
        }:
            raise ValueError(f"Invalid M1 status for {sample_id}")
        prediction_index[sample_id] = row
    if set(prediction_index) != set(manifest_index):
        raise ValueError("M1 predictions do not cover the complete M1 manifest")
    return manifest_rows, prediction_index, prediction_records[0]


def load_scene_observations(
    data_root: Path,
    scene_id: int,
) -> tuple[
    dict[int, list[Observation]],
    dict[str, Any],
]:
    scene_dir = data_root / "val" / f"{scene_id:06d}"
    gt = read_json(scene_dir / "scene_gt_realsense.json")
    info = read_json(scene_dir / "scene_gt_info_realsense.json")
    camera = read_json(scene_dir / "scene_camera_realsense.json")
    common_keys = sorted(set(gt) & set(info) & set(camera), key=int)
    if not common_keys:
        raise ValueError(f"No common GT/info/camera frames for scene {scene_id:06d}")
    if set(gt) != set(info):
        raise ValueError(f"GT/info frame mismatch in scene {scene_id:06d}")

    observations: dict[int, list[Observation]] = {}
    rotation_key_counts: Counter[str] = Counter()
    translation_key_counts: Counter[str] = Counter()
    camera_with_extrinsics = 0
    for image_key in common_keys:
        image_id = int(image_key)
        camera_entry = camera[image_key]
        camera_pose, aliases = camera_world_to_camera_pose_m(camera_entry)
        rotation_key_counts[aliases["rotation_key"]] += 1
        translation_key_counts[aliases["translation_key"]] += 1
        camera_with_extrinsics += 1
        gt_entries = gt[image_key]
        info_entries = info[image_key]
        if len(gt_entries) != len(info_entries):
            raise ValueError(
                f"GT/info instance mismatch: scene {scene_id:06d}, "
                f"image {image_id:06d}"
            )
        frame_observations: list[Observation] = []
        for gt_index, (gt_entry, info_entry) in enumerate(
            zip(gt_entries, info_entries, strict=True)
        ):
            model_camera = bop_pose_m(gt_entry)
            model_world = model_to_world_pose_m(camera_pose, model_camera)
            frame_observations.append(
                Observation(
                    scene_id=scene_id,
                    image_id=image_id,
                    gt_instance_index=gt_index,
                    object_id=int(gt_entry["obj_id"]),
                    gt_entry=gt_entry,
                    info_entry=info_entry,
                    camera_entry=camera_entry,
                    camera_pose_m=camera_pose,
                    model_to_camera_pose_m=model_camera,
                    model_to_world_pose_m=model_world,
                )
            )
        observations[image_id] = frame_observations
    return observations, {
        "scene_id": scene_id,
        "camera_entry_count": len(camera),
        "gt_frame_count": len(gt),
        "common_frame_count": len(common_keys),
        "common_frame_ids": [int(key) for key in common_keys],
        "camera_entries_with_extrinsics_in_common_frames": camera_with_extrinsics,
        "rotation_key_counts": dict(sorted(rotation_key_counts.items())),
        "translation_key_counts": dict(sorted(translation_key_counts.items())),
    }


def associate_scene_tracks(
    observations: dict[int, list[Observation]],
) -> tuple[
    dict[tuple[int, int], Observation],
    dict[str, list[Observation]],
    dict[str, Any],
]:
    reference_image_id = min(observations)
    reference = observations[reference_image_id]
    reference_by_object: dict[int, list[Observation]] = defaultdict(list)
    for observation in reference:
        reference_by_object[observation.object_id].append(observation)
    for values in reference_by_object.values():
        values.sort(key=lambda item: item.gt_instance_index)

    indexed: dict[tuple[int, int], Observation] = {}
    tracks: dict[str, list[Observation]] = defaultdict(list)
    assigned_distances_mm: list[float] = []
    second_nearest_mm: list[float] = []
    ambiguity_margins_mm: list[float] = []
    retained_gt_index_count = 0
    assignment_count = 0

    for image_id in sorted(observations):
        current_by_object: dict[int, list[Observation]] = defaultdict(list)
        for observation in observations[image_id]:
            current_by_object[observation.object_id].append(observation)
        for values in current_by_object.values():
            values.sort(key=lambda item: item.gt_instance_index)
        if set(current_by_object) != set(reference_by_object):
            raise ValueError(f"Object-set mismatch at frame {image_id}")

        for object_id in sorted(reference_by_object):
            anchors = reference_by_object[object_id]
            current = current_by_object[object_id]
            if len(anchors) != len(current):
                raise ValueError(
                    f"Object {object_id} instance-count mismatch at frame {image_id}"
                )
            anchor_positions = np.stack(
                [item.model_to_world_pose_m[:3, 3] for item in anchors],
                axis=0,
            )
            current_positions = np.stack(
                [item.model_to_world_pose_m[:3, 3] for item in current],
                axis=0,
            )
            cost_mm = (
                np.linalg.norm(
                    anchor_positions[:, None, :] - current_positions[None, :, :],
                    axis=2,
                )
                * 1000.0
            )
            anchor_indices, current_indices = linear_sum_assignment(cost_mm)
            if len(anchor_indices) != len(anchors):
                raise ValueError("Hungarian assignment did not cover every instance")
            for anchor_index, current_index in zip(
                anchor_indices,
                current_indices,
                strict=True,
            ):
                distance_mm = float(cost_mm[anchor_index, current_index])
                row_alternatives = np.delete(
                    cost_mm[anchor_index],
                    current_index,
                )
                column_alternatives = np.delete(
                    cost_mm[:, current_index],
                    anchor_index,
                )
                row_second_mm = (
                    float(np.min(row_alternatives))
                    if row_alternatives.size
                    else math.inf
                )
                column_second_mm = (
                    float(np.min(column_alternatives))
                    if column_alternatives.size
                    else math.inf
                )
                second_mm = min(row_second_mm, column_second_mm)
                margin_mm = second_mm - distance_mm
                if distance_mm > ASSOCIATION_MAX_DISTANCE_MM:
                    raise ValueError(
                        f"Association exceeds {ASSOCIATION_MAX_DISTANCE_MM} mm: "
                        f"frame={image_id} object={object_id} distance={distance_mm}"
                    )
                if (
                    math.isfinite(margin_mm)
                    and margin_mm < ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
                ):
                    raise ValueError(
                        f"Ambiguous association below "
                        f"{ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM} mm margin: "
                        f"frame={image_id} object={object_id} margin={margin_mm}"
                    )
                anchor = anchors[anchor_index]
                observation = current[current_index]
                track_id = (
                    f"xyzibd-val-s{observation.scene_id:06d}-"
                    f"o{object_id:06d}-r{anchor.gt_instance_index:06d}"
                )
                associated = Observation(
                    **{
                        **observation.__dict__,
                        "track_id": track_id,
                    }
                )
                key = (image_id, observation.gt_instance_index)
                if key in indexed:
                    raise ValueError(f"Observation assigned twice: {key}")
                indexed[key] = associated
                tracks[track_id].append(associated)
                assigned_distances_mm.append(distance_mm)
                if math.isfinite(second_mm):
                    second_nearest_mm.append(second_mm)
                    ambiguity_margins_mm.append(margin_mm)
                if anchor.gt_instance_index == observation.gt_instance_index:
                    retained_gt_index_count += 1
                assignment_count += 1

    expected_observations = sum(len(values) for values in observations.values())
    if assignment_count != expected_observations:
        raise ValueError(
            f"Global assignment incomplete: {assignment_count}/{expected_observations}"
        )
    return indexed, tracks, {
        "reference_image_id": reference_image_id,
        "assignment_count": assignment_count,
        "track_count": len(tracks),
        "assigned_world_center_distance_mm": percentile_summary(
            assigned_distances_mm
        ),
        "second_nearest_world_center_distance_mm": percentile_summary(
            second_nearest_mm
        ),
        "ambiguity_margin_mm": percentile_summary(ambiguity_margins_mm),
        "retained_gt_instance_index_count": retained_gt_index_count,
        "retained_gt_instance_index_fraction": (
            retained_gt_index_count / assignment_count
        ),
        "global_one_to_one_assignment": True,
        "association_max_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
        "minimum_ambiguity_margin_mm": ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM,
    }


def read_frame_arrays(
    scene_dir: Path,
    image_id: int,
    gt_instance_index: int,
    frame_cache: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]],
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, Path, Path, Path]:
    stem = f"{image_id:06d}"
    rgb_path = scene_dir / "rgb_realsense" / f"{stem}.png"
    depth_path = scene_dir / "depth_realsense" / f"{stem}.png"
    mask_path = (
        scene_dir
        / "mask_visib_realsense"
        / f"{stem}_{gt_instance_index:06d}.png"
    )
    for path in (rgb_path, depth_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_key = (int(scene_dir.name), image_id)
    if cache_key not in frame_cache:
        rgb = imageio.imread(rgb_path)
        depth = np.asarray(imageio.imread(depth_path))
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth.ndim != 2:
            raise ValueError(f"Invalid RGB/depth arrays for {rgb_path}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth shape mismatch for {rgb_path}")
        frame_cache[cache_key] = ((int(rgb.shape[0]), int(rgb.shape[1])), depth)
    image_shape, raw_depth = frame_cache[cache_key]
    mask = np.asarray(imageio.imread(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != image_shape:
        raise ValueError(f"Mask shape mismatch for {mask_path}")
    return image_shape, raw_depth, mask, rgb_path, depth_path, mask_path


def observation_to_manifest_row(
    observation: Observation,
    data_root: Path,
    selection_index: int,
    frame_cache: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]],
) -> dict[str, Any]:
    scene_dir = data_root / "val" / f"{observation.scene_id:06d}"
    (
        image_shape,
        raw_depth,
        mask,
        rgb_path,
        depth_path,
        mask_path,
    ) = read_frame_arrays(
        scene_dir,
        observation.image_id,
        observation.gt_instance_index,
        frame_cache,
    )
    visible_mask = mask > 0
    actual_mask_pixels = int(np.count_nonzero(visible_mask))
    declared_mask_pixels = int(observation.info_entry["px_count_visib"])
    if actual_mask_pixels <= 0 or actual_mask_pixels != declared_mask_pixels:
        raise ValueError(
            f"Visible-mask pixel mismatch for {mask_path}: "
            f"{actual_mask_pixels} != {declared_mask_pixels}"
        )
    depth_scale = float(observation.camera_entry["depth_scale"])
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"Invalid depth scale for {observation.sample_id}")
    physical_depth_mm = raw_depth.astype(np.float64) * depth_scale
    valid_depth = (
        visible_mask & np.isfinite(physical_depth_mm) & (physical_depth_mm > 0)
    )
    valid_depth_pixels = int(np.count_nonzero(valid_depth))
    camera_matrix = np.asarray(
        observation.camera_entry["cam_K"],
        dtype=np.float64,
    ).reshape(3, 3)
    model_path = data_root / "models" / f"obj_{observation.object_id:06d}.ply"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    visible_fraction = float(observation.info_entry["visib_fract"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_index": selection_index,
        "selection_seed": SELECTION_SEED,
        "sample_id": observation.sample_id,
        "dataset": "xyzibd",
        "dataset_split": "val",
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "sensor_modality": MODALITY,
        "visible_fraction": visible_fraction,
        "visibility_bin": visibility_bin(visible_fraction),
        "visible_mask_pixel_count": actual_mask_pixels,
        "valid_depth_pixel_count": valid_depth_pixels,
        "valid_depth_ratio_inside_mask": valid_depth_pixels / actual_mask_pixels,
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "rgb_path": str(rgb_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "model_path": str(model_path.resolve()),
        "camera_intrinsics_row_major": camera_matrix.tolist(),
        "raw_depth_scale": depth_scale,
        "gt_model_to_camera_pose_m": (
            observation.model_to_camera_pose_m.tolist()
        ),
    }


def assert_m1_row_matches_generated(
    m1_row: dict[str, Any],
    generated: dict[str, Any],
) -> None:
    exact_fields = (
        "sample_id",
        "scene_id",
        "image_id",
        "gt_instance_index",
        "object_id",
        "sensor_modality",
        "visible_mask_pixel_count",
        "valid_depth_pixel_count",
        "image_height",
        "image_width",
        "rgb_path",
        "depth_path",
        "mask_path",
        "model_path",
    )
    for field in exact_fields:
        if m1_row[field] != generated[field]:
            raise ValueError(
                f"M1/generated field mismatch for {m1_row['sample_id']}: {field}"
            )
    numeric_fields = (
        "visible_fraction",
        "valid_depth_ratio_inside_mask",
        "raw_depth_scale",
    )
    for field in numeric_fields:
        if not math.isclose(
            float(m1_row[field]),
            float(generated[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"M1/generated numeric mismatch for {m1_row['sample_id']}: {field}"
            )
    array_fields = (
        "camera_intrinsics_row_major",
        "gt_model_to_camera_pose_m",
    )
    for field in array_fields:
        if not np.allclose(
            np.asarray(m1_row[field], dtype=np.float64),
            np.asarray(generated[field], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"M1/generated array mismatch for {m1_row['sample_id']}: {field}"
            )


def view_record(
    observation: Observation,
    manifest_row: dict[str, Any],
    track_center_world_m: np.ndarray,
    prediction_source: str,
    acquisition_rank: int,
    diversity_score_camera_center_distance_m: float | None,
) -> dict[str, Any]:
    direction = unit_viewing_direction_world(
        observation.camera_pose_m,
        track_center_world_m,
    )
    return {
        "acquisition_rank": acquisition_rank,
        "sample_id": observation.sample_id,
        "scene_id": observation.scene_id,
        "image_id": observation.image_id,
        "gt_instance_index": observation.gt_instance_index,
        "object_id": observation.object_id,
        "prediction_source": prediction_source,
        "visible_fraction": float(manifest_row["visible_fraction"]),
        "visible_mask_pixel_count": int(
            manifest_row["visible_mask_pixel_count"]
        ),
        "camera_world_to_camera_pose_m": observation.camera_pose_m.tolist(),
        "gt_model_to_camera_pose_m": (
            observation.model_to_camera_pose_m.tolist()
        ),
        "camera_center_world_m": (
            camera_center_world_m(observation.camera_pose_m).tolist()
        ),
        "viewing_direction_world": direction.tolist(),
        "diversity_score_camera_center_distance_m": (
            diversity_score_camera_center_distance_m
        ),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    toolkit_root = args.toolkit_root.resolve()
    manifest_path = args.m1_manifest.resolve()
    predictions_path = args.m1_predictions.resolve()
    groups_path = ensure_raw_artifact_path(args.groups_output, repo_root)
    groups_summary_path = ensure_raw_artifact_path(
        args.groups_summary_output,
        repo_root,
    )
    candidate_manifest_path = ensure_raw_artifact_path(
        args.candidate_manifest_output,
        repo_root,
    )
    extrinsics_audit_path = ensure_raw_artifact_path(
        args.extrinsics_audit_output,
        repo_root,
    )

    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    toolkit_sha = toolkit_commit(toolkit_root)
    if toolkit_sha != BOP_TOOLKIT_COMMIT:
        raise RuntimeError("Pinned BOP Toolkit verification failed")
    model_params, model_info = load_official_models(data_root)
    object_data = {
        object_id: load_object_evaluation_data(
            object_id,
            model_params,
            model_info,
        )
        for object_id in sorted(int(key) for key in model_info)
    }
    (
        m1_rows,
        m1_prediction_index,
        m1_prediction_metadata,
    ) = validate_m1_inputs(manifest_path, predictions_path)
    target_scene_ids = sorted({int(row["scene_id"]) for row in m1_rows})
    observation_index: dict[tuple[int, int, int], Observation] = {}
    scene_tracks: dict[tuple[int, str], list[Observation]] = {}
    scene_camera_audits: list[dict[str, Any]] = []
    scene_association_audits: list[dict[str, Any]] = []
    for scene_id in target_scene_ids:
        observations, camera_audit = load_scene_observations(data_root, scene_id)
        indexed, tracks, association_audit = associate_scene_tracks(observations)
        for (image_id, gt_index), observation in indexed.items():
            key = (scene_id, image_id, gt_index)
            if key in observation_index:
                raise ValueError(f"Duplicate observation identity: {key}")
            observation_index[key] = observation
        for track_id, values in tracks.items():
            scene_tracks[(scene_id, track_id)] = values
        scene_camera_audits.append(camera_audit)
        scene_association_audits.append(
            {"scene_id": scene_id, **association_audit}
        )

    frame_cache: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]] = {}
    manifest_row_cache: dict[str, dict[str, Any]] = {}

    def manifest_row_for(observation: Observation) -> dict[str, Any]:
        if observation.sample_id not in manifest_row_cache:
            manifest_row_cache[observation.sample_id] = (
                observation_to_manifest_row(
                    observation,
                    data_root,
                    selection_index=-1,
                    frame_cache=frame_cache,
                )
            )
        return manifest_row_cache[observation.sample_id]

    groups: list[dict[str, Any]] = []
    candidate_reference_counts: Counter[str] = Counter()
    selected_cross_view_pairs: list[
        tuple[dict[str, Any], Observation, Observation]
    ] = []
    all_target_pair_association_distances_mm: list[float] = []
    all_target_pair_second_nearest_mm: list[float] = []
    all_target_pair_ambiguity_margins_mm: list[float] = []

    for target_row in m1_rows:
        target_key = (
            int(target_row["scene_id"]),
            int(target_row["image_id"]),
            int(target_row["gt_instance_index"]),
        )
        target_observation = observation_index.get(target_key)
        if target_observation is None:
            raise KeyError(f"M1 target has no calibrated observation: {target_key}")
        if target_observation.object_id != int(target_row["object_id"]):
            raise ValueError(f"M1 target object mismatch: {target_row['sample_id']}")
        generated_target = manifest_row_for(target_observation)
        assert_m1_row_matches_generated(target_row, generated_target)
        track_values = sorted(
            scene_tracks[(target_observation.scene_id, target_observation.track_id)],
            key=lambda item: (item.image_id, item.gt_instance_index),
        )
        track_center = np.mean(
            np.stack(
                [item.model_to_world_pose_m[:3, 3] for item in track_values],
                axis=0,
            ),
            axis=0,
        )
        target_camera_center = camera_center_world_m(
            target_observation.camera_pose_m
        )
        eligible_candidates: list[dict[str, Any]] = []
        for candidate_observation in track_values:
            if candidate_observation.image_id == target_observation.image_id:
                continue
            candidate_row = manifest_row_for(candidate_observation)
            visible_fraction = float(candidate_row["visible_fraction"])
            mask_pixels = int(candidate_row["visible_mask_pixel_count"])
            if (
                visible_fraction < M2_MIN_VISIBLE_FRACTION
                or mask_pixels <= 0
            ):
                continue
            direction = unit_viewing_direction_world(
                candidate_observation.camera_pose_m,
                track_center,
            )
            eligible_candidates.append(
                {
                    "sample_id": candidate_observation.sample_id,
                    "scene_id": candidate_observation.scene_id,
                    "image_id": candidate_observation.image_id,
                    "gt_instance_index": (
                        candidate_observation.gt_instance_index
                    ),
                    "object_id": candidate_observation.object_id,
                    "camera_center_world_m": (
                        camera_center_world_m(
                            candidate_observation.camera_pose_m
                        ).tolist()
                    ),
                    "viewing_direction_world": direction.tolist(),
                    "_observation": candidate_observation,
                }
            )

            target_position = target_observation.model_to_world_pose_m[:3, 3]
            same_object = [
                item
                for item in observation_index.values()
                if item.scene_id == target_observation.scene_id
                and item.image_id == candidate_observation.image_id
                and item.object_id == target_observation.object_id
            ]
            distances = sorted(
                float(
                    np.linalg.norm(
                        target_position - item.model_to_world_pose_m[:3, 3]
                    )
                    * 1000.0
                )
                for item in same_object
            )
            assigned = float(
                np.linalg.norm(
                    target_position
                    - candidate_observation.model_to_world_pose_m[:3, 3]
                )
                * 1000.0
            )
            row_alternatives = [
                value
                for value in distances
                if not math.isclose(
                    value,
                    assigned,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            candidate_position = (
                candidate_observation.model_to_world_pose_m[:3, 3]
            )
            same_object_in_target = [
                item
                for item in observation_index.values()
                if item.scene_id == target_observation.scene_id
                and item.image_id == target_observation.image_id
                and item.object_id == target_observation.object_id
            ]
            column_distances = sorted(
                float(
                    np.linalg.norm(
                        candidate_position - item.model_to_world_pose_m[:3, 3]
                    )
                    * 1000.0
                )
                for item in same_object_in_target
            )
            column_alternatives = [
                value
                for value in column_distances
                if not math.isclose(
                    value,
                    assigned,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            all_target_pair_association_distances_mm.append(assigned)
            finite_alternatives = row_alternatives + column_alternatives
            if finite_alternatives:
                second = min(finite_alternatives)
                all_target_pair_second_nearest_mm.append(second)
                all_target_pair_ambiguity_margins_mm.append(second - assigned)

        selected = greedy_camera_center_diversity(
            target_camera_center,
            eligible_candidates,
            count=M2_MAX_ADDITIONAL_VIEWS,
        )
        target_view = view_record(
            target_observation,
            target_row,
            track_center,
            prediction_source="m1",
            acquisition_rank=0,
            diversity_score_camera_center_distance_m=None,
        )
        views = [target_view]
        for acquisition_rank, candidate in enumerate(selected, start=1):
            candidate_observation = candidate.pop("_observation")
            candidate_row = manifest_row_for(candidate_observation)
            source = (
                "m1"
                if candidate_observation.sample_id in m1_prediction_index
                else "m2"
            )
            views.append(
                view_record(
                    candidate_observation,
                    candidate_row,
                    track_center,
                    prediction_source=source,
                    acquisition_rank=acquisition_rank,
                    diversity_score_camera_center_distance_m=float(
                        candidate[
                            "diversity_score_camera_center_distance_m"
                        ]
                    ),
                )
            )
            selected_cross_view_pairs.append(
                (target_row, target_observation, candidate_observation)
            )
            if source == "m2":
                candidate_reference_counts[candidate_observation.sample_id] += 1

        group = {
            "schema_version": M2_SCHEMA_VERSION,
            "group_id": str(target_row["sample_id"]),
            "target_sample_id": str(target_row["sample_id"]),
            "scene_id": int(target_row["scene_id"]),
            "object_id": int(target_row["object_id"]),
            "target_visibility_bin": str(target_row["visibility_bin"]),
            "target_visible_fraction": float(target_row["visible_fraction"]),
            "oracle_association": {
                "method": (
                    "GT model centres transformed to world coordinates with "
                    "per-object global Hungarian one-to-one matching"
                ),
                "uses_ground_truth": True,
                "deployable": False,
                "track_id": target_observation.track_id,
                "max_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
                "minimum_ambiguity_margin_mm": (
                    ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
                ),
            },
            "acquisition_order": {
                "method": (
                    "greedy farthest-point sampling over official world-frame "
                    "camera centres"
                ),
                "uses_gt_pose_error": False,
                "uses_gt_visible_fraction_for_order": False,
                "candidate_eligibility_min_visible_fraction": (
                    M2_MIN_VISIBLE_FRACTION
                ),
                "tie_break": "image_id, gt_instance_index, sample_id",
            },
            "eligible_additional_view_count": len(eligible_candidates),
            "available_additional_view_count": len(views) - 1,
            "available_view_count": len(views),
            "views": views,
        }
        groups.append(group)

    if len(groups) != 300:
        raise ValueError(f"Expected 300 groups, got {len(groups)}")
    if len({group["group_id"] for group in groups}) != 300:
        raise ValueError("M2 group IDs are not unique")

    candidate_rows: list[dict[str, Any]] = []
    for selection_index, sample_id in enumerate(
        sorted(
            candidate_reference_counts,
            key=lambda identity: (
                int(manifest_row_cache[identity]["object_id"]),
                int(manifest_row_cache[identity]["scene_id"]),
                int(manifest_row_cache[identity]["image_id"]),
                int(manifest_row_cache[identity]["gt_instance_index"]),
            ),
        )
    ):
        row = dict(manifest_row_cache[sample_id])
        row["selection_index"] = selection_index
        row["m2_group_reference_count"] = candidate_reference_counts[sample_id]
        candidate_rows.append(row)
    if any(row["sample_id"] in m1_prediction_index for row in candidate_rows):
        raise ValueError("Candidate manifest contains an existing M1 prediction")

    referenced_m2_ids = {
        str(view["sample_id"])
        for group in groups
        for view in group["views"][1:]
        if view["prediction_source"] == "m2"
    }
    if referenced_m2_ids != {str(row["sample_id"]) for row in candidate_rows}:
        raise ValueError("Candidate manifest does not exactly match M2 group references")

    mssd_values: list[float] = []
    normalized_mssd_values: list[float] = []
    mspd_values: list[float] = []
    audit_outliers: list[dict[str, Any]] = []
    for target_row, target_observation, candidate_observation in (
        selected_cross_view_pairs
    ):
        transformed_gt = transform_model_pose_to_target(
            target_observation.camera_pose_m,
            candidate_observation.camera_pose_m,
            candidate_observation.model_to_camera_pose_m,
        )
        object_id = int(target_row["object_id"])
        mssd_mm, mspd_px = official_errors(
            transformed_gt,
            np.asarray(target_row["gt_model_to_camera_pose_m"], dtype=np.float64),
            np.asarray(
                target_row["camera_intrinsics_row_major"],
                dtype=np.float64,
            ),
            object_data[object_id],
        )
        normalized = mssd_mm / float(object_data[object_id]["diameter_mm"])
        mssd_values.append(mssd_mm)
        mspd_values.append(mspd_px)
        normalized_mssd_values.append(normalized)
        if (
            mssd_mm > GT_AUDIT_MAX_MSSD_MM
            or normalized > GT_AUDIT_MAX_NORMALIZED_MSSD
            or mspd_px > GT_AUDIT_MAX_MSPD_PX
        ):
            audit_outliers.append(
                {
                    "target_sample_id": target_row["sample_id"],
                    "candidate_sample_id": candidate_observation.sample_id,
                    "mssd_mm": mssd_mm,
                    "normalized_mssd": normalized,
                    "mspd_px": mspd_px,
                }
            )
    if audit_outliers:
        raise RuntimeError(
            f"Cross-view GT audit failed for {len(audit_outliers)} pairs"
        )

    groups_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(groups_path, groups)
    write_jsonl_atomic(candidate_manifest_path, candidate_rows)

    available_distribution = Counter(
        int(group["available_view_count"]) for group in groups
    )
    additional_slot_sources = Counter(
        str(view["prediction_source"])
        for group in groups
        for view in group["views"][1:]
    )
    unique_additional_ids = {
        str(view["sample_id"])
        for group in groups
        for view in group["views"][1:]
    }
    pilot_groups: dict[str, str] = {}
    pilot_candidate_ids: set[str] = set()
    for object_id in sorted({int(group["object_id"]) for group in groups}):
        object_groups = [
            group for group in groups if int(group["object_id"]) == object_id
        ]
        ordered = sorted(
            object_groups,
            key=lambda group: (
                -int(group["available_additional_view_count"]),
                str(group["group_id"]),
            ),
        )
        group = next(
            (
                item
                for item in ordered
                if any(
                    view["prediction_source"] == "m2"
                    for view in item["views"][1:]
                )
            ),
            ordered[0],
        )
        pilot_groups[str(object_id)] = str(group["group_id"])
        pilot_candidate_ids.update(
            str(view["sample_id"])
            for view in group["views"][1:]
            if view["prediction_source"] == "m2"
        )

    groups_summary = {
        "schema_version": M2_SCHEMA_VERSION,
        "group_count": len(groups),
        "target_count": len(m1_rows),
        "object_count": len({int(group["object_id"]) for group in groups}),
        "target_visibility_bin_counts": dict(
            sorted(Counter(group["target_visibility_bin"] for group in groups).items())
        ),
        "available_view_count_distribution": {
            str(key): value for key, value in sorted(available_distribution.items())
        },
        "eligible_additional_view_count_distribution": {
            str(key): value
            for key, value in sorted(
                Counter(
                    int(group["eligible_additional_view_count"])
                    for group in groups
                ).items()
            )
        },
        "additional_view_slot_count": sum(additional_slot_sources.values()),
        "additional_view_slot_source_counts": dict(
            sorted(additional_slot_sources.items())
        ),
        "unique_additional_view_identity_count": len(unique_additional_ids),
        "unique_m1_reused_additional_identity_count": len(
            unique_additional_ids & set(m1_prediction_index)
        ),
        "deduplicated_missing_candidate_count": len(candidate_rows),
        "pilot_group_ids_by_object": pilot_groups,
        "pilot_candidate_count": len(pilot_candidate_ids),
        "pilot_candidate_ids": sorted(pilot_candidate_ids),
        "selection": {
            "max_additional_views": M2_MAX_ADDITIONAL_VIEWS,
            "minimum_visible_fraction": M2_MIN_VISIBLE_FRACTION,
            "requires_nonempty_decoded_visible_mask": True,
            "ordering": (
                "target first, then greedy camera-centre farthest-point sampling"
            ),
            "uses_gt_pose_error_for_order": False,
            "uses_gt_visible_fraction_for_order": False,
        },
        "oracle_cross_view_association": {
            "uses_ground_truth": True,
            "global_one_to_one_matching": True,
            "max_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
            "minimum_ambiguity_margin_mm": (
                ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
            ),
            "ambiguous_match_count": 0,
        },
        "source_sha256": {
            "m1_manifest": sha256_file(manifest_path),
            "m1_predictions": sha256_file(predictions_path),
        },
        "m1_batch_fingerprint": m1_prediction_metadata["batch_fingerprint"],
    }
    write_json_atomic(groups_summary_path, groups_summary)

    aggregate_camera_fields = {
        "rotation": dict(
            sorted(
                sum(
                    (
                        Counter(item["rotation_key_counts"])
                        for item in scene_camera_audits
                    ),
                    Counter(),
                ).items()
            )
        ),
        "translation": dict(
            sorted(
                sum(
                    (
                        Counter(item["translation_key_counts"])
                        for item in scene_camera_audits
                    ),
                    Counter(),
                ).items()
            )
        ),
    }
    association_sample_count = len(all_target_pair_association_distances_mm)
    extrinsics_audit = {
        "schema_version": M2_SCHEMA_VERSION,
        "status": "pass",
        "dataset": "xyzibd",
        "split": "val",
        "sensor_modality": MODALITY,
        "target_count": len(m1_rows),
        "scene_count": len(target_scene_ids),
        "official_field_audit": {
            "requested_bop_canonical_names": [
                "cam_R_w2c",
                "cam_t_w2c",
            ],
            "xyzibd_published_names": ["R_w2c", "t_w2c"],
            "resolved_field_counts": aggregate_camera_fields,
            "translation_input_unit": "millimetres",
            "internal_pose_unit": "metres",
            "direction": "world_to_camera",
            "formulae": {
                "model_to_camera": "T_c_m = T_c_w @ T_w_m",
                "model_to_world": "T_w_m = inverse(T_c_w) @ T_c_m",
                "candidate_to_target": (
                    "T_target_m = T_target_w @ inverse(T_candidate_w) "
                    "@ T_candidate_m"
                ),
                "rigid_inverse": "literal homogeneous numpy.linalg.inv(T)",
            },
        },
        "camera_scene_audits": scene_camera_audits,
        "all_m1_targets_have_official_extrinsics": True,
        "association": {
            "method": (
                "per-scene, per-object Hungarian matching of GT model centres "
                "in the shared world frame"
            ),
            "oracle_benchmark_association": True,
            "global_one_to_one": True,
            "target_candidate_pair_count": association_sample_count,
            "assigned_world_center_distance_mm": percentile_summary(
                all_target_pair_association_distances_mm
            ),
            "second_nearest_world_center_distance_mm": percentile_summary(
                all_target_pair_second_nearest_mm
            ),
            "ambiguity_margin_mm": percentile_summary(
                all_target_pair_ambiguity_margins_mm
            ),
            "ambiguous_match_count": 0,
            "thresholds": {
                "maximum_assigned_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
                "minimum_ambiguity_margin_mm": (
                    ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
                ),
            },
            "per_scene": scene_association_audits,
        },
        "official_bop_gt_transform_residual": {
            "toolkit_commit": toolkit_sha,
            "pair_count": len(selected_cross_view_pairs),
            "mssd_mm": percentile_summary(mssd_values),
            "normalized_mssd_by_diameter": percentile_summary(
                normalized_mssd_values
            ),
            "mspd_px": percentile_summary(mspd_values),
            "failure_thresholds": {
                "maximum_mssd_mm": GT_AUDIT_MAX_MSSD_MM,
                "maximum_normalized_mssd": GT_AUDIT_MAX_NORMALIZED_MSSD,
                "maximum_mspd_px": GT_AUDIT_MAX_MSPD_PX,
            },
            "outlier_count": len(audit_outliers),
            "outliers": audit_outliers,
        },
        "outputs": {
            "groups": str(groups_path),
            "groups_sha256": sha256_file(groups_path),
            "groups_summary": str(groups_summary_path),
            "candidate_manifest": str(candidate_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        },
        "source_sha256": groups_summary["source_sha256"],
    }
    write_json_atomic(extrinsics_audit_path, extrinsics_audit)

    print(
        f"pass: groups={len(groups)} selected_pairs="
        f"{len(selected_cross_view_pairs)} missing_candidates={len(candidate_rows)}"
    )
    print(
        "GT residual maxima: "
        f"MSSD={max(mssd_values):.9f} mm "
        f"MSPD={max(mspd_values):.9f} px"
    )
    print(f"saved: {extrinsics_audit_path}")
    print(f"saved: {groups_path}")
    print(f"saved: {groups_summary_path}")
    print(f"saved: {candidate_manifest_path}")


if __name__ == "__main__":
    main()
