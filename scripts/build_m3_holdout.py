#!/usr/bin/env python3
"""Build the physical-instance-disjoint PoseLoop M3 holdout."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_m2_groups import (
    Observation,
    associate_scene_tracks,
    load_scene_observations,
    observation_to_manifest_row,
    view_record,
)
from m1_common import (
    MODALITY,
    SELECTION_SEED,
    VISIBILITY_BIN_ORDER,
    load_jsonl,
    seeded_key,
    sha256_file,
    stable_sample_id,
    write_json_atomic,
    write_jsonl_atomic,
)
from m2_common import (
    ASSOCIATION_MAX_DISTANCE_MM,
    ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM,
    M2_MAX_ADDITIONAL_VIEWS,
    M2_MIN_VISIBLE_FRACTION,
    camera_center_world_m,
    greedy_camera_center_diversity,
)


M3_SCHEMA_VERSION = 1
M3_MIN_HOLDOUT_GROUPS = 100
M3_REQUIRED_OBJECT_COUNT = 15
M3_PREDICTION_SOURCE = "m3"


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
        "--m1-manifest",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--m2-groups",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=repo_root / "artifacts" / "m3" / "holdout_groups.jsonl",
    )
    parser.add_argument(
        "--candidate-manifest-output",
        type=Path,
        default=repo_root / "artifacts" / "m3" / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "artifacts" / "m3" / "holdout_summary.json",
    )
    parser.add_argument(
        "--leakage-audit-output",
        type=Path,
        default=repo_root / "artifacts" / "m3" / "leakage_audit.json",
    )
    return parser.parse_args()


def ensure_m3_artifact_path(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    artifact_root = (repo_root / "artifacts" / "m3").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(
            f"Raw M3 artifact must stay under {artifact_root}: {resolved}"
        )
    return resolved


def counter_json(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def visibility_counter_json(counter: Counter[str]) -> dict[str, int]:
    return {
        label: int(counter[label])
        for label in VISIBILITY_BIN_ORDER
    }


def validate_development_rows(
    m1_rows: list[dict[str, Any]],
    m2_groups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(m1_rows) != 300:
        raise ValueError(f"Expected 300 M1 targets, got {len(m1_rows)}")
    if len(m2_groups) != 300:
        raise ValueError(f"Expected 300 M2 target groups, got {len(m2_groups)}")

    m1_index: dict[str, dict[str, Any]] = {}
    for row in m1_rows:
        sample_id = str(row["sample_id"])
        expected = stable_sample_id(
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        if sample_id != expected or sample_id in m1_index:
            raise ValueError(f"Invalid or duplicate M1 target: {sample_id}")
        m1_index[sample_id] = row

    m2_index: dict[str, dict[str, Any]] = {}
    for group in m2_groups:
        sample_id = str(group["target_sample_id"])
        if sample_id != str(group["group_id"]) or sample_id in m2_index:
            raise ValueError(f"Invalid or duplicate M2 target group: {sample_id}")
        if int(group["object_id"]) != int(m1_index.get(sample_id, {}).get(
            "object_id", -1
        )):
            raise ValueError(f"M1/M2 target object mismatch: {sample_id}")
        if len(group.get("views", [])) != 5:
            raise ValueError(f"M2 group does not contain five views: {sample_id}")
        m2_index[sample_id] = group
    if set(m1_index) != set(m2_index):
        raise ValueError("M1 targets and M2 target groups do not match exactly")
    return m1_index, m2_index


def associate_catalogue(
    data_root: Path,
    scene_ids: list[int],
) -> tuple[
    dict[tuple[int, int, int], Observation],
    dict[str, list[Observation]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    observation_index: dict[tuple[int, int, int], Observation] = {}
    tracks: dict[str, list[Observation]] = {}
    camera_audits: list[dict[str, Any]] = []
    association_audits: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        observations, camera_audit = load_scene_observations(data_root, scene_id)
        indexed, scene_tracks, association_audit = associate_scene_tracks(
            observations
        )
        for (image_id, gt_index), observation in indexed.items():
            key = (scene_id, image_id, gt_index)
            if key in observation_index:
                raise ValueError(f"Duplicate observation identity: {key}")
            observation_index[key] = observation
        for track_id, values in scene_tracks.items():
            if track_id in tracks:
                raise ValueError(f"Duplicate physical instance ID: {track_id}")
            values = sorted(
                values,
                key=lambda item: (item.image_id, item.gt_instance_index),
            )
            if any(item.track_id != track_id for item in values):
                raise ValueError(f"Inconsistent track assignment: {track_id}")
            if len({item.object_id for item in values}) != 1:
                raise ValueError(f"Track spans multiple object IDs: {track_id}")
            tracks[track_id] = values
        camera_audits.append(camera_audit)
        association_audits.append({"scene_id": scene_id, **association_audit})
    return observation_index, tracks, camera_audits, association_audits


def development_track_ids(
    m1_index: dict[str, dict[str, Any]],
    m2_index: dict[str, dict[str, Any]],
    observation_index: dict[tuple[int, int, int], Observation],
) -> tuple[set[str], dict[int, set[str]], dict[str, Any]]:
    m1_tracks: set[str] = set()
    m2_tracks: set[str] = set()
    by_object: dict[int, set[str]] = defaultdict(set)
    declared_m2_tracks: set[str] = set()

    for sample_id, row in m1_index.items():
        key = (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
        )
        observation = observation_index.get(key)
        if observation is None or observation.sample_id != sample_id:
            raise KeyError(f"M1 target has no associated observation: {sample_id}")
        if observation.object_id != int(row["object_id"]):
            raise ValueError(f"M1 target object mismatch: {sample_id}")
        m1_tracks.add(observation.track_id)
        by_object[observation.object_id].add(observation.track_id)

    for sample_id, group in m2_index.items():
        target = group["views"][0]
        key = (
            int(target["scene_id"]),
            int(target["image_id"]),
            int(target["gt_instance_index"]),
        )
        observation = observation_index.get(key)
        if observation is None or observation.sample_id != sample_id:
            raise KeyError(f"M2 target has no associated observation: {sample_id}")
        declared = str(group["oracle_association"]["track_id"])
        if declared != observation.track_id:
            raise ValueError(
                f"M2 declared/recomputed track mismatch for {sample_id}: "
                f"{declared} != {observation.track_id}"
            )
        declared_m2_tracks.add(declared)
        m2_tracks.add(observation.track_id)

    if m1_tracks != m2_tracks or m2_tracks != declared_m2_tracks:
        raise ValueError("M1/M2/recomputed development track sets differ")
    return m1_tracks | m2_tracks, by_object, {
        "m1_unique_physical_instance_count": len(m1_tracks),
        "m2_unique_physical_instance_count": len(m2_tracks),
        "declared_m2_unique_physical_instance_count": len(declared_m2_tracks),
        "sets_match_exactly": True,
    }


def target_choice_key(
    track_id: str,
    observation: Observation,
    object_usage: Counter[int],
    global_usage: Counter[int],
) -> tuple[int, int, str, int, int, str]:
    return (
        int(object_usage[observation.image_id]),
        int(global_usage[observation.image_id]),
        seeded_key(
            "m3-holdout-target",
            track_id,
            observation.sample_id,
            seed=SELECTION_SEED,
        ),
        observation.image_id,
        observation.gt_instance_index,
        observation.sample_id,
    )


def decorate_view(
    view: dict[str, Any],
    *,
    track_id: str,
    role: str,
) -> dict[str, Any]:
    return {
        **view,
        "physical_instance_id": track_id,
        "m3_role": role,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    m1_path = args.m1_manifest.resolve()
    m2_groups_path = args.m2_groups.resolve()
    groups_path = ensure_m3_artifact_path(args.groups_output, repo_root)
    manifest_path = ensure_m3_artifact_path(
        args.candidate_manifest_output,
        repo_root,
    )
    summary_path = ensure_m3_artifact_path(args.summary_output, repo_root)
    leakage_path = ensure_m3_artifact_path(
        args.leakage_audit_output,
        repo_root,
    )

    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    for path in (m1_path, m2_groups_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    m1_rows = load_jsonl(m1_path)
    m2_groups = load_jsonl(m2_groups_path)
    m1_index, m2_index = validate_development_rows(m1_rows, m2_groups)
    scene_ids = sorted({int(row["scene_id"]) for row in m1_rows})
    (
        observation_index,
        tracks,
        camera_audits,
        association_audits,
    ) = associate_catalogue(data_root, scene_ids)
    development_ids, development_by_object, development_audit = (
        development_track_ids(m1_index, m2_index, observation_index)
    )

    tracks_by_object: dict[int, set[str]] = defaultdict(set)
    tracks_by_scene: dict[int, list[str]] = defaultdict(list)
    for track_id, observations in tracks.items():
        object_id = observations[0].object_id
        scene_id = observations[0].scene_id
        tracks_by_object[object_id].add(track_id)
        tracks_by_scene[scene_id].append(track_id)
    object_ids = sorted(tracks_by_object)
    if object_ids != sorted({int(row["object_id"]) for row in m1_rows}):
        raise ValueError("Associated catalogue and M1 object sets differ")

    remaining_ids = set(tracks) - development_ids
    if development_ids - set(tracks):
        raise ValueError("Development set contains an unknown physical instance")

    track_candidates: list[dict[str, Any]] = []
    track_eligibility_rows: list[dict[str, Any]] = []
    metadata_below_visibility = 0
    metadata_empty_mask = 0
    decoded_candidate_count = 0

    for scene_id in scene_ids:
        frame_cache: dict[
            tuple[int, int],
            tuple[tuple[int, int], np.ndarray],
        ] = {}
        for track_id in sorted(tracks_by_scene[scene_id]):
            if track_id not in remaining_ids:
                continue
            observations = tracks[track_id]
            valid: list[tuple[Observation, dict[str, Any]]] = []
            below_visibility = 0
            empty_declared_mask = 0
            for observation in observations:
                visible_fraction = float(
                    observation.info_entry["visib_fract"]
                )
                if (
                    not math.isfinite(visible_fraction)
                    or visible_fraction < M2_MIN_VISIBLE_FRACTION
                ):
                    below_visibility += 1
                    metadata_below_visibility += 1
                    continue
                if int(observation.info_entry["px_count_visib"]) <= 0:
                    empty_declared_mask += 1
                    metadata_empty_mask += 1
                    continue
                row = observation_to_manifest_row(
                    observation,
                    data_root,
                    selection_index=-1,
                    frame_cache=frame_cache,
                )
                if int(row["visible_mask_pixel_count"]) <= 0:
                    raise ValueError(
                        f"Decoded mask is empty despite eligibility: "
                        f"{observation.sample_id}"
                    )
                decoded_candidate_count += 1
                valid.append((observation, row))
            object_id = observations[0].object_id
            eligibility_row = {
                "physical_instance_id": track_id,
                "scene_id": observations[0].scene_id,
                "object_id": object_id,
                "associated_observation_count": len(observations),
                "valid_observation_count": len(valid),
                "below_minimum_visibility_count": below_visibility,
                "empty_declared_mask_count": empty_declared_mask,
                "holdout_capable": (
                    len(valid) >= M2_MAX_ADDITIONAL_VIEWS + 1
                ),
            }
            track_eligibility_rows.append(eligibility_row)
            if eligibility_row["holdout_capable"]:
                track_candidates.append(
                    {
                        "physical_instance_id": track_id,
                        "scene_id": observations[0].scene_id,
                        "object_id": object_id,
                        "observations": observations,
                        "valid": valid,
                    }
                )

    holdout_ids = {
        str(candidate["physical_instance_id"])
        for candidate in track_candidates
    }
    leakage = development_ids & holdout_ids
    holdout_object_ids = {
        int(candidate["object_id"]) for candidate in track_candidates
    }
    hard_gates = {
        "minimum_group_count": {
            "required": M3_MIN_HOLDOUT_GROUPS,
            "actual": len(track_candidates),
            "passed": len(track_candidates) >= M3_MIN_HOLDOUT_GROUPS,
        },
        "all_objects_present": {
            "required_object_count": M3_REQUIRED_OBJECT_COUNT,
            "actual_object_count": len(holdout_object_ids),
            "expected_object_ids": object_ids,
            "actual_object_ids": sorted(holdout_object_ids),
            "passed": (
                len(holdout_object_ids) == M3_REQUIRED_OBJECT_COUNT
                and holdout_object_ids == set(object_ids)
            ),
        },
        "physical_instance_disjoint": {
            "intersection_count": len(leakage),
            "passed": not leakage,
        },
        "all_remaining_capable_instances_selected": {
            "remaining_instance_count": len(remaining_ids),
            "capable_instance_count": len(holdout_ids),
            "passed": holdout_ids
            == {
                str(row["physical_instance_id"])
                for row in track_eligibility_rows
                if bool(row["holdout_capable"])
            },
        },
    }
    if not all(bool(gate["passed"]) for gate in hard_gates.values()):
        failures = [
            name for name, gate in hard_gates.items() if not gate["passed"]
        ]
        raise RuntimeError(f"M3 holdout hard gate failed: {failures}")

    global_target_image_usage: Counter[int] = Counter()
    object_target_image_usage: dict[int, Counter[int]] = defaultdict(Counter)
    target_by_track: dict[str, tuple[Observation, dict[str, Any]]] = {}
    for candidate in sorted(
        track_candidates,
        key=lambda item: (
            int(item["object_id"]),
            seeded_key(
                "m3-holdout-track-order",
                item["physical_instance_id"],
                seed=SELECTION_SEED,
            ),
            str(item["physical_instance_id"]),
        ),
    ):
        track_id = str(candidate["physical_instance_id"])
        object_id = int(candidate["object_id"])
        target = min(
            candidate["valid"],
            key=lambda item: target_choice_key(
                track_id,
                item[0],
                object_target_image_usage[object_id],
                global_target_image_usage,
            ),
        )
        target_by_track[track_id] = target
        image_id = target[0].image_id
        object_target_image_usage[object_id][image_id] += 1
        global_target_image_usage[image_id] += 1

    groups: list[dict[str, Any]] = []
    selected_row_by_sample: dict[str, dict[str, Any]] = {}
    for candidate in sorted(
        track_candidates,
        key=lambda item: (
            int(item["object_id"]),
            int(item["scene_id"]),
            str(item["physical_instance_id"]),
        ),
    ):
        track_id = str(candidate["physical_instance_id"])
        object_id = int(candidate["object_id"])
        target_observation, target_row = target_by_track[track_id]
        track_center = np.mean(
            np.stack(
                [
                    item.model_to_world_pose_m[:3, 3]
                    for item in candidate["observations"]
                ],
                axis=0,
            ),
            axis=0,
        )
        target_camera_center = camera_center_world_m(
            target_observation.camera_pose_m
        )
        diversity_candidates: list[dict[str, Any]] = []
        valid_row_by_sample = {
            observation.sample_id: row
            for observation, row in candidate["valid"]
        }
        for observation, _row in candidate["valid"]:
            if observation.sample_id == target_observation.sample_id:
                continue
            diversity_candidates.append(
                {
                    "sample_id": observation.sample_id,
                    "scene_id": observation.scene_id,
                    "image_id": observation.image_id,
                    "gt_instance_index": observation.gt_instance_index,
                    "object_id": observation.object_id,
                    "camera_center_world_m": camera_center_world_m(
                        observation.camera_pose_m
                    ).tolist(),
                    "_observation": observation,
                }
            )
        selected = greedy_camera_center_diversity(
            target_camera_center,
            diversity_candidates,
            count=M2_MAX_ADDITIONAL_VIEWS,
        )
        if len(selected) != M2_MAX_ADDITIONAL_VIEWS:
            raise ValueError(
                f"Track {track_id} did not yield four additional views"
            )

        target_view = decorate_view(
            view_record(
                target_observation,
                target_row,
                track_center,
                prediction_source=M3_PREDICTION_SOURCE,
                acquisition_rank=0,
                diversity_score_camera_center_distance_m=None,
            ),
            track_id=track_id,
            role="target",
        )
        views = [target_view]
        selected_row_by_sample[target_observation.sample_id] = target_row
        for acquisition_rank, selected_candidate in enumerate(
            selected,
            start=1,
        ):
            observation = selected_candidate.pop("_observation")
            row = valid_row_by_sample[observation.sample_id]
            views.append(
                decorate_view(
                    view_record(
                        observation,
                        row,
                        track_center,
                        prediction_source=M3_PREDICTION_SOURCE,
                        acquisition_rank=acquisition_rank,
                        diversity_score_camera_center_distance_m=float(
                            selected_candidate[
                                "diversity_score_camera_center_distance_m"
                            ]
                        ),
                    ),
                    track_id=track_id,
                    role="additional",
                )
            )
            selected_row_by_sample[observation.sample_id] = row

        groups.append(
            {
                "schema_version": M3_SCHEMA_VERSION,
                "group_id": target_observation.sample_id,
                "target_sample_id": target_observation.sample_id,
                "physical_instance_id": track_id,
                "scene_id": target_observation.scene_id,
                "object_id": object_id,
                "target_visibility_bin": str(target_row["visibility_bin"]),
                "target_visible_fraction": float(
                    target_row["visible_fraction"]
                ),
                "oracle_association": {
                    "method": (
                        "GT model centres transformed to world coordinates "
                        "with per-object global Hungarian one-to-one matching"
                    ),
                    "uses_ground_truth": True,
                    "deployable": False,
                    "track_id": track_id,
                    "max_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
                    "minimum_ambiguity_margin_mm": (
                        ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
                    ),
                },
                "benchmark_target_selection": {
                    "method": (
                        "seeded deterministic choice with per-object then "
                        "global image-ID load balancing"
                    ),
                    "seed": SELECTION_SEED,
                    "uses_inference_results": False,
                    "uses_gt_pose_error": False,
                    "uses_exact_visible_fraction_for_ranking": False,
                    "eligibility_min_visible_fraction": (
                        M2_MIN_VISIBLE_FRACTION
                    ),
                    "requires_nonempty_decoded_visible_mask": True,
                },
                "acquisition_order": {
                    "method": (
                        "greedy farthest-point sampling over official "
                        "world-frame camera centres"
                    ),
                    "uses_gt_pose_error": False,
                    "uses_gt_visible_fraction_for_order": False,
                    "candidate_eligibility_min_visible_fraction": (
                        M2_MIN_VISIBLE_FRACTION
                    ),
                    "tie_break": "image_id, gt_instance_index, sample_id",
                },
                "eligible_observation_count": len(candidate["valid"]),
                "eligible_additional_view_count": (
                    len(candidate["valid"]) - 1
                ),
                "available_additional_view_count": len(views) - 1,
                "available_view_count": len(views),
                "views": views,
            }
        )

    if len(groups) != len(holdout_ids):
        raise ValueError("Group count does not equal holdout physical-instance count")
    if len({str(group["group_id"]) for group in groups}) != len(groups):
        raise ValueError("M3 group IDs are not unique")
    if any(len(group["views"]) != 5 for group in groups):
        raise ValueError("Every M3 group must contain exactly five views")

    reference_counts: Counter[str] = Counter(
        str(view["sample_id"])
        for group in groups
        for view in group["views"]
    )
    if any(count != 1 for count in reference_counts.values()):
        raise ValueError("A selected observation is referenced by multiple groups")
    if set(reference_counts) != set(selected_row_by_sample):
        raise ValueError("Selected manifest cache does not match group references")

    manifest_rows: list[dict[str, Any]] = []
    for selection_index, group in enumerate(groups):
        for view in sorted(
            group["views"],
            key=lambda item: int(item["acquisition_rank"]),
        ):
            sample_id = str(view["sample_id"])
            row = dict(selected_row_by_sample[sample_id])
            row.update(
                {
                    "selection_index": len(manifest_rows),
                    "selection_seed": SELECTION_SEED,
                    "physical_instance_id": str(
                        group["physical_instance_id"]
                    ),
                    "m3_group_id": str(group["group_id"]),
                    "m3_group_selection_index": selection_index,
                    "m3_acquisition_rank": int(view["acquisition_rank"]),
                    "m3_role": str(view["m3_role"]),
                    "m3_group_reference_count": reference_counts[sample_id],
                }
            )
            manifest_rows.append(row)
    if len(manifest_rows) != len(groups) * 5:
        raise ValueError("M3 candidate manifest does not have five rows per group")

    target_visibility_counts = Counter(
        str(group["target_visibility_bin"]) for group in groups
    )
    target_visibility_by_object: dict[str, dict[str, int]] = {}
    group_count_by_object: Counter[int] = Counter()
    for object_id in object_ids:
        object_groups = [
            group for group in groups if int(group["object_id"]) == object_id
        ]
        group_count_by_object[object_id] = len(object_groups)
        target_visibility_by_object[str(object_id)] = visibility_counter_json(
            Counter(
                str(group["target_visibility_bin"])
                for group in object_groups
            )
        )

    pilot_group_ids: dict[str, str] = {}
    pilot_candidate_ids: set[str] = set()
    for object_id in object_ids:
        object_groups = [
            group for group in groups if int(group["object_id"]) == object_id
        ]
        pilot_group = min(
            object_groups,
            key=lambda group: (
                seeded_key(
                    "m3-pilot",
                    group["physical_instance_id"],
                    seed=SELECTION_SEED,
                ),
                str(group["group_id"]),
            ),
        )
        pilot_group_ids[str(object_id)] = str(pilot_group["group_id"])
        pilot_candidate_ids.update(
            str(view["sample_id"]) for view in pilot_group["views"]
        )

    source_sha256 = {
        "m1_manifest": sha256_file(m1_path),
        "m2_groups": sha256_file(m2_groups_path),
        "build_m2_groups_source": sha256_file(
            repo_root / "scripts" / "build_m2_groups.py"
        ),
        "m2_common_source": sha256_file(
            repo_root / "scripts" / "m2_common.py"
        ),
        "build_m3_holdout_source": sha256_file(Path(__file__).resolve()),
    }
    dataset_metadata_sha256: dict[str, dict[str, str]] = {}
    for scene_id in scene_ids:
        scene_dir = data_root / "val" / f"{scene_id:06d}"
        dataset_metadata_sha256[str(scene_id)] = {
            name: sha256_file(scene_dir / name)
            for name in (
                "scene_gt_realsense.json",
                "scene_gt_info_realsense.json",
                "scene_camera_realsense.json",
            )
        }

    groups_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(groups_path, groups)
    write_jsonl_atomic(manifest_path, manifest_rows)

    summary = {
        "schema_version": M3_SCHEMA_VERSION,
        "status": "pass",
        "dataset": "xyzibd",
        "split": "val",
        "sensor_modality": MODALITY,
        "holdout_group_count": len(groups),
        "holdout_target_count": len(groups),
        "object_count": len(holdout_object_ids),
        "selected_view_slot_count": len(manifest_rows),
        "deduplicated_candidate_count": len(manifest_rows),
        "holdout_group_count_by_object": counter_json(group_count_by_object),
        "target_visibility_bin_counts": visibility_counter_json(
            target_visibility_counts
        ),
        "target_visibility_bin_counts_by_object": (
            target_visibility_by_object
        ),
        "target_image_id_distribution": counter_json(
            global_target_image_usage
        ),
        "unique_target_image_id_count": len(global_target_image_usage),
        "pilot_group_ids_by_object": pilot_group_ids,
        "pilot_candidate_count": len(pilot_candidate_ids),
        "pilot_candidate_ids": sorted(pilot_candidate_ids),
        "selection": {
            "physical_instance_policy": (
                "every remaining development-disjoint associated physical "
                "instance with at least five valid observations"
            ),
            "target_selection_seed": SELECTION_SEED,
            "target_selection_order": (
                "seeded deterministic per-object/global image-ID load balance"
            ),
            "minimum_valid_observations_per_track": (
                M2_MAX_ADDITIONAL_VIEWS + 1
            ),
            "minimum_visible_fraction": M2_MIN_VISIBLE_FRACTION,
            "requires_nonempty_decoded_visible_mask": True,
            "additional_view_count": M2_MAX_ADDITIONAL_VIEWS,
            "additional_view_order": (
                "target first, then exact M2 greedy camera-centre "
                "farthest-point sampling"
            ),
            "uses_gt_pose_error_for_order": False,
            "uses_gt_visible_fraction_for_order": False,
        },
        "hard_gates": hard_gates,
        "source_sha256": source_sha256,
        "outputs": {
            "groups": str(groups_path),
            "groups_sha256": sha256_file(groups_path),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "leakage_audit": str(leakage_path),
        },
    }
    write_json_atomic(summary_path, summary)

    total_by_object = {
        object_id: tracks_by_object[object_id] for object_id in object_ids
    }
    remaining_by_object = {
        object_id: tracks_by_object[object_id] - development_by_object[object_id]
        for object_id in object_ids
    }
    holdout_by_object = {
        object_id: {
            track_id
            for track_id in holdout_ids
            if tracks[track_id][0].object_id == object_id
        }
        for object_id in object_ids
    }
    physical_counts_by_object = {
        str(object_id): {
            "catalogued": len(total_by_object[object_id]),
            "development": len(development_by_object[object_id]),
            "remaining_disjoint": len(remaining_by_object[object_id]),
            "holdout_capable_and_selected": len(
                holdout_by_object[object_id]
            ),
            "excluded_insufficient_valid_observations": (
                len(remaining_by_object[object_id])
                - len(holdout_by_object[object_id])
            ),
        }
        for object_id in object_ids
    }
    leakage_audit = {
        "schema_version": M3_SCHEMA_VERSION,
        "status": "pass",
        "dataset": "xyzibd",
        "split": "val",
        "sensor_modality": MODALITY,
        "association": {
            "method": (
                "validated M2 per-scene, per-object Hungarian matching of GT "
                "model centres in the shared world frame"
            ),
            "uses_ground_truth": True,
            "global_one_to_one": True,
            "maximum_assigned_distance_mm": ASSOCIATION_MAX_DISTANCE_MM,
            "minimum_ambiguity_margin_mm": (
                ASSOCIATION_MIN_AMBIGUITY_MARGIN_MM
            ),
            "camera_scene_audits": camera_audits,
            "per_scene": association_audits,
        },
        "development_source_validation": development_audit,
        "physical_instance_counts": {
            "catalogued": len(tracks),
            "development": len(development_ids),
            "remaining_disjoint": len(remaining_ids),
            "holdout_capable_and_selected": len(holdout_ids),
            "excluded_insufficient_valid_observations": (
                len(remaining_ids) - len(holdout_ids)
            ),
            "by_object": physical_counts_by_object,
        },
        "development_physical_instance_ids": sorted(development_ids),
        "holdout_physical_instance_ids": sorted(holdout_ids),
        "development_holdout_intersection": sorted(leakage),
        "development_holdout_intersection_count": len(leakage),
        "development_physical_instance_ids_by_object": {
            str(object_id): sorted(development_by_object[object_id])
            for object_id in object_ids
        },
        "holdout_physical_instance_ids_by_object": {
            str(object_id): sorted(holdout_by_object[object_id])
            for object_id in object_ids
        },
        "observation_eligibility": {
            "minimum_visible_fraction": M2_MIN_VISIBLE_FRACTION,
            "requires_nonempty_decoded_visible_mask": True,
            "minimum_valid_observation_count": (
                M2_MAX_ADDITIONAL_VIEWS + 1
            ),
            "decoded_metadata_eligible_observation_count": (
                decoded_candidate_count
            ),
            "below_minimum_visibility_count": metadata_below_visibility,
            "empty_declared_mask_count": metadata_empty_mask,
            "associated_observation_count_distribution": counter_json(
                Counter(
                    int(row["associated_observation_count"])
                    for row in track_eligibility_rows
                )
            ),
            "valid_observation_count_distribution": counter_json(
                Counter(
                    int(row["valid_observation_count"])
                    for row in track_eligibility_rows
                )
            ),
            "per_remaining_physical_instance": sorted(
                track_eligibility_rows,
                key=lambda row: (
                    int(row["object_id"]),
                    str(row["physical_instance_id"]),
                ),
            ),
        },
        "selected_target_audit": {
            "seed": SELECTION_SEED,
            "method": (
                "seeded deterministic choice with per-object then global "
                "image-ID load balancing"
            ),
            "target_image_id_distribution": counter_json(
                global_target_image_usage
            ),
            "unique_target_image_id_count": len(
                global_target_image_usage
            ),
            "target_visibility_bin_counts": visibility_counter_json(
                target_visibility_counts
            ),
            "target_visibility_bin_counts_by_object": (
                target_visibility_by_object
            ),
            "selected_targets_and_additional_views": [
                {
                    "group_id": str(group["group_id"]),
                    "physical_instance_id": str(
                        group["physical_instance_id"]
                    ),
                    "scene_id": int(group["scene_id"]),
                    "object_id": int(group["object_id"]),
                    "target_visibility_bin": str(
                        group["target_visibility_bin"]
                    ),
                    "target_visible_fraction": float(
                        group["target_visible_fraction"]
                    ),
                    "views": [
                        {
                            "acquisition_rank": int(
                                view["acquisition_rank"]
                            ),
                            "sample_id": str(view["sample_id"]),
                            "image_id": int(view["image_id"]),
                            "gt_instance_index": int(
                                view["gt_instance_index"]
                            ),
                            "visible_fraction": float(
                                view["visible_fraction"]
                            ),
                            "visible_mask_pixel_count": int(
                                view["visible_mask_pixel_count"]
                            ),
                            "diversity_score_camera_center_distance_m": (
                                view[
                                    "diversity_score_camera_center_distance_m"
                                ]
                            ),
                        }
                        for view in group["views"]
                    ],
                }
                for group in groups
            ],
        },
        "hard_gates": hard_gates,
        "dataset_metadata_sha256": dataset_metadata_sha256,
        "source_sha256": source_sha256,
        "outputs": {
            "groups": str(groups_path),
            "groups_sha256": sha256_file(groups_path),
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
        },
    }
    write_json_atomic(leakage_path, leakage_audit)

    print(
        f"pass: holdout_groups={len(groups)} objects={len(holdout_object_ids)} "
        f"selected_views={len(manifest_rows)} leakage={len(leakage)}"
    )
    print(
        "physical instances: "
        f"catalogued={len(tracks)} development={len(development_ids)} "
        f"remaining={len(remaining_ids)} holdout={len(holdout_ids)}"
    )
    print(
        "target visibility: "
        + " ".join(
            f"{label}={target_visibility_counts[label]}"
            for label in ("low", "mid", "high")
        )
    )
    for object_id in object_ids:
        counts = physical_counts_by_object[str(object_id)]
        print(
            f"object {object_id}: catalogued={counts['catalogued']} "
            f"development={counts['development']} "
            f"remaining={counts['remaining_disjoint']} "
            f"holdout={counts['holdout_capable_and_selected']}"
        )
    print(f"saved: {groups_path}")
    print(f"saved: {manifest_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {leakage_path}")


if __name__ == "__main__":
    main()
