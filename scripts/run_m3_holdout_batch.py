#!/usr/bin/env python3
"""Run the immutable PoseLoop M3 holdout universe with FoundationPose."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh

from m1_common import (
    ALLOWED_STATUSES,
    BATCH_SCHEMA_VERSION,
    FOUNDATIONPOSE_ITERATIONS,
    GT_ROTATION_ATOL,
    INFERENCE_SEED,
    MANIFEST_SCHEMA_VERSION,
    MODALITY,
    append_jsonl_durable,
    assert_pose,
    canonical_sha256,
    load_jsonl,
    raw_pose_errors,
    sha256_file,
    stable_sample_id,
    write_jsonl_atomic,
)
from run_xyzibd_batch import (
    SCORE_SEMANTICS,
    SystemicEnvironmentFailure,
    attempt_sample,
    choose_warmup_row,
    clear_per_sample_estimator_state,
    configure_logging,
    load_mesh,
    run_warmup,
)
from run_xyzibd_smoke import (
    FOUNDATIONPOSE_COMMIT,
    git_commit,
    package_version,
    verify_checkpoints,
    verify_dataset_structure,
)


M3_BATCH_SCHEMA_VERSION = 1
EXPECTED_OBJECT_COUNT = 15
MINIMUM_GROUP_COUNT = 100
VIEWS_PER_GROUP = 5


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "artifacts" / "m3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=artifact_dir / "candidate_manifest.jsonl",
        help="Every deduplicated M3 holdout view in the M1 inference schema.",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=artifact_dir / "holdout_groups.jsonl",
        help="The fixed five-view M3 holdout acquisition groups.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "POSELOOP_XYZIBD_ROOT",
                Path(os.environ.get("POSELOOP_DATA_ROOT", Path.home() / "datasets"))
                / "xyzibd",
            )
        ),
    )
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "view_predictions.jsonl",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=artifact_dir / "inference.log",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Attempt all five views from one deterministic group per object.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest/group universe without CUDA or output writes.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume",
        action="store_true",
        help="Validate the immutable full-universe batch and attempt missing rows.",
    )
    action.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output; accepted only for a pilot invocation.",
    )
    return parser.parse_args()


def ensure_m3_artifact_path(path: Path, repo_root: Path) -> Path:
    """Keep every M3 runtime output in the ignored artifact subtree."""
    resolved = path.resolve()
    artifact_root = (repo_root / "artifacts" / "m3").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(f"Raw M3 artifact must stay under {artifact_root}: {resolved}")
    return resolved


def _require_string(row: dict[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {field} for {label}")
    return value


def validate_candidate_manifest(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the full deduplicated holdout universe and M3 linkage fields."""
    if not rows:
        raise ValueError("M3 candidate manifest is empty")
    indexed: dict[str, dict[str, Any]] = {}
    identities: set[tuple[int, int, int, int]] = set()
    object_ids: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        sample_id = _require_string(row, "sample_id", f"manifest row {row_number}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate M3 candidate sample ID: {sample_id}")
        if int(row.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Invalid manifest schema for candidate {sample_id}")
        if row.get("sensor_modality") != MODALITY:
            raise ValueError(f"M3 supports only RealSense: {sample_id}")

        identity = (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        if sample_id != stable_sample_id(*identity):
            raise ValueError(
                f"Candidate sample ID does not encode its identity: {sample_id}"
            )
        if identity in identities:
            raise ValueError(f"Repeated M3 candidate instance identity: {identity}")
        identities.add(identity)
        object_ids.add(identity[3])

        physical_instance_id = _require_string(
            row,
            "physical_instance_id",
            sample_id,
        )
        group_id = _require_string(row, "m3_group_id", sample_id)
        role = _require_string(row, "m3_role", sample_id)
        rank = int(row.get("m3_acquisition_rank", -1))
        if rank not in range(VIEWS_PER_GROUP):
            raise ValueError(f"Invalid acquisition rank for {sample_id}: {rank}")
        expected_role = "target" if rank == 0 else "additional"
        if role != expected_role:
            raise ValueError(
                f"Role/rank mismatch for {sample_id}: {role!r} at rank {rank}"
            )
        if rank == 0 and group_id != sample_id:
            raise ValueError(f"Rank-zero candidate is not its group ID: {sample_id}")
        if int(row.get("m3_group_reference_count", -1)) != 1:
            raise ValueError(
                f"Candidate must belong to exactly one group: {sample_id}"
            )
        if int(row.get("selection_index", -1)) != row_number - 1:
            raise ValueError(
                f"Non-contiguous candidate selection index for {sample_id}"
            )
        if not physical_instance_id.strip():
            raise ValueError(f"Blank physical instance ID for {sample_id}")

        visible_fraction = float(row["visible_fraction"])
        if not 0.10 <= visible_fraction <= 1.01:
            raise ValueError(f"Invalid visible fraction for {sample_id}")
        if int(row["visible_mask_pixel_count"]) <= 0:
            raise ValueError(f"Candidate declares an empty mask: {sample_id}")
        depth_ratio = float(row["valid_depth_ratio_inside_mask"])
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"Invalid depth ratio for {sample_id}")
        if int(row["valid_depth_pixel_count"]) < 4:
            raise ValueError(f"Candidate has too few valid depth pixels: {sample_id}")
        if int(row["image_height"]) <= 0 or int(row["image_width"]) <= 0:
            raise ValueError(f"Candidate has invalid image dimensions: {sample_id}")
        depth_scale = float(row["raw_depth_scale"])
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError(f"Invalid raw depth scale for {sample_id}")
        for path_field in ("rgb_path", "depth_path", "mask_path", "model_path"):
            _require_string(row, path_field, sample_id)

        camera_matrix = np.asarray(
            row["camera_intrinsics_row_major"],
            dtype=np.float64,
        )
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError(f"Invalid camera matrix for {sample_id}")
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            f"candidate GT audit pose {sample_id}",
            GT_ROTATION_ATOL,
        )
        indexed[sample_id] = row

    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError(
            f"M3 candidate manifest must cover {EXPECTED_OBJECT_COUNT} objects; "
            f"found={sorted(object_ids)}"
        )
    return indexed


def _assert_matching_pose(
    first: Any,
    second: Any,
    *,
    label: str,
) -> None:
    first_pose = np.asarray(first, dtype=np.float64)
    second_pose = np.asarray(second, dtype=np.float64)
    assert_pose(first_pose, label, GT_ROTATION_ATOL)
    if second_pose.shape != (4, 4) or not np.allclose(
        first_pose,
        second_pose,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Pose mismatch for {label}")


def validate_groups(
    groups: list[dict[str, Any]],
    candidate_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the holdout group topology against the exact manifest universe."""
    if len(groups) < MINIMUM_GROUP_COUNT:
        raise ValueError(
            f"Expected at least {MINIMUM_GROUP_COUNT} M3 groups, got {len(groups)}"
        )
    group_ids: set[str] = set()
    physical_instance_ids: set[str] = set()
    declared_candidate_ids: set[str] = set()
    object_counts: Counter[int] = Counter()

    for row_number, group in enumerate(groups, start=1):
        group_id = _require_string(group, "group_id", f"group row {row_number}")
        target_sample_id = _require_string(group, "target_sample_id", group_id)
        physical_instance_id = _require_string(
            group,
            "physical_instance_id",
            group_id,
        )
        if group_id in group_ids:
            raise ValueError(f"Duplicate M3 group ID: {group_id}")
        if int(group.get("schema_version", -1)) != M3_BATCH_SCHEMA_VERSION:
            raise ValueError(f"Invalid M3 group schema for {group_id}")
        if physical_instance_id in physical_instance_ids:
            raise ValueError(
                f"Physical instance appears in multiple groups: {physical_instance_id}"
            )
        if group_id != target_sample_id:
            raise ValueError(f"Group ID is not its target sample ID: {group_id}")
        if target_sample_id not in candidate_index:
            raise ValueError(f"Unknown holdout target in group {group_id}")
        group_ids.add(group_id)
        physical_instance_ids.add(physical_instance_id)

        object_id = int(group["object_id"])
        scene_id = int(group["scene_id"])
        object_counts[object_id] += 1
        target_manifest_row = candidate_index[target_sample_id]
        if object_id != int(target_manifest_row["object_id"]):
            raise ValueError(f"Target object mismatch in group {group_id}")
        if scene_id != int(target_manifest_row["scene_id"]):
            raise ValueError(f"Target scene mismatch in group {group_id}")
        if physical_instance_id != str(
            target_manifest_row["physical_instance_id"]
        ):
            raise ValueError(f"Target physical instance mismatch in group {group_id}")
        if group.get("target_visibility_bin") != target_manifest_row.get(
            "visibility_bin"
        ):
            raise ValueError(f"Target visibility bin mismatch in group {group_id}")
        if not np.isclose(
            float(group["target_visible_fraction"]),
            float(target_manifest_row["visible_fraction"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Target visible fraction mismatch in group {group_id}"
            )

        views = group.get("views")
        if not isinstance(views, list) or len(views) != VIEWS_PER_GROUP:
            raise ValueError(
                f"Group {group_id} must contain exactly {VIEWS_PER_GROUP} views"
            )
        if (
            "available_view_count" in group
            and int(group["available_view_count"]) != VIEWS_PER_GROUP
        ):
            raise ValueError(f"available_view_count mismatch for {group_id}")
        if (
            "available_additional_view_count" in group
            and int(group["available_additional_view_count"]) != VIEWS_PER_GROUP - 1
        ):
            raise ValueError(
                f"available_additional_view_count mismatch for {group_id}"
            )
        ranks = [int(view["acquisition_rank"]) for view in views]
        if ranks != list(range(VIEWS_PER_GROUP)):
            raise ValueError(f"Non-contiguous acquisition ranks for {group_id}")

        local_ids: set[str] = set()
        for rank, view in enumerate(views):
            sample_id = _require_string(view, "sample_id", f"{group_id} rank {rank}")
            if sample_id in local_ids or sample_id in declared_candidate_ids:
                raise ValueError(f"Repeated view identity in M3 groups: {sample_id}")
            local_ids.add(sample_id)
            declared_candidate_ids.add(sample_id)
            if sample_id not in candidate_index:
                raise ValueError(
                    f"Group {group_id} references absent candidate {sample_id}"
                )
            if view.get("prediction_source") != "m3":
                raise ValueError(
                    f"Invalid prediction source for {group_id}/{sample_id}"
                )
            expected_role = "target" if rank == 0 else "additional"
            if view.get("m3_role") != expected_role:
                raise ValueError(f"Invalid role for {group_id}/{sample_id}")
            if rank == 0 and sample_id != target_sample_id:
                raise ValueError(f"Rank-zero target view is invalid for {group_id}")

            manifest_row = candidate_index[sample_id]
            if manifest_row["m3_group_id"] != group_id:
                raise ValueError(
                    f"Manifest group link mismatch for {group_id}/{sample_id}"
                )
            if int(manifest_row["m3_acquisition_rank"]) != rank:
                raise ValueError(
                    f"Manifest acquisition rank mismatch for {group_id}/{sample_id}"
                )
            if manifest_row["m3_role"] != expected_role:
                raise ValueError(
                    f"Manifest role mismatch for {group_id}/{sample_id}"
                )
            if str(manifest_row["physical_instance_id"]) != physical_instance_id:
                raise ValueError(
                    f"Manifest physical instance mismatch for {group_id}/{sample_id}"
                )
            for field in (
                "scene_id",
                "image_id",
                "gt_instance_index",
                "object_id",
            ):
                if int(view[field]) != int(manifest_row[field]):
                    raise ValueError(
                        f"Group identity mismatch for {group_id}/{sample_id}: {field}"
                    )
            if int(view["object_id"]) != object_id:
                raise ValueError(f"Mixed object IDs in group {group_id}")
            if int(view["scene_id"]) != scene_id:
                raise ValueError(f"Mixed scene IDs in group {group_id}")
            if str(view["physical_instance_id"]) != physical_instance_id:
                raise ValueError(
                    f"Mixed physical instance IDs in group {group_id}"
                )
            if not np.isclose(
                float(view["visible_fraction"]),
                float(manifest_row["visible_fraction"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"View visibility mismatch for {group_id}/{sample_id}"
                )
            if int(view["visible_mask_pixel_count"]) != int(
                manifest_row["visible_mask_pixel_count"]
            ):
                raise ValueError(
                    f"View mask area mismatch for {group_id}/{sample_id}"
                )
            assert_pose(
                np.asarray(
                    view["camera_world_to_camera_pose_m"],
                    dtype=np.float64,
                ),
                f"group camera pose {group_id}/{sample_id}",
                GT_ROTATION_ATOL,
            )
            _assert_matching_pose(
                view["gt_model_to_camera_pose_m"],
                manifest_row["gt_model_to_camera_pose_m"],
                label=f"group GT audit pose {group_id}/{sample_id}",
            )

    manifest_object_ids = {
        int(row["object_id"]) for row in candidate_index.values()
    }
    if (
        len(object_counts) != EXPECTED_OBJECT_COUNT
        or set(object_counts) != manifest_object_ids
    ):
        raise ValueError(
            f"M3 groups must cover the same {EXPECTED_OBJECT_COUNT} objects "
            "as the candidate manifest; "
            f"counts={dict(sorted(object_counts.items()))}"
        )
    if declared_candidate_ids != set(candidate_index):
        unreferenced = sorted(set(candidate_index) - declared_candidate_ids)
        absent = sorted(declared_candidate_ids - set(candidate_index))
        raise ValueError(
            "M3 group/candidate universe mismatch: "
            f"unreferenced={unreferenced[:3]} absent={absent[:3]}"
        )
    if len(candidate_index) != VIEWS_PER_GROUP * len(groups):
        raise ValueError(
            "M3 candidate count is not exactly five per physical-instance group"
        )
    return groups


def select_pilot_rows(
    candidate_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose one complete group per object without using visibility or results."""
    selected_groups: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    object_ids = sorted({int(group["object_id"]) for group in groups})
    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError(
            f"Pilot requires {EXPECTED_OBJECT_COUNT} objects, found {object_ids}"
        )
    for object_id in object_ids:
        object_groups = [
            group for group in groups if int(group["object_id"]) == object_id
        ]
        if not object_groups:
            raise ValueError(f"Pilot cannot cover absent object {object_id}")
        chosen = min(object_groups, key=lambda group: str(group["group_id"]))
        selected_groups.append(chosen)
        selected_ids.update(str(view["sample_id"]) for view in chosen["views"])

    selected_rows = [
        row for row in candidate_rows if str(row["sample_id"]) in selected_ids
    ]
    if len(selected_groups) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Pilot must select exactly one group per object")
    if len(selected_rows) != VIEWS_PER_GROUP * EXPECTED_OBJECT_COUNT:
        raise ValueError(
            f"Pilot must contain {VIEWS_PER_GROUP * EXPECTED_OBJECT_COUNT} rows"
        )
    if {str(row["sample_id"]) for row in selected_rows} != selected_ids:
        raise ValueError("Pilot groups reference candidates outside the manifest")
    return selected_rows, selected_groups


def expected_metadata(
    args: argparse.Namespace,
    candidate_manifest_path: Path,
    groups_path: Path,
    candidate_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    foundationpose_root: Path,
    foundationpose_commit: str,
    checkpoints: dict[str, Any],
    warmup_row: dict[str, Any] | None,
    pilot_groups: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one immutable full-universe fingerprint for pilot and full resume."""
    candidate_ids = [str(row["sample_id"]) for row in candidate_rows]
    group_ids = [str(group["group_id"]) for group in groups]
    script_dir = Path(__file__).resolve().parent
    runtime_contract = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "opencv": cv2.__version__,
        "trimesh": trimesh.__version__,
        "nvdiffrast": package_version("nvdiffrast"),
        "pytorch3d": package_version("pytorch3d"),
    }
    source_hashes = {
        "run_m3_holdout_batch.py": sha256_file(Path(__file__).resolve()),
        "run_xyzibd_batch.py": sha256_file(script_dir / "run_xyzibd_batch.py"),
        "m1_common.py": sha256_file(script_dir / "m1_common.py"),
        "run_xyzibd_smoke.py": sha256_file(script_dir / "run_xyzibd_smoke.py"),
    }
    provenance_hashes = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "groups_sha256": sha256_file(groups_path),
    }
    config = {
        "m3_batch_schema_version": M3_BATCH_SCHEMA_VERSION,
        "prediction_row_schema_version": BATCH_SCHEMA_VERSION,
        "poseloop_source_sha256": source_hashes,
        "runtime_contract": runtime_contract,
        **provenance_hashes,
        "candidate_universe_sample_ids_sha256": canonical_sha256(candidate_ids),
        "candidate_universe_sample_count": len(candidate_ids),
        "holdout_group_ids_sha256": canonical_sha256(group_ids),
        "holdout_group_count": len(group_ids),
        "foundationpose_commit_sha": foundationpose_commit,
        "checkpoint_sha256": {
            label: value["sha256"] for label, value in checkpoints["models"].items()
        },
        "registration_iteration_count": FOUNDATIONPOSE_ITERATIONS,
        "inference_seed": INFERENCE_SEED,
        "sensor_modality": MODALITY,
        "mesh_input_unit": "millimetres",
        "mesh_scale_to_metres": 0.001,
        "raw_depth_to_millimetres": "raw_depth * depth_scale",
        "depth_scale_to_metres": 0.001,
        "inference_symmetry_tfs": None,
        "warmup_registration_count_per_process_with_pending_rows": 1,
        "per_sample_retry_count": 0,
        "gt_pose_passed_to_foundationpose": False,
    }
    pilot_group_ids = [str(group["group_id"]) for group in pilot_groups]
    effective_ids = [str(row["sample_id"]) for row in effective_rows]
    return {
        "record_type": "metadata",
        "schema_version": M3_BATCH_SCHEMA_VERSION,
        "batch_fingerprint": canonical_sha256(config),
        "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initial_invocation": "pilot" if args.pilot else "full",
        "candidate_manifest": str(candidate_manifest_path),
        "groups": str(groups_path),
        **provenance_hashes,
        "candidate_universe_sample_count": len(candidate_ids),
        "candidate_universe_sample_ids_sha256": config[
            "candidate_universe_sample_ids_sha256"
        ],
        "holdout_group_count": len(group_ids),
        "holdout_group_ids_sha256": config["holdout_group_ids_sha256"],
        "initial_pilot_selection": (
            {
                "rule": (
                    "one complete group per object, selected by ascending group ID; "
                    "no visibility, pose error, or inference result is consulted"
                ),
                "group_ids": pilot_group_ids,
                "group_ids_sha256": canonical_sha256(pilot_group_ids),
                "candidate_sample_count": len(effective_ids),
                "candidate_sample_ids_sha256": canonical_sha256(effective_ids),
            }
            if args.pilot
            else None
        ),
        "warmup": (
            {
                "sample_id": str(warmup_row["sample_id"]),
                "included_in_runtime_statistics": False,
                "performed_once_before_new_timed_attempts_per_process": True,
            }
            if warmup_row is not None
            else None
        ),
        "foundationpose": {
            "root": str(foundationpose_root),
            "commit_sha": foundationpose_commit,
            "per_process_predictor_instantiation_count": {
                "ScorePredictor": 1,
                "PoseRefinePredictor": 1,
            },
            "per_process_raster_context_instantiation_count": 1,
            "per_process_estimator_instantiation_count": 1,
            "score_semantics": SCORE_SEMANTICS,
        },
        "checkpoints": checkpoints,
        "runtime_environment": {
            **runtime_contract,
            "cuda_device_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
        },
        "inference_input_contract": {
            "foundationpose_inputs": [
                "camera intrinsics",
                "RGB",
                "metric depth",
                "visible mask",
                "known object ID",
                "known CAD model",
            ],
            "gt_pose_passed_to_foundationpose": False,
            "gt_pose_uses": [
                "pre-inference unit-consistency audit",
                "post-inference diagnostic errors",
                "later evaluation",
            ],
            "gt_pose_substitution_on_failure": False,
        },
        "result_contract": {
            "allowed_statuses": sorted(ALLOWED_STATUSES),
            "failures_are_completed_rows": True,
            "failed_rows_are_not_retried": True,
            "resume_may_add_a_new_process_session": True,
            "shared_model_and_warmup_counts_are_per_process": True,
            "pilot_and_full_resume_share_full_universe_fingerprint": True,
            "output_contains_every_m3_holdout_candidate_row_after_full_run": True,
            "all_rows_are_attempted_exactly_once": True,
            "registration_seconds_is_synchronized_foundationpose_wall_time": True,
            "official_evaluation_failure_denominator": (
                "Failures are misses at every threshold."
            ),
        },
    }


def _validate_existing_prediction(
    row: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    sample_id: str,
) -> None:
    if row.get("status") not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid status in existing output for {sample_id}")
    if row.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid prediction schema in existing output for {sample_id}"
        )
    for field in (
        "scene_id",
        "image_id",
        "gt_instance_index",
        "object_id",
        "sensor_modality",
    ):
        if row.get(field) != manifest_row.get(field):
            raise ValueError(
                f"Existing row identity mismatch for {sample_id}: {field}"
            )
    _assert_matching_pose(
        row.get("gt_model_to_camera_pose_m"),
        manifest_row["gt_model_to_camera_pose_m"],
        label=f"existing GT audit pose {sample_id}",
    )
    for field in (
        "visible_fraction",
        "valid_depth_ratio_inside_mask",
    ):
        if not np.isclose(
            float(row.get(field)),
            float(manifest_row[field]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Existing row mismatch for {sample_id}: {field}")
    if int(row.get("visible_mask_pixel_count", -1)) != int(
        manifest_row["visible_mask_pixel_count"]
    ):
        raise ValueError(
            f"Existing row mismatch for {sample_id}: visible_mask_pixel_count"
        )
    if row.get("visibility_bin") != manifest_row.get("visibility_bin"):
        raise ValueError(f"Existing row mismatch for {sample_id}: visibility_bin")

    runtime = row.get("registration_seconds")
    if runtime is not None and (
        not np.isfinite(float(runtime)) or float(runtime) < 0
    ):
        raise ValueError(f"Invalid existing runtime for {sample_id}")
    if row["status"] == "success":
        if runtime is None:
            raise ValueError(f"Successful row lacks registration time: {sample_id}")
        predicted_pose = np.asarray(
            row.get("predicted_model_to_camera_pose_m"),
            dtype=np.float64,
        )
        assert_pose(predicted_pose, f"existing prediction {sample_id}")
        manifest_gt = np.asarray(
            manifest_row["gt_model_to_camera_pose_m"],
            dtype=np.float64,
        )
        expected_translation, expected_rotation = raw_pose_errors(
            predicted_pose,
            manifest_gt,
        )
        if not np.isclose(
            float(row.get("translation_error_mm")),
            expected_translation,
            rtol=1e-6,
            atol=1e-6,
        ) or not np.isclose(
            float(row.get("raw_rotation_error_degrees")),
            expected_rotation,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"Existing diagnostic errors mismatch for {sample_id}")
        scores = (
            row.get("foundationpose_top_score"),
            row.get("foundationpose_top_score_margin"),
        )
        if any(
            value is not None and not np.isfinite(float(value)) for value in scores
        ):
            raise ValueError(f"Invalid FoundationPose score for {sample_id}")
        if int(row.get("pose_hypothesis_count", 0)) <= 0:
            raise ValueError(f"Invalid hypothesis count for {sample_id}")
        if row.get("foundationpose_score_semantics") != SCORE_SEMANTICS:
            raise ValueError(f"Invalid score semantics for {sample_id}")
    else:
        if row["status"] in {"inference_error", "nonfinite_pose"} and runtime is None:
            raise ValueError(f"Timed failure lacks registration time: {sample_id}")
        if "predicted_model_to_camera_pose_m" in row:
            raise ValueError(
                f"Failed existing row unexpectedly has a pose: {sample_id}"
            )
        error = row.get("error")
        if not isinstance(error, dict) or not error.get("type"):
            raise ValueError(f"Failed existing row lacks explicit error: {sample_id}")


def prepare_output(
    output_path: Path,
    metadata: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    resume: bool,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    """Initialize or strictly resume the durable M3 prediction stream."""
    if output_path.exists() and overwrite:
        write_jsonl_atomic(output_path, [metadata])
        return {}
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists; use --resume or --overwrite: {output_path}"
        )
    if not output_path.exists():
        write_jsonl_atomic(output_path, [metadata])
        return {}

    existing = load_jsonl(output_path)
    if not existing or existing[0].get("record_type") != "metadata":
        raise ValueError("Existing M3 output does not start with metadata")
    stored_metadata = existing[0]
    if stored_metadata.get("schema_version") != M3_BATCH_SCHEMA_VERSION:
        raise ValueError("Existing M3 metadata schema version does not match")
    for field in (
        "batch_fingerprint",
        "candidate_manifest_sha256",
        "groups_sha256",
        "candidate_universe_sample_count",
        "candidate_universe_sample_ids_sha256",
        "holdout_group_count",
        "holdout_group_ids_sha256",
    ):
        if stored_metadata.get(field) != metadata.get(field):
            raise ValueError(f"Existing M3 metadata mismatch: {field}")

    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(existing[1:], start=2):
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected M3 record at output line {line_number}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in candidate_index:
            raise ValueError(
                f"Existing M3 output contains non-candidate sample {sample_id}"
            )
        if sample_id in completed:
            raise ValueError(
                f"Existing M3 output contains duplicate sample {sample_id}"
            )
        _validate_existing_prediction(
            row,
            candidate_index[sample_id],
            sample_id=sample_id,
        )
        completed[sample_id] = row
    return completed


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (repo_root / "artifacts" / "m3").resolve()
    candidate_manifest_path = ensure_m3_artifact_path(
        args.candidate_manifest,
        repo_root,
    )
    groups_path = ensure_m3_artifact_path(args.groups, repo_root)
    output_path = ensure_m3_artifact_path(args.output, repo_root)
    log_path = ensure_m3_artifact_path(args.log, repo_root)
    data_root = args.data_root.resolve()
    foundationpose_root = args.foundationpose_root.resolve()

    if args.validate_only and (args.pilot or args.resume or args.overwrite):
        raise ValueError(
            "--validate-only cannot be combined with pilot/resume/overwrite"
        )
    if output_path.exists() and args.overwrite and not args.pilot:
        raise ValueError(
            "A full M3 invocation may not overwrite existing attempts; "
            "use --resume so every completed row is preserved."
        )
    for label, path in (
        ("candidate manifest", candidate_manifest_path),
        ("holdout groups", groups_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    candidate_rows = load_jsonl(candidate_manifest_path)
    candidate_index = validate_candidate_manifest(candidate_rows)
    groups = validate_groups(load_jsonl(groups_path), candidate_index)
    object_counts = Counter(int(group["object_id"]) for group in groups)
    if args.validate_only:
        print(
            "validated M3 holdout universe: "
            f"groups={len(groups)} candidates={len(candidate_rows)} "
            f"objects={dict(sorted(object_counts.items()))}",
            flush=True,
        )
        print(
            f"candidate_manifest_sha256={sha256_file(candidate_manifest_path)}",
            flush=True,
        )
        print(f"groups_sha256={sha256_file(groups_path)}", flush=True)
        return

    required_pilot_rows, required_pilot_groups = select_pilot_rows(
        candidate_rows,
        groups,
    )
    pilot_rows = required_pilot_rows if args.pilot else []
    pilot_groups = required_pilot_groups if args.pilot else []
    effective_rows = pilot_rows if args.pilot else list(candidate_rows)
    if not args.pilot and not output_path.is_file():
        raise FileNotFoundError(
            "The full M3 batch requires the durable all-object pilot output; "
            "run --pilot first, then run --resume."
        )
    preexisting_ids: set[str] = set()
    if args.resume and output_path.is_file():
        preexisting_ids = {
            str(row.get("sample_id", ""))
            for row in load_jsonl(output_path)[1:]
            if row.get("record_type") == "prediction"
        }
    warmup_candidates = [
        row
        for row in effective_rows
        if str(row["sample_id"]) not in preexisting_ids
    ]
    warmup_row = (
        choose_warmup_row(warmup_candidates) if warmup_candidates else None
    )

    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(
            f"FoundationPose checkout not found: {foundationpose_root}"
        )
    dataset = verify_dataset_structure(data_root)
    foundationpose_commit = git_commit(foundationpose_root)
    if foundationpose_commit != FOUNDATIONPOSE_COMMIT:
        raise SystemicEnvironmentFailure(
            f"FoundationPose commit mismatch: {foundationpose_commit}"
        )
    checkpoints = verify_checkpoints(foundationpose_root)
    if not torch.cuda.is_available():
        raise SystemicEnvironmentFailure("PyTorch cannot access CUDA")
    torch.cuda.set_device(0)

    metadata = expected_metadata(
        args,
        candidate_manifest_path,
        groups_path,
        candidate_rows,
        groups,
        foundationpose_root,
        foundationpose_commit,
        checkpoints,
        warmup_row,
        pilot_groups,
        effective_rows,
    )
    completed = prepare_output(
        output_path,
        metadata,
        candidate_index,
        args.resume,
        args.overwrite,
    )
    if not args.pilot:
        stored_metadata = load_jsonl(output_path)[0]
        if stored_metadata.get("initial_invocation") != "pilot":
            raise ValueError(
                "The M3 output was not initialized by the required pilot"
            )
        required_pilot_ids = {
            str(row["sample_id"]) for row in required_pilot_rows
        }
        missing_pilot_ids = sorted(required_pilot_ids - set(completed))
        if missing_pilot_ids:
            raise ValueError(
                "Full M3 inference is blocked until every deterministic pilot row "
                f"has an explicit result; missing={missing_pilot_ids[:3]}"
            )
    pending_rows = [
        row for row in effective_rows if str(row["sample_id"]) not in completed
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path, append=args.resume and log_path.exists())
    invocation = "pilot" if args.pilot else "full"
    logging.info(
        "PoseLoop M3 %s invocation: universe=%d groups=%d output_completed=%d "
        "selected_now=%d pending_now=%d",
        invocation,
        len(candidate_rows),
        len(groups),
        len(completed),
        len(effective_rows),
        len(pending_rows),
    )
    logging.info(
        "Candidate manifest: %s sha256=%s",
        candidate_manifest_path,
        sha256_file(candidate_manifest_path),
    )
    logging.info("Groups: %s sha256=%s", groups_path, sha256_file(groups_path))
    logging.info("Output: %s", output_path)
    logging.info("Dataset: %s", dataset)
    logging.info(
        "GT pose contract: audit/post-hoc diagnostics only; never passed to "
        "FoundationPose and never substituted for a prediction"
    )
    if args.pilot:
        logging.info(
            "Pilot groups: %s",
            [str(group["group_id"]) for group in pilot_groups],
        )
    if not pending_rows:
        counts = Counter(row["status"] for row in completed.values())
        print(
            f"nothing pending for {invocation}: output has "
            f"{len(completed)}/{len(candidate_rows)} universe rows; "
            f"statuses={dict(sorted(counts.items()))}",
            flush=True,
        )
        return
    if warmup_row is None:
        raise RuntimeError("Pending candidates exist but no warm-up row is available")

    os.chdir(foundationpose_root)
    sys.path.insert(0, str(foundationpose_root))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import set_seed

    # FoundationPose's Utils module reloads Python logging at import time.
    configure_logging(log_path, append=True)
    logging.info("FoundationPose imports complete; durable M3 logging restored")

    from m1_common import read_json

    models_info = read_json(data_root / "models" / "models_info.json")
    mesh_cache: dict[int, trimesh.Trimesh] = {}

    def object_mesh(row: dict[str, Any]) -> trimesh.Trimesh:
        object_id = int(row["object_id"])
        if object_id not in mesh_cache:
            mesh_cache[object_id] = load_mesh(row, data_root, models_info)
        return mesh_cache[object_id]

    warmup_mesh = object_mesh(warmup_row)
    try:
        set_seed(INFERENCE_SEED)
        logging.info("Instantiating shared ScorePredictor")
        scorer = ScorePredictor()
        logging.info("Instantiating shared PoseRefinePredictor")
        refiner = PoseRefinePredictor()
        logging.info("Instantiating shared CUDA rasterization context")
        glctx = dr.RasterizeCudaContext(device=0)
        logging.info("Instantiating one shared FoundationPose estimator")
        estimator = FoundationPose(
            model_pts=warmup_mesh.vertices.copy(),
            model_normals=warmup_mesh.vertex_normals.copy(),
            symmetry_tfs=None,
            mesh=warmup_mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=str(artifact_dir / "foundationpose_debug"),
        )
        current_object_id = int(warmup_row["object_id"])
        run_warmup(warmup_row, estimator)
    except Exception as exc:
        logging.exception("Shared predictor/context/estimator warm-up failed")
        raise SystemicEnvironmentFailure(
            f"Predictor/context/estimator warm-up failed: {exc}"
        ) from exc

    newly_attempted = 0
    universe_count = len(candidate_rows)
    for row in effective_rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        object_id = int(row["object_id"])
        if object_id != current_object_id:
            mesh = object_mesh(row)
            try:
                set_seed(INFERENCE_SEED)
                estimator.reset_object(
                    model_pts=mesh.vertices.copy(),
                    model_normals=mesh.vertex_normals.copy(),
                    symmetry_tfs=None,
                    mesh=mesh,
                )
                clear_per_sample_estimator_state(estimator)
                current_object_id = object_id
            except Exception as exc:
                logging.exception("Object reset failed for object %d", object_id)
                raise SystemicEnvironmentFailure(
                    f"FoundationPose object reset failed for object {object_id}: {exc}"
                ) from exc

        result = attempt_sample(row, estimator)
        append_jsonl_durable(output_path, result)
        completed[sample_id] = result
        newly_attempted += 1
        print(
            f"[{len(completed):03d}/{universe_count:03d}] object={object_id:02d} "
            f"sample={sample_id} status={result['status']} "
            f"time={result.get('registration_seconds', float('nan')):.3f}s",
            flush=True,
        )
        logging.info(
            "Completed %s status=%s registration_seconds=%s",
            sample_id,
            result["status"],
            result.get("registration_seconds"),
        )

    missing_effective = {str(row["sample_id"]) for row in effective_rows} - set(
        completed
    )
    if missing_effective:
        raise RuntimeError(
            f"Invocation exited with {len(missing_effective)} unattempted rows"
        )
    if not args.pilot and set(completed) != set(candidate_index):
        missing = sorted(set(candidate_index) - set(completed))
        extra = sorted(set(completed) - set(candidate_index))
        raise RuntimeError(
            "Full M3 batch incomplete or contaminated: "
            f"completed={len(completed)}/{universe_count} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    counts = Counter(row["status"] for row in completed.values())
    logging.info(
        "PoseLoop M3 %s invocation complete: newly_attempted=%d "
        "output_completed=%d universe=%d statuses=%s",
        invocation,
        newly_attempted,
        len(completed),
        universe_count,
        dict(sorted(counts.items())),
    )
    print(
        f"{invocation} complete: output has {len(completed)}/{universe_count} "
        f"universe rows; statuses={dict(sorted(counts.items()))}",
        flush=True,
    )
    print(f"saved: {output_path}", flush=True)
    print(f"log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
