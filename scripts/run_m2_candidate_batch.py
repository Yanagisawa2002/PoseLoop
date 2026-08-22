#!/usr/bin/env python3
"""Run only the deduplicated PoseLoop M2 candidate views with FoundationPose."""

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
    validate_manifest as validate_m1_manifest,
)
from run_xyzibd_smoke import (
    FOUNDATIONPOSE_COMMIT,
    git_commit,
    package_version,
    verify_checkpoints,
    verify_dataset_structure,
)
from m2_common import ensure_raw_artifact_path


M2_BATCH_SCHEMA_VERSION = 1
EXPECTED_GROUP_COUNT = 300
EXPECTED_OBJECT_COUNT = 15


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "artifacts" / "m2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=artifact_dir / "candidate_manifest.jsonl",
        help="Deduplicated M2-only candidate rows in the M1 manifest schema.",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=artifact_dir / "groups.jsonl",
        help="The fixed 300-target M2 acquisition groups.",
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
        help=(
            "Attempt the union of new views from one deterministic group per object. "
            "The output still uses the full-universe fingerprint."
        ),
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
        help="Replace an existing output and log with a new batch.",
    )
    return parser.parse_args()


def validate_candidate_manifest(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate arbitrary-count candidate rows against the exact M1 input contract."""
    indexed: dict[str, dict[str, Any]] = {}
    frames: set[tuple[int, int, int, int]] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate candidate sample ID at row {index}")
        if int(row.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Invalid manifest schema for candidate {sample_id}")
        if row.get("sensor_modality") != MODALITY:
            raise ValueError(f"M2 supports only RealSense: {sample_id}")

        identity = (
            int(row["scene_id"]),
            int(row["image_id"]),
            int(row["gt_instance_index"]),
            int(row["object_id"]),
        )
        expected_sample_id = stable_sample_id(*identity)
        if sample_id != expected_sample_id:
            raise ValueError(
                f"Candidate sample ID does not encode its identity: {sample_id}"
            )
        if identity in frames:
            raise ValueError(f"Repeated candidate instance identity: {identity}")
        frames.add(identity)

        visible_fraction = float(row["visible_fraction"])
        if not 0.10 <= visible_fraction <= 1.01:
            raise ValueError(f"Invalid visible fraction for {sample_id}")
        if int(row["visible_mask_pixel_count"]) <= 0:
            raise ValueError(f"Candidate declares an empty mask: {sample_id}")
        depth_ratio = float(row["valid_depth_ratio_inside_mask"])
        if not 0.0 <= depth_ratio <= 1.0:
            raise ValueError(f"Invalid depth ratio for {sample_id}")
        camera_matrix = np.asarray(row["camera_intrinsics_row_major"], dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError(f"Invalid camera matrix for {sample_id}")
        assert_pose(
            np.asarray(row["gt_model_to_camera_pose_m"], dtype=np.float64),
            f"candidate GT {sample_id}",
            GT_ROTATION_ATOL,
        )
        indexed[sample_id] = row
    return indexed


def load_and_validate_m1(
    manifest_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_rows = load_jsonl(manifest_path)
    manifest_index = validate_m1_manifest(manifest_rows)
    prediction_records = load_jsonl(predictions_path)
    if not prediction_records or prediction_records[0].get("record_type") != "metadata":
        raise ValueError("M1 predictions do not start with metadata")
    metadata = prediction_records[0]
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("M1 prediction metadata does not match the M1 manifest")

    prediction_index: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(prediction_records[1:], start=2):
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected M1 prediction record at line {line_number}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in manifest_index:
            raise ValueError(f"M1 predictions contain unknown sample {sample_id}")
        if sample_id in prediction_index:
            raise ValueError(f"M1 predictions contain duplicate sample {sample_id}")
        if row.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"M1 prediction has invalid status for {sample_id}")
        if row.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError(f"M1 prediction has invalid schema for {sample_id}")
        manifest_row = manifest_index[sample_id]
        for field in (
            "scene_id",
            "image_id",
            "gt_instance_index",
            "object_id",
            "sensor_modality",
        ):
            if row.get(field) != manifest_row.get(field):
                raise ValueError(
                    f"M1 prediction identity mismatch for {sample_id}: {field}"
                )
        prediction_gt = np.asarray(
            row.get("gt_model_to_camera_pose_m"), dtype=np.float64
        )
        manifest_gt = np.asarray(
            manifest_row["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        if prediction_gt.shape != (4, 4) or not np.allclose(
            prediction_gt,
            manifest_gt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"M1 prediction GT mismatch for {sample_id}")
        prediction_index[sample_id] = row
    if set(prediction_index) != set(manifest_index):
        missing = sorted(set(manifest_index) - set(prediction_index))
        raise ValueError(
            f"M1 predictions are incomplete: {len(prediction_index)}/"
            f"{len(manifest_index)}; first missing={missing[:1]}"
        )
    return manifest_index, prediction_index


def _view_m2_ids(group: dict[str, Any]) -> list[str]:
    return [
        str(view["sample_id"])
        for view in group["views"]
        if view.get("prediction_source") == "m2"
    ]


def validate_groups(
    groups: list[dict[str, Any]],
    candidate_index: dict[str, dict[str, Any]],
    m1_manifest_index: dict[str, dict[str, Any]],
    m1_prediction_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(groups) != EXPECTED_GROUP_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GROUP_COUNT} M2 groups, got {len(groups)}"
        )
    group_ids: set[str] = set()
    object_counts: Counter[int] = Counter()
    declared_m2_ids: set[str] = set()
    for row_number, group in enumerate(groups, start=1):
        group_id = str(group.get("group_id", ""))
        target_sample_id = str(group.get("target_sample_id", ""))
        if not group_id or group_id in group_ids:
            raise ValueError(f"Missing or duplicate group ID at row {row_number}")
        if group_id != target_sample_id:
            raise ValueError(f"Group ID is not its target sample ID: {group_id}")
        if target_sample_id not in m1_manifest_index:
            raise ValueError(f"Unknown M1 target in group {group_id}")
        group_ids.add(group_id)

        object_id = int(group["object_id"])
        object_counts[object_id] += 1
        target_manifest_row = m1_manifest_index[target_sample_id]
        if object_id != int(target_manifest_row["object_id"]):
            raise ValueError(f"Target object mismatch in group {group_id}")
        if int(group["scene_id"]) != int(target_manifest_row["scene_id"]):
            raise ValueError(f"Target scene mismatch in group {group_id}")

        views = group.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"Group has no views: {group_id}")
        if len(views) > 5:
            raise ValueError(f"Group exceeds the five-view budget: {group_id}")
        if int(group["available_view_count"]) != len(views):
            raise ValueError(f"available_view_count mismatch for {group_id}")
        if int(group["available_additional_view_count"]) != len(views) - 1:
            raise ValueError(f"available_additional_view_count mismatch for {group_id}")
        ranks = [int(view["acquisition_rank"]) for view in views]
        if ranks != list(range(len(views))):
            raise ValueError(f"Non-contiguous acquisition ranks for {group_id}")
        target_view = views[0]
        if (
            str(target_view.get("sample_id")) != target_sample_id
            or target_view.get("prediction_source") != "m1"
        ):
            raise ValueError(f"Rank-zero target view is invalid for {group_id}")
        if group.get("target_visibility_bin") != target_manifest_row.get(
            "visibility_bin"
        ):
            raise ValueError(f"Target visibility bin mismatch for {group_id}")

        local_ids: set[str] = set()
        for view in views:
            sample_id = str(view.get("sample_id", ""))
            if not sample_id or sample_id in local_ids:
                raise ValueError(f"Duplicate/empty view in group {group_id}")
            local_ids.add(sample_id)
            source = view.get("prediction_source")
            if source == "m1":
                source_index = m1_manifest_index
                if sample_id not in m1_prediction_index:
                    raise ValueError(
                        f"Group {group_id} references absent M1 prediction {sample_id}"
                    )
            elif source == "m2":
                source_index = candidate_index
                declared_m2_ids.add(sample_id)
            else:
                raise ValueError(
                    f"Invalid prediction_source in group {group_id}: {source}"
                )
            if sample_id not in source_index:
                raise ValueError(
                    f"Group {group_id} references absent {source} row {sample_id}"
                )
            source_row = source_index[sample_id]
            for field in (
                "scene_id",
                "image_id",
                "gt_instance_index",
                "object_id",
            ):
                if int(view[field]) != int(source_row[field]):
                    raise ValueError(
                        f"Group {group_id} view identity mismatch for "
                        f"{sample_id}: {field}"
                    )
            if int(view["object_id"]) != object_id:
                raise ValueError(f"Mixed object IDs in group {group_id}")
            if int(view["scene_id"]) != int(group["scene_id"]):
                raise ValueError(f"Mixed scene IDs in group {group_id}")
            if not np.isclose(
                float(view["visible_fraction"]),
                float(source_row["visible_fraction"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"View visibility mismatch for {group_id}/{sample_id}")
            if int(view["visible_mask_pixel_count"]) != int(
                source_row["visible_mask_pixel_count"]
            ):
                raise ValueError(f"View mask area mismatch for {group_id}/{sample_id}")
            assert_pose(
                np.asarray(view["camera_world_to_camera_pose_m"], dtype=np.float64),
                f"group camera pose {group_id}/{sample_id}",
                GT_ROTATION_ATOL,
            )
            view_gt = np.asarray(view["gt_model_to_camera_pose_m"], dtype=np.float64)
            assert_pose(
                view_gt,
                f"group GT {group_id}/{sample_id}",
                GT_ROTATION_ATOL,
            )
            source_gt = np.asarray(
                source_row["gt_model_to_camera_pose_m"], dtype=np.float64
            )
            if not np.allclose(view_gt, source_gt, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Group GT differs from source manifest for {group_id}/{sample_id}"
                )

    if len(object_counts) != EXPECTED_OBJECT_COUNT or set(object_counts.values()) != {
        20
    }:
        raise ValueError(f"Expected 15 objects x 20 groups, got {dict(object_counts)}")
    if set(group_ids) != set(m1_manifest_index):
        raise ValueError("Group targets do not exactly reproduce the M1 target set")
    if declared_m2_ids != set(candidate_index):
        missing = sorted(set(candidate_index) - declared_m2_ids)
        extra = sorted(declared_m2_ids - set(candidate_index))
        raise ValueError(
            "M2 group/candidate universe mismatch: "
            f"unreferenced={missing[:3]} absent={extra[:3]}"
        )
    overlap = set(candidate_index) & set(m1_manifest_index)
    if overlap:
        raise ValueError(
            f"Candidate manifest would rerun existing M1 rows: {sorted(overlap)[:3]}"
        )
    return groups


def select_pilot_rows(
    candidate_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_groups: list[dict[str, Any]] = []
    selected_candidate_ids: set[str] = set()
    object_ids = sorted({int(group["object_id"]) for group in groups})
    for object_id in object_ids:
        ranked = sorted(
            (group for group in groups if int(group["object_id"]) == object_id),
            key=lambda group: (
                -int(group["available_additional_view_count"]),
                str(group["group_id"]),
            ),
        )
        with_new_candidates = [group for group in ranked if _view_m2_ids(group)]
        chosen = with_new_candidates[0] if with_new_candidates else ranked[0]
        selected_groups.append(chosen)
        selected_candidate_ids.update(_view_m2_ids(chosen))

    if len(selected_groups) != EXPECTED_OBJECT_COUNT:
        raise ValueError(f"Pilot selected {len(selected_groups)} groups, expected 15")
    selected_rows = [
        row for row in candidate_rows if str(row["sample_id"]) in selected_candidate_ids
    ]
    if {str(row["sample_id"]) for row in selected_rows} != selected_candidate_ids:
        raise ValueError("Pilot groups reference candidates outside the manifest")
    return selected_rows, selected_groups


def expected_metadata(
    args: argparse.Namespace,
    candidate_manifest_path: Path,
    groups_path: Path,
    m1_manifest_path: Path,
    m1_predictions_path: Path,
    candidate_rows: list[dict[str, Any]],
    foundationpose_root: Path,
    foundationpose_commit: str,
    checkpoints: dict[str, Any],
    warmup_row: dict[str, Any] | None,
    pilot_groups: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [str(row["sample_id"]) for row in candidate_rows]
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
        "run_m2_candidate_batch.py": sha256_file(Path(__file__).resolve()),
        "m2_common.py": sha256_file(script_dir / "m2_common.py"),
        "run_xyzibd_batch.py": sha256_file(script_dir / "run_xyzibd_batch.py"),
        "m1_common.py": sha256_file(script_dir / "m1_common.py"),
        "run_xyzibd_smoke.py": sha256_file(script_dir / "run_xyzibd_smoke.py"),
    }
    provenance_hashes = {
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "groups_sha256": sha256_file(groups_path),
        "m1_manifest_sha256": sha256_file(m1_manifest_path),
        "m1_predictions_sha256": sha256_file(m1_predictions_path),
    }
    config = {
        "m2_batch_schema_version": M2_BATCH_SCHEMA_VERSION,
        "prediction_row_schema_version": BATCH_SCHEMA_VERSION,
        "poseloop_source_sha256": source_hashes,
        "runtime_contract": runtime_contract,
        **provenance_hashes,
        "candidate_universe_sample_ids_sha256": canonical_sha256(candidate_ids),
        "candidate_universe_sample_count": len(candidate_ids),
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
        "target_rows_rerun": False,
    }
    pilot_group_ids = [str(group["group_id"]) for group in pilot_groups]
    effective_ids = [str(row["sample_id"]) for row in effective_rows]
    return {
        "record_type": "metadata",
        "schema_version": M2_BATCH_SCHEMA_VERSION,
        "batch_fingerprint": canonical_sha256(config),
        "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initial_invocation": "pilot" if args.pilot else "full",
        "candidate_manifest": str(candidate_manifest_path),
        "groups": str(groups_path),
        "m1_manifest": str(m1_manifest_path),
        "m1_predictions": str(m1_predictions_path),
        **provenance_hashes,
        "candidate_universe_sample_count": len(candidate_ids),
        "candidate_universe_sample_ids_sha256": config[
            "candidate_universe_sample_ids_sha256"
        ],
        "initial_pilot_selection": (
            {
                "rule": (
                    "one group per object ranked by descending available additional "
                    "views then group ID, preferring a group with an M2 candidate"
                ),
                "group_ids": pilot_group_ids,
                "group_ids_sha256": canonical_sha256(pilot_group_ids),
                "m2_candidate_sample_count": len(effective_ids),
                "m2_candidate_sample_ids_sha256": canonical_sha256(effective_ids),
                "m1_reused_views_count": sum(
                    1
                    for group in pilot_groups
                    for view in group["views"]
                    if view["prediction_source"] == "m1"
                ),
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
        "result_contract": {
            "allowed_statuses": sorted(ALLOWED_STATUSES),
            "failures_are_completed_rows": True,
            "failed_rows_are_not_retried": True,
            "resume_may_add_a_new_process_session": True,
            "shared_model_and_warmup_counts_are_per_process": True,
            "pilot_and_full_resume_share_full_universe_fingerprint": True,
            "output_contains_only_m2_candidate_rows": True,
            "m1_predictions_are_reused_without_rerun": True,
            "official_evaluation_failure_denominator": (
                "Failures are misses at every threshold."
            ),
        },
    }


def prepare_output(
    output_path: Path,
    metadata: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    resume: bool,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
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
        raise ValueError("Existing M2 output does not start with metadata")
    stored_metadata = existing[0]
    if stored_metadata.get("schema_version") != M2_BATCH_SCHEMA_VERSION:
        raise ValueError("Existing M2 metadata schema version does not match")
    for field in (
        "batch_fingerprint",
        "candidate_manifest_sha256",
        "groups_sha256",
        "m1_manifest_sha256",
        "m1_predictions_sha256",
        "candidate_universe_sample_count",
        "candidate_universe_sample_ids_sha256",
    ):
        if stored_metadata.get(field) != metadata.get(field):
            raise ValueError(f"Existing M2 metadata mismatch: {field}")

    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(existing[1:], start=2):
        if row.get("record_type") != "prediction":
            raise ValueError(f"Unexpected M2 record at output line {line_number}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in candidate_index:
            raise ValueError(
                f"Existing M2 output contains non-candidate sample {sample_id}"
            )
        if sample_id in completed:
            raise ValueError(
                f"Existing M2 output contains duplicate sample {sample_id}"
            )
        if row.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status in existing output for {sample_id}")
        if row.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"Invalid prediction schema in existing output for {sample_id}"
            )
        manifest_row = candidate_index[sample_id]
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
        stored_gt = np.asarray(row.get("gt_model_to_camera_pose_m"), dtype=np.float64)
        manifest_gt = np.asarray(
            manifest_row["gt_model_to_camera_pose_m"], dtype=np.float64
        )
        if stored_gt.shape != (4, 4) or not np.allclose(
            stored_gt,
            manifest_gt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Existing row GT mismatch for {sample_id}")
        runtime = row.get("registration_seconds")
        if runtime is not None and (
            not np.isfinite(float(runtime)) or float(runtime) < 0
        ):
            raise ValueError(f"Invalid existing runtime for {sample_id}")
        if row["status"] == "success":
            predicted_pose = np.asarray(
                row.get("predicted_model_to_camera_pose_m"), dtype=np.float64
            )
            assert_pose(predicted_pose, f"existing prediction {sample_id}")
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
                raise ValueError(
                    f"Invalid existing FoundationPose score for {sample_id}"
                )
        elif "predicted_model_to_camera_pose_m" in row:
            raise ValueError(
                f"Failed existing row unexpectedly has a pose: {sample_id}"
            )
        completed[sample_id] = row
    return completed


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (repo_root / "artifacts" / "m2").resolve()
    candidate_manifest_path = ensure_raw_artifact_path(
        args.candidate_manifest,
        repo_root,
    )
    groups_path = ensure_raw_artifact_path(args.groups, repo_root)
    m1_manifest_path = args.m1_manifest.resolve()
    m1_predictions_path = args.m1_predictions.resolve()
    data_root = args.data_root.resolve()
    foundationpose_root = args.foundationpose_root.resolve()
    output_path = ensure_raw_artifact_path(args.output, repo_root)
    log_path = ensure_raw_artifact_path(args.log, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and args.overwrite and not args.pilot:
        raise ValueError(
            "A full M2 invocation may not overwrite an existing pilot/output; "
            "use --resume so completed candidate rows are preserved."
        )

    for label, path in (
        ("candidate manifest", candidate_manifest_path),
        ("groups", groups_path),
        ("M1 manifest", m1_manifest_path),
        ("M1 predictions", m1_predictions_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    candidate_rows = load_jsonl(candidate_manifest_path)
    candidate_index = validate_candidate_manifest(candidate_rows)
    m1_manifest_index, m1_prediction_index = load_and_validate_m1(
        m1_manifest_path,
        m1_predictions_path,
    )
    groups = validate_groups(
        load_jsonl(groups_path),
        candidate_index,
        m1_manifest_index,
        m1_prediction_index,
    )
    pilot_rows: list[dict[str, Any]] = []
    pilot_groups: list[dict[str, Any]] = []
    if args.pilot:
        pilot_rows, pilot_groups = select_pilot_rows(candidate_rows, groups)
    effective_rows = pilot_rows if args.pilot else list(candidate_rows)
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
        m1_manifest_path,
        m1_predictions_path,
        candidate_rows,
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
    pending_rows = [
        row for row in effective_rows if str(row["sample_id"]) not in completed
    ]
    configure_logging(log_path, append=args.resume and log_path.exists())
    invocation = "pilot" if args.pilot else "full"
    logging.info(
        "PoseLoop M2 %s invocation: universe=%d output_completed=%d "
        "selected_now=%d pending_now=%d",
        invocation,
        len(candidate_rows),
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
    logging.info(
        "M1 reuse: manifest=%s predictions=%s",
        sha256_file(m1_manifest_path),
        sha256_file(m1_predictions_path),
    )
    logging.info("Output: %s", output_path)
    logging.info("Dataset: %s", dataset)
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
    logging.info("FoundationPose imports complete; durable M2 logging restored")

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
    if not args.pilot and len(completed) != universe_count:
        raise RuntimeError(
            f"Full M2 batch incomplete: {len(completed)}/{universe_count}"
        )
    counts = Counter(row["status"] for row in completed.values())
    logging.info(
        "PoseLoop M2 %s invocation complete: newly_attempted=%d "
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
