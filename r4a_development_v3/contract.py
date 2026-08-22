"""Immutable protocol, evidence, and prefreeze contracts for R4-A v3."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from r4a_development_repair.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development-multi-object.protocol.v3"
ID_PLAN_SCHEMA = "poseloop.r4a.development-id-plan.v3"
ID_PREFREEZE_SCHEMA = "poseloop.r4a.development-id-prefreeze.v3"
ASSET_PLAN_SCHEMA = "poseloop.r4a.development-asset-plan.v3"
ASSET_PREFREEZE_SCHEMA = "poseloop.r4a.development-asset-prefreeze.v3"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def git_value(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("R4-A v3 protocol schema mismatch")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 protocol identity mismatch")
    if protocol.get("state") != "frozen_before_any_new_v3_development_id_label_access":
        raise ContractError("R4-A v3 protocol is not frozen before ID access")
    role = protocol.get("role_declaration", {})
    if role.get("role") != "DEVELOPMENT_ONLY" or role.get("sealed_split_eligible") is not False:
        raise ContractError("R4-A v3 role boundary changed")
    failure = protocol.get("immutable_v2_failure", {})
    for key in (
        "protocol_sha256",
        "prefreeze_receipt_sha256",
        "job_log_sha256",
        "job_exit_sha256",
        "failure_receipt_sha256",
        "failure_receipt_lock_sha256",
        "object_diagnosis_sha256",
    ):
        if not _valid_sha256(failure.get(key)):
            raise ContractError(f"Invalid immutable v2 hash: {key}")
    if failure.get("exit_code") != 1 or failure.get("rerun_permitted") is not False:
        raise ContractError("R4-A v2 failure must remain non-rerunnable")
    catalog = protocol.get("source_catalog", {})
    if not _valid_sha256(catalog.get("sha256")) or catalog.get("bytes") != 640055050:
        raise ContractError("R4-A v3 source catalog changed")
    selection = protocol.get("id_only_selection", {})
    expected_scenes = [0, 3, 9, 12, 15, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    if selection.get("candidate_scene_ids") != expected_scenes:
        raise ContractError("R4-A v3 candidate scene order changed")
    required_selection = {
        "required_object_count": 5,
        "required_scene_count": 3,
        "required_images_per_object": 2,
        "maximum_selected_object_count": 5,
        "maximum_selected_image_count": 10,
        "maximum_selected_target_count": 10,
        "maximum_candidate_scene_gt_entries": 15,
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise ContractError(f"R4-A v3 selection gate changed: {key}")
    if selection.get("allowed_decoded_fields") != ["top_level_image_id", "obj_id", "instance_ordinal"]:
        raise ContractError("R4-A v3 decoded-field allowlist changed")
    assets = protocol.get("exact_asset_selection", {})
    if assets.get("inference_roles") != [
        "gray",
        "depth",
        "predicted_depth_component_mask",
        "selected_camera",
        "selected_public_cad",
    ]:
        raise ContractError("R4-A v3 no-GT inference roles changed")
    inference = protocol.get("bundle_contract", {}).get("inference_bundle", {})
    if inference.get("label_access_count") != 0:
        raise ContractError("R4-A v3 no-GT inference label access changed")
    if inference.get("input_mask_source_commit") != "9d75eb14fb0634490f8fe5de27a5d1138e0e7c66":
        raise ContractError("R4-A v3 input-mask implementation binding changed")
    forbidden = set(inference.get("forbidden", []))
    if not {"cam_R_m2c", "cam_t_m2c", "scene_gt", "scene_gt_info", "visib_fract", "evaluator_output"}.issubset(forbidden):
        raise ContractError("R4-A v3 no-GT forbidden-role boundary changed")
    evaluator = protocol.get("bundle_contract", {}).get("evaluator_bundle", {})
    if evaluator.get("must_not_leave_gpu_a") is not True:
        raise ContractError("R4-A v3 evaluator-only bundle isolation changed")
    boundaries = protocol.get("immutable_boundaries", {})
    for key in (
        "xyzibd_val_access_permitted",
        "xyzibd_val_evaluator_permitted",
        "xyzibd_val_score_json_access_permitted",
        "score_guided_pose_or_target_selection_permitted",
        "foundationpose_setup_on_gpu_a_permitted",
        "push_merge_tag_permitted",
        "release_or_delete_instance_or_disk_permitted",
    ):
        if boundaries.get(key) is not False:
            raise ContractError(f"R4-A v3 immutable boundary changed: {key}")
    if boundaries.get("gpu_a_poweroff_permitted_after_local_archive_hash_verification") is not True:
        raise ContractError("R4-A v3 poweroff completion gate changed")
    return protocol


def build_id_plan(*, protocol_path: Path, catalog_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if sha256_file(catalog_path) != protocol["source_catalog"]["sha256"]:
        raise ContractError("R4-A v3 catalog SHA mismatch")
    catalog = read_json(catalog_path)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ContractError("R4-A v3 catalog entries are missing")
    by_name = {str(row.get("name")): row for row in entries}
    selected: list[dict[str, Any]] = []
    for ordinal, scene_id in enumerate(protocol["id_only_selection"]["candidate_scene_ids"]):
        name = f"train_pbr/{scene_id:06d}/scene_gt.json"
        if name not in by_name:
            raise ContractError(f"R4-A v3 candidate scene_gt is absent: {name}")
        row = dict(by_name[name])
        row.update({"candidate_ordinal": ordinal, "scene_id": scene_id, "role": "id_only_scene_gt_source"})
        selected.append(row)
    compressed = sum(int(row["compressed_size"]) for row in selected)
    uncompressed = sum(int(row["uncompressed_size"]) for row in selected)
    limits = protocol["id_discovery_download_contract"]
    if compressed > int(limits["maximum_compressed_entry_bytes"]):
        raise ContractError("R4-A v3 ID plan exceeds compressed-byte gate")
    plan = {
        "schema_version": ID_PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "catalog_names_sizes_crc_offsets_only_no_new_label_bytes_opened",
        "candidate_scene_ids": protocol["id_only_selection"]["candidate_scene_ids"],
        "entry_count": len(selected),
        "compressed_entry_bytes": compressed,
        "uncompressed_entry_bytes": uncompressed,
        "entries": selected,
        "new_development_id_label_files_opened": 0,
        "pose_values_decoded": 0,
        "visibility_values_decoded": 0,
        "score_values_decoded": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
    }
    write_json_atomic(output_path, plan)
    return plan


def _evidence_hashes(
    *,
    v2_protocol_path: Path,
    v2_prefreeze_path: Path,
    v2_job_log_path: Path,
    v2_job_exit_path: Path,
    v2_failure_receipt_path: Path,
    v2_object_diagnosis_path: Path,
) -> dict[str, str]:
    return {
        "protocol_sha256": sha256_file(v2_protocol_path),
        "prefreeze_receipt_sha256": sha256_file(v2_prefreeze_path),
        "job_log_sha256": sha256_file(v2_job_log_path),
        "job_exit_sha256": sha256_file(v2_job_exit_path),
        "failure_receipt_sha256": sha256_file(v2_failure_receipt_path),
        "object_diagnosis_sha256": sha256_file(v2_object_diagnosis_path),
    }


def build_id_prefreeze_receipt(
    *,
    protocol_path: Path,
    catalog_path: Path,
    id_plan_path: Path,
    v2_protocol_path: Path,
    v2_prefreeze_path: Path,
    v2_job_log_path: Path,
    v2_job_exit_path: Path,
    v2_failure_receipt_path: Path,
    v2_object_diagnosis_path: Path,
    implementation_commit: str,
    repository_tree: str,
    repository_tracked_clean: bool,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if sha256_file(catalog_path) != protocol["source_catalog"]["sha256"]:
        raise ContractError("R4-A v3 catalog changed before ID freeze")
    plan = read_json(id_plan_path)
    if plan.get("schema_version") != ID_PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 ID plan identity mismatch")
    evidence = _evidence_hashes(
        v2_protocol_path=v2_protocol_path,
        v2_prefreeze_path=v2_prefreeze_path,
        v2_job_log_path=v2_job_log_path,
        v2_job_exit_path=v2_job_exit_path,
        v2_failure_receipt_path=v2_failure_receipt_path,
        v2_object_diagnosis_path=v2_object_diagnosis_path,
    )
    for key, value in evidence.items():
        if protocol["immutable_v2_failure"].get(key) != value:
            raise ContractError(f"R4-A v2 frozen evidence changed: {key}")
    v2_failure = read_json(v2_failure_receipt_path)
    if v2_failure.get("lock_sha256") != protocol["immutable_v2_failure"]["failure_receipt_lock_sha256"]:
        raise ContractError("R4-A v2 failure lock changed")
    if v2_failure.get("rerun_permitted") is not False:
        raise ContractError("R4-A v2 failure became rerunnable")
    if v2_job_exit_path.read_text(encoding="utf-8").strip() != "1":
        raise ContractError("R4-A v2 job exit content changed")
    if not repository_tracked_clean:
        raise ContractError("R4-A v3 repository has tracked changes")
    for value in (implementation_commit, repository_tree):
        if len(value) != 40 or not all(character in "0123456789abcdef" for character in value):
            raise ContractError("R4-A v3 Git binding is invalid")
    receipt = {
        "schema_version": ID_PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": protocol["state"],
        "role": "DEVELOPMENT_ONLY_ID_DISCOVERY",
        "protocol_sha256": sha256_file(protocol_path),
        "catalog_sha256": sha256_file(catalog_path),
        "id_plan_sha256": sha256_file(id_plan_path),
        "immutable_v2_evidence": evidence,
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "repository_tracked_clean": True,
        "selection_algorithm": protocol["id_only_selection"],
        "download_contract": protocol["id_discovery_download_contract"],
        "new_development_id_label_files_opened_at_freeze": 0,
        "pose_values_decoded_at_freeze": 0,
        "visibility_values_decoded_at_freeze": 0,
        "score_values_decoded_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt


def verify_prefreeze(path: Path, *, schema: str, protocol_path: Path) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("schema_version") != schema or receipt.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 prefreeze identity mismatch")
    unlocked = dict(receipt)
    lock = unlocked.pop("lock_sha256", None)
    if lock != canonical_sha256(unlocked):
        raise ContractError("R4-A v3 prefreeze lock mismatch")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("R4-A v3 protocol changed after freeze")
    if receipt.get("xyzibd_val_access_count") != 0 or receipt.get("evaluator_invocation_count") != 0:
        raise ContractError("R4-A v3 validation/evaluator boundary changed")
    return receipt


def build_asset_plan(
    *,
    protocol_path: Path,
    catalog_path: Path,
    exact_selection_path: Path,
    id_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Plan exact image/camera/evaluator assets without opening new label bytes."""

    protocol = load_protocol(protocol_path)
    if sha256_file(catalog_path) != protocol["source_catalog"]["sha256"]:
        raise ContractError("R4-A v3 catalog changed before asset planning")
    exact = read_json(exact_selection_path)
    if exact.get("schema_version") != "poseloop.r4a.development-exact-selection.v3":
        raise ContractError("R4-A v3 exact selection schema mismatch")
    if exact.get("protocol_sha256") != sha256_file(protocol_path) or exact.get("gate_pass") is not True:
        raise ContractError("R4-A v3 exact selection is not a passing frozen selection")
    id_result = read_json(id_result_path)
    if id_result.get("exact_selection_sha256") != sha256_file(exact_selection_path):
        raise ContractError("R4-A v3 ID result does not bind the exact selection")
    keys = sorted(
        {
            (int(row["scene_id"]), int(row["image_id"]))
            for row in exact["targets"]
        }
    )
    scenes = sorted({scene_id for scene_id, _ in keys})
    requested: dict[str, tuple[str, int, int | None]] = {}
    for scene_id in scenes:
        for filename, role in (
            ("scene_camera.json", "scene_camera"),
            ("scene_gt.json", "scene_gt"),
            ("scene_gt_info.json", "scene_gt_info"),
        ):
            requested[f"train_pbr/{scene_id:06d}/{filename}"] = (role, scene_id, None)
    for scene_id, image_id in keys:
        requested[f"train_pbr/{scene_id:06d}/gray/{image_id:06d}.png"] = ("gray", scene_id, image_id)
        requested[f"train_pbr/{scene_id:06d}/depth/{image_id:06d}.png"] = ("depth", scene_id, image_id)
    catalog = read_json(catalog_path)
    by_name = {str(row.get("name")): row for row in catalog.get("entries", [])}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ContractError(f"R4-A v3 exact assets are absent from catalog: {missing[:3]}")
    entries = []
    for name in sorted(requested):
        role, scene_id, image_id = requested[name]
        row = dict(by_name[name])
        row.update({"role": role, "scene_id": scene_id, "image_id": image_id})
        entries.append(row)
    compressed = sum(int(row["compressed_size"]) for row in entries)
    uncompressed = sum(int(row["uncompressed_size"]) for row in entries)
    limits = protocol["exact_asset_selection"]
    if any(int(row["compressed_size"]) > int(limits["maximum_compressed_entry_bytes"]) for row in entries):
        raise ContractError("R4-A v3 exact asset exceeds per-entry compressed gate")
    if any(int(row["uncompressed_size"]) > int(limits["maximum_uncompressed_entry_bytes"]) for row in entries):
        raise ContractError("R4-A v3 exact asset exceeds per-entry uncompressed gate")
    if compressed > int(limits["maximum_network_bytes_including_headers"]):
        raise ContractError("R4-A v3 exact asset plan exceeds network byte gate")
    plan = {
        "schema_version": ASSET_PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "exact_selection_plus_catalog_names_sizes_crc_offsets_no_new_label_bytes_opened",
        "protocol_sha256": sha256_file(protocol_path),
        "catalog_sha256": sha256_file(catalog_path),
        "exact_selection_sha256": sha256_file(exact_selection_path),
        "id_result_sha256": sha256_file(id_result_path),
        "scene_ids": scenes,
        "scene_image_keys": [{"scene_id": scene, "image_id": image} for scene, image in keys],
        "entry_count": len(entries),
        "compressed_entry_bytes": compressed,
        "uncompressed_entry_bytes": uncompressed,
        "entries": entries,
        "new_non_id_development_label_files_opened": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
    }
    write_json_atomic(output_path, plan)
    return plan


def build_asset_prefreeze_receipt(
    *,
    protocol_path: Path,
    catalog_path: Path,
    exact_selection_path: Path,
    id_result_path: Path,
    asset_plan_path: Path,
    id_prefreeze_path: Path,
    implementation_commit: str,
    repository_tree: str,
    repository_tracked_clean: bool,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    id_receipt = verify_prefreeze(id_prefreeze_path, schema=ID_PREFREEZE_SCHEMA, protocol_path=protocol_path)
    plan = read_json(asset_plan_path)
    if plan.get("schema_version") != ASSET_PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 asset plan identity mismatch")
    expected = {
        "catalog_sha256": sha256_file(catalog_path),
        "exact_selection_sha256": sha256_file(exact_selection_path),
        "id_result_sha256": sha256_file(id_result_path),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ContractError(f"R4-A v3 asset plan source changed: {key}")
    if expected["catalog_sha256"] != protocol["source_catalog"]["sha256"]:
        raise ContractError("R4-A v3 asset freeze catalog mismatch")
    if not repository_tracked_clean:
        raise ContractError("R4-A v3 repository has tracked changes at asset freeze")
    for value in (implementation_commit, repository_tree):
        if len(value) != 40 or not all(character in "0123456789abcdef" for character in value):
            raise ContractError("R4-A v3 asset Git binding is invalid")
    receipt = {
        "schema_version": ASSET_PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_after_id_gate_before_camera_info_pose_or_image_asset_access",
        "protocol_sha256": sha256_file(protocol_path),
        "asset_plan_sha256": sha256_file(asset_plan_path),
        **expected,
        "id_prefreeze_sha256": sha256_file(id_prefreeze_path),
        "id_prefreeze_lock_sha256": id_receipt["lock_sha256"],
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "repository_tracked_clean": True,
        "new_non_id_development_label_files_opened_at_freeze": 0,
        "inference_bundle_object_pose_gt_count_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt


def safe_archive_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "val" in {part.lower() for part in value.parts}:
        raise ContractError(f"Unsafe R4-A v3 archive path: {name}")
    return Path(*value.parts)


def deterministic_combinations(values: Sequence[int], size: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(values, size))
