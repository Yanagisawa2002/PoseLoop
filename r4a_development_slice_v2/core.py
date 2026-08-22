"""Frozen protocol and receipt for the exact R4-A gray development slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from r4a_development_repair.core import ContractError, canonical_sha256, read_json, sha256_file, write_json_atomic

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development-slice.protocol.v2"
PREFREEZE_SCHEMA = "poseloop.r4a.development-slice.prefreeze.v2"


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A slice-v2 protocol identity mismatch")
    if protocol.get("state") != "frozen_before_selected_development_label_bytes_download":
        raise ContractError("R4-A slice-v2 protocol is not pre-frozen")
    failure = protocol.get("immutable_slice_v1_plan_failure", {})
    for key in ("plan_log_sha256", "plan_exit_sha256"):
        if not _hash(failure.get(key)):
            raise ContractError(f"Invalid slice-v1 failure hash: {key}")
    if failure.get("exit_code") != 1 or failure.get("rerun_permitted") is not False:
        raise ContractError("Slice-v1 plan failure must remain non-rerunnable")
    source = protocol.get("source_catalog", {})
    for key in ("catalog_sha256", "v4_protocol_sha256", "v4_prefreeze_receipt_sha256"):
        if not _hash(source.get(key)):
            raise ContractError(f"Invalid source catalog hash: {key}")
    plan = protocol.get("slice_plan", {})
    if not _hash(plan.get("sha256")):
        raise ContractError("Invalid slice-v2 plan hash")
    required_plan = {
        "scene_ids": [0, 1, 2],
        "image_ids_per_scene": [0],
        "input_modality": "gray",
        "target_policy": "all_mask_visib_entries_for_selected_images",
        "target_count": 173,
    }
    for key, expected in required_plan.items():
        if plan.get(key) != expected:
            raise ContractError(f"Slice-v2 plan changed: {key}")
    limits = protocol.get("download_contract", {})
    required_limits = {
        "whole_archive_download_permitted": False,
        "maximum_compressed_entry_bytes": 536870912,
        "maximum_network_bytes_including_headers": 67108864,
        "maximum_target_count": 192,
        "range_chunk_bytes": 8388608,
        "maximum_attempts_per_chunk": 4,
        "retry_http_statuses": [429, 500, 502, 503, 504],
        "retry_backoff_seconds": [2, 5, 10],
        "short_body_retryable": True,
        "persistent_chunk_cache": True,
    }
    for key, expected in required_limits.items():
        if limits.get(key) != expected:
            raise ContractError(f"Slice-v2 download contract changed: {key}")
    boundary = protocol.get("immutable_boundaries", {})
    if boundary.get("xyzibd_val_access_permitted") is not False or boundary.get("official_evaluator_invocation_permitted") is not False:
        raise ContractError("Slice-v2 validation/evaluator boundary changed")
    return protocol


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    plan_path: Path,
    catalog_path: Path,
    v4_receipt_path: Path,
    implementation_commit: str,
    repository_tree: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    actual = {
        "catalog_sha256": sha256_file(catalog_path),
        "v4_prefreeze_receipt_sha256": sha256_file(v4_receipt_path),
        "plan_sha256": sha256_file(plan_path),
    }
    expected = {
        "catalog_sha256": protocol["source_catalog"]["catalog_sha256"],
        "v4_prefreeze_receipt_sha256": protocol["source_catalog"]["v4_prefreeze_receipt_sha256"],
        "plan_sha256": protocol["slice_plan"]["sha256"],
    }
    if actual != expected:
        raise ContractError(f"Slice-v2 source hashes changed: {actual!r}")
    for value in (implementation_commit, repository_tree):
        if len(value) != 40 or not all(c in "0123456789abcdef" for c in value):
            raise ContractError("Invalid implementation Git binding")
    receipt = {
        "schema_version": PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": protocol["state"],
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "source_hashes": actual,
        "slice_plan": protocol["slice_plan"],
        "download_contract": protocol["download_contract"],
        "development_label_bytes_downloaded_at_freeze": 0,
        "development_label_files_opened_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "official_evaluator_invocation_count": 0,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt
