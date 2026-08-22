"""Contracts for label-byte extraction after a label-free central catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from r4a_development_repair.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)

from . import PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r4a.development-slice.protocol.v1"
PREFREEZE_SCHEMA = "poseloop.r4a.development-slice.prefreeze.v1"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError("Development-slice protocol must be an object")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Development-slice protocol identity mismatch")
    if protocol.get("state") != "frozen_before_selected_development_label_bytes_download":
        raise ContractError("Development-slice protocol is not frozen")
    source = protocol.get("source_catalog", {})
    for key in ("catalog_sha256", "v2_protocol_sha256", "v2_prefreeze_receipt_sha256"):
        if not _valid_sha256(source.get(key)):
            raise ContractError(f"Invalid source catalog binding: {key}")
    plan = protocol.get("slice_plan", {})
    if not _valid_sha256(plan.get("sha256")):
        raise ContractError("Invalid slice-plan hash")
    if plan.get("scene_ids") != [0, 1, 2] or plan.get("image_ids_per_scene") != [0, 1, 2, 3]:
        raise ContractError("Slice selection drifted")
    if plan.get("target_policy") != "all_mask_visib_entries_for_selected_images":
        raise ContractError("Slice target policy drifted")
    limits = protocol.get("download_contract", {})
    if limits.get("whole_archive_download_permitted") is not False:
        raise ContractError("Whole archive download became permitted")
    if limits.get("maximum_compressed_entry_bytes") != 1610612736:
        raise ContractError("Compressed slice stop gate changed")
    if limits.get("maximum_target_count") != 128:
        raise ContractError("Target count stop gate changed")
    required_transport = {
        "range_chunk_bytes": 8388608,
        "maximum_attempts_per_chunk": 4,
        "retry_http_statuses": [429, 500, 502, 503, 504],
        "retry_backoff_seconds": [2, 5, 10],
        "persistent_chunk_cache": True,
    }
    for key, expected in required_transport.items():
        if limits.get(key) != expected:
            raise ContractError(f"Slice transport contract changed: {key}")
    boundaries = protocol.get("immutable_boundaries", {})
    if boundaries.get("xyzibd_val_access_permitted") is not False:
        raise ContractError("XYZ-IBD validation access became permitted")
    if boundaries.get("official_evaluator_invocation_permitted") is not False:
        raise ContractError("Evaluator invocation became permitted during extraction")
    return protocol


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    plan_path: Path,
    catalog_path: Path,
    v2_receipt_path: Path,
    implementation_commit: str,
    repository_tree: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    source = protocol["source_catalog"]
    plan = protocol["slice_plan"]
    actual = {
        "catalog_sha256": sha256_file(catalog_path),
        "v2_prefreeze_receipt_sha256": sha256_file(v2_receipt_path),
        "plan_sha256": sha256_file(plan_path),
    }
    expected = {
        "catalog_sha256": source["catalog_sha256"],
        "v2_prefreeze_receipt_sha256": source["v2_prefreeze_receipt_sha256"],
        "plan_sha256": plan["sha256"],
    }
    if actual != expected:
        raise ContractError(f"Slice pre-freeze source hash mismatch: {actual!r}")
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
        "selection": plan,
        "download_contract": protocol["download_contract"],
        "development_label_bytes_downloaded_at_freeze": 0,
        "development_label_files_opened_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "official_evaluator_invocation_count": 0,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt
