"""Immutable v3 pre-freeze contract after the frozen v2 HTTP 503."""

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


PROTOCOL_SCHEMA = "poseloop.r4a.development.protocol.v3"
PREFREEZE_SCHEMA = "poseloop.r4a.development.prefreeze.v3"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError("R4-A v3 protocol must be an object")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 protocol identity mismatch")
    if protocol.get("state") != "frozen_before_any_v3_development_label_access":
        raise ContractError("R4-A v3 protocol is not pre-frozen")
    v2 = protocol.get("immutable_v2_failure", {})
    for key in ("protocol_sha256", "prefreeze_receipt_sha256", "job_log_sha256", "job_exit_sha256"):
        if not _valid_sha256(v2.get(key)):
            raise ContractError(f"Invalid immutable v2 hash: {key}")
    if v2.get("exit_code") != 1 or v2.get("rerun_permitted") is not False:
        raise ContractError("R4-A v2 failure must remain non-rerunnable")
    if v2.get("last_completed_cumulative_bytes") != 109314104:
        raise ContractError("R4-A v2 frozen progress changed")
    role = protocol.get("role_declaration", {})
    if role.get("role") != "DEVELOPMENT_ONLY" or role.get("sealed_split_eligible") is not False:
        raise ContractError("R4-A v3 role must remain development-only")
    selection = protocol.get("development_split", {}).get("selection", {})
    if selection.get("scene_ids") != [0, 1, 2] or selection.get("image_ids_per_scene") != [0, 1, 2, 3]:
        raise ContractError("R4-A v3 selection differs from v1/v2")
    if selection.get("selection_expansion_after_label_access_permitted") is not False:
        raise ContractError("R4-A v3 selection expansion became permitted")
    network = protocol.get("development_split", {}).get("network_contract", {})
    required = {
        "whole_archive_download_permitted": False,
        "http_range_required": True,
        "central_directory_max_bytes": 536870912,
        "range_chunk_bytes": 8388608,
        "maximum_attempts_per_chunk": 4,
        "retry_http_statuses": [429, 500, 502, 503, 504],
        "retry_backoff_seconds": [2, 5, 10],
        "persistent_chunk_cache": True,
    }
    for key, expected in required.items():
        if network.get(key) != expected:
            raise ContractError(f"R4-A v3 network contract changed: {key}")
    boundaries = protocol.get("immutable_boundaries", {})
    for key in (
        "xyzibd_val_evaluator_permitted",
        "xyzibd_val_gt_access_permitted",
        "score_guided_pose_selection_permitted",
        "checkpoint_change_permitted",
    ):
        if boundaries.get(key) is not False:
            raise ContractError(f"R4-A v3 immutable boundary changed: {key}")
    return protocol


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    v2_protocol_path: Path,
    implementation_commit: str,
    repository_tree: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if sha256_file(v2_protocol_path) != protocol["immutable_v2_failure"]["protocol_sha256"]:
        raise ContractError("R4-A v2 protocol changed")
    for value in (implementation_commit, repository_tree):
        if len(value) != 40 or not all(c in "0123456789abcdef" for c in value):
            raise ContractError("Invalid implementation Git binding")
    receipt = {
        "schema_version": PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": protocol["state"],
        "protocol_sha256": sha256_file(protocol_path),
        "immutable_v2_failure": protocol["immutable_v2_failure"],
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "development_label_access_count_at_freeze": 0,
        "xyzibd_val_label_access_count": 0,
        "official_evaluator_invocation_count": 0,
        "selection": protocol["development_split"]["selection"],
        "archives": protocol["development_split"]["archives"],
        "network_contract": protocol["development_split"]["network_contract"],
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt
