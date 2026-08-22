"""Immutable v2 pre-freeze contract for the central-directory repair."""

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


PROTOCOL_SCHEMA = "poseloop.r4a.development.protocol.v2"
PREFREEZE_SCHEMA = "poseloop.r4a.development.prefreeze.v2"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError("R4-A v2 protocol must be an object")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("R4-A v2 protocol schema mismatch")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v2 protocol ID mismatch")
    if protocol.get("state") != "frozen_before_any_v2_development_label_access":
        raise ContractError("R4-A v2 protocol is not pre-frozen")
    base = protocol.get("immutable_v1_protocol", {})
    if base.get("protocol_id") != "poseloop.r4a.xyzibd-train-pbr.development.coverage-geometry.v1":
        raise ContractError("R4-A v1 protocol binding changed")
    if not _valid_sha256(base.get("protocol_sha256")):
        raise ContractError("R4-A v1 protocol SHA is invalid")
    failure = protocol.get("immutable_v1_catalog_failure", {})
    for key in ("prefreeze_receipt_sha256", "job_log_sha256", "job_exit_sha256"):
        if not _valid_sha256(failure.get(key)):
            raise ContractError(f"R4-A v1 failure hash is invalid: {key}")
    if failure.get("exit_code") != 1 or failure.get("rerun_permitted") is not False:
        raise ContractError("R4-A v1 failed job must remain non-rerunnable")
    role = protocol.get("role_declaration", {})
    if role.get("role") != "DEVELOPMENT_ONLY" or role.get("sealed_split_eligible") is not False:
        raise ContractError("R4-A v2 role must remain development-only")
    selection = protocol.get("development_split", {}).get("selection", {})
    if selection.get("scene_ids") != [0, 1, 2] or selection.get("image_ids_per_scene") != [0, 1, 2, 3]:
        raise ContractError("R4-A v2 data selection differs from frozen v1")
    if selection.get("selection_expansion_after_label_access_permitted") is not False:
        raise ContractError("R4-A v2 selection expansion must remain forbidden")
    network = protocol.get("development_split", {}).get("network_contract", {})
    required = {
        "whole_archive_download_permitted": False,
        "http_range_required": True,
        "central_directory_max_bytes": 536870912,
        "range_chunk_bytes": 8388608,
        "selected_compressed_payload_max_bytes": 1610612736,
        "total_download_stop_gate_bytes": 2281701376,
    }
    for key, expected in required.items():
        if network.get(key) != expected:
            raise ContractError(f"R4-A v2 network contract changed: {key}")
    boundaries = protocol.get("immutable_boundaries", {})
    for key in (
        "xyzibd_val_evaluator_permitted",
        "xyzibd_val_gt_access_permitted",
        "score_guided_pose_selection_permitted",
        "checkpoint_change_permitted",
    ):
        if boundaries.get(key) is not False:
            raise ContractError(f"R4-A v2 immutable boundary changed: {key}")
    return protocol


def build_prefreeze_receipt(
    *,
    protocol_path: Path,
    base_protocol_path: Path,
    implementation_commit: str,
    repository_tree: str,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    base = protocol["immutable_v1_protocol"]
    if sha256_file(base_protocol_path) != base["protocol_sha256"]:
        raise ContractError("R4-A v1 protocol no longer matches its frozen hash")
    for name, value in (("implementation_commit", implementation_commit), ("repository_tree", repository_tree)):
        if len(value) != 40 or not all(character in "0123456789abcdef" for character in value):
            raise ContractError(f"Invalid Git SHA: {name}")
    receipt = {
        "schema_version": PREFREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "state": protocol["state"],
        "protocol_sha256": sha256_file(protocol_path),
        "immutable_v1_protocol_sha256": base["protocol_sha256"],
        "immutable_v1_failure": protocol["immutable_v1_catalog_failure"],
        "implementation_commit": implementation_commit,
        "repository_tree": repository_tree,
        "development_label_access_count_at_freeze": 0,
        "xyzibd_val_label_access_count": 0,
        "official_evaluator_invocation_count": 0,
        "selection": protocol["development_split"]["selection"],
        "archives": protocol["development_split"]["archives"],
        "network_contract": protocol["development_split"]["network_contract"],
        "immutable_boundaries": protocol["immutable_boundaries"],
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output_path, receipt)
    return receipt
