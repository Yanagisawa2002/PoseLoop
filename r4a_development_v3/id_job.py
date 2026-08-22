"""Bounded ID-only discovery job for the R4-A v3 development slice."""

from __future__ import annotations

import binascii
from pathlib import Path
from typing import Any

from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic
from r4a_development_repair.splitzip import Entry, Part, SplitZip
from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient

from . import PROTOCOL_ID
from .contract import ID_PLAN_SCHEMA, load_protocol, safe_archive_relative, verify_prefreeze
from .id_parser import parse_scene_gt_ids, select_multi_object_targets


ID_RESULT_SCHEMA = "poseloop.r4a.development-id-result.v3"
EXACT_SELECTION_SCHEMA = "poseloop.r4a.development-exact-selection.v3"


def _parts(protocol: dict[str, Any]) -> list[Part]:
    return [
        Part(row["filename"], row["url"], int(row["bytes"]), row["sha256"])
        for row in protocol["official_distribution"]["archives"]
    ]


def _entry(row: dict[str, Any]) -> Entry:
    return Entry(
        name=str(row["name"]),
        disk_start=int(row["disk_start"]),
        local_offset=int(row["local_offset"]),
        compressed_size=int(row["compressed_size"]),
        uncompressed_size=int(row["uncompressed_size"]),
        compression=int(row["compression"]),
        crc32=int(row["crc32"]),
        flags=int(row["flags"]),
    )


def _crc32(path: Path) -> int:
    result = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result = binascii.crc32(block, result)
    return result & 0xFFFFFFFF


def discover_ids(
    *,
    protocol_path: Path,
    id_plan_path: Path,
    prefreeze_path: Path,
    output_root: Path,
    cache_root: Path,
    exact_selection_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = verify_prefreeze(prefreeze_path, schema="poseloop.r4a.development-id-prefreeze.v3", protocol_path=protocol_path)
    plan = read_json(id_plan_path)
    if plan.get("schema_version") != ID_PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("R4-A v3 ID job plan identity mismatch")
    if receipt.get("id_plan_sha256") != sha256_file(id_plan_path):
        raise ContractError("R4-A v3 ID plan changed after prefreeze")
    limits = protocol["id_discovery_download_contract"]
    client = ShortBodyRetryRangeClient(
        maximum_bytes=int(limits["maximum_network_bytes_including_headers"]),
        chunk_bytes=int(limits["range_chunk_bytes"]),
        cache_root=cache_root.resolve(),
        maximum_attempts=int(limits["maximum_attempts_per_chunk"]),
        retry_statuses=limits["retry_http_statuses"],
        backoff_seconds=limits["retry_backoff_seconds"],
    )
    archive = SplitZip(_parts(protocol), client)
    root = output_root.resolve()
    all_records = []
    opened = []
    selection = None
    for index, row in enumerate(plan["entries"]):
        path = root / safe_archive_relative(str(row["name"]))
        expected_size = int(row["uncompressed_size"])
        expected_crc = int(row["crc32"])
        if path.is_file() and path.stat().st_size == expected_size and _crc32(path) == expected_crc:
            status = "resumed_verified"
        else:
            archive.extract_entry(_entry(row), path)
            status = "downloaded_verified"
        payload = path.read_bytes()
        records, access = parse_scene_gt_ids(payload, scene_id=int(row["scene_id"]))
        all_records.append(records)
        opened.append(
            {
                "candidate_ordinal": index,
                "scene_id": int(row["scene_id"]),
                "entry": row["name"],
                "path": str(path),
                "bytes": len(payload),
                "sha256": sha256_file(path),
                "status": status,
                **access,
            }
        )
        selection = select_multi_object_targets(
            all_records,
            required_objects=int(protocol["id_only_selection"]["required_object_count"]),
            required_scenes=int(protocol["id_only_selection"]["required_scene_count"]),
            images_per_object=int(protocol["id_only_selection"]["required_images_per_object"]),
        )
        print(
            f"ID_PROGRESS {index + 1}/{len(plan['entries'])} scene={row['scene_id']} "
            f"rows={len(records)} gate={'pass' if selection else 'pending'}",
            flush=True,
        )
        if selection is not None:
            break
    gate_pass = selection is not None
    if not gate_pass:
        raise ContractError("R4-A v3 ID gate exhausted the frozen candidate prefix without five objects")
    assert selection is not None
    if selection["target_count"] != 10 or selection["unique_key_count"] != 10:
        raise ContractError("R4-A v3 exact target cap changed")
    exact = {
        "schema_version": EXACT_SELECTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "role": "DEVELOPMENT_ONLY_ID_SELECTED",
        "selection_algorithm": protocol["id_only_selection"],
        "scanned_scene_ids": [row["scene_id"] for row in opened],
        "object_ids": selection["object_ids"],
        "scene_ids": selection["scene_ids"],
        "target_count": selection["target_count"],
        "unique_key_count": selection["unique_key_count"],
        "per_object_count": selection["per_object_count"],
        "targets": selection["targets"],
        "pose_values_decoded": 0,
        "visibility_values_decoded": 0,
        "score_values_decoded": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gate_pass": True,
    }
    write_json_atomic(exact_selection_path, exact)
    result = {
        "schema_version": ID_RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "id_plan_sha256": sha256_file(id_plan_path),
        "prefreeze_sha256": sha256_file(prefreeze_path),
        "exact_selection_path": str(exact_selection_path.resolve()),
        "exact_selection_sha256": sha256_file(exact_selection_path),
        "opened_id_files": opened,
        "opened_id_file_count": len(opened),
        "network_bytes_downloaded": client.bytes_downloaded,
        "cache_bytes_reused": client.bytes_from_cache,
        "network_requests": client.requests,
        "pose_values_decoded": 0,
        "visibility_values_decoded": 0,
        "score_values_decoded": 0,
        "xyzibd_val_access_count": 0,
        "evaluator_invocation_count": 0,
        "gate_pass": True,
    }
    write_json_atomic(result_path, result)
    return result


__all__ = ["EXACT_SELECTION_SCHEMA", "ID_RESULT_SCHEMA", "discover_ids"]
