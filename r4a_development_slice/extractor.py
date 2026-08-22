"""Resumable exact-entry extraction under a frozen slice plan."""

from __future__ import annotations

import binascii
from pathlib import Path, PurePosixPath
from typing import Any

from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic
from r4a_development_repair.splitzip import Entry, Part, SplitZip
from r4a_development_repair_v3.splitzip import CachedRetryRangeClient

from .core import load_protocol


def _safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "val" in {part.lower() for part in value.parts}:
        raise ContractError(f"Unsafe or validation entry path: {name}")
    return Path(*value.parts)


def _crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = binascii.crc32(block, value)
    return value & 0xFFFFFFFF


def extract_plan(
    *,
    protocol_path: Path,
    plan_path: Path,
    output_root: Path,
    cache_root: Path,
    receipt_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    plan = read_json(plan_path)
    receipt = read_json(receipt_path)
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("Slice pre-freeze receipt does not bind this protocol")
    if receipt.get("source_hashes", {}).get("plan_sha256") != sha256_file(plan_path):
        raise ContractError("Slice pre-freeze receipt does not bind this plan")
    if receipt.get("development_label_bytes_downloaded_at_freeze") != 0:
        raise ContractError("Label bytes were downloaded before slice freeze")
    archives = protocol["archives"]
    parts = [Part(row["filename"], row["url"], int(row["bytes"]), row["sha256"]) for row in archives]
    limits = protocol["download_contract"]
    client = CachedRetryRangeClient(
        maximum_bytes=int(limits["maximum_network_bytes_including_headers"]),
        chunk_bytes=int(limits["range_chunk_bytes"]),
        cache_root=cache_root.resolve(),
        maximum_attempts=int(limits["maximum_attempts_per_chunk"]),
        retry_statuses=limits["retry_http_statuses"],
        backoff_seconds=limits["retry_backoff_seconds"],
    )
    archive = SplitZip(parts, client)
    extracted: list[dict[str, Any]] = []
    root = output_root.resolve()
    for index, row in enumerate(plan["entries"]):
        relative = _safe_relative(str(row["name"]))
        output = root / relative
        expected_size = int(row["uncompressed_size"])
        expected_crc = int(row["crc32"])
        if output.is_file() and output.stat().st_size == expected_size and _crc32_file(output) == expected_crc:
            item = {
                "entry": row["name"],
                "output_path": str(output),
                "uncompressed_bytes": expected_size,
                "sha256": sha256_file(output),
                "status": "resumed_verified",
            }
        else:
            entry = Entry(
                name=str(row["name"]),
                disk_start=int(row["disk_start"]),
                local_offset=int(row["local_offset"]),
                compressed_size=int(row["compressed_size"]),
                uncompressed_size=expected_size,
                compression=int(row["compression"]),
                crc32=expected_crc,
                flags=int(row["flags"]),
            )
            item = archive.extract_entry(entry, output)
            item["status"] = "downloaded_verified"
        item["index"] = index
        item["role"] = row["role"]
        extracted.append(item)
        print(f"ENTRY_PROGRESS {index + 1}/{len(plan['entries'])} {item['status']} {row['name']}", flush=True)
    result = {
        "schema_version": "poseloop.r4a.development-slice.extraction.v1",
        "protocol_sha256": sha256_file(protocol_path),
        "plan_sha256": sha256_file(plan_path),
        "prefreeze_receipt_sha256": sha256_file(receipt_path),
        "entry_count": len(extracted),
        "network_bytes_downloaded": client.bytes_downloaded,
        "cache_bytes_reused": client.bytes_from_cache,
        "network_requests": client.requests,
        "files": extracted,
        "xyzibd_val_access_count": 0,
        "official_evaluator_invocation_count": 0,
    }
    write_json_atomic(result_path, result)
    return result
