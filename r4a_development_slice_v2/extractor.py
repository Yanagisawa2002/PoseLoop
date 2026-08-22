"""Resumable exact-entry transfer for the frozen gray slice."""

from __future__ import annotations

from pathlib import Path

from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic
from r4a_development_repair.splitzip import Entry, Part, SplitZip
from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient
from r4a_development_slice.extractor import _crc32_file, _safe_relative

from .core import load_protocol


def extract_plan(
    *,
    protocol_path: Path,
    plan_path: Path,
    output_root: Path,
    cache_root: Path,
    receipt_path: Path,
    result_path: Path,
) -> dict:
    protocol = load_protocol(protocol_path)
    plan = read_json(plan_path)
    receipt = read_json(receipt_path)
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("Slice-v2 receipt protocol hash mismatch")
    if receipt.get("source_hashes", {}).get("plan_sha256") != sha256_file(plan_path):
        raise ContractError("Slice-v2 receipt plan hash mismatch")
    if receipt.get("development_label_bytes_downloaded_at_freeze") != 0:
        raise ContractError("Development label bytes predate slice-v2 freeze")
    parts = [
        Part(row["filename"], row["url"], int(row["bytes"]), row["sha256"])
        for row in protocol["archives"]
    ]
    limits = protocol["download_contract"]
    client = ShortBodyRetryRangeClient(
        maximum_bytes=int(limits["maximum_network_bytes_including_headers"]),
        chunk_bytes=int(limits["range_chunk_bytes"]),
        cache_root=cache_root.resolve(),
        maximum_attempts=int(limits["maximum_attempts_per_chunk"]),
        retry_statuses=limits["retry_http_statuses"],
        backoff_seconds=limits["retry_backoff_seconds"],
    )
    archive = SplitZip(parts, client)
    root = output_root.resolve()
    extracted = []
    for index, row in enumerate(plan["entries"]):
        output = root / _safe_relative(str(row["name"]))
        size = int(row["uncompressed_size"])
        crc = int(row["crc32"])
        if output.is_file() and output.stat().st_size == size and _crc32_file(output) == crc:
            item = {"entry": row["name"], "output_path": str(output), "uncompressed_bytes": size, "sha256": sha256_file(output), "status": "resumed_verified"}
        else:
            entry = Entry(
                name=str(row["name"]),
                disk_start=int(row["disk_start"]),
                local_offset=int(row["local_offset"]),
                compressed_size=int(row["compressed_size"]),
                uncompressed_size=size,
                compression=int(row["compression"]),
                crc32=crc,
                flags=int(row["flags"]),
            )
            item = archive.extract_entry(entry, output)
            item["status"] = "downloaded_verified"
        item.update({"index": index, "role": row["role"]})
        extracted.append(item)
        print(f"ENTRY_PROGRESS {index + 1}/{len(plan['entries'])} {item['status']} {row['name']}", flush=True)
    result = {
        "schema_version": "poseloop.r4a.development-slice.extraction.v2",
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
