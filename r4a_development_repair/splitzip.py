"""Auditable HTTP-range reader for the official two-part train_pbr ZIP.

Only the central directory and explicitly selected entries are transferred.
The caller supplies hard byte limits from the frozen protocol.  A server that
ignores Range is rejected before a whole archive can be downloaded.
"""

from __future__ import annotations

import binascii
import json
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import ContractError, sha256_file, write_json_atomic


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class Part:
    name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Entry:
    name: str
    disk_start: int
    local_offset: int
    compressed_size: int
    uncompressed_size: int
    compression: int
    crc32: int
    flags: int


class RangeClient:
    def __init__(self, maximum_bytes: int, timeout_seconds: int = 30):
        self.maximum_bytes = int(maximum_bytes)
        self.timeout_seconds = int(timeout_seconds)
        self.bytes_downloaded = 0
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ContractError("Invalid HTTP byte range")
        expected = end - start + 1
        if self.bytes_downloaded + expected > self.maximum_bytes:
            raise ContractError("Frozen HTTP byte stop gate would be exceeded")
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = int(response.status)
            content_range = response.headers.get("Content-Range", "")
            if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                raise ContractError(
                    f"Server did not honor the exact byte range: status={status} content-range={content_range!r}"
                )
            data = response.read(expected + 1)
            if len(data) != expected:
                raise ContractError(f"Short or oversized range response: expected={expected} actual={len(data)}")
            self.bytes_downloaded += len(data)
            self.requests.append(
                {
                    "url": url,
                    "range_start": start,
                    "range_end": end,
                    "status": status,
                    "content_range": content_range,
                    "bytes": len(data),
                }
            )
            return data


def _zip64_values(extra: bytes, needs: tuple[bool, bool, bool, bool]) -> list[int | None]:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        payload = extra[cursor : cursor + size]
        cursor += size
        if header_id != 0x0001:
            continue
        values: list[int | None] = []
        offset = 0
        widths = (8, 8, 8, 4)
        for required, width in zip(needs, widths):
            if not required:
                values.append(None)
                continue
            if offset + width > len(payload):
                raise ContractError("Truncated ZIP64 extra field")
            fmt = "<Q" if width == 8 else "<L"
            values.append(int(struct.unpack_from(fmt, payload, offset)[0]))
            offset += width
        return values
    if any(needs):
        raise ContractError("Missing ZIP64 extra values")
    return [None, None, None, None]


class SplitZip:
    def __init__(self, parts: list[Part], client: RangeClient):
        if len(parts) != 2:
            raise ContractError("R4-A expects exactly two official split ZIP parts")
        self.parts = parts
        self.client = client

    def _read_across_parts(self, disk: int, offset: int, count: int) -> bytes:
        if disk < 0 or disk >= len(self.parts) or count < 0:
            raise ContractError("Invalid split ZIP location")
        chunks: list[bytes] = []
        remaining = count
        current_disk = disk
        current_offset = offset
        while current_disk < len(self.parts) and current_offset >= self.parts[current_disk].size:
            current_offset -= self.parts[current_disk].size
            current_disk += 1
        while remaining:
            if current_disk >= len(self.parts):
                raise ContractError("ZIP record extends beyond the final part")
            part = self.parts[current_disk]
            available = part.size - current_offset
            if available <= 0:
                current_disk += 1
                current_offset = 0
                continue
            take = min(remaining, available)
            chunks.append(self.client.get(part.url, current_offset, current_offset + take - 1))
            remaining -= take
            current_disk += 1
            current_offset = 0
        return b"".join(chunks)

    def read_central_directory(self) -> tuple[list[Entry], dict[str, Any]]:
        final_disk = len(self.parts) - 1
        final = self.parts[final_disk]
        tail_size = min(final.size, 256 * 1024)
        tail_start = final.size - tail_size
        tail = self.client.get(final.url, tail_start, final.size - 1)
        eocd_index = tail.rfind(EOCD_SIGNATURE)
        if eocd_index < 0 or eocd_index + 22 > len(tail):
            raise ContractError("End-of-central-directory record was not found")
        (
            disk_number,
            cd_start_disk,
            entries_on_disk,
            entry_count,
            cd_size,
            cd_offset,
            comment_length,
        ) = struct.unpack_from("<4H2LH", tail, eocd_index + 4)
        if eocd_index + 22 + comment_length > len(tail):
            raise ContractError("Truncated EOCD comment")

        zip64 = any(
            value == maximum
            for value, maximum in (
                (disk_number, 0xFFFF),
                (cd_start_disk, 0xFFFF),
                (entries_on_disk, 0xFFFF),
                (entry_count, 0xFFFF),
                (cd_size, 0xFFFFFFFF),
                (cd_offset, 0xFFFFFFFF),
            )
        )
        if zip64:
            locator_index = tail.rfind(ZIP64_LOCATOR_SIGNATURE, 0, eocd_index)
            if locator_index < 0 or locator_index + 20 > len(tail):
                raise ContractError("ZIP64 locator was not found")
            eocd64_disk, eocd64_offset, total_disks = struct.unpack_from("<LQL", tail, locator_index + 4)
            if total_disks != len(self.parts):
                raise ContractError("ZIP64 part count differs from frozen archive contract")
            eocd64_head = self._read_across_parts(eocd64_disk, eocd64_offset, 56)
            if not eocd64_head.startswith(ZIP64_EOCD_SIGNATURE):
                raise ContractError("ZIP64 EOCD signature mismatch")
            (
                _record_size,
                _version_made,
                _version_needed,
                disk_number,
                cd_start_disk,
                entries_on_disk,
                entry_count,
                cd_size,
                cd_offset,
            ) = struct.unpack_from("<Q2H2L4Q", eocd64_head, 4)
        if disk_number != final_disk:
            raise ContractError("EOCD is not on the expected final split disk")
        central = self._read_across_parts(int(cd_start_disk), int(cd_offset), int(cd_size))
        entries: list[Entry] = []
        cursor = 0
        while cursor < len(central):
            if central[cursor : cursor + 4] != CENTRAL_SIGNATURE:
                raise ContractError(f"Central directory signature mismatch at byte {cursor}")
            if cursor + 46 > len(central):
                raise ContractError("Truncated central directory record")
            fields = struct.unpack_from("<6H3L5H2L", central, cursor + 4)
            (
                _version_made,
                _version_needed,
                flags,
                compression,
                _mod_time,
                _mod_date,
                crc32,
                compressed_size,
                uncompressed_size,
                name_length,
                extra_length,
                comment_length,
                disk_start,
                _internal_attributes,
                _external_attributes,
                local_offset,
            ) = fields
            record_end = cursor + 46 + name_length + extra_length + comment_length
            if record_end > len(central):
                raise ContractError("Central directory variable fields are truncated")
            name_bytes = central[cursor + 46 : cursor + 46 + name_length]
            extra = central[cursor + 46 + name_length : cursor + 46 + name_length + extra_length]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding)
            needs = (
                uncompressed_size == 0xFFFFFFFF,
                compressed_size == 0xFFFFFFFF,
                local_offset == 0xFFFFFFFF,
                disk_start == 0xFFFF,
            )
            z_uncompressed, z_compressed, z_offset, z_disk = _zip64_values(extra, needs)
            entries.append(
                Entry(
                    name=name,
                    disk_start=int(z_disk if z_disk is not None else disk_start),
                    local_offset=int(z_offset if z_offset is not None else local_offset),
                    compressed_size=int(z_compressed if z_compressed is not None else compressed_size),
                    uncompressed_size=int(z_uncompressed if z_uncompressed is not None else uncompressed_size),
                    compression=int(compression),
                    crc32=int(crc32),
                    flags=int(flags),
                )
            )
            cursor = record_end
        if len(entries) != int(entry_count):
            raise ContractError(f"Central directory count mismatch: parsed={len(entries)} expected={entry_count}")
        metadata = {
            "disk_number": int(disk_number),
            "central_directory_start_disk": int(cd_start_disk),
            "central_directory_offset": int(cd_offset),
            "central_directory_bytes": int(cd_size),
            "entry_count": int(entry_count),
            "zip64": zip64,
        }
        return entries, metadata

    def extract_entry(self, entry: Entry, output_path: Path) -> dict[str, Any]:
        header = self._read_across_parts(entry.disk_start, entry.local_offset, 30)
        if not header.startswith(LOCAL_SIGNATURE):
            raise ContractError(f"Local header signature mismatch: {entry.name}")
        (
            _version,
            flags,
            compression,
            _mod_time,
            _mod_date,
            _crc,
            _compressed_size,
            _uncompressed_size,
            name_length,
            extra_length,
        ) = struct.unpack_from("<5H3L2H", header, 4)
        if flags & 0x1:
            raise ContractError("Encrypted ZIP entries are unsupported")
        if int(compression) != entry.compression:
            raise ContractError("Local and central compression methods differ")
        data_offset = entry.local_offset + 30 + name_length + extra_length
        compressed = self._read_across_parts(entry.disk_start, data_offset, entry.compressed_size)
        if entry.compression == 0:
            raw = compressed
        elif entry.compression == 8:
            raw = zlib.decompress(compressed, -15)
        else:
            raise ContractError(f"Unsupported ZIP compression method: {entry.compression}")
        if len(raw) != entry.uncompressed_size:
            raise ContractError("Uncompressed entry size mismatch")
        if (binascii.crc32(raw) & 0xFFFFFFFF) != entry.crc32:
            raise ContractError("Entry CRC32 mismatch")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(output_path)
        return {
            "entry": entry.name,
            "output_path": str(output_path.resolve()),
            "compressed_bytes": entry.compressed_size,
            "uncompressed_bytes": entry.uncompressed_size,
            "sha256": sha256_file(output_path),
        }


def write_catalog(path: Path, entries: Iterable[Entry], metadata: Mapping[str, Any], client: RangeClient) -> None:
    rows = [entry.__dict__ for entry in entries]
    write_json_atomic(
        path,
        {
            "schema_version": "poseloop.r4a.split-zip-catalog.v1",
            "metadata": dict(metadata),
            "entries": rows,
            "network": {
                "bytes_downloaded": client.bytes_downloaded,
                "requests": client.requests,
            },
        },
    )
