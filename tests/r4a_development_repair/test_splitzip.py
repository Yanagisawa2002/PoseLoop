from __future__ import annotations

import binascii
import io
import struct
import zipfile
from pathlib import Path

from r4a_development_repair.splitzip import Entry, Part, RangeClient, SplitZip


class MemoryRangeClient(RangeClient):
    def __init__(self, urls: dict[str, bytes]):
        super().__init__(maximum_bytes=1024 * 1024)
        self.urls = urls

    def get(self, url: str, start: int, end: int) -> bytes:
        data = self.urls[url][start : end + 1]
        assert len(data) == end - start + 1
        self.bytes_downloaded += len(data)
        self.requests.append({"url": url, "range_start": start, "range_end": end, "bytes": len(data)})
        return data


def test_extract_deflated_entry_across_two_parts(tmp_path: Path) -> None:
    payload = b"pose-loop-r4a" * 100
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xyzibd/train_pbr/000000/scene_camera.json", payload)
    whole = buffer.getvalue()
    local_offset = whole.index(b"PK\x03\x04")
    central_offset = whole.index(b"PK\x01\x02")
    fields = struct.unpack_from("<6H3L5H2L", whole, central_offset + 4)
    compressed_size = fields[7]
    uncompressed_size = fields[8]
    name_length = fields[9]
    extra_length = fields[10]
    compression = fields[3]
    crc32 = fields[6]
    split = 40
    parts_data = {"mem://z01": whole[:split], "mem://zip": whole[split:]}
    parts = [
        Part("x.z01", "mem://z01", split, "0" * 64),
        Part("x.zip", "mem://zip", len(whole) - split, "1" * 64),
    ]
    client = MemoryRangeClient(parts_data)
    archive = SplitZip(parts, client)
    entry = Entry(
        name="xyzibd/train_pbr/000000/scene_camera.json",
        disk_start=0,
        local_offset=local_offset,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        compression=compression,
        crc32=crc32,
        flags=0,
    )
    output = tmp_path / "scene_camera.json"
    result = archive.extract_entry(entry, output)
    assert output.read_bytes() == payload
    assert result["uncompressed_bytes"] == len(payload)
