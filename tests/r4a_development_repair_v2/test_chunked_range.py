from __future__ import annotations

from r4a_development_repair_v2.splitzip import ChunkedRangeClient


class MemoryClient(ChunkedRangeClient):
    def __init__(self, payload: bytes):
        super().__init__(maximum_bytes=100, chunk_bytes=4)
        self.payload = payload
        self.calls: list[tuple[int, int]] = []

    def _request(self, url: str, start: int, end: int):
        self.calls.append((start, end))
        return self.payload[start : end + 1], 206, f"bytes {start}-{end}/{len(self.payload)}", url


def test_large_logical_range_is_chunked() -> None:
    client = MemoryClient(b"0123456789abcdef")
    assert client.get("memory://archive", 2, 13) == b"23456789abcd"
    assert client.calls == [(2, 5), (6, 9), (10, 13)]
    assert client.bytes_downloaded == 12
