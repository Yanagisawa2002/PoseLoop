"""Chunked HTTP range transport for the immutable R4-A v2 repair."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from r4a_development_repair.core import ContractError
from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog


class ChunkedRangeClient:
    def __init__(self, maximum_bytes: int, chunk_bytes: int, timeout_seconds: int = 30):
        self.maximum_bytes = int(maximum_bytes)
        self.chunk_bytes = int(chunk_bytes)
        self.timeout_seconds = int(timeout_seconds)
        if self.maximum_bytes <= 0 or self.chunk_bytes <= 0:
            raise ContractError("Range limits must be positive")
        self.bytes_downloaded = 0
        self.requests: list[dict[str, Any]] = []

    def _request(self, url: str, start: int, end: int) -> tuple[bytes, int, str, str]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = int(response.status)
            content_range = response.headers.get("Content-Range", "")
            data = response.read(end - start + 2)
            return data, status, content_range, response.url

    def get(self, url: str, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ContractError("Invalid HTTP byte range")
        expected_total = end - start + 1
        if self.bytes_downloaded + expected_total > self.maximum_bytes:
            raise ContractError("Frozen R4-A v2 HTTP byte stop gate would be exceeded")
        chunks: list[bytes] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + self.chunk_bytes - 1)
            expected = chunk_end - cursor + 1
            data, status, content_range, final_url = self._request(url, cursor, chunk_end)
            if status != 206 or not content_range.startswith(f"bytes {cursor}-{chunk_end}/"):
                raise ContractError(
                    f"Server ignored exact Range: status={status} content-range={content_range!r}"
                )
            if len(data) != expected:
                raise ContractError(f"Range size mismatch: expected={expected} actual={len(data)}")
            self.bytes_downloaded += len(data)
            record = {
                "request_index": len(self.requests),
                "range_start": cursor,
                "range_end": chunk_end,
                "status": status,
                "content_range": content_range,
                "bytes": len(data),
                "cumulative_bytes": self.bytes_downloaded,
                "final_host": urllib.request.urlparse(final_url).netloc if hasattr(urllib.request, "urlparse") else "redirected",
            }
            self.requests.append(record)
            print("RANGE_PROGRESS " + json.dumps(record, sort_keys=True), flush=True)
            chunks.append(data)
            cursor = chunk_end + 1
        return b"".join(chunks)


__all__ = ["ChunkedRangeClient", "Entry", "Part", "SplitZip", "write_catalog"]
