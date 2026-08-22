"""Bounded-retry, persistent-cache HTTP range transport for R4-A v3."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from r4a_development_repair.core import ContractError
from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog


class CachedRetryRangeClient:
    def __init__(
        self,
        *,
        maximum_bytes: int,
        chunk_bytes: int,
        cache_root: Path,
        maximum_attempts: int,
        retry_statuses: Sequence[int],
        backoff_seconds: Sequence[int],
        timeout_seconds: int = 45,
    ):
        self.maximum_bytes = int(maximum_bytes)
        self.chunk_bytes = int(chunk_bytes)
        self.cache_root = cache_root.resolve()
        self.maximum_attempts = int(maximum_attempts)
        self.retry_statuses = {int(value) for value in retry_statuses}
        self.backoff_seconds = [int(value) for value in backoff_seconds]
        self.timeout_seconds = int(timeout_seconds)
        if min(self.maximum_bytes, self.chunk_bytes, self.maximum_attempts) <= 0:
            raise ContractError("R4-A v3 range/retry limits must be positive")
        if len(self.backoff_seconds) != self.maximum_attempts - 1:
            raise ContractError("R4-A v3 retry backoff count mismatch")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.bytes_downloaded = 0
        self.bytes_from_cache = 0
        self.requests: list[dict[str, Any]] = []

    def _cache_path(self, url: str, start: int, end: int) -> Path:
        digest = hashlib.sha256(f"{url}\n{start}\n{end}\n".encode("utf-8")).hexdigest()
        return self.cache_root / f"{digest}.{start}-{end}.range"

    def _request(self, url: str, start: int, end: int) -> tuple[bytes, int, str, str]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = int(response.status)
            content_range = response.headers.get("Content-Range", "")
            data = response.read(end - start + 2)
            return data, status, content_range, response.url

    def _download_chunk(self, url: str, start: int, end: int) -> bytes:
        expected = end - start + 1
        cache_path = self._cache_path(url, start, end)
        if cache_path.is_file() and cache_path.stat().st_size == expected:
            data = cache_path.read_bytes()
            self.bytes_from_cache += len(data)
            record = {
                "range_start": start,
                "range_end": end,
                "bytes": len(data),
                "source": "persistent_cache",
                "cumulative_network_bytes": self.bytes_downloaded,
                "cumulative_cache_bytes": self.bytes_from_cache,
            }
            self.requests.append(record)
            print("RANGE_PROGRESS " + json.dumps(record, sort_keys=True), flush=True)
            return data
        last_error: BaseException | None = None
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                data, status, content_range, final_url = self._request(url, start, end)
                if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ContractError(
                        f"Server ignored exact Range: status={status} content-range={content_range!r}"
                    )
                if len(data) != expected:
                    raise ContractError(f"Range size mismatch: expected={expected} actual={len(data)}")
                temporary = cache_path.with_name(cache_path.name + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(cache_path)
                self.bytes_downloaded += len(data)
                record = {
                    "range_start": start,
                    "range_end": end,
                    "status": status,
                    "bytes": len(data),
                    "attempt": attempt,
                    "source": "network",
                    "cumulative_network_bytes": self.bytes_downloaded,
                    "cumulative_cache_bytes": self.bytes_from_cache,
                    "final_host": urllib.parse.urlparse(final_url).netloc,
                }
                self.requests.append(record)
                print("RANGE_PROGRESS " + json.dumps(record, sort_keys=True), flush=True)
                return data
            except urllib.error.HTTPError as error:
                last_error = error
                retryable = int(error.code) in self.retry_statuses
                print(
                    "RANGE_RETRY "
                    + json.dumps(
                        {
                            "range_start": start,
                            "range_end": end,
                            "attempt": attempt,
                            "http_status": int(error.code),
                            "retryable": retryable,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not retryable or attempt >= self.maximum_attempts:
                    raise
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
                print(
                    "RANGE_RETRY "
                    + json.dumps(
                        {
                            "range_start": start,
                            "range_end": end,
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "retryable": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if attempt >= self.maximum_attempts:
                    raise
            time.sleep(self.backoff_seconds[attempt - 1])
        raise ContractError(f"Unreachable retry loop exit: {last_error!r}")

    def get(self, url: str, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ContractError("Invalid HTTP byte range")
        expected_total = end - start + 1
        if self.bytes_downloaded + self.bytes_from_cache + expected_total > self.maximum_bytes:
            raise ContractError("Frozen R4-A v3 HTTP byte stop gate would be exceeded")
        chunks: list[bytes] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + self.chunk_bytes - 1)
            chunks.append(self._download_chunk(url, cursor, chunk_end))
            cursor = chunk_end + 1
        return b"".join(chunks)


__all__ = ["CachedRetryRangeClient", "Entry", "Part", "SplitZip", "write_catalog"]
