from __future__ import annotations

import urllib.error
from pathlib import Path

from r4a_development_repair_v3.splitzip import CachedRetryRangeClient


class FlakyClient(CachedRetryRangeClient):
    def __init__(self, root: Path):
        super().__init__(
            maximum_bytes=100,
            chunk_bytes=4,
            cache_root=root,
            maximum_attempts=2,
            retry_statuses=[503],
            backoff_seconds=[0],
        )
        self.calls = 0

    def _request(self, url: str, start: int, end: int):
        self.calls += 1
        if self.calls == 1:
            raise urllib.error.HTTPError(url, 503, "transient", {}, None)
        payload = b"0123456789abcdef"[start : end + 1]
        return payload, 206, f"bytes {start}-{end}/16", url


def test_transient_503_retries_then_uses_cache(tmp_path: Path) -> None:
    client = FlakyClient(tmp_path)
    assert client.get("memory://archive", 2, 5) == b"2345"
    assert client.calls == 2
    second = FlakyClient(tmp_path)
    assert second.get("memory://archive", 2, 5) == b"2345"
    assert second.calls == 0
    assert second.bytes_from_cache == 4
