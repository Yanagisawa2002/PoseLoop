from __future__ import annotations

import urllib.error
from pathlib import Path

from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient


class ShortThenComplete(ShortBodyRetryRangeClient):
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
            raise urllib.error.ContentTooShortError("short", b"12")
        payload = b"0123456789"[start : end + 1]
        return payload, 206, f"bytes {start}-{end}/10", url


def test_short_body_is_retried(tmp_path: Path) -> None:
    client = ShortThenComplete(tmp_path)
    assert client.get("memory://archive", 2, 5) == b"2345"
    assert client.calls == 2
