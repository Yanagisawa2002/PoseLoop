"""Treat a prematurely closed Range body as a bounded-retry event."""

from __future__ import annotations

import urllib.error
import urllib.request

from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog
from r4a_development_repair_v3.splitzip import CachedRetryRangeClient


class ShortBodyRetryRangeClient(CachedRetryRangeClient):
    def _request(self, url: str, start: int, end: int) -> tuple[bytes, int, str, str]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = int(response.status)
            content_range = response.headers.get("Content-Range", "")
            expected = end - start + 1
            data = response.read(expected + 1)
            if len(data) != expected:
                raise urllib.error.ContentTooShortError(
                    f"premature Range body: expected={expected} actual={len(data)}",
                    data,
                )
            return data, status, content_range, response.url


__all__ = ["ShortBodyRetryRangeClient", "Entry", "Part", "SplitZip", "write_catalog"]
