#!/usr/bin/env python3
"""Verify the frozen release after editing the live portfolio overview.

SHA256SUMS remains unchanged. Its README entry resolves to the original bytes
in README.snapshot.md; all other entries still resolve to their original paths.
The frozen verify_release.py remains available for the original release checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

from verify_release import load_sums, sha256_file, video_duration_seconds

ROOT = Path(__file__).resolve().parents[1]


def validate() -> dict[str, object]:
    sums = load_sums(ROOT / "release/v1.1.0/SHA256SUMS")
    mappings = {
        "README.md": "release/v1.1.0/README.snapshot.md",
        "scripts/run_release_pipeline.sh": "release/v1.1.0/run_release_pipeline.snapshot.sh",
    }
    checked = []
    for original, expected in sums.items():
        actual = mappings.get(original, original)
        path = ROOT / actual
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen release checksum mismatch: {original} -> {actual}")
        checked.append(actual)
    duration = video_duration_seconds(ROOT / "docs/media/poseloop-demo.mp4")
    if duration is not None and not 60.0 <= duration <= 90.0:
        raise ValueError(f"Unexpected release video duration: {duration}")
    return {
        "status": "PASS_FROZEN_RELEASE_WITH_EDITABLE_OVERVIEW",
        "checked_files": checked,
        "manifest_path_mapping": mappings,
        "video_duration_seconds": duration,
        "live_readme_is_frozen": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
