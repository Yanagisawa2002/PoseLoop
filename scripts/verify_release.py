#!/usr/bin/env python3
"""Validate the compact public release bundle and demo media."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "release/v1.1.0/results.json"
SUMS_PATH = ROOT / "release/v1.1.0/SHA256SUMS"
VIDEO_PATH = ROOT / "docs/media/poseloop-demo.mp4"
POSTER_PATH = ROOT / "docs/media/poseloop-demo-poster.jpg"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        rows[relative] = digest
    return rows


def video_duration_seconds(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(process.stdout.strip())


def validate() -> dict[str, object]:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_PACKAGE_AND_CLOSE"
    assert result["dataset"]["scene9_read_count"] == 0
    assert result["detector"]["prediction_count"] == 820
    assert result["detector"]["f1_iou50"] == 0.7257861635220125
    assert result["pose_pipeline"]["runtime_completed"] == 820
    assert result["pose_pipeline"]["joint_pose_successes"] == 482
    assert result["pose_pipeline"]["combined_ar_mssd_mspd"] == 0.6381168831168831
    assert result["retired_routes_frozen"] is True

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    forbidden_internal_labels = re.findall(r"\b(?:M[1-6]|R(?:1[0-7]|[1-9]))\b", readme)
    assert forbidden_internal_labels == [], forbidden_internal_labels

    sums = load_sums(SUMS_PATH)
    checked: list[str] = []
    for relative, expected in sums.items():
        candidate = ROOT / relative
        assert candidate.is_file(), relative
        assert sha256_file(candidate) == expected, relative
        checked.append(relative)

    duration = video_duration_seconds(VIDEO_PATH)
    if duration is not None:
        assert 60.0 <= duration <= 90.0, duration

    assert POSTER_PATH.stat().st_size > 50_000
    return {
        "status": "PASS_RELEASE_VALIDATION",
        "checked_files": checked,
        "video_duration_seconds": duration,
    }


def main() -> int:
    print(json.dumps(validate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
