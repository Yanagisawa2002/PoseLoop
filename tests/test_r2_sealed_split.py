from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r2_sealed_photoneo as r2  # noqa: E402


def test_round_robin_uses_each_track_before_reuse() -> None:
    tracks = {
        "track-a": ["a1", "a2", "a3"],
        "track-b": ["b1", "b2", "b3"],
        "track-c": ["c1", "c2", "c3"],
    }
    selected = r2.round_robin_targets(tracks, 5)
    first_tracks = {sample[0] for sample in selected[:3]}
    assert first_tracks == {"a", "b", "c"}
    assert len(selected) == len(set(selected)) == 5


def test_round_robin_is_deterministic() -> None:
    tracks = {"x": ["x1", "x2"], "y": ["y1", "y2"]}
    assert r2.round_robin_targets(tracks, 4) == r2.round_robin_targets(tracks, 4)
