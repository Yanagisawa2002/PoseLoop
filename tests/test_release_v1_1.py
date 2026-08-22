from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = ROOT / "scripts/verify_release.py"
    spec = importlib.util.spec_from_file_location("verify_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_release_bundle() -> None:
    verifier = _load_verifier()
    result = verifier.validate()
    assert result["status"] == "PASS_RELEASE_VALIDATION"
    assert 60.0 <= result["video_duration_seconds"] <= 90.0
