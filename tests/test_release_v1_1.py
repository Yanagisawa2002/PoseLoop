from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/verify_portfolio.py"
    spec = importlib.util.spec_from_file_location("verify_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_release_bundle() -> None:
    verifier = _load_verifier()
    result = verifier.validate()
    assert result["status"] == "PASS_FROZEN_RELEASE_WITH_EDITABLE_OVERVIEW"
    if result["video_duration_seconds"] is not None:
        assert 60.0 <= result["video_duration_seconds"] <= 90.0
