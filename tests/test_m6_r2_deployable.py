from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_m6_r1  # noqa: E402


def test_r2_protocol_allows_only_selected_output_features() -> None:
    protocol = json.loads((REPO_ROOT / "protocols" / "poseloop_r2_protocol.json").read_text())
    allowed = sorted(protocol["M6-R2"]["allowed_feature_names"])
    source = json.loads(
        (REPO_ROOT / "artifacts" / "r1" / "m6_r1" / "features.jsonl")
        .read_text()
        .splitlines()[0]
    )
    families = run_m6_r1.feature_families(sorted(source["features"]))
    assert allowed == families["selected_output_only"]
    assert all("policy_" not in name for name in allowed)
    assert all(not name.startswith(("pair_", "cad_", "geometry_")) for name in allowed)
