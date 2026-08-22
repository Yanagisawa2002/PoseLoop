from __future__ import annotations

from pathlib import Path

from r4a_development_slice_v2.core import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_slice_v2.json"


def test_slice_v2_is_exact_gray_development_only() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["role_declaration"]["role"] == "DEVELOPMENT_ONLY"
    assert protocol["slice_plan"]["target_count"] == 173
    assert protocol["slice_plan"]["entry_count"] == 188
    assert protocol["slice_plan"]["input_modality"] == "gray"
    assert protocol["immutable_slice_v1_plan_failure"]["rerun_permitted"] is False
    assert protocol["immutable_boundaries"]["xyzibd_val_access_permitted"] is False
