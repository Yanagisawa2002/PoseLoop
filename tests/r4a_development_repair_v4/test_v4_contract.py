from __future__ import annotations

from pathlib import Path

from r4a_development_repair_v4.core import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_protocol_v4.json"


def test_v4_binds_short_v3_body_and_keeps_selection() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["immutable_v3_failure"]["rerun_permitted"] is False
    assert protocol["immutable_v3_failure"]["actual_body_bytes"] == 4893689
    assert protocol["development_split"]["selection"]["scene_ids"] == [0, 1, 2]
    assert protocol["repair_contract"]["data_selection_changed"] is False
