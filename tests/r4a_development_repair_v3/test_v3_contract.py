from __future__ import annotations

from pathlib import Path

from r4a_development_repair_v3.core import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_protocol_v3.json"


def test_v3_binds_nonrerunnable_v2_failure() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["immutable_v2_failure"]["rerun_permitted"] is False
    assert protocol["immutable_v2_failure"]["last_completed_cumulative_bytes"] == 109314104


def test_v3_only_changes_transient_transport() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["repair_contract"]["data_selection_changed"] is False
    assert protocol["repair_contract"]["byte_stop_gates_changed"] is False
    assert protocol["repair_contract"]["pose_or_model_changed"] is False
    assert protocol["development_split"]["network_contract"]["maximum_attempts_per_chunk"] == 4
