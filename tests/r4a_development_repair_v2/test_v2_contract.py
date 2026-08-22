from __future__ import annotations

from pathlib import Path

from r4a_development_repair.core import sha256_file
from r4a_development_repair_v2.core import load_protocol


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_protocol.json"
V2 = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_protocol_v2.json"


def test_v2_binds_immutable_v1_and_failure() -> None:
    protocol = load_protocol(V2)
    assert sha256_file(V1) == protocol["immutable_v1_protocol"]["protocol_sha256"]
    assert protocol["immutable_v1_catalog_failure"]["rerun_permitted"] is False
    assert protocol["immutable_v1_catalog_failure"]["exit_code"] == 1


def test_v2_changes_transport_not_data_selection() -> None:
    protocol = load_protocol(V2)
    assert protocol["development_split"]["selection"]["scene_ids"] == [0, 1, 2]
    assert protocol["development_split"]["selection"]["image_ids_per_scene"] == [0, 1, 2, 3]
    assert protocol["repair_contract"]["data_selection_changed"] is False
    assert protocol["repair_contract"]["pose_or_model_changed"] is False
