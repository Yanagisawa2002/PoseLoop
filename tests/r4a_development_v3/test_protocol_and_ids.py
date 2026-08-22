from __future__ import annotations

import json
from pathlib import Path

import pytest

from r4a_development_repair.core import ContractError
from r4a_development_v3.contract import load_protocol
from r4a_development_v3.id_parser import IdRecord, parse_scene_gt_ids, select_multi_object_targets


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_multi_object_v3.json"


def test_protocol_freezes_no_gt_multi_object_boundary() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["id_only_selection"]["required_object_count"] == 5
    assert protocol["id_only_selection"]["required_scene_count"] == 3
    assert protocol["bundle_contract"]["inference_bundle"]["label_access_count"] == 0
    assert "selected_mask_visib" not in protocol["exact_asset_selection"]["inference_roles"]
    assert protocol["bundle_contract"]["evaluator_bundle"]["must_not_leave_gpu_a"] is True
    assert protocol["immutable_boundaries"]["xyzibd_val_access_permitted"] is False


def test_selective_parser_skips_pose_values_lexically() -> None:
    payload = (
        b'{"17":[{"cam_R_m2c":[1e9999999,0,0,0,1,0,0,0,1],'
        b'"cam_t_m2c":[1e9999999,-1e9999999,0],"obj_id":12},'
        b'{"obj_id":13,"cam_R_m2c":[1,0,0,0,1,0,0,0,1],"cam_t_m2c":[0,0,1]}]}'
    )
    rows, audit = parse_scene_gt_ids(payload, scene_id=3)
    assert rows == [IdRecord(3, 17, 12, 0), IdRecord(3, 17, 13, 1)]
    assert audit["pose_values_decoded"] == 0
    assert audit["visibility_values_decoded"] == 0
    assert audit["score_values_decoded"] == 0


def test_selective_parser_rejects_score_or_visibility_fields() -> None:
    with pytest.raises(ContractError, match="Forbidden or unknown"):
        parse_scene_gt_ids(b'{"0":[{"obj_id":1,"score":0.9}]}', scene_id=0)
    with pytest.raises(ContractError, match="Forbidden or unknown"):
        parse_scene_gt_ids(b'{"0":[{"obj_id":1,"visib_fract":0.9}]}', scene_id=0)


def test_deterministic_gate_yields_five_objects_ten_unique_keys_three_scenes() -> None:
    scenes = [
        [IdRecord(0, object_id, object_id, object_id) for object_id in range(1, 6)],
        [IdRecord(3, 100 + object_id, object_id, object_id) for object_id in (1, 2)],
        [IdRecord(9, 200 + object_id, object_id, object_id) for object_id in (3, 4, 5)],
    ]
    result = select_multi_object_targets(scenes)
    assert result is not None
    assert result["object_ids"] == [1, 2, 3, 4, 5]
    assert result["target_count"] == 10
    assert result["unique_key_count"] == 10
    assert len(result["scene_ids"]) >= 3
    assert set(result["per_object_count"].values()) == {2}


def test_protocol_json_contains_no_oracle_mask_disclosure() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "selected_mask_visib" not in text
    assert "official DEVELOPMENT_ONLY mask_visib used" not in text
    parsed = json.loads(text)
    assert parsed["bundle_contract"]["inference_bundle"]["label_access_count"] == 0
