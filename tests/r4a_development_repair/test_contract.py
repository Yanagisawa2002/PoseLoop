from __future__ import annotations

import json
from pathlib import Path

import pytest

from r4a_development_repair.core import ContractError, load_protocol
from r4a_development_repair.development import (
    camera_world_to_camera_pose_m,
    coverage_audit,
    foundationpose_to_bop_row,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_protocol.json"


def test_protocol_is_development_only_and_val_forbidden() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["role_declaration"]["role"] == "DEVELOPMENT_ONLY"
    assert protocol["role_declaration"]["sealed_split_eligible"] is False
    assert protocol["immutable_boundaries"]["xyzibd_val_evaluator_permitted"] is False
    assert protocol["immutable_boundaries"]["xyzibd_val_gt_access_permitted"] is False
    assert protocol["development_split"]["network_contract"]["whole_archive_download_permitted"] is False


def test_protocol_rejects_selection_drift(tmp_path: Path) -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["development_split"]["selection"]["scene_ids"] = [0, 1, 3]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="scene selection"):
        load_protocol(changed)


def test_foundationpose_to_bop_uses_model_to_camera_and_mm() -> None:
    pose = [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, -0.2],
        [0.0, 0.0, 1.0, 0.65],
        [0.0, 0.0, 0.0, 1.0],
    ]
    row = foundationpose_to_bop_row(
        scene_id=1,
        image_id=2,
        object_id=4,
        score=0.8,
        predicted_model_to_camera_pose_m=pose,
        elapsed_seconds=0.5,
    )
    assert row["R"] == "1 0 0 0 1 0 0 0 1"
    assert row["t"] == "100 -200 650"


def test_camera_translation_converts_mm_to_m() -> None:
    pose = camera_world_to_camera_pose_m(
        {
            "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_w2c": [100, -200, 650],
        }
    )
    assert [pose[row][3] for row in range(3)] == [0.1, -0.2, 0.65]


def test_no_detection_is_explicit_coverage_failure() -> None:
    targets = [
        {"scene_id": 0, "image_id": 0, "object_id": 1, "instance_id": 0},
        {"scene_id": 0, "image_id": 0, "object_id": 2, "instance_id": 1},
    ]
    results = [
        {"scene_id": 0, "image_id": 0, "object_id": 1, "instance_id": 0, "status": "success"},
        {"scene_id": 0, "image_id": 0, "object_id": 2, "instance_id": 1, "status": "no_detection"},
    ]
    audit = coverage_audit(targets, results)
    assert audit["coverage_fraction"] == 0.5
    assert audit["gate_pass"] is False
    assert audit["missing"] == [(0, 0, 2, 1)]
