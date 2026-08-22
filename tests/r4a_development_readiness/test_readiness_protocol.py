from __future__ import annotations

from pathlib import Path

from r4a_development_readiness.core import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_readiness_v1.json"


def test_readiness_protocol_excludes_gt_pose_from_manifest() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["role"] == "DEVELOPMENT_ONLY"
    assert protocol["frozen_slice"]["target_count"] == 173
    assert protocol["manifest_contract"]["candidate_selection_from_gt"] is False
    assert protocol["boundaries"]["gt_pose_export_to_prediction_manifest"] is False
    assert protocol["boundaries"]["xyzibd_val_access_permitted"] is False


def test_target_manifest_has_no_development_label_fields() -> None:
    source = (ROOT / "r4a_development_readiness" / "core.py").read_text(encoding="utf-8")
    assert '"cam_R_m2c"' not in source[source.index("targets.append(") :]
    assert '"cam_t_m2c"' not in source[source.index("targets.append(") :]
    assert '"development_visibility' not in source
