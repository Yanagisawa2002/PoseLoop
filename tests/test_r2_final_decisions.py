from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TOOLKIT = REPO_ROOT / "third_party" / "bop_toolkit"
for path in (SCRIPTS, TOOLKIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finalize_r2_sealed_decisions import build_final_row  # noqa: E402


def test_target_only_route_preserves_unavailable_k3_score() -> None:
    identity = np.eye(4).tolist()
    views = [
        {
            "acquisition_rank": rank,
            "sample_id": f"s{rank}",
            "visible_mask_pixel_count": 10 + rank,
            "camera_world_to_camera_pose_m": identity,
        }
        for rank in range(5)
    ]
    group = {"group_id": "g", "object_id": 1, "target_sample_id": "s0", "views": views}
    m3 = {
        "selected_budget": 1,
        "physical_instance_id": "track",
        "k1_predicted_marginal_value": 0.1,
        "k3_predicted_marginal_value": None,
    }
    predictions = {
        f"s{rank}": {"status": "success", "predicted_model_to_camera_pose_m": identity}
        for rank in range(5)
    }
    row = build_final_row(group, m3, predictions, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, 2)
    assert row["m3_k3_predicted_marginal_value"] is None
    assert row["m3_k3_score_available"] is False
    assert row["final"]["route"] == "m3_stop_target_only"
