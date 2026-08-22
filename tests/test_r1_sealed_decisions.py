from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TOOLKIT = Path(__file__).resolve().parents[1] / "third_party" / "bop_toolkit"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))

from evaluate_r1_sealed_once import macro_object  # noqa: E402
from finalize_r1_sealed_decisions import build_final_row, rank_slots  # noqa: E402


class _LinearModel:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[:, 0] + 0.5 * matrix[:, 1]


def _ranking_rows() -> list[dict]:
    values = {
        1: (0.0, 0.0),
        2: (1.0, 0.2),
        3: (1.2, 0.4),
        4: (0.5, 0.1),
    }
    return [
        {
            "group_id": "g",
            "candidate_slot": slot,
            "features": {"camera_baseline_over_diameter": values[slot][0]},
            "occlusion_features": {"occ_new_surface_fraction": values[slot][1]},
        }
        for slot in (1, 2, 3, 4)
    ]


def test_rank_slots_uses_anchor_differences_and_minimum_gain() -> None:
    names = [
        "camera_baseline_over_diameter",
        "occ_new_surface_fraction",
        "candidate_slot_is_1",
        "candidate_slot_is_3",
        "candidate_slot_is_4",
    ]
    scores, selected = rank_slots(_ranking_rows(), [_LinearModel()], names, 0.01)
    assert scores[2] == 0.0
    assert np.isclose(scores[3], 0.3)
    assert selected == 3


def test_rank_slots_falls_back_to_anchor_below_minimum_gain() -> None:
    names = [
        "camera_baseline_over_diameter",
        "occ_new_surface_fraction",
        "candidate_slot_is_1",
        "candidate_slot_is_3",
        "candidate_slot_is_4",
    ]
    scores, selected = rank_slots(_ranking_rows(), [_LinearModel()], names, 0.5)
    assert max(scores.values()) < 0.5
    assert selected == 2


def _pose(z: float) -> list[list[float]]:
    value = np.eye(4)
    value[2, 3] = z
    return value.tolist()


def test_build_final_row_applies_max_mask_without_pose_reselection() -> None:
    identity = np.eye(4).tolist()
    views = [
        {
            "acquisition_rank": rank,
            "sample_id": f"s{rank}",
            "visible_mask_pixel_count": count,
            "camera_world_to_camera_pose_m": identity,
        }
        for rank, count in enumerate((10, 11, 40, 12, 13))
    ]
    group = {
        "group_id": "g",
        "object_id": 1,
        "target_sample_id": "s0",
        "views": views,
    }
    m3 = {
        "selected_budget": 3,
        "physical_instance_id": "track",
        "k1_predicted_marginal_value": 0.2,
        "k3_predicted_marginal_value": -0.1,
    }
    predictions = {
        f"s{rank}": {
            "status": "success",
            "predicted_model_to_camera_pose_m": _pose(0.5 + rank * 0.01),
        }
        for rank in range(5)
    }
    result = build_final_row(group, m3, predictions, {1: -0.1, 2: 0.0, 3: 0.2, 4: 0.1}, 2)
    assert result["final"]["m4_selected_slot"] == 2
    assert result["final"]["selected_sample_id"] == "s2"
    assert result["final"]["selected_acquisition_rank"] == 2


def test_macro_object_weights_objects_equally() -> None:
    rows = [
        {"object_id": 1, "value": 0.0},
        {"object_id": 1, "value": 1.0},
        {"object_id": 2, "value": 1.0},
    ]
    assert macro_object(rows, "value") == 0.75
