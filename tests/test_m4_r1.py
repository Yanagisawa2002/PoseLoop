from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_m4_r1_cad_visibility as visibility  # noqa: E402
import build_m4_r1_inputs as inputs  # noqa: E402
import run_m4_r1 as m4_r1  # noqa: E402


def test_feature_outcome_split_keeps_labels_out_of_feature_stream() -> None:
    row = {
        "group_id": "g1",
        "object_id": 1,
        "physical_instance_id": "track-1",
        "candidate_slot": 1,
        "features": {"relative_view_angle_rad": 0.2},
        "actual_utility": 0.8,
        "sample_ar_mssd": 0.7,
        "sample_ar_mspd": 0.9,
        "joint_success": True,
        "outcome_evidence": {"selected_acquisition_rank": 1},
    }
    feature_rows, outcome_rows = inputs.split_rows([row], {"g1": 0.5})
    assert feature_rows[0]["features"] == {"relative_view_angle_rad": 0.2}
    assert "candidate_pair_utility" not in feature_rows[0]
    assert "outcome_evidence" not in feature_rows[0]
    assert outcome_rows[0]["candidate_pair_utility"] == 0.8
    assert "features" not in outcome_rows[0]


def test_occlusion_pair_features_measure_new_and_overlapping_surface() -> None:
    areas = np.asarray([1.0, 1.0, 2.0])
    target = {
        "visible_faces": np.asarray([True, True, False]),
        "silhouette_pixels": 100,
    }
    candidate = {
        "visible_faces": np.asarray([False, True, True]),
        "silhouette_pixels": 150,
    }
    result = visibility.pair_features(target, candidate, areas, 4.0, 32)
    assert result["occ_target_visible_surface_fraction"] == 0.5
    assert result["occ_candidate_visible_surface_fraction"] == 0.75
    assert result["occ_new_surface_fraction"] == 0.5
    assert result["occ_lost_surface_fraction"] == 0.25
    assert result["occ_union_surface_fraction"] == 1.0
    assert result["occ_visible_surface_jaccard"] == 0.25
    assert result["occ_overlap_over_target"] == 0.5
    assert result["occ_silhouette_ratio"] == 1.5


def test_camera_basis_is_orthonormal() -> None:
    direction = np.asarray([0.2, -0.3, 0.9])
    direction = direction / np.linalg.norm(direction)
    right, up = visibility._camera_basis(direction)
    basis = np.stack((right, up, direction))
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)


def test_ranker_difference_vector_uses_candidate_minus_anchor() -> None:
    rows = [
        {
            "group_id": "g1",
            "candidate_slot": 1,
            "features": {"relative_view_angle_rad": 0.7},
            "occlusion_features": {"occ_new_surface_fraction": 0.4},
        },
        {
            "group_id": "g1",
            "candidate_slot": 2,
            "features": {"relative_view_angle_rad": 0.2},
            "occlusion_features": {"occ_new_surface_fraction": 0.1},
        },
    ]
    vector = m4_r1.difference_vector(
        rows,
        0,
        1,
        ("relative_view_angle_rad", "occ_new_surface_fraction"),
    )
    assert np.allclose(vector, [0.5, 0.3, 1.0, 0.0, 0.0])


def test_ranker_falls_back_to_static_anchor_when_no_gain_is_positive() -> None:
    predictions = {
        "g1": {1: -0.1, 2: 0.0, 3: -0.2, 4: -0.3},
        "g2": {1: -0.1, 2: 0.0, 3: 0.4, 4: 0.2},
        "g3": {1: 0.009, 2: 0.0, 3: 0.008, 4: 0.007},
    }
    selected = m4_r1.select_slots(predictions)
    assert selected == {"g1": 2, "g2": 3, "g3": 2}


def test_m4_gate_requires_both_baselines_and_both_bootstrap_bounds() -> None:
    primary = {
        "gain_vs_fixed_slot_2_pp": 1.0,
        "gain_vs_uniform_random_pp": 2.0,
    }
    bootstrap = {
        "gain_vs_fixed_slot_2_pp": {"one_sided_90pct_lower": 0.1},
        "gain_vs_uniform_random_pp": {"one_sided_90pct_lower": 0.2},
    }
    gate = {
        "gain_pp_min": 0.5,
        "one_sided_group_bootstrap_90pct_lower_gain_pp_min": 0.0,
        "minimum_predicted_gain_to_deviate_from_fixed_slot": 0.01,
        "all_conditions_required": True,
    }
    assert m4_r1.gate_decision(primary, bootstrap, gate)["passed"] is True
    bootstrap["gain_vs_fixed_slot_2_pp"]["one_sided_90pct_lower"] = -0.1
    assert m4_r1.gate_decision(primary, bootstrap, gate)["passed"] is False
