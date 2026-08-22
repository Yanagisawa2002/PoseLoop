from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_m6_r1_features as features  # noqa: E402
import run_m6_r1 as m6_r1  # noqa: E402


def test_transform_pose_to_target_camera_respects_extrinsics() -> None:
    candidate_pose = np.eye(4)
    candidate_pose[0, 3] = 2.0
    candidate_extrinsic = np.eye(4)
    candidate_extrinsic[0, 3] = 1.0
    target_extrinsic = np.eye(4)
    transformed = features.transform_pose_to_target_camera(
        candidate_pose, candidate_extrinsic, target_extrinsic
    )
    assert np.allclose(transformed[:3, 3], [1.0, 0.0, 0.0])


def test_rotation_geodesic_is_zero_for_equal_rotations() -> None:
    pose = np.eye(4)
    assert features.rotation_geodesic_radians(pose, pose) == 0.0


def test_feature_label_separation_rejects_evaluator_feature() -> None:
    feature_rows = [
        {
            "group_id": "g1",
            "features": {"selected_raw_score": 1.0, "joint_success": 1.0},
        }
    ]
    label_rows = [{"group_id": "g1", "y_failure": 0}]
    try:
        features.validate_separation(feature_rows, label_rows)
    except ValueError as error:
        assert "separation failed" in str(error)
    else:
        raise AssertionError("Evaluator feature should have been rejected")


def test_grouped_fold_assignments_keep_tracks_intact() -> None:
    y = np.asarray([0, 1] * 10)
    groups = np.asarray([f"track-{index // 2}" for index in range(20)])
    assignments = m6_r1.make_fold_assignments(y, groups, 5, 42)
    for group in set(groups):
        assert len(set(assignments[groups == group])) == 1


def test_gate_requires_all_three_conditions() -> None:
    candidate = {"auroc": 0.70, "aurc": 0.20}
    baseline = {"auroc": 0.60, "aurc": 0.25}
    folds = [
        {"positive_direction_both_metrics": value}
        for value in (True, True, True, True, False)
    ]
    thresholds = {
        "nested_grouped_cv_auroc_gain_over_raw_score_min": 0.05,
        "aurc_relative_reduction_min": 0.1,
        "positive_direction_outer_folds_min": 4,
    }
    assert m6_r1.gate_decision(candidate, baseline, folds, thresholds)["passed"]
    folds[3]["positive_direction_both_metrics"] = False
    assert not m6_r1.gate_decision(candidate, baseline, folds, thresholds)["passed"]
