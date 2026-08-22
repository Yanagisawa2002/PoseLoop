from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m5_g0_core as legacy  # noqa: E402
import m5_g0_metrics as metrics  # noqa: E402
import audit_m5_r1_real_replay as replay_audit  # noqa: E402
import run_m5_r1_phase1 as phase1  # noqa: E402
from m5_r1_core import QuotientCVKalmanConfig, run_quotient_cv_kalman  # noqa: E402


def test_phase1_manifest_is_balanced_and_fold_grouped() -> None:
    manifest = phase1.synthetic_manifest()
    assert len(manifest) == 768
    assert {row["outer_fold"] for row in manifest} == {0, 1, 2, 3}
    for replicate in range(phase1.REPLICATES):
        assert {
            row["outer_fold"] for row in manifest if row["replicate"] == replicate
        } == {replicate % phase1.OUTER_FOLDS}


def test_methods_have_equal_configuration_budgets() -> None:
    grids = phase1.configuration_grid()
    assert set(grids) == {phase1.METHOD_NEAREST, phase1.METHOD_PROPOSED}
    assert {len(grid) for grid in grids.values()} == {4}


def test_quotient_kalman_stays_finite_for_continuous_symmetry() -> None:
    truth = legacy.generate_ground_truth(
        legacy.SymmetryClass.CONTINUOUS_AXIAL,
        legacy.MotionFamily.CURVED_ACCELERATING,
        1234,
    )
    sequence = legacy.corrupt_measurements(
        truth, legacy.StressFamily.COMBINED, 5678
    )
    outputs = run_quotient_cv_kalman(sequence, QuotientCVKalmanConfig())
    assert all(np.isfinite(output.pose).all() for output in outputs)
    assert all(output.body_twist[5] == 0.0 for output in outputs)


def test_robust_gate_does_not_accept_injected_outlier_burst() -> None:
    truth = legacy.generate_ground_truth("C2", "CONSTANT_TWIST", 1234)
    sequence = legacy.corrupt_measurements(truth, "OUTLIER_BURST", 5678)
    config = phase1.proposed_grid()["qcvk_balanced"]
    outputs = run_quotient_cv_kalman(sequence, config)
    outlier_indices = [
        index for index, frame in enumerate(sequence.evaluator_frames) if frame.outlier
    ]
    assert outlier_indices
    assert not any(outputs[index].accepted for index in outlier_indices)


def test_gate_requires_effect_size_and_bootstrap_lower_bound() -> None:
    thresholds = {
        "relative_improvement_min": 0.1,
        "one_sided_sequence_bootstrap_90pct_lower_improvement_min": 0.0,
    }
    bootstrap = {"one_sided_90pct_lower_relative_improvement": 0.05}
    assert phase1.gate_decision(1.0, 0.8, bootstrap, thresholds)["passed"]
    bootstrap["one_sided_90pct_lower_relative_improvement"] = -0.01
    assert not phase1.gate_decision(1.0, 0.8, bootstrap, thresholds)["passed"]


def test_real_replay_selection_uses_median_support_then_track_id() -> None:
    rows = [
        {"track_id": "track-c", "available_observation_count": 5},
        {"track_id": "track-b", "available_observation_count": 9},
        {"track_id": "track-a", "available_observation_count": 9},
        {"track_id": "track-d", "available_observation_count": 20},
    ]
    selected = replay_audit.choose_median_support_track(rows)
    assert selected["track_id"] == "track-a"


def test_real_replay_input_gate_requires_observed_missing_sequences() -> None:
    selected = [
        {"available_observation_count": 10, "observed_missing_frame_count": 0}
        for _ in range(10)
    ]
    phase2 = {
        "minimum_sequence_count": 10,
        "minimum_sequences_with_observed_missing_frames": 5,
        "minimum_total_observed_missing_frames": 15,
    }
    gate = replay_audit.input_gate(selected, phase2)
    assert gate["conditions"]["minimum_sequence_count"]
    assert not gate["passed"]
