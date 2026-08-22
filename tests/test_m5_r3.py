from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_m5_r3_development as development  # noqa: E402
import build_m5_r3_photoneo_replay as builder  # noqa: E402
import evaluate_m5_r3_once as evaluator  # noqa: E402
import m5_g0_core as legacy  # noqa: E402
import run_m5_r3_inference as inference  # noqa: E402
from m5_r3_core import (  # noqa: E402
    StaticQuotientMedoidConfig,
    StaticQuotientMedoidEstimator,
    normalized_quotient_distance,
    quotient_medoid_index,
)


def _pose(x_m: float, yaw_deg: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x_m
    pose[:3, :3] = legacy.z_rotation_transform(np.radians(yaw_deg))[:3, :3]
    return pose


def _input(
    timestamp_s: float, pose: np.ndarray | None, symmetry: legacy.SymmetrySpec
) -> legacy.EstimatorInput:
    return legacy.EstimatorInput(
        timestamp_s=timestamp_s,
        measurement_pose=pose,
        missing=pose is None,
        symmetry=symmetry,
    )


def test_medoid_uses_frozen_normalized_quotient_metric() -> None:
    symmetry = legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC)
    config = StaticQuotientMedoidConfig()
    assert normalized_quotient_distance(_pose(0.0), _pose(0.01), symmetry, config) == pytest.approx(1.0)
    assert normalized_quotient_distance(
        _pose(0.0), _pose(0.0, 5.0), symmetry, config
    ) == pytest.approx(1.0)


def test_all_history_medoid_rejects_one_static_outlier_and_holds_dropout() -> None:
    symmetry = legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC)
    estimator = StaticQuotientMedoidEstimator(StaticQuotientMedoidConfig())
    outputs = [
        estimator.step(_input(0.00, _pose(0.000), symmetry)),
        estimator.step(_input(0.05, _pose(0.001), symmetry)),
        estimator.step(_input(0.10, _pose(0.100), symmetry)),
        estimator.step(_input(0.15, None, symmetry)),
    ]
    assert outputs[2].pose[0, 3] == pytest.approx(0.001)
    assert np.array_equal(outputs[2].pose, outputs[3].pose)
    assert outputs[3].reason == "static_dropout_hold"
    assert not outputs[3].accepted


def test_discrete_equivalent_representatives_have_zero_medoid_distance() -> None:
    symmetry = legacy.symmetry_spec(legacy.SymmetryClass.C2)
    first = _pose(0.0, 0.0)
    equivalent = _pose(0.0, 180.0)
    config = StaticQuotientMedoidConfig()
    assert normalized_quotient_distance(first, equivalent, symmetry, config) == pytest.approx(
        0.0, abs=1e-12
    )
    assert quotient_medoid_index([first, equivalent], symmetry, config) == 0


def test_history_limit_is_measurement_count_not_frame_count() -> None:
    symmetry = legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC)
    estimator = StaticQuotientMedoidEstimator(
        StaticQuotientMedoidConfig(history_limit=2)
    )
    estimator.step(_input(0.00, _pose(0.0), symmetry))
    estimator.step(_input(0.05, None, symmetry))
    estimator.step(_input(0.10, _pose(0.1), symmetry))
    estimator.step(_input(0.15, None, symmetry))
    estimator.step(_input(0.20, _pose(0.2), symmetry))
    assert [pose[0, 3] for pose in estimator.history] == pytest.approx([0.1, 0.2])


def test_development_macro_weights_objects_equally() -> None:
    rows = [
        {"object_id": 1, "loss": 1.0},
        {"object_id": 2, "loss": 10.0},
        {"object_id": 2, "loss": 20.0},
        {"object_id": 2, "loss": 30.0},
    ]
    assert development.macro_object_mean(rows, "loss") == pytest.approx(10.5)


def test_candidate_ids_make_all_history_explicit() -> None:
    assert development.history_id(None) == "all_history"
    assert development.history_id(20) == "last_20_measurements"


def test_prior_group_boundaries_are_unioned_by_track() -> None:
    first = [
        {"oracle_association": {"track_id": "track-a"}},
        {"oracle_association": {"track_id": "track-a"}},
        {"oracle_association": {"track_id": "track-b"}},
    ]
    second = [{"oracle_association": {"track_id": "track-c"}}]
    consumed = builder.consumed_track_ids(first) | builder.consumed_track_ids(second)
    assert consumed == {"track-a", "track-b", "track-c"}


def test_builder_rejects_evaluator_fields_in_inference_stream() -> None:
    with pytest.raises(RuntimeError, match="Evaluator fields leaked"):
        builder.inference_row(
            {
                "sample_id": "sample",
                "input_available": True,
                "gt_model_to_camera_pose_m": np.eye(4).tolist(),
            }
        )


def test_inference_adapter_is_photoneo_and_label_blind() -> None:
    row = {
        "record_type": "m5_r3_inference_sample",
        "sample_id": "sample",
        "sensor_modality": "photoneo",
        "input_available": True,
        "evaluator_label_read": False,
        "input_unit_check_pose_m": np.eye(4).tolist(),
        "input_mask_area_fraction": 0.02,
    }
    effective = inference.effective_row(row)
    assert effective["gt_model_to_camera_pose_m"] == row["input_unit_check_pose_m"]
    assert effective["visibility_bin"] == "input_only"
    assert "gt_model_to_camera_pose_m" not in row


def test_expected_count_validation_checks_per_object_tracks() -> None:
    observed = {
        "track_count": 2,
        "object_count": 2,
        "replay_frame_count": 10,
        "inference_sample_count": 8,
        "natural_missing_frame_count": 2,
        "preinitialization_missing_frame_count": 1,
        "object_ids": [1, 2],
        "by_object": {"1": {"track_count": 1}, "2": {"track_count": 1}},
    }
    expected = {
        "track_count": 2,
        "object_count": 2,
        "replay_frame_count": 10,
        "inference_sample_count": 8,
        "natural_missing_frame_count": 2,
        "preinitialization_missing_frame_count": 1,
        "object_ids": [1, 2],
        "track_counts_by_object": {"1": 1, "2": 1},
    }
    builder.validate_expected_counts(observed, expected)
    expected["track_counts_by_object"]["2"] = 2
    with pytest.raises(RuntimeError, match="per-object track counts"):
        builder.validate_expected_counts(observed, expected)


def test_evaluator_macro_helper_is_object_balanced() -> None:
    rows = [
        {"object_id": 1, "loss": 1.0},
        {"object_id": 2, "loss": 10.0},
        {"object_id": 2, "loss": 20.0},
    ]
    assert evaluator.r2_evaluator.macro_object_mean(rows, "loss") == pytest.approx(8.0)
