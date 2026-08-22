"""Contract tests for the M5-R2 observed-dropout recorded replay."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_m5_r2_real_replay as builder  # noqa: E402
import evaluate_m5_r2_once as evaluator  # noqa: E402
import m5_g0_core as core  # noqa: E402
import m5_r1_core as r1_core  # noqa: E402
import m5_r2_core as r2_core  # noqa: E402
import run_m5_r2_inference as inference  # noqa: E402


def _audit_frame(available: bool) -> dict[str, bool]:
    return {"observation_available": available}


def test_track_selection_requires_natural_missing_after_initialization() -> None:
    assert builder.track_is_selected(
        [_audit_frame(False), _audit_frame(True), _audit_frame(False)]
        + [_audit_frame(True)] * 4
    )
    assert not builder.track_is_selected([_audit_frame(True)] * 6)
    assert not builder.track_is_selected(
        [_audit_frame(False), _audit_frame(True), _audit_frame(True)]
    )


def test_observation_availability_uses_only_frozen_input_thresholds() -> None:
    row = {
        "image_height": 720,
        "image_width": 1280,
        "visible_mask_pixel_count": math.ceil(0.001 * 720 * 1280),
        "valid_depth_pixel_count": 4,
        "valid_depth_ratio_inside_mask": 0.5,
    }
    assert builder.observation_is_available(row)
    assert not builder.observation_is_available(
        {**row, "valid_depth_ratio_inside_mask": 0.499999}
    )
    assert not builder.observation_is_available(
        {**row, "visible_mask_pixel_count": 1}
    )
    assert not builder.observation_is_available(
        {**row, "valid_depth_pixel_count": 3}
    )


def test_inference_adapter_rejects_evaluator_fields() -> None:
    row = {
        "record_type": "m5_r2_inference_sample",
        "sample_id": "sample",
        "sensor_modality": "realsense",
        "input_available": True,
        "input_unit_check_pose_m": np.eye(4).tolist(),
        "input_mask_area_fraction": 0.01,
    }
    effective = inference.effective_row(row)
    np.testing.assert_array_equal(effective["gt_model_to_camera_pose_m"], np.eye(4))
    assert effective["visibility_bin"] == "input_only"

    leaked = {**row, "gt_model_to_camera_pose_m": np.eye(4).tolist()}
    try:
        inference.effective_row(leaked)
    except ValueError as error:
        assert "Evaluator field leaked" in str(error)
    else:  # pragma: no cover - assertion path
        raise AssertionError("Evaluator leakage was accepted")


def test_official_continuous_x_axis_is_preserved() -> None:
    symmetry = evaluator.symmetry_from_model_info(
        {
            "symmetries_continuous": [
                {"axis": [1.0, 0.0, 0.0], "offset": [0.0, 0.0, 0.0]}
            ]
        }
    )
    assert symmetry.continuous
    np.testing.assert_array_equal(symmetry.axis_object, [1.0, 0.0, 0.0])
    reference = core.se3_exp(np.array([0.01, -0.02, 0.7, 0.1, -0.2, 0.3]))
    equivalent = reference @ r2_core.axial_rotation_transform(
        symmetry.axis_object, math.radians(127.0)
    )
    basis, filter_symmetry = r2_core.canonicalization_for_symmetry(symmetry)
    np.testing.assert_allclose(basis[:3, :3] @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    residual, _, _, _ = core.quotient_residual(
        r2_core.to_canonical_pose(reference, basis),
        r2_core.to_canonical_pose(equivalent, basis),
        filter_symmetry,
    )
    assert np.linalg.norm(residual) < 2e-10


def test_x_axis_kalman_observes_z_tilt_instead_of_suppressing_it() -> None:
    symmetry = core.SymmetrySpec(
        core.SymmetryClass.CONTINUOUS_AXIAL,
        (np.eye(4),),
        axis_object=np.array([1.0, 0.0, 0.0]),
    )
    estimator = r1_core.QuotientCVKalmanEstimator(
        r1_core.QuotientCVKalmanConfig(innovation_gate_sigma=1e6)
    )
    identity = np.eye(4)
    basis, filter_symmetry = r2_core.canonicalization_for_symmetry(symmetry)
    estimator.step(
        core.EstimatorInput(
            0.0,
            r2_core.to_canonical_pose(identity, basis),
            missing=False,
            symmetry=filter_symmetry,
        )
    )
    tilted = r2_core.axial_rotation_transform(
        np.array([0.0, 0.0, 1.0]), math.radians(5.0)
    )
    output = estimator.step(
        core.EstimatorInput(
            0.05,
            r2_core.to_canonical_pose(tilted, basis),
            missing=False,
            symmetry=filter_symmetry,
        )
    )

    restored = r2_core.from_canonical_pose(output.pose, basis)
    observable_tilt = core.quotient_rotation_error_degrees(
        identity, restored, symmetry
    )
    assert observable_tilt > 0.1
    assert output.body_twist[5] == 0.0


def test_official_four_element_discrete_group_is_used_exactly() -> None:
    transforms = []
    for axis in np.eye(3):
        transform = r2_core.axial_rotation_transform(axis, math.pi)
        transforms.append(transform.reshape(-1).tolist())
    symmetry = evaluator.symmetry_from_model_info(
        {"symmetries_discrete": transforms}
    )
    assert symmetry.symmetry_class is core.SymmetryClass.C4
    assert len(symmetry.finite_transforms) == 4
    reference = core.se3_exp(np.array([0.01, -0.02, 0.7, 0.1, -0.2, 0.3]))
    for transform in symmetry.finite_transforms:
        residual, _, _, _ = core.quotient_residual(
            reference, reference @ transform, symmetry
        )
        assert np.linalg.norm(residual) < 2e-10


def test_macro_object_mean_does_not_overweight_track_rich_objects() -> None:
    rows = [
        {"object_id": 1, "value": 1.0},
        {"object_id": 1, "value": 3.0},
        {"object_id": 1, "value": 5.0},
        {"object_id": 2, "value": 9.0},
    ]
    assert evaluator.macro_object_mean(rows, "value") == 6.0


def test_hierarchical_bootstrap_is_deterministic_and_paired() -> None:
    rows = [
        {"object_id": 1, "baseline_loss": 2.0, "proposed_loss": 1.0},
        {"object_id": 1, "baseline_loss": 4.0, "proposed_loss": 2.0},
        {"object_id": 2, "baseline_loss": 8.0, "proposed_loss": 4.0},
    ]
    first = evaluator.hierarchical_paired_bootstrap(rows, seed=5202, resamples=100)
    second = evaluator.hierarchical_paired_bootstrap(rows, seed=5202, resamples=100)
    assert first == second
    assert math.isclose(first["mean_relative_improvement"], 0.5, abs_tol=1e-15)
    assert math.isclose(
        first["one_sided_90pct_lower_relative_improvement"], 0.5, abs_tol=1e-15
    )
