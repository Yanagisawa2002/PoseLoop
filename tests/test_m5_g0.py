"""Focused, fast contract tests for the M5-G0 synthetic estimation kernel."""

from __future__ import annotations

import math
import subprocess
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_m5_g0_manifests as manifests  # noqa: E402
import m1_common  # noqa: E402
import m5_g0_core as core  # noqa: E402
import m5_g0_metrics as metrics  # noqa: E402
import run_m5_g0 as runner  # noqa: E402
import render_m5_g0_zero_aware_plot as zero_plot  # noqa: E402


def _input(
    frame: int,
    pose: np.ndarray | None,
    symmetry: core.SymmetrySpec,
) -> core.EstimatorInput:
    return core.EstimatorInput(
        timestamp_s=frame * core.DT_SECONDS,
        measurement_pose=None if pose is None else np.asarray(pose).copy(),
        missing=pose is None,
        symmetry=symmetry,
    )


def _pose_with_z_degrees(angle_deg: float) -> np.ndarray:
    return core.z_rotation_transform(math.radians(angle_deg))


def test_se3_exp_log_round_trip() -> None:
    twist = np.array([0.12, -0.08, 0.035, 0.41, -0.27, 0.19])

    transform = core.se3_exp(twist)

    np.testing.assert_allclose(core.se3_log(transform), twist, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(
        core.se3_exp(core.se3_log(transform)), transform, atol=2e-12, rtol=0.0
    )


def test_se3_exp_log_is_stable_close_to_zero() -> None:
    twists = (
        np.zeros(6),
        np.array([1e-11, -2e-11, 3e-11, 1e-12, -2e-12, 3e-12]),
    )

    for twist in twists:
        recovered = core.se3_log(core.se3_exp(twist))
        assert np.isfinite(recovered).all()
        np.testing.assert_allclose(recovered, twist, atol=1e-14, rtol=0.0)


def test_se3_exp_log_is_stable_close_to_pi() -> None:
    axis = np.array([0.31, -0.72, 0.62])
    axis /= np.linalg.norm(axis)
    twist = np.concatenate([np.array([0.025, -0.04, 0.015]), axis * (math.pi - 1e-9)])

    transform = core.se3_exp(twist)
    recovered = core.se3_log(transform)

    assert np.isfinite(recovered).all()
    np.testing.assert_allclose(core.se3_exp(recovered), transform, atol=2e-8, rtol=0.0)
    assert math.isclose(np.linalg.norm(recovered[3:]), math.pi - 1e-9, abs_tol=2e-8)


def test_asymmetric_quotient_residual_is_the_standard_se3_residual() -> None:
    reference = core.se3_exp(np.array([0.04, -0.02, 0.7, 0.17, -0.11, 0.08]))
    increment = np.array([0.006, -0.004, 0.002, 0.07, 0.02, -0.03])
    measurement = reference @ core.se3_exp(increment)
    symmetry = core.symmetry_spec(core.SymmetryClass.ASYMMETRIC)

    residual, representative, index, _ = core.quotient_residual(
        reference, measurement, symmetry
    )
    expected = core.se3_log(core.rigid_inverse(reference) @ measurement)

    np.testing.assert_allclose(residual, expected, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(representative, measurement, atol=0.0, rtol=0.0)
    assert index == 0


def test_c2_quotient_accepts_180_degrees_but_not_90_degrees() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.C2)
    identity = np.eye(4)

    equivalent, _, _, _ = core.quotient_residual(
        identity, _pose_with_z_degrees(180.0), symmetry
    )
    non_equivalent, _, _, _ = core.quotient_residual(
        identity, _pose_with_z_degrees(90.0), symmetry
    )

    assert np.linalg.norm(equivalent) < 1e-10
    assert math.isclose(
        np.linalg.norm(non_equivalent[3:]), math.pi / 2.0, abs_tol=1e-10
    )


def test_every_c4_group_element_is_quotient_equivalent() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.C4)
    reference = core.se3_exp(np.array([0.08, -0.03, 0.5, 0.2, -0.1, 0.35]))

    for element in symmetry.finite_transforms:
        residual, _, _, _ = core.quotient_residual(
            reference, reference @ element, symmetry
        )
        assert np.linalg.norm(residual) < 2e-10


def test_continuous_quotient_is_axial_gauge_invariant() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.CONTINUOUS_AXIAL)
    reference = core.se3_exp(np.array([0.04, -0.03, 0.6, 0.3, -0.2, 0.1]))

    for gauge_deg in (-179.0, -91.0, -7.0, 0.0, 63.0, 177.0):
        equivalent = reference @ _pose_with_z_degrees(gauge_deg)
        residual, _, index, _ = core.quotient_residual(reference, equivalent, symmetry)
        translation, rotation = core.quotient_pose_error(
            reference, equivalent, symmetry
        )
        assert np.linalg.norm(residual) < 2e-10
        assert index is None
        assert translation < 1e-12
        assert rotation < 2e-6


def test_continuous_quotient_remains_sensitive_to_axis_tilt() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.CONTINUOUS_AXIAL)
    reference = core.se3_exp(np.array([0.02, -0.01, 0.5, 0.13, -0.19, 0.07]))
    tilt_deg = 20.0
    tilt = core.se3_exp(np.array([0.0, 0.0, 0.0, math.radians(tilt_deg), 0.0, 0.0]))
    measured = reference @ tilt @ _pose_with_z_degrees(137.0)

    residual, _, _, _ = core.quotient_residual(reference, measured, symmetry)
    error_deg = core.quotient_rotation_error_degrees(reference, measured, symmetry)

    assert math.isclose(error_deg, tilt_deg, abs_tol=1e-8)
    assert np.linalg.norm(residual[3:5]) > math.radians(15.0)


def test_finite_group_composition_inverse_and_right_action_convention() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.C4)
    elements = symmetry.finite_transforms

    for first_index, first in enumerate(elements):
        np.testing.assert_allclose(
            core.rigid_inverse(first), elements[-first_index % 4], atol=2e-15
        )
        for second_index, second in enumerate(elements):
            np.testing.assert_allclose(
                first @ second,
                elements[(first_index + second_index) % 4],
                atol=3e-15,
            )

    pose = core.se3_exp(np.array([0.11, -0.07, 0.6, 0.25, -0.18, 0.31]))
    right_equivalent = pose @ elements[1]
    left_applied = elements[1] @ pose
    right_translation, right_rotation = core.quotient_pose_error(
        pose, right_equivalent, symmetry
    )
    left_translation, _ = core.quotient_pose_error(pose, left_applied, symmetry)

    assert right_translation < 1e-12
    assert right_rotation < 1e-6
    assert left_translation > 0.05


def test_existing_raw_pose_error_and_vendored_bop_mssd_are_unchanged() -> None:
    identity = np.eye(4)
    predicted = _pose_with_z_degrees(30.0)
    predicted[:3, 3] = np.array([0.003, -0.004, 0.0])

    translation_mm, rotation_deg = m1_common.raw_pose_errors(predicted, identity)

    assert math.isclose(translation_mm, 5.0, abs_tol=1e-12)
    assert math.isclose(rotation_deg, 30.0, abs_tol=1e-12)

    model_points = np.array(
        [
            [-0.04, -0.03, -0.02],
            [0.05, -0.02, 0.01],
            [0.02, 0.06, -0.01],
            [-0.03, 0.01, 0.07],
        ]
    )
    gt_poses = np.stack([identity, identity])
    translated = identity.copy()
    translated[0, 3] = 0.01
    output_poses = np.stack([translated, _pose_with_z_degrees(180.0)])
    symmetry = core.symmetry_spec(core.SymmetryClass.C2)
    vendored = metrics.bop_normalized_mssd_diagnostic(
        gt_poses,
        output_poses,
        symmetry,
        model_points,
        0.2,
        prefer_vendored=True,
    )
    local = metrics.bop_normalized_mssd_diagnostic(
        gt_poses,
        output_poses,
        symmetry,
        model_points,
        0.2,
        prefer_vendored=False,
    )

    assert vendored["implementation"] == "vendored_bop_pose_error"
    assert vendored["official_benchmark_claim"] is False
    np.testing.assert_allclose(
        vendored["per_frame_normalized_mssd"], [0.05, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        vendored["per_frame_normalized_mssd"],
        local["per_frame_normalized_mssd"],
        atol=1e-12,
    )


@pytest.mark.parametrize("symmetry", list(core.SymmetryClass))
@pytest.mark.parametrize("motion", list(core.MotionFamily))
def test_ground_truth_generation_is_deterministic(
    symmetry: core.SymmetryClass, motion: core.MotionFamily
) -> None:
    first = core.generate_ground_truth(symmetry, motion, 31415)
    second = core.generate_ground_truth(symmetry, motion, 31415)
    different = core.generate_ground_truth(symmetry, motion, 31416)

    np.testing.assert_array_equal(first.timestamps, second.timestamps)
    np.testing.assert_array_equal(first.poses, second.poses)
    assert not np.array_equal(first.poses, different.poses)
    assert first.poses.shape == (core.FRAME_COUNT, 4, 4)
    np.testing.assert_allclose(np.diff(first.timestamps), core.DT_SECONDS, atol=1e-15)
    assert np.max(np.linalg.norm(first.poses[:, :3, 3], axis=1)) < 2.0
    assert first.reversal_frame == (
        core.FRAME_COUNT // 2
        if motion is core.MotionFamily.DIRECTION_REVERSAL
        else None
    )
    if symmetry is core.SymmetryClass.CONTINUOUS_AXIAL:
        assert abs(first.observable_rotation_axis_object[2]) < 1e-15


@pytest.mark.parametrize("stress", list(core.StressFamily))
def test_corruption_generation_is_deterministic(stress: core.StressFamily) -> None:
    truth = core.generate_ground_truth(core.SymmetryClass.C4, "CONSTANT_TWIST", 123)
    first = core.corrupt_measurements(truth, stress, 456)
    second = core.corrupt_measurements(truth, stress, 456)
    different = core.corrupt_measurements(truth, stress, 457)

    np.testing.assert_allclose(
        core.measurement_stream(first),
        core.measurement_stream(second),
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )
    assert not np.allclose(
        core.measurement_stream(first),
        core.measurement_stream(different),
        equal_nan=True,
    )
    assert first.dropout_intervals == second.dropout_intervals
    assert first.outlier_intervals == second.outlier_intervals
    assert [
        (
            frame.representative_index,
            frame.axial_gauge_rad,
            frame.dropout,
            frame.outlier,
        )
        for frame in first.evaluator_frames
    ] == [
        (
            frame.representative_index,
            frame.axial_gauge_rad,
            frame.dropout,
            frame.outlier,
        )
        for frame in second.evaluator_frames
    ]
    for left, right in zip(first.evaluator_frames, second.evaluator_frames):
        np.testing.assert_array_equal(left.ground_truth_pose, right.ground_truth_pose)
        np.testing.assert_array_equal(
            left.noisy_canonical_measurement_pose,
            right.noisy_canonical_measurement_pose,
        )

    expected_dropout = {
        core.StressFamily.DROPOUT_5: 5,
        core.StressFamily.DROPOUT_10: 10,
        core.StressFamily.COMBINED: 10,
    }.get(stress)
    expected_outlier = {
        core.StressFamily.OUTLIER_BURST: 3,
        core.StressFamily.COMBINED: 3,
    }.get(stress)
    assert sum(end - start for start, end in first.dropout_intervals) == (
        expected_dropout or 0
    )
    assert sum(end - start for start, end in first.outlier_intervals) == (
        expected_outlier or 0
    )


def test_estimator_input_schema_is_strict_and_evaluator_data_is_separate() -> None:
    input_fields = tuple(field.name for field in fields(core.EstimatorInput))
    evaluator_fields = tuple(field.name for field in fields(core.EvaluatorFrame))

    assert input_fields == ("timestamp_s", "measurement_pose", "missing", "symmetry")
    assert evaluator_fields == (
        "ground_truth_pose",
        "noisy_canonical_measurement_pose",
        "representative_index",
        "axial_gauge_rad",
        "dropout",
        "outlier",
    )
    assert not set(input_fields) & set(evaluator_fields)

    item = _input(0, np.eye(4), core.symmetry_spec("C2"))
    for forbidden in (
        "ground_truth_pose",
        "corruption_type",
        "representative_index",
        "dropout_end_time",
        "outlier",
        "motion_family",
    ):
        assert not hasattr(item, forbidden)
    with pytest.raises(TypeError):
        core.EstimatorInput(  # type: ignore[call-arg]
            timestamp_s=0.0,
            measurement_pose=np.eye(4),
            missing=False,
            symmetry=core.symmetry_spec("C2"),
            ground_truth_pose=np.eye(4),
        )


def test_standard_se3_ct_reacquires_a_persistent_raw_switch() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.C2)
    estimator = core.create_estimator(core.EstimatorKind.STANDARD_SE3_CT)
    estimator.step(_input(0, np.eye(4), symmetry))

    switched = _pose_with_z_degrees(180.0)
    outputs = [
        estimator.step(_input(frame, switched, symmetry)) for frame in range(1, 5)
    ]

    assert [output.reason for output in outputs[:3]] == ["innovation_rejected"] * 3
    assert not any(output.accepted for output in outputs[:3])
    assert outputs[3].accepted
    assert outputs[3].reason == "persistent_innovation_reacquired"
    assert core.rotation_angle_degrees(outputs[3].pose[:3, :3], switched[:3, :3]) < 1e-8


def test_nearest_representative_is_prefix_causal_and_truth_independent() -> None:
    truth = core.generate_ground_truth(
        core.SymmetryClass.C2, core.MotionFamily.CONSTANT_TWIST, 73, frame_count=18
    )
    baseline = core.corrupt_measurements(truth, core.StressFamily.NOMINAL, 91)
    prefix_length = 9
    changed_inputs = tuple(baseline.inputs[:prefix_length]) + tuple(
        replace(
            item,
            measurement_pose=item.measurement_pose
            @ core.symmetry_spec("C2").finite_transforms[1],
        )
        for item in baseline.inputs[prefix_length:]
    )
    changed_truth = replace(
        truth,
        poses=np.stack(
            [pose @ _pose_with_z_degrees(71.0) for pose in truth.poses], axis=0
        ),
    )
    changed_evaluator = tuple(
        replace(frame, ground_truth_pose=changed_truth.poses[index])
        for index, frame in enumerate(baseline.evaluator_frames)
    )
    variant = replace(
        baseline,
        inputs=changed_inputs,
        evaluator_frames=changed_evaluator,
        ground_truth=changed_truth,
    )

    first = core.run_estimator(baseline, core.EstimatorKind.NEAREST_REPRESENTATIVE_CT)
    second = core.run_estimator(variant, core.EstimatorKind.NEAREST_REPRESENTATIVE_CT)

    for left, right in zip(first[:prefix_length], second[:prefix_length]):
        np.testing.assert_array_equal(left.pose, right.pose)
        np.testing.assert_array_equal(left.body_twist, right.body_twist)
        assert left.accepted == right.accepted
        assert left.reason == right.reason
        assert left.selected_symmetry_index == right.selected_symmetry_index


def test_symquot_finite_choice_is_relative_to_prediction() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.C2)
    config = core.EstimatorConfig(innovation_gate_sigma=1e9, velocity_update_gain=0.0)
    estimator = core.create_estimator(core.EstimatorKind.SYMQUOT_CT, config)
    initial = _pose_with_z_degrees(80.0)
    estimator.step(_input(0, initial, symmetry))
    estimator.twist[5] = math.radians(-90.0) / core.DT_SECONDS
    measurement = _pose_with_z_degrees(100.0)
    predicted = initial @ core.se3_exp(estimator.twist * core.DT_SECONDS)

    prior_choice = core.quotient_residual(initial, measurement, symmetry)[2]
    predicted_choice = core.quotient_residual(predicted, measurement, symmetry)[2]
    output = estimator.step(_input(1, measurement, symmetry))

    assert prior_choice == 0
    assert predicted_choice == 1
    assert output.selected_symmetry_index == predicted_choice


def test_continuous_arbitrary_gauge_does_not_inject_axial_twist() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.CONTINUOUS_AXIAL)
    estimator = core.create_estimator(core.EstimatorKind.SYMQUOT_CT)
    outputs = []
    for frame, gauge_deg in enumerate((70.0, -120.0, 179.0, -30.0, 121.0)):
        outputs.append(
            estimator.step(_input(frame, _pose_with_z_degrees(gauge_deg), symmetry))
        )

    first_axis = outputs[0].pose[:3, :3] @ symmetry.axis_object
    for output in outputs:
        np.testing.assert_allclose(
            output.pose[:3, :3] @ symmetry.axis_object,
            first_axis,
            atol=2e-12,
        )
        assert abs(output.body_twist[5]) < 1e-14
        assert (
            core.quotient_rotation_error_degrees(outputs[0].pose, output.pose, symmetry)
            < 2e-6
        )


def test_dropout_uncertainty_growth_is_monotonic() -> None:
    truth = core.generate_ground_truth("C2", "CONSTANT_TWIST", 13)
    sequence = core.corrupt_measurements(truth, "DROPOUT_10", 21)
    outputs = core.run_estimator(sequence, core.EstimatorKind.SYMQUOT_CT)
    arrays = core.output_arrays(outputs)
    output_error = metrics.compute_per_frame_pose_errors(
        truth.poses, arrays["poses"], truth.symmetry
    )["normalized_pose_error"]
    measurement_error = metrics.compute_per_frame_pose_errors(
        truth.poses, core.measurement_stream(sequence), truth.symmetry
    )["normalized_pose_error"]
    result = metrics.compute_dropout_metrics(
        output_error,
        measurement_error,
        arrays["accepted"],
        arrays["uncertainty"],
        sequence.dropout_intervals,
    )
    start, end = sequence.dropout_intervals[0]
    scalar_uncertainty = arrays["uncertainty"].mean(axis=1)

    assert np.all(np.diff(scalar_uncertainty[start - 1 : end]) > 0.0)
    assert result["uncertainty_monotonic_all_intervals"] is True
    assert result["uncertainty_monotonic_failure_count"] == 0


def test_runtime_metrics_count_nonfinite_uncertainty_explicitly() -> None:
    poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    uncertainty = np.ones((2, 6), dtype=np.float64)
    uncertainty[1, 2] = np.nan
    uncertainty[1, 4] = np.inf

    result = metrics.compute_runtime_metrics(
        [0.001, 0.002], poses, uncertainty=uncertainty
    )

    assert result["nan_uncertainty_value_count"] == 1
    assert result["inf_uncertainty_value_count"] == 1
    assert result["nonfinite_uncertainty_value_count"] == 2


def test_estimator_rejects_nonfinite_uncertainty_state() -> None:
    symmetry = core.symmetry_spec(core.SymmetryClass.ASYMMETRIC)
    estimator = core.ConstantTwistEstimator(
        core.EstimatorKind.SYMQUOT_CT, core.EstimatorConfig()
    )
    estimator.step(_input(0, np.eye(4), symmetry))
    estimator.pose_variance[0] = np.nan

    with pytest.raises(FloatingPointError, match="state became non-finite"):
        estimator.step(_input(1, None, symmetry))


def test_flattened_trajectory_row_deduplicates_matching_symmetry_metadata() -> None:
    row = {
        "trajectory_id": "synthetic-row",
        "method": "SYMQUOT_CT",
        "config_id": "balanced",
        "symmetry_class": "C2",
        "motion_family": "STATIC",
        "stress_family": "NOMINAL",
        "status": "success",
        "metrics": {
            "symmetry_class": "C2",
            "accuracy": {"median_quotient_pose_error": 0.25},
        },
    }

    flattened = runner._flat_rows([row])

    assert flattened[0]["symmetry_class"] == "C2"
    assert flattened[0]["accuracy__median_quotient_pose_error"] == 0.25
    mismatch = dict(row)
    mismatch["metrics"] = {"symmetry_class": "C4"}
    with pytest.raises(ValueError, match="symmetry_class disagree"):
        runner._flat_rows([mismatch])


def test_zero_aware_dropout_plot_retains_explicit_zero_values() -> None:
    rows = []
    for method in runner.SEALED_METHODS:
        rows.append(
            {
                "method": method,
                "status": "success",
                "symmetry_class": "C2",
                "motion_family": "CONSTANT_TWIST",
                "stress_family": "DROPOUT_10",
                "metrics": {
                    "dropout": {"frames_to_return_within_nominal": {"median": 0.0}}
                },
            }
        )

    assert zero_plot.dropout_recovery_values(rows) == {
        method: 0.0 for method in runner.SEALED_METHODS
    }


def test_clear_outlier_is_rejected_and_estimator_recovers() -> None:
    truth = core.generate_ground_truth("C2", "CONSTANT_TWIST", 13)
    sequence = core.corrupt_measurements(truth, "COMBINED", 21)
    outputs = core.run_estimator(sequence, core.EstimatorKind.SYMQUOT_CT)
    dropout_start, dropout_end = sequence.dropout_intervals[0]
    outlier_start, outlier_end = sequence.outlier_intervals[0]

    assert all(not output.accepted for output in outputs[dropout_start:dropout_end])
    assert outputs[dropout_end].accepted
    assert outputs[dropout_end].reason == "measurement_updated"
    assert all(not output.accepted for output in outputs[outlier_start:outlier_end])
    assert all(
        output.reason == "innovation_rejected"
        for output in outputs[outlier_start:outlier_end]
    )
    assert outputs[outlier_end].accepted
    assert outputs[outlier_end].reason == "measurement_updated"


def test_all_methods_receive_the_identical_input_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = core.generate_ground_truth("C4", "CONSTANT_TWIST", 9, frame_count=12)
    sequence = core.corrupt_measurements(truth, "NOMINAL", 17)
    snapshot = core.measurement_stream(sequence).copy()
    original_create = core.create_estimator
    observed: dict[core.EstimatorKind, list[core.EstimatorInput]] = {}

    def recording_factory(
        kind: core.EstimatorKind | str,
        config: core.EstimatorConfig | None = None,
    ) -> object:
        key = kind if isinstance(kind, core.EstimatorKind) else core.EstimatorKind(kind)
        observed[key] = []
        real_estimator = original_create(key, config)

        class RecordingEstimator:
            def step(self, item: core.EstimatorInput) -> core.EstimatorOutput:
                observed[key].append(item)
                return real_estimator.step(item)

        return RecordingEstimator()

    monkeypatch.setattr(core, "create_estimator", recording_factory)
    for kind in core.EstimatorKind:
        core.run_estimator(sequence, kind)

    expected_ids = [id(item) for item in sequence.inputs]
    for kind in core.EstimatorKind:
        assert [id(item) for item in observed[kind]] == expected_ids
    np.testing.assert_allclose(
        core.measurement_stream(sequence), snapshot, atol=0.0, rtol=0.0, equal_nan=True
    )


def test_catastrophic_jump_metric_is_exact_on_hand_checked_trace() -> None:
    poses = np.stack(
        [
            _pose_with_z_degrees(0.0),
            _pose_with_z_degrees(180.0),
            _pose_with_z_degrees(185.0),
            _pose_with_z_degrees(5.0),
        ]
    )
    result = metrics.compute_representation_stability(
        np.arange(len(poses)) * core.DT_SECONDS, poses, core.symmetry_spec("C2")
    )

    np.testing.assert_array_equal(result["catastrophic_jump_mask"], [True, False, True])
    assert result["catastrophic_representation_jump_count"] == 2
    np.testing.assert_allclose(
        result["raw_rotation_step_deg"], [180.0, 5.0, 180.0], atol=1e-6
    )
    np.testing.assert_allclose(
        result["quotient_rotation_step_deg"], [0.0, 5.0, 0.0], atol=1e-6
    )


def test_direction_change_lag_metric_is_exact_on_hand_checked_trace() -> None:
    reference = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    estimated = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.1, -0.3, -1.0])

    assert metrics.direction_change_lag(reference, estimated, reversal_frame=4) == 2
    assert metrics.direction_change_lag(reference, reference, reversal_frame=4) == 0


def test_development_and_sealed_manifests_are_disjoint_and_complete() -> None:
    development = manifests._seed_rows(  # noqa: SLF001
        "development", manifests.DEVELOPMENT_REPLICATES
    )
    sealed = manifests._seed_rows("sealed", manifests.SEALED_REPLICATES)  # noqa: SLF001
    manifests._validate_seed_rows(development, "development")  # noqa: SLF001
    manifests._validate_seed_rows(sealed, "sealed")  # noqa: SLF001

    assert len(development) == manifests.EXPECTED_DEVELOPMENT_TRAJECTORIES == 1152
    assert len(sealed) == manifests.EXPECTED_SEALED_TRAJECTORIES == 2304
    assert not (
        {row["trajectory_id"] for row in development}
        & {row["trajectory_id"] for row in sealed}
    )
    assert not (
        {row["trajectory_seed"] for row in development}
        & {row["trajectory_seed"] for row in sealed}
    )
    assert not (
        {row["corruption_seed"] for row in development}
        & {row["corruption_seed"] for row in sealed}
    )

    for rows, expected_replicates in (
        (development, manifests.DEVELOPMENT_REPLICATES),
        (sealed, manifests.SEALED_REPLICATES),
    ):
        counts = Counter(
            (row["symmetry_class"], row["motion_family"], row["stress_family"])
            for row in rows
        )
        assert len(counts) == 4 * 4 * 6
        assert set(counts.values()) == {expected_replicates}


def test_existing_release_summary_check_still_passes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPTS_DIR / "render_precomputed_report.py"),
            "--check",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS:" in completed.stdout
    assert "matches results.json" in completed.stdout
