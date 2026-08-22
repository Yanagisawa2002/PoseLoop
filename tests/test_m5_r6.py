from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m5_g0_core as legacy  # noqa: E402
import build_m5_r6_ruapc_development as builder  # noqa: E402
import evaluate_m5_r5_development as evaluator  # noqa: E402
import run_m5_r5_inference as inference  # noqa: E402
from build_m5_r6_ruapc_development import (  # noqa: E402
    FrameObservation,
    select_object_window,
    summarize_visible_mask_metadata,
    validate_visible_mask_count,
)
from m5_r6_core import (  # noqa: E402
    MeasurementFirstConfidenceGate,
    MeasurementFirstConfig,
    MeasurementFirstObservation,
    run_measurement_first_confidence_gate,
)


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _frame(image_id: int, available: bool) -> FrameObservation:
    return FrameObservation(
        scene_id=1,
        image_id=image_id,
        gt_instance_index=0,
        object_id=1,
        gt_pose_m=_pose(),
        camera_entry={
            "cam_K": [531.15, 0.0, 320.0, 0.0, 531.15, 240.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        },
        visible_fraction=0.5,
        mask_pixels=100,
        mask_metadata_delta_pixels=0,
        valid_depth_pixels=90,
        valid_depth_ratio=0.9,
        mask_area_fraction=0.01,
        median_depth_m=0.8,
        available=available,
        rgb_path=Path(f"rgb/{image_id:06d}.png"),
        depth_path=Path(f"depth/{image_id:06d}.png"),
        mask_path=Path(f"mask_visib/{image_id:06d}_000000.png"),
    )


def _available(
    index: int,
    *,
    anchor_x: float = 0.0,
    measurement_x: float = 0.0,
    mask: float = 0.1,
    depth: float = 0.9,
) -> MeasurementFirstObservation:
    return MeasurementFirstObservation(
        timestamp_s=index / 30.0,
        anchor_pose=_pose(anchor_x),
        measurement_pose=_pose(measurement_x),
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=mask,
        valid_depth_ratio=depth,
    )


def _missing(index: int, *, anchor_x: float) -> MeasurementFirstObservation:
    return MeasurementFirstObservation(
        timestamp_s=index / 30.0,
        anchor_pose=_pose(anchor_x),
        measurement_pose=None,
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=None,
        valid_depth_ratio=None,
    )


def test_large_anchor_disagreement_does_not_reject_reliable_measurement() -> None:
    estimator = MeasurementFirstConfidenceGate(
        MeasurementFirstConfig(blend_gain=1.0, minimum_measurement_confidence=0.35)
    )
    estimator.step(_available(0, anchor_x=0.0, measurement_x=0.0))
    output = estimator.step(_available(1, anchor_x=0.0, measurement_x=0.25))
    assert output.correction_applied
    assert output.reason == "measurement_first_correction"
    assert output.anchor_measurement_disagreement == pytest.approx(25.0)
    np.testing.assert_allclose(output.pose, _pose(0.25), atol=1e-12)


def test_measurement_output_never_becomes_persistent_pose_owner() -> None:
    outputs = run_measurement_first_confidence_gate(
        [
            _available(0, anchor_x=0.0, measurement_x=0.0),
            _available(1, anchor_x=0.0, measurement_x=0.1),
            _available(2, anchor_x=-0.2, measurement_x=-0.2),
        ],
        MeasurementFirstConfig(blend_gain=1.0, minimum_measurement_confidence=0.0),
    )
    assert outputs[1].pose[0, 3] == pytest.approx(0.1)
    assert outputs[2].pose[0, 3] == pytest.approx(-0.2)


def test_missing_pose_is_bit_exact_anchor_and_uncertainty_grows() -> None:
    observations = [
        _available(0, anchor_x=0.0, measurement_x=0.2),
        _missing(1, anchor_x=0.01),
        _missing(2, anchor_x=0.02),
    ]
    outputs = run_measurement_first_confidence_gate(
        observations, MeasurementFirstConfig()
    )
    assert np.array_equal(outputs[1].pose, observations[1].anchor_pose)
    assert np.array_equal(outputs[2].pose, observations[2].anchor_pose)
    assert outputs[1].reason == outputs[2].reason == "missing_anchor_passthrough"
    assert outputs[1].confidence > outputs[2].confidence
    assert outputs[1].uncertainty < outputs[2].uncertainty


def test_anchor_staleness_reduces_missing_confidence() -> None:
    aligned = run_measurement_first_confidence_gate(
        [_available(0), _missing(1, anchor_x=0.0)], MeasurementFirstConfig()
    )
    stale = run_measurement_first_confidence_gate(
        [
            _available(0, anchor_x=0.0, measurement_x=0.2),
            _missing(1, anchor_x=0.0),
        ],
        MeasurementFirstConfig(),
    )
    assert stale[1].confidence < aligned[1].confidence
    assert stale[1].uncertainty > aligned[1].uncertainty


def test_temporally_inconsistent_measurement_can_fall_back_to_anchor() -> None:
    estimator = MeasurementFirstConfidenceGate(
        MeasurementFirstConfig(minimum_measurement_confidence=0.5)
    )
    estimator.step(_available(0, measurement_x=0.00))
    estimator.step(_available(1, measurement_x=0.01))
    estimator.step(_available(2, measurement_x=0.02))
    output = estimator.step(_available(3, anchor_x=0.03, measurement_x=1.0))
    assert output.confidence < 0.5
    assert not output.correction_applied
    assert output.reason == "measurement_confidence_anchor_fallback"
    assert np.array_equal(output.pose, _pose(0.03))


def test_object_local_support_drop_reduces_measurement_confidence() -> None:
    estimator = MeasurementFirstConfidenceGate(MeasurementFirstConfig())
    first = estimator.step(_available(0))
    second = estimator.step(_available(1, mask=0.025, depth=0.6))
    assert first.confidence == 1.0
    assert second.support_quality is not None
    assert 0.0 < second.support_quality < 1.0
    assert second.confidence < first.confidence


def test_anchor_only_candidate_never_changes_pose() -> None:
    observations = [
        _available(0, anchor_x=0.0, measurement_x=0.1),
        _available(1, anchor_x=0.2, measurement_x=0.4),
        _missing(2, anchor_x=0.3),
    ]
    outputs = run_measurement_first_confidence_gate(
        observations,
        MeasurementFirstConfig(blend_gain=0.0, minimum_measurement_confidence=1.0),
    )
    for observation, output in zip(observations, outputs, strict=True):
        assert np.array_equal(output.pose, observation.anchor_pose)


def test_missing_observation_rejects_support_and_missing_initialization() -> None:
    with pytest.raises(ValueError, match="missing observations"):
        MeasurementFirstObservation(
            timestamp_s=0.0,
            anchor_pose=_pose(),
            measurement_pose=None,
            symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
            mask_area_fraction=0.1,
            valid_depth_ratio=None,
        )
    with pytest.raises(RuntimeError, match="cannot initialize"):
        MeasurementFirstConfidenceGate(MeasurementFirstConfig()).step(
            _missing(0, anchor_x=0.0)
        )


def test_raw_foundationpose_score_is_not_a_gate_input() -> None:
    fields = set(MeasurementFirstObservation.__dataclass_fields__)
    assert "foundationpose_top_score" not in fields
    assert "foundationpose_top_score_margin" not in fields


def test_frozen_protocol_has_all_ruapc_objects_and_sealed_stop_rule() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "poseloop_m5_r6_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    object_ids = protocol["source"]["development_object_ids"]
    scene_map = protocol["source"]["scene_object_map"]
    gate = protocol["evaluation"]["development_gate"]
    family = protocol["methods"]["proposed_family"]

    assert protocol["protocol_id"] == "poseloop-m5-r6-v1"
    assert object_ids == list(range(1, 15))
    assert scene_map == {str(index): index for index in range(1, 15)}
    assert gate["represented_object_count_min"] == 14
    assert gate["track_count_min"] == 14
    assert gate["missing_represented_object_count_min"] == 14
    assert family["method"] == "MEASUREMENT_FIRST_OBJECT_CONFIDENCE_GATE"
    assert family["candidate_order"] == [
        "anchor_calibrated",
        "measurement_cautious",
        "measurement_balanced",
        "measurement_first",
    ]
    assert protocol["source"]["archives"]["sealed_ycbv"]["development_access"] == (
        "FORBIDDEN"
    )
    assert protocol["evaluation"]["stop_rule"].endswith("forbids YCB-V access.")


def test_r6a_retains_all_objects_without_relaxing_numerical_effect_gates() -> None:
    protocol_root = Path(__file__).resolve().parents[1] / "protocols"
    r6 = json.loads(
        (protocol_root / "poseloop_m5_r6_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    r6a = json.loads(
        (protocol_root / "poseloop_m5_r6a_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    r6_gate = r6["evaluation"]["development_gate"]
    r6a_gate = r6a["evaluation"]["development_gate"]

    assert builder.PROTOCOL_ID == r6a["protocol_id"] == "poseloop-m5-r6a-v1"
    assert builder.STAGE_ID == r6a["stage_id"] == "M5-R6A"
    assert r6a["predecessor"]["evidence_commit"] == "95cd5f6"
    assert r6a["selection"]["eligible_window"]["natural_missing_frame_count_min"] == 0
    assert r6a_gate["represented_object_count_min"] == 14
    assert r6a_gate["track_count_min"] == 14
    assert r6a_gate["missing_represented_object_count_min"] == 12
    assert r6a["source"]["always_available_object_ids"] == [2, 8]
    assert r6a["source"]["natural_missing_object_ids"] == [
        1,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
    ]
    assert r6a["methods"] == r6["methods"]
    unchanged_gates = {
        "natural_missing_frame_count_min",
        "cross_fitted_macro_object_relative_improvement_min",
        "cross_fitted_natural_missing_frame_relative_improvement_min",
        "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min",
        "missing_frame_uncertainty_error_spearman_min",
        "missing_anchor_mismatch_count_max",
        "unsafe_forced_reacquisition_count_max",
        "nonfinite_output_count_max",
    }
    for name in unchanged_gates:
        assert r6a_gate[name] == r6_gate[name]
    assert r6a["source"]["archives"]["sealed_ycbv"]["development_access"] == (
        "FORBIDDEN"
    )


def test_ruapc_window_selection_maximizes_natural_missing_and_starts_available() -> None:
    rows = [_frame(index, value) for index, value in enumerate([True] * 5 + [False] * 3)]
    selection = {
        "window_length_frames": 4,
        "eligible_window": {
            "available_frame_count_min": 1,
            "natural_missing_frame_count_min": 1,
        },
    }
    selected = select_object_window(rows, selection)
    assert selected is not None
    assert [row.image_id for row in selected] == [4, 5, 6, 7]
    assert selected[0].available
    assert sum(not row.available for row in selected) == 3


def test_ruapc_mask_metadata_tolerance_is_strictly_one_pixel() -> None:
    assert validate_visible_mask_count(17, 17, Path("mask.png")) == 0
    assert validate_visible_mask_count(17, 16, Path("mask.png")) == 1
    assert validate_visible_mask_count(17, 18, Path("mask.png")) == -1
    with pytest.raises(ValueError, match="actual=17 declared=15"):
        validate_visible_mask_count(17, 15, Path("mask.png"))
    audit = summarize_visible_mask_metadata([-1, 0, 0, 1])
    assert audit["allowed_absolute_delta_pixels"] == 1
    assert audit["availability_uses_actual_png_count"]
    assert audit["signed_delta_counts"] == {"-1": 1, "0": 2, "1": 1}


def test_evaluator_dispatches_r6_to_measurement_first_runtime() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "poseloop_m5_r6a_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = evaluator._config_from_protocol(protocol, "measurement_first")
    observation_type, runner = evaluator._proposed_runtime(protocol)

    assert isinstance(config, MeasurementFirstConfig)
    assert config.blend_gain == 1.0
    assert config.minimum_measurement_confidence == 0.35
    assert observation_type is MeasurementFirstObservation
    assert runner is run_measurement_first_confidence_gate


def test_generalized_inference_row_requires_frozen_ruapc_modality() -> None:
    row = {
        "record_type": "m5_r5_development_inference_sample",
        "sample_id": "ruapc-test",
        "sensor_modality": "ruapc_rgbd",
        "input_available": True,
        "evaluator_label_read": False,
        "input_unit_check_pose_m": _pose().tolist(),
        "input_mask_area_fraction": 0.01,
    }
    effective = inference.effective_row(row, "ruapc_rgbd")
    assert effective["gt_model_to_camera_pose_m"] == _pose().tolist()
    assert effective["visibility_bin"] == "input_only"
    with pytest.raises(ValueError, match="Invalid M5-R5 inference sample"):
        inference.effective_row(row, "linemod_rgbd")
