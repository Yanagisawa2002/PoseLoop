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
from build_m5_r5_lmo_development import (  # noqa: E402
    FrameObservation,
    _consecutive_segments,
    lmo_bop_pose_m,
    missing_gap_frame_counts,
    select_best_window,
    validate_visible_mask_count,
)
from m5_r5_core import (  # noqa: E402
    AnchoredGateConfig,
    AnchoredGateObservation,
    NearestAnchoredConfidenceGate,
    run_nearest_anchored_confidence_gate,
)
from evaluate_m5_r5_development import _candidate_summary  # noqa: E402


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _available(
    index: int,
    *,
    anchor_x: float = 0.0,
    measurement_x: float = 0.0,
    mask: float = 0.1,
    depth: float = 0.9,
) -> AnchoredGateObservation:
    return AnchoredGateObservation(
        timestamp_s=index / 30.0,
        anchor_pose=_pose(anchor_x),
        measurement_pose=_pose(measurement_x),
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=mask,
        valid_depth_ratio=depth,
    )


def _missing(index: int, *, anchor_x: float) -> AnchoredGateObservation:
    return AnchoredGateObservation(
        timestamp_s=index / 30.0,
        anchor_pose=_pose(anchor_x),
        measurement_pose=None,
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=None,
        valid_depth_ratio=None,
    )


def test_missing_pose_is_bit_exact_anchor_and_uncertainty_grows() -> None:
    observations = [_available(0), _missing(1, anchor_x=0.01), _missing(2, anchor_x=0.02)]
    outputs = run_nearest_anchored_confidence_gate(
        observations, AnchoredGateConfig()
    )
    assert np.array_equal(outputs[1].pose, observations[1].anchor_pose)
    assert np.array_equal(outputs[2].pose, observations[2].anchor_pose)
    assert outputs[1].reason == outputs[2].reason == "missing_anchor_passthrough"
    assert outputs[1].confidence > outputs[2].confidence
    assert outputs[1].uncertainty < outputs[2].uncertainty


def test_large_low_confidence_outlier_cannot_take_pose_ownership() -> None:
    estimator = NearestAnchoredConfidenceGate(AnchoredGateConfig())
    estimator.step(_available(0))
    outlier = estimator.step(_available(1, anchor_x=0.001, measurement_x=1.0))
    assert outlier.reason == "confidence_anchor_fallback"
    assert not outlier.correction_applied
    assert np.array_equal(outlier.pose, _pose(0.001))
    assert outlier.confidence == pytest.approx(0.05)
    assert "reacquir" not in outlier.reason


def test_high_confidence_correction_is_bounded_and_not_persistent() -> None:
    config = AnchoredGateConfig(
        blend_gain=0.5,
        minimum_correction_confidence=0.0,
        maximum_anchor_measurement_disagreement=10.0,
    )
    outputs = run_nearest_anchored_confidence_gate(
        [
            _available(0),
            _available(1, anchor_x=0.0, measurement_x=0.01),
            _available(2, anchor_x=0.10, measurement_x=0.10),
        ],
        config,
    )
    assert outputs[1].correction_applied
    assert 0.0 < outputs[1].pose[0, 3] < 0.01
    assert outputs[2].pose[0, 3] == pytest.approx(0.10)


def test_object_local_support_drop_reduces_confidence_causally() -> None:
    estimator = NearestAnchoredConfidenceGate(AnchoredGateConfig())
    first = estimator.step(_available(0))
    second = estimator.step(_available(1, mask=0.025, depth=0.6))
    assert first.confidence == 1.0
    assert second.support_quality is not None
    assert 0.0 < second.support_quality < 1.0
    assert second.confidence < first.confidence


def test_missing_observation_rejects_support_and_missing_initialization() -> None:
    with pytest.raises(ValueError, match="missing observations"):
        AnchoredGateObservation(
            timestamp_s=0.0,
            anchor_pose=_pose(),
            measurement_pose=None,
            symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
            mask_area_fraction=0.1,
            valid_depth_ratio=None,
        )
    with pytest.raises(RuntimeError, match="cannot initialize"):
        NearestAnchoredConfidenceGate(AnchoredGateConfig()).step(
            _missing(0, anchor_x=0.0)
        )


def test_anchor_only_candidate_never_changes_pose() -> None:
    config = AnchoredGateConfig(
        blend_gain=0.0,
        minimum_correction_confidence=1.0,
        maximum_anchor_measurement_disagreement=0.0,
    )
    observations = [
        _available(0, anchor_x=0.0, measurement_x=0.005),
        _available(1, anchor_x=0.01, measurement_x=0.02),
        _missing(2, anchor_x=0.03),
    ]
    outputs = run_nearest_anchored_confidence_gate(observations, config)
    for observation, output in zip(observations, outputs, strict=True):
        assert np.array_equal(output.pose, observation.anchor_pose)


def test_window_selection_maximizes_natural_missing_and_starts_available() -> None:
    availability = [True] * 14
    availability[2:4] = [False, False]
    availability[8:11] = [False, False, False]
    selected = select_best_window(
        availability,
        window_length=6,
        minimum_available=3,
        minimum_missing=2,
    )
    assert selected == (5, 11)


def test_missing_gap_frame_strata_count_frames_not_runs() -> None:
    availability = [True, False, True, False, False, False, True] + [False] * 8
    assert missing_gap_frame_counts(availability) == {
        "short_gap_1_2": 1,
        "medium_gap_3_7": 3,
        "long_gap_8_plus": 8,
    }


def _frame(image_id: int) -> FrameObservation:
    return FrameObservation(
        scene_id=2,
        image_id=image_id,
        gt_instance_index=0,
        object_id=1,
        gt_pose_m=_pose(),
        camera_entry={},
        visible_fraction=1.0,
        mask_pixels=100,
        valid_depth_pixels=100,
        valid_depth_ratio=1.0,
        mask_area_fraction=0.1,
        median_depth_m=0.7,
        available=True,
        rgb_path=Path("rgb.png"),
        depth_path=Path("depth.png"),
        mask_path=Path("mask.png"),
    )


def test_gt_present_rows_split_at_source_frame_gaps() -> None:
    segments = _consecutive_segments([_frame(3), _frame(1), _frame(2), _frame(7)])
    assert [[row.image_id for row in segment] for segment in segments] == [
        [1, 2, 3],
        [7],
    ]


def test_visible_mask_metadata_tolerance_is_strictly_two_pixels() -> None:
    path = Path("mask.png")
    assert validate_visible_mask_count(100, 100, path) == 0
    assert validate_visible_mask_count(99, 100, path) == -1
    assert validate_visible_mask_count(102, 100, path) == 2
    with pytest.raises(ValueError, match="beyond two-pixel archive tolerance"):
        validate_visible_mask_count(103, 100, path)


def test_raw_foundationpose_score_is_not_a_gate_input() -> None:
    fields = set(AnchoredGateObservation.__dataclass_fields__)
    assert "foundationpose_top_score" not in fields
    assert "foundationpose_top_score_margin" not in fields


def test_lmo_rotation_adapter_projects_bounded_scale_to_so3() -> None:
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose, audit = lmo_bop_pose_m(
        {
            "cam_R_m2c": (rotation * 1.004).reshape(-1).tolist(),
            "cam_t_m2c": [10.0, -20.0, 500.0],
        }
    )
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], [0.01, -0.02, 0.5], atol=1e-12)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    assert audit["projection_frobenius"] > 0


def test_lmo_rotation_adapter_rejects_reflection_and_excessive_scale() -> None:
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="orientation preserving"):
        lmo_bop_pose_m(
            {
                "cam_R_m2c": reflection.reshape(-1).tolist(),
                "cam_t_m2c": [0.0, 0.0, 0.0],
            }
        )
    with pytest.raises(ValueError, match="orthogonality adapter bound"):
        lmo_bop_pose_m(
            {
                "cam_R_m2c": (np.eye(3) * 1.02).reshape(-1).tolist(),
                "cam_t_m2c": [0.0, 0.0, 0.0],
            }
        )


def test_available_only_object_is_excluded_only_from_missing_macro() -> None:
    common = {
        "candidate_id": "candidate",
        "nonfinite_output_count": 0,
        "missing_anchor_mismatch_count": 0,
        "unsafe_forced_reacquisition_count": 0,
        "correction_applied_frame_count": 0,
    }
    summary = _candidate_summary(
        [
            {
                **common,
                "object_id": 1,
                "natural_missing_frame_count": 4,
                "baseline_loss": 2.0,
                "proposed_loss": 1.0,
                "baseline_natural_missing_loss": 3.0,
                "proposed_natural_missing_loss": 3.0,
            },
            {
                **common,
                "object_id": 8,
                "natural_missing_frame_count": 0,
                "baseline_loss": 4.0,
                "proposed_loss": 2.0,
                "baseline_natural_missing_loss": None,
                "proposed_natural_missing_loss": None,
            },
        ],
        "candidate",
    )
    assert summary["baseline_macro_object_loss"] == pytest.approx(3.0)
    assert summary["proposed_macro_object_loss"] == pytest.approx(1.5)
    assert summary["macro_object_relative_improvement"] == pytest.approx(0.5)
    assert summary["missing_represented_object_count"] == 1
    assert summary["natural_missing_macro_object_relative_improvement"] == 0.0


def test_r5a_changes_only_outcome_blind_sparse_missing_eligibility() -> None:
    root = Path(__file__).resolve().parents[1]
    r5 = json.loads((root / "protocols" / "poseloop_m5_r5_protocol.json").read_text())
    r5a = json.loads(
        (root / "protocols" / "poseloop_m5_r5a_protocol.json").read_text()
    )
    assert r5a["methods"]["baseline"]["method"] == r5["methods"]["baseline"][
        "method"
    ]
    assert r5a["methods"]["baseline"]["configuration_id"] == r5["methods"][
        "baseline"
    ]["configuration_id"]
    assert r5a["methods"]["baseline"]["configuration_source"] == r5["methods"][
        "baseline"
    ]["configuration_source"]
    for key in ("shared_configuration", "candidate_order", "candidates"):
        assert r5a["methods"]["proposed_family"][key] == r5["methods"][
            "proposed_family"
        ][key]
    for key, value in r5["evaluation"]["development_gate"].items():
        assert r5a["evaluation"]["development_gate"][key] == value
    assert r5a["selection"]["window_length_frames"] == 192
    assert r5a["selection"]["minimum_mask_area_fraction"] == 0.001
    assert r5a["selection"]["minimum_valid_depth_ratio"] == 0.5
    assert r5a["selection"]["minimum_valid_depth_pixels"] == 4
    assert r5a["selection"]["eligible_window"] == {
        "available_frame_count_min": 96,
        "natural_missing_frame_count_min": 1,
    }
    assert r5a["evaluation"]["development_gate"][
        "missing_represented_object_count_min"
    ] == 8
