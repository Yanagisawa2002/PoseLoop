from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_m5_r6a_ycbv_sealed_failure as failure_audit  # noqa: E402
import build_m5_r6a_ycbv_sealed as builder  # noqa: E402
import run_m5_r6a_ycbv_inference as inference  # noqa: E402
from m5_r6_core import MeasurementFirstConfig  # noqa: E402


def _protocol(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "protocols" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(rotation: np.ndarray) -> dict:
    return {
        "obj_id": 1,
        "cam_R_m2c": np.asarray(rotation, dtype=np.float64).reshape(-1).tolist(),
        "cam_t_m2c": [0.0, 0.0, 1000.0],
    }


def _frame(scene_id: int, image_id: int, available: bool) -> builder.FrameObservation:
    pose = np.eye(4, dtype=np.float64)
    return builder.FrameObservation(
        scene_id=scene_id,
        image_id=image_id,
        gt_instance_index=0,
        object_id=1,
        gt_pose_m=pose,
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


def test_sealed_candidate_and_numerical_gates_equal_development_freeze() -> None:
    development = _protocol("poseloop_m5_r6a_protocol.json")
    sealed = _protocol("poseloop_m5_r6a_ycbv_sealed_protocol.json")
    development_family = development["methods"]["proposed_family"]
    expected_config = {
        **development_family["shared_configuration"],
        **development_family["candidates"]["measurement_first"],
    }
    development_gate = development["evaluation"]["development_gate"]
    sealed_gate = sealed["evaluation"]["sealed_gate"]

    assert sealed["protocol_id"] == builder.PROTOCOL_ID == inference.PROTOCOL_ID
    assert sealed["stage_id"] == builder.STAGE_ID == "M5-R6A-S1"
    assert sealed["development_freeze"]["result_commit"] == "38d861f"
    assert sealed["development_freeze"]["result_sha256"] == (
        "1ee295ccc8aa2c2c06675b30694819e01e5ca23b61bdffb7d60190a3b5247aa7"
    )
    assert sealed["development_freeze"]["fold_winner_counts"] == {
        "measurement_first": 14
    }
    assert sealed["methods"]["proposed"]["candidate_id"] == "measurement_first"
    assert sealed["methods"]["proposed"]["configuration"] == expected_config
    assert MeasurementFirstConfig(**expected_config).blend_gain == 1.0
    assert sealed_gate["represented_object_count_min"] == development_gate[
        "represented_object_count_min"
    ]
    assert sealed_gate["track_count_min"] == development_gate["track_count_min"]
    assert sealed_gate["missing_represented_object_count_min"] == development_gate[
        "missing_represented_object_count_min"
    ]
    assert sealed_gate["natural_missing_frame_count_min"] == development_gate[
        "natural_missing_frame_count_min"
    ]
    assert sealed_gate["macro_object_relative_improvement_min"] == development_gate[
        "cross_fitted_macro_object_relative_improvement_min"
    ]
    assert sealed_gate["natural_missing_frame_relative_improvement_min"] == (
        development_gate[
            "cross_fitted_natural_missing_frame_relative_improvement_min"
        ]
    )
    assert sealed_gate[
        "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"
    ] == development_gate[
        "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"
    ]
    assert sealed_gate["missing_frame_uncertainty_error_spearman_min"] == (
        development_gate["missing_frame_uncertainty_error_spearman_min"]
    )


def test_sealed_source_and_selection_are_outcome_blind_and_object_disjoint() -> None:
    protocol = _protocol("poseloop_m5_r6a_ycbv_sealed_protocol.json")
    assert protocol["source"]["object_ids"] == list(range(1, 22))
    assert protocol["source"]["physical_object_overlap_with_ruapc"] == 0
    assert protocol["selection"]["window_length_frames"] == 128
    assert protocol["selection"]["eligible_window"] == {
        "available_frame_count_min": 64,
        "natural_missing_frame_count_min": 0,
    }
    assert not protocol["selection"]["artificial_frame_deletion"]
    assert not protocol["selection"]["selection_reads_method_prediction_or_pose_error"]
    assert protocol["single_open_policy"]["archive_and_labels_unavailable_at_freeze"]
    assert not protocol["single_open_policy"]["reselection_or_retuning_after_open"]
    assert protocol["methods"]["proposed"]["sealed_candidate_reselection"] is False


def test_ycbv_mask_metadata_tolerance_is_strictly_two_pixels() -> None:
    path = Path("mask.png")
    assert builder.validate_visible_mask_count(20, 20, path) == 0
    assert builder.validate_visible_mask_count(20, 18, path) == 2
    assert builder.validate_visible_mask_count(20, 22, path) == -2
    with pytest.raises(ValueError, match="actual=20 declared=17"):
        builder.validate_visible_mask_count(20, 17, path)


def test_conditional_rotation_adapter_preserves_strict_and_projects_bounded() -> None:
    strict_pose, strict_audit = builder.sealed_bop_pose_m(_entry(np.eye(3)))
    np.testing.assert_array_equal(strict_pose[:3, :3], np.eye(3))
    assert strict_audit["conditional_projection_applied"] is False

    scaled = np.diag([1.002, 1.0, 1.0])
    projected_pose, projected_audit = builder.sealed_bop_pose_m(_entry(scaled))
    np.testing.assert_allclose(projected_pose[:3, :3], np.eye(3), atol=1e-12)
    assert projected_audit["conditional_projection_applied"] is True

    with pytest.raises(ValueError, match="orthogonality adapter bound"):
        builder.sealed_bop_pose_m(_entry(np.diag([1.2, 1.0, 1.0])))


def test_sealed_window_prefers_more_missing_then_lower_scene_and_start() -> None:
    rows = [
        *[_frame(2, index, value) for index, value in enumerate([True, True, False, False])],
        *[_frame(1, index, value) for index, value in enumerate([True, False, False, False])],
        *[
            _frame(1, index + 10, value)
            for index, value in enumerate([True, False, False, False])
        ],
    ]
    selection = {
        "window_length_frames": 4,
        "eligible_window": {
            "available_frame_count_min": 1,
            "natural_missing_frame_count_min": 0,
        },
    }
    selected = builder.select_object_window(rows, selection)
    assert selected is not None
    assert selected[0].scene_id == 1
    assert selected[0].image_id == 0
    assert sum(not row.available for row in selected) == 3


def test_failure_audit_uses_frozen_gap_keys_in_checks_and_report() -> None:
    protocol = _protocol("poseloop_m5_r6a_ycbv_sealed_protocol.json")
    summary = {
        "represented_object_count": 21,
        "missing_represented_object_count": 7,
        "natural_missing_frame_count": 357,
        "missing_gap_frame_counts": {
            "short_gap_1_2": 105,
            "medium_gap_3_7": 131,
            "long_gap_8_plus": 121,
        },
    }
    checks = failure_audit.build_gate_checks(summary, protocol)
    assert checks["missing_gap_short_gap_1_2_frame_count"]["observed"] == 105
    assert checks["missing_gap_medium_gap_3_7_frame_count"]["observed"] == 131
    assert checks["missing_gap_long_gap_8_plus_frame_count"]["observed"] == 121

    report = failure_audit.render_report(
        {
            "builder_failure": {"message": failure_audit.EXPECTED_BUILDER_FAILURE},
            "gate_checks": checks,
            "by_object": {
                "13": {
                    "scene_id": 53,
                    "source_start_image_id": 709,
                    "available_frame_count": 105,
                    "natural_missing_frame_count": 23,
                    "missing_gap_frame_counts": {
                        "short_gap_1_2": 2,
                        "medium_gap_3_7": 0,
                        "long_gap_8_plus": 21,
                    },
                }
            },
        }
    )
    assert "| Short-gap frames | 8 | 105 | PASS |" in report
    assert "| 13 | 53 | 709 | 105 | 23 | 2 | 0 | 21 |" in report
