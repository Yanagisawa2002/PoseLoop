from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m5_g0_core as legacy  # noqa: E402
from build_m5_r4_hb_development import (  # noqa: E402
    difficulty_stratum,
    select_best_window,
    select_tracks,
    validate_visible_mask_count,
)
from evaluate_m5_r4_development import spearman_correlation  # noqa: E402
from m5_r4_core import (  # noqa: E402
    AdaptiveConfidenceConfig,
    AdaptiveObservation,
    ObjectLocalAdaptiveConfidenceEstimator,
    run_object_local_adaptive_confidence,
)


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _available(index: int, *, mask: float = 0.1, depth: float = 0.9) -> AdaptiveObservation:
    return AdaptiveObservation(
        timestamp_s=index / 30.0,
        measurement_pose=_pose(index * 0.001),
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=mask,
        valid_depth_ratio=depth,
    )


def _missing(index: int) -> AdaptiveObservation:
    return AdaptiveObservation(
        timestamp_s=index / 30.0,
        measurement_pose=None,
        symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
        mask_area_fraction=None,
        valid_depth_ratio=None,
    )


def test_adaptive_observation_rejects_support_on_missing() -> None:
    with pytest.raises(ValueError, match="missing observations"):
        AdaptiveObservation(
            timestamp_s=0.0,
            measurement_pose=None,
            symmetry=legacy.symmetry_spec(legacy.SymmetryClass.ASYMMETRIC),
            mask_area_fraction=0.1,
            valid_depth_ratio=None,
        )


def test_object_local_support_reduces_confidence_causally() -> None:
    estimator = ObjectLocalAdaptiveConfidenceEstimator(AdaptiveConfidenceConfig())
    first = estimator.step(_available(0))
    second = estimator.step(_available(1, mask=0.025, depth=0.6))
    assert first.confidence == 1.0
    assert second.support_quality is not None
    assert 0.0 < second.support_quality < 1.0
    assert 0.0 < second.confidence < 1.0


def test_missing_confidence_decays_and_uncertainty_grows() -> None:
    config = AdaptiveConfidenceConfig(max_predict_gap_frames=2)
    observations = [_available(index) for index in range(8)]
    observations.extend([_missing(8), _missing(9), _missing(10)])
    outputs = run_object_local_adaptive_confidence(observations, config)
    missing = outputs[-3:]
    assert missing[0].reason == "adaptive_dropout_prediction"
    assert missing[1].reason == "adaptive_dropout_prediction"
    assert missing[2].reason == "adaptive_dropout_hold"
    assert missing[0].confidence > missing[1].confidence > missing[2].confidence
    assert missing[0].uncertainty < missing[1].uncertainty < missing[2].uncertainty


def test_hold_candidate_never_extrapolates_missing_input() -> None:
    config = AdaptiveConfidenceConfig(max_predict_gap_frames=0)
    outputs = run_object_local_adaptive_confidence(
        [_available(index) for index in range(8)] + [_missing(8)], config
    )
    assert outputs[-1].reason == "adaptive_dropout_hold"
    assert np.array_equal(outputs[-1].pose, outputs[-2].pose)


def test_runner_is_deterministic_and_finite() -> None:
    observations = [_available(index) for index in range(6)] + [_missing(6)]
    config = AdaptiveConfidenceConfig(max_predict_gap_frames=2)
    first = run_object_local_adaptive_confidence(observations, config)
    second = run_object_local_adaptive_confidence(observations, config)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.pose, right.pose)
        assert np.array_equal(left.uncertainty_diag, right.uncertainty_diag)
        assert np.isfinite(left.pose).all()
        assert np.isfinite(left.uncertainty_diag).all()


def test_best_window_maximizes_missing_and_breaks_ties_earliest() -> None:
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


def test_track_selection_is_stable_and_enforces_object_cap() -> None:
    candidates = [
        {"track_id": f"track-{index}", "object_id": 1 if index < 4 else index}
        for index in range(8)
    ]
    first = select_tracks(
        candidates,
        namespace="test:",
        track_cap=5,
        per_object_cap=2,
    )
    second = select_tracks(
        list(reversed(candidates)),
        namespace="test:",
        track_cap=5,
        per_object_cap=2,
    )
    assert first == second
    assert sum(int(row["object_id"]) == 1 for row in first) <= 2
    assert len(first) == 5


def test_hb_visible_mask_count_allows_only_observed_one_pixel_archive_delta() -> None:
    path = Path("mask.png")
    validate_visible_mask_count(100, 100, path)
    validate_visible_mask_count(99, 100, path)
    validate_visible_mask_count(101, 100, path)
    with pytest.raises(ValueError, match="beyond one-pixel archive tolerance"):
        validate_visible_mask_count(102, 100, path)


def test_dropout_difficulty_strata_are_exclusive_and_cover_eligible_tracks() -> None:
    strata = {
        "sparse": {
            "natural_missing_frame_count_min": 2,
            "natural_missing_frame_count_max": 7,
        },
        "stress": {
            "natural_missing_frame_count_min": 8,
            "natural_missing_frame_count_max": None,
        },
    }
    assert difficulty_stratum(2, strata) == "sparse"
    assert difficulty_stratum(7, strata) == "sparse"
    assert difficulty_stratum(8, strata) == "stress"
    with pytest.raises(ValueError, match="exactly one difficulty stratum"):
        difficulty_stratum(1, strata)


def test_spearman_uses_average_tie_ranks() -> None:
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    tied = spearman_correlation([1, 1, 2, 3], [1, 2, 3, 4])
    assert 0.8 < tied < 1.0


def test_raw_foundationpose_score_is_not_an_estimator_input() -> None:
    fields = set(AdaptiveObservation.__dataclass_fields__)
    assert "foundationpose_top_score" not in fields
    assert "foundationpose_top_score_margin" not in fields
