from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_m3_r1 as m3_r1  # noqa: E402


def _row(
    object_id: int,
    track: str,
    outcomes: tuple[float, float, float],
    visibility: str = "mid",
) -> dict:
    return {
        "object_id": object_id,
        "physical_instance_track_id": track,
        "target_visibility_bin": visibility,
        "k1": {"marginal_value": outcomes[1] - outcomes[0]},
        "k3": {"marginal_value": outcomes[2] - outcomes[1]},
        "outcomes": dict(zip((1, 3, 5), outcomes, strict=True)),
    }


def test_apply_thresholds_is_sequential() -> None:
    p1 = np.asarray([-0.1, 0.2, 0.2])
    p3 = np.asarray([0.9, -0.1, 0.4])
    budgets = m3_r1.apply_thresholds(p1, p3, 0.0, 0.0)
    assert budgets.tolist() == [1, 3, 5]


def test_threshold_candidates_include_finite_all_action_sentinels() -> None:
    predictions = np.asarray([-0.25, 0.0, 0.75])
    candidates = m3_r1.threshold_candidates(predictions)
    assert np.all(np.isfinite(candidates))
    assert candidates[0] < np.min(predictions)
    assert candidates[-1] > np.max(predictions)


def test_exact_budget_random_expectation_matches_enumeration() -> None:
    rows = [
        _row(1, "a", (0.0, 0.5, 1.0)),
        _row(1, "b", (1.0, 0.5, 0.0)),
        _row(2, "c", (0.2, 0.4, 0.8)),
    ]
    budgets = np.asarray([1, 3, 5])
    expected = m3_r1.expected_matched_random_score(rows, budgets)
    enumerated = []
    import itertools

    for assignment in itertools.permutations(budgets):
        enumerated.append(m3_r1.policy_score(rows, np.asarray(assignment)))
    assert np.isclose(expected, np.mean(enumerated))


def test_allocation_gain_rewards_correct_ranking() -> None:
    rows = [
        _row(1, "a", (0.0, 0.0, 1.0)),
        _row(1, "b", (1.0, 1.0, 0.0)),
    ]
    good = np.asarray([5, 1])
    bad = np.asarray([1, 5])
    assert m3_r1.allocation_gain(rows, good) > 0.0
    assert m3_r1.allocation_gain(rows, bad) < 0.0


def test_feature_contract_excludes_evaluation_fields() -> None:
    audit = m3_r1.feature_audit()
    assert audit["passed"] is True
    names = audit["k1_feature_names"] + audit["k3_feature_names"]
    assert not any("visibility" in name for name in names)
    assert not any("object_id" in name for name in names)
    assert not any("future" in name for name in names)


def test_grouped_bootstrap_never_splits_a_track() -> None:
    rows = [
        _row(1, "track-a", (0.0, 0.5, 1.0)),
        _row(1, "track-a", (0.0, 0.5, 1.0)),
        _row(2, "track-b", (1.0, 0.5, 0.0)),
        _row(2, "track-c", (1.0, 0.5, 0.0)),
    ]
    budgets = np.asarray([5, 5, 1, 1])
    result = m3_r1.grouped_bootstrap_gain(rows, budgets, 20)
    assert result["unit"] == "physical_instance_track_id"
    assert result["track_count"] == 3
