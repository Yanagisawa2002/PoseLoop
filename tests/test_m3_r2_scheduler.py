from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_m3_r2_scheduler as r2  # noqa: E402


def _rows() -> list[dict[str, float | str]]:
    return [
        {"group_id": "a", "p1": 0.9, "p3": 0.1},
        {"group_id": "b", "p1": 0.8, "p3": 0.9},
        {"group_id": "c", "p1": 0.2, "p3": 5.0},
        {"group_id": "d", "p1": -0.1, "p3": 9.0},
    ]


def test_scheduler_is_sequential_nested_and_budget_bounded() -> None:
    budgets, audit = r2.schedule_budgets(
        _rows(),
        {"numerator": 2, "denominator": 4},
        {"numerator": 1, "denominator": 4},
        k1_field="p1",
        k3_field="p3",
        score_floor=0.0,
    )
    assert budgets.tolist() == [3, 5, 1, 1]
    assert audit["q1_selected"] == 2
    assert audit["q2_selected"] == 1
    assert set(audit["stage2_group_ids"]) <= set(audit["stage1_group_ids"])
    assert float(np.mean(budgets)) <= 3.0


def test_scheduler_never_fills_quota_with_nonpositive_scores() -> None:
    rows = [
        {"group_id": "a", "p1": -1.0, "p3": 2.0},
        {"group_id": "b", "p1": 0.0, "p3": 2.0},
    ]
    budgets, audit = r2.schedule_budgets(
        rows,
        {"numerator": 1, "denominator": 1},
        {"numerator": 1, "denominator": 1},
        k1_field="p1",
        k3_field="p3",
        score_floor=0.0,
    )
    assert budgets.tolist() == [1, 1]
    assert audit["q1_selected"] == 0
    assert audit["q2_selected"] == 0


def test_round_half_up_quota_matches_frozen_sealed_contract() -> None:
    assert r2.rounded_quota(150, {"numerator": 144, "denominator": 300}) == 72
    assert r2.rounded_quota(150, {"numerator": 33, "denominator": 300}) == 17


def test_development_json_budget_keys_are_normalized() -> None:
    rows = r2.normalize_development_rows(
        [{"group_id": "g", "outcomes": {"1": 0.1, "3": 0.2, "5": 0.3}}]
    )
    assert rows[0]["outcomes"] == {1: 0.1, 3: 0.2, 5: 0.3}


def test_positive_ranking_is_deterministic_and_drops_nonpositive() -> None:
    rows = [
        {"group_id": "b", "p": 0.2},
        {"group_id": "a", "p": -1.0},
        {"group_id": "c", "p": 0.5},
    ]
    assert r2.rank_positive_indices(rows, "p", 0.0) == [2, 0]
