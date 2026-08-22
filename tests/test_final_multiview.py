from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_r1_final_multiview as final_multiview  # noqa: E402


def _m3(group_id: str, budget: int) -> dict:
    return {
        "group_id": group_id,
        "object_id": 1,
        "physical_instance_track_id": f"track-{group_id}",
        "target_visibility_bin": "mid",
        "selected_budgets": {"3.0": budget},
        "k1_predicted_marginal_value": 0.2,
        "k3_predicted_marginal_value": 0.1,
        "outcomes": {"1": 0.5, "3": 0.7, "5": 0.8},
    }


def _m4(group_id: str, continue_: bool, slot: int = 3) -> dict:
    return {
        "group_id": group_id,
        "m3_primary_continue": continue_,
        "selected_slot": slot,
        "predicted_differences_from_slot_2": {
            "1": -0.1,
            "2": 0.0,
            "3": 0.2,
            "4": 0.1,
        },
    }


def _outcome(group_id: str, slot: int, utility: float) -> dict:
    return {
        "candidate_pair_utility": utility,
        "sample_ar_mssd": utility,
        "sample_ar_mspd": utility,
        "joint_success": utility > 0.5,
        "outcome_evidence": {
            "selected_sample_id": f"sample-{group_id}-{slot}",
            "selected_acquisition_rank": slot,
            "normalized_mssd": 0.05,
            "mspd_px": 5.0,
            "finite_pose": True,
            "status": "success",
        },
    }


def test_final_stream_composes_stop_and_ranked_pair_without_reselection() -> None:
    m3 = {"stop": _m3("stop", 1), "go": _m3("go", 3)}
    m4 = {"stop": _m4("stop", False), "go": _m4("go", True)}
    outcomes = {
        (group_id, slot): _outcome(group_id, slot, 0.4 + 0.1 * slot)
        for group_id in ("stop", "go")
        for slot in (1, 2, 3, 4)
    }
    groups = {
        group_id: {"target_sample_id": f"target-{group_id}"}
        for group_id in ("stop", "go")
    }
    target_metrics = {
        group_id: {
            "selected_sample_id": f"target-{group_id}",
            "sample_ar_mssd": 0.5,
            "sample_ar_mspd": 0.5,
            "diagnostic_success": {"joint": False},
            "normalized_mssd": 0.2,
            "mspd_px": 20.0,
            "finite_pose": True,
            "status": "success",
        }
        for group_id in ("stop", "go")
    }
    rows = final_multiview.build_final_rows(
        m3, m4, outcomes, groups, target_metrics
    )
    indexed = {row["group_id"]: row for row in rows}
    assert indexed["stop"]["final"]["route"] == "m3_stop_target_only"
    assert indexed["stop"]["final"]["acquired_view_count"] == 1
    assert indexed["go"]["final"]["route"] == "m3_continue_m4_ranked_pair"
    assert indexed["go"]["final"]["m4_selected_slot"] == 3
    assert indexed["go"]["final"]["selected_sample_id"] == "sample-go-3"
