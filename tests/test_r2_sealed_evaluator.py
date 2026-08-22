from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TOOLKIT = REPO_ROOT / "third_party" / "bop_toolkit"
for path in (SCRIPTS, TOOLKIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_r2_sealed_once import (  # noqa: E402
    macro_object,
    render_report,
    transform_audit_limits,
)


def test_transform_audit_limit_is_resolution_normalized() -> None:
    contract = {
        "evaluator_transform_audit": {
            "photoneo_image_width": 2064,
            "photoneo_mspd_px_max": 0.016125,
            "normalized_mssd_max": 0.0005,
            "use_for_pose_success_thresholds": False,
        }
    }
    assert transform_audit_limits(contract) == (0.0005, 0.016125)


def test_r2_macro_object_weights_objects_equally() -> None:
    rows = [
        {"object_id": 1, "value": 0.0},
        {"object_id": 1, "value": 1.0},
        {"object_id": 2, "value": 1.0},
    ]
    assert macro_object(rows, "value") == 0.75


def test_r2_report_uses_prepare_groups_transform_audit_shape() -> None:
    result = {
        "status": "PASS_R2_SEALED_VALIDATION",
        "target_count": 150,
        "final_mean_acquired_views": 1.48,
        "gt_transform_audit": {
            "normalized_mssd": {"max": 0.0001},
            "mspd_px": {"max": 0.01234},
        },
        "gt_transform_audit_limits": {
            "normalized_mssd_max": 0.0005,
            "mspd_px_max": 0.016125,
        },
        "stages": {
            "M3-R2": {
                "passed": True,
                "active_macro_combined": 0.8,
                "matched_random_macro_combined": 0.7,
                "gain_over_matched_random_pp": 10.0,
                "mean_views": 2.0,
            },
            "M4-R2": {
                "passed": True,
                "continue_target_count": 72,
                "active_macro_combined": 0.8,
                "fixed_slot_2_macro_combined": 0.7,
                "uniform_random_macro_combined": 0.75,
                "gain_over_fixed_slot_2_pp": 10.0,
                "gain_over_uniform_random_pp": 5.0,
            },
            "M6-R2": {
                "passed": True,
                "failure_count": 20,
                "learned": {"auroc": 0.8, "aurc": 0.2},
                "raw_score_baseline": {"auroc": 0.5, "aurc": 0.4},
                "auroc_gain_over_raw": 0.3,
                "aurc_relative_reduction_vs_raw": 0.5,
            },
        },
    }
    report = render_report(result)
    assert "0.0001 / 0.0005" in report
    assert "0.01234 / 0.01613 px" in report
