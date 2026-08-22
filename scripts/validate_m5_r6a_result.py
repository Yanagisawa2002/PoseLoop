#!/usr/bin/env python3
"""Independently validate the frozen M5-R6A development result artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_m5_r2_once as r2_evaluator
from evaluate_m5_r5_development import spearman_correlation
from m1_common import load_jsonl, sha256_file


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r6a"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "contract.json")
    parser.add_argument("--result", type=Path, default=root / "development_result.json")
    return parser.parse_args()


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"M5-R6A result mismatch for {name}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    result_path = args.result.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != "poseloop-m5-r6a-v1"
        or result.get("protocol_id") != contract["protocol_id"]
        or result.get("stage_id") != "M5-R6A"
        or result.get("status") != "PASS_M5_R6A_DEVELOPMENT"
        or not bool(result.get("development_gate", {}).get("passed"))
        or bool(result.get("sealed_archive_or_label_read"))
        or bool(result.get("raw_foundationpose_scores_used_for_filtering_or_confidence"))
    ):
        raise RuntimeError("Invalid M5-R6A result identity, status, or claim boundary")
    if not all(bool(value) for value in result["development_gate"]["conditions"].values()):
        raise RuntimeError("M5-R6A result contains a failed development condition")

    track_path = Path(str(result["track_results"]["path"]))
    if sha256_file(track_path) != str(result["track_results"]["sha256"]):
        raise RuntimeError("M5-R6A cross-fitted track-result hash mismatch")
    rows = load_jsonl(track_path)
    if len(rows) != 14 or {int(row["object_id"]) for row in rows} != set(range(1, 15)):
        raise RuntimeError("M5-R6A cross-fitted object/track coverage differs")
    if any(
        row.get("candidate_id") != "measurement_first"
        or not str(row.get("track_id", "")).startswith("ruapc-test-")
        for row in rows
    ):
        raise RuntimeError("M5-R6A cross-fitted candidate or dataset identity differs")

    baseline_macro = r2_evaluator.macro_object_mean(rows, "baseline_loss")
    proposed_macro = r2_evaluator.macro_object_mean(rows, "proposed_loss")
    improvement = (baseline_macro - proposed_macro) / baseline_macro
    observed = result["development_gate"]["observed"]
    _assert_close(
        improvement,
        observed["cross_fitted_macro_object_relative_improvement"],
        "cross-fitted all-frame improvement",
    )
    missing_rows = [row for row in rows if int(row["natural_missing_frame_count"]) > 0]
    missing_baseline = r2_evaluator.macro_object_mean(
        missing_rows, "baseline_natural_missing_loss"
    )
    missing_proposed = r2_evaluator.macro_object_mean(
        missing_rows, "proposed_natural_missing_loss"
    )
    missing_improvement = (missing_baseline - missing_proposed) / missing_baseline
    _assert_close(
        missing_improvement,
        observed["cross_fitted_natural_missing_frame_relative_improvement"],
        "cross-fitted missing improvement",
    )

    missing_uncertainty: list[float] = []
    missing_error: list[float] = []
    missing_anchor_mismatches = 0
    nonfinite = 0
    correction_count = 0
    for row in rows:
        correction_count += int(row["correction_applied_frame_count"])
        nonfinite += int(row["nonfinite_output_count"])
        for frame in row["frames"]:
            if not bool(frame["natural_input_missing"]):
                continue
            missing_uncertainty.append(float(frame["proposed_uncertainty"]))
            missing_error.append(float(frame["proposed_normalized_pose_error"]))
            baseline_pose = np.asarray(frame["baseline_output_pose_m"], dtype=np.float64)
            proposed_pose = np.asarray(frame["proposed_output_pose_m"], dtype=np.float64)
            missing_anchor_mismatches += int(not np.array_equal(baseline_pose, proposed_pose))
    if len(missing_error) != 481:
        raise RuntimeError("M5-R6A missing-frame evidence count differs")
    confidence_spearman = spearman_correlation(missing_uncertainty, missing_error)
    _assert_close(
        confidence_spearman,
        observed["missing_frame_uncertainty_error_spearman"],
        "missing uncertainty/error Spearman",
    )
    if missing_anchor_mismatches != 0 or nonfinite != 0:
        raise RuntimeError("M5-R6A structural output validation failed")

    bootstrap_spec = json.loads(
        Path(str(contract["protocol"]["path"])).read_text(encoding="utf-8")
    )["evaluation"]["bootstrap"]
    bootstrap = r2_evaluator.hierarchical_paired_bootstrap(
        rows,
        seed=int(bootstrap_spec["seed"]),
        resamples=int(bootstrap_spec["resamples"]),
    )
    stored_bootstrap = result["hierarchical_paired_bootstrap"]
    for name in (
        "mean_relative_improvement",
        "one_sided_90pct_lower_relative_improvement",
        "two_sided_90pct_upper_relative_improvement",
    ):
        _assert_close(bootstrap[name], stored_bootstrap[name], f"bootstrap {name}")

    folds = result["leave_one_object_out"]
    if (
        len(folds) != 14
        or result.get("fold_winner_counts") != {"measurement_first": 14}
        or result.get("modal_candidate_id") != "measurement_first"
        or any(fold["selected_candidate_id"] != "measurement_first" for fold in folds)
    ):
        raise RuntimeError("M5-R6A fold winner selection differs")
    per_object = sorted(result["per_object"], key=lambda row: int(row["object_id"]))
    improvements = np.asarray(
        [float(row["relative_improvement"]) for row in per_object], dtype=np.float64
    )
    if len(per_object) != 14 or not np.all(improvements > 0):
        raise RuntimeError("M5-R6A does not improve every held-out object")
    fold_by_object = {int(row["held_out_object_id"]): row for row in folds}
    for row in per_object:
        object_id = int(row["object_id"])
        _assert_close(
            row["relative_improvement"],
            fold_by_object[object_id]["held_out_relative_improvement"],
            f"held-out object {object_id}",
        )

    thresholds = result["development_gate"]["thresholds"]
    if (
        improvement < float(thresholds["cross_fitted_macro_object_relative_improvement_min"])
        or missing_improvement
        < float(thresholds["cross_fitted_natural_missing_frame_relative_improvement_min"])
        or float(bootstrap["one_sided_90pct_lower_relative_improvement"])
        < float(
            thresholds[
                "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"
            ]
        )
        or not math.isfinite(confidence_spearman)
        or confidence_spearman
        < float(thresholds["missing_frame_uncertainty_error_spearman_min"])
    ):
        raise RuntimeError("M5-R6A recomputed numerical gate failed")

    validation = {
        "status": "PASS_M5_R6A_RESULT_VALIDATION",
        "protocol_id": result["protocol_id"],
        "result_sha256": sha256_file(result_path),
        "track_results_sha256": sha256_file(track_path),
        "cross_fitted_macro_object_relative_improvement": improvement,
        "one_sided_90pct_lower_relative_improvement": bootstrap[
            "one_sided_90pct_lower_relative_improvement"
        ],
        "missing_frame_relative_improvement": missing_improvement,
        "missing_frame_uncertainty_error_spearman": confidence_spearman,
        "held_out_object_improvement_min": float(np.min(improvements)),
        "held_out_object_improvement_median": float(np.median(improvements)),
        "held_out_object_improvement_max": float(np.max(improvements)),
        "positive_held_out_object_count": int(np.sum(improvements > 0)),
        "modal_candidate_id": result["modal_candidate_id"],
        "correction_applied_frame_count": correction_count,
        "missing_anchor_mismatch_count": missing_anchor_mismatches,
        "nonfinite_output_count": nonfinite,
        "cosmetic_stale_stage_label_count": int("LM-O" in str(result.get("stage"))),
        "cosmetic_stale_stage_label_effect": "none; protocol, stage ID, track IDs, contract, and source hashes identify RU-APC",
        "sealed_archive_or_label_read": False,
    }
    print(json.dumps(validation, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
