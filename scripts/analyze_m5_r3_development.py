#!/usr/bin/env python3
"""Select and freeze M5-R3 using only the consumed M5-R2 development replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m5_g0_core as legacy
import m5_g0_metrics as metrics
import m5_r2_core as r2_core
from evaluate_m5_r2_once import hierarchical_paired_bootstrap, symmetry_from_model_info
from m1_common import load_jsonl, sha256_file, write_json_atomic
from m5_r3_core import StaticQuotientMedoidConfig, run_static_quotient_medoid


SCHEMA_VERSION = 1
CANDIDATE_HISTORY_LIMITS: tuple[int | None, ...] = (5, 10, 20, None)
DEVELOPMENT_SEED = 5303
BOOTSTRAP_RESAMPLES = 5000


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m5-r2-root", type=Path, default=repo_root / "artifacts" / "r2" / "m5_r2"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r3_development.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r3_development.md",
    )
    return parser.parse_args()


def history_id(limit: int | None) -> str:
    return "all_history" if limit is None else f"last_{limit}_measurements"


def macro_object_mean(
    rows: Sequence[Mapping[str, Any]], field: str, object_ids: set[int] | None = None
) -> float:
    by_object: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        object_id = int(row["object_id"])
        if object_ids is None or object_id in object_ids:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite development metric: {field}")
            by_object[object_id].append(value)
    if not by_object:
        raise ValueError("Cannot aggregate an empty development result")
    return float(np.mean([np.mean(by_object[key]) for key in sorted(by_object)]))


def build_development_sequence(
    row: Mapping[str, Any], models_info: Mapping[str, Any]
) -> legacy.CorruptedSequence:
    symmetry_original = symmetry_from_model_info(models_info[str(row["object_id"])])
    canonical_to_original, symmetry = r2_core.canonicalization_for_symmetry(
        symmetry_original
    )
    inputs: list[legacy.EstimatorInput] = []
    evaluator: list[legacy.EvaluatorFrame] = []
    gt_poses: list[np.ndarray] = []
    timestamps: list[float] = []
    for frame in row["frames"]:
        measurement_raw = frame["measurement_model_to_world_pose_m"]
        measurement = (
            None
            if measurement_raw is None
            else r2_core.to_canonical_pose(
                np.asarray(measurement_raw, dtype=np.float64), canonical_to_original
            )
        )
        gt_pose = r2_core.to_canonical_pose(
            np.asarray(frame["gt_model_to_world_pose_m"], dtype=np.float64),
            canonical_to_original,
        )
        timestamp = float(frame["timestamp_s"])
        inputs.append(
            legacy.EstimatorInput(
                timestamp_s=timestamp,
                measurement_pose=None if measurement is None else measurement.copy(),
                missing=measurement is None,
                symmetry=symmetry,
            )
        )
        evaluator.append(
            legacy.EvaluatorFrame(
                ground_truth_pose=gt_pose.copy(),
                noisy_canonical_measurement_pose=(
                    gt_pose.copy() if measurement is None else measurement.copy()
                ),
                representative_index=None,
                axial_gauge_rad=None,
                dropout=measurement is None,
                outlier=False,
            )
        )
        gt_poses.append(gt_pose)
        timestamps.append(timestamp)
    return legacy.CorruptedSequence(
        inputs=tuple(inputs),
        evaluator_frames=tuple(evaluator),
        ground_truth=legacy.GroundTruthSequence(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            poses=np.stack(gt_poses),
            symmetry=symmetry,
            motion_family=legacy.MotionFamily.STATIC,
            seed=0,
            reversal_frame=None,
            translation_direction_camera=np.zeros(3, dtype=np.float64),
            observable_rotation_axis_object=np.asarray(symmetry.axis_object, dtype=np.float64),
        ),
        stress_family=legacy.StressFamily.NOMINAL,
        corruption_seed=0,
        dropout_intervals=(),
        outlier_intervals=(),
    )


def evaluate_candidate(
    row: Mapping[str, Any], sequence: legacy.CorruptedSequence, limit: int | None
) -> dict[str, Any]:
    outputs = run_static_quotient_medoid(
        sequence, StaticQuotientMedoidConfig(history_limit=limit)
    )
    poses = np.stack([output.pose for output in outputs])
    frame_metrics = metrics.compute_per_frame_pose_errors(
        sequence.ground_truth.poses, poses, sequence.ground_truth.symmetry
    )
    loss = np.asarray(frame_metrics["normalized_pose_error"], dtype=np.float64)
    missing = np.asarray(
        [bool(frame["natural_input_missing"]) for frame in row["frames"]], dtype=bool
    )
    return {
        "record_type": "m5_r3_development_track",
        "track_id": str(row["track_id"]),
        "object_id": int(row["object_id"]),
        "baseline_loss": float(row["baseline_loss"]),
        "baseline_natural_missing_loss": float(row["baseline_natural_missing_loss"]),
        "proposed_loss": float(np.mean(loss)),
        "proposed_natural_missing_loss": float(np.mean(loss[missing])),
        "nonfinite_output_count": int(np.sum(~np.isfinite(poses).all(axis=(1, 2)))),
    }


def relative_improvement(
    rows: Sequence[Mapping[str, Any]], object_ids: set[int] | None = None
) -> tuple[float, float, float, float]:
    baseline = macro_object_mean(rows, "baseline_loss", object_ids)
    proposed = macro_object_mean(rows, "proposed_loss", object_ids)
    baseline_missing = macro_object_mean(
        rows, "baseline_natural_missing_loss", object_ids
    )
    proposed_missing = macro_object_mean(
        rows, "proposed_natural_missing_loss", object_ids
    )
    return (
        baseline,
        proposed,
        (baseline - proposed) / baseline,
        (baseline_missing - proposed_missing) / baseline_missing,
    )


def render_report(result: Mapping[str, Any]) -> str:
    selected = result["selected_candidate"]
    lines = [
        "# PoseLoop M5-R3 development and candidate freeze",
        "",
        f"**{result['status']}**",
        "",
        "Only the already-consumed M5-R2 RealSense replay was used here. No M5-R3 "
        "Photoneo prediction, evaluator label, pose error, or temporal outcome was read.",
        "",
        "| Candidate | All-frame improvement | Missing-frame improvement |",
        "| --- | ---: | ---: |",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['candidate_id']} | {row['macro_object_relative_improvement']:+.2%} | "
            f"{row['natural_missing_macro_object_relative_improvement']:+.2%} |"
        )
    lines.extend(
        [
            "",
            "Leave-one-object-out selection chose the same candidate in every fold:",
            "",
            "| Held-out object | Selected on other four | Held-out all | Held-out missing |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for fold in result["leave_one_object_out"]:
        lines.append(
            f"| {fold['held_out_object_id']:02d} | {fold['selected_candidate_id']} | "
            f"{fold['held_out_relative_improvement']:+.2%} | "
            f"{fold['held_out_missing_relative_improvement']:+.2%} |"
        )
    lines.extend(
        [
            "",
            f"Frozen candidate: **{selected['candidate_id']}**, all frames "
            f"{selected['cross_fitted_relative_improvement']:+.2%}, natural missing "
            f"{selected['cross_fitted_missing_relative_improvement']:+.2%}.",
            "",
            f"The development hierarchical-bootstrap 10th percentile is "
            f"{selected['one_sided_90pct_lower_relative_improvement']:+.2%}. It is a "
            "recorded uncertainty warning, not positive evidence; the candidate must pass "
            "all gates on the untouched Photoneo split before M5-R3 can be called successful.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.m5_r2_root.resolve()
    result_path = (root / "result.json").resolve()
    track_results_path = (root / "track_results.jsonl").resolve()
    contract_path = (root / "contract.json").resolve()
    for path in (result_path, track_results_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    predecessor = json.loads(result_path.read_text(encoding="utf-8"))
    if predecessor.get("status") != "FAIL_M5_R2_REAL_REPLAY" or not predecessor.get(
        "labels_opened"
    ):
        raise RuntimeError("M5-R2 is not the consumed negative development result")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    models_info_path = Path(str(contract["inputs"]["models_eval_info"]["path"]))
    if sha256_file(models_info_path) != contract["inputs"]["models_eval_info"]["sha256"]:
        raise RuntimeError("Symmetry metadata changed after M5-R2")
    models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
    source_rows = load_jsonl(track_results_path)
    sequences = {
        str(row["track_id"]): build_development_sequence(row, models_info)
        for row in source_rows
    }
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    for limit in CANDIDATE_HISTORY_LIMITS:
        candidate_id = history_id(limit)
        rows = [
            evaluate_candidate(row, sequences[str(row["track_id"])], limit)
            for row in source_rows
        ]
        by_candidate[candidate_id] = rows
        baseline, proposed, relative, missing_relative = relative_improvement(rows)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "history_limit": limit,
                "baseline_macro_object_loss": baseline,
                "proposed_macro_object_loss": proposed,
                "macro_object_relative_improvement": relative,
                "natural_missing_macro_object_relative_improvement": missing_relative,
                "nonfinite_output_count": sum(
                    int(row["nonfinite_output_count"]) for row in rows
                ),
            }
        )
    object_ids = sorted({int(row["object_id"]) for row in source_rows})
    cross_fitted_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for held_out in object_ids:
        training_ids = set(object_ids) - {held_out}
        training_scores = []
        for order, candidate in enumerate(candidates):
            candidate_id = str(candidate["candidate_id"])
            _, _, score, _ = relative_improvement(
                by_candidate[candidate_id], training_ids
            )
            training_scores.append((score, -order, candidate_id))
        _, _, selected_id = max(training_scores)
        held_rows = [
            row
            for row in by_candidate[selected_id]
            if int(row["object_id"]) == held_out
        ]
        cross_fitted_rows.extend(held_rows)
        _, _, held_relative, held_missing = relative_improvement(held_rows)
        folds.append(
            {
                "held_out_object_id": held_out,
                "selected_candidate_id": selected_id,
                "training_relative_improvement_by_candidate": {
                    candidate_id: score for score, _, candidate_id in training_scores
                },
                "held_out_relative_improvement": held_relative,
                "held_out_missing_relative_improvement": held_missing,
            }
        )
    selected_ids = {str(row["selected_candidate_id"]) for row in folds}
    if len(selected_ids) != 1:
        raise RuntimeError("M5-R3 leave-one-object-out selection is not stable")
    selected_id = next(iter(selected_ids))
    baseline, proposed, relative, missing_relative = relative_improvement(
        cross_fitted_rows
    )
    bootstrap = hierarchical_paired_bootstrap(
        cross_fitted_rows, seed=DEVELOPMENT_SEED, resamples=BOOTSTRAP_RESAMPLES
    )
    selected_candidate = next(
        row for row in candidates if row["candidate_id"] == selected_id
    )
    gate = {
        "same_candidate_selected_in_every_leave_one_object_out_fold": True,
        "cross_fitted_macro_object_relative_improvement_at_least_5pct": relative >= 0.05,
        "cross_fitted_natural_missing_relative_improvement_nonnegative": (
            missing_relative >= 0.0
        ),
        "nonfinite_output_count_zero": int(selected_candidate["nonfinite_output_count"])
        == 0,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R3 consumed RealSense development and candidate freeze",
        "status": (
            "FROZEN_M5_R3_CANDIDATE" if all(gate.values()) else "REJECTED_M5_R3_CANDIDATE"
        ),
        "development_data_status": "consumed_M5_R2_labels_opened",
        "photoneo_r3_prediction_or_evaluator_outcome_read": False,
        "object_count": len(object_ids),
        "track_count": len(source_rows),
        "candidate_selection_gate": gate,
        "candidates": candidates,
        "leave_one_object_out": folds,
        "selected_candidate": {
            "candidate_id": selected_id,
            "history_limit": selected_candidate["history_limit"],
            "translation_scale_m": 0.01,
            "rotation_scale_deg": 5.0,
            "cross_fitted_baseline_macro_object_loss": baseline,
            "cross_fitted_proposed_macro_object_loss": proposed,
            "cross_fitted_relative_improvement": relative,
            "cross_fitted_missing_relative_improvement": missing_relative,
            "one_sided_90pct_lower_relative_improvement": bootstrap[
                "one_sided_90pct_lower_relative_improvement"
            ],
            "bootstrap": bootstrap,
        },
        "provenance": {
            "m5_r2_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
            "m5_r2_track_results": {
                "path": str(track_results_path),
                "sha256": sha256_file(track_results_path),
            },
            "models_eval_info": {
                "path": str(models_info_path),
                "sha256": sha256_file(models_info_path),
            },
            "analysis_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_json.resolve(), result)
    args.output_report.resolve().write_text(render_report(result), encoding="utf-8")
    print(result["status"])
    print(
        f"selected={selected_id}; all={relative:+.2%}; missing={missing_relative:+.2%}; "
        f"bootstrap_lower={bootstrap['one_sided_90pct_lower_relative_improvement']:+.2%}"
    )
    print(args.output_json.resolve())
    print(args.output_report.resolve())


if __name__ == "__main__":
    main()
