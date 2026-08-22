#!/usr/bin/env python3
"""Run M5-R1 phase 1: paired synthetic quotient filtering vs nearest."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m5_g0_core as legacy
import m5_g0_metrics as metrics
from m1_common import sha256_file, write_json_atomic, write_jsonl_atomic
from m5_r1_core import (
    QuotientCVKalmanConfig,
    output_arrays as kalman_output_arrays,
    run_quotient_cv_kalman,
)


SCHEMA_VERSION = 1
METHOD_NEAREST = "NEAREST_REPRESENTATIVE_CT"
METHOD_PROPOSED = "QUOTIENT_CV_KALMAN"
METHODS = (METHOD_NEAREST, METHOD_PROPOSED)
OUTER_FOLDS = 4
REPLICATES = 8
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20261201


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "m5_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r1_protocol.json"
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "m5_r1_phase1_development.md",
    )
    return parser.parse_args()


def nearest_grid() -> dict[str, legacy.EstimatorConfig]:
    common = {
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "reacquire_after": 4,
    }
    return {
        "nearest_smooth": legacy.EstimatorConfig(
            process_translation_sigma_m=0.001,
            process_rotation_sigma_deg=0.5,
            velocity_process_sigma=0.02,
            innovation_gate_sigma=4.5,
            velocity_update_gain=0.20,
            **common,
        ),
        "nearest_balanced": legacy.EstimatorConfig(
            process_translation_sigma_m=0.002,
            process_rotation_sigma_deg=1.0,
            velocity_process_sigma=0.04,
            innovation_gate_sigma=5.5,
            velocity_update_gain=0.35,
            **common,
        ),
        "nearest_agile": legacy.EstimatorConfig(
            process_translation_sigma_m=0.003,
            process_rotation_sigma_deg=1.5,
            velocity_process_sigma=0.08,
            innovation_gate_sigma=5.5,
            velocity_update_gain=0.55,
            **common,
        ),
        "nearest_fast": legacy.EstimatorConfig(
            process_translation_sigma_m=0.004,
            process_rotation_sigma_deg=2.0,
            velocity_process_sigma=0.12,
            innovation_gate_sigma=6.0,
            velocity_update_gain=0.70,
            **common,
        ),
    }


def proposed_grid() -> dict[str, QuotientCVKalmanConfig]:
    common = {
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "reacquire_after": 4,
    }
    return {
        "qcvk_smooth": QuotientCVKalmanConfig(
            acceleration_translation_sigma_m_s2=0.002,
            acceleration_rotation_sigma_deg_s2=1.5,
            initial_velocity_translation_sigma_m_s=0.02,
            initial_velocity_rotation_sigma_deg_s=15.0,
            innovation_gate_sigma=4.5,
            **common,
        ),
        "qcvk_balanced": QuotientCVKalmanConfig(
            acceleration_translation_sigma_m_s2=0.005,
            acceleration_rotation_sigma_deg_s2=3.0,
            initial_velocity_translation_sigma_m_s=0.02,
            initial_velocity_rotation_sigma_deg_s=20.0,
            innovation_gate_sigma=4.5,
            **common,
        ),
        "qcvk_dynamic": QuotientCVKalmanConfig(
            acceleration_translation_sigma_m_s2=0.010,
            acceleration_rotation_sigma_deg_s2=5.0,
            initial_velocity_translation_sigma_m_s=0.05,
            initial_velocity_rotation_sigma_deg_s=25.0,
            innovation_gate_sigma=5.0,
            **common,
        ),
        "qcvk_wide_gate": QuotientCVKalmanConfig(
            acceleration_translation_sigma_m_s2=0.005,
            acceleration_rotation_sigma_deg_s2=3.0,
            initial_velocity_translation_sigma_m_s=0.05,
            initial_velocity_rotation_sigma_deg_s=20.0,
            innovation_gate_sigma=5.5,
            **common,
        ),
    }


def configuration_grid() -> dict[str, dict[str, Any]]:
    return {METHOD_NEAREST: nearest_grid(), METHOD_PROPOSED: proposed_grid()}


def synthetic_manifest() -> list[dict[str, Any]]:
    rows = []
    for symmetry_index, symmetry in enumerate(legacy.SymmetryClass):
        for motion_index, motion in enumerate(legacy.MotionFamily):
            for stress_index, stress in enumerate(legacy.StressFamily):
                for replicate in range(REPLICATES):
                    trajectory_seed = (
                        510_000_000
                        + symmetry_index * 1_000_000
                        + motion_index * 10_000
                        + stress_index * 100
                        + replicate
                    )
                    corruption_seed = (
                        610_000_000
                        + symmetry_index * 1_000_000
                        + motion_index * 10_000
                        + stress_index * 100
                        + replicate
                    )
                    rows.append(
                        {
                            "trajectory_id": (
                                f"r1-{symmetry.value.lower()}-{motion.value.lower()}-"
                                f"{stress.value.lower()}-r{replicate:02d}"
                            ),
                            "symmetry_class": symmetry.value,
                            "motion_family": motion.value,
                            "stress_family": stress.value,
                            "replicate": replicate,
                            "outer_fold": replicate % OUTER_FOLDS,
                            "trajectory_seed": trajectory_seed,
                            "corruption_seed": corruption_seed,
                        }
                    )
    expected = (
        len(legacy.SymmetryClass)
        * len(legacy.MotionFamily)
        * len(legacy.StressFamily)
        * REPLICATES
    )
    if len(rows) != expected or len({row["trajectory_id"] for row in rows}) != expected:
        raise ValueError("M5-R1 synthetic manifest identity failure")
    return rows


def _evaluate_outputs(
    truth: legacy.GroundTruthSequence,
    sequence: legacy.CorruptedSequence,
    outputs: Sequence[legacy.EstimatorOutput],
) -> dict[str, Any]:
    poses = np.stack([output.pose for output in outputs])
    pose_metrics = metrics.compute_per_frame_pose_errors(
        truth.poses, poses, truth.symmetry
    )
    normalized = np.asarray(pose_metrics["normalized_pose_error"], dtype=float)
    dropout_mask = np.asarray(
        [frame.dropout for frame in sequence.evaluator_frames], dtype=bool
    )
    outlier_mask = np.asarray(
        [frame.outlier for frame in sequence.evaluator_frames], dtype=bool
    )
    runtime = np.asarray([output.runtime_s for output in outputs], dtype=float)
    return {
        "primary_trajectory_loss": float(np.mean(normalized)),
        "median_normalized_pose_error": float(np.median(normalized)),
        "p95_normalized_pose_error": float(np.quantile(normalized, 0.95)),
        "translation_rmse_mm": 1000.0
        * float(pose_metrics["translation"]["rmse"]),
        "rotation_rmse_deg": float(pose_metrics["rotation"]["rmse"]),
        "dropout_mean_loss": (
            float(np.mean(normalized[dropout_mask])) if np.any(dropout_mask) else None
        ),
        "outlier_mean_loss": (
            float(np.mean(normalized[outlier_mask])) if np.any(outlier_mask) else None
        ),
        "accepted_frame_count": int(sum(output.accepted for output in outputs)),
        "rejected_frame_count": int(sum(not output.accepted for output in outputs)),
        "nonfinite_output_count": int(np.sum(~np.isfinite(poses).all(axis=(1, 2)))),
        "p95_runtime_ms": 1000.0 * float(np.quantile(runtime, 0.95)),
    }


def evaluate_grid(
    manifest: Sequence[Mapping[str, Any]],
    grids: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_index, item in enumerate(manifest):
        truth = legacy.generate_ground_truth(
            item["symmetry_class"],
            item["motion_family"],
            int(item["trajectory_seed"]),
            frame_count=120,
            control_hz=20.0,
        )
        sequence = legacy.corrupt_measurements(
            truth, item["stress_family"], int(item["corruption_seed"])
        )
        for method in METHODS:
            for config_id, config in grids[method].items():
                try:
                    if method == METHOD_NEAREST:
                        outputs = legacy.run_estimator(
                            sequence, legacy.EstimatorKind.NEAREST_REPRESENTATIVE_CT, config
                        )
                    else:
                        outputs = run_quotient_cv_kalman(sequence, config)
                    result = _evaluate_outputs(truth, sequence, outputs)
                    status = "success"
                    failure = None
                except Exception as error:  # pragma: no cover - retained evidence path
                    result = {
                        "primary_trajectory_loss": 100.0,
                        "median_normalized_pose_error": None,
                        "p95_normalized_pose_error": None,
                        "translation_rmse_mm": None,
                        "rotation_rmse_deg": None,
                        "dropout_mean_loss": None,
                        "outlier_mean_loss": None,
                        "accepted_frame_count": 0,
                        "rejected_frame_count": 120,
                        "nonfinite_output_count": 120,
                        "p95_runtime_ms": None,
                    }
                    status = "failure"
                    failure = {"type": type(error).__name__, "message": str(error)}
                rows.append(
                    {
                        "record_type": "m5_r1_synthetic_grid_result",
                        "schema_version": SCHEMA_VERSION,
                        **dict(item),
                        "method": method,
                        "config_id": config_id,
                        "status": status,
                        "failure": failure,
                        "metrics": result,
                    }
                )
        if (manifest_index + 1) % 96 == 0:
            print(f"M5-R1 phase1 progress: {manifest_index + 1}/{len(manifest)} trajectories", flush=True)
    return rows


def cross_fitted_comparison(
    grid_rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (str(row["trajectory_id"]), str(row["method"]), str(row["config_id"])): row
        for row in grid_rows
    }
    selections = []
    oof_rows = []
    for fold in range(OUTER_FOLDS):
        train_ids = {
            str(row["trajectory_id"]) for row in manifest if int(row["outer_fold"]) != fold
        }
        test_rows = [row for row in manifest if int(row["outer_fold"]) == fold]
        selected: dict[str, str] = {}
        for method in METHODS:
            config_ids = sorted(
                {str(row["config_id"]) for row in grid_rows if row["method"] == method}
            )
            scores = {}
            for config_id in config_ids:
                losses = [
                    float(by_key[(trajectory_id, method, config_id)]["metrics"]["primary_trajectory_loss"])
                    for trajectory_id in train_ids
                ]
                scores[config_id] = float(np.mean(losses))
            selected[method] = min(scores, key=lambda key: (scores[key], key))
            selections.append(
                {
                    "outer_fold": fold,
                    "method": method,
                    "selected_config_id": selected[method],
                    "train_trajectory_count": len(train_ids),
                    "test_trajectory_count": len(test_rows),
                    "mean_training_loss_by_config": scores,
                    "outer_test_used_for_selection": False,
                }
            )
        for item in test_rows:
            trajectory_id = str(item["trajectory_id"])
            nearest = by_key[(trajectory_id, METHOD_NEAREST, selected[METHOD_NEAREST])]
            proposed = by_key[(trajectory_id, METHOD_PROPOSED, selected[METHOD_PROPOSED])]
            nearest_loss = float(nearest["metrics"]["primary_trajectory_loss"])
            proposed_loss = float(proposed["metrics"]["primary_trajectory_loss"])
            oof_rows.append(
                {
                    "record_type": "m5_r1_cross_fitted_pair",
                    "schema_version": SCHEMA_VERSION,
                    **dict(item),
                    "nearest_config_id": selected[METHOD_NEAREST],
                    "proposed_config_id": selected[METHOD_PROPOSED],
                    "nearest_loss": nearest_loss,
                    "proposed_loss": proposed_loss,
                    "absolute_improvement": nearest_loss - proposed_loss,
                    "relative_improvement": (
                        (nearest_loss - proposed_loss) / nearest_loss
                        if nearest_loss > 0
                        else None
                    ),
                    "nearest_metrics": nearest["metrics"],
                    "proposed_metrics": proposed["metrics"],
                }
            )
    if len(oof_rows) != len(manifest):
        raise ValueError("M5-R1 cross-fitted comparison lacks exact coverage")
    return sorted(oof_rows, key=lambda row: row["trajectory_id"]), selections


def paired_bootstrap(oof_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nearest = np.asarray([float(row["nearest_loss"]) for row in oof_rows])
    proposed = np.asarray([float(row["proposed_loss"]) for row in oof_rows])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(oof_rows), size=len(oof_rows))
        baseline = float(np.mean(nearest[sampled]))
        values[index] = (baseline - float(np.mean(proposed[sampled]))) / baseline
    return {
        "unit": "complete paired synthetic trajectory",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "mean_relative_improvement": float(np.mean(values)),
        "one_sided_90pct_lower_relative_improvement": float(np.quantile(values, 0.10)),
        "two_sided_90pct_upper_relative_improvement": float(np.quantile(values, 0.95)),
    }


def gate_decision(
    nearest_mean: float,
    proposed_mean: float,
    bootstrap: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    relative = (nearest_mean - proposed_mean) / nearest_mean
    lower = float(bootstrap["one_sided_90pct_lower_relative_improvement"])
    conditions = {
        "relative_improvement": relative
        >= float(thresholds["relative_improvement_min"]),
        "one_sided_sequence_bootstrap_90pct_lower": lower
        >= float(thresholds["one_sided_sequence_bootstrap_90pct_lower_improvement_min"]),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "observed": {
            "nearest_mean_loss": nearest_mean,
            "proposed_mean_loss": proposed_mean,
            "relative_improvement": relative,
            "one_sided_90pct_lower_relative_improvement": lower,
        },
        "thresholds": dict(thresholds),
    }


def difficulty_tables(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    output = []
    for value in sorted({str(row[field]) for row in rows}):
        selected = [row for row in rows if str(row[field]) == value]
        nearest = float(np.mean([float(row["nearest_loss"]) for row in selected]))
        proposed = float(np.mean([float(row["proposed_loss"]) for row in selected]))
        output.append(
            {
                field: value,
                "trajectory_count": len(selected),
                "nearest_mean_loss": nearest,
                "proposed_mean_loss": proposed,
                "relative_improvement": (nearest - proposed) / nearest,
            }
        )
    return output


def select_full_configuration(
    rows: Sequence[Mapping[str, Any]], method: str
) -> dict[str, Any]:
    config_ids = sorted({str(row["config_id"]) for row in rows if row["method"] == method})
    means = {
        config_id: float(
            np.mean(
                [
                    float(row["metrics"]["primary_trajectory_loss"])
                    for row in rows
                    if row["method"] == method and row["config_id"] == config_id
                ]
            )
        )
        for config_id in config_ids
    }
    selected = min(means, key=lambda key: (means[key], key))
    return {"selected_config_id": selected, "mean_loss_by_config": means}


def render_report(result: Mapping[str, Any]) -> str:
    gate = result["development_gate"]
    lines = [
        "# PoseLoop M5-R1 phase 1 development result",
        "",
        f"**{result['status']}**",
        "",
        "The proposed method is a causal 12-state quotient constant-velocity "
        "Kalman filter with pose/twist cross-covariance. It is compared with "
        "nearest-representative CT under equal four-configuration budgets; every "
        "reported pair is selected without its outer replicate fold.",
        "",
        "| Metric | Nearest | Quotient CV Kalman |",
        "| --- | ---: | ---: |",
        f"| Cross-fitted mean trajectory loss | {gate['observed']['nearest_mean_loss']:.4f} | {gate['observed']['proposed_mean_loss']:.4f} |",
        f"| Relative improvement | — | {gate['observed']['relative_improvement']:+.1%} |",
        f"| One-sided 90% paired-bootstrap lower | — | {gate['observed']['one_sided_90pct_lower_relative_improvement']:+.1%} |",
        "",
        "The benchmark contains 768 fresh deterministic trajectories: four symmetry "
        "classes, four motions, six stress families, and eight replicates. Evaluator "
        "truth is never passed to either estimator.",
        "",
        "This phase establishes synthetic mechanism value only. Real sequence replay "
        "is a separate phase and cannot inherit a positive result unless its timestamp "
        "and observed-dropout input gate is satisfied.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(r1_root):
        raise ValueError("M5-R1 outputs must stay under artifacts/r1")
    report_path = args.report.resolve()
    if not report_path.is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("M5-R1 report must stay under reports/r1")
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage = protocol["stages"]["M5-R1"]
    grids = configuration_grid()
    if {method: len(grid) for method, grid in grids.items()} != {
        METHOD_NEAREST: 4,
        METHOD_PROPOSED: 4,
    }:
        raise ValueError("M5-R1 methods do not have equal four-configuration budgets")
    manifest = synthetic_manifest()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "synthetic_manifest.jsonl"
    write_jsonl_atomic(manifest_path, manifest)
    grid_rows = evaluate_grid(manifest, grids)
    grid_path = output_root / "phase1_grid_results.jsonl"
    write_jsonl_atomic(grid_path, grid_rows)
    oof_rows, selections = cross_fitted_comparison(grid_rows, manifest)
    oof_path = output_root / "phase1_cross_fitted_pairs.jsonl"
    write_jsonl_atomic(oof_path, oof_rows)
    nearest_mean = float(np.mean([float(row["nearest_loss"]) for row in oof_rows]))
    proposed_mean = float(np.mean([float(row["proposed_loss"]) for row in oof_rows]))
    bootstrap = paired_bootstrap(oof_rows)
    gate = gate_decision(nearest_mean, proposed_mean, bootstrap, stage["development_gate"])
    frozen = {
        METHOD_NEAREST: select_full_configuration(grid_rows, METHOD_NEAREST),
        METHOD_PROPOSED: select_full_configuration(grid_rows, METHOD_PROPOSED),
    }
    frozen_configs = {
        method: {
            **selection,
            "configuration": dataclasses.asdict(
                grids[method][str(selection["selected_config_id"])]
            ),
        }
        for method, selection in frozen.items()
    }
    config_path = output_root / "frozen_phase1_configurations.json"
    write_json_atomic(
        config_path,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "M5-R1 phase 1",
            "status": "frozen_for_real_replay" if gate["passed"] else "development_gate_failed",
            "equal_configuration_budget_per_method": 4,
            "selected": frozen_configs,
        },
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R1 phase 1",
        "status": "PASS_CONTINUE_TO_REAL_REPLAY" if gate["passed"] else "STOP_M5_R1_PHASE1_GATE_FAILED",
        "trajectory_count": len(manifest),
        "grid_result_count": len(grid_rows),
        "design": {
            "symmetry_classes": [item.value for item in legacy.SymmetryClass],
            "motion_families": [item.value for item in legacy.MotionFamily],
            "stress_families": [item.value for item in legacy.StressFamily],
            "replicates": REPLICATES,
            "outer_folds": OUTER_FOLDS,
            "equal_configuration_budget_per_method": 4,
            "outer_fold_group": "replicate",
        },
        "development_gate": gate,
        "paired_bootstrap": bootstrap,
        "outer_selections": selections,
        "selection_counts": {
            method: dict(
                Counter(
                    row["selected_config_id"]
                    for row in selections
                    if row["method"] == method
                )
            )
            for method in METHODS
        },
        "per_symmetry": difficulty_tables(oof_rows, "symmetry_class"),
        "per_motion": difficulty_tables(oof_rows, "motion_family"),
        "per_stress": difficulty_tables(oof_rows, "stress_family"),
        "failure_counts": dict(Counter(row["method"] for row in grid_rows if row["status"] != "success")),
        "frozen_full_development_configurations": frozen_configs,
        "phase2_real_replay_started": False,
        "claim_boundary": "Synthetic mechanism development evidence only.",
        "input_provenance": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        },
        "output_provenance": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "grid_results": {"path": str(grid_path), "sha256": sha256_file(grid_path)},
            "cross_fitted_pairs": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
            "frozen_configurations": {"path": str(config_path), "sha256": sha256_file(config_path)},
        },
    }
    result_path = output_root / "phase1_development_result.json"
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(
        f"{result['status']}: nearest={nearest_mean:.4f}, proposed={proposed_mean:.4f}, "
        f"relative={gate['observed']['relative_improvement']:+.2%}, "
        f"lower90={gate['observed']['one_sided_90pct_lower_relative_improvement']:+.2%}"
    )
    print(result_path)


if __name__ == "__main__":
    main()
