#!/usr/bin/env python3
"""Evaluate the frozen M5-R4A candidate family on new HB development data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import evaluate_m5_r2_once as r2_evaluator
import m5_g0_core as legacy
import m5_g0_metrics as metrics
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from m5_r4_core import (
    AdaptiveConfidenceConfig,
    AdaptiveObservation,
    run_object_local_adaptive_confidence,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "poseloop-m5-r4a-v1"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r4a"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "contract.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.jsonl")
    parser.add_argument(
        "--inference-receipt", type=Path, default=root / "inference_receipt.json"
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r4a_development.md",
    )
    return parser.parse_args()


def _config_from_protocol(
    protocol: Mapping[str, Any], candidate_id: str
) -> AdaptiveConfidenceConfig:
    family = protocol["methods"]["proposed_family"]
    values = dict(family["shared_configuration"])
    values.update(family["candidates"][candidate_id])
    return AdaptiveConfidenceConfig(**values)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    xr = _average_ranks(x)
    yr = _average_ranks(y)
    x_centered = xr - float(np.mean(xr))
    y_centered = yr - float(np.mean(yr))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def _load_and_validate(
    contract: Mapping[str, Any], predictions_path: Path, receipt_path: Path
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    for key in ("track_manifest", "evaluator_labels"):
        path = Path(str(contract["files"][key]["path"]))
        if sha256_file(path) != str(contract["files"][key]["sha256"]):
            raise RuntimeError(f"M5-R4 {key} hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or bool(
        receipt.get("evaluator_labels_path_read")
    ):
        raise RuntimeError("M5-R4 inference receipt is invalid")
    if sha256_file(predictions_path) != str(receipt["predictions"]["sha256"]):
        raise RuntimeError("M5-R4 predictions hash mismatch")
    prediction_rows = load_jsonl(predictions_path)
    if not prediction_rows or prediction_rows[0].get("record_type") != "m5_r4_inference_metadata":
        raise ValueError("M5-R4 prediction metadata missing")
    predictions = {
        str(row["sample_id"]): row
        for row in prediction_rows[1:]
        if row.get("record_type") == "m5_r4_development_prediction"
    }
    tracks = load_jsonl(Path(str(contract["files"]["track_manifest"]["path"])))
    labels_list = load_jsonl(Path(str(contract["files"]["evaluator_labels"]["path"])))
    labels = {
        (str(row["track_id"]), int(row["replay_frame_index"])): row
        for row in labels_list
    }
    if len(predictions) != int(contract["selection_summary"]["inference_sample_count"]):
        raise ValueError("M5-R4 prediction count mismatch")
    if len(labels) != int(contract["selection_summary"]["replay_frame_count"]):
        raise ValueError("M5-R4 evaluator label count mismatch")
    for track in tracks:
        frames = list(track["frames"])
        if not frames or not bool(frames[0]["input_available"]):
            raise RuntimeError(f"M5-R4 track cannot initialize: {track['track_id']}")
        first_prediction = predictions[str(frames[0]["sample_id"])]
        if first_prediction.get("status") != "success":
            raise RuntimeError(
                f"M5-R4 first measurement inference failed: {track['track_id']}"
            )
    return tracks, labels, predictions, receipt


def _build_track_inputs(
    track: Mapping[str, Any],
    labels: Mapping[tuple[str, int], Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    symmetry: legacy.SymmetrySpec,
) -> tuple[legacy.CorruptedSequence, list[AdaptiveObservation], list[dict[str, Any]]]:
    legacy_inputs: list[legacy.EstimatorInput] = []
    adaptive_inputs: list[AdaptiveObservation] = []
    evaluator_frames: list[legacy.EvaluatorFrame] = []
    gt_poses: list[np.ndarray] = []
    timestamps: list[float] = []
    detail: list[dict[str, Any]] = []
    track_id = str(track["track_id"])
    for frame in track["frames"]:
        replay_index = int(frame["replay_frame_index"])
        label = labels[(track_id, replay_index)]
        if str(label["sample_id"]) != str(frame["sample_id"]):
            raise ValueError(f"M5-R4 label/track mismatch: {track_id}/{replay_index}")
        natural_missing = bool(label["natural_input_missing"])
        if natural_missing == bool(frame["input_available"]):
            raise ValueError(f"M5-R4 availability mismatch: {track_id}/{replay_index}")
        measurement: np.ndarray | None = None
        prediction_status = "input_unavailable"
        if bool(frame["input_available"]):
            prediction = predictions[str(frame["sample_id"])]
            prediction_status = str(prediction["status"])
            if prediction_status == "success":
                measurement = np.asarray(
                    prediction["predicted_model_to_camera_pose_m"], dtype=np.float64
                )
        gt_pose = np.asarray(label["gt_model_to_camera_pose_m"], dtype=np.float64)
        timestamp = float(frame["timestamp_s"])
        legacy_inputs.append(
            legacy.EstimatorInput(
                timestamp_s=timestamp,
                measurement_pose=None if measurement is None else measurement.copy(),
                missing=measurement is None,
                symmetry=symmetry,
            )
        )
        adaptive_inputs.append(
            AdaptiveObservation(
                timestamp_s=timestamp,
                measurement_pose=None if measurement is None else measurement.copy(),
                symmetry=symmetry,
                mask_area_fraction=(
                    float(frame["input_mask_area_fraction"])
                    if measurement is not None
                    else None
                ),
                valid_depth_ratio=(
                    float(frame["valid_depth_ratio_inside_mask"])
                    if measurement is not None
                    else None
                ),
            )
        )
        evaluator_frames.append(
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
        detail.append(
            {
                "replay_frame_index": replay_index,
                "source_frame_ordinal": int(frame["source_frame_ordinal"]),
                "timestamp_s": timestamp,
                "sample_id": str(frame["sample_id"]),
                "natural_input_missing": natural_missing,
                "prediction_status": prediction_status,
                "measurement_model_to_camera_pose_m": (
                    None if measurement is None else measurement.tolist()
                ),
                "gt_model_to_camera_pose_m": gt_pose.tolist(),
            }
        )
    truth = legacy.GroundTruthSequence(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        poses=np.stack(gt_poses),
        symmetry=symmetry,
        motion_family=legacy.MotionFamily.CONSTANT_TWIST,
        seed=0,
        reversal_frame=None,
        translation_direction_camera=np.zeros(3, dtype=np.float64),
        observable_rotation_axis_object=np.asarray(symmetry.axis_object, dtype=np.float64),
    )
    sequence = legacy.CorruptedSequence(
        inputs=tuple(legacy_inputs),
        evaluator_frames=tuple(evaluator_frames),
        ground_truth=truth,
        stress_family=legacy.StressFamily.NOMINAL,
        corruption_seed=0,
        dropout_intervals=(),
        outlier_intervals=(),
    )
    return sequence, adaptive_inputs, detail


def _evaluate_track(
    track: Mapping[str, Any],
    sequence: legacy.CorruptedSequence,
    adaptive_inputs: list[AdaptiveObservation],
    detail: list[dict[str, Any]],
    nearest_config: legacy.EstimatorConfig,
    proposed_config: AdaptiveConfidenceConfig,
    candidate_id: str,
) -> dict[str, Any]:
    baseline_outputs = legacy.run_estimator(
        sequence, legacy.EstimatorKind.NEAREST_REPRESENTATIVE_CT, nearest_config
    )
    proposed_outputs = run_object_local_adaptive_confidence(
        adaptive_inputs, proposed_config
    )
    baseline_poses = np.stack([row.pose for row in baseline_outputs])
    proposed_poses = np.stack([row.pose for row in proposed_outputs])
    baseline_metrics = metrics.compute_per_frame_pose_errors(
        sequence.ground_truth.poses, baseline_poses, sequence.ground_truth.symmetry
    )
    proposed_metrics = metrics.compute_per_frame_pose_errors(
        sequence.ground_truth.poses, proposed_poses, sequence.ground_truth.symmetry
    )
    baseline_loss = np.asarray(baseline_metrics["normalized_pose_error"], dtype=np.float64)
    proposed_loss = np.asarray(proposed_metrics["normalized_pose_error"], dtype=np.float64)
    natural_missing = np.asarray(
        [bool(row["natural_input_missing"]) for row in detail], dtype=bool
    )
    if not np.any(natural_missing):
        raise ValueError(f"M5-R4 track has no natural missing frames: {track['track_id']}")
    frames: list[dict[str, Any]] = []
    for index, base in enumerate(detail):
        proposed = proposed_outputs[index]
        frames.append(
            {
                **base,
                "baseline_output_pose_m": baseline_poses[index].tolist(),
                "proposed_output_pose_m": proposed_poses[index].tolist(),
                "baseline_normalized_pose_error": float(baseline_loss[index]),
                "proposed_normalized_pose_error": float(proposed_loss[index]),
                "proposed_uncertainty": proposed.uncertainty,
                "proposed_confidence": proposed.confidence,
                "proposed_support_quality": proposed.support_quality,
                "proposed_gap_frames": proposed.gap_frames,
                "baseline_reason": baseline_outputs[index].reason,
                "proposed_reason": proposed.reason,
                "baseline_accepted": bool(baseline_outputs[index].accepted),
                "proposed_accepted": bool(proposed.accepted),
            }
        )
    baseline_mean = float(np.mean(baseline_loss))
    proposed_mean = float(np.mean(proposed_loss))
    baseline_missing = float(np.mean(baseline_loss[natural_missing]))
    proposed_missing = float(np.mean(proposed_loss[natural_missing]))
    nonfinite = int(np.sum(~np.isfinite(proposed_poses).all(axis=(1, 2))))
    nonfinite += int(np.sum(~np.isfinite([row.uncertainty for row in proposed_outputs])))
    return {
        "record_type": "m5_r4_candidate_track_result",
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "track_id": str(track["track_id"]),
        "scene_id": int(track["scene_id"]),
        "object_id": int(track["object_id"]),
        "frame_count": len(frames),
        "natural_missing_frame_count": int(np.sum(natural_missing)),
        "predictor_failure_frame_count": sum(
            row["prediction_status"] not in ("success", "input_unavailable") for row in frames
        ),
        "nonfinite_output_count": nonfinite,
        "baseline_loss": baseline_mean,
        "proposed_loss": proposed_mean,
        "relative_improvement": (baseline_mean - proposed_mean) / baseline_mean,
        "baseline_natural_missing_loss": baseline_missing,
        "proposed_natural_missing_loss": proposed_missing,
        "natural_missing_relative_improvement": (
            baseline_missing - proposed_missing
        )
        / baseline_missing,
        "baseline_translation_rmse_mm": 1000.0
        * float(baseline_metrics["translation"]["rmse"]),
        "proposed_translation_rmse_mm": 1000.0
        * float(proposed_metrics["translation"]["rmse"]),
        "baseline_rotation_rmse_deg": float(baseline_metrics["rotation"]["rmse"]),
        "proposed_rotation_rmse_deg": float(proposed_metrics["rotation"]["rmse"]),
        "frames": frames,
    }


def _macro(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return r2_evaluator.macro_object_mean(rows, field)


def _candidate_summary(rows: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    baseline = _macro(rows, "baseline_loss")
    proposed = _macro(rows, "proposed_loss")
    missing_baseline = _macro(rows, "baseline_natural_missing_loss")
    missing_proposed = _macro(rows, "proposed_natural_missing_loss")
    return {
        "candidate_id": candidate_id,
        "baseline_macro_object_loss": baseline,
        "proposed_macro_object_loss": proposed,
        "macro_object_relative_improvement": (baseline - proposed) / baseline,
        "baseline_natural_missing_macro_object_loss": missing_baseline,
        "proposed_natural_missing_macro_object_loss": missing_proposed,
        "natural_missing_macro_object_relative_improvement": (
            missing_baseline - missing_proposed
        )
        / missing_baseline,
        "nonfinite_output_count": sum(int(row["nonfinite_output_count"]) for row in rows),
    }


def render_report(result: Mapping[str, Any]) -> str:
    observed = result["development_gate"]["observed"]
    thresholds = result["development_gate"]["thresholds"]
    lines = [
        "# PoseLoop M5-R4A HB Primesense development",
        "",
        f"Status: **{result['status']}**",
        "",
        "M5-R4A was frozen before inference after M5-R4 exposed only an input-feasibility shortfall. It uses new HB Primesense scenes 1-8 with object-grouped cross-fitting. The estimator uses only causal object-local mask/depth support and innovation consistency; raw FoundationPose scores are excluded from confidence. Kinect 2 scenes 9-13 remain untouched.",
        "",
        "| Development condition | Observed | Required |",
        "|---|---:|---:|",
        f"| Cross-fitted all-frame improvement | {observed['cross_fitted_macro_object_relative_improvement']:+.2%} | >= {thresholds['cross_fitted_macro_object_relative_improvement_min']:.0%} |",
        f"| Cross-fitted missing-frame improvement | {observed['cross_fitted_natural_missing_frame_relative_improvement']:+.2%} | >= {thresholds['cross_fitted_natural_missing_frame_relative_improvement_min']:.0%} |",
        f"| Bootstrap 10th percentile | {observed['one_sided_90pct_lower_relative_improvement']:+.2%} | >= {thresholds['one_sided_hierarchical_bootstrap_90pct_lower_improvement_min']:.0%} |",
        f"| Missing uncertainty/error Spearman | {observed['missing_frame_uncertainty_error_spearman']:+.3f} | >= {thresholds['missing_frame_uncertainty_error_spearman_min']:.2f} |",
        f"| Nonfinite outputs | {observed['nonfinite_output_count']} | <= {thresholds['nonfinite_output_count_max']} |",
        "",
        "| Candidate | All-frame improvement | Missing improvement |",
        "|---|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['candidate_id']} | {row['macro_object_relative_improvement']:+.2%} | "
            f"{row['natural_missing_macro_object_relative_improvement']:+.2%} |"
        )
    lines.extend(
        [
            "",
            f"Modal fold winner for any future sealed freeze: `{result['modal_candidate_id']}`.",
            "",
            "This is development evidence only. A failed gate forbids downloading or inspecting the Kinect 2 sealed split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    predictions_path = args.predictions.resolve()
    receipt_path = args.inference_receipt.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    result_path = output_root / "development_result.json"
    track_results_path = output_root / "development_track_results.jsonl"
    if result_path.exists() or track_results_path.exists():
        raise FileExistsError("M5-R4 development result already exists")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("protocol_id") != PROTOCOL_ID or bool(
        contract.get("sealed_archive_or_label_read")
    ):
        raise RuntimeError("Invalid M5-R4 development contract")
    if sha256_file(Path(__file__).resolve()) != contract["code"]["evaluator"]["sha256"]:
        raise RuntimeError("M5-R4 evaluator changed after bundle freeze")
    core_path = Path(str(contract["code"]["core"]["path"]))
    if sha256_file(core_path) != contract["code"]["core"]["sha256"]:
        raise RuntimeError("M5-R4 core changed after bundle freeze")
    protocol_path = Path(str(contract["protocol"]["path"]))
    if sha256_file(protocol_path) != contract["protocol"]["sha256"]:
        raise RuntimeError("M5-R4 protocol changed after bundle freeze")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tracks, labels, predictions, inference_receipt = _load_and_validate(
        contract, predictions_path, receipt_path
    )
    models_info_path = Path(str(contract["dataset"]["models_info"]["path"]))
    if sha256_file(models_info_path) != contract["dataset"]["models_info"]["sha256"]:
        raise RuntimeError("M5-R4 models_info hash mismatch")
    models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
    config_path = Path(
        str(protocol["methods"]["baseline"]["configuration_source"])
    )
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    frozen_configs = json.loads(config_path.read_text(encoding="utf-8"))
    nearest_config = legacy.EstimatorConfig(
        **frozen_configs["selected"]["NEAREST_REPRESENTATIVE_CT"]["configuration"]
    )
    candidate_order = list(protocol["methods"]["proposed_family"]["candidate_order"])
    results_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidate_order
    }
    for track_index, track in enumerate(tracks, start=1):
        object_id = int(track["object_id"])
        symmetry = r2_evaluator.symmetry_from_model_info(models_info[str(object_id)])
        sequence, adaptive_inputs, detail = _build_track_inputs(
            track, labels, predictions, symmetry
        )
        for candidate_id in candidate_order:
            results_by_candidate[candidate_id].append(
                _evaluate_track(
                    track,
                    sequence,
                    adaptive_inputs,
                    detail,
                    nearest_config,
                    _config_from_protocol(protocol, candidate_id),
                    candidate_id,
                )
            )
        print(f"M5-R4 track {track_index:02d}/{len(tracks):02d}: {track['track_id']}")

    object_ids = sorted({int(track["object_id"]) for track in tracks})
    fold_results: list[dict[str, Any]] = []
    cross_fitted: list[dict[str, Any]] = []
    for held_out in object_ids:
        training_scores: dict[str, float] = {}
        for candidate_id in candidate_order:
            training = [
                row
                for row in results_by_candidate[candidate_id]
                if int(row["object_id"]) != held_out
            ]
            training_scores[candidate_id] = _candidate_summary(
                training, candidate_id
            )["macro_object_relative_improvement"]
        selected = max(
            candidate_order,
            key=lambda candidate_id: (
                training_scores[candidate_id],
                -candidate_order.index(candidate_id),
            ),
        )
        held_rows = [
            row
            for row in results_by_candidate[selected]
            if int(row["object_id"]) == held_out
        ]
        cross_fitted.extend(held_rows)
        held_summary = _candidate_summary(held_rows, selected)
        fold_results.append(
            {
                "held_out_object_id": held_out,
                "selected_candidate_id": selected,
                "training_relative_improvement_by_candidate": training_scores,
                "held_out_relative_improvement": held_summary[
                    "macro_object_relative_improvement"
                ],
                "held_out_missing_relative_improvement": held_summary[
                    "natural_missing_macro_object_relative_improvement"
                ],
            }
        )
    winner_counts = Counter(row["selected_candidate_id"] for row in fold_results)
    modal_candidate = max(
        candidate_order,
        key=lambda candidate_id: (
            winner_counts[candidate_id], -candidate_order.index(candidate_id)
        ),
    )
    candidate_summaries = [
        _candidate_summary(results_by_candidate[candidate_id], candidate_id)
        for candidate_id in candidate_order
    ]
    cross_summary = _candidate_summary(cross_fitted, "cross_fitted")
    bootstrap_spec = protocol["evaluation"]["bootstrap"]
    bootstrap = r2_evaluator.hierarchical_paired_bootstrap(
        cross_fitted,
        seed=int(bootstrap_spec["seed"]),
        resamples=int(bootstrap_spec["resamples"]),
    )
    missing_uncertainty: list[float] = []
    missing_error: list[float] = []
    for track_result in cross_fitted:
        for frame in track_result["frames"]:
            if frame["natural_input_missing"]:
                missing_uncertainty.append(float(frame["proposed_uncertainty"]))
                missing_error.append(float(frame["proposed_normalized_pose_error"]))
    confidence_spearman = spearman_correlation(missing_uncertainty, missing_error)
    gate_spec = protocol["evaluation"]["development_gate"]
    observed = {
        "represented_object_count": len(object_ids),
        "track_count": len(cross_fitted),
        "natural_missing_frame_count": len(missing_error),
        "cross_fitted_macro_object_relative_improvement": cross_summary[
            "macro_object_relative_improvement"
        ],
        "cross_fitted_natural_missing_frame_relative_improvement": cross_summary[
            "natural_missing_macro_object_relative_improvement"
        ],
        "one_sided_90pct_lower_relative_improvement": bootstrap[
            "one_sided_90pct_lower_relative_improvement"
        ],
        "missing_frame_uncertainty_error_spearman": confidence_spearman,
        "nonfinite_output_count": sum(
            int(row["nonfinite_output_count"]) for row in cross_fitted
        ),
    }
    conditions = {
        "represented_object_count": observed["represented_object_count"]
        >= int(gate_spec["represented_object_count_min"]),
        "track_count": observed["track_count"] >= int(gate_spec["track_count_min"]),
        "natural_missing_frame_count": observed["natural_missing_frame_count"]
        >= int(gate_spec["natural_missing_frame_count_min"]),
        "cross_fitted_macro_object_relative_improvement": observed[
            "cross_fitted_macro_object_relative_improvement"
        ]
        >= float(gate_spec["cross_fitted_macro_object_relative_improvement_min"]),
        "cross_fitted_natural_missing_frame_relative_improvement": observed[
            "cross_fitted_natural_missing_frame_relative_improvement"
        ]
        >= float(
            gate_spec["cross_fitted_natural_missing_frame_relative_improvement_min"]
        ),
        "one_sided_hierarchical_bootstrap_90pct_lower": observed[
            "one_sided_90pct_lower_relative_improvement"
        ]
        >= float(
            gate_spec[
                "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"
            ]
        ),
        "missing_frame_uncertainty_error_spearman": math.isfinite(confidence_spearman)
        and confidence_spearman
        >= float(gate_spec["missing_frame_uncertainty_error_spearman_min"]),
        "nonfinite_output_count": observed["nonfinite_output_count"]
        <= int(gate_spec["nonfinite_output_count_max"]),
    }
    passed = all(conditions.values())
    per_object: list[dict[str, Any]] = []
    for object_id in object_ids:
        rows = [row for row in cross_fitted if int(row["object_id"]) == object_id]
        summary = _candidate_summary(rows, "cross_fitted")
        per_object.append(
            {
                "object_id": object_id,
                "track_count": len(rows),
                "selected_candidate_ids": sorted({str(row["candidate_id"]) for row in rows}),
                "baseline_loss": summary["baseline_macro_object_loss"],
                "proposed_loss": summary["proposed_macro_object_loss"],
                "relative_improvement": summary["macro_object_relative_improvement"],
                "natural_missing_relative_improvement": summary[
                    "natural_missing_macro_object_relative_improvement"
                ],
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "M5-R4A HB Primesense object-grouped development",
        "status": "PASS_M5_R4A_DEVELOPMENT" if passed else "FAIL_M5_R4A_DEVELOPMENT",
        "development_gate": {
            "passed": passed,
            "conditions": conditions,
            "observed": observed,
            "thresholds": gate_spec,
        },
        "candidates": candidate_summaries,
        "leave_one_object_out": fold_results,
        "fold_winner_counts": dict(sorted(winner_counts.items())),
        "modal_candidate_id": modal_candidate,
        "hierarchical_paired_bootstrap": bootstrap,
        "per_object": per_object,
        "inference_status_counts": inference_receipt["status_counts"],
        "track_count": len(cross_fitted),
        "object_count": len(object_ids),
        "natural_missing_frame_count": len(missing_error),
        "raw_foundationpose_scores_used_for_filtering_or_confidence": False,
        "sealed_archive_or_label_read": False,
        "claim_scope": protocol["claim_boundary"]["development_allowed"],
    }
    write_jsonl_atomic(track_results_path, cross_fitted)
    result["track_results"] = {
        "path": str(track_results_path),
        "sha256": sha256_file(track_results_path),
    }
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["development_gate"], indent=2, sort_keys=True))
    print(result_path)
    print(report_path)


if __name__ == "__main__":
    main()
