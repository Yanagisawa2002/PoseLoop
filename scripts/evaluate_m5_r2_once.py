#!/usr/bin/env python3
"""Open the frozen M5-R2 evaluator stream exactly once and compare methods."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m5_g0_core as legacy
import m5_g0_metrics as metrics
import m5_r2_core as r2_core
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from m5_r1_core import QuotientCVKalmanConfig, run_quotient_cv_kalman


SCHEMA_VERSION = 1
BASELINE = "NEAREST_REPRESENTATIVE_CT"
PROPOSED = "QUOTIENT_CV_KALMAN"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "contract.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.jsonl")
    parser.add_argument("--inference-receipt", type=Path, default=root / "inference_receipt.json")
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m5_r2_real_replay.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def symmetry_from_model_info(info: Mapping[str, Any]) -> legacy.SymmetrySpec:
    continuous = list(info.get("symmetries_continuous", []))
    discrete = list(info.get("symmetries_discrete", []))
    if continuous and discrete:
        raise ValueError("Combined continuous/discrete symmetry is unsupported in M5-R2")
    if continuous:
        if len(continuous) != 1:
            raise ValueError("M5-R2 expects one continuous symmetry generator")
        axis = np.asarray(continuous[0]["axis"], dtype=np.float64).reshape(3)
        offset = np.asarray(continuous[0].get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        if np.linalg.norm(offset) > 1e-12:
            raise ValueError("M5-R2 continuous symmetry requires a zero object-frame offset")
        axis /= np.linalg.norm(axis)
        return legacy.SymmetrySpec(
            legacy.SymmetryClass.CONTINUOUS_AXIAL,
            (np.eye(4, dtype=np.float64),),
            axis_object=axis,
        )
    transforms = [np.eye(4, dtype=np.float64)]
    for raw in discrete:
        transform = np.asarray(raw, dtype=np.float64).reshape(4, 4).copy()
        # BOP model-space translations are millimetres; replay poses are metres.
        transform[:3, 3] *= 0.001
        transforms.append(transform)
    class_by_count = {
        1: legacy.SymmetryClass.ASYMMETRIC,
        2: legacy.SymmetryClass.C2,
        4: legacy.SymmetryClass.C4,
    }
    if len(transforms) not in class_by_count:
        raise ValueError(f"Unsupported finite symmetry count: {len(transforms)}")
    return legacy.SymmetrySpec(class_by_count[len(transforms)], tuple(transforms))


def macro_object_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    by_object: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite M5-R2 metric: {field}")
        by_object[int(row["object_id"])].append(value)
    if not by_object:
        raise ValueError("Cannot aggregate an empty M5-R2 result")
    return float(
        np.mean([np.mean(by_object[object_id]) for object_id in sorted(by_object)])
    )


def hierarchical_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> dict[str, Any]:
    by_object: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[int(row["object_id"])].append(row)
    object_ids = np.asarray(sorted(by_object), dtype=np.int64)
    if len(object_ids) < 2:
        raise ValueError("Hierarchical bootstrap requires at least two objects")
    rng = np.random.default_rng(int(seed))
    improvements = np.empty(int(resamples), dtype=np.float64)
    for bootstrap_index in range(int(resamples)):
        sampled_objects = rng.choice(object_ids, size=len(object_ids), replace=True)
        baseline_objects: list[float] = []
        proposed_objects: list[float] = []
        for object_id in sampled_objects:
            object_rows = by_object[int(object_id)]
            indices = rng.integers(0, len(object_rows), size=len(object_rows))
            baseline_objects.append(
                float(np.mean([float(object_rows[index]["baseline_loss"]) for index in indices]))
            )
            proposed_objects.append(
                float(np.mean([float(object_rows[index]["proposed_loss"]) for index in indices]))
            )
        baseline = float(np.mean(baseline_objects))
        proposed = float(np.mean(proposed_objects))
        improvements[bootstrap_index] = (baseline - proposed) / baseline
    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "unit": "paired hierarchical object then complete-track resampling",
        "mean_relative_improvement": float(np.mean(improvements)),
        "one_sided_90pct_lower_relative_improvement": float(
            np.quantile(improvements, 0.10)
        ),
        "two_sided_90pct_upper_relative_improvement": float(
            np.quantile(improvements, 0.95)
        ),
    }


def validate_unopened_contract(contract: Mapping[str, Any], contract_path: Path) -> None:
    if (
        contract.get("protocol_id") != "poseloop-m5-r2-v1"
        or contract.get("status") != "development_labels_unopened"
        or bool(contract.get("labels_opened"))
        or int(contract.get("evaluation_invocation_count", -1)) != 0
    ):
        raise RuntimeError("M5-R2 evaluator contract is not unopened")
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    for name, record in contract["code"].items():
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"M5-R2 frozen code changed: {name}")


def load_label_blind_inputs(
    contract: Mapping[str, Any], prediction_path: Path, receipt_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    track_path = Path(str(contract["files"]["track_manifest"]["path"]))
    manifest_path = Path(str(contract["files"]["inference_manifest"]["path"]))
    for name, path in (("track_manifest", track_path), ("inference_manifest", manifest_path)):
        if sha256_file(path) != str(contract["files"][name]["sha256"]):
            raise RuntimeError(f"M5-R2 {name} hash changed")
    tracks = load_jsonl(track_path)
    manifest = load_jsonl(manifest_path)
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if len(manifest_ids) != int(contract["selection_summary"]["inference_sample_count"]):
        raise ValueError("M5-R2 inference manifest identity count changed")
    prediction_rows = load_jsonl(prediction_path)
    if not prediction_rows or prediction_rows[0].get("record_type") != "m5_r2_inference_metadata":
        raise ValueError("M5-R2 predictions lack metadata")
    if (
        prediction_rows[0].get("manifest_sha256")
        != contract["files"]["inference_manifest"]["sha256"]
        or int(prediction_rows[0].get("sample_count", -1)) != len(manifest_ids)
        or bool(prediction_rows[0].get("evaluator_labels_path_read"))
        or bool(prediction_rows[0].get("evaluator_pose_or_error_computed"))
    ):
        raise RuntimeError("M5-R2 prediction metadata failed label-blind validation")
    predictions: dict[str, dict[str, Any]] = {}
    for row in prediction_rows[1:]:
        if row.get("record_type") != "m5_r2_prediction" or bool(
            row.get("evaluator_label_read")
        ):
            raise ValueError("Invalid M5-R2 prediction row")
        sample_id = str(row["sample_id"])
        if sample_id not in manifest_ids or sample_id in predictions:
            raise ValueError(f"Invalid M5-R2 prediction identity: {sample_id}")
        predictions[sample_id] = row
    if set(predictions) != manifest_ids:
        raise ValueError("M5-R2 predictions do not cover the complete manifest")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or bool(receipt.get("labels_opened"))
        or int(receipt.get("sample_count", -1)) != len(manifest_ids)
        or receipt.get("manifest_sha256")
        != contract["files"]["inference_manifest"]["sha256"]
        or sha256_file(prediction_path) != str(receipt["predictions"]["sha256"])
    ):
        raise RuntimeError("M5-R2 inference receipt failed validation")
    track_ids = [str(row["track_id"]) for row in tracks]
    if len(track_ids) != len(set(track_ids)) or len(tracks) != int(
        contract["selection_summary"]["track_count"]
    ):
        raise ValueError("M5-R2 track manifest identity count changed")
    referenced_available_ids = {
        str(frame["sample_id"])
        for track in tracks
        for frame in track["frames"]
        if bool(frame["observation_available"])
    }
    if referenced_available_ids != manifest_ids:
        raise ValueError("M5-R2 track availability differs from the inference manifest")
    for track in tracks:
        frames = track["frames"]
        if not frames or not bool(frames[0]["observation_available"]):
            raise RuntimeError(f"M5-R2 track cannot initialize: {track['track_id']}")
        first_prediction = predictions[str(frames[0]["sample_id"])]
        if first_prediction.get("status") != "success":
            raise RuntimeError(
                f"M5-R2 first measurement inference failed: {track['track_id']}"
            )
    return tracks, predictions, receipt


def build_sequence(
    track: Mapping[str, Any],
    labels: Mapping[tuple[str, int], Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    symmetry: legacy.SymmetrySpec,
) -> tuple[legacy.CorruptedSequence, list[dict[str, Any]], np.ndarray]:
    inputs: list[legacy.EstimatorInput] = []
    evaluator_frames: list[legacy.EvaluatorFrame] = []
    gt_poses: list[np.ndarray] = []
    detail: list[dict[str, Any]] = []
    timestamps: list[float] = []
    track_id = str(track["track_id"])
    canonical_to_original, filter_symmetry = r2_core.canonicalization_for_symmetry(
        symmetry
    )
    for frame in track["frames"]:
        replay_index = int(frame["replay_frame_index"])
        label = labels[(track_id, replay_index)]
        if (
            str(label["sample_id"]) != str(frame["sample_id"])
            or bool(label["observation_available"])
            != bool(frame["observation_available"])
        ):
            raise ValueError(f"M5-R2 label/track mismatch: {track_id}/{replay_index}")
        gt_world_original = np.asarray(
            label["gt_model_to_world_pose_m"], dtype=np.float64
        )
        measurement_world_original: np.ndarray | None = None
        prediction_status = "input_unavailable"
        if bool(frame["observation_available"]):
            prediction = predictions[str(frame["sample_id"])]
            prediction_status = str(prediction["status"])
            if prediction_status == "success":
                predicted_camera = np.asarray(
                    prediction["predicted_model_to_camera_pose_m"], dtype=np.float64
                )
                world_to_camera = np.asarray(
                    frame["camera_world_to_camera_pose_m"], dtype=np.float64
                )
                measurement_world_original = (
                    legacy.rigid_inverse(world_to_camera) @ predicted_camera
                )
        gt_world = r2_core.to_canonical_pose(
            gt_world_original, canonical_to_original
        )
        measurement_world = (
            None
            if measurement_world_original is None
            else r2_core.to_canonical_pose(
                measurement_world_original, canonical_to_original
            )
        )
        timestamp = float(frame["timestamp_s"])
        inputs.append(
            legacy.EstimatorInput(
                timestamp_s=timestamp,
                measurement_pose=(
                    None if measurement_world is None else measurement_world.copy()
                ),
                missing=measurement_world is None,
                symmetry=filter_symmetry,
            )
        )
        evaluator_frames.append(
            legacy.EvaluatorFrame(
                ground_truth_pose=gt_world.copy(),
                noisy_canonical_measurement_pose=(
                    gt_world.copy()
                    if measurement_world is None
                    else measurement_world.copy()
                ),
                representative_index=None,
                axial_gauge_rad=None,
                dropout=measurement_world is None,
                outlier=False,
            )
        )
        gt_poses.append(gt_world)
        timestamps.append(timestamp)
        detail.append(
            {
                "replay_frame_index": replay_index,
                "source_frame_ordinal": int(frame["source_frame_ordinal"]),
                "timestamp_s": timestamp,
                "sample_id": str(frame["sample_id"]),
                "natural_input_missing": not bool(frame["observation_available"]),
                "prediction_status": prediction_status,
                "measurement_model_to_world_pose_m": (
                    None
                    if measurement_world_original is None
                    else measurement_world_original.tolist()
                ),
                "gt_model_to_world_pose_m": gt_world_original.tolist(),
            }
        )
    truth = legacy.GroundTruthSequence(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        poses=np.stack(gt_poses),
        symmetry=filter_symmetry,
        motion_family=legacy.MotionFamily.STATIC,
        seed=0,
        reversal_frame=None,
        translation_direction_camera=np.zeros(3, dtype=np.float64),
        observable_rotation_axis_object=np.asarray(
            filter_symmetry.axis_object, dtype=np.float64
        ),
    )
    sequence = legacy.CorruptedSequence(
        inputs=tuple(inputs),
        evaluator_frames=tuple(evaluator_frames),
        ground_truth=truth,
        stress_family=legacy.StressFamily.NOMINAL,
        corruption_seed=0,
        dropout_intervals=(),
        outlier_intervals=(),
    )
    return sequence, detail, canonical_to_original


def evaluate_track(
    track: Mapping[str, Any],
    sequence: legacy.CorruptedSequence,
    detail: list[dict[str, Any]],
    canonical_to_original: np.ndarray,
    nearest_config: legacy.EstimatorConfig,
    proposed_config: QuotientCVKalmanConfig,
) -> dict[str, Any]:
    baseline_outputs = legacy.run_estimator(
        sequence, legacy.EstimatorKind.NEAREST_REPRESENTATIVE_CT, nearest_config
    )
    proposed_outputs = run_quotient_cv_kalman(sequence, proposed_config)
    baseline_poses = np.stack([row.pose for row in baseline_outputs])
    proposed_poses = np.stack([row.pose for row in proposed_outputs])
    baseline_poses_original = np.stack(
        [
            r2_core.from_canonical_pose(pose, canonical_to_original)
            for pose in baseline_poses
        ]
    )
    proposed_poses_original = np.stack(
        [
            r2_core.from_canonical_pose(pose, canonical_to_original)
            for pose in proposed_poses
        ]
    )
    nonfinite = int(np.sum(~np.isfinite(baseline_poses).all(axis=(1, 2)))) + int(
        np.sum(~np.isfinite(proposed_poses).all(axis=(1, 2)))
    )
    baseline_metrics = metrics.compute_per_frame_pose_errors(
        sequence.ground_truth.poses, baseline_poses, sequence.ground_truth.symmetry
    )
    proposed_metrics = metrics.compute_per_frame_pose_errors(
        sequence.ground_truth.poses, proposed_poses, sequence.ground_truth.symmetry
    )
    baseline_loss = np.asarray(
        baseline_metrics["normalized_pose_error"], dtype=np.float64
    )
    proposed_loss = np.asarray(
        proposed_metrics["normalized_pose_error"], dtype=np.float64
    )
    natural_missing = np.asarray(
        [bool(row["natural_input_missing"]) for row in detail], dtype=bool
    )
    if not np.any(natural_missing):
        raise RuntimeError(f"M5-R2 selected track lacks natural missing frames: {track['track_id']}")
    for index, row in enumerate(detail):
        row.update(
            {
                "baseline_output_pose_m": baseline_poses_original[index].tolist(),
                "proposed_output_pose_m": proposed_poses_original[index].tolist(),
                "baseline_normalized_pose_error": float(baseline_loss[index]),
                "proposed_normalized_pose_error": float(proposed_loss[index]),
                "baseline_reason": baseline_outputs[index].reason,
                "proposed_reason": proposed_outputs[index].reason,
                "baseline_accepted": bool(baseline_outputs[index].accepted),
                "proposed_accepted": bool(proposed_outputs[index].accepted),
            }
        )
    first_gt = sequence.ground_truth.poses[0]
    gt_spans = [
        legacy.quotient_pose_error(first_gt, pose, sequence.ground_truth.symmetry)
        for pose in sequence.ground_truth.poses
    ]
    return {
        "record_type": "m5_r2_track_result",
        "schema_version": SCHEMA_VERSION,
        "track_id": str(track["track_id"]),
        "scene_id": int(track["scene_id"]),
        "object_id": int(track["object_id"]),
        "frame_count": len(detail),
        "natural_missing_frame_count": int(np.sum(natural_missing)),
        "predictor_failure_frame_count": int(
            sum(
                row["prediction_status"] not in {"success", "input_unavailable"}
                for row in detail
            )
        ),
        "baseline_loss": float(np.mean(baseline_loss)),
        "proposed_loss": float(np.mean(proposed_loss)),
        "relative_improvement": float(
            (np.mean(baseline_loss) - np.mean(proposed_loss))
            / np.mean(baseline_loss)
        ),
        "baseline_natural_missing_loss": float(np.mean(baseline_loss[natural_missing])),
        "proposed_natural_missing_loss": float(np.mean(proposed_loss[natural_missing])),
        "natural_missing_relative_improvement": float(
            (np.mean(baseline_loss[natural_missing]) - np.mean(proposed_loss[natural_missing]))
            / np.mean(baseline_loss[natural_missing])
        ),
        "baseline_translation_rmse_mm": 1000.0
        * float(baseline_metrics["translation"]["rmse"]),
        "proposed_translation_rmse_mm": 1000.0
        * float(proposed_metrics["translation"]["rmse"]),
        "baseline_rotation_rmse_deg": float(baseline_metrics["rotation"]["rmse"]),
        "proposed_rotation_rmse_deg": float(proposed_metrics["rotation"]["rmse"]),
        "nonfinite_output_count": nonfinite,
        "gt_world_translation_span_mm_max": 1000.0
        * max(value[0] for value in gt_spans),
        "gt_world_quotient_rotation_span_deg_max": max(value[1] for value in gt_spans),
        "frames": detail,
    }


def render_report(result: Mapping[str, Any]) -> str:
    gate = result["development_gate"]
    lines = [
        "# PoseLoop M5-R2 recorded RealSense replay",
        "",
        f"**{result['status']}**",
        "",
        "The two phase-1-frozen temporal methods were replayed on 21 recorded "
        "physical tracks selected only because their input streams contain natural "
        "post-initialization dropouts. FoundationPose was rerun independently on every "
        "available frame before the evaluator stream was opened once.",
        "",
        "| Macro-object metric | Nearest | Quotient CV Kalman | Relative delta |",
        "| --- | ---: | ---: | ---: |",
        f"| All replay frames | {gate['observed']['baseline_macro_object_loss']:.4f} | "
        f"{gate['observed']['proposed_macro_object_loss']:.4f} | "
        f"{gate['observed']['macro_object_relative_improvement']:+.1%} |",
        f"| Natural missing frames | {gate['observed']['baseline_natural_missing_macro_object_loss']:.4f} | "
        f"{gate['observed']['proposed_natural_missing_macro_object_loss']:.4f} | "
        f"{gate['observed']['natural_missing_macro_object_relative_improvement']:+.1%} |",
        f"| One-sided 90% hierarchical-bootstrap lower | — | — | "
        f"{gate['observed']['one_sided_90pct_lower_relative_improvement']:+.1%} |",
        "",
        "| Object | Tracks | Missing frames | Nearest loss | Proposed loss | Delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["per_object"]:
        lines.append(
            f"| {row['object_id']:02d} | {row['track_count']} | "
            f"{row['natural_missing_frame_count']} | {row['baseline_loss']:.4f} | "
            f"{row['proposed_loss']:.4f} | {row['relative_improvement']:+.1%} |"
        )
    lines.extend(
        [
            "",
            f"Inference statuses: {result['inference_status_counts']}. Non-finite "
            f"temporal outputs: {gate['observed']['nonfinite_output_count']}.",
            "",
            "Claim boundary: this is observed-dropout-enriched development replay on "
            "recorded XYZ-IBD RealSense multi-view data with a nominal 20 Hz frame "
            "clock and oracle association. The objects are static in the world frame; "
            "it is not sealed, unseen-sensor, hardware-timestamp, deployable-association, "
            "or moving-object evidence. The old M5-R1 real-replay result remains blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    expected_root = (repo_root / "artifacts" / "r2" / "m5_r2").resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if output_root != expected_root:
        raise ValueError("M5-R2 evaluator outputs must stay in artifacts/r2/m5_r2")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M5-R2 evaluator report must stay under reports/r2")
    contract_path = args.contract.resolve()
    prediction_path = args.predictions.resolve()
    receipt_path = args.inference_receipt.resolve()
    for path in (contract_path, prediction_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_unopened_contract(contract, contract_path)
    tracks, predictions, inference_receipt = load_label_blind_inputs(
        contract, prediction_path, receipt_path
    )
    labels_path = Path(str(contract["files"]["evaluator_labels"]["path"]))
    models_info_path = Path(str(contract["inputs"]["models_eval_info"]["path"]))
    configs_path = Path(str(contract["inputs"]["frozen_phase1_configurations"]["path"]))
    protocol_path = Path(str(contract["inputs"]["protocol"]["path"]))
    for path in (labels_path, models_info_path, configs_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(configs_path) != str(
        contract["inputs"]["frozen_phase1_configurations"]["sha256"]
    ):
        raise RuntimeError("M5-R2 frozen temporal configurations changed")
    if sha256_file(protocol_path) != str(contract["inputs"]["protocol"]["sha256"]):
        raise RuntimeError("M5-R2 protocol changed after bundle freeze")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    open_receipt_path = output_root / "open_receipt.json"
    result_path = output_root / "result.json"
    if open_receipt_path.exists() or result_path.exists():
        raise RuntimeError("M5-R2 evaluator has already been opened or evaluated")
    if args.preflight_only:
        print(
            "M5-R2 evaluator preflight passed: "
            f"tracks={len(tracks)}, predictions={len(predictions)}, labels_opened=False"
        )
        return

    opened_utc = datetime.now(timezone.utc).isoformat()
    open_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": contract["protocol_id"],
        "stage": "M5-R2 single development-label open",
        "status": "opening_committed_before_label_read",
        "evaluation_invocation_count": 1,
        "labels_opened": True,
        "opened_utc": opened_utc,
        "claim_scope": contract["claim_scope"],
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "expected_evaluator_labels_sha256": contract["files"]["evaluator_labels"]["sha256"],
    }
    write_json_atomic(open_receipt_path, open_receipt)

    try:
        if sha256_file(labels_path) != str(contract["files"]["evaluator_labels"]["sha256"]):
            raise RuntimeError("M5-R2 evaluator-label hash changed")
        if sha256_file(models_info_path) != str(
            contract["inputs"]["models_eval_info"]["sha256"]
        ):
            raise RuntimeError("M5-R2 models_eval symmetry metadata changed")
        labels_rows = load_jsonl(labels_path)
        expected_label_count = int(contract["selection_summary"]["replay_frame_count"])
        if len(labels_rows) != expected_label_count:
            raise ValueError("M5-R2 evaluator-label count changed")
        labels: dict[tuple[str, int], dict[str, Any]] = {}
        for row in labels_rows:
            key = str(row["track_id"]), int(row["replay_frame_index"])
            if key in labels:
                raise ValueError(f"Duplicate M5-R2 evaluator frame: {key}")
            labels[key] = row
        models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
        configs = json.loads(configs_path.read_text(encoding="utf-8"))
        nearest_config = legacy.EstimatorConfig(
            **configs["selected"][BASELINE]["configuration"]
        )
        proposed_config = QuotientCVKalmanConfig(
            **configs["selected"][PROPOSED]["configuration"]
        )
        symmetries = {
            int(object_id): symmetry_from_model_info(models_info[str(object_id)])
            for object_id in contract["selection_summary"]["object_ids"]
        }
        track_results: list[dict[str, Any]] = []
        for track in tracks:
            object_id = int(track["object_id"])
            sequence, detail, canonical_to_original = build_sequence(
                track, labels, predictions, symmetries[object_id]
            )
            result = evaluate_track(
                track,
                sequence,
                detail,
                canonical_to_original,
                nearest_config,
                proposed_config,
            )
            track_results.append(result)
            print(
                f"M5-R2 track {len(track_results):02d}/{len(tracks):02d} "
                f"obj={object_id:02d} delta={result['relative_improvement']:+.1%}",
                flush=True,
            )

        baseline_macro = macro_object_mean(track_results, "baseline_loss")
        proposed_macro = macro_object_mean(track_results, "proposed_loss")
        missing_baseline_macro = macro_object_mean(
            track_results, "baseline_natural_missing_loss"
        )
        missing_proposed_macro = macro_object_mean(
            track_results, "proposed_natural_missing_loss"
        )
        relative = (baseline_macro - proposed_macro) / baseline_macro
        missing_relative = (
            missing_baseline_macro - missing_proposed_macro
        ) / missing_baseline_macro
        bootstrap_spec = protocol["evaluation"]["bootstrap"]
        bootstrap = hierarchical_paired_bootstrap(
            track_results,
            seed=int(bootstrap_spec["seed"]),
            resamples=int(bootstrap_spec["resamples"]),
        )
        thresholds = protocol["evaluation"]["development_gate"]
        nonfinite = sum(int(row["nonfinite_output_count"]) for row in track_results)
        conditions = {
            "macro_object_relative_trajectory_loss_improvement": relative
            >= float(thresholds["macro_object_relative_trajectory_loss_improvement_min"]),
            "one_sided_hierarchical_bootstrap_90pct_lower": float(
                bootstrap["one_sided_90pct_lower_relative_improvement"]
            )
            >= float(
                thresholds[
                    "one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"
                ]
            ),
            "natural_missing_frame_macro_object_relative_improvement": missing_relative
            >= float(
                thresholds[
                    "natural_missing_frame_macro_object_relative_improvement_min"
                ]
            ),
            "nonfinite_output_count": nonfinite
            <= int(thresholds["nonfinite_output_count_max"]),
        }
        gate = {
            "passed": all(conditions.values()),
            "conditions": conditions,
            "thresholds": thresholds,
            "observed": {
                "baseline_macro_object_loss": baseline_macro,
                "proposed_macro_object_loss": proposed_macro,
                "macro_object_relative_improvement": relative,
                "baseline_natural_missing_macro_object_loss": missing_baseline_macro,
                "proposed_natural_missing_macro_object_loss": missing_proposed_macro,
                "natural_missing_macro_object_relative_improvement": missing_relative,
                "one_sided_90pct_lower_relative_improvement": bootstrap[
                    "one_sided_90pct_lower_relative_improvement"
                ],
                "nonfinite_output_count": nonfinite,
            },
        }
        by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in track_results:
            by_object[int(row["object_id"])].append(row)
        per_object = []
        for object_id in sorted(by_object):
            rows = by_object[object_id]
            baseline = float(np.mean([row["baseline_loss"] for row in rows]))
            proposed = float(np.mean([row["proposed_loss"] for row in rows]))
            per_object.append(
                {
                    "object_id": object_id,
                    "track_count": len(rows),
                    "natural_missing_frame_count": sum(
                        int(row["natural_missing_frame_count"]) for row in rows
                    ),
                    "baseline_loss": baseline,
                    "proposed_loss": proposed,
                    "relative_improvement": (baseline - proposed) / baseline,
                    "baseline_natural_missing_loss": float(
                        np.mean([row["baseline_natural_missing_loss"] for row in rows])
                    ),
                    "proposed_natural_missing_loss": float(
                        np.mean([row["proposed_natural_missing_loss"] for row in rows])
                    ),
                }
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": contract["protocol_id"],
            "stage": "M5-R2 observed-dropout recorded replay",
            "status": (
                "PASS_M5_R2_REAL_REPLAY"
                if gate["passed"]
                else "FAIL_M5_R2_REAL_REPLAY"
            ),
            "claim_scope": contract["claim_scope"],
            "track_count": len(track_results),
            "object_count": len(per_object),
            "replay_frame_count": int(
                contract["selection_summary"]["replay_frame_count"]
            ),
            "natural_missing_frame_count": int(
                contract["selection_summary"]["natural_missing_frame_count"]
            ),
            "evaluation_invocation_count": 1,
            "labels_opened": True,
            "opened_utc": opened_utc,
            "development_gate": gate,
            "hierarchical_paired_bootstrap": bootstrap,
            "per_object": per_object,
            "inference_status_counts": dict(
                sorted(
                    Counter(str(row["status"]) for row in predictions.values()).items()
                )
            ),
            "predictor_failure_frame_count": sum(
                int(row["predictor_failure_frame_count"]) for row in track_results
            ),
            "gt_world_audit": {
                "max_translation_span_mm": max(
                    row["gt_world_translation_span_mm_max"] for row in track_results
                ),
                "max_quotient_rotation_span_deg": max(
                    row["gt_world_quotient_rotation_span_deg_max"]
                    for row in track_results
                ),
            },
            "frozen_methods": {
                "baseline": {
                    "method": BASELINE,
                    "configuration_id": contract["configuration_ids"][BASELINE],
                },
                "proposed": {
                    "method": PROPOSED,
                    "configuration_id": contract["configuration_ids"][PROPOSED],
                },
            },
            "claim_boundary": protocol["claim_boundary"],
            "predecessor_m5_r1_real_replay_status": "BLOCKED_REAL_REPLAY_INPUT_GATE",
            "provenance": {
                "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
                "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
                "inference_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
                "evaluator_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            },
        }
        track_results_path = output_root / "track_results.jsonl"
        write_jsonl_atomic(track_results_path, track_results)
        result["track_results"] = {
            "path": str(track_results_path),
            "sha256": sha256_file(track_results_path),
        }
        write_json_atomic(result_path, result)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(result), encoding="utf-8")
        opened_contract = {
            **contract,
            "status": "development_evaluated",
            "labels_opened": True,
            "evaluation_invocation_count": 1,
            "immutable_source_contract_sha256": sha256_file(contract_path),
            "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        }
        write_json_atomic(output_root / "opened_contract.json", opened_contract)
        open_receipt.update(
            {
                "status": "evaluation_complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
                "track_results": {
                    "path": str(track_results_path),
                    "sha256": sha256_file(track_results_path),
                },
                "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            }
        )
        write_json_atomic(open_receipt_path, open_receipt)
        print(result["status"])
        print(
            f"all={relative:+.1%}, missing={missing_relative:+.1%}, "
            f"bootstrap_lower={bootstrap['one_sided_90pct_lower_relative_improvement']:+.1%}"
        )
        print(result_path)
        print(report_path)
    except Exception as exc:
        open_receipt.update(
            {
                "status": "evaluation_failed_after_open",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            }
        )
        write_json_atomic(open_receipt_path, open_receipt)
        raise


if __name__ == "__main__":
    main()
