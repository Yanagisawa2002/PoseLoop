#!/usr/bin/env python3
"""Open the frozen M5-R3 Photoneo evaluator stream exactly once."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import evaluate_m5_r2_once as r2_evaluator
import m5_g0_core as legacy
import m5_g0_metrics as metrics
import m5_r2_core as r2_core
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from m5_r3_core import StaticQuotientMedoidConfig, run_static_quotient_medoid


SCHEMA_VERSION = 1
BASELINE = "NEAREST_REPRESENTATIVE_CT"
PROPOSED = "STATIC_QUOTIENT_ALL_HISTORY_MEDOID"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r3"
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
        default=repo_root / "reports" / "r2" / "m5_r3_photoneo_replay.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_unopened_contract(contract: Mapping[str, Any], contract_path: Path) -> None:
    if (
        contract.get("protocol_id") != "poseloop-m5-r3-v1"
        or contract.get("status") != "sealed_labels_unopened"
        or bool(contract.get("labels_opened"))
        or int(contract.get("evaluation_invocation_count", -1)) != 0
        or int(contract.get("prior_track_overlap", -1)) != 0
    ):
        raise RuntimeError("M5-R3 evaluator contract is not sealed and unopened")
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    for name, record in contract["code"].items():
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"M5-R3 frozen code changed: {name}")


def load_label_blind_inputs(
    contract: Mapping[str, Any], prediction_path: Path, receipt_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    track_path = Path(str(contract["files"]["track_manifest"]["path"]))
    manifest_path = Path(str(contract["files"]["inference_manifest"]["path"]))
    for name, path in (("track_manifest", track_path), ("inference_manifest", manifest_path)):
        if sha256_file(path) != str(contract["files"][name]["sha256"]):
            raise RuntimeError(f"M5-R3 {name} hash changed")
    tracks = load_jsonl(track_path)
    manifest = load_jsonl(manifest_path)
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if len(manifest_ids) != int(contract["selection_summary"]["inference_sample_count"]):
        raise ValueError("M5-R3 inference manifest identity count changed")
    prediction_rows = load_jsonl(prediction_path)
    if not prediction_rows or prediction_rows[0].get("record_type") != "m5_r3_inference_metadata":
        raise ValueError("M5-R3 predictions lack metadata")
    metadata = prediction_rows[0]
    if (
        metadata.get("manifest_sha256")
        != contract["files"]["inference_manifest"]["sha256"]
        or int(metadata.get("sample_count", -1)) != len(manifest_ids)
        or bool(metadata.get("evaluator_labels_path_read"))
        or bool(metadata.get("evaluator_pose_or_error_computed"))
    ):
        raise RuntimeError("M5-R3 prediction metadata failed label-blind validation")
    predictions: dict[str, dict[str, Any]] = {}
    for row in prediction_rows[1:]:
        if row.get("record_type") != "m5_r3_prediction" or bool(
            row.get("evaluator_label_read")
        ):
            raise ValueError("Invalid M5-R3 prediction row")
        sample_id = str(row["sample_id"])
        if sample_id not in manifest_ids or sample_id in predictions:
            raise ValueError(f"Invalid M5-R3 prediction identity: {sample_id}")
        predictions[sample_id] = row
    if set(predictions) != manifest_ids:
        raise ValueError("M5-R3 predictions do not cover the complete manifest")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or bool(receipt.get("labels_opened"))
        or bool(receipt.get("evaluator_pose_or_error_computed"))
        or int(receipt.get("sample_count", -1)) != len(manifest_ids)
        or receipt.get("manifest_sha256")
        != contract["files"]["inference_manifest"]["sha256"]
        or sha256_file(prediction_path) != str(receipt["predictions"]["sha256"])
    ):
        raise RuntimeError("M5-R3 inference receipt failed validation")
    track_ids = [str(row["track_id"]) for row in tracks]
    if len(track_ids) != len(set(track_ids)) or len(tracks) != int(
        contract["selection_summary"]["track_count"]
    ):
        raise ValueError("M5-R3 track manifest identity count changed")
    referenced_ids = {
        str(frame["sample_id"])
        for track in tracks
        for frame in track["frames"]
        if bool(frame["observation_available"])
    }
    if referenced_ids != manifest_ids:
        raise ValueError("M5-R3 track availability differs from the inference manifest")
    for track in tracks:
        frames = track["frames"]
        if not frames or not bool(frames[0]["observation_available"]):
            raise RuntimeError(f"M5-R3 track cannot initialize: {track['track_id']}")
        first_prediction = predictions[str(frames[0]["sample_id"])]
        if first_prediction.get("status") != "success":
            raise RuntimeError(
                f"M5-R3 first measurement inference failed: {track['track_id']}"
            )
    return tracks, predictions, receipt


def evaluate_track(
    track: Mapping[str, Any],
    sequence: legacy.CorruptedSequence,
    detail: list[dict[str, Any]],
    canonical_to_original: np.ndarray,
    nearest_config: legacy.EstimatorConfig,
    proposed_config: StaticQuotientMedoidConfig,
) -> dict[str, Any]:
    baseline_outputs = legacy.run_estimator(
        sequence, legacy.EstimatorKind.NEAREST_REPRESENTATIVE_CT, nearest_config
    )
    proposed_outputs = run_static_quotient_medoid(sequence, proposed_config)
    baseline_poses = np.stack([row.pose for row in baseline_outputs])
    proposed_poses = np.stack([row.pose for row in proposed_outputs])
    baseline_original = np.stack(
        [r2_core.from_canonical_pose(pose, canonical_to_original) for pose in baseline_poses]
    )
    proposed_original = np.stack(
        [r2_core.from_canonical_pose(pose, canonical_to_original) for pose in proposed_poses]
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
    baseline_loss = np.asarray(baseline_metrics["normalized_pose_error"], dtype=np.float64)
    proposed_loss = np.asarray(proposed_metrics["normalized_pose_error"], dtype=np.float64)
    natural_missing = np.asarray(
        [bool(row["natural_input_missing"]) for row in detail], dtype=bool
    )
    if not np.any(natural_missing):
        raise RuntimeError(f"M5-R3 selected track lacks natural missing: {track['track_id']}")
    for index, row in enumerate(detail):
        row.update(
            {
                "baseline_output_pose_m": baseline_original[index].tolist(),
                "proposed_output_pose_m": proposed_original[index].tolist(),
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
    baseline_mean = float(np.mean(baseline_loss))
    proposed_mean = float(np.mean(proposed_loss))
    baseline_missing = float(np.mean(baseline_loss[natural_missing]))
    proposed_missing = float(np.mean(proposed_loss[natural_missing]))
    return {
        "record_type": "m5_r3_track_result",
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
        "nonfinite_output_count": nonfinite,
        "gt_world_translation_span_mm_max": 1000.0 * max(value[0] for value in gt_spans),
        "gt_world_quotient_rotation_span_deg_max": max(value[1] for value in gt_spans),
        "frames": detail,
    }


def render_report(result: Mapping[str, Any]) -> str:
    gate = result["sealed_gate"]
    observed = gate["observed"]
    lines = [
        "# PoseLoop M5-R3 sealed Photoneo replay",
        "",
        f"**{result['status']}**",
        "",
        "The RealSense-developed static all-history quotient medoid was evaluated "
        "once on 23 naturally incomplete Photoneo tracks disjoint from every prior "
        "R1/R2 Photoneo group track.",
        "",
        "| Macro-object metric | Nearest | Static quotient medoid | Relative delta |",
        "| --- | ---: | ---: | ---: |",
        f"| All replay frames | {observed['baseline_macro_object_loss']:.4f} | "
        f"{observed['proposed_macro_object_loss']:.4f} | "
        f"{observed['macro_object_relative_improvement']:+.1%} |",
        f"| Natural missing frames | {observed['baseline_natural_missing_macro_object_loss']:.4f} | "
        f"{observed['proposed_natural_missing_macro_object_loss']:.4f} | "
        f"{observed['natural_missing_macro_object_relative_improvement']:+.1%} |",
        f"| One-sided 90% hierarchical-bootstrap lower | — | — | "
        f"{observed['one_sided_90pct_lower_relative_improvement']:+.1%} |",
        "",
        "| Object | Tracks | Missing | Nearest loss | Proposed loss | Delta |",
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
            f"temporal outputs: {observed['nonfinite_output_count']}.",
            "",
            "Claim boundary: a single static-world, nominal-clock, oracle-associated "
            "Photoneo replay on previously unused tracks. It is not a project-wide "
            "unseen-sensor claim, unseen-object claim, hardware-timestamp test, moving-"
            "object test, or deployable-association result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    expected_root = (repo_root / "artifacts" / "r2" / "m5_r3").resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if output_root != expected_root:
        raise ValueError("M5-R3 evaluator outputs must stay in artifacts/r2/m5_r3")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M5-R3 evaluator report must stay under reports/r2")
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
    development_path = Path(str(contract["inputs"]["development_result"]["path"]))
    for path in (labels_path, models_info_path, configs_path, protocol_path, development_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in (
        ("frozen_phase1_configurations", configs_path),
        ("protocol", protocol_path),
        ("development_result", development_path),
    ):
        if sha256_file(path) != str(contract["inputs"][name]["sha256"]):
            raise RuntimeError(f"M5-R3 frozen input changed: {name}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    open_receipt_path = output_root / "open_receipt.json"
    result_path = output_root / "result.json"
    if open_receipt_path.exists() or result_path.exists():
        raise RuntimeError("M5-R3 evaluator has already been opened or completed")

    if args.preflight_only:
        print(
            "M5-R3 evaluator preflight passed: "
            f"tracks={len(tracks)}, predictions={len(predictions)}, labels_opened=False"
        )
        return

    opened_utc = datetime.now(timezone.utc).isoformat()
    open_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": contract["protocol_id"],
        "stage": "M5-R3 single sealed-label open",
        "status": "opening_committed_before_label_read",
        "evaluation_invocation_count": 1,
        "labels_opened": True,
        "opened_utc": opened_utc,
        "claim_scope": contract["claim_scope"],
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "expected_evaluator_labels_sha256": contract["files"]["evaluator_labels"]["sha256"],
    }
    write_json_atomic(open_receipt_path, open_receipt)

    try:
        if sha256_file(labels_path) != str(contract["files"]["evaluator_labels"]["sha256"]):
            raise RuntimeError("M5-R3 evaluator-label hash changed")
        if sha256_file(models_info_path) != str(contract["inputs"]["models_eval_info"]["sha256"]):
            raise RuntimeError("M5-R3 symmetry metadata changed")
        label_rows = load_jsonl(labels_path)
        expected_labels = int(contract["selection_summary"]["replay_frame_count"])
        if len(label_rows) != expected_labels:
            raise ValueError("M5-R3 evaluator-label count changed")
        labels: dict[tuple[str, int], dict[str, Any]] = {}
        for row in label_rows:
            key = str(row["track_id"]), int(row["replay_frame_index"])
            if key in labels:
                raise ValueError(f"Duplicate M5-R3 evaluator frame: {key}")
            labels[key] = row
        models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
        configs = json.loads(configs_path.read_text(encoding="utf-8"))
        nearest_config = legacy.EstimatorConfig(
            **configs["selected"][BASELINE]["configuration"]
        )
        proposed_config = StaticQuotientMedoidConfig(history_limit=None)
        symmetries = {
            int(object_id): r2_evaluator.symmetry_from_model_info(
                models_info[str(object_id)]
            )
            for object_id in contract["selection_summary"]["object_ids"]
        }
        track_results: list[dict[str, Any]] = []
        for track in tracks:
            object_id = int(track["object_id"])
            sequence, detail, canonical_to_original = r2_evaluator.build_sequence(
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
                f"M5-R3 track {len(track_results):02d}/{len(tracks):02d} "
                f"obj={object_id:02d} delta={result['relative_improvement']:+.1%}",
                flush=True,
            )

        baseline_macro = r2_evaluator.macro_object_mean(track_results, "baseline_loss")
        proposed_macro = r2_evaluator.macro_object_mean(track_results, "proposed_loss")
        missing_baseline = r2_evaluator.macro_object_mean(
            track_results, "baseline_natural_missing_loss"
        )
        missing_proposed = r2_evaluator.macro_object_mean(
            track_results, "proposed_natural_missing_loss"
        )
        relative = (baseline_macro - proposed_macro) / baseline_macro
        missing_relative = (missing_baseline - missing_proposed) / missing_baseline
        bootstrap_spec = protocol["evaluation"]["bootstrap"]
        bootstrap = r2_evaluator.hierarchical_paired_bootstrap(
            track_results,
            seed=int(bootstrap_spec["seed"]),
            resamples=int(bootstrap_spec["resamples"]),
        )
        thresholds = protocol["evaluation"]["sealed_gate"]
        nonfinite = sum(int(row["nonfinite_output_count"]) for row in track_results)
        conditions = {
            "macro_object_relative_trajectory_loss_improvement": relative
            >= float(thresholds["macro_object_relative_trajectory_loss_improvement_min"]),
            "one_sided_hierarchical_bootstrap_90pct_lower": float(
                bootstrap["one_sided_90pct_lower_relative_improvement"]
            )
            >= float(
                thresholds["one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"]
            ),
            "natural_missing_frame_macro_object_relative_improvement": missing_relative
            >= float(thresholds["natural_missing_frame_macro_object_relative_improvement_min"]),
            "nonfinite_output_count": nonfinite <= int(thresholds["nonfinite_output_count_max"]),
        }
        gate = {
            "passed": all(conditions.values()),
            "conditions": conditions,
            "thresholds": thresholds,
            "observed": {
                "baseline_macro_object_loss": baseline_macro,
                "proposed_macro_object_loss": proposed_macro,
                "macro_object_relative_improvement": relative,
                "baseline_natural_missing_macro_object_loss": missing_baseline,
                "proposed_natural_missing_macro_object_loss": missing_proposed,
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
            "stage": "M5-R3 track-disjoint Photoneo sealed replay",
            "status": "PASS_M5_R3_SEALED" if gate["passed"] else "FAIL_M5_R3_SEALED",
            "claim_scope": contract["claim_scope"],
            "track_count": len(track_results),
            "object_count": len(per_object),
            "replay_frame_count": int(contract["selection_summary"]["replay_frame_count"]),
            "natural_missing_frame_count": int(
                contract["selection_summary"]["natural_missing_frame_count"]
            ),
            "prior_photoneo_track_count": int(contract["prior_photoneo_track_count"]),
            "prior_track_overlap": 0,
            "evaluation_invocation_count": 1,
            "labels_opened": True,
            "opened_utc": opened_utc,
            "sealed_gate": gate,
            "hierarchical_paired_bootstrap": bootstrap,
            "per_object": per_object,
            "inference_status_counts": dict(
                sorted(Counter(str(row["status"]) for row in predictions.values()).items())
            ),
            "predictor_failure_frame_count": sum(
                int(row["predictor_failure_frame_count"]) for row in track_results
            ),
            "gt_world_audit": {
                "max_translation_span_mm": max(
                    row["gt_world_translation_span_mm_max"] for row in track_results
                ),
                "max_quotient_rotation_span_deg": max(
                    row["gt_world_quotient_rotation_span_deg_max"] for row in track_results
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
            "predecessor_m5_r2_status": "FAIL_M5_R2_REAL_REPLAY",
            "provenance": {
                "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
                "predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
                "inference_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
                "evaluator_labels": {
                    "path": str(labels_path),
                    "sha256": sha256_file(labels_path),
                },
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
            "status": "sealed_evaluated",
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
