#!/usr/bin/env python3
"""Perform the single fixed-candidate M5-R6A YCB-V sealed evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import evaluate_m5_r2_once as r2_evaluator
import m5_g0_core as legacy
from evaluate_m5_r5_development import (
    _build_track_sequence,
    _candidate_summary,
    _evaluate_track,
    _load_and_validate,
    spearman_correlation,
)
from m1_common import sha256_file, write_json_atomic, write_jsonl_atomic
from m5_r6_core import (
    MeasurementFirstConfig,
    MeasurementFirstObservation,
    run_measurement_first_confidence_gate,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "poseloop-m5-r6a-ycbv-sealed-v1"
STAGE_ID = "M5-R6A-S1"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r6a_ycbv_sealed"
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
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "r2"
        / "m5_r6a_ycbv_sealed.md",
    )
    return parser.parse_args()


def render_report(result: Mapping[str, Any]) -> str:
    observed = result["sealed_gate"]["observed"]
    thresholds = result["sealed_gate"]["thresholds"]
    lines = [
        "# PoseLoop M5-R6A-S1 YCB-V sealed validation",
        "",
        f"Status: **{result['status']}**",
        "",
        "The candidate was fixed to `measurement_first` before YCB-V download. No sealed candidate selection or threshold adjustment was performed.",
        "",
        "| Sealed condition | Observed | Required |",
        "|---|---:|---:|",
        f"| Macro-object all-frame improvement | {observed['macro_object_relative_improvement']:+.2%} | >= {thresholds['macro_object_relative_improvement_min']:.0%} |",
        f"| Natural-missing improvement | {observed['natural_missing_frame_relative_improvement']:+.2%} | >= {thresholds['natural_missing_frame_relative_improvement_min']:.0%} |",
        f"| Bootstrap 10th percentile | {observed['one_sided_90pct_lower_relative_improvement']:+.2%} | >= {thresholds['one_sided_hierarchical_bootstrap_90pct_lower_improvement_min']:.0%} |",
        f"| Missing uncertainty/error Spearman | {observed['missing_frame_uncertainty_error_spearman']:+.3f} | >= {thresholds['missing_frame_uncertainty_error_spearman_min']:.2f} |",
        f"| Represented objects/tracks | {observed['represented_object_count']} / {observed['track_count']} | >= {thresholds['represented_object_count_min']} / {thresholds['track_count_min']} |",
        f"| Missing-represented objects | {observed['missing_represented_object_count']} | >= {thresholds['missing_represented_object_count_min']} |",
        f"| Natural missing frames | {observed['natural_missing_frame_count']} | >= {thresholds['natural_missing_frame_count_min']} |",
        f"| Missing-anchor mismatches | {observed['missing_anchor_mismatch_count']} | <= {thresholds['missing_anchor_mismatch_count_max']} |",
        f"| Unsafe forced reacquisitions | {observed['unsafe_forced_reacquisition_count']} | <= {thresholds['unsafe_forced_reacquisition_count_max']} |",
        f"| Nonfinite outputs | {observed['nonfinite_output_count']} | <= {thresholds['nonfinite_output_count_max']} |",
        "",
        f"Corrections applied on available frames: {result['correction_applied_frame_count']}.",
        "",
        "This is the one-open sealed result. It is not eligible for YCB-V-driven retuning or a second evaluation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    predictions_path = args.predictions.resolve()
    receipt_path = args.inference_receipt.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    result_path = output_root / "sealed_result.json"
    track_results_path = output_root / "sealed_track_results.jsonl"
    if result_path.exists() or track_results_path.exists():
        raise FileExistsError("M5-R6A YCB-V sealed result already exists")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("stage_id") != STAGE_ID
        or contract.get("status") != "sealed_labels_opened_bundle_ready"
        or not bool(contract.get("sealed_archive_or_label_read"))
        or not bool(contract.get("sealed_labels_opened"))
        or int(contract.get("evaluation_invocation_count", -1)) != 0
        or contract.get("candidate_id") != "measurement_first"
    ):
        raise RuntimeError("Invalid M5-R6A YCB-V sealed contract")
    code_checks = {
        "evaluator": Path(__file__).resolve(),
        "development_evaluation_core": Path(__file__).resolve().parent
        / "evaluate_m5_r5_development.py",
        "core": Path(__file__).resolve().parent / "m5_r6_core.py",
    }
    for name, path in code_checks.items():
        if sha256_file(path) != str(contract["code"][name]["sha256"]):
            raise RuntimeError(f"M5-R6A sealed code changed: {name}")
    protocol_path = Path(str(contract["protocol"]["path"]))
    if sha256_file(protocol_path) != str(contract["protocol"]["sha256"]):
        raise RuntimeError("M5-R6A sealed protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development_path = Path(str(contract["development_result"]["path"]))
    if sha256_file(development_path) != str(contract["development_result"]["sha256"]):
        raise RuntimeError("M5-R6A development result changed before sealed evaluation")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if (
        development.get("status") != "PASS_M5_R6A_DEVELOPMENT"
        or development.get("modal_candidate_id") != "measurement_first"
        or development.get("fold_winner_counts") != {"measurement_first": 14}
    ):
        raise RuntimeError("M5-R6A development candidate freeze is invalid")

    tracks, labels, predictions, inference_receipt = _load_and_validate(
        contract, predictions_path, receipt_path
    )
    if (
        not bool(inference_receipt.get("sealed_source_labels_opened"))
        or not bool(inference_receipt.get("sealed_archive_or_label_read"))
        or bool(inference_receipt.get("evaluator_labels_path_read"))
        or bool(inference_receipt.get("evaluator_pose_or_error_computed"))
    ):
        raise RuntimeError("M5-R6A sealed inference receipt boundary is invalid")
    models_info_path = Path(str(contract["dataset"]["models_info"]["path"]))
    if sha256_file(models_info_path) != str(contract["dataset"]["models_info"]["sha256"]):
        raise RuntimeError("YCB-V models_info hash mismatch")
    models_info = json.loads(models_info_path.read_text(encoding="utf-8"))
    config_path = Path(str(protocol["methods"]["baseline"]["configuration_source"]))
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    frozen_configs = json.loads(config_path.read_text(encoding="utf-8"))
    nearest_config = legacy.EstimatorConfig(
        **frozen_configs["selected"]["NEAREST_REPRESENTATIVE_CT"]["configuration"]
    )
    proposed_config = MeasurementFirstConfig(
        **protocol["methods"]["proposed"]["configuration"]
    )

    track_results: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks, start=1):
        object_id = int(track["object_id"])
        symmetry = r2_evaluator.symmetry_from_model_info(models_info[str(object_id)])
        sequence, detail = _build_track_sequence(track, labels, predictions, symmetry)
        track_results.append(
            _evaluate_track(
                track,
                sequence,
                detail,
                nearest_config,
                proposed_config,
                MeasurementFirstObservation,
                run_measurement_first_confidence_gate,
                "measurement_first",
            )
        )
        print(
            f"{STAGE_ID} track {track_index:02d}/{len(tracks):02d}: "
            f"{track['track_id']}"
        )
    summary = _candidate_summary(track_results, "measurement_first")
    bootstrap_spec = protocol["evaluation"]["bootstrap"]
    bootstrap = r2_evaluator.hierarchical_paired_bootstrap(
        track_results,
        seed=int(bootstrap_spec["seed"]),
        resamples=int(bootstrap_spec["resamples"]),
    )
    missing_uncertainty: list[float] = []
    missing_error: list[float] = []
    for track_result in track_results:
        for frame in track_result["frames"]:
            if bool(frame["natural_input_missing"]):
                missing_uncertainty.append(float(frame["proposed_uncertainty"]))
                missing_error.append(float(frame["proposed_normalized_pose_error"]))
    confidence_spearman = spearman_correlation(missing_uncertainty, missing_error)
    gate = protocol["evaluation"]["sealed_gate"]
    observed = {
        "represented_object_count": len({int(row["object_id"]) for row in track_results}),
        "track_count": len(track_results),
        "missing_represented_object_count": summary["missing_represented_object_count"],
        "natural_missing_frame_count": len(missing_error),
        "macro_object_relative_improvement": summary[
            "macro_object_relative_improvement"
        ],
        "natural_missing_frame_relative_improvement": summary[
            "natural_missing_macro_object_relative_improvement"
        ],
        "one_sided_90pct_lower_relative_improvement": bootstrap[
            "one_sided_90pct_lower_relative_improvement"
        ],
        "missing_frame_uncertainty_error_spearman": confidence_spearman,
        "nonfinite_output_count": summary["nonfinite_output_count"],
        "missing_anchor_mismatch_count": summary["missing_anchor_mismatch_count"],
        "unsafe_forced_reacquisition_count": summary[
            "unsafe_forced_reacquisition_count"
        ],
    }
    conditions = {
        "represented_object_count": observed["represented_object_count"]
        >= int(gate["represented_object_count_min"]),
        "track_count": observed["track_count"] >= int(gate["track_count_min"]),
        "missing_represented_object_count": observed["missing_represented_object_count"]
        >= int(gate["missing_represented_object_count_min"]),
        "natural_missing_frame_count": observed["natural_missing_frame_count"]
        >= int(gate["natural_missing_frame_count_min"]),
        "macro_object_relative_improvement": observed[
            "macro_object_relative_improvement"
        ]
        >= float(gate["macro_object_relative_improvement_min"]),
        "natural_missing_frame_relative_improvement": observed[
            "natural_missing_frame_relative_improvement"
        ]
        >= float(gate["natural_missing_frame_relative_improvement_min"]),
        "one_sided_hierarchical_bootstrap_90pct_lower": observed[
            "one_sided_90pct_lower_relative_improvement"
        ]
        >= float(
            gate["one_sided_hierarchical_bootstrap_90pct_lower_improvement_min"]
        ),
        "missing_frame_uncertainty_error_spearman": math.isfinite(
            confidence_spearman
        )
        and confidence_spearman
        >= float(gate["missing_frame_uncertainty_error_spearman_min"]),
        "nonfinite_output_count": observed["nonfinite_output_count"]
        <= int(gate["nonfinite_output_count_max"]),
        "missing_anchor_mismatch_count": observed["missing_anchor_mismatch_count"]
        <= int(gate["missing_anchor_mismatch_count_max"]),
        "unsafe_forced_reacquisition_count": observed[
            "unsafe_forced_reacquisition_count"
        ]
        <= int(gate["unsafe_forced_reacquisition_count_max"]),
    }
    passed = all(conditions.values())
    per_object: list[dict[str, Any]] = []
    for object_id in sorted({int(row["object_id"]) for row in track_results}):
        object_rows = [row for row in track_results if int(row["object_id"]) == object_id]
        object_summary = _candidate_summary(object_rows, "measurement_first")
        per_object.append(
            {
                "object_id": object_id,
                "track_count": len(object_rows),
                "baseline_loss": object_summary["baseline_macro_object_loss"],
                "proposed_loss": object_summary["proposed_macro_object_loss"],
                "relative_improvement": object_summary[
                    "macro_object_relative_improvement"
                ],
                "natural_missing_relative_improvement": object_summary[
                    "natural_missing_macro_object_relative_improvement"
                ],
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "stage": "M5-R6A fixed measurement-first YCB-V one-open sealed validation",
        "status": (
            "PASS_M5_R6A_YCBV_SEALED" if passed else "FAIL_M5_R6A_YCBV_SEALED"
        ),
        "sealed_gate": {
            "passed": passed,
            "conditions": conditions,
            "observed": observed,
            "thresholds": gate,
        },
        "candidate_id": "measurement_first",
        "candidate_configuration": protocol["methods"]["proposed"]["configuration"],
        "candidate_reselected_on_ycbv": False,
        "summary": summary,
        "hierarchical_paired_bootstrap": bootstrap,
        "per_object": per_object,
        "correction_applied_frame_count": sum(
            int(row["correction_applied_frame_count"]) for row in track_results
        ),
        "inference_status_counts": inference_receipt["status_counts"],
        "raw_foundationpose_scores_used_for_filtering_or_confidence": False,
        "persistent_pose_owner": "NEAREST_REPRESENTATIVE_CT nearest_smooth",
        "sealed_archive_or_label_read": True,
        "sealed_evaluation_invocation_count": 1,
        "reopen_allowed": False,
        "claim_scope": (
            protocol["claim_boundary"]["allowed_if_passed"] if passed else None
        ),
    }
    write_jsonl_atomic(track_results_path, track_results)
    result["track_results"] = {
        "path": str(track_results_path),
        "sha256": sha256_file(track_results_path),
    }
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result["sealed_gate"], indent=2, sort_keys=True))
    print(result_path)
    print(report_path)


if __name__ == "__main__":
    main()
