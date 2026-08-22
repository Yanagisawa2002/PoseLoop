#!/usr/bin/env python3
"""Open and evaluate the sealed Photoneo labels exactly once."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import evaluate_m2
import fit_m4_voi
import run_m6_r1
from build_r1_sealed_policy_inputs import (
    EXPECTED_PREDICTIONS,
    EXPECTED_TARGETS,
    _contract_input_paths,
    load_predictions,
    validate_groups,
    validate_unopened_contract,
)
from evaluate_m1 import load_official_models
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
M3_METHOD = "symmetry_aware_medoid"
VIEW_BUDGETS = (1, 3, 5)
SLOTS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    sealed = repo_root / "artifacts" / "r1" / "sealed_photoneo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=sealed / "sealed_contract.json")
    parser.add_argument("--predictions", type=Path, default=sealed / "predictions.jsonl")
    parser.add_argument(
        "--decisions", type=Path, default=sealed / "label_blind_decisions.jsonl"
    )
    parser.add_argument(
        "--decision-receipt",
        type=Path,
        default=sealed / "label_blind_decision_receipt.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument("--output-root", type=Path, default=sealed)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "sealed_photoneo_validation.md",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen label-blind inputs without opening evaluator files.",
    )
    return parser.parse_args()


def macro_object(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite macro field: {field}")
        grouped[int(row["object_id"])].append(value)
    if not grouped:
        raise ValueError("Cannot compute an empty macro-object metric")
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def combined_utility(row: Mapping[str, Any]) -> float:
    return 0.5 * (float(row["sample_ar_mssd"]) + float(row["sample_ar_mspd"]))


def _method_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = str(row["group_id"]), int(row["requested_view_budget"]), str(row["method"])
        if key in output:
            raise ValueError(f"Duplicate sealed method result: {key}")
        output[key] = row
    return output


def _decision_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("record_type") != "r1_sealed_final_decision":
            raise ValueError("Unexpected label-blind decision record")
        group_id = str(row["group_id"])
        if group_id in output:
            raise ValueError(f"Duplicate sealed decision: {group_id}")
        if bool(row.get("evaluator_label_read")):
            raise RuntimeError("Label-blind decision declares evaluator access")
        output[group_id] = row
    return output


def render_report(result: Mapping[str, Any]) -> str:
    m3 = result["stages"]["M3-R1"]
    m4 = result["stages"]["M4-R1"]
    m6 = result["stages"]["M6-R1"]
    return "\n".join(
        [
            "# PoseLoop R1 single sealed Photoneo validation",
            "",
            f"**{result['status']}**",
            "",
            "The evaluator was opened once after all FoundationPose predictions, M3/M4 "
            "decisions, final pose selections, and M6 risks were frozen and hashed.",
            "",
            "| Stage | Frozen method | Baseline | Delta | Gate |",
            "| --- | ---: | ---: | ---: | :---: |",
            f"| M3 macro combined | {100*m3['active_macro_combined']:.2f}% | "
            f"{100*m3['matched_random_macro_combined']:.2f}% | "
            f"{m3['gain_over_matched_random_pp']:+.2f} pp | {'pass' if m3['passed'] else 'fail'} |",
            f"| M4 continue macro combined vs fixed | {100*m4['active_macro_combined']:.2f}% | "
            f"{100*m4['fixed_slot_2_macro_combined']:.2f}% | "
            f"{m4['gain_over_fixed_slot_2_pp']:+.2f} pp | {'pass' if m4['passed'] else 'fail'} |",
            f"| M4 continue macro combined vs random | {100*m4['active_macro_combined']:.2f}% | "
            f"{100*m4['uniform_random_macro_combined']:.2f}% | "
            f"{m4['gain_over_uniform_random_pp']:+.2f} pp | {'pass' if m4['passed'] else 'fail'} |",
            f"| M6 AUROC | {m6['learned']['auroc']:.4f} | {m6['raw_score_baseline']['auroc']:.4f} | "
            f"{m6['auroc_gain_over_raw']:+.4f} | {'pass' if m6['passed'] else 'fail'} |",
            f"| M6 AURC | {m6['learned']['aurc']:.4f} | {m6['raw_score_baseline']['aurc']:.4f} | "
            f"{m6['aurc_relative_reduction_vs_raw']:+.1%} relative | {'pass' if m6['passed'] else 'fail'} |",
            "",
            f"M3 mean acquired views before M4 collapse: {m3['mean_views']:.3f}. "
            f"M4 continue targets: {m4['continue_target_count']}. "
            f"Final failure labels: {m6['failure_count']}/{result['target_count']}.",
            "",
            "Claim boundary: this validates sensor-and-observation generalization to "
            "Photoneo. It does not establish unseen physical-instance generalization.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sealed_root = (repo_root / "artifacts" / "r1" / "sealed_photoneo").resolve()
    output_root = args.output_root.resolve()
    if output_root != sealed_root:
        raise ValueError("Sealed evaluation outputs must stay in the sealed root")
    report_path = args.report.resolve()
    if not report_path.is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("Sealed report must stay under reports/r1")
    contract_path = args.contract.resolve()
    prediction_path = args.predictions.resolve()
    decision_path = args.decisions.resolve()
    decision_receipt_path = args.decision_receipt.resolve()
    dataset_root = args.dataset_root.resolve()
    for path in (contract_path, prediction_path, decision_path, decision_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_inputs = _contract_input_paths(repo_root)
    validate_unopened_contract(contract, contract_inputs)
    manifest_rows = load_jsonl(contract_inputs["inference_manifest"])
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if len(manifest_ids) != EXPECTED_PREDICTIONS:
        raise ValueError("Unexpected sealed inference identity count")
    _, predictions = load_predictions(prediction_path, manifest_ids)
    target_input_rows = load_jsonl(contract_inputs["target_manifest"])
    target_inputs = {str(row["sample_id"]): row for row in target_input_rows}
    groups_inference = load_jsonl(contract_inputs["groups_inference"])
    validate_groups(groups_inference, target_inputs, set(predictions))
    decisions = _decision_index(load_jsonl(decision_path))
    if len(decisions) != EXPECTED_TARGETS or set(decisions) != {
        str(group["group_id"]) for group in groups_inference
    }:
        raise ValueError("Label-blind decisions do not cover all sealed targets")
    decision_receipt = json.loads(decision_receipt_path.read_text(encoding="utf-8"))
    if (
        decision_receipt.get("status") != "complete_ready_for_single_open"
        or bool(decision_receipt.get("evaluator_label_files_opened"))
        or sha256_file(decision_path)
        != str(decision_receipt["outputs"]["label_blind_decisions"]["sha256"])
    ):
        raise RuntimeError("Label-blind decision receipt failed validation")

    evaluator_labels_path = sealed_root / "evaluator_labels.jsonl"
    groups_evaluator_path = sealed_root / "groups_evaluator.jsonl"
    if not evaluator_labels_path.is_file() or not groups_evaluator_path.is_file():
        raise FileNotFoundError("Sealed evaluator package is missing")
    open_receipt_path = sealed_root / "sealed_open_receipt.json"
    result_path = sealed_root / "sealed_result.json"
    if open_receipt_path.exists() or result_path.exists():
        raise RuntimeError("Sealed evaluator has already been opened or evaluated")
    if args.preflight_only:
        print(
            "sealed evaluator preflight passed: "
            f"targets={len(decisions)}, predictions={len(predictions)}, labels_opened=False"
        )
        return

    opened_utc = datetime.now(timezone.utc).isoformat()
    open_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1 single sealed Photoneo validation",
        "status": "opening_committed_before_label_read",
        "evaluation_invocation_count": 1,
        "labels_opened": True,
        "opened_utc": opened_utc,
        "claim_scope": contract["claim_scope"],
        "sealed_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "frozen_decisions": {"path": str(decision_path), "sha256": sha256_file(decision_path)},
        "frozen_decision_receipt": {
            "path": str(decision_receipt_path),
            "sha256": sha256_file(decision_receipt_path),
        },
        "expected_evaluator_hashes": {
            "evaluator_labels": contract["files"]["evaluator_labels"]["sha256"],
            "groups_evaluator": contract["files"]["groups_evaluator"]["sha256"],
        },
    }
    write_json_atomic(open_receipt_path, open_receipt)

    try:
        # This is the sole point at which the evaluator package is opened.
        if sha256_file(evaluator_labels_path) != str(
            contract["files"]["evaluator_labels"]["sha256"]
        ):
            raise RuntimeError("Sealed evaluator-label hash changed")
        if sha256_file(groups_evaluator_path) != str(
            contract["files"]["groups_evaluator"]["sha256"]
        ):
            raise RuntimeError("Sealed evaluator-group hash changed")
        evaluator_labels = load_jsonl(evaluator_labels_path)
        groups_evaluator = load_jsonl(groups_evaluator_path)
        label_index = {
            str(row["sample_id"]): row for row in evaluator_labels
        }
        if len(label_index) != EXPECTED_PREDICTIONS or set(label_index) != manifest_ids:
            raise ValueError("Sealed evaluator-label identities differ from inference")
        if len(groups_evaluator) != EXPECTED_TARGETS:
            raise ValueError("Sealed evaluator-group count changed")
        evaluator_group_index = {
            str(row["group_id"]): row for row in groups_evaluator
        }
        if set(evaluator_group_index) != set(decisions):
            raise ValueError("Evaluator-group identities differ from frozen decisions")

        evaluator_manifest = {
            sample_id: {**target_inputs[sample_id], **label_index[sample_id]}
            for sample_id in target_inputs
        }
        model_params, model_info = load_official_models(dataset_root)
        prepared, transform_audit = evaluate_m2.prepare_groups(
            groups_evaluator,
            evaluator_manifest,
            predictions,
            predictions,
            model_params,
            model_info,
            evaluate_m2.GT_TRANSFORM_MAX_NORMALIZED_MSSD,
            evaluate_m2.GT_TRANSFORM_MAX_MSPD_PX,
        )
        prepared_index = {
            str(row["group"]["group_id"]): row for row in prepared
        }
        method_rows = evaluate_m2.evaluate_methods(prepared)
        methods = _method_index(method_rows)

        m3_active_rows = []
        outcome_by_group_budget: dict[tuple[str, int], float] = {}
        budgets = []
        for group_id in sorted(decisions):
            decision = decisions[group_id]
            budget = int(decision["m3_primary_budget_before_m4_collapse"])
            budgets.append(budget)
            for candidate_budget in VIEW_BUDGETS:
                outcome_by_group_budget[(group_id, candidate_budget)] = combined_utility(
                    methods[(group_id, candidate_budget, M3_METHOD)]
                )
            m3_active_rows.append(
                {
                    "object_id": int(decision["object_id"]),
                    "value": outcome_by_group_budget[(group_id, budget)],
                }
            )
        budget_counts = Counter(budgets)
        probabilities = {
            budget: budget_counts[budget] / len(budgets) for budget in VIEW_BUDGETS
        }
        m3_random_rows = []
        for group_id in sorted(decisions):
            m3_random_rows.append(
                {
                    "object_id": int(decisions[group_id]["object_id"]),
                    "value": sum(
                        probabilities[budget] * outcome_by_group_budget[(group_id, budget)]
                        for budget in VIEW_BUDGETS
                    ),
                }
            )
        m3_active = macro_object(m3_active_rows, "value")
        m3_random = macro_object(m3_random_rows, "value")
        mean_views = float(np.mean(budgets))
        m3_gain_pp = 100.0 * (m3_active - m3_random)
        gates = contract["single_open_gates"]
        m3_passed = bool(
            m3_gain_pp
            >= float(gates["M3_gain_over_exact_budget_matched_random_pp_min"])
            and mean_views <= float(gates["M3_mean_views_max"])
        )

        pair_outcomes: dict[tuple[str, int], dict[str, Any]] = {}
        continue_ids = [
            group_id
            for group_id, decision in decisions.items()
            if int(decision["m3_primary_budget_before_m4_collapse"]) > 1
        ]
        for group_id in sorted(continue_ids):
            for slot in SLOTS:
                pair_outcomes[(group_id, slot)] = fit_m4_voi._pair_outcome(
                    prepared_index[group_id], slot
                )
        if not continue_ids:
            raise RuntimeError("M4 sealed evaluation has no M3-continue targets")
        active_pair_rows = []
        fixed_pair_rows = []
        random_pair_rows = []
        for group_id in sorted(continue_ids):
            decision = decisions[group_id]
            slot = int(decision["final"]["m4_selected_slot"])
            active_pair_rows.append(
                {
                    "object_id": int(decision["object_id"]),
                    "value": pair_outcomes[(group_id, slot)]["actual_utility"],
                }
            )
            fixed_pair_rows.append(
                {
                    "object_id": int(decision["object_id"]),
                    "value": pair_outcomes[(group_id, 2)]["actual_utility"],
                }
            )
            random_pair_rows.append(
                {
                    "object_id": int(decision["object_id"]),
                    "value": float(
                        np.mean(
                            [pair_outcomes[(group_id, slot)]["actual_utility"] for slot in SLOTS]
                        )
                    ),
                }
            )
        m4_active = macro_object(active_pair_rows, "value")
        m4_fixed = macro_object(fixed_pair_rows, "value")
        m4_random = macro_object(random_pair_rows, "value")
        m4_gain_fixed_pp = 100.0 * (m4_active - m4_fixed)
        m4_gain_random_pp = 100.0 * (m4_active - m4_random)
        m4_passed = bool(
            m4_gain_fixed_pp >= float(gates["M4_gain_over_fixed_slot_pp_min"])
            and m4_gain_random_pp >= float(gates["M4_gain_over_uniform_random_pp_min"])
        )

        evaluated_final_rows = []
        y_failure = []
        learned_risk = []
        raw_risk = []
        ids = []
        for group_id in sorted(decisions):
            decision = decisions[group_id]
            prepared_group = prepared_index[group_id]
            final = decision["final"]
            selected_id = str(final["selected_sample_id"])
            selected = next(
                view
                for view in prepared_group["views"]
                if str(view["view"]["sample_id"]) == selected_id
            )
            if final["route"] == "m3_stop_target_only":
                acquired = [prepared_group["views"][0]]
            else:
                slot = int(final["m4_selected_slot"])
                acquired = [prepared_group["views"][0], prepared_group["views"][slot]]
            stored_pose = np.asarray(
                final["predicted_model_to_target_camera_pose_m"], dtype=np.float64
            )
            if not np.allclose(
                stored_pose,
                np.asarray(selected["transformed_pose_m"], dtype=np.float64),
                rtol=0.0,
                atol=1e-8,
            ):
                raise RuntimeError(f"Frozen final pose changed at evaluation: {group_id}")
            outcome = evaluate_m2.result_from_selected(
                prepared_group,
                acquired,
                selected,
                "r1_sealed_final",
                int(final["acquired_view_count"]),
            )
            failure = int(not bool(outcome["diagnostic_success"]["joint"]))
            y_failure.append(failure)
            learned_risk.append(float(decision["m6_risk"]["failure_probability"]))
            raw_risk.append(float(decision["m6_risk"]["raw_score_risk_baseline"]))
            ids.append(group_id)
            evaluated_final_rows.append(
                {
                    "record_type": "r1_sealed_evaluated_outcome",
                    "schema_version": SCHEMA_VERSION,
                    "group_id": group_id,
                    "object_id": int(decision["object_id"]),
                    "physical_instance_id": str(label_index[selected_id]["physical_instance_id"]),
                    "selected_sample_id": selected_id,
                    "m3_budget": int(decision["m3_primary_budget_before_m4_collapse"]),
                    "m4_selected_slot": final["m4_selected_slot"],
                    "sample_ar_mssd": float(outcome["sample_ar_mssd"]),
                    "sample_ar_mspd": float(outcome["sample_ar_mspd"]),
                    "combined_utility": combined_utility(outcome),
                    "joint_success": bool(outcome["diagnostic_success"]["joint"]),
                    "normalized_mssd": outcome["normalized_mssd"],
                    "mspd_px": outcome["mspd_px"],
                    "learned_failure_risk": learned_risk[-1],
                    "raw_score_risk_baseline": raw_risk[-1],
                }
            )
        y = np.asarray(y_failure, dtype=np.int64)
        learned = np.asarray(learned_risk, dtype=np.float64)
        raw = np.asarray(raw_risk, dtype=np.float64)
        learned_metrics = run_m6_r1._metric_summary(y, learned, learned, ids)
        raw_metrics = run_m6_r1._metric_summary(y, raw, None, ids)
        auroc_gain = float(learned_metrics["auroc"] - raw_metrics["auroc"])
        aurc_reduction = float(
            (raw_metrics["aurc"] - learned_metrics["aurc"])
            / max(raw_metrics["aurc"], 1e-12)
        )
        m6_passed = bool(
            auroc_gain >= float(gates["M6_auroc_gain_over_raw_score_min"])
            and aurc_reduction >= float(gates["M6_aurc_relative_reduction_min"])
        )

        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": contract["protocol_id"],
            "stage": "R1 single sealed Photoneo validation",
            "status": (
                "PASS_SEALED_VALIDATION" if m3_passed and m4_passed and m6_passed
                else "FAIL_SEALED_VALIDATION"
            ),
            "target_count": EXPECTED_TARGETS,
            "sensor_modality": contract["sensor_modality"],
            "claim_scope": contract["claim_scope"],
            "physical_instance_overlap_not_excluded": True,
            "evaluation_invocation_count": 1,
            "labels_opened": True,
            "opened_utc": opened_utc,
            "stages": {
                "M3-R1": {
                    "passed": m3_passed,
                    "active_macro_combined": m3_active,
                    "matched_random_macro_combined": m3_random,
                    "gain_over_matched_random_pp": m3_gain_pp,
                    "mean_views": mean_views,
                    "budget_counts": {str(key): int(value) for key, value in sorted(budget_counts.items())},
                    "gates": {
                        "gain_pp_min": gates["M3_gain_over_exact_budget_matched_random_pp_min"],
                        "mean_views_max": gates["M3_mean_views_max"],
                    },
                },
                "M4-R1": {
                    "passed": m4_passed,
                    "continue_target_count": len(continue_ids),
                    "active_macro_combined": m4_active,
                    "fixed_slot_2_macro_combined": m4_fixed,
                    "uniform_random_macro_combined": m4_random,
                    "gain_over_fixed_slot_2_pp": m4_gain_fixed_pp,
                    "gain_over_uniform_random_pp": m4_gain_random_pp,
                    "gates": {
                        "gain_fixed_pp_min": gates["M4_gain_over_fixed_slot_pp_min"],
                        "gain_random_pp_min": gates["M4_gain_over_uniform_random_pp_min"],
                    },
                },
                "M6-R1": {
                    "passed": m6_passed,
                    "failure_count": int(np.sum(y)),
                    "learned": learned_metrics,
                    "raw_score_baseline": raw_metrics,
                    "auroc_gain_over_raw": auroc_gain,
                    "aurc_relative_reduction_vs_raw": aurc_reduction,
                    "gates": {
                        "auroc_gain_min": gates["M6_auroc_gain_over_raw_score_min"],
                        "aurc_relative_reduction_min": gates["M6_aurc_relative_reduction_min"],
                    },
                    "pose_reselection_after_risk": False,
                },
            },
            "gt_transform_audit": transform_audit,
            "provenance": {
                "sealed_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
                "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
                "frozen_decisions": {"path": str(decision_path), "sha256": sha256_file(decision_path)},
                "evaluator_labels": {"path": str(evaluator_labels_path), "sha256": sha256_file(evaluator_labels_path)},
                "groups_evaluator": {"path": str(groups_evaluator_path), "sha256": sha256_file(groups_evaluator_path)},
            },
        }
        outcome_path = output_root / "sealed_evaluated_outcomes.jsonl"
        write_jsonl_atomic(outcome_path, evaluated_final_rows)
        result["evaluated_outcomes"] = {
            "path": str(outcome_path),
            "sha256": sha256_file(outcome_path),
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
                "evaluated_outcomes": {
                    "path": str(outcome_path),
                    "sha256": sha256_file(outcome_path),
                },
                "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            }
        )
        write_json_atomic(open_receipt_path, open_receipt)
        print(result["status"])
        print(
            f"M3 gain={m3_gain_pp:+.3f}pp mean_views={mean_views:.3f}; "
            f"M4 gains=fixed {m4_gain_fixed_pp:+.3f}pp/random {m4_gain_random_pp:+.3f}pp; "
            f"M6 AUROC gain={auroc_gain:+.4f}, AURC reduction={aurc_reduction:+.1%}"
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
