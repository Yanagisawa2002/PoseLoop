#!/usr/bin/env python3
"""Validate M6-R2 using only features available from the frozen final output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

import run_m6_r1
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
FAMILY = "selected_output_only"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1" / "m6_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=r1 / "features.jsonl")
    parser.add_argument("--labels", type=Path, default=r1 / "labels.jsonl")
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r2_protocol.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=repo_root / "artifacts" / "r2" / "m6_r2"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m6_r2_deployable_development.md",
    )
    return parser.parse_args()


def render_report(result: Mapping[str, Any]) -> str:
    learned = result["metrics"]["nested_risk_model"]
    raw = result["metrics"]["raw_score_baseline"]
    gate = result["development_gate"]
    fold_lines = [
        "| Fold | N | AUROC gain | AURC reduction | Both improve |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in result["outer_fold_metrics"]:
        fold_lines.append(
            f"| {row['outer_fold']} | {row['target_count']} | "
            f"{row['auroc_gain_over_raw']:+.3f} | "
            f"{row['aurc_relative_reduction_vs_raw']:+.1%} | "
            f"{'yes' if row['positive_direction_both_metrics'] else 'no'} |"
        )
    return "\n".join(
        [
            "# PoseLoop M6-R2 deployable-feature development",
            "",
            f"**{result['status']}**",
            "",
            "All inner and outer models are restricted to `selected_output_only`. "
            "No M3 k3 score, unacquired-view prediction, pair feature, CAD feature, "
            "or object identity is available to this risk model.",
            "",
            "| Metric | Learned | Raw-score baseline | Difference |",
            "| --- | ---: | ---: | ---: |",
            f"| AUROC | {learned['auroc']:.4f} | {raw['auroc']:.4f} | {gate['observed']['auroc_gain_over_raw']:+.4f} |",
            f"| AURC | {learned['aurc']:.4f} | {raw['aurc']:.4f} | {gate['observed']['aurc_relative_reduction_vs_raw']:+.1%} relative |",
            f"| Positive outer folds | {gate['observed']['positive_direction_outer_folds']}/5 | — | — |",
            "",
            *fold_lines,
            "",
            "This replaces the broader R1 OOF claim for deployable M6 evidence.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    inputs = {
        "features": args.features.resolve(),
        "labels": args.labels.resolve(),
        "protocol": args.protocol.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r2").resolve()):
        raise ValueError("M6-R2 outputs must stay under artifacts/r2")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M6-R2 report must stay under reports/r2")

    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    stage = protocol["M6-R2"]
    feature_rows = load_jsonl(inputs["features"])
    labels = {str(row["group_id"]): row for row in load_jsonl(inputs["labels"])}
    if len(feature_rows) != 300 or len(labels) != 300:
        raise ValueError("M6-R2 requires the 300-row development streams")
    label_rows = [labels[str(row["group_id"])] for row in feature_rows]
    all_names = sorted(feature_rows[0]["features"])
    families = run_m6_r1.feature_families(all_names)
    allowed = [str(name) for name in stage["allowed_feature_names"]]
    if families[FAMILY] != sorted(allowed):
        raise RuntimeError("Protocol deployable feature names differ from the source stream")
    restricted_families = {FAMILY: families[FAMILY]}
    candidates = run_m6_r1.model_candidates(restricted_families)
    if any(candidate["family"] != FAMILY for candidate in candidates):
        raise RuntimeError("A forbidden M6-R2 feature family entered candidate selection")

    oof, assignments, selections, fold_rows = run_m6_r1.nested_grouped_oof(
        feature_rows, label_rows, restricted_families, candidates
    )
    if set(
        row["selected"]["candidate"]["family"] for row in selections
    ) != {FAMILY}:
        raise RuntimeError("Outer selection used a non-deployable feature family")
    y = np.asarray([int(row["y_failure"]) for row in label_rows], dtype=int)
    groups = np.asarray([str(row["physical_instance_id"]) for row in label_rows])
    ids = [str(row["group_id"]) for row in label_rows]
    raw = -np.asarray([float(row["features"]["selected_raw_score"]) for row in feature_rows])
    learned_metrics = run_m6_r1._metric_summary(y, oof, oof, ids)
    raw_metrics = run_m6_r1._metric_summary(y, raw, None, ids)
    gate = run_m6_r1.gate_decision(
        learned_metrics, raw_metrics, fold_rows, stage["development_gate"]
    )
    bootstrap = run_m6_r1.grouped_bootstrap(y, oof, raw, groups)

    full_assignments = run_m6_r1.make_fold_assignments(
        y, groups, run_m6_r1.OUTER_FOLDS, run_m6_r1.CV_SEED + 1
    )
    frozen_selection, candidate_scores = run_m6_r1.select_candidate(
        feature_rows,
        y,
        groups,
        restricted_families,
        candidates,
        full_assignments,
        run_m6_r1.MODEL_SEED_BASE + 900_000,
    )
    frozen_candidate = frozen_selection["candidate"]
    matrix = run_m6_r1.feature_matrix(feature_rows, allowed)
    model = run_m6_r1.build_model(frozen_candidate, run_m6_r1.MODEL_SEED_BASE + 999_999)
    model.fit(matrix, y)

    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "frozen_risk_model.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "M6-R2",
            "candidate": frozen_candidate,
            "feature_names": allowed,
            "model": model,
            "pose_reselection_allowed": False,
        },
        model_path,
    )
    oof_rows = [
        {
            "record_type": "m6_r2_nested_oof_risk",
            "schema_version": SCHEMA_VERSION,
            "group_id": feature_rows[index]["group_id"],
            "physical_instance_id": label_rows[index]["physical_instance_id"],
            "object_id": label_rows[index]["object_id"],
            "outer_fold": int(assignments[index]),
            "y_failure": int(y[index]),
            "raw_score_risk": float(raw[index]),
            "predicted_failure_risk": float(oof[index]),
        }
        for index in range(len(y))
    ]
    oof_path = output_root / "nested_oof_predictions.jsonl"
    write_jsonl_atomic(oof_path, oof_rows)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M6-R2",
        "status": "frozen_for_sealed_validation" if gate["passed"] else "development_gate_failed",
        "selected_candidate": frozen_selection,
        "candidate_scores": candidate_scores,
        "feature_family": FAMILY,
        "feature_names": allowed,
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "pose_reselection_allowed": False,
        "future_or_unacquired_view_feature_used": False,
    }
    contract_path = output_root / "frozen_model_contract.json"
    write_json_atomic(contract_path, contract)
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M6-R2 deployable selected-output-only risk",
        "status": "PASS_FREEZE_M6_R2" if gate["passed"] else "STOP_M6_R2",
        "target_count": len(y),
        "failure_count": int(y.sum()),
        "physical_instance_count": len(set(groups)),
        "candidate_count": len(candidates),
        "feature_family": FAMILY,
        "feature_names": allowed,
        "metrics": {
            "nested_risk_model": learned_metrics,
            "raw_score_baseline": raw_metrics,
        },
        "development_gate": gate,
        "grouped_bootstrap": bootstrap,
        "outer_fold_metrics": fold_rows,
        "outer_model_selections": selections,
        "selected_model_families": dict(
            Counter(row["selected"]["candidate"]["family"] for row in selections)
        ),
        "future_or_unacquired_view_feature_used": False,
        "pose_reselection_after_risk": False,
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "oof_predictions": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
        },
    }
    result_path = output_root / "development_result.json"
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(
        f"{result['status']}: AUROC={learned_metrics['auroc']:.4f} "
        f"(raw={raw_metrics['auroc']:.4f}), AURC={learned_metrics['aurc']:.4f} "
        f"(raw={raw_metrics['aurc']:.4f}), positive_folds="
        f"{gate['observed']['positive_direction_outer_folds']}/5"
    )
    print(result_path)
    print(report_path)


if __name__ == "__main__":
    main()
