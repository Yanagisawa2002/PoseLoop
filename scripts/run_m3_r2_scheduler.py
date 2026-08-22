#!/usr/bin/env python3
"""Validate and freeze the R2 sequential label-free M3 budget scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import run_m3_r1
from m1_common import load_jsonl, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
VIEW_BUDGETS = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_r2_protocol.json",
    )
    parser.add_argument(
        "--development-rows",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m3_r1" / "nested_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--photoneo-calibration-rows",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "sealed_photoneo" / "m3_decisions.jsonl",
    )
    parser.add_argument(
        "--r1-model",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m3_r1" / "frozen_policy.joblib",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "m3_r2",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r2" / "m3_r2_scheduler_development.md",
    )
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rounded_quota(batch_size: int, spec: Mapping[str, Any]) -> int:
    numerator = int(spec["numerator"])
    denominator = int(spec["denominator"])
    if batch_size < 1 or numerator < 0 or denominator <= 0:
        raise ValueError("Invalid scheduler quota")
    return int(math.floor(batch_size * numerator / denominator + 0.5))


def normalize_development_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        outcomes = {int(key): float(value) for key, value in row["outcomes"].items()}
        if set(outcomes) != set(VIEW_BUDGETS):
            raise ValueError(f"Incomplete development outcomes: {row.get('group_id')}")
        output.append({**row, "outcomes": outcomes})
    return output


def rank_positive_indices(
    rows: Sequence[Mapping[str, Any]], field: str, score_floor: float
) -> list[int]:
    ranked = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        group_id = str(row["group_id"])
        if not group_id or group_id in seen:
            raise ValueError(f"Missing or duplicate scheduler group ID: {group_id!r}")
        seen.add(group_id)
        score = float(row[field])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite scheduler score: {group_id} {field}")
        if score > score_floor:
            ranked.append(index)
    return sorted(
        ranked,
        key=lambda index: (
            -float(rows[index][field]),
            stable_hash(str(rows[index]["group_id"])),
            str(rows[index]["group_id"]),
        ),
    )


def schedule_budgets(
    rows: Sequence[Mapping[str, Any]],
    q1_spec: Mapping[str, Any],
    q2_spec: Mapping[str, Any],
    *,
    k1_field: str,
    k3_field: str,
    score_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot schedule an empty batch")
    group_ids = [str(row["group_id"]) for row in rows]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Scheduler group IDs are not unique")
    q1 = min(len(rows), rounded_quota(len(rows), q1_spec))
    q2 = min(q1, rounded_quota(len(rows), q2_spec))

    stage1_ranked = rank_positive_indices(rows, k1_field, score_floor)
    stage1 = set(stage1_ranked[:q1])
    stage1_rows = [rows[index] for index in sorted(stage1)]
    stage2_local = rank_positive_indices(stage1_rows, k3_field, score_floor)
    stage1_indices = sorted(stage1)
    stage2_ranked = [stage1_indices[index] for index in stage2_local]
    stage2 = set(stage2_ranked[:q2])
    if not stage2 <= stage1:
        raise RuntimeError("Stage-two scheduler selections are not nested")

    budgets = np.ones(len(rows), dtype=np.int64)
    if stage1:
        budgets[np.asarray(sorted(stage1), dtype=np.int64)] = 3
    if stage2:
        budgets[np.asarray(sorted(stage2), dtype=np.int64)] = 5
    mean_views = float(np.mean(budgets))
    if mean_views > 3.0 + 1e-12:
        raise RuntimeError(f"Scheduler exceeded the fixed mean-view cap: {mean_views}")
    audit = {
        "batch_size": len(rows),
        "q1_limit": q1,
        "q2_limit": q2,
        "q1_selected": len(stage1),
        "q2_selected": len(stage2),
        "budget_counts": {
            str(budget): int(np.sum(budgets == budget)) for budget in VIEW_BUDGETS
        },
        "mean_views": mean_views,
        "stage1_group_ids": [group_ids[index] for index in stage1_ranked[:q1]],
        "stage2_group_ids": [group_ids[index] for index in stage2_ranked[:q2]],
    }
    return budgets, audit


def render_report(result: Mapping[str, Any]) -> str:
    dev = result["development"]
    gate = result["development_gate"]
    calibration = result["photoneo_label_free_dry_run"]
    return "\n".join(
        [
            "# PoseLoop M3-R2 scheduler development",
            "",
            f"**Decision: `{gate['decision']}`.**",
            "",
            "The R1 marginal-value model is unchanged. R2 replaces sensor-sensitive "
            "absolute thresholds with sequential batch quantiles whose quotas were copied "
            "from the already frozen R1 primary OOF operating point. Stage two is ranked "
            "only after stage-one views are acquired.",
            "",
            "## RealSense nested-OOF validation",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Active macro combined | {100*dev['active_score']:.2f}% |",
            f"| Exact-budget matched random | {100*dev['matched_random_expected_score']:.2f}% |",
            f"| Gain | {dev['gain_pp']:+.2f} pp |",
            f"| One-sided 90% grouped-bootstrap lower gain | {dev['grouped_bootstrap']['gain_pp_one_sided_90pct_lower']:+.2f} pp |",
            f"| Mean views | {dev['mean_views']:.3f} |",
            f"| k=1 / k=3 / k=5 | {dev['budget_counts']['1']} / {dev['budget_counts']['3']} / {dev['budget_counts']['5']} |",
            "",
            "## Consumed Photoneo label-free dry run",
            "",
            f"The scheduler selected k=1/3/5 counts "
            f"{calibration['budget_counts']['1']}/{calibration['budget_counts']['3']}/"
            f"{calibration['budget_counts']['5']} with mean views "
            f"{calibration['mean_views']:.3f}. No evaluator artifact or pose outcome was read.",
            "",
            "## Gate",
            "",
            *[
                f"- `{name}`: {'pass' if passed else 'fail'}"
                for name, passed in gate["checks"].items()
            ],
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol.resolve()
    development_path = args.development_rows.resolve()
    calibration_path = args.photoneo_calibration_rows.resolve()
    model_path = args.r1_model.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    for path in (protocol_path, development_path, calibration_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not output_root.is_relative_to((repo_root / "artifacts" / "r2").resolve()):
        raise ValueError("M3-R2 outputs must stay under artifacts/r2")
    if not report_path.is_relative_to((repo_root / "reports" / "r2").resolve()):
        raise ValueError("M3-R2 report must stay under reports/r2")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "poseloop-r2-v1":
        raise ValueError("Unexpected R2 protocol")
    stage = protocol["M3-R2"]
    scheduler = stage["scheduler"]
    if sha256_file(model_path) != str(stage["marginal_value_model"]["sha256"]):
        raise RuntimeError("Frozen R1 M3 model hash changed")

    development_rows = normalize_development_rows(load_jsonl(development_path))
    if any(row.get("record_type") != "m3_r1_nested_oof" for row in development_rows):
        raise ValueError("Unexpected M3 development row")
    dev_budgets, dev_schedule = schedule_budgets(
        development_rows,
        scheduler["q1"],
        scheduler["q2"],
        k1_field="k1_predicted_marginal_value",
        k3_field="k3_predicted_marginal_value",
        score_floor=float(scheduler["score_floor"]),
    )
    development = run_m3_r1.evaluate_cap(
        development_rows,
        dev_budgets,
        3.0,
        random_assignments=2000,
        bootstrap_resamples=5000,
    )
    gate_spec = stage["development_gate"]
    checks = {
        "gain": development["gain_pp"]
        >= float(gate_spec["gain_over_exact_budget_matched_random_pp_min"]),
        "bootstrap_lower": development["grouped_bootstrap"][
            "gain_pp_one_sided_90pct_lower"
        ]
        >= float(gate_spec["one_sided_group_bootstrap_90pct_lower_gain_pp_min"]),
        "mean_view_cap": development["mean_views"]
        <= float(gate_spec["mean_views_max"]) + 1e-12,
        "difficulty_guard": all(
            (not value["supported_for_gate"])
            or value["gain_pp"]
            >= -float(gate_spec["max_supported_visibility_stratum_regression_pp"])
            for value in development["difficulty"].values()
        ),
    }
    passed = all(checks.values())

    calibration_rows = load_jsonl(calibration_path)
    if any(
        row.get("record_type") != "r1_sealed_m3_decision"
        or bool(row.get("evaluator_label_read"))
        for row in calibration_rows
    ):
        raise RuntimeError("Photoneo calibration input is not label-blind")
    _, calibration_schedule = schedule_budgets(
        calibration_rows,
        scheduler["q1"],
        scheduler["q2"],
        k1_field="k1_predicted_marginal_value",
        k3_field="k3_predicted_marginal_value",
        score_floor=float(scheduler["score_floor"]),
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M3-R2 sequential label-free batch scheduler",
        "status": "PASS_FREEZE_FOR_R2" if passed else "STOP_M3_R2",
        "marginal_value_model_changed": False,
        "scheduler": {
            **scheduler,
            "development_schedule": dev_schedule,
        },
        "development": development,
        "development_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gate_spec,
            "decision": "PASS_FREEZE_FOR_R2" if passed else "STOP_M3_R2",
        },
        "photoneo_label_free_dry_run": {
            **calibration_schedule,
            "evaluator_artifact_read": False,
            "pose_outcome_read": False,
        },
        "provenance": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "development_rows": {
                "path": str(development_path),
                "sha256": sha256_file(development_path),
            },
            "photoneo_calibration_rows": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "frozen_m3_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "development_result.json"
    write_json_atomic(result_path, result)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M3-R2 scheduler contract",
        "status": "frozen_after_development_gate_pass" if passed else "not_frozen",
        "marginal_value_model": result["provenance"]["frozen_m3_model"],
        "scheduler": scheduler,
        "development_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "evaluator_label_used_for_scheduler": False,
    }
    write_json_atomic(output_root / "frozen_scheduler.json", contract)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(result["status"])
    print(
        f"development gain={development['gain_pp']:+.3f}pp, "
        f"bootstrap lower={development['grouped_bootstrap']['gain_pp_one_sided_90pct_lower']:+.3f}pp, "
        f"mean_views={development['mean_views']:.3f}"
    )
    print(
        "Photoneo label-free dry run: "
        f"budgets={calibration_schedule['budget_counts']}, "
        f"mean_views={calibration_schedule['mean_views']:.3f}"
    )
    print(result_path)
    print(report_path)


if __name__ == "__main__":
    main()
