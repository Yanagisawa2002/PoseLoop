#!/usr/bin/env python3
"""Freeze the cross-fitted M3-R1 + M4-R1 final multi-view output stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
PRIMARY_M3_CAP = "3.0"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m3-oof",
        type=Path,
        default=r1 / "m3_r1" / "nested_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--m3-result", type=Path, default=r1 / "m3_r1" / "development_result.json"
    )
    parser.add_argument(
        "--m4-oof", type=Path, default=r1 / "m4_r1" / "oof_rankings.jsonl"
    )
    parser.add_argument(
        "--m4-outcomes",
        type=Path,
        default=r1 / "m4_r1" / "development_outcomes.jsonl",
    )
    parser.add_argument(
        "--m4-result", type=Path, default=r1 / "m4_r1" / "development_result.json"
    )
    parser.add_argument(
        "--m2-groups",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--m2-metrics",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "metrics.jsonl",
    )
    parser.add_argument(
        "--output-root", type=Path, default=r1 / "final_multiview"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "final_multiview_output.md",
    )
    return parser.parse_args()


def index_by_group(rows: Sequence[dict[str, Any]], record_type: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected {record_type} stream row")
        group_id = str(row["group_id"])
        if group_id in output:
            raise ValueError(f"Duplicate group ID: {group_id}")
        output[group_id] = row
    return output


def index_outcomes(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("record_type") != "m4_r1_development_outcome":
            raise ValueError("Unexpected M4-R1 outcome row")
        key = str(row["group_id"]), int(row["candidate_slot"])
        if key in output:
            raise ValueError(f"Duplicate M4-R1 outcome: {key}")
        output[key] = row
    return output


def index_target_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("record_type") != "method_result":
            raise ValueError("Unexpected M2 metric row")
        if row.get("method") != "target_only" or int(row["requested_view_budget"]) != 1:
            continue
        group_id = str(row["group_id"])
        if group_id in output:
            raise ValueError(f"Duplicate target-only metric: {group_id}")
        output[group_id] = row
    return output


def macro_object(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["object_id"])].append(float(row[field]))
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def build_final_rows(
    m3: Mapping[str, dict[str, Any]],
    m4: Mapping[str, dict[str, Any]],
    outcomes: Mapping[tuple[str, int], dict[str, Any]],
    groups: Mapping[str, dict[str, Any]],
    target_metrics: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    identities = set(m3)
    if identities != set(m4) or identities != set(groups) or identities != set(target_metrics):
        raise ValueError("M3/M4/group/metric target identities differ")
    rows = []
    for group_id in sorted(identities):
        m3_row = m3[group_id]
        m4_row = m4[group_id]
        group = groups[group_id]
        m3_budget = int(m3_row["selected_budgets"][PRIMARY_M3_CAP])
        if bool(m4_row["m3_primary_continue"]) != (m3_budget > 1):
            raise ValueError(f"M3/M4 eligibility mismatch: {group_id}")
        rank_scores = {
            str(slot): float(value)
            for slot, value in m4_row["predicted_differences_from_slot_2"].items()
        }
        sorted_scores = sorted(rank_scores.values(), reverse=True)
        if m3_budget == 1:
            metric = target_metrics[group_id]
            final = {
                "route": "m3_stop_target_only",
                "acquired_view_count": 1,
                "m4_selected_slot": None,
                "selected_sample_id": str(metric["selected_sample_id"]),
                "selected_acquisition_rank": 0,
                "sample_ar_mssd": float(metric["sample_ar_mssd"]),
                "sample_ar_mspd": float(metric["sample_ar_mspd"]),
                "joint_success": bool(metric["diagnostic_success"]["joint"]),
                "normalized_mssd": metric["normalized_mssd"],
                "mspd_px": metric["mspd_px"],
                "finite_pose": bool(metric["finite_pose"]),
                "status": str(metric["status"]),
            }
        else:
            slot = int(m4_row["selected_slot"])
            outcome = outcomes[(group_id, slot)]
            evidence = outcome["outcome_evidence"]
            final = {
                "route": "m3_continue_m4_ranked_pair",
                "acquired_view_count": 2,
                "m4_selected_slot": slot,
                "selected_sample_id": str(evidence["selected_sample_id"]),
                "selected_acquisition_rank": int(evidence["selected_acquisition_rank"]),
                "sample_ar_mssd": float(outcome["sample_ar_mssd"]),
                "sample_ar_mspd": float(outcome["sample_ar_mspd"]),
                "joint_success": bool(outcome["joint_success"]),
                "normalized_mssd": evidence["normalized_mssd"],
                "mspd_px": evidence["mspd_px"],
                "finite_pose": bool(evidence["finite_pose"]),
                "status": str(evidence["status"]),
            }
        final["combined_utility"] = 0.5 * (
            final["sample_ar_mssd"] + final["sample_ar_mspd"]
        )
        rows.append(
            {
                "record_type": "r1_final_multiview_output",
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "object_id": int(m3_row["object_id"]),
                "physical_instance_id": str(m3_row["physical_instance_track_id"]),
                "target_sample_id": str(group["target_sample_id"]),
                "target_visibility_bin": str(m3_row["target_visibility_bin"]),
                "m3_primary_budget_before_m4_collapse": m3_budget,
                "m3_k1_predicted_marginal_value": float(
                    m3_row["k1_predicted_marginal_value"]
                ),
                "m3_k3_predicted_marginal_value": float(
                    m3_row["k3_predicted_marginal_value"]
                ),
                "m4_predicted_differences_from_slot_2": rank_scores,
                "m4_top_predicted_difference": float(sorted_scores[0]),
                "m4_predicted_rank_margin": float(sorted_scores[0] - sorted_scores[1]),
                "final": final,
                "comparison": {
                    "m3_primary_utility": float(m3_row["outcomes"][str(m3_budget)]),
                    "m4_fixed_slot_2_utility": (
                        float(outcomes[(group_id, 2)]["candidate_pair_utility"])
                        if m3_budget > 1
                        else final["combined_utility"]
                    ),
                    "m4_uniform_random_expected_utility": (
                        float(
                            np.mean(
                                [
                                    outcomes[(group_id, slot)]["candidate_pair_utility"]
                                    for slot in (1, 2, 3, 4)
                                ]
                            )
                        )
                        if m3_budget > 1
                        else final["combined_utility"]
                    ),
                },
            }
        )
    return rows


def render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    return "\n".join(
        [
            "# PoseLoop R1 final multi-view development output",
            "",
            "The frozen development output composes the nested-OOF M3-R1 continue "
            "decision with the nested-OOF M4-R1 CAD ranker. M3 stops retain the "
            "target pose; M3 continues acquire one M4-ranked candidate and use the "
            "frozen max-mask post-acquisition rule.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Targets | {summary['target_count']} |",
            f"| Mean acquired views | {metrics['mean_acquired_view_count']:.3f} |",
            f"| Final macro combined | {100 * metrics['macro_object_combined']:.2f}% |",
            f"| M3-R1 primary macro combined | {100 * metrics['m3_primary_macro_object_combined']:.2f}% |",
            f"| Gain vs M3-R1 primary | {metrics['gain_vs_m3_primary_pp']:+.2f} pp |",
            f"| Gain vs fixed-slot composition | {metrics['gain_vs_fixed_slot_composition_pp']:+.2f} pp |",
            f"| Gain vs random composition | {metrics['gain_vs_random_composition_pp']:+.2f} pp |",
            "",
            "This stream is the only pose-output contract permitted as M6-R1 input. "
            "M6-R1 may predict risk or abstain, but may not reselect a pose.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    inputs = {
        "m3_oof": args.m3_oof.resolve(),
        "m3_result": args.m3_result.resolve(),
        "m4_oof": args.m4_oof.resolve(),
        "m4_outcomes": args.m4_outcomes.resolve(),
        "m4_result": args.m4_result.resolve(),
        "m2_groups": args.m2_groups.resolve(),
        "m2_metrics": args.m2_metrics.resolve(),
    }
    for name, path in inputs.items():
        allowed_root = r1_root if name.startswith(("m3_", "m4_")) else (
            repo_root / "artifacts" / "m2"
        ).resolve()
        if not path.is_file() or not path.is_relative_to(allowed_root):
            raise ValueError(f"Invalid final-multiview input {name}: {path}")
    if not json.loads(inputs["m3_result"].read_text())["development_gate"]["passed"]:
        raise RuntimeError("M3-R1 development gate did not pass")
    if not json.loads(inputs["m4_result"].read_text())["development_gate"]["passed"]:
        raise RuntimeError("M4-R1 development gate did not pass")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(r1_root):
        raise ValueError("Final multi-view outputs must stay under artifacts/r1")
    if not args.report.resolve().is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("Final multi-view report must stay under reports/r1")

    m3 = index_by_group(load_jsonl(inputs["m3_oof"]), "m3_r1_nested_oof")
    m4 = index_by_group(load_jsonl(inputs["m4_oof"]), "m4_r1_oof_ranking")
    outcomes = index_outcomes(load_jsonl(inputs["m4_outcomes"]))
    groups = {
        str(row["group_id"]): row for row in load_jsonl(inputs["m2_groups"])
    }
    target_metrics = index_target_metrics(load_jsonl(inputs["m2_metrics"]))
    rows = build_final_rows(m3, m4, outcomes, groups, target_metrics)
    if len(rows) != 300:
        raise ValueError(f"Final multi-view stream must have 300 rows, got {len(rows)}")

    final_score = macro_object(
        [{**row, "value": row["final"]["combined_utility"]} for row in rows],
        "value",
    )
    m3_score = macro_object(
        [{**row, "value": row["comparison"]["m3_primary_utility"]} for row in rows],
        "value",
    )
    fixed_score = macro_object(
        [
            {**row, "value": row["comparison"]["m4_fixed_slot_2_utility"]}
            for row in rows
        ],
        "value",
    )
    random_score = macro_object(
        [
            {
                **row,
                "value": row["comparison"]["m4_uniform_random_expected_utility"],
            }
            for row in rows
        ],
        "value",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1 final multi-view output",
        "status": "frozen_before_m6_r1",
        "m6_pose_reselection_allowed": False,
        "target_count": len(rows),
        "physical_instance_count": len({row["physical_instance_id"] for row in rows}),
        "route_counts": dict(
            sorted(
                {
                    route: sum(row["final"]["route"] == route for row in rows)
                    for route in {row["final"]["route"] for row in rows}
                }.items()
            )
        ),
        "metrics": {
            "mean_acquired_view_count": float(
                np.mean([row["final"]["acquired_view_count"] for row in rows])
            ),
            "macro_object_combined": final_score,
            "m3_primary_macro_object_combined": m3_score,
            "gain_vs_m3_primary_pp": 100.0 * (final_score - m3_score),
            "fixed_slot_composition_macro_object_combined": fixed_score,
            "gain_vs_fixed_slot_composition_pp": 100.0 * (final_score - fixed_score),
            "random_composition_macro_object_combined": random_score,
            "gain_vs_random_composition_pp": 100.0 * (final_score - random_score),
        },
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "outputs.jsonl"
    write_jsonl_atomic(output_path, rows)
    summary["output"] = {"path": str(output_path), "sha256": sha256_file(output_path)}
    summary["configuration_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json_atomic(output_root / "frozen_contract.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary), encoding="utf-8")
    print(
        f"Final multi-view frozen: score={final_score:.6f}, "
        f"mean_views={summary['metrics']['mean_acquired_view_count']:.3f}, "
        f"gain_vs_m3={summary['metrics']['gain_vs_m3_primary_pp']:+.3f}pp"
    )
    print(output_path)


if __name__ == "__main__":
    main()
