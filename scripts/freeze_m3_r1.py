#!/usr/bin/env python3
"""Fit and freeze the deployable M3-R1 marginal-value policy on all development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

import fit_m3_policy as legacy_m3
import run_m3_r1 as m3
from m1_common import load_jsonl, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
FULL_CV_SEED = 20261301
FULL_MODEL_SEED = 20261351


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "m3_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", type=Path, default=repo_root / "artifacts" / "m2" / "groups.jsonl"
    )
    parser.add_argument(
        "--metrics", type=Path, default=repo_root / "artifacts" / "m2" / "metrics.jsonl"
    )
    parser.add_argument(
        "--m1-predictions", type=Path, default=repo_root / "artifacts" / "m1" / "predictions.jsonl"
    )
    parser.add_argument(
        "--m2-predictions", type=Path, default=repo_root / "artifacts" / "m2" / "view_predictions.jsonl"
    )
    parser.add_argument(
        "--development-result", type=Path, default=root / "development_result.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=repo_root / "protocols" / "poseloop_r1_protocol.json"
    )
    parser.add_argument("--output-root", type=Path, default=root)
    return parser.parse_args()


def select_full_development_candidate(
    rows: list[dict[str, Any]], mean_view_cap: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for index, spec in enumerate(m3.MODEL_SPECS):
        predictions = m3.crossfit_spec(
            rows, spec, m3.OUTER_FOLDS, FULL_CV_SEED + index
        )
        operation = m3.select_operating_point(rows, predictions, mean_view_cap)
        candidates.append(
            {
                "spec": dict(spec),
                "cross_fitted_operation": operation,
            }
        )
    chosen = min(
        candidates,
        key=lambda item: (
            -item["cross_fitted_operation"]["allocation_gain"],
            -item["cross_fitted_operation"]["active_score"],
            item["spec"]["name"],
        ),
    )
    return chosen, candidates


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(r1_root):
        raise ValueError("M3-R1 frozen model must stay under artifacts/r1")
    inputs = {
        "groups": args.groups.resolve(),
        "metrics": args.metrics.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
        "m2_predictions": args.m2_predictions.resolve(),
        "development_result": args.development_result.resolve(),
        "protocol": args.protocol.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    development = json.loads(inputs["development_result"].read_text(encoding="utf-8"))
    if not development["development_gate"]["passed"]:
        raise RuntimeError("M3-R1 development gate did not pass")
    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    mean_view_cap = float(
        protocol["stages"]["M3-R1"]["validation"]["primary_mean_view_cap"]
    )
    groups = load_jsonl(inputs["groups"])
    indexed_metrics = legacy_m3._index_metrics(load_jsonl(inputs["metrics"]))
    prediction_sources = {
        "m1": m3._load_prediction_index(inputs["m1_predictions"]),
        "m2": m3._load_prediction_index(inputs["m2_predictions"]),
    }
    rows = m3.build_development_rows(groups, indexed_metrics, prediction_sources)
    chosen, candidates = select_full_development_candidate(rows, mean_view_cap)
    models = {}
    for stage_index, stage in enumerate(("k1", "k3")):
        models[stage] = m3.fit_regressor(
            m3._matrix(rows, stage),
            m3._targets(rows, stage),
            rows,
            chosen["spec"],
            FULL_MODEL_SEED + stage_index,
        )
    operation = chosen["cross_fitted_operation"]
    model_path = output_root / "frozen_policy.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3-R1",
            "model_spec": chosen["spec"],
            "models": models,
            "feature_names": {
                "k1": list(m3.K1_FEATURE_NAMES),
                "k3": list(m3.K3_FEATURE_NAMES),
            },
            "thresholds": {
                "k1": float(operation["threshold_k1"]),
                "k3": float(operation["threshold_k3"]),
            },
            "mean_view_cap": mean_view_cap,
        },
        model_path,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M3-R1",
        "status": "frozen_for_single_sealed_validation",
        "selection_evidence": {
            "evaluation": "full-development grouped OOF candidate/threshold selection",
            "chosen": chosen,
            "all_candidates": candidates,
            "sealed_data_read": False,
        },
        "feature_names": {
            "k1": list(m3.K1_FEATURE_NAMES),
            "k3": list(m3.K3_FEATURE_NAMES),
        },
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    write_json_atomic(output_root / "frozen_policy.json", contract)
    print(
        f"M3-R1 policy frozen: model={chosen['spec']['name']}, "
        f"thresholds=({operation['threshold_k1']:.6g}, {operation['threshold_k3']:.6g}), "
        f"cv_gain={100 * operation['allocation_gain']:+.3f}pp, "
        f"mean_views={operation['mean_views']:.3f}"
    )
    print(model_path)


if __name__ == "__main__":
    main()
