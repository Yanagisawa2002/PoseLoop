#!/usr/bin/env python3
"""Rebuild M4-R1 development features/outcomes from raw M1/M2 evidence.

The output is deliberately split into a pre-acquisition feature stream and a
label-only outcome stream.  The downstream CAD renderer receives only the
feature stream and therefore cannot consume candidate outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fit_m4_voi as legacy_m4
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "artifacts" / "r1" / "m4_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--toolkit-root", type=Path, default=repo_root / "third_party" / "bop_toolkit"
    )
    parser.add_argument("--output-root", type=Path, default=output_root)
    return parser.parse_args()


def _input_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "m2_groups": repo_root / "artifacts" / "m2" / "groups.jsonl",
        "m2_candidate_manifest": repo_root
        / "artifacts"
        / "m2"
        / "candidate_manifest.jsonl",
        "m2_groups_summary": repo_root / "artifacts" / "m2" / "groups_summary.json",
        "m2_extrinsics_audit": repo_root
        / "artifacts"
        / "m2"
        / "extrinsics_audit.json",
        "m2_view_predictions": repo_root
        / "artifacts"
        / "m2"
        / "view_predictions.jsonl",
        "m2_metrics": repo_root / "artifacts" / "m2" / "metrics.jsonl",
        "m1_manifest": repo_root / "artifacts" / "m1" / "manifest.jsonl",
        "m1_predictions": repo_root / "artifacts" / "m1" / "predictions.jsonl",
    }


def _target_only_scores(metrics_path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in load_jsonl(metrics_path):
        if row.get("record_type") != "method_result":
            raise ValueError("Unexpected M2 metrics row")
        if row.get("method") != "target_only" or int(row["requested_view_budget"]) != 1:
            continue
        group_id = str(row["group_id"])
        if group_id in scores:
            raise ValueError(f"Duplicate target-only outcome for {group_id}")
        scores[group_id] = 0.5 * (
            float(row["sample_ar_mssd"]) + float(row["sample_ar_mspd"])
        )
    if len(scores) != legacy_m4.EXPECTED_TARGET_COUNT:
        raise ValueError(f"Expected 300 target-only outcomes, got {len(scores)}")
    return scores


def split_rows(
    rows: list[dict[str, Any]], target_only: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows = []
    outcome_rows = []
    for row in rows:
        identity = {
            "schema_version": 1,
            "group_id": row["group_id"],
            "object_id": row["object_id"],
            "physical_instance_id": row["physical_instance_id"],
            "candidate_slot": row["candidate_slot"],
        }
        feature_rows.append(
            {
                "record_type": "m4_r1_preacquisition_feature",
                **identity,
                "features": row["features"],
            }
        )
        outcome_rows.append(
            {
                "record_type": "m4_r1_development_outcome",
                **identity,
                "target_only_utility": target_only[str(row["group_id"])],
                "candidate_pair_utility": row["actual_utility"],
                "sample_ar_mssd": row["sample_ar_mssd"],
                "sample_ar_mspd": row["sample_ar_mspd"],
                "joint_success": row["joint_success"],
                "outcome_evidence": row["outcome_evidence"],
            }
        )
    return feature_rows, outcome_rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r1" / "m4_r1").resolve()):
        raise ValueError("M4-R1 inputs must stay under artifacts/r1/m4_r1")
    if not args.dataset_root.resolve().is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.toolkit_root.resolve().is_dir():
        raise FileNotFoundError(args.toolkit_root)

    inputs = {name: path.resolve() for name, path in _input_paths(repo_root).items()}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    loader_args = SimpleNamespace(
        dataset_root=args.dataset_root.resolve(), toolkit_root=args.toolkit_root.resolve()
    )
    prepared, _, preparation_audit = legacy_m4._load_prepared_m2(loader_args, inputs)
    rows, cad_cache = legacy_m4._build_development_rows(
        prepared, args.dataset_root.resolve()
    )
    feature_rows, outcome_rows = split_rows(
        rows, _target_only_scores(inputs["m2_metrics"])
    )

    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / "preacquisition_features.jsonl"
    outcome_path = output_root / "development_outcomes.jsonl"
    write_jsonl_atomic(feature_path, feature_rows)
    write_jsonl_atomic(outcome_path, outcome_rows)
    summary = {
        "schema_version": 1,
        "stage": "M4-R1",
        "source": "raw M1/M2 development evidence",
        "legacy_m4_artifacts_read": False,
        "candidate_target_count": legacy_m4.EXPECTED_TARGET_COUNT,
        "candidate_rows_per_target": legacy_m4.EXPECTED_CANDIDATES_PER_TARGET,
        "candidate_row_count": len(rows),
        "physical_instance_count": len(
            {str(row["physical_instance_id"]) for row in rows}
        ),
        "preacquisition_features": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "contains_candidate_outcome": False,
        },
        "development_outcomes": {
            "path": str(outcome_path),
            "sha256": sha256_file(outcome_path),
            "contains_runtime_features": False,
        },
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "cad_provenance": {
            str(object_id): {
                "model_path": str(cad["model_path"]),
                "model_sha256": cad["model_sha256"],
            }
            for object_id, cad in sorted(cad_cache.items())
        },
        "preparation_audit": preparation_audit,
    }
    write_json_atomic(output_root / "input_summary.json", summary)
    print(
        f"M4-R1 inputs rebuilt: targets={summary['candidate_target_count']}, "
        f"rows={summary['candidate_row_count']}, "
        f"tracks={summary['physical_instance_count']}"
    )
    print(feature_path)
    print(outcome_path)


if __name__ == "__main__":
    main()
