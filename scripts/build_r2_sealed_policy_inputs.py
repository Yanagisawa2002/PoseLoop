#!/usr/bin/env python3
"""Build sequential label-blind M3-R2 decisions and M4 inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

import fit_m4_voi
import run_m3_r1
import run_m3_r2_scheduler as scheduler_lib
from build_r1_sealed_policy_inputs import (
    FORBIDDEN_PREDICTION_FIELDS,
    VIEW_BUDGETS,
    predict_marginal,
    prefix_result,
    prepare_views,
)
from evaluate_m1 import load_object_evaluation_data, load_official_models
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    r2 = repo_root / "artifacts" / "r2"
    sealed = r2 / "sealed_photoneo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=sealed / "sealed_contract.json")
    parser.add_argument("--predictions", type=Path, default=sealed / "predictions.jsonl")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--m3-model", type=Path, default=r1 / "m3_r1" / "frozen_policy.joblib"
    )
    parser.add_argument(
        "--scheduler-contract", type=Path, default=r2 / "m3_r2" / "frozen_scheduler.json"
    )
    parser.add_argument("--output-root", type=Path, default=sealed)
    parser.add_argument(
        "--m4-features-output",
        type=Path,
        default=r2 / "m4_r2" / "sealed_photoneo_preacquisition_features.jsonl",
    )
    return parser.parse_args()


def contract_input_paths(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "artifacts" / "r2" / "sealed_photoneo"
    return {
        "inference_manifest": root / "inference_manifest.jsonl",
        "target_manifest": root / "target_manifest.jsonl",
        "groups_inference": root / "groups_inference.jsonl",
        "protocol": repo_root / "protocols" / "poseloop_r2_protocol.json",
    }


def validate_unopened_contract(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    root = paths["inference_manifest"].resolve().parent
    if (root / "sealed_open_receipt.json").exists():
        raise RuntimeError("R2 evaluator already has an open receipt")
    if (
        contract.get("protocol_id") != "poseloop-r2-v1"
        or contract.get("status") != "sealed_unopened"
        or bool(contract.get("labels_opened"))
        or int(contract.get("evaluation_invocation_count", -1)) != 0
    ):
        raise RuntimeError("R2 sealed contract is not unopened")
    for name in ("inference_manifest", "target_manifest", "groups_inference"):
        path = paths[name].resolve()
        if not path.is_file() or sha256_file(path) != str(contract["files"][name]["sha256"]):
            raise RuntimeError(f"R2 sealed {name} changed")
    if sha256_file(paths["protocol"].resolve()) != str(contract["protocol"]["sha256"]):
        raise RuntimeError("R2 protocol hash changed")


def load_predictions(
    path: Path, expected_ids: set[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    if not rows or rows[0].get("record_type") != "r1_sealed_inference_metadata":
        raise ValueError("R2 prediction metadata is missing")
    metadata = rows[0]
    if (
        bool(metadata.get("evaluator_labels_path_read"))
        or bool(metadata.get("evaluator_pose_or_error_computed"))
        or int(metadata.get("sample_count", -1)) != len(expected_ids)
    ):
        raise RuntimeError("R2 predictions violate the label-blind contract")
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if row.get("record_type") != "r1_sealed_prediction":
            raise ValueError("Unexpected R2 prediction record")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in predictions:
            raise ValueError(f"Missing or duplicate R2 prediction ID: {sample_id!r}")
        if bool(row.get("evaluator_label_read")):
            raise RuntimeError(f"R2 prediction declares evaluator access: {sample_id}")
        forbidden = FORBIDDEN_PREDICTION_FIELDS & set(row)
        if forbidden:
            raise RuntimeError(f"R2 prediction contains evaluator fields: {sorted(forbidden)}")
        predictions[sample_id] = row
    if set(predictions) != expected_ids:
        raise ValueError("R2 prediction identities differ from the sealed manifest")
    return metadata, predictions


def validate_groups(
    groups: list[dict[str, Any]],
    targets: Mapping[str, dict[str, Any]],
    prediction_ids: set[str],
    expected_targets: int,
) -> None:
    if len(groups) != expected_targets or len(targets) != expected_targets:
        raise ValueError("R2 target/group count mismatch")
    seen: set[str] = set()
    for group in groups:
        group_id = str(group.get("group_id", ""))
        target_id = str(group.get("target_sample_id", ""))
        views = sorted(group.get("views", []), key=lambda row: int(row["acquisition_rank"]))
        if (
            not group_id
            or group_id in seen
            or target_id not in targets
            or len(views) != 5
            or [int(row["acquisition_rank"]) for row in views] != list(range(5))
            or str(views[0]["sample_id"]) != target_id
            or any(str(view["sample_id"]) not in prediction_ids for view in views)
        ):
            raise ValueError(f"Invalid R2 group: {group_id}")
        seen.add(group_id)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sealed_root = (repo_root / "artifacts" / "r2" / "sealed_photoneo").resolve()
    output_root = args.output_root.resolve()
    m4_root = (repo_root / "artifacts" / "r2" / "m4_r2").resolve()
    m4_feature_path = args.m4_features_output.resolve()
    if output_root != sealed_root or not m4_feature_path.is_relative_to(m4_root):
        raise ValueError("R2 policy outputs are outside their frozen namespaces")

    contract_path = args.contract.resolve()
    predictions_path = args.predictions.resolve()
    model_path = args.m3_model.resolve()
    scheduler_path = args.scheduler_contract.resolve()
    dataset_root = args.dataset_root.resolve()
    for path in (contract_path, predictions_path, model_path, scheduler_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_paths = contract_input_paths(repo_root)
    validate_unopened_contract(contract, input_paths)
    manifest_rows = load_jsonl(input_paths["inference_manifest"])
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if len(manifest_ids) != int(contract["inference_sample_count"]):
        raise ValueError("R2 manifest identity count changed")
    metadata, predictions = load_predictions(predictions_path, manifest_ids)
    target_rows = load_jsonl(input_paths["target_manifest"])
    targets = {str(row["sample_id"]): row for row in target_rows}
    groups = load_jsonl(input_paths["groups_inference"])
    validate_groups(groups, targets, set(predictions), int(contract["target_count"]))

    if sha256_file(model_path) != str(contract["components"]["m3_model"]["sha256"]):
        raise RuntimeError("R2 frozen M3 model changed")
    if sha256_file(scheduler_path) != str(contract["components"]["m3_scheduler"]["sha256"]):
        raise RuntimeError("R2 frozen M3 scheduler changed")
    frozen_m3 = joblib.load(model_path)
    scheduler_contract = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if scheduler_contract.get("status") != "frozen_after_development_gate_pass":
        raise RuntimeError("R2 scheduler is not frozen")
    scheduler = scheduler_contract["scheduler"]
    names = {
        stage: [str(name) for name in frozen_m3["feature_names"][stage]]
        for stage in ("k1", "k3")
    }
    prediction_sources = {"m1": predictions, "m2": predictions}
    model_params, model_info = load_official_models(dataset_root)
    object_data_cache: dict[int, dict[str, Any]] = {}
    cad_cache: dict[int, dict[str, Any]] = {}
    prepared_by_group: dict[str, list[dict[str, Any]]] = {}
    decision_rows: list[dict[str, Any]] = []
    m4_rows: list[dict[str, Any]] = []

    sorted_groups = sorted(groups, key=lambda row: str(row["group_id"]))
    for index, group in enumerate(sorted_groups, start=1):
        group_id = str(group["group_id"])
        object_id = int(group["object_id"])
        target_id = str(group["target_sample_id"])
        track_id = str(group.get("oracle_association", {}).get("track_id", ""))
        if not track_id:
            raise ValueError(f"R2 group lacks split-isolation track ID: {group_id}")
        if object_id not in object_data_cache:
            object_data_cache[object_id] = load_object_evaluation_data(
                object_id, model_params, model_info
            )
            cad_cache[object_id] = fit_m4_voi.load_cad_geometry(dataset_root, object_id)
        prepared = prepare_views(group, predictions)
        prepared_by_group[group_id] = prepared
        prefix1 = prefix_result(prepared, 1, object_data_cache[object_id])
        k1_features = run_m3_r1.extract_r1_prefix_features(
            group, prefix1, 1, prediction_sources
        )
        p1 = predict_marginal(frozen_m3["models"]["k1"], k1_features, names["k1"])
        decision_rows.append(
            {
                "record_type": "r2_sealed_m3_decision",
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "object_id": object_id,
                "physical_instance_id": track_id,
                "target_sample_id": target_id,
                "k1_predicted_marginal_value": p1,
                "k3_predicted_marginal_value": None,
                "selected_budget": 1,
                "k1_features": k1_features,
                "k3_features": None,
                "prefix_results": {"1": prefix1},
                "evaluator_label_read": False,
            }
        )
        context = fit_m4_voi.extract_target_context(
            group,
            targets[target_id],
            predictions[target_id],
            cad_cache[object_id],
            dataset_root,
        )
        for slot in (1, 2, 3, 4):
            m4_rows.append(
                {
                    "record_type": "m4_r1_preacquisition_feature",
                    "schema_version": SCHEMA_VERSION,
                    "group_id": group_id,
                    "object_id": object_id,
                    "physical_instance_id": track_id,
                    "candidate_slot": slot,
                    "features": fit_m4_voi.extract_candidate_features(
                        group, slot, context, cad_cache[object_id]
                    ),
                }
            )
        if index % 25 == 0 or index == len(sorted_groups):
            print(f"R2 target-only features: {index}/{len(sorted_groups)}", flush=True)

    q1 = min(
        len(decision_rows),
        scheduler_lib.rounded_quota(len(decision_rows), scheduler["q1"]),
    )
    stage1_ranked = scheduler_lib.rank_positive_indices(
        decision_rows, "k1_predicted_marginal_value", float(scheduler["score_floor"])
    )
    stage1_indices = stage1_ranked[:q1]
    for rank, row_index in enumerate(stage1_indices, start=1):
        row = decision_rows[row_index]
        group = sorted_groups[row_index]
        prepared = prepared_by_group[str(row["group_id"])]
        prefix3 = prefix_result(
            prepared, 3, object_data_cache[int(row["object_id"])]
        )
        k3_features = run_m3_r1.extract_r1_prefix_features(
            group, prefix3, 3, prediction_sources
        )
        row["k3_features"] = k3_features
        row["k3_predicted_marginal_value"] = predict_marginal(
            frozen_m3["models"]["k3"], k3_features, names["k3"]
        )
        row["prefix_results"]["3"] = prefix3
        row["selected_budget"] = 3
        row["stage1_scheduler_rank"] = rank

    q2 = min(
        len(stage1_indices),
        scheduler_lib.rounded_quota(len(decision_rows), scheduler["q2"]),
    )
    stage1_rows = [decision_rows[index] for index in stage1_indices]
    stage2_local_ranked = scheduler_lib.rank_positive_indices(
        stage1_rows, "k3_predicted_marginal_value", float(scheduler["score_floor"])
    )
    stage2_indices = [stage1_indices[index] for index in stage2_local_ranked[:q2]]
    for rank, row_index in enumerate(stage2_indices, start=1):
        row = decision_rows[row_index]
        prepared = prepared_by_group[str(row["group_id"])]
        row["prefix_results"]["5"] = prefix_result(
            prepared, 5, object_data_cache[int(row["object_id"])]
        )
        row["selected_budget"] = 5
        row["stage2_scheduler_rank"] = rank

    budgets = np.asarray([int(row["selected_budget"]) for row in decision_rows])
    mean_views = float(np.mean(budgets))
    if mean_views > float(contract["single_open_gates"]["M3_mean_views_max"]) + 1e-12:
        raise RuntimeError("R2 scheduler exceeded the sealed mean-view cap")
    if any(
        (row["selected_budget"] == 1 and row["k3_features"] is not None)
        or (row["selected_budget"] > 1 and row["k3_features"] is None)
        or (row["selected_budget"] < 5 and "5" in row["prefix_results"])
        for row in decision_rows
    ):
        raise RuntimeError("R2 acquired-prefix availability audit failed")

    output_root.mkdir(parents=True, exist_ok=True)
    m4_feature_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path = output_root / "m3_decisions.jsonl"
    write_jsonl_atomic(decision_path, decision_rows)
    write_jsonl_atomic(m4_feature_path, m4_rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": contract["protocol_id"],
        "stage": "R2 sequential label-blind policy inputs",
        "status": "complete",
        "target_count": len(decision_rows),
        "m4_candidate_row_count": len(m4_rows),
        "budget_counts": {
            str(budget): int(np.sum(budgets == budget)) for budget in VIEW_BUDGETS
        },
        "mean_m3_views": mean_views,
        "stage1_k3_feature_count": sum(row["k3_features"] is not None for row in decision_rows),
        "stage2_k5_prefix_count": sum("5" in row["prefix_results"] for row in decision_rows),
        "future_prefix_feature_used_before_acquisition": False,
        "evaluator_label_files_opened": False,
        "evaluator_pose_or_error_computed": False,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            "sealed_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
            "m3_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "scheduler": {"path": str(scheduler_path), "sha256": sha256_file(scheduler_path)},
            **{
                key: {"path": str(path), "sha256": sha256_file(path)}
                for key, path in input_paths.items()
            },
        },
        "outputs": {
            "m3_decisions": {"path": str(decision_path), "sha256": sha256_file(decision_path)},
            "m4_preacquisition_features": {
                "path": str(m4_feature_path),
                "sha256": sha256_file(m4_feature_path),
                "contains_candidate_outcome": False,
            },
        },
        "inference_metadata": {
            "manifest_sha256": metadata["manifest_sha256"],
            "evaluator_labels_path_read": metadata["evaluator_labels_path_read"],
        },
    }
    write_json_atomic(output_root / "policy_input_receipt.json", receipt)
    print(
        f"R2 label-blind inputs complete: targets={len(decision_rows)}, "
        f"budgets={receipt['budget_counts']}, mean_views={mean_views:.3f}"
    )
    print(decision_path)
    print(m4_feature_path)


if __name__ == "__main__":
    main()
