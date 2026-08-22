#!/usr/bin/env python3
"""Build label-blind M3 decisions and M4 inputs for sealed Photoneo data."""

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

import evaluate_m2
import fit_m4_voi
import run_m3_r1
from evaluate_m1 import load_object_evaluation_data, load_official_models
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
EXPECTED_TARGETS = 150
EXPECTED_PREDICTIONS = 712
VIEW_BUDGETS = (1, 3, 5)
FORBIDDEN_PREDICTION_FIELDS = {
    "gt_model_to_camera_pose_m",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    sealed = repo_root / "artifacts" / "r1" / "sealed_photoneo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=sealed / "sealed_contract.json")
    parser.add_argument("--predictions", type=Path, default=sealed / "predictions.jsonl")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")),
    )
    parser.add_argument(
        "--m3-contract",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m3_r1" / "frozen_policy.json",
    )
    parser.add_argument(
        "--m3-model",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m3_r1" / "frozen_policy.joblib",
    )
    parser.add_argument("--output-root", type=Path, default=sealed)
    parser.add_argument(
        "--m4-features-output",
        type=Path,
        default=repo_root
        / "artifacts"
        / "r1"
        / "m4_r1"
        / "sealed_photoneo_preacquisition_features.jsonl",
    )
    return parser.parse_args()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _contract_input_paths(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "artifacts" / "r1" / "sealed_photoneo"
    return {
        "inference_manifest": root / "inference_manifest.jsonl",
        "target_manifest": root / "target_manifest.jsonl",
        "groups_inference": root / "groups_inference.jsonl",
        "protocol": repo_root / "protocols" / "poseloop_r1_protocol.json",
    }


def validate_unopened_contract(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    open_receipt = paths["inference_manifest"].resolve().parent / "sealed_open_receipt.json"
    if open_receipt.exists():
        raise RuntimeError(f"Sealed labels have an open receipt: {open_receipt}")
    if (
        contract.get("status") != "sealed_unopened"
        or bool(contract.get("labels_opened"))
        or int(contract.get("evaluation_invocation_count", -1)) != 0
    ):
        raise RuntimeError("Sealed contract is no longer unopened")
    if int(contract.get("target_count", -1)) != EXPECTED_TARGETS:
        raise ValueError("Unexpected sealed target count")
    if int(contract.get("inference_sample_count", -1)) != EXPECTED_PREDICTIONS:
        raise ValueError("Unexpected sealed inference sample count")
    for name in ("inference_manifest", "target_manifest", "groups_inference"):
        path = paths[name].resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(contract["files"][name]["sha256"]):
            raise RuntimeError(f"Sealed {name} hash changed")
    protocol_path = paths["protocol"].resolve()
    if sha256_file(protocol_path) != str(contract["protocol"]["sha256"]):
        raise RuntimeError("Sealed protocol hash changed")


def load_predictions(
    path: Path, expected_ids: set[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    if not rows or rows[0].get("record_type") != "r1_sealed_inference_metadata":
        raise ValueError("Sealed prediction metadata is missing")
    metadata = rows[0]
    if (
        bool(metadata.get("evaluator_labels_path_read"))
        or bool(metadata.get("evaluator_pose_or_error_computed"))
        or int(metadata.get("sample_count", -1)) != EXPECTED_PREDICTIONS
    ):
        raise RuntimeError("Prediction metadata violates label-blind contract")
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if row.get("record_type") != "r1_sealed_prediction":
            raise ValueError("Unexpected sealed prediction record")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in predictions:
            raise ValueError(f"Missing or duplicate prediction ID: {sample_id!r}")
        if bool(row.get("evaluator_label_read")):
            raise RuntimeError(f"Prediction declares evaluator access: {sample_id}")
        forbidden = FORBIDDEN_PREDICTION_FIELDS & set(row)
        if forbidden:
            raise RuntimeError(f"Prediction contains evaluator fields: {sorted(forbidden)}")
        predictions[sample_id] = row
    if set(predictions) != expected_ids:
        raise ValueError(
            "Prediction identities differ from sealed inference manifest: "
            f"missing={len(expected_ids - set(predictions))}, "
            f"extra={len(set(predictions) - expected_ids)}"
        )
    return metadata, predictions


def validate_groups(
    groups: list[dict[str, Any]],
    targets: Mapping[str, dict[str, Any]],
    prediction_ids: set[str],
) -> None:
    if len(groups) != EXPECTED_TARGETS or len(targets) != EXPECTED_TARGETS:
        raise ValueError("Sealed target/group count mismatch")
    seen: set[str] = set()
    for group in groups:
        group_id = str(group.get("group_id", ""))
        target_id = str(group.get("target_sample_id", ""))
        if not group_id or group_id in seen or target_id not in targets:
            raise ValueError(f"Invalid sealed group identity: {group_id}")
        seen.add(group_id)
        views = sorted(group.get("views", []), key=lambda row: int(row["acquisition_rank"]))
        if len(views) != 5 or [int(row["acquisition_rank"]) for row in views] != list(range(5)):
            raise ValueError(f"Sealed group does not contain ranks 0..4: {group_id}")
        if str(views[0]["sample_id"]) != target_id:
            raise ValueError(f"Rank-zero view is not target: {group_id}")
        if any(str(view["sample_id"]) not in prediction_ids for view in views):
            raise ValueError(f"Group references an unknown prediction: {group_id}")


def prepare_views(
    group: Mapping[str, Any], predictions: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
    target_world_to_camera = np.asarray(
        views[0]["camera_world_to_camera_pose_m"], dtype=np.float64
    )
    output = []
    for view in views:
        prediction = predictions[str(view["sample_id"])]
        pose = np.asarray(
            prediction.get("predicted_model_to_camera_pose_m"), dtype=np.float64
        )
        finite_pose = bool(
            prediction.get("status") == "success"
            and pose.shape == (4, 4)
            and np.all(np.isfinite(pose))
        )
        transformed = (
            evaluate_m2.transform_to_target(
                pose,
                np.asarray(view["camera_world_to_camera_pose_m"], dtype=np.float64),
                target_world_to_camera,
            )
            if finite_pose
            else None
        )
        output.append(
            {
                "view": view,
                "prediction": prediction,
                "status": str(prediction.get("status", "invalid_input")),
                "finite_pose": finite_pose,
                "transformed_pose_m": transformed,
            }
        )
    return output


def prefix_result(
    prepared_views: list[dict[str, Any]], budget: int, object_data: Mapping[str, Any]
) -> dict[str, Any]:
    acquired = prepared_views[:budget]
    selected, scores = evaluate_m2.choose_medoid(acquired, dict(object_data))
    finite_pose = selected is not None and bool(selected["finite_pose"])
    return {
        "status": str(selected["status"]) if selected is not None else "no_usable_prediction",
        "finite_pose": finite_pose,
        "selected_sample_id": (
            str(selected["view"]["sample_id"]) if selected is not None else None
        ),
        "selected_acquisition_rank": (
            int(selected["view"]["acquisition_rank"]) if selected is not None else None
        ),
        "predicted_model_to_target_camera_pose_m": (
            np.asarray(selected["transformed_pose_m"], dtype=np.float64).tolist()
            if finite_pose
            else None
        ),
        "selection_scores_normalized_symmetric_mssd": scores,
        "acquired_view_count": budget,
        "usable_view_count": sum(bool(view["finite_pose"]) for view in acquired),
    }


def predict_marginal(
    model: Any, features: Mapping[str, float], names: list[str]
) -> float:
    if set(features) != set(names):
        raise ValueError("M3 feature names differ from frozen model contract")
    matrix = np.asarray([[features[name] for name in names]], dtype=np.float64)
    value = float(model.predict(matrix)[0])
    return _finite(value, "predicted marginal value")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sealed_root = (repo_root / "artifacts" / "r1" / "sealed_photoneo").resolve()
    output_root = args.output_root.resolve()
    if output_root != sealed_root:
        raise ValueError("Sealed policy outputs must stay in the sealed Photoneo root")
    m4_root = (repo_root / "artifacts" / "r1" / "m4_r1").resolve()
    m4_feature_path = args.m4_features_output.resolve()
    if not m4_feature_path.is_relative_to(m4_root):
        raise ValueError("Sealed M4 features must stay under artifacts/r1/m4_r1")

    contract_path = args.contract.resolve()
    predictions_path = args.predictions.resolve()
    m3_contract_path = args.m3_contract.resolve()
    m3_model_path = args.m3_model.resolve()
    dataset_root = args.dataset_root.resolve()
    for path in (contract_path, predictions_path, m3_contract_path, m3_model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_paths = _contract_input_paths(repo_root)
    validate_unopened_contract(contract, input_paths)
    manifest_rows = load_jsonl(input_paths["inference_manifest"])
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if len(manifest_ids) != EXPECTED_PREDICTIONS:
        raise ValueError("Sealed inference manifest identity count changed")
    metadata, predictions = load_predictions(predictions_path, manifest_ids)
    target_rows = load_jsonl(input_paths["target_manifest"])
    targets = {str(row["sample_id"]): row for row in target_rows}
    groups = load_jsonl(input_paths["groups_inference"])
    validate_groups(groups, targets, set(predictions))

    m3_contract = json.loads(m3_contract_path.read_text(encoding="utf-8"))
    if m3_contract.get("status") != "frozen_for_single_sealed_validation":
        raise RuntimeError("M3-R1 model is not frozen for sealed validation")
    if sha256_file(m3_model_path) != str(m3_contract["model"]["sha256"]):
        raise RuntimeError("Frozen M3-R1 model hash changed")
    frozen_m3 = joblib.load(m3_model_path)
    names = {
        stage: [str(name) for name in frozen_m3["feature_names"][stage]]
        for stage in ("k1", "k3")
    }
    if names != {
        stage: [str(name) for name in m3_contract["feature_names"][stage]]
        for stage in ("k1", "k3")
    }:
        raise RuntimeError("M3-R1 joblib/JSON feature contracts differ")
    thresholds = {stage: float(frozen_m3["thresholds"][stage]) for stage in ("k1", "k3")}
    prediction_sources = {"m1": predictions, "m2": predictions}

    model_params, model_info = load_official_models(dataset_root)
    object_data_cache: dict[int, dict[str, Any]] = {}
    cad_cache: dict[int, dict[str, Any]] = {}
    decision_rows: list[dict[str, Any]] = []
    m4_rows: list[dict[str, Any]] = []
    for index, group in enumerate(sorted(groups, key=lambda row: str(row["group_id"])), start=1):
        group_id = str(group["group_id"])
        object_id = int(group["object_id"])
        target_id = str(group["target_sample_id"])
        if object_id not in object_data_cache:
            object_data_cache[object_id] = load_object_evaluation_data(
                object_id, model_params, model_info
            )
            cad_cache[object_id] = fit_m4_voi.load_cad_geometry(dataset_root, object_id)
        prepared = prepare_views(group, predictions)
        prefix = {
            budget: prefix_result(prepared, budget, object_data_cache[object_id])
            for budget in VIEW_BUDGETS
        }
        k1_features = run_m3_r1.extract_r1_prefix_features(
            group, prefix[1], 1, prediction_sources
        )
        k3_features = run_m3_r1.extract_r1_prefix_features(
            group, prefix[3], 3, prediction_sources
        )
        p1 = predict_marginal(frozen_m3["models"]["k1"], k1_features, names["k1"])
        p3 = predict_marginal(frozen_m3["models"]["k3"], k3_features, names["k3"])
        budget = 1 if p1 <= thresholds["k1"] else (5 if p3 > thresholds["k3"] else 3)
        track_id = str(group.get("oracle_association", {}).get("track_id", ""))
        if not track_id:
            raise ValueError(f"Sealed group lacks association identity: {group_id}")
        decision_rows.append(
            {
                "record_type": "r1_sealed_m3_decision",
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "object_id": object_id,
                "physical_instance_id": track_id,
                "target_sample_id": target_id,
                "k1_predicted_marginal_value": p1,
                "k3_predicted_marginal_value": p3,
                "thresholds": thresholds,
                "selected_budget": budget,
                "k1_features": k1_features,
                "k3_features": k3_features,
                "prefix_results": {str(key): value for key, value in prefix.items()},
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
        if index % 25 == 0 or index == len(groups):
            print(f"label-blind M3/M4 inputs: {index}/{len(groups)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    m4_feature_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path = output_root / "m3_decisions.jsonl"
    write_jsonl_atomic(decision_path, decision_rows)
    write_jsonl_atomic(m4_feature_path, m4_rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1 sealed Photoneo label-blind policy inputs",
        "status": "complete",
        "target_count": len(decision_rows),
        "m4_candidate_row_count": len(m4_rows),
        "budget_counts": {
            str(budget): sum(row["selected_budget"] == budget for row in decision_rows)
            for budget in VIEW_BUDGETS
        },
        "mean_m3_views": float(np.mean([row["selected_budget"] for row in decision_rows])),
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
            "m3_contract": {"path": str(m3_contract_path), "sha256": sha256_file(m3_contract_path)},
            "m3_model": {"path": str(m3_model_path), "sha256": sha256_file(m3_model_path)},
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
        f"sealed label-blind inputs complete: targets={len(decision_rows)}, "
        f"budgets={receipt['budget_counts']}, mean_views={receipt['mean_m3_views']:.3f}"
    )
    print(decision_path)
    print(m4_feature_path)


if __name__ == "__main__":
    main()
