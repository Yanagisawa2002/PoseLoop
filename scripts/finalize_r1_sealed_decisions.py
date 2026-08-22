#!/usr/bin/env python3
"""Apply frozen M4/M6 models to label-blind sealed Photoneo inputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

import build_m6_r1_features as m6_features
import evaluate_m2
import run_m4_r1
from build_r1_sealed_policy_inputs import (
    EXPECTED_PREDICTIONS,
    EXPECTED_TARGETS,
    _contract_input_paths,
    load_predictions,
    validate_groups,
    validate_unopened_contract,
)
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
SLOTS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    sealed = r1 / "sealed_photoneo"
    m4 = r1 / "m4_r1"
    m6 = r1 / "m6_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=sealed / "sealed_contract.json")
    parser.add_argument("--predictions", type=Path, default=sealed / "predictions.jsonl")
    parser.add_argument("--m3-decisions", type=Path, default=sealed / "m3_decisions.jsonl")
    parser.add_argument(
        "--policy-input-receipt", type=Path, default=sealed / "policy_input_receipt.json"
    )
    parser.add_argument(
        "--m4-features",
        type=Path,
        default=m4 / "sealed_photoneo_preacquisition_features.jsonl",
    )
    parser.add_argument(
        "--m4-occlusion",
        type=Path,
        default=m4 / "sealed_photoneo_cad_occlusion_features.jsonl",
    )
    parser.add_argument(
        "--m4-occlusion-summary",
        type=Path,
        default=m4 / "sealed_photoneo_cad_occlusion_summary.json",
    )
    parser.add_argument("--m4-contract", type=Path, default=m4 / "frozen_ranker.json")
    parser.add_argument("--m4-model", type=Path, default=m4 / "frozen_ranker.joblib")
    parser.add_argument(
        "--m6-contract", type=Path, default=m6 / "frozen_model_contract.json"
    )
    parser.add_argument("--m6-model", type=Path, default=m6 / "frozen_risk_model.joblib")
    parser.add_argument("--output-root", type=Path, default=sealed)
    return parser.parse_args()


def _unique_index(
    rows: Sequence[dict[str, Any]], key: str, record_type: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected record type for {key}")
        value = str(row[key])
        if not value or value in output:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        output[value] = row
    return output


def _candidate_index(
    rows: Sequence[dict[str, Any]], record_type: str
) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected candidate row: {record_type}")
        key = str(row["group_id"]), int(row["candidate_slot"])
        if key in output:
            raise ValueError(f"Duplicate candidate row: {key}")
        output[key] = row
    return output


def join_m4_rows(
    group_id: str,
    feature_index: Mapping[tuple[str, int], dict[str, Any]],
    occlusion_index: Mapping[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for slot in SLOTS:
        feature = feature_index[(group_id, slot)]
        occlusion = occlusion_index[(group_id, slot)]
        if int(feature["object_id"]) != int(occlusion["object_id"]):
            raise ValueError(f"M4 object mismatch: {group_id}/{slot}")
        rows.append(
            {
                "group_id": group_id,
                "candidate_slot": slot,
                "features": feature["features"],
                "occlusion_features": occlusion["features"],
            }
        )
    return rows


def rank_slots(
    rows: Sequence[dict[str, Any]],
    models: Sequence[Any],
    frozen_feature_names: Sequence[str],
    minimum_gain: float,
) -> tuple[dict[int, float], int]:
    slot_names = [f"candidate_slot_is_{slot}" for slot in run_m4_r1.NON_ANCHOR_SLOTS]
    if list(frozen_feature_names[-len(slot_names) :]) != slot_names:
        raise ValueError("Frozen M4 slot-one-hot suffix changed")
    base_names = list(frozen_feature_names[: -len(slot_names)])
    anchor_index = next(
        index for index, row in enumerate(rows) if int(row["candidate_slot"]) == 2
    )
    scores = {2: 0.0}
    candidate_indices = [index for index in range(len(rows)) if index != anchor_index]
    matrix = np.asarray(
        [
            run_m4_r1.difference_vector(rows, index, anchor_index, base_names)
            for index in candidate_indices
        ],
        dtype=np.float64,
    )
    member_values = np.asarray([model.predict(matrix) for model in models], dtype=np.float64)
    if member_values.shape != (len(models), len(candidate_indices)):
        raise ValueError("Frozen M4 ensemble returned an unexpected shape")
    values = np.mean(member_values, axis=0)
    for index, value in zip(candidate_indices, values, strict=True):
        scores[int(rows[index]["candidate_slot"])] = float(value)
    if set(scores) != set(SLOTS) or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Frozen M4 score map is incomplete or non-finite")
    best = min(scores, key=lambda slot: (-scores[slot], slot))
    selected = best if scores[best] >= minimum_gain else 2
    return scores, selected


def _selected_pose(
    prediction: Mapping[str, Any],
    view: Mapping[str, Any],
    target_view: Mapping[str, Any],
) -> tuple[bool, list[list[float]] | None]:
    pose = np.asarray(prediction.get("predicted_model_to_camera_pose_m"), dtype=np.float64)
    usable = bool(
        prediction.get("status") == "success"
        and pose.shape == (4, 4)
        and np.all(np.isfinite(pose))
    )
    if not usable:
        return False, None
    transformed = evaluate_m2.transform_to_target(
        pose,
        np.asarray(view["camera_world_to_camera_pose_m"], dtype=np.float64),
        np.asarray(target_view["camera_world_to_camera_pose_m"], dtype=np.float64),
    )
    return True, transformed.tolist()


def build_final_row(
    group: Mapping[str, Any],
    m3: Mapping[str, Any],
    predictions: Mapping[str, dict[str, Any]],
    rank_scores: Mapping[int, float],
    selected_slot: int,
) -> dict[str, Any]:
    views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
    target = views[0]
    budget = int(m3["selected_budget"])
    if budget == 1:
        selected_view = target
        route = "m3_stop_target_only"
        acquired_count = 1
        final_slot = None
    else:
        candidate = views[selected_slot]
        selected_view = max(
            (target, candidate),
            key=lambda view: (
                int(view["visible_mask_pixel_count"]),
                -int(view["acquisition_rank"]),
            ),
        )
        route = "m3_continue_m4_ranked_pair"
        acquired_count = 2
        final_slot = selected_slot
    sample_id = str(selected_view["sample_id"])
    prediction = predictions[sample_id]
    finite_pose, transformed_pose = _selected_pose(prediction, selected_view, target)
    sorted_scores = sorted((float(value) for value in rank_scores.values()), reverse=True)
    return {
        "record_type": "r1_sealed_final_decision",
        "schema_version": SCHEMA_VERSION,
        "group_id": str(group["group_id"]),
        "object_id": int(group["object_id"]),
        "physical_instance_id": str(m3["physical_instance_id"]),
        "target_sample_id": str(group["target_sample_id"]),
        "m3_primary_budget_before_m4_collapse": budget,
        "m3_k1_predicted_marginal_value": float(m3["k1_predicted_marginal_value"]),
        "m3_k3_predicted_marginal_value": float(m3["k3_predicted_marginal_value"]),
        "m4_predicted_differences_from_slot_2": {
            str(slot): float(rank_scores[slot]) for slot in SLOTS
        },
        "m4_top_predicted_difference": sorted_scores[0],
        "m4_predicted_rank_margin": sorted_scores[0] - sorted_scores[1],
        "final": {
            "route": route,
            "acquired_view_count": acquired_count,
            "m4_selected_slot": final_slot,
            "selected_sample_id": sample_id,
            "selected_acquisition_rank": int(selected_view["acquisition_rank"]),
            "predicted_model_to_target_camera_pose_m": transformed_pose,
            "finite_pose": finite_pose,
            "status": str(prediction.get("status", "invalid_input")),
        },
        "evaluator_label_read": False,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sealed_root = (repo_root / "artifacts" / "r1" / "sealed_photoneo").resolve()
    output_root = args.output_root.resolve()
    if output_root != sealed_root:
        raise ValueError("Sealed decisions must stay in the sealed Photoneo root")
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "contract",
            "predictions",
            "m3_decisions",
            "policy_input_receipt",
            "m4_features",
            "m4_occlusion",
            "m4_occlusion_summary",
            "m4_contract",
            "m4_model",
            "m6_contract",
            "m6_model",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract_inputs = _contract_input_paths(repo_root)
    validate_unopened_contract(contract, contract_inputs)
    manifest_rows = load_jsonl(contract_inputs["inference_manifest"])
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if len(manifest_ids) != EXPECTED_PREDICTIONS:
        raise ValueError("Unexpected inference identity count")
    _, predictions = load_predictions(paths["predictions"], manifest_ids)
    targets = {
        str(row["sample_id"]): row for row in load_jsonl(contract_inputs["target_manifest"])
    }
    groups_list = load_jsonl(contract_inputs["groups_inference"])
    validate_groups(groups_list, targets, set(predictions))
    groups = {str(row["group_id"]): row for row in groups_list}

    m3 = _unique_index(
        load_jsonl(paths["m3_decisions"]), "group_id", "r1_sealed_m3_decision"
    )
    if set(m3) != set(groups) or len(m3) != EXPECTED_TARGETS:
        raise ValueError("M3 sealed decision identities are incomplete")
    policy_receipt = json.loads(paths["policy_input_receipt"].read_text(encoding="utf-8"))
    if (
        policy_receipt.get("status") != "complete"
        or bool(policy_receipt.get("evaluator_label_files_opened"))
        or sha256_file(paths["m3_decisions"])
        != str(policy_receipt["outputs"]["m3_decisions"]["sha256"])
        or sha256_file(paths["m4_features"])
        != str(policy_receipt["outputs"]["m4_preacquisition_features"]["sha256"])
    ):
        raise RuntimeError("Sealed policy-input receipt failed validation")

    m4_features_index = _candidate_index(
        load_jsonl(paths["m4_features"]), "m4_r1_preacquisition_feature"
    )
    m4_occlusion_index = _candidate_index(
        load_jsonl(paths["m4_occlusion"]), "m4_r1_cad_occlusion_feature"
    )
    expected_candidate_keys = {(group_id, slot) for group_id in groups for slot in SLOTS}
    if set(m4_features_index) != expected_candidate_keys or set(m4_occlusion_index) != expected_candidate_keys:
        raise ValueError("Sealed M4 candidate features are incomplete")
    occlusion_summary = json.loads(
        paths["m4_occlusion_summary"].read_text(encoding="utf-8")
    )
    if bool(occlusion_summary.get("candidate_outcome_stream_read")):
        raise RuntimeError("M4 CAD renderer accessed candidate outcomes")
    if (
        int(occlusion_summary.get("row_count", -1)) != len(expected_candidate_keys)
        or sha256_file(paths["m4_features"])
        != str(occlusion_summary["input_feature_stream"]["sha256"])
    ):
        raise RuntimeError("M4 CAD occlusion provenance failed validation")

    m4_contract = json.loads(paths["m4_contract"].read_text(encoding="utf-8"))
    if m4_contract.get("status") != "frozen_after_development_gate_pass":
        raise RuntimeError("M4-R1 ranker is not frozen")
    if sha256_file(paths["m4_model"]) != str(m4_contract["model"]["sha256"]):
        raise RuntimeError("Frozen M4-R1 model hash changed")
    m4_models = joblib.load(paths["m4_model"])
    if not isinstance(m4_models, list) or len(m4_models) != len(
        m4_contract["model"]["member_seeds"]
    ):
        raise ValueError("Frozen M4 ensemble member count changed")

    final_rows: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        joined = join_m4_rows(group_id, m4_features_index, m4_occlusion_index)
        rank_scores, selected_slot = rank_slots(
            joined,
            m4_models,
            [str(name) for name in m4_contract["feature_names"]],
            float(m4_contract["minimum_predicted_gain_to_deviate"]),
        )
        final_rows.append(
            build_final_row(groups[group_id], m3[group_id], predictions, rank_scores, selected_slot)
        )

    m6_contract = json.loads(paths["m6_contract"].read_text(encoding="utf-8"))
    if m6_contract.get("status") != "frozen_for_sealed_validation":
        raise RuntimeError("M6-R1 risk model is not frozen")
    if sha256_file(paths["m6_model"]) != str(m6_contract["model"]["sha256"]):
        raise RuntimeError("Frozen M6-R1 model hash changed")
    frozen_m6 = joblib.load(paths["m6_model"])
    if bool(frozen_m6.get("pose_reselection_allowed")):
        raise RuntimeError("Frozen M6 model unexpectedly permits pose reselection")
    feature_names = [str(name) for name in frozen_m6["feature_names"]]
    if feature_names != [str(name) for name in m6_contract["feature_names"]]:
        raise RuntimeError("M6 joblib/JSON feature contracts differ")

    m6_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for final_row in final_rows:
        group_id = str(final_row["group_id"])
        feature_row, label_row = m6_features.build_row(
            final_row,
            groups[group_id],
            predictions,
            m4_features_index,
            m4_occlusion_index,
        )
        if label_row is not None:
            raise RuntimeError("Label-blind M6 construction unexpectedly returned a label")
        vector = np.asarray(
            [[feature_row["features"][name] for name in feature_names]], dtype=np.float64
        )
        risk = float(frozen_m6["model"].predict_proba(vector)[0, 1])
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"Invalid sealed M6 risk: {group_id}")
        selected_prediction = predictions[str(final_row["final"]["selected_sample_id"])]
        raw_score = float(selected_prediction.get("foundationpose_top_score", 0.0))
        m6_rows.append(feature_row)
        decision_rows.append(
            {
                **final_row,
                "m6_risk": {
                    "failure_probability": risk,
                    "raw_foundationpose_score": raw_score,
                    "raw_score_risk_baseline": -raw_score,
                    "pose_reselected": False,
                    "feature_names": feature_names,
                },
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / "label_blind_decisions.jsonl"
    m6_path = output_root / "m6_inference_features.jsonl"
    write_jsonl_atomic(final_path, decision_rows)
    write_jsonl_atomic(m6_path, m6_rows)
    routes = Counter(str(row["final"]["route"]) for row in decision_rows)
    risks = np.asarray(
        [row["m6_risk"]["failure_probability"] for row in decision_rows], dtype=np.float64
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1 sealed Photoneo frozen label-blind decisions",
        "status": "complete_ready_for_single_open",
        "target_count": len(decision_rows),
        "route_counts": dict(sorted(routes.items())),
        "mean_acquired_view_count": float(
            np.mean([row["final"]["acquired_view_count"] for row in decision_rows])
        ),
        "m6_risk_summary_without_labels": {
            "minimum": float(np.min(risks)),
            "median": float(np.median(risks)),
            "maximum": float(np.max(risks)),
        },
        "evaluator_label_files_opened": False,
        "evaluator_pose_or_error_computed": False,
        "pose_reselection_after_risk": False,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "label_blind_decisions": {"path": str(final_path), "sha256": sha256_file(final_path)},
            "m6_inference_features": {"path": str(m6_path), "sha256": sha256_file(m6_path)},
        },
    }
    write_json_atomic(output_root / "label_blind_decision_receipt.json", receipt)
    print(
        f"sealed decisions ready: targets={len(decision_rows)}, "
        f"routes={dict(routes)}, mean_views={receipt['mean_acquired_view_count']:.3f}"
    )
    print(final_path)


if __name__ == "__main__":
    main()
