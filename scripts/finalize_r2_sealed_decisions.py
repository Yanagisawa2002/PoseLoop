#!/usr/bin/env python3
"""Apply frozen M4-R1 and deployable M6-R2 models to R2 inputs."""

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
from build_r2_sealed_policy_inputs import (
    contract_input_paths,
    load_predictions,
    validate_groups,
    validate_unopened_contract,
)
from finalize_r1_sealed_decisions import (
    _candidate_index,
    _selected_pose,
    join_m4_rows,
    rank_slots,
)
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
SLOTS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    r2 = repo_root / "artifacts" / "r2"
    sealed = r2 / "sealed_photoneo"
    m4 = r2 / "m4_r2"
    m6 = r2 / "m6_r2"
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
    parser.add_argument(
        "--m4-contract", type=Path, default=r1 / "m4_r1" / "frozen_ranker.json"
    )
    parser.add_argument(
        "--m4-model", type=Path, default=r1 / "m4_r1" / "frozen_ranker.joblib"
    )
    parser.add_argument(
        "--m6-contract", type=Path, default=m6 / "frozen_model_contract.json"
    )
    parser.add_argument("--m6-model", type=Path, default=m6 / "frozen_risk_model.joblib")
    parser.add_argument("--output-root", type=Path, default=sealed)
    return parser.parse_args()


def unique_index(
    rows: Sequence[dict[str, Any]], key: str, record_type: str
) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected R2 record type for {key}")
        value = str(row[key])
        if not value or value in output:
            raise ValueError(f"Missing or duplicate R2 {key}: {value!r}")
        output[value] = row
    return output


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
    k3_value = m3.get("k3_predicted_marginal_value")
    return {
        "record_type": "r2_sealed_final_decision",
        "schema_version": SCHEMA_VERSION,
        "group_id": str(group["group_id"]),
        "object_id": int(group["object_id"]),
        "physical_instance_id": str(m3["physical_instance_id"]),
        "target_sample_id": str(group["target_sample_id"]),
        "m3_primary_budget_before_m4_collapse": budget,
        "m3_k1_predicted_marginal_value": float(m3["k1_predicted_marginal_value"]),
        "m3_k3_predicted_marginal_value": (
            float(k3_value) if k3_value is not None else None
        ),
        "m3_k3_score_available": k3_value is not None,
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
    sealed_root = (repo_root / "artifacts" / "r2" / "sealed_photoneo").resolve()
    output_root = args.output_root.resolve()
    if output_root != sealed_root:
        raise ValueError("R2 decisions must stay in the sealed R2 root")
    paths = {
        name: path.resolve()
        for name, path in {
            "contract": args.contract,
            "predictions": args.predictions,
            "m3_decisions": args.m3_decisions,
            "policy_input_receipt": args.policy_input_receipt,
            "m4_features": args.m4_features,
            "m4_occlusion": args.m4_occlusion,
            "m4_occlusion_summary": args.m4_occlusion_summary,
            "m4_contract": args.m4_contract,
            "m4_model": args.m4_model,
            "m6_contract": args.m6_contract,
            "m6_model": args.m6_model,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    input_paths = contract_input_paths(repo_root)
    validate_unopened_contract(contract, input_paths)
    manifest_rows = load_jsonl(input_paths["inference_manifest"])
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    _, predictions = load_predictions(paths["predictions"], manifest_ids)
    targets = {
        str(row["sample_id"]): row for row in load_jsonl(input_paths["target_manifest"])
    }
    groups_list = load_jsonl(input_paths["groups_inference"])
    validate_groups(
        groups_list, targets, set(predictions), int(contract["target_count"])
    )
    groups = {str(row["group_id"]): row for row in groups_list}
    m3 = unique_index(
        load_jsonl(paths["m3_decisions"]), "group_id", "r2_sealed_m3_decision"
    )
    if set(m3) != set(groups):
        raise ValueError("R2 M3 decisions are incomplete")
    policy_receipt = json.loads(paths["policy_input_receipt"].read_text(encoding="utf-8"))
    if (
        policy_receipt.get("status") != "complete"
        or bool(policy_receipt.get("evaluator_label_files_opened"))
        or bool(policy_receipt.get("future_prefix_feature_used_before_acquisition"))
        or sha256_file(paths["m3_decisions"])
        != str(policy_receipt["outputs"]["m3_decisions"]["sha256"])
        or sha256_file(paths["m4_features"])
        != str(policy_receipt["outputs"]["m4_preacquisition_features"]["sha256"])
    ):
        raise RuntimeError("R2 policy-input receipt failed validation")

    m4_features = _candidate_index(
        load_jsonl(paths["m4_features"]), "m4_r1_preacquisition_feature"
    )
    m4_occlusion = _candidate_index(
        load_jsonl(paths["m4_occlusion"]), "m4_r1_cad_occlusion_feature"
    )
    expected_keys = {(group_id, slot) for group_id in groups for slot in SLOTS}
    if set(m4_features) != expected_keys or set(m4_occlusion) != expected_keys:
        raise ValueError("R2 M4 candidate features are incomplete")
    occlusion_summary = json.loads(paths["m4_occlusion_summary"].read_text(encoding="utf-8"))
    if (
        bool(occlusion_summary.get("candidate_outcome_stream_read"))
        or int(occlusion_summary.get("row_count", -1)) != len(expected_keys)
        or sha256_file(paths["m4_features"])
        != str(occlusion_summary["input_feature_stream"]["sha256"])
    ):
        raise RuntimeError("R2 M4 CAD provenance failed validation")
    m4_contract = json.loads(paths["m4_contract"].read_text(encoding="utf-8"))
    if (
        m4_contract.get("status") != "frozen_after_development_gate_pass"
        or sha256_file(paths["m4_model"]) != str(m4_contract["model"]["sha256"])
    ):
        raise RuntimeError("Frozen M4 ranker changed")
    m4_models = joblib.load(paths["m4_model"])
    final_rows = []
    for group_id in sorted(groups):
        joined = join_m4_rows(group_id, m4_features, m4_occlusion)
        scores, slot = rank_slots(
            joined,
            m4_models,
            [str(name) for name in m4_contract["feature_names"]],
            float(m4_contract["minimum_predicted_gain_to_deviate"]),
        )
        final_rows.append(
            build_final_row(groups[group_id], m3[group_id], predictions, scores, slot)
        )

    m6_contract = json.loads(paths["m6_contract"].read_text(encoding="utf-8"))
    if (
        m6_contract.get("status") != "frozen_for_sealed_validation"
        or bool(m6_contract.get("future_or_unacquired_view_feature_used"))
        or sha256_file(paths["m6_model"]) != str(m6_contract["model"]["sha256"])
    ):
        raise RuntimeError("Frozen deployable M6-R2 model changed")
    frozen_m6 = joblib.load(paths["m6_model"])
    if bool(frozen_m6.get("pose_reselection_allowed")):
        raise RuntimeError("M6-R2 unexpectedly permits pose reselection")
    feature_names = [str(name) for name in frozen_m6["feature_names"]]
    if feature_names != [str(name) for name in m6_contract["feature_names"]]:
        raise RuntimeError("M6-R2 model/contract feature order differs")
    if any(name.startswith(("policy_", "pair_", "cad_", "geometry_")) for name in feature_names):
        raise RuntimeError("M6-R2 selected a future or non-final feature")

    feature_rows = []
    decision_rows = []
    for final_row in final_rows:
        group_id = str(final_row["group_id"])
        feature_row, label_row = m6_features.build_row(
            final_row,
            groups[group_id],
            predictions,
            m4_features,
            m4_occlusion,
        )
        if label_row is not None:
            raise RuntimeError("R2 label-blind M6 construction returned a label")
        vector = np.asarray(
            [[feature_row["features"][name] for name in feature_names]], dtype=np.float64
        )
        risk = float(frozen_m6["model"].predict_proba(vector)[0, 1])
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"Invalid R2 M6 risk: {group_id}")
        selected_prediction = predictions[str(final_row["final"]["selected_sample_id"])]
        raw_score = float(selected_prediction.get("foundationpose_top_score", 0.0))
        feature_rows.append(feature_row)
        decision_rows.append(
            {
                **final_row,
                "m6_risk": {
                    "failure_probability": risk,
                    "raw_foundationpose_score": raw_score,
                    "raw_score_risk_baseline": -raw_score,
                    "pose_reselected": False,
                    "feature_names": feature_names,
                    "future_or_unacquired_view_feature_used": False,
                },
            }
        )

    final_path = output_root / "label_blind_decisions.jsonl"
    feature_path = output_root / "m6_inference_features.jsonl"
    write_jsonl_atomic(final_path, decision_rows)
    write_jsonl_atomic(feature_path, feature_rows)
    routes = Counter(str(row["final"]["route"]) for row in decision_rows)
    risks = np.asarray([row["m6_risk"]["failure_probability"] for row in decision_rows])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": contract["protocol_id"],
        "stage": "R2 frozen label-blind decisions",
        "status": "complete_ready_for_single_open",
        "target_count": len(decision_rows),
        "route_counts": dict(sorted(routes.items())),
        "mean_final_acquired_view_count": float(
            np.mean([row["final"]["acquired_view_count"] for row in decision_rows])
        ),
        "mean_m3_evaluation_budget": float(
            np.mean([row["m3_primary_budget_before_m4_collapse"] for row in decision_rows])
        ),
        "m6_risk_summary_without_labels": {
            "minimum": float(np.min(risks)),
            "median": float(np.median(risks)),
            "maximum": float(np.max(risks)),
        },
        "evaluator_label_files_opened": False,
        "evaluator_pose_or_error_computed": False,
        "future_or_unacquired_view_feature_used_by_m6": False,
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
            "m6_inference_features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
        },
    }
    write_json_atomic(output_root / "label_blind_decision_receipt.json", receipt)
    print(
        f"R2 decisions ready: targets={len(decision_rows)}, routes={dict(routes)}, "
        f"mean_final_views={receipt['mean_final_acquired_view_count']:.3f}"
    )
    print(final_path)


if __name__ == "__main__":
    main()
