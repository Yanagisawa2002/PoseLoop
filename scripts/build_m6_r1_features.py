#!/usr/bin/env python3
"""Build leakage-separated M6-R1 inference features and evaluator labels.

The feature stream is computed from the exact frozen R1 multi-view output and
signals available at inference time.  Outcome fields remain in a separate
label stream and are never exposed to the risk-model feature matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
EXPECTED_TARGET_COUNT = 300
M4_SLOTS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    r1 = repo_root / "artifacts" / "r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-output", type=Path, default=r1 / "final_multiview" / "outputs.jsonl"
    )
    parser.add_argument(
        "--final-contract",
        type=Path,
        default=r1 / "final_multiview" / "frozen_contract.json",
    )
    parser.add_argument(
        "--m2-groups", type=Path, default=repo_root / "artifacts" / "m2" / "groups.jsonl"
    )
    parser.add_argument(
        "--m1-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m1" / "predictions.jsonl",
    )
    parser.add_argument(
        "--m2-predictions",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "view_predictions.jsonl",
    )
    parser.add_argument(
        "--m4-features", type=Path, default=r1 / "m4_r1" / "preacquisition_features.jsonl"
    )
    parser.add_argument(
        "--m4-occlusion",
        type=Path,
        default=r1 / "m4_r1" / "cad_occlusion_features.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=r1 / "m6_r1")
    return parser.parse_args()


def _unique_index(
    rows: Sequence[dict[str, Any]], key: str, record_type: str | None = None
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if record_type is not None and row.get("record_type") != record_type:
            raise ValueError(f"Unexpected record type while indexing {key}")
        value = str(row[key])
        if value in output:
            raise ValueError(f"Duplicate {key}: {value}")
        output[value] = row
    return output


def _prediction_index(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if row.get("record_type") != "prediction":
                continue
            sample_id = str(row["sample_id"])
            if sample_id in output:
                raise ValueError(f"Duplicate prediction sample ID: {sample_id}")
            output[sample_id] = row
    return output


def _candidate_index(
    rows: Sequence[dict[str, Any]], record_type: str
) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected {record_type} row")
        key = str(row["group_id"]), int(row["candidate_slot"])
        if key in output:
            raise ValueError(f"Duplicate candidate identity: {key}")
        output[key] = row
    return output


def _pose(row: Mapping[str, Any]) -> np.ndarray:
    pose = np.asarray(row.get("predicted_model_to_camera_pose_m"), dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Prediction has no finite 4x4 pose: {row.get('sample_id')}")
    return pose


def _extrinsic(view: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(view["camera_world_to_camera_pose_m"], dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"View has no finite 4x4 extrinsic: {view.get('sample_id')}")
    return value


def transform_pose_to_target_camera(
    candidate_pose_m: np.ndarray,
    candidate_world_to_camera_m: np.ndarray,
    target_world_to_camera_m: np.ndarray,
) -> np.ndarray:
    """Express a candidate-camera model pose in the target-camera frame."""

    result = target_world_to_camera_m @ np.linalg.inv(candidate_world_to_camera_m) @ candidate_pose_m
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("Transformed candidate pose is not finite")
    return result


def rotation_geodesic_radians(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _safe_number(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _prediction_signals(
    prefix: str,
    prediction: Mapping[str, Any],
    mask_fraction: float,
    diameter_m: float,
) -> dict[str, float]:
    pose = _pose(prediction)
    translation = pose[:3, 3]
    depth = float(translation[2])
    lateral = float(np.linalg.norm(translation[:2]))
    norm = float(np.linalg.norm(translation))
    return {
        f"{prefix}_raw_score": _safe_number(prediction.get("foundationpose_top_score")),
        f"{prefix}_raw_score_margin": _safe_number(
            prediction.get("foundationpose_top_score_margin")
        ),
        f"{prefix}_valid_depth_ratio": _safe_number(
            prediction.get("valid_depth_ratio_inside_mask")
        ),
        f"{prefix}_mask_area_fraction": float(mask_fraction),
        f"{prefix}_predicted_depth_over_diameter": depth / diameter_m,
        f"{prefix}_predicted_lateral_over_depth": lateral / max(abs(depth), 1e-9),
        f"{prefix}_predicted_translation_norm_over_diameter": norm / diameter_m,
        f"{prefix}_pose_usable": float(
            prediction.get("status") == "success" and np.all(np.isfinite(pose))
        ),
    }


def build_row(
    final_row: Mapping[str, Any],
    group: Mapping[str, Any],
    predictions: Mapping[str, dict[str, Any]],
    m4_features: Mapping[tuple[str, int], dict[str, Any]],
    m4_occlusion: Mapping[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    group_id = str(final_row["group_id"])
    final = final_row["final"]
    target_sample_id = str(final_row["target_sample_id"])
    selected_sample_id = str(final["selected_sample_id"])
    target_prediction = predictions[target_sample_id]
    selected_prediction = predictions[selected_sample_id]
    views = {int(view["acquisition_rank"]): view for view in group["views"]}
    samples = {str(view["sample_id"]): view for view in group["views"]}
    if target_sample_id not in samples or selected_sample_id not in samples:
        raise ValueError(f"Frozen output references a sample outside group {group_id}")
    if str(group["target_sample_id"]) != target_sample_id:
        raise ValueError(f"Target identity mismatch: {group_id}")

    target_view = samples[target_sample_id]
    selected_view = samples[selected_sample_id]
    reference_features = m4_features[(group_id, 1)]["features"]
    target_depth = float(_pose(target_prediction)[2, 3])
    target_depth_over_diameter = _safe_number(
        reference_features["target_predicted_depth_over_diameter"]
    )
    diameter_m = abs(target_depth / target_depth_over_diameter)
    if not math.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError(f"Could not recover CAD diameter for {group_id}")
    target_mask_fraction = _safe_number(reference_features["target_mask_area_fraction"])
    target_mask_pixels = max(_safe_number(target_prediction["visible_mask_pixel_count"]), 1.0)
    image_pixels = target_mask_pixels / max(target_mask_fraction, 1e-12)
    selected_mask_fraction = _safe_number(selected_prediction["visible_mask_pixel_count"]) / image_pixels

    features: dict[str, float] = {}
    features.update(
        _prediction_signals(
            "selected", selected_prediction, selected_mask_fraction, diameter_m
        )
    )
    features.update(
        _prediction_signals(
            "target", target_prediction, target_mask_fraction, diameter_m
        )
    )
    features.update(
        {
            "acquired_view_count": float(final["acquired_view_count"]),
            "has_ranked_pair": float(final["route"] == "m3_continue_m4_ranked_pair"),
            "selected_is_target": float(selected_sample_id == target_sample_id),
            "policy_budget_is_1": float(final_row["m3_primary_budget_before_m4_collapse"] == 1),
            "policy_budget_is_3": float(final_row["m3_primary_budget_before_m4_collapse"] == 3),
            "policy_budget_is_5": float(final_row["m3_primary_budget_before_m4_collapse"] == 5),
            "policy_k1_predicted_marginal_value": _safe_number(
                final_row["m3_k1_predicted_marginal_value"]
            ),
            "policy_k3_predicted_marginal_value": _safe_number(
                final_row["m3_k3_predicted_marginal_value"]
            ),
            "target_bbox_area_fraction": _safe_number(
                reference_features["target_bbox_area_fraction"]
            ),
            "target_bbox_aspect_log": _safe_number(
                reference_features["target_bbox_aspect_log"]
            ),
            "target_mask_bbox_fill_fraction": _safe_number(
                reference_features["target_mask_bbox_fill_fraction"]
            ),
            "target_depth_iqr_over_diameter": _safe_number(
                reference_features["target_depth_iqr_over_diameter"]
            ),
            "target_depth_std_over_diameter": _safe_number(
                reference_features["target_depth_std_over_diameter"]
            ),
            "cad_extent_min_over_max": _safe_number(
                reference_features["cad_min_over_max_extent"]
            ),
            "cad_surface_area_over_diameter_sq": _safe_number(
                reference_features["cad_surface_area_over_diameter_sq"]
            ),
        }
    )

    pair_defaults = {
        "pair_raw_score_max": features["selected_raw_score"],
        "pair_raw_score_min": features["selected_raw_score"],
        "pair_raw_score_range": 0.0,
        "pair_score_margin_max": features["selected_raw_score_margin"],
        "pair_mask_area_max": features["selected_mask_area_fraction"],
        "pair_mask_area_ratio": 1.0,
        "pair_translation_disagreement_over_diameter": 0.0,
        "pair_depth_disagreement_over_diameter": 0.0,
        "pair_rotation_disagreement_rad": 0.0,
        "ranking_top_predicted_difference": 0.0,
        "ranking_predicted_margin": 0.0,
        "ranking_selected_difference_from_anchor": 0.0,
        "geometry_relative_view_angle_rad": 0.0,
        "geometry_camera_baseline_over_diameter": 0.0,
        "cad_analytic_new_surface_fraction": 0.0,
        "cad_analytic_visible_area_ratio": 0.0,
        "cad_occ_new_surface_fraction": 0.0,
        "cad_occ_lost_surface_fraction": 0.0,
        "cad_occ_union_surface_fraction": 0.0,
        "cad_occ_visible_surface_jaccard": 0.0,
        "cad_occ_overlap_over_target": 0.0,
        "cad_occ_silhouette_ratio": 0.0,
    }
    features.update(pair_defaults)

    candidate_sample_id: str | None = None
    selected_slot = final.get("m4_selected_slot")
    if final["route"] == "m3_continue_m4_ranked_pair":
        if int(final["acquired_view_count"]) != 2 or selected_slot is None:
            raise ValueError(f"Invalid pair route contract: {group_id}")
        slot = int(selected_slot)
        if slot not in M4_SLOTS or slot not in views:
            raise ValueError(f"Invalid selected acquisition slot: {group_id}/{slot}")
        candidate_view = views[slot]
        candidate_sample_id = str(candidate_view["sample_id"])
        candidate_prediction = predictions[candidate_sample_id]
        candidate_mask_fraction = (
            _safe_number(candidate_prediction["visible_mask_pixel_count"]) / image_pixels
        )
        target_pose = _pose(target_prediction)
        candidate_pose_target = transform_pose_to_target_camera(
            _pose(candidate_prediction), _extrinsic(candidate_view), _extrinsic(target_view)
        )
        translation_delta = candidate_pose_target[:3, 3] - target_pose[:3, 3]
        score_values = np.asarray(
            [
                _safe_number(target_prediction.get("foundationpose_top_score")),
                _safe_number(candidate_prediction.get("foundationpose_top_score")),
            ],
            dtype=float,
        )
        margin_values = np.asarray(
            [
                _safe_number(target_prediction.get("foundationpose_top_score_margin")),
                _safe_number(candidate_prediction.get("foundationpose_top_score_margin")),
            ],
            dtype=float,
        )
        mask_values = np.asarray([target_mask_fraction, candidate_mask_fraction], dtype=float)
        analytic = m4_features[(group_id, slot)]["features"]
        occlusion = m4_occlusion[(group_id, slot)]["features"]
        rank_scores = {
            int(key): _safe_number(value)
            for key, value in final_row["m4_predicted_differences_from_slot_2"].items()
        }
        features.update(
            {
                "pair_raw_score_max": float(np.max(score_values)),
                "pair_raw_score_min": float(np.min(score_values)),
                "pair_raw_score_range": float(np.ptp(score_values)),
                "pair_score_margin_max": float(np.max(margin_values)),
                "pair_mask_area_max": float(np.max(mask_values)),
                "pair_mask_area_ratio": float(
                    np.max(mask_values) / max(np.min(mask_values), 1e-12)
                ),
                "pair_translation_disagreement_over_diameter": float(
                    np.linalg.norm(translation_delta) / diameter_m
                ),
                "pair_depth_disagreement_over_diameter": float(
                    abs(translation_delta[2]) / diameter_m
                ),
                "pair_rotation_disagreement_rad": rotation_geodesic_radians(
                    target_pose, candidate_pose_target
                ),
                "ranking_top_predicted_difference": _safe_number(
                    final_row["m4_top_predicted_difference"]
                ),
                "ranking_predicted_margin": _safe_number(
                    final_row["m4_predicted_rank_margin"]
                ),
                "ranking_selected_difference_from_anchor": rank_scores[slot],
                "geometry_relative_view_angle_rad": _safe_number(
                    analytic["relative_view_angle_rad"]
                ),
                "geometry_camera_baseline_over_diameter": _safe_number(
                    analytic["camera_baseline_over_diameter"]
                ),
                "cad_analytic_new_surface_fraction": _safe_number(
                    analytic["cad_new_surface_fraction"]
                ),
                "cad_analytic_visible_area_ratio": _safe_number(
                    analytic["cad_visible_area_ratio"]
                ),
                "cad_occ_new_surface_fraction": _safe_number(
                    occlusion["occ_new_surface_fraction"]
                ),
                "cad_occ_lost_surface_fraction": _safe_number(
                    occlusion["occ_lost_surface_fraction"]
                ),
                "cad_occ_union_surface_fraction": _safe_number(
                    occlusion["occ_union_surface_fraction"]
                ),
                "cad_occ_visible_surface_jaccard": _safe_number(
                    occlusion["occ_visible_surface_jaccard"]
                ),
                "cad_occ_overlap_over_target": _safe_number(
                    occlusion["occ_overlap_over_target"]
                ),
                "cad_occ_silhouette_ratio": _safe_number(
                    occlusion["occ_silhouette_ratio"]
                ),
            }
        )
        if selected_sample_id not in {target_sample_id, candidate_sample_id}:
            raise ValueError(f"Final pose was reselected outside acquired pair: {group_id}")
    elif selected_slot is not None or selected_sample_id != target_sample_id:
        raise ValueError(f"Invalid target-only route contract: {group_id}")

    if any(not math.isfinite(float(value)) for value in features.values()):
        raise ValueError(f"Non-finite M6-R1 feature in {group_id}")
    feature_row = {
        "record_type": "m6_r1_inference_features",
        "schema_version": SCHEMA_VERSION,
        "group_id": group_id,
        "selected_sample_id": selected_sample_id,
        "acquired_candidate_sample_id": candidate_sample_id,
        "features": dict(sorted(features.items())),
    }
    label_row = None
    if "joint_success" in final:
        label_row = {
            "record_type": "m6_r1_evaluator_label",
            "schema_version": SCHEMA_VERSION,
            "group_id": group_id,
            "object_id": int(final_row["object_id"]),
            "physical_instance_id": str(final_row["physical_instance_id"]),
            "target_visibility_bin": str(final_row["target_visibility_bin"]),
            "y_failure": int(not bool(final["joint_success"])),
        }
    return feature_row, label_row


def validate_separation(
    feature_rows: Sequence[dict[str, Any]], label_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    forbidden_tokens = (
        "object_id",
        "physical_instance",
        "visibility",
        "joint_success",
        "normalized_mssd",
        "mspd_px",
        "failure",
        "ground_truth",
        "pose_error",
        "oracle",
    )
    feature_names = sorted(feature_rows[0]["features"])
    consistent = all(sorted(row["features"]) == feature_names for row in feature_rows)
    forbidden = [
        name for name in feature_names if any(token in name.lower() for token in forbidden_tokens)
    ]
    identities_match = {row["group_id"] for row in feature_rows} == {
        row["group_id"] for row in label_rows
    }
    result = {
        "passed": bool(consistent and not forbidden and identities_match),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "consistent_schema": consistent,
        "forbidden_feature_names": forbidden,
        "feature_label_identities_match": identities_match,
        "evaluation_fields_are_metadata_only": True,
        "pose_reselection_after_risk_allowed": False,
    }
    if not result["passed"]:
        raise ValueError(f"M6-R1 feature separation failed: {result}")
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    inputs = {
        "final_output": args.final_output.resolve(),
        "final_contract": args.final_contract.resolve(),
        "m2_groups": args.m2_groups.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
        "m2_predictions": args.m2_predictions.resolve(),
        "m4_features": args.m4_features.resolve(),
        "m4_occlusion": args.m4_occlusion.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = json.loads(inputs["final_contract"].read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_m6_r1":
        raise RuntimeError("Final multi-view contract is not frozen for M6-R1")
    if bool(contract.get("m6_pose_reselection_allowed")):
        raise RuntimeError("M6-R1 input contract unexpectedly permits pose reselection")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(r1_root):
        raise ValueError("M6-R1 outputs must stay under artifacts/r1")

    final_rows = load_jsonl(inputs["final_output"])
    groups = _unique_index(load_jsonl(inputs["m2_groups"]), "group_id")
    predictions = _prediction_index(
        [inputs["m1_predictions"], inputs["m2_predictions"]]
    )
    m4_features = _candidate_index(
        load_jsonl(inputs["m4_features"]), "m4_r1_preacquisition_feature"
    )
    m4_occlusion = _candidate_index(
        load_jsonl(inputs["m4_occlusion"]), "m4_r1_cad_occlusion_feature"
    )
    if len(final_rows) != EXPECTED_TARGET_COUNT:
        raise ValueError(f"Expected {EXPECTED_TARGET_COUNT} final rows, got {len(final_rows)}")

    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for row in sorted(final_rows, key=lambda item: str(item["group_id"])):
        if row.get("record_type") != "r1_final_multiview_output":
            raise ValueError("Unexpected final multi-view row")
        group_id = str(row["group_id"])
        feature_row, label_row = build_row(
            row, groups[group_id], predictions, m4_features, m4_occlusion
        )
        if label_row is None:
            raise RuntimeError("Development final output unexpectedly lacks evaluator label")
        feature_rows.append(feature_row)
        label_rows.append(label_row)
    audit = validate_separation(feature_rows, label_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / "features.jsonl"
    label_path = output_root / "labels.jsonl"
    write_jsonl_atomic(feature_path, feature_rows)
    write_jsonl_atomic(label_path, label_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M6-R1 feature construction",
        "target_count": len(feature_rows),
        "failure_count": sum(row["y_failure"] for row in label_rows),
        "physical_instance_count": len(
            {row["physical_instance_id"] for row in label_rows}
        ),
        "feature_separation_audit": audit,
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "labels": {"path": str(label_path), "sha256": sha256_file(label_path)},
        },
        "legacy_m6_artifacts_read": False,
    }
    write_json_atomic(output_root / "feature_summary.json", summary)
    print(
        f"M6-R1 features built: targets={len(feature_rows)}, "
        f"failures={summary['failure_count']}, features={audit['feature_count']}"
    )
    print(feature_path)
    print(label_path)


if __name__ == "__main__":
    main()
