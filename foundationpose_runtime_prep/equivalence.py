"""Exact c252 prediction equivalence gate with a diagnostic tolerance layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    EQUIVALENCE_SCHEMA,
    PREP_PROTOCOL_ID,
    PrepError,
    canonical_bytes,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from .manifest import validate_manifest
from .producer import load_protocol
from .results import pose_values, prediction_projection


def _by_id(rows: list[dict[str, Any]], *, side: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in indexed:
            raise PrepError(f"{side} results contain a blank or duplicate item ID")
        prediction_projection(row)
        indexed[item_id] = row
    return indexed


def compare_exact(
    *,
    protocol_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    _, manifest_items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
    )
    expected_ids = [row["item_id"] for row in manifest_items]
    expected_set = set(expected_ids)
    baseline = _by_id(read_jsonl(baseline_path.resolve()), side="Baseline")
    candidate = _by_id(read_jsonl(candidate_path.resolve()), side="Candidate")
    baseline_missing = sorted(expected_set - set(baseline))
    baseline_extra = sorted(set(baseline) - expected_set)
    candidate_missing = sorted(expected_set - set(candidate))
    candidate_extra = sorted(set(candidate) - expected_set)
    common = [
        item_id
        for item_id in expected_ids
        if item_id in baseline and item_id in candidate
    ]
    tolerance = protocol["equivalence"]["diagnostic_numeric_tolerance"]
    item_reports: list[dict[str, Any]] = []
    for item_id in common:
        left = prediction_projection(baseline[item_id])
        right = prediction_projection(candidate[item_id])
        left_pose = pose_values(left["predicted_model_to_camera_pose_m"])
        right_pose = pose_values(right["predicted_model_to_camera_pose_m"])
        pose_max = max(abs(a - b) for a, b in zip(left_pose, right_pose))
        score_diff = abs(
            float(left["foundationpose_top_score"])
            - float(right["foundationpose_top_score"])
        )
        margin_diff = abs(
            float(left["foundationpose_top_score_margin"])
            - float(right["foundationpose_top_score_margin"])
        )
        discrete_equal = all(
            left[field] == right[field]
            for field in (
                "selected_candidate_index",
                "candidate_limit",
                "pose_hypothesis_count",
            )
        )
        bitwise_equal = canonical_bytes(left) == canonical_bytes(right)
        within_tolerance = (
            discrete_equal
            and pose_max <= tolerance["pose_abs"]
            and score_diff <= tolerance["score_abs"]
            and margin_diff <= tolerance["margin_abs"]
        )
        item_reports.append(
            {
                "item_id": item_id,
                "bitwise_equal": bitwise_equal,
                "within_numeric_tolerance": within_tolerance,
                "pose_max_abs_difference": pose_max,
                "top_score_abs_difference": score_diff,
                "margin_abs_difference": margin_diff,
                "selection_and_candidate_counts_equal": discrete_equal,
            }
        )
    coverage_exact = not any(
        (baseline_missing, baseline_extra, candidate_missing, candidate_extra)
    ) and len(baseline) == len(candidate) == len(expected_ids)
    all_bitwise = coverage_exact and all(row["bitwise_equal"] for row in item_reports)
    all_numeric = coverage_exact and all(
        row["within_numeric_tolerance"] for row in item_reports
    )
    report = {
        "schema_version": EQUIVALENCE_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "baseline_results_sha256": sha256_file(baseline_path.resolve()),
        "candidate_results_sha256": sha256_file(candidate_path.resolve()),
        "expected_item_count": len(expected_ids),
        "compared_item_count": len(item_reports),
        "coverage": {
            "exact": coverage_exact,
            "baseline_missing": baseline_missing,
            "baseline_extra": baseline_extra,
            "candidate_missing": candidate_missing,
            "candidate_extra": candidate_extra,
        },
        "diagnostic_numeric_tolerance": tolerance,
        "all_predictions_bitwise_equal": all_bitwise,
        "all_predictions_within_numeric_tolerance": all_numeric,
        "passed_exact_guard": all_bitwise,
        "tolerance_cannot_promote_drift": True,
        "items": item_reports,
        "accuracy_claim": "unavailable-label-free-equivalence-only",
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    write_json_atomic(output_path, report)
    return report
