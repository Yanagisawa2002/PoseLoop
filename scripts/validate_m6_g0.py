#!/usr/bin/env python3
"""Formally validate the complete PoseLoop M6-G0 artifact set."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import build_m6_g0_manifests as manifest_builder
import m6_g0_analysis as analysis
import m6_g0_features as feature_module
import m6_g0_labels as labels
from m1_common import load_jsonl, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
EXPECTED_TARGET_COUNT = 300
EXPECTED_PHYSICAL_INSTANCE_COUNT = 169
EXPECTED_OBJECT_COUNT = 15
EXPECTED_FAILURE_COUNT = 97
EXPECTED_SUCCESS_COUNT = 203
EXPECTED_VIEW_BUDGET = 5

FORMAL_METHODS = manifest_builder.FORMAL_METHODS
ARTIFACT_DIR = Path("artifacts/m6_g0")
FROZEN_DIR = ARTIFACT_DIR / "frozen"
REPORT_PATH = Path("reports/m6_g0/existing_data_confidence_signal_audit.md")
PRECOMPUTED_PATH = Path("precomputed/m6_g0/results.json")
ACCESS_BOUNDARY_DISCLOSURE_PATH = Path(
    "precomputed/m6_g0/access_boundary_disclosure.json"
)
FORMAL_VALIDATION_PATH = ARTIFACT_DIR / "formal_validation.json"

JSONL_ARTIFACTS = {
    "features": ARTIFACT_DIR / "features.jsonl",
    "evaluator_rows": ARTIFACT_DIR / "evaluator_rows.jsonl",
    "oof_predictions": ARTIFACT_DIR / "oof_predictions.jsonl",
    "ablation_oof_predictions": ARTIFACT_DIR / "ablation_oof_predictions.jsonl",
    "per_fold_metrics": ARTIFACT_DIR / "per_fold_metrics.jsonl",
}
JSON_ARTIFACTS = {
    "aggregate_metrics": ARTIFACT_DIR / "aggregate_metrics.json",
    "grouped_bootstrap": ARTIFACT_DIR / "grouped_bootstrap.json",
    "object_jackknife": ARTIFACT_DIR / "object_jackknife.json",
    "feature_ablations": ARTIFACT_DIR / "feature_ablations.json",
    "feature_contributions": ARTIFACT_DIR / "feature_contributions.json",
    "model_selections": ARTIFACT_DIR / "model_selections.json",
    "prediction_validation": ARTIFACT_DIR / "prediction_validation.json",
    "decision_inputs": ARTIFACT_DIR / "decision_inputs.json",
    "decision": ARTIFACT_DIR / "decision.json",
    "qualitative_error_table": ARTIFACT_DIR / "qualitative_error_table.json",
    "chronology": ARTIFACT_DIR / "chronology.json",
    "source_hashes": ARTIFACT_DIR / "source_hashes.json",
    "access_boundary_disclosure": ACCESS_BOUNDARY_DISCLOSURE_PATH,
}
REQUIRED_PLOTS = (
    "failure_prevalence_by_object.png",
    "raw_score_by_outcome.png",
    "risk_coverage_curves.png",
    "calibration_reliability.png",
    "aurc_brier_bootstrap.png",
    "feature_ablation.png",
    "disagreement_vs_actual_error.png",
    "per_object_aurc_difference.png",
    "leave_one_object_out.png",
)

TARGET_ID_KEYS = {
    "target_id",
    "target_sample_id",
    "group_id",
}
TARGET_ID_LIST_KEYS = {"target_ids", "target_sample_ids", "group_ids"}
SAMPLE_ID_KEYS = {
    "sample_id",
    "selected_sample_id",
    "top_score_sample_id",
    "source_sample_id",
    "candidate_sample_id",
}
SAMPLE_ID_LIST_KEYS = {
    "sample_ids",
    "selected_sample_ids",
    "top_score_sample_ids",
    "source_sample_ids",
    "candidate_sample_ids",
}
INSTANCE_ID_KEYS = {"physical_instance_id", "track_id"}
INSTANCE_ID_LIST_KEYS = {"physical_instance_ids", "track_ids"}
DATASET_ID_PREFIX = "xyzibd-"
M3_INPUT_PATTERN = re.compile(
    r"(?:^|[\\/])artifacts[\\/]m3(?:[\\/]|$)|"
    r"(?:^|[\\/])reports[\\/]m3(?:[_\\/]|$)",
    re.IGNORECASE,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing formal result without rewriting it.",
    )
    return parser.parse_args(argv)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard non-finite JSON constant: {value}")


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number at {label}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")


def _strict_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_constant)
    _finite_tree(value, str(path))
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            _finite_tree(row, f"{path}:{line_number}")
            rows.append(row)
    return rows


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required M6-G0 {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty required M6-G0 {label}: {path}")
    return path


def _load_required_artifacts(repo_root: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    for name, relative in (*JSONL_ARTIFACTS.items(), *JSON_ARTIFACTS.items()):
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
            continue
        values[name] = (
            _strict_jsonl(path) if name in JSONL_ARTIFACTS else _strict_json(path)
        )
    fold_path = repo_root / FROZEN_DIR / "fold_manifest.json"
    if not fold_path.is_file():
        missing.append((FROZEN_DIR / "fold_manifest.json").as_posix())
    else:
        values["fold_manifest"] = _strict_json(fold_path)
    for required in (REPORT_PATH, PRECOMPUTED_PATH):
        if not (repo_root / required).is_file():
            missing.append(required.as_posix())
    for plot in REQUIRED_PLOTS:
        path = repo_root / "reports" / "m6_g0" / plot
        if not path.is_file():
            missing.append(path.relative_to(repo_root).as_posix())
    if missing:
        raise FileNotFoundError(
            "M6-G0 downstream artifact set is incomplete; run "
            "`python -B scripts/run_m6_g0.py` first. Missing: "
            + ", ".join(sorted(missing))
        )
    return values


def _assert_tree_close(
    observed: Any,
    expected: Any,
    label: str,
    *,
    atol: float = 1e-12,
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(
                f"{label} mapping keys differ: "
                f"observed={sorted(observed) if isinstance(observed, Mapping) else type(observed)}, "
                f"expected={sorted(expected)}"
            )
        for key in expected:
            _assert_tree_close(
                observed[key], expected[key], f"{label}.{key}", atol=atol
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{label} list length differs")
        for index, item in enumerate(expected):
            _assert_tree_close(observed[index], item, f"{label}[{index}]", atol=atol)
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
    ):
        if not math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=atol,
        ):
            raise ValueError(f"{label} differs: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise ValueError(f"{label} differs: {observed!r} != {expected!r}")


def _feature_names_from_module() -> list[str]:
    raw = feature_module.FEATURE_SPEC
    if "features" in raw and isinstance(raw["features"], Mapping):
        return [str(name) for name in raw["features"]]
    return [str(name) for name in raw]


def _target_id_from_feature(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Feature row lacks a separate metadata mapping")
    target_id = str(metadata.get("target_sample_id") or metadata.get("target_id") or "")
    if not target_id:
        raise ValueError("Feature row has no target ID in metadata")
    return target_id


def _target_id_from_evaluator(row: Mapping[str, Any]) -> str:
    target_id = str(row.get("target_id") or row.get("target_sample_id") or "")
    if not target_id:
        raise ValueError("Evaluator row has no target ID")
    return target_id


def _validate_feature_and_evaluator_rows(
    repo_root: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    allowed_targets = manifest_builder.target_ids_from_m2(repo_root)
    if len(allowed_targets) != EXPECTED_TARGET_COUNT:
        raise ValueError("M2 target universe no longer contains exactly 300 IDs")
    if len(feature_rows) != EXPECTED_TARGET_COUNT:
        raise ValueError(f"Expected 300 feature rows, found {len(feature_rows)}")
    frozen_schema = _strict_json(repo_root / FROZEN_DIR / "feature_schema.json")
    expected_names = list(frozen_schema["features"]["required"])
    if expected_names != _feature_names_from_module():
        raise ValueError(
            "Frozen feature schema differs from m6_g0_features.FEATURE_SPEC"
        )

    indexed_features: dict[str, Mapping[str, Any]] = {}
    prohibited = set(frozen_schema["prohibited_feature_fields"])
    for row_number, row in enumerate(feature_rows, start=1):
        if set(row) != {"features", "metadata"}:
            raise ValueError(
                f"Feature row {row_number} must contain only features and metadata"
            )
        features = row["features"]
        metadata = row["metadata"]
        if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError(f"Malformed feature row {row_number}")
        if set(features) != set(expected_names):
            raise ValueError(f"Feature keys differ at row {row_number}")
        if prohibited & set(features):
            raise ValueError(
                f"Prohibited fields entered feature matrix: {sorted(prohibited & set(features))}"
            )
        for name, value in features.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"Feature {name} at row {row_number} is not finite numeric/null"
                )
        leaked_metadata = {
            "y_failure",
            "correct",
            "normalized_mssd",
            "mspd_px",
            "diagnostic_success",
            "gt_model_to_camera_pose_m",
        } & set(metadata)
        if leaked_metadata:
            raise ValueError(
                f"Evaluator fields leaked into feature metadata: {sorted(leaked_metadata)}"
            )
        target_id = _target_id_from_feature(row)
        if target_id in indexed_features:
            raise ValueError(f"Duplicate feature target: {target_id}")
        if target_id not in allowed_targets:
            raise ValueError(f"Non-M2 target appears in features: {target_id}")
        if metadata.get("frozen_method") != "symmetry_aware_medoid":
            raise ValueError(f"Feature row uses the wrong frozen output: {target_id}")
        if int(metadata.get("view_budget", -1)) != EXPECTED_VIEW_BUDGET:
            raise ValueError(f"Feature row uses the wrong view budget: {target_id}")
        indexed_features[target_id] = row
    if set(indexed_features) != allowed_targets:
        raise ValueError("Feature rows do not exactly cover the M2 target universe")

    validator = getattr(feature_module, "validate_feature_schema", None)
    if callable(validator):
        for row in feature_rows:
            result = validator(row)
            if result is False:
                raise ValueError("m6_g0_features.validate_feature_schema failed")

    expected_evaluator = labels.build_evaluator_rows(
        load_jsonl(repo_root / "artifacts" / "m2" / "metrics.jsonl"),
        feature_rows,
    )
    _assert_tree_close(
        list(evaluator_rows),
        expected_evaluator,
        "evaluator rows",
    )
    indexed_evaluator: dict[str, Mapping[str, Any]] = {}
    for row in evaluator_rows:
        target_id = _target_id_from_evaluator(row)
        if target_id in indexed_evaluator:
            raise ValueError(f"Duplicate evaluator target: {target_id}")
        if target_id not in allowed_targets:
            raise ValueError(f"Non-M2 target appears in evaluator rows: {target_id}")
        if int(row.get("y_failure", -1)) not in {0, 1}:
            raise ValueError(f"Invalid failure label: {target_id}")
        indexed_evaluator[target_id] = row
    if set(indexed_evaluator) != set(indexed_features):
        raise ValueError("Feature/evaluator target sets differ")

    support = labels.label_support_summary(evaluator_rows)
    expected_support = {
        "target_count": EXPECTED_TARGET_COUNT,
        "failure_count": EXPECTED_FAILURE_COUNT,
        "success_count": EXPECTED_SUCCESS_COUNT,
        "physical_instance_count": EXPECTED_PHYSICAL_INSTANCE_COUNT,
        "object_count": EXPECTED_OBJECT_COUNT,
        "passed": True,
    }
    for key, expected in expected_support.items():
        if support.get(key) != expected:
            raise ValueError(
                f"Frozen label support {key} changed: {support.get(key)} != {expected}"
            )
    return indexed_features, indexed_evaluator, support


def _call_fold_validator(
    manifest: Mapping[str, Any],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    validator = getattr(analysis, "validate_fold_manifest", None)
    if not callable(validator):
        raise AttributeError("m6_g0_analysis.validate_fold_manifest is missing")
    signature = inspect.signature(validator)
    parameter_names = set(signature.parameters)
    kwargs: dict[str, Any] = {}
    if "manifest" in parameter_names:
        kwargs["manifest"] = manifest
    if "fold_manifest" in parameter_names:
        kwargs["fold_manifest"] = manifest
    if "evaluator_rows" in parameter_names:
        kwargs["evaluator_rows"] = evaluator_rows
    if "rows" in parameter_names:
        kwargs["rows"] = evaluator_rows
    if kwargs:
        result = validator(**kwargs)
    else:
        result = validator(manifest, evaluator_rows)
    if not isinstance(result, Mapping) or result.get("passed") is not True:
        errors = result.get("errors") if isinstance(result, Mapping) else result
        raise ValueError(f"m6_g0_analysis.validate_fold_manifest failed: {errors}")
    return result


def _validate_fold_manifest(
    manifest: Mapping[str, Any],
    evaluator_by_target: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    evaluator_rows = list(evaluator_by_target.values())
    library_validation = _call_fold_validator(manifest, evaluator_rows)

    hash_function = getattr(analysis, "fold_manifest_hash", None)
    if not callable(hash_function):
        raise AttributeError("m6_g0_analysis.fold_manifest_hash is missing")
    observed_hash = str(hash_function(manifest))
    stored_hash = str(
        manifest.get("manifest_hash")
        or manifest.get("manifest_sha256")
        or manifest.get("fold_manifest_sha256")
        or manifest.get("assignment_sha256")
        or ""
    )
    if not stored_hash or observed_hash != stored_hash:
        raise ValueError("Frozen fold-manifest canonical hash mismatch")

    instance_by_target = {
        target_id: str(row["physical_instance_id"])
        for target_id, row in evaluator_by_target.items()
    }
    outer_fold_count = int(manifest.get("outer_n_splits", 0))
    if outer_fold_count not in {4, 5}:
        raise ValueError(
            "Fold manifest must contain five folds or one documented fallback to four"
        )
    if outer_fold_count == 4 and not manifest.get("five_fold_fallback_reason"):
        raise ValueError("Four-fold fallback lacks the required exact reason")

    outer_rows = manifest.get("outer_assignments")
    if not isinstance(outer_rows, list) or len(outer_rows) != len(evaluator_by_target):
        raise ValueError("Fold manifest outer assignments are incomplete")
    outer_assignment: dict[str, int] = {}
    groups_to_fold: dict[str, set[int]] = defaultdict(set)
    for row in outer_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed outer assignment row")
        target_id = str(row.get("target_id", ""))
        fold_id = int(row.get("outer_fold", -1))
        if (
            target_id not in evaluator_by_target
            or target_id in outer_assignment
            or fold_id not in range(outer_fold_count)
        ):
            raise ValueError(f"Invalid outer assignment: {target_id}/{fold_id}")
        if str(row.get("physical_instance_id")) != instance_by_target[target_id]:
            raise ValueError(f"Outer assignment metadata mismatch: {target_id}")
        outer_assignment[target_id] = fold_id
        groups_to_fold[instance_by_target[target_id]].add(fold_id)
    if set(outer_assignment) != set(evaluator_by_target):
        raise ValueError("Every target must appear in exactly one outer-test fold")
    if any(len(folds) != 1 for folds in groups_to_fold.values()):
        raise ValueError("A physical instance crosses an outer boundary")

    inner_rows = manifest.get("inner_assignments")
    if not isinstance(inner_rows, list):
        raise ValueError("Fold manifest inner assignments are missing")
    for outer_fold in range(outer_fold_count):
        outer_train = {
            target_id
            for target_id, assigned_fold in outer_assignment.items()
            if assigned_fold != outer_fold
        }
        selected = [
            row
            for row in inner_rows
            if isinstance(row, Mapping) and int(row.get("outer_fold", -1)) == outer_fold
        ]
        if len(selected) != len(outer_train):
            raise ValueError(f"Outer fold {outer_fold} inner coverage is incomplete")
        inner_by_target = {
            str(row.get("target_id")): int(row.get("inner_fold", -1))
            for row in selected
        }
        if set(inner_by_target) != outer_train or set(inner_by_target.values()) != set(
            range(4)
        ):
            raise ValueError(f"Outer fold {outer_fold} inner assignments are invalid")
        inner_groups: dict[str, set[int]] = defaultdict(set)
        for target_id, inner_fold in inner_by_target.items():
            inner_groups[instance_by_target[target_id]].add(inner_fold)
        if any(len(folds) != 1 for folds in inner_groups.values()):
            raise ValueError(f"A physical instance crosses inner fold {outer_fold}")
    return outer_assignment, {
        "outer_fold_count": outer_fold_count,
        "inner_fold_count_per_outer": 4,
        "target_count": len(outer_assignment),
        "physical_instance_count": len(set(instance_by_target.values())),
        "canonical_sha256": observed_hash,
        "library_checks": library_validation["checks"],
    }


def _prediction_field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"Prediction row lacks one of {names}")


def _prediction_target(row: Mapping[str, Any]) -> str:
    return str(_prediction_field(row, "target_id", "target_sample_id"))


def _prediction_method(row: Mapping[str, Any]) -> str:
    return str(_prediction_field(row, "method", "method_name"))


def _risk_score(row: Mapping[str, Any]) -> float:
    value = float(_prediction_field(row, "risk_score", "risk"))
    if not math.isfinite(value):
        raise ValueError(f"Invalid prediction risk score: {value}")
    return value


def _failure_probability_or_none(row: Mapping[str, Any]) -> float | None:
    value = row.get(
        "failure_probability",
        row.get("predicted_failure_probability", row.get("probability_failure")),
    )
    if value is None:
        return None
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid predicted failure probability: {probability}")
    return probability


def _validate_oof_predictions(
    predictions: Sequence[Mapping[str, Any]],
    feature_by_target: Mapping[str, Mapping[str, Any]],
    evaluator_by_target: Mapping[str, Mapping[str, Any]],
    outer_assignment: Mapping[str, int],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    expected_count = len(FORMAL_METHODS) * EXPECTED_TARGET_COUNT
    if len(predictions) != expected_count:
        raise ValueError(
            f"OOF predictions have {len(predictions)} rows, expected {expected_count}"
        )
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str]] = set()
    for row in predictions:
        method = _prediction_method(row)
        target_id = _prediction_target(row)
        if method not in FORMAL_METHODS:
            raise ValueError(f"Unknown formal method in OOF predictions: {method}")
        if target_id not in evaluator_by_target:
            raise ValueError(f"Non-M2 target in OOF predictions: {target_id}")
        key = (method, target_id)
        if key in keys:
            raise ValueError(f"Duplicate OOF prediction: {key}")
        keys.add(key)
        risk_score = _risk_score(row)
        probability = _failure_probability_or_none(row)
        feature_values = feature_by_target[target_id].get("features")
        if not isinstance(feature_values, Mapping):
            raise ValueError(
                f"Feature mapping is malformed for OOF target: {target_id}"
            )
        frozen_raw_score = feature_values.get("raw_selected_score")
        if frozen_raw_score is None or not math.isfinite(float(frozen_raw_score)):
            raise ValueError(f"Frozen raw selected score is unavailable: {target_id}")
        stored_raw_score = float(_prediction_field(row, "raw_score"))
        if not math.isfinite(stored_raw_score) or not math.isclose(
            stored_raw_score,
            float(frozen_raw_score),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"OOF raw score differs from frozen feature: {key}")
        if method == "RAW_SCORE_RANK" and probability is not None:
            raise ValueError("RAW_SCORE_RANK must not claim calibrated probabilities")
        if method == "RAW_SCORE_RANK" and not math.isclose(
            risk_score,
            -float(frozen_raw_score),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "RAW_SCORE_RANK risk must be the negative frozen confidence score: "
                f"{target_id}"
            )
        if method != "RAW_SCORE_RANK" and probability is None:
            raise ValueError(f"Calibrated OOF method lacks probability: {method}")
        if method != "RAW_SCORE_RANK" and not math.isclose(
            risk_score,
            float(probability),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Calibrated OOF risk differs from failure probability: {key}"
            )
        if int(_prediction_field(row, "y_failure", "label")) != int(
            evaluator_by_target[target_id]["y_failure"]
        ):
            raise ValueError(f"OOF label mismatch: {key}")
        stored_outer = int(_prediction_field(row, "outer_fold", "fold"))
        if stored_outer != outer_assignment[target_id]:
            raise ValueError(f"OOF row is assigned to the wrong outer fold: {key}")
        by_method[method].append(row)
    if set(by_method) != set(FORMAL_METHODS):
        raise ValueError("OOF predictions do not cover every formal method")
    target_universe = set(evaluator_by_target)
    for method, rows in by_method.items():
        if {_prediction_target(row) for row in rows} != target_universe:
            raise ValueError(f"OOF method does not cover every target: {method}")
    return dict(by_method), {
        "raw_score_rows_linked_to_frozen_feature": EXPECTED_TARGET_COUNT,
        "raw_score_rank_rows_with_frozen_negative_orientation": EXPECTED_TARGET_COUNT,
        "calibrated_rows_with_risk_equal_probability": (
            (len(FORMAL_METHODS) - 1) * EXPECTED_TARGET_COUNT
        ),
        "raw_score_higher_means_more_confident": True,
    }


def _call_prediction_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    function = getattr(analysis, "compute_prediction_metrics", None)
    if not callable(function):
        raise AttributeError("m6_g0_analysis.compute_prediction_metrics is missing")
    labels_values = [int(_prediction_field(row, "y_failure", "label")) for row in rows]
    risk_scores = [_risk_score(row) for row in rows]
    probabilities = [_failure_probability_or_none(row) for row in rows]
    method = _prediction_method(rows[0])
    probability_vector: list[float] | None
    if method == "RAW_SCORE_RANK":
        probability_vector = None
    else:
        if any(value is None for value in probabilities):
            raise ValueError(f"{method} has missing probabilities")
        probability_vector = [
            float(value) for value in probabilities if value is not None
        ]
    result = function(
        labels_values,
        risk_scores,
        failure_probability=probability_vector,
        tie_breaker=[_prediction_target(row) for row in rows],
    )
    if not isinstance(result, Mapping):
        raise ValueError("compute_prediction_metrics did not return a mapping")
    return result


def _metric_entries(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("methods", "by_method", "metrics"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and set(FORMAL_METHODS).issubset(nested):
                return {method: nested[method] for method in FORMAL_METHODS}
            if isinstance(nested, list):
                return _metric_entries(nested)
        if set(FORMAL_METHODS).issubset(value):
            return {method: value[method] for method in FORMAL_METHODS}
    if isinstance(value, list):
        indexed: dict[str, Mapping[str, Any]] = {}
        for row in value:
            if not isinstance(row, Mapping):
                continue
            method = str(row.get("method") or row.get("method_name") or "")
            if method:
                indexed[method] = row
        if set(FORMAL_METHODS).issubset(indexed):
            return {method: indexed[method] for method in FORMAL_METHODS}
    raise ValueError("Could not index aggregate metrics by formal method")


def _strip_metric_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if key
        not in {
            "record_type",
            "schema_version",
            "method",
            "method_name",
            "outer_fold",
            "fold",
        }
    }


def _validate_metric_reconstruction(
    aggregate_metrics: Any,
    per_fold_metrics: Sequence[Mapping[str, Any]],
    by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    aggregate_by_method = _metric_entries(aggregate_metrics)
    reconstructed: dict[str, Mapping[str, Any]] = {}
    raw_aurc = float(aggregate_by_method["RAW_SCORE_RANK"]["aurc"])
    isotonic_aurc = float(aggregate_by_method["SCORE_ISOTONIC"]["aurc"])
    for method in FORMAL_METHODS:
        expected = _call_prediction_metrics(by_method[method])
        observed = _strip_metric_identity(aggregate_by_method[method])
        observed_core = {
            key: value
            for key, value in observed.items()
            if key
            not in {
                "relative_aurc_improvement_vs_raw_score_rank",
                "relative_aurc_improvement_vs_score_isotonic",
            }
        }
        expected_stripped = _strip_metric_identity(expected)
        _assert_tree_close(
            observed_core,
            expected_stripped,
            f"pooled metric reconstruction for {method}",
        )
        expected_vs_raw = analysis.relative_aurc_improvement(
            raw_aurc,
            float(expected["aurc"]),
        )
        expected_vs_isotonic = analysis.relative_aurc_improvement(
            isotonic_aurc,
            float(expected["aurc"]),
        )
        _assert_tree_close(
            observed.get("relative_aurc_improvement_vs_raw_score_rank"),
            expected_vs_raw,
            f"relative AURC vs raw for {method}",
        )
        _assert_tree_close(
            observed.get("relative_aurc_improvement_vs_score_isotonic"),
            expected_vs_isotonic,
            f"relative AURC vs isotonic for {method}",
        )
        reconstructed[method] = expected

    indexed_folds: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in per_fold_metrics:
        method = str(row.get("method") or row.get("method_name") or "")
        fold = int(row.get("outer_fold", row.get("fold", -1)))
        if not method or fold < 0 or (method, fold) in indexed_folds:
            raise ValueError("Malformed or duplicate per-fold metric row")
        indexed_folds[(method, fold)] = row
    prediction_fold_keys = {
        (
            method,
            int(_prediction_field(row, "outer_fold", "fold")),
        )
        for method, rows in by_method.items()
        for row in rows
    }
    if set(indexed_folds) != prediction_fold_keys:
        raise ValueError("Per-fold metrics do not match OOF method/fold coverage")
    for key, stored in indexed_folds.items():
        method, fold = key
        rows = [
            row
            for row in by_method[method]
            if int(_prediction_field(row, "outer_fold", "fold")) == fold
        ]
        expected = _strip_metric_identity(_call_prediction_metrics(rows))
        observed_value = stored.get("metrics", stored)
        if not isinstance(observed_value, Mapping):
            raise ValueError(f"Per-fold metric payload is malformed: {method}/{fold}")
        observed = _strip_metric_identity(observed_value)
        _assert_tree_close(
            observed,
            expected,
            f"fold metric reconstruction for {method}/{fold}",
        )
    return {
        "formal_method_count": len(reconstructed),
        "per_fold_metric_row_count": len(indexed_folds),
        "pooled_and_fold_metrics_exact": True,
    }


def _validate_ablation_metric_reconstruction(
    feature_ablations: Any,
    predictions: Sequence[Mapping[str, Any]],
    evaluator_by_target: Mapping[str, Mapping[str, Any]],
    outer_assignment: Mapping[str, int],
    formal_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reconstruct every ablation metric from persisted per-target OOF rows."""

    if not isinstance(feature_ablations, Mapping) or set(feature_ablations) != set(
        analysis.ABLATION_NAMES
    ):
        raise ValueError("Feature ablations do not match the frozen four-way design")
    expected_row_fields = {
        "ablation",
        "target_id",
        "y_failure",
        "outer_fold",
        "method",
        "risk_score",
        "failure_probability",
    }
    expected_count = len(analysis.ABLATION_NAMES) * len(evaluator_by_target)
    if len(predictions) != expected_count:
        raise ValueError(
            "Ablation OOF predictions have "
            f"{len(predictions)} rows, expected {expected_count}"
        )

    rows_by_ablation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    observed_order: list[tuple[str, str]] = []
    for row in predictions:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("Ablation OOF prediction row schema changed")
        ablation = str(row["ablation"])
        if ablation not in analysis.ABLATION_NAMES:
            raise ValueError(f"Unknown ablation in OOF predictions: {ablation}")
        if _prediction_method(row) != "NESTED_MULTIFEATURE":
            raise ValueError(
                "Ablation OOF evidence must contain only the nested method"
            )
        target_id = _prediction_target(row)
        if target_id not in evaluator_by_target:
            raise ValueError(f"Non-M2 target in ablation OOF predictions: {target_id}")
        key = (ablation, target_id)
        if key in seen:
            raise ValueError(f"Duplicate ablation OOF prediction: {key}")
        seen.add(key)
        observed_order.append(key)
        _risk_score(row)
        if _failure_probability_or_none(row) is None:
            raise ValueError(f"Ablation OOF prediction lacks probability: {key}")
        if int(_prediction_field(row, "y_failure", "label")) != int(
            evaluator_by_target[target_id]["y_failure"]
        ):
            raise ValueError(f"Ablation OOF label mismatch: {key}")
        if int(_prediction_field(row, "outer_fold", "fold")) != int(
            outer_assignment[target_id]
        ):
            raise ValueError(f"Ablation OOF fold mismatch: {key}")
        rows_by_ablation[ablation].append(row)

    target_universe = set(evaluator_by_target)
    if set(rows_by_ablation) != set(analysis.ABLATION_NAMES) or any(
        {_prediction_target(row) for row in rows} != target_universe
        for rows in rows_by_ablation.values()
    ):
        raise ValueError(
            "Ablation OOF evidence does not cover every target exactly once"
        )
    expected_order = [
        (ablation, target_id)
        for ablation in analysis.ABLATION_NAMES
        for target_id in sorted(target_universe)
    ]
    if observed_order != expected_order:
        raise ValueError("Ablation OOF rows are not in frozen deterministic order")

    required_formal_evidence = {
        "RAW_SCORE_RANK",
        "SCORE_ISOTONIC",
        "NESTED_MULTIFEATURE",
    }
    if not required_formal_evidence.issubset(formal_by_method):
        raise ValueError(
            "Formal OOF evidence required by ablation validation is missing"
        )
    formal_nested_by_target = {
        _prediction_target(row): row for row in formal_by_method["NESTED_MULTIFEATURE"]
    }
    if set(formal_nested_by_target) != target_universe:
        raise ValueError(
            "Formal nested OOF evidence does not cover the ablation targets"
        )
    for row in rows_by_ablation["all"]:
        target_id = _prediction_target(row)
        formal_row = formal_nested_by_target[target_id]
        _assert_tree_close(
            _risk_score(row),
            _risk_score(formal_row),
            f"all-feature ablation risk matches formal nested OOF for {target_id}",
        )
        _assert_tree_close(
            _failure_probability_or_none(row),
            _failure_probability_or_none(formal_row),
            f"all-feature ablation probability matches formal nested OOF for {target_id}",
        )

    raw_aurc = float(
        _call_prediction_metrics(formal_by_method["RAW_SCORE_RANK"])["aurc"]
    )
    isotonic_aurc = float(
        _call_prediction_metrics(formal_by_method["SCORE_ISOTONIC"])["aurc"]
    )
    expected_folds = set(outer_assignment.values())
    reconstructed_fold_count = 0
    for ablation in analysis.ABLATION_NAMES:
        payload = feature_ablations[ablation]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Ablation payload is malformed: {ablation}")
        rows = rows_by_ablation[ablation]
        expected_aggregate = _call_prediction_metrics(rows)
        observed_aggregate_value = payload.get("aggregate_metrics")
        if not isinstance(observed_aggregate_value, Mapping):
            raise ValueError(f"Ablation aggregate metrics are malformed: {ablation}")
        observed_aggregate = _strip_metric_identity(observed_aggregate_value)
        observed_core = {
            key: value
            for key, value in observed_aggregate.items()
            if key
            not in {
                "relative_aurc_improvement_vs_raw_score_rank",
                "relative_aurc_improvement_vs_score_isotonic",
            }
        }
        _assert_tree_close(
            observed_core,
            _strip_metric_identity(expected_aggregate),
            f"ablation pooled metric reconstruction for {ablation}",
        )
        expected_vs_raw = analysis.relative_aurc_improvement(
            raw_aurc, float(expected_aggregate["aurc"])
        )
        expected_vs_isotonic = analysis.relative_aurc_improvement(
            isotonic_aurc, float(expected_aggregate["aurc"])
        )
        _assert_tree_close(
            observed_aggregate.get("relative_aurc_improvement_vs_raw_score_rank"),
            expected_vs_raw,
            f"ablation relative AURC vs raw for {ablation}",
        )
        _assert_tree_close(
            observed_aggregate.get("relative_aurc_improvement_vs_score_isotonic"),
            expected_vs_isotonic,
            f"ablation relative AURC vs isotonic for {ablation}",
        )

        stored_fold_rows = payload.get("per_fold_metrics")
        if not isinstance(stored_fold_rows, list):
            raise ValueError(f"Ablation fold metrics are malformed: {ablation}")
        indexed_folds: dict[int, Mapping[str, Any]] = {}
        for stored in stored_fold_rows:
            if not isinstance(stored, Mapping):
                raise ValueError(f"Ablation fold row is malformed: {ablation}")
            fold = int(stored.get("outer_fold", stored.get("fold", -1)))
            if (
                _prediction_method(stored) != "NESTED_MULTIFEATURE"
                or fold not in expected_folds
                or fold in indexed_folds
            ):
                raise ValueError(f"Duplicate or invalid ablation fold row: {ablation}")
            indexed_folds[fold] = stored
        if set(indexed_folds) != expected_folds:
            raise ValueError(f"Ablation fold metrics omit a frozen fold: {ablation}")
        for fold, stored in indexed_folds.items():
            fold_predictions = [
                row
                for row in rows
                if int(_prediction_field(row, "outer_fold", "fold")) == fold
            ]
            expected_fold = _strip_metric_identity(
                _call_prediction_metrics(fold_predictions)
            )
            observed_fold_value = stored.get("metrics")
            if not isinstance(observed_fold_value, Mapping):
                raise ValueError(
                    f"Ablation fold metric payload is malformed: {ablation}/{fold}"
                )
            _assert_tree_close(
                _strip_metric_identity(observed_fold_value),
                expected_fold,
                f"ablation fold metric reconstruction for {ablation}/{fold}",
            )
            reconstructed_fold_count += 1

    return {
        "ablation_count": len(rows_by_ablation),
        "prediction_row_count": len(predictions),
        "per_fold_metric_row_count": reconstructed_fold_count,
        "each_target_once_per_ablation": True,
        "all_ablation_matches_formal_nested_oof": True,
        "pooled_and_fold_metrics_exact": True,
    }


def _validate_bootstrap_and_robustness(
    grouped_bootstrap: Any,
    object_jackknife: Any,
    predictions: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(grouped_bootstrap, Mapping):
        raise ValueError("Grouped bootstrap artifact must be a mapping")
    resamples = int(
        grouped_bootstrap.get("n_resamples")
        or grouped_bootstrap.get("resample_count")
        or grouped_bootstrap.get("bootstrap_resamples")
        or 0
    )
    if resamples < 2000:
        raise ValueError("Grouped bootstrap uses fewer than 2,000 resamples")
    serialized = json.dumps(grouped_bootstrap, sort_keys=True).lower()
    if "physical_instance" not in serialized:
        raise ValueError(
            "Grouped bootstrap does not declare physical-instance resampling"
        )
    evaluator_by_target = {
        _target_id_from_evaluator(row): row for row in evaluator_rows
    }
    targets = sorted(evaluator_by_target)
    prediction_index = {
        (_prediction_method(row), _prediction_target(row)): row for row in predictions
    }
    y_failure = [int(evaluator_by_target[target]["y_failure"]) for target in targets]
    physical_instances = [
        str(evaluator_by_target[target]["physical_instance_id"]) for target in targets
    ]
    object_ids = [evaluator_by_target[target]["object_id"] for target in targets]
    risk_scores = {
        method: [_risk_score(prediction_index[(method, target)]) for target in targets]
        for method in FORMAL_METHODS
    }
    probabilities: dict[str, list[float] | None] = {}
    for method in FORMAL_METHODS:
        values = [
            _failure_probability_or_none(prediction_index[(method, target)])
            for target in targets
        ]
        probabilities[method] = (
            None
            if method == "RAW_SCORE_RANK"
            else [float(value) for value in values if value is not None]
        )
    expected_bootstrap = analysis.grouped_bootstrap(
        y_failure,
        risk_scores,
        probabilities,
        physical_instances,
        targets,
        n_resamples=resamples,
        seed=int(grouped_bootstrap.get("seed", analysis.BOOTSTRAP_SEED)),
    )
    _assert_tree_close(
        grouped_bootstrap,
        expected_bootstrap,
        "grouped bootstrap reconstruction",
    )
    expected_jackknife = analysis.object_robustness_analysis(
        y_failure,
        risk_scores["NESTED_MULTIFEATURE"],
        risk_scores["SCORE_ISOTONIC"],
        object_ids,
        physical_instances,
        targets,
    )
    _assert_tree_close(
        object_jackknife,
        expected_jackknife,
        "object robustness reconstruction",
    )
    if not isinstance(object_jackknife, (Mapping, list)):
        raise ValueError("Object-jackknife artifact is malformed")
    return {
        "resample_count": resamples,
        "resampling_unit": "physical_instance_id",
        "bootstrap_exactly_reconstructed": True,
        "object_robustness_exactly_reconstructed": True,
    }


def _stable_disagreement_benefit(feature_ablations: Mapping[str, Any]) -> bool:
    score_only = feature_ablations["score_only"]
    score_disagreement = feature_ablations["score_disagreement"]
    pooled_improvement = float(score_only["aggregate_metrics"]["aurc"]) - float(
        score_disagreement["aggregate_metrics"]["aurc"]
    )
    score_folds = {
        int(row["outer_fold"]): float(row["metrics"]["aurc"])
        for row in score_only["per_fold_metrics"]
    }
    disagreement_folds = {
        int(row["outer_fold"]): float(row["metrics"]["aurc"])
        for row in score_disagreement["per_fold_metrics"]
    }
    if set(score_folds) != set(disagreement_folds) or not score_folds:
        raise ValueError("Ablation fold sets differ or are empty")
    positive_folds = sum(
        disagreement_folds[fold] < score_folds[fold] for fold in sorted(score_folds)
    )
    required_positive = int(math.ceil(0.60 * len(score_folds)))
    return bool(pooled_improvement > 0.0 and positive_folds >= required_positive)


def _decision_classification(value: Mapping[str, Any]) -> str:
    for key in ("classification", "decision", "result", "status"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate in {
            "SIGNAL GO",
            "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED",
            "NO-GO",
        }:
            return candidate
    raise ValueError("Machine-readable decision has no formal classification")


def _independent_relative_aurc_improvement(
    baseline_aurc: float | None,
    candidate_aurc: float | None,
) -> float | None:
    """Reconstruct the frozen relative-AURC definition without analysis helpers."""

    if baseline_aurc is None or candidate_aurc is None or baseline_aurc <= 0.0:
        return None
    return float((baseline_aurc - candidate_aurc) / baseline_aurc)


def _independent_decision_classification(
    go_conditions: Mapping[str, bool],
    weak_conditions: Mapping[str, bool],
    hard_validity_conditions: Mapping[str, bool],
) -> str:
    """Apply the frozen precedence without calling the production decision code."""

    if not all(hard_validity_conditions.values()):
        return "NO-GO"
    if all(go_conditions.values()):
        return "SIGNAL GO"
    if all(weak_conditions.values()):
        return "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED"
    return "NO-GO"


def _reconstruct_frozen_decision(
    aggregate_metrics: Mapping[str, Any],
    grouped_bootstrap: Mapping[str, Any],
    object_robustness: Mapping[str, Any],
    *,
    stable_disagreement_benefit: bool,
    leakage_checks_passed: bool,
    label_support_passed: bool,
    oof_complete: bool,
    m3_access_boundary_passed: bool,
) -> dict[str, Any]:
    """Independently reconstruct every frozen M6-G0 decision threshold."""

    candidate = aggregate_metrics["NESTED_MULTIFEATURE"]
    raw = aggregate_metrics["RAW_SCORE_RANK"]
    isotonic = aggregate_metrics["SCORE_ISOTONIC"]
    candidate_aurc = float(candidate["aurc"])
    raw_aurc = float(raw["aurc"])
    isotonic_aurc = float(isotonic["aurc"])
    improvement_raw = _independent_relative_aurc_improvement(
        raw_aurc,
        candidate_aurc,
    )
    improvement_isotonic = _independent_relative_aurc_improvement(
        isotonic_aurc,
        candidate_aurc,
    )
    bootstrap_ci = grouped_bootstrap["paired_differences"][
        "aurc_improvement_vs_score_isotonic"
    ]
    ci_excludes_zero_positive = (
        bootstrap_ci.get("lower") is not None and float(bootstrap_ci["lower"]) > 0.0
    )
    risk_005_coverage = candidate.get("predicted_risk_le_0_05_coverage")
    risk_005_empirical = candidate.get("predicted_risk_le_0_05_empirical_failure_rate")
    risk_005_count = candidate.get("predicted_risk_le_0_05_count")
    candidate_brier = candidate.get("brier")
    isotonic_brier = isotonic.get("brier")
    if candidate_brier is None or isotonic_brier is None:
        brier_condition = False
        brier_relative_change = None
    elif float(isotonic_brier) > 0.0:
        brier_relative_change = (
            float(candidate_brier) - float(isotonic_brier)
        ) / float(isotonic_brier)
        brier_condition = brier_relative_change <= 0.02
    else:
        brier_relative_change = 0.0 if float(candidate_brier) == 0.0 else math.inf
        brier_condition = float(candidate_brier) == 0.0

    leave_one_object_out = object_robustness.get("leave_one_object_out", [])
    all_object_jackknives_non_negative = bool(leave_one_object_out) and all(
        bool(row.get("improvement_remains_non_negative"))
        for row in leave_one_object_out
    )
    not_object_driven = not bool(object_robustness.get("object_driven", True))

    go_conditions = {
        "auroc_at_least_0_75": candidate.get("auroc") is not None
        and float(candidate["auroc"]) >= 0.75,
        "relative_aurc_improvement_vs_raw_at_least_0_10": (
            improvement_raw is not None and improvement_raw >= 0.10
        ),
        "relative_aurc_improvement_vs_isotonic_at_least_0_05": (
            improvement_isotonic is not None and improvement_isotonic >= 0.05
        ),
        "bootstrap_isotonic_aurc_improvement_ci_excludes_zero": (
            ci_excludes_zero_positive
        ),
        "predicted_risk_0_05_coverage_at_least_0_40": (
            risk_005_coverage is not None and float(risk_005_coverage) >= 0.40
        ),
        "predicted_risk_0_05_empirical_failure_at_most_0_05": (
            risk_005_empirical is not None and float(risk_005_empirical) <= 0.05
        ),
        "predicted_risk_0_05_accepts_at_least_30": (
            risk_005_count is not None and int(risk_005_count) >= 30
        ),
        "brier_not_more_than_2_percent_worse_than_isotonic": brier_condition,
        "ece_at_most_0_05": candidate.get("ece") is not None
        and float(candidate["ece"]) <= 0.05,
        "every_object_jackknife_non_negative": all_object_jackknives_non_negative,
        "not_object_or_instance_driven": not_object_driven,
        "leakage_checks_passed": bool(leakage_checks_passed),
        "label_support_passed": bool(label_support_passed),
        "oof_complete": bool(oof_complete),
        "m3_access_boundary_passed": bool(m3_access_boundary_passed),
    }
    weak_conditions = {
        "auroc_at_least_0_70": candidate.get("auroc") is not None
        and float(candidate["auroc"]) >= 0.70,
        "relative_aurc_improvement_vs_raw_at_least_0_05": (
            improvement_raw is not None and improvement_raw >= 0.05
        ),
        "stable_disagreement_benefit": bool(stable_disagreement_benefit),
    }
    hard_validity = {
        "leakage_checks_passed": bool(leakage_checks_passed),
        "label_support_passed": bool(label_support_passed),
        "oof_complete": bool(oof_complete),
        "m3_access_boundary_passed": bool(m3_access_boundary_passed),
    }
    classification = _independent_decision_classification(
        go_conditions,
        weak_conditions,
        hard_validity,
    )
    counterfactual_go = dict(go_conditions)
    counterfactual_go["m3_access_boundary_passed"] = True
    counterfactual_hard = dict(hard_validity)
    counterfactual_hard["m3_access_boundary_passed"] = True
    counterfactual = _independent_decision_classification(
        counterfactual_go,
        weak_conditions,
        counterfactual_hard,
    )
    reasons = [name for name, passed in go_conditions.items() if not passed]
    if not m3_access_boundary_passed:
        classification = "NO-GO"
        reasons.insert(
            0,
            "m3_access_boundary_failed_signal_go_forbidden_conservative_no_go",
        )
    if classification == "NO-GO":
        for name, passed in weak_conditions.items():
            if not passed and name not in reasons:
                reasons.append(name)
    return {
        "classification": classification,
        "reasons": reasons,
        "counterfactual_classification_without_m3_boundary": counterfactual,
        "m3_holdout_authorized": False,
        "m3_access_boundary_passed": bool(m3_access_boundary_passed),
        "signal_go_forbidden_by_m3_boundary": not bool(m3_access_boundary_passed),
        "go_conditions": go_conditions,
        "weak_conditions": weak_conditions,
        "hard_validity_conditions": hard_validity,
        "derived_values": {
            "candidate_auroc": candidate.get("auroc"),
            "relative_aurc_improvement_vs_raw": improvement_raw,
            "relative_aurc_improvement_vs_isotonic": improvement_isotonic,
            "isotonic_aurc_improvement_bootstrap_ci": bootstrap_ci,
            "predicted_risk_0_05_coverage": risk_005_coverage,
            "predicted_risk_0_05_empirical_failure_rate": risk_005_empirical,
            "predicted_risk_0_05_accepted_count": risk_005_count,
            "candidate_brier_relative_change_vs_isotonic": brier_relative_change,
            "candidate_ece": candidate.get("ece"),
            "object_driven": not not_object_driven,
        },
        "rules_version": "m6_g0_frozen_decision_v1",
    }


def _validate_decision(
    decision: Any,
    decision_inputs: Any,
    aggregate_metrics: Any,
    grouped_bootstrap: Any,
    object_jackknife: Any,
    feature_ablations: Any,
) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise ValueError("Decision artifact must be a mapping")
    if not isinstance(decision_inputs, Mapping) or set(decision_inputs) != {
        "stable_disagreement_benefit",
        "leakage_checks_passed",
        "label_support_passed",
        "oof_complete",
    }:
        raise ValueError("Decision-input artifact does not have the frozen schema")
    if any(not isinstance(value, bool) for value in decision_inputs.values()):
        raise ValueError("Every frozen decision input must be Boolean")
    if not isinstance(aggregate_metrics, Mapping):
        raise ValueError("Aggregate metrics must be a mapping")
    if not isinstance(grouped_bootstrap, Mapping):
        raise ValueError("Grouped bootstrap must be a mapping")
    if not isinstance(object_jackknife, Mapping):
        raise ValueError("Object robustness must be a mapping")
    if not isinstance(feature_ablations, Mapping):
        raise ValueError("Feature ablations must be a mapping")

    stable_disagreement = _stable_disagreement_benefit(feature_ablations)
    if decision_inputs["stable_disagreement_benefit"] != stable_disagreement:
        raise ValueError(
            "Stored stable-disagreement decision input is not reproducible"
        )
    for field in ("leakage_checks_passed", "label_support_passed", "oof_complete"):
        if decision_inputs[field] is not True:
            raise ValueError(f"Required causal/support decision input failed: {field}")

    expected = _reconstruct_frozen_decision(
        aggregate_metrics,
        grouped_bootstrap,
        object_jackknife,
        stable_disagreement_benefit=stable_disagreement,
        leakage_checks_passed=True,
        label_support_passed=True,
        oof_complete=True,
        m3_access_boundary_passed=False,
    )
    _assert_tree_close(decision, expected, "formal decision")
    classification = _decision_classification(decision)
    if classification != "NO-GO":
        raise ValueError(
            "The disclosed M3 documentation access must conservatively force NO-GO"
        )
    serialized = json.dumps(decision, sort_keys=True).lower()
    if "holdout" not in serialized or "false" not in serialized:
        raise ValueError(
            "Decision does not explicitly keep the M3 holdout unauthorized"
        )
    if "m3_access_boundary_passed" not in serialized:
        raise ValueError("Decision omits the false M3 access-boundary gate")
    if decision.get("m3_access_boundary_passed") is not False:
        raise ValueError("Decision did not preserve the failed M3 access boundary")
    if decision.get("signal_go_forbidden_by_m3_boundary") is not True:
        raise ValueError("Decision does not mechanically forbid SIGNAL GO")
    return {
        "classification": classification,
        "m3_holdout_authorized": False,
        "stable_disagreement_benefit": stable_disagreement,
        "decision_exactly_reconstructed": True,
    }


def _walk(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))


def _m2_identifier_universes(repo_root: Path) -> dict[str, set[str]]:
    """Derive allowed identifiers from the protected M2 development manifest."""

    allowed_targets = set(manifest_builder.target_ids_from_m2(repo_root))
    allowed_samples: set[str] = set()
    allowed_instances: set[str] = set()
    group_rows = _strict_jsonl(repo_root / "artifacts" / "m2" / "groups.jsonl")
    for row in group_rows:
        group_id = str(row.get("group_id", ""))
        target_id = str(row.get("target_sample_id", ""))
        if group_id not in allowed_targets or target_id not in allowed_targets:
            raise ValueError(
                "Protected M2 group identifiers differ from the target universe"
            )
        views = row.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"Protected M2 group has no candidate views: {group_id}")
        for view in views:
            if not isinstance(view, Mapping) or not isinstance(
                view.get("sample_id"), str
            ):
                raise ValueError(f"Malformed protected M2 candidate view: {group_id}")
            allowed_samples.add(str(view["sample_id"]))
        association = row.get("oracle_association")
        if not isinstance(association, Mapping) or not isinstance(
            association.get("track_id"), str
        ):
            raise ValueError(
                f"Protected M2 group lacks its physical instance: {group_id}"
            )
        allowed_instances.add(str(association["track_id"]))
    if len(allowed_targets) != EXPECTED_TARGET_COUNT:
        raise ValueError("M2 target universe no longer contains exactly 300 IDs")
    if len(allowed_instances) != EXPECTED_PHYSICAL_INSTANCE_COUNT:
        raise ValueError("M2 physical-instance universe no longer contains 169 IDs")
    return {
        "target": allowed_targets,
        "sample": allowed_samples,
        "instance": allowed_instances,
        "all": allowed_targets | allowed_samples | allowed_instances,
    }


def _identifier_context(path: Sequence[str]) -> str:
    for component in reversed(path):
        if not component.isdigit():
            return component.lower()
    return ""


def _expected_identifier_universe(
    context: str,
    universes: Mapping[str, set[str]],
) -> set[str] | None:
    if context in TARGET_ID_KEYS | TARGET_ID_LIST_KEYS:
        return universes["target"]
    if context in SAMPLE_ID_KEYS | SAMPLE_ID_LIST_KEYS:
        return universes["sample"]
    if context in INSTANCE_ID_KEYS | INSTANCE_ID_LIST_KEYS:
        return universes["instance"]
    return None


def _validate_no_m3_inputs_or_ids(
    repo_root: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    universes = _m2_identifier_universes(repo_root)
    disclosure_artifacts = {"source_audit", "access_boundary_disclosure"}
    checked_ids = 0
    checked_strings = 0
    for artifact_name, value in artifacts.items():
        for path, item in _walk(value):
            if artifact_name not in disclosure_artifacts:
                for component in path:
                    if M3_INPUT_PATTERN.search(component.replace("\\", "/")):
                        raise ValueError(
                            "M3 path appears as an M6 mapping key: "
                            f"{artifact_name}:{'.'.join(path)}"
                        )
            if not isinstance(item, str):
                continue
            checked_strings += 1
            if artifact_name not in disclosure_artifacts and M3_INPUT_PATTERN.search(
                item.replace("\\", "/")
            ):
                raise ValueError(
                    f"M3 path appears anywhere in M6 artifact {artifact_name}: {item}"
                )
            context = _identifier_context(path)
            expected_universe = _expected_identifier_universe(context, universes)
            if expected_universe is not None:
                if item not in expected_universe:
                    raise ValueError(
                        "Identifier is outside the protected M1/M2-derived universe in "
                        f"{artifact_name}:{'.'.join(path)}: {item}"
                    )
                checked_ids += 1
            elif (
                item.lower().startswith(DATASET_ID_PREFIX)
                and item not in universes["all"]
            ):
                raise ValueError(
                    "Unknown dataset identifier appears in M6 artifact "
                    f"{artifact_name}:{'.'.join(path)}: {item}"
                )

        for path, _item in _walk(value):
            for component in path:
                if (
                    component.lower().startswith(DATASET_ID_PREFIX)
                    and component not in universes["all"]
                ):
                    raise ValueError(
                        "Unknown dataset identifier appears as an M6 mapping key in "
                        f"{artifact_name}:{'.'.join(path)}: {component}"
                    )

    return {
        "allowed_m2_target_count": len(universes["target"]),
        "allowed_m1_m2_candidate_sample_count": len(universes["sample"]),
        "allowed_m2_physical_instance_count": len(universes["instance"]),
        "checked_output_id_occurrences": checked_ids,
        "checked_string_occurrences": checked_strings,
        "m3_input_paths_absent": True,
        "non_m1_m2_identifiers_absent": True,
        "raw_structured_m3_access": False,
    }


def _validate_access_boundary_disclosure(
    value: Any,
    frozen_disclosure: Any,
) -> dict[str, Any]:
    """Strictly bind the tracked disclosure to the frozen source audit."""

    if not isinstance(frozen_disclosure, Mapping):
        raise ValueError("Frozen M3 access disclosure must be a mapping")
    expected = {
        "record_type": "m6_g0_access_boundary_disclosure",
        "schema_version": 1,
        "strict_m3_access_boundary_passed": False,
        "m3_raw_artifact_structured_content_accessed": False,
        "m3_target_level_rows_accessed": False,
        "m3_target_ids_accessed": False,
        "m3_target_labels_or_predictions_accessed": False,
        "m3_candidate_values_accessed": False,
        "m3_per_target_metrics_accessed": False,
        "m3_object_level_report_context_displayed": True,
        "affected_path": "reports/m3_active_budget.md",
        "m3_information_used_for_features_models_thresholds_folds_or_decisions": False,
        "required_consequence": (
            "SIGNAL GO is prohibited and the M3 holdout remains unauthorized."
        ),
        "note": (
            "Read-only documentation searches returned broader context than intended. "
            "No raw artifacts/m3 file was parsed, no displayed M3 outcome value is "
            "reproduced here, and all M6 computation remains M2-only."
        ),
    }
    _assert_tree_close(value, expected, "tracked M3 access-boundary disclosure")
    linked_fields = {
        "strict_m3_access_boundary_passed": "m3_access_boundary_passed",
        "m3_raw_artifact_structured_content_accessed": "raw_structured_m3_access",
        "m3_target_level_rows_accessed": "m3_artifact_rows_accessed",
        "m3_target_ids_accessed": "m3_target_ids_accessed",
        "m3_target_labels_or_predictions_accessed": "m3_predictions_or_labels_accessed",
        "m3_information_used_for_features_models_thresholds_folds_or_decisions": (
            "displayed_values_used_by_m6_g0"
        ),
        "affected_path": "accidental_documentation_display",
    }
    for tracked_key, frozen_key in linked_fields.items():
        if value[tracked_key] != frozen_disclosure.get(frozen_key):
            raise ValueError(
                f"Tracked/frozen M3 disclosure mismatch: {tracked_key} != {frozen_key}"
            )
    if (
        frozen_disclosure.get("displayed_scope")
        != "aggregate and per-object context only"
    ):
        raise ValueError("Frozen M3 displayed-scope disclosure changed")
    if frozen_disclosure.get("m3_hashes_unchanged") is not True:
        raise ValueError("Frozen disclosure no longer records unchanged M3 hashes")
    return {
        "tracked_disclosure_sha256_required": True,
        "strictly_matches_frozen_source_audit": True,
        "m3_access_boundary_passed": False,
        "raw_structured_m3_access": False,
        "displayed_values_used": False,
        "passed": True,
    }


def _hash_mapping(value: Any) -> Mapping[str, str]:
    if isinstance(value, Mapping):
        for key in ("sha256", "source_sha256", "files", "source_hashes"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and all(
                isinstance(path, str) and isinstance(digest, str)
                for path, digest in nested.items()
            ):
                return nested
    raise ValueError("Source-hash artifact contains no path-to-SHA256 mapping")


def _validate_source_hashes_and_publication(
    repo_root: Path,
    source_hashes: Any,
) -> dict[str, Any]:
    hashes = _hash_mapping(source_hashes)
    if not isinstance(source_hashes, Mapping) or source_hashes.get(
        "canonical_sha256"
    ) != manifest_builder.canonical_sha256(hashes):
        raise ValueError("M6-G0 source-hash canonical receipt mismatch")
    if len(hashes) < 8:
        raise ValueError("M6-G0 source-hash receipt is unexpectedly small")
    disclosure_relative = ACCESS_BOUNDARY_DISCLOSURE_PATH.as_posix()
    disclosure_path = repo_root / ACCESS_BOUNDARY_DISCLOSURE_PATH
    disclosure_sha256 = sha256_file(disclosure_path)
    if hashes.get(disclosure_relative) != disclosure_sha256:
        raise ValueError(
            "M6-G0 source-hash receipt omits or mismatches the access disclosure"
        )
    for relative, expected in hashes.items():
        normalized = relative.replace("\\", "/")
        if M3_INPUT_PATTERN.search(normalized):
            raise ValueError(f"M3 path appears in M6 source hashes: {relative}")
        path = repo_root / Path(normalized)
        if not path.is_file():
            raise FileNotFoundError(f"Source-hash path is missing: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA-256 text for {relative}: {expected}")
        if sha256_file(path) != expected:
            raise ValueError(f"M6-G0 source hash mismatch: {relative}")

    precomputed_path = _required_file(
        repo_root / PRECOMPUTED_PATH,
        "tracked precomputed result",
    )
    precomputed = _strict_json(precomputed_path)
    if not isinstance(precomputed, Mapping):
        raise ValueError("Tracked precomputed result must be a JSON object")
    precomputed_hashes = precomputed.get("source_hashes")
    if not isinstance(precomputed_hashes, Mapping):
        raise ValueError("Tracked precomputed result omits source hashes")
    if precomputed_hashes.get(disclosure_relative) != disclosure_sha256:
        raise ValueError(
            "Tracked precomputed result omits or mismatches the access disclosure"
        )
    for relative, expected in precomputed_hashes.items():
        path = repo_root / Path(str(relative).replace("\\", "/"))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Precomputed source hash mismatch: {relative}")

    report_path = _required_file(repo_root / REPORT_PATH, "paper-style report")
    report = report_path.read_text(encoding="utf-8")
    required_report_text = (
        "Existing-Data Confidence Signal Audit",
        "M5-G0 remained NO-GO",
        "M2 development",
        "M3",
        "no FoundationPose inference",
        "no new pose estimates",
        "grouped cross-validation",
        "NO-GO",
    )
    missing = [
        text for text in required_report_text if text.lower() not in report.lower()
    ]
    if missing:
        raise ValueError(f"M6-G0 report omits required statements: {missing}")
    for plot in REQUIRED_PLOTS:
        path = _required_file(
            repo_root / "reports" / "m6_g0" / plot,
            f"plot {plot}",
        )
        if path.stat().st_size < 1024:
            raise ValueError(f"M6-G0 plot is implausibly small: {plot}")

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    for command in (
        "python -B scripts/build_m6_g0_manifests.py",
        "python -B scripts/run_m6_g0.py",
        "python -B scripts/validate_m6_g0.py",
        "python -B scripts/validate_m6_g0.py --check",
    ):
        if command not in readme:
            raise ValueError(f"README M6-G0 reproduction omits: {command}")
    return {
        "source_hash_count": len(hashes),
        "precomputed_source_hash_count": len(precomputed_hashes),
        "access_boundary_disclosure_sha256": disclosure_sha256,
        "report_sha256": sha256_file(report_path),
        "precomputed_sha256": sha256_file(precomputed_path),
        "plot_count": len(REQUIRED_PLOTS),
        "readme_reproduction_present": True,
    }


def _run_existing_check(repo_root: Path, command: Sequence[str], label: str) -> str:
    completed = subprocess.run(
        list(command),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = "\n".join(
        item.strip() for item in (completed.stdout, completed.stderr) if item.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Existing {label} validation failed with exit {completed.returncode}:\n{output}"
        )
    return output


def _validate_prior_milestones(repo_root: Path) -> dict[str, Any]:
    release_output = _run_existing_check(
        repo_root,
        ["python", "-B", "scripts/render_precomputed_report.py", "--check"],
        "M1-M4",
    )
    m5_output = _run_existing_check(
        repo_root,
        ["python", "-B", "scripts/validate_m5_g0.py", "--check"],
        "M5-G0",
    )
    protected = manifest_builder.validate_protected_baseline(repo_root)
    return {
        "m1_m4_check": "PASS",
        "m1_m4_check_output": release_output,
        "m5_g0_check": "PASS",
        "m5_g0_check_output": m5_output,
        "protected_entry_count": protected["entry_count"],
        "protected_root_sha256": protected["root_sha256"],
        "m3_hashes_unchanged": protected["m3_hashes_unchanged"],
    }


def _canonical_parameters(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_model_selection_rows(
    value: Any,
    outer_assignment: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(value, list):
        raise ValueError("Model-selection audit must be a list")
    fold_count = len(set(outer_assignment.values()))
    expected_folds = set(range(fold_count))
    by_fold: dict[int, Mapping[str, Any]] = {}
    logistic_grid = {
        _canonical_parameters({"C": float(c_value)})
        for c_value in analysis.LOGISTIC_C_GRID
    }
    tree_grid = {
        _canonical_parameters(parameters) for parameters in analysis.TREE_PARAM_GRID
    }
    required_fields = {
        "outer_fold",
        "outer_train_target_count",
        "outer_test_target_count",
        "selection_data",
        "objective",
        "logistic_candidates",
        "tree_candidates",
        "fixed_logistic_selected_parameters",
        "fixed_tree_selected_parameters",
        "nested_selected_family",
        "nested_selected_parameters",
        "best_logistic_inner_mean_aurc",
        "best_tree_inner_mean_aurc",
        "tree_inner_aurc_advantage",
        "logistic_tie_tolerance",
        "logistic_tie_rule_applied",
        "outer_test_labels_used_for_selection",
        "tree_calibration",
    }
    for row in value:
        if not isinstance(row, Mapping) or set(row) != required_fields:
            raise ValueError("Model-selection row does not have the frozen schema")
        fold = int(row["outer_fold"])
        if fold not in expected_folds or fold in by_fold:
            raise ValueError(f"Duplicate or invalid model-selection fold: {fold}")
        by_fold[fold] = row
        outer_test_count = sum(
            assigned_fold == fold for assigned_fold in outer_assignment.values()
        )
        if (
            int(row["outer_test_target_count"]) != outer_test_count
            or int(row["outer_train_target_count"])
            != len(outer_assignment) - outer_test_count
        ):
            raise ValueError(f"Model-selection fold counts differ at fold {fold}")
        if row["selection_data"] != "outer_train_inner_grouped_cv_only":
            raise ValueError("Hyperparameters were not declared training-only")
        if row["objective"] != "inner_mean_aurc":
            raise ValueError("Model-selection objective is not frozen inner AURC")
        if row["outer_test_labels_used_for_selection"] is not False:
            raise ValueError("Outer-test labels entered model selection")
        if row["tree_calibration"] != "sigmoid_from_outer_train_inner_oof_only":
            raise ValueError("Tree calibration is not training-only inner OOF")

        logistic_candidates = row["logistic_candidates"]
        tree_candidates = row["tree_candidates"]
        if not isinstance(logistic_candidates, list) or not isinstance(
            tree_candidates, list
        ):
            raise ValueError("Model candidate audits must be lists")
        observed_logistic: set[str] = set()
        for candidate in logistic_candidates:
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "family",
                "parameters",
                "inner_mean_aurc",
                "inner_fold_aurc",
            }:
                raise ValueError("Logistic candidate audit schema changed")
            if candidate["family"] != "logistic" or not isinstance(
                candidate["parameters"], Mapping
            ):
                raise ValueError("Malformed logistic candidate")
            observed_logistic.add(_canonical_parameters(candidate["parameters"]))
            folds = candidate["inner_fold_aurc"]
            if not isinstance(folds, list) or len(folds) != 4:
                raise ValueError("Logistic candidate does not cover four inner folds")
            if not math.isclose(
                float(candidate["inner_mean_aurc"]),
                sum(float(metric) for metric in folds) / 4.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Logistic inner mean AURC is not reproducible")
        if observed_logistic != logistic_grid:
            raise ValueError("Logistic hyperparameter grid differs from frozen grid")

        observed_tree: set[str] = set()
        for candidate in tree_candidates:
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "family",
                "parameters",
                "inner_mean_aurc",
                "inner_fold_aurc",
                "calibration",
            }:
                raise ValueError("Tree candidate audit schema changed")
            if candidate["family"] != "tree" or not isinstance(
                candidate["parameters"], Mapping
            ):
                raise ValueError("Malformed tree candidate")
            if candidate["calibration"] != "sigmoid_fit_to_training_only_inner_oof":
                raise ValueError("Tree candidate calibration is not inner-OOF-only")
            observed_tree.add(_canonical_parameters(candidate["parameters"]))
            folds = candidate["inner_fold_aurc"]
            if not isinstance(folds, list) or len(folds) != 4:
                raise ValueError("Tree candidate does not cover four inner folds")
            if not math.isclose(
                float(candidate["inner_mean_aurc"]),
                sum(float(metric) for metric in folds) / 4.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Tree inner mean AURC is not reproducible")
        if observed_tree != tree_grid:
            raise ValueError("Tree hyperparameter grid differs from frozen grid")

        best_logistic = min(
            logistic_candidates,
            key=lambda candidate: (
                float(candidate["inner_mean_aurc"]),
                float(candidate["parameters"]["C"]),
            ),
        )
        best_tree = min(
            tree_candidates,
            key=lambda candidate: (
                float(candidate["inner_mean_aurc"]),
                int(candidate["parameters"]["max_depth"]),
                int(candidate["parameters"]["n_estimators"]),
                -int(candidate["parameters"]["min_samples_leaf"]),
            ),
        )
        _assert_tree_close(
            row["fixed_logistic_selected_parameters"],
            best_logistic["parameters"],
            f"fold {fold} fixed logistic selection",
        )
        _assert_tree_close(
            row["fixed_tree_selected_parameters"],
            best_tree["parameters"],
            f"fold {fold} fixed tree selection",
        )
        _assert_tree_close(
            row["best_logistic_inner_mean_aurc"],
            best_logistic["inner_mean_aurc"],
            f"fold {fold} best logistic AURC",
        )
        _assert_tree_close(
            row["best_tree_inner_mean_aurc"],
            best_tree["inner_mean_aurc"],
            f"fold {fold} best tree AURC",
        )
        advantage = float(best_logistic["inner_mean_aurc"]) - float(
            best_tree["inner_mean_aurc"]
        )
        _assert_tree_close(
            row["tree_inner_aurc_advantage"],
            advantage,
            f"fold {fold} tree advantage",
        )
        if float(row["logistic_tie_tolerance"]) != 0.005:
            raise ValueError("Nested family tie tolerance changed")
        expected_family = "tree" if advantage > 0.005 else "logistic"
        expected_parameters = (
            best_tree["parameters"]
            if expected_family == "tree"
            else best_logistic["parameters"]
        )
        if row["nested_selected_family"] != expected_family:
            raise ValueError(f"Nested family selection is wrong at fold {fold}")
        _assert_tree_close(
            row["nested_selected_parameters"],
            expected_parameters,
            f"fold {fold} nested parameters",
        )
        if row["logistic_tie_rule_applied"] is not (abs(advantage) <= 0.005):
            raise ValueError(f"Nested tie-rule audit is wrong at fold {fold}")
    if set(by_fold) != expected_folds:
        raise ValueError("Model-selection audit does not cover every outer fold")
    return {
        "outer_fold_count": fold_count,
        "logistic_grid_size": len(logistic_grid),
        "tree_grid_size": len(tree_grid),
    }


def _validate_ablations_and_contributions(
    repo_root: Path,
    feature_ablations: Any,
    feature_contributions: Any,
    model_selections: Any,
    aggregate_metrics: Any,
    per_fold_metrics: Any,
    outer_assignment: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(feature_ablations, Mapping) or set(feature_ablations) != set(
        analysis.ABLATION_NAMES
    ):
        raise ValueError("Feature ablations do not match the frozen four-way design")
    frozen_specification = _strict_json(
        repo_root / FROZEN_DIR / "feature_specification.json"
    )
    frozen_ablations = frozen_specification["ablations"]
    main_nested_metrics = _metric_entries(aggregate_metrics)["NESTED_MULTIFEATURE"]
    expected_nested_folds = [
        row
        for row in per_fold_metrics
        if _prediction_method(row) == "NESTED_MULTIFEATURE"
    ]
    selection_evidence: dict[str, Any] = {}
    for name in analysis.ABLATION_NAMES:
        payload = feature_ablations[name]
        if not isinstance(payload, Mapping) or set(payload) != {
            "feature_names",
            "aggregate_metrics",
            "per_fold_metrics",
            "model_selections",
        }:
            raise ValueError(f"Ablation payload schema changed: {name}")
        expected_features = list(frozen_ablations[name])
        if payload["feature_names"] != expected_features:
            raise ValueError(f"Ablation feature list differs from frozen spec: {name}")
        fold_rows = payload["per_fold_metrics"]
        if not isinstance(fold_rows, list) or {
            int(row.get("outer_fold", -1))
            for row in fold_rows
            if isinstance(row, Mapping)
        } != set(outer_assignment.values()):
            raise ValueError(f"Ablation does not use every frozen outer fold: {name}")
        if any(
            not isinstance(row, Mapping)
            or row.get("method") != "NESTED_MULTIFEATURE"
            or not isinstance(row.get("metrics"), Mapping)
            for row in fold_rows
        ):
            raise ValueError(f"Ablation fold metrics are malformed: {name}")
        selection_evidence[name] = _validate_model_selection_rows(
            payload["model_selections"], outer_assignment
        )
    _assert_tree_close(
        feature_ablations["all"]["model_selections"],
        model_selections,
        "all-feature model selections",
    )
    _assert_tree_close(
        feature_ablations["all"]["aggregate_metrics"],
        main_nested_metrics,
        "all-feature candidate aggregate metrics",
    )
    _assert_tree_close(
        feature_ablations["all"]["per_fold_metrics"],
        expected_nested_folds,
        "all-feature candidate fold metrics",
    )

    if not isinstance(feature_contributions, Mapping) or set(feature_contributions) != {
        "logistic_outer_fold_coefficients",
        "logistic_coefficient_stability",
        "tree_outer_test_permutation_importance",
    }:
        raise ValueError("Feature-contribution artifact does not have frozen schema")
    all_features = list(frozen_ablations["all"])
    expected_folds = set(outer_assignment.values())
    coefficient_rows = feature_contributions["logistic_outer_fold_coefficients"]
    if not isinstance(coefficient_rows, list):
        raise ValueError("Logistic coefficient audit must be a list")
    coefficient_keys: set[tuple[int, str]] = set()
    features_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in coefficient_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "outer_fold",
            "feature",
            "standardized_coefficient",
        }:
            raise ValueError("Logistic coefficient row schema changed")
        fold = int(row["outer_fold"])
        feature = str(row["feature"])
        key = (fold, feature)
        if fold not in expected_folds or key in coefficient_keys:
            raise ValueError("Duplicate or invalid logistic coefficient row")
        coefficient_keys.add(key)
        base_feature = feature.removeprefix("missingindicator_")
        if base_feature not in all_features:
            raise ValueError(f"Unknown transformed logistic feature: {feature}")
        if not math.isfinite(float(row["standardized_coefficient"])):
            raise ValueError("Non-finite standardized logistic coefficient")
        features_by_fold[fold].add(feature)
    if set(features_by_fold) != expected_folds or any(
        not set(all_features).issubset(features)
        for features in features_by_fold.values()
    ):
        raise ValueError("Logistic contribution rows omit frozen features or folds")
    coefficient_stability_function = getattr(analysis, "_coefficient_stability", None)
    if not callable(coefficient_stability_function):
        raise AttributeError("m6_g0_analysis._coefficient_stability is missing")
    expected_stability = coefficient_stability_function(
        coefficient_rows, len(expected_folds)
    )
    _assert_tree_close(
        feature_contributions["logistic_coefficient_stability"],
        expected_stability,
        "logistic coefficient stability",
    )

    permutation_rows = feature_contributions["tree_outer_test_permutation_importance"]
    if not isinstance(permutation_rows, list) or len(permutation_rows) != len(
        expected_folds
    ) * len(all_features):
        raise ValueError("Tree permutation audit does not cover every fold/feature")
    permutation_keys: set[tuple[int, str]] = set()
    for row in permutation_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "outer_fold",
            "feature",
            "metric",
            "repeats",
            "importance_mean",
            "importance_std",
        }:
            raise ValueError("Tree permutation-importance row schema changed")
        fold = int(row["outer_fold"])
        feature = str(row["feature"])
        key = (fold, feature)
        if (
            fold not in expected_folds
            or feature not in all_features
            or key in permutation_keys
        ):
            raise ValueError("Duplicate or invalid tree permutation row")
        permutation_keys.add(key)
        if row["metric"] != "outer_test_aurc_increase_when_permuted":
            raise ValueError("Tree importance is not outer-test permutation AURC")
        if int(row["repeats"]) != 10:
            raise ValueError("Tree permutation repeat count changed")
        if not all(
            math.isfinite(float(row[field]))
            for field in ("importance_mean", "importance_std")
        ):
            raise ValueError("Tree permutation importance is non-finite")
    return {
        "ablation_count": len(feature_ablations),
        "selection_audits": selection_evidence,
        "logistic_coefficient_row_count": len(coefficient_rows),
        "tree_permutation_row_count": len(permutation_rows),
        "training_only_model_selection_and_preprocessing": True,
    }


def _validate_prediction_audit(
    value: Any,
    by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Prediction-validation artifact must be a mapping")
    rows = [row for method in FORMAL_METHODS for row in by_method[method]]
    target_ids = sorted({_prediction_target(row) for row in rows})
    expected = analysis.validate_oof_predictions(rows, target_ids)
    _assert_tree_close(value, expected, "OOF prediction-validation audit")
    if value.get("passed") is not True:
        raise ValueError(f"OOF prediction validation failed: {value.get('errors')}")
    return {
        "formal_method_count": len(by_method),
        "prediction_row_count": sum(len(rows) for rows in by_method.values()),
        "target_count": len(target_ids),
        "each_target_once_per_method": True,
        "causal_audit_passed": True,
    }


def _validate_nested_prediction_linkage(
    by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    model_selections: Any,
) -> dict[str, Any]:
    """Link each nested OOF row to the fixed family selected in its outer fold."""

    if not isinstance(model_selections, list):
        raise ValueError("Model-selection audit must be a list")
    family_by_fold: dict[int, str] = {}
    for row in model_selections:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed model-selection row")
        fold = int(row.get("outer_fold", -1))
        family = str(row.get("nested_selected_family", ""))
        if fold < 0 or fold in family_by_fold or family not in {"logistic", "tree"}:
            raise ValueError("Nested family linkage has an invalid fold selection")
        family_by_fold[fold] = family

    fixed_method_by_family = {
        "logistic": "LOGISTIC_MULTIFEATURE",
        "tree": "SHALLOW_TREE_MULTIFEATURE",
    }
    fixed_indices = {
        method: {_prediction_target(row): row for row in by_method[method]}
        for method in fixed_method_by_family.values()
    }
    linked_by_family = {"logistic": 0, "tree": 0}
    nested_rows = by_method["NESTED_MULTIFEATURE"]
    for nested_row in nested_rows:
        target_id = _prediction_target(nested_row)
        outer_fold = int(_prediction_field(nested_row, "outer_fold", "fold"))
        if outer_fold not in family_by_fold:
            raise ValueError(
                f"Nested OOF row has no fold family selection: {target_id}"
            )
        family = family_by_fold[outer_fold]
        fixed_method = fixed_method_by_family[family]
        fixed_row = fixed_indices[fixed_method].get(target_id)
        if fixed_row is None:
            raise ValueError(
                f"Nested OOF row has no fixed-family counterpart: {target_id}"
            )
        _assert_tree_close(
            _risk_score(nested_row),
            _risk_score(fixed_row),
            f"nested risk linkage for {target_id}",
        )
        _assert_tree_close(
            _failure_probability_or_none(nested_row),
            _failure_probability_or_none(fixed_row),
            f"nested probability linkage for {target_id}",
        )
        linked_by_family[family] += 1
    if sum(linked_by_family.values()) != EXPECTED_TARGET_COUNT:
        raise ValueError("Nested linkage does not cover every target")
    return {
        "nested_rows_linked_to_selected_fixed_family": EXPECTED_TARGET_COUNT,
        "linked_row_count_by_family": linked_by_family,
        "risk_and_probability_exactly_match_selected_family": True,
    }


def _validate_chronology(
    repo_root: Path,
    chronology: Any,
    classification: str,
) -> dict[str, Any]:
    if not isinstance(chronology, Mapping):
        raise ValueError("M6-G0 chronology must be a mapping")
    fields = (
        "run_started_utc",
        "static_protocol_frozen_utc",
        "feature_and_label_rows_written_utc",
        "fold_manifest_written_utc",
        "model_evaluation_started_utc",
        "run_completed_utc",
    )
    times: list[datetime] = []
    for field in fields:
        value = chronology.get(field)
        if not isinstance(value, str):
            raise ValueError(f"M6-G0 chronology omits {field}")
        try:
            times.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError as exc:
            raise ValueError(f"Invalid chronology timestamp {field}: {value}") from exc
    if times != sorted(times):
        raise ValueError("M6-G0 freeze/data/fold/model chronology is out of order")
    fold_path = repo_root / FROZEN_DIR / "fold_manifest.json"
    if chronology.get("fold_manifest_sha256") != sha256_file(fold_path):
        raise ValueError("Chronology fold-manifest file hash mismatch")
    if chronology.get("classification") != classification:
        raise ValueError("Chronology classification differs from decision")
    if int(chronology.get("bootstrap_resamples", 0)) < 2000:
        raise ValueError("Chronology records fewer than 2,000 bootstraps")
    return {
        "fold_frozen_before_model_evaluation": True,
        "fold_manifest_sha256": chronology["fold_manifest_sha256"],
        "bootstrap_resamples": chronology["bootstrap_resamples"],
        "passed": True,
    }


def validate(repo_root: Path, *, check_only: bool = False) -> dict[str, Any]:
    """Validate every frozen contract, output row, metric, and decision gate."""

    repo_root = repo_root.resolve()
    manifest_receipt = manifest_builder.freeze(repo_root, check_only=True)
    source_audit = _strict_json(repo_root / FROZEN_DIR / "source_audit.json")
    disclosure = source_audit.get("m3_access_disclosure", {})
    if disclosure.get("m3_access_boundary_passed") is not False:
        raise ValueError("M3 documentation-access disclosure was not preserved")
    if disclosure.get("raw_structured_m3_access") is not False:
        raise ValueError("Raw structured M3 access must remain false")
    if disclosure.get("displayed_values_used_by_m6_g0") is not False:
        raise ValueError("Displayed M3 documentation values must not enter M6-G0")

    artifacts = _load_required_artifacts(repo_root)
    disclosure_evidence = _validate_access_boundary_disclosure(
        artifacts["access_boundary_disclosure"],
        disclosure,
    )
    feature_by_target, evaluator_by_target, support = (
        _validate_feature_and_evaluator_rows(
            repo_root,
            artifacts["features"],
            artifacts["evaluator_rows"],
        )
    )
    outer_assignment, fold_evidence = _validate_fold_manifest(
        artifacts["fold_manifest"],
        evaluator_by_target,
    )
    by_method, score_semantics_evidence = _validate_oof_predictions(
        artifacts["oof_predictions"],
        feature_by_target,
        evaluator_by_target,
        outer_assignment,
    )
    metric_evidence = _validate_metric_reconstruction(
        artifacts["aggregate_metrics"],
        artifacts["per_fold_metrics"],
        by_method,
    )
    ablation_metric_evidence = _validate_ablation_metric_reconstruction(
        artifacts["feature_ablations"],
        artifacts["ablation_oof_predictions"],
        evaluator_by_target,
        outer_assignment,
        by_method,
    )
    bootstrap_evidence = _validate_bootstrap_and_robustness(
        artifacts["grouped_bootstrap"],
        artifacts["object_jackknife"],
        artifacts["oof_predictions"],
        artifacts["evaluator_rows"],
    )
    prediction_evidence = _validate_prediction_audit(
        artifacts["prediction_validation"],
        by_method,
    )
    model_evidence = _validate_ablations_and_contributions(
        repo_root,
        artifacts["feature_ablations"],
        artifacts["feature_contributions"],
        artifacts["model_selections"],
        artifacts["aggregate_metrics"],
        artifacts["per_fold_metrics"],
        outer_assignment,
    )
    nested_linkage_evidence = _validate_nested_prediction_linkage(
        by_method,
        artifacts["model_selections"],
    )
    prediction_evidence = {
        **prediction_evidence,
        "score_probability_semantics": score_semantics_evidence,
        "nested_family_linkage": nested_linkage_evidence,
    }
    decision_evidence = _validate_decision(
        artifacts["decision"],
        artifacts["decision_inputs"],
        artifacts["aggregate_metrics"],
        artifacts["grouped_bootstrap"],
        artifacts["object_jackknife"],
        artifacts["feature_ablations"],
    )
    chronology_evidence = _validate_chronology(
        repo_root,
        artifacts["chronology"],
        decision_evidence["classification"],
    )
    m3_evidence = _validate_no_m3_inputs_or_ids(repo_root, artifacts)
    publication_evidence = _validate_source_hashes_and_publication(
        repo_root,
        artifacts["source_hashes"],
    )
    prior_evidence = _validate_prior_milestones(repo_root)

    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m6_g0_formal_validation",
        "status": "PASS",
        "classification": decision_evidence["classification"],
        "m3_holdout_authorized": False,
        "checks": {
            "static_freeze": {
                "passed": True,
                "file_count": manifest_receipt["static_frozen_file_count"],
                "source_hash_count": len(manifest_receipt["source_sha256"]),
            },
            "protected_m2_m5": prior_evidence,
            "source_audit_disclosure": disclosure_evidence,
            "feature_evaluator_separation": {
                "feature_row_count": len(feature_by_target),
                "evaluator_row_count": len(evaluator_by_target),
                "feature_name_count": len(_feature_names_from_module()),
                "one_row_per_target": True,
                "model_api_receives_features_only": True,
                "passed": True,
            },
            "label_support": support,
            "fold_manifest": fold_evidence,
            "oof_predictions": prediction_evidence,
            "model_selection_and_feature_contributions": model_evidence,
            "metric_reconstruction": metric_evidence,
            "ablation_metric_reconstruction": ablation_metric_evidence,
            "bootstrap_and_robustness": bootstrap_evidence,
            "m3_exclusion": m3_evidence,
            "decision": decision_evidence,
            "chronology": chronology_evidence,
            "publication": publication_evidence,
        },
        "source_sha256": {
            "scripts/build_m6_g0_manifests.py": sha256_file(
                repo_root / "scripts" / "build_m6_g0_manifests.py"
            ),
            "scripts/m6_g0_features.py": sha256_file(
                repo_root / "scripts" / "m6_g0_features.py"
            ),
            "scripts/m6_g0_labels.py": sha256_file(
                repo_root / "scripts" / "m6_g0_labels.py"
            ),
            "scripts/m6_g0_analysis.py": sha256_file(
                repo_root / "scripts" / "m6_g0_analysis.py"
            ),
            "scripts/m6_g0_report.py": sha256_file(
                repo_root / "scripts" / "m6_g0_report.py"
            ),
            "scripts/run_m6_g0.py": sha256_file(repo_root / "scripts" / "run_m6_g0.py"),
            "scripts/validate_m6_g0.py": sha256_file(
                repo_root / "scripts" / "validate_m6_g0.py"
            ),
        },
    }
    formal_path = repo_root / FORMAL_VALIDATION_PATH
    if check_only:
        stored = _strict_json(_required_file(formal_path, "formal validation result"))
        _assert_tree_close(stored, result, "stored formal validation")
    else:
        write_json_atomic(formal_path, result)
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = validate(repo_root, check_only=args.check)
    action = "validated" if args.check else "wrote"
    print(
        f"{action} M6-G0 formal validation: status={result['status']}, "
        f"classification={result['classification']}, "
        "M3 holdout authorized=false"
    )


if __name__ == "__main__":
    main()
