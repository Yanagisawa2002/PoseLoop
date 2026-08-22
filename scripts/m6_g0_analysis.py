"""Leakage-safe statistical analysis utilities for the M6-G0 signal audit.

The module deliberately has no repository I/O.  Callers must construct the
inference-time feature records and the separate evaluator records, persist the
returned fold manifest, and only then call :func:`run_evaluation`.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTER_SEED = 6001
INNER_SEED_BASE = 6100
BOOTSTRAP_SEED = 6201
PERMUTATION_SEED_BASE = 6300
LOGISTIC_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
TREE_PARAM_GRID: tuple[dict[str, int], ...] = tuple(
    {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
    }
    for max_depth in (2, 3)
    for n_estimators in (50, 100)
    for min_samples_leaf in (10, 20)
)
FORMAL_METHODS: tuple[str, ...] = (
    "RAW_SCORE_RANK",
    "SCORE_ISOTONIC",
    "LOGISTIC_MULTIFEATURE",
    "SHALLOW_TREE_MULTIFEATURE",
    "NESTED_MULTIFEATURE",
)
ABLATION_NAMES: tuple[str, ...] = (
    "score_only",
    "disagreement_only",
    "score_disagreement",
    "all",
)
PROHIBITED_FEATURE_TOKENS: tuple[str, ...] = (
    "target_id",
    "object_id",
    "physical_instance_id",
    "instance_id",
    "scene_id",
    "split_id",
    "m3",
    "ground_truth",
    "groundtruth",
    "gt_pose",
    "pose_error",
    "bop_error",
    "correct",
    "success",
    "failure",
    "label",
    "oracle",
)


class FoldSupportError(ValueError):
    """Raised when the frozen grouped split design cannot support both classes."""


class AnalysisContractError(ValueError):
    """Raised when caller-provided rows violate the frozen analysis contract."""


@dataclass(frozen=True)
class _ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = float(np.clip(self.probability, 0.0, 1.0))
        positive = np.full(len(x), probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


def _as_records(rows: Any, *, name: str) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        try:
            converted = rows.to_dict(orient="records")
        except TypeError:
            converted = rows.to_dict("records")
    else:
        converted = list(rows)
    if not all(isinstance(row, Mapping) for row in converted):
        raise AnalysisContractError(f"{name} must be a sequence of mappings")
    return [dict(row) for row in converted]


def _require_columns(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *, name: str
) -> None:
    if not rows:
        raise AnalysisContractError(f"{name} must not be empty")
    missing = sorted(
        {column for column in columns if any(column not in row for row in rows)}
    )
    if missing:
        raise AnalysisContractError(f"{name} missing columns: {missing}")


def _project_evaluator_records(evaluator_rows: Any) -> list[dict[str, Any]]:
    rows = _as_records(evaluator_rows, name="evaluator_rows")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if "target_id" in row:
            projected.append(row)
        elif "target_sample_id" in row:
            projected.append({"target_id": row["target_sample_id"], **row})
        else:
            projected.append(row)
    return projected


def _id_key(value: Any) -> str:
    return _canonical_json(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def fold_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 digest of a fold manifest.

    Any existing ``manifest_hash`` field is excluded, which makes validation
    idempotent after the digest is embedded in the manifest.
    """

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _binary_vector(values: Sequence[Any], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=int)
    if vector.ndim != 1:
        raise AnalysisContractError(f"{name} must be one-dimensional")
    if not np.all(np.isin(vector, (0, 1))):
        raise AnalysisContractError(f"{name} must contain only 0 and 1")
    return vector


def validate_label_support(evaluator_rows: Any) -> dict[str, Any]:
    """Evaluate the frozen minimum class, object, and instance support gates."""

    rows = _project_evaluator_records(evaluator_rows)
    _require_columns(
        rows,
        ("target_id", "y_failure", "physical_instance_id", "object_id"),
        name="evaluator_rows",
    )
    y = _binary_vector([row["y_failure"] for row in rows], name="y_failure")
    failure_instances = {
        _id_key(row["physical_instance_id"])
        for row, label in zip(rows, y, strict=True)
        if label == 1
    }
    failure_objects = {
        _id_key(row["object_id"])
        for row, label in zip(rows, y, strict=True)
        if label == 1
    }
    counts = {
        "targets": int(len(rows)),
        "failures": int(y.sum()),
        "successes": int(len(y) - y.sum()),
        "physical_instances_with_failures": len(failure_instances),
        "objects_with_failures": len(failure_objects),
    }
    conditions = {
        "at_least_30_failures": counts["failures"] >= 30,
        "at_least_30_successes": counts["successes"] >= 30,
        "at_least_10_failure_instances": (
            counts["physical_instances_with_failures"] >= 10
        ),
        "at_least_5_failure_objects": counts["objects_with_failures"] >= 5,
    }

    def grouped_counts(field: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        keys = sorted({_id_key(row[field]) for row in rows})
        display = {_id_key(row[field]): row[field] for row in rows}
        for key in keys:
            labels = np.asarray(
                [int(row["y_failure"]) for row in rows if _id_key(row[field]) == key],
                dtype=int,
            )
            result.append(
                {
                    field: display[key],
                    "target_count": int(len(labels)),
                    "failure_count": int(labels.sum()),
                    "success_count": int(len(labels) - labels.sum()),
                    "failure_prevalence": float(np.mean(labels)),
                }
            )
        return result

    return _jsonable(
        {
            "passed": all(conditions.values()),
            "counts": counts,
            "conditions": conditions,
            "counts_by_object": grouped_counts("object_id"),
            "counts_by_physical_instance": grouped_counts("physical_instance_id"),
        }
    )


def _split_support_reason(
    y: np.ndarray, groups: np.ndarray, n_splits: int
) -> str | None:
    if len(np.unique(groups)) < n_splits:
        return (
            f"only {len(np.unique(groups))} physical-instance groups for "
            f"{n_splits} folds"
        )
    for label in (0, 1):
        supporting_groups = len(np.unique(groups[y == label]))
        if supporting_groups < n_splits:
            return (
                f"class {label} occurs in only {supporting_groups} groups, "
                f"fewer than {n_splits}"
            )
    return None


def _make_sgkf_assignments(
    y: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray | None, str | None]:
    support_reason = _split_support_reason(y, groups, n_splits)
    if support_reason is not None:
        return None, support_reason
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    assignments = np.full(len(y), -1, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            splits = list(splitter.split(np.zeros((len(y), 1)), strata, groups))
        except ValueError as exc:
            return None, f"StratifiedGroupKFold failed: {exc}"
    for fold, (train_index, test_index) in enumerate(splits):
        if set(np.unique(y[train_index])) != {0, 1}:
            return None, f"fold {fold} outer-train lacks one binary class"
        if set(np.unique(y[test_index])) != {0, 1}:
            return None, f"fold {fold} outer-test lacks one binary class"
        assignments[test_index] = fold
    if np.any(assignments < 0):
        return None, "at least one target was not assigned"
    return assignments, None


def _count_by(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def make_fold_manifest(
    evaluator_rows: Any,
    *,
    outer_seed: int = OUTER_SEED,
    inner_seed_base: int = INNER_SEED_BASE,
) -> dict[str, Any]:
    """Create every outer and inner assignment before any model is fitted.

    Five outer folds are attempted first.  A single fallback to four is made
    only when the five-fold assignment cannot place both binary classes in
    every grouped train and test partition.  Inner folds are always four-fold
    grouped splits wholly inside their corresponding outer-training set.
    """

    rows = _project_evaluator_records(evaluator_rows)
    _require_columns(
        rows,
        ("target_id", "y_failure", "physical_instance_id", "object_id"),
        name="evaluator_rows",
    )
    rows = sorted(rows, key=lambda row: _id_key(row["target_id"]))
    target_keys = [_id_key(row["target_id"]) for row in rows]
    if len(target_keys) != len(set(target_keys)):
        raise AnalysisContractError("evaluator_rows target_id values must be unique")

    y = _binary_vector([row["y_failure"] for row in rows], name="y_failure")
    groups = np.asarray(
        [_id_key(row["physical_instance_id"]) for row in rows], dtype=object
    )
    strata = np.asarray(
        [
            f"{_id_key(row['object_id'])}|y={label}"
            for row, label in zip(rows, y, strict=True)
        ],
        dtype=object,
    )

    outer_assignments, five_fold_reason = _make_sgkf_assignments(
        y,
        groups,
        strata,
        n_splits=5,
        seed=outer_seed,
    )
    fallback_reason: str | None = None
    outer_n_splits = 5
    if outer_assignments is None:
        fallback_reason = five_fold_reason
        outer_n_splits = 4
        outer_assignments, four_fold_reason = _make_sgkf_assignments(
            y,
            groups,
            strata,
            n_splits=4,
            seed=outer_seed,
        )
        if outer_assignments is None:
            raise FoldSupportError(
                "frozen outer split failed at five folds "
                f"({five_fold_reason}) and at the single four-fold fallback "
                f"({four_fold_reason})"
            )

    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        outer_rows.append(
            {
                "target_id": row["target_id"],
                "physical_instance_id": row["physical_instance_id"],
                "object_id": row["object_id"],
                "y_failure": int(y[index]),
                "outer_fold": int(outer_assignments[index]),
            }
        )

    for outer_fold in range(outer_n_splits):
        test_mask = outer_assignments == outer_fold
        train_index = np.flatnonzero(~test_mask)
        test_index = np.flatnonzero(test_mask)
        inner_y = y[train_index]
        inner_groups = groups[train_index]
        inner_strata = strata[train_index]
        inner_assignments, inner_reason = _make_sgkf_assignments(
            inner_y,
            inner_groups,
            inner_strata,
            n_splits=4,
            seed=inner_seed_base + outer_fold,
        )
        if inner_assignments is None:
            raise FoldSupportError(
                f"outer fold {outer_fold} cannot support the frozen four-fold "
                f"inner split: {inner_reason}"
            )
        for local_index, global_index in enumerate(train_index):
            row = rows[int(global_index)]
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "target_id": row["target_id"],
                    "physical_instance_id": row["physical_instance_id"],
                    "inner_fold": int(inner_assignments[local_index]),
                }
            )
        summaries.append(
            {
                "outer_fold": outer_fold,
                "train_target_count": int(len(train_index)),
                "test_target_count": int(len(test_index)),
                "train_label_counts": _count_by(y[train_index]),
                "test_label_counts": _count_by(y[test_index]),
                "train_object_counts": _count_by(
                    [rows[int(index)]["object_id"] for index in train_index]
                ),
                "test_object_counts": _count_by(
                    [rows[int(index)]["object_id"] for index in test_index]
                ),
                "train_instance_count": int(len(np.unique(groups[train_index]))),
                "test_instance_count": int(len(np.unique(groups[test_index]))),
                "inner_seed": inner_seed_base + outer_fold,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "m6_g0_fold_manifest_v1",
        "splitter": "StratifiedGroupKFold",
        "strata": "object_id_x_y_failure",
        "group": "physical_instance_id",
        "outer_seed": outer_seed,
        "inner_seed_base": inner_seed_base,
        "outer_n_splits": outer_n_splits,
        "inner_n_splits": 4,
        "five_fold_fallback_reason": fallback_reason,
        "outer_assignments": outer_rows,
        "inner_assignments": inner_rows,
        "fold_summaries": summaries,
    }
    manifest["manifest_hash"] = fold_manifest_hash(manifest)
    validation = validate_fold_manifest(manifest, rows)
    if not validation["passed"]:
        raise AnalysisContractError(
            f"generated fold manifest failed validation: {validation['errors']}"
        )
    return _jsonable(manifest)


def validate_fold_manifest(
    manifest: Mapping[str, Any], evaluator_rows: Any
) -> dict[str, Any]:
    """Validate coverage, hashes, class support, and all group boundaries."""

    rows = _project_evaluator_records(evaluator_rows)
    _require_columns(
        rows,
        ("target_id", "y_failure", "physical_instance_id", "object_id"),
        name="evaluator_rows",
    )
    errors: list[str] = []
    checks: dict[str, bool] = {}
    expected_hash = fold_manifest_hash(manifest)
    checks["manifest_hash_matches"] = manifest.get("manifest_hash") == expected_hash
    if not checks["manifest_hash_matches"]:
        errors.append("manifest hash mismatch")

    evaluator_by_target = {_id_key(row["target_id"]): row for row in rows}
    outer_rows = list(manifest.get("outer_assignments", []))
    outer_keys = [_id_key(row.get("target_id")) for row in outer_rows]
    checks["one_outer_assignment_per_target"] = len(outer_keys) == len(
        set(outer_keys)
    ) == len(evaluator_by_target) and set(outer_keys) == set(evaluator_by_target)
    if not checks["one_outer_assignment_per_target"]:
        errors.append("outer target coverage is not exactly once")

    instance_folds: dict[str, set[int]] = {}
    metadata_matches = True
    for assignment in outer_rows:
        key = _id_key(assignment.get("target_id"))
        source = evaluator_by_target.get(key)
        if source is None:
            metadata_matches = False
            continue
        metadata_matches &= (
            _id_key(assignment.get("physical_instance_id"))
            == _id_key(source["physical_instance_id"])
            and _id_key(assignment.get("object_id")) == _id_key(source["object_id"])
            and int(assignment.get("y_failure", -1)) == int(source["y_failure"])
        )
        group_key = _id_key(source["physical_instance_id"])
        instance_folds.setdefault(group_key, set()).add(
            int(assignment.get("outer_fold", -1))
        )
    checks["outer_metadata_matches"] = bool(metadata_matches)
    checks["physical_instances_do_not_cross_outer_folds"] = all(
        len(folds) == 1 for folds in instance_folds.values()
    )
    if not checks["outer_metadata_matches"]:
        errors.append("outer assignments disagree with evaluator metadata")
    if not checks["physical_instances_do_not_cross_outer_folds"]:
        errors.append("a physical instance crosses outer folds")

    n_outer = int(manifest.get("outer_n_splits", 0))
    expected_folds = set(range(n_outer))
    observed_folds = {int(row.get("outer_fold", -1)) for row in outer_rows}
    checks["outer_fold_ids_complete"] = observed_folds == expected_folds
    if not checks["outer_fold_ids_complete"]:
        errors.append("outer fold IDs are incomplete")

    class_support = True
    for fold in expected_folds:
        test_labels = [
            int(row["y_failure"])
            for row in outer_rows
            if int(row["outer_fold"]) == fold
        ]
        train_labels = [
            int(row["y_failure"])
            for row in outer_rows
            if int(row["outer_fold"]) != fold
        ]
        class_support &= set(test_labels) == {0, 1} and set(train_labels) == {0, 1}
    checks["outer_binary_class_support"] = bool(class_support)
    if not class_support:
        errors.append("an outer partition lacks binary class support")

    inner_rows = list(manifest.get("inner_assignments", []))
    inner_valid = True
    inner_group_valid = True
    for outer_fold in expected_folds:
        outer_train = {
            _id_key(row["target_id"])
            for row in outer_rows
            if int(row["outer_fold"]) != outer_fold
        }
        outer_test = {
            _id_key(row["target_id"])
            for row in outer_rows
            if int(row["outer_fold"]) == outer_fold
        }
        selected = [
            row for row in inner_rows if int(row.get("outer_fold", -1)) == outer_fold
        ]
        selected_keys = [_id_key(row.get("target_id")) for row in selected]
        inner_valid &= (
            len(selected_keys) == len(set(selected_keys))
            and set(selected_keys) == outer_train
            and not set(selected_keys).intersection(outer_test)
            and {int(row.get("inner_fold", -1)) for row in selected} == set(range(4))
        )
        group_assignments: dict[str, set[int]] = {}
        for row in selected:
            group_assignments.setdefault(
                _id_key(row.get("physical_instance_id")), set()
            ).add(int(row.get("inner_fold", -1)))
        inner_group_valid &= all(
            len(folds) == 1 for folds in group_assignments.values()
        )
        for inner_fold in range(4):
            validation_keys = {
                _id_key(row["target_id"])
                for row in selected
                if int(row["inner_fold"]) == inner_fold
            }
            train_keys = outer_train - validation_keys
            validation_labels = {
                int(evaluator_by_target[key]["y_failure"]) for key in validation_keys
            }
            train_labels = {
                int(evaluator_by_target[key]["y_failure"]) for key in train_keys
            }
            inner_valid &= validation_labels == {0, 1} and train_labels == {0, 1}
    checks["inner_partitions_inside_outer_train"] = bool(inner_valid)
    checks["physical_instances_do_not_cross_inner_folds"] = bool(inner_group_valid)
    if not inner_valid:
        errors.append(
            "inner assignments violate coverage, containment, or class support"
        )
    if not inner_group_valid:
        errors.append("a physical instance crosses inner folds")

    return {"passed": all(checks.values()), "checks": checks, "errors": errors}


def _deterministic_order(
    risk_score: Sequence[float], tie_breaker: Sequence[Any] | None = None
) -> np.ndarray:
    risk = np.asarray(risk_score, dtype=float)
    if risk.ndim != 1:
        raise AnalysisContractError("risk_score must be one-dimensional")
    if tie_breaker is None:
        tie_breaker = list(range(len(risk)))
    if len(tie_breaker) != len(risk):
        raise AnalysisContractError("tie_breaker length does not match risk_score")
    safe_risk = np.where(np.isfinite(risk), risk, np.inf)
    # Preserve input order only as a deterministic serialization choice.  Equal
    # risks are handled analytically by ``risk_coverage_curve`` and must never
    # acquire a secondary ranking from a target/object/instance identifier.
    return np.argsort(safe_risk, kind="stable")


def _expected_tied_cumulative_failures(
    y_failure: np.ndarray,
    risk_score: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Return tie-invariant expected cumulative failures at every prefix.

    Within each equal-risk block, all permutations are equally likely.  At the
    ``j``-th position of a block containing ``f`` failures among ``m`` targets,
    the expected number of newly accepted failures is ``j * f / m``.  This
    preserves the complete discrete risk-coverage curve without allowing an ID
    or row-order convention to act as an undeclared secondary confidence score.
    """

    ordered_risk = np.asarray(risk_score, dtype=float)[order]
    ordered_risk = np.where(np.isfinite(ordered_risk), ordered_risk, np.inf)
    ordered_y = np.asarray(y_failure, dtype=int)[order]
    expected = np.empty(len(order), dtype=float)
    failures_before = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ordered_risk[end] == ordered_risk[start]:
            end += 1
        block_size = end - start
        block_failures = float(np.sum(ordered_y[start:end]))
        for offset in range(1, block_size + 1):
            expected[start + offset - 1] = (
                failures_before + offset * block_failures / block_size
            )
        failures_before += block_failures
        start = end
    return expected


def risk_coverage_curve(
    y_failure: Sequence[int],
    risk_score: Sequence[float],
    *,
    tie_breaker: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return the complete empirical risk curve from safest to riskiest.

    The AURC is the arithmetic mean of the cumulative empirical failure rates
    at every non-empty prefix.  Equal-risk blocks use the expectation over all
    within-block permutations, so identifiers never act as secondary ranking
    inputs.  ``ordered_indices`` retains stable input order only as a
    deterministic output representation.
    """

    y = _binary_vector(y_failure, name="y_failure")
    risk = np.asarray(risk_score, dtype=float)
    if len(y) != len(risk) or len(y) == 0:
        raise AnalysisContractError(
            "y_failure and risk_score need equal nonzero length"
        )
    order = _deterministic_order(risk, tie_breaker)
    accepted = np.arange(1, len(y) + 1, dtype=int)
    coverage = accepted / len(y)
    cumulative_failures = _expected_tied_cumulative_failures(y, risk, order)
    empirical_risk = cumulative_failures / accepted
    return {
        "accepted_count": accepted.tolist(),
        "coverage": coverage.tolist(),
        "empirical_failure_risk": empirical_risk.tolist(),
        "ordered_indices": order.tolist(),
        "equal_risk_tie_policy": "expected_over_all_within_block_permutations",
        "aurc": float(np.mean(empirical_risk)),
    }


def aurc(
    y_failure: Sequence[int],
    risk_score: Sequence[float],
    *,
    tie_breaker: Sequence[Any] | None = None,
) -> float:
    """Compute exact area under the complete empirical risk-coverage curve."""

    return float(
        risk_coverage_curve(
            y_failure,
            risk_score,
            tie_breaker=tie_breaker,
        )["aurc"]
    )


def _per_target_aurc_weights(
    y_failure: Sequence[int], risk_score: Sequence[float]
) -> np.ndarray:
    """Decompose tie-invariant AURC into additive per-target weights.

    Correct targets have zero weight.  Failures in the same equal-risk block
    receive the same expected contribution, matching the uniform within-block
    tie policy used by :func:`risk_coverage_curve`.
    """

    y = _binary_vector(y_failure, name="y_failure")
    risk = np.asarray(risk_score, dtype=float)
    if len(y) != len(risk) or len(y) == 0:
        raise AnalysisContractError(
            "AURC contribution vectors need equal nonzero length"
        )
    order = _deterministic_order(risk)
    ordered_risk = np.where(np.isfinite(risk[order]), risk[order], np.inf)
    inverse_prefix = 1.0 / np.arange(1, len(y) + 1, dtype=float)
    weights = np.zeros(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered_risk[end] == ordered_risk[start]:
            end += 1
        block_size = end - start
        within_block = float(
            np.sum(
                (np.arange(1, block_size + 1, dtype=float) / block_size)
                * inverse_prefix[start:end]
            )
        )
        after_block = float(np.sum(inverse_prefix[end:]))
        failure_weight = (within_block + after_block) / len(y)
        block_indices = order[start:end]
        weights[block_indices[y[block_indices] == 1]] = failure_weight
        start = end
    expected = aurc(y, risk)
    if not math.isclose(float(np.sum(weights)), expected, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisContractError("Per-target AURC weights do not reconstruct AURC")
    return weights


def _per_target_aurc_gain_contributions(
    y_failure: Sequence[int],
    candidate_risk: Sequence[float],
    baseline_risk: Sequence[float],
) -> np.ndarray:
    """Return additive baseline-minus-candidate AURC gain per target."""

    candidate_weights = _per_target_aurc_weights(y_failure, candidate_risk)
    baseline_weights = _per_target_aurc_weights(y_failure, baseline_risk)
    return baseline_weights - candidate_weights


def _expected_deferred_failure_contributions(
    y_failure: Sequence[int],
    risk_score: Sequence[float],
    *,
    accepted_count: int,
) -> tuple[np.ndarray, bool]:
    """Return tie-invariant expected deferred-failure contributions.

    Targets strictly above the accepted-prefix boundary are fully deferred and
    targets strictly below it are fully accepted.  If the boundary splits an
    equal-risk block, every target in that block receives the same deferred
    probability: the number of deferred slots in the block divided by its
    size.  Multiplying that probability by the failure indicator yields the
    frozen uniform-within-block expectation without using row or identifier
    order as a secondary confidence signal.
    """

    y = _binary_vector(y_failure, name="y_failure")
    risk = np.asarray(risk_score, dtype=float)
    if len(y) != len(risk) or len(y) == 0:
        raise AnalysisContractError(
            "Deferred-failure contribution vectors need equal nonzero length"
        )
    if accepted_count < 0 or accepted_count > len(y):
        raise AnalysisContractError("accepted_count is outside the target range")

    order = _deterministic_order(risk)
    ordered_risk = np.where(np.isfinite(risk[order]), risk[order], np.inf)
    deferred_probability = np.zeros(len(y), dtype=float)
    boundary_tie_split = False
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered_risk[end] == ordered_risk[start]:
            end += 1
        block_size = end - start
        accepted_in_block = min(max(accepted_count - start, 0), block_size)
        deferred_in_block = block_size - accepted_in_block
        deferred_probability[order[start:end]] = deferred_in_block / block_size
        boundary_tie_split |= 0 < accepted_in_block < block_size
        start = end

    return y * deferred_probability, boundary_tie_split


def _risk_at_coverage(curve: Mapping[str, Any], requested: float) -> float:
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["empirical_failure_risk"], dtype=float)
    index = min(int(np.searchsorted(coverage, requested, side="left")), len(risk) - 1)
    return float(risk[index])


def _coverage_at_empirical_risk(curve: Mapping[str, Any], maximum_risk: float) -> float:
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["empirical_failure_risk"], dtype=float)
    valid = coverage[risk <= maximum_risk]
    return float(np.max(valid)) if len(valid) else 0.0


def equal_frequency_calibration_error(
    y_failure: Sequence[int],
    failure_probability: Sequence[float],
    *,
    n_bins: int = 10,
    tie_breaker: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute ECE/MCE using deterministic, equal-frequency probability bins."""

    y = _binary_vector(y_failure, name="y_failure")
    probability = np.asarray(failure_probability, dtype=float)
    if len(y) != len(probability) or len(y) == 0:
        raise AnalysisContractError(
            "y_failure and failure_probability need equal nonzero length"
        )
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise AnalysisContractError("failure_probability must be finite and in [0, 1]")
    if n_bins < 1:
        raise AnalysisContractError("n_bins must be positive")
    order = _deterministic_order(probability, tie_breaker)
    ordered_probability = probability[order]
    ordered_y = y[order]
    expected_ordered_y = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered_probability[end] == ordered_probability[start]:
            end += 1
        expected_ordered_y[start:end] = float(np.mean(ordered_y[start:end]))
        start = end
    bins = np.array_split(np.arange(len(y), dtype=int), min(n_bins, len(y)))
    details: list[dict[str, Any]] = []
    weighted_error = 0.0
    maximum_error = 0.0
    for bin_index, positions in enumerate(bins):
        mean_probability = float(np.mean(ordered_probability[positions]))
        empirical_failure_rate = float(np.mean(expected_ordered_y[positions]))
        absolute_error = abs(mean_probability - empirical_failure_rate)
        weight = len(positions) / len(y)
        weighted_error += weight * absolute_error
        maximum_error = max(maximum_error, absolute_error)
        details.append(
            {
                "bin": bin_index,
                "count": int(len(positions)),
                "mean_probability": mean_probability,
                "empirical_failure_rate": empirical_failure_rate,
                "absolute_error": absolute_error,
                "weight": weight,
            }
        )
    return {
        "ece": float(weighted_error),
        "mce": float(maximum_error),
        "bins": details,
        "equal_probability_tie_policy": (
            "expected_failure_allocation_over_within_tie_permutations"
        ),
    }


def _calibration_slope_intercept(
    y: np.ndarray, probability: np.ndarray
) -> tuple[float | None, float | None]:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    if (
        positives < 10
        or negatives < 10
        or len(np.unique(np.round(logit, 12))) < 3
        or float(np.std(logit)) <= 1e-12
    ):
        return None, None
    x = logit.reshape(-1, 1)
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5000,
            random_state=OUTER_SEED,
        )
        model.fit(x, y)
    except (TypeError, ValueError):
        try:
            model = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=5000,
                random_state=OUTER_SEED,
            )
            model.fit(x, y)
        except (TypeError, ValueError):
            return None, None
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _safe_discrimination(
    y: np.ndarray, risk_score: np.ndarray
) -> tuple[float | None, float | None]:
    if set(np.unique(y)) != {0, 1}:
        return None, None
    finite = np.isfinite(risk_score)
    if not np.all(finite):
        finite_values = risk_score[finite]
        replacement = float(np.max(finite_values) + 1.0) if len(finite_values) else 0.0
        risk_score = np.where(finite, risk_score, replacement)
    return (
        float(roc_auc_score(y, risk_score)),
        float(average_precision_score(y, risk_score)),
    )


def compute_prediction_metrics(
    y_failure: Sequence[int],
    risk_score: Sequence[float],
    *,
    failure_probability: Sequence[float] | None,
    tie_breaker: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute all frozen discrimination, calibration, and selective metrics.

    Pass ``failure_probability=None`` for ``RAW_SCORE_RANK``.  Its ranking and
    selective metrics remain available while every calibration field is null.
    """

    y = _binary_vector(y_failure, name="y_failure")
    risk = np.asarray(risk_score, dtype=float)
    if len(y) != len(risk) or len(y) == 0:
        raise AnalysisContractError("metric vectors need equal nonzero length")
    if tie_breaker is not None and len(tie_breaker) != len(y):
        raise AnalysisContractError("tie_breaker length does not match metric vectors")
    curve = risk_coverage_curve(y, risk, tie_breaker=tie_breaker)
    auroc, auprc = _safe_discrimination(y, risk)
    prevalence = float(np.mean(y))
    metrics: dict[str, Any] = {
        "target_count": int(len(y)),
        "failure_count": int(y.sum()),
        "failure_prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift_over_prevalence": (
            float(auprc / prevalence) if auprc is not None and prevalence > 0 else None
        ),
        "risk_coverage_curve": curve,
        "aurc": float(curve["aurc"]),
        "risk_at_coverage_0_50": _risk_at_coverage(curve, 0.50),
        "risk_at_coverage_0_80": _risk_at_coverage(curve, 0.80),
        "risk_at_coverage_0_90": _risk_at_coverage(curve, 0.90),
        "coverage_at_empirical_risk_0_05": _coverage_at_empirical_risk(curve, 0.05),
        "coverage_at_empirical_risk_0_10": _coverage_at_empirical_risk(curve, 0.10),
        "coverage_at_empirical_risk_0_20": _coverage_at_empirical_risk(curve, 0.20),
        "brier": None,
        "log_loss": None,
        "ece": None,
        "mce": None,
        "calibration_slope": None,
        "calibration_intercept": None,
        "calibration_bins": None,
        "calibration_equal_probability_tie_policy": None,
        "predicted_risk_le_0_05_count": None,
        "predicted_risk_le_0_05_coverage": None,
        "predicted_risk_le_0_05_empirical_failure_rate": None,
    }
    if failure_probability is None:
        return _jsonable(metrics)

    probability = np.asarray(failure_probability, dtype=float)
    if len(probability) != len(y):
        raise AnalysisContractError("failure_probability length does not match labels")
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise AnalysisContractError("failure_probability must be finite and in [0, 1]")
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    calibration = equal_frequency_calibration_error(
        y,
        probability,
        n_bins=10,
        tie_breaker=tie_breaker,
    )
    slope, intercept = _calibration_slope_intercept(y, probability)
    accepted = probability <= 0.05
    accepted_count = int(accepted.sum())
    metrics.update(
        {
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
            "ece": calibration["ece"],
            "mce": calibration["mce"],
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibration_bins": calibration["bins"],
            "calibration_equal_probability_tie_policy": calibration[
                "equal_probability_tie_policy"
            ],
            "predicted_risk_le_0_05_count": accepted_count,
            "predicted_risk_le_0_05_coverage": accepted_count / len(y),
            "predicted_risk_le_0_05_empirical_failure_rate": (
                float(np.mean(y[accepted])) if accepted_count else None
            ),
        }
    )
    return _jsonable(metrics)


def relative_aurc_improvement(
    baseline_aurc: float | None, candidate_aurc: float | None
) -> float | None:
    """Return ``(baseline - candidate) / baseline`` with explicit edge handling."""

    if baseline_aurc is None or candidate_aurc is None or baseline_aurc <= 0:
        return None
    return float((baseline_aurc - candidate_aurc) / baseline_aurc)


def _oriented_raw_risk(
    raw_score: Sequence[float], higher_is_confident: bool
) -> np.ndarray:
    score = np.asarray(raw_score, dtype=float)
    risk = -score if higher_is_confident else score.copy()
    finite = np.isfinite(risk)
    if np.any(finite):
        finite_values = risk[finite]
        span = max(float(np.ptp(finite_values)), 1.0)
        risk[~finite] = float(np.max(finite_values) + span)
    else:
        risk[:] = 0.0
    return risk


def _project_feature_records(feature_rows: Any) -> list[dict[str, Any]]:
    """Project persisted nested rows to target ID plus inference-time features."""

    rows = _as_records(feature_rows, name="feature_rows")
    projected: list[dict[str, Any]] = []
    for row in rows:
        if "features" not in row:
            projected.append(row)
            continue
        values = row["features"]
        if not isinstance(values, Mapping):
            raise AnalysisContractError(
                "nested feature_rows.features must be a mapping"
            )
        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise AnalysisContractError(
                "nested feature_rows.metadata must be a mapping"
            )
        target_id = row.get(
            "target_id",
            row.get(
                "target_sample_id",
                metadata.get("target_id", metadata.get("target_sample_id")),
            ),
        )
        if target_id is None:
            raise AnalysisContractError("nested feature row has no target_id")
        if "target_id" in values:
            raise AnalysisContractError(
                "target_id must not be inside the feature mapping"
            )
        projected.append({"target_id": target_id, **dict(values)})
    return projected


def validate_feature_schema(
    feature_rows: Any,
    feature_names: Sequence[str],
    *,
    target_id_field: str = "target_id",
) -> dict[str, Any]:
    """Prove that evaluator fields and identifiers cannot enter the matrix."""

    rows = _project_feature_records(feature_rows)
    _require_columns(rows, (target_id_field, *feature_names), name="feature_rows")
    errors: list[str] = []
    target_keys = [_id_key(row[target_id_field]) for row in rows]
    if len(target_keys) != len(set(target_keys)):
        errors.append("feature rows do not have unique target IDs")

    selected_lower = [name.lower() for name in feature_names]
    prohibited_selected = sorted(
        {
            name
            for name, lowered in zip(feature_names, selected_lower, strict=True)
            if any(token in lowered for token in PROHIBITED_FEATURE_TOKENS)
        }
    )
    if prohibited_selected:
        errors.append(f"prohibited model features selected: {prohibited_selected}")

    evaluator_only_fields = {
        "y_failure",
        "physical_instance_id",
        "object_id",
        "scene_id",
        "split_id",
        "correct",
        "success",
        "pose_error",
        "ground_truth_pose",
    }
    present_fields = {field.lower() for row in rows for field in row}
    present_evaluator_fields = sorted(
        evaluator_only_fields.intersection(present_fields)
    )
    if present_evaluator_fields:
        errors.append(
            f"feature rows contain evaluator-only fields: {present_evaluator_fields}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "feature_names": list(feature_names),
        "target_count": len(rows),
    }


def _coerce_feature_value(value: Any, feature_name: str) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"feature {feature_name!r} contains a non-numeric value {value!r}"
        ) from exc
    return result


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    if not feature_names:
        raise AnalysisContractError("at least one model feature is required")
    return np.asarray(
        [
            [_coerce_feature_value(row[name], name) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )


def _build_logistic(c_value: float) -> Pipeline:
    return Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=OUTER_SEED,
                ),
            ),
        )
    )


def _build_tree(parameters: Mapping[str, int], *, seed: int) -> Pipeline:
    return Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(parameters["n_estimators"]),
                    max_depth=int(parameters["max_depth"]),
                    min_samples_leaf=int(parameters["min_samples_leaf"]),
                    criterion="log_loss",
                    class_weight=None,
                    bootstrap=True,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        )
    )


def _positive_probability(model: BaseEstimator, x: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x), dtype=float)
    classes = (
        np.asarray(model.classes_) if hasattr(model, "classes_") else np.asarray([0, 1])
    )
    positive_locations = np.flatnonzero(classes == 1)
    if len(positive_locations) != 1:
        raise AnalysisContractError("fitted classifier has no unique positive class")
    return probability[:, int(positive_locations[0])]


def _fit_sigmoid_calibrator(
    raw_probability: np.ndarray, y: np.ndarray
) -> BaseEstimator:
    if len(np.unique(y)) < 2:
        return _ConstantProbabilityModel(float(np.mean(y)))
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=OUTER_SEED,
    )
    calibrator.fit(logits, y)
    return calibrator


def _apply_sigmoid_calibrator(
    calibrator: BaseEstimator, raw_probability: np.ndarray
) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if isinstance(calibrator, _ConstantProbabilityModel):
        return calibrator.predict_proba(logits)[:, 1]
    return _positive_probability(calibrator, logits)


def _inner_oof_logistic(
    x: np.ndarray,
    y: np.ndarray,
    inner_folds: np.ndarray,
    c_value: float,
) -> np.ndarray:
    prediction = np.full(len(y), np.nan, dtype=float)
    for inner_fold in range(4):
        validation = inner_folds == inner_fold
        training = ~validation
        model = _build_logistic(c_value)
        model.fit(x[training], y[training])
        prediction[validation] = _positive_probability(model, x[validation])
    if np.any(~np.isfinite(prediction)):
        raise AnalysisContractError("logistic inner OOF predictions are incomplete")
    return prediction


def _inner_oof_tree(
    x: np.ndarray,
    y: np.ndarray,
    inner_folds: np.ndarray,
    parameters: Mapping[str, int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, BaseEstimator]:
    raw_prediction = np.full(len(y), np.nan, dtype=float)
    for inner_fold in range(4):
        validation = inner_folds == inner_fold
        training = ~validation
        model = _build_tree(parameters, seed=seed + inner_fold)
        model.fit(x[training], y[training])
        raw_prediction[validation] = _positive_probability(model, x[validation])
    if np.any(~np.isfinite(raw_prediction)):
        raise AnalysisContractError("tree inner OOF predictions are incomplete")
    calibrator = _fit_sigmoid_calibrator(raw_prediction, y)
    calibrated = _apply_sigmoid_calibrator(calibrator, raw_prediction)
    return raw_prediction, calibrated, calibrator


def _mean_fold_aurc(
    y: np.ndarray,
    prediction: np.ndarray,
    inner_folds: np.ndarray,
    target_ids: Sequence[Any],
) -> tuple[float, list[float]]:
    values: list[float] = []
    ids = np.asarray(target_ids, dtype=object)
    for inner_fold in range(4):
        selected = inner_folds == inner_fold
        values.append(
            aurc(
                y[selected],
                prediction[selected],
                tie_breaker=ids[selected].tolist(),
            )
        )
    return float(np.mean(values)), values


def _select_logistic(
    x: np.ndarray,
    y: np.ndarray,
    inner_folds: np.ndarray,
    target_ids: Sequence[Any],
) -> tuple[float, list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    for c_value in LOGISTIC_C_GRID:
        prediction = _inner_oof_logistic(x, y, inner_folds, c_value)
        mean_value, fold_values = _mean_fold_aurc(
            y, prediction, inner_folds, target_ids
        )
        audit.append(
            {
                "family": "logistic",
                "parameters": {"C": c_value},
                "inner_mean_aurc": mean_value,
                "inner_fold_aurc": fold_values,
            }
        )
    chosen = min(
        audit, key=lambda row: (row["inner_mean_aurc"], row["parameters"]["C"])
    )
    return float(chosen["parameters"]["C"]), audit


def _select_tree(
    x: np.ndarray,
    y: np.ndarray,
    inner_folds: np.ndarray,
    target_ids: Sequence[Any],
    *,
    seed: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    for grid_index, parameters in enumerate(TREE_PARAM_GRID):
        raw_prediction, _, _ = _inner_oof_tree(
            x,
            y,
            inner_folds,
            parameters,
            seed=seed + 100 * grid_index,
        )
        mean_value, fold_values = _mean_fold_aurc(
            y, raw_prediction, inner_folds, target_ids
        )
        audit.append(
            {
                "family": "tree",
                "parameters": dict(parameters),
                "inner_mean_aurc": mean_value,
                "inner_fold_aurc": fold_values,
                "calibration": "sigmoid_fit_to_training_only_inner_oof",
            }
        )
    chosen = min(
        audit,
        key=lambda row: (
            row["inner_mean_aurc"],
            row["parameters"]["max_depth"],
            row["parameters"]["n_estimators"],
            -row["parameters"]["min_samples_leaf"],
        ),
    )
    return dict(chosen["parameters"]), audit


def _logistic_coefficients(
    fitted_model: Pipeline, feature_names: Sequence[str], outer_fold: int
) -> list[dict[str, Any]]:
    imputer = fitted_model.named_steps["imputer"]
    transformed_names = list(imputer.get_feature_names_out(feature_names))
    coefficients = fitted_model.named_steps["classifier"].coef_[0]
    if len(transformed_names) != len(coefficients):
        transformed_names = [
            f"transformed_feature_{index}" for index in range(len(coefficients))
        ]
    return [
        {
            "outer_fold": outer_fold,
            "feature": name,
            "standardized_coefficient": float(value),
        }
        for name, value in zip(transformed_names, coefficients, strict=True)
    ]


def _tree_permutation_importance(
    model: Pipeline,
    calibrator: BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    target_ids: Sequence[Any],
    *,
    outer_fold: int,
    n_repeats: int = 10,
) -> list[dict[str, Any]]:
    baseline_raw = _positive_probability(model, x_test)
    baseline_probability = _apply_sigmoid_calibrator(calibrator, baseline_raw)
    baseline_aurc = aurc(y_test, baseline_probability, tie_breaker=target_ids)
    rng = np.random.default_rng(PERMUTATION_SEED_BASE + outer_fold)
    records: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        increases: list[float] = []
        for _ in range(n_repeats):
            permuted = x_test.copy()
            order = rng.permutation(len(x_test))
            permuted[:, feature_index] = permuted[order, feature_index]
            raw = _positive_probability(model, permuted)
            probability = _apply_sigmoid_calibrator(calibrator, raw)
            increases.append(
                aurc(y_test, probability, tie_breaker=target_ids) - baseline_aurc
            )
        records.append(
            {
                "outer_fold": outer_fold,
                "feature": feature_name,
                "metric": "outer_test_aurc_increase_when_permuted",
                "repeats": n_repeats,
                "importance_mean": float(np.mean(increases)),
                "importance_std": float(np.std(increases, ddof=1)),
            }
        )
    return records


def _training_only_confidence(
    raw_score: np.ndarray,
    training_indices: np.ndarray,
    *,
    higher_is_confident: bool,
) -> np.ndarray:
    confidence = raw_score.copy() if higher_is_confident else -raw_score
    training_values = confidence[training_indices]
    finite_training = training_values[np.isfinite(training_values)]
    if not len(finite_training):
        raise AnalysisContractError(
            "raw score is entirely missing in an outer-training partition"
        )
    span = max(float(np.ptp(finite_training)), 1.0)
    missing_value = float(np.min(finite_training) - span)
    return np.where(np.isfinite(confidence), confidence, missing_value)


def _coefficient_stability(
    coefficient_rows: Sequence[Mapping[str, Any]], outer_n_splits: int
) -> list[dict[str, Any]]:
    by_feature: dict[str, dict[int, float]] = {}
    for row in coefficient_rows:
        by_feature.setdefault(str(row["feature"]), {})[int(row["outer_fold"])] = float(
            row["standardized_coefficient"]
        )
    result: list[dict[str, Any]] = []
    for feature, fold_values in sorted(by_feature.items()):
        values = np.asarray(
            [fold_values.get(fold, 0.0) for fold in range(outer_n_splits)],
            dtype=float,
        )
        nonzero = values[np.abs(values) > 1e-12]
        if len(nonzero):
            positive_fraction = float(np.mean(nonzero > 0))
            sign_consistency = max(positive_fraction, 1.0 - positive_fraction)
        else:
            sign_consistency = 1.0
        result.append(
            {
                "feature": feature,
                "outer_fold_count": outer_n_splits,
                "coefficient_mean": float(np.mean(values)),
                "coefficient_std": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "coefficient_min": float(np.min(values)),
                "coefficient_max": float(np.max(values)),
                "sign_consistency": sign_consistency,
                "fold_coefficients": values.tolist(),
            }
        )
    return result


def _cross_validated_models(
    x: np.ndarray,
    y: np.ndarray,
    raw_score: np.ndarray,
    target_ids: Sequence[Any],
    outer_folds: np.ndarray,
    inner_folds_by_outer: Mapping[int, Mapping[str, int]],
    feature_names: Sequence[str],
    *,
    raw_score_higher_is_confident: bool,
    collect_contributions: bool,
) -> dict[str, Any]:
    n_targets = len(y)
    predictions = {
        method: np.full(n_targets, np.nan, dtype=float) for method in FORMAL_METHODS
    }
    predictions["RAW_SCORE_RANK"] = _oriented_raw_risk(
        raw_score, raw_score_higher_is_confident
    )
    probabilities: dict[str, np.ndarray | None] = {
        "RAW_SCORE_RANK": None,
        **{
            method: np.full(n_targets, np.nan, dtype=float)
            for method in FORMAL_METHODS
            if method != "RAW_SCORE_RANK"
        },
    }
    selection_audit: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    ids_array = np.asarray(target_ids, dtype=object)
    outer_n_splits = len(np.unique(outer_folds))

    for outer_fold in range(outer_n_splits):
        test_indices = np.flatnonzero(outer_folds == outer_fold)
        train_indices = np.flatnonzero(outer_folds != outer_fold)
        train_id_to_inner = inner_folds_by_outer[outer_fold]
        inner_folds = np.asarray(
            [train_id_to_inner[_id_key(target_ids[index])] for index in train_indices],
            dtype=int,
        )
        if set(np.unique(inner_folds)) != {0, 1, 2, 3}:
            raise AnalysisContractError(
                f"outer fold {outer_fold} does not contain all four inner folds"
            )
        x_train = x[train_indices]
        y_train = y[train_indices]
        train_ids = ids_array[train_indices].tolist()
        x_test = x[test_indices]
        y_test = y[test_indices]
        test_ids = ids_array[test_indices].tolist()

        confidence = _training_only_confidence(
            raw_score,
            train_indices,
            higher_is_confident=raw_score_higher_is_confident,
        )
        isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
        isotonic.fit(confidence[train_indices], y_train)
        isotonic_probability = np.clip(
            isotonic.predict(confidence[test_indices]), 0.0, 1.0
        )
        predictions["SCORE_ISOTONIC"][test_indices] = isotonic_probability
        probabilities["SCORE_ISOTONIC"][test_indices] = isotonic_probability

        selected_c, logistic_audit = _select_logistic(
            x_train,
            y_train,
            inner_folds,
            train_ids,
        )
        logistic_model = _build_logistic(selected_c)
        logistic_model.fit(x_train, y_train)
        logistic_probability = _positive_probability(logistic_model, x_test)
        predictions["LOGISTIC_MULTIFEATURE"][test_indices] = logistic_probability
        probabilities["LOGISTIC_MULTIFEATURE"][test_indices] = logistic_probability

        selected_tree, tree_audit = _select_tree(
            x_train,
            y_train,
            inner_folds,
            train_ids,
            seed=INNER_SEED_BASE + outer_fold,
        )
        _, _, tree_calibrator = _inner_oof_tree(
            x_train,
            y_train,
            inner_folds,
            selected_tree,
            seed=INNER_SEED_BASE + outer_fold,
        )
        tree_model = _build_tree(
            selected_tree,
            seed=OUTER_SEED + outer_fold,
        )
        tree_model.fit(x_train, y_train)
        tree_raw_probability = _positive_probability(tree_model, x_test)
        tree_probability = _apply_sigmoid_calibrator(
            tree_calibrator, tree_raw_probability
        )
        predictions["SHALLOW_TREE_MULTIFEATURE"][test_indices] = tree_probability
        probabilities["SHALLOW_TREE_MULTIFEATURE"][test_indices] = tree_probability

        best_logistic = min(
            logistic_audit,
            key=lambda row: (row["inner_mean_aurc"], row["parameters"]["C"]),
        )
        best_tree = min(
            tree_audit,
            key=lambda row: (
                row["inner_mean_aurc"],
                row["parameters"]["max_depth"],
                row["parameters"]["n_estimators"],
                -row["parameters"]["min_samples_leaf"],
            ),
        )
        tree_advantage = float(
            best_logistic["inner_mean_aurc"] - best_tree["inner_mean_aurc"]
        )
        if tree_advantage > 0.005:
            nested_family = "tree"
            nested_probability = tree_probability
            nested_parameters: dict[str, Any] = dict(selected_tree)
            tie_rule_applied = False
        else:
            nested_family = "logistic"
            nested_probability = logistic_probability
            nested_parameters = {"C": selected_c}
            tie_rule_applied = abs(tree_advantage) <= 0.005
        predictions["NESTED_MULTIFEATURE"][test_indices] = nested_probability
        probabilities["NESTED_MULTIFEATURE"][test_indices] = nested_probability

        selection_audit.append(
            {
                "outer_fold": outer_fold,
                "outer_train_target_count": int(len(train_indices)),
                "outer_test_target_count": int(len(test_indices)),
                "selection_data": "outer_train_inner_grouped_cv_only",
                "objective": "inner_mean_aurc",
                "logistic_candidates": logistic_audit,
                "tree_candidates": tree_audit,
                "fixed_logistic_selected_parameters": {"C": selected_c},
                "fixed_tree_selected_parameters": selected_tree,
                "nested_selected_family": nested_family,
                "nested_selected_parameters": nested_parameters,
                "best_logistic_inner_mean_aurc": float(
                    best_logistic["inner_mean_aurc"]
                ),
                "best_tree_inner_mean_aurc": float(best_tree["inner_mean_aurc"]),
                "tree_inner_aurc_advantage": tree_advantage,
                "logistic_tie_tolerance": 0.005,
                "logistic_tie_rule_applied": tie_rule_applied,
                "outer_test_labels_used_for_selection": False,
                "tree_calibration": "sigmoid_from_outer_train_inner_oof_only",
            }
        )
        if collect_contributions:
            coefficient_rows.extend(
                _logistic_coefficients(logistic_model, feature_names, outer_fold)
            )
            permutation_rows.extend(
                _tree_permutation_importance(
                    tree_model,
                    tree_calibrator,
                    x_test,
                    y_test,
                    feature_names,
                    test_ids,
                    outer_fold=outer_fold,
                )
            )

    for method in FORMAL_METHODS:
        if np.any(~np.isfinite(predictions[method])):
            raise AnalysisContractError(f"{method} OOF predictions are incomplete")
        probability = probabilities[method]
        if probability is not None and np.any(~np.isfinite(probability)):
            raise AnalysisContractError(f"{method} OOF probabilities are incomplete")
    return {
        "risk_scores": predictions,
        "probabilities": probabilities,
        "model_selections": selection_audit,
        "coefficient_rows": coefficient_rows,
        "coefficient_stability": _coefficient_stability(
            coefficient_rows, outer_n_splits
        ),
        "tree_permutation_importance": permutation_rows,
    }


def validate_oof_predictions(
    prediction_rows: Any,
    target_ids: Sequence[Any],
    *,
    methods: Sequence[str] = FORMAL_METHODS,
) -> dict[str, Any]:
    """Check that each target has exactly one prediction for every method."""

    rows = _as_records(prediction_rows, name="prediction_rows")
    expected_targets = {_id_key(target_id) for target_id in target_ids}
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("method")), _id_key(row.get("target_id")))
        counts[key] = counts.get(key, 0) + 1
    errors: list[str] = []
    for method in methods:
        observed = {
            target_key
            for (observed_method, target_key), count in counts.items()
            if observed_method == method and count == 1
        }
        if observed != expected_targets:
            errors.append(f"{method} target coverage is not exactly once")
        duplicates = [
            target_key
            for (observed_method, target_key), count in counts.items()
            if observed_method == method and count != 1
        ]
        if duplicates:
            errors.append(f"{method} has duplicate target predictions")
    unexpected_methods = sorted({str(row.get("method")) for row in rows} - set(methods))
    if unexpected_methods:
        errors.append(f"unexpected methods: {unexpected_methods}")
    expected_row_count = len(expected_targets) * len(methods)
    if len(rows) != expected_row_count:
        errors.append(
            f"prediction row count {len(rows)} != expected {expected_row_count}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "target_count": len(expected_targets),
        "method_count": len(methods),
        "prediction_row_count": len(rows),
    }


def grouped_bootstrap_indices(
    physical_instance_ids: Sequence[Any],
    *,
    n_resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Draw deterministic resamples whose atomic units are whole instances."""

    if n_resamples < 1:
        raise AnalysisContractError("n_resamples must be positive")
    group_keys = np.asarray(
        [_id_key(value) for value in physical_instance_ids], dtype=object
    )
    unique_groups = sorted(set(group_keys.tolist()))
    if not unique_groups:
        raise AnalysisContractError("physical_instance_ids must not be empty")
    group_indices = {
        group: np.flatnonzero(group_keys == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    resamples: list[np.ndarray] = []
    for _ in range(n_resamples):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        resamples.append(
            np.concatenate([group_indices[unique_groups[index]] for index in selected])
        )
    return resamples


def _bootstrap_metric_values(
    y: np.ndarray,
    risk_score: np.ndarray,
    probability: np.ndarray | None,
    tie_breaker: Sequence[Any],
) -> dict[str, float | None]:
    auroc, auprc = _safe_discrimination(y, risk_score)
    curve = risk_coverage_curve(y, risk_score, tie_breaker=tie_breaker)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(brier_score_loss(y, probability))
        if probability is not None
        else None,
        "aurc": float(curve["aurc"]),
        "risk_at_coverage_0_80": _risk_at_coverage(curve, 0.80),
        "predicted_risk_le_0_05_coverage": (
            float(np.mean(probability <= 0.05)) if probability is not None else None
        ),
    }


def _percentile_interval(values: Sequence[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(value)
        ],
        dtype=float,
    )
    if not len(finite):
        return {"lower": None, "upper": None, "valid_resamples": 0}
    lower, upper = np.percentile(finite, (2.5, 97.5))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_resamples": int(len(finite)),
    }


def grouped_bootstrap(
    y_failure: Sequence[int],
    risk_scores_by_method: Mapping[str, Sequence[float]],
    probabilities_by_method: Mapping[str, Sequence[float] | None],
    physical_instance_ids: Sequence[Any],
    target_ids: Sequence[Any],
    *,
    candidate_method: str = "NESTED_MULTIFEATURE",
    n_resamples: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run paired percentile bootstrap over complete physical instances."""

    if n_resamples < 2000:
        raise AnalysisContractError(
            "the frozen analysis requires at least 2,000 resamples"
        )
    y = _binary_vector(y_failure, name="y_failure")
    if not (len(y) == len(physical_instance_ids) == len(target_ids)):
        raise AnalysisContractError("bootstrap metadata lengths do not match labels")
    risk = {
        method: np.asarray(values, dtype=float)
        for method, values in risk_scores_by_method.items()
    }
    probability = {
        method: None if values is None else np.asarray(values, dtype=float)
        for method, values in probabilities_by_method.items()
    }
    for method, values in risk.items():
        if len(values) != len(y):
            raise AnalysisContractError(
                f"{method} bootstrap prediction length mismatch"
            )
        if method not in probability:
            raise AnalysisContractError(f"{method} missing bootstrap probability entry")
        if probability[method] is not None and len(probability[method]) != len(y):
            raise AnalysisContractError(
                f"{method} bootstrap probability length mismatch"
            )
    for required in (candidate_method, "RAW_SCORE_RANK", "SCORE_ISOTONIC"):
        if required not in risk:
            raise AnalysisContractError(f"bootstrap missing required method {required}")

    resampled_indices = grouped_bootstrap_indices(
        physical_instance_ids,
        n_resamples=n_resamples,
        seed=seed,
    )
    per_method_values: dict[str, dict[str, list[float | None]]] = {
        method: {
            "auroc": [],
            "auprc": [],
            "brier": [],
            "aurc": [],
            "risk_at_coverage_0_80": [],
            "predicted_risk_le_0_05_coverage": [],
        }
        for method in risk
    }
    paired_values: dict[str, list[float | None]] = {
        "aurc_improvement_vs_raw_score_rank": [],
        "aurc_improvement_vs_score_isotonic": [],
        "risk_at_coverage_0_80_difference_vs_raw_score_rank": [],
        "risk_at_coverage_0_80_difference_vs_score_isotonic": [],
        "coverage_at_predicted_risk_0_05": [],
    }
    target_array = np.asarray(target_ids, dtype=object)
    for indices in resampled_indices:
        sample_metrics: dict[str, dict[str, float | None]] = {}
        sampled_ids = [
            f"bootstrap_position={position}|target={target_array[index]}"
            for position, index in enumerate(indices)
        ]
        for method in risk:
            sampled_probability = (
                None if probability[method] is None else probability[method][indices]
            )
            values = _bootstrap_metric_values(
                y[indices],
                risk[method][indices],
                sampled_probability,
                sampled_ids,
            )
            sample_metrics[method] = values
            for metric_name, metric_value in values.items():
                per_method_values[method][metric_name].append(metric_value)

        candidate = sample_metrics[candidate_method]
        raw = sample_metrics["RAW_SCORE_RANK"]
        isotonic = sample_metrics["SCORE_ISOTONIC"]
        paired_values["aurc_improvement_vs_raw_score_rank"].append(
            float(raw["aurc"] - candidate["aurc"])
        )
        paired_values["aurc_improvement_vs_score_isotonic"].append(
            float(isotonic["aurc"] - candidate["aurc"])
        )
        paired_values["risk_at_coverage_0_80_difference_vs_raw_score_rank"].append(
            float(candidate["risk_at_coverage_0_80"] - raw["risk_at_coverage_0_80"])
        )
        paired_values["risk_at_coverage_0_80_difference_vs_score_isotonic"].append(
            float(
                candidate["risk_at_coverage_0_80"] - isotonic["risk_at_coverage_0_80"]
            )
        )
        paired_values["coverage_at_predicted_risk_0_05"].append(
            candidate["predicted_risk_le_0_05_coverage"]
        )

    method_intervals = {
        method: {
            metric_name: _percentile_interval(values)
            for metric_name, values in metrics.items()
        }
        for method, metrics in per_method_values.items()
    }
    paired_intervals = {
        name: _percentile_interval(values) for name, values in paired_values.items()
    }
    return _jsonable(
        {
            "resampling_unit": "physical_instance_id",
            "paired": True,
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence_level": 0.95,
            "interval": "percentile",
            "method_intervals": method_intervals,
            "paired_differences": paired_intervals,
        }
    )


def _gain_after_removal(
    y: np.ndarray,
    candidate_risk: np.ndarray,
    isotonic_risk: np.ndarray,
    target_ids: np.ndarray,
    keep: np.ndarray,
) -> float | None:
    if int(np.sum(keep)) == 0:
        return None
    candidate_aurc = aurc(
        y[keep], candidate_risk[keep], tie_breaker=target_ids[keep].tolist()
    )
    isotonic_aurc = aurc(
        y[keep], isotonic_risk[keep], tie_breaker=target_ids[keep].tolist()
    )
    return float(isotonic_aurc - candidate_aurc)


def object_robustness_analysis(
    y_failure: Sequence[int],
    candidate_risk: Sequence[float],
    isotonic_risk: Sequence[float],
    object_ids: Sequence[Any],
    physical_instance_ids: Sequence[Any],
    target_ids: Sequence[Any],
) -> dict[str, Any]:
    """Compute object metrics, jackknives, and frozen object-driven flags."""

    y = _binary_vector(y_failure, name="y_failure")
    candidate = np.asarray(candidate_risk, dtype=float)
    isotonic = np.asarray(isotonic_risk, dtype=float)
    if not (
        len(y)
        == len(candidate)
        == len(isotonic)
        == len(object_ids)
        == len(physical_instance_ids)
        == len(target_ids)
    ):
        raise AnalysisContractError("object robustness vector lengths do not match")
    objects = np.asarray([_id_key(value) for value in object_ids], dtype=object)
    instances = np.asarray(
        [_id_key(value) for value in physical_instance_ids], dtype=object
    )
    targets = np.asarray(target_ids, dtype=object)
    object_display = {_id_key(value): value for value in object_ids}
    instance_display = {_id_key(value): value for value in physical_instance_ids}
    full_candidate_aurc = aurc(y, candidate, tie_breaker=target_ids)
    full_isotonic_aurc = aurc(y, isotonic, tie_breaker=target_ids)
    full_gain = float(full_isotonic_aurc - full_candidate_aurc)
    target_gain_contributions = _per_target_aurc_gain_contributions(
        y,
        candidate,
        isotonic,
    )
    reconstructed_gain = float(np.sum(target_gain_contributions))
    if not math.isclose(reconstructed_gain, full_gain, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisContractError(
            "Per-target AURC-gain contributions do not reconstruct aggregate gain"
        )

    per_object: list[dict[str, Any]] = []
    jackknife: list[dict[str, Any]] = []
    for object_key in sorted(set(objects.tolist())):
        selected = objects == object_key
        object_y = y[selected]
        object_candidate = candidate[selected]
        object_isotonic = isotonic[selected]
        object_targets = targets[selected].tolist()
        object_auroc, _ = _safe_discrimination(object_y, object_candidate)
        object_curve = risk_coverage_curve(
            object_y,
            object_candidate,
            tie_breaker=object_targets,
        )
        object_isotonic_aurc = aurc(
            object_y,
            object_isotonic,
            tie_breaker=object_targets,
        )
        per_object.append(
            {
                "object_id": object_display[object_key],
                "target_count": int(np.sum(selected)),
                "failure_count": int(np.sum(object_y)),
                "failure_prevalence": float(np.mean(object_y)),
                "auroc": object_auroc,
                "aurc": float(object_curve["aurc"]),
                "isotonic_aurc": object_isotonic_aurc,
                "candidate_vs_isotonic_absolute_aurc_gain": float(
                    object_isotonic_aurc - object_curve["aurc"]
                ),
                "risk_at_coverage_0_80": (
                    _risk_at_coverage(object_curve, 0.80)
                    if int(np.sum(selected)) >= 5
                    else None
                ),
            }
        )
        keep = ~selected
        leave_out_gain = _gain_after_removal(
            y,
            candidate,
            isotonic,
            targets,
            keep,
        )
        leave_out_gain_change = (
            float(full_gain - leave_out_gain) if leave_out_gain is not None else None
        )
        additive_contribution = float(np.sum(target_gain_contributions[selected]))
        fraction = float(additive_contribution / full_gain) if full_gain > 0 else None
        jackknife.append(
            {
                "removed_object_id": object_display[object_key],
                "remaining_target_count": int(np.sum(keep)),
                "candidate_minus_isotonic_aurc_improvement": leave_out_gain,
                "relative_improvement": (
                    relative_aurc_improvement(
                        aurc(
                            y[keep], isotonic[keep], tie_breaker=targets[keep].tolist()
                        ),
                        aurc(
                            y[keep], candidate[keep], tie_breaker=targets[keep].tolist()
                        ),
                    )
                    if int(np.sum(keep))
                    else None
                ),
                "improvement_remains_non_negative": (
                    leave_out_gain is not None and leave_out_gain >= 0
                ),
                "aggregate_gain_change": leave_out_gain_change,
                "additive_aurc_gain_contribution": additive_contribution,
                "fraction_of_aggregate_gain": fraction,
            }
        )

    accepted_count = int(math.ceil(0.80 * len(y)))
    expected_deferred_failures, boundary_tie_split = (
        _expected_deferred_failure_contributions(
            y,
            candidate,
            accepted_count=accepted_count,
        )
    )
    total_expected_correctly_deferred = float(np.sum(expected_deferred_failures))

    def exact_or_expected_count(value: float) -> int | float:
        nearest_integer = round(value)
        if math.isclose(value, nearest_integer, rel_tol=0.0, abs_tol=1e-12):
            return int(nearest_integer)
        return float(value)

    deferred_contributions: list[dict[str, Any]] = []
    object_expected_counts: list[float] = []
    for object_key in sorted(set(objects.tolist())):
        expected_count = float(
            np.sum(expected_deferred_failures[objects == object_key])
        )
        object_expected_counts.append(expected_count)
        serialized_count = exact_or_expected_count(expected_count)
        fraction = (
            expected_count / total_expected_correctly_deferred
            if total_expected_correctly_deferred
            else None
        )
        deferred_contributions.append(
            {
                "object_id": object_display[object_key],
                "expected_correctly_deferred_failure_count": serialized_count,
                "fraction_of_expected_correctly_deferred_failures": fraction,
                # Compatibility aliases retain the historical report schema;
                # their expected-value semantics are declared in the metadata.
                "correctly_deferred_failure_count": serialized_count,
                "fraction_of_correctly_deferred_failures": fraction,
                "definition": (
                    "expected failure contribution outside the safest "
                    "ceil(80 percent * target count) candidate prefix"
                ),
            }
        )

    object_expected_count_sum = float(sum(object_expected_counts))
    deferred_reconstruction_matches = math.isclose(
        object_expected_count_sum,
        total_expected_correctly_deferred,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if not deferred_reconstruction_matches:
        raise AnalysisContractError(
            "Object expected deferred-failure counts do not reconstruct total"
        )
    object_expected_fraction_sum = (
        float(
            sum(
                row["fraction_of_expected_correctly_deferred_failures"]
                for row in deferred_contributions
            )
        )
        if total_expected_correctly_deferred
        else None
    )

    instance_contributions: list[dict[str, Any]] = []
    for instance_key in sorted(set(instances.tolist())):
        keep = instances != instance_key
        leave_out_gain = _gain_after_removal(
            y,
            candidate,
            isotonic,
            targets,
            keep,
        )
        leave_out_gain_change = (
            float(full_gain - leave_out_gain) if leave_out_gain is not None else None
        )
        selected = instances == instance_key
        additive_contribution = float(np.sum(target_gain_contributions[selected]))
        fraction = float(additive_contribution / full_gain) if full_gain > 0 else None
        instance_contributions.append(
            {
                "physical_instance_id": instance_display[instance_key],
                "target_count": int(np.sum(selected)),
                "leave_one_instance_out_gain": leave_out_gain,
                "leave_one_instance_out_gain_change": leave_out_gain_change,
                "aggregate_gain_contribution": additive_contribution,
                "fraction_of_aggregate_gain": fraction,
            }
        )

    removal_reverses = any(
        row["candidate_minus_isotonic_aurc_improvement"] is not None
        and row["candidate_minus_isotonic_aurc_improvement"] < 0
        for row in jackknife
    )
    object_over_40 = any(
        row["fraction_of_aggregate_gain"] is not None
        and row["fraction_of_aggregate_gain"] > 0.40
        for row in jackknife
    )
    instance_over_20 = any(
        row["fraction_of_aggregate_gain"] is not None
        and row["fraction_of_aggregate_gain"] > 0.20
        for row in instance_contributions
    )
    flags = {
        "removing_one_object_reverses_improvement": removal_reverses,
        "one_object_exceeds_40_percent_of_gain": object_over_40,
        "one_instance_exceeds_20_percent_of_gain": instance_over_20,
    }
    return _jsonable(
        {
            "candidate_method": "NESTED_MULTIFEATURE",
            "baseline_method": "SCORE_ISOTONIC",
            "aggregate_candidate_aurc": full_candidate_aurc,
            "aggregate_isotonic_aurc": full_isotonic_aurc,
            "aggregate_absolute_aurc_gain": full_gain,
            "per_object_metrics": per_object,
            "leave_one_object_out": jackknife,
            "object_correctly_deferred_failure_contributions": deferred_contributions,
            "object_correctly_deferred_failure_contributions_metadata": {
                "requested_accepted_coverage": 0.80,
                "accepted_count": accepted_count,
                "realized_accepted_coverage": accepted_count / len(y),
                "deferred_count": len(y) - accepted_count,
                "equal_risk_boundary_policy": (
                    "uniform_expected_allocation_within_boundary_block"
                ),
                "identifiers_used_as_secondary_ranking_inputs": False,
                "cutoff_splits_equal_risk_block": boundary_tie_split,
                "count_semantics": (
                    "expected count; exact integer whenever the cutoff does not "
                    "split an equal-risk block"
                ),
                "expected_count_field": ("expected_correctly_deferred_failure_count"),
                "expected_fraction_field": (
                    "fraction_of_expected_correctly_deferred_failures"
                ),
                "legacy_aliases_use_expected_value_semantics": True,
                "total_expected_correctly_deferred_failure_count": (
                    exact_or_expected_count(total_expected_correctly_deferred)
                ),
                "object_expected_count_sum": exact_or_expected_count(
                    object_expected_count_sum
                ),
                "object_expected_fraction_sum": object_expected_fraction_sum,
                "object_counts_reconstruct_total": deferred_reconstruction_matches,
            },
            "physical_instance_gain_contributions": instance_contributions,
            "additive_gain_reconstruction": {
                "per_target_contribution_sum": reconstructed_gain,
                "object_contribution_sum": float(
                    sum(row["additive_aurc_gain_contribution"] for row in jackknife)
                ),
                "physical_instance_contribution_sum": float(
                    sum(
                        row["aggregate_gain_contribution"]
                        for row in instance_contributions
                    )
                ),
                "matches_aggregate_gain": True,
                "share_denominator": "aggregate_isotonic_aurc_minus_candidate_aurc",
            },
            "object_driven_flags": flags,
            "object_driven": any(flags.values()),
        }
    )


def build_ablation_feature_families(
    feature_family_by_name: Mapping[str, str],
    *,
    raw_score_feature: str = "raw_score",
) -> dict[str, list[str]]:
    """Expand a feature-to-primary-family map into the four frozen ablations."""

    all_features = list(dict.fromkeys(str(name) for name in feature_family_by_name))
    score_features = [
        name
        for name in all_features
        if name == raw_score_feature
        or "score" in str(feature_family_by_name[name]).lower()
    ]
    disagreement_features = [
        name
        for name in all_features
        if any(
            token in str(feature_family_by_name[name]).lower()
            for token in ("disagreement", "agreement", "view")
        )
    ]
    if raw_score_feature not in score_features:
        score_features.insert(0, raw_score_feature)
    if not disagreement_features:
        raise AnalysisContractError(
            "feature-to-family map contains no disagreement/agreement family"
        )
    score_disagreement = list(dict.fromkeys([*score_features, *disagreement_features]))
    return {
        "score_only": score_features,
        "disagreement_only": disagreement_features,
        "score_disagreement": score_disagreement,
        "all": all_features,
    }


def _normalize_feature_families(
    feature_families: Mapping[str, Sequence[str] | str], raw_score_feature: str
) -> dict[str, list[str]]:
    string_values = [isinstance(value, str) for value in feature_families.values()]
    if feature_families and all(string_values):
        primary_map = {
            str(name): str(value) for name, value in feature_families.items()
        }
        feature_families = build_ablation_feature_families(
            primary_map,
            raw_score_feature=raw_score_feature,
        )
    elif any(string_values):
        raise AnalysisContractError(
            "feature_families cannot mix feature-to-family strings with ablation lists"
        )
    aliases = {
        "score-only": "score_only",
        "score": "score_only",
        "disagreement-only": "disagreement_only",
        "disagreement": "disagreement_only",
        "score+disagreement": "score_disagreement",
        "score_and_disagreement": "score_disagreement",
        "all_features": "all",
    }
    normalized: dict[str, list[str]] = {}
    for name, features in feature_families.items():
        key = aliases.get(str(name).lower(), str(name).lower())
        normalized[key] = list(dict.fromkeys(str(feature) for feature in features))
    missing = [name for name in ABLATION_NAMES if name not in normalized]
    if missing:
        raise AnalysisContractError(
            f"feature_families missing frozen ablations: {missing}"
        )
    for name, features in normalized.items():
        if not features:
            raise AnalysisContractError(f"feature family {name!r} must not be empty")
    if raw_score_feature not in normalized["score_only"]:
        raise AnalysisContractError(
            f"score_only must include raw score feature {raw_score_feature!r}"
        )
    expected_score_disagreement = set(normalized["score_only"]).union(
        normalized["disagreement_only"]
    )
    if not expected_score_disagreement.issubset(normalized["score_disagreement"]):
        raise AnalysisContractError(
            "score_disagreement must contain every score-only and disagreement-only feature"
        )
    if not set(normalized["score_disagreement"]).issubset(normalized["all"]):
        raise AnalysisContractError(
            "all feature family must contain the score+disagreement family"
        )
    return {name: normalized[name] for name in ABLATION_NAMES}


def _metric_table_for_predictions(
    y: np.ndarray,
    target_ids: Sequence[Any],
    outer_folds: np.ndarray,
    risk_scores: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray | None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate: dict[str, Any] = {}
    per_fold: list[dict[str, Any]] = []
    ids = np.asarray(target_ids, dtype=object)
    for method in FORMAL_METHODS:
        aggregate[method] = compute_prediction_metrics(
            y,
            risk_scores[method],
            failure_probability=probabilities[method],
            tie_breaker=target_ids,
        )
        for outer_fold in sorted(np.unique(outer_folds).tolist()):
            selected = outer_folds == outer_fold
            per_fold.append(
                {
                    "outer_fold": int(outer_fold),
                    "method": method,
                    "metrics": compute_prediction_metrics(
                        y[selected],
                        risk_scores[method][selected],
                        failure_probability=(
                            None
                            if probabilities[method] is None
                            else probabilities[method][selected]
                        ),
                        tie_breaker=ids[selected].tolist(),
                    ),
                }
            )
    raw_aurc = aggregate["RAW_SCORE_RANK"]["aurc"]
    isotonic_aurc = aggregate["SCORE_ISOTONIC"]["aurc"]
    for method in FORMAL_METHODS:
        aggregate[method]["relative_aurc_improvement_vs_raw_score_rank"] = (
            relative_aurc_improvement(raw_aurc, aggregate[method]["aurc"])
        )
        aggregate[method]["relative_aurc_improvement_vs_score_isotonic"] = (
            relative_aurc_improvement(isotonic_aurc, aggregate[method]["aurc"])
        )
    return _jsonable(aggregate), _jsonable(per_fold)


def _build_prediction_rows(
    evaluator_rows: Sequence[Mapping[str, Any]],
    outer_folds: np.ndarray,
    raw_score: np.ndarray,
    risk_scores: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, evaluator in enumerate(evaluator_rows):
        for method in FORMAL_METHODS:
            probability = probabilities[method]
            rows.append(
                {
                    "target_id": evaluator["target_id"],
                    "physical_instance_id": evaluator["physical_instance_id"],
                    "object_id": evaluator["object_id"],
                    "y_failure": int(evaluator["y_failure"]),
                    "outer_fold": int(outer_folds[index]),
                    "method": method,
                    "risk_score": float(risk_scores[method][index]),
                    "failure_probability": (
                        None if probability is None else float(probability[index])
                    ),
                    "raw_score": (
                        float(raw_score[index])
                        if math.isfinite(float(raw_score[index]))
                        else None
                    ),
                }
            )
    return _jsonable(rows)


def _ablation_metrics(
    feature_rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    raw_score: np.ndarray,
    target_ids: Sequence[Any],
    outer_folds: np.ndarray,
    inner_folds_by_outer: Mapping[int, Mapping[str, int]],
    feature_families: Mapping[str, Sequence[str]],
    *,
    raw_score_higher_is_confident: bool,
    all_model_result: Mapping[str, Any],
    all_aggregate: Mapping[str, Any],
    all_per_fold: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for ablation_name in ABLATION_NAMES:
        feature_names = list(feature_families[ablation_name])
        if ablation_name == "all":
            model_result = all_model_result
            aggregate = all_aggregate
            per_fold = all_per_fold
        else:
            x = _feature_matrix(feature_rows, feature_names)
            model_result = _cross_validated_models(
                x,
                y,
                raw_score,
                target_ids,
                outer_folds,
                inner_folds_by_outer,
                feature_names,
                raw_score_higher_is_confident=raw_score_higher_is_confident,
                collect_contributions=False,
            )
            aggregate, per_fold = _metric_table_for_predictions(
                y,
                target_ids,
                outer_folds,
                model_result["risk_scores"],
                model_result["probabilities"],
            )
        nested_risk = np.asarray(
            model_result["risk_scores"]["NESTED_MULTIFEATURE"], dtype=float
        )
        nested_probability_value = model_result["probabilities"]["NESTED_MULTIFEATURE"]
        if nested_probability_value is None:
            raise AnalysisContractError(
                f"Ablation {ablation_name!r} lacks nested failure probabilities"
            )
        nested_probability = np.asarray(nested_probability_value, dtype=float)
        if not (
            len(target_ids)
            == y.size
            == outer_folds.size
            == nested_risk.size
            == nested_probability.size
        ):
            raise AnalysisContractError(
                f"Ablation {ablation_name!r} OOF vectors are misaligned"
            )
        for index, target_id in enumerate(target_ids):
            prediction_rows.append(
                {
                    "ablation": ablation_name,
                    "target_id": target_id,
                    "y_failure": int(y[index]),
                    "outer_fold": int(outer_folds[index]),
                    "method": "NESTED_MULTIFEATURE",
                    "risk_score": float(nested_risk[index]),
                    "failure_probability": float(nested_probability[index]),
                }
            )
        nested_fold_metrics = [
            row for row in per_fold if row["method"] == "NESTED_MULTIFEATURE"
        ]
        result[ablation_name] = {
            "feature_names": feature_names,
            "aggregate_metrics": aggregate["NESTED_MULTIFEATURE"],
            "per_fold_metrics": nested_fold_metrics,
            "model_selections": model_result["model_selections"],
        }
    return _jsonable(result), _jsonable(prediction_rows)


def _stable_disagreement_benefit(feature_ablations: Mapping[str, Any]) -> bool:
    score_only = feature_ablations["score_only"]
    score_disagreement = feature_ablations["score_disagreement"]
    pooled_improvement = (
        score_only["aggregate_metrics"]["aurc"]
        - score_disagreement["aggregate_metrics"]["aurc"]
    )
    score_folds = {
        int(row["outer_fold"]): row["metrics"]["aurc"]
        for row in score_only["per_fold_metrics"]
    }
    disagreement_folds = {
        int(row["outer_fold"]): row["metrics"]["aurc"]
        for row in score_disagreement["per_fold_metrics"]
    }
    common = sorted(set(score_folds).intersection(disagreement_folds))
    positive_folds = sum(
        disagreement_folds[fold] < score_folds[fold] for fold in common
    )
    required_positive = int(math.ceil(0.60 * len(common))) if common else 1
    return bool(pooled_improvement > 0 and positive_folds >= required_positive)


def run_evaluation(
    feature_rows: Any,
    evaluator_rows: Any,
    fold_manifest: Mapping[str, Any],
    feature_families: Mapping[str, Sequence[str] | str],
    *,
    raw_score_feature: str = "raw_score",
    raw_score_higher_is_confident: bool = True,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Run the complete frozen M6-G0 grouped nested-CV analysis.

    The caller must write the output of :func:`make_fold_manifest` before
    invoking this function.  This API performs no file or repository access.
    """

    features = _project_feature_records(feature_rows)
    evaluators = _project_evaluator_records(evaluator_rows)
    _require_columns(
        evaluators,
        ("target_id", "y_failure", "physical_instance_id", "object_id"),
        name="evaluator_rows",
    )
    normalized_families = _normalize_feature_families(
        feature_families, raw_score_feature
    )
    all_features = normalized_families["all"]
    feature_schema = validate_feature_schema(features, all_features)
    if not feature_schema["passed"]:
        raise AnalysisContractError(
            f"feature schema validation failed: {feature_schema['errors']}"
        )
    manifest_validation = validate_fold_manifest(fold_manifest, evaluators)
    if not manifest_validation["passed"]:
        raise AnalysisContractError(
            f"fold manifest validation failed: {manifest_validation['errors']}"
        )

    feature_by_target = {_id_key(row["target_id"]): row for row in features}
    evaluator_by_target = {_id_key(row["target_id"]): row for row in evaluators}
    if set(feature_by_target) != set(evaluator_by_target):
        missing_features = sorted(set(evaluator_by_target) - set(feature_by_target))
        extra_features = sorted(set(feature_by_target) - set(evaluator_by_target))
        raise AnalysisContractError(
            "feature/evaluator target mismatch: "
            f"missing={missing_features}, extra={extra_features}"
        )
    outer_assignment_by_target = {
        _id_key(row["target_id"]): int(row["outer_fold"])
        for row in fold_manifest["outer_assignments"]
    }
    target_keys = sorted(evaluator_by_target)
    aligned_evaluators = [evaluator_by_target[key] for key in target_keys]
    aligned_features = [feature_by_target[key] for key in target_keys]
    target_ids = [row["target_id"] for row in aligned_evaluators]
    y = _binary_vector(
        [row["y_failure"] for row in aligned_evaluators], name="y_failure"
    )
    instance_ids = [row["physical_instance_id"] for row in aligned_evaluators]
    object_ids = [row["object_id"] for row in aligned_evaluators]
    outer_folds = np.asarray(
        [outer_assignment_by_target[key] for key in target_keys], dtype=int
    )
    inner_folds_by_outer: dict[int, dict[str, int]] = {
        fold: {} for fold in range(int(fold_manifest["outer_n_splits"]))
    }
    for row in fold_manifest["inner_assignments"]:
        inner_folds_by_outer[int(row["outer_fold"])][_id_key(row["target_id"])] = int(
            row["inner_fold"]
        )
    raw_score = np.asarray(
        [
            _coerce_feature_value(row[raw_score_feature], raw_score_feature)
            for row in aligned_features
        ],
        dtype=float,
    )
    x_all = _feature_matrix(aligned_features, all_features)
    main_models = _cross_validated_models(
        x_all,
        y,
        raw_score,
        target_ids,
        outer_folds,
        inner_folds_by_outer,
        all_features,
        raw_score_higher_is_confident=raw_score_higher_is_confident,
        collect_contributions=True,
    )
    aggregate_metrics, per_fold_metrics = _metric_table_for_predictions(
        y,
        target_ids,
        outer_folds,
        main_models["risk_scores"],
        main_models["probabilities"],
    )
    prediction_rows = _build_prediction_rows(
        aligned_evaluators,
        outer_folds,
        raw_score,
        main_models["risk_scores"],
        main_models["probabilities"],
    )
    prediction_validation = validate_oof_predictions(prediction_rows, target_ids)

    ablations, ablation_prediction_rows = _ablation_metrics(
        aligned_features,
        y,
        raw_score,
        target_ids,
        outer_folds,
        inner_folds_by_outer,
        normalized_families,
        raw_score_higher_is_confident=raw_score_higher_is_confident,
        all_model_result=main_models,
        all_aggregate=aggregate_metrics,
        all_per_fold=per_fold_metrics,
    )
    stable_disagreement = _stable_disagreement_benefit(ablations)
    bootstrap = grouped_bootstrap(
        y,
        main_models["risk_scores"],
        main_models["probabilities"],
        instance_ids,
        target_ids,
        n_resamples=bootstrap_resamples,
    )
    robustness = object_robustness_analysis(
        y,
        main_models["risk_scores"]["NESTED_MULTIFEATURE"],
        main_models["risk_scores"]["SCORE_ISOTONIC"],
        object_ids,
        instance_ids,
        target_ids,
    )
    label_support = validate_label_support(aligned_evaluators)
    leakage_checks_passed = bool(
        feature_schema["passed"] and manifest_validation["passed"]
    )
    decision_inputs = {
        "stable_disagreement_benefit": stable_disagreement,
        "leakage_checks_passed": leakage_checks_passed,
        "label_support_passed": bool(label_support["passed"]),
        "oof_complete": bool(prediction_validation["passed"]),
    }
    return _jsonable(
        {
            "schema_version": "m6_g0_analysis_result_v1",
            "fold_manifest_hash": fold_manifest.get("manifest_hash"),
            "hyperparameter_grid": {
                "logistic_C": list(LOGISTIC_C_GRID),
                "random_forest": list(TREE_PARAM_GRID),
                "family_tie_tolerance_aurc": 0.005,
                "family_tie_preference": "logistic",
            },
            "feature_families": normalized_families,
            "oof_predictions": prediction_rows,
            "per_fold_metrics": per_fold_metrics,
            "aggregate_metrics": aggregate_metrics,
            "grouped_bootstrap": bootstrap,
            "object_jackknife": robustness,
            "feature_ablations": ablations,
            "ablation_oof_predictions": ablation_prediction_rows,
            "feature_contributions": {
                "logistic_outer_fold_coefficients": main_models["coefficient_rows"],
                "logistic_coefficient_stability": main_models["coefficient_stability"],
                "tree_outer_test_permutation_importance": main_models[
                    "tree_permutation_importance"
                ],
            },
            "model_selections": main_models["model_selections"],
            "prediction_validation": prediction_validation,
            "fold_manifest_validation": manifest_validation,
            "feature_schema_validation": feature_schema,
            "label_support": label_support,
            "decision_inputs": decision_inputs,
        }
    )


def _decision_classification(
    go_conditions: Mapping[str, bool],
    weak_conditions: Mapping[str, bool],
    hard_validity_conditions: Mapping[str, bool],
) -> str:
    if not all(hard_validity_conditions.values()):
        return "NO-GO"
    if all(go_conditions.values()):
        return "SIGNAL GO"
    if all(weak_conditions.values()):
        return "WEAK SIGNAL / HOLDOUT NOT AUTHORIZED"
    return "NO-GO"


def frozen_signal_decision(
    aggregate_metrics: Mapping[str, Any],
    grouped_bootstrap: Mapping[str, Any],
    object_robustness: Mapping[str, Any],
    *,
    feature_ablations: Mapping[str, Any],
    stable_disagreement_benefit: bool,
    leakage_checks_passed: bool,
    label_support_passed: bool,
    oof_complete: bool,
    m3_access_boundary_passed: bool,
) -> dict[str, Any]:
    """Apply the immutable SIGNAL GO / WEAK / NO-GO rules mechanically.

    ``m3_access_boundary_passed`` is intentionally explicit.  If it is false,
    SIGNAL GO is forbidden and the final classification is conservatively
    forced to ``NO-GO``.  The numerical counterfactual is reported separately
    and never changes that final classification.
    """

    del feature_ablations  # The stable ablation conclusion is passed explicitly.
    candidate = aggregate_metrics["NESTED_MULTIFEATURE"]
    raw = aggregate_metrics["RAW_SCORE_RANK"]
    isotonic = aggregate_metrics["SCORE_ISOTONIC"]
    candidate_aurc = float(candidate["aurc"])
    raw_aurc = float(raw["aurc"])
    isotonic_aurc = float(isotonic["aurc"])
    improvement_raw = relative_aurc_improvement(raw_aurc, candidate_aurc)
    improvement_isotonic = relative_aurc_improvement(isotonic_aurc, candidate_aurc)
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
    elif float(isotonic_brier) > 0:
        brier_relative_change = (
            float(candidate_brier) - float(isotonic_brier)
        ) / float(isotonic_brier)
        brier_condition = brier_relative_change <= 0.02
    else:
        brier_relative_change = 0.0 if float(candidate_brier) == 0 else math.inf
        brier_condition = float(candidate_brier) == 0
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
    classification = _decision_classification(
        go_conditions, weak_conditions, hard_validity
    )
    counterfactual_go = dict(go_conditions)
    counterfactual_go["m3_access_boundary_passed"] = True
    counterfactual_hard = dict(hard_validity)
    counterfactual_hard["m3_access_boundary_passed"] = True
    counterfactual = _decision_classification(
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
    return _jsonable(
        {
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
                "candidate_brier_relative_change_vs_isotonic": (brier_relative_change),
                "candidate_ece": candidate.get("ece"),
                "object_driven": not not_object_driven,
            },
            "rules_version": "m6_g0_frozen_decision_v1",
        }
    )


def select_qualitative_audit_rows(
    oof_prediction_rows: Any,
    *,
    candidate_method: str = "NESTED_MULTIFEATURE",
    baseline_method: str = "SCORE_ISOTONIC",
    count: int = 5,
) -> dict[str, Any]:
    """Select representative errors using frozen, non-favorable ordering rules."""

    if count < 1:
        raise AnalysisContractError("count must be positive")
    rows = _as_records(oof_prediction_rows, name="oof_prediction_rows")
    candidate = {
        _id_key(row["target_id"]): row
        for row in rows
        if row.get("method") == candidate_method
    }
    baseline = {
        _id_key(row["target_id"]): row
        for row in rows
        if row.get("method") == baseline_method
    }
    if not candidate or set(candidate) != set(baseline):
        raise AnalysisContractError(
            "candidate and baseline qualitative rows need identical target coverage"
        )

    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        key = _id_key(row["target_id"])
        return {
            "target_id": row["target_id"],
            "object_id": row.get("object_id"),
            "physical_instance_id": row.get("physical_instance_id"),
            "y_failure": int(row["y_failure"]),
            "candidate_risk": float(row["risk_score"]),
            "isotonic_risk": float(baseline[key]["risk_score"]),
            "candidate_minus_isotonic_risk": float(
                row["risk_score"] - baseline[key]["risk_score"]
            ),
        }

    candidate_rows = list(candidate.values())
    low_risk_failures = sorted(
        (row for row in candidate_rows if int(row["y_failure"]) == 1),
        key=lambda row: (float(row["risk_score"]), _id_key(row["target_id"])),
    )[:count]
    high_risk_successes = sorted(
        (row for row in candidate_rows if int(row["y_failure"]) == 0),
        key=lambda row: (-float(row["risk_score"]), _id_key(row["target_id"])),
    )[:count]
    largest_disagreements = sorted(
        candidate_rows,
        key=lambda row: (
            -abs(
                float(row["risk_score"])
                - float(baseline[_id_key(row["target_id"])]["risk_score"])
            ),
            _id_key(row["target_id"]),
        ),
    )[:count]
    return _jsonable(
        {
            "lowest_predicted_risk_failures": [
                compact(row) for row in low_risk_failures
            ],
            "highest_predicted_risk_successes": [
                compact(row) for row in high_risk_successes
            ],
            "largest_candidate_vs_isotonic_disagreements": [
                compact(row) for row in largest_disagreements
            ],
            "selection_rule": "deterministic_extremes_no_manual_selection",
        }
    )


def summarize_m6_g0_analysis(
    evaluation: Mapping[str, Any],
    *,
    m3_access_boundary_passed: bool,
    leakage_checks_passed: bool | None = None,
) -> dict[str, Any]:
    """Return a compact machine-readable final summary and frozen decision."""

    decision_inputs = dict(evaluation["decision_inputs"])
    if leakage_checks_passed is not None:
        decision_inputs["leakage_checks_passed"] = bool(leakage_checks_passed)
    decision = frozen_signal_decision(
        evaluation["aggregate_metrics"],
        evaluation["grouped_bootstrap"],
        evaluation["object_jackknife"],
        feature_ablations=evaluation["feature_ablations"],
        stable_disagreement_benefit=bool(
            decision_inputs["stable_disagreement_benefit"]
        ),
        leakage_checks_passed=bool(decision_inputs["leakage_checks_passed"]),
        label_support_passed=bool(decision_inputs["label_support_passed"]),
        oof_complete=bool(decision_inputs["oof_complete"]),
        m3_access_boundary_passed=m3_access_boundary_passed,
    )
    nested = evaluation["aggregate_metrics"]["NESTED_MULTIFEATURE"]
    return _jsonable(
        {
            "classification": decision["classification"],
            "decision": decision,
            "formal_candidate": "NESTED_MULTIFEATURE",
            "formal_candidate_metrics": nested,
            "formal_method_metrics": evaluation["aggregate_metrics"],
            "label_support": evaluation["label_support"],
            "fold_manifest_hash": evaluation["fold_manifest_hash"],
            "model_selection_audit": evaluation["model_selections"],
            "qualitative_audit": select_qualitative_audit_rows(
                evaluation["oof_predictions"]
            ),
        }
    )


__all__ = [
    "ABLATION_NAMES",
    "AnalysisContractError",
    "BOOTSTRAP_SEED",
    "FORMAL_METHODS",
    "FoldSupportError",
    "INNER_SEED_BASE",
    "LOGISTIC_C_GRID",
    "OUTER_SEED",
    "TREE_PARAM_GRID",
    "aurc",
    "build_ablation_feature_families",
    "compute_prediction_metrics",
    "equal_frequency_calibration_error",
    "fold_manifest_hash",
    "frozen_signal_decision",
    "grouped_bootstrap",
    "grouped_bootstrap_indices",
    "make_fold_manifest",
    "object_robustness_analysis",
    "relative_aurc_improvement",
    "risk_coverage_curve",
    "run_evaluation",
    "select_qualitative_audit_rows",
    "summarize_m6_g0_analysis",
    "validate_feature_schema",
    "validate_fold_manifest",
    "validate_label_support",
    "validate_oof_predictions",
]
