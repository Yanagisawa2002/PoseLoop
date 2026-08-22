#!/usr/bin/env python3
"""Run the PoseLoop M3-R1 marginal-value policy development gate.

The script reads only M1/M2 development evidence.  It uses nested,
physical-instance-grouped cross-validation to predict the incremental value of
k=3 over k=1 and k=5 over k=3.  The resulting sequential allocation is compared
with an exact-budget-matched random allocation before any legacy M3 holdout is
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import fit_m3_policy as legacy_m3
from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
CV_SEED = 20260815
MODEL_SEED = 20260816
RANDOM_BASELINE_SEED = 20260817
BOOTSTRAP_SEED = 20260818
OUTER_FOLDS = 5
INNER_FOLDS = 4
THRESHOLD_QUANTILES = tuple(np.linspace(0.0, 1.0, 17))
SUPPORTED_VISIBILITY_MIN_ROWS = 20

RAW_K1_FEATURE_NAMES = (
    "raw_score",
    "raw_score_margin",
    "valid_depth_ratio",
)
RAW_K3_FEATURE_NAMES = tuple(
    f"{base}_{stat}"
    for base in ("raw_score", "raw_score_margin", "valid_depth_ratio")
    for stat in ("min", "median", "max", "std")
) + (
    "selected_raw_score",
    "selected_raw_score_margin",
    "selected_valid_depth_ratio",
)
K1_FEATURE_NAMES = tuple(legacy_m3.K1_FEATURE_NAMES) + RAW_K1_FEATURE_NAMES
K3_FEATURE_NAMES = tuple(legacy_m3.K3_FEATURE_NAMES) + RAW_K3_FEATURE_NAMES

FORBIDDEN_FEATURE_TOKENS = (
    "gt_",
    "ground_truth",
    "visible_fraction",
    "visibility_bin",
    "object_id",
    "sample_id",
    "scene_id",
    "group_id",
    "track_id",
    "success",
    "label",
    "error",
    "future",
    "holdout",
)

MODEL_SPECS = (
    {
        "name": "ridge_alpha_1",
        "family": "ridge",
        "alpha": 1.0,
        "difficulty_weighted": False,
    },
    {
        "name": "ridge_alpha_10_balanced",
        "family": "ridge",
        "alpha": 10.0,
        "difficulty_weighted": True,
    },
    {
        "name": "rf_depth_2_leaf_12_balanced",
        "family": "random_forest",
        "max_depth": 2,
        "min_samples_leaf": 12,
        "max_features": 0.75,
        "n_estimators": 200,
        "difficulty_weighted": True,
    },
    {
        "name": "rf_depth_3_leaf_8_balanced",
        "family": "random_forest",
        "max_depth": 3,
        "min_samples_leaf": 8,
        "max_features": 0.75,
        "n_estimators": 200,
        "difficulty_weighted": True,
    },
)


@dataclass(frozen=True)
class FittedRegressor:
    model: Any
    scaler: Any | None

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(matrix) if self.scaler is not None else matrix
        prediction = np.asarray(self.model.predict(transformed), dtype=np.float64)
        if prediction.shape != (len(matrix),) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("Marginal-value model produced invalid predictions")
        return prediction


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "artifacts" / "r1" / "m3_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", type=Path, default=repo_root / "artifacts" / "m2" / "groups.jsonl"
    )
    parser.add_argument(
        "--metrics", type=Path, default=repo_root / "artifacts" / "m2" / "metrics.jsonl"
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
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_r1_protocol.json",
    )
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "m3_r1_development.md",
    )
    return parser.parse_args()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _load_prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    metadata_count = 0
    for row_number, row in enumerate(load_jsonl(path), start=1):
        record_type = row.get("record_type")
        if record_type == "metadata":
            metadata_count += 1
            continue
        if record_type != "prediction":
            raise ValueError(f"Unexpected prediction record at {path}:{row_number}")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in index:
            raise ValueError(f"Missing or duplicate prediction sample ID: {sample_id!r}")
        index[sample_id] = row
    if metadata_count != 1 or not index:
        raise ValueError(f"Prediction stream needs one metadata row and data: {path}")
    return index


def _prediction_features(prediction: Mapping[str, Any]) -> tuple[float, float, float]:
    if prediction.get("status") != "success":
        return 0.0, 0.0, 0.0
    return (
        _finite(prediction.get("foundationpose_top_score", 0.0), "raw score"),
        _finite(prediction.get("foundationpose_top_score_margin", 0.0), "raw margin"),
        _finite(prediction.get("valid_depth_ratio_inside_mask", 0.0), "depth ratio"),
    )


def _summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Cannot summarize empty/non-finite {prefix}")
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_std": float(np.std(array)),
    }


def extract_r1_prefix_features(
    group: Mapping[str, Any],
    method_result: Mapping[str, Any],
    budget: int,
    prediction_sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    """Extract only observations available after acquiring the declared prefix."""
    if budget not in (1, 3):
        raise ValueError("M3-R1 features are defined only for k=1 and k=3")
    legacy = legacy_m3.extract_prefix_features(group, method_result, budget)
    views = sorted(group["views"], key=lambda row: int(row["acquisition_rank"]))
    raw = []
    for view in views[:budget]:
        source = str(view.get("prediction_source", ""))
        sample_id = str(view.get("sample_id", ""))
        if source not in prediction_sources or sample_id not in prediction_sources[source]:
            raise ValueError(f"Missing {source} prefix prediction for {sample_id}")
        raw.append(_prediction_features(prediction_sources[source][sample_id]))

    if budget == 1:
        score, margin, depth_ratio = raw[0]
        features = {
            **legacy,
            "raw_score": score,
            "raw_score_margin": margin,
            "valid_depth_ratio": depth_ratio,
        }
        names = K1_FEATURE_NAMES
    else:
        scores, margins, depth_ratios = zip(*raw, strict=True)
        selected_sample_id = str(method_result.get("selected_sample_id", ""))
        selected_prediction = None
        for view in views[:budget]:
            if str(view.get("sample_id")) == selected_sample_id:
                selected_prediction = prediction_sources[str(view["prediction_source"])][
                    selected_sample_id
                ]
                break
        if selected_prediction is None:
            raise ValueError(f"Selected k3 sample is outside acquired prefix: {selected_sample_id}")
        selected_score, selected_margin, selected_depth = _prediction_features(
            selected_prediction
        )
        features = {
            **legacy,
            **_summary("raw_score", scores),
            **_summary("raw_score_margin", margins),
            **_summary("valid_depth_ratio", depth_ratios),
            "selected_raw_score": selected_score,
            "selected_raw_score_margin": selected_margin,
            "selected_valid_depth_ratio": selected_depth,
        }
        names = K3_FEATURE_NAMES
    if set(features) != set(names):
        raise AssertionError(f"Feature contract mismatch for k={budget}")
    return {name: _finite(features[name], name) for name in names}


def combined_score(result: Mapping[str, Any]) -> float:
    return (
        _finite(result["sample_ar_mssd"], "sample_ar_mssd")
        + _finite(result["sample_ar_mspd"], "sample_ar_mspd")
    ) / 2.0


def build_development_rows(
    groups: list[dict[str, Any]],
    metrics: Mapping[tuple[str, str, int], dict[str, Any]],
    prediction_sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    legacy_rows = legacy_m3._build_development_rows(groups, metrics)
    group_index = {str(group["group_id"]): group for group in groups}
    rows = []
    for legacy_row in legacy_rows:
        group_id = str(legacy_row["group_id"])
        group = group_index[group_id]
        outcomes = {
            budget: combined_score(legacy_row["_policy_results"][budget])
            for budget in (1, 3, 5)
        }
        rows.append(
            {
                "group_id": group_id,
                "object_id": int(legacy_row["object_id"]),
                "physical_instance_track_id": str(
                    legacy_row["physical_instance_track_id"]
                ),
                "target_visibility_bin": str(group["target_visibility_bin"]),
                "k1": {
                    "features": extract_r1_prefix_features(
                        group,
                        legacy_row["_policy_results"][1],
                        1,
                        prediction_sources,
                    ),
                    "marginal_value": outcomes[3] - outcomes[1],
                },
                "k3": {
                    "features": extract_r1_prefix_features(
                        group,
                        legacy_row["_policy_results"][3],
                        3,
                        prediction_sources,
                    ),
                    "marginal_value": outcomes[5] - outcomes[3],
                },
                "outcomes": outcomes,
            }
        )
    rows.sort(key=lambda row: row["group_id"])
    return rows


def _matrix(rows: Sequence[dict[str, Any]], stage: str) -> np.ndarray:
    names = K1_FEATURE_NAMES if stage == "k1" else K3_FEATURE_NAMES
    matrix = np.asarray(
        [[row[stage]["features"][name] for name in names] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(names)) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid {stage} feature matrix")
    return matrix


def _targets(rows: Sequence[dict[str, Any]], stage: str) -> np.ndarray:
    values = np.asarray([row[stage]["marginal_value"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {stage} marginal-value targets")
    return values


def _stratification(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    visibility_code = {"high": 0, "mid": 1, "low": 2}
    values = []
    for row in rows:
        visibility = str(row["target_visibility_bin"])
        if visibility not in visibility_code:
            raise ValueError(f"Unknown visibility bin: {visibility}")
        opportunity = int(
            row["k1"]["marginal_value"] > 1e-12
            or row["k3"]["marginal_value"] > 1e-12
        )
        values.append(2 * visibility_code[visibility] + opportunity)
    return np.asarray(values, dtype=np.int64)


def make_grouped_folds(
    rows: Sequence[dict[str, Any]], n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(
        [row["physical_instance_track_id"] for row in rows], dtype=object
    )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(
        splitter.split(
            np.zeros((len(rows), 1), dtype=np.float64),
            _stratification(rows),
            groups,
        )
    )
    tested: list[int] = []
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Physical-instance leakage in grouped CV")
        tested.extend(int(index) for index in test)
    if sorted(tested) != list(range(len(rows))):
        raise RuntimeError("Grouped CV must test every row exactly once")
    return folds


def _difficulty_weights(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["target_visibility_bin"]) for row in rows)
    weights = np.asarray(
        [len(rows) / (len(counts) * counts[str(row["target_visibility_bin"])]) for row in rows],
        dtype=np.float64,
    )
    return weights / float(np.mean(weights))


def fit_regressor(
    matrix: np.ndarray,
    targets: np.ndarray,
    rows: Sequence[dict[str, Any]],
    spec: Mapping[str, Any],
    seed: int,
) -> FittedRegressor:
    weights = _difficulty_weights(rows) if spec["difficulty_weighted"] else None
    if spec["family"] == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        transformed = scaler.fit_transform(matrix)
        model = Ridge(alpha=float(spec["alpha"]), fit_intercept=True)
        model.fit(transformed, targets, sample_weight=weights)
        return FittedRegressor(model=model, scaler=scaler)
    if spec["family"] == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            criterion="squared_error",
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(matrix, targets, sample_weight=weights)
        return FittedRegressor(model=model, scaler=None)
    raise ValueError(f"Unknown model family: {spec['family']}")


def _subset(rows: Sequence[dict[str, Any]], indices: np.ndarray) -> list[dict[str, Any]]:
    return [rows[int(index)] for index in indices]


def crossfit_spec(
    rows: Sequence[dict[str, Any]],
    spec: Mapping[str, Any],
    n_splits: int,
    seed: int,
) -> dict[str, np.ndarray]:
    folds = make_grouped_folds(rows, n_splits, seed)
    matrices = {stage: _matrix(rows, stage) for stage in ("k1", "k3")}
    targets = {stage: _targets(rows, stage) for stage in ("k1", "k3")}
    output = {
        stage: np.full(len(rows), np.nan, dtype=np.float64) for stage in ("k1", "k3")
    }
    for fold_index, (train, test) in enumerate(folds):
        train_rows = _subset(rows, train)
        for stage in ("k1", "k3"):
            fitted = fit_regressor(
                matrices[stage][train],
                targets[stage][train],
                train_rows,
                spec,
                seed + 100 * fold_index + (1 if stage == "k1" else 3),
            )
            output[stage][test] = fitted.predict(matrices[stage][test])
    if any(not np.all(np.isfinite(values)) for values in output.values()):
        raise RuntimeError("Cross-fitted marginal predictions are incomplete")
    return output


def macro_object_mean(rows: Sequence[dict[str, Any]], values: Sequence[float]) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[int(row["object_id"])].append(float(value))
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def outcomes_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row["outcomes"][budget] for budget in (1, 3, 5)] for row in rows],
        dtype=np.float64,
    )


def budget_indices(budgets: np.ndarray) -> np.ndarray:
    mapping = {1: 0, 3: 1, 5: 2}
    try:
        return np.asarray([mapping[int(value)] for value in budgets], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown view budget: {exc.args[0]}") from exc


def policy_score(rows: Sequence[dict[str, Any]], budgets: np.ndarray) -> float:
    matrix = outcomes_matrix(rows)
    values = matrix[np.arange(len(rows)), budget_indices(budgets)]
    return macro_object_mean(rows, values)


def expected_matched_random_score(
    rows: Sequence[dict[str, Any]], budgets: np.ndarray
) -> float:
    counts = Counter(int(value) for value in budgets)
    probabilities = np.asarray([counts[budget] / len(rows) for budget in (1, 3, 5)])
    expected = outcomes_matrix(rows) @ probabilities
    return macro_object_mean(rows, expected)


def allocation_gain(rows: Sequence[dict[str, Any]], budgets: np.ndarray) -> float:
    return policy_score(rows, budgets) - expected_matched_random_score(rows, budgets)


def threshold_candidates(predictions: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(predictions, THRESHOLD_QUANTILES)
    # Finite sentinels preserve the all-continue/all-stop candidates while
    # keeping every audit artifact valid under strict JSON (allow_nan=False).
    below_minimum = np.nextafter(float(np.min(predictions)), -np.inf)
    above_maximum = np.nextafter(float(np.max(predictions)), np.inf)
    values = np.unique(
        np.concatenate(([below_minimum], quantiles, [above_maximum]))
    )
    return values.astype(np.float64)


def apply_thresholds(
    p1: np.ndarray, p3: np.ndarray, threshold_k1: float, threshold_k3: float
) -> np.ndarray:
    if p1.shape != p3.shape:
        raise ValueError("k1 and k3 predictions are mis-shaped")
    return np.where(
        p1 > threshold_k1,
        np.where(p3 > threshold_k3, 5, 3),
        1,
    ).astype(np.int64)


def select_operating_point(
    rows: Sequence[dict[str, Any]],
    predictions: Mapping[str, np.ndarray],
    mean_view_cap: float,
) -> dict[str, Any]:
    candidates = []
    for threshold_k1 in threshold_candidates(predictions["k1"]):
        for threshold_k3 in threshold_candidates(predictions["k3"]):
            budgets = apply_thresholds(
                predictions["k1"], predictions["k3"], threshold_k1, threshold_k3
            )
            mean_views = float(np.mean(budgets))
            if mean_views > mean_view_cap + 1e-12:
                continue
            active = policy_score(rows, budgets)
            random_score = expected_matched_random_score(rows, budgets)
            candidates.append(
                {
                    "threshold_k1": float(threshold_k1),
                    "threshold_k3": float(threshold_k3),
                    "mean_views": mean_views,
                    "active_score": active,
                    "matched_random_expected_score": random_score,
                    "allocation_gain": active - random_score,
                    "budget_counts": {
                        str(budget): int(np.sum(budgets == budget))
                        for budget in (1, 3, 5)
                    },
                }
            )
    if not candidates:
        raise RuntimeError(f"No operating point satisfies mean views <= {mean_view_cap}")
    return min(
        candidates,
        key=lambda row: (
            -row["allocation_gain"],
            -row["active_score"],
            row["mean_views"],
            -row["threshold_k1"],
            -row["threshold_k3"],
        ),
    )


def nested_oof(
    rows: Sequence[dict[str, Any]], mean_view_caps: Sequence[float]
) -> tuple[dict[float, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    outer = make_grouped_folds(rows, OUTER_FOLDS, CV_SEED)
    budgets_by_cap = {
        float(cap): np.full(len(rows), -1, dtype=np.int64) for cap in mean_view_caps
    }
    predictions = {
        stage: np.full(len(rows), np.nan, dtype=np.float64) for stage in ("k1", "k3")
    }
    audit = []
    matrices = {stage: _matrix(rows, stage) for stage in ("k1", "k3")}
    targets = {stage: _targets(rows, stage) for stage in ("k1", "k3")}

    for fold_index, (train, test) in enumerate(outer):
        train_rows = _subset(rows, train)
        test_rows = _subset(rows, test)
        per_spec = []
        for spec_index, spec in enumerate(MODEL_SPECS):
            inner_predictions = crossfit_spec(
                train_rows,
                spec,
                INNER_FOLDS,
                CV_SEED + 1000 + 100 * fold_index + spec_index,
            )
            operations = {
                float(cap): select_operating_point(train_rows, inner_predictions, cap)
                for cap in mean_view_caps
            }
            per_spec.append(
                {
                    "spec": spec,
                    "inner_predictions": inner_predictions,
                    "operations": operations,
                }
            )
        chosen = min(
            per_spec,
            key=lambda item: (
                -item["operations"][3.0]["allocation_gain"],
                -item["operations"][3.0]["active_score"],
                item["spec"]["name"],
            ),
        )

        test_predictions: dict[str, np.ndarray] = {}
        for stage in ("k1", "k3"):
            fitted = fit_regressor(
                matrices[stage][train],
                targets[stage][train],
                train_rows,
                chosen["spec"],
                MODEL_SEED + 100 * fold_index + (1 if stage == "k1" else 3),
            )
            test_predictions[stage] = fitted.predict(matrices[stage][test])
            predictions[stage][test] = test_predictions[stage]

        applied = {}
        for cap in mean_view_caps:
            operation = chosen["operations"][float(cap)]
            budgets = apply_thresholds(
                test_predictions["k1"],
                test_predictions["k3"],
                operation["threshold_k1"],
                operation["threshold_k3"],
            )
            budgets_by_cap[float(cap)][test] = budgets
            applied[str(cap)] = {
                "threshold_k1": operation["threshold_k1"],
                "threshold_k3": operation["threshold_k3"],
                "test_mean_views": float(np.mean(budgets)),
                "test_allocation_gain": allocation_gain(test_rows, budgets),
                "test_budget_counts": {
                    str(budget): int(np.sum(budgets == budget)) for budget in (1, 3, 5)
                },
            }
        train_tracks = {
            rows[int(index)]["physical_instance_track_id"] for index in train
        }
        test_tracks = {rows[int(index)]["physical_instance_track_id"] for index in test}
        audit.append(
            {
                "fold": fold_index,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_tracks": len(train_tracks),
                "test_tracks": len(test_tracks),
                "track_intersection_count": len(train_tracks & test_tracks),
                "chosen_model": chosen["spec"],
                "inner_primary_candidates": [
                    {
                        "model": item["spec"]["name"],
                        **item["operations"][3.0],
                    }
                    for item in per_spec
                ],
                "outer_application": applied,
            }
        )
    if any(np.any(values < 0) for values in budgets_by_cap.values()):
        raise RuntimeError("Nested OOF budgets are incomplete")
    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("Nested OOF predictions are incomplete")
    return budgets_by_cap, predictions, audit


def seeded_random_baseline(
    rows: Sequence[dict[str, Any]], budgets: np.ndarray, assignments: int
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    scores = np.asarray(
        [policy_score(rows, rng.permutation(budgets)) for _ in range(assignments)],
        dtype=np.float64,
    )
    return {
        "assignments": assignments,
        "seed": RANDOM_BASELINE_SEED,
        "exact_budget_counts_each_assignment": {
            str(budget): int(np.sum(budgets == budget)) for budget in (1, 3, 5)
        },
        "mean_score": float(np.mean(scores)),
        "score_95pct_interval": [
            float(np.quantile(scores, 0.025)),
            float(np.quantile(scores, 0.975)),
        ],
        "analytic_expected_score": expected_matched_random_score(rows, budgets),
    }


def grouped_bootstrap_gain(
    rows: Sequence[dict[str, Any]], budgets: np.ndarray, resamples: int
) -> dict[str, Any]:
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[str(row["physical_instance_track_id"])].append(index)
    track_ids = sorted(by_track)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        selected_tracks = rng.choice(track_ids, size=len(track_ids), replace=True)
        indices = np.asarray(
            [index for track_id in selected_tracks for index in by_track[str(track_id)]],
            dtype=np.int64,
        )
        sampled_rows = [rows[int(index)] for index in indices]
        sampled_budgets = budgets[indices]
        gains[sample_index] = allocation_gain(sampled_rows, sampled_budgets)
    return {
        "unit": "physical_instance_track_id",
        "track_count": len(track_ids),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "gain_pp_one_sided_90pct_lower": 100.0 * float(np.quantile(gains, 0.10)),
        "gain_pp_95pct_interval": [
            100.0 * float(np.quantile(gains, 0.025)),
            100.0 * float(np.quantile(gains, 0.975)),
        ],
        "gain_pp_median": 100.0 * float(np.median(gains)),
    }


def difficulty_metrics(
    rows: Sequence[dict[str, Any]], budgets: np.ndarray
) -> dict[str, dict[str, Any]]:
    output = {}
    for visibility in ("low", "mid", "high"):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["target_visibility_bin"] == visibility
            ],
            dtype=np.int64,
        )
        stratum_rows = [rows[int(index)] for index in indices]
        stratum_budgets = budgets[indices]
        output[visibility] = {
            "row_count": len(indices),
            "supported_for_gate": len(indices) >= SUPPORTED_VISIBILITY_MIN_ROWS,
            "mean_views": float(np.mean(stratum_budgets)),
            "active_score": policy_score(stratum_rows, stratum_budgets),
            "matched_random_expected_score": expected_matched_random_score(
                stratum_rows, stratum_budgets
            ),
            "gain_pp": 100.0 * allocation_gain(stratum_rows, stratum_budgets),
        }
    return output


def evaluate_cap(
    rows: Sequence[dict[str, Any]],
    budgets: np.ndarray,
    cap: float,
    random_assignments: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    active = policy_score(rows, budgets)
    random_expected = expected_matched_random_score(rows, budgets)
    return {
        "mean_view_cap": cap,
        "mean_views": float(np.mean(budgets)),
        "budget_counts": {
            str(budget): int(np.sum(budgets == budget)) for budget in (1, 3, 5)
        },
        "active_score": active,
        "matched_random_expected_score": random_expected,
        "gain_pp": 100.0 * (active - random_expected),
        "seeded_random_audit": seeded_random_baseline(
            rows, budgets, random_assignments
        ),
        "grouped_bootstrap": grouped_bootstrap_gain(
            rows, budgets, bootstrap_resamples
        ),
        "difficulty": difficulty_metrics(rows, budgets),
    }


def evaluate_gate(primary: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "gain": primary["gain_pp"]
        >= float(gate["primary_gain_over_exact_budget_matched_random_pp_min"]),
        "bootstrap_lower": primary["grouped_bootstrap"][
            "gain_pp_one_sided_90pct_lower"
        ]
        >= float(gate["primary_one_sided_group_bootstrap_90pct_lower_gain_pp_min"]),
        "mean_view_cap": primary["mean_views"] <= primary["mean_view_cap"] + 1e-12,
        "difficulty_guard": all(
            (not value["supported_for_gate"])
            or value["gain_pp"] >= -float(gate["max_supported_visibility_stratum_regression_pp"])
            for value in primary["difficulty"].values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(gate),
        "decision": "PASS_CONTINUE_TO_M4_R1" if all(checks.values()) else "STOP_BEFORE_M4_R1",
    }


def feature_audit() -> dict[str, Any]:
    names = list(K1_FEATURE_NAMES) + list(K3_FEATURE_NAMES)
    hits = {
        name: [token for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()]
        for name in names
    }
    hits = {name: value for name, value in hits.items() if value}
    if hits:
        raise RuntimeError(f"Forbidden M3-R1 feature names: {hits}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "k1_feature_names": list(K1_FEATURE_NAMES),
        "k3_feature_names": list(K3_FEATURE_NAMES),
        "runtime_features_use_only_acquired_prefix": True,
        "ground_truth_pose_or_pose_error_used_as_feature": False,
        "future_view_prediction_used_as_feature": False,
        "difficulty_fields_used_as_runtime_features": False,
        "difficulty_weighting_note": (
            "Visibility bins may balance training weights for predeclared model "
            "candidates but are absent from every inference feature vector."
        ),
        "allowed_raw_prediction_fields": [
            "foundationpose_top_score",
            "foundationpose_top_score_margin",
            "valid_depth_ratio_inside_mask",
        ],
        "labels": [
            "combined_AR(k3)-combined_AR(k1)",
            "combined_AR(k5)-combined_AR(k3)",
        ],
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "forbidden_feature_token_hits": hits,
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def fold_audit(rows: Sequence[dict[str, Any]], nested: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "splitter": "StratifiedGroupKFold",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "group_key": "physical_instance_track_id",
        "stratification": "2 * target_visibility_code + any_positive_marginal_value",
        "row_count": len(rows),
        "physical_instance_track_count": len(
            {row["physical_instance_track_id"] for row in rows}
        ),
        "all_outer_track_intersections_empty": all(
            fold["track_intersection_count"] == 0 for fold in nested
        ),
        "folds": nested,
    }


def development_rows_for_output(
    rows: Sequence[dict[str, Any]],
    predictions: Mapping[str, np.ndarray],
    budgets_by_cap: Mapping[float, np.ndarray],
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "record_type": "m3_r1_nested_oof",
                "schema_version": SCHEMA_VERSION,
                "group_id": row["group_id"],
                "object_id": row["object_id"],
                "physical_instance_track_id": row["physical_instance_track_id"],
                "target_visibility_bin": row["target_visibility_bin"],
                "k1_marginal_value": row["k1"]["marginal_value"],
                "k1_predicted_marginal_value": float(predictions["k1"][index]),
                "k3_marginal_value": row["k3"]["marginal_value"],
                "k3_predicted_marginal_value": float(predictions["k3"][index]),
                "outcomes": row["outcomes"],
                "selected_budgets": {
                    str(cap): int(budgets[index]) for cap, budgets in budgets_by_cap.items()
                },
            }
        )
    return output


def render_report(result: Mapping[str, Any]) -> str:
    primary = result["operating_points"]["3.0"]
    gate = result["development_gate"]
    lines = [
        "# PoseLoop M3-R1 development result",
        "",
        f"**Decision: `{gate['decision']}`.**",
        "",
        "M3-R1 predicts the marginal value of acquiring more views, not the "
        "success probability of the current pose. Every score below is nested, "
        "physical-instance-grouped out-of-fold development evidence from M2 only; "
        "the frozen legacy M3 holdout was not read.",
        "",
        "## Primary result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Macro-object combined score | {100 * primary['active_score']:.2f}% |",
        f"| Exact-budget matched-random expectation | {100 * primary['matched_random_expected_score']:.2f}% |",
        f"| Gain over matched random | {primary['gain_pp']:+.2f} pp |",
        f"| One-sided 90% grouped-bootstrap lower gain | {primary['grouped_bootstrap']['gain_pp_one_sided_90pct_lower']:+.2f} pp |",
        f"| Mean acquired views | {primary['mean_views']:.3f} / cap 3.0 |",
        f"| k=1 / k=3 / k=5 counts | {primary['budget_counts']['1']} / {primary['budget_counts']['3']} / {primary['budget_counts']['5']} |",
        "",
        "## Gate checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Difficulty strata",
            "",
            "| Visibility | Rows | Mean views | Active | Matched random | Gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for visibility in ("low", "mid", "high"):
        value = primary["difficulty"][visibility]
        lines.append(
            f"| {visibility} | {value['row_count']} | {value['mean_views']:.3f} | "
            f"{100 * value['active_score']:.2f}% | "
            f"{100 * value['matched_random_expected_score']:.2f}% | "
            f"{value['gain_pp']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A development PASS authorizes M4-R1 work; it is not a holdout or sealed "
            "claim. A development FAIL stops before M4-R1 and leaves the legacy "
            "M3-M6 baseline unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_paths(args: argparse.Namespace, repo_root: Path) -> None:
    input_roots = {
        "groups": repo_root / "artifacts" / "m2",
        "metrics": repo_root / "artifacts" / "m2",
        "m1_predictions": repo_root / "artifacts" / "m1",
        "m2_predictions": repo_root / "artifacts" / "m2",
        "protocol": repo_root / "protocols",
    }
    for name, root in input_roots.items():
        path = getattr(args, name).resolve()
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            raise ValueError(f"{name} must be an existing development input under {root}")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to((repo_root / "artifacts" / "r1").resolve()):
        raise ValueError("M3-R1 artifacts must stay under artifacts/r1")
    report = args.report.resolve()
    if not report.is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("M3-R1 report must stay under reports/r1")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    validate_paths(args, repo_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    stage_protocol = protocol["stages"]["M3-R1"]
    if int(stage_protocol["validation"]["outer_folds"]) != OUTER_FOLDS:
        raise ValueError("Protocol/code outer-fold mismatch")
    if int(stage_protocol["validation"]["inner_folds"]) != INNER_FOLDS:
        raise ValueError("Protocol/code inner-fold mismatch")

    inputs = {
        "groups": args.groups.resolve(),
        "metrics": args.metrics.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
        "m2_predictions": args.m2_predictions.resolve(),
        "protocol": args.protocol.resolve(),
    }
    groups = load_jsonl(inputs["groups"])
    metrics = legacy_m3._index_metrics(load_jsonl(inputs["metrics"]))
    prediction_sources = {
        "m1": _load_prediction_index(inputs["m1_predictions"]),
        "m2": _load_prediction_index(inputs["m2_predictions"]),
    }
    rows = build_development_rows(groups, metrics, prediction_sources)
    feature_contract = feature_audit()

    caps = [
        float(value)
        for value in stage_protocol["validation"]["secondary_mean_view_caps"]
    ] + [float(stage_protocol["validation"]["primary_mean_view_cap"])]
    caps = sorted(set(caps))
    budgets_by_cap, predictions, nested_audit = nested_oof(rows, caps)
    operating_points = {
        str(cap): evaluate_cap(
            rows,
            budgets_by_cap[cap],
            cap,
            int(stage_protocol["validation"]["matched_random_assignments"]),
            int(stage_protocol["validation"]["bootstrap_resamples"]),
        )
        for cap in caps
    }
    primary = operating_points[str(stage_protocol["validation"]["primary_mean_view_cap"])]
    gate = evaluate_gate(primary, stage_protocol["development_gate"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M3-R1",
        "evaluation": "nested physical-instance-grouped OOF development",
        "legacy_m3_holdout_read": False,
        "row_count": len(rows),
        "object_count": len({row["object_id"] for row in rows}),
        "physical_instance_track_count": len(
            {row["physical_instance_track_id"] for row in rows}
        ),
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "targets": stage_protocol["targets"],
        "candidate_models": list(MODEL_SPECS),
        "feature_audit": feature_contract,
        "operating_points": operating_points,
        "development_gate": gate,
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "development_result.json", result)
    write_json_atomic(output_root / "feature_audit.json", feature_contract)
    write_json_atomic(output_root / "fold_audit.json", fold_audit(rows, nested_audit))
    write_jsonl_atomic(
        output_root / "nested_oof_predictions.jsonl",
        development_rows_for_output(rows, predictions, budgets_by_cap),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")

    print(f"M3-R1 decision: {gate['decision']}")
    print(
        f"primary: score={primary['active_score']:.6f}, "
        f"matched_random={primary['matched_random_expected_score']:.6f}, "
        f"gain={primary['gain_pp']:+.3f}pp, "
        f"bootstrap_lower={primary['grouped_bootstrap']['gain_pp_one_sided_90pct_lower']:+.3f}pp, "
        f"mean_views={primary['mean_views']:.3f}"
    )
    print(output_root / "development_result.json")
    print(args.report.resolve())


if __name__ == "__main__":
    main()
