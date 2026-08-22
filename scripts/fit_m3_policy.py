#!/usr/bin/env python3
"""Fit and freeze PoseLoop M3 confidence models using M2 development data only.

The reusable feature extraction and frozen-model application helpers in this
module intentionally depend only on the Python standard library and NumPy.
Scikit-learn is imported only by the development fitting path in ``main`` so a
later BOP evaluator can apply the frozen models without adding dependencies.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from m1_common import (
    canonical_sha256,
    load_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
CV_SEED = 20260730
CV_FOLDS = 5
LOGISTIC_C = 1.0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
IMAGE_AREA_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
THRESHOLD_GRID = tuple(round(index * 0.05, 2) for index in range(1, 20))
OPERATING_POINT_CAPS = (
    ("fast", 2.0),
    ("balanced", 3.0),
    ("conservative", 4.0),
)

K1_FEATURE_NAMES = (
    "mask_area_fraction",
    "pose_usable_fraction",
    "predicted_depth_m",
    "predicted_translation_norm_m",
    "predicted_lateral_offset_over_depth",
)
K3_FEATURE_NAMES = (
    "mask_area_fraction_min",
    "mask_area_fraction_median",
    "mask_area_fraction_max",
    "mask_area_fraction_std",
    "pose_usable_fraction",
    "predicted_depth_m",
    "predicted_translation_norm_m",
    "predicted_lateral_offset_over_depth",
    "pairwise_norm_mssd_min",
    "pairwise_norm_mssd_median",
    "pairwise_norm_mssd_max",
    "pairwise_norm_mssd_std",
    "medoid_mean_norm_mssd",
)

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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "artifacts" / "m3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "groups.jsonl",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=repo_root / "artifacts" / "m2" / "metrics.jsonl",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=artifact_root / "development_features.jsonl",
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=artifact_root / "development_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=artifact_root / "development_confidence_metrics.json",
    )
    parser.add_argument(
        "--fold-audit-out",
        type=Path,
        default=artifact_root / "development_fold_audit.json",
    )
    parser.add_argument(
        "--grid-out",
        type=Path,
        default=artifact_root / "development_policy_grid.jsonl",
    )
    parser.add_argument(
        "--feature-audit-out",
        type=Path,
        default=artifact_root / "feature_audit.json",
    )
    parser.add_argument(
        "--frozen-policy-out",
        type=Path,
        default=artifact_root / "frozen_policy.json",
    )
    return parser.parse_args()


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def pose_is_usable(method_result: Mapping[str, Any]) -> bool:
    """Return whether a selected prefix pose can safely receive confidence."""
    if (
        method_result.get("status") != "success"
        or not bool(method_result.get("finite_pose"))
    ):
        return False
    pose = np.asarray(
        method_result.get("predicted_model_to_target_camera_pose_m"),
        dtype=np.float64,
    )
    return bool(pose.shape == (4, 4) and np.all(np.isfinite(pose)))


def _pose_features(method_result: Mapping[str, Any]) -> dict[str, float]:
    """Create finite predicted-pose features, with zeros for unusable poses."""
    if not pose_is_usable(method_result):
        return {
            "pose_usable_fraction": 0.0,
            "predicted_depth_m": 0.0,
            "predicted_translation_norm_m": 0.0,
            "predicted_lateral_offset_over_depth": 0.0,
        }
    pose = np.asarray(
        method_result["predicted_model_to_target_camera_pose_m"],
        dtype=np.float64,
    )
    translation = pose[:3, 3]
    depth = float(translation[2])
    lateral = float(np.linalg.norm(translation[:2]))
    lateral_ratio = lateral / max(abs(depth), 1e-6)
    return {
        "pose_usable_fraction": 1.0,
        "predicted_depth_m": depth,
        "predicted_translation_norm_m": float(np.linalg.norm(translation)),
        "predicted_lateral_offset_over_depth": lateral_ratio,
    }


def _pairwise_disagreements_from_medoid_scores(
    raw_scores: Any,
) -> tuple[list[float], float, float]:
    """Recover finite pairwise distances from M2's mean-disagreement scores."""
    if not isinstance(raw_scores, Mapping):
        return [0.0], 0.0, 0.0
    scores = sorted(
        _finite_float(value, "medoid score")
        for value in raw_scores.values()
    )
    usable_fraction = min(len(scores), 3) / 3.0
    if len(scores) >= 3:
        first, second, third = scores[:3]
        # For three views each stored score is the mean of its two incident
        # edges. Sorting changes vertex names but not the recovered edge set.
        pairwise = [
            first + second - third,
            first + third - second,
            second + third - first,
        ]
        if min(pairwise) < -1e-8:
            raise ValueError(
                "M2 medoid scores cannot represent non-negative pairwise distances"
            )
        pairwise = [max(0.0, float(value)) for value in pairwise]
    elif len(scores) == 2:
        pairwise = [float(np.mean(scores))]
    else:
        pairwise = [0.0]
    medoid_mean = min(scores) if scores else 0.0
    return pairwise, medoid_mean, usable_fraction


def extract_prefix_features(
    group: Mapping[str, Any],
    method_result: Mapping[str, Any],
    budget: int,
) -> dict[str, float]:
    """Extract the audited k=1 or k=3 feature mapping from acquired views only."""
    if budget not in (1, 3):
        raise ValueError(f"Confidence features are defined only at k=1/k=3: {budget}")
    raw_views = group.get("views")
    if not isinstance(raw_views, list) or len(raw_views) < budget:
        raise ValueError(f"Group lacks the requested acquired prefix k={budget}")
    views = sorted(raw_views, key=lambda row: int(row["acquisition_rank"]))
    expected_ranks = list(range(len(views)))
    if [int(row["acquisition_rank"]) for row in views] != expected_ranks:
        raise ValueError("Acquisition ranks must be consecutive and unique")

    mask_fractions = np.asarray(
        [
            _finite_float(
                views[index]["visible_mask_pixel_count"],
                "visible mask pixel count",
            )
            / IMAGE_AREA_PIXELS
            for index in range(budget)
        ],
        dtype=np.float64,
    )
    if np.any(mask_fractions <= 0.0) or np.any(mask_fractions > 1.0):
        raise ValueError("Acquired visible-mask area fraction is outside (0, 1]")

    pose_features = _pose_features(method_result)
    if budget == 1:
        features = {
            "mask_area_fraction": float(mask_fractions[0]),
            **pose_features,
        }
        names = K1_FEATURE_NAMES
    else:
        pairwise, medoid_mean, usable_fraction = (
            _pairwise_disagreements_from_medoid_scores(
                method_result.get(
                    "selection_scores_normalized_symmetric_mssd"
                )
            )
        )
        pairwise_array = np.asarray(pairwise, dtype=np.float64)
        features = {
            "mask_area_fraction_min": float(np.min(mask_fractions)),
            "mask_area_fraction_median": float(np.median(mask_fractions)),
            "mask_area_fraction_max": float(np.max(mask_fractions)),
            "mask_area_fraction_std": float(np.std(mask_fractions)),
            **pose_features,
            "pairwise_norm_mssd_min": float(np.min(pairwise_array)),
            "pairwise_norm_mssd_median": float(np.median(pairwise_array)),
            "pairwise_norm_mssd_max": float(np.max(pairwise_array)),
            "pairwise_norm_mssd_std": float(np.std(pairwise_array)),
            "medoid_mean_norm_mssd": float(medoid_mean),
        }
        # M2 scores also reveal how many prefix poses entered the medoid.
        features["pose_usable_fraction"] = float(usable_fraction)
        names = K3_FEATURE_NAMES

    if set(features) != set(names):
        raise AssertionError(
            f"Feature contract mismatch: expected={names}, actual={tuple(features)}"
        )
    for name, value in features.items():
        _finite_float(value, f"feature {name}")
    return {name: float(features[name]) for name in names}


def ordered_feature_vector(
    features: Mapping[str, Any] | Sequence[float],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Return one finite feature vector in the frozen model's declared order."""
    if isinstance(features, Mapping):
        if set(features) != set(feature_names):
            raise ValueError(
                "Feature mapping keys do not exactly match the frozen feature names"
            )
        vector = np.asarray(
            [features[name] for name in feature_names],
            dtype=np.float64,
        )
    else:
        vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (len(feature_names),) or not np.all(np.isfinite(vector)):
        raise ValueError("Frozen confidence feature vector is non-finite or mis-shaped")
    return vector


def apply_frozen_logistic(
    features: Mapping[str, Any] | Sequence[float],
    frozen_model: Mapping[str, Any],
    *,
    pose_usable: bool,
) -> float:
    """Apply a stored StandardScaler plus binary logistic model using NumPy."""
    if not pose_usable:
        return 0.0
    names = tuple(str(name) for name in frozen_model["feature_names"])
    vector = ordered_feature_vector(features, names)
    scaler = frozen_model["standard_scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    coefficient = np.asarray(
        frozen_model["logistic_regression"]["coefficient"],
        dtype=np.float64,
    )
    intercept = _finite_float(
        frozen_model["logistic_regression"]["intercept"],
        "logistic intercept",
    )
    if mean.shape != vector.shape or scale.shape != vector.shape:
        raise ValueError("Frozen scaler shape does not match feature vector")
    if coefficient.shape != vector.shape or np.any(scale <= 0.0):
        raise ValueError("Frozen logistic/scaler parameters are invalid")
    logit = float(np.dot((vector - mean) / scale, coefficient) + intercept)
    if logit >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probability = exp_logit / (1.0 + exp_logit)
    return float(probability)


def _index_metrics(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        if row.get("record_type") != "method_result":
            raise ValueError(f"Unexpected M2 metric record at row {row_number}")
        key = (
            str(row["group_id"]),
            str(row["method"]),
            int(row["requested_view_budget"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate M2 metric identity: {key}")
        indexed[key] = row
    return indexed


def _required_result(
    indexed: Mapping[tuple[str, str, int], dict[str, Any]],
    group_id: str,
    method: str,
    budget: int,
) -> dict[str, Any]:
    key = (group_id, method, budget)
    if key not in indexed:
        raise ValueError(f"Missing required M2 metric result: {key}")
    return indexed[key]


def _build_development_rows(
    groups: list[dict[str, Any]],
    metrics: Mapping[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    track_objects: dict[str, int] = {}
    for group in groups:
        group_id = str(group.get("group_id", ""))
        if not group_id or group_id in seen_groups:
            raise ValueError(f"Missing or duplicate M2 group ID: {group_id!r}")
        seen_groups.add(group_id)
        object_id = int(group["object_id"])
        track_id = str(group["oracle_association"]["track_id"])
        if not track_id:
            raise ValueError(f"Missing physical-instance track for {group_id}")
        if track_id in track_objects and track_objects[track_id] != object_id:
            raise ValueError(f"Physical track crosses object IDs: {track_id}")
        track_objects[track_id] = object_id

        k1 = _required_result(metrics, group_id, "target_only", 1)
        k3 = _required_result(metrics, group_id, "symmetry_aware_medoid", 3)
        k5 = _required_result(metrics, group_id, "symmetry_aware_medoid", 5)
        if str(k1["target_sample_id"]) != str(group["target_sample_id"]):
            raise ValueError(f"M2 group/metric target mismatch for {group_id}")
        labels = {}
        for name, result in (("k1", k1), ("k3", k3)):
            diagnostic = result.get("diagnostic_success")
            if not isinstance(diagnostic, Mapping) or "joint" not in diagnostic:
                raise ValueError(f"Missing GT-derived development label for {group_id}")
            labels[name] = bool(diagnostic["joint"])

        rows.append(
            {
                "record_type": "development_feature",
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "target_sample_id": str(group["target_sample_id"]),
                "object_id": object_id,
                "physical_instance_track_id": track_id,
                "k1": {
                    "features": extract_prefix_features(group, k1, 1),
                    "feature_names": list(K1_FEATURE_NAMES),
                    "pose_usable": pose_is_usable(k1),
                    "joint_success_label": labels["k1"],
                },
                "k3": {
                    "features": extract_prefix_features(group, k3, 3),
                    "feature_names": list(K3_FEATURE_NAMES),
                    "pose_usable": pose_is_usable(k3),
                    "joint_success_label": labels["k3"],
                },
                "_policy_results": {1: k1, 3: k3, 5: k5},
            }
        )
    rows.sort(key=lambda row: row["group_id"])
    if len(rows) != 300:
        raise ValueError(
            f"M3 development set must contain 300 M2 targets, got {len(rows)}"
        )
    return rows


def _feature_matrix(
    rows: Sequence[dict[str, Any]],
    stage: str,
    names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            ordered_feature_vector(row[stage]["features"], names)
            for row in rows
        ],
        dtype=np.float64,
    )


def _labels(rows: Sequence[dict[str, Any]], stage: str) -> np.ndarray:
    return np.asarray(
        [int(bool(row[stage]["joint_success_label"])) for row in rows],
        dtype=np.int64,
    )


def _serialize_fitted_model(
    scaler: Any,
    logistic: Any,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    if list(logistic.classes_) != [0, 1]:
        raise ValueError(f"Unexpected fitted logistic classes: {logistic.classes_}")
    return {
        "feature_names": list(feature_names),
        "standard_scaler": {
            "with_mean": True,
            "with_std": True,
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
            "variance": [float(value) for value in scaler.var_],
            "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).reshape(-1)[0]),
        },
        "logistic_regression": {
            "classes": [0, 1],
            "coefficient": [float(value) for value in logistic.coef_[0]],
            "intercept": float(logistic.intercept_[0]),
            "C": LOGISTIC_C,
            "penalty": "l2",
            "l1_ratio": 0.0,
            "solver": "lbfgs",
            "fit_intercept": True,
            "class_weight": None,
            "max_iter": 2000,
            "tolerance": 1e-4,
            "random_state": CV_SEED,
            "n_iter": int(logistic.n_iter_[0]),
        },
    }


def _fit_one_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[Any, Any, dict[str, Any]]:
    # Deferred imports keep inference-time reuse free of scikit-learn.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if set(int(value) for value in train_y) != {0, 1}:
        raise ValueError("Each confidence-model training split needs both classes")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(train_x)
    logistic = LogisticRegression(
        C=LOGISTIC_C,
        l1_ratio=0.0,
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=2000,
        tol=1e-4,
        random_state=CV_SEED,
    )
    logistic.fit(scaled, train_y)
    return scaler, logistic, _serialize_fitted_model(
        scaler, logistic, feature_names
    )


def _class_counts(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {"negative": counts[0], "positive": counts[1]}


def _make_folds(
    rows: Sequence[dict[str, Any]],
    stratification: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(
        [row["physical_instance_track_id"] for row in rows],
        dtype=object,
    )
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_SEED,
    )
    folds = list(
        splitter.split(
            np.zeros((len(rows), 1), dtype=np.float64),
            stratification,
            groups,
        )
    )
    seen_test_indices: list[int] = []
    per_fold = []
    track_to_fold: dict[str, int] = {}
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_tracks = set(groups[train_indices])
        test_tracks = set(groups[test_indices])
        overlap = sorted(train_tracks & test_tracks)
        if overlap:
            raise RuntimeError(
                f"Physical-instance leakage in fold {fold_index}: {overlap}"
            )
        for track_id in test_tracks:
            if str(track_id) in track_to_fold:
                raise RuntimeError(
                    f"Physical track assigned to multiple test folds: {track_id}"
                )
            track_to_fold[str(track_id)] = fold_index
        seen_test_indices.extend(int(value) for value in test_indices)
        per_fold.append(
            {
                "fold": fold_index,
                "train_row_count": int(len(train_indices)),
                "test_row_count": int(len(test_indices)),
                "train_physical_instance_count": len(train_tracks),
                "test_physical_instance_count": len(test_tracks),
                "physical_instance_intersection": overlap,
                "physical_instance_intersection_count": len(overlap),
                "stratification_counts_train": dict(
                    sorted(
                        Counter(
                            int(stratification[index])
                            for index in train_indices
                        ).items()
                    )
                ),
                "stratification_counts_test": dict(
                    sorted(
                        Counter(
                            int(stratification[index])
                            for index in test_indices
                        ).items()
                    )
                ),
                "objects_test": dict(
                    sorted(
                        Counter(
                            int(rows[index]["object_id"])
                            for index in test_indices
                        ).items()
                    )
                ),
            }
        )
    if sorted(seen_test_indices) != list(range(len(rows))):
        raise RuntimeError("Each development row must appear in exactly one test fold")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": CV_FOLDS,
        "shuffle": True,
        "random_state": CV_SEED,
        "group_field": "physical_instance_track_id",
        "stratification_target": "2 * k1_joint_success + k3_joint_success",
        "row_count": len(rows),
        "physical_instance_count": len(set(groups)),
        "all_fold_group_intersections_empty": True,
        "each_row_tested_exactly_once": True,
        "track_to_test_fold": dict(sorted(track_to_fold.items())),
        "folds": per_fold,
    }
    return folds, audit


def _ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    weighted_error = 0.0
    rows = []
    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(np.sum(mask))
        confidence = float(np.mean(probabilities[mask])) if count else None
        accuracy = float(np.mean(labels[mask])) if count else None
        absolute_gap = (
            abs(float(confidence) - float(accuracy)) if count else None
        )
        if count:
            weighted_error += count / len(labels) * float(absolute_gap)
        rows.append(
            {
                "bin_index": index,
                "lower_inclusive": lower,
                "upper_inclusive_only_for_last_bin": upper,
                "count": count,
                "mean_probability": confidence,
                "empirical_positive_rate": accuracy,
                "absolute_gap": absolute_gap,
            }
        )
    return float(weighted_error), rows


def _confidence_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if labels.shape != probabilities.shape or len(labels) == 0:
        raise ValueError("Labels/probabilities are empty or mis-shaped")
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("Confidence metrics require both classes")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Confidence probabilities must be finite in [0, 1]")
    ece, bins = _ece(labels, probabilities)
    return {
        "sample_count": int(len(labels)),
        "class_counts": _class_counts(labels),
        "positive_rate": float(np.mean(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": ece,
        "ece_bin_count": 10,
        "ece_definition": (
            "Equal-width [0,1] bins; sample-weighted absolute difference "
            "between mean probability and empirical positive rate."
        ),
        "calibration_bins": bins,
    }


def _confusion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "comparison": "probability >= threshold",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _fit_oof(
    rows: Sequence[dict[str, Any]],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    oof: dict[str, np.ndarray] = {}
    fold_models: dict[str, Any] = {}
    final_models: dict[str, Any] = {}
    for stage, names in (("k1", K1_FEATURE_NAMES), ("k3", K3_FEATURE_NAMES)):
        matrix = _feature_matrix(rows, stage, names)
        labels = _labels(rows, stage)
        probabilities = np.full(len(rows), np.nan, dtype=np.float64)
        serialized_folds = []
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            scaler, logistic, config = _fit_one_model(
                matrix[train_indices],
                labels[train_indices],
                names,
            )
            test_probabilities = logistic.predict_proba(
                scaler.transform(matrix[test_indices])
            )[:, 1]
            # Deterministic safety override: no usable current pose may stop.
            usability = np.asarray(
                [bool(rows[index][stage]["pose_usable"]) for index in test_indices]
            )
            test_probabilities = np.where(usability, test_probabilities, 0.0)
            probabilities[test_indices] = test_probabilities
            serialized_folds.append(
                {
                    "fold": fold_index,
                    "train_class_counts": _class_counts(labels[train_indices]),
                    "test_class_counts": _class_counts(labels[test_indices]),
                    "model": config,
                }
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"OOF probabilities are incomplete for {stage}")
        oof[stage] = probabilities
        fold_models[stage] = serialized_folds

    # The two complete OOF prediction vectors exist before either all-M2 final
    # fit is created. These final fits are the only models used on holdout.
    for stage, names in (("k1", K1_FEATURE_NAMES), ("k3", K3_FEATURE_NAMES)):
        matrix = _feature_matrix(rows, stage, names)
        labels = _labels(rows, stage)
        scaler, logistic, final_config = _fit_one_model(matrix, labels, names)
        sklearn_probabilities = logistic.predict_proba(scaler.transform(matrix))[:, 1]
        manual_probabilities = np.asarray(
            [
                apply_frozen_logistic(
                    row[stage]["features"],
                    final_config,
                    pose_usable=bool(row[stage]["pose_usable"]),
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        usability = np.asarray([bool(row[stage]["pose_usable"]) for row in rows])
        expected = np.where(usability, sklearn_probabilities, 0.0)
        max_difference = float(np.max(np.abs(manual_probabilities - expected)))
        if max_difference > 1e-12:
            raise RuntimeError(
                "Manual frozen-model application disagrees with sklearn: "
                f"{max_difference}"
            )
        final_config["training_row_count"] = len(rows)
        final_config["training_class_counts"] = _class_counts(labels)
        final_config["manual_application_validation"] = {
            "max_absolute_probability_difference": max_difference,
            "tolerance": 1e-12,
            "passed": True,
        }
        final_models[stage] = final_config
    return oof, fold_models, final_models


def _macro_object_mean(
    object_ids: Sequence[int],
    values: Sequence[float],
) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for object_id, value in zip(object_ids, values, strict=True):
        grouped[int(object_id)].append(float(value))
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def _simulate_policy(
    rows: Sequence[dict[str, Any]],
    p1: np.ndarray,
    p3: np.ndarray,
    t1: float,
    t3: float,
) -> dict[str, Any]:
    budgets = []
    ar_mssd = []
    ar_mspd = []
    joint = []
    latencies = []
    object_ids = []
    for index, row in enumerate(rows):
        if p1[index] >= t1:
            budget = 1
        elif p3[index] >= t3:
            budget = 3
        else:
            budget = 5
        result = row["_policy_results"][budget]
        budgets.append(budget)
        object_ids.append(int(row["object_id"]))
        ar_mssd.append(_finite_float(result["sample_ar_mssd"], "sample AR_MSSD"))
        ar_mspd.append(_finite_float(result["sample_ar_mspd"], "sample AR_MSPD"))
        joint.append(float(bool(result["diagnostic_success"]["joint"])))
        runtime = result.get("sequential_runtime_seconds")
        latencies.append(
            _finite_float(runtime, "sequential registration runtime")
        )
    combined = (np.asarray(ar_mssd) + np.asarray(ar_mspd)) / 2.0
    budget_counts = Counter(budgets)
    return {
        "schema_version": SCHEMA_VERSION,
        "t1": float(t1),
        "t3": float(t3),
        "mean_acquired_view_count": float(np.mean(budgets)),
        "macro_object_ar_mssd": _macro_object_mean(object_ids, ar_mssd),
        "macro_object_ar_mspd": _macro_object_mean(object_ids, ar_mspd),
        "macro_object_combined": _macro_object_mean(object_ids, combined),
        "macro_object_joint_success_rate": _macro_object_mean(object_ids, joint),
        "mean_registration_seconds": float(np.mean(latencies)),
        "stopping_counts": {
            "k1": budget_counts[1],
            "k3": budget_counts[3],
            "k5": budget_counts[5],
        },
        "stopping_fractions": {
            "k1": budget_counts[1] / len(rows),
            "k3": budget_counts[3] / len(rows),
            "k5": budget_counts[5] / len(rows),
        },
    }


def _pareto_frontier(grid: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in grid:
        dominated = False
        for other in grid:
            no_more_views = (
                other["mean_acquired_view_count"]
                <= candidate["mean_acquired_view_count"] + 1e-12
            )
            no_less_score = (
                other["macro_object_combined"]
                >= candidate["macro_object_combined"] - 1e-12
            )
            one_strict = (
                other["mean_acquired_view_count"]
                < candidate["mean_acquired_view_count"] - 1e-12
                or other["macro_object_combined"]
                > candidate["macro_object_combined"] + 1e-12
            )
            if no_more_views and no_less_score and one_strict:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            row["mean_acquired_view_count"],
            -row["macro_object_combined"],
            row["t1"],
            row["t3"],
        ),
    )


def _select_operating_points(
    frontier: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for name, cap in OPERATING_POINT_CAPS:
        eligible = [
            row for row in frontier if row["mean_acquired_view_count"] <= cap + 1e-12
        ]
        if not eligible:
            raise RuntimeError(
                f"No Pareto operating point satisfies mean views <= {cap}"
            )
        best = min(
            eligible,
            key=lambda row: (
                -row["macro_object_combined"],
                row["mean_acquired_view_count"],
                -row["t1"],
                -row["t3"],
            ),
        )
        selected.append(
            {
                "name": name,
                "mean_view_cap": cap,
                "selection_rule": (
                    "Maximize macro-object combined development score on the "
                    "OOF Pareto frontier subject to the mean-view cap; ties use "
                    "lower mean views, then higher t1, then higher t3."
                ),
                **{
                    key: value
                    for key, value in best.items()
                    if key != "schema_version"
                },
            }
        )
    return selected


def _feature_audit() -> dict[str, Any]:
    all_names = list(K1_FEATURE_NAMES) + list(K3_FEATURE_NAMES)
    token_hits = {
        name: [
            token for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()
        ]
        for name in all_names
    }
    token_hits = {name: hits for name, hits in token_hits.items() if hits}
    if token_hits:
        raise RuntimeError(f"Forbidden feature-name audit failed: {token_hits}")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "policy_feature_inputs_use_ground_truth": False,
        "ground_truth_use": (
            "M2 diagnostic_success.joint is used only as the development label; "
            "M2 sample AR values are used only to select thresholds on the "
            "development trade-off, never as confidence-model inputs."
        ),
        "known_oracle_mask_scope": (
            "visible_mask_pixel_count is treated as the currently observed mask "
            "area, consistent with PoseLoop's known GT visible-mask diagnostic "
            "scope; visible_fraction and visibility_bin are rejected."
        ),
        "prefix_contract": {
            "k1": "Only acquisition rank 0 is accessed.",
            "k3": "Only acquisition ranks 0, 1, and 2 are accessed.",
            "future_view_access": False,
        },
        "k1_feature_names": list(K1_FEATURE_NAMES),
        "k3_feature_names": list(K3_FEATURE_NAMES),
        "allowed_source_fields": {
            "group.views[acquired_prefix].visible_mask_pixel_count": (
                "currently observed mask geometry"
            ),
            "method_result.predicted_model_to_target_camera_pose_m": (
                "currently selected predicted pose"
            ),
            "method_result.status_and_finite_pose": (
                "current-pose usability and deterministic continue rule"
            ),
            "k3_method_result.selection_scores_normalized_symmetric_mssd": (
                "prediction-to-prediction symmetry-aware disagreement only"
            ),
        },
        "explicitly_rejected_candidates": {
            "gt_model_to_camera_pose_m": "ground-truth pose",
            "normalized_mssd": "ground-truth pose error",
            "mspd_px": "ground-truth pose error",
            "diagnostic_success": "label only, never a model input",
            "sample_ar_mssd_or_mspd": "threshold selection score only",
            "visible_fraction": "ground-truth visibility",
            "visibility_bin": "ground-truth visibility",
            "future_view_masks_or_predictions": "not yet acquired",
            "object_id": "categorical shortcut; metadata/scoring strata only",
            "sample_scene_group_or_track_id": (
                "identity metadata/CV grouping only, never model inputs"
            ),
            "holdout_artifacts": "not read by this script",
        },
        "feature_name_forbidden_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "feature_name_forbidden_token_hits": token_hits,
        "failure_handling": (
            "If the current k1 prediction or k3 medoid is unusable, all features "
            "remain finite, confidence is forcibly set to 0, and the policy "
            "continues to the next budget."
        ),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def _clean_feature_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "_policy_results"}
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    m2_root = (repo_root / "artifacts" / "m2").resolve()
    m3_root = (repo_root / "artifacts" / "m3").resolve()
    input_paths = {
        "groups": args.groups.resolve(),
        "metrics": args.metrics.resolve(),
    }
    output_paths = {
        "features": args.features_out.resolve(),
        "oof_predictions": args.oof_out.resolve(),
        "confidence_metrics": args.metrics_out.resolve(),
        "fold_audit": args.fold_audit_out.resolve(),
        "policy_grid": args.grid_out.resolve(),
        "feature_audit": args.feature_audit_out.resolve(),
        "frozen_policy": args.frozen_policy_out.resolve(),
    }
    for label, path in input_paths.items():
        if not path.is_file() or not path.is_relative_to(m2_root):
            raise ValueError(f"{label} must be an existing M2 artifact: {path}")
    for label, path in output_paths.items():
        if not path.is_relative_to(m3_root):
            raise ValueError(f"{label} must stay under the ignored M3 artifact root")
    if len(set(output_paths.values())) != len(output_paths):
        raise ValueError("M3 policy output paths must be unique")

    groups = load_jsonl(input_paths["groups"])
    metrics = _index_metrics(load_jsonl(input_paths["metrics"]))
    rows = _build_development_rows(groups, metrics)
    k1_labels = _labels(rows, "k1")
    k3_labels = _labels(rows, "k3")
    stratification = 2 * k1_labels + k3_labels
    folds, fold_audit = _make_folds(rows, stratification)
    oof, fold_models, final_models = _fit_oof(rows, folds)

    grid = [
        _simulate_policy(rows, oof["k1"], oof["k3"], t1, t3)
        for t1 in THRESHOLD_GRID
        for t3 in THRESHOLD_GRID
    ]
    frontier = _pareto_frontier(grid)
    operating_points = _select_operating_points(frontier)

    confidence_metrics = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": (
            "Physical-instance-grouped out-of-fold development predictions only"
        ),
        "k1": _confidence_metrics(k1_labels, oof["k1"]),
        "k3": _confidence_metrics(k3_labels, oof["k3"]),
        "confusion_matrices_at_frozen_thresholds": {
            point["name"]: {
                "k1": _confusion(k1_labels, oof["k1"], point["t1"]),
                "k3": _confusion(k3_labels, oof["k3"], point["t3"]),
            }
            for point in operating_points
        },
    }
    oof_rows = [
        {
            "record_type": "development_oof_prediction",
            "schema_version": SCHEMA_VERSION,
            "group_id": row["group_id"],
            "object_id": row["object_id"],
            "physical_instance_track_id": row["physical_instance_track_id"],
            "fold": fold_audit["track_to_test_fold"][
                row["physical_instance_track_id"]
            ],
            "k1_joint_success_label": bool(row["k1"]["joint_success_label"]),
            "k1_pose_usable": bool(row["k1"]["pose_usable"]),
            "k1_probability": float(oof["k1"][index]),
            "k3_joint_success_label": bool(row["k3"]["joint_success_label"]),
            "k3_pose_usable": bool(row["k3"]["pose_usable"]),
            "k3_probability": float(oof["k3"][index]),
        }
        for index, row in enumerate(rows)
    ]
    feature_audit = _feature_audit()
    input_provenance = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in input_paths.items()
    }
    frozen_policy = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PoseLoop-AB confidence-triggered active view budgeting",
        "freeze_status": "frozen_before_holdout_pose_result_evaluation",
        "holdout_artifacts_read": False,
        "development_source": "M2 targets only",
        "input_provenance": input_provenance,
        "development_contract": {
            "row_count": len(rows),
            "object_count": len(set(int(row["object_id"]) for row in rows)),
            "physical_instance_count": len(
                set(row["physical_instance_track_id"] for row in rows)
            ),
            "k1_class_counts": _class_counts(k1_labels),
            "k3_class_counts": _class_counts(k3_labels),
        },
        "feature_audit": feature_audit,
        "cross_validation": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": CV_FOLDS,
            "shuffle": True,
            "random_state": CV_SEED,
            "group_field": "physical_instance_track_id",
            "stratification_target": "2 * k1_joint_success + k3_joint_success",
            "all_fold_group_intersections_empty": True,
        },
        "confidence_models": final_models,
        "oof_fold_models": fold_models,
        "oof_confidence_metrics": confidence_metrics,
        "policy": {
            "sequence": (
                "k1 target; if p1 >= t1 stop, else acquire ranks 1-2 and use "
                "k3 symmetry-aware medoid; if p3 >= t3 stop, else acquire "
                "ranks 3-4 and use k5 symmetry-aware medoid"
            ),
            "confidence_comparison": "probability >= threshold",
            "unusable_pose_probability": 0.0,
            "unusable_pose_action": "continue unless already at k5",
            "additional_view_order": "fixed M2 camera-diversity order",
            "learned_next_best_view": False,
            "threshold_grid": list(THRESHOLD_GRID),
            "grid_size": len(grid),
            "pareto_frontier_size": len(frontier),
            "pareto_objective": (
                "maximize macro-object development combined score while "
                "minimizing mean acquired views"
            ),
            "operating_points": operating_points,
        },
        "runtime_application": {
            "implementation": "apply_frozen_logistic",
            "dependencies": "Python standard library plus NumPy; no sklearn required",
        },
    }
    frozen_policy["configuration_sha256"] = canonical_sha256(frozen_policy)
    # This receipt is deliberately excluded from the deterministic
    # configuration hash above. It proves the completed development-only
    # freeze predates the first holdout prediction stream.
    frozen_policy["frozen_utc"] = datetime.now(timezone.utc).isoformat()

    # All artifacts below are development-only. No holdout path is opened.
    write_jsonl_atomic(output_paths["features"], _clean_feature_rows(rows))
    write_jsonl_atomic(output_paths["oof_predictions"], oof_rows)
    write_json_atomic(output_paths["confidence_metrics"], confidence_metrics)
    write_json_atomic(output_paths["fold_audit"], fold_audit)
    write_jsonl_atomic(output_paths["policy_grid"], grid)
    write_json_atomic(output_paths["feature_audit"], feature_audit)
    # Write the complete frozen configuration last, after all referenced
    # development evidence is durable and before any holdout pose evaluation.
    write_json_atomic(output_paths["frozen_policy"], frozen_policy)

    print(
        "M3 development confidence freeze complete: "
        f"rows={len(rows)}, tracks={fold_audit['physical_instance_count']}, "
        f"frontier={len(frontier)}",
        flush=True,
    )
    for stage in ("k1", "k3"):
        metric = confidence_metrics[stage]
        print(
            f"{stage}: AUROC={metric['auroc']:.6f}, "
            f"AUPRC={metric['auprc']:.6f}, "
            f"Brier={metric['brier_score']:.6f}, ECE={metric['ece']:.6f}",
            flush=True,
        )
    for point in operating_points:
        print(
            f"{point['name']}: t1={point['t1']:.2f}, t3={point['t3']:.2f}, "
            f"score={point['macro_object_combined']:.6f}, "
            f"mean_views={point['mean_acquired_view_count']:.6f}",
            flush=True,
        )
    print(
        f"frozen policy: {output_paths['frozen_policy']} "
        f"sha256={frozen_policy['configuration_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
