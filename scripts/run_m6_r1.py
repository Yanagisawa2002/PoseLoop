#!/usr/bin/env python3
"""Run leakage-safe nested grouped CV for the M6-R1 risk model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from m6_g0_analysis import aurc, compute_prediction_metrics, relative_aurc_improvement


SCHEMA_VERSION = 1
OUTER_FOLDS = 5
INNER_FOLDS = 4
CV_SEED = 20260831
INNER_SEED_BASE = 20260900
MODEL_SEED_BASE = 20261000
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_RESAMPLES = 5000
INNER_TIE_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "m6_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=root / "features.jsonl")
    parser.add_argument("--labels", type=Path, default=root / "labels.jsonl")
    parser.add_argument(
        "--feature-summary", type=Path, default=root / "feature_summary.json"
    )
    parser.add_argument(
        "--final-contract",
        type=Path,
        default=repo_root
        / "artifacts"
        / "r1"
        / "final_multiview"
        / "frozen_contract.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_r1_protocol.json",
    )
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "r1" / "m6_r1_development.md",
    )
    return parser.parse_args()


def feature_families(feature_names: Sequence[str]) -> dict[str, list[str]]:
    names = sorted(feature_names)
    selected = [name for name in names if name.startswith("selected_")]
    selected += ["acquired_view_count", "selected_is_target"]
    output_plus_policy = [
        name
        for name in names
        if not name.startswith(("pair_", "cad_", "geometry_"))
    ]
    result = {
        "selected_output_only": sorted(set(selected)),
        "output_plus_prefix_policy": sorted(set(output_plus_policy)),
        "full_multiview_cad": names,
    }
    if any(not values for values in result.values()):
        raise ValueError("M6-R1 feature family unexpectedly empty")
    return result


def model_candidates(families: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for family in families:
        for c_value in (0.1, 1.0, 10.0):
            candidates.append(
                {
                    "family": family,
                    "model": "logistic",
                    "parameters": {"C": c_value, "class_weight": "balanced"},
                }
            )
        for max_depth, min_samples_leaf in ((2, 12), (3, 10), (4, 8)):
            candidates.append(
                {
                    "family": family,
                    "model": "extra_trees",
                    "parameters": {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "n_estimators": 250,
                        "max_features": 0.75,
                        "class_weight": "balanced",
                    },
                }
            )
    return candidates


def candidate_name(candidate: Mapping[str, Any]) -> str:
    parameters = ",".join(
        f"{key}={candidate['parameters'][key]}" for key in sorted(candidate["parameters"])
    )
    return f"{candidate['family']}|{candidate['model']}|{parameters}"


def candidate_complexity(candidate: Mapping[str, Any]) -> tuple[int, int, float]:
    family_order = {
        "selected_output_only": 0,
        "output_plus_prefix_policy": 1,
        "full_multiview_cad": 2,
    }
    model_order = 0 if candidate["model"] == "logistic" else 1
    if candidate["model"] == "logistic":
        parameter_order = abs(math.log10(float(candidate["parameters"]["C"])))
    else:
        parameter_order = float(candidate["parameters"]["max_depth"])
    return family_order[str(candidate["family"])], model_order, parameter_order


def build_model(candidate: Mapping[str, Any], seed: int) -> Pipeline:
    if candidate["model"] == "logistic":
        classifier = LogisticRegression(
            C=float(candidate["parameters"]["C"]),
            class_weight=str(candidate["parameters"]["class_weight"]),
            max_iter=5000,
            random_state=seed,
            solver="liblinear",
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if candidate["model"] == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=int(candidate["parameters"]["n_estimators"]),
            max_depth=int(candidate["parameters"]["max_depth"]),
            min_samples_leaf=int(candidate["parameters"]["min_samples_leaf"]),
            max_features=float(candidate["parameters"]["max_features"]),
            class_weight=str(candidate["parameters"]["class_weight"]),
            random_state=seed,
            n_jobs=-1,
            bootstrap=False,
        )
        return Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        )
    raise ValueError(f"Unknown model family: {candidate['model']}")


def make_fold_assignments(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Insufficient physical-instance groups for grouped CV")
    for label in (0, 1):
        if len(np.unique(groups[y == label])) < n_splits:
            raise ValueError(f"Class {label} occurs in too few groups")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    assignments = np.full(len(y), -1, dtype=int)
    for fold, (_, test_indices) in enumerate(
        splitter.split(np.zeros((len(y), 1)), y, groups)
    ):
        assignments[test_indices] = fold
    if np.any(assignments < 0):
        raise ValueError("Grouped CV did not assign every row")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for group, fold in zip(groups, assignments, strict=True):
        group_folds[str(group)].add(int(fold))
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A physical-instance group crossed folds")
    for fold in range(n_splits):
        if set(y[assignments == fold]) != {0, 1}:
            raise ValueError(f"Fold {fold} lacks binary class support")
    return assignments


def feature_matrix(
    feature_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in feature_names]
            for row in feature_rows
        ],
        dtype=float,
    )
    if matrix.shape != (len(feature_rows), len(feature_names)):
        raise ValueError("Unexpected M6-R1 feature matrix shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("M6-R1 feature matrix contains non-finite values")
    return matrix


def select_candidate(
    feature_rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    groups: np.ndarray,
    families: Mapping[str, Sequence[str]],
    candidates: Sequence[Mapping[str, Any]],
    assignments: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    matrices = {
        family: feature_matrix(feature_rows, names) for family, names in families.items()
    }
    for candidate_index, candidate in enumerate(candidates):
        probabilities = np.full(len(y), np.nan, dtype=float)
        matrix = matrices[str(candidate["family"])]
        for fold in sorted(np.unique(assignments)):
            train = assignments != fold
            validation = assignments == fold
            model = build_model(candidate, seed + 1000 * candidate_index + int(fold))
            model.fit(matrix[train], y[train])
            probabilities[validation] = model.predict_proba(matrix[validation])[:, 1]
        if np.any(~np.isfinite(probabilities)):
            raise ValueError("Inner OOF probabilities are incomplete")
        scores.append(
            {
                "candidate": dict(candidate),
                "candidate_name": candidate_name(candidate),
                "inner_oof_aurc": aurc(y, probabilities),
            }
        )
    best_aurc = min(float(row["inner_oof_aurc"]) for row in scores)
    eligible = [
        row
        for row in scores
        if float(row["inner_oof_aurc"]) <= best_aurc + INNER_TIE_TOLERANCE
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            candidate_complexity(row["candidate"]),
            float(row["inner_oof_aurc"]),
            str(row["candidate_name"]),
        ),
    )
    selection = {
        **chosen,
        "best_inner_oof_aurc": best_aurc,
        "tie_tolerance": INNER_TIE_TOLERANCE,
        "eligible_within_tolerance": len(eligible),
        "tie_rule": "prefer smaller feature family, then logistic, within 0.005 AURC",
    }
    return selection, scores


def _metric_summary(
    y: np.ndarray,
    risk: np.ndarray,
    probability: np.ndarray | None,
    ids: Sequence[str],
) -> dict[str, Any]:
    return compute_prediction_metrics(
        y,
        risk,
        failure_probability=probability,
        tie_breaker=ids,
    )


def nested_grouped_oof(
    feature_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    families: Mapping[str, Sequence[str]],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray([int(row["y_failure"]) for row in label_rows], dtype=int)
    groups = np.asarray([str(row["physical_instance_id"]) for row in label_rows])
    ids = [str(row["group_id"]) for row in label_rows]
    outer_assignments = make_fold_assignments(y, groups, OUTER_FOLDS, CV_SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    raw_risk = -np.asarray(
        [float(row["features"]["selected_raw_score"]) for row in feature_rows]
    )
    matrices = {
        family: feature_matrix(feature_rows, names) for family, names in families.items()
    }
    for outer_fold in range(OUTER_FOLDS):
        train_indices = np.flatnonzero(outer_assignments != outer_fold)
        test_indices = np.flatnonzero(outer_assignments == outer_fold)
        inner_assignments = make_fold_assignments(
            y[train_indices],
            groups[train_indices],
            INNER_FOLDS,
            INNER_SEED_BASE + outer_fold,
        )
        inner_features = [feature_rows[index] for index in train_indices]
        selection, candidate_scores = select_candidate(
            inner_features,
            y[train_indices],
            groups[train_indices],
            families,
            candidates,
            inner_assignments,
            MODEL_SEED_BASE + outer_fold * 100_000,
        )
        chosen = selection["candidate"]
        matrix = matrices[str(chosen["family"])]
        model = build_model(chosen, MODEL_SEED_BASE + outer_fold)
        model.fit(matrix[train_indices], y[train_indices])
        oof[test_indices] = model.predict_proba(matrix[test_indices])[:, 1]
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        if train_groups.intersection(test_groups):
            raise ValueError("Outer grouped split leakage detected")
        selection_row = {
            "outer_fold": outer_fold,
            "train_target_count": len(train_indices),
            "test_target_count": len(test_indices),
            "train_physical_instance_count": len(train_groups),
            "test_physical_instance_count": len(test_groups),
            "group_intersection_count": 0,
            "selected": selection,
            "candidate_scores": candidate_scores,
        }
        selections.append(selection_row)
        candidate_metrics = _metric_summary(
            y[test_indices], oof[test_indices], oof[test_indices], [ids[i] for i in test_indices]
        )
        baseline_metrics = _metric_summary(
            y[test_indices], raw_risk[test_indices], None, [ids[i] for i in test_indices]
        )
        auroc_gain = float(candidate_metrics["auroc"] - baseline_metrics["auroc"])
        aurc_reduction = relative_aurc_improvement(
            baseline_metrics["aurc"], candidate_metrics["aurc"]
        )
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "target_count": len(test_indices),
                "physical_instance_count": len(test_groups),
                "candidate": candidate_metrics,
                "raw_score_baseline": baseline_metrics,
                "auroc_gain_over_raw": auroc_gain,
                "aurc_relative_reduction_vs_raw": aurc_reduction,
                "positive_direction_both_metrics": bool(
                    auroc_gain > 0 and aurc_reduction is not None and aurc_reduction > 0
                ),
            }
        )
    if np.any(~np.isfinite(oof)):
        raise ValueError("Nested OOF prediction stream is incomplete")
    return oof, outer_assignments, selections, fold_rows


def grouped_bootstrap(
    y: np.ndarray,
    candidate_risk: np.ndarray,
    raw_risk: np.ndarray,
    groups: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique_groups = np.asarray(sorted(set(groups)))
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    auroc_gains: list[float] = []
    aurc_reductions: list[float] = []
    for _ in range(resamples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([indices[group] for group in sampled_groups])
        sampled_y = y[sampled]
        if len(np.unique(sampled_y)) < 2:
            continue
        auroc_gains.append(
            float(
                roc_auc_score(sampled_y, candidate_risk[sampled])
                - roc_auc_score(sampled_y, raw_risk[sampled])
            )
        )
        baseline_aurc = aurc(sampled_y, raw_risk[sampled])
        candidate_aurc = aurc(sampled_y, candidate_risk[sampled])
        aurc_reductions.append((baseline_aurc - candidate_aurc) / baseline_aurc)

    def interval(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "two_sided_90pct_lower": float(np.quantile(array, 0.05)),
            "two_sided_90pct_upper": float(np.quantile(array, 0.95)),
        }

    return {
        "unit": "physical_instance_id",
        "seed": BOOTSTRAP_SEED,
        "requested_resamples": resamples,
        "valid_resamples": len(auroc_gains),
        "auroc_gain_over_raw": interval(auroc_gains),
        "aurc_relative_reduction_vs_raw": interval(aurc_reductions),
    }


def gate_decision(
    candidate_metrics: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
    fold_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    auroc_gain = float(candidate_metrics["auroc"] - raw_metrics["auroc"])
    aurc_reduction = relative_aurc_improvement(
        float(raw_metrics["aurc"]), float(candidate_metrics["aurc"])
    )
    positive_folds = sum(bool(row["positive_direction_both_metrics"]) for row in fold_rows)
    conditions = {
        "auroc_gain_over_raw": auroc_gain
        >= float(gate["nested_grouped_cv_auroc_gain_over_raw_score_min"]),
        "aurc_relative_reduction_vs_raw": aurc_reduction is not None
        and aurc_reduction >= float(gate["aurc_relative_reduction_min"]),
        "positive_direction_outer_folds": positive_folds
        >= int(gate["positive_direction_outer_folds_min"]),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "observed": {
            "auroc_gain_over_raw": auroc_gain,
            "aurc_relative_reduction_vs_raw": aurc_reduction,
            "positive_direction_outer_folds": positive_folds,
        },
        "thresholds": dict(gate),
    }


def stratum_metrics(
    label_rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    risk: np.ndarray,
    raw_risk: np.ndarray,
    field: str,
) -> list[dict[str, Any]]:
    output = []
    values = sorted({str(row[field]) for row in label_rows})
    for value in values:
        indices = np.asarray(
            [index for index, row in enumerate(label_rows) if str(row[field]) == value],
            dtype=int,
        )
        if len(indices) < 2 or len(np.unique(y[indices])) < 2:
            output.append(
                {
                    field: value,
                    "target_count": len(indices),
                    "failure_count": int(y[indices].sum()),
                    "candidate_auroc": None,
                    "candidate_aurc": None,
                    "raw_auroc": None,
                    "raw_aurc": None,
                }
            )
            continue
        candidate = _metric_summary(y[indices], risk[indices], risk[indices], [str(i) for i in indices])
        baseline = _metric_summary(y[indices], raw_risk[indices], None, [str(i) for i in indices])
        output.append(
            {
                field: value,
                "target_count": len(indices),
                "failure_count": int(y[indices].sum()),
                "candidate_auroc": candidate["auroc"],
                "candidate_aurc": candidate["aurc"],
                "raw_auroc": baseline["auroc"],
                "raw_aurc": baseline["aurc"],
            }
        )
    return output


def render_report(result: Mapping[str, Any]) -> str:
    candidate = result["metrics"]["nested_risk_model"]
    raw = result["metrics"]["raw_score_baseline"]
    gate = result["development_gate"]
    selections = Counter(
        row["selected"]["candidate"]["family"] for row in result["outer_model_selections"]
    )
    fold_lines = [
        "| Fold | N | Learned AUROC | Raw AUROC | Learned AURC | Raw AURC | Both improve |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in result["outer_fold_metrics"]:
        fold_lines.append(
            f"| {row['outer_fold']} | {row['target_count']} | "
            f"{row['candidate']['auroc']:.3f} | {row['raw_score_baseline']['auroc']:.3f} | "
            f"{row['candidate']['aurc']:.3f} | {row['raw_score_baseline']['aurc']:.3f} | "
            f"{'yes' if row['positive_direction_both_metrics'] else 'no'} |"
        )
    return "\n".join(
        [
            "# PoseLoop M6-R1 development result",
            "",
            f"**{result['status']}**",
            "",
            "M6-R1 predicts failure risk for the exact frozen M3-R1/M4-R1 output. "
            "Every reported learned prediction is outer-fold OOF by physical-instance "
            "track. The risk model is not allowed to reselect a pose.",
            "",
            "| Metric | Learned risk | Frozen raw-score risk | Difference |",
            "| --- | ---: | ---: | ---: |",
            f"| AUROC | {candidate['auroc']:.4f} | {raw['auroc']:.4f} | {gate['observed']['auroc_gain_over_raw']:+.4f} |",
            f"| AURC (lower is better) | {candidate['aurc']:.4f} | {raw['aurc']:.4f} | {gate['observed']['aurc_relative_reduction_vs_raw']:+.1%} relative |",
            f"| Risk at 80% coverage | {candidate['risk_at_coverage_0_80']:.4f} | {raw['risk_at_coverage_0_80']:.4f} | — |",
            "",
            f"Positive-direction outer folds: {gate['observed']['positive_direction_outer_folds']}/{OUTER_FOLDS}. "
            f"Selected feature families by outer fold: `{dict(selections)}`.",
            "",
            *fold_lines,
            "",
            "This is a grouped development result, not sealed evidence. The final model "
            "and feature order are frozen for the single fresh-sensor validation; opening "
            "that split before data are available is forbidden by the R1 protocol.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    inputs = {
        "features": args.features.resolve(),
        "labels": args.labels.resolve(),
        "feature_summary": args.feature_summary.resolve(),
        "final_contract": args.final_contract.resolve(),
        "protocol": args.protocol.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(r1_root):
        raise ValueError("M6-R1 outputs must stay under artifacts/r1")
    report_path = args.report.resolve()
    if not report_path.is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("M6-R1 report must stay under reports/r1")
    feature_summary = json.loads(inputs["feature_summary"].read_text(encoding="utf-8"))
    if not feature_summary["feature_separation_audit"]["passed"]:
        raise RuntimeError("M6-R1 feature separation audit did not pass")
    final_contract = json.loads(inputs["final_contract"].read_text(encoding="utf-8"))
    if final_contract.get("status") != "frozen_before_m6_r1":
        raise RuntimeError("Final multi-view output was not frozen before M6-R1")
    protocol = json.loads(inputs["protocol"].read_text(encoding="utf-8"))
    stage_protocol = protocol["stages"]["M6-R1"]

    feature_rows = load_jsonl(inputs["features"])
    label_index = {
        str(row["group_id"]): row for row in load_jsonl(inputs["labels"])
    }
    if len(feature_rows) != 300 or len(label_index) != 300:
        raise ValueError("M6-R1 requires exactly 300 feature/label rows")
    label_rows = [label_index[str(row["group_id"])] for row in feature_rows]
    if any(row.get("record_type") != "m6_r1_inference_features" for row in feature_rows):
        raise ValueError("Unexpected M6-R1 feature row")
    if any(row.get("record_type") != "m6_r1_evaluator_label" for row in label_rows):
        raise ValueError("Unexpected M6-R1 label row")
    feature_names = sorted(feature_rows[0]["features"])
    if any(sorted(row["features"]) != feature_names for row in feature_rows):
        raise ValueError("M6-R1 feature schema differs across rows")
    families = feature_families(feature_names)
    candidates = model_candidates(families)

    oof, outer_assignments, selections, fold_rows = nested_grouped_oof(
        feature_rows, label_rows, families, candidates
    )
    y = np.asarray([int(row["y_failure"]) for row in label_rows], dtype=int)
    groups = np.asarray([str(row["physical_instance_id"]) for row in label_rows])
    ids = [str(row["group_id"]) for row in label_rows]
    raw_risk = -np.asarray(
        [float(row["features"]["selected_raw_score"]) for row in feature_rows]
    )
    candidate_metrics = _metric_summary(y, oof, oof, ids)
    raw_metrics = _metric_summary(y, raw_risk, None, ids)
    gate = gate_decision(
        candidate_metrics, raw_metrics, fold_rows, stage_protocol["development_gate"]
    )
    bootstrap = grouped_bootstrap(y, oof, raw_risk, groups)

    full_assignments = make_fold_assignments(y, groups, OUTER_FOLDS, CV_SEED + 1)
    frozen_selection, frozen_candidate_scores = select_candidate(
        feature_rows,
        y,
        groups,
        families,
        candidates,
        full_assignments,
        MODEL_SEED_BASE + 900_000,
    )
    frozen_candidate = frozen_selection["candidate"]
    frozen_feature_names = families[str(frozen_candidate["family"])]
    frozen_matrix = feature_matrix(feature_rows, frozen_feature_names)
    frozen_model = build_model(frozen_candidate, MODEL_SEED_BASE + 999_999)
    frozen_model.fit(frozen_matrix, y)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "frozen_risk_model.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "M6-R1",
            "candidate": frozen_candidate,
            "feature_names": list(frozen_feature_names),
            "model": frozen_model,
            "pose_reselection_allowed": False,
        },
        model_path,
    )

    oof_rows = []
    for index, (feature_row, label_row) in enumerate(zip(feature_rows, label_rows, strict=True)):
        oof_rows.append(
            {
                "record_type": "m6_r1_nested_oof_risk",
                "schema_version": SCHEMA_VERSION,
                "group_id": feature_row["group_id"],
                "physical_instance_id": label_row["physical_instance_id"],
                "object_id": label_row["object_id"],
                "target_visibility_bin": label_row["target_visibility_bin"],
                "outer_fold": int(outer_assignments[index]),
                "y_failure": int(y[index]),
                "raw_score_risk": float(raw_risk[index]),
                "predicted_failure_risk": float(oof[index]),
            }
        )
    oof_path = output_root / "nested_oof_predictions.jsonl"
    write_jsonl_atomic(oof_path, oof_rows)
    write_json_atomic(output_root / "fold_audit.json", {"outer_model_selections": selections})
    write_json_atomic(
        output_root / "frozen_model_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "M6-R1",
            "status": "frozen_for_sealed_validation" if gate["passed"] else "development_gate_failed",
            "selected_candidate": frozen_selection,
            "candidate_scores": frozen_candidate_scores,
            "feature_names": list(frozen_feature_names),
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "pose_reselection_allowed": False,
        },
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M6-R1",
        "status": "PASS_FREEZE_M6_R1" if gate["passed"] else "STOP_M6_R1_DEVELOPMENT_GATE_FAILED",
        "target_count": len(y),
        "failure_count": int(y.sum()),
        "physical_instance_count": len(set(groups)),
        "validation": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "splitter": "StratifiedGroupKFold",
            "group_key": "physical_instance_id",
            "inner_selection_metric": "pooled inner-OOF AURC",
            "inner_tie_tolerance": INNER_TIE_TOLERANCE,
            "candidate_count": len(candidates),
            "all_outer_group_intersections_zero": all(
                row["group_intersection_count"] == 0 for row in selections
            ),
        },
        "metrics": {
            "nested_risk_model": candidate_metrics,
            "raw_score_baseline": raw_metrics,
        },
        "development_gate": gate,
        "grouped_bootstrap": bootstrap,
        "outer_fold_metrics": fold_rows,
        "outer_model_selections": selections,
        "per_visibility": stratum_metrics(
            label_rows, y, oof, raw_risk, "target_visibility_bin"
        ),
        "frozen_full_development_model": {
            "selection": frozen_selection,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "feature_names": list(frozen_feature_names),
        },
        "pose_reselection_after_risk_allowed": False,
        "sealed_validation_opened": False,
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "output_provenance": {
            "oof_predictions": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        },
    }
    result_path = output_root / "development_result.json"
    write_json_atomic(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    print(
        f"{result['status']}: AUROC={candidate_metrics['auroc']:.4f} "
        f"(raw={raw_metrics['auroc']:.4f}), AURC={candidate_metrics['aurc']:.4f} "
        f"(raw={raw_metrics['aurc']:.4f}), positive_folds="
        f"{gate['observed']['positive_direction_outer_folds']}/{OUTER_FOLDS}"
    )
    print(result_path)


if __name__ == "__main__":
    main()
