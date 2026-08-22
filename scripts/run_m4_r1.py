#!/usr/bin/env python3
"""Run the M4-R1 CAD visibility/ranking ablation and development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m1_common import load_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


SCHEMA_VERSION = 1
CV_SEED = 20260815
MODEL_SEED = 20260821
MODEL_SEEDS = (1000, 2000, 3000, MODEL_SEED)
BOOTSTRAP_SEED = 20260822
CV_FOLDS = 5
BOOTSTRAP_RESAMPLES = 5000
ANCHOR_SLOT = 2
MINIMUM_PREDICTED_GAIN = 0.01
CANDIDATE_SLOTS = (1, 2, 3, 4)
NON_ANCHOR_SLOTS = (1, 3, 4)

GEOMETRY_PREFIXES = (
    "relative_",
    "camera_",
    "candidate_distance",
    "target_view_direction",
    "candidate_view_direction",
    "candidate_center_delta",
)
ANALYTIC_CAD_TOKENS = (
    "visible",
    "front_facing",
    "new_surface",
    "projected_size",
)
OCCLUSION_FEATURE_NAMES = (
    "occ_target_visible_surface_fraction",
    "occ_candidate_visible_surface_fraction",
    "occ_new_surface_fraction",
    "occ_lost_surface_fraction",
    "occ_union_surface_fraction",
    "occ_visible_surface_jaccard",
    "occ_overlap_over_target",
    "occ_candidate_silhouette_fraction",
    "occ_silhouette_ratio",
)
FAMILIES = (
    "camera_geometry_only",
    "camera_geometry_plus_front_facing_cad_visibility",
    "camera_geometry_plus_occlusion_aware_cad_visibility",
)
PRIMARY_PIPELINE = (
    "camera_geometry_plus_occlusion_aware_cad_visibility_plus_m3_prefix_gate"
)
MODEL_PARAMETERS = {
    "n_estimators": 1000,
    "max_depth": 4,
    "min_samples_leaf": 8,
    "max_features": 1.0,
    "criterion": "squared_error",
    "bootstrap": False,
    "n_jobs": 1,
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "m4_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=root / "preacquisition_features.jsonl"
    )
    parser.add_argument(
        "--outcomes", type=Path, default=root / "development_outcomes.jsonl"
    )
    parser.add_argument(
        "--occlusion-features",
        type=Path,
        default=root / "cad_occlusion_features.jsonl",
    )
    parser.add_argument(
        "--occlusion-summary",
        type=Path,
        default=root / "cad_occlusion_summary.json",
    )
    parser.add_argument(
        "--m3-oof",
        type=Path,
        default=repo_root
        / "artifacts"
        / "r1"
        / "m3_r1"
        / "nested_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--m3-result",
        type=Path,
        default=repo_root / "artifacts" / "r1" / "m3_r1" / "development_result.json",
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
        default=repo_root / "reports" / "r1" / "m4_r1_development.md",
    )
    return parser.parse_args()


def identity(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["group_id"]), int(row["candidate_slot"])


def load_unique(path: Path, record_type: str) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in load_jsonl(path):
        if row.get("record_type") != record_type:
            raise ValueError(f"Unexpected record type in {path}")
        key = identity(row)
        if key in output:
            raise ValueError(f"Duplicate M4-R1 identity: {key}")
        output[key] = row
    if len(output) != 1200:
        raise ValueError(f"M4-R1 stream must have 1200 rows: {path}")
    return output


def load_m3(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in load_jsonl(path):
        if row.get("record_type") != "m3_r1_nested_oof":
            raise ValueError("Unexpected M3-R1 OOF row")
        group_id = str(row["group_id"])
        if group_id in output:
            raise ValueError(f"Duplicate M3-R1 group: {group_id}")
        output[group_id] = row
    if len(output) != 300:
        raise ValueError(f"M3-R1 OOF stream must have 300 rows, got {len(output)}")
    return output


def join_rows(
    features: Mapping[tuple[str, int], dict[str, Any]],
    outcomes: Mapping[tuple[str, int], dict[str, Any]],
    occlusion: Mapping[tuple[str, int], dict[str, Any]],
    m3: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(features) != set(outcomes) or set(features) != set(occlusion):
        raise ValueError("M4-R1 feature/outcome identities differ")
    rows = []
    for key in sorted(features):
        feature_row = features[key]
        outcome_row = outcomes[key]
        occlusion_row = occlusion[key]
        for field in ("object_id", "physical_instance_id"):
            if feature_row[field] != outcome_row[field] or feature_row[field] != occlusion_row[field]:
                raise ValueError(f"M4-R1 identity mismatch for {key}: {field}")
        group_id, slot = key
        if group_id not in m3:
            raise ValueError(f"M4-R1 group absent from M3-R1 OOF: {group_id}")
        m3_row = m3[group_id]
        if int(feature_row["object_id"]) != int(m3_row["object_id"]):
            raise ValueError(f"M3/M4 object mismatch for {group_id}")
        rows.append(
            {
                "group_id": group_id,
                "candidate_slot": slot,
                "object_id": int(feature_row["object_id"]),
                "physical_instance_id": str(feature_row["physical_instance_id"]),
                "features": feature_row["features"],
                "occlusion_features": occlusion_row["features"],
                "utility": float(outcome_row["candidate_pair_utility"]),
                "target_only_utility": float(outcome_row["target_only_utility"]),
                "sample_ar_mssd": float(outcome_row["sample_ar_mssd"]),
                "sample_ar_mspd": float(outcome_row["sample_ar_mspd"]),
                "joint_success": bool(outcome_row["joint_success"]),
                "m3_primary_budget": int(m3_row["selected_budgets"]["3.0"]),
                "m3_k1_predicted_marginal_value": float(
                    m3_row["k1_predicted_marginal_value"]
                ),
                "target_visibility_bin": str(m3_row["target_visibility_bin"]),
            }
        )
    return rows


def feature_names(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    base_names = tuple(sorted(rows[0]["features"]))
    geometry = tuple(name for name in base_names if name.startswith(GEOMETRY_PREFIXES))
    analytic = tuple(
        name
        for name in base_names
        if name.startswith("cad_") and any(token in name for token in ANALYTIC_CAD_TOKENS)
    )
    observed_occ = tuple(sorted(rows[0]["occlusion_features"]))
    if set(observed_occ) != set(OCCLUSION_FEATURE_NAMES):
        raise ValueError("CAD occlusion feature contract mismatch")
    if not geometry or not analytic:
        raise ValueError("M4-R1 geometry/CAD feature families are empty")
    return {
        FAMILIES[0]: geometry,
        FAMILIES[1]: geometry + analytic,
        FAMILIES[2]: geometry + analytic + tuple(OCCLUSION_FEATURE_NAMES),
    }


def by_group(rows: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        output[str(row["group_id"])].append(index)
    for group_id, indices in output.items():
        indices.sort(key=lambda index: int(rows[index]["candidate_slot"]))
        slots = [int(rows[index]["candidate_slot"]) for index in indices]
        if slots != list(CANDIDATE_SLOTS):
            raise ValueError(f"M4-R1 target lacks slots 1..4: {group_id}")
    if len(output) != 300:
        raise ValueError(f"M4-R1 needs 300 targets, got {len(output)}")
    return dict(sorted(output.items()))


def _raw_value(row: Mapping[str, Any], name: str) -> float:
    source = row["occlusion_features"] if name.startswith("occ_") else row["features"]
    value = float(source[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite M4-R1 feature: {name}")
    return value


def difference_vector(
    rows: Sequence[dict[str, Any]],
    candidate_index: int,
    anchor_index: int,
    names: Sequence[str],
) -> np.ndarray:
    candidate = rows[candidate_index]
    anchor = rows[anchor_index]
    if candidate["group_id"] != anchor["group_id"]:
        raise ValueError("M4-R1 feature difference crosses targets")
    differences = [_raw_value(candidate, name) - _raw_value(anchor, name) for name in names]
    slot_one_hot = [float(candidate["candidate_slot"] == slot) for slot in NON_ANCHOR_SLOTS]
    vector = np.asarray(differences + slot_one_hot, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Non-finite M4-R1 difference vector")
    return vector


def target_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = by_group(rows)
    table = []
    for group_id, indices in groups.items():
        first = rows[indices[0]]
        if len({rows[index]["physical_instance_id"] for index in indices}) != 1:
            raise ValueError(f"Target candidates cross physical tracks: {group_id}")
        table.append(
            {
                "group_id": group_id,
                "object_id": int(first["object_id"]),
                "physical_instance_id": str(first["physical_instance_id"]),
                "eligible": int(first["m3_primary_budget"]) > 1,
                "visibility": str(first["target_visibility_bin"]),
            }
        )
    return table


def make_folds(targets: Sequence[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    objects = np.asarray([row["object_id"] for row in targets], dtype=np.int64)
    tracks = np.asarray([row["physical_instance_id"] for row in targets], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED
    )
    folds = list(
        splitter.split(
            np.zeros((len(targets), 1), dtype=np.float64), objects, tracks
        )
    )
    tested = []
    for train, test in folds:
        if set(tracks[train]) & set(tracks[test]):
            raise RuntimeError("Physical-instance leakage in M4-R1 grouped CV")
        tested.extend(int(index) for index in test)
    if sorted(tested) != list(range(len(targets))):
        raise RuntimeError("M4-R1 grouped CV does not cover every target")
    return folds


def fit_model(matrix: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(random_state=seed, **MODEL_PARAMETERS)
    model.fit(matrix, labels)
    return model


def training_pairs(
    rows: Sequence[dict[str, Any]],
    groups: Mapping[str, list[int]],
    group_ids: set[str],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = []
    labels = []
    for group_id in sorted(group_ids):
        indices = groups[group_id]
        anchor = next(index for index in indices if rows[index]["candidate_slot"] == ANCHOR_SLOT)
        for index in indices:
            if index == anchor:
                continue
            matrix.append(difference_vector(rows, index, anchor, names))
            labels.append(rows[index]["utility"] - rows[anchor]["utility"])
    return np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def crossfit_family(
    rows: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    names: Sequence[str],
) -> tuple[
    dict[str, dict[int, float]],
    list[dict[str, dict[int, float]]],
    list[dict[str, Any]],
]:
    groups = by_group(rows)
    predictions: dict[str, dict[int, float]] = {}
    member_predictions: list[dict[str, dict[int, float]]] = [
        {} for _ in MODEL_SEEDS
    ]
    audit = []
    for fold_index, (train, test) in enumerate(folds):
        train_ids = {targets[int(index)]["group_id"] for index in train}
        test_ids = {targets[int(index)]["group_id"] for index in test}
        matrix, labels = training_pairs(rows, groups, train_ids, names)
        models = [
            fit_model(matrix, labels, seed + fold_index) for seed in MODEL_SEEDS
        ]
        for group_id in sorted(test_ids):
            indices = groups[group_id]
            anchor = next(
                index for index in indices if rows[index]["candidate_slot"] == ANCHOR_SLOT
            )
            predictions[group_id] = {ANCHOR_SLOT: 0.0}
            for member in member_predictions:
                member[group_id] = {ANCHOR_SLOT: 0.0}
            candidate_indices = [index for index in indices if index != anchor]
            candidate_matrix = np.asarray(
                [difference_vector(rows, index, anchor, names) for index in candidate_indices]
            )
            member_values = [model.predict(candidate_matrix) for model in models]
            values = np.mean(np.asarray(member_values, dtype=np.float64), axis=0)
            for candidate_position, (index, value) in enumerate(
                zip(candidate_indices, values, strict=True)
            ):
                slot = int(rows[index]["candidate_slot"])
                predictions[group_id][slot] = float(value)
                for member_index, member in enumerate(member_predictions):
                    member[group_id][slot] = float(
                        member_values[member_index][candidate_position]
                    )
        train_tracks = {targets[int(index)]["physical_instance_id"] for index in train}
        test_tracks = {targets[int(index)]["physical_instance_id"] for index in test}
        audit.append(
            {
                "fold": fold_index,
                "train_target_count": len(train),
                "test_target_count": len(test),
                "train_track_count": len(train_tracks),
                "test_track_count": len(test_tracks),
                "track_intersection_count": len(train_tracks & test_tracks),
                "training_pair_count": len(labels),
                "ensemble_member_count": len(MODEL_SEEDS),
            }
        )
    if len(predictions) != len(targets):
        raise RuntimeError("M4-R1 OOF predictions are incomplete")
    if any(len(member) != len(targets) for member in member_predictions):
        raise RuntimeError("M4-R1 member OOF predictions are incomplete")
    return predictions, member_predictions, audit


def select_slots(
    predictions: Mapping[str, Mapping[int, float]],
    minimum_predicted_gain: float = MINIMUM_PREDICTED_GAIN,
) -> dict[str, int]:
    selected = {}
    for group_id, scores in predictions.items():
        if set(scores) != set(CANDIDATE_SLOTS):
            raise ValueError(f"M4-R1 target lacks four ranking scores: {group_id}")
        best = min(scores, key=lambda slot: (-float(scores[slot]), int(slot)))
        selected[group_id] = (
            best
            if float(scores[best]) >= float(minimum_predicted_gain)
            else ANCHOR_SLOT
        )
    return selected


def utility_maps(
    rows: Sequence[dict[str, Any]], groups: Mapping[str, list[int]]
) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    utilities = {}
    random = {}
    for group_id, indices in groups.items():
        values = []
        for index in indices:
            key = (group_id, int(rows[index]["candidate_slot"]))
            utilities[key] = float(rows[index]["utility"])
            values.append(float(rows[index]["utility"]))
        random[group_id] = float(np.mean(values))
    return utilities, random


def macro_score(
    group_ids: Sequence[str],
    targets: Mapping[str, dict[str, Any]],
    values: Mapping[str, float],
) -> float:
    by_object: dict[int, list[float]] = defaultdict(list)
    for group_id in group_ids:
        by_object[int(targets[group_id]["object_id"])].append(float(values[group_id]))
    return float(np.mean([np.mean(by_object[key]) for key in sorted(by_object)]))


def selected_values(
    group_ids: Sequence[str],
    selected: Mapping[str, int],
    utilities: Mapping[tuple[str, int], float],
) -> dict[str, float]:
    return {group_id: utilities[(group_id, selected[group_id])] for group_id in group_ids}


def rank_correlation(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(set(float(value) for value in actual)) <= 1:
        return None
    value = float(spearmanr(actual, predicted).statistic)
    return value if math.isfinite(value) else None


def evaluate_population(
    group_ids: Sequence[str],
    targets: Mapping[str, dict[str, Any]],
    selected: Mapping[str, int],
    predictions: Mapping[str, Mapping[int, float]],
    utilities: Mapping[tuple[str, int], float],
    random_values: Mapping[str, float],
) -> dict[str, Any]:
    active_values = selected_values(group_ids, selected, utilities)
    anchor_values = {
        group_id: utilities[(group_id, ANCHOR_SLOT)] for group_id in group_ids
    }
    active = macro_score(group_ids, targets, active_values)
    anchor = macro_score(group_ids, targets, anchor_values)
    random = macro_score(group_ids, targets, random_values)
    correlations = []
    oracle_values = {}
    regrets = []
    hits = []
    for group_id in group_ids:
        actual = [utilities[(group_id, slot)] for slot in CANDIDATE_SLOTS]
        predicted = [predictions[group_id][slot] for slot in CANDIDATE_SLOTS]
        correlation = rank_correlation(actual, predicted)
        if correlation is not None:
            correlations.append(correlation)
        oracle = max(actual)
        oracle_values[group_id] = oracle
        selected_value = active_values[group_id]
        regrets.append(oracle - selected_value)
        hits.append(math.isclose(selected_value, oracle, rel_tol=0.0, abs_tol=1e-12))
    return {
        "target_count": len(group_ids),
        "object_count": len({targets[group_id]["object_id"] for group_id in group_ids}),
        "selected_slot_counts": {
            str(slot): sum(selected[group_id] == slot for group_id in group_ids)
            for slot in CANDIDATE_SLOTS
        },
        "macro_object_active": active,
        "macro_object_fixed_slot_2": anchor,
        "macro_object_uniform_random_expected": random,
        "gain_vs_fixed_slot_2_pp": 100.0 * (active - anchor),
        "gain_vs_uniform_random_pp": 100.0 * (active - random),
        "macro_object_oracle": macro_score(group_ids, targets, oracle_values),
        "mean_regret_to_oracle": float(np.mean(regrets)),
        "oracle_hit_rate_tie_aware": float(np.mean(hits)),
        "per_target_spearman": {
            "defined_count": len(correlations),
            "undefined_count": len(group_ids) - len(correlations),
            "mean_where_defined": float(np.mean(correlations)) if correlations else None,
            "median_where_defined": float(np.median(correlations)) if correlations else None,
        },
    }


def bootstrap_primary(
    group_ids: Sequence[str],
    targets: Mapping[str, dict[str, Any]],
    selected: Mapping[str, int],
    utilities: Mapping[tuple[str, int], float],
    random_values: Mapping[str, float],
) -> dict[str, Any]:
    by_track: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        by_track[str(targets[group_id]["physical_instance_id"])].append(group_id)
    tracks = sorted(by_track)
    active_values = selected_values(group_ids, selected, utilities)
    anchor_values = {
        group_id: utilities[(group_id, ANCHOR_SLOT)] for group_id in group_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = np.empty((BOOTSTRAP_RESAMPLES, 2), dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sampled_tracks = rng.choice(tracks, size=len(tracks), replace=True)
        sampled_groups = [
            group_id
            for track in sampled_tracks
            for group_id in by_track[str(track)]
        ]
        active = macro_score(sampled_groups, targets, active_values)
        anchor = macro_score(sampled_groups, targets, anchor_values)
        random = macro_score(sampled_groups, targets, random_values)
        gains[index] = (active - anchor, active - random)
    return {
        "unit": "physical_instance_id",
        "track_count": len(tracks),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "gain_vs_fixed_slot_2_pp": {
            "one_sided_90pct_lower": 100.0 * float(np.quantile(gains[:, 0], 0.10)),
            "interval_95pct": [
                100.0 * float(np.quantile(gains[:, 0], 0.025)),
                100.0 * float(np.quantile(gains[:, 0], 0.975)),
            ],
        },
        "gain_vs_uniform_random_pp": {
            "one_sided_90pct_lower": 100.0 * float(np.quantile(gains[:, 1], 0.10)),
            "interval_95pct": [
                100.0 * float(np.quantile(gains[:, 1], 0.025)),
                100.0 * float(np.quantile(gains[:, 1], 0.975)),
            ],
        },
    }


def gate_decision(
    primary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gate: Mapping[str, Any],
    member_gains_pp: Sequence[float] | None = None,
) -> dict[str, Any]:
    minimum_gain = float(gate["gain_pp_min"])
    minimum_lower = float(gate["one_sided_group_bootstrap_90pct_lower_gain_pp_min"])
    if not math.isclose(
        float(gate["minimum_predicted_gain_to_deviate_from_fixed_slot"]),
        MINIMUM_PREDICTED_GAIN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("M4-R1 protocol/code safety-margin mismatch")
    checks = {
        "gain_vs_fixed_slot": primary["gain_vs_fixed_slot_2_pp"] >= minimum_gain,
        "gain_vs_uniform_random": primary["gain_vs_uniform_random_pp"] >= minimum_gain,
        "bootstrap_vs_fixed_slot": bootstrap["gain_vs_fixed_slot_2_pp"][
            "one_sided_90pct_lower"
        ]
        >= minimum_lower,
        "bootstrap_vs_uniform_random": bootstrap["gain_vs_uniform_random_pp"][
            "one_sided_90pct_lower"
        ]
        >= minimum_lower,
    }
    if "each_seed_ensemble_member_gain_vs_fixed_slot_pp_min" in gate:
        if not member_gains_pp:
            raise ValueError("M4-R1 ensemble-member gains are required by the gate")
        checks["each_seed_member_gain"] = min(member_gains_pp) >= float(
            gate["each_seed_ensemble_member_gain_vs_fixed_slot_pp_min"]
        )
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "thresholds": dict(gate),
        "decision": "PASS_FREEZE_M4_R1" if passed else "STOP_BEFORE_M6_R1",
    }


def dump_joblib(path: Path, value: Any) -> None:
    import joblib

    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def render_report(result: Mapping[str, Any]) -> str:
    primary = result["primary"]
    bootstrap = result["primary_bootstrap"]
    gate = result["development_gate"]
    lines = [
        "# PoseLoop M4-R1 CAD visibility/ranking development result",
        "",
        f"**Decision: `{gate['decision']}`.**",
        "",
        "M4-R1 ranks candidate views only for targets where the cross-fitted "
        "M3-R1 primary policy requests more than one view. The candidate ranker "
        "predicts utility differences from frozen static slot 2 and falls back "
        "to that slot unless another candidate exceeds the frozen 0.01 predicted-gain margin.",
        "",
        "## Primary staged result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| M3-continue targets | {primary['target_count']} |",
        f"| CAD-ranker macro combined | {100 * primary['macro_object_active']:.2f}% |",
        f"| Fixed slot 2 | {100 * primary['macro_object_fixed_slot_2']:.2f}% |",
        f"| Uniform-random expectation | {100 * primary['macro_object_uniform_random_expected']:.2f}% |",
        f"| Gain vs fixed slot 2 | {primary['gain_vs_fixed_slot_2_pp']:+.2f} pp |",
        f"| Gain vs random | {primary['gain_vs_uniform_random_pp']:+.2f} pp |",
        f"| 90% lower gain vs fixed | {bootstrap['gain_vs_fixed_slot_2_pp']['one_sided_90pct_lower']:+.2f} pp |",
        f"| 90% lower gain vs random | {bootstrap['gain_vs_uniform_random_pp']['one_sided_90pct_lower']:+.2f} pp |",
        "",
        "## Ablation on the same M3-continue population",
        "",
        "| Feature family | Macro combined | vs fixed slot 2 | vs random |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, value in result["ablations"].items():
        lines.append(
            f"| `{family}` | {100 * value['eligible']['macro_object_active']:.2f}% | "
            f"{value['eligible']['gain_vs_fixed_slot_2_pp']:+.2f} pp | "
            f"{value['eligible']['gain_vs_uniform_random_pp']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            "| Check | Pass |",
            "| --- | ---: |",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is grouped-CV development evidence, not sealed validation. "
            "Candidate RGB, depth, masks, predictions, scores, and outcomes were "
            "not used by the pre-acquisition feature builders.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_paths(args: argparse.Namespace, repo_root: Path) -> None:
    r1_root = (repo_root / "artifacts" / "r1").resolve()
    for name in (
        "features",
        "outcomes",
        "occlusion_features",
        "occlusion_summary",
        "m3_oof",
        "m3_result",
    ):
        path = getattr(args, name).resolve()
        if not path.is_file() or not path.is_relative_to(r1_root):
            raise ValueError(f"{name} must be an existing R1 artifact")
    if not args.protocol.resolve().is_relative_to((repo_root / "protocols").resolve()):
        raise ValueError("Protocol path must stay under protocols")
    if not args.output_root.resolve().is_relative_to(
        (repo_root / "artifacts" / "r1" / "m4_r1").resolve()
    ):
        raise ValueError("M4-R1 outputs must stay under artifacts/r1/m4_r1")
    if not args.report.resolve().is_relative_to((repo_root / "reports" / "r1").resolve()):
        raise ValueError("M4-R1 report must stay under reports/r1")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    validate_paths(args, repo_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    stage_protocol = protocol["stages"]["M4-R1"]
    m3_result = json.loads(args.m3_result.read_text(encoding="utf-8"))
    if not bool(m3_result["development_gate"]["passed"]):
        raise RuntimeError("M4-R1 cannot run before M3-R1 passes")
    occlusion_summary = json.loads(args.occlusion_summary.read_text(encoding="utf-8"))
    if bool(occlusion_summary["candidate_outcome_stream_read"]):
        raise RuntimeError("CAD visibility renderer read candidate outcomes")

    inputs = {
        "features": args.features.resolve(),
        "outcomes": args.outcomes.resolve(),
        "occlusion_features": args.occlusion_features.resolve(),
        "occlusion_summary": args.occlusion_summary.resolve(),
        "m3_oof": args.m3_oof.resolve(),
        "m3_result": args.m3_result.resolve(),
        "protocol": args.protocol.resolve(),
    }
    rows = join_rows(
        load_unique(inputs["features"], "m4_r1_preacquisition_feature"),
        load_unique(inputs["outcomes"], "m4_r1_development_outcome"),
        load_unique(inputs["occlusion_features"], "m4_r1_cad_occlusion_feature"),
        load_m3(inputs["m3_oof"]),
    )
    groups = by_group(rows)
    targets_list = target_table(rows)
    targets = {row["group_id"]: row for row in targets_list}
    folds = make_folds(targets_list)
    names_by_family = feature_names(rows)
    utilities, random_values = utility_maps(rows, groups)
    all_group_ids = sorted(groups)
    eligible_group_ids = [
        row["group_id"] for row in targets_list if bool(row["eligible"])
    ]

    ablations = {}
    predictions_by_family = {}
    member_predictions_by_family = {}
    fold_audit = None
    for family in FAMILIES:
        predictions, member_predictions, family_audit = crossfit_family(
            rows, targets_list, folds, names_by_family[family]
        )
        selected = select_slots(predictions)
        predictions_by_family[family] = predictions
        member_predictions_by_family[family] = member_predictions
        ablations[family] = {
            "feature_names": list(names_by_family[family])
            + [f"candidate_slot_is_{slot}" for slot in NON_ANCHOR_SLOTS],
            "all_targets": evaluate_population(
                all_group_ids,
                targets,
                selected,
                predictions,
                utilities,
                random_values,
            ),
            "eligible": evaluate_population(
                eligible_group_ids,
                targets,
                selected,
                predictions,
                utilities,
                random_values,
            ),
        }
        if fold_audit is None:
            fold_audit = family_audit
        elif family_audit != fold_audit:
            raise RuntimeError("M4-R1 ablations used different fold assignments")

    final_family = FAMILIES[2]
    final_predictions = predictions_by_family[final_family]
    final_selected = select_slots(final_predictions)
    member_stability = []
    for seed, member_predictions in zip(
        MODEL_SEEDS, member_predictions_by_family[final_family], strict=True
    ):
        member_selected = select_slots(member_predictions)
        member_metric = evaluate_population(
            eligible_group_ids,
            targets,
            member_selected,
            member_predictions,
            utilities,
            random_values,
        )
        member_stability.append(
            {
                "seed": seed,
                "gain_vs_fixed_slot_2_pp": member_metric[
                    "gain_vs_fixed_slot_2_pp"
                ],
                "gain_vs_uniform_random_pp": member_metric[
                    "gain_vs_uniform_random_pp"
                ],
                "selected_slot_counts": member_metric["selected_slot_counts"],
            }
        )
    primary = ablations[final_family]["eligible"]
    primary_bootstrap = bootstrap_primary(
        eligible_group_ids,
        targets,
        final_selected,
        utilities,
        random_values,
    )
    gate = gate_decision(
        primary,
        primary_bootstrap,
        stage_protocol["development_gate"],
        [row["gain_vs_fixed_slot_2_pp"] for row in member_stability],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage": "M4-R1",
        "evaluation": "physical-instance-grouped OOF development",
        "legacy_m4_artifacts_read": False,
        "primary_pipeline": PRIMARY_PIPELINE,
        "m3_continue_definition": "selected_budgets['3.0'] > 1",
        "anchor_slot": ANCHOR_SLOT,
        "minimum_predicted_gain_to_deviate": MINIMUM_PREDICTED_GAIN,
        "model": {
            "type": "mean ExtraTreesRegressor ensemble",
            "member_seeds": list(MODEL_SEEDS),
            "parameters_per_member": MODEL_PARAMETERS,
        },
        "primary": primary,
        "primary_bootstrap": primary_bootstrap,
        "seed_stability": {
            "aggregation": "arithmetic mean of four member predictions",
            "members": member_stability,
            "minimum_member_gain_vs_fixed_slot_2_pp": min(
                row["gain_vs_fixed_slot_2_pp"] for row in member_stability
            ),
            "maximum_member_gain_vs_fixed_slot_2_pp": max(
                row["gain_vs_fixed_slot_2_pp"] for row in member_stability
            ),
        },
        "ablations": ablations,
        "development_gate": gate,
        "input_provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "development_result.json", result)
    write_json_atomic(
        output_root / "fold_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "splitter": "StratifiedGroupKFold",
            "fold_count": CV_FOLDS,
            "group_key": "physical_instance_id",
            "all_track_intersections_empty": all(
                row["track_intersection_count"] == 0 for row in fold_audit or []
            ),
            "folds": fold_audit,
        },
    )
    oof_rows = []
    for group_id in all_group_ids:
        oof_rows.append(
            {
                "record_type": "m4_r1_oof_ranking",
                "schema_version": SCHEMA_VERSION,
                "group_id": group_id,
                "object_id": targets[group_id]["object_id"],
                "physical_instance_id": targets[group_id]["physical_instance_id"],
                "target_visibility_bin": targets[group_id]["visibility"],
                "m3_primary_continue": targets[group_id]["eligible"],
                "predicted_differences_from_slot_2": {
                    str(slot): float(final_predictions[group_id][slot])
                    for slot in CANDIDATE_SLOTS
                },
                "selected_slot": int(final_selected[group_id]),
                "utilities": {
                    str(slot): utilities[(group_id, slot)] for slot in CANDIDATE_SLOTS
                },
            }
        )
    write_jsonl_atomic(output_root / "oof_rankings.jsonl", oof_rows)

    feature_audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "candidate_observation_or_prediction_used": False,
        "candidate_outcome_used_as_feature": False,
        "cad_renderer_read_candidate_outcomes": False,
        "known_object_cad_used": True,
        "camera_extrinsics_used": True,
        "m3_prefix_role": "cross-fitted continue/stop eligibility only",
        "anchor_slot": ANCHOR_SLOT,
        "minimum_predicted_gain_to_deviate": MINIMUM_PREDICTED_GAIN,
        "family_feature_names": {
            family: ablations[family]["feature_names"] for family in FAMILIES
        },
    }
    feature_audit["audit_sha256"] = hashlib.sha256(
        json.dumps(feature_audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json_atomic(output_root / "feature_audit.json", feature_audit)

    if gate["passed"]:
        matrix, labels = training_pairs(
            rows, groups, set(all_group_ids), names_by_family[final_family]
        )
        final_models = [fit_model(matrix, labels, seed) for seed in MODEL_SEEDS]
        model_path = output_root / "frozen_ranker.joblib"
        dump_joblib(model_path, final_models)
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": protocol["protocol_id"],
            "stage": "M4-R1",
            "status": "frozen_after_development_gate_pass",
            "legacy_holdout_read": False,
            "primary_pipeline": PRIMARY_PIPELINE,
            "m3_continue_definition": "selected primary budget > 1",
            "anchor_slot": ANCHOR_SLOT,
            "minimum_predicted_gain_to_deviate": MINIMUM_PREDICTED_GAIN,
            "feature_names": ablations[final_family]["feature_names"],
            "model": {
                "type": "mean ExtraTreesRegressor ensemble",
                "member_seeds": list(MODEL_SEEDS),
                "parameters_per_member": MODEL_PARAMETERS,
                "path": str(model_path),
                "sha256": sha256_file(model_path),
                "training_pair_count": len(labels),
            },
            "development_gate": gate,
            "development_result_sha256": sha256_file(
                output_root / "development_result.json"
            ),
            "feature_audit_sha256": sha256_file(output_root / "feature_audit.json"),
        }
        write_json_atomic(output_root / "frozen_ranker.json", frozen)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(f"M4-R1 decision: {gate['decision']}")
    print(
        f"primary: score={primary['macro_object_active']:.6f}, "
        f"fixed={primary['macro_object_fixed_slot_2']:.6f}, "
        f"random={primary['macro_object_uniform_random_expected']:.6f}, "
        f"gain_fixed={primary['gain_vs_fixed_slot_2_pp']:+.3f}pp, "
        f"lower_fixed={primary_bootstrap['gain_vs_fixed_slot_2_pp']['one_sided_90pct_lower']:+.3f}pp"
    )
    print(output_root / "development_result.json")
    print(args.report.resolve())


if __name__ == "__main__":
    main()
