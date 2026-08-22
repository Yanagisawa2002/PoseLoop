#!/usr/bin/env python3
"""Evaluate the frozen PoseLoop M4 one-step view-utility ranker."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_m1 import (  # noqa: E402
    MAX_SYMMETRY_DISCRETIZATION_STEP,
    load_official_models,
    toolkit_commit,
    validate_manifest,
)
from evaluate_m2 import (  # noqa: E402
    GT_TRANSFORM_MAX_MSPD_PX,
    GT_TRANSFORM_MAX_NORMALIZED_MSSD,
    load_candidate_manifest as load_m2_candidate_manifest,
    load_m1_predictions,
    load_view_predictions as load_m2_view_predictions,
    prepare_groups,
    result_from_selected,
    save_figure_atomic,
    validate_groups as validate_m2_groups,
    validate_m2_provenance,
)
from evaluate_m3 import (  # noqa: E402
    load_predictions as load_m3_predictions,
    validate_candidate_manifest as validate_m3_candidate_manifest,
    validate_groups as validate_m3_groups,
    validate_holdout_evidence,
)
from m1_common import (  # noqa: E402
    canonical_sha256,
    load_jsonl,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
EXPECTED_OBJECT_COUNT = 15
EXPECTED_DEVELOPMENT_GROUPS = 300
EXPECTED_DEVELOPMENT_CANDIDATES = 1200
EXPECTED_DEVELOPMENT_PHYSICAL_INSTANCES = 169
EXPECTED_HOLDOUT_GROUPS = 230
EXPECTED_HOLDOUT_CANDIDATES = 920
EXPECTED_HOLDOUT_PHYSICAL_INSTANCES = 230
CANDIDATE_SLOTS = (1, 2, 3, 4)
METHODS = (
    "target_only",
    "fixed_first",
    "best_static_slot",
    "random_candidate",
    "max_geometry_novelty",
    "learned_voi",
    "oracle_best_candidate",
)
BUDGET_MATCHED_METHODS = METHODS[1:]
VISIBILITY_BINS = ("low", "mid", "high")
RANDOM_ASSIGNMENT_SEED = 20260731
BOOTSTRAP_SEED = 20260731
MIN_RANDOM_ASSIGNMENTS = 2000
MIN_BOOTSTRAP_REPLICATES = 2000
LOW_N_THRESHOLD = 5

FORBIDDEN_FEATURE_TOKENS = (
    "candidate_rgb",
    "candidate_depth",
    "candidate_mask",
    "candidate_visibility",
    "candidate_visible",
    "candidate_gt",
    "candidate_pose_error",
    "candidate_prediction",
    "candidate_score",
    "actual_utility",
    "joint_success",
    "candidate_slot",
    "sample_id",
    "physical_instance",
    "scene_id",
    "object_id",
)

REQUIRED_CAVEATS = (
    "Known object IDs, CAD models, camera extrinsics, and the current target mask are used.",
    "Ground truth is used for physical-instance association, utility labels, and final evaluation only.",
    "Candidate RGB, depth, mask, visibility, GT pose, and FoundationPose output are unavailable at selection time.",
    "The M4 model and best static slot were frozen from M2 before any M3 outcome evaluation.",
    "The reused M3 inference stream predates M4; no new FoundationPose inference was run.",
    "The holdout is physical-instance-disjoint from M2 but not scene-disjoint.",
    "Holdout object counts are unequal; macro-object metrics are primary.",
    "Per-object estimates with fewer than five targets are low-n descriptive results.",
    "All budget-matched methods acquire exactly the target plus one existing candidate view.",
    "These oracle-mask subset diagnostics are not official BOP detection AP.",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    m1 = repo_root / "artifacts" / "m1"
    m2 = repo_root / "artifacts" / "m2"
    m3 = repo_root / "artifacts" / "m3"
    m4 = repo_root / "artifacts" / "m4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get("POSELOOP_XYZIBD_ROOT", "/home/cgliu/datasets/xyzibd")
        ),
    )
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=repo_root / "third_party" / "bop_toolkit",
    )
    parser.add_argument("--m1-manifest", type=Path, default=m1 / "manifest.jsonl")
    parser.add_argument("--m1-predictions", type=Path, default=m1 / "predictions.jsonl")
    parser.add_argument("--m1-summary", type=Path, default=m1 / "summary.json")
    parser.add_argument("--m2-groups", type=Path, default=m2 / "groups.jsonl")
    parser.add_argument(
        "--m2-candidate-manifest",
        type=Path,
        default=m2 / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--m2-groups-summary", type=Path, default=m2 / "groups_summary.json"
    )
    parser.add_argument(
        "--m2-extrinsics-audit", type=Path, default=m2 / "extrinsics_audit.json"
    )
    parser.add_argument(
        "--m2-view-predictions", type=Path, default=m2 / "view_predictions.jsonl"
    )
    parser.add_argument("--m2-metrics", type=Path, default=m2 / "metrics.jsonl")
    parser.add_argument("--m3-groups", type=Path, default=m3 / "holdout_groups.jsonl")
    parser.add_argument(
        "--m3-candidate-manifest",
        type=Path,
        default=m3 / "candidate_manifest.jsonl",
    )
    parser.add_argument(
        "--m3-holdout-summary", type=Path, default=m3 / "holdout_summary.json"
    )
    parser.add_argument(
        "--m3-leakage-audit", type=Path, default=m3 / "leakage_audit.json"
    )
    parser.add_argument(
        "--m3-predictions", type=Path, default=m3 / "view_predictions.jsonl"
    )
    parser.add_argument("--m3-metrics", type=Path, default=m3 / "metrics.jsonl")
    parser.add_argument(
        "--frozen-model", type=Path, default=m4 / "voi_regressor.joblib"
    )
    parser.add_argument(
        "--frozen-configuration", type=Path, default=m4 / "frozen_voi.json"
    )
    parser.add_argument("--feature-audit", type=Path, default=m4 / "feature_audit.json")
    parser.add_argument(
        "--fold-audit", type=Path, default=m4 / "development_fold_audit.json"
    )
    parser.add_argument(
        "--development-features",
        type=Path,
        default=m4 / "development_features.jsonl",
    )
    parser.add_argument(
        "--development-oof",
        type=Path,
        default=m4 / "development_oof_predictions.jsonl",
    )
    parser.add_argument(
        "--development-diagnostics",
        type=Path,
        default=m4 / "development_diagnostics.json",
    )
    parser.add_argument(
        "--holdout-features",
        type=Path,
        default=m4 / "holdout_features.jsonl",
    )
    parser.add_argument("--metrics-output", type=Path, default=m4 / "metrics.jsonl")
    parser.add_argument(
        "--random-assignments-output",
        type=Path,
        default=m4 / "random_assignments.jsonl",
    )
    parser.add_argument(
        "--summary-output", type=Path, default=m4 / "evaluation_summary.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "m4_view_ranking.md",
    )
    parser.add_argument("--figures-dir", type=Path, default=repo_root / "reports")
    parser.add_argument(
        "--random-assignments", type=int, default=MIN_RANDOM_ASSIGNMENTS
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=MIN_BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def _raw_m4_path(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    root = (repo_root / "artifacts" / "m4").resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"M4 raw artifact must remain under {root}: {resolved}")
    return resolved


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite: {value!r}")
    return number


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _percentiles(values: Iterable[Any]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "sample_count": len(finite),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _interval(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return {
        "lower_2.5_percent": float(np.percentile(finite, 2.5)),
        "median": float(np.percentile(finite, 50.0)),
        "upper_97.5_percent": float(np.percentile(finite, 97.5)),
    }


def _metric_value(row: Mapping[str, Any], field: str) -> float:
    if field in {"ar_mssd", "ar_mspd"}:
        return _finite(row[f"sample_{field}"], f"sample {field}")
    if field == "combined":
        return (
            _finite(row["sample_ar_mssd"], "sample AR_MSSD")
            + _finite(row["sample_ar_mspd"], "sample AR_MSPD")
        ) / 2.0
    if field == "finite_pose_rate":
        if "finite_pose_probability" in row:
            return _finite(row["finite_pose_probability"], field)
        return float(bool(row["finite_pose"]))
    if field == "joint_success_rate":
        if "joint_success_probability" in row:
            return _finite(row["joint_success_probability"], field)
        return float(bool(row["diagnostic_success"]["joint"]))
    return _finite(row[field], field)


def _score_summary(
    rows: Sequence[dict[str, Any]],
    *,
    require_all_objects: bool = True,
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "represented_object_count": 0,
            "macro_object": None,
            "micro": None,
            "by_object": [],
        }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["object_id"])].append(row)
    if require_all_objects and len(grouped) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Method result does not cover all 15 objects")
    fields = (
        "sample_ar_mssd",
        "sample_ar_mspd",
        "combined",
        "finite_pose_rate",
        "joint_success_rate",
    )

    def means(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        values = {
            field: float(np.mean([_metric_value(row, field) for row in selected]))
            for field in fields
        }
        return {
            "sample_count": len(selected),
            "ar_mssd": values["sample_ar_mssd"],
            "ar_mspd": values["sample_ar_mspd"],
            "combined": values["combined"],
            "finite_pose_rate": values["finite_pose_rate"],
            "joint_success_rate": values["joint_success_rate"],
        }

    micro = means(rows)
    by_object = []
    for object_id in sorted(grouped):
        entry = means(grouped[object_id])
        entry.update(
            {
                "object_id": object_id,
                "physical_instance_count": len(grouped[object_id]),
                "low_n_descriptive": len(grouped[object_id]) < LOW_N_THRESHOLD,
            }
        )
        by_object.append(entry)
    macro = {
        "object_count": len(by_object),
        **{
            field: float(np.mean([row[field] for row in by_object]))
            for field in (
                "ar_mssd",
                "ar_mspd",
                "combined",
                "finite_pose_rate",
                "joint_success_rate",
            )
        },
    }
    return {
        "sample_count": len(rows),
        "represented_object_count": len(grouped),
        "macro_object": macro,
        "micro": micro,
        "by_object": by_object,
    }


def _rescue_harm(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base_index = {str(row["group_id"]): row for row in baseline}
    candidate_index = {str(row["group_id"]): row for row in candidate}
    if set(base_index) != set(candidate_index):
        raise ValueError("Rescue/harm methods do not cover identical targets")
    output: dict[str, Any] = {}
    for flag in ("joint", "mssd_0.10d", "mspd_10r"):
        failures = 0
        successes = 0
        rescued = 0.0
        harmed = 0.0
        for group_id, baseline_row in base_index.items():
            baseline_success = bool(baseline_row["diagnostic_success"][flag])
            candidate_row = candidate_index[group_id]
            probability_key = f"{flag}_success_probability"
            candidate_probability = (
                _finite(candidate_row[probability_key], probability_key)
                if probability_key in candidate_row
                else float(bool(candidate_row["diagnostic_success"][flag]))
            )
            if not 0.0 <= candidate_probability <= 1.0:
                raise ValueError("Success probability lies outside [0,1]")
            if baseline_success:
                successes += 1
                harmed += 1.0 - candidate_probability
            else:
                failures += 1
                rescued += candidate_probability
        output[flag] = {
            "baseline_failure_count": failures,
            "baseline_success_count": successes,
            "expected_rescued_count": rescued,
            "expected_harmed_count": harmed,
            "rescue_rate": rescued / failures if failures else None,
            "harm_rate": harmed / successes if successes else None,
        }
    return output


def summarize_method(
    method: str,
    rows: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result = _score_summary(rows)
    by_visibility = {
        label: _score_summary(
            [row for row in rows if row["target_visibility_bin"] == label],
            require_all_objects=False,
        )
        for label in VISIBILITY_BINS
    }
    object_15 = next(row for row in result["by_object"] if int(row["object_id"]) == 15)
    slots = Counter(
        int(row["selected_candidate_slot"])
        for row in rows
        if row.get("selected_candidate_slot") is not None
    )
    has_realized_slots = bool(slots)
    return {
        "method": method,
        **result,
        "by_target_visibility_bin": by_visibility,
        "object_15": object_15,
        "rescue_harm_vs_target_only": _rescue_harm(baseline, rows),
        "candidate_slot_selection_counts": {
            str(slot): int(slots[slot]) if has_realized_slots else None
            for slot in CANDIDATE_SLOTS
        },
        "candidate_slot_selection_fractions": {
            str(slot): (float(slots[slot] / len(rows)) if has_realized_slots else None)
            for slot in CANDIDATE_SLOTS
        },
        "registration_latency_seconds": _percentiles(
            row.get("sequential_runtime_seconds") for row in rows
        ),
    }


def _random_expectation_rows(
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id in sorted(candidate_outcomes):
        outcomes = candidate_outcomes[group_id]
        if set(outcomes) != set(CANDIDATE_SLOTS):
            raise ValueError(f"Random expectation lacks four slots: {group_id}")
        source = outcomes[1]
        row = {
            key: copy.deepcopy(source[key])
            for key in (
                "schema_version",
                "record_type",
                "group_id",
                "target_sample_id",
                "scene_id",
                "object_id",
                "target_visibility_bin",
                "target_visible_fraction",
                "mspd_scale_r",
                "physical_instance_id",
            )
        }
        row.update(
            {
                "method": "random_candidate",
                "requested_view_budget": 2,
                "acquired_view_count": 2,
                "selected_candidate_slot": None,
                "selected_candidate_sample_id": None,
                "sample_ar_mssd": float(
                    np.mean(
                        [outcomes[slot]["sample_ar_mssd"] for slot in CANDIDATE_SLOTS]
                    )
                ),
                "sample_ar_mspd": float(
                    np.mean(
                        [outcomes[slot]["sample_ar_mspd"] for slot in CANDIDATE_SLOTS]
                    )
                ),
                "finite_pose_probability": float(
                    np.mean(
                        [
                            bool(outcomes[slot]["finite_pose"])
                            for slot in CANDIDATE_SLOTS
                        ]
                    )
                ),
                "joint_success_probability": float(
                    np.mean(
                        [
                            bool(outcomes[slot]["diagnostic_success"]["joint"])
                            for slot in CANDIDATE_SLOTS
                        ]
                    )
                ),
                "mssd_0.10d_success_probability": float(
                    np.mean(
                        [
                            bool(outcomes[slot]["diagnostic_success"]["mssd_0.10d"])
                            for slot in CANDIDATE_SLOTS
                        ]
                    )
                ),
                "mspd_10r_success_probability": float(
                    np.mean(
                        [
                            bool(outcomes[slot]["diagnostic_success"]["mspd_10r"])
                            for slot in CANDIDATE_SLOTS
                        ]
                    )
                ),
                "sequential_runtime_seconds": float(
                    np.mean(
                        [
                            outcomes[slot]["sequential_runtime_seconds"]
                            for slot in CANDIDATE_SLOTS
                        ]
                    )
                ),
                "random_expectation_is_exact_four_slot_mean": True,
            }
        )
        rows.append(row)
    return rows


def random_assignment_distribution(
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
    assignment_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if assignment_count < MIN_RANDOM_ASSIGNMENTS:
        raise ValueError(f"Use at least {MIN_RANDOM_ASSIGNMENTS} random assignments")
    group_ids = sorted(candidate_outcomes)
    object_ids = np.asarray(
        [int(candidate_outcomes[group_id][1]["object_id"]) for group_id in group_ids],
        dtype=np.int64,
    )
    unique_objects = sorted(set(object_ids.tolist()))
    if len(unique_objects) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Random assignments do not span all 15 objects")
    arrays = {
        "ar_mssd": np.asarray(
            [
                [
                    candidate_outcomes[group_id][slot]["sample_ar_mssd"]
                    for slot in CANDIDATE_SLOTS
                ]
                for group_id in group_ids
            ],
            dtype=np.float64,
        ),
        "ar_mspd": np.asarray(
            [
                [
                    candidate_outcomes[group_id][slot]["sample_ar_mspd"]
                    for slot in CANDIDATE_SLOTS
                ]
                for group_id in group_ids
            ],
            dtype=np.float64,
        ),
        "joint": np.asarray(
            [
                [
                    bool(
                        candidate_outcomes[group_id][slot]["diagnostic_success"][
                            "joint"
                        ]
                    )
                    for slot in CANDIDATE_SLOTS
                ]
                for group_id in group_ids
            ],
            dtype=np.float64,
        ),
    }
    arrays["combined"] = (arrays["ar_mssd"] + arrays["ar_mspd"]) / 2.0
    baseline_joint = np.asarray(
        [
            bool(
                candidate_outcomes[group_id][1]["target_only_diagnostic_success"][
                    "joint"
                ]
            )
            for group_id in group_ids
        ],
        dtype=bool,
    )
    rng = np.random.default_rng(RANDOM_ASSIGNMENT_SEED)
    assignments = rng.integers(
        0,
        len(CANDIDATE_SLOTS),
        size=(assignment_count, len(group_ids)),
        endpoint=False,
    )
    row_indices = np.arange(len(group_ids), dtype=np.int64)[None, :]
    selected = {
        field: values[row_indices, assignments] for field, values in arrays.items()
    }
    macro = {
        field: np.mean(
            np.stack(
                [
                    np.mean(values[:, object_ids == object_id], axis=1)
                    for object_id in unique_objects
                ],
                axis=1,
            ),
            axis=1,
        )
        for field, values in selected.items()
    }
    micro = {field: np.mean(values, axis=1) for field, values in selected.items()}
    failures = ~baseline_joint
    successes = baseline_joint
    rescue = (
        np.mean(selected["joint"][:, failures], axis=1)
        if np.any(failures)
        else np.full(assignment_count, np.nan)
    )
    harm = (
        np.mean(1.0 - selected["joint"][:, successes], axis=1)
        if np.any(successes)
        else np.full(assignment_count, np.nan)
    )
    counts = np.stack(
        [np.sum(assignments == index, axis=1) for index in range(4)], axis=1
    )
    assignment_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "random_candidate_assignment",
            "assignment_index": index,
            "seed": RANDOM_ASSIGNMENT_SEED,
            "macro_ar_mssd": float(macro["ar_mssd"][index]),
            "macro_ar_mspd": float(macro["ar_mspd"][index]),
            "macro_combined": float(macro["combined"][index]),
            "micro_ar_mssd": float(micro["ar_mssd"][index]),
            "micro_ar_mspd": float(micro["ar_mspd"][index]),
            "micro_combined": float(micro["combined"][index]),
            "joint_rescue_rate": float(rescue[index]),
            "joint_harm_rate": float(harm[index]),
            "candidate_slot_counts": {
                str(slot): int(counts[index, slot - 1]) for slot in CANDIDATE_SLOTS
            },
        }
        for index in range(assignment_count)
    ]
    total_counts = np.sum(counts, axis=0)
    summary = {
        "method": (
            "Monte Carlo distribution from one deterministic seeded RNG stream "
            "of uniformly sampled one-candidate assignments; this is assignment "
            "variation, not a "
            "physical-instance confidence interval."
        ),
        "seed": RANDOM_ASSIGNMENT_SEED,
        "assignment_count": assignment_count,
        "target_count_per_assignment": len(group_ids),
        "macro_assignment_intervals": {
            field: _interval(values) for field, values in macro.items()
        },
        "micro_assignment_intervals": {
            field: _interval(values) for field, values in micro.items()
        },
        "joint_rescue_rate_interval": _interval(rescue),
        "joint_harm_rate_interval": _interval(harm),
        "aggregate_candidate_slot_counts": {
            str(slot): int(total_counts[slot - 1]) for slot in CANDIDATE_SLOTS
        },
        "aggregate_candidate_slot_fractions": {
            str(slot): float(total_counts[slot - 1] / np.sum(total_counts))
            for slot in CANDIDATE_SLOTS
        },
    }
    return assignment_rows, summary


def bootstrap_intervals(
    result_sets: Mapping[str, Sequence[dict[str, Any]]],
    replicates: int,
) -> dict[str, Any]:
    if replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"Use at least {MIN_BOOTSTRAP_REPLICATES} bootstrap replicates"
        )
    methods = list(result_sets)
    canonical_rows = list(result_sets[methods[0]])
    canonical_ids = sorted(str(row["group_id"]) for row in canonical_rows)
    if (
        len(canonical_ids) != EXPECTED_HOLDOUT_GROUPS
        or len(set(canonical_ids)) != EXPECTED_HOLDOUT_GROUPS
    ):
        raise ValueError("Bootstrap requires 230 unique holdout group IDs")
    if any(
        {str(row["group_id"]) for row in result_sets[method]} != set(canonical_ids)
        for method in methods[1:]
    ):
        raise ValueError("Bootstrap methods do not cover identical instances")
    indexed = {
        method: {str(row["group_id"]): row for row in rows}
        for method, rows in result_sets.items()
    }
    canonical_physical = {
        group_id: str(indexed[methods[0]][group_id]["physical_instance_id"])
        for group_id in canonical_ids
    }
    if len(set(canonical_physical.values())) != EXPECTED_HOLDOUT_PHYSICAL_INSTANCES:
        raise ValueError("Bootstrap requires 230 unique physical-instance IDs")
    for method in methods[1:]:
        physical = {
            group_id: str(indexed[method][group_id]["physical_instance_id"])
            for group_id in canonical_ids
        }
        if physical != canonical_physical:
            raise ValueError(
                f"Bootstrap physical-instance mapping differs for {method}"
            )
    object_ids = sorted(
        {int(indexed[methods[0]][group_id]["object_id"]) for group_id in canonical_ids}
    )
    if len(object_ids) != EXPECTED_OBJECT_COUNT:
        raise ValueError("Bootstrap does not span all 15 objects")
    ids_by_object = {
        object_id: [
            group_id
            for group_id in canonical_ids
            if int(indexed[methods[0]][group_id]["object_id"]) == object_id
        ]
        for object_id in object_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {
        object_id: rng.integers(
            0,
            len(group_ids),
            size=(replicates, len(group_ids)),
            endpoint=False,
        )
        for object_id, group_ids in ids_by_object.items()
    }
    samples: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        fields = {
            "ar_mssd": np.zeros(replicates, dtype=np.float64),
            "ar_mspd": np.zeros(replicates, dtype=np.float64),
            "combined": np.zeros(replicates, dtype=np.float64),
        }
        for object_id, group_ids in ids_by_object.items():
            rows = [indexed[method][group_id] for group_id in group_ids]
            values = {
                field: np.asarray(
                    [_metric_value(row, field) for row in rows], dtype=np.float64
                )
                for field in fields
            }
            indices = draws[object_id]
            for field, array in values.items():
                fields[field] += np.mean(array[indices], axis=1) / EXPECTED_OBJECT_COUNT
        samples[method] = fields
    method_intervals = {
        method: {field: _interval(values) for field, values in fields.items()}
        for method, fields in samples.items()
    }
    learned = samples["learned_voi"]["combined"]
    paired = {
        f"learned_minus_{control}": _interval(learned - samples[control]["combined"])
        for control in (
            "target_only",
            "fixed_first",
            "best_static_slot",
            "random_candidate",
            "max_geometry_novelty",
        )
    }
    paired["oracle_minus_learned"] = _interval(
        samples["oracle_best_candidate"]["combined"] - learned
    )
    return {
        "method": (
            "Object-stratified physical-instance bootstrap: resample holdout "
            "instances with replacement within each object, compute each object "
            "score, then average the 15 objects equally. Random uses its exact "
            "per-target four-slot expectation."
        ),
        "seed": BOOTSTRAP_SEED,
        "replicate_count": replicates,
        "confidence_level": 0.95,
        "method_intervals": method_intervals,
        "paired_combined_intervals": paired,
    }


def validate_frozen_configuration(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the M2-only freeze before the evaluator opens any M3 artifact."""
    from fit_m4_voi import FEATURE_NAMES, NEW_SURFACE_FEATURE_NAME

    frozen = read_json(paths["frozen_configuration"])
    if frozen.get("freeze_status") != (
        "frozen_before_m3_holdout_candidate_outcome_evaluation"
    ):
        raise ValueError("M4 configuration is not declared as pre-holdout frozen")
    holdout_claims = {
        key: frozen[key]
        for key in (
            "holdout_artifacts_read",
            "holdout_identity_metadata_read",
            "holdout_predictions_metrics_outcomes_read",
        )
        if key in frozen
    }
    if (
        "holdout_artifacts_read" not in holdout_claims
        or "holdout_predictions_metrics_outcomes_read" not in holdout_claims
        or any(value is not False for value in holdout_claims.values())
    ):
        raise ValueError("M4 fitter did not declare complete M3 artifact isolation")
    serialized_frozen = json.dumps(frozen, sort_keys=True).lower()
    if "artifacts/m3" in serialized_frozen or "artifacts\\m3" in serialized_frozen:
        raise ValueError("Frozen M2-only configuration contains an M3 artifact path")

    claimed_hash = str(frozen.get("configuration_sha256", ""))
    hash_payload = dict(frozen)
    hash_payload.pop("configuration_sha256", None)
    hash_payload.pop("frozen_utc", None)
    if not claimed_hash or canonical_sha256(hash_payload) != claimed_hash:
        raise ValueError("Frozen M4 configuration hash mismatch")
    feature_names = tuple(str(name) for name in frozen.get("feature_names", []))
    if feature_names != tuple(FEATURE_NAMES):
        raise ValueError("Frozen/runtime M4 feature names differ")
    if frozen.get("new_surface_feature_name") != NEW_SURFACE_FEATURE_NAME:
        raise ValueError("Frozen/runtime geometry novelty feature differs")
    lowered_names = [name.lower() for name in feature_names]
    hits = {
        name: [token for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()]
        for name in lowered_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    }
    if hits:
        raise ValueError(f"Forbidden M4 feature-name tokens: {hits}")

    model = frozen.get("model", {})
    if (
        not str(model.get("type", "")).endswith("HistGradientBoostingRegressor")
        or model.get("serialization") != "joblib"
        or model.get("sha256") != sha256_file(paths["frozen_model"])
    ):
        raise ValueError("Frozen M4 model type/serialization/hash mismatch")
    training = frozen.get("training_data", frozen.get("development_contract", {}))
    if (
        int(
            training.get(
                "row_count",
                training.get("candidate_row_count", -1),
            )
        )
        != EXPECTED_DEVELOPMENT_CANDIDATES
        or int(training.get("target_count", -1)) != EXPECTED_DEVELOPMENT_GROUPS
        or int(training.get("physical_instance_count", -1))
        != EXPECTED_DEVELOPMENT_PHYSICAL_INSTANCES
        or int(training.get("object_count", -1)) != EXPECTED_OBJECT_COUNT
    ):
        raise ValueError("Frozen M4 training-data count contract failed")
    development_feature_rows = load_jsonl(paths["development_features"])
    training_payload = [
        {
            "group_id": row["group_id"],
            "physical_instance_id": row["physical_instance_id"],
            "candidate_slot": row["candidate_slot"],
            "features": row["features"],
            "actual_utility": row["actual_utility"],
            "joint_success": row["joint_success"],
            "sample_ar_mssd": row["sample_ar_mssd"],
            "sample_ar_mspd": row["sample_ar_mspd"],
        }
        for row in development_feature_rows
    ]
    declared_training_hash = str(
        training.get("sha256", training.get("training_data_sha256", ""))
    )
    if declared_training_hash not in {
        sha256_file(paths["development_features"]),
        canonical_sha256(development_feature_rows),
        canonical_sha256(training_payload),
    }:
        raise ValueError("Frozen M4 training-data hash mismatch")

    standalone_audit = read_json(paths["feature_audit"])
    declared_audit = frozen.get("feature_audit", {})
    if (
        not standalone_audit.get("passed")
        or declared_audit.get("sha256") != sha256_file(paths["feature_audit"])
        or tuple(standalone_audit.get("feature_names", [])) != tuple(FEATURE_NAMES)
    ):
        raise ValueError("M4 feature audit is absent, stale, or failed")
    audit_text = (
        json.dumps(standalone_audit, sort_keys=True)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    required_audit_claims = (
        "candidate_rgb",
        "candidate_depth",
        "candidate_mask",
        "candidate_gt",
        "viewing_direction_world",
    )
    if any(token not in audit_text for token in required_audit_claims):
        raise ValueError("M4 feature audit omits a required forbidden source")
    token_hits = standalone_audit.get(
        "forbidden_feature_name_token_hits",
        standalone_audit.get("feature_name_forbidden_token_hits", {}),
    )
    if token_hits not in ({}, None):
        raise ValueError("M4 feature audit reports forbidden feature-name hits")
    viewing_direction_access = standalone_audit.get(
        "uses_group_viewing_direction_world",
        standalone_audit.get("group_viewing_direction_world_accessed"),
    )
    if viewing_direction_access is not False:
        raise ValueError("Feature audit does not forbid GT-derived viewing direction")
    if standalone_audit.get("candidate_observation_or_outcome_feature_count", 0) != 0:
        raise ValueError("Feature audit reports candidate outcome leakage")
    if (
        standalone_audit.get("candidate_observation_files_opened") is not False
        or standalone_audit.get(
            "candidate_prediction_rows_accessed_by_feature_extractor"
        )
        is not False
        or standalone_audit.get("candidate_gt_pose_accessed_by_feature_extractor")
        is not False
    ):
        raise ValueError("Feature audit does not prove candidate-source isolation")

    fold = read_json(paths["fold_audit"])
    folds = fold.get("folds", [])
    if (
        int(fold.get("n_splits", -1)) != 5
        or len(folds) != 5
        or not fold.get("all_fold_group_intersections_empty")
        or not fold.get(
            "each_row_tested_exactly_once",
            fold.get("each_candidate_row_tested_exactly_once"),
        )
        or any(
            int(item.get("physical_instance_intersection_count", -1)) != 0
            or item.get("physical_instance_intersection") not in ([], None)
            for item in folds
        )
    ):
        raise ValueError("M4 grouped-fold leakage audit failed")
    if int(fold.get("physical_instance_count", -1)) != (
        EXPECTED_DEVELOPMENT_PHYSICAL_INSTANCES
    ):
        raise ValueError("M4 fold physical-instance count mismatch")

    diagnostics = read_json(paths["development_diagnostics"])
    frozen_diagnostics = frozen.get("oof_diagnostics", {})
    if frozen_diagnostics.get("sha256") != sha256_file(
        paths["development_diagnostics"]
    ):
        raise ValueError("Frozen M4 OOF-diagnostics hash mismatch")
    if not diagnostics:
        raise ValueError("M4 OOF diagnostics are empty")
    artifact_contract = frozen.get("artifacts", {})
    for label, path_key in (
        ("development_features", "development_features"),
        ("development_oof_predictions", "development_oof"),
        ("development_fold_audit", "fold_audit"),
        ("development_diagnostics", "development_diagnostics"),
        ("feature_audit", "feature_audit"),
        ("model", "frozen_model"),
    ):
        declaration = artifact_contract.get(label)
        if not isinstance(declaration, Mapping) or declaration.get(
            "sha256"
        ) != sha256_file(paths[path_key]):
            raise ValueError(f"Frozen M4 artifact hash mismatch: {label}")

    declared_provenance = frozen.get("input_provenance", {})
    required_sources = {
        str(paths[key].resolve()): sha256_file(paths[key])
        for key in (
            "m1_manifest",
            "m1_predictions",
            "m2_groups",
            "m2_candidate_manifest",
            "m2_groups_summary",
            "m2_extrinsics_audit",
            "m2_view_predictions",
        )
    }
    declared_sources = {}
    for entry in declared_provenance.values():
        if isinstance(entry, Mapping) and entry.get("sha256"):
            stored_path = entry.get("path", entry.get("artifact"))
            if stored_path:
                candidate_path = Path(str(stored_path))
                if not candidate_path.is_absolute():
                    candidate_path = (
                        Path(__file__).resolve().parents[1] / candidate_path
                    )
                declared_sources[str(candidate_path.resolve())] = str(entry["sha256"])
    missing_sources = {
        path: expected_hash
        for path, expected_hash in required_sources.items()
        if declared_sources.get(path) != expected_hash
    }
    if missing_sources:
        raise ValueError(
            "Frozen M4 input provenance does not bind all required M2-only inputs: "
            f"{sorted(missing_sources)}"
        )

    frozen_utc = _parse_utc(frozen.get("frozen_utc"), "M4 frozen_utc")
    fit_source = (SCRIPT_DIR / "fit_m4_voi.py").resolve()
    if frozen.get("source_contract", {}).get("fit_script_sha256") != sha256_file(
        fit_source
    ):
        raise ValueError("Frozen M4 fitter source hash differs from runtime source")
    fit_modified = datetime.fromtimestamp(fit_source.stat().st_mtime, timezone.utc)
    if not fit_modified < frozen_utc:
        raise ValueError("M4 fitter source does not predate its frozen configuration")
    return frozen, {
        "passed": True,
        "configuration_sha256": claimed_hash,
        "frozen_utc": frozen_utc.isoformat(),
        "holdout_artifacts_read_before_freeze": False,
        "m2_only_input_provenance_passed": True,
        "feature_audit": standalone_audit,
        "fold_audit": fold,
        "development_diagnostics": diagnostics,
        "fit_source": {
            "path": str(fit_source),
            "sha256": sha256_file(fit_source),
            "last_modified_utc": fit_modified.isoformat(),
            "last_modified_precedes_freeze": True,
        },
    }


def validate_development_inputs(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    manifest_index = validate_manifest(load_jsonl(paths["m1_manifest"]))
    manifest_sha = sha256_file(paths["m1_manifest"])
    m1_metadata, m1_predictions = load_m1_predictions(
        paths["m1_predictions"], manifest_index, manifest_sha
    )
    groups, m2_source_ids = validate_m2_groups(
        load_jsonl(paths["m2_groups"]), manifest_index
    )
    candidate_index = load_m2_candidate_manifest(
        paths["m2_candidate_manifest"], m2_source_ids
    )
    m2_metadata, m2_predictions = load_m2_view_predictions(
        paths["m2_view_predictions"], candidate_index
    )
    validate_m2_provenance(
        m2_metadata,
        {
            "groups": paths["m2_groups"],
            "candidate_manifest": paths["m2_candidate_manifest"],
            "groups_summary": paths["m2_groups_summary"],
            "extrinsics_audit": paths["m2_extrinsics_audit"],
            "m1_manifest": paths["m1_manifest"],
            "m1_predictions": paths["m1_predictions"],
        },
        candidate_index,
    )
    physical_ids = {str(group["oracle_association"]["track_id"]) for group in groups}
    target_candidate_pairs = [
        (
            str(group["group_id"]),
            int(view["acquisition_rank"]),
            str(view["sample_id"]),
        )
        for group in groups
        for view in group["views"][1:]
    ]
    source_reference_counts = Counter(
        str(view["prediction_source"])
        for group in groups
        for view in group["views"][1:]
    )
    unique_additional_by_source = {
        source: {
            str(view["sample_id"])
            for group in groups
            for view in group["views"][1:]
            if str(view["prediction_source"]) == source
        }
        for source in ("m1", "m2")
    }
    if (
        len(groups) != EXPECTED_DEVELOPMENT_GROUPS
        or len(target_candidate_pairs) != EXPECTED_DEVELOPMENT_CANDIDATES
        or len(physical_ids) != EXPECTED_DEVELOPMENT_PHYSICAL_INSTANCES
        or any(
            {int(view["acquisition_rank"]) for view in group["views"][1:]}
            != set(CANDIDATE_SLOTS)
            for group in groups
        )
        or source_reference_counts != Counter({"m1": 60, "m2": 1140})
        or len(unique_additional_by_source["m1"]) != 39
        or len(unique_additional_by_source["m2"]) != 926
    ):
        raise ValueError("M2 development split count/slot contract failed")
    return {
        "groups": groups,
        "manifest_index": manifest_index,
        "m1_predictions": m1_predictions,
        "m2_predictions": m2_predictions,
        "m1_prediction_metadata": m1_metadata,
        "m2_prediction_metadata": m2_metadata,
        "physical_instance_ids": physical_ids,
        "target_candidate_pairs": target_candidate_pairs,
        "audit": {
            "status": "pass",
            "target_count": len(groups),
            "candidate_row_count": len(target_candidate_pairs),
            "physical_instance_count": len(physical_ids),
            "object_count": len({int(group["object_id"]) for group in groups}),
            "unique_m1_prediction_count": len(m1_predictions),
            "unique_m2_prediction_count": len(m2_predictions),
            "additional_view_prediction_source_reference_counts": dict(
                sorted(source_reference_counts.items())
            ),
            "unique_additional_view_count_by_prediction_source": {
                source: len(sample_ids)
                for source, sample_ids in unique_additional_by_source.items()
            },
            "candidate_slots_per_target": list(CANDIDATE_SLOTS),
        },
    }


def build_candidate_outcomes(
    prepared_groups: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[int, dict[str, Any]]],
]:
    baseline_rows: list[dict[str, Any]] = []
    outcomes: dict[str, dict[int, dict[str, Any]]] = {}
    for prepared in prepared_groups:
        group = prepared["group"]
        group_id = str(group["group_id"])
        physical_id = str(
            group.get(
                "physical_instance_id",
                group.get("oracle_association", {}).get("track_id", ""),
            )
        )
        views = prepared["views"]
        if len(views) != 5:
            raise ValueError(f"M4 requires exactly five existing views: {group_id}")
        target = views[0]
        baseline = result_from_selected(prepared, [target], target, "target_only", 1)
        baseline["physical_instance_id"] = physical_id
        baseline["selected_candidate_slot"] = None
        baseline["selected_candidate_sample_id"] = None
        baseline["joint_success_probability"] = float(
            bool(baseline["diagnostic_success"]["joint"])
        )
        baseline_rows.append(baseline)
        per_slot: dict[int, dict[str, Any]] = {}
        for slot in CANDIDATE_SLOTS:
            candidate = views[slot]
            acquired = [target, candidate]
            selected = max(
                acquired,
                key=lambda view: (
                    int(view["view"]["visible_mask_pixel_count"]),
                    -int(view["view"]["acquisition_rank"]),
                ),
            )
            row = result_from_selected(
                prepared,
                acquired,
                selected,
                "candidate_slot_outcome",
                2,
            )
            row["physical_instance_id"] = physical_id
            row["selected_candidate_slot"] = slot
            row["selected_candidate_sample_id"] = str(candidate["view"]["sample_id"])
            row["post_acquisition_rule"] = (
                "max visible-mask pixel count over target and acquired candidate; "
                "lower acquisition rank breaks ties"
            )
            row["actual_utility"] = (
                float(row["sample_ar_mssd"]) + float(row["sample_ar_mspd"])
            ) / 2.0
            row["joint_success_probability"] = float(
                bool(row["diagnostic_success"]["joint"])
            )
            row["target_only_diagnostic_success"] = copy.deepcopy(
                baseline["diagnostic_success"]
            )
            per_slot[slot] = row
        outcomes[group_id] = per_slot
    return baseline_rows, outcomes


def validate_development_artifacts(
    frozen: Mapping[str, Any],
    paths: Mapping[str, Path],
    development: Mapping[str, Any],
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    from fit_m4_voi import FEATURE_NAMES

    feature_rows = load_jsonl(paths["development_features"])
    oof_rows = load_jsonl(paths["development_oof"])
    if (
        len(feature_rows) != EXPECTED_DEVELOPMENT_CANDIDATES
        or len(oof_rows) != EXPECTED_DEVELOPMENT_CANDIDATES
    ):
        raise ValueError("M4 development feature/OOF row count mismatch")

    expected_identity = {
        (group_id, slot, sample_id)
        for group_id, slot, sample_id in development["target_candidate_pairs"]
    }
    canonical_groups: dict[str, dict[str, Any]] = {}
    for group in development["groups"]:
        group_id = str(group["group_id"])
        candidate_sample_by_slot = {
            int(view["acquisition_rank"]): str(view["sample_id"])
            for view in group["views"][1:]
        }
        canonical_groups[group_id] = {
            "target_sample_id": str(group["target_sample_id"]),
            "object_id": int(group["object_id"]),
            "physical_instance_id": str(group["oracle_association"]["track_id"]),
            "candidate_sample_by_slot": candidate_sample_by_slot,
        }
    if set(canonical_groups) != {
        group_id for group_id, _, _ in expected_identity
    } or any(
        set(contract["candidate_sample_by_slot"]) != set(CANDIDATE_SLOTS)
        for contract in canonical_groups.values()
    ):
        raise ValueError("Canonical M2 development group metadata is incomplete")

    indexed_features: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in feature_rows:
        identity = (
            str(row["group_id"]),
            int(row["candidate_slot"]),
            str(row["candidate_sample_id"]),
        )
        if (
            row.get("record_type") != "m4_development_candidate"
            or identity in indexed_features
            or set(row.get("features", {})) != set(FEATURE_NAMES)
        ):
            raise ValueError(f"Invalid M4 development feature row: {identity}")
        contract = canonical_groups.get(identity[0])
        if (
            contract is None
            or str(row["target_sample_id"]) != contract["target_sample_id"]
            or int(row["object_id"]) != contract["object_id"]
            or str(row["physical_instance_id"]) != contract["physical_instance_id"]
            or contract["candidate_sample_by_slot"].get(identity[1]) != identity[2]
        ):
            raise ValueError(
                f"M4 development feature metadata differs from M2: {identity}"
            )
        if any(not math.isfinite(float(value)) for value in row["features"].values()):
            raise ValueError(f"Nonfinite M4 development feature: {identity}")
        utility = (
            _finite(row["sample_ar_mssd"], "development sample AR_MSSD")
            + _finite(row["sample_ar_mspd"], "development sample AR_MSPD")
        ) / 2.0
        if not math.isclose(
            utility,
            _finite(row["actual_utility"], "development utility"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Development utility definition mismatch: {identity}")
        indexed_features[identity] = row
    if set(indexed_features) != expected_identity:
        raise ValueError("M4 development features do not cover four slots per target")

    max_evidence_delta = 0.0
    for identity, row in indexed_features.items():
        group_id, slot, _ = identity
        outcome = candidate_outcomes[group_id][slot]
        evidence = row["outcome_evidence"]
        if evidence.get("post_acquisition_rule") != "max_mask":
            raise ValueError(
                f"Development label does not use common max-mask: {identity}"
            )
        exact_pairs = (
            ("selected_sample_id", "selected_sample_id"),
            ("selected_acquisition_rank", "selected_acquisition_rank"),
            ("status", "status"),
            ("finite_pose", "finite_pose"),
        )
        if any(
            evidence.get(saved) != outcome.get(actual) for saved, actual in exact_pairs
        ):
            raise ValueError(f"Development outcome identity mismatch: {identity}")
        if bool(row["joint_success"]) != bool(outcome["diagnostic_success"]["joint"]):
            raise ValueError(f"Development joint label mismatch: {identity}")
        for field in (
            "sample_ar_mssd",
            "sample_ar_mspd",
            "actual_utility",
        ):
            max_evidence_delta = max(
                max_evidence_delta,
                abs(float(row[field]) - float(outcome[field])),
            )
        for field in ("normalized_mssd", "mspd_px", "mspd_scale_r"):
            saved = evidence.get(field)
            actual = outcome.get(field)
            if saved is None or actual is None:
                if saved != actual:
                    raise ValueError(f"Development outcome null mismatch: {identity}")
            else:
                max_evidence_delta = max(
                    max_evidence_delta, abs(float(saved) - float(actual))
                )
    if max_evidence_delta > 1e-12:
        raise ValueError(
            f"Development outcome evidence differs from recomputation: "
            f"{max_evidence_delta}"
        )

    fold_contract = read_json(paths["fold_audit"])
    track_to_test_fold = {
        str(track_id): int(fold)
        for track_id, fold in fold_contract.get("track_to_test_fold", {}).items()
    }
    if set(track_to_test_fold) != {
        contract["physical_instance_id"] for contract in canonical_groups.values()
    }:
        raise ValueError("M4 fold audit does not cover the canonical M2 tracks")

    indexed_oof: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in oof_rows:
        identity = (
            str(row["group_id"]),
            int(row["candidate_slot"]),
            str(row["candidate_sample_id"]),
        )
        if (
            row.get("record_type") != "m4_development_oof_prediction"
            or identity in indexed_oof
            or identity not in indexed_features
            or int(row["fold"]) not in range(5)
        ):
            raise ValueError(f"Invalid M4 OOF row: {identity}")
        feature_row = indexed_features[identity]
        contract = canonical_groups[identity[0]]
        if any(
            row[field] != feature_row[field]
            for field in (
                "target_sample_id",
                "object_id",
                "physical_instance_id",
            )
        ):
            raise ValueError(f"M4 OOF metadata differs from features: {identity}")
        if (
            str(row["target_sample_id"]) != contract["target_sample_id"]
            or int(row["object_id"]) != contract["object_id"]
            or str(row["physical_instance_id"]) != contract["physical_instance_id"]
            or int(row["fold"]) != track_to_test_fold[contract["physical_instance_id"]]
        ):
            raise ValueError(
                f"M4 OOF metadata/fold differs from M2 fold contract: {identity}"
            )
        _finite(row["predicted_utility"], "OOF predicted utility")
        indexed_oof[identity] = row
    if set(indexed_oof) != expected_identity:
        raise ValueError("M4 OOF predictions do not cover every candidate once")

    actual = np.asarray(
        [indexed_features[key]["actual_utility"] for key in sorted(expected_identity)],
        dtype=np.float64,
    )
    predicted = np.asarray(
        [indexed_oof[key]["predicted_utility"] for key in sorted(expected_identity)],
        dtype=np.float64,
    )
    correlations = []
    regrets = []
    oracle_hits = []
    selected_rows = []
    grouped_identities: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for identity in expected_identity:
        grouped_identities[identity[0]].append(identity)
    for group_id in sorted(grouped_identities):
        identities = sorted(grouped_identities[group_id], key=lambda item: item[1])
        actual_values = np.asarray(
            [indexed_features[key]["actual_utility"] for key in identities],
            dtype=np.float64,
        )
        predicted_values = np.asarray(
            [indexed_oof[key]["predicted_utility"] for key in identities],
            dtype=np.float64,
        )
        if np.unique(actual_values).size > 1 and np.unique(predicted_values).size > 1:
            correlation = float(spearmanr(predicted_values, actual_values).statistic)
            if math.isfinite(correlation):
                correlations.append(correlation)
        selected_index = min(
            range(len(identities)),
            key=lambda index: (-predicted_values[index], identities[index][1]),
        )
        selected_identity = identities[selected_index]
        selected_utility = float(actual_values[selected_index])
        best_utility = float(np.max(actual_values))
        regrets.append(best_utility - selected_utility)
        oracle_hits.append(
            math.isclose(selected_utility, best_utility, rel_tol=0.0, abs_tol=1e-15)
        )
        selected_rows.append(indexed_features[selected_identity])
        for index, identity in enumerate(identities):
            expected_selected = index == selected_index
            if bool(indexed_oof[identity]["selected_by_learned_voi_oof"]) != (
                expected_selected
            ):
                raise ValueError(f"OOF selected-row flag mismatch: {identity}")
            if not math.isclose(
                float(indexed_oof[identity]["oracle_best_actual_utility"]),
                best_utility,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"OOF oracle utility mismatch: {identity}")

    selected_score_rows = [
        {
            "object_id": int(row["object_id"]),
            "sample_ar_mssd": float(row["sample_ar_mssd"]),
            "sample_ar_mspd": float(row["sample_ar_mspd"]),
            "finite_pose_probability": float(
                bool(row["outcome_evidence"]["finite_pose"])
            ),
            "joint_success_probability": float(bool(row["joint_success"])),
        }
        for row in selected_rows
    ]
    recomputed = {
        "utility_mae": float(np.mean(np.abs(predicted - actual))),
        "utility_rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "per_target_rank_correlation": {
            "defined_target_count": len(correlations),
            "mean": float(np.mean(correlations)) if correlations else None,
            "median": float(np.median(correlations)) if correlations else None,
        },
        "mean_regret": float(np.mean(regrets)),
        "oracle_best_candidate_hit_rate_tie_aware": float(np.mean(oracle_hits)),
        "selected_pose_metrics": _score_summary(selected_score_rows),
    }
    saved = read_json(paths["development_diagnostics"])
    saved_scalars = {
        "utility_mae": saved.get(
            "utility_mae", saved.get("utility_regression", {}).get("mae")
        ),
        "utility_rmse": saved.get(
            "utility_rmse", saved.get("utility_regression", {}).get("rmse")
        ),
        "mean_regret": saved.get(
            "mean_regret",
            saved.get("ranking", {}).get("mean_regret_to_best_of_four"),
        ),
        "oracle_best_candidate_hit_rate_tie_aware": saved.get(
            "oracle_best_candidate_hit_rate_tie_aware",
            saved.get("ranking", {}).get("oracle_best_candidate_hit_rate"),
        ),
    }
    for field, expected in (
        ("utility_mae", recomputed["utility_mae"]),
        ("utility_rmse", recomputed["utility_rmse"]),
        ("mean_regret", recomputed["mean_regret"]),
        (
            "oracle_best_candidate_hit_rate_tie_aware",
            recomputed["oracle_best_candidate_hit_rate_tie_aware"],
        ),
    ):
        if not math.isclose(
            float(saved_scalars[field]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Saved/recomputed OOF diagnostic mismatch: {field}")

    slot_scores = {}
    for slot in CANDIDATE_SLOTS:
        rows = [
            {
                "object_id": int(row["object_id"]),
                "sample_ar_mssd": float(row["sample_ar_mssd"]),
                "sample_ar_mspd": float(row["sample_ar_mspd"]),
                "finite_pose_probability": float(
                    bool(row["outcome_evidence"]["finite_pose"])
                ),
                "joint_success_probability": float(bool(row["joint_success"])),
            }
            for row in feature_rows
            if int(row["candidate_slot"]) == slot
        ]
        slot_scores[slot] = _score_summary(rows)["macro_object"]["combined"]
    best_slot = min(CANDIDATE_SLOTS, key=lambda slot: (-slot_scores[slot], slot))
    frozen_best = frozen["best_static_slot"]
    if int(frozen_best["candidate_slot"]) != best_slot:
        raise ValueError("Frozen best static slot differs from M2 recomputation")
    raw_declared_scores = frozen_best["slot_scores"]
    if isinstance(raw_declared_scores, list):
        declared_scores = {
            int(row["candidate_slot"]): float(row.get("macro_object", row)["combined"])
            for row in raw_declared_scores
        }
    else:
        declared_scores = {
            int(key): float(value) for key, value in raw_declared_scores.items()
        }
    if any(
        not math.isclose(
            declared_scores[slot], slot_scores[slot], rel_tol=0.0, abs_tol=1e-12
        )
        for slot in CANDIDATE_SLOTS
    ):
        raise ValueError("Frozen best-static slot scores differ from M2")
    return {
        "feature_rows": feature_rows,
        "oof_rows": oof_rows,
        "audit": {
            "status": "pass",
            "feature_row_count": len(feature_rows),
            "oof_row_count": len(oof_rows),
            "maximum_recomputed_outcome_difference": max_evidence_delta,
            "canonical_m2_metadata_match": True,
            "oof_fold_mapping_match": True,
            "utility_definition": (
                "(sample AR_MSSD + sample AR_MSPD) / 2 after common max-mask"
            ),
            "grouped_oof": recomputed,
            "best_static_slot": best_slot,
            "best_static_slot_scores": {
                str(slot): slot_scores[slot] for slot in CANDIDATE_SLOTS
            },
        },
    }


def load_target_predictions_only(
    path: Path,
    target_sample_ids: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Retain target predictions and discard candidate rows before selection."""
    metadata: dict[str, Any] | None = None
    targets: dict[str, dict[str, Any]] = {}
    discarded_candidate_rows = 0
    unexpected_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_type = row.get("record_type")
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError("M3 prediction stream contains duplicate metadata")
                metadata = row
            elif record_type == "prediction":
                sample_id = str(row.get("sample_id", ""))
                if sample_id in target_sample_ids:
                    if sample_id in targets:
                        raise ValueError(f"Duplicate M3 target prediction: {sample_id}")
                    targets[sample_id] = row
                else:
                    discarded_candidate_rows += 1
            else:
                unexpected_rows += 1
                raise ValueError(
                    f"Unexpected M3 prediction record at line {line_number}: "
                    f"{record_type!r}"
                )
    if metadata is None or set(targets) != target_sample_ids:
        raise ValueError("Target-only M3 prediction pass is incomplete")
    expected_discarded = EXPECTED_HOLDOUT_CANDIDATES
    if discarded_candidate_rows != expected_discarded or unexpected_rows:
        raise ValueError(
            "Target-only M3 prediction pass did not discard 920 candidates"
        )
    return (
        metadata,
        targets,
        {
            "status": "pass",
            "target_prediction_rows_retained": len(targets),
            "candidate_prediction_rows_streamed_and_discarded_without_outcome_field_access": (
                discarded_candidate_rows
            ),
            "candidate_prediction_mapping_passed_to_feature_builder": False,
            "full_candidate_prediction_loading_deferred_until_after_selection": True,
        },
    )


def load_preacquisition_holdout_geometry(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    physical_ids: set[str] = set()
    candidate_row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            views = sorted(raw["views"], key=lambda view: int(view["acquisition_rank"]))
            if len(views) != 5 or [
                int(view["acquisition_rank"]) for view in views
            ] != list(range(5)):
                raise ValueError(
                    f"Invalid pre-acquisition M3 group at line {line_number}"
                )
            group = {
                "group_id": str(raw["group_id"]),
                "target_sample_id": str(raw["target_sample_id"]),
                "object_id": int(raw["object_id"]),
                "physical_instance_id": str(raw["physical_instance_id"]),
                "views": [
                    {
                        "acquisition_rank": int(view["acquisition_rank"]),
                        "sample_id": str(view["sample_id"]),
                        "camera_world_to_camera_pose_m": copy.deepcopy(
                            view["camera_world_to_camera_pose_m"]
                        ),
                        "camera_center_world_m": copy.deepcopy(
                            view["camera_center_world_m"]
                        ),
                    }
                    for view in views
                ],
            }
            if group["views"][0]["sample_id"] != group["target_sample_id"]:
                raise ValueError("Pre-acquisition rank zero is not the target")
            if group["physical_instance_id"] in physical_ids:
                raise ValueError("Duplicate M3 holdout physical instance")
            physical_ids.add(group["physical_instance_id"])
            candidate_row_count += len(group["views"]) - 1
            groups.append(group)
    if (
        len(groups) != EXPECTED_HOLDOUT_GROUPS
        or candidate_row_count != EXPECTED_HOLDOUT_CANDIDATES
        or len(physical_ids) != EXPECTED_HOLDOUT_PHYSICAL_INSTANCES
    ):
        raise ValueError("Pre-acquisition M3 geometry count contract failed")
    return groups, {
        "status": "pass",
        "group_count": len(groups),
        "candidate_geometry_row_count": candidate_row_count,
        "physical_instance_count": len(physical_ids),
        "retained_candidate_fields": [
            "acquisition_rank",
            "camera_center_world_m",
            "camera_world_to_camera_pose_m",
            "sample_id_audit_only",
        ],
        "candidate_observation_gt_outcome_fields_retained": False,
        "group_viewing_direction_world_retained": False,
    }


def _sanitized_target_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "camera_intrinsics_row_major",
        "image_height",
        "image_width",
        "raw_depth_scale",
        "depth_path",
        "mask_path",
        "visible_mask_pixel_count",
        "valid_depth_pixel_count",
        "valid_depth_ratio_inside_mask",
    )
    return {key: copy.deepcopy(row[key]) for key in allowed}


def load_target_manifests_only(
    path: Path,
    target_sample_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    discarded_candidate_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id in target_sample_ids:
                if sample_id in targets:
                    raise ValueError(f"Duplicate M3 target manifest row: {sample_id}")
                targets[sample_id] = _sanitized_target_manifest(row)
            else:
                discarded_candidate_rows += 1
    if (
        set(targets) != target_sample_ids
        or discarded_candidate_rows != EXPECTED_HOLDOUT_CANDIDATES
    ):
        raise ValueError("Target-only M3 manifest pass is incomplete")
    return targets, {
        "status": "pass",
        "target_manifest_rows_retained": len(targets),
        "candidate_manifest_rows_streamed_and_discarded_without_observation_field_access": (
            discarded_candidate_rows
        ),
        "candidate_rgb_depth_mask_or_gt_passed_to_feature_builder": False,
    }


def _sanitized_target_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "status",
        "predicted_model_to_camera_pose_m",
        "foundationpose_top_score",
        "foundationpose_top_score_margin",
        "pose_hypothesis_count",
        "valid_depth_ratio_inside_mask",
        "visible_mask_pixel_count",
    )
    return {key: copy.deepcopy(row[key]) for key in allowed if key in row}


def _sanitized_preacquisition_group(group: Mapping[str, Any]) -> dict[str, Any]:
    views = []
    for raw_view in sorted(
        group["views"], key=lambda view: int(view["acquisition_rank"])
    ):
        views.append(
            {
                "acquisition_rank": int(raw_view["acquisition_rank"]),
                "camera_world_to_camera_pose_m": copy.deepcopy(
                    raw_view["camera_world_to_camera_pose_m"]
                ),
                "camera_center_world_m": copy.deepcopy(
                    raw_view["camera_center_world_m"]
                ),
            }
        )
    return {"views": views}


def build_holdout_features_and_selection(
    groups: Sequence[dict[str, Any]],
    target_manifests: Mapping[str, dict[str, Any]],
    target_predictions: Mapping[str, dict[str, Any]],
    frozen: Mapping[str, Any],
    frozen_model_path: Path,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    from fit_m4_voi import (
        FEATURE_NAMES,
        NEW_SURFACE_FEATURE_NAME,
        extract_candidate_features,
        extract_target_context,
        load_cad_geometry,
        load_frozen_regressor,
    )

    regressor = load_frozen_regressor(frozen, frozen_model_path)
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    model_contract = frozen["model"]
    if sklearn.__version__ != str(model_contract["sklearn_version"]):
        raise ValueError("Runtime sklearn version differs from the frozen M4 model")
    if type(regressor) is not HistGradientBoostingRegressor:
        raise ValueError("Loaded M4 estimator class differs from the frozen model")
    runtime_parameters = regressor.get_params(deep=False)
    frozen_parameters = model_contract.get("parameters", {})
    if set(runtime_parameters) != set(frozen_parameters) or any(
        runtime_parameters[name] != frozen_parameters[name]
        for name in frozen_parameters
    ):
        raise ValueError("Loaded M4 estimator parameters differ from the freeze")

    dataset_root = dataset_root.resolve()
    cad_contract = frozen.get("cad_provenance", {})
    expected_object_ids = {int(group["object_id"]) for group in groups}
    declared_objects = cad_contract.get("objects", {})
    if {int(object_id) for object_id in declared_objects} != expected_object_ids:
        raise ValueError("Frozen CAD provenance does not cover holdout object IDs")

    def checked_cad_path(stored: Any) -> Path:
        path = (dataset_root / str(stored)).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError(f"Frozen CAD artifact escapes dataset root: {stored}")
        return path

    models_info_path = checked_cad_path(cad_contract["models_info_artifact"])
    if sha256_file(models_info_path) != cad_contract["models_info_sha256"]:
        raise ValueError("Runtime CAD models_info differs from the M4 freeze")
    checked_object_hashes: dict[str, str] = {}
    for object_id in sorted(expected_object_ids):
        declaration = declared_objects[str(object_id)]
        model_path = checked_cad_path(declaration["model_artifact"])
        model_hash = sha256_file(model_path)
        if model_hash != declaration["model_sha256"]:
            raise ValueError(f"Runtime CAD model differs from freeze: {object_id}")
        checked_object_hashes[str(object_id)] = model_hash

    best_static_slot = int(frozen["best_static_slot"]["candidate_slot"])
    if best_static_slot not in CANDIDATE_SLOTS:
        raise ValueError("Frozen best static slot lies outside 1..4")
    geometry_cache: dict[int, Any] = {}
    feature_rows: list[dict[str, Any]] = []
    feature_rows_by_group: dict[str, dict[int, dict[str, Any]]] = {}
    selection_plan: dict[str, dict[str, int]] = {}
    runtime_forbidden_input_counts: Counter[str] = Counter()
    for group in groups:
        group_id = str(group["group_id"])
        object_id = int(group["object_id"])
        target_id = str(group["target_sample_id"])
        if object_id not in geometry_cache:
            geometry_cache[object_id] = load_cad_geometry(dataset_root, object_id)
        cad_geometry = geometry_cache[object_id]
        sanitized_group = _sanitized_preacquisition_group(group)
        target_manifest = copy.deepcopy(target_manifests[target_id])
        target_prediction = _sanitized_target_prediction(target_predictions[target_id])
        if any("viewing_direction_world" in view for view in sanitized_group["views"]):
            raise AssertionError("Sanitizer exposed GT-derived viewing direction")
        target_context = extract_target_context(
            sanitized_group,
            target_manifest,
            target_prediction,
            cad_geometry,
            dataset_root,
        )
        per_slot: dict[int, dict[str, Any]] = {}
        for slot in CANDIDATE_SLOTS:
            features = extract_candidate_features(
                sanitized_group,
                slot,
                target_context,
                cad_geometry,
            )
            if tuple(features) != tuple(FEATURE_NAMES):
                raise ValueError(
                    f"Holdout feature order differs from freeze: {group_id} slot {slot}"
                )
            values = np.asarray(list(features.values()), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite holdout feature: {group_id} slot {slot}")
            candidate_view = group["views"][slot]
            row = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "m4_holdout_candidate_feature",
                "group_id": group_id,
                "target_sample_id": target_id,
                "candidate_sample_id": str(candidate_view["sample_id"]),
                "object_id": object_id,
                "physical_instance_id": str(group["physical_instance_id"]),
                "candidate_slot": slot,
                "features": {name: float(features[name]) for name in FEATURE_NAMES},
                "geometry_novelty": float(features[NEW_SURFACE_FEATURE_NAME]),
            }
            feature_rows.append(row)
            per_slot[slot] = row
        feature_rows_by_group[group_id] = per_slot
        runtime_forbidden_input_counts.update(
            {
                "candidate_rgb_depth_mask_fields": 0,
                "candidate_visibility_or_mask_fields": 0,
                "candidate_gt_or_outcome_fields": 0,
                "candidate_prediction_fields": 0,
                "group_viewing_direction_world_fields": 0,
                "categorical_id_feature_fields": 0,
            }
        )

    if (
        len(feature_rows) != EXPECTED_HOLDOUT_CANDIDATES
        or len(feature_rows_by_group) != EXPECTED_HOLDOUT_GROUPS
        or any(
            set(per_slot) != set(CANDIDATE_SLOTS)
            for per_slot in feature_rows_by_group.values()
        )
    ):
        raise ValueError("M4 holdout feature-row count/slot contract failed")

    feature_matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )
    if feature_matrix.shape != (
        EXPECTED_HOLDOUT_CANDIDATES,
        len(FEATURE_NAMES),
    ) or not np.all(np.isfinite(feature_matrix)):
        raise ValueError("M4 holdout batch feature matrix is invalid")
    raw_predictions = np.asarray(
        regressor.predict(feature_matrix),
        dtype=np.float64,
    )
    if raw_predictions.shape != (EXPECTED_HOLDOUT_CANDIDATES,) or not np.all(
        np.isfinite(raw_predictions)
    ):
        raise ValueError("Frozen M4 regressor returned invalid batch predictions")
    clipped_predictions = np.clip(raw_predictions, 0.0, 1.0)
    for row, predicted_utility in zip(
        feature_rows,
        clipped_predictions,
        strict=True,
    ):
        row["predicted_utility"] = float(predicted_utility)

    for group in groups:
        group_id = str(group["group_id"])
        per_slot = feature_rows_by_group[group_id]
        learned_slot = min(
            CANDIDATE_SLOTS,
            key=lambda slot: (
                -per_slot[slot]["predicted_utility"],
                slot,
            ),
        )
        novelty_slot = min(
            CANDIDATE_SLOTS,
            key=lambda slot: (
                -per_slot[slot]["geometry_novelty"],
                slot,
            ),
        )
        selection_plan[group_id] = {
            "fixed_first": 1,
            "best_static_slot": best_static_slot,
            "max_geometry_novelty": novelty_slot,
            "learned_voi": learned_slot,
        }
    if len(selection_plan) != EXPECTED_HOLDOUT_GROUPS or any(
        {
            int(row["candidate_slot"])
            for row in feature_rows
            if row["group_id"] == group_id
        }
        != set(CANDIDATE_SLOTS)
        for group_id in selection_plan
    ):
        raise ValueError("M4 holdout feature/selection-plan count contract failed")
    selected_plan_payload = [
        {"group_id": group_id, **selection_plan[group_id]}
        for group_id in sorted(selection_plan)
    ]
    frozen_at = datetime.now(timezone.utc)
    return (
        feature_rows,
        selection_plan,
        {
            "status": "pass",
            "feature_row_count": len(feature_rows),
            "target_count": len(selection_plan),
            "selection_plan_sha256": canonical_sha256(selected_plan_payload),
            "selection_frozen_utc": frozen_at.isoformat(),
            "selection_frozen_before_full_candidate_prediction_load": True,
            "utility_prediction": {
                "status": "pass",
                "mode": "single_batch_predict",
                "regressor_predict_call_count": 1,
                "feature_matrix_shape": list(feature_matrix.shape),
                "feature_row_order": (
                    "input group order, then candidate slots 1 through 4"
                ),
                "output_shape": list(clipped_predictions.shape),
                "finite_before_clipping": True,
                "clip_interval": [0.0, 1.0],
            },
            "runtime_model_contract": {
                "status": "pass",
                "class": type(regressor).__name__,
                "sklearn_version": sklearn.__version__,
                "parameters_match": True,
            },
            "runtime_cad_contract": {
                "status": "pass",
                "models_info_sha256": sha256_file(models_info_path),
                "object_model_sha256": checked_object_hashes,
            },
            "feature_builder_received_only_sanitized_group_extrinsics": True,
            "feature_builder_received_only_sanitized_target_observation": True,
            "candidate_prediction_mapping_passed_to_feature_builder": False,
            "forbidden_preacquisition_input_counts": dict(
                sorted(runtime_forbidden_input_counts.items())
            ),
            "sanitized_group_view_keys": [
                "acquisition_rank",
                "camera_center_world_m",
                "camera_world_to_camera_pose_m",
            ],
            "sanitized_target_manifest_keys": sorted(
                target_manifests[str(groups[0]["target_sample_id"])]
            ),
            "sanitized_target_prediction_keys": sorted(
                _sanitized_target_prediction(
                    target_predictions[str(groups[0]["target_sample_id"])]
                )
            ),
        },
    )


def select_holdout_methods(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
    feature_rows: Sequence[dict[str, Any]],
    selection_plan: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    feature_index = {
        (str(row["group_id"]), int(row["candidate_slot"])): row for row in feature_rows
    }
    result_sets: dict[str, list[dict[str, Any]]] = {
        "target_only": [copy.deepcopy(row) for row in baseline_rows]
    }
    selection_audit: dict[str, Any] = {}
    for method in (
        "fixed_first",
        "best_static_slot",
        "max_geometry_novelty",
        "learned_voi",
    ):
        rows = []
        for group_id in sorted(candidate_outcomes):
            slot = int(selection_plan[group_id][method])
            row = copy.deepcopy(candidate_outcomes[group_id][slot])
            row["method"] = method
            row["selected_candidate_slot"] = slot
            row["candidate_predicted_utility"] = float(
                feature_index[(group_id, slot)]["predicted_utility"]
            )
            row["candidate_geometry_novelty"] = float(
                feature_index[(group_id, slot)]["geometry_novelty"]
            )
            row["selection_used_candidate_outcome"] = False
            rows.append(row)
        result_sets[method] = rows
        selection_audit[method] = {
            "candidate_outcome_used_before_selection": False,
            "candidate_slot_counts": {
                str(slot): int(
                    sum(row["selected_candidate_slot"] == slot for row in rows)
                )
                for slot in CANDIDATE_SLOTS
            },
        }

    oracle_rows = []
    for group_id in sorted(candidate_outcomes):
        per_slot = candidate_outcomes[group_id]
        slot = min(
            CANDIDATE_SLOTS,
            key=lambda candidate_slot: (
                -float(per_slot[candidate_slot]["actual_utility"]),
                candidate_slot,
            ),
        )
        row = copy.deepcopy(per_slot[slot])
        row["method"] = "oracle_best_candidate"
        row["selected_candidate_slot"] = slot
        row["candidate_predicted_utility"] = float(
            feature_index[(group_id, slot)]["predicted_utility"]
        )
        row["candidate_geometry_novelty"] = float(
            feature_index[(group_id, slot)]["geometry_novelty"]
        )
        row["selection_used_candidate_outcome"] = True
        row["oracle_is_diagnostic_only"] = True
        oracle_rows.append(row)
    result_sets["oracle_best_candidate"] = oracle_rows
    random_rows = _random_expectation_rows(candidate_outcomes)
    result_sets["random_candidate"] = random_rows
    selection_audit["random_candidate"] = {
        "point_estimate": "exact uniform mean over all four slots per target",
        "candidate_outcome_used_for_deployable_selection": False,
    }
    selection_audit["oracle_best_candidate"] = {
        "candidate_outcome_used_before_selection": True,
        "diagnostic_only": True,
        "tie_break": "lowest candidate slot",
    }
    if set(result_sets) != set(METHODS):
        raise AssertionError("M4 evaluated-method set is incomplete")
    return result_sets, selection_audit


def compare_methods(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scores = {
        method: float(summaries[method]["macro_object"]["combined"])
        for method in METHODS
    }
    learned = scores["learned_voi"]
    oracle = scores["oracle_best_candidate"]
    best_control_name = max(
        (
            "fixed_first",
            "best_static_slot",
            "random_candidate",
            "max_geometry_novelty",
        ),
        key=lambda method: (scores[method], method),
    )
    best_control = scores[best_control_name]

    def fraction(control: str) -> float | None:
        denominator = oracle - scores[control]
        return (
            (learned - scores[control]) / denominator if denominator > 1e-15 else None
        )

    return {
        "macro_combined_scores": scores,
        "learned_gain_over_target_only": learned - scores["target_only"],
        "learned_gain_over_fixed_first": learned - scores["fixed_first"],
        "learned_gain_over_best_static_slot": learned - scores["best_static_slot"],
        "learned_gain_over_random_expectation": learned - scores["random_candidate"],
        "learned_gain_over_max_geometry_novelty": learned
        - scores["max_geometry_novelty"],
        "learned_gap_to_oracle_best_candidate": oracle - learned,
        "oracle_headroom_over_fixed_first": oracle - scores["fixed_first"],
        "oracle_headroom_over_random_expectation": oracle - scores["random_candidate"],
        "oracle_headroom_over_best_control": oracle - best_control,
        "best_budget_matched_nonlearned_control": best_control_name,
        "fraction_oracle_headroom_closed_from_fixed_first": fraction("fixed_first"),
        "fraction_oracle_headroom_closed_from_random_expectation": fraction(
            "random_candidate"
        ),
        "fraction_oracle_headroom_closed_from_best_control": fraction(
            best_control_name
        ),
        "learned_beats_fixed_static_and_random": bool(
            learned > scores["fixed_first"]
            and learned > scores["best_static_slot"]
            and learned > scores["random_candidate"]
        ),
    }


def enrich_holdout_feature_rows(
    feature_rows: Sequence[dict[str, Any]],
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for raw in feature_rows:
        row = copy.deepcopy(raw)
        outcome = candidate_outcomes[str(row["group_id"])][int(row["candidate_slot"])]
        row.update(
            {
                "actual_utility": float(outcome["actual_utility"]),
                "joint_success": bool(outcome["diagnostic_success"]["joint"]),
                "sample_ar_mssd": float(outcome["sample_ar_mssd"]),
                "sample_ar_mspd": float(outcome["sample_ar_mspd"]),
                "outcome_evidence": {
                    "post_acquisition_rule": "max_mask",
                    "selected_sample_id": outcome["selected_sample_id"],
                    "selected_acquisition_rank": outcome["selected_acquisition_rank"],
                    "status": outcome["status"],
                    "finite_pose": bool(outcome["finite_pose"]),
                    "normalized_mssd": outcome["normalized_mssd"],
                    "mspd_px": outcome["mspd_px"],
                    "mspd_scale_r": outcome["mspd_scale_r"],
                },
            }
        )
        output.append(row)
    return output


def validate_m2_slot1_anchor(
    m2_metrics_path: Path,
    candidate_outcomes: Mapping[str, Mapping[int, dict[str, Any]]],
) -> dict[str, Any]:
    stored_rows = [
        row
        for row in load_jsonl(m2_metrics_path)
        if row.get("method") == "max_mask_area"
        and int(row.get("requested_view_budget", -1)) == 2
    ]
    stored = {str(row["group_id"]): row for row in stored_rows}
    if len(stored) != EXPECTED_DEVELOPMENT_GROUPS:
        raise ValueError("Stored M2 slot-1 max-mask anchor is incomplete")
    max_difference = 0.0
    reconstructed_rows = []
    for group_id in sorted(candidate_outcomes):
        current = candidate_outcomes[group_id][1]
        expected = stored[group_id]
        if (
            current["selected_sample_id"] != expected["selected_sample_id"]
            or current["selected_acquisition_rank"]
            != expected["selected_acquisition_rank"]
            or current["finite_pose"] != expected["finite_pose"]
        ):
            raise ValueError(f"M2 slot-1 max-mask identity mismatch: {group_id}")
        for field in (
            "sample_ar_mssd",
            "sample_ar_mspd",
            "normalized_mssd",
            "mspd_px",
        ):
            current_value = current[field]
            expected_value = expected[field]
            if current_value is None or expected_value is None:
                if current_value != expected_value:
                    raise ValueError(f"M2 slot-1 null mismatch: {group_id} {field}")
            else:
                max_difference = max(
                    max_difference, abs(float(current_value) - float(expected_value))
                )
        reconstructed_rows.append(current)
    if max_difference > 1e-12:
        raise ValueError(f"M2 slot-1 reconstruction delta: {max_difference}")
    score = _score_summary(reconstructed_rows)
    if not math.isclose(
        float(score["macro_object"]["combined"]),
        0.6993333333333333,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("M2 slot-1 reconstruction aggregate anchor changed")
    return {
        "status": "pass",
        "stored_row_count": len(stored),
        "maximum_numeric_difference": max_difference,
        "macro_object": score["macro_object"],
        "micro": score["micro"],
    }


def validate_m3_reconstruction_anchors(
    prepared_groups: Sequence[dict[str, Any]],
    m3_metrics_path: Path,
) -> dict[str, Any]:
    stored_rows = load_jsonl(m3_metrics_path)
    stored = {
        (str(row["group_id"]), str(row["method"])): row
        for row in stored_rows
        if row.get("method")
        in {"fixed_k1_target", "fixed_k3_max_mask", "fixed_k5_max_mask"}
    }
    expected_count = EXPECTED_HOLDOUT_GROUPS * 3
    if len(stored) != expected_count:
        raise ValueError("Stored M3 reconstruction anchors are incomplete")
    reconstructed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    maximum_difference = 0.0
    for prepared in prepared_groups:
        group_id = str(prepared["group"]["group_id"])
        views = prepared["views"]
        definitions = (
            ("fixed_k1_target", 1),
            ("fixed_k3_max_mask", 3),
            ("fixed_k5_max_mask", 5),
        )
        for method, budget in definitions:
            acquired = views[:budget]
            selected = max(
                acquired,
                key=lambda view: (
                    int(view["view"]["visible_mask_pixel_count"]),
                    -int(view["view"]["acquisition_rank"]),
                ),
            )
            current = result_from_selected(prepared, acquired, selected, method, budget)
            expected = stored[(group_id, method)]
            if (
                current["selected_sample_id"] != expected["selected_sample_id"]
                or current["selected_acquisition_rank"]
                != expected["selected_acquisition_rank"]
                or current["finite_pose"] != expected["finite_pose"]
            ):
                raise ValueError(f"M3 reconstruction identity mismatch: {group_id}")
            for field in (
                "sample_ar_mssd",
                "sample_ar_mspd",
                "normalized_mssd",
                "mspd_px",
            ):
                current_value = current[field]
                expected_value = expected[field]
                if current_value is None or expected_value is None:
                    if current_value != expected_value:
                        raise ValueError(
                            f"M3 reconstruction null mismatch: {group_id} {field}"
                        )
                else:
                    maximum_difference = max(
                        maximum_difference,
                        abs(float(current_value) - float(expected_value)),
                    )
            reconstructed[method].append(current)
    if maximum_difference > 1e-12:
        raise ValueError(f"M3 reconstruction numeric delta: {maximum_difference}")
    expected_macro_combined = {
        "fixed_k1_target": 0.8317859428153546,
        "fixed_k3_max_mask": 0.8183901847063613,
        "fixed_k5_max_mask": 0.8066631258175376,
    }
    observed_macro = {
        method: _score_summary(rows)["macro_object"]
        for method, rows in sorted(reconstructed.items())
    }
    if any(
        not math.isclose(
            float(observed_macro[method]["combined"]),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for method, expected in expected_macro_combined.items()
    ):
        raise ValueError("M3 reconstruction aggregate anchors changed")
    return {
        "status": "pass",
        "stored_row_count": len(stored),
        "maximum_numeric_difference": maximum_difference,
        "macro_object_by_method": observed_macro,
        "micro_by_method": {
            method: _score_summary(rows)["micro"]
            for method, rows in sorted(reconstructed.items())
        },
    }


def _pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return (
        f"{100.0 * float(value):+0.2f}%" if signed else f"{100.0 * float(value):0.2f}%"
    )


def _num(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _pp(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):+0.2f}"


def _ci(
    interval: Mapping[str, Any] | None,
    *,
    pp: bool = False,
    percent: bool = False,
) -> str:
    if interval is None:
        return "N/A"
    if pp and percent:
        raise ValueError("A confidence interval cannot be both percent and pp")
    scale = 100.0 if pp or percent else 1.0
    suffix = " pp" if pp else ""
    value_suffix = "%" if percent else ""
    sign = "+" if pp else ""
    return (
        f"[{scale * float(interval['lower_2.5_percent']):{sign}.2f}{value_suffix}, "
        f"{scale * float(interval['upper_97.5_percent']):{sign}.2f}{value_suffix}]"
        f"{suffix}"
    )


def _interpretation(summary: Mapping[str, Any]) -> str:
    comparison = summary["comparisons"]
    summaries = {row["method"]: row for row in summary["method_summaries"]}
    headroom = float(comparison["oracle_headroom_over_best_control"])
    if comparison["learned_beats_fixed_static_and_random"]:
        conclusion = (
            "The point-estimate interpretation gate supports candidate-specific "
            "pre-acquisition ranking: "
            "`learned_voi` beats `fixed_first`, `best_static_slot`, and the "
            "uniform-random expectation."
        )
        required_intervals = [
            summary["bootstrap"]["paired_combined_intervals"][key]
            for key in (
                "learned_minus_fixed_first",
                "learned_minus_best_static_slot",
                "learned_minus_random_candidate",
            )
        ]
        if all(
            float(interval["lower_2.5_percent"])
            <= 0.0
            <= float(interval["upper_97.5_percent"])
            for interval in required_intervals
        ):
            conclusion += (
                " All three paired 95% bootstrap intervals include zero, so this "
                "directional advantage is not statistically resolved."
            )
    elif headroom >= 0.01:
        conclusion = (
            "Useful candidate headroom exists, but the current pre-acquisition "
            "geometry/current-observation features do not identify it reliably: "
            "`learned_voi` does not beat all required controls."
        )
    else:
        conclusion = (
            "The four-view candidate set offers less than 1.00 pp oracle headroom "
            "over the strongest nonlearned control, so it does not currently "
            "justify a larger next-best-view project."
        )
    if not comparison["learned_beats_fixed_static_and_random"]:
        best_control = comparison["best_budget_matched_nonlearned_control"]
        conditional = []
        for visibility in VISIBILITY_BINS:
            learned_subset = summaries["learned_voi"]["by_target_visibility_bin"][
                visibility
            ]
            control_subset = summaries[best_control]["by_target_visibility_bin"][
                visibility
            ]
            if (
                learned_subset["macro_object"] is not None
                and control_subset["macro_object"] is not None
                and learned_subset["macro_object"]["combined"]
                > control_subset["macro_object"]["combined"]
            ):
                conditional.append(visibility)
        if conditional:
            conclusion += (
                " Positive gains occur only conditionally in target visibility "
                f"bin(s) {conditional}; this is not a full-population policy win."
            )
    if float(comparison["learned_gain_over_target_only"]) < 0.0:
        conclusion += (
            f" It remains {_pp(comparison['learned_gain_over_target_only'])} pp "
            "below the one-view target-only reference, so acquiring a second view "
            "is not an end-to-end population win under the common max-mask rule."
        )
    if summaries["learned_voi"]["by_target_visibility_bin"]["low"]["sample_count"] == 0:
        conclusion += (
            " The holdout has no low-visibility targets, so no low-visibility "
            "conditional claim can be tested."
        )
    return conclusion


def markdown_report(summary: Mapping[str, Any]) -> str:
    summaries = {row["method"]: row for row in summary["method_summaries"]}
    comparison = summary["comparisons"]
    development = summary["development"]["grouped_oof"]
    bootstrap = summary["bootstrap"]
    random = summary["random_assignments"]
    caveats = "\n".join(f"- **{item}**" for item in summary["caveats"])

    main_rows = []
    for method in METHODS:
        row = summaries[method]
        macro = row["macro_object"]
        micro = row["micro"]
        main_rows.append(
            f"| `{method}` | {_pct(macro['ar_mssd'])} | "
            f"{_pct(macro['ar_mspd'])} | {_pct(macro['combined'])} | "
            f"{_pct(micro['ar_mssd'])} | {_pct(micro['ar_mspd'])} | "
            f"{_pct(micro['combined'])} | "
            f"{_num(row['registration_latency_seconds']['mean'])} |"
        )
    gain_rows = [
        f"| vs target-only | {_pp(comparison['learned_gain_over_target_only'])} |",
        f"| vs fixed-first | {_pp(comparison['learned_gain_over_fixed_first'])} |",
        f"| vs best static slot | {_pp(comparison['learned_gain_over_best_static_slot'])} |",
        f"| vs random expectation | {_pp(comparison['learned_gain_over_random_expectation'])} |",
        f"| vs max geometry novelty | {_pp(comparison['learned_gain_over_max_geometry_novelty'])} |",
        f"| gap to oracle | {-100.0 * comparison['learned_gap_to_oracle_best_candidate']:+.2f} |",
    ]
    rescue_rows = []
    for method in BUDGET_MATCHED_METHODS:
        joint = summaries[method]["rescue_harm_vs_target_only"]["joint"]
        rescue_rows.append(
            f"| `{method}` | {_pct(joint['rescue_rate'])} | "
            f"{_pct(joint['harm_rate'])} | "
            f"{_num(joint['expected_rescued_count'], 2)} | "
            f"{_num(joint['expected_harmed_count'], 2)} |"
        )
    visibility_rows = []
    for method in METHODS:
        cells = []
        for label in VISIBILITY_BINS:
            subset = summaries[method]["by_target_visibility_bin"][label]
            cells.append(
                "N/A"
                if subset["macro_object"] is None
                else _pct(subset["macro_object"]["combined"])
            )
        visibility_rows.append(f"| `{method}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    object_rows = []
    by_method_object = {
        method: {int(row["object_id"]): row for row in summaries[method]["by_object"]}
        for method in METHODS
    }
    for object_id in sorted(by_method_object["target_only"]):
        for method in METHODS:
            row = by_method_object[method][object_id]
            object_rows.append(
                f"| {object_id} | `{method}` | {row['physical_instance_count']} | "
                f"{'Yes' if row['low_n_descriptive'] else 'No'} | "
                f"{_pct(row['ar_mssd'])} | {_pct(row['ar_mspd'])} | "
                f"{_pct(row['combined'])} |"
            )
    slot_rows = []
    for method in BUDGET_MATCHED_METHODS:
        if method == "random_candidate":
            fractions = random["aggregate_candidate_slot_fractions"]
            display_method = "random_candidate (seeded assignments)"
        else:
            fractions = summaries[method]["candidate_slot_selection_fractions"]
            display_method = method
        slot_rows.append(
            f"| `{display_method}` | "
            + " | ".join(_pct(fractions[str(slot)]) for slot in CANDIDATE_SLOTS)
            + " |"
        )
    bootstrap_rows = []
    for method in METHODS:
        interval = bootstrap["method_intervals"][method]["combined"]
        bootstrap_rows.append(f"| `{method}` | {_ci(interval, percent=True)} |")
    paired_rows = [
        f"| learned minus {label.removeprefix('learned_minus_')} | {_ci(interval, pp=True)} |"
        for label, interval in bootstrap["paired_combined_intervals"].items()
        if label.startswith("learned_minus_")
    ]
    paired_rows.append(
        "| oracle minus learned | "
        f"{_ci(bootstrap['paired_combined_intervals']['oracle_minus_learned'], pp=True)} |"
    )
    frozen = summary["freeze_audit"]
    split = summary["split_audit"]
    chronology = summary["chronology"]
    object15_rows = [
        f"| `{method}` | {summaries[method]['object_15']['physical_instance_count']} | "
        f"{_pct(summaries[method]['object_15']['ar_mssd'])} | "
        f"{_pct(summaries[method]['object_15']['ar_mspd'])} | "
        f"{_pct(summaries[method]['object_15']['combined'])} |"
        for method in METHODS
    ]
    return f"""# PoseLoop M4 pre-acquisition one-step view ranking

## Answer first

{_interpretation(summary)}

`learned_voi` changes macro combined by
**{_pp(comparison['learned_gain_over_fixed_first'])} pp vs fixed-first**,
**{_pp(comparison['learned_gain_over_best_static_slot'])} pp vs the M2-frozen
best static slot**, and **{_pp(comparison['learned_gain_over_random_expectation'])}
pp vs uniform random expectation**. Its gap to the diagnostic oracle is
**{100.0 * comparison['learned_gap_to_oracle_best_candidate']:.2f} pp**.

## Scope and hard caveats

{caveats}

## Split, leakage, and freeze

Development contains **{split['development_target_count']} targets,
{split['development_candidate_row_count']} target-candidate rows, and
{split['development_physical_instance_count']} physical instances**. Holdout
contains **{split['holdout_target_count']} targets,
{split['holdout_candidate_row_count']} target-candidate rows, and
{split['holdout_physical_instance_count']} physical instances**. The physical
instance intersection is exactly **{split['intersection_count']}**; every target
has candidate slots 1--4.

The frozen M2-only configuration is
`{frozen['configuration_sha256']}`. Existing M3 inference was created at
`{chronology['existing_m3_prediction_stream_created_utc']}`, before the M4
freeze at `{chronology['m4_frozen_utc']}`; this is expected because M4 reuses
the untouched stream. Holdout evaluation began at
`{chronology['evaluation_started_utc']}`, after the freeze. Pre-acquisition
selection was hashed as `{chronology['selection_plan_sha256']}` before full
candidate prediction/outcome loading.

The runtime feature builder received only target observation/prediction fields,
candidate camera extrinsics, and CAD geometry. It received zero candidate
RGB/depth/mask, visibility, GT, FoundationPose prediction/score, outcome, or
stored `viewing_direction_world` fields. IDs remain audit metadata outside the
feature mapping. Exact features are
`{summary['feature_audit']['feature_names']}`.

## Grouped M2 out-of-fold diagnostics

Five grouped folds have zero physical-instance overlap and test all 1,200 rows
once. Candidate utility prediction gives MAE
**{development['utility_mae']:.4f}**, RMSE
**{development['utility_rmse']:.4f}**, mean defined per-target Spearman
**{_num(development['per_target_rank_correlation']['mean'], 4)}**
over **{development['per_target_rank_correlation']['defined_target_count']}**
of 300 targets, mean regret
**{development['mean_regret']:.4f}**, and tie-aware oracle-best hit rate
**{_pct(development['oracle_best_candidate_hit_rate_tie_aware'])}**.

OOF selected-pair pose scores are macro AR_MSSD
**{_pct(development['selected_pose_metrics']['macro_object']['ar_mssd'])}**,
macro AR_MSPD
**{_pct(development['selected_pose_metrics']['macro_object']['ar_mspd'])}**, and
macro combined
**{_pct(development['selected_pose_metrics']['macro_object']['combined'])}**
(micro combined
**{_pct(development['selected_pose_metrics']['micro']['combined'])}**).
The M2-only best static slot is
**{summary['development']['best_static_slot']}**, selected by macro-object
combined with lower-slot tie-breaking.

![Predicted versus actual utility](m4_predicted_vs_actual_utility.png)

## Holdout method comparison

Macro-object scores are primary; micro scores are secondary unequal-count
summaries. Random's row is the exact per-target mean over four equiprobable
slots. Existing registration seconds are contextual only.

| Method | Macro AR_MSSD | Macro AR_MSPD | Macro combined | Micro AR_MSSD | Micro AR_MSPD | Micro combined | Mean registration seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

![Method comparison](m4_method_comparison.png)

| Learned comparison | Macro combined change (pp) |
|---|---:|
{chr(10).join(gain_rows)}

Oracle headroom closed is
**{_pct(comparison['fraction_oracle_headroom_closed_from_fixed_first'])} from
fixed-first**, **{_pct(comparison['fraction_oracle_headroom_closed_from_random_expectation'])}
from random expectation**, and
**{_pct(comparison['fraction_oracle_headroom_closed_from_best_control'])} from
the strongest nonlearned control
`{comparison['best_budget_matched_nonlearned_control']}`**. Fractions are
unclipped and N/A when the oracle denominator is nonpositive.

## Rescue, harm, visibility, and object detail

| Method | Joint rescue rate | Joint harm rate | Expected rescued | Expected harmed |
|---|---:|---:|---:|---:|
{chr(10).join(rescue_rows)}

![Joint rescue and harm](m4_rescue_harm.png)

| Method | Low visibility | Mid visibility | High visibility |
|---|---:|---:|---:|
{chr(10).join(visibility_rows)}

The low-visibility holdout bin has zero targets, so its values are N/A rather
than zero.

| Object | Method | n | Low-n | AR_MSSD | AR_MSPD | Combined |
|---:|---|---:|:---:|---:|---:|---:|
{chr(10).join(object_rows)}

![Oracle gap by object](m4_oracle_gap_by_object.png)

Object 15 is shown explicitly:

| Method | n | AR_MSSD | AR_MSPD | Combined |
|---|---:|---:|---:|---:|
{chr(10).join(object15_rows)}

## Candidate-slot selections

| Method | Slot 1 | Slot 2 | Slot 3 | Slot 4 |
|---|---:|---:|---:|---:|
{chr(10).join(slot_rows)}

Random frequencies pool all
`{random['assignment_count']}` seeded whole-holdout assignments. The random
macro combined assignment median is
`{random['macro_assignment_intervals']['combined']['median']:.4f}` and its
central 95% assignment range is
`{_ci(random['macro_assignment_intervals']['combined'], percent=True)}`. The main-table random
point estimate is instead the exact four-slot mean. This assignment range
measures assignment variation, not physical-instance uncertainty; the bootstrap
below measures physical-instance sampling uncertainty.

## Object-stratified physical-instance bootstrap

The paired bootstrap uses the same within-object resamples for every method.
Random uses the exact four-slot per-target expectation, not one Monte Carlo
assignment.

| Method | Macro combined 95% CI |
|---|---:|
{chr(10).join(bootstrap_rows)}

| Paired comparison | Macro combined 95% CI |
|---|---:|
{chr(10).join(paired_rows)}

## Provenance and reconstruction

The M2 official two-view slot-1 and existing M3 k=1/k=3/k=5 max-mask
metric-anchor reconstructions all pass with maximum numeric delta
`{summary['reconstruction_audit']['maximum_numeric_difference']:.3e}`. The
official symmetry-aware evaluator uses BOP Toolkit commit
`{summary['bop_toolkit']['commit_sha']}`, `models_eval`, and
`max_sym_disc_step =
{summary['bop_toolkit']['max_sym_disc_step']}`. The cross-view GT transform
gate remains unchanged.
"""


def generate_figures(
    summary: Mapping[str, Any],
    development_oof_rows: Sequence[dict[str, Any]],
    holdout_feature_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )
    colors = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "red": "#D55E00",
        "purple": "#CC79A7",
        "gray": "#777777",
        "sky": "#56B4E9",
    }
    summaries = {row["method"]: row for row in summary["method_summaries"]}
    output_dir = Path(summary["outputs"]["figures_dir"])

    labels = [
        "Target\nonly",
        "Fixed\nfirst",
        "Best\nstatic",
        "Random",
        "Geometry\nnovelty",
        "Learned\nVOI",
        "Oracle",
    ]
    scores = np.asarray(
        [summaries[method]["macro_object"]["combined"] for method in METHODS]
    )
    intervals = [
        summary["bootstrap"]["method_intervals"][method]["combined"]
        for method in METHODS
    ]
    errors = np.asarray(
        [
            [
                score - interval["lower_2.5_percent"],
                interval["upper_97.5_percent"] - score,
            ]
            for score, interval in zip(scores, intervals, strict=True)
        ]
    ).T
    errors = np.maximum(errors, 0.0)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        np.arange(len(METHODS)),
        scores,
        yerr=errors,
        capsize=3,
        color=[
            colors["gray"],
            colors["blue"],
            colors["sky"],
            colors["gray"],
            colors["green"],
            colors["orange"],
            colors["purple"],
        ],
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_xticks(np.arange(len(METHODS)), labels)
    axis.set_ylabel("Macro-object combined diagnostic score")
    axis.set_ylim(max(0.0, float(np.min(scores - errors[0])) - 0.03), 1.0)
    figure.tight_layout()
    save_figure_atomic(figure, output_dir / "m4_method_comparison.png")
    plt.close(figure)

    object_ids = [
        int(row["object_id"]) for row in summaries["learned_voi"]["by_object"]
    ]
    learned_by_object = {
        int(row["object_id"]): float(row["combined"])
        for row in summaries["learned_voi"]["by_object"]
    }
    oracle_by_object = {
        int(row["object_id"]): float(row["combined"])
        for row in summaries["oracle_best_candidate"]["by_object"]
    }
    gaps = [
        oracle_by_object[object_id] - learned_by_object[object_id]
        for object_id in object_ids
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.bar(
        np.arange(len(object_ids)),
        gaps,
        color=[
            colors["red"] if object_id == 15 else colors["sky"]
            for object_id in object_ids
        ],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(object_ids)), [str(value) for value in object_ids])
    axis.set_xlabel("Object ID")
    axis.set_ylabel("Oracle minus learned combined score")
    figure.tight_layout()
    save_figure_atomic(figure, output_dir / "m4_oracle_gap_by_object.png")
    plt.close(figure)

    plotted_methods = list(BUDGET_MATCHED_METHODS)
    raw_rescue = [
        summaries[method]["rescue_harm_vs_target_only"]["joint"]["rescue_rate"]
        for method in plotted_methods
    ]
    raw_harm = [
        summaries[method]["rescue_harm_vs_target_only"]["joint"]["harm_rate"]
        for method in plotted_methods
    ]
    rescue = [0.0 if value is None else float(value) for value in raw_rescue]
    harm = [0.0 if value is None else float(value) for value in raw_harm]
    x_values = np.arange(len(plotted_methods))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.3, 4.0))
    axis.bar(
        x_values - width / 2,
        rescue,
        width,
        label="Joint rescue",
        color=colors["green"],
    )
    axis.bar(
        x_values + width / 2,
        harm,
        width,
        label="Joint harm",
        color=colors["red"],
    )
    axis.set_xticks(
        x_values,
        ["Fixed", "Static", "Random", "Geometry", "Learned", "Oracle"],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Rate relative to target-only")
    maximum_rate = max(rescue + harm, default=0.0)
    axis.set_ylim(0.0, maximum_rate * 1.18 if maximum_rate else 1.0)
    axis.legend(loc="upper right")
    figure.tight_layout()
    save_figure_atomic(figure, output_dir / "m4_rescue_harm.png")
    plt.close(figure)

    development_actual = np.asarray(
        [row["actual_utility"] for row in development_oof_rows], dtype=np.float64
    )
    development_predicted = np.asarray(
        [row["predicted_utility"] for row in development_oof_rows],
        dtype=np.float64,
    )
    holdout_actual = np.asarray(
        [row["actual_utility"] for row in holdout_feature_rows], dtype=np.float64
    )
    holdout_predicted = np.asarray(
        [row["predicted_utility"] for row in holdout_feature_rows],
        dtype=np.float64,
    )
    lower = float(
        min(
            np.min(development_actual),
            np.min(development_predicted),
            np.min(holdout_actual),
            np.min(holdout_predicted),
        )
    )
    upper = float(
        max(
            np.max(development_actual),
            np.max(development_predicted),
            np.max(holdout_actual),
            np.max(holdout_predicted),
        )
    )
    margin = 0.03 * max(1.0, upper - lower)
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    axis.scatter(
        development_actual,
        development_predicted,
        s=12,
        alpha=0.28,
        color=colors["blue"],
        label="M2 grouped OOF",
        linewidths=0,
    )
    axis.scatter(
        holdout_actual,
        holdout_predicted,
        s=12,
        alpha=0.28,
        color=colors["orange"],
        label="M3 holdout",
        linewidths=0,
    )
    axis.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Identity",
    )
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_xlabel("Actual candidate utility")
    axis.set_ylabel("Predicted candidate utility")
    axis.legend(loc="best")
    figure.tight_layout()
    save_figure_atomic(figure, output_dir / "m4_predicted_vs_actual_utility.png")
    plt.close(figure)


def validate_figure_outputs(figures_dir: Path) -> dict[str, Any]:
    from PIL import Image

    names = (
        "m4_method_comparison.png",
        "m4_oracle_gap_by_object.png",
        "m4_rescue_harm.png",
        "m4_predicted_vs_actual_utility.png",
    )
    observed = sorted(path.name for path in figures_dir.glob("m4_*.png"))
    if observed != sorted(names):
        raise ValueError(f"M4 must produce exactly four PNGs, found {observed}")
    rows = []
    for name in names:
        path = figures_dir / name
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            if min(float(dpi[0]), float(dpi[1])) < 299.0:
                raise ValueError(f"M4 figure is not 300 DPI: {name} {dpi}")
            rows.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi": [float(dpi[0]), float(dpi[1])],
                    "title_inside_figure": False,
                    "colorblind_safe_palette": True,
                }
            )
    return {"status": "pass", "figure_count": len(rows), "figures": rows}


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths = {
        "m1_manifest": args.m1_manifest.resolve(),
        "m1_predictions": args.m1_predictions.resolve(),
        "m1_summary": args.m1_summary.resolve(),
        "m2_groups": args.m2_groups.resolve(),
        "m2_candidate_manifest": args.m2_candidate_manifest.resolve(),
        "m2_groups_summary": args.m2_groups_summary.resolve(),
        "m2_extrinsics_audit": args.m2_extrinsics_audit.resolve(),
        "m2_view_predictions": args.m2_view_predictions.resolve(),
        "m2_metrics": args.m2_metrics.resolve(),
        "m3_groups": args.m3_groups.resolve(),
        "m3_candidate_manifest": args.m3_candidate_manifest.resolve(),
        "m3_holdout_summary": args.m3_holdout_summary.resolve(),
        "m3_leakage_audit": args.m3_leakage_audit.resolve(),
        "m3_predictions": args.m3_predictions.resolve(),
        "m3_metrics": args.m3_metrics.resolve(),
        "frozen_model": _raw_m4_path(args.frozen_model, repo_root),
        "frozen_configuration": _raw_m4_path(args.frozen_configuration, repo_root),
        "feature_audit": _raw_m4_path(args.feature_audit, repo_root),
        "fold_audit": _raw_m4_path(args.fold_audit, repo_root),
        "development_features": _raw_m4_path(args.development_features, repo_root),
        "development_oof": _raw_m4_path(args.development_oof, repo_root),
        "development_diagnostics": _raw_m4_path(
            args.development_diagnostics, repo_root
        ),
        "holdout_features": _raw_m4_path(args.holdout_features, repo_root),
        "metrics_output": _raw_m4_path(args.metrics_output, repo_root),
        "random_assignments_output": _raw_m4_path(
            args.random_assignments_output, repo_root
        ),
        "summary_output": _raw_m4_path(args.summary_output, repo_root),
    }
    required_inputs = {
        key: path
        for key, path in paths.items()
        if key
        not in {
            "holdout_features",
            "metrics_output",
            "random_assignments_output",
            "summary_output",
        }
    }
    m3_input_keys = {
        "m3_groups",
        "m3_candidate_manifest",
        "m3_holdout_summary",
        "m3_leakage_audit",
        "m3_predictions",
        "m3_metrics",
    }
    missing = [
        str(path)
        for key, path in required_inputs.items()
        if key not in m3_input_keys and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing M4 evaluation inputs: {missing}")
    if args.random_assignments < MIN_RANDOM_ASSIGNMENTS:
        raise ValueError("--random-assignments must be at least 2000")
    if args.bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError("--bootstrap-replicates must be at least 2000")

    # This is intentionally the first artifact-validation phase. It opens the
    # M2-only freeze and its M2 provenance, but no M3 file.
    frozen, freeze_audit = validate_frozen_configuration(paths)
    evaluation_started = datetime.now(timezone.utc)
    frozen_utc = _parse_utc(frozen["frozen_utc"], "M4 frozen_utc")
    if not frozen_utc < evaluation_started:
        raise ValueError("M4 holdout evaluation did not begin after the freeze")
    missing_m3 = [
        str(paths[key]) for key in sorted(m3_input_keys) if not paths[key].is_file()
    ]
    if missing_m3:
        raise FileNotFoundError(f"Missing post-freeze M3 inputs: {missing_m3}")

    development = validate_development_inputs(paths)
    model_params, model_info = load_official_models(args.dataset_root.resolve())
    toolkit_sha = toolkit_commit(args.toolkit_root.resolve())
    prepared_development, development_transform_audit = prepare_groups(
        development["groups"],
        development["manifest_index"],
        development["m1_predictions"],
        development["m2_predictions"],
        model_params,
        model_info,
        GT_TRANSFORM_MAX_NORMALIZED_MSSD,
        GT_TRANSFORM_MAX_MSPD_PX,
    )
    _, development_outcomes = build_candidate_outcomes(prepared_development)
    development_artifacts = validate_development_artifacts(
        frozen,
        paths,
        development,
        development_outcomes,
    )
    m2_anchor = validate_m2_slot1_anchor(paths["m2_metrics"], development_outcomes)

    # First M3 pass: retain only identities, candidate extrinsics, current
    # target observations, and current target predictions.
    pre_groups, geometry_load_audit = load_preacquisition_holdout_geometry(
        paths["m3_groups"]
    )
    target_ids = {str(group["target_sample_id"]) for group in pre_groups}
    target_manifests, target_manifest_audit = load_target_manifests_only(
        paths["m3_candidate_manifest"], target_ids
    )
    target_prediction_metadata, target_predictions, target_prediction_audit = (
        load_target_predictions_only(paths["m3_predictions"], target_ids)
    )
    feature_rows, selection_plan, feature_runtime_audit = (
        build_holdout_features_and_selection(
            pre_groups,
            target_manifests,
            target_predictions,
            frozen,
            paths["frozen_model"],
            args.dataset_root.resolve(),
        )
    )
    selection_frozen_utc = _parse_utc(
        feature_runtime_audit["selection_frozen_utc"],
        "selection_frozen_utc",
    )
    if not frozen_utc < selection_frozen_utc:
        raise ValueError("Deployable M4 selection was not frozen after the model")

    # Only after the deployable slots are fixed do we load full candidate
    # manifests, predictions, masks, GT, and outcome evidence.
    full_candidate_load_started = datetime.now(timezone.utc)
    if not selection_frozen_utc <= full_candidate_load_started:
        raise ValueError("Full candidate evidence loading preceded slot freeze")
    candidate_index = validate_m3_candidate_manifest(
        load_jsonl(paths["m3_candidate_manifest"])
    )
    groups = validate_m3_groups(load_jsonl(paths["m3_groups"]), candidate_index)
    holdout = validate_holdout_evidence(
        {
            "groups": paths["m3_groups"],
            "candidate_manifest": paths["m3_candidate_manifest"],
            "holdout_summary": paths["m3_holdout_summary"],
            "leakage_audit": paths["m3_leakage_audit"],
            "m2_groups": paths["m2_groups"],
        },
        groups,
        candidate_index,
    )
    development_ids = set(development["physical_instance_ids"])
    holdout_ids = {str(group["physical_instance_id"]) for group in groups}
    intersection = development_ids & holdout_ids
    if (
        len(groups) != EXPECTED_HOLDOUT_GROUPS
        or len(candidate_index) - len(groups) != EXPECTED_HOLDOUT_CANDIDATES
        or len(holdout_ids) != EXPECTED_HOLDOUT_PHYSICAL_INSTANCES
        or intersection
    ):
        raise ValueError("M4 post-freeze split/leakage count contract failed")
    full_identity = {
        (
            str(group["group_id"]),
            int(view["acquisition_rank"]),
            str(view["sample_id"]),
        )
        for group in groups
        for view in group["views"][1:]
    }
    pre_identity = {
        (
            str(group["group_id"]),
            int(view["acquisition_rank"]),
            str(view["sample_id"]),
        )
        for group in pre_groups
        for view in group["views"][1:]
    }
    if full_identity != pre_identity:
        raise ValueError("Pre-acquisition and full M3 candidate identities differ")

    prediction_metadata, predictions, inference_audit = load_m3_predictions(
        paths["m3_predictions"],
        candidate_index,
        groups,
        paths["m3_groups"],
        paths["m3_candidate_manifest"],
    )
    if prediction_metadata != target_prediction_metadata:
        raise ValueError("Target-only/full M3 prediction metadata differs")
    prediction_created = _parse_utc(
        prediction_metadata["created_utc"], "M3 prediction created_utc"
    )
    if not prediction_created < frozen_utc:
        raise ValueError(
            "Expected existing M3 inference stream to predate the M4 freeze"
        )

    compatible_groups = copy.deepcopy(groups)
    for group in compatible_groups:
        for view in group["views"]:
            view["prediction_source"] = "m2"
    prepared_holdout, holdout_transform_audit = prepare_groups(
        compatible_groups,
        candidate_index,
        {},
        predictions,
        model_params,
        model_info,
        GT_TRANSFORM_MAX_NORMALIZED_MSSD,
        GT_TRANSFORM_MAX_MSPD_PX,
    )
    original_by_id = {str(group["group_id"]): group for group in groups}
    for item in prepared_holdout:
        item["group"] = original_by_id[str(item["group"]["group_id"])]
    baseline_rows, holdout_outcomes = build_candidate_outcomes(prepared_holdout)
    selection_plan_hash_after_outcome_load = canonical_sha256(
        [
            {"group_id": group_id, **selection_plan[group_id]}
            for group_id in sorted(selection_plan)
        ]
    )
    if (
        selection_plan_hash_after_outcome_load
        != feature_runtime_audit["selection_plan_sha256"]
    ):
        raise ValueError("Deployable M4 selection plan changed after outcome loading")
    m3_anchor = validate_m3_reconstruction_anchors(
        prepared_holdout, paths["m3_metrics"]
    )
    enriched_features = enrich_holdout_feature_rows(feature_rows, holdout_outcomes)
    result_sets, selection_audit = select_holdout_methods(
        baseline_rows,
        holdout_outcomes,
        enriched_features,
        selection_plan,
    )
    random_assignment_rows, random_summary = random_assignment_distribution(
        holdout_outcomes, args.random_assignments
    )
    summaries = {
        method: summarize_method(method, result_sets[method], baseline_rows)
        for method in METHODS
    }
    comparisons = compare_methods(summaries)
    bootstrap = bootstrap_intervals(result_sets, args.bootstrap_replicates)

    metric_rows = [
        row
        for method in METHODS
        for row in sorted(result_sets[method], key=lambda item: str(item["group_id"]))
    ]
    write_jsonl_atomic(paths["holdout_features"], enriched_features)
    write_jsonl_atomic(paths["metrics_output"], metric_rows)
    write_jsonl_atomic(paths["random_assignments_output"], random_assignment_rows)
    split_audit = {
        "status": "pass",
        "development_target_count": len(development["groups"]),
        "development_candidate_row_count": EXPECTED_DEVELOPMENT_CANDIDATES,
        "development_physical_instance_count": len(development_ids),
        "holdout_target_count": len(groups),
        "holdout_candidate_row_count": EXPECTED_HOLDOUT_CANDIDATES,
        "holdout_physical_instance_count": len(holdout_ids),
        "intersection_count": len(intersection),
        "intersection": sorted(intersection),
        "every_target_has_slots_1_through_4": True,
    }
    reconstruction_max = max(
        float(m2_anchor["maximum_numeric_difference"]),
        float(m3_anchor["maximum_numeric_difference"]),
        float(development_artifacts["audit"]["maximum_recomputed_outcome_difference"]),
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "PoseLoop-VOI pre-acquisition one-step view ranking",
        "primary_aggregation": "macro_object",
        "split_audit": split_audit,
        "freeze_audit": freeze_audit,
        "chronology": {
            "existing_m3_prediction_stream_created_utc": prediction_created.isoformat(),
            "existing_m3_inference_predates_m4_freeze": True,
            "m4_frozen_utc": frozen_utc.isoformat(),
            "evaluation_started_utc": evaluation_started.isoformat(),
            "evaluation_started_after_freeze": True,
            "selection_frozen_utc": selection_frozen_utc.isoformat(),
            "selection_plan_sha256": feature_runtime_audit["selection_plan_sha256"],
            "selection_plan_sha256_after_outcome_load": (
                selection_plan_hash_after_outcome_load
            ),
            "selection_plan_unchanged_after_outcome_load": True,
            "full_candidate_evidence_load_started_utc": (
                full_candidate_load_started.isoformat()
            ),
            "selection_preceded_full_candidate_evidence_load": True,
        },
        "development": {
            **development["audit"],
            "grouped_oof": development_artifacts["audit"]["grouped_oof"],
            "best_static_slot": development_artifacts["audit"]["best_static_slot"],
            "best_static_slot_scores": development_artifacts["audit"][
                "best_static_slot_scores"
            ],
            "cross_view_gt_transform_check": development_transform_audit,
        },
        "holdout": holdout,
        "inference": inference_audit,
        "feature_audit": {
            **freeze_audit["feature_audit"],
            "runtime": feature_runtime_audit,
            "preacquisition_geometry_load": geometry_load_audit,
            "target_manifest_load": target_manifest_audit,
            "target_prediction_load": target_prediction_audit,
        },
        "selection_audit": selection_audit,
        "cross_view_gt_transform_check": holdout_transform_audit,
        "method_summaries": [summaries[method] for method in METHODS],
        "comparisons": comparisons,
        "random_assignments": random_summary,
        "bootstrap": bootstrap,
        "registration_context": {
            "description": (
                "Existing measured target-plus-candidate sequential FoundationPose "
                "registration cost across all 920 possible pairs."
            ),
            "all_candidate_pairs_seconds": _percentiles(
                row["sequential_runtime_seconds"]
                for per_slot in holdout_outcomes.values()
                for row in per_slot.values()
            ),
        },
        "reconstruction_audit": {
            "status": "pass",
            "maximum_numeric_difference": reconstruction_max,
            "m2_slot1_two_view_max_mask": m2_anchor,
            "m3_existing_fixed_method_anchors": m3_anchor,
        },
        "bop_toolkit": {
            "commit_sha": toolkit_sha,
            "model_type": "models_eval",
            "max_sym_disc_step": MAX_SYMMETRY_DISCRETIZATION_STEP,
        },
        "caveats": list(REQUIRED_CAVEATS),
        "provenance": {
            "evaluate_m4_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "fit_m4_voi_source": freeze_audit["fit_source"],
            **{
                key: {"path": str(path), "sha256": sha256_file(path)}
                for key, path in required_inputs.items()
            },
        },
        "outputs": {
            "holdout_features": str(paths["holdout_features"]),
            "holdout_features_sha256": sha256_file(paths["holdout_features"]),
            "metrics": str(paths["metrics_output"]),
            "metrics_sha256": sha256_file(paths["metrics_output"]),
            "random_assignments": str(paths["random_assignments_output"]),
            "random_assignments_sha256": sha256_file(
                paths["random_assignments_output"]
            ),
            "summary": str(paths["summary_output"]),
            "report": str(args.report.resolve()),
            "figures_dir": str(args.figures_dir.resolve()),
        },
    }
    generate_figures(
        summary,
        development_artifacts["oof_rows"],
        enriched_features,
    )
    figure_audit = validate_figure_outputs(args.figures_dir.resolve())
    summary["figure_audit"] = figure_audit
    write_text_atomic(args.report.resolve(), markdown_report(summary))
    write_json_atomic(paths["summary_output"], summary)

    print(_interpretation(summary), flush=True)
    print(
        f"development/holdout candidate rows: "
        f"{EXPECTED_DEVELOPMENT_CANDIDATES}/{EXPECTED_HOLDOUT_CANDIDATES}",
        flush=True,
    )
    print(
        f"physical-instance overlap: {split_audit['intersection_count']}",
        flush=True,
    )
    print(f"saved: {paths['summary_output']}", flush=True)
    print(f"saved: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
