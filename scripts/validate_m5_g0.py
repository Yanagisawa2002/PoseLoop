#!/usr/bin/env python3
"""Formally validate the complete PoseLoop M5-G0 synthetic artifact set."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_m5_g0_manifests as manifest_builder
import m5_g0_core as core
import run_m5_g0 as runner
from m1_common import canonical_sha256, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
EXPECTED_DEVELOPMENT_ROWS = 1152 * 3 * 4
EXPECTED_SEALED_ROWS = 2304 * 4
REQUIRED_PLOTS = (
    "timeline_c2_representation_switch.png",
    "timeline_continuous_axial.png",
    "timeline_dropout_10.png",
    "timeline_outlier_burst.png",
    "timeline_direction_reversal.png",
    "aggregate_quotient_error_by_symmetry.png",
    "aggregate_jump_counts.png",
    "aggregate_dropout_recovery.png",
    "aggregate_asymmetric_guardrail.png",
    "aggregate_runtime_distribution.png",
)
TEMPORAL_METHODS = (
    "STANDARD_SE3_CT",
    "NEAREST_REPRESENTATIVE_CT",
    "SYMQUOT_CT",
)
SEALED_METHODS = ("RAW_HOLD", *TEMPORAL_METHODS)


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
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            _finite_tree(value, f"{path}:{line_number}")
            rows.append(value)
    return rows


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} does not match deterministic recomputation")


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _paths(repo_root: Path) -> runner.Paths:
    return runner.Paths.create(repo_root, None, None, None)


def _condition_counts(
    rows: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            str(row["symmetry_class"]),
            str(row["motion_family"]),
            str(row["stress_family"]),
        )
        for row in rows
    )


def _validate_manifest(
    rows: Sequence[Mapping[str, Any]], split: str, expected_count: int, replicates: int
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(
            f"{split} manifest has {len(rows)} rows, expected {expected_count}"
        )
    expected_conditions = {
        (symmetry, motion, stress)
        for symmetry in manifest_builder.SYMMETRY_CLASSES
        for motion in manifest_builder.MOTION_FAMILIES
        for stress in manifest_builder.STRESS_FAMILIES
    }
    counts = _condition_counts(rows)
    if set(counts) != expected_conditions or set(counts.values()) != {replicates}:
        raise ValueError(f"{split} manifest does not have the frozen condition product")
    ids = [str(row["trajectory_id"]) for row in rows]
    trajectory_seeds = [int(row["trajectory_seed"]) for row in rows]
    corruption_seeds = [int(row["corruption_seed"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split} manifest has duplicate trajectory IDs")
    if len(set(trajectory_seeds)) != len(trajectory_seeds):
        raise ValueError(f"{split} manifest has duplicate trajectory seeds")
    if len(set(corruption_seeds)) != len(corruption_seeds):
        raise ValueError(f"{split} manifest has duplicate corruption seeds")
    for row in rows:
        if row.get("split") != split or row.get("record_type") != "m5_g0_seed":
            raise ValueError(
                f"Malformed {split} manifest row: {row.get('trajectory_id')}"
            )
    return {
        "row_count": len(rows),
        "condition_count": len(counts),
        "trajectory_id_count": len(set(ids)),
        "trajectory_seed_count": len(set(trajectory_seeds)),
        "corruption_seed_count": len(set(corruption_seeds)),
        "canonical_sha256": canonical_sha256(list(rows)),
    }


def _validate_frozen(
    paths: runner.Paths,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = (
        "development_seed_manifest.jsonl",
        "sealed_seed_manifest.jsonl",
        "corruption_manifest.json",
        "algorithm_grid.json",
        "metric_definitions.json",
        "decision_thresholds.json",
        "conventions.json",
        "protected_source_hashes.json",
        "manifest_hashes.json",
    )
    for name in required:
        _required_file(paths.frozen / name, f"frozen {name}")
    receipt = manifest_builder.freeze(paths.artifacts, check_only=True)
    strict_receipt = _strict_json(paths.frozen / "manifest_hashes.json")
    _assert_equal(receipt, strict_receipt, "manifest hash receipt")
    for name, expected_hash in strict_receipt["sha256"].items():
        if sha256_file(paths.frozen / name) != expected_hash:
            raise ValueError(f"Frozen receipt hash mismatch: {name}")
    development = _strict_jsonl(paths.frozen / "development_seed_manifest.jsonl")
    sealed = _strict_jsonl(paths.frozen / "sealed_seed_manifest.jsonl")
    development_evidence = _validate_manifest(
        development, "development", 1152, manifest_builder.DEVELOPMENT_REPLICATES
    )
    sealed_evidence = _validate_manifest(
        sealed, "sealed", 2304, manifest_builder.SEALED_REPLICATES
    )
    if (
        development_evidence["canonical_sha256"]
        != strict_receipt["development_manifest_canonical_sha256"]
    ):
        raise ValueError("Development canonical manifest hash mismatch")
    if (
        sealed_evidence["canonical_sha256"]
        != strict_receipt["sealed_manifest_canonical_sha256"]
    ):
        raise ValueError("Sealed canonical manifest hash mismatch")
    if {int(row["trajectory_seed"]) for row in development} & {
        int(row["trajectory_seed"]) for row in sealed
    }:
        raise ValueError("Development and sealed trajectory seeds overlap")
    if {int(row["corruption_seed"]) for row in development} & {
        int(row["corruption_seed"]) for row in sealed
    }:
        raise ValueError("Development and sealed corruption seeds overlap")
    grid = _strict_json(paths.frozen / "algorithm_grid.json")
    if tuple(grid.get("methods", ())) != TEMPORAL_METHODS:
        raise ValueError("Frozen grid does not name exactly the three temporal methods")
    if (
        len(grid.get("configurations", {})) != 4
        or grid.get("equal_search_budget_per_method") != 4
    ):
        raise ValueError(
            "Frozen grid does not provide exactly four equal-budget configs"
        )
    metric_definitions = _strict_json(paths.frozen / "metric_definitions.json")
    required_metric_sections = {
        "interval_convention",
        "finite_summary",
        "translation_error_m",
        "quotient_rotation_error_deg",
        "quotient_pose_error",
        "representation_stability",
        "dropout",
        "outlier",
        "dynamic_guardrails",
        "asymmetric_guardrails",
        "runtime_and_finiteness",
        "condition_aggregation",
        "development_objective",
        "bop_compatibility",
    }
    if not required_metric_sections.issubset(metric_definitions):
        raise ValueError("Frozen metric definitions omit exact required sections")
    decision_thresholds = _strict_json(paths.frozen / "decision_thresholds.json")
    if "at least one" not in decision_thresholds.get(
        "mechanism_pass_rule", ""
    ) or "no post-sealed" not in decision_thresholds.get("outlier_diagnostic_rule", ""):
        raise ValueError("Frozen classification and diagnostic rules are incomplete")
    return (
        strict_receipt,
        development,
        sealed,
        {
            "development": development_evidence,
            "sealed": sealed_evidence,
            "seed_sets_disjoint": True,
            "frozen_file_count": len(required),
        },
    )


def _validate_result_row(row: Mapping[str, Any], split: str) -> None:
    if row.get("record_type") != "m5_g0_trajectory_result" or row.get("split") != split:
        raise ValueError(f"Malformed {split} result row: {row.get('trajectory_id')}")
    status = row.get("status")
    if status not in {"success", "failure"}:
        raise ValueError(f"Unknown result status: {status}")
    if status == "success":
        if not isinstance(row.get("metrics"), dict) or row.get("failure") is not None:
            raise ValueError("Successful result does not contain metrics cleanly")
        runtime = row.get("frame_runtime_s")
        if not isinstance(runtime, list) or len(runtime) != 120:
            raise ValueError("Successful result does not retain 120 frame runtimes")
        finite_counts = row["metrics"].get("runtime_and_finite_counts", {})
        if finite_counts.get("nonfinite_output_value_count") != 0:
            raise ValueError("Successful result contains a non-finite output pose")
        if finite_counts.get("nonfinite_uncertainty_value_count") != 0:
            raise ValueError("Successful result contains non-finite uncertainty")
    else:
        if row.get("metrics") is not None or not isinstance(row.get("failure"), dict):
            raise ValueError("Failed result was not retained with failure evidence")
        if row.get("frame_runtime_s") not in ([], None):
            raise ValueError("Failed result has an unexpected partial runtime trace")


def _validate_development_results(
    paths: runner.Paths,
    manifest: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _required_file(
        paths.artifacts / "development_results.jsonl", "development results"
    )
    if (paths.artifacts / "development_results.partial.jsonl").exists():
        raise ValueError("Completed development results still have a partial JSONL")
    rows = _strict_jsonl(path)
    if len(rows) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError(
            f"Development results have {len(rows)} rows, expected {EXPECTED_DEVELOPMENT_ROWS}"
        )
    config_ids = set(grid["configurations"])
    expected = {
        (str(seed["trajectory_id"]), method, config_id)
        for seed in manifest
        for method in TEMPORAL_METHODS
        for config_id in config_ids
    }
    actual = []
    manifest_ids = {str(row["trajectory_id"]) for row in manifest}
    for row in rows:
        _validate_result_row(row, "development")
        if str(row["trajectory_id"]) not in manifest_ids:
            raise ValueError("Development result references an unknown trajectory")
        actual.append(runner._result_key(row))
    if len(actual) != len(set(actual)):
        raise ValueError("Development results contain duplicate keys")
    if set(actual) != expected:
        raise ValueError(
            "Development results do not exactly cover manifest x method x config"
        )
    grouped_hashes: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        grouped_hashes[str(row["trajectory_id"])].add(row.get("measurement_sha256"))
    bad_hashes = [
        key
        for key, hashes in grouped_hashes.items()
        if len(hashes) != 1 or None in hashes
    ]
    if bad_hashes:
        raise ValueError(
            "Development methods/configs did not receive identical measurement streams"
        )
    statuses = Counter(str(row["status"]) for row in rows)
    return rows, {
        "row_count": len(rows),
        "unique_key_count": len(set(actual)),
        "status_counts": dict(sorted(statuses.items())),
        "retained_failure_count": statuses.get("failure", 0),
        "measurement_identity_trajectory_count": len(grouped_hashes),
        "sha256": sha256_file(path),
    }


def _validate_selection(
    paths: runner.Paths,
    receipt: Mapping[str, Any],
    development_rows: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _required_file(paths.artifacts / "selected_configs.json", "selected configs")
    selected = _strict_json(path)
    if set(selected.get("selected", {})) != set(TEMPORAL_METHODS):
        raise ValueError("Selected config file does not cover every temporal method")
    if selected.get("sealed_evaluation_had_started") is not False:
        raise ValueError("Selection was not recorded before sealed evaluation")
    selected_time = _parse_time(selected.get("selected_at_utc"), "selected_at_utc")
    frozen_time = _parse_time(receipt.get("frozen_at_utc"), "frozen_at_utc")
    if selected_time < frozen_time:
        raise ValueError("Selected configs predate the frozen manifests")
    source_paths = {
        "development_seed_manifest": paths.frozen / "development_seed_manifest.jsonl",
        "algorithm_grid": paths.frozen / "algorithm_grid.json",
        "metric_definitions": paths.frozen / "metric_definitions.json",
        "manifest_hash_receipt": paths.frozen / "manifest_hashes.json",
        "development_results": paths.artifacts / "development_results.jsonl",
        "runner_source": paths.repo_root / "scripts" / "run_m5_g0.py",
        "core_source": paths.repo_root / "scripts" / "m5_g0_core.py",
        "metrics_source": paths.repo_root / "scripts" / "m5_g0_metrics.py",
    }
    for field, source_path in source_paths.items():
        if selected.get("source_hashes", {}).get(field) != sha256_file(source_path):
            raise ValueError(f"Selected-config source hash mismatch: {field}")
    objectives = {
        (method, config_id): runner._development_objective(
            development_rows, method, config_id
        )
        for method in TEMPORAL_METHODS
        for config_id in sorted(grid["configurations"])
    }
    stored_objectives = {
        (str(item["method"]), str(item["config_id"])): item
        for item in selected.get("all_objectives", [])
    }
    _assert_equal(stored_objectives, objectives, "development objectives")
    for method, choice in selected["selected"].items():
        config_id = str(choice["config_id"])
        if config_id not in grid["configurations"]:
            raise ValueError(f"Selected unknown config for {method}: {config_id}")
        _assert_equal(
            choice["configuration"],
            grid["configurations"][config_id],
            f"{method} configuration",
        )
        if choice["configuration_sha256"] != canonical_sha256(choice["configuration"]):
            raise ValueError(f"Selected configuration hash mismatch: {method}")
        _assert_equal(
            choice["objective"], objectives[(method, config_id)], f"{method} objective"
        )
        candidates = [
            objectives[(method, candidate)]
            for candidate in sorted(grid["configurations"])
        ]
        if any(item["objective"] is None for item in candidates):
            raise ValueError(f"Unavailable development objective for {method}")
        minimum = min(float(item["objective"]) for item in candidates)
        expected_id = min(
            item["config_id"]
            for item in candidates
            if abs(float(item["objective"]) - minimum) <= runner.OBJECTIVE_TIE_TOLERANCE
        )
        if config_id != expected_id:
            raise ValueError(
                f"Selected config does not satisfy frozen tie rule: {method}"
            )
    return selected, {
        "selected_at_utc": selected["selected_at_utc"],
        "selected_config_ids": {
            method: value["config_id"] for method, value in selected["selected"].items()
        },
        "source_hash_count": len(source_paths),
        "sha256": sha256_file(path),
    }


def _validate_chronology(
    paths: runner.Paths,
    receipt: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _required_file(
        paths.artifacts / "sealed_evaluation_receipt.json",
        "sealed evaluation chronology receipt",
    )
    chronology = _strict_json(path)
    frozen_time = _parse_time(
        chronology.get("frozen_at_utc"), "chronology frozen_at_utc"
    )
    selected_time = _parse_time(
        chronology.get("selected_at_utc"), "chronology selected_at_utc"
    )
    sealed_time = _parse_time(
        chronology.get("sealed_started_at_utc"), "sealed_started_at_utc"
    )
    if not frozen_time <= selected_time <= sealed_time:
        raise ValueError("Frozen/select/sealed timestamps are out of order")
    if chronology["frozen_at_utc"] != receipt["frozen_at_utc"]:
        raise ValueError("Chronology frozen timestamp differs from receipt")
    if chronology["selected_at_utc"] != selected["selected_at_utc"]:
        raise ValueError("Chronology selected timestamp differs from selection")
    selected_path = paths.artifacts / "selected_configs.json"
    if chronology.get("selected_configs_sha256") != sha256_file(selected_path):
        raise ValueError("Chronology selected-config hash mismatch")
    if chronology.get("sealed_manifest_sha256") != sha256_file(
        paths.frozen / "sealed_seed_manifest.jsonl"
    ):
        raise ValueError("Chronology sealed-manifest hash mismatch")
    if chronology.get("manifest_hash_receipt_sha256") != sha256_file(
        paths.frozen / "manifest_hashes.json"
    ):
        raise ValueError("Chronology manifest-receipt hash mismatch")
    source_paths = {
        "core": paths.repo_root / "scripts" / "m5_g0_core.py",
        "metrics": paths.repo_root / "scripts" / "m5_g0_metrics.py",
        "runner": paths.repo_root / "scripts" / "run_m5_g0.py",
    }
    for field, source_path in source_paths.items():
        if chronology.get("source_hashes", {}).get(field) != sha256_file(source_path):
            raise ValueError(f"Sealed chronology source hash mismatch: {field}")
    if chronology.get("sealed_tuning_allowed") is not False:
        raise ValueError("Sealed chronology does not forbid tuning")
    sealed_path = _required_file(
        paths.artifacts / "sealed_results.jsonl", "sealed results"
    )
    if path.stat().st_mtime_ns > sealed_path.stat().st_mtime_ns:
        raise ValueError("Sealed chronology receipt does not precede sealed results")
    return chronology, {
        "frozen_at_utc": chronology["frozen_at_utc"],
        "selected_at_utc": chronology["selected_at_utc"],
        "sealed_started_at_utc": chronology["sealed_started_at_utc"],
        "receipt_precedes_results": True,
        "source_hash_count": len(source_paths),
        "sha256": sha256_file(path),
    }


def _validate_sealed_results(
    paths: runner.Paths,
    manifest: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _required_file(paths.artifacts / "sealed_results.jsonl", "sealed results")
    if (paths.artifacts / "sealed_results.partial.jsonl").exists():
        raise ValueError("Completed sealed results still have a partial JSONL")
    rows = _strict_jsonl(path)
    if len(rows) != EXPECTED_SEALED_ROWS:
        raise ValueError(
            f"Sealed results have {len(rows)} rows, expected {EXPECTED_SEALED_ROWS}"
        )
    configs = {
        "RAW_HOLD": "raw_hold_no_tuning",
        **{
            method: str(selected["selected"][method]["config_id"])
            for method in TEMPORAL_METHODS
        },
    }
    expected = {
        (str(seed["trajectory_id"]), method, configs[method])
        for seed in manifest
        for method in SEALED_METHODS
    }
    actual = []
    manifest_ids = {str(row["trajectory_id"]) for row in manifest}
    for row in rows:
        _validate_result_row(row, "sealed")
        if str(row["trajectory_id"]) not in manifest_ids:
            raise ValueError("Sealed result references an unknown trajectory")
        actual.append(runner._result_key(row))
    if len(actual) != len(set(actual)):
        raise ValueError("Sealed results contain duplicate keys")
    if set(actual) != expected:
        raise ValueError("Sealed results do not exactly cover manifest x four methods")
    identity = runner._validate_measurement_identity(rows)
    if identity.get("pass") is not True or identity.get("mismatch_count") != 0:
        raise ValueError(
            "Sealed methods do not have four identical non-null measurement hashes"
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    for trajectory_id, group in grouped.items():
        if len(group) != 4 or len({row["measurement_sha256"] for row in group}) != 1:
            raise ValueError(f"Sealed measurement mismatch: {trajectory_id}")
    statuses = Counter(str(row["status"]) for row in rows)
    return rows, {
        "row_count": len(rows),
        "unique_key_count": len(set(actual)),
        "unique_trajectory_count": len(grouped),
        "status_counts": dict(sorted(statuses.items())),
        "retained_failure_count": statuses.get("failure", 0),
        "measurement_identity": identity,
        "sha256": sha256_file(path),
    }


def _validate_protected(paths: runner.Paths) -> dict[str, Any]:
    payload = _strict_json(paths.frozen / "protected_source_hashes.json")
    hashes = payload.get("sha256", {})
    if not hashes:
        raise ValueError("Protected M2/M3/M4 hash inventory is empty")
    for relative, expected in hashes.items():
        path = paths.repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protected M2/M3/M4 file is missing: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"Protected M2/M3/M4 file changed: {relative}")
    return {"file_count": len(hashes), "all_hashes_unchanged": True}


def _validate_estimator_contract() -> dict[str, Any]:
    fields = tuple(field.name for field in dataclasses.fields(core.EstimatorInput))
    expected = ("timestamp_s", "measurement_pose", "missing", "symmetry")
    if fields != expected:
        raise ValueError(
            f"EstimatorInput fields changed or leak evaluator data: {fields}"
        )
    forbidden = {
        "ground_truth_pose",
        "stress_family",
        "corruption_type",
        "representative_index",
        "axial_gauge_rad",
        "dropout_end",
        "outlier",
        "motion_family",
        "future_measurement",
        "evaluator_frames",
    }
    if forbidden & set(fields):
        raise ValueError("EstimatorInput exposes evaluator-only metadata")
    evaluator_fields = tuple(
        field.name for field in dataclasses.fields(core.EvaluatorFrame)
    )
    if not {
        "ground_truth_pose",
        "representative_index",
        "axial_gauge_rad",
        "dropout",
        "outlier",
    }.issubset(evaluator_fields):
        raise ValueError("EvaluatorFrame no longer holds the evaluator-only contract")
    corrupted_fields = tuple(
        field.name for field in dataclasses.fields(core.CorruptedSequence)
    )
    if not {
        "inputs",
        "evaluator_frames",
        "dropout_intervals",
        "outlier_intervals",
    }.issubset(corrupted_fields):
        raise ValueError(
            "CorruptedSequence no longer separates inputs and evaluator metadata"
        )
    step_classes = (core.RawHoldEstimator, core.ConstantTwistEstimator)
    for estimator_type in step_classes:
        signature = inspect.signature(estimator_type.step)
        parameters = list(signature.parameters.values())
        if len(parameters) != 2 or parameters[1].name not in {"item", "input", "frame"}:
            raise ValueError(
                f"Unexpected estimator step API: {estimator_type.__name__}{signature}"
            )
        annotations = inspect.get_annotations(estimator_type.step, eval_str=True)
        if annotations.get(parameters[1].name) is not core.EstimatorInput:
            raise ValueError(
                f"Estimator step no longer accepts EstimatorInput: {estimator_type.__name__}"
            )
    return {
        "estimator_input_fields": list(fields),
        "evaluator_frame_fields": list(evaluator_fields),
        "corrupted_sequence_fields": list(corrupted_fields),
        "hidden_evaluator_field_count": 0,
        "step_contract_class_count": len(step_classes),
    }


def _validate_recomputation(
    paths: runner.Paths,
    sealed_rows: Sequence[Mapping[str, Any]],
    selected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aggregate_path = _required_file(
        paths.artifacts / "aggregate_statistics.json", "aggregate statistics"
    )
    acceptance_path = _required_file(paths.artifacts / "acceptance.json", "acceptance")
    stored_aggregate = _strict_json(aggregate_path)
    stored_acceptance = _strict_json(acceptance_path)
    with tempfile.TemporaryDirectory(prefix="poseloop-m5-validation-") as temporary:
        temporary_root = Path(temporary)
        temporary_paths = runner.Paths(
            repo_root=paths.repo_root,
            artifacts=temporary_root,
            frozen=paths.frozen,
            reports=temporary_root / "reports",
            precomputed=temporary_root / "precomputed.json",
        )
        recomputed_aggregate = runner._aggregate_statistics(
            temporary_paths,
            sealed_rows,
            selected_sha256,
            paths.artifacts / "sealed_results.jsonl",
        )
        recomputed_acceptance = runner._acceptance(
            temporary_paths,
            sealed_rows,
            recomputed_aggregate,
            selected_sha256,
        )
    _assert_equal(stored_aggregate, recomputed_aggregate, "aggregate statistics")
    _assert_equal(stored_acceptance, recomputed_acceptance, "acceptance result")
    outlier_values = (
        stored_acceptance.get("criteria", {})
        .get("outlier_diagnostic", {})
        .get("values", {})
    )
    if set(outlier_values) != set(SEALED_METHODS):
        raise ValueError("Outlier diagnostic does not compare all four methods")
    for method, value in outlier_values.items():
        if value.get("trajectory_count") != 768:
            raise ValueError(
                f"Outlier diagnostic silently lost trajectories for {method}"
            )
        count = value.get("recovery_failure_count")
        rate = value.get("recovery_failure_rate")
        if not isinstance(count, int) or not isinstance(rate, (int, float)):
            raise ValueError(f"Outlier failure accounting is incomplete for {method}")
        if not math.isclose(float(rate), count / 768.0, abs_tol=1e-15):
            raise ValueError(f"Outlier failure rate is inconsistent for {method}")
    return (
        stored_aggregate,
        stored_acceptance,
        {
            "aggregate_sha256": sha256_file(aggregate_path),
            "acceptance_sha256": sha256_file(acceptance_path),
            "classification": stored_acceptance["classification"],
            "aggregate_exact_recomputation": True,
            "acceptance_exact_recomputation": True,
        },
    )


def _validate_report_artifacts(
    paths: runner.Paths,
    aggregate: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    selected_sha256: str,
    sealed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for name in REQUIRED_PLOTS:
        path = _required_file(paths.reports / name, f"required plot {name}")
        with path.open("rb") as handle:
            signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n" or path.stat().st_size <= 1024:
            raise ValueError(f"Required plot is not a substantive PNG: {path}")
    report_path = _required_file(
        paths.repo_root / "reports" / "m5_g0_synthetic_mechanism_validation.md",
        "paper-style M5-G0 report",
    )
    report_text = report_path.read_text(encoding="utf-8")
    required_report_text = (
        "Synthetic Mechanism Validation",
        acceptance["classification"],
        "Anti-cherry-picking answers",
        "real-video",
        "NEAREST_REPRESENTATIVE_CT",
    )
    if any(text not in report_text for text in required_report_text):
        raise ValueError(
            "Paper-style report is missing required conclusions or claim boundaries"
        )
    compact_path = _required_file(paths.precomputed, "tracked compact M5-G0 result")
    compact = _strict_json(compact_path)
    if compact.get("classification") != acceptance.get("classification"):
        raise ValueError("Compact result classification differs from acceptance")
    if compact.get("sealed_result_row_count") != EXPECTED_SEALED_ROWS:
        raise ValueError("Compact result sealed row count is incorrect")
    if set(compact.get("plots", ())) != set(REQUIRED_PLOTS):
        raise ValueError("Compact result does not list exactly the ten required plots")
    representative_path = _required_file(
        paths.artifacts / "representative_traces.json", "representative traces"
    )
    representative = _strict_json(representative_path)
    dropout_trace = representative.get("c4_dropout_10", {})
    dropout_intervals = dropout_trace.get("intervals", {}).get("dropout_intervals", [])
    if len(dropout_intervals) != 1:
        raise ValueError("Frozen dropout representative lacks its exact interval")
    dropout_end = int(dropout_intervals[0][1])
    methods = dropout_trace.get("methods", {})
    if set(methods) != set(SEALED_METHODS):
        raise ValueError("Dropout representative does not contain all four methods")
    for method, payload in methods.items():
        accepted = payload.get("accepted")
        if (
            not isinstance(accepted, list)
            or len(accepted) != 120
            or not any(value is True for value in accepted[dropout_end:])
        ):
            raise ValueError(
                f"Dropout timeline cannot show post-dropout reacquisition for {method}"
            )
    zero_receipt_path = _required_file(
        paths.artifacts / "zero_aware_plot_receipt.json",
        "zero-aware dropout plot receipt",
    )
    zero_receipt = _strict_json(zero_receipt_path)
    expected_zero_values = {
        method: runner._median(
            runner._finite_values(
                runner._filter_rows(
                    sealed_rows,
                    method,
                    symmetries=set(runner.SYMMETRIC_CLASSES),
                    stresses={"DROPOUT_10"},
                ),
                "metrics.dropout.frames_to_return_within_nominal.median",
            )
        )
        for method in SEALED_METHODS
    }
    _assert_equal(
        zero_receipt.get("values"),
        expected_zero_values,
        "zero-aware dropout plot values",
    )
    finite_zero_values = [
        float(value) for value in expected_zero_values.values() if value is not None
    ]
    expected_all_zero = bool(finite_zero_values) and all(
        abs(value) <= 1e-15 for value in finite_zero_values
    )
    if zero_receipt.get("all_finite_values_zero") is not expected_all_zero:
        raise ValueError("Zero-aware dropout annotation state is incorrect")
    zero_renderer = paths.repo_root / "scripts" / "render_m5_g0_zero_aware_plot.py"
    _assert_equal(
        zero_receipt.get("source_hashes"),
        {
            "sealed_results": sha256_file(paths.artifacts / "sealed_results.jsonl"),
            "renderer": sha256_file(zero_renderer),
            "plot": sha256_file(paths.reports / "aggregate_dropout_recovery.png"),
        },
        "zero-aware dropout plot source hashes",
    )
    expected_hashes = {
        "sealed_results": sha256_file(paths.artifacts / "sealed_results.jsonl"),
        "selected_configs": selected_sha256,
        "aggregate_statistics": sha256_file(
            paths.artifacts / "aggregate_statistics.json"
        ),
        "acceptance": sha256_file(paths.artifacts / "acceptance.json"),
        "representative_traces": sha256_file(representative_path),
    }
    _assert_equal(
        compact.get("source_hashes"), expected_hashes, "compact result source hashes"
    )
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            str(paths.precomputed.relative_to(paths.repo_root)),
        ],
        cwd=paths.repo_root,
        capture_output=True,
        text=True,
    )
    if ignored.returncode == 0:
        raise ValueError("Compact precomputed result is ignored and cannot be tracked")
    if ignored.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed: {ignored.stderr.strip()}")
    return {
        "required_plot_count": len(REQUIRED_PLOTS),
        "plots": list(REQUIRED_PLOTS),
        "report_sha256": sha256_file(report_path),
        "precomputed_sha256": sha256_file(compact_path),
        "representative_traces_sha256": sha256_file(representative_path),
        "compact_result_trackable": True,
        "aggregate_record_type": aggregate.get("record_type"),
    }


def _strict_all_json(paths: runner.Paths) -> dict[str, Any]:
    files = sorted(
        [
            path
            for path in paths.artifacts.rglob("*")
            if path.is_file()
            and path.name != "formal_validation.json"
            and path.suffix in {".json", ".jsonl"}
        ]
        + [paths.precomputed]
    )
    total_rows = 0
    for path in files:
        if path.suffix == ".jsonl":
            total_rows += len(_strict_jsonl(path))
        else:
            _strict_json(path)
    return {
        "json_file_count": len(files),
        "jsonl_row_count": total_rows,
        "nonfinite_json_number_count": 0,
    }


def _formal_body(repo_root: Path) -> dict[str, Any]:
    paths = _paths(repo_root)
    receipt, development_manifest, sealed_manifest, frozen_evidence = _validate_frozen(
        paths
    )
    grid = _strict_json(paths.frozen / "algorithm_grid.json")
    development_rows, development_evidence = _validate_development_results(
        paths, development_manifest, grid
    )
    selected, selection_evidence = _validate_selection(
        paths, receipt, development_rows, grid
    )
    _, chronology_evidence = _validate_chronology(paths, receipt, selected)
    sealed_rows, sealed_evidence = _validate_sealed_results(
        paths, sealed_manifest, selected
    )
    protected_evidence = _validate_protected(paths)
    contract_evidence = _validate_estimator_contract()
    aggregate, acceptance, recomputation_evidence = _validate_recomputation(
        paths, sealed_rows, selection_evidence["sha256"]
    )
    report_evidence = _validate_report_artifacts(
        paths,
        aggregate,
        acceptance,
        selection_evidence["sha256"],
        sealed_rows,
    )
    json_evidence = _strict_all_json(paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_formal_validation",
        "status": "PASS",
        "classification": acceptance["classification"],
        "claim_boundary": (
            "Synthetic mechanism validation only; no real-video, FoundationPose, "
            "official BOP benchmark, robot-control, or hardware claim."
        ),
        "checks": {
            "frozen_manifests": frozen_evidence,
            "development_results": development_evidence,
            "selection": selection_evidence,
            "sealed_chronology": chronology_evidence,
            "sealed_results": sealed_evidence,
            "protected_m2_m3_m4": protected_evidence,
            "estimator_input_contract": contract_evidence,
            "deterministic_recomputation": recomputation_evidence,
            "reports_and_compact_result": report_evidence,
            "strict_json_finiteness": json_evidence,
        },
        "source_hashes": {
            "validator": sha256_file(Path(__file__).resolve()),
            "manifest_builder": sha256_file(
                repo_root / "scripts" / "build_m5_g0_manifests.py"
            ),
            "core": sha256_file(repo_root / "scripts" / "m5_g0_core.py"),
            "metrics": sha256_file(repo_root / "scripts" / "m5_g0_metrics.py"),
            "runner": sha256_file(repo_root / "scripts" / "run_m5_g0.py"),
            "zero_aware_plot_renderer": sha256_file(
                repo_root / "scripts" / "render_m5_g0_zero_aware_plot.py"
            ),
            "frozen_receipt": sha256_file(paths.frozen / "manifest_hashes.json"),
        },
    }


def validate(repo_root: Path, *, check_only: bool) -> dict[str, Any]:
    body = _formal_body(repo_root)
    path = repo_root / "artifacts" / "m5_g0" / "formal_validation.json"
    if check_only:
        stored = _strict_json(_required_file(path, "formal validation result"))
        observed = dict(stored)
        validated_at = observed.pop("validated_at_utc", None)
        _parse_time(validated_at, "formal validation validated_at_utc")
        _assert_equal(observed, body, "existing formal validation result")
        return stored
    result = {
        **body,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    result = validate(repo_root, check_only=bool(args.check))
    action = "checked" if args.check else "wrote"
    print(
        f"{action} M5-G0 formal validation: {result['status']} "
        f"({result['classification']})"
    )


if __name__ == "__main__":
    main()
