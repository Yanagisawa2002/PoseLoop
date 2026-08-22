#!/usr/bin/env python3
"""Tune, seal, and report the PoseLoop M5-G0 synthetic mechanism gate.

The manifest freeze is intentionally a separate command implemented by
``build_m5_g0_manifests.py``.  This runner never changes frozen inputs and never
uses evaluator-only fields when constructing estimator inputs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import m5_g0_core as core
import m5_g0_metrics as metric_lib
from m1_common import (
    append_jsonl_durable,
    canonical_sha256,
    load_jsonl,
    read_json,
    sha256_file,
    write_json_atomic,
)


SCHEMA_VERSION = 1
EXPECTED_DEVELOPMENT_TRAJECTORIES = 1152
EXPECTED_SEALED_TRAJECTORIES = 2304
TEMPORAL_METHODS = (
    "STANDARD_SE3_CT",
    "NEAREST_REPRESENTATIVE_CT",
    "SYMQUOT_CT",
)
SEALED_METHODS = ("RAW_HOLD", *TEMPORAL_METHODS)
SYMMETRIC_CLASSES = ("C2", "C4", "CONTINUOUS_AXIAL")
SWITCH_STRESSES = ("REPRESENTATION_SWITCH", "COMBINED")
OBJECTIVE_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    artifacts: Path
    frozen: Path
    reports: Path
    precomputed: Path

    @classmethod
    def create(
        cls,
        repo_root: Path,
        artifacts: Path | None,
        reports: Path | None,
        precomputed: Path | None,
    ) -> "Paths":
        artifact_root = (artifacts or repo_root / "artifacts" / "m5_g0").resolve()
        required_artifact_root = (repo_root / "artifacts" / "m5_g0").resolve()
        if not artifact_root.is_relative_to(required_artifact_root):
            raise ValueError(
                f"M5-G0 raw outputs must remain under {required_artifact_root}"
            )
        report_root = (reports or repo_root / "reports" / "m5_g0").resolve()
        required_report_root = (repo_root / "reports" / "m5_g0").resolve()
        if not report_root.is_relative_to(required_report_root):
            raise ValueError(f"M5-G0 plots must remain under {required_report_root}")
        precomputed_path = (
            precomputed or repo_root / "precomputed" / "m5_g0" / "results.json"
        ).resolve()
        required_precomputed_root = (repo_root / "precomputed" / "m5_g0").resolve()
        if not precomputed_path.is_relative_to(required_precomputed_root):
            raise ValueError(
                f"M5-G0 compact results must remain under {required_precomputed_root}"
            )
        return cls(
            repo_root=repo_root.resolve(),
            artifacts=artifact_root,
            frozen=artifact_root / "frozen",
            reports=report_root,
            precomputed=precomputed_path,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "sealed", "report"))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--precomputed-result", type=Path)
    return parser.parse_args(argv)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_frozen(paths: Paths) -> dict[str, Any]:
    receipt_path = paths.frozen / "manifest_hashes.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Frozen M5-G0 inputs are absent; run build_m5_g0_manifests.py first: {receipt_path}"
        )
    receipt = read_json(receipt_path)
    for name, expected in receipt.get("sha256", {}).items():
        path = paths.frozen / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    if (
        int(receipt.get("development_trajectory_count", -1))
        != EXPECTED_DEVELOPMENT_TRAJECTORIES
    ):
        raise ValueError("Frozen development trajectory count is not 1152")
    if int(receipt.get("sealed_trajectory_count", -1)) != EXPECTED_SEALED_TRAJECTORIES:
        raise ValueError("Frozen sealed trajectory count is not 2304")
    return receipt


def _enum(enum_type: Any, name: str) -> Any:
    try:
        return enum_type[name]
    except (KeyError, TypeError):
        return enum_type(name)


def _symmetry_spec(name: str) -> Any:
    kind = _enum(core.SymmetryClass, name)
    return core.symmetry_spec(kind)


def _config(config: Mapping[str, Any]) -> Any:
    return core.EstimatorConfig(**dict(config))


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _frame_pose(frame: Any) -> np.ndarray:
    pose = _field(frame, "pose", "gt_pose", "ground_truth_pose", "transform")
    if pose is None:
        raise ValueError(f"Frame has no pose field: {type(frame).__name__}")
    result = np.asarray(pose, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError("Frame pose is not 4x4")
    return result


def _frame_timestamp(frame: Any) -> float:
    value = _field(frame, "timestamp", "timestamp_s", "time")
    if value is None:
        raise ValueError(f"Frame has no timestamp field: {type(frame).__name__}")
    return float(value)


def _input_measurement(frame: Any) -> np.ndarray | None:
    value = _field(frame, "measurement", "pose_measurement", "measurement_pose", "pose")
    missing = bool(
        _field(frame, "missing", "measurement_missing", default=value is None)
    )
    if missing or value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("Estimator-input measurement is not 4x4")
    return pose


def _sequence_arrays(ground_truth: Any, corrupted: Any) -> dict[str, Any]:
    timestamps = np.asarray(_field(ground_truth, "timestamps"), dtype=np.float64)
    gt_poses = np.asarray(_field(ground_truth, "poses"), dtype=np.float64)
    inputs = list(_field(corrupted, "estimator_inputs", "inputs", default=[]))
    evaluator = list(_field(corrupted, "evaluator_frames", default=[]))
    if gt_poses.ndim != 3 or gt_poses.shape[1:] != (4, 4):
        raise ValueError("Ground-truth poses do not have shape (N,4,4)")
    if len(gt_poses) == 0 or len(gt_poses) != len(inputs):
        raise ValueError("Ground-truth and estimator-input frame counts differ")
    if evaluator and len(evaluator) != len(gt_poses):
        raise ValueError("Evaluator-only frame count differs from input frame count")
    input_timestamps = np.asarray(
        [_frame_timestamp(frame) for frame in inputs], dtype=np.float64
    )
    if not np.array_equal(timestamps, input_timestamps):
        raise ValueError(
            "Ground truth and estimator inputs do not share exact timestamps"
        )
    measurements: list[np.ndarray | None] = [
        _input_measurement(frame) for frame in inputs
    ]
    return {
        "timestamps": timestamps,
        "gt_poses": gt_poses,
        "measurements": measurements,
        "inputs": inputs,
        "evaluator_frames": evaluator,
    }


def _measurement_hash(inputs: Sequence[Any], symmetry_class: str) -> str:
    visible = []
    for frame in inputs:
        measurement = _input_measurement(frame)
        visible.append(
            {
                "timestamp_s": _frame_timestamp(frame),
                "missing": measurement is None,
                "measurement_pose": None
                if measurement is None
                else measurement.tolist(),
                "declared_symmetry": symmetry_class,
            }
        )
    return canonical_sha256(visible)


def _synthetic_model(symmetry_class: str) -> tuple[np.ndarray, float] | None:
    if symmetry_class == "CONTINUOUS_AXIAL":
        return None
    if symmetry_class == "C4":
        extents = np.asarray((0.030, 0.030, 0.025), dtype=np.float64)
    elif symmetry_class == "C2":
        extents = np.asarray((0.040, 0.022, 0.025), dtype=np.float64)
    else:
        extents = np.asarray((0.041, 0.027, 0.019), dtype=np.float64)
    corners = np.asarray(
        [
            (sx * extents[0], sy * extents[1], sz * extents[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    if symmetry_class == "ASYMMETRIC":
        corners = np.vstack((corners, np.asarray([[0.013, -0.007, 0.031]])))
    pairwise = corners[:, None, :] - corners[None, :, :]
    diameter = float(np.max(np.linalg.norm(pairwise, axis=2)))
    return corners, diameter


def _interval_payload(corrupted: Any, motion_family: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    dropouts = _field(corrupted, "dropout_intervals", default=())
    outliers = _field(corrupted, "outlier_intervals", default=())
    if dropouts:
        payload["dropout_intervals"] = [list(interval) for interval in dropouts]
    if outliers:
        payload["outlier_intervals"] = [list(interval) for interval in outliers]
    if motion_family == "DIRECTION_REVERSAL":
        payload["reversal_frame"] = int(
            _field(_field(corrupted, "ground_truth"), "reversal_frame", default=60)
        )
    return payload


def _run_timed(
    corrupted: Any, method: str, config: Any
) -> tuple[list[Any], list[float]]:
    kind = _enum(core.EstimatorKind, method)
    estimator = core.create_estimator(kind, config)
    outputs: list[Any] = []
    runtimes: list[float] = []
    for frame in _field(corrupted, "estimator_inputs", "inputs", default=[]):
        start = time.perf_counter_ns()
        output = estimator.step(frame)
        stop = time.perf_counter_ns()
        outputs.append(output)
        runtimes.append((stop - start) * 1e-9)
    return outputs, runtimes


def _output_arrays(outputs: Sequence[Any]) -> dict[str, Any]:
    poses: list[np.ndarray] = []
    accepted: list[bool] = []
    residuals: list[float] = []
    uncertainties: list[np.ndarray] = []
    initialized: list[bool] = []
    statuses: list[str] = []
    for output in outputs:
        pose = _field(output, "pose")
        poses.append(
            np.full((4, 4), np.nan, dtype=np.float64)
            if pose is None
            else np.asarray(pose, dtype=np.float64)
        )
        accepted.append(bool(_field(output, "accepted", default=False)))
        residual = _field(
            output,
            "runtime_residual_norm",
            "residual_norm",
            "innovation_norm",
            "residual",
            default=math.nan,
        )
        if residual is None:
            residual = math.nan
        residual_array = np.asarray(residual, dtype=np.float64)
        residuals.append(
            float(residual_array)
            if residual_array.ndim == 0
            else float(np.linalg.norm(residual_array))
        )
        uncertainty = _field(
            output, "uncertainty_diag", "uncertainty", default=[math.nan]
        )
        uncertainties.append(np.asarray(uncertainty, dtype=np.float64).reshape(-1))
        initialized.append(
            bool(
                pose is not None
                and np.isfinite(np.asarray(pose, dtype=np.float64)).all()
            )
        )
        status = _field(output, "status", "reason", default="unknown")
        statuses.append(str(getattr(status, "value", status)))
    if not poses:
        raise ValueError("Estimator returned no frames")
    widths = {len(value) for value in uncertainties}
    if len(widths) != 1:
        raise ValueError("Estimator uncertainty width changes within a trajectory")
    return {
        "poses": np.stack(poses),
        "accepted": np.asarray(accepted, dtype=bool),
        "residuals": np.asarray(residuals, dtype=np.float64),
        "uncertainty": np.stack(uncertainties),
        "initialized": np.asarray(initialized, dtype=bool),
        "statuses": statuses,
    }


def _generate(row: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], str]:
    symmetry = _symmetry_spec(str(row["symmetry_class"]))
    motion = _enum(core.MotionFamily, str(row["motion_family"]))
    stress = _enum(core.StressFamily, str(row["stress_family"]))
    truth = core.generate_ground_truth(
        symmetry,
        motion,
        int(row["trajectory_seed"]),
        frame_count=120,
        control_hz=20.0,
    )
    corrupted = core.corrupt_measurements(
        truth,
        stress,
        int(row["corruption_seed"]),
    )
    arrays = _sequence_arrays(truth, corrupted)
    digest = _measurement_hash(arrays["inputs"], str(row["symmetry_class"]))
    return truth, corrupted, arrays, digest


def _execute(
    manifest_row: Mapping[str, Any],
    corrupted: Any,
    arrays: Mapping[str, Any],
    measurement_hash: str,
    method: str,
    config_id: str,
    config_values: Mapping[str, Any],
    *,
    include_mssd: bool = True,
) -> dict[str, Any]:
    estimator_config = _config(config_values)
    outputs, runtimes = _run_timed(corrupted, method, estimator_config)
    if len(outputs) != len(arrays["timestamps"]):
        raise ValueError("Estimator output frame count differs from the common input")
    output_arrays = _output_arrays(outputs)
    symmetry_spec = _symmetry_spec(str(manifest_row["symmetry_class"]))
    model = (
        _synthetic_model(str(manifest_row["symmetry_class"])) if include_mssd else None
    )
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model_points"], kwargs["object_diameter"] = model
    metrics = metric_lib.compute_trajectory_metrics(
        arrays["timestamps"],
        arrays["gt_poses"],
        arrays["measurements"],
        output_arrays["poses"],
        output_arrays["accepted"],
        output_arrays["residuals"],
        output_arrays["uncertainty"],
        runtimes,
        symmetry_spec,
        evaluator_intervals=_interval_payload(
            corrupted, str(manifest_row["motion_family"])
        ),
        metadata={
            "trajectory_id": manifest_row["trajectory_id"],
            "method": method,
            "config_id": config_id,
        },
        include_frame_metrics=False,
        initialization_failures=int(not bool(np.any(output_arrays["initialized"]))),
        **kwargs,
    )
    return {
        "metrics": metrics,
        "frame_runtime_s": [float(value) for value in runtimes],
        "outputs": outputs,
        "output_arrays": output_arrays,
        "measurement_sha256": measurement_hash,
    }


def _base_row(
    split: str,
    manifest_row: Mapping[str, Any],
    method: str,
    config_id: str,
    measurement_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_trajectory_result",
        "split": split,
        "trajectory_id": manifest_row["trajectory_id"],
        "symmetry_class": manifest_row["symmetry_class"],
        "motion_family": manifest_row["motion_family"],
        "stress_family": manifest_row["stress_family"],
        "replicate_index": int(manifest_row["replicate_index"]),
        "trajectory_seed": int(manifest_row["trajectory_seed"]),
        "corruption_seed": int(manifest_row["corruption_seed"]),
        "method": method,
        "config_id": config_id,
        "measurement_sha256": measurement_hash,
    }


def _success_row(
    split: str,
    manifest_row: Mapping[str, Any],
    method: str,
    config_id: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    row = _base_row(
        split,
        manifest_row,
        method,
        config_id,
        str(execution["measurement_sha256"]),
    )
    row.update(
        {
            "status": "success",
            "failure": None,
            "metrics": execution["metrics"],
            "frame_runtime_s": execution["frame_runtime_s"],
        }
    )
    return row


def _failure_row(
    split: str,
    manifest_row: Mapping[str, Any],
    method: str,
    config_id: str,
    measurement_hash: str | None,
    exc: Exception,
) -> dict[str, Any]:
    row = _base_row(split, manifest_row, method, config_id, measurement_hash)
    row.update(
        {
            "status": "failure",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "metrics": None,
            "frame_runtime_s": [],
        }
    )
    return row


def _result_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["trajectory_id"]), str(row["method"]), str(row["config_id"])


def _load_partial(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    if not path.exists():
        return [], set()
    rows = load_jsonl(path)
    keys = [_result_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate result keys in resumable JSONL: {path}")
    return rows, set(keys)


def _complete_results(
    final_path: Path,
    partial_path: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    if final_path.exists():
        rows = load_jsonl(final_path)
        if len(rows) != expected_count:
            raise ValueError(
                f"Completed result count is not {expected_count}: {final_path}"
            )
        return rows
    rows, _ = _load_partial(partial_path)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Stage stopped with {len(rows)}/{expected_count} durable rows in {partial_path}"
        )
    partial_path.replace(final_path)
    return rows


def _path_value(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_values(rows: Iterable[Mapping[str, Any]], path: str) -> list[float]:
    values = []
    for row in rows:
        value = _path_value(row, path)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return values


def _median(values: Sequence[float]) -> float | None:
    return (
        None if not values else float(np.median(np.asarray(values, dtype=np.float64)))
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _development_objective(
    rows: Sequence[Mapping[str, Any]], method: str, config_id: str
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["method"] == method and row["config_id"] == config_id
    ]
    if len(selected) != EXPECTED_DEVELOPMENT_TRAJECTORIES:
        raise ValueError(f"Development cell {method}/{config_id} is incomplete")
    successes = [row for row in selected if row["status"] == "success"]
    median_error = _median(
        _finite_values(successes, "metrics.accuracy.median_quotient_pose_error")
    )
    jumps = []
    for row in successes:
        stability = _path_value(row, "metrics.representation_stability") or {}
        jumps.append(
            float(stability.get("catastrophic_representation_jump_count", 0))
            + float(stability.get("axial_gauge_jump_count", 0))
        )
    reversal_lags = []
    for row in successes:
        if row["motion_family"] != "DIRECTION_REVERSAL":
            continue
        translation = _path_value(
            row, "metrics.dynamic_guardrails.translation_direction_change_lag_frames"
        )
        rotation = _path_value(
            row,
            "metrics.dynamic_guardrails.observable_rotation_direction_change_lag_frames",
        )
        if translation is not None and rotation is not None:
            reversal_lags.append(float(max(translation, rotation)))
    contaminations = []
    for row in successes:
        if row["stress_family"] not in {"OUTLIER_BURST", "COMBINED"}:
            continue
        value = _path_value(row, "metrics.outlier.contamination_duration_frames")
        if value is not None:
            contaminations.append(float(value))
    components = {
        "median_quotient_pose_error": median_error,
        "mean_representation_jumps": _mean(jumps),
        "median_max_reversal_lag_frames": _median(reversal_lags),
        "mean_outlier_contamination_frames": _mean(contaminations),
        "trajectory_failure_rate": 1.0 - len(successes) / len(selected),
    }
    if any(
        components[key] is None
        for key in components
        if key != "trajectory_failure_rate"
    ):
        objective = None
    else:
        objective = float(
            components["median_quotient_pose_error"]
            + components["mean_representation_jumps"]
            + components["median_max_reversal_lag_frames"] / 4.0
            + components["mean_outlier_contamination_frames"] / 3.0
            + 100.0 * components["trajectory_failure_rate"]
        )
    return {
        "method": method,
        "config_id": config_id,
        "trajectory_count": len(selected),
        "successful_trajectory_count": len(successes),
        "components": components,
        "objective": objective,
    }


def _select_configs(
    paths: Paths,
    rows: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    receipt: Mapping[str, Any],
    development_result_path: Path,
) -> dict[str, Any]:
    selected_path = paths.artifacts / "selected_configs.json"
    if selected_path.exists():
        return read_json(selected_path)
    config_ids = sorted(grid["configurations"])
    objectives = [
        _development_objective(rows, method, config_id)
        for method in TEMPORAL_METHODS
        for config_id in config_ids
    ]
    selected: dict[str, Any] = {}
    for method in TEMPORAL_METHODS:
        candidates = [row for row in objectives if row["method"] == method]
        if any(row["objective"] is None for row in candidates):
            raise RuntimeError(
                f"Cannot select {method}: at least one objective is unavailable"
            )
        minimum = min(float(row["objective"]) for row in candidates)
        tied = sorted(
            row["config_id"]
            for row in candidates
            if abs(float(row["objective"]) - minimum) <= OBJECTIVE_TIE_TOLERANCE
        )
        config_id = tied[0]
        selected[method] = {
            "config_id": config_id,
            "configuration": grid["configurations"][config_id],
            "configuration_sha256": canonical_sha256(grid["configurations"][config_id]),
            "objective": next(
                row for row in candidates if row["config_id"] == config_id
            ),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_selected_configurations",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_evaluation_had_started": False,
        "equal_search_budget_per_method": len(config_ids),
        "objective_formula": (
            "median_error + mean_jumps + median_max_reversal_lag/4 + "
            "mean_outlier_contamination/3 + 100*failure_rate"
        ),
        "tie_tolerance": OBJECTIVE_TIE_TOLERANCE,
        "tie_break": "lexicographically smallest config_id",
        "source_hashes": {
            "development_seed_manifest": sha256_file(
                paths.frozen / "development_seed_manifest.jsonl"
            ),
            "algorithm_grid": sha256_file(paths.frozen / "algorithm_grid.json"),
            "metric_definitions": sha256_file(paths.frozen / "metric_definitions.json"),
            "manifest_hash_receipt": sha256_file(paths.frozen / "manifest_hashes.json"),
            "development_results": sha256_file(development_result_path),
            "runner_source": sha256_file(Path(__file__).resolve()),
            "core_source": sha256_file(paths.repo_root / "scripts" / "m5_g0_core.py"),
            "metrics_source": sha256_file(
                paths.repo_root / "scripts" / "m5_g0_metrics.py"
            ),
        },
        "frozen_receipt_sha256": receipt.get("sha256", {}),
        "all_objectives": objectives,
        "selected": selected,
    }
    write_json_atomic(selected_path, payload)
    return payload


def tune(paths: Paths) -> dict[str, Any]:
    receipt = _verify_frozen(paths)
    manifest = load_jsonl(paths.frozen / "development_seed_manifest.jsonl")
    if len(manifest) != EXPECTED_DEVELOPMENT_TRAJECTORIES:
        raise ValueError("Development manifest does not contain 1152 trajectories")
    grid = read_json(paths.frozen / "algorithm_grid.json")
    config_ids = sorted(grid["configurations"])
    if len(config_ids) != 4:
        raise ValueError(
            "The frozen development grid must contain exactly four configs"
        )
    expected = len(manifest) * len(TEMPORAL_METHODS) * len(config_ids)
    final_path = paths.artifacts / "development_results.jsonl"
    partial_path = paths.artifacts / "development_results.partial.jsonl"
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        rows = _complete_results(final_path, partial_path, expected)
        return _select_configs(paths, rows, grid, receipt, final_path)
    _, complete = _load_partial(partial_path)
    for manifest_row in manifest:
        missing = [
            (method, config_id)
            for method in TEMPORAL_METHODS
            for config_id in config_ids
            if (
                str(manifest_row["trajectory_id"]),
                method,
                config_id,
            )
            not in complete
        ]
        if not missing:
            continue
        measurement_hash: str | None = None
        try:
            _, corrupted, arrays, measurement_hash = _generate(manifest_row)
        except Exception as exc:
            for method, config_id in missing:
                row = _failure_row(
                    "development",
                    manifest_row,
                    method,
                    config_id,
                    None,
                    exc,
                )
                append_jsonl_durable(partial_path, row)
            continue
        for method, config_id in missing:
            try:
                execution = _execute(
                    manifest_row,
                    corrupted,
                    arrays,
                    measurement_hash,
                    method,
                    config_id,
                    grid["configurations"][config_id],
                    include_mssd=False,
                )
                row = _success_row(
                    "development", manifest_row, method, config_id, execution
                )
            except Exception as exc:
                row = _failure_row(
                    "development",
                    manifest_row,
                    method,
                    config_id,
                    measurement_hash,
                    exc,
                )
            append_jsonl_durable(partial_path, row)
    rows = _complete_results(final_path, partial_path, expected)
    return _select_configs(paths, rows, grid, receipt, final_path)


def _selected_configs(paths: Paths) -> tuple[dict[str, Any], str]:
    path = paths.artifacts / "selected_configs.json"
    if not path.is_file():
        raise FileNotFoundError("Run the tune stage before sealed evaluation")
    payload = read_json(path)
    if set(payload.get("selected", {})) != set(TEMPORAL_METHODS):
        raise ValueError("Selected configuration file is incomplete")
    for method, selection in payload["selected"].items():
        if (
            canonical_sha256(selection["configuration"])
            != selection["configuration_sha256"]
        ):
            raise ValueError(f"Selected configuration hash mismatch: {method}")
    if (
        sha256_file(paths.frozen / "algorithm_grid.json")
        != payload["source_hashes"]["algorithm_grid"]
    ):
        raise ValueError("Frozen grid changed after development selection")
    return payload, sha256_file(path)


def _flat_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        metadata = {
            "trajectory_id": row["trajectory_id"],
            "method": row["method"],
            "config_id": row["config_id"],
            "symmetry_class": row["symmetry_class"],
            "motion_family": row["motion_family"],
            "stress_family": row["stress_family"],
            "status": row["status"],
        }
        if row.get("metrics") is None:
            flattened.append(metadata)
        else:
            metric_payload = dict(row["metrics"])
            metric_symmetry = metric_payload.pop("symmetry_class", None)
            if metric_symmetry != metadata["symmetry_class"]:
                raise ValueError(
                    "Trajectory metadata and metric symmetry_class disagree: "
                    f"{metadata['trajectory_id']}"
                )
            flattened.append(metric_lib.make_trajectory_row(metric_payload, metadata))
    return flattened


def _validate_measurement_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    bad = []
    for trajectory_id, group in grouped.items():
        hashes = {row.get("measurement_sha256") for row in group}
        methods = {row["method"] for row in group}
        if (
            len(group) != len(SEALED_METHODS)
            or methods != set(SEALED_METHODS)
            or len(hashes) != 1
            or None in hashes
        ):
            bad.append(trajectory_id)
    return {
        "trajectory_count": len(grouped),
        "identical_stream_trajectory_count": len(grouped) - len(bad),
        "mismatch_count": len(bad),
        "mismatched_trajectory_ids": sorted(bad),
        "pass": not bad and len(grouped) == EXPECTED_SEALED_TRAJECTORIES,
    }


def _aggregate_statistics(
    paths: Paths,
    rows: Sequence[Mapping[str, Any]],
    selected_sha256: str,
    sealed_path: Path,
) -> dict[str, Any]:
    flat = _flat_rows(rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_aggregate_statistics",
        "trajectory_result_count": len(rows),
        "unique_trajectory_count": len({row["trajectory_id"] for row in rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "measurement_identity": _validate_measurement_identity(rows),
        "source_hashes": {
            "sealed_results": sha256_file(sealed_path),
            "selected_configs": selected_sha256,
            "sealed_manifest": sha256_file(paths.frozen / "sealed_seed_manifest.jsonl"),
        },
        "by_condition_method": metric_lib.aggregate_trajectory_rows(
            flat,
            group_by=("symmetry_class", "motion_family", "stress_family", "method"),
        ),
        "by_symmetry_method": metric_lib.aggregate_trajectory_rows(
            flat, group_by=("symmetry_class", "method")
        ),
        "by_method": metric_lib.aggregate_trajectory_rows(flat, group_by=("method",)),
        "asymmetric_guardrails": metric_lib.aggregate_asymmetric_rows(
            flat, group_by=("method",)
        ),
    }
    write_json_atomic(paths.artifacts / "aggregate_statistics.json", payload)
    return payload


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    symmetries: set[str] | None = None,
    motions: set[str] | None = None,
    stresses: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and row["status"] == "success"
        and (symmetries is None or row["symmetry_class"] in symmetries)
        and (motions is None or row["motion_family"] in motions)
        and (stresses is None or row["stress_family"] in stresses)
    ]


def _relative_reduction(
    proposed: float | None, comparator: float | None, minimum: float
) -> dict[str, Any]:
    if proposed is None or comparator is None:
        return {
            "value": None,
            "threshold": minimum,
            "comparator_zero": False,
            "pass": False,
        }
    if comparator <= 0.0:
        return {
            "value": None,
            "threshold": minimum,
            "comparator_zero": True,
            "pass": False,
        }
    value = (comparator - proposed) / comparator
    return {
        "value": float(value),
        "threshold": float(minimum),
        "comparator_zero": False,
        "pass": bool(value >= minimum),
    }


def _relative_degradation(
    proposed: float | None, comparator: float | None, maximum: float
) -> dict[str, Any]:
    if proposed is None or comparator is None:
        return {
            "value": None,
            "threshold": maximum,
            "comparator_zero": False,
            "pass": False,
        }
    if comparator <= 0.0:
        return {
            "value": 0.0 if proposed <= 0.0 else None,
            "threshold": maximum,
            "comparator_zero": True,
            "pass": bool(proposed <= 0.0),
        }
    value = (proposed - comparator) / comparator
    return {
        "value": float(value),
        "threshold": float(maximum),
        "comparator_zero": False,
        "pass": bool(value <= maximum),
    }


def _jump_total(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(
        sum(
            float(
                _path_value(
                    row,
                    "metrics.representation_stability.catastrophic_representation_jump_count",
                )
                or 0
            )
            + float(
                _path_value(
                    row, "metrics.representation_stability.axial_gauge_jump_count"
                )
                or 0
            )
            for row in rows
        )
    )


def _rms(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _max_lag_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        translation = _path_value(
            row, "metrics.dynamic_guardrails.translation_direction_change_lag_frames"
        )
        rotation = _path_value(
            row,
            "metrics.dynamic_guardrails.observable_rotation_direction_change_lag_frames",
        )
        if translation is not None and rotation is not None:
            values.append(float(max(translation, rotation)))
    return values


def _acceptance(
    paths: Paths,
    rows: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    selected_sha256: str,
) -> dict[str, Any]:
    jump_conditions = []
    for symmetry in SYMMETRIC_CLASSES:
        for stress in SWITCH_STRESSES:
            filters = {symmetry}, {stress}
            proposed = _jump_total(
                _filter_rows(
                    rows, "SYMQUOT_CT", symmetries=filters[0], stresses=filters[1]
                )
            )
            standard = _jump_total(
                _filter_rows(
                    rows, "STANDARD_SE3_CT", symmetries=filters[0], stresses=filters[1]
                )
            )
            nearest = _jump_total(
                _filter_rows(
                    rows,
                    "NEAREST_REPRESENTATIVE_CT",
                    symmetries=filters[0],
                    stresses=filters[1],
                )
            )
            vs_standard = _relative_reduction(proposed, standard, 0.80)
            vs_nearest = _relative_reduction(proposed, nearest, 0.30)
            jump_conditions.append(
                {
                    "symmetry_class": symmetry,
                    "stress_family": stress,
                    "jump_totals": {
                        "SYMQUOT_CT": proposed,
                        "STANDARD_SE3_CT": standard,
                        "NEAREST_REPRESENTATIVE_CT": nearest,
                    },
                    "reduction_vs_standard": vs_standard,
                    "reduction_vs_nearest": vs_nearest,
                    "pass": bool(vs_standard["pass"] and vs_nearest["pass"]),
                }
            )
    jump_gate = all(condition["pass"] for condition in jump_conditions)

    combined_values: dict[str, float | None] = {}
    for method in TEMPORAL_METHODS:
        subset = _filter_rows(
            rows,
            method,
            symmetries=set(SYMMETRIC_CLASSES),
            stresses={"COMBINED"},
        )
        combined_values[method] = _median(
            _finite_values(subset, "metrics.accuracy.median_quotient_pose_error")
        )
    combined_vs_standard = _relative_reduction(
        combined_values["SYMQUOT_CT"], combined_values["STANDARD_SE3_CT"], 0.15
    )
    combined_vs_nearest = _relative_degradation(
        combined_values["SYMQUOT_CT"],
        combined_values["NEAREST_REPRESENTATIVE_CT"],
        0.03,
    )
    accuracy_gate = bool(combined_vs_standard["pass"] and combined_vs_nearest["pass"])

    dropout_values: dict[str, float | None] = {}
    for method in ("STANDARD_SE3_CT", "SYMQUOT_CT"):
        subset = _filter_rows(
            rows,
            method,
            symmetries=set(SYMMETRIC_CLASSES),
            stresses={"DROPOUT_10"},
        )
        dropout_values[method] = _median(
            _finite_values(
                subset,
                "metrics.dropout.first_accepted_post_dropout_output_error.median",
            )
        )
    dropout_vs_standard = _relative_reduction(
        dropout_values["SYMQUOT_CT"], dropout_values["STANDARD_SE3_CT"], 0.20
    )
    dropout_gate = bool(dropout_vs_standard["pass"])

    outlier_values: dict[str, Any] = {}
    for method in SEALED_METHODS:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["stress_family"] in {"OUTLIER_BURST", "COMBINED"}
        ]
        successful = [row for row in subset if row["status"] == "success"]
        accepted = _finite_values(successful, "metrics.outlier.outliers_accepted")
        rejected = _finite_values(successful, "metrics.outlier.outliers_rejected")
        recovery_failures = _finite_values(
            successful, "metrics.outlier.recovery_failure_count"
        )
        accepted_total = float(sum(accepted))
        rejected_total = float(sum(rejected))
        injected_total = accepted_total + rejected_total
        evaluation_failures = len(subset) - len(successful)
        recovery_failure_count = int(sum(recovery_failures)) + evaluation_failures
        outlier_values[method] = {
            "trajectory_count": len(subset),
            "evaluation_failure_count": evaluation_failures,
            "accepted_outlier_frame_count": accepted_total,
            "rejected_outlier_frame_count": rejected_total,
            "injected_frame_rejection_rate": (
                None if injected_total == 0.0 else rejected_total / injected_total
            ),
            "mean_contamination_duration_frames": _mean(
                _finite_values(
                    successful, "metrics.outlier.contamination_duration_frames"
                )
            ),
            "median_maximum_error_after_onset": _median(
                _finite_values(
                    successful,
                    "metrics.outlier.maximum_error_after_outlier_onset",
                )
            ),
            "median_recovery_frames_after_burst": _median(
                _finite_values(
                    successful,
                    "metrics.outlier.frames_until_recovery_after_burst.median",
                )
            ),
            "recovery_failure_count": recovery_failure_count,
            "recovery_failure_rate": (
                None if not subset else recovery_failure_count / len(subset)
            ),
        }

    asymmetric_values: dict[str, Any] = {}
    for method in ("STANDARD_SE3_CT", "SYMQUOT_CT"):
        subset = _filter_rows(rows, method, symmetries={"ASYMMETRIC"})
        reversal = [
            row for row in subset if row["motion_family"] == "DIRECTION_REVERSAL"
        ]
        asymmetric_values[method] = {
            "translation_rmse_m": _rms(
                _finite_values(
                    subset, "metrics.asymmetric_guardrails.translation_rmse_m"
                )
            ),
            "rotation_rmse_deg": _rms(
                _finite_values(
                    subset, "metrics.asymmetric_guardrails.rotation_rmse_deg"
                )
            ),
            "median_max_direction_lag_frames": _median(_max_lag_values(reversal)),
        }
    asym_translation = _relative_degradation(
        asymmetric_values["SYMQUOT_CT"]["translation_rmse_m"],
        asymmetric_values["STANDARD_SE3_CT"]["translation_rmse_m"],
        0.03,
    )
    asym_rotation = _relative_degradation(
        asymmetric_values["SYMQUOT_CT"]["rotation_rmse_deg"],
        asymmetric_values["STANDARD_SE3_CT"]["rotation_rmse_deg"],
        0.03,
    )
    asym_lag_value = None
    asym_lag_pass = False
    if (
        asymmetric_values["SYMQUOT_CT"]["median_max_direction_lag_frames"] is not None
        and asymmetric_values["STANDARD_SE3_CT"]["median_max_direction_lag_frames"]
        is not None
    ):
        asym_lag_value = (
            asymmetric_values["SYMQUOT_CT"]["median_max_direction_lag_frames"]
            - asymmetric_values["STANDARD_SE3_CT"]["median_max_direction_lag_frames"]
        )
        asym_lag_pass = asym_lag_value <= 1.0
    asymmetric_gate = bool(
        asym_translation["pass"] and asym_rotation["pass"] and asym_lag_pass
    )

    dynamic_values = {}
    for method in TEMPORAL_METHODS:
        subset = _filter_rows(
            rows,
            method,
            symmetries=set(SYMMETRIC_CLASSES),
            motions={"DIRECTION_REVERSAL"},
        )
        dynamic_values[method] = _median(_max_lag_values(subset))
    baseline_lags = [
        dynamic_values[method]
        for method in ("STANDARD_SE3_CT", "NEAREST_REPRESENTATIVE_CT")
        if dynamic_values[method] is not None
    ]
    best_baseline_lag = None if not baseline_lags else min(baseline_lags)
    dynamic_worsening = (
        None
        if best_baseline_lag is None or dynamic_values["SYMQUOT_CT"] is None
        else dynamic_values["SYMQUOT_CT"] - best_baseline_lag
    )
    dynamic_gate = dynamic_worsening is not None and dynamic_worsening <= 1.0

    continuous_values = {}
    for method in ("STANDARD_SE3_CT", "SYMQUOT_CT"):
        subset = _filter_rows(
            rows,
            method,
            symmetries={"CONTINUOUS_AXIAL"},
            stresses=set(SWITCH_STRESSES),
        )
        continuous_values[method] = _rms(
            _finite_values(
                subset,
                "metrics.representation_stability.unobservable_axial_angular_velocity_rms_rad_s",
            )
        )
    axial_reduction = _relative_reduction(
        continuous_values["SYMQUOT_CT"], continuous_values["STANDARD_SE3_CT"], 0.80
    )
    continuous_failures = [
        row
        for row in rows
        if row["method"] == "SYMQUOT_CT"
        and row["symmetry_class"] == "CONTINUOUS_AXIAL"
        and (
            row["status"] != "success"
            or float(
                _path_value(
                    row,
                    "metrics.runtime_and_finite_counts.nonfinite_output_value_count",
                )
                or 0
            )
            > 0
        )
    ]
    continuous_gate = bool(axial_reduction["pass"] and not continuous_failures)

    symquot_rows = [row for row in rows if row["method"] == "SYMQUOT_CT"]
    failed_symquot = sum(row["status"] != "success" for row in symquot_rows)
    nonfinite_symquot = sum(
        int(
            _path_value(
                row, "metrics.runtime_and_finite_counts.nonfinite_output_value_count"
            )
            or 0
        )
        for row in symquot_rows
        if row["status"] == "success"
    )
    nonfinite_symquot_uncertainty = sum(
        int(
            _path_value(
                row,
                "metrics.runtime_and_finite_counts.nonfinite_uncertainty_value_count",
            )
            or 0
        )
        for row in symquot_rows
        if row["status"] == "success"
    )
    runtimes = [
        float(value)
        for row in symquot_rows
        for value in row.get("frame_runtime_s", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    p95_runtime_s = None if not runtimes else float(np.percentile(runtimes, 95))
    measurement_identity = bool(aggregates["measurement_identity"]["pass"])
    numerical_runtime_gate = bool(
        failed_symquot == 0
        and nonfinite_symquot == 0
        and nonfinite_symquot_uncertainty == 0
        and p95_runtime_s is not None
        and p95_runtime_s < 0.010
        and measurement_identity
    )

    algorithm_go = bool(
        jump_gate
        and accuracy_gate
        and dropout_gate
        and asymmetric_gate
        and dynamic_gate
        and continuous_gate
        and numerical_runtime_gate
    )
    mechanism_core = bool(
        jump_gate
        and asymmetric_gate
        and dynamic_gate
        and continuous_gate
        and numerical_runtime_gate
    )
    mechanism_value = bool(accuracy_gate or dropout_gate)
    if algorithm_go:
        classification = "ALGORITHM GO"
    elif mechanism_core and mechanism_value:
        classification = "MECHANISM PASS / REAL-DATA VALUE UNPROVEN"
    else:
        classification = "NO-GO"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_acceptance",
        "classification": classification,
        "all_thresholds_predeclared": True,
        "sealed_tuning_performed": False,
        "zero_comparator_rule": (
            "Requested relative reductions fail when comparator is zero, including zero versus zero."
        ),
        "source_hashes": {
            "sealed_results": aggregates["source_hashes"]["sealed_results"],
            "aggregate_statistics": sha256_file(
                paths.artifacts / "aggregate_statistics.json"
            ),
            "selected_configs": selected_sha256,
            "decision_thresholds": sha256_file(
                paths.frozen / "decision_thresholds.json"
            ),
        },
        "criteria": {
            "symmetric_switch_jump_reduction": {
                "conditions": jump_conditions,
                "pass": jump_gate,
            },
            "symmetric_combined_accuracy": {
                "median_errors": combined_values,
                "improvement_vs_standard": combined_vs_standard,
                "degradation_vs_nearest": combined_vs_nearest,
                "pass": accuracy_gate,
            },
            "symmetric_dropout_10": {
                "median_first_accepted_errors": dropout_values,
                "improvement_vs_standard": dropout_vs_standard,
                "pass": dropout_gate,
            },
            "outlier_diagnostic": {
                "formal_classification_gate": False,
                "scope": "OUTLIER_BURST and COMBINED across all symmetry classes",
                "values": outlier_values,
            },
            "asymmetric_guardrails": {
                "values": asymmetric_values,
                "translation_degradation": asym_translation,
                "rotation_degradation": asym_rotation,
                "lag_worsening_frames": asym_lag_value,
                "pass": asymmetric_gate,
            },
            "dynamic_symmetric_guardrail": {
                "median_max_lag_frames": dynamic_values,
                "best_non_oracle_baseline_lag_frames": best_baseline_lag,
                "symquot_worsening_frames": dynamic_worsening,
                "pass": dynamic_gate,
            },
            "continuous_observability": {
                "axial_velocity_rms_rad_s": continuous_values,
                "reduction_vs_standard": axial_reduction,
                "track_explosion_count": len(continuous_failures),
                "unobservable_yaw_accuracy_claimed": False,
                "pass": continuous_gate,
            },
            "numerical_runtime_and_contract": {
                "failed_symquot_trajectory_count": failed_symquot,
                "nonfinite_symquot_output_value_count": nonfinite_symquot,
                "nonfinite_symquot_uncertainty_value_count": (
                    nonfinite_symquot_uncertainty
                ),
                "p95_symquot_cpu_latency_s": p95_runtime_s,
                "p95_limit_s": 0.010,
                "hidden_truth_access": False,
                "identical_measurement_streams": measurement_identity,
                "pass": numerical_runtime_gate,
            },
        },
        "mechanism_core_pass": mechanism_core,
        "mechanism_value_gate_pass": mechanism_value,
    }
    write_json_atomic(paths.artifacts / "acceptance.json", payload)
    return payload


def sealed(paths: Paths) -> dict[str, Any]:
    frozen_receipt = _verify_frozen(paths)
    selected, selected_hash = _selected_configs(paths)
    manifest = load_jsonl(paths.frozen / "sealed_seed_manifest.jsonl")
    if len(manifest) != EXPECTED_SEALED_TRAJECTORIES:
        raise ValueError("Sealed manifest does not contain 2304 trajectories")
    expected = len(manifest) * len(SEALED_METHODS)
    final_path = paths.artifacts / "sealed_results.jsonl"
    partial_path = paths.artifacts / "sealed_results.partial.jsonl"
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    chronology_path = paths.artifacts / "sealed_evaluation_receipt.json"
    if not chronology_path.exists():
        if final_path.exists() or partial_path.exists():
            raise RuntimeError(
                "Sealed results exist without the pre-evaluation chronology receipt"
            )
        chronology = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "m5_g0_sealed_evaluation_receipt",
            "sealed_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_at_utc": selected["selected_at_utc"],
            "selected_configs_sha256": selected_hash,
            "sealed_manifest_sha256": sha256_file(
                paths.frozen / "sealed_seed_manifest.jsonl"
            ),
            "manifest_hash_receipt_sha256": sha256_file(
                paths.frozen / "manifest_hashes.json"
            ),
            "frozen_at_utc": frozen_receipt["frozen_at_utc"],
            "source_hashes": {
                "core": sha256_file(paths.repo_root / "scripts" / "m5_g0_core.py"),
                "metrics": sha256_file(
                    paths.repo_root / "scripts" / "m5_g0_metrics.py"
                ),
                "runner": sha256_file(Path(__file__).resolve()),
            },
            "sealed_tuning_allowed": False,
        }
        if not (
            chronology["frozen_at_utc"]
            <= chronology["selected_at_utc"]
            <= chronology["sealed_started_at_utc"]
        ):
            raise ValueError("M5-G0 freeze/select/sealed chronology is invalid")
        write_json_atomic(chronology_path, chronology)
    else:
        chronology = read_json(chronology_path)
        if chronology["selected_configs_sha256"] != selected_hash:
            raise ValueError("Selected configurations changed after sealed start")
        if chronology["sealed_manifest_sha256"] != sha256_file(
            paths.frozen / "sealed_seed_manifest.jsonl"
        ):
            raise ValueError("Sealed manifest changed after sealed start")
    if not final_path.exists():
        _, complete = _load_partial(partial_path)
        for manifest_row in manifest:
            method_configs: list[tuple[str, str, Mapping[str, Any]]] = []
            raw_config_id = "raw_hold_no_tuning"
            raw_config = selected["selected"]["STANDARD_SE3_CT"]["configuration"]
            method_configs.append(("RAW_HOLD", raw_config_id, raw_config))
            for method in TEMPORAL_METHODS:
                choice = selected["selected"][method]
                method_configs.append(
                    (method, str(choice["config_id"]), choice["configuration"])
                )
            missing = [
                item
                for item in method_configs
                if (
                    str(manifest_row["trajectory_id"]),
                    item[0],
                    item[1],
                )
                not in complete
            ]
            if not missing:
                continue
            measurement_hash: str | None = None
            try:
                _, corrupted, arrays, measurement_hash = _generate(manifest_row)
            except Exception as exc:
                for method, config_id, _ in missing:
                    append_jsonl_durable(
                        partial_path,
                        _failure_row(
                            "sealed",
                            manifest_row,
                            method,
                            config_id,
                            None,
                            exc,
                        ),
                    )
                continue
            for method, config_id, config_values in missing:
                try:
                    execution = _execute(
                        manifest_row,
                        corrupted,
                        arrays,
                        measurement_hash,
                        method,
                        config_id,
                        config_values,
                    )
                    row = _success_row(
                        "sealed", manifest_row, method, config_id, execution
                    )
                except Exception as exc:
                    row = _failure_row(
                        "sealed",
                        manifest_row,
                        method,
                        config_id,
                        measurement_hash,
                        exc,
                    )
                append_jsonl_durable(partial_path, row)
    rows = _complete_results(final_path, partial_path, expected)
    aggregates = _aggregate_statistics(paths, rows, selected_hash, final_path)
    return _acceptance(paths, rows, aggregates, selected_hash)


def _representative_manifest_rows(
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    definitions = {
        "c2_representation_switch": ("C2", "CONSTANT_TWIST", "REPRESENTATION_SWITCH"),
        "continuous_representation_switch": (
            "CONTINUOUS_AXIAL",
            "CONSTANT_TWIST",
            "REPRESENTATION_SWITCH",
        ),
        "c4_dropout_10": ("C4", "CONSTANT_TWIST", "DROPOUT_10"),
        "c2_outlier_burst": ("C2", "CONSTANT_TWIST", "OUTLIER_BURST"),
        "c4_direction_reversal": ("C4", "DIRECTION_REVERSAL", "NOMINAL"),
    }
    selected = {}
    for label, condition in definitions.items():
        candidates = sorted(
            (
                row
                for row in manifest
                if (
                    row["symmetry_class"],
                    row["motion_family"],
                    row["stress_family"],
                )
                == condition
            ),
            key=lambda row: (int(row["trajectory_seed"]), str(row["trajectory_id"])),
        )
        if not candidates:
            raise ValueError(f"No sealed representative candidates for {label}")
        selected[label] = candidates[len(candidates) // 2]
    return selected


def _yaw_degrees(poses: Sequence[Any]) -> np.ndarray:
    result = np.full(len(poses), np.nan, dtype=np.float64)
    for index, pose in enumerate(poses):
        if pose is None:
            continue
        rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
        if np.isfinite(rotation).all():
            result[index] = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return np.unwrap(np.radians(result)) * 180.0 / math.pi


def _relative_axial_gauge_degrees(gt: np.ndarray, poses: Sequence[Any]) -> np.ndarray:
    result = np.full(len(gt), np.nan, dtype=np.float64)
    for index, pose in enumerate(poses):
        if pose is None:
            continue
        rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
        relative = gt[index, :3, :3].T @ rotation
        result[index] = math.degrees(math.atan2(relative[1, 0], relative[0, 0]))
    return np.unwrap(np.radians(result)) * 180.0 / math.pi


def _trace_execution(
    row: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    _, corrupted, arrays, digest = _generate(row)
    methods: dict[str, Any] = {}
    for method in SEALED_METHODS:
        if method == "RAW_HOLD":
            config_id = "raw_hold_no_tuning"
            config_values = selected["selected"]["STANDARD_SE3_CT"]["configuration"]
        else:
            choice = selected["selected"][method]
            config_id = choice["config_id"]
            config_values = choice["configuration"]
        execution = _execute(
            row,
            corrupted,
            arrays,
            digest,
            method,
            config_id,
            config_values,
        )
        # Recompute frames only for five frozen representative traces.
        output = execution["output_arrays"]
        spec = _symmetry_spec(str(row["symmetry_class"]))
        frame_metrics = metric_lib.compute_trajectory_metrics(
            arrays["timestamps"],
            arrays["gt_poses"],
            arrays["measurements"],
            output["poses"],
            output["accepted"],
            output["residuals"],
            output["uncertainty"],
            execution["frame_runtime_s"],
            spec,
            evaluator_intervals=_interval_payload(corrupted, str(row["motion_family"])),
            include_frame_metrics=True,
        )
        methods[method] = {
            "config_id": config_id,
            "poses": output["poses"],
            "accepted": output["accepted"],
            "residuals": output["residuals"],
            "uncertainty": output["uncertainty"],
            "metrics": frame_metrics,
        }
    return {
        "manifest": dict(row),
        "timestamps": arrays["timestamps"],
        "gt_poses": arrays["gt_poses"],
        "measurements": arrays["measurements"],
        "measurement_sha256": digest,
        "intervals": _interval_payload(corrupted, str(row["motion_family"])),
        "methods": methods,
    }


def _plot_representatives(paths: Paths, traces: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.reports.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    colors = {
        "RAW_HOLD": "#777777",
        "STANDARD_SE3_CT": "#d95f02",
        "NEAREST_REPRESENTATIVE_CT": "#1b9e77",
        "SYMQUOT_CT": "#386cb0",
    }

    trace = traces["c2_representation_switch"]
    times = trace["timestamps"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        times, _yaw_degrees(trace["gt_poses"]), label="GT canonical", color="black"
    )
    axes[0].plot(
        times,
        _yaw_degrees(trace["measurements"]),
        label="measurement representation",
        alpha=0.55,
    )
    for method, payload in trace["methods"].items():
        axes[0].plot(
            times,
            _yaw_degrees(payload["poses"]),
            label=method,
            color=colors[method],
            linewidth=1.2,
        )
        error = [
            frame["normalized_pose_error"]
            for frame in payload["metrics"]["frame_metrics"]
        ]
        axes[1].plot(times, error, label=method, color=colors[method])
        jumps = [
            frame["catastrophic_jump"] for frame in payload["metrics"]["frame_metrics"]
        ]
        indices = np.flatnonzero(jumps)
        if len(indices):
            axes[1].scatter(
                times[indices],
                np.asarray(error)[indices],
                color=colors[method],
                marker="x",
            )
    axes[0].set_ylabel("raw yaw gauge (deg)")
    axes[1].set_ylabel("normalized quotient error")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = paths.reports / "timeline_c2_representation_switch.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    trace = traces["continuous_representation_switch"]
    times = trace["timestamps"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(
        times,
        _relative_axial_gauge_degrees(trace["gt_poses"], trace["measurements"]),
        label="input gauge",
    )
    for method, payload in trace["methods"].items():
        frames = payload["metrics"]["frame_metrics"]
        axes[0].plot(
            times,
            _relative_axial_gauge_degrees(trace["gt_poses"], payload["poses"]),
            label=method,
            color=colors[method],
        )
        axes[1].plot(
            times,
            [frame["rotation_error_deg"] for frame in frames],
            label=method,
            color=colors[method],
        )
        axes[2].plot(
            times,
            [frame["axial_angular_velocity_rad_s"] for frame in frames],
            label=method,
            color=colors[method],
        )
    axes[0].set_ylabel("axial gauge (deg)")
    axes[1].set_ylabel("axis error (deg)")
    axes[2].set_ylabel("unobs. axial velocity (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = paths.reports / "timeline_continuous_axial.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    trace = traces["c4_dropout_10"]
    times = trace["timestamps"]
    availability = np.asarray(
        [pose is not None for pose in trace["measurements"]], dtype=float
    )
    dropout_end = int(trace["intervals"]["dropout_intervals"][0][1])
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].step(times, availability, where="post", color="black")
    for method, payload in trace["methods"].items():
        frames = payload["metrics"]["frame_metrics"]
        axes[1].plot(
            times,
            [frame["normalized_pose_error"] for frame in frames],
            label=method,
            color=colors[method],
        )
        axes[2].plot(
            times,
            [frame["uncertainty_scalar"] for frame in frames],
            label=method,
            color=colors[method],
        )
        accepted = np.asarray(payload["accepted"], dtype=bool)
        axes[3].step(
            times,
            accepted.astype(float),
            where="post",
            label=method,
            color=colors[method],
        )
        reacquired = np.flatnonzero(accepted[dropout_end:])
        if len(reacquired):
            frame = dropout_end + int(reacquired[0])
            axes[3].scatter(
                times[frame],
                1.0,
                color=colors[method],
                marker="o",
                s=24,
                zorder=5,
            )
    axes[0].set_ylabel("measurement")
    axes[1].set_ylabel("quotient error")
    axes[2].set_ylabel("uncertainty")
    axes[3].set_ylabel("accepted")
    axes[3].set_xlabel("time (s)")
    axes[3].set_yticks([0.0, 1.0])
    axes[1].legend(ncol=2, fontsize=8)
    axes[3].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = paths.reports / "timeline_dropout_10.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    trace = traces["c2_outlier_burst"]
    times = trace["timestamps"]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for method, payload in trace["methods"].items():
        axes[0].plot(times, payload["residuals"], label=method, color=colors[method])
        axes[1].step(
            times,
            payload["accepted"].astype(float),
            where="post",
            label=method,
            color=colors[method],
        )
        axes[2].plot(
            times,
            [
                frame["normalized_pose_error"]
                for frame in payload["metrics"]["frame_metrics"]
            ],
            label=method,
            color=colors[method],
        )
    axes[0].set_ylabel("innovation")
    axes[1].set_ylabel("accepted")
    axes[2].set_ylabel("quotient error")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = paths.reports / "timeline_outlier_burst.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    trace = traces["c4_direction_reversal"]
    times = trace["timestamps"]
    spec = _symmetry_spec("C4")
    gt_velocity, _ = metric_lib.observable_velocity_series(
        times, trace["gt_poses"], spec
    )
    pre = np.nanmean(gt_velocity[50:60], axis=0)
    direction = pre / max(np.linalg.norm(pre), 1e-12)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, gt_velocity @ direction, label="GT", color="black", linewidth=2)
    for method, payload in trace["methods"].items():
        velocity, _ = metric_lib.observable_velocity_series(
            times, payload["poses"], spec
        )
        lag = payload["metrics"]["dynamic_guardrails"][
            "translation_direction_change_lag_frames"
        ]
        axis.plot(
            times,
            velocity @ direction,
            label=f"{method} (lag={lag})",
            color=colors[method],
        )
    axis.axvline(times[60], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("signed translation velocity (m/s)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = paths.reports / "timeline_direction_reversal.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)
    return created


def _plot_aggregates(paths: Paths, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[str] = []
    methods = list(SEALED_METHODS)
    symmetries = ["ASYMMETRIC", *SYMMETRIC_CLASSES]

    figure, axes = plt.subplots(1, len(symmetries), figsize=(14, 4), sharey=True)
    for axis, symmetry in zip(axes, symmetries):
        data = [
            _finite_values(
                _filter_rows(rows, method, symmetries={symmetry}),
                "metrics.accuracy.median_quotient_pose_error",
            )
            for method in methods
        ]
        axis.boxplot(data, showfliers=False)
        axis.set_title(symmetry)
        axis.set_xticks(
            range(1, len(methods) + 1),
            [method.replace("_CT", "") for method in methods],
            rotation=65,
            fontsize=7,
        )
    axes[0].set_ylabel("trajectory median quotient error")
    figure.tight_layout()
    path = paths.reports / "aggregate_quotient_error_by_symmetry.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    jump_totals = [
        _jump_total(
            _filter_rows(
                rows,
                method,
                symmetries=set(SYMMETRIC_CLASSES),
                stresses=set(SWITCH_STRESSES),
            )
        )
        or 0
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(methods, jump_totals)
    axis.set_ylabel("catastrophic + gauge jumps")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path = paths.reports / "aggregate_jump_counts.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    dropout = [
        _median(
            _finite_values(
                _filter_rows(
                    rows,
                    method,
                    symmetries=set(SYMMETRIC_CLASSES),
                    stresses={"DROPOUT_10"},
                ),
                "metrics.dropout.frames_to_return_within_nominal.median",
            )
        )
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(methods, [math.nan if value is None else value for value in dropout])
    axis.set_ylabel("median recovery frames")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path = paths.reports / "aggregate_dropout_recovery.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    translation = []
    rotation = []
    for method in methods:
        subset = _filter_rows(rows, method, symmetries={"ASYMMETRIC"})
        translation.append(
            1000.0
            * (
                _rms(
                    _finite_values(
                        subset, "metrics.asymmetric_guardrails.translation_rmse_m"
                    )
                )
                or math.nan
            )
        )
        rotation.append(
            _rms(
                _finite_values(
                    subset, "metrics.asymmetric_guardrails.rotation_rmse_deg"
                )
            )
            or math.nan
        )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(methods, translation)
    axes[0].set_ylabel("translation RMSE (mm)")
    axes[1].bar(methods, rotation)
    axes[1].set_ylabel("rotation RMSE (deg)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    path = paths.reports / "aggregate_asymmetric_guardrail.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)

    runtimes = [
        [
            1000.0 * float(value)
            for row in rows
            if row["method"] == method
            for value in row.get("frame_runtime_s", [])
        ]
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot(runtimes, showfliers=False)
    axis.axhline(10.0, color="red", linestyle="--", label="10 ms gate")
    axis.set_xticks(range(1, len(methods) + 1), methods, rotation=30)
    axis.set_ylabel("CPU estimator latency (ms/frame)")
    axis.legend()
    figure.tight_layout()
    path = paths.reports / "aggregate_runtime_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)
    return created


def _trace_json(traces: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for label, trace in traces.items():
        payload[label] = {
            "manifest": trace["manifest"],
            "measurement_sha256": trace["measurement_sha256"],
            "timestamps_s": trace["timestamps"].tolist(),
            "intervals": trace["intervals"],
            "methods": {
                method: {
                    "config_id": method_trace["config_id"],
                    "accepted": [bool(value) for value in method_trace["accepted"]],
                    "frame_metrics": method_trace["metrics"]["frame_metrics"],
                }
                for method, method_trace in trace["methods"].items()
            },
        }
    return payload


def _anti_cherry_answers(acceptance: Mapping[str, Any]) -> list[tuple[str, str]]:
    criteria = acceptance["criteria"]
    outlier = criteria["outlier_diagnostic"]["values"]

    def formatted(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    outlier_summary = "; ".join(
        f"{method}: rejection={formatted(value['injected_frame_rejection_rate'])}, "
        "mean contamination="
        f"{formatted(value['mean_contamination_duration_frames'])} frames, "
        f"recovery failures={value['recovery_failure_count']} "
        f"({formatted(value['recovery_failure_rate'])})"
        for method, value in outlier.items()
    )
    return [
        (
            "Would nearest-equivalent representation achieve the same result?",
            "The sealed per-condition comparison against NEAREST_REPRESENTATIVE_CT "
            + (
                "met the 30% incremental jump gate."
                if criteria["symmetric_switch_jump_reduction"]["pass"]
                else "did not establish the required incremental benefit."
            ),
        ),
        (
            "Physical quotient accuracy or only coordinate continuity?",
            "The combined-stress quotient-accuracy gate "
            + (
                "passed."
                if criteria["symmetric_combined_accuracy"]["pass"]
                else "failed; continuity alone is insufficient."
            ),
        ),
        (
            "Useful through dropout and outliers?",
            "The formal ten-frame dropout gate "
            + ("passed. " if criteria["symmetric_dropout_10"]["pass"] else "failed. ")
            + "The threshold-free sealed outlier diagnostic (not a classification "
            "gate) is: " + outlier_summary + ".",
        ),
        (
            "Material observable-motion lag?",
            "The dynamic symmetric lag guardrail "
            + (
                "passed."
                if criteria["dynamic_symmetric_guardrail"]["pass"]
                else "failed."
            ),
        ),
        (
            "Asymmetric objects preserved?",
            "The asymmetric translation, rotation, and lag guardrails "
            + ("passed." if criteria["asymmetric_guardrails"]["pass"] else "failed."),
        ),
        (
            "Continuous yaw treated as unobservable?",
            "Yes. Accuracy uses transformed-axis direction only; arbitrary axial gauge is diagnosed, never scored as physical yaw.",
        ),
        (
            "Temporally structured representation switching?",
            "Yes. The frozen corruption is a persistent seeded Markov process with minimum dwell, not independent labels.",
        ),
        (
            "Survives a sealed seed set?",
            f"The mechanical sealed classification is {acceptance['classification']} over the complete retained result set.",
        ),
        (
            "Synthetic upper-bound evidence?",
            "All evidence in M5-G0 is synthetic mechanism evidence; none is real-video, FoundationPose, BOP benchmark, robot, or hardware evidence.",
        ),
        (
            "Evidence still needed for a real-video claim?",
            "A frozen replay on real pose-predictor streams with timestamped dropouts/outliers, object symmetries, labeled pose truth, sealed thresholds, and latency measurements.",
        ),
    ]


def _report_markdown(
    acceptance: Mapping[str, Any],
    selected: Mapping[str, Any],
    plots: Sequence[str],
    sealed_rows: Sequence[Mapping[str, Any]],
) -> str:
    criteria = acceptance["criteria"]

    def number(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.{digits}f}"

    config_lines = [
        f"| `{method}` | `{entry['config_id']}` | {entry['objective']['objective']:.6f} |"
        for method, entry in selected["selected"].items()
    ]
    jump_lines = []
    for condition in criteria["symmetric_switch_jump_reduction"]["conditions"]:
        totals = condition["jump_totals"]
        jump_lines.append(
            "| {sym} | {stress} | {standard} | {nearest} | {proposed} | {status} |".format(
                sym=condition["symmetry_class"],
                stress=condition["stress_family"],
                standard=totals["STANDARD_SE3_CT"],
                nearest=totals["NEAREST_REPRESENTATIVE_CT"],
                proposed=totals["SYMQUOT_CT"],
                status="PASS" if condition["pass"] else "FAIL",
            )
        )
    answers = "\n\n".join(
        f"{index}. **{question}** {answer}"
        for index, (question, answer) in enumerate(
            _anti_cherry_answers(acceptance), start=1
        )
    )
    failures = Counter(
        row["method"] for row in sealed_rows if row["status"] != "success"
    )
    overall_lines = []
    dropout_lines = []
    outlier_lines = []
    for method in SEALED_METHODS:
        method_rows = _filter_rows(sealed_rows, method)
        switch_rows = _filter_rows(
            sealed_rows,
            method,
            symmetries=set(SYMMETRIC_CLASSES),
            stresses=set(SWITCH_STRESSES),
        )
        runtime_values = [
            1000.0 * float(value)
            for row in method_rows
            for value in row.get("frame_runtime_s", [])
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        overall_lines.append(
            "| {method} | {qpose} | {translation} | {rotation} | {jumps} | "
            "{latency} | {failures} |".format(
                method=method,
                qpose=number(
                    _median(
                        _finite_values(
                            method_rows,
                            "metrics.accuracy.median_quotient_pose_error",
                        )
                    )
                ),
                translation=number(
                    1000.0
                    * (
                        _rms(
                            _finite_values(
                                method_rows, "metrics.accuracy.translation.rmse"
                            )
                        )
                        or math.nan
                    ),
                    3,
                ),
                rotation=number(
                    _rms(_finite_values(method_rows, "metrics.accuracy.rotation.rmse")),
                    3,
                ),
                jumps=number(_jump_total(switch_rows), 0),
                latency=number(
                    None
                    if not runtime_values
                    else float(np.percentile(runtime_values, 95)),
                    3,
                ),
                failures=failures.get(method, 0),
            )
        )
        dropout_rows = _filter_rows(
            sealed_rows,
            method,
            symmetries=set(SYMMETRIC_CLASSES),
            stresses={"DROPOUT_10"},
        )
        dropout_lines.append(
            "| {method} | {final} | {accepted} | {recovery} |".format(
                method=method,
                final=number(
                    _median(
                        _finite_values(
                            dropout_rows,
                            "metrics.dropout.final_missing_pose_error.median",
                        )
                    )
                ),
                accepted=number(
                    _median(
                        _finite_values(
                            dropout_rows,
                            "metrics.dropout.first_accepted_post_dropout_output_error.median",
                        )
                    )
                ),
                recovery=number(
                    _median(
                        _finite_values(
                            dropout_rows,
                            "metrics.dropout.frames_to_return_within_nominal.median",
                        )
                    ),
                    1,
                ),
            )
        )
        outlier_rows = _filter_rows(
            sealed_rows,
            method,
            stresses={"OUTLIER_BURST", "COMBINED"},
        )
        outlier_lines.append(
            "| {method} | {accepted} | {rejected} | {maximum} | {recovery} | "
            "{contamination} |".format(
                method=method,
                accepted=int(
                    sum(
                        _finite_values(
                            outlier_rows, "metrics.outlier.outliers_accepted"
                        )
                    )
                ),
                rejected=int(
                    sum(
                        _finite_values(
                            outlier_rows, "metrics.outlier.outliers_rejected"
                        )
                    )
                ),
                maximum=number(
                    _median(
                        _finite_values(
                            outlier_rows,
                            "metrics.outlier.maximum_error_after_outlier_onset",
                        )
                    )
                ),
                recovery=number(
                    _median(
                        _finite_values(
                            outlier_rows,
                            "metrics.outlier.frames_until_recovery_after_burst.median",
                        )
                    ),
                    1,
                ),
                contamination=number(
                    _mean(
                        _finite_values(
                            outlier_rows,
                            "metrics.outlier.contamination_duration_frames",
                        )
                    ),
                    2,
                ),
            )
        )

    accuracy = criteria["symmetric_combined_accuracy"]
    dropout = criteria["symmetric_dropout_10"]
    asymmetric = criteria["asymmetric_guardrails"]
    dynamic = criteria["dynamic_symmetric_guardrail"]
    continuous = criteria["continuous_observability"]
    runtime = criteria["numerical_runtime_and_contract"]
    return f"""# PoseLoop M5-G0: Synthetic Mechanism Validation

## Claim boundary and decision

**{acceptance["classification"]}**

This report contains deterministic synthetic mechanism evidence only. It does
not claim real-video performance, FoundationPose improvement, official BOP
benchmark improvement, robot-control improvement, or hardware validation.

## Frozen benchmark

The sealed evaluation contains 2,304 trajectories (4 symmetries x 4 motions x
6 stresses x 24 seeds), each 120 frames at 20 Hz. All four methods saw the same
measurement hash per trajectory. Failed trajectories were retained; failures by
method are `{dict(sorted(failures.items()))}`.

## Development selection

Each temporal method received the same 1,152 development trajectories and four
configuration candidates. The exact frozen normalized objective and
lexicographic tie rule selected:

| Method | Config | Objective |
|---|---|---:|
{chr(10).join(config_lines)}

## Sealed aggregate results

| Method | Median quotient error | Translation RMSE (mm) | Rotation RMSE (deg) | Symmetric switch jumps | p95 CPU ms | Failed trajectories |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_lines)}

The formal symmetric combined-stress median errors are
`{accuracy["median_errors"]}`. The relative comparison versus standard is
`{accuracy["improvement_vs_standard"]}` and versus nearest is
`{accuracy["degradation_vs_nearest"]}`.

## Representation stability against both baselines

| Symmetry | Stress | Standard jumps | Nearest jumps | SymQuot jumps | Gate |
|---|---|---:|---:|---:|---|
{chr(10).join(jump_lines)}

The jump criterion is **{"PASS" if criteria["symmetric_switch_jump_reduction"]["pass"] else "FAIL"}**.
The zero-comparator rule is enforced: zero-versus-zero cannot satisfy a requested
relative reduction.

## Ten-frame dropout

| Method | Final-missing quotient error | First-accepted quotient error | Recovery frames |
|---|---:|---:|---:|
{chr(10).join(dropout_lines)}

The formal first-accepted comparison is `{dropout["improvement_vs_standard"]}`.

## Outlier bursts

| Method | Accepted outlier frames | Rejected outlier frames | Median max error | Median post-burst recovery | Mean contamination frames |
|---|---:|---:|---:|---:|---:|
{chr(10).join(outlier_lines)}

## Quotient accuracy, dropout, dynamics, and asymmetry

- Symmetric combined-stress accuracy: **{"PASS" if criteria["symmetric_combined_accuracy"]["pass"] else "FAIL"}**.
- Ten-frame symmetric dropout recovery: **{"PASS" if criteria["symmetric_dropout_10"]["pass"] else "FAIL"}**.
- Asymmetric translation/rotation/lag guardrail: **{"PASS" if criteria["asymmetric_guardrails"]["pass"] else "FAIL"}**.
- Dynamic symmetric direction-change guardrail: **{"PASS" if criteria["dynamic_symmetric_guardrail"]["pass"] else "FAIL"}**.
- Continuous-axis observability/gauge criterion: **{"PASS" if criteria["continuous_observability"]["pass"] else "FAIL"}**.
- Numerical, contract, and p95 CPU latency criterion: **{"PASS" if criteria["numerical_runtime_and_contract"]["pass"] else "FAIL"}**.

Exact guardrail inputs are:

- Asymmetric: `{asymmetric["values"]}`; translation degradation
  `{asymmetric["translation_degradation"]}`, rotation degradation
  `{asymmetric["rotation_degradation"]}`, lag worsening
  `{asymmetric["lag_worsening_frames"]}` frames.
- Dynamic symmetric median maximum lags: `{dynamic["median_max_lag_frames"]}`;
  SymQuot worsening `{dynamic["symquot_worsening_frames"]}` frames.
- Continuous axial velocity RMS: `{continuous["axial_velocity_rms_rad_s"]}`;
  reduction `{continuous["reduction_vs_standard"]}`.
- SymQuot p95 CPU latency: `{number(1000.0 * runtime["p95_symquot_cpu_latency_s"], 3) if runtime["p95_symquot_cpu_latency_s"] is not None else "N/A"}` ms/frame;
  failed trajectories `{runtime["failed_symquot_trajectory_count"]}` and non-finite
  output values `{runtime["nonfinite_symquot_output_value_count"]}`.

Continuous axial yaw is unobservable. The estimator output gauge is a continuity
choice only; accuracy scores transformed-axis direction and never claims yaw
recovery.

## Frozen representative and aggregate figures

{chr(10).join(f"- `{name}`" for name in plots)}

Representative trajectories are the middle seed in sorted sealed-seed order for
five conditions declared before outcomes were inspected.

## Anti-cherry-picking answers

{answers}

## Limitations and next evidence

The generator is bounded synthetic SE(3), noise and corruptions are declared,
and model-point normalized MSSD is diagnostic only. The result cannot establish
benefit on real predictor failure modes. A subsequent G1 would need frozen,
causal real-prediction replay and labeled symmetry-aware truth before any
real-video claim.
"""


def report(paths: Paths) -> dict[str, Any]:
    _verify_frozen(paths)
    selected, selected_hash = _selected_configs(paths)
    sealed_path = paths.artifacts / "sealed_results.jsonl"
    if not sealed_path.is_file():
        raise FileNotFoundError("Run the sealed stage before report generation")
    sealed_rows = load_jsonl(sealed_path)
    expected = EXPECTED_SEALED_TRAJECTORIES * len(SEALED_METHODS)
    if len(sealed_rows) != expected:
        raise ValueError(
            f"Sealed results contain {len(sealed_rows)} rows, expected {expected}"
        )
    aggregate_path = paths.artifacts / "aggregate_statistics.json"
    acceptance_path = paths.artifacts / "acceptance.json"
    if not aggregate_path.is_file() or not acceptance_path.is_file():
        aggregates = _aggregate_statistics(
            paths, sealed_rows, selected_hash, sealed_path
        )
        acceptance = _acceptance(paths, sealed_rows, aggregates, selected_hash)
    else:
        aggregates = read_json(aggregate_path)
        acceptance = read_json(acceptance_path)
    manifest = load_jsonl(paths.frozen / "sealed_seed_manifest.jsonl")
    representatives = _representative_manifest_rows(manifest)
    traces = {
        label: _trace_execution(row, selected) for label, row in representatives.items()
    }
    write_json_atomic(
        paths.artifacts / "representative_traces.json", _trace_json(traces)
    )
    plots = [
        *_plot_representatives(paths, traces),
        *_plot_aggregates(paths, sealed_rows),
    ]
    compact = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_compact_results",
        "title": "PoseLoop M5-G0 Synthetic Mechanism Validation",
        "classification": acceptance["classification"],
        "claim_boundary": (
            "Synthetic only; no real-video, FoundationPose, official BOP benchmark, "
            "robot-control, or hardware claim."
        ),
        "sealed_trajectory_count": EXPECTED_SEALED_TRAJECTORIES,
        "sealed_result_row_count": len(sealed_rows),
        "selected_config_ids": {
            method: value["config_id"] for method, value in selected["selected"].items()
        },
        "criteria": acceptance["criteria"],
        "measurement_identity": aggregates["measurement_identity"],
        "representative_rule": "middle seed in sorted sealed-seed order",
        "representative_trajectory_ids": {
            label: row["trajectory_id"] for label, row in representatives.items()
        },
        "plots": plots,
        "source_hashes": {
            "sealed_results": sha256_file(sealed_path),
            "selected_configs": selected_hash,
            "aggregate_statistics": sha256_file(aggregate_path),
            "acceptance": sha256_file(acceptance_path),
            "representative_traces": sha256_file(
                paths.artifacts / "representative_traces.json"
            ),
        },
    }
    write_json_atomic(paths.precomputed, compact)
    report_path = (
        paths.repo_root / "reports" / "m5_g0_synthetic_mechanism_validation.md"
    )
    _write_text_atomic(
        report_path,
        _report_markdown(acceptance, selected, plots, sealed_rows),
    )
    return compact


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths.create(
        repo_root,
        args.artifacts_dir,
        args.reports_dir,
        args.precomputed_result,
    )
    if args.stage == "tune":
        result = tune(paths)
        print(
            "selected M5-G0 configs: "
            + ", ".join(
                f"{method}={entry['config_id']}"
                for method, entry in result["selected"].items()
            )
        )
    elif args.stage == "sealed":
        result = sealed(paths)
        print(f"sealed M5-G0 classification: {result['classification']}")
    else:
        result = report(paths)
        print(f"reported M5-G0 classification: {result['classification']}")


if __name__ == "__main__":
    main()
