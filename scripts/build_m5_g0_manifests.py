#!/usr/bin/env python3
"""Freeze the auditable manifests and protocol inputs for PoseLoop M5-G0."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m1_common import canonical_sha256, read_json, sha256_file, write_json_atomic


SCHEMA_VERSION = 1
SYMMETRY_CLASSES = ("ASYMMETRIC", "C2", "C4", "CONTINUOUS_AXIAL")
MOTION_FAMILIES = (
    "STATIC",
    "CONSTANT_TWIST",
    "CURVED_ACCELERATING",
    "DIRECTION_REVERSAL",
)
STRESS_FAMILIES = (
    "NOMINAL",
    "REPRESENTATION_SWITCH",
    "DROPOUT_5",
    "DROPOUT_10",
    "OUTLIER_BURST",
    "COMBINED",
)
METHODS = (
    "RAW_HOLD",
    "STANDARD_SE3_CT",
    "NEAREST_REPRESENTATIVE_CT",
    "SYMQUOT_CT",
)
TEMPORAL_METHODS = METHODS[1:]
DEVELOPMENT_REPLICATES = 12
SEALED_REPLICATES = 24
EXPECTED_DEVELOPMENT_TRAJECTORIES = 1152
EXPECTED_SEALED_TRAJECTORIES = 2304


ALGORITHM_GRID: dict[str, dict[str, float | int]] = {
    "agile": {
        "process_translation_sigma_m": 0.003,
        "process_rotation_sigma_deg": 1.5,
        "velocity_process_sigma": 0.08,
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "innovation_gate_sigma": 6.5,
        "velocity_update_gain": 0.50,
        "reacquire_after": 4,
    },
    "balanced": {
        "process_translation_sigma_m": 0.002,
        "process_rotation_sigma_deg": 1.0,
        "velocity_process_sigma": 0.04,
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "innovation_gate_sigma": 5.5,
        "velocity_update_gain": 0.35,
        "reacquire_after": 4,
    },
    "conservative": {
        "process_translation_sigma_m": 0.001,
        "process_rotation_sigma_deg": 0.5,
        "velocity_process_sigma": 0.02,
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "innovation_gate_sigma": 4.5,
        "velocity_update_gain": 0.20,
        "reacquire_after": 4,
    },
    "robust_agile": {
        "process_translation_sigma_m": 0.003,
        "process_rotation_sigma_deg": 1.5,
        "velocity_process_sigma": 0.08,
        "measurement_translation_sigma_m": 0.002,
        "measurement_rotation_sigma_deg": 1.0,
        "innovation_gate_sigma": 4.5,
        "velocity_update_gain": 0.50,
        "reacquire_after": 4,
    },
}


CORRUPTION_MANIFEST: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "record_type": "m5_g0_corruption_manifest",
    "nominal_noise": {
        "translation_sigma_m": 0.002,
        "rotation_sigma_deg": 1.0,
        "application": "right-multiplicative body-frame SE(3) noise",
    },
    "representation_switch": {
        "finite": {
            "process": "seeded persistent Markov chain over group indices",
            "stay_probability": 0.92,
            "minimum_dwell_frames": 5,
            "application": "T_measurement @ S_index",
        },
        "continuous": {
            "process": "seeded persistent axial gauge with smooth drift and jumps",
            "application": "T_measurement @ Rz(gauge)",
        },
    },
    "dropout_5": {
        "length_frames": 5,
        "start_rule": "38 + corruption_seed % 17",
    },
    "dropout_10": {
        "length_frames": 10,
        "start_rule": "38 + corruption_seed % 17",
    },
    "outlier_burst": {
        "length_frames": 3,
        "start_rule": "72 + corruption_seed % 13",
        "translation_m": 0.030,
        "observable_rotation_deg": 20.0,
    },
    "combined": {
        "dropout_length_frames": 10,
        "dropout_start_rule": "40 + corruption_seed % 9",
        "outlier_length_frames": 3,
        "outlier_start_rule": "76 + corruption_seed % 11",
        "requires_disjoint_intervals": True,
        "includes_representation_switch": True,
    },
}


METRIC_DEFINITIONS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "record_type": "m5_g0_metric_definitions",
    "interval_convention": "half-open [start_frame, end_frame_exclusive)",
    "finite_summary": (
        "All scalar summaries retain the original count and failure counts but use "
        "only finite values: mean=sum(x)/n, median=50th percentile, "
        "p95=95th percentile, and RMSE=sqrt(mean(x^2))."
    ),
    "translation_error_m": "Euclidean distance between output and truth translations",
    "quotient_rotation_error_deg": (
        "minimum SO(3) geodesic over declared finite right symmetries; for "
        "CONTINUOUS_AXIAL, angle between transformed +z axes"
    ),
    "quotient_pose_error": (
        "sqrt((translation_error_m / 0.01)^2 + (quotient_rotation_error_deg / 5.0)^2)"
    ),
    "catastrophic_finite_jump": "raw_step_deg > 90 and quotient_step_deg < 10",
    "continuous_gauge_jump": "abs(axial_spin_deg) > 45 and quotient_step_deg < 10",
    "representation_stability": {
        "raw_step": "SO(3) geodesic angle between consecutive output rotations",
        "finite_quotient_step": (
            "minimum SO(3) geodesic from the previous output to the current "
            "output right-multiplied by every declared finite symmetry"
        ),
        "continuous_quotient_step": (
            "angle between consecutive transformed directed object +z axes"
        ),
        "continuous_axial_spin": (
            "swing-align the previous transformed axis to the current axis, project "
            "the residual quaternion vector onto object +z, compute "
            "2*atan2(projected_vector,w), and wrap to [-pi,pi)"
        ),
        "continuous_axial_velocity_rms": (
            "sqrt(mean((axial_spin_rad / dt_s)^2)) over finite consecutive frames"
        ),
    },
    "nominal_recovery_threshold": "quotient_pose_error <= 1.0",
    "dropout": {
        "final_missing_pose_error": "quotient pose error at end_frame_exclusive - 1",
        "first_return_measurement_error": (
            "measurement quotient pose error at the first finite measurement frame "
            "at or after end_frame_exclusive"
        ),
        "first_accepted_post_dropout_output_error": (
            "output quotient pose error at the first accepted update frame at or "
            "after end_frame_exclusive; missing if no update is accepted"
        ),
        "frames_to_return_within_nominal": (
            "first frame at or after end_frame_exclusive with quotient_pose_error "
            "<= 1.0, minus end_frame_exclusive"
        ),
        "uncertainty_scalar": (
            "mean of the six diagonal pose-uncertainty entries per frame; covariance "
            "inputs use trace and scalar inputs are unchanged"
        ),
        "uncertainty_monotonic": (
            "all successive differences from the frame immediately before dropout "
            "through the final missing frame are >= -1e-12"
        ),
        "maximum_stable_dropout_length": (
            "longest prefix of missing frames, starting at dropout onset, whose "
            "output quotient_pose_error remains finite and <= 1.0"
        ),
    },
    "outlier": {
        "accepted_rejected": (
            "counts of estimator accepted flags within the declared outlier interval"
        ),
        "maximum_error_after_onset": (
            "maximum finite quotient_pose_error from outlier onset through recovery, "
            "inclusive; through trajectory end if recovery fails"
        ),
        "frames_until_recovery_after_burst": (
            "first post-burst frame with quotient_pose_error <= 1.0, minus "
            "end_frame_exclusive"
        ),
        "contamination_duration": (
            "frames_until_recovery_after_burst, or remaining post-burst frames when "
            "recovery fails"
        ),
    },
    "dynamic_guardrails": {
        "observable_velocity": (
            "frame-aligned finite differences; continuous axial angular velocity "
            "has its unobservable body-axis component removed"
        ),
        "direction_change_lag": (
            "using the mean reference direction over the ten frames before the "
            "declared reversal, max(0, first estimated negative crossing - first "
            "reference negative crossing)"
        ),
        "peak_velocity_ratio": "max output vector norm / max truth vector norm",
        "peak_velocity_attenuation": "max(0, 1 - peak_velocity_ratio)",
        "oversmoothing_ratio": (
            "RMS output observable normalized speed / RMS truth observable normalized "
            "speed, with speed=sqrt((translation_m_s/0.01)^2 + "
            "(rotation_deg_s/5)^2)"
        ),
        "dynamic_trajectory_rmse": "RMSE of quotient_pose_error on the dynamic interval",
    },
    "asymmetric_guardrails": {
        "translation_rmse_m": "sqrt(mean(translation_error_m^2))",
        "rotation_rmse_deg": "sqrt(mean(rotation_error_deg^2))",
        "direction_change_lag": "the dynamic direction-change definitions above",
        "numerical_failures": "count of output frames containing any non-finite pose value",
    },
    "runtime_and_finiteness": {
        "runtime": (
            "per-frame perf_counter CPU wall duration; report mean, numpy linear "
            "p95, and maximum over finite non-negative values"
        ),
        "pose_counts": "exact NaN, Inf, non-finite value, and non-finite frame counts",
        "uncertainty_counts": (
            "exact NaN, Inf, and non-finite value counts over the reported uncertainty"
        ),
        "failures": "exact initialization and recovery failure counts",
    },
    "condition_aggregation": (
        "Group by method, symmetry, motion, and stress; retain every trajectory and "
        "failure count; finite-summarize every numeric trajectory scalar; also sum "
        "fields denoting counts, jumps, accepted/rejected outliers, failures, or "
        "contamination."
    ),
    "development_objective": {
        "formula": (
            "median_quotient_pose_error/1.0 + mean_representation_jumps/1.0 + "
            "median_max_reversal_lag_frames/4.0 + "
            "mean_outlier_contamination_frames/3.0 + 100*trajectory_failure_rate"
        ),
        "tie_break": "lexicographically smallest config_id within 1e-12",
    },
    "bop_compatibility": (
        "For asymmetric/C2/C4 synthetic model points, min over right symmetries S "
        "of max_p ||T_output*p - (T_truth*S)*p|| divided by synthetic object "
        "diameter; unavailable for continuous symmetry and not an official BOP "
        "benchmark result."
    ),
}


DECISION_THRESHOLDS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "record_type": "m5_g0_decision_thresholds",
    "classification_order": [
        "ALGORITHM GO",
        "MECHANISM PASS / REAL-DATA VALUE UNPROVEN",
        "NO-GO",
    ],
    "mechanism_pass_rule": (
        "All jump, asymmetric, dynamic, continuous, and numerical/runtime gates pass, "
        "and at least one of the combined quotient-accuracy or ten-frame-dropout "
        "gates passes. If both value gates fail, classify NO-GO."
    ),
    "outlier_diagnostic_rule": (
        "Report mean contamination duration, accepted outliers, and recovery-failure "
        "rate against both temporal baselines; no post-sealed outlier threshold is a "
        "formal classification gate."
    ),
    "algorithm_go": {
        "jump_reduction_vs_standard_min": 0.80,
        "jump_reduction_vs_nearest_min": 0.30,
        "combined_median_error_improvement_vs_standard_min": 0.15,
        "combined_median_error_degradation_vs_nearest_max": 0.03,
        "dropout_first_accepted_improvement_vs_standard_min": 0.20,
        "asymmetric_translation_rmse_degradation_max": 0.03,
        "asymmetric_rotation_rmse_degradation_max": 0.03,
        "asymmetric_direction_lag_worsening_frames_max": 1,
        "dynamic_symmetric_lag_worsening_frames_max": 1,
        "continuous_axial_velocity_rms_reduction_vs_standard_min": 0.80,
        "p95_latency_ms_max": 10.0,
        "nan_inf_trajectories_max": 0,
        "hidden_truth_access_allowed": False,
    },
    "zero_comparator_rule": (
        "A requested relative reduction cannot pass when the comparator count "
        "is zero; zero-versus-zero is not evidence of reduction."
    ),
    "mechanism_pass": (
        "Use only when the exact jump criterion, continuous criterion, asymmetric "
        "and dynamic guardrails, and numerical/runtime criterion pass, while the "
        "combined-accuracy or dropout criterion fails. Otherwise use NO-GO."
    ),
}


CONVENTIONS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "record_type": "m5_g0_mathematical_conventions",
    "pose": "T_c_o maps object/model-frame column vectors into the camera frame",
    "serialization": "nested 4x4 row-major lists; translation in T[:3,3]; metres",
    "symmetry": "object-frame transforms act on the right: T_equivalent = T @ S",
    "body_prediction": "T_next = T @ Exp(body_twist * dt)",
    "innovation": "Log(inv(T_predicted) @ T_measurement_representative)",
    "continuous_axial": (
        "transformed object +z direction is observable; right axial rotation is "
        "unobservable and output gauge continuity is not physical accuracy"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "m5_g0",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing freeze without writing anything.",
    )
    return parser.parse_args()


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _seed_rows(split: str, replicates: int) -> list[dict[str, Any]]:
    split_base = 1_000_000 if split == "development" else 9_000_000
    corruption_base = 2_000_000 if split == "development" else 10_000_000
    rows: list[dict[str, Any]] = []
    for symmetry_index, symmetry in enumerate(SYMMETRY_CLASSES):
        for motion_index, motion in enumerate(MOTION_FAMILIES):
            for stress_index, stress in enumerate(STRESS_FAMILIES):
                condition_index = (
                    symmetry_index * len(MOTION_FAMILIES) * len(STRESS_FAMILIES)
                    + motion_index * len(STRESS_FAMILIES)
                    + stress_index
                )
                for replicate in range(replicates):
                    seed_offset = condition_index * 100 + replicate
                    trajectory_id = (
                        f"{split[:3]}-{symmetry.lower()}-{motion.lower()}-"
                        f"{stress.lower()}-{replicate:02d}"
                    )
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "m5_g0_seed",
                            "split": split,
                            "trajectory_id": trajectory_id,
                            "symmetry_class": symmetry,
                            "motion_family": motion,
                            "stress_family": stress,
                            "replicate_index": replicate,
                            "trajectory_seed": split_base + seed_offset,
                            "corruption_seed": corruption_base + seed_offset,
                        }
                    )
    return rows


def _canonical_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    tracked = _git(repo_root, "ls-files").splitlines()
    protected = [
        path
        for path in tracked
        if path == "precomputed/v1.0.0/results.json"
        or path == "reports/release_summary.md"
        or path.startswith(("reports/m2_", "reports/m3_", "reports/m4_"))
        or path.startswith(
            (
                "scripts/evaluate_m2.py",
                "scripts/evaluate_m3.py",
                "scripts/evaluate_m4.py",
                "scripts/fit_m3_policy.py",
                "scripts/fit_m4_voi.py",
            )
        )
    ]
    hashes = {path: sha256_file(repo_root / path) for path in sorted(protected)}
    for milestone in ("m2", "m3", "m4"):
        root = repo_root / "artifacts" / milestone
        if root.exists():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                relative = path.relative_to(repo_root).as_posix()
                hashes[relative] = sha256_file(path)
    return hashes


def _source_audit(repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_source_audit",
        "branch": _git(repo_root, "branch", "--show-current"),
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "initial_status_short_branch": "## master",
        "freeze_status_short_branch": _git(repo_root, "status", "--short", "--branch"),
        "log_3_oneline": _git(repo_root, "log", "-3", "--oneline").splitlines(),
        "branch_vv": _git(repo_root, "branch", "-vv").splitlines(),
        "remote_v": _git(repo_root, "remote", "-v").splitlines(),
        "baseline_check": {
            "command": "python -B scripts/render_precomputed_report.py --check",
            "result": "PASS: reports\\release_summary.md matches results.json",
        },
        "reused": [
            "scripts/m1_common.py atomic JSON/JSONL, hashing, assert_pose, raw errors",
            "scripts/m2_common.py calibrated column-vector transform convention",
            "scripts/evaluate_m1.py and vendored BOP right-applied symmetry convention",
        ],
        "scope": "Synthetic mechanism validation only; M2/M3/M4 are immutable.",
    }


def _validate_seed_rows(rows: list[dict[str, Any]], split: str) -> None:
    expected = (
        EXPECTED_DEVELOPMENT_TRAJECTORIES
        if split == "development"
        else EXPECTED_SEALED_TRAJECTORIES
    )
    if len(rows) != expected:
        raise ValueError(f"{split} manifest has {len(rows)} rows, expected {expected}")
    ids = {row["trajectory_id"] for row in rows}
    seeds = {row["trajectory_seed"] for row in rows}
    corruption = {row["corruption_seed"] for row in rows}
    if len(ids) != expected or len(seeds) != expected or len(corruption) != expected:
        raise ValueError(f"{split} manifest identifiers or seeds are not unique")


def freeze(output_dir: Path, *, check_only: bool) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = output_dir.resolve()
    required_root = (repo_root / "artifacts" / "m5_g0").resolve()
    if not resolved.is_relative_to(required_root):
        raise ValueError(f"M5-G0 outputs must remain under {required_root}: {resolved}")

    frozen_dir = resolved / "frozen"
    result_markers = [
        resolved / "selected_configs.json",
        resolved / "sealed_results.jsonl",
        resolved / "acceptance.json",
    ]
    if not check_only and any(path.exists() for path in result_markers):
        raise RuntimeError("Refusing to rewrite frozen inputs after evaluation began")

    development_rows = _seed_rows("development", DEVELOPMENT_REPLICATES)
    sealed_rows = _seed_rows("sealed", SEALED_REPLICATES)
    _validate_seed_rows(development_rows, "development")
    _validate_seed_rows(sealed_rows, "sealed")
    if {row["trajectory_seed"] for row in development_rows} & {
        row["trajectory_seed"] for row in sealed_rows
    }:
        raise ValueError("Development and sealed trajectory seeds overlap")
    if {row["corruption_seed"] for row in development_rows} & {
        row["corruption_seed"] for row in sealed_rows
    }:
        raise ValueError("Development and sealed corruption seeds overlap")

    deterministic_payloads: dict[str, Any] = {
        "development_seed_manifest.jsonl": _canonical_jsonl(development_rows),
        "sealed_seed_manifest.jsonl": _canonical_jsonl(sealed_rows),
        "corruption_manifest.json": CORRUPTION_MANIFEST,
        "algorithm_grid.json": {
            "schema_version": SCHEMA_VERSION,
            "record_type": "m5_g0_algorithm_grid",
            "methods": list(TEMPORAL_METHODS),
            "equal_search_budget_per_method": len(ALGORITHM_GRID),
            "shared_across_symmetry_classes": True,
            "configurations": ALGORITHM_GRID,
        },
        "metric_definitions.json": METRIC_DEFINITIONS,
        "decision_thresholds.json": DECISION_THRESHOLDS,
        "conventions.json": CONVENTIONS,
    }

    if check_only:
        if not frozen_dir.exists():
            raise FileNotFoundError(f"Frozen directory does not exist: {frozen_dir}")
        for name, expected in deterministic_payloads.items():
            path = frozen_dir / name
            if isinstance(expected, str):
                observed = path.read_text(encoding="utf-8")
                if observed != expected:
                    raise ValueError(
                        f"Frozen file differs from protocol generator: {path}"
                    )
            elif read_json(path) != expected:
                raise ValueError(f"Frozen file differs from protocol generator: {path}")
        receipt = read_json(frozen_dir / "manifest_hashes.json")
        for name, expected_hash in receipt["sha256"].items():
            if sha256_file(frozen_dir / name) != expected_hash:
                raise ValueError(f"Frozen hash mismatch: {name}")
        return receipt

    frozen_dir.mkdir(parents=True, exist_ok=True)
    for name, value in deterministic_payloads.items():
        path = frozen_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen file: {path}")
        if isinstance(value, str):
            _write_text_atomic(path, value)
        else:
            write_json_atomic(path, value)

    source_audit = _source_audit(repo_root)
    write_json_atomic(resolved / "source_audit.json", source_audit)
    protected_hashes = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_protected_source_hashes",
        "sha256": _protected_hashes(repo_root),
    }
    write_json_atomic(frozen_dir / "protected_source_hashes.json", protected_hashes)
    frozen_names = sorted([*deterministic_payloads, "protected_source_hashes.json"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "m5_g0_manifest_hashes",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": source_audit["head"],
        "development_trajectory_count": len(development_rows),
        "sealed_trajectory_count": len(sealed_rows),
        "development_manifest_canonical_sha256": canonical_sha256(development_rows),
        "sealed_manifest_canonical_sha256": canonical_sha256(sealed_rows),
        "sha256": {name: sha256_file(frozen_dir / name) for name in frozen_names},
    }
    write_json_atomic(frozen_dir / "manifest_hashes.json", receipt)
    return receipt


def main() -> None:
    args = parse_args()
    receipt = freeze(args.output_dir, check_only=args.check)
    action = "validated" if args.check else "frozen"
    print(
        f"{action} M5-G0 protocol: "
        f"development={receipt['development_trajectory_count']}, "
        f"sealed={receipt['sealed_trajectory_count']}"
    )


if __name__ == "__main__":
    main()
