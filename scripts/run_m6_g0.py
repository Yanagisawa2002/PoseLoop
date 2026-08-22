#!/usr/bin/env python3
"""Run the PoseLoop M6-G0 existing-data confidence signal audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_m6_g0_manifests as manifest_builder  # noqa: E402
import m6_g0_analysis as analysis  # noqa: E402
import m6_g0_features as feature_builder  # noqa: E402
from m1_common import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from m6_g0_labels import label_support_summary, load_evaluator_rows  # noqa: E402


SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_dataset_root() -> Path:
    configured = os.environ.get("POSELOOP_XYZIBD_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path(r"\\wsl.localhost\Ubuntu-24.04\home\cgliu\datasets\xyzibd")
    return Path("/home/cgliu/datasets/xyzibd")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=repo_root / "third_party" / "bop_toolkit",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
    )
    return parser.parse_args()


def artifact_paths(repo_root: Path) -> dict[str, Path]:
    artifact_root = (repo_root / "artifacts" / "m6_g0").resolve()
    expected_root = (repo_root / "artifacts" / "m6_g0").resolve()
    if artifact_root != expected_root:
        raise AssertionError("M6-G0 artifact root changed unexpectedly")
    return {
        "root": artifact_root,
        "frozen": artifact_root / "frozen",
        "features": artifact_root / "features.jsonl",
        "evaluator_rows": artifact_root / "evaluator_rows.jsonl",
        "oof_predictions": artifact_root / "oof_predictions.jsonl",
        "ablation_oof_predictions": artifact_root / "ablation_oof_predictions.jsonl",
        "per_fold_metrics": artifact_root / "per_fold_metrics.jsonl",
        "aggregate_metrics": artifact_root / "aggregate_metrics.json",
        "grouped_bootstrap": artifact_root / "grouped_bootstrap.json",
        "object_jackknife": artifact_root / "object_jackknife.json",
        "feature_ablations": artifact_root / "feature_ablations.json",
        "feature_contributions": artifact_root / "feature_contributions.json",
        "model_selections": artifact_root / "model_selections.json",
        "prediction_validation": artifact_root / "prediction_validation.json",
        "decision_inputs": artifact_root / "decision_inputs.json",
        "qualitative": artifact_root / "qualitative_error_table.json",
        "decision": artifact_root / "decision.json",
        "chronology": artifact_root / "chronology.json",
        "source_hashes": artifact_root / "source_hashes.json",
        "formal_validation": artifact_root / "formal_validation.json",
    }


def _json_safe(value: Any) -> Any:
    """Convert common NumPy scalar/array values into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _write_result(path: Path, value: Any, *, jsonl: bool = False) -> None:
    safe = _json_safe(value)
    if jsonl:
        if not isinstance(safe, list):
            raise TypeError(
                f"Expected a row list for {path}, got {type(safe).__name__}"
            )
        write_jsonl_atomic(path, safe)
    else:
        write_json_atomic(path, safe)


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, Mapping):
        raise ValueError("Feature row lacks metadata")
    return value


def _target_id(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    target_id = str(
        metadata.get("target_id")
        or metadata.get("target_sample_id")
        or metadata.get("group_id")
        or ""
    )
    if not target_id:
        raise ValueError("Feature row has no target identity")
    return target_id


def frozen_ablation_feature_sets() -> dict[str, list[str]]:
    """Return the exact predeclared M6-G0 ablation feature sets.

    Availability-only diagnostics are deliberately reserved for the ``all``
    family.  This mirrors the write-once feature specification frozen before
    any model fitting.
    """

    feature_names = list(feature_builder.FEATURE_SPEC)
    score_only = [
        name
        for name in feature_names
        if feature_builder.FEATURE_FAMILIES[name] == "score"
    ]
    disagreement_only = [
        name
        for name in feature_names
        if feature_builder.FEATURE_FAMILIES[name]
        in {"disagreement", "agreement", "view"}
    ]
    return {
        "score_only": score_only,
        "disagreement_only": disagreement_only,
        "score_disagreement": [*score_only, *disagreement_only],
        "all": feature_names,
    }


def validate_feature_label_separation(
    feature_rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prohibited = {
        "target_id",
        "target_sample_id",
        "group_id",
        "object_id",
        "physical_instance_id",
        "physical_instance_track_id",
        "scene_id",
        "split_id",
        "correct",
        "success",
        "y_failure",
        "normalized_mssd",
        "mssd_mm",
        "mspd_px",
        "ground_truth",
        "gt_pose",
    }
    feature_names: set[str] | None = None
    targets: list[str] = []
    for row in feature_rows:
        features = row.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("Inference feature row lacks a feature mapping")
        names = set(str(name) for name in features)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Feature rows do not share an identical schema")
        forbidden_hits = sorted(names & prohibited)
        token_hits = sorted(
            name
            for name in names
            if any(
                token in name.lower()
                for token in feature_builder.PROHIBITED_FEATURE_TOKENS
            )
        )
        if forbidden_hits or token_hits:
            raise ValueError(
                f"Prohibited feature fields: exact={forbidden_hits}, tokens={token_hits}"
            )
        targets.append(_target_id(row))
    if len(targets) != len(set(targets)):
        raise ValueError("Feature table has duplicate target rows")
    label_targets = [str(row["target_id"]) for row in evaluator_rows]
    if set(targets) != set(label_targets) or len(label_targets) != len(
        set(label_targets)
    ):
        raise ValueError("Feature/evaluator target identity mismatch")
    return {
        "passed": True,
        "target_count": len(targets),
        "feature_count": len(feature_names or ()),
        "one_row_per_target": True,
        "feature_and_evaluator_structures_separate": True,
        "prohibited_feature_hits": [],
    }


def _flat_oof_rows(oof_predictions: Any) -> list[dict[str, Any]]:
    if not isinstance(oof_predictions, list):
        raise TypeError("OOF predictions must be a list")
    if not oof_predictions:
        return []
    if all(isinstance(row, Mapping) and "method" in row for row in oof_predictions):
        return [dict(row) for row in oof_predictions]
    flattened: list[dict[str, Any]] = []
    for row in oof_predictions:
        if not isinstance(row, Mapping):
            raise TypeError("OOF prediction row is not a mapping")
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError("Unrecognized OOF prediction schema")
        base = {key: value for key, value in row.items() if key != "predictions"}
        for method, prediction in predictions.items():
            if isinstance(prediction, Mapping):
                flattened.append({**base, "method": method, **prediction})
            else:
                flattened.append(
                    {**base, "method": method, "failure_probability": prediction}
                )
    return flattened


def _risk_value(row: Mapping[str, Any]) -> float:
    for name in (
        "failure_probability",
        "predicted_failure_probability",
        "risk",
        "risk_score",
        "score",
    ):
        value = row.get(name)
        if value is not None:
            return float(value)
    raise ValueError(f"OOF row lacks a risk value: {row.get('method')}")


def qualitative_error_table(
    oof_predictions: Any,
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select frozen deterministic qualitative rows without favorable filtering."""
    rows = _flat_oof_rows(oof_predictions)
    nested = [row for row in rows if row.get("method") == "NESTED_MULTIFEATURE"]
    isotonic = {
        str(row["target_id"]): _risk_value(row)
        for row in rows
        if row.get("method") == "SCORE_ISOTONIC"
    }
    labels = {str(row["target_id"]): row for row in evaluator_rows}
    enriched = []
    for row in nested:
        target_id = str(row["target_id"])
        label = labels[target_id]
        risk = _risk_value(row)
        enriched.append(
            {
                "target_id": target_id,
                "object_id": int(label["object_id"]),
                "physical_instance_id": str(label["physical_instance_id"]),
                "y_failure": int(label["y_failure"]),
                "nested_failure_risk": risk,
                "isotonic_failure_risk": isotonic.get(target_id),
                "absolute_candidate_isotonic_difference": (
                    abs(risk - isotonic[target_id]) if target_id in isotonic else None
                ),
            }
        )
    low_risk_failures = sorted(
        (row for row in enriched if row["y_failure"] == 1),
        key=lambda row: (row["nested_failure_risk"], row["target_id"]),
    )[:5]
    high_risk_successes = sorted(
        (row for row in enriched if row["y_failure"] == 0),
        key=lambda row: (-row["nested_failure_risk"], row["target_id"]),
    )[:5]
    largest_disagreements = sorted(
        enriched,
        key=lambda row: (
            -float(row["absolute_candidate_isotonic_difference"] or 0.0),
            row["target_id"],
        ),
    )[:5]
    return {
        "record_type": "m6_g0_qualitative_error_table",
        "schema_version": SCHEMA_VERSION,
        "selection_rules": {
            "lowest_risk_failures": "five failures with lowest nested OOF risk",
            "highest_risk_successes": "five successes with highest nested OOF risk",
            "largest_candidate_isotonic_disagreements": (
                "five largest absolute nested-versus-isotonic OOF risk differences"
            ),
        },
        "lowest_risk_failures": low_risk_failures,
        "highest_risk_successes": high_risk_successes,
        "largest_candidate_isotonic_disagreements": largest_disagreements,
    }


def _decision_boolean(
    decision_inputs: Mapping[str, Any],
    name: str,
    fallback: bool,
) -> bool:
    value = decision_inputs.get(name, fallback)
    return bool(value)


def _write_evaluation_results(
    paths: Mapping[str, Path],
    evaluation: Mapping[str, Any],
) -> None:
    _write_result(paths["oof_predictions"], evaluation["oof_predictions"], jsonl=True)
    _write_result(
        paths["ablation_oof_predictions"],
        evaluation["ablation_oof_predictions"],
        jsonl=True,
    )
    _write_result(paths["per_fold_metrics"], evaluation["per_fold_metrics"], jsonl=True)
    for key in (
        "aggregate_metrics",
        "grouped_bootstrap",
        "object_jackknife",
        "feature_ablations",
        "feature_contributions",
        "model_selections",
        "prediction_validation",
        "decision_inputs",
    ):
        _write_result(paths[key], evaluation[key])


def _source_hash_payload(
    repo_root: Path,
    paths: Mapping[str, Path],
    extra_paths: Sequence[Path],
) -> dict[str, Any]:
    candidates = [
        path
        for key, path in paths.items()
        if key not in {"root", "frozen", "source_hashes", "formal_validation"}
        and path.is_file()
    ]
    candidates.extend(path for path in paths["frozen"].glob("*") if path.is_file())
    candidates.extend(path for path in extra_paths if path.is_file())
    unique = sorted(
        set(path.resolve() for path in candidates), key=lambda path: path.as_posix()
    )
    hashes = {
        path.relative_to(repo_root).as_posix(): sha256_file(path) for path in unique
    }
    return {
        "record_type": "m6_g0_source_hashes",
        "schema_version": SCHEMA_VERSION,
        "sha256": hashes,
        "canonical_sha256": canonical_sha256(hashes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.bootstrap_resamples) < BOOTSTRAP_RESAMPLES:
        raise ValueError("Formal M6-G0 requires at least 2,000 grouped bootstraps")
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    toolkit_root = args.toolkit_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"XYZ-IBD root is unavailable: {dataset_root}")
    if not toolkit_root.is_dir():
        raise FileNotFoundError(f"BOP Toolkit root is unavailable: {toolkit_root}")

    paths = artifact_paths(repo_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    chronology: dict[str, Any] = {
        "record_type": "m6_g0_chronology",
        "schema_version": SCHEMA_VERSION,
        "run_started_utc": utc_now(),
        "bootstrap_resamples": int(args.bootstrap_resamples),
    }
    freeze_receipt = manifest_builder.freeze(repo_root, check_only=False)
    chronology["static_protocol_frozen_utc"] = utc_now()
    chronology["static_freeze_receipt"] = freeze_receipt

    groups_path = repo_root / "artifacts" / "m2" / "groups.jsonl"
    m1_predictions_path = repo_root / "artifacts" / "m1" / "predictions.jsonl"
    m2_predictions_path = repo_root / "artifacts" / "m2" / "view_predictions.jsonl"
    metrics_path = repo_root / "artifacts" / "m2" / "metrics.jsonl"
    feature_rows = feature_builder.build_feature_rows(
        groups_path=groups_path,
        m1_predictions_path=m1_predictions_path,
        m2_predictions_path=m2_predictions_path,
        dataset_root=dataset_root,
        toolkit_root=toolkit_root,
    )
    if len(feature_rows) != 300:
        raise ValueError(
            f"M6-G0 requires exactly 300 M2 targets, got {len(feature_rows)}"
        )
    for row in feature_rows:
        feature_builder.validate_feature_schema(row, require_complete=True)
    evaluator_rows = load_evaluator_rows(metrics_path, feature_rows)
    support = label_support_summary(evaluator_rows)
    if not support["passed"]:
        raise RuntimeError(f"LABEL-SUPPORT NO-GO: {support['criteria']}")
    separation = validate_feature_label_separation(feature_rows, evaluator_rows)
    _write_result(paths["features"], feature_rows, jsonl=True)
    _write_result(paths["evaluator_rows"], evaluator_rows, jsonl=True)
    chronology["feature_and_label_rows_written_utc"] = utc_now()

    fold_manifest = analysis.make_fold_manifest(evaluator_rows)
    analysis.validate_fold_manifest(fold_manifest, evaluator_rows)
    fold_manifest_path = paths["frozen"] / "fold_manifest.json"
    manifest_builder.freeze_json(fold_manifest_path, _json_safe(fold_manifest))
    chronology["fold_manifest_written_utc"] = utc_now()
    chronology["fold_manifest_sha256"] = sha256_file(fold_manifest_path)
    chronology["model_evaluation_started_utc"] = utc_now()
    _write_result(paths["chronology"], chronology)

    evaluation = analysis.run_evaluation(
        feature_rows,
        evaluator_rows,
        fold_manifest,
        frozen_ablation_feature_sets(),
        raw_score_feature="raw_selected_score",
        raw_score_higher_is_confident=True,
        bootstrap_resamples=int(args.bootstrap_resamples),
    )
    _write_evaluation_results(paths, evaluation)
    decision_inputs = evaluation["decision_inputs"]
    decision = analysis.frozen_signal_decision(
        evaluation["aggregate_metrics"],
        evaluation["grouped_bootstrap"],
        evaluation["object_jackknife"],
        feature_ablations=evaluation["feature_ablations"],
        stable_disagreement_benefit=_decision_boolean(
            decision_inputs,
            "stable_disagreement_benefit",
            False,
        ),
        leakage_checks_passed=_decision_boolean(
            decision_inputs,
            "leakage_checks_passed",
            separation["passed"],
        ),
        label_support_passed=bool(support["passed"]),
        oof_complete=_decision_boolean(
            decision_inputs,
            "oof_complete",
            bool(evaluation["prediction_validation"].get("passed", False)),
        ),
        m3_access_boundary_passed=False,
    )
    qualitative = qualitative_error_table(evaluation["oof_predictions"], evaluator_rows)
    _write_result(paths["decision"], decision)
    _write_result(paths["qualitative"], qualitative)
    chronology["model_evaluation_completed_utc"] = utc_now()
    chronology["classification"] = decision["classification"]
    _write_result(paths["chronology"], chronology)

    # Reporting is imported late so unit tests of the causal core do not need a
    # Matplotlib backend.
    from m6_g0_report import render_m6_g0_outputs

    report_result = render_m6_g0_outputs(
        repo_root=repo_root,
        feature_rows=feature_rows,
        evaluator_rows=evaluator_rows,
        evaluation=evaluation,
        label_support=support,
        decision=decision,
        qualitative=qualitative,
        chronology=chronology,
        protected_receipt=freeze_receipt,
    )
    chronology["report_outputs_completed_utc"] = utc_now()
    chronology["run_completed_utc"] = chronology["report_outputs_completed_utc"]
    _write_result(paths["chronology"], chronology)
    source_hashes = _source_hash_payload(
        repo_root,
        paths,
        [Path(path) for path in report_result["tracked_paths"]],
    )
    _write_result(paths["source_hashes"], source_hashes)
    return {
        "classification": decision["classification"],
        "counterfactual_classification_without_m3_boundary": decision.get(
            "counterfactual_classification_without_m3_boundary"
        ),
        "target_count": len(feature_rows),
        "failure_count": support["failure_count"],
        "fold_manifest_sha256": chronology["fold_manifest_sha256"],
        "report": report_result["report"],
        "plots": report_result["plots"],
    }


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
