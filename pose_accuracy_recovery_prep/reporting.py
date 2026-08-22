"""Unified official-capability and internal diagnostic reporting."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import ContractError, EVALUATOR_VARIANTS, canonical_sha256, write_json
from .se3 import load_cad_asset, load_pose_asset, perturb_pose, pose_errors

OFFICIAL_METRICS = ("AR", "MSSD", "MSPD", "VSD")
INTERNAL_METRICS = ("add_s_mm", "rotation_error_degrees", "translation_error_mm")
STAGES = ("baseline", "final")


def official_metric_status(
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    *,
    fixture_mode: bool,
) -> dict[str, dict[str, Any]]:
    capabilities = protocol.get("official_metric_capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != set(OFFICIAL_METRICS):
        raise ContractError("Official metric capability matrix is incomplete")
    if payload is None:
        reason = (
            "synthetic fixture dry-run never invokes the official evaluator"
            if fixture_mode
            else "official output not provided"
        )
        return {
            name: {
                "status": "not_run_fixture" if fixture_mode else "not_run",
                "value": None,
                "reason": reason,
                "capability": capabilities[name],
            }
            for name in OFFICIAL_METRICS
        }
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ContractError("Official payload requires a metrics object")
    output: dict[str, dict[str, Any]] = {}
    for name in OFFICIAL_METRICS:
        raw = raw_metrics.get(name)
        if not isinstance(raw, dict) or raw.get("status") not in {
            "available",
            "unavailable",
            "not_run",
        }:
            raise ContractError(f"Official metric {name} requires an explicit status")
        status = str(raw["status"])
        value = raw.get("value")
        if status == "available":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ContractError(
                    f"Official metric {name} available status requires a finite value"
                )
            value = float(value)
        elif value is not None:
            raise ContractError(
                f"Official metric {name} cannot carry a value when {status}"
            )
        reason = raw.get("reason")
        if status != "available" and (not isinstance(reason, str) or not reason):
            raise ContractError(
                f"Official metric {name} {status} status requires a reason"
            )
        output[name] = {
            "status": status,
            "value": value,
            "reason": reason,
            "capability": capabilities[name],
        }
    if output["AR"]["status"] == "available" and any(
        output[name]["status"] != "available" for name in capabilities["AR"]["requires"]
    ):
        raise ContractError(
            "Official AR cannot be available without every required component"
        )
    return output


def _prediction_key(row: Mapping[str, Any]) -> tuple[int, int, int, str, str]:
    values = []
    for name in ("scene_id", "image_id", "object_id"):
        value = row.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"prediction.{name} must be a non-negative integer")
        values.append(value)
    variant = row.get("mask_variant")
    stage = row.get("stage")
    if variant not in EVALUATOR_VARIANTS:
        raise ContractError(f"Unknown prediction mask variant: {variant}")
    if stage not in STAGES:
        raise ContractError(f"Unknown prediction stage: {stage}")
    return values[0], values[1], values[2], str(variant), str(stage)


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def _aggregate(
    rows: Sequence[Mapping[str, Any]], grouping: str, key: Any
) -> dict[str, Any]:
    total = len(rows)
    success = [row for row in rows if row["status"] == "success"]
    missing = sum(row["status"] == "missing" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    return {
        "grouping": grouping,
        "group": key,
        "row_count": total,
        "success_count": len(success),
        "missing_count": missing,
        "failure_count": failed,
        "missing_rate": missing / total if total else 0.0,
        "failure_rate": failed / total if total else 0.0,
        **{f"mean_{name}": _mean(success, name) for name in INTERNAL_METRICS},
    }


def evaluate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_by_key = {
        (
            sample["key"]["scene_id"],
            sample["key"]["image_id"],
            sample["key"]["object_id"],
        ): sample
        for sample in manifest["samples"]
    }
    expected = {
        (*key, variant, stage)
        for key in sample_by_key
        for variant in EVALUATOR_VARIANTS
        for stage in STAGES
    }
    given: dict[tuple[int, int, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _prediction_key(row)
        if key in given:
            raise ContractError(f"Duplicate prediction key: {key}")
        if key not in expected:
            raise ContractError(f"Prediction key is outside manifest: {key}")
        given[key] = row
    normalized: list[dict[str, Any]] = []
    for key in sorted(expected):
        scene_id, image_id, object_id, variant, stage = key
        row = given.get(key)
        status = "missing" if row is None else row.get("status")
        if status not in {"success", "missing", "failed"}:
            raise ContractError(f"Invalid prediction status for {key}: {status}")
        output = {
            "scene_id": scene_id,
            "image_id": image_id,
            "object_id": object_id,
            "mask_variant": variant,
            "stage": stage,
            "status": status,
            **{name: None for name in INTERNAL_METRICS},
        }
        if status == "success":
            if row is None or "model_to_camera_pose_m" not in row:
                raise ContractError(f"Successful prediction lacks a pose: {key}")
            sample = sample_by_key[(scene_id, image_id, object_id)]
            gt_pose = load_pose_asset(data_root, sample["evaluator_only"]["gt_pose"])
            points, _ = load_cad_asset(data_root, sample["producer_inputs"]["cad"])
            output.update(
                pose_errors(
                    points,
                    row["model_to_camera_pose_m"],
                    gt_pose,
                    symmetric=sample["symmetric_object"],
                )
            )
        normalized.append(output)
    grouped: list[dict[str, Any]] = [_aggregate(normalized, "all", "all")]
    groupers = {
        "object": lambda row: row["object_id"],
        "scene": lambda row: row["scene_id"],
        "mask_variant": lambda row: row["mask_variant"],
        "stage": lambda row: row["stage"],
        "mask_variant_stage": lambda row: f"{row['mask_variant']}::{row['stage']}",
    }
    for name, function in groupers.items():
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in normalized:
            buckets[function(row)].append(row)
        grouped.extend(
            _aggregate(bucket, name, key)
            for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0]))
        )
    paired = _paired_deltas(normalized)
    return normalized, grouped, paired


def _paired_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successful = {
        (
            row["scene_id"],
            row["image_id"],
            row["object_id"],
            row["mask_variant"],
            row["stage"],
        ): row
        for row in rows
        if row["status"] == "success"
    }
    outputs: list[dict[str, Any]] = []
    sample_keys = sorted({key[:3] for key in successful})
    for stage in STAGES:
        for variant in EVALUATOR_VARIANTS:
            if variant == "predicted_mask":
                continue
            deltas = []
            for sample_key in sample_keys:
                reference = successful.get((*sample_key, "predicted_mask", stage))
                candidate = successful.get((*sample_key, variant, stage))
                if reference is None or candidate is None:
                    continue
                deltas.append(
                    {
                        name: float(candidate[name]) - float(reference[name])
                        for name in INTERNAL_METRICS
                    }
                )
            outputs.append(
                {
                    "comparison": f"{variant}-minus-predicted_mask",
                    "stage": stage,
                    "paired_count": len(deltas),
                    **{
                        f"mean_delta_{name}": float(
                            np.mean([row[name] for row in deltas])
                        )
                        if deltas
                        else None
                        for name in INTERNAL_METRICS
                    },
                }
            )
    for variant in EVALUATOR_VARIANTS:
        deltas = []
        for sample_key in sample_keys:
            baseline = successful.get((*sample_key, variant, "baseline"))
            final = successful.get((*sample_key, variant, "final"))
            if baseline is None or final is None:
                continue
            deltas.append(
                {
                    name: float(final[name]) - float(baseline[name])
                    for name in INTERNAL_METRICS
                }
            )
        outputs.append(
            {
                "comparison": "final-minus-baseline",
                "mask_variant": variant,
                "paired_count": len(deltas),
                **{
                    f"mean_delta_{name}": float(np.mean([row[name] for row in deltas]))
                    if deltas
                    else None
                    for name in INTERNAL_METRICS
                },
            }
        )
    return outputs


def synthetic_prediction_rows(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any], *, data_root: Path
) -> list[dict[str, Any]]:
    """Fixture-only rows that exercise schemas and calculations without a model."""
    magnitudes = protocol.get("fixture_prediction_perturbations")
    if not isinstance(magnitudes, dict) or set(magnitudes) != set(EVALUATOR_VARIANTS):
        raise ContractError("Fixture prediction perturbations are incomplete")
    rows = []
    for sample in manifest["samples"]:
        gt_pose = load_pose_asset(data_root, sample["evaluator_only"]["gt_pose"])
        for variant in EVALUATOR_VARIANTS:
            parameters = magnitudes[variant]
            baseline_rotation = float(parameters["baseline_rotation_degrees"])
            baseline_translation = float(parameters["baseline_translation_mm"])
            final_rotation = float(parameters["final_rotation_degrees"])
            final_translation = float(parameters["final_translation_mm"])
            for stage, rotation, translation in (
                ("baseline", baseline_rotation, baseline_translation),
                ("final", final_rotation, final_translation),
            ):
                rows.append(
                    {
                        **sample["key"],
                        "mask_variant": variant,
                        "stage": stage,
                        "status": "success",
                        "model_to_camera_pose_m": perturb_pose(
                            gt_pose,
                            rotation_axis="y",
                            rotation_degrees=rotation,
                            translation_axis="x",
                            translation_mm=translation,
                        ).tolist(),
                        "source": "CPU_SYNTHETIC_FIXTURE_DRY_RUN",
                    }
                )
    return rows


def write_report(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    output_root: Path,
    official_payload: Mapping[str, Any] | None = None,
    fixture_mode: bool,
) -> dict[str, Any]:
    normalized, grouped, paired = evaluate_prediction_rows(
        rows, manifest, data_root=data_root
    )
    official = official_metric_status(
        protocol, official_payload, fixture_mode=fixture_mode
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "internal-metrics.csv", normalized)
    _write_csv(output_root / "grouped-metrics.csv", grouped)
    _write_csv(output_root / "paired-deltas.csv", paired)
    _write_csv(
        output_root / "official-metrics.csv",
        [{"metric": name, **value} for name, value in official.items()],
    )
    report = {
        "schema_version": "poseloop.pose-accuracy-recovery.metrics.v1",
        "protocol_id": protocol["protocol_id"],
        "execution_mode": "CPU_SYNTHETIC_FIXTURE_DRY_RUN"
        if fixture_mode
        else "DEVELOPMENT_EVALUATOR_ONLY",
        "official_metrics": official,
        "internal_metrics": {
            "definitions": {
                "add_s_mm": "ADD for asymmetric objects and ADI-style closest-point distance for symmetric objects",
                "rotation_error_degrees": "SO(3) geodesic angle",
                "translation_error_mm": "Euclidean translation difference",
            },
            "rows": normalized,
            "grouped": grouped,
            "paired_deltas": paired,
        },
        "accuracy_claim_permitted": False,
        "oracle_deployment_conclusion_permitted": False,
        "dry_run_is_result": False,
    }
    report["lock_sha256"] = canonical_sha256(report)
    write_json(output_root / "metrics.json", report)
    return report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ContractError(
                    f"JSONL row at {path}:{line_number} must be an object"
                )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            for row in materialized:
                writer.writerow(
                    {
                        name: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                        for name, value in row.items()
                    }
                )
