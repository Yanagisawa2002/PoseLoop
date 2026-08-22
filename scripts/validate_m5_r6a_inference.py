#!/usr/bin/env python3
"""Validate completed label-blind M5-R6A inference before evaluator-label use."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from m1_common import assert_pose, load_jsonl, sha256_file


FORBIDDEN_PREDICTION_FIELDS = {
    "gt_model_to_camera_pose_m",
    "natural_input_missing",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r6a"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "contract.json")
    parser.add_argument("--predictions", type=Path, default=root / "predictions.jsonl")
    parser.add_argument("--receipt", type=Path, default=root / "inference_receipt.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.resolve().read_text(encoding="utf-8"))
    predictions_path = args.predictions.resolve()
    if (
        contract.get("protocol_id") != "poseloop-m5-r6a-v1"
        or receipt.get("protocol_id") != contract["protocol_id"]
        or receipt.get("status") != "complete"
        or int(receipt.get("sample_count", -1))
        != int(contract["selection_summary"]["inference_sample_count"])
        or bool(receipt.get("evaluator_labels_path_read"))
        or bool(receipt.get("evaluator_pose_or_error_computed"))
        or bool(receipt.get("sealed_archive_or_label_read"))
    ):
        raise RuntimeError("Invalid M5-R6A inference receipt boundary")
    if sha256_file(predictions_path) != str(receipt["predictions"]["sha256"]):
        raise RuntimeError("M5-R6A predictions hash mismatch")
    manifest_path = Path(str(contract["files"]["inference_manifest"]["path"]))
    if (
        sha256_file(manifest_path) != str(receipt["manifest_sha256"])
        or str(receipt.get("sensor_modality")) != "ruapc_rgbd"
        or receipt.get("status_counts") != {"success": receipt["sample_count"]}
    ):
        raise RuntimeError("M5-R6A receipt manifest, modality, or status mismatch")

    manifest = load_jsonl(manifest_path)
    output = load_jsonl(predictions_path)
    if not output or output[0].get("record_type") != "m5_r5_inference_metadata":
        raise RuntimeError("M5-R6A inference metadata is missing")
    metadata = output[0]
    if (
        metadata.get("protocol_id") != contract["protocol_id"]
        or metadata.get("stage_id") != "M5-R6A"
        or metadata.get("dataset") != "RU-APC"
        or metadata.get("sensor_modality") != "ruapc_rgbd"
        or bool(metadata.get("evaluator_labels_path_read"))
        or bool(metadata.get("evaluator_pose_or_error_computed"))
    ):
        raise RuntimeError("Invalid M5-R6A inference metadata boundary")
    prediction_rows = output[1:]
    if len(prediction_rows) != len(manifest):
        raise RuntimeError("M5-R6A prediction count does not match manifest")

    statuses = Counter()
    runtimes: list[float] = []
    peak_memory: list[int] = []
    hypothesis_counts: list[int] = []
    for expected, row in zip(manifest, prediction_rows, strict=True):
        sample_id = str(expected["sample_id"])
        if row.get("record_type") != "m5_r5_development_prediction":
            raise RuntimeError(f"Invalid M5-R6A prediction record: {sample_id}")
        for field in (
            "sample_id",
            "scene_id",
            "image_id",
            "gt_instance_index",
            "object_id",
            "sensor_modality",
        ):
            if row.get(field) != expected.get(field):
                raise RuntimeError(
                    f"M5-R6A prediction identity mismatch: {sample_id}/{field}"
                )
        leaked = FORBIDDEN_PREDICTION_FIELDS.intersection(row)
        if leaked or bool(row.get("evaluator_label_read")):
            raise RuntimeError(
                f"M5-R6A evaluator field leaked into prediction {sample_id}: {leaked}"
            )
        status = str(row.get("status"))
        statuses[status] += 1
        if status != "success":
            raise RuntimeError(f"M5-R6A prediction failed: {sample_id}/{status}")
        predicted_pose = np.asarray(
            row["predicted_model_to_camera_pose_m"], dtype=np.float64
        )
        assert_pose(predicted_pose, f"M5-R6A prediction {sample_id}")
        if predicted_pose[2, 3] <= 0:
            raise RuntimeError(f"M5-R6A pose is behind camera: {sample_id}")
        runtime = float(row["registration_seconds"])
        memory = int(row["cuda_peak_memory_bytes"])
        hypotheses = int(row["pose_hypothesis_count"])
        scores = (
            row.get("foundationpose_top_score"),
            row.get("foundationpose_top_score_margin"),
        )
        if (
            not np.isfinite(runtime)
            or runtime < 0
            or memory <= 0
            or hypotheses <= 0
            or any(value is not None and not np.isfinite(float(value)) for value in scores)
        ):
            raise RuntimeError(f"Invalid M5-R6A runtime diagnostics: {sample_id}")
        runtimes.append(runtime)
        peak_memory.append(memory)
        hypothesis_counts.append(hypotheses)

    runtime_array = np.asarray(runtimes, dtype=np.float64)
    result = {
        "status": "PASS_M5_R6A_INFERENCE_VALIDATION",
        "protocol_id": contract["protocol_id"],
        "prediction_sha256": sha256_file(predictions_path),
        "sample_count": len(prediction_rows),
        "status_counts": dict(sorted(statuses.items())),
        "registration_seconds": {
            "mean": float(np.mean(runtime_array)),
            "median": float(np.median(runtime_array)),
            "p95": float(np.quantile(runtime_array, 0.95)),
            "maximum": float(np.max(runtime_array)),
        },
        "cuda_peak_memory_bytes_max": max(peak_memory),
        "pose_hypothesis_count_values": sorted(set(hypothesis_counts)),
        "evaluator_field_leak_count": 0,
        "raw_foundationpose_scores_used_for_filtering_or_confidence": False,
        "sealed_archive_or_label_read": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
