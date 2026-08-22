#!/usr/bin/env python3
"""Validate the frozen M5-R6A RU-APC bundle before GPU inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_m5_r5_lmo_development import missing_gap_frame_counts
from m1_common import load_jsonl, sha256_file


FORBIDDEN_INFERENCE_FIELDS = {
    "gt_model_to_camera_pose_m",
    "natural_input_missing",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=repo_root / "artifacts" / "r2" / "m5_r6a" / "contract.json",
    )
    return parser.parse_args()


def main() -> None:
    contract_path = parse_args().contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != "poseloop-m5-r6a-v1"
        or contract.get("stage_id") != "M5-R6A"
        or contract.get("status") != "development_labels_opened_bundle_ready"
        or not bool(contract.get("development_labels_opened"))
        or bool(contract.get("sealed_archive_or_label_read"))
    ):
        raise RuntimeError("Invalid M5-R6A contract identity or label boundary")

    expected_bundle_files = {
        "contract.json",
        "evaluator_labels.jsonl",
        "inference_manifest.jsonl",
        "selection_summary.json",
        "track_manifest.jsonl",
    }
    observed_bundle_files = {
        path.name for path in contract_path.parent.iterdir() if path.is_file()
    }
    if observed_bundle_files != expected_bundle_files:
        raise RuntimeError(
            f"Unexpected M5-R6A bundle files: {sorted(observed_bundle_files)}"
        )

    protocol_path = Path(str(contract["protocol"]["path"]))
    if sha256_file(protocol_path) != str(contract["protocol"]["sha256"]):
        raise RuntimeError("M5-R6A protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != contract["protocol_id"]:
        raise RuntimeError("M5-R6A protocol identity mismatch")
    for row in contract["source_archives"].values():
        if sha256_file(Path(str(row["path"]))) != str(row["sha256"]):
            raise RuntimeError("M5-R6A source archive hash mismatch")
    for name, row in contract["code"].items():
        if sha256_file(Path(str(row["path"]))) != str(row["sha256"]):
            raise RuntimeError(f"M5-R6A code hash mismatch: {name}")
    for name, row in contract["files"].items():
        if sha256_file(Path(str(row["path"]))) != str(row["sha256"]):
            raise RuntimeError(f"M5-R6A bundle file hash mismatch: {name}")

    summary_path = Path(str(contract["files"]["selection_summary"]["path"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary != contract["selection_summary"]:
        raise RuntimeError("M5-R6A embedded selection summary mismatch")
    tracks = load_jsonl(Path(str(contract["files"]["track_manifest"]["path"])))
    labels_list = load_jsonl(
        Path(str(contract["files"]["evaluator_labels"]["path"]))
    )
    inference_rows = load_jsonl(
        Path(str(contract["files"]["inference_manifest"]["path"]))
    )
    if len(tracks) != int(summary["track_count"]):
        raise RuntimeError("M5-R6A track count mismatch")
    if len(labels_list) != int(summary["replay_frame_count"]):
        raise RuntimeError("M5-R6A evaluator-label count mismatch")
    if len(inference_rows) != int(summary["inference_sample_count"]):
        raise RuntimeError("M5-R6A inference count mismatch")

    inference_ids: set[str] = set()
    for row in inference_rows:
        sample_id = str(row["sample_id"])
        if sample_id in inference_ids:
            raise RuntimeError(f"Duplicate M5-R6A inference ID: {sample_id}")
        inference_ids.add(sample_id)
        if row.get("sensor_modality") != "ruapc_rgbd" or not bool(
            row.get("input_available")
        ):
            raise RuntimeError(f"Invalid M5-R6A inference input: {sample_id}")
        leaked = FORBIDDEN_INFERENCE_FIELDS.intersection(row)
        if leaked or bool(row.get("evaluator_label_read")):
            raise RuntimeError(
                f"M5-R6A evaluator field leaked into inference {sample_id}: {leaked}"
            )

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for row in labels_list:
        key = (str(row["track_id"]), int(row["replay_frame_index"]))
        if key in labels or "gt_model_to_camera_pose_m" not in row:
            raise RuntimeError(f"Invalid M5-R6A evaluator label: {key}")
        labels[key] = row

    track_ids: set[str] = set()
    replay_ids: set[str] = set()
    reconstructed_inference_ids: set[str] = set()
    reconstructed_gap_counts = Counter()
    by_object: dict[str, dict[str, int]] = {}
    for track in tracks:
        track_id = str(track["track_id"])
        if track_id in track_ids:
            raise RuntimeError(f"Duplicate M5-R6A track ID: {track_id}")
        track_ids.add(track_id)
        frames = list(track["frames"])
        if len(frames) != 128 or not bool(frames[0]["input_available"]):
            raise RuntimeError(f"Invalid M5-R6A track initialization: {track_id}")
        availability: list[bool] = []
        for index, frame in enumerate(frames):
            if int(frame["replay_frame_index"]) != index:
                raise RuntimeError(f"Non-contiguous M5-R6A replay index: {track_id}")
            if float(frame["timestamp_s"]) != index / 30.0:
                raise RuntimeError(f"Invalid M5-R6A nominal timestamp: {track_id}/{index}")
            sample_id = str(frame["sample_id"])
            if sample_id in replay_ids:
                raise RuntimeError(f"Duplicate M5-R6A replay ID: {sample_id}")
            replay_ids.add(sample_id)
            available = bool(frame["input_available"])
            availability.append(available)
            label = labels[(track_id, index)]
            if str(label["sample_id"]) != sample_id or bool(
                label["natural_input_missing"]
            ) == available:
                raise RuntimeError(f"M5-R6A label/track mismatch: {track_id}/{index}")
            support = (
                frame["input_mask_area_fraction"],
                frame["valid_depth_ratio_inside_mask"],
            )
            if available:
                if any(value is None for value in support):
                    raise RuntimeError(f"Missing M5-R6A support diagnostics: {sample_id}")
                reconstructed_inference_ids.add(sample_id)
            elif any(value is not None for value in support):
                raise RuntimeError(f"Unavailable M5-R6A frame carries support: {sample_id}")
        reconstructed_gap_counts.update(missing_gap_frame_counts(availability))
        object_id = str(int(track["object_id"]))
        by_object[object_id] = {
            "available_frame_count": sum(availability),
            "natural_missing_frame_count": sum(not value for value in availability),
        }

    if reconstructed_inference_ids != inference_ids:
        raise RuntimeError("M5-R6A inference/track ID sets differ")
    if len(labels) != len(replay_ids):
        raise RuntimeError("M5-R6A evaluator/track key sets differ")
    if dict(sorted(reconstructed_gap_counts.items())) != summary[
        "missing_gap_frame_counts"
    ]:
        raise RuntimeError("M5-R6A missing-gap counts differ")
    for object_id, counts in by_object.items():
        expected = summary["by_object"][object_id]
        if counts["available_frame_count"] != int(expected["available_frame_count"]) or counts[
            "natural_missing_frame_count"
        ] != int(expected["natural_missing_frame_count"]):
            raise RuntimeError(f"M5-R6A object summary differs: {object_id}")

    gate = protocol["evaluation"]["development_gate"]
    checks = {
        "represented_object_count": len(by_object)
        >= int(gate["represented_object_count_min"]),
        "track_count": len(tracks) >= int(gate["track_count_min"]),
        "missing_represented_object_count": sum(
            row["natural_missing_frame_count"] > 0 for row in by_object.values()
        )
        >= int(gate["missing_represented_object_count_min"]),
        "natural_missing_frame_count": sum(
            row["natural_missing_frame_count"] for row in by_object.values()
        )
        >= int(gate["natural_missing_frame_count_min"]),
    }
    for name, minimum in protocol["selection"]["missing_gap_frame_minimums"].items():
        checks[name] = int(reconstructed_gap_counts[name]) >= int(minimum)
    if not all(checks.values()):
        raise RuntimeError(f"M5-R6A input gate failed: {checks}")

    receipt = {
        "status": "PASS_M5_R6A_BUNDLE_VALIDATION",
        "protocol_id": contract["protocol_id"],
        "contract_sha256": sha256_file(contract_path),
        "track_count": len(tracks),
        "replay_frame_count": len(replay_ids),
        "inference_sample_count": len(inference_ids),
        "natural_missing_frame_count": sum(
            row["natural_missing_frame_count"] for row in by_object.values()
        ),
        "missing_gap_frame_counts": dict(sorted(reconstructed_gap_counts.items())),
        "input_gate_checks": checks,
        "inference_evaluator_field_leak_count": 0,
        "sealed_archive_or_label_read": False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
