#!/usr/bin/env python3
"""Persist the final YCB-V sealed input failure without running inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from build_m5_r5_lmo_development import missing_gap_frame_counts
from build_m5_r6a_ycbv_sealed import (
    PROTOCOL_ID,
    STAGE_ID,
    load_target_rows,
    select_object_window,
)
from m1_common import sha256_file, write_json_atomic


EXPECTED_BUILDER_FAILURE = "YCB-V sealed input has too few missing-represented objects"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/cgliu/datasets/ycbv")
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/cgliu/datasets/archives/ycbv"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root
        / "protocols"
        / "poseloop_m5_r6a_ycbv_sealed_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "artifacts"
        / "r2"
        / "m5_r6a_ycbv_sealed_failure"
        / "input_failure_audit.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root
        / "reports"
        / "r2"
        / "m5_r6a_ycbv_sealed_input_failure.md",
    )
    return parser.parse_args()


def build_gate_checks(
    summary: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    gate = protocol["evaluation"]["sealed_gate"]
    selection = protocol["selection"]
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, observed: int, required: int) -> None:
        checks[name] = {
            "observed": int(observed),
            "required_minimum": int(required),
            "passed": int(observed) >= int(required),
        }

    add(
        "represented_object_count",
        int(summary["represented_object_count"]),
        int(gate["represented_object_count_min"]),
    )
    add(
        "missing_represented_object_count",
        int(summary["missing_represented_object_count"]),
        int(gate["missing_represented_object_count_min"]),
    )
    add(
        "natural_missing_frame_count",
        int(summary["natural_missing_frame_count"]),
        int(gate["natural_missing_frame_count_min"]),
    )
    for name, minimum in selection["missing_gap_frame_minimums"].items():
        add(
            f"missing_gap_{name}_frame_count",
            int(summary["missing_gap_frame_counts"].get(name, 0)),
            int(minimum),
        )
    return checks


def render_report(result: Mapping[str, Any]) -> str:
    checks = result["gate_checks"]
    by_object = result["by_object"]
    zero_missing = [
        object_id
        for object_id, row in by_object.items()
        if int(row["natural_missing_frame_count"]) == 0
    ]
    lines = [
        "# PoseLoop M5-R6A-S1 YCB-V sealed input failure",
        "",
        "Status: **FAIL_M5_R6A_SEALED_INPUT**",
        "",
        (
            "The one permitted frozen builder invocation stopped before FoundationPose "
            f"inference with `{result['builder_failure']['message']}`. The sealed labels "
            "were opened, so this protocol is final and was not relaxed or rerun."
        ),
        "",
        "| Frozen input gate | Required | Observed | Result |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "represented_object_count": "Represented objects",
        "missing_represented_object_count": "Objects with natural missing frames",
        "natural_missing_frame_count": "Natural missing frames",
        "missing_gap_short_gap_1_2_frame_count": "Short-gap frames",
        "missing_gap_medium_gap_3_7_frame_count": "Medium-gap frames",
        "missing_gap_long_gap_8_plus_frame_count": "Long-gap frames",
    }
    for name, check in checks.items():
        lines.append(
            f"| {labels.get(name, name)} | {check['required_minimum']} | "
            f"{check['observed']} | {'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Objects with zero natural missing frames in their frozen best windows: {', '.join(zero_missing) or 'none'}.",
            "",
            "| Object | Scene | Start frame | Available | Missing | Short | Medium | Long |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for object_id, row in sorted(by_object.items(), key=lambda item: int(item[0])):
        gaps = row["missing_gap_frame_counts"]
        lines.append(
            f"| {object_id} | {row['scene_id']} | {row['source_start_image_id']} | "
            f"{row['available_frame_count']} | {row['natural_missing_frame_count']} | "
            f"{gaps.get('short_gap_1_2', 0)} | "
            f"{gaps.get('medium_gap_3_7', 0)} | "
            f"{gaps.get('long_gap_8_plus', 0)} |"
        )
    lines.extend(
        [
            "",
            "No prediction, pose error, candidate comparison, GPU inference, or sealed numerical evaluation was run. This audit only replays the already-frozen source-support selection to preserve the input failure evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    archive_root = args.archive_root.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    frozen_output = repo_root / "artifacts" / "r2" / "m5_r6a_ycbv_sealed"
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to replace sealed failure evidence: {path}")
    if frozen_output.exists():
        raise RuntimeError(f"Unexpected sealed bundle exists after input failure: {frozen_output}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected YCB-V sealed protocol")

    source_receipt: dict[str, dict[str, Any]] = {}
    for key in ("base", "models", "sealed"):
        expected = protocol["source"]["archives"][key]
        path = archive_root / str(expected["filename"])
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(expected["size_bytes"]) or actual_hash != str(
            expected["sha256"]
        ):
            raise RuntimeError(f"YCB-V archive provenance mismatch: {path}")
        source_receipt[key] = {
            "path": str(path),
            "size_bytes": actual_size,
            "sha256": actual_hash,
        }

    rows_by_object, source_adapter_audit, scene_ids = load_target_rows(
        dataset_root, protocol
    )
    selection = protocol["selection"]
    selected = {
        object_id: window
        for object_id, rows in rows_by_object.items()
        if (window := select_object_window(rows, selection)) is not None
    }
    by_object: dict[str, dict[str, Any]] = {}
    gap_counts: Counter[str] = Counter()
    for object_id, rows in sorted(selected.items()):
        availability = [row.available for row in rows]
        gaps = missing_gap_frame_counts(availability)
        gap_counts.update(gaps)
        by_object[str(object_id)] = {
            "source_gt_present_frame_count": len(rows_by_object[object_id]),
            "scene_id": rows[0].scene_id,
            "source_start_image_id": rows[0].image_id,
            "source_end_image_id": rows[-1].image_id,
            "replay_frame_count": len(rows),
            "available_frame_count": sum(availability),
            "natural_missing_frame_count": sum(not value for value in availability),
            "missing_gap_frame_counts": dict(sorted(gaps.items())),
        }

    summary = {
        "represented_object_count": len(selected),
        "missing_represented_object_count": sum(
            int(row["natural_missing_frame_count"]) > 0 for row in by_object.values()
        ),
        "natural_missing_frame_count": sum(
            int(row["natural_missing_frame_count"]) for row in by_object.values()
        ),
        "missing_gap_frame_counts": dict(sorted(gap_counts.items())),
    }
    gate_checks = build_gate_checks(summary, protocol)
    failed_gates = [name for name, row in gate_checks.items() if not row["passed"]]
    if failed_gates != ["missing_represented_object_count"]:
        raise RuntimeError(
            "Failure audit does not reproduce the observed frozen builder stop: "
            f"failed_gates={failed_gates}"
        )

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "FAIL_M5_R6A_SEALED_INPUT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder_failure": {
            "exception_type": "RuntimeError",
            "message": EXPECTED_BUILDER_FAILURE,
            "exit_code": 1,
        },
        "failed_gates": failed_gates,
        "gate_checks": gate_checks,
        "scene_ids": scene_ids,
        **summary,
        "by_object": by_object,
        "source_adapter_audit": source_adapter_audit,
        "source_archives": source_receipt,
        "provenance": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "frozen_builder": {
                "path": str(repo_root / "scripts" / "build_m5_r6a_ycbv_sealed.py"),
                "sha256": sha256_file(
                    repo_root / "scripts" / "build_m5_r6a_ycbv_sealed.py"
                ),
            },
            "failure_auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "sealed_archive_or_label_read": True,
        "sealed_labels_opened": True,
        "protocol_relaxed_or_rerun": False,
        "selection_reads_method_prediction_or_pose_error": False,
        "foundationpose_inference_invocation_count": 0,
        "sealed_evaluation_invocation_count": 0,
        "prediction_or_pose_error_read": False,
        "artificial_frame_deletion": False,
        "sealed_candidate_id": "measurement_first",
    }
    write_json_atomic(output_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
