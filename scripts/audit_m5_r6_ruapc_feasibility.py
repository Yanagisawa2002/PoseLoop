#!/usr/bin/env python3
"""Audit frozen M5-R6 RU-APC replay feasibility without running inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_m5_r5_lmo_development import (
    _consecutive_segments,
    missing_gap_frame_counts,
    select_best_window,
)
from build_m5_r6_ruapc_development import FrameObservation, load_target_rows


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/cgliu/datasets/ruapc")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_m5_r6_protocol.json",
    )
    return parser.parse_args()


def best_window(
    rows: Sequence[FrameObservation], selection: Mapping[str, Any]
) -> list[FrameObservation] | None:
    length = int(selection["window_length_frames"])
    minimum_available = int(selection["eligible_window"]["available_frame_count_min"])
    candidates: list[tuple[int, int, list[FrameObservation]]] = []
    for segment in _consecutive_segments(rows):
        window = select_best_window(
            [row.available for row in segment],
            window_length=length,
            minimum_available=minimum_available,
            minimum_missing=0,
        )
        if window is None:
            continue
        start, stop = window
        selected = segment[start:stop]
        candidates.append(
            (-sum(not row.available for row in selected), selected[0].image_id, selected)
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    rows_by_object, source_audit = load_target_rows(
        args.dataset_root.resolve(), protocol
    )
    selection = protocol["selection"]
    by_object: dict[str, dict[str, Any]] = {}
    gap_counts = Counter()
    missing_objects = 0
    total_missing = 0
    for object_id, rows in sorted(rows_by_object.items()):
        window = best_window(rows, selection)
        if window is None:
            by_object[str(object_id)] = {
                "source_gt_present_frame_count": len(rows),
                "has_128_frame_window_starting_available": False,
            }
            continue
        availability = [row.available for row in window]
        missing = sum(not value for value in availability)
        gaps = missing_gap_frame_counts(availability)
        gap_counts.update(gaps)
        total_missing += missing
        missing_objects += int(missing > 0)
        by_object[str(object_id)] = {
            "source_gt_present_frame_count": len(rows),
            "has_128_frame_window_starting_available": True,
            "source_start_image_id": window[0].image_id,
            "available_frame_count": sum(availability),
            "natural_missing_frame_count": missing,
            "current_r6_window_eligible": missing
            >= int(selection["eligible_window"]["natural_missing_frame_count_min"]),
            "missing_gap_frame_counts": gaps,
            "source_mask_area_fraction_min": min(row.mask_area_fraction for row in rows),
            "source_valid_depth_ratio_min": min(row.valid_depth_ratio for row in rows),
            "source_valid_depth_pixels_min": min(row.valid_depth_pixels for row in rows),
        }
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "FAIL_INPUT_FEASIBILITY",
        "audit_scope": "support and source-frame continuity only; no FoundationPose inference or pose error",
        "represented_object_count_with_128_frame_window": sum(
            bool(row["has_128_frame_window_starting_available"])
            for row in by_object.values()
        ),
        "current_r6_eligible_object_count": sum(
            bool(row.get("current_r6_window_eligible")) for row in by_object.values()
        ),
        "missing_represented_object_count_if_all_windows_retained": missing_objects,
        "natural_missing_frame_count_if_all_windows_retained": total_missing,
        "missing_gap_frame_counts_if_all_windows_retained": dict(
            sorted(gap_counts.items())
        ),
        "by_object": by_object,
        "source_adapter_audit": source_audit,
        "method_prediction_or_error_read": False,
        "sealed_archive_or_label_read": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
