#!/usr/bin/env python3
"""Evaluator-only frozen failure labels for PoseLoop M6-G0.

This module deliberately has no feature-matrix API.  It reads the existing M2
method-result rows only after inference-time feature construction has finished,
then joins labels to those rows by the frozen M2 group identity.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from m1_common import load_jsonl


SCHEMA_VERSION = 1
FROZEN_METHOD = "symmetry_aware_medoid"
FROZEN_VIEW_BUDGET = 5
MSSD_THRESHOLD_DIAMETERS = 0.10
MSPD_THRESHOLD_R = 10.0


def assert_m2_label_path(path: Path) -> Path:
    """Accept only the exact M2 metrics artifact and reject M3-like paths."""
    resolved = path.resolve()
    normalized = resolved.as_posix().lower()
    if "/m3/" in normalized or "/m3_" in normalized or "reports/m3" in normalized:
        raise ValueError(f"M3 paths are forbidden for M6-G0 labels: {resolved}")
    if normalized.endswith("/artifacts/m2/metrics.jsonl") is False:
        raise ValueError(
            f"M6-G0 labels must come from artifacts/m2/metrics.jsonl, got {resolved}"
        )
    return resolved


def _feature_metadata_index(
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(feature_rows, start=1):
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Feature row {row_number} has no metadata mapping")
        target_id = str(
            metadata.get("target_id")
            or metadata.get("target_sample_id")
            or metadata.get("group_id")
            or ""
        )
        if not target_id or target_id in indexed:
            raise ValueError(f"Missing or duplicate feature target ID: {target_id!r}")
        indexed[target_id] = metadata
    return indexed


def _pose_matches(metadata: Mapping[str, Any], metric: Mapping[str, Any]) -> bool:
    expected = metadata.get("output_pose_m")
    if expected is None:
        expected = metadata.get("predicted_model_to_target_camera_pose_m")
    observed = metric.get("predicted_model_to_target_camera_pose_m")
    if expected is None or observed is None:
        return expected is None and observed is None
    expected_array = np.asarray(expected, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    return bool(
        expected_array.shape == (4, 4)
        and observed_array.shape == (4, 4)
        and np.allclose(expected_array, observed_array, rtol=0.0, atol=1e-9)
    )


def build_evaluator_rows(
    metrics_rows: Iterable[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create evaluator-only labels for the frozen k=5 medoid output."""
    metadata_by_target = _feature_metadata_index(feature_rows)
    selected: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(metrics_rows, start=1):
        if row.get("record_type") != "method_result":
            raise ValueError(f"Unexpected M2 metric record at row {row_number}")
        if (
            str(row.get("method")) != FROZEN_METHOD
            or int(row.get("requested_view_budget", -1)) != FROZEN_VIEW_BUDGET
        ):
            continue
        target_id = str(row.get("target_sample_id", ""))
        if not target_id or target_id in selected:
            raise ValueError(
                f"Missing or duplicate frozen metric target: {target_id!r}"
            )
        selected[target_id] = row

    if set(selected) != set(metadata_by_target):
        missing = sorted(set(metadata_by_target) - set(selected))
        extra = sorted(set(selected) - set(metadata_by_target))
        raise ValueError(
            "Frozen label/feature target mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    evaluator_rows: list[dict[str, Any]] = []
    for target_id in sorted(metadata_by_target):
        metadata = metadata_by_target[target_id]
        metric = selected[target_id]
        diagnostic = metric.get("diagnostic_success")
        if not isinstance(diagnostic, Mapping) or "joint" not in diagnostic:
            raise ValueError(f"Missing frozen joint correctness for {target_id}")
        correct = bool(diagnostic["joint"])
        selected_sample = str(metric.get("selected_sample_id", ""))
        feature_selected = str(
            metadata.get("selected_sample_id") or metadata.get("output_sample_id") or ""
        )
        if feature_selected and selected_sample != feature_selected:
            raise ValueError(
                f"Frozen medoid selection mismatch for {target_id}: "
                f"feature={feature_selected}, metric={selected_sample}"
            )
        if not _pose_matches(metadata, metric):
            raise ValueError(f"Frozen medoid output-pose mismatch for {target_id}")

        normalized_mssd = metric.get("normalized_mssd")
        mspd_px = metric.get("mspd_px")
        scale_r = float(metric["mspd_scale_r"])
        if bool(metric.get("finite_pose")):
            if normalized_mssd is None or mspd_px is None:
                raise ValueError(f"Finite frozen output lacks pose errors: {target_id}")
            normalized_mssd = float(normalized_mssd)
            mspd_px = float(mspd_px)
            if not math.isfinite(normalized_mssd) or not math.isfinite(mspd_px):
                raise ValueError(f"Non-finite frozen output error: {target_id}")
            reconstructed = bool(
                normalized_mssd <= MSSD_THRESHOLD_DIAMETERS
                and mspd_px <= MSPD_THRESHOLD_R * scale_r
            )
        else:
            normalized_mssd = None
            mspd_px = None
            reconstructed = False
        if reconstructed != correct:
            raise ValueError(
                f"Frozen joint correctness reconstruction failed: {target_id}"
            )

        object_id = int(metadata.get("object_id", metric["object_id"]))
        physical_instance_id = str(
            metadata.get("physical_instance_id")
            or metadata.get("physical_instance_track_id")
            or metadata.get("track_id")
            or ""
        )
        if not physical_instance_id:
            raise ValueError(f"Missing physical-instance group for {target_id}")
        evaluator_rows.append(
            {
                "record_type": "m6_g0_evaluator_row",
                "schema_version": SCHEMA_VERSION,
                "target_id": target_id,
                "object_id": object_id,
                "physical_instance_id": physical_instance_id,
                "outer_split_source": "M2 development only",
                "frozen_method": FROZEN_METHOD,
                "frozen_view_budget": FROZEN_VIEW_BUDGET,
                "selected_sample_id": selected_sample,
                "correct": correct,
                "y_failure": int(not correct),
                "finite_pose": bool(metric.get("finite_pose")),
                "normalized_mssd": normalized_mssd,
                "mspd_px": mspd_px,
                "mspd_scale_r": scale_r,
                "mssd_threshold_diameters": MSSD_THRESHOLD_DIAMETERS,
                "mspd_threshold_r": MSPD_THRESHOLD_R,
            }
        )
    return evaluator_rows


def load_evaluator_rows(
    metrics_path: Path,
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Load the exact allowlisted M2 metrics input and construct labels."""
    return build_evaluator_rows(
        load_jsonl(assert_m2_label_path(metrics_path)),
        feature_rows,
    )


def label_support_summary(
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the frozen label and evaluate the minimum support gate."""
    if not evaluator_rows:
        raise ValueError("Evaluator row set is empty")
    failures = sum(int(row["y_failure"]) for row in evaluator_rows)
    successes = len(evaluator_rows) - failures
    by_object: dict[int, Counter[int]] = defaultdict(Counter)
    by_instance: dict[str, Counter[int]] = defaultdict(Counter)
    for row in evaluator_rows:
        label = int(row["y_failure"])
        by_object[int(row["object_id"])][label] += 1
        by_instance[str(row["physical_instance_id"])][label] += 1
    object_rows = [
        {
            "object_id": object_id,
            "target_count": counts[0] + counts[1],
            "success_count": counts[0],
            "failure_count": counts[1],
            "failure_prevalence": counts[1] / (counts[0] + counts[1]),
        }
        for object_id, counts in sorted(by_object.items())
    ]
    instance_rows = [
        {
            "physical_instance_id": instance_id,
            "target_count": counts[0] + counts[1],
            "success_count": counts[0],
            "failure_count": counts[1],
        }
        for instance_id, counts in sorted(by_instance.items())
    ]
    failure_instances = sum(row["failure_count"] > 0 for row in instance_rows)
    failure_objects = sum(row["failure_count"] > 0 for row in object_rows)
    criteria = {
        "failure_count_at_least_30": failures >= 30,
        "success_count_at_least_30": successes >= 30,
        "failure_instance_count_at_least_10": failure_instances >= 10,
        "failure_object_count_at_least_5": failure_objects >= 5,
    }
    return {
        "target_count": len(evaluator_rows),
        "failure_count": failures,
        "success_count": successes,
        "failure_prevalence": failures / len(evaluator_rows),
        "physical_instance_count": len(instance_rows),
        "physical_instances_with_failures": failure_instances,
        "object_count": len(object_rows),
        "objects_with_failures": failure_objects,
        "by_object": object_rows,
        "by_physical_instance": instance_rows,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
