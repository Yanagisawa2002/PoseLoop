"""Reproducible 720p warmup/steady profiling schema and summarizer."""

from __future__ import annotations

import csv
import io
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import (
    PREP_PROTOCOL_ID,
    PROFILE_SAMPLE_SCHEMA,
    PROFILE_SUMMARY_SCHEMA,
    RESULT_SCHEMA,
    PrepError,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from .producer import STAGE_NAMES, load_protocol


PHASES = {"warmup", "steady"}


def _nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        raise PrepError("Cannot summarize an empty metric")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _validate_sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != PROFILE_SAMPLE_SCHEMA:
        raise PrepError("Profile sample schema is invalid")
    if value.get("protocol_id") != PREP_PROTOCOL_ID:
        raise PrepError("Profile sample protocol differs")
    if value.get("phase") not in PHASES:
        raise PrepError("Profile phase is invalid")
    resolution = value.get("resolution")
    if resolution != {"width": 1280, "height": 720}:
        raise PrepError("Profile sample is not 720p")
    if value.get("candidate_limit") != 252 or value.get("pose_hypothesis_count") != 252:
        raise PrepError("Profile sample pruned the c252 candidate set")
    stages = value.get("stage_timings_ms")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_NAMES):
        raise PrepError("Profile stage timing schema differs")
    for name in STAGE_NAMES:
        timing = stages[name]
        if isinstance(timing, bool) or not isinstance(timing, (int, float)) or timing < 0:
            raise PrepError(f"Invalid profile stage timing: {name}")
    total = value.get("total_latency_ms")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        raise PrepError("Profile total latency is invalid")
    if not math.isclose(float(total), sum(float(stages[name]) for name in STAGE_NAMES), rel_tol=0.0, abs_tol=1e-9):
        raise PrepError("Profile total does not equal the declared stage timings")
    allocated = value.get("peak_allocated_vram_bytes")
    reserved = value.get("peak_reserved_vram_bytes")
    if not isinstance(allocated, int) or not isinstance(reserved, int) or allocated < 0 or reserved < allocated:
        raise PrepError("Profile VRAM counters are invalid")
    if value.get("label_access_count") != 0 or value.get("official_scorer_run") is not False:
        raise PrepError("Profile sample crossed the label/scorer boundary")
    return dict(value)


def fixture_samples(
    *,
    results_path: Path,
    output_path: Path,
    warmup_repeats: int,
    steady_repeats: int,
) -> list[dict[str, Any]]:
    if warmup_repeats < 1 or steady_repeats < 1:
        raise PrepError("Warmup and steady repeat counts must be positive")
    results = read_jsonl(results_path.resolve())
    samples: list[dict[str, Any]] = []
    for result in results:
        if result.get("schema_version") != RESULT_SCHEMA or result.get("status") != "success":
            raise PrepError("Fixture profile requires successful PREP fixture results")
        if result.get("candidate_limit") != 252 or result.get("pose_hypothesis_count") != 252:
            raise PrepError("Fixture profile observed non-c252 output")
        base_stages = result.get("stage_timings_ms")
        if not isinstance(base_stages, dict) or set(base_stages) != set(STAGE_NAMES):
            raise PrepError("Fixture result stage timing schema differs")
        for phase, count in (("warmup", warmup_repeats), ("steady", steady_repeats)):
            for repeat_index in range(count):
                multiplier = (
                    1.20 + 0.01 * repeat_index
                    if phase == "warmup"
                    else 1.00 + 0.005 * repeat_index
                )
                stages = {
                    name: float(base_stages[name]) * multiplier for name in STAGE_NAMES
                }
                sample = {
                    "schema_version": PROFILE_SAMPLE_SCHEMA,
                    "protocol_id": PREP_PROTOCOL_ID,
                    "measurement_source": "synthetic-fixture-not-gpu",
                    "item_id": result["item_id"],
                    "mask_variant_id": result["mask_variant_id"],
                    "phase": phase,
                    "repeat_index": repeat_index,
                    "resolution": {"width": 1280, "height": 720},
                    "candidate_limit": 252,
                    "pose_hypothesis_count": 252,
                    "stage_timings_ms": stages,
                    "total_latency_ms": sum(stages.values()),
                    "peak_allocated_vram_bytes": int(result["cuda_peak_allocated_bytes"]),
                    "peak_reserved_vram_bytes": int(result["cuda_peak_reserved_bytes"]),
                    "label_access_count": 0,
                    "official_scorer_run": False,
                }
                samples.append(_validate_sample(sample))
    write_jsonl_atomic(output_path, samples)
    return samples


def _phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(row["total_latency_ms"]) for row in rows]
    wall_ms = sum(totals)
    stages = {
        name: {
            "p50_ms": statistics.median(
                float(row["stage_timings_ms"][name]) for row in rows
            ),
            "p95_ms": _nearest_rank(
                [float(row["stage_timings_ms"][name]) for row in rows], 0.95
            ),
        }
        for name in STAGE_NAMES
    }
    return {
        "sample_count": len(rows),
        "p50_ms": statistics.median(totals),
        "p95_ms": _nearest_rank(totals, 0.95),
        "throughput_items_per_second": 1000.0 * len(rows) / wall_ms,
        "wall_time_ms": wall_ms,
        "peak_allocated_vram_bytes": max(
            int(row["peak_allocated_vram_bytes"]) for row in rows
        ),
        "peak_reserved_vram_bytes": max(
            int(row["peak_reserved_vram_bytes"]) for row in rows
        ),
        "stage_timings": stages,
    }


def _csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    fieldnames = [
        "item_id",
        "mask_variant_id",
        "phase",
        "repeat_index",
        "width",
        "height",
        "candidate_limit",
        "pose_hypothesis_count",
        *STAGE_NAMES,
        "total_latency_ms",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "measurement_source",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "item_id": row["item_id"],
                "mask_variant_id": row["mask_variant_id"],
                "phase": row["phase"],
                "repeat_index": row["repeat_index"],
                "width": row["resolution"]["width"],
                "height": row["resolution"]["height"],
                "candidate_limit": row["candidate_limit"],
                "pose_hypothesis_count": row["pose_hypothesis_count"],
                **row["stage_timings_ms"],
                "total_latency_ms": row["total_latency_ms"],
                "peak_allocated_vram_bytes": row["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": row["peak_reserved_vram_bytes"],
                "measurement_source": row["measurement_source"],
            }
        )
    return output.getvalue()


def summarize_profile(
    *,
    protocol_path: Path,
    samples_path: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    rows = [_validate_sample(value) for value in read_jsonl(samples_path.resolve())]
    warmup = [row for row in rows if row["phase"] == "warmup"]
    steady = [row for row in rows if row["phase"] == "steady"]
    if not warmup or not steady:
        raise PrepError("Profile requires separate non-empty warmup and steady samples")
    item_ids = {row["item_id"] for row in rows}
    variants = {row["mask_variant_id"] for row in rows}
    for phase_rows in (warmup, steady):
        if {row["item_id"] for row in phase_rows} != item_ids:
            raise PrepError("Profile phase coverage differs")
    summary = {
        "schema_version": PROFILE_SUMMARY_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path.resolve()),
        "samples_sha256": sha256_file(samples_path.resolve()),
        "resolution": {"width": 1280, "height": 720},
        "candidate_limit": 252,
        "pose_hypothesis_count": 252,
        "item_count": len(item_ids),
        "mask_variant_ids": sorted(variants),
        "warmup": _phase_summary(warmup),
        "steady": _phase_summary(steady),
        "external_reference": {
            "target_ms": protocol["profiling"]["external_reference_target_ms"],
            "scope": "external-reference-target-only",
            "comparable": False,
            "speedup_or_claim_computed": False,
        },
        "measurement_sources": sorted({row["measurement_source"] for row in rows}),
        "accuracy_claim": "unavailable-no-label-access",
        "label_access_count": 0,
        "official_scorer_run": False,
    }
    write_json_atomic(output_json, summary)
    write_text_atomic(output_csv, _csv_text(rows))
    return summary
