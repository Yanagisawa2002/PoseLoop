"""Evaluator-only scorer and refiner diagnostics around a frozen GT perturbation grid."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import PROTOCOL_ID
from .core import ContractError, canonical_sha256, read_json, write_json, write_jsonl
from .se3 import (
    as_pose,
    interpolate_perturbation,
    load_cad_asset,
    load_pose_asset,
    perturb_pose,
    pose_errors,
)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise ContractError("PREP protocol must be a JSON object")
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("schema_version")
        != "poseloop.pose-accuracy-recovery.protocol.v1"
    ):
        raise ContractError("PREP protocol identity mismatch")
    if (
        protocol.get("state") != "frozen_pre_execution_template"
        or protocol.get("auto_deploy") is not False
    ):
        raise ContractError("PREP protocol is not frozen with AUTO_DEPLOY=false")
    boundaries = protocol.get("immutable_boundaries", {})
    required_false = (
        "server_connection_permitted",
        "download_permitted",
        "training_permitted",
        "inference_permitted",
        "sealed_split_access_permitted",
        "accuracy_claim_permitted",
    )
    if any(boundaries.get(name) is not False for name in required_false):
        raise ContractError("PREP immutable execution boundaries changed")
    diagnostics = protocol.get("diagnostics", {})
    if diagnostics.get("namespace_role") != "EVALUATOR_ONLY":
        raise ContractError("SE(3) diagnostics must remain evaluator-only")
    if diagnostics.get("scorer_order") != "descending_higher_is_better":
        raise ContractError("Scorer ordering must remain explicit and descending")
    if (
        diagnostics.get("refiner_trace_selection")
        != "largest_l1_perturbation_then_candidate_id"
    ):
        raise ContractError("Refiner trace selection changed")
    known = protocol.get("known_capability_constraints", {})
    if known.get("xyzibd_pinned_toolkit_supported_error_types") != [
        "ad",
        "add",
        "adi",
        "mssd",
        "mspd",
    ]:
        raise ContractError("Known XYZ-IBD toolkit capability changed")
    if (
        known.get("xyzibd_vsd") != "unavailable"
        or known.get("xyzibd_bop19_ar_requiring_vsd") != "unavailable"
        or known.get("reduced_error_set_may_be_relabelled_official_ar") is not False
    ):
        raise ContractError("Known VSD/AR unavailable boundary changed")
    return protocol


def _axis_values(
    values: Sequence[Any], axes: Sequence[str], signs: Sequence[int]
) -> list[tuple[str, float]]:
    output = [("x", 0.0)]
    for raw in values:
        magnitude = float(raw)
        if magnitude == 0.0:
            continue
        for axis in axes:
            for sign in signs:
                output.append((axis, float(sign) * magnitude))
    return output


def generate_perturbation_grid(
    gt_pose: np.ndarray, configuration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if configuration.get("composition") != "cartesian_axis_aligned_camera_frame":
        raise ContractError("Unsupported perturbation-grid composition")
    axes = configuration.get("axes")
    signs = configuration.get("signs")
    if axes != ["x", "y", "z"] or signs != [-1, 1]:
        raise ContractError("Perturbation axes/signs differ from the frozen grid")
    rotations = _axis_values(configuration.get("rotation_degrees", []), axes, signs)
    translations = _axis_values(configuration.get("translation_mm", []), axes, signs)
    candidates = []
    for rotation_axis, rotation_degrees in rotations:
        for translation_axis, translation_mm in translations:
            metadata = {
                "rotation_axis": rotation_axis,
                "rotation_degrees": rotation_degrees,
                "translation_axis": translation_axis,
                "translation_mm": translation_mm,
            }
            candidates.append(
                {
                    "candidate_id": f"perturb-{len(candidates):04d}",
                    **metadata,
                    "model_to_camera_pose_m": perturb_pose(
                        gt_pose, **metadata
                    ).tolist(),
                }
            )
    expected = configuration.get("expected_hypothesis_count")
    if expected != len(candidates):
        raise ContractError(
            f"Perturbation-grid count mismatch: expected {expected}, got {len(candidates)}"
        )
    return candidates


def prepare_diagnostics(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    namespace_role: str,
) -> list[dict[str, Any]]:
    if namespace_role != "EVALUATOR_ONLY":
        raise ContractError(
            "GT perturbations may only be prepared in EVALUATOR_ONLY namespace"
        )
    configuration = protocol["diagnostics"]["perturbation_grid"]
    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        gt_pose = load_pose_asset(data_root, sample["evaluator_only"]["gt_pose"])
        key = sample["key"]
        for candidate in generate_perturbation_grid(gt_pose, configuration):
            rows.append({**key, **candidate, "namespace_role": namespace_role})
    return rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _average_ranks(np.asarray(left, dtype=np.float64))
    right_rank = _average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def evaluate_scorer(
    candidates: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    gt_pose: np.ndarray,
    points: np.ndarray,
    symmetric: bool,
    top_k: int,
    minimum_rank_correlation: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(scores) != {str(candidate["candidate_id"]) for candidate in candidates}:
        raise ContractError("Scorer output must cover the exact frozen candidate grid")
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        errors = pose_errors(
            points, candidate["model_to_camera_pose_m"], gt_pose, symmetric=symmetric
        )
        score = float(scores[candidate_id])
        if not math.isfinite(score):
            raise ContractError(f"Non-finite scorer value for {candidate_id}")
        rows.append({"candidate_id": candidate_id, "score": score, **errors})
    ranked = sorted(rows, key=lambda row: (-row["score"], row["candidate_id"]))
    nearest = min(rows, key=lambda row: (row["add_s_mm"], row["candidate_id"]))
    nearest_rank = next(
        index
        for index, row in enumerate(ranked, start=1)
        if row["candidate_id"] == nearest["candidate_id"]
    )
    correlation = _spearman(
        [-row["add_s_mm"] for row in rows], [row["score"] for row in rows]
    )
    summary = {
        "nearest_candidate_id": nearest["candidate_id"],
        "nearest_candidate_rank": nearest_rank,
        "nearest_candidate_in_top_k": nearest_rank <= top_k,
        "top_k": top_k,
        "negative_add_s_score_rank_correlation": correlation,
        "minimum_rank_correlation": minimum_rank_correlation,
        "top1_add_s_mm": ranked[0]["add_s_mm"],
        "median_add_s_mm": float(np.median([row["add_s_mm"] for row in rows])),
    }
    summary["passed"] = bool(
        summary["nearest_candidate_in_top_k"]
        and correlation >= minimum_rank_correlation
        and summary["top1_add_s_mm"] <= summary["median_add_s_mm"]
    )
    return summary, ranked


def evaluate_refiner(
    traces: Sequence[Mapping[str, Any]],
    *,
    gt_pose: np.ndarray,
    points: np.ndarray,
    symmetric: bool,
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not traces:
        raise ContractError("Refiner diagnostics require at least one trace")
    metric_names = ("add_s_mm", "rotation_error_degrees", "translation_error_mm")
    output_rows: list[dict[str, Any]] = []
    trace_passes: list[bool] = []
    for trace in traces:
        candidate_id = str(trace.get("candidate_id"))
        poses = trace.get("poses_model_to_camera_m")
        if not isinstance(poses, list) or len(poses) < 2:
            raise ContractError(
                f"Refiner trace {candidate_id} requires at least two poses"
            )
        metric_rows = [
            pose_errors(points, pose, gt_pose, symmetric=symmetric) for pose in poses
        ]
        monotonic = {
            name: all(
                current[name] <= previous[name] + tolerance
                for previous, current in zip(metric_rows, metric_rows[1:])
            )
            for name in metric_names
        }
        improved = {
            name: metric_rows[-1][name] <= metric_rows[0][name] + tolerance
            for name in metric_names
        }
        passed = all(monotonic.values()) and all(improved.values())
        trace_passes.append(passed)
        for step, metrics in enumerate(metric_rows):
            output_rows.append({"candidate_id": candidate_id, "step": step, **metrics})
        output_rows.append(
            {
                "candidate_id": candidate_id,
                "step": "summary",
                **{f"monotonic_{name}": value for name, value in monotonic.items()},
                **{
                    f"nonincreasing_final_{name}": value
                    for name, value in improved.items()
                },
                "passed": passed,
            }
        )
    return {
        "trace_count": len(traces),
        "passed_trace_count": sum(trace_passes),
        "failed_trace_count": len(traces) - sum(trace_passes),
        "passed": all(trace_passes),
        "tolerance": tolerance,
    }, output_rows


def synthetic_outputs(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gt_pose: np.ndarray,
    points: np.ndarray,
    symmetric: bool,
    trace_count: int,
    refiner_scales: Sequence[float],
    selection_algorithm: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Generate deterministic fixture-only outputs; never use this as a model result."""
    scores = {}
    for candidate in candidates:
        errors = pose_errors(
            points, candidate["model_to_camera_pose_m"], gt_pose, symmetric=symmetric
        )
        scores[str(candidate["candidate_id"])] = -errors["add_s_mm"]
    selected = select_refiner_candidates(
        candidates, count=trace_count, algorithm=selection_algorithm
    )
    traces = []
    for candidate in selected:
        metadata = {
            "rotation_axis": candidate["rotation_axis"],
            "rotation_degrees": candidate["rotation_degrees"],
            "translation_axis": candidate["translation_axis"],
            "translation_mm": candidate["translation_mm"],
        }
        traces.append(
            {
                "candidate_id": candidate["candidate_id"],
                "poses_model_to_camera_m": [
                    interpolate_perturbation(gt_pose, metadata, float(scale)).tolist()
                    for scale in refiner_scales
                ],
            }
        )
    return scores, traces


def select_refiner_candidates(
    candidates: Sequence[Mapping[str, Any]], *, count: int, algorithm: str
) -> list[Mapping[str, Any]]:
    if algorithm != "largest_l1_perturbation_then_candidate_id":
        raise ContractError("Unsupported refiner trace selection algorithm")
    non_identity = [
        candidate
        for candidate in candidates
        if candidate["rotation_degrees"] != 0.0 or candidate["translation_mm"] != 0.0
    ]
    selected = sorted(
        non_identity,
        key=lambda candidate: (
            -(abs(candidate["rotation_degrees"]) + abs(candidate["translation_mm"])),
            candidate["candidate_id"],
        ),
    )[:count]
    if len(selected) != count:
        raise ContractError("Refiner trace selection exceeds non-identity grid")
    return selected


def run_fixture_diagnostics(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    configuration = protocol["diagnostics"]
    grid = configuration["perturbation_grid"]
    all_summaries = []
    all_scorer_rows = []
    all_refiner_rows = []
    grid_rows = prepare_diagnostics(
        protocol, manifest, data_root=data_root, namespace_role="EVALUATOR_ONLY"
    )
    for sample in manifest["samples"]:
        key = sample["key"]
        candidates = [
            row for row in grid_rows if all(row[name] == key[name] for name in key)
        ]
        gt_pose = load_pose_asset(data_root, sample["evaluator_only"]["gt_pose"])
        points, _ = load_cad_asset(data_root, sample["producer_inputs"]["cad"])
        scores, traces = synthetic_outputs(
            candidates,
            gt_pose=gt_pose,
            points=points,
            symmetric=sample["symmetric_object"],
            trace_count=int(configuration["refiner_trace_count"]),
            refiner_scales=configuration["fixture_refiner_scales"],
            selection_algorithm=configuration["refiner_trace_selection"],
        )
        scorer_summary, scorer_rows = evaluate_scorer(
            candidates,
            scores,
            gt_pose=gt_pose,
            points=points,
            symmetric=sample["symmetric_object"],
            top_k=int(configuration["scorer_top_k"]),
            minimum_rank_correlation=float(configuration["minimum_rank_correlation"]),
        )
        refiner_summary, refiner_rows = evaluate_refiner(
            traces,
            gt_pose=gt_pose,
            points=points,
            symmetric=sample["symmetric_object"],
            tolerance=float(configuration["monotonic_tolerance"]),
        )
        all_summaries.append(
            {**key, "scorer": scorer_summary, "refiner": refiner_summary}
        )
        all_scorer_rows.extend([{**key, **row} for row in scorer_rows])
        all_refiner_rows.extend([{**key, **row} for row in refiner_rows])
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "perturbation-grid.jsonl", grid_rows)
    write_jsonl(output_root / "scorer-diagnostic.jsonl", all_scorer_rows)
    write_jsonl(output_root / "refiner-diagnostic.jsonl", all_refiner_rows)
    summary = {
        "schema_version": "poseloop.pose-accuracy-recovery.diagnostic-summary.v1",
        "execution_mode": "CPU_SYNTHETIC_FIXTURE_DRY_RUN",
        "namespace_role": "EVALUATOR_ONLY",
        "protocol_id": protocol["protocol_id"],
        "random_seed": protocol["random_seed"],
        "sample_count": len(manifest["samples"]),
        "hypothesis_count_per_sample": grid["expected_hypothesis_count"],
        "scorer_passed": all(row["scorer"]["passed"] for row in all_summaries),
        "refiner_passed": all(row["refiner"]["passed"] for row in all_summaries),
        "accuracy_claim_permitted": False,
        "dry_run_is_result": False,
        "samples": all_summaries,
    }
    summary["lock_sha256"] = canonical_sha256(summary)
    write_json(output_root / "diagnostic-summary.json", summary)
    _write_summary_csv(output_root / "diagnostic-summary.csv", all_summaries)
    return summary


def evaluate_prepared_diagnostics(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    grid_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    refiner_traces: Sequence[Mapping[str, Any]],
    namespace_role: str,
    output_root: Path,
) -> dict[str, Any]:
    """Evaluate real scorer/refiner outputs against the exact recomputed grid."""
    if namespace_role != "EVALUATOR_ONLY":
        raise ContractError(
            "Scorer/refiner diagnostics may only run in EVALUATOR_ONLY namespace"
        )
    expected_grid = prepare_diagnostics(
        protocol, manifest, data_root=data_root, namespace_role=namespace_role
    )
    if canonical_sha256(list(grid_rows)) != canonical_sha256(expected_grid):
        raise ContractError(
            "Prepared perturbation grid differs from the frozen protocol and manifest"
        )
    sample_keys = {
        (
            sample["key"]["scene_id"],
            sample["key"]["image_id"],
            sample["key"]["object_id"],
        ): sample
        for sample in manifest["samples"]
    }
    expected_scorer_keys = {
        (row["scene_id"], row["image_id"], row["object_id"], row["candidate_id"])
        for row in expected_grid
    }
    scorer_by_key: dict[tuple[int, int, int, str], Mapping[str, Any]] = {}
    for row in scorer_rows:
        key = _diagnostic_row_key(row)
        if row.get("namespace_role") != namespace_role:
            raise ContractError(f"Scorer row {key} is outside EVALUATOR_ONLY namespace")
        if key in scorer_by_key:
            raise ContractError(f"Duplicate scorer diagnostic key: {key}")
        score = row.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
        ):
            raise ContractError(f"Scorer row {key} has a non-finite score")
        scorer_by_key[key] = row
    if set(scorer_by_key) != expected_scorer_keys:
        raise ContractError(
            "Scorer output does not exactly cover the frozen perturbation grid"
        )

    refiner_by_sample: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {
        key: [] for key in sample_keys
    }
    seen_refiner: set[tuple[int, int, int, str]] = set()
    candidate_ids = {key[3] for key in expected_scorer_keys}
    for row in refiner_traces:
        full_key = _diagnostic_row_key(row)
        sample_key = full_key[:3]
        if row.get("namespace_role") != namespace_role:
            raise ContractError(
                f"Refiner row {full_key} is outside EVALUATOR_ONLY namespace"
            )
        if sample_key not in sample_keys or full_key[3] not in candidate_ids:
            raise ContractError(f"Refiner row is outside the frozen grid: {full_key}")
        if full_key in seen_refiner:
            raise ContractError(f"Duplicate refiner diagnostic key: {full_key}")
        seen_refiner.add(full_key)
        refiner_by_sample[sample_key].append(row)
    expected_trace_count = int(protocol["diagnostics"]["refiner_trace_count"])
    if any(len(rows) != expected_trace_count for rows in refiner_by_sample.values()):
        raise ContractError("Refiner trace count differs from the frozen protocol")

    summaries = []
    ranked_rows = []
    refiner_metric_rows = []
    for sample_key, sample in sorted(sample_keys.items()):
        candidates = [
            row
            for row in expected_grid
            if (row["scene_id"], row["image_id"], row["object_id"]) == sample_key
        ]
        expected_refiner_ids = {
            str(row["candidate_id"])
            for row in select_refiner_candidates(
                candidates,
                count=expected_trace_count,
                algorithm=protocol["diagnostics"]["refiner_trace_selection"],
            )
        }
        actual_refiner_ids = {
            str(row["candidate_id"]) for row in refiner_by_sample[sample_key]
        }
        if actual_refiner_ids != expected_refiner_ids:
            raise ContractError(
                "Refiner candidates differ from the frozen trace selection"
            )
        candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
        for trace in refiner_by_sample[sample_key]:
            poses = trace.get("poses_model_to_camera_m")
            if not isinstance(poses, list) or not poses:
                raise ContractError("Refiner trace requires an initial pose")
            candidate_pose = candidate_by_id[str(trace["candidate_id"])][
                "model_to_camera_pose_m"
            ]
            if not np.allclose(
                as_pose(poses[0], label="refiner_initial_pose"),
                as_pose(candidate_pose, label="grid_candidate_pose"),
                atol=1e-9,
            ):
                raise ContractError(
                    "Refiner trace does not start from its frozen candidate"
                )
        scores = {
            str(row["candidate_id"]): float(
                scorer_by_key[(*sample_key, str(row["candidate_id"]))]["score"]
            )
            for row in candidates
        }
        gt_pose = load_pose_asset(data_root, sample["evaluator_only"]["gt_pose"])
        points, _ = load_cad_asset(data_root, sample["producer_inputs"]["cad"])
        scorer_summary, ranked = evaluate_scorer(
            candidates,
            scores,
            gt_pose=gt_pose,
            points=points,
            symmetric=sample["symmetric_object"],
            top_k=int(protocol["diagnostics"]["scorer_top_k"]),
            minimum_rank_correlation=float(
                protocol["diagnostics"]["minimum_rank_correlation"]
            ),
        )
        refiner_summary, trace_metrics = evaluate_refiner(
            refiner_by_sample[sample_key],
            gt_pose=gt_pose,
            points=points,
            symmetric=sample["symmetric_object"],
            tolerance=float(protocol["diagnostics"]["monotonic_tolerance"]),
        )
        key_fields = {
            "scene_id": sample_key[0],
            "image_id": sample_key[1],
            "object_id": sample_key[2],
        }
        summaries.append(
            {**key_fields, "scorer": scorer_summary, "refiner": refiner_summary}
        )
        ranked_rows.extend([{**key_fields, **row} for row in ranked])
        refiner_metric_rows.extend([{**key_fields, **row} for row in trace_metrics])
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "scorer-diagnostic.jsonl", ranked_rows)
    write_jsonl(output_root / "refiner-diagnostic.jsonl", refiner_metric_rows)
    summary = {
        "schema_version": "poseloop.pose-accuracy-recovery.diagnostic-summary.v1",
        "execution_mode": "DEVELOPMENT_EVALUATOR_ONLY",
        "namespace_role": namespace_role,
        "protocol_id": protocol["protocol_id"],
        "random_seed": protocol["random_seed"],
        "sample_count": len(sample_keys),
        "hypothesis_count_per_sample": protocol["diagnostics"]["perturbation_grid"][
            "expected_hypothesis_count"
        ],
        "scorer_passed": all(row["scorer"]["passed"] for row in summaries),
        "refiner_passed": all(row["refiner"]["passed"] for row in summaries),
        "accuracy_claim_permitted": False,
        "samples": summaries,
    }
    summary["lock_sha256"] = canonical_sha256(summary)
    write_json(output_root / "diagnostic-summary.json", summary)
    _write_summary_csv(output_root / "diagnostic-summary.csv", summaries)
    return summary


def _diagnostic_row_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    values = []
    for name in ("scene_id", "image_id", "object_id"):
        value = row.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"Diagnostic row {name} must be a non-negative integer")
        values.append(value)
    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ContractError("Diagnostic row candidate_id must be a non-empty string")
    return values[0], values[1], values[2], candidate_id


def _write_summary_csv(path: Path, summaries: Iterable[Mapping[str, Any]]) -> None:
    fields = [
        "scene_id",
        "image_id",
        "object_id",
        "scorer_passed",
        "nearest_candidate_rank",
        "rank_correlation",
        "refiner_passed",
        "refiner_trace_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "scene_id": row["scene_id"],
                    "image_id": row["image_id"],
                    "object_id": row["object_id"],
                    "scorer_passed": row["scorer"]["passed"],
                    "nearest_candidate_rank": row["scorer"]["nearest_candidate_rank"],
                    "rank_correlation": row["scorer"][
                        "negative_add_s_score_rank_correlation"
                    ],
                    "refiner_passed": row["refiner"]["passed"],
                    "refiner_trace_count": row["refiner"]["trace_count"],
                }
            )


def save_prepared_grid(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    write_jsonl(path, rows)
    receipt = {
        "schema_version": "poseloop.pose-accuracy-recovery.perturbation-receipt.v1",
        "namespace_role": "EVALUATOR_ONLY",
        "row_count": len(rows),
        "content_sha256": canonical_sha256(list(rows)),
        "accuracy_claim_permitted": False,
    }
    write_json(path.with_suffix(".receipt.json"), receipt)
    return receipt
