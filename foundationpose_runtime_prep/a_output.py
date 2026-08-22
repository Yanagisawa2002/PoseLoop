"""Export and strictly self-validate GPU-C rows for GPU-A accuracy recovery."""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .common import (
    A_PRODUCER_OUTPUT_SCHEMA,
    PREP_PROTOCOL_ID,
    RESULT_SCHEMA,
    RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
    PrepError,
    canonical_bytes,
    canonical_sha256,
    is_sha256,
    read_jsonl,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from .manifest import V2_VARIANT_IDS, validate_manifest
from .producer import STAGE_NAMES
from .results import prediction_projection


A_OUTPUT_VALIDATION_SCHEMA = (
    "poseloop.pose-accuracy-recovery.producer-output-validation.v1"
)
A_OUTPUT_KEYS = {
    "schema_version",
    "item_id",
    "sample_key",
    "mask_variant_id",
    "manifest_lock_sha256",
    "input_sha256",
    "implementation_commit",
    "implementation_sha256",
    "model_sha256",
    "checkpoint_sha256",
    "initial_model_to_camera_pose_m",
    "final_model_to_camera_pose_m",
    "top_k",
    "refiner_trace",
    "status",
    "attempt",
    "latency_ms",
    "failure",
    "oom",
    "failure_reason",
    "access_counters",
    "visualization_inventory",
}
INPUT_SHA_KEYS = {"rgb", "depth", "mask", "camera", "cad"}
ACCESS_COUNTER_KEYS = {
    "label_access_count_on_gpu_c",
    "gt_path_open_count_on_gpu_c",
    "evaluator_path_open_count_on_gpu_c",
    "scorer_path_open_count_on_gpu_c",
}
VISUALIZATION_ROLES = {
    "rgb",
    "input_mask",
    "initial_pose_overlay",
    "top_k_overlay",
    "final_pose_overlay",
}
FORBIDDEN_VISUAL_PATH_TOKENS = {
    "gt",
    "groundtruth",
    "ground_truth",
    "evaluator",
    "evaluation",
    "sealed",
    "official_score",
    "official_scorer",
}
CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
TOP_K_COUNT = 5
REFINER_TRACE_COUNT = 6


def _finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrepError(f"A-output {field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise PrepError(f"A-output {field} is invalid")
    return number


def _legal_se3(value: Any, *, field: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise PrepError(f"A-output {field} must be 4x4")
    matrix = [[_finite(cell, field=field) for cell in row] for row in value]
    if any(
        abs(matrix[3][index] - expected) > 1e-8
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        raise PrepError(f"A-output {field} homogeneous row differs")
    rotation = [row[:3] for row in matrix[:3]]
    for row_index in range(3):
        for other_index in range(3):
            dot = sum(
                rotation[row_index][axis] * rotation[other_index][axis]
                for axis in range(3)
            )
            expected = 1.0 if row_index == other_index else 0.0
            if abs(dot - expected) > 1e-5:
                raise PrepError(f"A-output {field} rotation is not orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise PrepError(f"A-output {field} rotation determinant differs")
    return matrix


def _safe_visual_path(value: Any, *, item_id: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PrepError("A-output visualization path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 3
        or path.parts[0] != "visualizations"
        or path.parts[1] != item_id
    ):
        raise PrepError("A-output visualization path is not item-isolated")
    tokens = {
        token
        for part in path.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token
    }
    if tokens & FORBIDDEN_VISUAL_PATH_TOKENS:
        raise PrepError("A-output visualization path exposes GT/evaluator data")
    return path.as_posix()


def _validate_top_k(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != TOP_K_COUNT:
        raise PrepError("A-output top_k must contain exactly five candidates")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_score = math.inf
    for index, entry in enumerate(value, 1):
        if not isinstance(entry, dict) or set(entry) != {
            "rank",
            "candidate_id",
            "score",
            "model_to_camera_pose_m",
        }:
            raise PrepError("A-output top_k entry fields differ")
        if entry.get("rank") != index:
            raise PrepError("A-output top_k ranks are not stable and consecutive")
        candidate_id = entry.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not CANDIDATE_ID.fullmatch(candidate_id)
            or candidate_id in seen_ids
        ):
            raise PrepError("A-output top_k candidate ID is invalid or duplicate")
        seen_ids.add(candidate_id)
        score = _finite(entry.get("score"), field="top_k score")
        if score > previous_score:
            raise PrepError("A-output top_k scores are not descending")
        previous_score = score
        _legal_se3(
            entry.get("model_to_camera_pose_m"),
            field=f"top_k[{index - 1}] pose",
        )
        normalized.append(dict(entry))
    return normalized


def _validate_trace(
    value: Any,
    *,
    initial_pose: Any,
    final_pose: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != REFINER_TRACE_COUNT:
        raise PrepError("A-output refiner trace must contain all six states")
    normalized: list[dict[str, Any]] = []
    poses: list[list[list[float]]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != {
            "iteration",
            "model_to_camera_pose_m",
            "objective",
        }:
            raise PrepError("A-output refiner trace entry fields differ")
        if entry.get("iteration") != index:
            raise PrepError("A-output refiner trace iterations are not contiguous")
        poses.append(
            _legal_se3(
                entry.get("model_to_camera_pose_m"),
                field=f"refiner_trace[{index}] pose",
            )
        )
        _finite(entry.get("objective"), field=f"refiner_trace[{index}] objective")
        normalized.append(dict(entry))
    if canonical_bytes(poses[0]) != canonical_bytes(initial_pose):
        raise PrepError("A-output refiner trace does not start at the initial pose")
    if canonical_bytes(poses[-1]) != canonical_bytes(final_pose):
        raise PrepError("A-output refiner trace does not end at the final pose")
    return normalized


def _validate_visualizations(
    value: Any,
    *,
    item_id: str,
    result_root: Path,
    verify_assets: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(VISUALIZATION_ROLES):
        raise PrepError("A-output visualization inventory is incomplete")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    root = result_root.resolve()
    for member in value:
        if not isinstance(member, dict) or set(member) != {
            "role",
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise PrepError("A-output visualization member fields differ")
        role = member.get("role")
        if role not in VISUALIZATION_ROLES or role in seen:
            raise PrepError("A-output visualization role is invalid or duplicate")
        seen.add(str(role))
        relative = _safe_visual_path(member.get("relative_path"), item_id=item_id)
        if not is_sha256(member.get("sha256")):
            raise PrepError("A-output visualization member SHA-256 is invalid")
        byte_count = member.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise PrepError("A-output visualization member byte count is invalid")
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PrepError("A-output visualization escapes its result root") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise PrepError(f"A-output visualization is missing: {relative}")
        if verify_assets:
            if candidate.stat().st_size != byte_count:
                raise PrepError(f"A-output visualization byte drift: {relative}")
            if sha256_file(candidate) != member["sha256"]:
                raise PrepError(f"A-output visualization hash drift: {relative}")
        normalized.append(dict(member))
    if seen != VISUALIZATION_ROLES:
        raise PrepError("A-output visualization role coverage differs")
    return normalized


def _synthetic_visualizations(
    row: Mapping[str, Any],
    *,
    result_root: Path,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for role in sorted(VISUALIZATION_ROLES):
        relative = f"visualizations/{row['item_id']}/{role}.synthetic.json"
        path = result_root.resolve() / Path(*PurePosixPath(relative).parts)
        payload = {
            "schema_version": "poseloop.r4c.prep.synthetic-visualization.v1",
            "item_id": row["item_id"],
            "role": role,
            "synthetic_backend": True,
            "prediction_fingerprint": row["prediction_fingerprint"],
        }
        write_json_atomic(path, payload)
        members.append(
            {
                "role": role,
                "relative_path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return members


def _validate_internal_success(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != RESULT_SCHEMA or row.get("status") != "success":
        raise PrepError("A-output export requires successful PREP producer rows")
    initial = _legal_se3(
        row.get("initial_model_to_camera_pose_m"), field="internal initial pose"
    )
    final = _legal_se3(
        row.get("final_model_to_camera_pose_m"), field="internal final pose"
    )
    top_k = _validate_top_k(row.get("top_k"))
    if float(top_k[0]["score"]) != float(row.get("foundationpose_top_score")):
        raise PrepError("Internal top_k rank-one score differs from producer score")
    _validate_trace(
        row.get("refiner_trace_model_to_camera_m"),
        initial_pose=initial,
        final_pose=final,
    )
    if canonical_bytes(final) != canonical_bytes(
        row.get("predicted_model_to_camera_pose_m")
    ):
        raise PrepError("Internal final pose differs from the selected prediction")
    for field in (
        "label_access_count",
        "gt_path_open_count",
        "evaluator_path_open_count",
        "scorer_path_open_count",
        "official_scorer_run_count",
    ):
        if row.get(field) != 0:
            raise PrepError(f"Internal producer access counter is nonzero: {field}")
    if row.get("official_scorer_run") is not False:
        raise PrepError("Internal producer declares official scorer execution")
    if row.get("producer_runtime_isolated") is not True:
        raise PrepError("Internal producer is not runtime-isolated")
    if row.get("synthetic_backend") is not True:
        raise PrepError("PREP export accepts only explicit synthetic fixture rows")
    if not all(
        isinstance(row.get(field), str) and row[field]
        for field in (
            "implementation_commit",
            "implementation_sha256",
            "model_sha256",
            "checkpoint_sha256",
        )
    ):
        raise PrepError("Internal producer runtime identities are incomplete")


def validate_internal_live_success(
    row: Mapping[str, Any], *, visualization_root: Path
) -> None:
    """Validate a real backend row without weakening the synthetic exporter."""

    if row.get("schema_version") != RESULT_SCHEMA or row.get("status") != "success":
        raise PrepError("Live A-output export requires a successful producer row")
    # Reject the fixture path before interpreting any candidate evidence.  A
    # malformed synthetic row must never be able to reach live export by
    # looking structurally similar to a backend result.
    if row.get("synthetic_backend") is not False:
        raise PrepError("Live export rejects synthetic backend rows")
    if row.get("backend_id") != "official-foundationpose-python-live-v1":
        raise PrepError("Live internal backend identity differs")
    projection = prediction_projection(row)
    initial = _legal_se3(
        row.get("initial_model_to_camera_pose_m"), field="live internal initial pose"
    )
    final = _legal_se3(
        row.get("final_model_to_camera_pose_m"), field="live internal final pose"
    )
    top_k = _validate_top_k(row.get("top_k"))
    if float(top_k[0]["score"]) != float(row.get("foundationpose_top_score")):
        raise PrepError("Live internal top_k rank-one score differs")
    if canonical_bytes(top_k[0]["model_to_camera_pose_m"]) != canonical_bytes(final):
        raise PrepError("Live internal top_k rank-one pose differs from final pose")
    selected = projection["selected_candidate_index"]
    if top_k[0]["candidate_id"] != f"foundationpose-c{selected:03d}":
        raise PrepError("Live internal top_k rank-one candidate ID differs")
    expected_margin = float(top_k[0]["score"]) - float(top_k[1]["score"])
    if expected_margin != float(row.get("foundationpose_top_score_margin")):
        raise PrepError("Live internal top score margin differs from top_k")
    trace = _validate_trace(
        row.get("refiner_trace_model_to_camera_m"),
        initial_pose=initial,
        final_pose=final,
    )
    if float(trace[-1]["objective"]) != float(top_k[0]["score"]):
        raise PrepError("Live internal final trace objective differs from top score")
    if canonical_bytes(final) != canonical_bytes(
        row.get("predicted_model_to_camera_pose_m")
    ):
        raise PrepError("Live internal final pose differs from selected prediction")
    for field in (
        "label_access_count",
        "gt_path_open_count",
        "evaluator_path_open_count",
        "scorer_path_open_count",
        "official_scorer_run_count",
    ):
        if row.get(field) != 0:
            raise PrepError(f"Live internal access counter is nonzero: {field}")
    if row.get("official_scorer_run") is not False:
        raise PrepError("Live internal producer declares official scorer execution")
    if row.get("producer_runtime_isolated") is not True:
        raise PrepError("Live internal producer is not runtime-isolated")
    evidence = row.get("backend_evidence")
    if evidence != {
        "synthetic": False,
        "primary_register_iterations": 5,
        "primary_candidate_count": 252,
        "foundationpose_candidate_scorer_calls_primary": 1,
        "trace_capture_mode": "exact-refiner-replay-plus-full-c252-scorer",
        "trace_refiner_replay_calls": 5,
        "trace_objective_scorer_calls": 6,
        "trace_final_pose_population_bitwise_equal": True,
        "trace_final_score_population_bitwise_equal": True,
        "official_evaluator_scorer_run": False,
    }:
        raise PrepError("Live internal backend evidence is incomplete")
    if row.get("candidate_limit") != 252 or row.get("pose_hypothesis_count") != 252:
        raise PrepError("Live internal producer did not retain c252")
    if row.get("resource_batches") != {
        "warp": 32,
        "score_data": 8,
        "score_feature": 32,
        "refine": 32,
    }:
        raise PrepError("Live internal producer resource batches differ")
    commit = row.get("implementation_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PrepError("Live internal implementation commit is invalid")
    for field in ("implementation_sha256", "model_sha256", "checkpoint_sha256"):
        if not is_sha256(row.get(field)):
            raise PrepError("Live internal runtime identities are incomplete")
    stages = row.get("stage_timings_ms")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_NAMES):
        raise PrepError("Live internal stage timing fields differ")
    for name in STAGE_NAMES:
        _finite(stages[name], field=f"live stage {name}", minimum=0.0)
    total = _finite(row.get("total_latency_ms"), field="live latency", minimum=0.0)
    if not math.isclose(
        total,
        sum(float(stages[name]) for name in STAGE_NAMES),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PrepError("Live internal latency does not equal its stage timings")
    wall = _finite(row.get("wall_time_ms"), field="wall_time_ms", minimum=0.0)
    _finite(row.get("evidence_capture_ms"), field="evidence_capture_ms", minimum=0.0)
    _finite(row.get("visualization_ms"), field="visualization_ms", minimum=0.0)
    if wall < total:
        raise PrepError("Live internal wall time is smaller than primary latency")
    allocated = row.get("cuda_peak_allocated_bytes")
    reserved = row.get("cuda_peak_reserved_bytes")
    evidence_allocated = row.get("evidence_peak_allocated_bytes")
    evidence_reserved = row.get("evidence_peak_reserved_bytes")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (allocated, reserved, evidence_allocated, evidence_reserved)
        )
        or reserved < allocated
        or evidence_reserved < evidence_allocated
    ):
        raise PrepError("Live internal VRAM counters are invalid")
    attempt = row.get("attempt")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or row.get("oom") is not False
        or row.get("failure_reason") is not None
    ):
        raise PrepError("Live internal success/failure fields differ")
    inventory = _validate_visualizations(
        row.get("visualization_inventory"),
        item_id=str(row.get("item_id", "")),
        result_root=visualization_root,
        verify_assets=True,
    )
    if any(
        PurePosixPath(member["relative_path"]).suffix.lower() != ".png"
        for member in inventory
    ):
        raise PrepError("Live internal visualization is not a PNG")


def export_a_results(
    *,
    protocol_path: Path,
    manifest_path: Path,
    producer_results_path: Path,
    output_path: Path,
    validation_output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
    )
    if manifest.get("schema_version") != RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        raise PrepError(
            "Formal A-output export requires the V2 runtime-isolated manifest"
        )
    internal_rows = read_jsonl(producer_results_path.resolve())
    by_id: dict[str, dict[str, Any]] = {}
    for row in internal_rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in by_id:
            raise PrepError(
                "PREP producer results contain a blank or duplicate item ID"
            )
        _validate_internal_success(row)
        by_id[item_id] = row
    expected_ids = [item["item_id"] for item in items]
    if set(by_id) != set(expected_ids):
        raise PrepError("PREP producer result coverage differs from the V2 manifest")
    runtime = manifest["producer_runtime_lock"]
    combined_checkpoint_sha = canonical_sha256(runtime["checkpoint_sha256"])
    output_root = output_path.resolve().parent
    output_rows: list[dict[str, Any]] = []
    for item in items:
        internal = by_id[item["item_id"]]
        if internal["sample_key"] != item["sample_key"]:
            raise PrepError("Internal producer sample key differs from manifest")
        if internal["mask_variant_id"] != item["mask_variant_id"]:
            raise PrepError("Internal producer mask variant differs from manifest")
        if internal["input_mask_sha256"] != item["inputs"]["mask"]["sha256"]:
            raise PrepError("Internal producer mask hash differs from manifest")
        expected_runtime = {
            "implementation_commit": runtime["implementation_commit"],
            "implementation_sha256": runtime["implementation_sha256"],
            "model_sha256": runtime["model_sha256"],
            "checkpoint_sha256": combined_checkpoint_sha,
        }
        if any(
            internal[field] != expected for field, expected in expected_runtime.items()
        ):
            raise PrepError("Internal producer runtime identity differs from manifest")
        output_rows.append(
            {
                "schema_version": A_PRODUCER_OUTPUT_SCHEMA,
                "item_id": item["item_id"],
                "sample_key": dict(item["sample_key"]),
                "mask_variant_id": item["mask_variant_id"],
                "manifest_lock_sha256": manifest["manifest_lock_sha256"],
                "input_sha256": {
                    slot: item["inputs"][slot]["sha256"]
                    for slot in ("rgb", "depth", "mask", "camera", "cad")
                },
                **expected_runtime,
                "initial_model_to_camera_pose_m": internal[
                    "initial_model_to_camera_pose_m"
                ],
                "final_model_to_camera_pose_m": internal[
                    "final_model_to_camera_pose_m"
                ],
                "top_k": internal["top_k"],
                "refiner_trace": internal["refiner_trace_model_to_camera_m"],
                "status": "success",
                "attempt": internal["attempt"],
                "latency_ms": internal["total_latency_ms"],
                "failure": False,
                "oom": internal["oom"],
                "failure_reason": internal["failure_reason"],
                "access_counters": {
                    "label_access_count_on_gpu_c": internal["label_access_count"],
                    "gt_path_open_count_on_gpu_c": internal["gt_path_open_count"],
                    "evaluator_path_open_count_on_gpu_c": internal[
                        "evaluator_path_open_count"
                    ],
                    "scorer_path_open_count_on_gpu_c": internal[
                        "scorer_path_open_count"
                    ],
                },
                "visualization_inventory": _synthetic_visualizations(
                    internal,
                    result_root=output_root,
                ),
            }
        )
    write_jsonl_atomic(output_path, output_rows)
    report = validate_a_results(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        results_path=output_path,
    )
    report["producer_results_sha256"] = sha256_file(producer_results_path.resolve())
    write_json_atomic(validation_output_path, report)
    return output_rows, report


def export_a_live_results(
    *,
    protocol_path: Path,
    manifest_path: Path,
    producer_results_path: Path,
    output_path: Path,
    validation_output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export only verified non-synthetic rows through a separate live path."""

    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
    )
    if manifest.get("schema_version") != RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        raise PrepError(
            "Live A-output export requires the V2 runtime-isolated manifest"
        )
    producer_root = producer_results_path.resolve().parent
    output_root = output_path.resolve().parent
    internal_rows = read_jsonl(producer_results_path.resolve())
    by_id: dict[str, dict[str, Any]] = {}
    for row in internal_rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in by_id:
            raise PrepError(
                "Live producer results contain a blank or duplicate item ID"
            )
        validate_internal_live_success(row, visualization_root=producer_root)
        by_id[item_id] = row
    expected_ids = [item["item_id"] for item in items]
    if set(by_id) != set(expected_ids):
        raise PrepError("Live producer result coverage differs from the V2 manifest")
    runtime = manifest["producer_runtime_lock"]
    expected_runtime = {
        "implementation_commit": runtime["implementation_commit"],
        "implementation_sha256": runtime["implementation_sha256"],
        "model_sha256": runtime["model_sha256"],
        "checkpoint_sha256": canonical_sha256(runtime["checkpoint_sha256"]),
    }
    output_rows: list[dict[str, Any]] = []
    for item in items:
        internal = by_id[item["item_id"]]
        if internal.get("sample_key") != item["sample_key"]:
            raise PrepError("Live producer sample key differs from manifest")
        if internal.get("mask_variant_id") != item["mask_variant_id"]:
            raise PrepError("Live producer mask variant differs from manifest")
        if internal.get("input_mask_sha256") != item["inputs"]["mask"]["sha256"]:
            raise PrepError("Live producer mask SHA differs from manifest")
        for field, expected in expected_runtime.items():
            if internal.get(field) != expected:
                raise PrepError(f"Live producer runtime identity differs: {field}")
        inventory = _validate_visualizations(
            internal.get("visualization_inventory"),
            item_id=item["item_id"],
            result_root=producer_root,
            verify_assets=True,
        )
        copied_inventory: list[dict[str, Any]] = []
        for member in inventory:
            relative = Path(*PurePosixPath(member["relative_path"]).parts)
            source = producer_root / relative
            destination = output_root / relative
            if source.resolve() != destination.resolve():
                write_bytes_atomic(destination, source.read_bytes())
            if (
                destination.stat().st_size != member["bytes"]
                or sha256_file(destination) != member["sha256"]
            ):
                raise PrepError("Live visualization changed during A-output export")
            copied_inventory.append(dict(member))
        output_rows.append(
            {
                "schema_version": A_PRODUCER_OUTPUT_SCHEMA,
                "item_id": item["item_id"],
                "sample_key": dict(item["sample_key"]),
                "mask_variant_id": item["mask_variant_id"],
                "manifest_lock_sha256": manifest["manifest_lock_sha256"],
                "input_sha256": {
                    slot: item["inputs"][slot]["sha256"]
                    for slot in ("rgb", "depth", "mask", "camera", "cad")
                },
                **expected_runtime,
                "initial_model_to_camera_pose_m": internal[
                    "initial_model_to_camera_pose_m"
                ],
                "final_model_to_camera_pose_m": internal[
                    "final_model_to_camera_pose_m"
                ],
                "top_k": internal["top_k"],
                "refiner_trace": internal["refiner_trace_model_to_camera_m"],
                "status": "success",
                "attempt": internal["attempt"],
                "latency_ms": internal["total_latency_ms"],
                "failure": False,
                "oom": False,
                "failure_reason": None,
                "access_counters": {
                    "label_access_count_on_gpu_c": internal["label_access_count"],
                    "gt_path_open_count_on_gpu_c": internal["gt_path_open_count"],
                    "evaluator_path_open_count_on_gpu_c": internal[
                        "evaluator_path_open_count"
                    ],
                    "scorer_path_open_count_on_gpu_c": internal[
                        "scorer_path_open_count"
                    ],
                },
                "visualization_inventory": copied_inventory,
            }
        )
    write_jsonl_atomic(output_path, output_rows)
    report = validate_a_results(
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        results_path=output_path,
    )
    if report.get("synthetic_fixture_row_count") != 0:
        raise PrepError("Live A-output unexpectedly contains synthetic candidate IDs")
    report["producer_results_sha256"] = sha256_file(producer_results_path.resolve())
    report["live_backend"] = True
    write_json_atomic(validation_output_path, report)
    return output_rows, report


def validate_a_results(
    *,
    protocol_path: Path,
    manifest_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    manifest, items = validate_manifest(
        manifest_path,
        protocol_path=protocol_path,
    )
    if manifest.get("schema_version") != RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        raise PrepError("A-output validation requires the V2 runtime-isolated manifest")
    rows = read_jsonl(results_path.resolve())
    if len(rows) != len(items):
        raise PrepError(
            f"A-output coverage is incomplete: expected={len(items)}, got={len(rows)}"
        )
    runtime = manifest["producer_runtime_lock"]
    expected_runtime = {
        "implementation_commit": runtime["implementation_commit"],
        "implementation_sha256": runtime["implementation_sha256"],
        "model_sha256": runtime["model_sha256"],
        "checkpoint_sha256": canonical_sha256(runtime["checkpoint_sha256"]),
    }
    result_root = results_path.resolve().parent
    seen: set[str] = set()
    status_counts = {"success": 0, "failed": 0, "oom": 0}
    per_variant = {variant_id: 0 for variant_id in V2_VARIANT_IDS}
    synthetic_rows = 0
    for index, (row, item) in enumerate(zip(rows, items)):
        if set(row) != A_OUTPUT_KEYS:
            raise PrepError("A-output row fields differ from the frozen A schema")
        if row.get("schema_version") != A_PRODUCER_OUTPUT_SCHEMA:
            raise PrepError("A-output row schema differs")
        item_id = row.get("item_id")
        if item_id != item["item_id"] or item_id in seen:
            raise PrepError("A-output item ID or ordering differs")
        seen.add(str(item_id))
        if row.get("sample_key") != item["sample_key"]:
            raise PrepError(f"A-output sample key differs: {item_id}")
        variant_id = row.get("mask_variant_id")
        if variant_id != item["mask_variant_id"] or variant_id not in V2_VARIANT_IDS:
            raise PrepError(f"A-output mask variant differs: {item_id}")
        per_variant[str(variant_id)] += 1
        if row.get("manifest_lock_sha256") != manifest["manifest_lock_sha256"]:
            raise PrepError(f"A-output manifest lock differs: {item_id}")
        input_sha = row.get("input_sha256")
        if not isinstance(input_sha, dict) or set(input_sha) != INPUT_SHA_KEYS:
            raise PrepError("A-output input SHA fields differ")
        for slot in INPUT_SHA_KEYS:
            if input_sha.get(slot) != item["inputs"][slot]["sha256"]:
                raise PrepError(f"A-output input SHA drift: {item_id}:{slot}")
        for field, expected in expected_runtime.items():
            if row.get(field) != expected:
                raise PrepError(f"A-output runtime identity differs: {field}")
        counters = row.get("access_counters")
        if not isinstance(counters, dict) or set(counters) != ACCESS_COUNTER_KEYS:
            raise PrepError("A-output access counter fields differ")
        for field in ACCESS_COUNTER_KEYS:
            if counters.get(field) != 0:
                raise PrepError(f"A-output access counter is nonzero: {field}")
        attempt = row.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise PrepError("A-output attempt is invalid")
        _finite(row.get("latency_ms"), field="latency_ms", minimum=0.0)
        if not isinstance(row.get("oom"), bool) or not isinstance(
            row.get("failure"), bool
        ):
            raise PrepError("A-output failure/OOM markers are invalid")
        status = row.get("status")
        if status == "success":
            if (
                row["failure"] is not False
                or row["oom"] is not False
                or row["failure_reason"] is not None
            ):
                raise PrepError("Successful A-output row declares failure/OOM")
            initial = _legal_se3(
                row.get("initial_model_to_camera_pose_m"), field="initial pose"
            )
            final = _legal_se3(
                row.get("final_model_to_camera_pose_m"), field="final pose"
            )
            top_k = _validate_top_k(row.get("top_k"))
            _validate_trace(
                row.get("refiner_trace"),
                initial_pose=initial,
                final_pose=final,
            )
            _validate_visualizations(
                row.get("visualization_inventory"),
                item_id=str(item_id),
                result_root=result_root,
                verify_assets=True,
            )
            if all(
                str(candidate["candidate_id"]).startswith("synthetic-fixture-")
                for candidate in top_k
            ):
                synthetic_rows += 1
        elif status in {"failed", "oom"}:
            if row["failure"] is not True:
                raise PrepError("Failed A-output row lacks the failure marker")
            if row["oom"] is not (status == "oom"):
                raise PrepError("Failed A-output row has inconsistent OOM marker")
            if (
                not isinstance(row.get("failure_reason"), str)
                or not row["failure_reason"]
            ):
                raise PrepError("Failed A-output row lacks a failure reason")
            if any(
                row[field] is not None
                for field in (
                    "initial_model_to_camera_pose_m",
                    "final_model_to_camera_pose_m",
                )
            ):
                raise PrepError("Failed A-output row fabricates a pose")
            if row["top_k"] != [] or row["refiner_trace"] != []:
                raise PrepError(
                    "Failed A-output row fabricates candidate/refiner evidence"
                )
            if row["visualization_inventory"] != []:
                raise PrepError("Failed A-output row fabricates visualization evidence")
        else:
            raise PrepError("A-output status is invalid")
        status_counts[str(status)] += 1
    if seen != {item["item_id"] for item in items}:
        raise PrepError("A-output coverage is incomplete")
    if per_variant != manifest["coverage"]["per_variant_item_count"]:
        raise PrepError("A-output per-variant coverage differs")
    return {
        "schema_version": A_OUTPUT_VALIDATION_SCHEMA,
        "protocol_id": PREP_PROTOCOL_ID,
        "manifest_schema": RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
        "manifest_lock_sha256": manifest["manifest_lock_sha256"],
        "results_sha256": sha256_file(results_path.resolve()),
        "row_count": len(rows),
        "base_sample_count": manifest["coverage"]["base_sample_count"],
        "variant_count": len(V2_VARIANT_IDS),
        "per_variant_item_count": per_variant,
        "status_counts": status_counts,
        **expected_runtime,
        "coverage_percent": 100.0,
        "synthetic_fixture_row_count": synthetic_rows,
        "status": "valid",
        "label_access_count_on_gpu_c": 0,
        "gt_path_open_count_on_gpu_c": 0,
        "evaluator_path_open_count_on_gpu_c": 0,
        "scorer_path_open_count_on_gpu_c": 0,
        "accuracy_claim_permitted": False,
    }
