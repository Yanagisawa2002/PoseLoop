"""Independent strict validator for the GPU-C producer return contract."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .c_handoff import load_and_validate_c_handoff
from .core import ContractError, canonical_json_bytes, canonical_sha256, sha256_file, write_json

C_RESULTS_SCHEMA = "poseloop.pose-accuracy-recovery.producer-output.v1"
C_RESULTS_VALIDATION_SCHEMA = (
    "poseloop.pose-accuracy-recovery.producer-output-validation.v1"
)

_ROW_KEYS = {
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
_INPUT_NAMES = {"rgb", "depth", "mask", "camera", "cad"}
_COUNTER_NAMES = {
    "label_access_count_on_gpu_c",
    "gt_path_open_count_on_gpu_c",
    "evaluator_path_open_count_on_gpu_c",
    "scorer_path_open_count_on_gpu_c",
}
_VISUAL_ROLES = {
    "rgb",
    "input_mask",
    "initial_pose_overlay",
    "top_k_overlay",
    "final_pose_overlay",
}
_FORBIDDEN_VISUAL_PATH_TOKENS = {
    "gt",
    "groundtruth",
    "ground_truth",
    "evaluator",
    "evaluation",
    "sealed",
    "official_score",
    "official_scorer",
}
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ContractError(f"{label} must be at least {minimum}")
    return number


def _pose(value: Any, label: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ContractError(f"{label} must be a row-major 4x4 matrix")
    matrix = [
        [
            _finite(cell, f"{label}[{row_index}][{column_index}]")
            for column_index, cell in enumerate(row)
        ]
        for row_index, row in enumerate(value)
    ]
    if any(
        not math.isclose(matrix[3][index], expected, abs_tol=1e-8)
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        raise ContractError(f"{label} homogeneous last row is invalid")
    rotation = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[left][axis] * rotation[right][axis] for axis in range(3))
            expected = 1.0 if left == right else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-5):
                raise ContractError(f"{label} rotation is not orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-5):
        raise ContractError(f"{label} rotation determinant must be +1")
    return matrix


def read_c_result_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"Cannot read C result JSONL {path}: {exc}") from exc
    if not lines:
        raise ContractError("C result JSONL is empty")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ContractError(f"C result JSONL line {index} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"C result JSONL line {index} is invalid: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"C result JSONL line {index} must be an object")
        rows.append(value)
    return rows


def _validate_top_k(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 5:
        raise ContractError(f"{label} must contain exactly five candidates")
    seen: set[str] = set()
    previous_score = math.inf
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        candidate = _strict_keys(
            raw,
            {"rank", "candidate_id", "score", "model_to_camera_pose_m"},
            f"{label}[{index}]",
        )
        if candidate["rank"] != index + 1:
            raise ContractError(f"{label} ranks must be contiguous and one-based")
        candidate_id = candidate["candidate_id"]
        if (
            not isinstance(candidate_id, str)
            or not _CANDIDATE_ID.fullmatch(candidate_id)
            or candidate_id in seen
        ):
            raise ContractError(f"{label} candidate IDs are invalid or duplicate")
        seen.add(candidate_id)
        score = _finite(candidate["score"], f"{label}[{index}].score")
        if score > previous_score:
            raise ContractError(f"{label} scores must be descending")
        previous_score = score
        _pose(
            candidate["model_to_camera_pose_m"],
            f"{label}[{index}].model_to_camera_pose_m",
        )
        normalized.append(candidate)
    return normalized


def _validate_refiner_trace(
    value: Any,
    *,
    initial_pose: Sequence[Sequence[float]],
    final_pose: Sequence[Sequence[float]],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != 6:
        raise ContractError(f"{label} must contain all six frozen states")
    poses: list[list[list[float]]] = []
    for index, raw in enumerate(value):
        entry = _strict_keys(
            raw,
            {"iteration", "model_to_camera_pose_m", "objective"},
            f"{label}[{index}]",
        )
        if entry["iteration"] != index:
            raise ContractError(f"{label} iterations must start at zero and be contiguous")
        poses.append(
            _pose(
                entry["model_to_camera_pose_m"],
                f"{label}[{index}].model_to_camera_pose_m",
            )
        )
        _finite(entry["objective"], f"{label}[{index}].objective")
    if canonical_json_bytes(poses[0]) != canonical_json_bytes(initial_pose):
        raise ContractError(f"{label} does not start at the initial pose")
    if canonical_json_bytes(poses[-1]) != canonical_json_bytes(final_pose):
        raise ContractError(f"{label} does not end at the final pose")


def _safe_visual_path(value: Any, label: str, *, item_id: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 3
        or path.parts[0] != "visualizations"
        or path.parts[1] != item_id
    ):
        raise ContractError(f"{label} must be isolated below visualizations/{item_id}/")
    tokens = {
        token
        for part in path.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token
    }
    if tokens & _FORBIDDEN_VISUAL_PATH_TOKENS:
        raise ContractError(f"{label} exposes a forbidden GT/evaluator role")
    return path


def _validate_visualizations(
    value: Any,
    *,
    item_id: str,
    result_root: Path,
    verify_assets: bool,
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(_VISUAL_ROLES):
        raise ContractError(f"{label} must contain all five producer views")
    seen: set[str] = set()
    root = result_root.resolve()
    for index, raw in enumerate(value):
        asset = _strict_keys(
            raw,
            {"role", "relative_path", "sha256", "bytes"},
            f"{label}[{index}]",
        )
        role = asset["role"]
        if role not in _VISUAL_ROLES or role in seen:
            raise ContractError(f"{label} has a missing, duplicate, or unknown role")
        seen.add(role)
        relative = _safe_visual_path(
            asset["relative_path"], f"{label}[{index}].relative_path", item_id=item_id
        )
        if not _is_sha256(asset["sha256"]):
            raise ContractError(f"{label}[{index}].sha256 must be SHA-256")
        if (
            not isinstance(asset["bytes"], int)
            or isinstance(asset["bytes"], bool)
            or asset["bytes"] <= 0
        ):
            raise ContractError(f"{label}[{index}].bytes must be positive")
        candidate = (root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"{label}[{index}] escapes result root") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError(f"Missing or symlinked visualization: {relative}")
        if verify_assets:
            if candidate.stat().st_size != asset["bytes"]:
                raise ContractError(f"Visualization byte mismatch: {relative}")
            if sha256_file(candidate) != asset["sha256"]:
                raise ContractError(f"Visualization SHA-256 mismatch: {relative}")
    if seen != _VISUAL_ROLES:
        raise ContractError(f"{label} visualization role coverage is incomplete")


def _validate_success_row(
    row: Mapping[str, Any], *, result_root: Path, verify_assets: bool, label: str
) -> None:
    if (
        row["failure"] is not False
        or row["oom"] is not False
        or row["failure_reason"] is not None
    ):
        raise ContractError(f"{label} success row declares failure/OOM")
    initial = _pose(row["initial_model_to_camera_pose_m"], f"{label}.initial_pose")
    final = _pose(row["final_model_to_camera_pose_m"], f"{label}.final_pose")
    _validate_top_k(row["top_k"], f"{label}.top_k")
    _validate_refiner_trace(
        row["refiner_trace"],
        initial_pose=initial,
        final_pose=final,
        label=f"{label}.refiner_trace",
    )
    _validate_visualizations(
        row["visualization_inventory"],
        item_id=row["item_id"],
        result_root=result_root,
        verify_assets=verify_assets,
        label=f"{label}.visualization_inventory",
    )


def _validate_failure_row(row: Mapping[str, Any], *, label: str) -> None:
    if row["status"] not in {"failed", "oom"}:
        raise ContractError(f"{label}.status is unsupported")
    if row["failure"] is not True:
        raise ContractError(f"{label}.failure must be true on failed/OOM rows")
    if row["oom"] is not (row["status"] == "oom"):
        raise ContractError(f"{label}.oom is inconsistent with status")
    if not isinstance(row["failure_reason"], str) or not row["failure_reason"]:
        raise ContractError(f"{label}.failure_reason must explain the failure")
    if any(
        row[name] is not None
        for name in (
            "initial_model_to_camera_pose_m",
            "final_model_to_camera_pose_m",
        )
    ):
        raise ContractError(f"{label} failed pose fields must be explicit null")
    if row["top_k"] != [] or row["refiner_trace"] != []:
        raise ContractError(f"{label} failed list outputs must be explicit empty lists")
    if row["visualization_inventory"] != []:
        raise ContractError(f"{label} failed visualization inventory must be explicit []")


def validate_c_result_rows(
    handoff_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    result_root: Path,
    verify_assets: bool = True,
) -> dict[str, Any]:
    """Validate exact ordered coverage and all producer return provenance."""
    expected_items = handoff_manifest["items"]
    if len(rows) != len(expected_items):
        raise ContractError(
            f"C result coverage mismatch: expected {len(expected_items)}, got {len(rows)}"
        )
    runtime = handoff_manifest["producer_runtime_lock"]
    expected_checkpoint_sha = canonical_sha256(runtime["checkpoint_sha256"])
    seen: set[str] = set()
    status_counts = {"success": 0, "failed": 0, "oom": 0}
    variant_counts = {
        variant["mask_variant_id"]: 0 for variant in handoff_manifest["mask_variants"]
    }
    for index, (raw_row, item) in enumerate(zip(rows, expected_items)):
        label = f"results[{index}]"
        row = _strict_keys(raw_row, _ROW_KEYS, label)
        if row["schema_version"] != C_RESULTS_SCHEMA:
            raise ContractError(f"{label}.schema_version mismatch")
        if row["item_id"] != item["item_id"]:
            raise ContractError(f"{label}.item_id or result ordering mismatch")
        if row["item_id"] in seen:
            raise ContractError(f"Duplicate C result item_id: {row['item_id']}")
        seen.add(row["item_id"])
        if row["sample_key"] != item["sample_key"]:
            raise ContractError(f"{label}.sample_key differs from handoff")
        if row["mask_variant_id"] != item["mask_variant_id"]:
            raise ContractError(f"{label}.mask_variant_id differs from handoff")
        if row["manifest_lock_sha256"] != handoff_manifest["manifest_lock_sha256"]:
            raise ContractError(f"{label}.manifest_lock_sha256 mismatch")
        input_hashes = _strict_keys(row["input_sha256"], _INPUT_NAMES, f"{label}.input_sha256")
        expected_hashes = {
            name: item["inputs"][name]["sha256"] for name in _INPUT_NAMES
        }
        if input_hashes != expected_hashes:
            raise ContractError(f"{label}.input_sha256 differs from handoff")
        for name in ("implementation_commit", "implementation_sha256", "model_sha256"):
            if row[name] != runtime[name]:
                raise ContractError(f"{label}.{name} differs from runtime lock")
        if row["checkpoint_sha256"] != expected_checkpoint_sha:
            raise ContractError(f"{label}.checkpoint_sha256 differs from runtime lock")
        status = row["status"]
        if status not in status_counts:
            raise ContractError(f"{label}.status is unsupported")
        if (
            not isinstance(row["attempt"], int)
            or isinstance(row["attempt"], bool)
            or row["attempt"] <= 0
        ):
            raise ContractError(f"{label}.attempt must be a positive integer")
        _finite(row["latency_ms"], f"{label}.latency_ms", minimum=0.0)
        if not isinstance(row["oom"], bool) or not isinstance(row["failure"], bool):
            raise ContractError(f"{label}.oom and failure must be boolean")
        counters = _strict_keys(
            row["access_counters"], _COUNTER_NAMES, f"{label}.access_counters"
        )
        if any(counters[name] != 0 for name in _COUNTER_NAMES):
            raise ContractError(f"{label} reports label/GT/evaluator/scorer access")
        if status == "success":
            _validate_success_row(
                row, result_root=result_root, verify_assets=verify_assets, label=label
            )
        else:
            _validate_failure_row(row, label=label)
        status_counts[status] += 1
        variant_counts[row["mask_variant_id"]] += 1
    if seen != {item["item_id"] for item in expected_items}:
        raise ContractError("C result item coverage is not exact")
    if variant_counts != handoff_manifest["coverage"]["per_variant_item_count"]:
        raise ContractError("C result per-variant coverage differs from handoff")
    summary: dict[str, Any] = {
        "schema_version": C_RESULTS_VALIDATION_SCHEMA,
        "status": "valid",
        "manifest_lock_sha256": handoff_manifest["manifest_lock_sha256"],
        "item_count": len(rows),
        "base_sample_count": handoff_manifest["coverage"]["base_sample_count"],
        "variant_count": len(variant_counts),
        "per_variant_item_count": variant_counts,
        "status_counts": status_counts,
        "implementation_commit": runtime["implementation_commit"],
        "implementation_sha256": runtime["implementation_sha256"],
        "model_sha256": runtime["model_sha256"],
        "checkpoint_sha256": expected_checkpoint_sha,
        "label_access_count_on_gpu_c": 0,
        "gt_path_open_count_on_gpu_c": 0,
        "evaluator_path_open_count_on_gpu_c": 0,
        "scorer_path_open_count_on_gpu_c": 0,
        "accuracy_claim_permitted": False,
    }
    summary["validation_lock_sha256"] = canonical_sha256(summary)
    return summary


def validate_c_results(
    *,
    handoff_manifest_path: Path,
    handoff_bundle_root: Path,
    results_path: Path,
    result_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    handoff, _ = load_and_validate_c_handoff(
        handoff_manifest_path.resolve(),
        bundle_root=handoff_bundle_root.resolve(),
        verify_assets=True,
    )
    rows = read_c_result_rows(results_path.resolve())
    summary = validate_c_result_rows(
        handoff, rows, result_root=result_root.resolve(), verify_assets=True
    )
    summary["results_sha256"] = sha256_file(results_path.resolve())
    unlocked = dict(summary)
    unlocked.pop("validation_lock_sha256", None)
    summary["validation_lock_sha256"] = canonical_sha256(unlocked)
    if output_path is not None:
        write_json(output_path.resolve(), summary)
    return summary
