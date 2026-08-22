"""Immutable ten-row CNOS producer with planned crash/resume semantics."""

from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import PROPOSAL_SCHEMA
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    ALLOWED_INPUT_ROLES,
    RANKING_ALGORITHM_ID,
    SELECTION_REASON,
    validate_proposal_bundle,
)

from . import RUN_RECEIPT_SCHEMA, RUN_STATE_SCHEMA, SCORE_TRACE_SCHEMA
from .adapter import Candidate, rank_candidates, store_raw_cosine
from .contracts import DEPLOYMENT_BOUNDARY_ZERO, validate_deployment


class CnosBackend(Protocol):
    def infer(
        self,
        *,
        rgb_relative_path: str,
        descriptor_relative_path: str,
        target_object_id: int,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[Candidate], list[dict[str, int]]]: ...


class PlannedCrash(RuntimeError):
    """Raised after the frozen number of completed items for resume testing."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise ContractError(f"Immutable CNOS output already differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"Immutable CNOS output raced with different data: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _asset(path: Path, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _write_image(path: Path, image: Image.Image) -> None:
    _write_immutable(path, _png_bytes(image))


def _count_components_8(mask: np.ndarray) -> int:
    data = np.asarray(mask, dtype=bool)
    height, width = data.shape
    parent: list[int] = []

    def new_label() -> int:
        parent.append(len(parent))
        return len(parent) - 1

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> int:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root
        return left_root

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = data[y]
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        current: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            overlaps = [
                label
                for previous_start, previous_end, label in previous
                if previous_start <= end and previous_end >= start
            ]
            component = new_label() if not overlaps else find(overlaps[0])
            for overlap in overlaps[1:]:
                component = union(component, overlap)
            current.append((start, end, component))
        previous = current
    return len({find(label) for label in range(len(parent))})


def _mask_record(mask: np.ndarray, path: Path, output_root: Path) -> tuple[dict[str, Any], list[int]]:
    values = np.asarray(mask)
    if values.shape != (1080, 1440) or values.dtype != np.bool_ or not values.any():
        raise ContractError("CNOS producer mask must be non-empty bool 1080x1440")
    ys, xs = np.nonzero(values)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    image = Image.fromarray(values.astype(np.uint8) * 255, mode="L")
    _write_image(path, image)
    pixels = int(values.sum())
    asset = {
        **_asset(path, output_root),
        "mask_pixels": pixels,
        "coverage": pixels / (1440 * 1080),
        "connected_components": _count_components_8(values),
    }
    return asset, bbox


def _contour(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    interior = values.copy()
    interior[1:, :] &= values[:-1, :]
    interior[:-1, :] &= values[1:, :]
    interior[:, 1:] &= values[:, :-1]
    interior[:, :-1] &= values[:, 1:]
    return np.logical_and(values, np.logical_not(interior))


def _overlay(rgb: np.ndarray, masks: list[np.ndarray], selected: int | None) -> Image.Image:
    output = rgb.astype(np.float32).copy()
    palette = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
    ]
    for index, mask in enumerate(masks):
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        alpha = 0.45 if selected is None or index == selected else 0.18
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
        edge = _contour(mask)
        output[edge] = color
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _item_receipt_lock(receipt: Mapping[str, Any]) -> str:
    unlocked = dict(receipt)
    unlocked.pop("item_receipt_lock_sha256", None)
    return canonical_sha256(unlocked)


def _output_path(
    root: Path, value: Any, *, required_root: str | None
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError("CNOS output asset path must be POSIX-relative")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or (required_root is not None and relative.parts[0] != required_root)
    ):
        location = "the output root" if required_root is None else f"{required_root}/"
        raise ContractError(f"CNOS output asset must remain under {location}")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError("CNOS output asset escapes output root") from exc
    return candidate


def _verify_asset(
    asset: Any, output_root: Path, *, required_root: str | None, label: str
) -> Path:
    if not isinstance(asset, Mapping) or set(asset) != {
        "relative_path",
        "bytes",
        "sha256",
    }:
        raise ContractError(f"{label} asset schema changed")
    path = _output_path(
        output_root, asset["relative_path"], required_root=required_root
    )
    if (
        not path.is_file()
        or path.stat().st_size != asset["bytes"]
        or sha256_file(path) != asset["sha256"]
    ):
        raise ContractError(f"{label} immutable asset changed on disk")
    return path


def _verify_mask_asset(
    asset: Any, bbox: Any, output_root: Path, *, label: str
) -> None:
    if not isinstance(asset, Mapping) or set(asset) != {
        "relative_path",
        "bytes",
        "sha256",
        "mask_pixels",
        "coverage",
        "connected_components",
    }:
        raise ContractError(f"{label} mask asset schema changed")
    basic = {
        name: asset[name] for name in ("relative_path", "bytes", "sha256")
    }
    path = _verify_asset(basic, output_root, required_root="masks", label=label)
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (1440, 1080):
                raise ContractError(f"{label} mask encoding/frame changed")
            values = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} mask is not a decodable PNG") from exc
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 255}) or 255 not in unique:
        raise ContractError(f"{label} mask must remain non-empty strictly binary")
    mask = values == 255
    ys, xs = np.nonzero(mask)
    actual_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    pixels = int(mask.sum())
    if bbox != actual_bbox:
        raise ContractError(f"{label} tight half-open bbox changed")
    if (
        asset["mask_pixels"] != pixels
        or not isinstance(asset["coverage"], (int, float))
        or isinstance(asset["coverage"], bool)
        or not math.isclose(
            float(asset["coverage"]), pixels / (1440 * 1080), abs_tol=1e-12
        )
        or asset["connected_components"] != _count_components_8(mask)
    ):
        raise ContractError(f"{label} mask statistics changed")


def _verify_visualization(asset: Any, output_root: Path, *, label: str) -> None:
    path = _verify_asset(
        asset, output_root, required_root="visualizations", label=label
    )
    try:
        with Image.open(path) as image:
            if image.size != (1440, 1080):
                raise ContractError(f"{label} visualization frame changed")
            image.verify()
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} visualization is not decodable") from exc


def _score_trace_lock(trace: Mapping[str, Any]) -> str:
    unlocked = dict(trace)
    unlocked.pop("score_trace_lock_sha256", None)
    return canonical_sha256(unlocked)


def _candidate_sort_key(candidate: Candidate) -> list[float | int]:
    return [
        -candidate.cad_similarity,
        -candidate.proposal_score,
        -candidate.mask_stability,
        candidate.proposal_index,
    ]


def _build_score_trace(
    *,
    item_id: str,
    sample_key: Mapping[str, Any],
    candidates: list[Candidate],
    run_identity: str,
) -> dict[str, Any]:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "proposal_index": candidate.proposal_index,
                "rank": rank,
                "raw_cad_cosine": candidate.raw_cad_cosine,
                "official_top5_template_cosines": list(
                    candidate.top5_template_cosines
                ),
                "official_top5_template_indices": list(
                    candidate.top5_template_indices
                ),
                "normalized_cad_similarity": candidate.cad_similarity,
                "proposal_score": candidate.proposal_score,
                "mask_stability": candidate.mask_stability,
                "sorting_key": _candidate_sort_key(candidate),
            }
        )
    trace = {
        "schema_version": SCORE_TRACE_SCHEMA,
        "run_identity_sha256": run_identity,
        "item_id": item_id,
        "sample_key": dict(sample_key),
        "formula": {
            "raw": "avg_top_5_l2_normalized_dinov2_template_cosines",
            "normalized": "raw_cosine_plus_one_divide_two",
            "clamp_permitted": False,
            "ranking_order": [
                "normalized_cad_similarity_desc",
                "proposal_score_desc",
                "mask_stability_desc",
                "proposal_index_asc",
            ],
        },
        "candidates": rows,
        "score_trace_lock_sha256": "pending",
    }
    trace["score_trace_lock_sha256"] = _score_trace_lock(trace)
    return trace


def _validate_score_trace(
    trace: Any,
    *,
    run_identity: str,
    expected_item_id: str,
    expected_sample_key: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    if not isinstance(trace, Mapping) or set(trace) != {
        "schema_version",
        "run_identity_sha256",
        "item_id",
        "sample_key",
        "formula",
        "candidates",
        "score_trace_lock_sha256",
    }:
        raise ContractError("CNOS candidate score trace schema changed")
    if (
        trace["schema_version"] != SCORE_TRACE_SCHEMA
        or trace["run_identity_sha256"] != run_identity
        or trace["item_id"] != expected_item_id
        or trace["sample_key"] != expected_sample_key
        or trace["formula"]
        != {
            "raw": "avg_top_5_l2_normalized_dinov2_template_cosines",
            "normalized": "raw_cosine_plus_one_divide_two",
            "clamp_permitted": False,
            "ranking_order": [
                "normalized_cad_similarity_desc",
                "proposal_score_desc",
                "mask_stability_desc",
                "proposal_index_asc",
            ],
        }
        or trace["score_trace_lock_sha256"] != _score_trace_lock(trace)
    ):
        raise ContractError("CNOS candidate score trace identity/formula changed")
    rows = trace["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ContractError("CNOS candidate score trace is empty")
    parsed: dict[int, dict[str, Any]] = {}
    normalized_keys: list[list[float | int]] = []
    raw_keys: list[list[float | int]] = []
    for expected_rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != {
            "proposal_index",
            "rank",
            "raw_cad_cosine",
            "official_top5_template_cosines",
            "official_top5_template_indices",
            "normalized_cad_similarity",
            "proposal_score",
            "mask_stability",
            "sorting_key",
        }:
            raise ContractError("CNOS candidate score row schema changed")
        index = row["proposal_index"]
        values = row["official_top5_template_cosines"]
        indices = row["official_top5_template_indices"]
        raw = row["raw_cad_cosine"]
        normalized = row["normalized_cad_similarity"]
        proposal_score = row["proposal_score"]
        stability = row["mask_stability"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in parsed
            or row["rank"] != expected_rank
            or not isinstance(values, list)
            or len(values) != 5
            or not isinstance(indices, list)
            or len(indices) != 5
            or len(set(indices)) != 5
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not -1.0 <= float(value) <= 1.0
                for value in values
            )
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in indices
            )
            or values != sorted(values, reverse=True)
            or not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(raw)
            or not -1.0 <= float(raw) <= 1.0
            or not math.isclose(float(raw), sum(values) / 5.0, abs_tol=1e-7)
            or not isinstance(normalized, (int, float))
            or isinstance(normalized, bool)
            or not math.isclose(
                float(normalized), store_raw_cosine(raw), abs_tol=1e-7
            )
            or any(
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0.0 <= float(score) <= 1.0
                for score in (proposal_score, stability)
            )
        ):
            raise ContractError("CNOS candidate score trace values changed")
        normalized_key = [-normalized, -proposal_score, -stability, index]
        raw_key = [-raw, -proposal_score, -stability, index]
        if row["sorting_key"] != normalized_key:
            raise ContractError("CNOS candidate score sorting key changed")
        normalized_keys.append(normalized_key)
        raw_keys.append(raw_key)
        parsed[index] = dict(row)
    if normalized_keys != sorted(normalized_keys) or [
        key[-1] for key in sorted(raw_keys)
    ] != [key[-1] for key in normalized_keys]:
        raise ContractError("CNOS normalized ranking differs from raw-cosine ranking")
    return parsed


def _validate_item_receipt(
    path: Path,
    expected_run_identity: str,
    *,
    output_root: Path,
    expected_sample: Mapping[str, Any],
    expected_runtime_item: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_identity_sha256",
        "item_id",
        "item",
        "candidate_score_trace",
        "oom_events",
        "latency_ms",
        "boundary",
        "item_receipt_lock_sha256",
    }:
        raise ContractError("CNOS item receipt schema changed")
    item_id = expected_sample["item_id"]
    if (
        value["schema_version"] != RUN_RECEIPT_SCHEMA
        or value["run_identity_sha256"] != expected_run_identity
        or value["item_id"] != item_id
        or value["boundary"] != DEPLOYMENT_BOUNDARY_ZERO
    ):
        raise ContractError("CNOS item receipt run identity changed")
    if value.get("item_receipt_lock_sha256") != _item_receipt_lock(value):
        raise ContractError("CNOS item receipt canonical lock mismatch")
    if (
        not isinstance(value["latency_ms"], (int, float))
        or isinstance(value["latency_ms"], bool)
        or not math.isfinite(value["latency_ms"])
        or value["latency_ms"] < 0
    ):
        raise ContractError("CNOS item receipt latency is invalid")
    oom_events = value["oom_events"]
    if not isinstance(oom_events, list):
        raise ContractError("CNOS item OOM events must be a list")
    for event in oom_events:
        if not isinstance(event, Mapping) or set(event) != {
            "cursor",
            "failed_chunk_size",
            "retry_chunk_size",
        }:
            raise ContractError("CNOS item OOM event schema changed")
        if (
            any(not isinstance(event[name], int) or isinstance(event[name], bool) for name in event)
            or event["cursor"] < 0
            or event["retry_chunk_size"] < 1
            or event["retry_chunk_size"] >= event["failed_chunk_size"]
        ):
            raise ContractError("CNOS item OOM event values are invalid")

    item = value["item"]
    if not isinstance(item, Mapping) or set(item) != {
        "item_id",
        "sample_key",
        "input_hashes",
        "proposals",
        "selected_proposal_index",
        "selected_mask",
        "selection_reason",
        "visualizations",
    }:
        raise ContractError("CNOS completed item schema changed")
    expected_key = {
        name: expected_sample[name]
        for name in ("scene_id", "image_id", "object_id")
    }
    expected_hashes = {
        role: expected_runtime_item["inputs"][role]["sha256"]
        for role in ALLOWED_INPUT_ROLES
    }
    if (
        item["item_id"] != item_id
        or item["sample_key"] != expected_key
        or item["input_hashes"] != expected_hashes
        or item["selection_reason"] != SELECTION_REASON
    ):
        raise ContractError("CNOS completed item input/sample identity changed")
    trace_path = _verify_asset(
        value["candidate_score_trace"],
        output_root,
        required_root="score-traces",
        label=f"{item_id}.candidate_score_trace",
    )
    trace = read_json(trace_path)
    score_rows = _validate_score_trace(
        trace,
        run_identity=expected_run_identity,
        expected_item_id=item_id,
        expected_sample_key=expected_key,
    )
    proposals = item["proposals"]
    if not isinstance(proposals, list) or not proposals:
        raise ContractError("CNOS completed item has no proposals")
    sort_keys: list[tuple[float, float, float, int]] = []
    seen_indices: set[int] = set()
    for rank, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, Mapping) or set(proposal) != {
            "proposal_index",
            "rank",
            "cad_object_id",
            "cad_similarity",
            "proposal_score",
            "mask_stability",
            "bbox_xyxy",
            "mask",
        }:
            raise ContractError("CNOS completed proposal schema changed")
        index = proposal["proposal_index"]
        scores = [
            proposal["cad_similarity"],
            proposal["proposal_score"],
            proposal["mask_stability"],
        ]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in seen_indices
            or proposal["rank"] != rank
            or proposal["cad_object_id"] != expected_sample["object_id"]
            or any(
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0.0 <= float(score) <= 1.0
                for score in scores
            )
        ):
            raise ContractError("CNOS completed proposal identity/rank/score changed")
        seen_indices.add(index)
        sort_keys.append((-scores[0], -scores[1], -scores[2], index))
        _verify_mask_asset(
            proposal["mask"],
            proposal["bbox_xyxy"],
            output_root,
            label=f"{item_id}.proposal[{index}]",
        )
        score_row = score_rows.get(index)
        if (
            score_row is None
            or score_row["rank"] != rank
            or not math.isclose(
                score_row["normalized_cad_similarity"],
                proposal["cad_similarity"],
                abs_tol=1e-7,
            )
            or score_row["proposal_score"] != proposal["proposal_score"]
            or score_row["mask_stability"] != proposal["mask_stability"]
        ):
            raise ContractError("CNOS proposal differs from candidate score trace")
    if set(score_rows) != seen_indices:
        raise ContractError("CNOS score trace candidate coverage changed")
    if sort_keys != sorted(sort_keys):
        raise ContractError("CNOS completed proposals changed frozen ranking")
    winner = proposals[0]
    if (
        item["selected_proposal_index"] != winner["proposal_index"]
        or item["selected_mask"] != winner["mask"]
    ):
        raise ContractError("CNOS completed rank-one selection changed")
    visualizations = item["visualizations"]
    expected_visuals = {"rgb", "proposal_overview", "selected_mask", "contours"}
    if not isinstance(visualizations, Mapping) or set(visualizations) != expected_visuals:
        raise ContractError("CNOS completed visualization inventory changed")
    for role, asset in visualizations.items():
        _verify_visualization(
            asset, output_root, label=f"{item_id}.visualizations.{role}"
        )
    return value


def _run_identity(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "route_lock_sha256": route["route_lock_sha256"],
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "deployment_lock_sha256": deployment["deployment_lock_sha256"],
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "ranking_algorithm_id": RANKING_ALGORITHM_ID,
        }
    )


def _new_state(run_identity: str, deployment: Mapping[str, Any], runtime_lock: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "run_identity_sha256": run_identity,
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "status": "IN_PROGRESS",
        "attempt_count": 0,
        "completed_items": [],
        "state_lock_sha256": "pending",
    }
    state["state_lock_sha256"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "state_lock_sha256"}
    )
    return state


def _validate_state(
    state: Mapping[str, Any],
    run_identity: str,
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "run_identity_sha256",
        "deployment_lock_sha256",
        "runtime_lock_sha256",
        "status",
        "attempt_count",
        "completed_items",
        "state_lock_sha256",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ContractError("CNOS run state schema changed")
    if (
        state["schema_version"] != RUN_STATE_SCHEMA
        or state["run_identity_sha256"] != run_identity
        or state["deployment_lock_sha256"] != deployment["deployment_lock_sha256"]
        or state["runtime_lock_sha256"] != runtime_lock["runtime_lock_sha256"]
        or state["status"] not in {"IN_PROGRESS", "PLANNED_CRASH", "FAILED", "COMPLETE"}
        or not isinstance(state["attempt_count"], int)
        or isinstance(state["attempt_count"], bool)
        or state["attempt_count"] < 0
    ):
        raise ContractError("CNOS run state identity changed")
    unlocked = dict(state)
    lock = unlocked.pop("state_lock_sha256")
    if lock != canonical_sha256(unlocked):
        raise ContractError("CNOS run state canonical lock mismatch")
    if not isinstance(state["completed_items"], list):
        raise ContractError("CNOS completed item journal must be a list")


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["state_lock_sha256"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "state_lock_sha256"}
    )
    write_json(path, state)


def _attempt_receipt(
    *,
    attempt: int,
    run_identity: str,
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    status: str,
    started_at: str,
    completed_item_count: int,
    error: str | None,
) -> dict[str, Any]:
    value = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "run_identity_sha256": run_identity,
        "attempt": attempt,
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "implementation_commit": deployment["implementation"]["commit"],
        "implementation_tree": deployment["implementation"]["tree"],
        "implementation_source_archive_sha256": deployment["implementation"][
            "source_archive"
        ]["sha256"],
        "source_commit": deployment["source"]["cnos_commit"],
        "source_tree": deployment["source"]["cnos_tree"],
        "model_identity_sha256": deployment["models"][
            "composite_checkpoint_sha256"
        ],
        "completed_item_count": completed_item_count,
        "error": error,
        "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
        "attempt_receipt_lock_sha256": "pending",
    }
    value["attempt_receipt_lock_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "attempt_receipt_lock_sha256"
        }
    )
    return value


def _validate_final_receipt(
    receipt: Any,
    *,
    run_identity: str,
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    output_root: Path,
    completed: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "run_identity_sha256",
        "status",
        "attempt_count",
        "route_lock_sha256",
        "protocol_lock_sha256",
        "deployment_lock_sha256",
        "runtime_lock_sha256",
        "implementation_commit",
        "implementation_tree",
        "implementation_source_archive_sha256",
        "source_commit",
        "source_tree",
        "model_identity_sha256",
        "proposal_bundle",
        "candidate_score_trace_inventory",
        "item_receipt_inventory",
        "completed_item_count",
        "failure_count",
        "oom_events",
        "boundary",
        "run_receipt_lock_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise ContractError("CNOS final receipt schema changed")
    if (
        receipt["schema_version"] != RUN_RECEIPT_SCHEMA
        or receipt["run_identity_sha256"] != run_identity
        or receipt["status"] != "COMPLETE"
        or receipt["attempt_count"] != state["attempt_count"]
        or receipt["route_lock_sha256"] != route["route_lock_sha256"]
        or receipt["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or receipt["deployment_lock_sha256"]
        != deployment["deployment_lock_sha256"]
        or receipt["runtime_lock_sha256"] != runtime_lock["runtime_lock_sha256"]
        or receipt["implementation_commit"]
        != deployment["implementation"]["commit"]
        or receipt["implementation_tree"] != deployment["implementation"]["tree"]
        or receipt["implementation_source_archive_sha256"]
        != deployment["implementation"]["source_archive"]["sha256"]
        or receipt["source_commit"] != deployment["source"]["cnos_commit"]
        or receipt["source_tree"] != deployment["source"]["cnos_tree"]
        or receipt["model_identity_sha256"]
        != deployment["models"]["composite_checkpoint_sha256"]
        or receipt["completed_item_count"] != 10
        or receipt["failure_count"] != 0
        or receipt["boundary"] != DEPLOYMENT_BOUNDARY_ZERO
    ):
        raise ContractError("CNOS final receipt identity/status changed")
    unlocked = dict(receipt)
    lock = unlocked.pop("run_receipt_lock_sha256")
    if lock != canonical_sha256(unlocked):
        raise ContractError("CNOS final receipt canonical lock mismatch")
    _verify_asset(
        receipt["proposal_bundle"],
        output_root,
        required_root=None,
        label="final proposal bundle",
    )
    expected_traces = [
        completed[sample["item_id"]]["candidate_score_trace"]
        for sample in protocol["input_lock"]["samples"]
    ]
    if receipt["candidate_score_trace_inventory"] != expected_traces:
        raise ContractError("CNOS final score-trace inventory changed")
    for index, asset in enumerate(expected_traces):
        _verify_asset(
            asset,
            output_root,
            required_root="score-traces",
            label=f"final score trace[{index}]",
        )
    if receipt["item_receipt_inventory"] != state["completed_items"]:
        raise ContractError("CNOS final item-receipt inventory changed")
    expected_oom = [
        event
        for sample in protocol["input_lock"]["samples"]
        for event in completed[sample["item_id"]]["oom_events"]
    ]
    if receipt["oom_events"] != expected_oom:
        raise ContractError("CNOS final OOM evidence changed")
    return dict(receipt)


def run_producer(
    *,
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    deployment_root: Path,
    output_root: Path,
    backend: CnosBackend,
    planned_crash_after_items: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or resume the exact ten-row producer without changing scientific inputs."""
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    if planned_crash_after_items is not None and (
        not isinstance(planned_crash_after_items, int)
        or isinstance(planned_crash_after_items, bool)
        or not 1 <= planned_crash_after_items <= 10
    ):
        raise ContractError("planned_crash_after_items must be in [1, 10]")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_identity = _run_identity(route, protocol, deployment, runtime_lock)
    state_path = output_root / "run-state.json"
    if state_path.exists():
        state = read_json(state_path)
        _validate_state(state, run_identity, deployment, runtime_lock)
        state = dict(state)
    else:
        state = _new_state(run_identity, deployment, runtime_lock)
        _save_state(state_path, state)

    completed: dict[str, dict[str, Any]] = {}
    expected_samples = {
        sample["item_id"]: sample for sample in protocol["input_lock"]["samples"]
    }
    runtime_items = {item["item_id"]: item for item in runtime_lock["data"]["items"]}
    for journal in state["completed_items"]:
        if not isinstance(journal, dict) or set(journal) != {
            "item_id",
            "receipt_relative_path",
            "receipt_sha256",
        }:
            raise ContractError("CNOS completed-item journal entry changed")
        receipt_path = _output_path(
            output_root,
            journal["receipt_relative_path"],
            required_root="receipts",
        )
        if not receipt_path.is_file() or sha256_file(receipt_path) != journal[
            "receipt_sha256"
        ]:
            raise ContractError("CNOS completed item receipt changed on disk")
        if journal["item_id"] not in expected_samples:
            raise ContractError("CNOS completed item is outside the frozen workload")
        receipt = _validate_item_receipt(
            receipt_path,
            run_identity,
            output_root=output_root,
            expected_sample=expected_samples[journal["item_id"]],
            expected_runtime_item=runtime_items[journal["item_id"]],
        )
        if receipt["item_id"] != journal["item_id"] or receipt["item_id"] in completed:
            raise ContractError("CNOS completed item identity is duplicate or mismatched")
        completed[receipt["item_id"]] = receipt
    if state["status"] == "COMPLETE":
        bundle_path = output_root / "proposal-bundle.json"
        final_receipt_path = output_root / "run-receipt.json"
        if not bundle_path.is_file() or not final_receipt_path.is_file():
            raise ContractError("Completed CNOS state is missing frozen final artifacts")
        bundle = read_json(bundle_path)
        receipt = read_json(final_receipt_path)
        if not isinstance(bundle, dict) or not isinstance(receipt, dict):
            raise ContractError("Completed CNOS final artifacts are malformed")
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime_lock,
            input_root=deployment_root,
            asset_root=output_root,
        )
        return bundle, _validate_final_receipt(
            receipt,
            run_identity=run_identity,
            route=route,
            protocol=protocol,
            deployment=deployment,
            runtime_lock=runtime_lock,
            output_root=output_root,
            completed=completed,
            state=state,
        )

    state["attempt_count"] += 1
    attempt = state["attempt_count"]
    state["status"] = "IN_PROGRESS"
    _save_state(state_path, state)
    started_at = _utc_now()
    attempt_path = output_root / "attempts" / f"attempt-{attempt:04d}.json"
    try:
        for sample in protocol["input_lock"]["samples"]:
            item_id = sample["item_id"]
            if item_id in completed:
                continue
            runtime_item = runtime_items[item_id]
            started_item = time.perf_counter()
            candidates, oom_events = backend.infer(
                rgb_relative_path=runtime_item["inputs"]["rgb"]["relative_path"],
                descriptor_relative_path=runtime_item["inputs"][
                    "cad_render_descriptors"
                ]["relative_path"],
                target_object_id=sample["object_id"],
                proposal_chunk_size=deployment["runtime"]["proposal_chunk_size"],
                minimum_chunk_size=deployment["runtime"][
                    "minimum_proposal_chunk_size"
                ],
            )
            ranked = rank_candidates(candidates)
            if not ranked:
                raise ContractError(f"CNOS produced no target candidates: {item_id}")
            proposals: list[dict[str, Any]] = []
            for rank, candidate in enumerate(ranked, start=1):
                mask_path = (
                    output_root
                    / "masks"
                    / item_id
                    / f"proposal-{candidate.proposal_index:04d}.png"
                )
                mask_asset, bbox = _mask_record(
                    candidate.mask, mask_path, output_root
                )
                proposals.append(
                    {
                        "proposal_index": candidate.proposal_index,
                        "rank": rank,
                        "cad_object_id": sample["object_id"],
                        "cad_similarity": candidate.cad_similarity,
                        "proposal_score": candidate.proposal_score,
                        "mask_stability": candidate.mask_stability,
                        "bbox_xyxy": bbox,
                        "mask": mask_asset,
                    }
                )
            rgb_path = deployment_root / Path(
                *runtime_item["inputs"]["rgb"]["relative_path"].split("/")
            )
            with Image.open(rgb_path) as rgb_image:
                rgb = np.asarray(rgb_image.convert("RGB"))
            if rgb.shape != (1080, 1440, 3):
                raise ContractError(f"CNOS RGB frame changed: {item_id}")
            masks = [candidate.mask for candidate in ranked]
            visual_paths = {
                "rgb": output_root / "visualizations" / "rgb" / f"{item_id}.png",
                "proposal_overview": output_root
                / "visualizations"
                / "proposal_overview"
                / f"{item_id}.png",
                "selected_mask": output_root
                / "visualizations"
                / "selected_mask"
                / f"{item_id}.png",
                "contours": output_root
                / "visualizations"
                / "contours"
                / f"{item_id}.png",
            }
            _write_image(visual_paths["rgb"], Image.fromarray(rgb, mode="RGB"))
            _write_image(visual_paths["proposal_overview"], _overlay(rgb, masks, None))
            _write_image(visual_paths["selected_mask"], _overlay(rgb, [masks[0]], 0))
            contour_rgb = rgb.copy()
            contour_rgb[_contour(masks[0])] = np.asarray([255, 0, 0], dtype=np.uint8)
            _write_image(
                visual_paths["contours"], Image.fromarray(contour_rgb, mode="RGB")
            )
            input_hashes = {
                role: runtime_item["inputs"][role]["sha256"]
                for role in ALLOWED_INPUT_ROLES
            }
            item_payload = {
                "item_id": item_id,
                "sample_key": {
                    name: sample[name]
                    for name in ("scene_id", "image_id", "object_id")
                },
                "input_hashes": input_hashes,
                "proposals": proposals,
                "selected_proposal_index": proposals[0]["proposal_index"],
                "selected_mask": dict(proposals[0]["mask"]),
                "selection_reason": SELECTION_REASON,
                "visualizations": {
                    role: _asset(path, output_root)
                    for role, path in visual_paths.items()
                },
            }
            trace = _build_score_trace(
                item_id=item_id,
                sample_key=item_payload["sample_key"],
                candidates=ranked,
                run_identity=run_identity,
            )
            trace_path = output_root / "score-traces" / f"{item_id}.json"
            _write_immutable(trace_path, _json_bytes(trace))
            item_receipt = {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "run_identity_sha256": run_identity,
                "item_id": item_id,
                "item": item_payload,
                "candidate_score_trace": _asset(trace_path, output_root),
                "oom_events": oom_events,
                "latency_ms": (time.perf_counter() - started_item) * 1000.0,
                "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
                "item_receipt_lock_sha256": "pending",
            }
            item_receipt["item_receipt_lock_sha256"] = _item_receipt_lock(
                item_receipt
            )
            item_receipt_path = output_root / "receipts" / f"{item_id}.json"
            _write_immutable(item_receipt_path, _json_bytes(item_receipt))
            item_receipt = _validate_item_receipt(
                item_receipt_path,
                run_identity,
                output_root=output_root,
                expected_sample=sample,
                expected_runtime_item=runtime_item,
            )
            journal = {
                "item_id": item_id,
                "receipt_relative_path": item_receipt_path.relative_to(
                    output_root
                ).as_posix(),
                "receipt_sha256": sha256_file(item_receipt_path),
            }
            state["completed_items"].append(journal)
            completed[item_id] = item_receipt
            _save_state(state_path, state)
            if (
                planned_crash_after_items is not None
                and len(completed) >= planned_crash_after_items
            ):
                raise PlannedCrash(
                    f"Planned crash after {planned_crash_after_items} completed items"
                )

        ordered_items = [
            completed[sample["item_id"]]["item"]
            for sample in protocol["input_lock"]["samples"]
        ]
        bundle = {
            "schema_version": PROPOSAL_SCHEMA,
            "protocol_id": protocol["protocol_id"],
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "proposal_bundle_lock_sha256": "pending",
            "role": "DEVELOPMENT_ONLY_INDEPENDENT_INSTANCE_PROPOSAL",
            "input_roles": list(ALLOWED_INPUT_ROLES),
            "ranking_algorithm_id": RANKING_ALGORITHM_ID,
            "boundary": dict(runtime_lock["boundary"]),
            "items": ordered_items,
        }
        bundle["proposal_bundle_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in bundle.items()
                if key != "proposal_bundle_lock_sha256"
            }
        )
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime_lock,
            input_root=deployment_root,
            asset_root=output_root,
        )
        bundle_path = output_root / "proposal-bundle.json"
        _write_immutable(bundle_path, _json_bytes(bundle))
        final_receipt = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "run_identity_sha256": run_identity,
            "status": "COMPLETE",
            "attempt_count": attempt,
            "route_lock_sha256": route["route_lock_sha256"],
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "deployment_lock_sha256": deployment["deployment_lock_sha256"],
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "implementation_commit": deployment["implementation"]["commit"],
            "implementation_tree": deployment["implementation"]["tree"],
            "implementation_source_archive_sha256": deployment[
                "implementation"
            ]["source_archive"]["sha256"],
            "source_commit": deployment["source"]["cnos_commit"],
            "source_tree": deployment["source"]["cnos_tree"],
            "model_identity_sha256": deployment["models"][
                "composite_checkpoint_sha256"
            ],
            "proposal_bundle": _asset(bundle_path, output_root),
            "candidate_score_trace_inventory": [
                completed[sample["item_id"]]["candidate_score_trace"]
                for sample in protocol["input_lock"]["samples"]
            ],
            "item_receipt_inventory": [
                dict(item) for item in state["completed_items"]
            ],
            "completed_item_count": 10,
            "failure_count": 0,
            "oom_events": [
                event
                for sample in protocol["input_lock"]["samples"]
                for event in completed[sample["item_id"]]["oom_events"]
            ],
            "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
            "run_receipt_lock_sha256": "pending",
        }
        final_receipt["run_receipt_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in final_receipt.items()
                if key != "run_receipt_lock_sha256"
            }
        )
        final_receipt_path = output_root / "run-receipt.json"
        state["status"] = "COMPLETE"
        _save_state(state_path, state)
        _validate_final_receipt(
            final_receipt,
            run_identity=run_identity,
            route=route,
            protocol=protocol,
            deployment=deployment,
            runtime_lock=runtime_lock,
            output_root=output_root,
            completed=completed,
            state=state,
        )
        _write_immutable(final_receipt_path, _json_bytes(final_receipt))
        _write_immutable(
            attempt_path,
            _json_bytes(
                _attempt_receipt(
                    attempt=attempt,
                    run_identity=run_identity,
                    deployment=deployment,
                    runtime_lock=runtime_lock,
                    status="COMPLETE",
                    started_at=started_at,
                    completed_item_count=10,
                    error=None,
                )
            ),
        )
        return bundle, final_receipt
    except PlannedCrash as exc:
        state["status"] = "PLANNED_CRASH"
        _save_state(state_path, state)
        _write_immutable(
            attempt_path,
            _json_bytes(
                _attempt_receipt(
                    attempt=attempt,
                    run_identity=run_identity,
                    deployment=deployment,
                    runtime_lock=runtime_lock,
                    status="PLANNED_CRASH",
                    started_at=started_at,
                    completed_item_count=len(completed),
                    error=str(exc),
                )
            ),
        )
        raise
    except Exception as exc:
        state["status"] = "FAILED"
        _save_state(state_path, state)
        _write_immutable(
            attempt_path,
            _json_bytes(
                _attempt_receipt(
                    attempt=attempt,
                    run_identity=run_identity,
                    deployment=deployment,
                    runtime_lock=runtime_lock,
                    status="FAILED",
                    started_at=started_at,
                    completed_item_count=len(completed),
                    error=f"{type(exc).__name__}: {exc}",
                )
            ),
        )
        raise
