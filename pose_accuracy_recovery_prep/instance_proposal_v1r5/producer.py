"""Create-only/resumable A-R5 all-catalog instance-proposal producer."""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)

from . import (
    ITEM_RECEIPT_SCHEMA,
    OUTPUT_BUNDLE_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    RUN_STATE_SCHEMA,
    SCORE_TRACE_SCHEMA,
)
from .adapter import InstanceProposal, rank_proposals
from .contracts import (
    BOUNDARY_ZERO,
    CATALOG_OBJECT_IDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VISUALIZATION_ROLES,
    decoded_image_binding,
    mask_statistics,
    proposal_adjacency_pairs,
    proposal_state,
    validate_output_bundle,
    validate_runtime_request,
)


class CnosCatalogBackend(Protocol):
    def infer_frame(
        self,
        *,
        rgb_relative_path: str,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[InstanceProposal], list[dict[str, Any]]]: ...


class PlannedCrash(RuntimeError):
    """Intentional stop after a completed prefix for resume validation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(f"A-R5 create-only artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ContractError(f"A-R5 create-only artifact raced: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_image_create_only(path: Path, image: Image.Image) -> None:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    _write_create_only(path, stream.getvalue())


def _output_asset(path: Path, output_root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.resolve().relative_to(output_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _input_path(data_root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or "\\" in relative_path:
        raise ContractError("A-R5 input path must be POSIX-relative")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ContractError("A-R5 input path is unsafe")
    root = data_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError("A-R5 input path escapes data root") from exc
    return path


def _contour(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    interior = values.copy()
    interior[1:] &= values[:-1]
    interior[:-1] &= values[1:]
    interior[:, 1:] &= values[:, :-1]
    interior[:, :-1] &= values[:, 1:]
    return np.logical_and(values, np.logical_not(interior))


def _palette(index: int) -> np.ndarray:
    colors = (
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 224),
        (255, 96, 192),
        (192, 255, 64),
    )
    return np.asarray(colors[index % len(colors)], dtype=np.uint8)


def _label_panel(image: Image.Image, lines: list[str]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    height = 12 + 22 * len(lines)
    draw.rectangle((0, 0, min(FRAME_WIDTH, 720), height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((8, 8 + 22 * index), line, fill=(255, 255, 255))
    return result


def _all_contours(rgb: np.ndarray, proposals: list[InstanceProposal]) -> Image.Image:
    output = rgb.copy()
    lines = [f"A-R5 independent FastSAM instances: {len(proposals)}"]
    for rank, proposal in enumerate(proposals, start=1):
        color = _palette(rank - 1)
        output[_contour(proposal.mask)] = color
        selected = proposal.cad_ranking[0]
        lines.append(
            f"p{proposal.proposal_index} conf={proposal.proposal_score:.4f} "
            f"obj={selected.object_id} cad={selected.normalized_similarity:.4f}"
        )
    return _label_panel(Image.fromarray(output, mode="RGB"), lines)


def _selected_overlay(
    rgb: np.ndarray,
    proposals: list[InstanceProposal],
    selected_index: int | None,
) -> Image.Image:
    output = rgb.astype(np.float32).copy()
    lines = ["A-R5 selected instance (selection is label-blind)"]
    selected = next(
        (
            proposal
            for proposal in proposals
            if proposal.proposal_index == selected_index
        ),
        None,
    )
    if selected is None:
        lines.append("NO_PROPOSAL: no GT fill or fallback")
    else:
        color = np.asarray((255, 48, 48), dtype=np.float32)
        mask = selected.mask
        output[mask] = output[mask] * 0.45 + color * 0.55
        output[_contour(mask)] = color
        score = selected.cad_ranking[0]
        lines.append(
            f"p{selected.proposal_index} -> obj={score.object_id}, "
            f"cad={score.normalized_similarity:.4f}"
        )
    return _label_panel(
        Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB"),
        lines,
    )


def _cad_overlay(rgb: np.ndarray, proposals: list[InstanceProposal]) -> Image.Image:
    output = rgb.copy()
    lines = ["A-R5 full CAD catalog ranks (top-5 per independent instance)"]
    for rank, proposal in enumerate(proposals, start=1):
        output[_contour(proposal.mask)] = _palette(rank - 1)
        values = ", ".join(
            f"{score.object_id}:{score.normalized_similarity:.3f}"
            for score in proposal.cad_ranking
        )
        lines.append(f"p{proposal.proposal_index} [{values}]")
    if not proposals:
        lines.append("NO_PROPOSAL: catalog ranking is empty by construction")
    return _label_panel(Image.fromarray(output, mode="RGB"), lines)


def _adjacency_overlay(
    rgb: np.ndarray,
    proposals: list[InstanceProposal],
    adjacency: list[list[int]],
) -> Image.Image:
    image = _all_contours(rgb, proposals)
    draw = ImageDraw.Draw(image)
    centers: list[tuple[int, int]] = []
    for proposal in proposals:
        stats = mask_statistics(proposal.mask)
        x0, y0, x1, y1 = stats["bbox_xyxy_half_open"]
        centers.append(((x0 + x1) // 2, (y0 + y1) // 2))
    for left, right in adjacency:
        draw.line((centers[left], centers[right]), fill=(255, 255, 0), width=4)
    draw.text(
        (8, FRAME_HEIGHT - 26),
        f"adjacent/touching proposal pairs: {adjacency}",
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return image


def _producer_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "implementation_commit": request["implementation"]["commit"],
        "implementation_tree": request["implementation"]["tree"],
        "implementation_source_archive_sha256": request["implementation"][
            "source_archive"
        ]["sha256"],
        "cnos_commit": request["source"]["cnos_commit"],
        "cnos_tree": request["source"]["cnos_tree"],
        "dinov2_commit": request["source"]["dinov2_commit"],
        "dinov2_tree": request["source"]["dinov2_tree"],
        "adapter_config_sha256": request["adapter_config"]["sha256"],
        "fastsam_x_checkpoint_sha256": request["models"]["fastsam_x_checkpoint"][
            "sha256"
        ],
        "dinov2_vitl14_checkpoint_sha256": request["models"][
            "dinov2_vitl14_checkpoint"
        ]["sha256"],
        "composite_model_sha256": request["models"]["composite_model_sha256"],
        "template_manifest_sha256": request["template_manifest"]["sha256"],
        "descriptor_generation_receipt_sha256": request[
            "descriptor_generation_receipt"
        ]["sha256"],
        "catalog_cad_sha256": {
            str(entry["object_id"]): entry["cad"]["sha256"]
            for entry in request["catalog"]
        },
        "catalog_descriptor_sha256": {
            str(entry["object_id"]): entry["descriptor"]["sha256"]
            for entry in request["catalog"]
        },
    }


def _cad_ranking(proposal: InstanceProposal) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "object_id": score.object_id,
            "raw_cosine": score.raw_cosine,
            "normalized_similarity": score.normalized_similarity,
            "top5_template_cosines": list(score.top5_template_cosines),
            "top5_template_indices": list(score.top5_template_indices),
            "descriptor_sha256": score.descriptor_sha256,
        }
        for rank, score in enumerate(proposal.cad_ranking, start=1)
    ]


def _mask_asset(
    mask: np.ndarray,
    path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    statistics = mask_statistics(mask)
    _write_image_create_only(
        path, Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    )
    asset = {
        **_output_asset(path, output_root, "independent_instance_mask"),
        "decoded_shape": [FRAME_HEIGHT, FRAME_WIDTH],
        "decoded_mode": "L",
        "foreground_pixels": statistics["mask_pixels"],
        "support_bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
        "mask_pixels": statistics["mask_pixels"],
        "coverage": statistics["coverage"],
        "connected_components": statistics["connected_components"],
    }
    return asset, statistics


def _score_trace(
    item_id: str,
    run_identity: str,
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    trace = {
        "schema_version": SCORE_TRACE_SCHEMA,
        "run_identity_sha256": run_identity,
        "item_id": item_id,
        "normalization": "raw_cosine_plus_one_divide_two_without_clamp",
        "catalog_order": list(CATALOG_OBJECT_IDS),
        "proposals": [
            {
                "proposal_index": proposal["proposal_index"],
                "cad_ranking": proposal["cad_ranking"],
            }
            for proposal in proposals
        ],
        "score_trace_lock_sha256": "pending",
    }
    trace["score_trace_lock_sha256"] = canonical_sha256(
        {key: value for key, value in trace.items() if key != "score_trace_lock_sha256"}
    )
    return trace


def _run_identity(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "frame_manifest_lock_sha256": manifest["frame_manifest_lock_sha256"],
            "runtime_request_lock_sha256": request["runtime_request_lock_sha256"],
        }
    )


def _new_state(run_identity: str) -> dict[str, Any]:
    return {
        "schema_version": RUN_STATE_SCHEMA,
        "run_identity_sha256": run_identity,
        "status": "READY",
        "attempt_count": 0,
        "completed_frames": [],
        "failure": None,
    }


def _validate_state(state: Any, run_identity: str) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "run_identity_sha256",
        "status",
        "attempt_count",
        "completed_frames",
        "failure",
    }:
        raise ContractError("A-R5 run state schema changed")
    if (
        state["schema_version"] != RUN_STATE_SCHEMA
        or state["run_identity_sha256"] != run_identity
        or state["status"]
        not in {"READY", "IN_PROGRESS", "PLANNED_CRASH", "FAILED", "COMPLETE"}
        or not isinstance(state["attempt_count"], int)
        or state["attempt_count"] < 0
        or not isinstance(state["completed_frames"], list)
    ):
        raise ContractError("A-R5 run state identity/status changed")
    return state


def _validate_completed_receipt(
    journal: Mapping[str, Any],
    *,
    expected_frame: Mapping[str, Any],
    run_identity: str,
    output_root: Path,
) -> dict[str, Any]:
    if set(journal) != {"item_id", "receipt_relative_path", "receipt_sha256"}:
        raise ContractError("A-R5 completed-frame journal changed")
    path = _input_path(output_root, journal["receipt_relative_path"])
    if (
        not path.is_file()
        or sha256_file(path) != journal["receipt_sha256"]
        or journal["item_id"] != expected_frame["item_id"]
    ):
        raise ContractError("A-R5 completed-frame receipt changed")
    receipt = read_json(path)
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "run_identity_sha256",
        "ordinal",
        "item",
        "boundary",
        "item_receipt_lock_sha256",
    }:
        raise ContractError("A-R5 item receipt schema changed")
    if (
        receipt["schema_version"] != ITEM_RECEIPT_SCHEMA
        or receipt["run_identity_sha256"] != run_identity
        or receipt["item"].get("item_id") != expected_frame["item_id"]
        or receipt["boundary"] != BOUNDARY_ZERO
        or receipt["item_receipt_lock_sha256"]
        != canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "item_receipt_lock_sha256"
            }
        )
    ):
        raise ContractError("A-R5 item receipt identity/self-lock changed")
    # Disk-check every completed-prefix output before skipping model execution.
    item = receipt["item"]
    for proposal in item["proposals"]:
        asset = proposal["mask"]
        mask_path = _input_path(output_root, asset["relative_path"])
        if (
            not mask_path.is_file()
            or mask_path.stat().st_size != asset["bytes"]
            or sha256_file(mask_path) != asset["sha256"]
        ):
            raise ContractError("A-R5 completed proposal mask changed before resume")
    for asset in [item["score_trace"], *item["visualizations"].values()]:
        asset_path = _input_path(output_root, asset["relative_path"])
        if (
            not asset_path.is_file()
            or asset_path.stat().st_size != asset["bytes"]
            or sha256_file(asset_path) != asset["sha256"]
        ):
            raise ContractError("A-R5 completed evidence changed before resume")
    return receipt


def _make_frame_item(
    *,
    frame: Mapping[str, Any],
    request: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    run_identity: str,
    backend: CnosCatalogBackend,
) -> dict[str, Any]:
    start = time.perf_counter()
    ranked, oom_events = backend.infer_frame(
        rgb_relative_path=frame["inputs"]["rgb"]["relative_path"],
        proposal_chunk_size=request["runtime"]["proposal_chunk_size"],
        minimum_chunk_size=request["runtime"]["minimum_proposal_chunk_size"],
    )
    ranked = rank_proposals(ranked)
    item_id = frame["item_id"]
    proposals: list[dict[str, Any]] = []
    for proposal_rank, proposal in enumerate(ranked, start=1):
        mask_path = (
            output_root
            / "outputs"
            / "masks"
            / item_id
            / f"proposal-{proposal.proposal_index:04d}.png"
        )
        mask_asset, statistics = _mask_asset(proposal.mask, mask_path, output_root)
        ranking = _cad_ranking(proposal)
        proposals.append(
            {
                "proposal_index": proposal.proposal_index,
                "proposal_rank": proposal_rank,
                "proposal_score": proposal.proposal_score,
                "mask_stability": proposal.mask_stability,
                "bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
                "mask": mask_asset,
                "cad_ranking": ranking,
                "selected_object_id": ranking[0]["object_id"],
                "selected_raw_cosine": ranking[0]["raw_cosine"],
                "selected_cad_similarity": ranking[0]["normalized_similarity"],
            }
        )
    adjacency = proposal_adjacency_pairs([proposal.mask for proposal in ranked])
    selected_index = None
    if proposals:
        selected_index = sorted(
            proposals,
            key=lambda proposal: (
                -proposal["selected_cad_similarity"],
                -proposal["proposal_score"],
                -proposal["mask_stability"],
                proposal["proposal_index"],
            ),
        )[0]["proposal_index"]
    rgb_path = _input_path(data_root, frame["inputs"]["rgb"]["relative_path"])
    depth_path = _input_path(data_root, frame["inputs"]["depth"]["relative_path"])
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    visuals = {
        "rgb_all_instance_contours": _all_contours(rgb, ranked),
        "selected_mask": _selected_overlay(rgb, ranked, selected_index),
        "cad_topk_overlay": _cad_overlay(rgb, ranked),
        "adjacency_overlay": _adjacency_overlay(rgb, ranked, adjacency),
    }
    visual_assets: dict[str, Any] = {}
    for role in VISUALIZATION_ROLES:
        path = output_root / "outputs" / "visualizations" / role / f"{item_id}.png"
        _write_image_create_only(path, visuals[role])
        visual_assets[role] = _output_asset(path, output_root, role)
    trace = _score_trace(item_id, run_identity, proposals)
    trace_path = output_root / "outputs" / "score-traces" / f"{item_id}.json"
    _write_create_only(trace_path, _json_bytes(trace))
    rgb_binding = decoded_image_binding(
        rgb_path,
        role="rgb",
        relative_path=frame["inputs"]["rgb"]["relative_path"],
    )
    depth_binding = decoded_image_binding(
        depth_path,
        role="raw_sensor_depth",
        relative_path=frame["inputs"]["depth"]["relative_path"],
    )
    return {
        "item_id": item_id,
        "frame_key": dict(frame["frame_key"]),
        "input_hashes": {
            role: frame["inputs"][role]["sha256"] for role in ("rgb", "depth", "camera")
        },
        "input_bindings": {
            "rgb": rgb_binding,
            "depth": depth_binding,
            "camera": dict(frame["inputs"]["camera"]),
        },
        "producer_identity": _producer_identity(request),
        "proposal_state": proposal_state(len(proposals), adjacency),
        "proposal_count": len(proposals),
        "adjacency_pairs": adjacency,
        "proposals": proposals,
        "selected_proposal_index": selected_index,
        "score_trace": _output_asset(trace_path, output_root, "candidate_score_trace"),
        "visualizations": visual_assets,
        "oom_events": oom_events,
        "latency_ms": (time.perf_counter() - start) * 1000.0,
    }


def validate_run_receipt(
    receipt: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_identity_sha256",
        "status",
        "attempt_count",
        "frame_count",
        "scene_count",
        "proposal_state_counts",
        "output_bundle",
        "boundary",
        "run_receipt_lock_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ContractError("A-R5 final receipt schema changed")
    if (
        receipt["schema_version"] != RUN_RECEIPT_SCHEMA
        or receipt["run_identity_sha256"] != bundle["run_identity_sha256"]
        or receipt["status"] != "COMPLETE"
        or receipt["frame_count"] != 10
        or receipt["scene_count"] != 5
        or receipt["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("A-R5 final receipt identity/status changed")
    expected_counts = {
        state: sum(frame["proposal_state"] == state for frame in bundle["frames"])
        for state in (
            "NO_PROPOSAL",
            "ONE_PROPOSAL",
            "MULTIPLE_PROPOSALS_DISJOINT",
            "MULTIPLE_PROPOSALS_ADJACENT",
        )
    }
    if receipt["proposal_state_counts"] != expected_counts:
        raise ContractError("A-R5 final proposal-state inventory changed")
    asset = receipt["output_bundle"]
    if not isinstance(asset, Mapping) or set(asset) != {
        "role",
        "relative_path",
        "bytes",
        "sha256",
    }:
        raise ContractError("A-R5 final bundle asset schema changed")
    path = _input_path(output_root, asset["relative_path"])
    if (
        asset["role"] != "producer_output_manifest"
        or not path.is_file()
        or path.stat().st_size != asset["bytes"]
        or sha256_file(path) != asset["sha256"]
    ):
        raise ContractError("A-R5 final bundle asset changed")
    if receipt["run_receipt_lock_sha256"] != canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "run_receipt_lock_sha256"
        }
    ):
        raise ContractError("A-R5 final receipt canonical lock mismatch")
    return dict(receipt)


def run_producer(
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    backend: CnosCatalogBackend,
    planned_crash_after_frames: int | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exactly ten frames, or resume a disk-verified planned-crash prefix."""
    validate_runtime_request(request, protocol, manifest, data_root=data_root)
    if planned_crash_after_frames is not None and (
        not isinstance(planned_crash_after_frames, int)
        or isinstance(planned_crash_after_frames, bool)
        or not 1 <= planned_crash_after_frames <= len(manifest["frames"])
    ):
        raise ContractError("planned_crash_after_frames must be in [1,10]")
    output_root = output_root.resolve()
    run_identity = _run_identity(protocol, manifest, request)
    state_path = output_root / "run-state.json"
    if state_path.exists():
        if not resume:
            raise ContractError(
                "A-R5 output root already has state; explicit --resume required"
            )
        state = _validate_state(read_json(state_path), run_identity)
        if state["status"] not in {"PLANNED_CRASH", "COMPLETE"}:
            raise ContractError("A-R5 resume is only permitted after a planned crash")
    else:
        if resume:
            raise ContractError("A-R5 --resume requested without an existing run state")
        if output_root.exists() and any(output_root.iterdir()):
            raise ContractError("A-R5 create-only output root is not empty")
        output_root.mkdir(parents=True, exist_ok=True)
        state = _new_state(run_identity)
        write_json(state_path, state)
    completed: dict[str, dict[str, Any]] = {}
    journals = state["completed_frames"]
    if len(journals) > len(manifest["frames"]):
        raise ContractError("A-R5 completed prefix exceeds frozen frame count")
    for ordinal, journal in enumerate(journals):
        expected = manifest["frames"][ordinal]
        receipt = _validate_completed_receipt(
            journal,
            expected_frame=expected,
            run_identity=run_identity,
            output_root=output_root,
        )
        if receipt["ordinal"] != ordinal or receipt["item"]["item_id"] in completed:
            raise ContractError("A-R5 completed prefix is out of order or duplicated")
        completed[receipt["item"]["item_id"]] = receipt["item"]
    if state["status"] == "COMPLETE":
        bundle_path = output_root / "outputs" / "producer-manifest.json"
        receipt_path = output_root / "outputs" / "run-receipt.json"
        if not bundle_path.is_file() or not receipt_path.is_file():
            raise ContractError("A-R5 completed state is missing final artifacts")
        bundle = read_json(bundle_path)
        receipt = read_json(receipt_path)
        validate_output_bundle(
            bundle,
            protocol,
            manifest,
            request,
            data_root=data_root,
            output_root=output_root,
        )
        validate_run_receipt(receipt, bundle, output_root=output_root)
        return bundle, receipt

    state["attempt_count"] += 1
    attempt_number = state["attempt_count"]
    state["status"] = "IN_PROGRESS"
    state["failure"] = None
    write_json(state_path, state)
    attempt_path = (
        output_root / "outputs" / "attempts" / f"attempt-{attempt_number:04d}.json"
    )
    try:
        for ordinal, frame in enumerate(manifest["frames"]):
            if frame["item_id"] in completed:
                continue
            item = _make_frame_item(
                frame=frame,
                request=request,
                data_root=data_root,
                output_root=output_root,
                run_identity=run_identity,
                backend=backend,
            )
            receipt = {
                "schema_version": ITEM_RECEIPT_SCHEMA,
                "run_identity_sha256": run_identity,
                "ordinal": ordinal,
                "item": item,
                "boundary": dict(BOUNDARY_ZERO),
                "item_receipt_lock_sha256": "pending",
            }
            receipt["item_receipt_lock_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "item_receipt_lock_sha256"
                }
            )
            receipt_path = (
                output_root / "outputs" / "item-receipts" / f"{frame['item_id']}.json"
            )
            _write_create_only(receipt_path, _json_bytes(receipt))
            journal = {
                "item_id": frame["item_id"],
                "receipt_relative_path": receipt_path.relative_to(
                    output_root
                ).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
            }
            state["completed_frames"].append(journal)
            completed[frame["item_id"]] = item
            write_json(state_path, state)
            if (
                planned_crash_after_frames is not None
                and len(completed) >= planned_crash_after_frames
            ):
                state["status"] = "PLANNED_CRASH"
                state["failure"] = {
                    "kind": "PLANNED_CRASH",
                    "completed_frame_count": len(completed),
                }
                write_json(state_path, state)
                attempt = {
                    "attempt": attempt_number,
                    "status": "PLANNED_CRASH",
                    "completed_frame_count": len(completed),
                    "run_identity_sha256": run_identity,
                    "boundary": dict(BOUNDARY_ZERO),
                }
                _write_create_only(attempt_path, _json_bytes(attempt))
                raise PlannedCrash(
                    f"Planned crash after {len(completed)} completed frames"
                )
        ordered = [completed[frame["item_id"]] for frame in manifest["frames"]]
        bundle = {
            "schema_version": OUTPUT_BUNDLE_SCHEMA,
            "protocol_id": protocol["protocol_id"],
            "protocol_lock_sha256": protocol["protocol_lock_sha256"],
            "frame_manifest_lock_sha256": manifest["frame_manifest_lock_sha256"],
            "runtime_request_lock_sha256": request["runtime_request_lock_sha256"],
            "run_identity_sha256": run_identity,
            "role": "DEVELOPMENT_ONLY_LABEL_BLIND_INSTANCE_PROPOSALS",
            "status": "COMPLETE",
            "frame_count": 10,
            "scene_count": 5,
            "catalog_object_ids": list(CATALOG_OBJECT_IDS),
            "boundary": dict(BOUNDARY_ZERO),
            "frames": ordered,
            "output_bundle_lock_sha256": "pending",
        }
        bundle["output_bundle_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in bundle.items()
                if key != "output_bundle_lock_sha256"
            }
        )
        validate_output_bundle(
            bundle,
            protocol,
            manifest,
            request,
            data_root=data_root,
            output_root=output_root,
        )
        bundle_path = output_root / "outputs" / "producer-manifest.json"
        _write_create_only(bundle_path, _json_bytes(bundle))
        final_receipt = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "run_identity_sha256": run_identity,
            "status": "COMPLETE",
            "attempt_count": attempt_number,
            "frame_count": 10,
            "scene_count": 5,
            "proposal_state_counts": {
                state_name: sum(
                    frame["proposal_state"] == state_name for frame in ordered
                )
                for state_name in (
                    "NO_PROPOSAL",
                    "ONE_PROPOSAL",
                    "MULTIPLE_PROPOSALS_DISJOINT",
                    "MULTIPLE_PROPOSALS_ADJACENT",
                )
            },
            "output_bundle": _output_asset(
                bundle_path, output_root, "producer_output_manifest"
            ),
            "boundary": dict(BOUNDARY_ZERO),
            "run_receipt_lock_sha256": "pending",
        }
        final_receipt["run_receipt_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in final_receipt.items()
                if key != "run_receipt_lock_sha256"
            }
        )
        final_receipt_path = output_root / "outputs" / "run-receipt.json"
        _write_create_only(final_receipt_path, _json_bytes(final_receipt))
        validate_run_receipt(final_receipt, bundle, output_root=output_root)
        attempt = {
            "attempt": attempt_number,
            "status": "COMPLETE",
            "completed_frame_count": 10,
            "run_identity_sha256": run_identity,
            "boundary": dict(BOUNDARY_ZERO),
        }
        _write_create_only(attempt_path, _json_bytes(attempt))
        state["status"] = "COMPLETE"
        state["failure"] = None
        write_json(state_path, state)
        return bundle, final_receipt
    except PlannedCrash:
        raise
    except Exception as exc:
        state["status"] = "FAILED"
        state["failure"] = {
            "kind": type(exc).__name__,
            "message": str(exc),
            "completed_frame_count": len(completed),
        }
        write_json(state_path, state)
        if not attempt_path.exists():
            _write_create_only(
                attempt_path,
                _json_bytes(
                    {
                        "attempt": attempt_number,
                        "status": "FAILED",
                        "completed_frame_count": len(completed),
                        "failure": state["failure"],
                        "run_identity_sha256": run_identity,
                        "boundary": dict(BOUNDARY_ZERO),
                    }
                ),
            )
        raise


__all__ = [
    "CnosCatalogBackend",
    "PlannedCrash",
    "run_producer",
    "validate_run_receipt",
]
