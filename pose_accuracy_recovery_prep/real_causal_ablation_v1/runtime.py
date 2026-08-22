"""A-R8 real-development causal ablation.

The producer is label blind: it opens only RGB and raw Realsense depth.  The
evaluator is a separate command which opens the already-consumed development
labels after every model prediction is on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.core import ContractError
from pose_accuracy_recovery_prep.instance_deblend_v1r7.adapter import (
    DEPTH_EDGE_NOISE_MULTIPLIER,
)
from pose_accuracy_recovery_prep.instance_deblend_v1r7.synthetic_runtime import (
    SyntheticGateRuntime,
    load_a_r5_assets,
    verify_sam2_freeze,
)
from pose_accuracy_recovery_prep.instance_selection_v1r6 import (
    FILTER_REASON_ORDER,
    POLICY,
)


PROTOCOL_SCHEMA = "poseloop.pose-accuracy-recovery.real-causal-ablation.v1"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.real-causal-ablation.v1"
EXPECTED_A_R7_COMMIT = "1d9a2d650bada890db35d940aec7d63d57437f47"
EXPECTED_SCENES = (10, 25, 30, 40, 65)
EXPECTED_IMAGES = (0, 10, 20, 30, 40)
VARIANTS = (
    "a_r6_baseline",
    "raw_fastsam_eligible",
    "geometry_instances",
    "geometry_plus_sam2",
)


@dataclass
class Candidate:
    """One instance-mask hypothesis."""

    mask: np.ndarray
    score: float
    source: str
    mask_stability: float = 0.0
    seed_xy: tuple[int, int] | None = None
    proposal_index: int | None = None
    predicted_object_id: int | None = None
    cad_similarity: float | None = None
    cad_ranking: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    sam2_status: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    """Load the exact A-R8 protocol and reject a mutated experimental scope."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("A-R8 protocol must be a JSON object")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("A-R8 protocol schema changed")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("A-R8 protocol identity changed")
    freeze = value.get("a_r7_freeze")
    if not isinstance(freeze, dict) or freeze != {
        "commit": EXPECTED_A_R7_COMMIT,
        "status": "SYNTHETIC_POSITIVE_FORMAL_NO_GO_REAL_UNTESTED",
        "threshold_or_gate_mutation_permitted": False,
        "successor_threshold_tuning_permitted": False,
    }:
        raise ContractError("A-R7 freeze boundary changed")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict):
        raise ContractError("A-R8 dataset contract is missing")
    if (
        tuple(dataset.get("scene_ids", ())) != EXPECTED_SCENES
        or tuple(dataset.get("image_ids_per_scene", ())) != EXPECTED_IMAGES
        or dataset.get("frame_count") != 25
        or dataset.get("sensor") != "realsense"
        or dataset.get("split") != "val"
        or dataset.get("train_pbr_read_permitted") is not False
        or dataset.get("scene9_replay_permitted") is not False
        or dataset.get("sealed_claim_permitted") is not False
    ):
        raise ContractError("A-R8 consumed-real frame scope changed")
    if value.get("variants") != [
        "a_r6_baseline",
        "geometry_instances",
        "geometry_plus_sam2",
        "geometry_plus_sam2_plus_causal_cad",
    ]:
        raise ContractError("A-R8 causal variants changed")
    return value


def _planned_frames(protocol: Mapping[str, Any]) -> list[dict[str, int | str]]:
    dataset = protocol["dataset"]
    frames = [
        {
            "scene_id": int(scene_id),
            "image_id": int(image_id),
            "frame_id": f"s{scene_id:06d}-i{image_id:06d}",
        }
        for scene_id in dataset["scene_ids"]
        for image_id in dataset["image_ids_per_scene"]
    ]
    if len(frames) != 25 or len({row["frame_id"] for row in frames}) != 25:
        raise ContractError("A-R8 frame plan is not the frozen 25-frame product")
    return frames


def _sensor_paths(
    dataset_root: Path, scene_id: int, image_id: int
) -> tuple[Path, Path]:
    scene = dataset_root / "val" / f"{scene_id:06d}"
    rgb = scene / "rgb_realsense" / f"{image_id:06d}.png"
    depth = scene / "depth_realsense" / f"{image_id:06d}.png"
    if not rgb.is_file() or not depth.is_file():
        raise ContractError(f"A-R8 sensor assets are missing for {scene_id}/{image_id}")
    return rgb, depth


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    locations = np.argwhere(mask)
    if not len(locations):
        raise ContractError("A-R8 candidate mask is empty")
    y_min, x_min = locations.min(axis=0)
    y_max, x_max = locations.max(axis=0)
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


def _seed(mask: np.ndarray) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ContractError("A-R8 OpenCV dependency is unavailable") from exc
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    maximum = float(distances.max())
    if maximum <= 0:
        y, x = np.argwhere(mask)[0]
        return int(x), int(y)
    y, x = min((int(y), int(x)) for y, x in np.argwhere(distances == maximum))
    return x, y


def _robust_depth_boundary(support: np.ndarray, depth: np.ndarray) -> np.ndarray:
    valid = support & (depth > 0)
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    horizontal_raw = np.abs(
        depth[:, 1:].astype(np.float64) - depth[:, :-1].astype(np.float64)
    )
    vertical_raw = np.abs(
        depth[1:, :].astype(np.float64) - depth[:-1, :].astype(np.float64)
    )
    differences = np.concatenate(
        [horizontal_raw[horizontal_valid], vertical_raw[vertical_valid]]
    )
    noise = 0.0 if not differences.size else float(np.median(differences))
    threshold = max(1.0, DEPTH_EDGE_NOISE_MULTIPLIER * noise)
    boundary = np.zeros(support.shape, dtype=bool)
    horizontal_edges = horizontal_valid & (horizontal_raw > threshold)
    vertical_edges = vertical_valid & (vertical_raw > threshold)
    boundary[:, 1:] |= horizontal_edges
    boundary[:, :-1] |= horizontal_edges
    boundary[1:, :] |= vertical_edges
    boundary[:-1, :] |= vertical_edges
    return boundary


def geometry_instances(
    support_mask: np.ndarray,
    depth: np.ndarray,
    *,
    source_masks: Sequence[np.ndarray] = (),
    source_scores: Sequence[float] = (),
    minimum_pixels: int = 9,
) -> list[Candidate]:
    """Split sensor support using robust depth edges and marker watershed."""

    support = np.asarray(support_mask, dtype=bool)
    values = np.asarray(depth)
    if support.ndim != 2 or not support.any():
        return []
    if values.shape != support.shape or not np.issubdtype(values.dtype, np.number):
        raise ContractError("A-R8 geometry depth/support shapes differ")
    if len(source_masks) != len(source_scores):
        raise ContractError("A-R8 geometry source mask/score coverage differs")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ContractError("A-R8 OpenCV dependency is unavailable") from exc

    valid = support & (values > 0)
    if not valid.any():
        return []
    boundary = _robust_depth_boundary(support, values)
    interior = (valid & ~boundary).astype(np.uint8)
    component_count, component_labels = cv2.connectedComponents(
        interior, connectivity=8
    )
    components: list[np.ndarray] = []
    for component_id in range(1, component_count):
        component = component_labels == component_id
        if int(component.sum()) >= minimum_pixels:
            components.append(component)
    if not components:
        return []

    markers = np.zeros(support.shape, dtype=np.int32)
    markers[~support] = 1
    marker_ids: list[int] = []
    for offset, component in enumerate(components, start=2):
        markers[component] = offset
        marker_ids.append(offset)
    positive_depth = values[valid].astype(np.float64)
    low = float(positive_depth.min())
    high = float(positive_depth.max())
    if high > low:
        normalized = np.clip((values.astype(np.float64) - low) / (high - low), 0, 1)
    else:
        normalized = np.zeros(values.shape, dtype=np.float64)
    depth_u8 = np.rint(normalized * 255.0).astype(np.uint8)
    watershed_image = np.repeat(depth_u8[:, :, None], 3, axis=2)
    resolved = cv2.watershed(watershed_image, markers.copy())

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for marker_id in marker_ids:
        mask = (resolved == marker_id) & support
        if int(mask.sum()) < minimum_pixels:
            continue
        digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        seed_xy = _seed(mask)
        confidence = 0.0
        for source_mask, source_score in zip(source_masks, source_scores, strict=True):
            x, y = seed_xy
            if np.asarray(source_mask, dtype=bool)[y, x]:
                confidence = max(confidence, float(source_score))
        candidates.append(
            Candidate(
                mask=mask,
                score=confidence,
                source="raw_depth_watershed",
                mask_stability=confidence,
                seed_xy=seed_xy,
            )
        )
    return sorted(
        candidates,
        key=lambda row: (
            row.seed_xy[1] if row.seed_xy else -1,
            row.seed_xy[0] if row.seed_xy else -1,
        ),
    )


def _sam2_refine(
    predictor: Any,
    image: np.ndarray,
    support: np.ndarray,
    geometry: Sequence[Candidate],
) -> list[Candidate]:
    if not geometry:
        return []
    predictor.set_image(image)
    seeds = [candidate.seed_xy for candidate in geometry]
    if any(seed_xy is None for seed_xy in seeds):
        raise ContractError("A-R8 geometry candidate lacks a SAM2 seed")
    concrete_seeds = [seed_xy for seed_xy in seeds if seed_xy is not None]
    choices: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for index, candidate in enumerate(geometry):
        positive = concrete_seeds[index]
        rivals = [seed_xy for i, seed_xy in enumerate(concrete_seeds) if i != index]
        coordinates = np.asarray([positive, *rivals], dtype=np.float32)
        labels = np.asarray([1, *([0] * len(rivals))], dtype=np.int32)
        result = predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            box=np.asarray(_mask_bbox(candidate.mask), dtype=np.float32),
            multimask_output=True,
            return_logits=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise ContractError("A-R8 SAM2 predictor returned an invalid result")
        logits, scores, low_resolution = (np.asarray(value) for value in result)
        if (
            logits.ndim != 3
            or scores.ndim != 1
            or low_resolution.ndim != 3
            or len(logits) != len(scores)
            or len(logits) != len(low_resolution)
            or not np.isfinite(logits).all()
            or not np.isfinite(scores).all()
        ):
            raise ContractError("A-R8 SAM2 output shapes changed")
        valid: list[tuple[float, int, np.ndarray]] = []
        px, py = positive
        for mask_index, raw_logits in enumerate(logits):
            mask = (raw_logits > 0) & support
            if not mask[py, px] or any(mask[y, x] for x, y in rivals):
                continue
            score = float(scores[mask_index])
            if not 0.0 <= score <= 1.0:
                raise ContractError("A-R8 SAM2 predicted IoU is outside [0,1]")
            valid.append((score, mask_index, raw_logits.astype(np.float32)))
        if valid:
            score, _, raw_logits = sorted(valid, key=lambda row: (-row[0], row[1]))[0]
            mask = (raw_logits > 0) & support
            choices.append((mask, raw_logits, score, "SAM2_REFINED"))
        else:
            fallback_logits = np.full(support.shape, -np.inf, dtype=np.float32)
            fallback_logits[candidate.mask] = 0.0
            choices.append(
                (
                    candidate.mask.copy(),
                    fallback_logits,
                    candidate.score,
                    "GEOMETRY_FALLBACK",
                )
            )

    stack = np.stack([row[1] for row in choices], axis=0)
    valid_stack = np.stack([row[0] for row in choices], axis=0)
    stack[~valid_stack] = -np.inf
    owner = np.argmax(stack, axis=0)
    any_valid = valid_stack.any(axis=0)
    output: list[Candidate] = []
    for index, (raw_mask, _logits, score, status) in enumerate(choices):
        mask = raw_mask & any_valid & (owner == index)
        if not mask.any():
            mask = geometry[index].mask.copy()
            status = "GEOMETRY_FALLBACK_EMPTY_REFINED_MASK"
            score = geometry[index].score
        output.append(
            Candidate(
                mask=mask,
                score=float(score),
                source="geometry_plus_sam2",
                mask_stability=float(score),
                seed_xy=geometry[index].seed_xy,
                sam2_status=status,
            )
        )
    return output


def _score_in_chunks(
    runtime: SyntheticGateRuntime,
    image: np.ndarray,
    candidates: Sequence[Candidate],
    *,
    chunk_size: int = 8,
) -> list[Candidate]:
    output: list[Candidate] = []
    for start in range(0, len(candidates), chunk_size):
        chunk = list(candidates[start : start + chunk_size])
        scored = runtime.score_masks(
            image,
            [row.mask for row in chunk],
            proposal_scores=[row.score for row in chunk],
            stabilities=[row.mask_stability for row in chunk],
            proposal_indices=list(range(start, start + len(chunk))),
        )
        for original, result in zip(chunk, scored, strict=True):
            ranking = tuple(dict(row) for row in result.cad_ranking)
            output.append(
                Candidate(
                    mask=original.mask,
                    score=original.score,
                    source=original.source,
                    mask_stability=original.mask_stability,
                    seed_xy=original.seed_xy,
                    proposal_index=original.proposal_index,
                    predicted_object_id=int(ranking[0]["object_id"]),
                    cad_similarity=float(ranking[0]["normalized_similarity"]),
                    cad_ranking=ranking,
                    sam2_status=original.sam2_status,
                )
            )
    return output


def _candidate_metadata(candidate: Candidate) -> dict[str, Any]:
    return {
        "score": float(candidate.score),
        "mask_stability": float(candidate.mask_stability),
        "source": candidate.source,
        "seed_xy": None if candidate.seed_xy is None else list(candidate.seed_xy),
        "proposal_index": candidate.proposal_index,
        "mask_pixels": int(candidate.mask.sum()),
        "bbox_xyxy_half_open": list(_mask_bbox(candidate.mask)),
        "predicted_object_id": candidate.predicted_object_id,
        "cad_similarity": candidate.cad_similarity,
        "cad_ranking": list(candidate.cad_ranking),
        "sam2_status": candidate.sam2_status,
    }


def _pack_masks(path: Path, variants: Mapping[str, Sequence[Candidate]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    shape = next(
        (
            candidate.mask.shape
            for candidates in variants.values()
            for candidate in candidates
        ),
        None,
    )
    if shape is None:
        raise ContractError("A-R8 frame produced no masks in any variant")
    for name, candidates in variants.items():
        if candidates:
            current_shape = candidates[0].mask.shape
            if any(candidate.mask.shape != current_shape for candidate in candidates):
                raise ContractError("A-R8 candidate mask shapes differ")
            if shape != current_shape:
                raise ContractError("A-R8 variant mask shapes differ")
            stack = np.stack([candidate.mask for candidate in candidates], axis=0)
        else:
            stack = np.zeros((0, *shape), dtype=bool)
        flat_pixels = int(shape[0]) * int(shape[1])
        arrays[f"{name}_packed"] = np.packbits(
            stack.reshape((len(stack), flat_pixels)), axis=1
        )
        arrays[f"{name}_count"] = np.asarray([len(stack)], dtype=np.int32)
    arrays["height_width"] = np.asarray(shape, dtype=np.int32)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _unpack_masks(path: Path, variant: str) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        height, width = (int(value) for value in bundle["height_width"])
        count = int(bundle[f"{variant}_count"][0])
        packed = bundle[f"{variant}_packed"]
        if count == 0:
            return []
        values = np.unpackbits(packed, axis=1, count=height * width)
        return [row.reshape(height, width).astype(bool) for row in values[:count]]


def _prepare_model_paths(sam2_root: Path) -> None:
    for path in (sam2_root / "python_deps", sam2_root / "source"):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def _native_mask_statistics(mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        raise ContractError("A-R8 native A-R6 mask must be non-empty HxW")
    x0, y0, x1, y1 = _mask_bbox(values)
    pixels = int(values.sum())
    return {
        "mask_pixels": pixels,
        "coverage": pixels / values.size,
        "bbox_xyxy_half_open": (x0, y0, x1, y1),
    }


def _native_contained_children(candidates: Sequence[Candidate], index: int) -> int:
    parent = candidates[index]
    parent_stats = _native_mask_statistics(parent.mask)
    parent_area = int(parent_stats["mask_pixels"])
    px0, py0, px1, py1 = parent_stats["bbox_xyxy_half_open"]
    tolerance = int(POLICY["container_bbox_tolerance_px"])
    count = 0
    for child_index, child in enumerate(candidates):
        if child_index == index:
            continue
        child_stats = _native_mask_statistics(child.mask)
        child_area = int(child_stats["mask_pixels"])
        if child_area > parent_area * POLICY["container_child_max_area_ratio"]:
            continue
        cx0, cy0, cx1, cy1 = child_stats["bbox_xyxy_half_open"]
        if (
            cx0 < px0 - tolerance
            or cy0 < py0 - tolerance
            or cx1 > px1 + tolerance
            or cy1 > py1 + tolerance
        ):
            continue
        overlap = int(np.logical_and(parent.mask, child.mask).sum())
        if overlap / max(1, child_area) >= POLICY["container_containment_threshold"]:
            count += 1
    return count


def _native_a_r6_decision(
    frame_id: str, candidates: Sequence[Candidate]
) -> tuple[Candidate | None, dict[str, Any]]:
    """Replay the frozen A-R6 policy at native sensor HxW."""

    audits: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if len(candidate.cad_ranking) < 2 or candidate.proposal_index is None:
            raise ContractError("A-R8 native A-R6 candidate lacks CAD evidence")
        statistics = _native_mask_statistics(candidate.mask)
        height, width = candidate.mask.shape
        x0, y0, x1, y1 = statistics["bbox_xyxy_half_open"]
        bbox_fraction = ((x1 - x0) * (y1 - y0)) / (width * height)
        bbox_fill = statistics["coverage"] / max(bbox_fraction, 1e-12)
        normalized_span = max((x1 - x0) / width, (y1 - y0) / height)
        edge = int(POLICY["frame_edge_margin_px"])
        touches_edge = (
            x0 <= edge or y0 <= edge or x1 >= width - edge or y1 >= height - edge
        )
        children = _native_contained_children(candidates, index)
        similarity = float(candidate.cad_ranking[0]["normalized_similarity"])
        margin = similarity - float(candidate.cad_ranking[1]["normalized_similarity"])
        reasons: list[str] = []
        if statistics["coverage"] > POLICY["maximum_mask_coverage"]:
            reasons.append("MAX_COVERAGE")
        if touches_edge:
            reasons.append("FRAME_EDGE_TOUCH")
        if (
            statistics["coverage"] >= POLICY["container_minimum_coverage"]
            and children >= POLICY["container_minimum_contained_children"]
        ):
            reasons.append("CONTAINS_MULTIPLE_CHILD_PROPOSALS")
        if (
            statistics["coverage"] >= POLICY["container_minimum_coverage"]
            and bbox_fill >= POLICY["rectangular_container_fill_threshold"]
            and normalized_span
            >= POLICY["rectangular_container_normalized_span_threshold"]
        ):
            reasons.append("RECTANGULAR_CONTAINER_SPAN")
        if candidate.score < POLICY["minimum_selection_proposal_score"]:
            reasons.append("LOW_PROPOSAL_SCORE")
        reasons = [reason for reason in FILTER_REASON_ORDER if reason in reasons]
        weights = POLICY["calibration_weights"]
        calibrated = (
            weights["cad_similarity"] * similarity
            + weights["proposal_score"] * candidate.score
            + weights["cad_margin"] * margin
            + weights["mask_stability"] * candidate.mask_stability
        )
        audits.append(
            {
                "proposal_index": candidate.proposal_index,
                "selected_object_id": int(candidate.cad_ranking[0]["object_id"]),
                "coverage": statistics["coverage"],
                "bbox_fill_ratio": bbox_fill,
                "maximum_normalized_bbox_span": normalized_span,
                "touches_frame_edge": touches_edge,
                "contained_child_count": children,
                "proposal_score": candidate.score,
                "mask_stability": candidate.mask_stability,
                "cad_similarity": similarity,
                "cad_margin": margin,
                "calibrated_score": calibrated,
                "filter_reasons": reasons,
                "eligible": not reasons,
            }
        )
    eligible = [row for row in audits if row["eligible"]]
    best = None
    if eligible:
        best = sorted(
            eligible,
            key=lambda row: (
                -row["calibrated_score"],
                -row["cad_similarity"],
                -row["proposal_score"],
                -row["mask_stability"],
                row["proposal_index"],
            ),
        )[0]
    if best is None:
        state = "NO_ELIGIBLE_PROPOSAL"
        selected_index = None
    elif best["cad_margin"] < POLICY["minimum_cad_margin_for_selection"]:
        state = "ABSTAIN_LOW_CAD_MARGIN"
        selected_index = None
    else:
        state = "SELECTED"
        selected_index = int(best["proposal_index"])
    selected = None
    if selected_index is not None:
        selected = next(
            row for row in candidates if row.proposal_index == selected_index
        )
    decision = {
        "item_id": frame_id,
        "native_frame_height_width": list(candidates[0].mask.shape)
        if candidates
        else None,
        "policy_source": "A-R6 b9e6e4b normalized policy at native sensor HxW",
        "source_proposal_count": len(candidates),
        "eligible_proposal_count": len(eligible),
        "rejected_proposal_count": len(candidates) - len(eligible),
        "best_candidate_proposal_index": None
        if best is None
        else best["proposal_index"],
        "selected_proposal_index": selected_index,
        "selected_object_id": None
        if state != "SELECTED"
        else best["selected_object_id"],
        "decision_state": state,
        "candidate_audits": audits,
    }
    return selected, decision


def produce(
    *,
    protocol_path: Path,
    dataset_root: Path,
    a_r5_root: Path,
    sam2_root: Path,
    output_root: Path,
    device: str,
    frame_limit: int | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if output_root.exists():
        raise ContractError("A-R8 producer output root is create-only")
    output_root.mkdir(parents=True)
    _prepare_model_paths(sam2_root)
    sam2_identity = verify_sam2_freeze(sam2_root)
    request, _template_manifest = load_a_r5_assets(a_r5_root)
    runtime = SyntheticGateRuntime(
        data_root=a_r5_root,
        sam2_asset_root=sam2_root,
        request=request,
        device=device,
    )
    frames = _planned_frames(protocol)
    mode = "FORMAL_25_FRAME_DEVELOPMENT"
    if frame_limit is not None:
        if frame_limit <= 0 or frame_limit >= len(frames):
            raise ContractError("A-R8 smoke frame limit must be between 1 and 24")
        frames = frames[:frame_limit]
        mode = "SMOKE_NOT_DECISION_ELIGIBLE"
    manifest_frames: list[dict[str, Any]] = []
    for frame in frames:
        scene_id = int(frame["scene_id"])
        image_id = int(frame["image_id"])
        frame_id = str(frame["frame_id"])
        rgb_path, depth_path = _sensor_paths(dataset_root, scene_id, image_id)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.ndim != 2
            or rgb.shape[:2] != depth.shape
        ):
            raise ContractError("A-R8 Realsense RGB/depth geometry differs")

        fastsam_rows = runtime.infer_fastsam(rgb)
        raw_candidates = [
            Candidate(
                mask=np.asarray(row["mask"], dtype=bool),
                score=float(row["proposal_score"]),
                source="frozen_fastsam",
                mask_stability=float(row["mask_stability"]),
                proposal_index=int(row["proposal_index"]),
            )
            for row in fastsam_rows
        ]
        scored_raw = _score_in_chunks(runtime, rgb, raw_candidates)
        selected, a_r6_decision = _native_a_r6_decision(frame_id, scored_raw)
        a_r6_candidates = [] if selected is None else [selected]

        audits = {
            int(row["proposal_index"]): row for row in a_r6_decision["candidate_audits"]
        }
        eligible = [
            row
            for row in scored_raw
            if bool(audits[int(row.proposal_index)]["eligible"])
        ]
        if eligible:
            support = np.logical_or.reduce([row.mask for row in eligible])
        else:
            support = np.zeros(depth.shape, dtype=bool)
        minimum_pixels = max(
            9,
            int(
                math.ceil(
                    runtime.cnos._config["postprocessing"]["min_mask_area_relative"]
                    * depth.size
                )
            ),
        )
        geometry = geometry_instances(
            support,
            depth,
            source_masks=[row.mask for row in eligible],
            source_scores=[row.score for row in eligible],
            minimum_pixels=minimum_pixels,
        )
        sam2 = _sam2_refine(runtime.sam2_predictor, rgb, support, geometry)
        scored_sam2 = _score_in_chunks(runtime, rgb, sam2)
        variants: dict[str, Sequence[Candidate]] = {
            "a_r6_baseline": a_r6_candidates,
            "raw_fastsam_eligible": eligible,
            "geometry_instances": geometry,
            "geometry_plus_sam2": scored_sam2,
        }
        mask_path = output_root / "frames" / f"{frame_id}.npz"
        metadata_path = output_root / "frames" / f"{frame_id}.json"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        _pack_masks(mask_path, variants)
        metadata = {
            "frame_id": frame_id,
            "scene_id": scene_id,
            "image_id": image_id,
            "rgb": {
                "relative_path": rgb_path.relative_to(dataset_root).as_posix(),
                "bytes": rgb_path.stat().st_size,
                "sha256": _sha256_file(rgb_path),
            },
            "depth": {
                "relative_path": depth_path.relative_to(dataset_root).as_posix(),
                "bytes": depth_path.stat().st_size,
                "sha256": _sha256_file(depth_path),
            },
            "gt_access_count": 0,
            "a_r6_decision": a_r6_decision,
            "support_pixels": int(support.sum()),
            "minimum_geometry_pixels": minimum_pixels,
            "variants": {
                name: [_candidate_metadata(row) for row in candidates]
                for name, candidates in variants.items()
            },
        }
        _write_json(metadata_path, metadata)
        manifest_frames.append(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "image_id": image_id,
                "mask_bundle": {
                    "relative_path": mask_path.relative_to(output_root).as_posix(),
                    "bytes": mask_path.stat().st_size,
                    "sha256": _sha256_file(mask_path),
                },
                "metadata": {
                    "relative_path": metadata_path.relative_to(output_root).as_posix(),
                    "bytes": metadata_path.stat().st_size,
                    "sha256": _sha256_file(metadata_path),
                },
            }
        )
        print(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "fastsam": len(raw_candidates),
                    "eligible": len(eligible),
                    "geometry": len(geometry),
                    "sam2": len(scored_sam2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = {
        "schema_version": "poseloop.a-r8.prediction-manifest.v1",
        "mode": mode,
        "protocol_sha256": _sha256_file(protocol_path),
        "frame_count": len(manifest_frames),
        "frames": manifest_frames,
        "sam2_identity": sam2_identity,
        "labels_opened": False,
        "gt_access_count": 0,
        "scene9_read_count": 0,
    }
    manifest_path = output_root / "prediction-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _iou_matrix(
    ground_truth: Sequence[np.ndarray], predicted: Sequence[np.ndarray]
) -> np.ndarray:
    matrix = np.zeros((len(ground_truth), len(predicted)), dtype=np.float64)
    for gt_index, gt_mask in enumerate(ground_truth):
        for pred_index, pred_mask in enumerate(predicted):
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            matrix[gt_index, pred_index] = 0.0 if union == 0 else intersection / union
    return matrix


def _greedy_matches(
    matrix: np.ndarray, threshold: float
) -> list[tuple[int, int, float]]:
    pairs = [
        (float(matrix[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(matrix.shape[0])
        for pred_index in range(matrix.shape[1])
        if float(matrix[gt_index, pred_index]) >= threshold
    ]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in sorted(
        pairs, key=lambda row: (-row[0], row[1], row[2])
    ):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def _frame_metrics(
    ground_truth: Sequence[np.ndarray], predicted: Sequence[np.ndarray]
) -> dict[str, float | int]:
    matrix = _iou_matrix(ground_truth, predicted)
    matches50 = _greedy_matches(matrix, 0.50)
    matches75 = _greedy_matches(matrix, 0.75)
    tp = len(matches50)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = 0.0 if not predicted else tp / len(predicted)
    recall = 0.0 if not ground_truth else tp / len(ground_truth)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    pq_denominator = tp + 0.5 * fp + 0.5 * fn
    pq = (
        0.0
        if pq_denominator == 0
        else sum(row[2] for row in matches50) / pq_denominator
    )
    merge_count = 0
    for pred_mask in predicted:
        covered = sum(
            int(np.logical_and(pred_mask, gt_mask).sum()) / max(1, int(gt_mask.sum()))
            >= 0.5
            for gt_mask in ground_truth
        )
        merge_count += int(covered >= 2)
    split_count = 0
    for gt_mask in ground_truth:
        fragments = sum(
            int(np.logical_and(pred_mask, gt_mask).sum()) / max(1, int(pred_mask.sum()))
            >= 0.5
            for pred_mask in predicted
        )
        split_count += int(fragments >= 2)
    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predicted),
        "tp_iou50": tp,
        "tp_iou75": len(matches75),
        "fp_iou50": fp,
        "fn_iou50": fn,
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "pq_iou50": pq,
        "merge_count": merge_count,
        "split_count": split_count,
    }


def _average_precision(frames: Sequence[Mapping[str, Any]], threshold: float) -> float:
    total_gt = sum(len(frame["gt_masks"]) for frame in frames)
    records: list[tuple[float, str, int, np.ndarray]] = []
    gt_by_frame: dict[str, Sequence[np.ndarray]] = {}
    for frame in frames:
        frame_id = str(frame["frame_id"])
        gt_by_frame[frame_id] = frame["gt_masks"]
        for index, (score, mask) in enumerate(
            zip(frame["scores"], frame["pred_masks"], strict=True)
        ):
            records.append((float(score), frame_id, index, mask))
    records.sort(key=lambda row: (-row[0], row[1], row[2]))
    used: dict[str, set[int]] = {frame_id: set() for frame_id in gt_by_frame}
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _score, frame_id, _index, mask in records:
        ground_truth = gt_by_frame[frame_id]
        best: tuple[float, int] | None = None
        for gt_index, gt_mask in enumerate(ground_truth):
            if gt_index in used[frame_id]:
                continue
            intersection = int(np.logical_and(mask, gt_mask).sum())
            union = int(np.logical_or(mask, gt_mask).sum())
            iou = 0.0 if union == 0 else intersection / union
            if best is None or iou > best[0]:
                best = (iou, gt_index)
        if best is not None and best[0] >= threshold:
            used[frame_id].add(best[1])
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    if total_gt == 0 or not records:
        return 0.0
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def evaluate_predictions(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_frame = []
    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    total_merge = total_split = 0
    pq_numerator = 0.0
    for frame in frames:
        metrics = _frame_metrics(frame["gt_masks"], frame["pred_masks"])
        per_frame.append({"frame_id": frame["frame_id"], **metrics})
        total_tp += int(metrics["tp_iou50"])
        total_fp += int(metrics["fp_iou50"])
        total_fn += int(metrics["fn_iou50"])
        total_gt += int(metrics["ground_truth_count"])
        total_pred += int(metrics["prediction_count"])
        total_merge += int(metrics["merge_count"])
        total_split += int(metrics["split_count"])
        matrix = _iou_matrix(frame["gt_masks"], frame["pred_masks"])
        pq_numerator += sum(row[2] for row in _greedy_matches(matrix, 0.50))
    precision = 0.0 if total_pred == 0 else total_tp / total_pred
    recall = 0.0 if total_gt == 0 else total_tp / total_gt
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    pq_denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    return {
        "frame_count": len(frames),
        "ground_truth_count": total_gt,
        "prediction_count": total_pred,
        "tp_iou50": total_tp,
        "fp_iou50": total_fp,
        "fn_iou50": total_fn,
        "instance_precision_iou50": precision,
        "instance_recall_iou50": recall,
        "instance_f1_iou50": f1,
        "ap50": _average_precision(frames, 0.50),
        "ap75": _average_precision(frames, 0.75),
        "pq_iou50": 0.0 if pq_denominator == 0 else pq_numerator / pq_denominator,
        "merge_count": total_merge,
        "split_count": total_split,
        "per_frame": per_frame,
    }


def _binary_f1(labels: Sequence[bool], selected: Sequence[bool]) -> float:
    tp = sum(label and keep for label, keep in zip(labels, selected, strict=True))
    fp = sum((not label) and keep for label, keep in zip(labels, selected, strict=True))
    fn = sum(label and (not keep) for label, keep in zip(labels, selected, strict=True))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _choose_cad_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if not scores:
        return 1.0
    if not any(labels):
        return math.nextafter(max(float(score) for score in scores), math.inf)
    thresholds = [
        math.inf,
        *sorted(set(float(score) for score in scores), reverse=True),
    ]
    best = max(
        (
            _binary_f1(labels, [score >= threshold for score in scores]),
            threshold,
        )
        for threshold in thresholds
    )
    return float(best[1])


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def _bootstrap_interval(
    values: Sequence[float], *, seed: int, draws: int
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.integers(0, len(array), size=len(array))
        means[index] = float(array[sample].mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired_delta(
    left: Mapping[str, Any], right: Mapping[str, Any], *, seed: int, draws: int
) -> dict[str, Any]:
    left_rows = {row["frame_id"]: row for row in left["per_frame"]}
    right_rows = {row["frame_id"]: row for row in right["per_frame"]}
    if set(left_rows) != set(right_rows):
        raise ContractError("A-R8 paired metric frame coverage differs")
    deltas = [
        float(right_rows[frame_id]["f1_iou50"]) - float(left_rows[frame_id]["f1_iou50"])
        for frame_id in sorted(left_rows)
    ]
    return {
        "mean_frame_f1_delta": float(np.mean(deltas)),
        "bootstrap_95_interval": _bootstrap_interval(deltas, seed=seed, draws=draws),
        "positive_frame_count": sum(delta > 0 for delta in deltas),
    }


def _positive_scene_count(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_rows = {row["frame_id"]: row for row in left["per_frame"]}
    right_rows = {row["frame_id"]: row for row in right["per_frame"]}
    count = 0
    for scene_id in EXPECTED_SCENES:
        prefix = f"s{scene_id:06d}-"
        frame_ids = [frame_id for frame_id in left_rows if frame_id.startswith(prefix)]
        left_mean = np.mean([left_rows[frame_id]["f1_iou50"] for frame_id in frame_ids])
        right_mean = np.mean(
            [right_rows[frame_id]["f1_iou50"] for frame_id in frame_ids]
        )
        count += int(right_mean > left_mean)
    return count


def make_decisions(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    cad_auc: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    geometry_delta = _paired_delta(
        metrics["a_r6_baseline"],
        metrics["geometry_instances"],
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    sam2_delta = _paired_delta(
        metrics["geometry_instances"],
        metrics["geometry_plus_sam2"],
        seed=bootstrap_seed + 1,
        draws=bootstrap_draws,
    )
    cad_delta = _paired_delta(
        metrics["geometry_plus_sam2"],
        metrics["geometry_plus_sam2_plus_causal_cad"],
        seed=bootstrap_seed + 2,
        draws=bootstrap_draws,
    )
    sam2_scene_count = _positive_scene_count(
        metrics["geometry_instances"], metrics["geometry_plus_sam2"]
    )
    cad_scene_count = _positive_scene_count(
        metrics["geometry_plus_sam2"],
        metrics["geometry_plus_sam2_plus_causal_cad"],
    )
    sam2_before = metrics["geometry_instances"]
    sam2_after = metrics["geometry_plus_sam2"]
    retain_sam2 = bool(
        sam2_delta["bootstrap_95_interval"][0] > 0
        and sam2_scene_count >= 4
        and sam2_after["ap50"] >= sam2_before["ap50"]
        and sam2_after["pq_iou50"] >= sam2_before["pq_iou50"]
    )
    cad_before = metrics["geometry_plus_sam2"]
    cad_after = metrics["geometry_plus_sam2_plus_causal_cad"]
    retain_cad = bool(
        cad_auc["bootstrap_95_interval"][0] > 0.5
        and cad_delta["bootstrap_95_interval"][0] > 0
        and cad_scene_count >= 4
        and cad_after["ap50"] >= cad_before["ap50"]
        and cad_after["pq_iou50"] >= cad_before["pq_iou50"]
    )
    chosen = "geometry_instances"
    if retain_sam2:
        chosen = "geometry_plus_sam2"
    if retain_sam2 and retain_cad:
        chosen = "geometry_plus_sam2_plus_causal_cad"
    chosen_metrics = metrics[chosen]
    versus_baseline = _paired_delta(
        metrics["a_r6_baseline"],
        chosen_metrics,
        seed=bootstrap_seed + 3,
        draws=bootstrap_draws,
    )
    versus_raw = _paired_delta(
        metrics["raw_fastsam_eligible"],
        chosen_metrics,
        seed=bootstrap_seed + 4,
        draws=bootstrap_draws,
    )
    route_positive = bool(
        versus_baseline["bootstrap_95_interval"][0] > 0
        and versus_raw["bootstrap_95_interval"][0] > 0
        and chosen_metrics["ap50"] >= metrics["raw_fastsam_eligible"]["ap50"]
        and chosen_metrics["pq_iou50"] >= metrics["raw_fastsam_eligible"]["pq_iou50"]
    )
    return {
        "sam2": {
            "decision": "RETAIN_SAM2" if retain_sam2 else "DROP_SAM2",
            "positive_scene_count": sam2_scene_count,
            "paired_delta": sam2_delta,
        },
        "cad": {
            "decision": "RETAIN_CAD" if retain_cad else "DROP_CAD",
            "positive_scene_count": cad_scene_count,
            "paired_delta": cad_delta,
            "discrimination": dict(cad_auc),
        },
        "geometry_vs_a_r6": geometry_delta,
        "selected_successor": chosen,
        "selected_vs_a_r6": versus_baseline,
        "selected_vs_raw_fastsam": versus_raw,
        "route_decision": (
            "REAL_DEVELOPMENT_ROUTE_POSITIVE_BUILD_PHYSICAL_SYNTHETIC_NEXT"
            if route_positive
            else "ABANDON_ROUTE_SWITCH_INSTANCE_DETECTOR_OR_3D_PROPOSALS"
        ),
        "scene9_replay_permitted": False,
    }


def _load_gt(
    dataset_root: Path, scene_id: int, image_id: int
) -> tuple[list[np.ndarray], list[int]]:
    scene = dataset_root / "val" / f"{scene_id:06d}"
    gt_path = scene / "scene_gt_realsense.json"
    if not gt_path.is_file():
        raise ContractError("A-R8 consumed development GT is missing")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    rows = gt.get(str(image_id))
    if not isinstance(rows, list) or not rows:
        raise ContractError("A-R8 consumed development GT row is missing")
    masks: list[np.ndarray] = []
    object_ids: list[int] = []
    for gt_index, row in enumerate(rows):
        mask_path = (
            scene / "mask_visib_realsense" / f"{image_id:06d}_{gt_index:06d}.png"
        )
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if not mask.any():
            continue
        masks.append(mask)
        object_ids.append(int(row["obj_id"]))
    return masks, object_ids


def _cad_candidate_labels(
    gt_masks: Sequence[np.ndarray],
    gt_object_ids: Sequence[int],
    pred_masks: Sequence[np.ndarray],
    pred_object_ids: Sequence[int],
) -> list[bool]:
    labels: list[bool] = []
    for pred_mask, pred_object_id in zip(pred_masks, pred_object_ids, strict=True):
        best = 0.0
        for gt_mask, gt_object_id in zip(gt_masks, gt_object_ids, strict=True):
            if pred_object_id != gt_object_id:
                continue
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            union = int(np.logical_or(pred_mask, gt_mask).sum())
            best = max(best, 0.0 if union == 0 else intersection / union)
        labels.append(best >= 0.50)
    return labels


def _outline(
    image: np.ndarray, masks: Sequence[np.ndarray], color: tuple[int, int, int]
) -> np.ndarray:
    result = image.copy()
    for mask in masks:
        padded = np.pad(mask, 1)
        eroded = np.logical_and.reduce(
            [
                padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
        )
        contour = mask & ~eroded
        result[contour] = color
    return result


def _write_visual(
    path: Path,
    rgb: np.ndarray,
    gt_masks: Sequence[np.ndarray],
    variants: Mapping[str, Sequence[np.ndarray]],
) -> None:
    panels = [
        ("ground truth", _outline(rgb, gt_masks, (0, 255, 0))),
        ("A-R6", _outline(rgb, variants["a_r6_baseline"], (255, 80, 80))),
        ("geometry", _outline(rgb, variants["geometry_instances"], (80, 160, 255))),
        (
            "geometry + SAM2",
            _outline(rgb, variants["geometry_plus_sam2"], (255, 200, 0)),
        ),
        (
            "geometry + SAM2 + CAD",
            _outline(
                rgb,
                variants["geometry_plus_sam2_plus_causal_cad"],
                (255, 0, 255),
            ),
        ),
    ]
    width, height = rgb.shape[1], rgb.shape[0]
    canvas = Image.new("RGB", (width * 3, height * 2), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = (index % 3) * width
        y = (index // 3) * height
        canvas.paste(Image.fromarray(panel), (x, y))
        draw.rectangle((x, y, x + 420, y + 34), fill=(0, 0, 0))
        draw.text((x + 8, y + 8), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(
    *,
    protocol_path: Path,
    dataset_root: Path,
    predictions_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if output_root.exists():
        raise ContractError("A-R8 evaluation output root is create-only")
    manifest_path = predictions_root / "prediction-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != "FORMAL_25_FRAME_DEVELOPMENT"
        or manifest.get("frame_count") != 25
        or manifest.get("labels_opened") is not False
        or manifest.get("gt_access_count") != 0
        or manifest.get("scene9_read_count") != 0
    ):
        raise ContractError("A-R8 prediction manifest is not formal label-blind output")
    expected = _planned_frames(protocol)
    observed = manifest.get("frames")
    if not isinstance(observed, list) or [row["frame_id"] for row in observed] != [
        row["frame_id"] for row in expected
    ]:
        raise ContractError("A-R8 prediction frame coverage changed")
    output_root.mkdir(parents=True)
    frame_records: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for frame in observed:
        frame_id = str(frame["frame_id"])
        scene_id = int(frame["scene_id"])
        image_id = int(frame["image_id"])
        metadata_path = predictions_root / frame["metadata"]["relative_path"]
        masks_path = predictions_root / frame["mask_bundle"]["relative_path"]
        if (
            _sha256_file(metadata_path) != frame["metadata"]["sha256"]
            or _sha256_file(masks_path) != frame["mask_bundle"]["sha256"]
        ):
            raise ContractError("A-R8 prediction artifact changed before evaluation")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gt_masks, gt_object_ids = _load_gt(dataset_root, scene_id, image_id)
        variants: dict[str, list[np.ndarray]] = {
            name: _unpack_masks(masks_path, name) for name in VARIANTS
        }
        scores: dict[str, list[float]] = {
            name: [float(row["score"]) for row in metadata["variants"][name]]
            for name in VARIANTS
        }
        sam2_metadata = metadata["variants"]["geometry_plus_sam2"]
        pred_object_ids = [int(row["predicted_object_id"]) for row in sam2_metadata]
        cad_scores = [float(row["cad_similarity"]) for row in sam2_metadata]
        labels = _cad_candidate_labels(
            gt_masks,
            gt_object_ids,
            variants["geometry_plus_sam2"],
            pred_object_ids,
        )
        for index, (score, label) in enumerate(zip(cad_scores, labels, strict=True)):
            candidate_rows.append(
                {
                    "frame_id": frame_id,
                    "scene_id": scene_id,
                    "candidate_index": index,
                    "score": score,
                    "label": label,
                }
            )
        rgb_path, _depth_path = _sensor_paths(dataset_root, scene_id, image_id)
        frame_records.append(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "image_id": image_id,
                "rgb_path": rgb_path,
                "gt_masks": gt_masks,
                "gt_object_ids": gt_object_ids,
                "variant_masks": variants,
                "variant_scores": scores,
                "cad_scores": cad_scores,
            }
        )

    thresholds: dict[int, float] = {}
    for holdout_scene in EXPECTED_SCENES:
        training = [row for row in candidate_rows if row["scene_id"] != holdout_scene]
        thresholds[holdout_scene] = _choose_cad_threshold(
            [float(row["score"]) for row in training],
            [bool(row["label"]) for row in training],
        )
    evaluation_frames: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            *VARIANTS,
            "geometry_plus_sam2_plus_causal_cad",
        )
    }
    for frame in frame_records:
        for name in VARIANTS:
            evaluation_frames[name].append(
                {
                    "frame_id": frame["frame_id"],
                    "gt_masks": frame["gt_masks"],
                    "pred_masks": frame["variant_masks"][name],
                    "scores": frame["variant_scores"][name],
                }
            )
        threshold = thresholds[int(frame["scene_id"])]
        keep = [score >= threshold for score in frame["cad_scores"]]
        cad_masks = [
            mask
            for mask, selected in zip(
                frame["variant_masks"]["geometry_plus_sam2"], keep, strict=True
            )
            if selected
        ]
        cad_scores = [
            score
            for score, selected in zip(frame["cad_scores"], keep, strict=True)
            if selected
        ]
        frame["variant_masks"]["geometry_plus_sam2_plus_causal_cad"] = cad_masks
        evaluation_frames["geometry_plus_sam2_plus_causal_cad"].append(
            {
                "frame_id": frame["frame_id"],
                "gt_masks": frame["gt_masks"],
                "pred_masks": cad_masks,
                "scores": cad_scores,
            }
        )
        with Image.open(frame["rgb_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        _write_visual(
            output_root / "visualizations" / f"{frame['frame_id']}.png",
            rgb,
            frame["gt_masks"],
            frame["variant_masks"],
        )
    metrics = {
        name: evaluate_predictions(frames) for name, frames in evaluation_frames.items()
    }
    scores = [float(row["score"]) for row in candidate_rows]
    labels = [bool(row["label"]) for row in candidate_rows]
    auc = _roc_auc(scores, labels)
    rng = np.random.default_rng(int(protocol["metrics"]["bootstrap_seed"]))
    auc_draws: list[float] = []
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_frame.setdefault(str(row["frame_id"]), []).append(row)
    frame_ids = sorted(by_frame)
    for _ in range(int(protocol["metrics"]["paired_bootstrap_frames"])):
        sampled = rng.choice(frame_ids, size=len(frame_ids), replace=True)
        rows = [row for frame_id in sampled for row in by_frame[str(frame_id)]]
        auc_draws.append(
            _roc_auc(
                [float(row["score"]) for row in rows],
                [bool(row["label"]) for row in rows],
            )
        )
    cad_auc = {
        "auroc": auc,
        "bootstrap_95_interval": [
            float(np.quantile(auc_draws, 0.025)),
            float(np.quantile(auc_draws, 0.975)),
        ],
        "positive_candidate_count": sum(labels),
        "negative_candidate_count": len(labels) - sum(labels),
        "leave_one_scene_out_thresholds": {
            f"{scene_id:06d}": threshold for scene_id, threshold in thresholds.items()
        },
    }
    decisions = make_decisions(
        metrics,
        cad_auc=cad_auc,
        bootstrap_seed=int(protocol["metrics"]["bootstrap_seed"]),
        bootstrap_draws=int(protocol["metrics"]["paired_bootstrap_frames"]),
    )
    result = {
        "schema_version": "poseloop.a-r8.real-causal-ablation-result.v1",
        "protocol_sha256": _sha256_file(protocol_path),
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
        "frame_count": 25,
        "labels_opened_during_inference": False,
        "labels_opened_during_evaluation": True,
        "scene9_read_count": 0,
        "metrics": metrics,
        "cad_discrimination": cad_auc,
        "decisions": decisions,
    }
    _write_json(output_root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("protocol-check")
    check.add_argument("--protocol", type=Path, required=True)

    producer = subparsers.add_parser("produce")
    producer.add_argument("--protocol", type=Path, required=True)
    producer.add_argument("--dataset-root", type=Path, required=True)
    producer.add_argument("--a-r5-root", type=Path, required=True)
    producer.add_argument("--sam2-root", type=Path, required=True)
    producer.add_argument("--output-root", type=Path, required=True)
    producer.add_argument("--device", default="cuda:0")
    producer.add_argument("--frame-limit", type=int)

    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--protocol", type=Path, required=True)
    evaluator.add_argument("--dataset-root", type=Path, required=True)
    evaluator.add_argument("--predictions-root", type=Path, required=True)
    evaluator.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "protocol-check":
        protocol = load_protocol(args.protocol)
        print(
            json.dumps(
                {
                    "status": "PASS_A_R8_PROTOCOL",
                    "protocol_id": protocol["protocol_id"],
                    "a_r7_status": protocol["a_r7_freeze"]["status"],
                    "frame_count": protocol["dataset"]["frame_count"],
                    "scene9_replay_permitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "produce":
        manifest = produce(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            a_r5_root=args.a_r5_root,
            sam2_root=args.sam2_root,
            output_root=args.output_root,
            device=args.device,
            frame_limit=args.frame_limit,
        )
        print(json.dumps({"status": "PRODUCED", "mode": manifest["mode"]}))
        return 0
    result = evaluate(
        protocol_path=args.protocol,
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        output_root=args.output_root,
    )
    print(json.dumps(result["decisions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
