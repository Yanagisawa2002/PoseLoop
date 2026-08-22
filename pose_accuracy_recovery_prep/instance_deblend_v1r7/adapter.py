"""Model-injected RGB-D prompt construction for A-R7.

No SAM2 package is imported here.  The reviewed server successor will bind the
official runtime and pass an already-initialized ``SAM2ImagePredictor``-like
object.  This keeps local tests deterministic and prevents PREP from silently
downloading weights.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pose_accuracy_recovery_prep.core import ContractError

from .core import DeblendCandidate, partition_prompted_candidates


# Six robust noise scales is fixed before replay and is only eligible for
# synthetic calibration.  It is not an A-R6 selection threshold.
DEPTH_EDGE_NOISE_MULTIPLIER = 6.0
MINIMUM_SEED_REGION_PIXELS = 9


def derive_depth_discontinuity_seeds(
    support_mask: np.ndarray, depth: np.ndarray
) -> tuple[tuple[int, int], ...]:
    """Find deterministic interior seeds separated by robust depth edges."""

    support = np.asarray(support_mask)
    values = np.asarray(depth)
    if support.ndim != 2 or support.dtype != np.bool_ or not support.any():
        raise ContractError("A-R7 seed support must be a non-empty boolean HxW mask")
    if values.shape != support.shape or values.ndim != 2:
        raise ContractError("A-R7 depth/support shapes differ")
    if not np.issubdtype(values.dtype, np.number):
        raise ContractError("A-R7 depth must be numeric")
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise ContractError("A-R7 depth contains non-finite values")
    valid = support & (values > 0)
    if not valid.any():
        raise ContractError("A-R7 support has no valid raw depth")

    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    horizontal_diff = np.abs(
        values[:, 1:].astype(np.float64) - values[:, :-1].astype(np.float64)
    )[horizontal_valid]
    vertical_diff = np.abs(
        values[1:, :].astype(np.float64) - values[:-1, :].astype(np.float64)
    )[vertical_valid]
    neighbor_differences = np.concatenate([horizontal_diff, vertical_diff])
    robust_noise = (
        0.0
        if neighbor_differences.size == 0
        else float(np.median(neighbor_differences))
    )
    edge_threshold = max(1.0, DEPTH_EDGE_NOISE_MULTIPLIER * robust_noise)

    boundary = np.zeros(support.shape, dtype=bool)
    horizontal_edge = np.zeros_like(horizontal_valid)
    vertical_edge = np.zeros_like(vertical_valid)
    if horizontal_diff.size:
        raw = np.abs(
            values[:, 1:].astype(np.float64) - values[:, :-1].astype(np.float64)
        )
        horizontal_edge = horizontal_valid & (raw > edge_threshold)
        boundary[:, 1:] |= horizontal_edge
        boundary[:, :-1] |= horizontal_edge
    if vertical_diff.size:
        raw = np.abs(
            values[1:, :].astype(np.float64) - values[:-1, :].astype(np.float64)
        )
        vertical_edge = vertical_valid & (raw > edge_threshold)
        boundary[1:, :] |= vertical_edge
        boundary[:-1, :] |= vertical_edge

    interior = (valid & ~boundary).astype(np.uint8)
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - remote preflight owns dependency
        raise ContractError("A-R7 OpenCV dependency is unavailable") from exc
    component_count, labels = cv2.connectedComponents(interior, connectivity=8)
    seeds: list[tuple[int, int]] = []
    for component in range(1, component_count):
        region = labels == component
        if int(region.sum()) < MINIMUM_SEED_REGION_PIXELS:
            continue
        distances = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        maximum = float(distances.max())
        locations = np.argwhere(distances == maximum)
        if not len(locations):
            raise ContractError("A-R7 depth component has no deterministic seed")
        y, x = min((int(y), int(x)) for y, x in locations)
        seeds.append((x, y))
    seeds = sorted(set(seeds), key=lambda point: (point[1], point[0]))
    if not seeds:
        raise ContractError("A-R7 depth support produced no seed region")
    return tuple(seeds)


def merge_seed_sources(
    depth_seeds: Sequence[tuple[int, int]],
    automatic_mask_seeds: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Union independent seed sources without a distance/NMS tuning knob."""

    merged: list[tuple[int, int]] = []
    for point in [*depth_seeds, *automatic_mask_seeds]:
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in point
            )
        ):
            raise ContractError("A-R7 seed sources must contain integer (x,y) tuples")
        if point not in merged:
            merged.append(point)
    if len(merged) < 2:
        raise ContractError("A-R7 needs at least two independent deblend seeds")
    return tuple(merged)


def _stability_from_logits(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("A-R7 SAM2 logits must be finite HxW")
    low = values > -1.0
    high = values > 1.0
    union = int(low.sum())
    return 0.0 if union == 0 else float(high.sum() / union)


def predict_prompt_exclusive_children(
    predictor: Any,
    seeds: Sequence[tuple[int, int]],
) -> list[DeblendCandidate]:
    """Prompt once per seed with every rival seed explicitly negative."""

    seed_tuple = tuple(seeds)
    if len(seed_tuple) < 2 or len(set(seed_tuple)) != len(seed_tuple):
        raise ContractError(
            "A-R7 prompted seeds must be unique and contain two or more"
        )
    candidates: list[DeblendCandidate] = []
    for positive_index, positive in enumerate(seed_tuple):
        ordered = [positive, *[seed for seed in seed_tuple if seed != positive]]
        coordinates = np.asarray(ordered, dtype=np.float32)
        labels = np.asarray([1, *([0] * (len(ordered) - 1))], dtype=np.int32)
        result = predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            multimask_output=True,
            return_logits=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise ContractError("A-R7 SAM2 predictor returned an invalid result")
        full_res_logits, scores, low_res_logits = (
            np.asarray(value) for value in result
        )
        if (
            full_res_logits.ndim != 3
            or scores.ndim != 1
            or low_res_logits.ndim != 3
            or len(full_res_logits) != len(scores)
            or len(full_res_logits) != len(low_res_logits)
            or not len(full_res_logits)
            or not np.isfinite(scores).all()
            or not np.isfinite(full_res_logits).all()
            or not np.isfinite(low_res_logits).all()
        ):
            raise ContractError("A-R7 SAM2 multimask result shapes changed")
        valid: list[tuple[float, float, int, np.ndarray]] = []
        for index in range(len(full_res_logits)):
            mask = full_res_logits[index] > 0
            px, py = positive
            if not 0 <= py < mask.shape[0] or not 0 <= px < mask.shape[1]:
                raise ContractError("A-R7 prompt seed is outside SAM2 output")
            rivals = tuple(seed for seed in seed_tuple if seed != positive)
            if not mask[py, px] or any(mask[ny, nx] for nx, ny in rivals):
                continue
            predicted_iou = float(scores[index])
            if not np.isfinite(predicted_iou) or not 0.0 <= predicted_iou <= 1.0:
                raise ContractError("A-R7 SAM2 predicted IoU is outside [0,1]")
            stability = _stability_from_logits(full_res_logits[index])
            valid.append((predicted_iou, stability, index, mask))
        if not valid:
            raise ContractError(
                f"A-R7 seed {positive_index} has no prompt-exclusive SAM2 hypothesis"
            )
        predicted_iou, stability, _, mask = sorted(
            valid, key=lambda item: (-item[0], -item[1], item[2])
        )[0]
        candidates.append(
            DeblendCandidate(
                candidate_id=f"seed-{positive_index:04d}",
                mask=mask,
                positive_seed_xy=positive,
                negative_seed_xys=tuple(
                    seed for seed in seed_tuple if seed != positive
                ),
                predicted_iou=predicted_iou,
                stability_score=stability,
            )
        )
    return partition_prompted_candidates(candidates)


__all__ = [
    "DEPTH_EDGE_NOISE_MULTIPLIER",
    "MINIMUM_SEED_REGION_PIXELS",
    "derive_depth_discontinuity_seeds",
    "merge_seed_sources",
    "predict_prompt_exclusive_children",
]
