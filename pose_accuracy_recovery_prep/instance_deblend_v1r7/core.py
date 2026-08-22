"""Threshold-independent invariants and synthetic A-R7 content gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from pose_accuracy_recovery_prep.core import ContractError

from .contracts import SYNTHETIC_STRATA


@dataclass(frozen=True)
class DeblendCandidate:
    """One SAM2 child prompted against all rival instance seeds."""

    candidate_id: str
    mask: np.ndarray
    positive_seed_xy: tuple[int, int]
    negative_seed_xys: tuple[tuple[int, int], ...]
    predicted_iou: float
    stability_score: float


def _mask(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.bool_ or not array.any():
        raise ContractError(f"{label} must be a non-empty boolean HxW mask")
    return array


def _point(point: tuple[int, int], shape: tuple[int, int], label: str) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in point)
    ):
        raise ContractError(f"{label} must be an integer (x,y) tuple")
    x, y = point
    if not 0 <= x < shape[1] or not 0 <= y < shape[0]:
        raise ContractError(f"{label} is outside the frame")


def _quality(value: float, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ContractError(f"{label} must be finite in [0,1]")
    return float(value)


def partition_prompted_candidates(
    candidates: Sequence[DeblendCandidate],
) -> list[DeblendCandidate]:
    """Validate prompt exclusion and make visible masks mutually exclusive.

    This is intentionally not an IoU/NMS threshold.  Each child must contain
    its own positive seed and exclude every declared rival seed.  Remaining
    overlap pixels are assigned deterministically to the higher model-quality
    hypothesis; the ten replay frames cannot alter this rule.
    """

    if len(candidates) < 2:
        raise ContractError("A-R7 deblending needs at least two prompted children")
    shape = np.asarray(candidates[0].mask).shape
    seen_ids: set[str] = set()
    seen_seeds: set[tuple[int, int]] = set()
    parsed: list[DeblendCandidate] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
            raise ContractError("A-R7 candidate_id must be non-empty")
        if candidate.candidate_id in seen_ids:
            raise ContractError("A-R7 candidate IDs must be unique")
        seen_ids.add(candidate.candidate_id)
        mask = _mask(candidate.mask, f"candidate[{index}].mask")
        if mask.shape != shape:
            raise ContractError("A-R7 candidate mask shapes differ")
        _point(candidate.positive_seed_xy, shape, "positive seed")
        if candidate.positive_seed_xy in seen_seeds:
            raise ContractError("A-R7 positive seeds must be unique")
        seen_seeds.add(candidate.positive_seed_xy)
        negative = tuple(candidate.negative_seed_xys)
        if len(negative) != len(candidates) - 1 or len(set(negative)) != len(negative):
            raise ContractError("A-R7 must declare every rival seed exactly once")
        _quality(candidate.predicted_iou, "predicted_iou")
        _quality(candidate.stability_score, "stability_score")
        parsed.append(replace(candidate, mask=mask, negative_seed_xys=negative))

    all_seeds = {candidate.positive_seed_xy for candidate in parsed}
    for candidate in parsed:
        if set(candidate.negative_seed_xys) != all_seeds - {candidate.positive_seed_xy}:
            raise ContractError("A-R7 negative prompts do not equal all rival seeds")
        x, y = candidate.positive_seed_xy
        if not candidate.mask[y, x]:
            raise ContractError("A-R7 child excludes its own positive seed")
        if any(candidate.mask[ny, nx] for nx, ny in candidate.negative_seed_xys):
            raise ContractError("A-R7 child includes a rival negative seed")

    ordered = sorted(
        parsed,
        key=lambda item: (
            -item.predicted_iou,
            -item.stability_score,
            item.candidate_id,
        ),
    )
    claimed = np.zeros(shape, dtype=bool)
    result: list[DeblendCandidate] = []
    for candidate in ordered:
        visible = np.logical_and(candidate.mask, ~claimed)
        x, y = candidate.positive_seed_xy
        if not visible[y, x]:
            raise ContractError("A-R7 overlap partition removed a positive seed")
        claimed |= visible
        result.append(replace(candidate, mask=visible))
    for left_index, left in enumerate(result):
        for right in result[left_index + 1 :]:
            if np.logical_and(left.mask, right.mask).any():
                raise ContractError("A-R7 visible masks are not mutually exclusive")
    return sorted(result, key=lambda item: item.candidate_id)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return 0.0 if union == 0 else intersection / union


def _maximum_matches(matrix: np.ndarray, threshold: float) -> int:
    rows, columns = matrix.shape
    matched_row_by_column = [-1] * columns

    def augment(row: int, visited: set[int]) -> bool:
        for column in range(columns):
            if column in visited or matrix[row, column] < threshold:
                continue
            visited.add(column)
            previous = matched_row_by_column[column]
            if previous == -1 or augment(previous, visited):
                matched_row_by_column[column] = row
                return True
        return False

    return sum(augment(row, set()) for row in range(rows))


def evaluate_instance_set(
    predicted_masks: Sequence[np.ndarray], ground_truth_masks: Sequence[np.ndarray]
) -> dict[str, float | int]:
    """Evaluate one synthetic scene with visible-instance ground truth."""

    if not ground_truth_masks:
        raise ContractError("A-R7 synthetic scene has no ground-truth instances")
    ground_truth = [
        _mask(mask, f"ground_truth[{index}]")
        for index, mask in enumerate(ground_truth_masks)
    ]
    shape = ground_truth[0].shape
    if any(mask.shape != shape for mask in ground_truth):
        raise ContractError("A-R7 ground-truth mask shapes differ")
    predicted = [
        _mask(mask, f"predicted[{index}]") for index, mask in enumerate(predicted_masks)
    ]
    if any(mask.shape != shape for mask in predicted):
        raise ContractError("A-R7 prediction/ground-truth shapes differ")

    matrix = np.zeros((len(ground_truth), len(predicted)), dtype=np.float64)
    for gt_index, gt_mask in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predicted):
            matrix[gt_index, prediction_index] = _iou(gt_mask, prediction)
    matches50 = _maximum_matches(matrix, 0.50)
    matches75 = _maximum_matches(matrix, 0.75)

    merge_count = 0
    for prediction in predicted:
        covered = sum(
            int(np.logical_and(prediction, gt_mask).sum()) / int(gt_mask.sum()) >= 0.5
            for gt_mask in ground_truth
        )
        merge_count += int(covered >= 2)
    split_count = 0
    for gt_mask in ground_truth:
        fragments = sum(
            int(np.logical_and(prediction, gt_mask).sum()) / int(prediction.sum())
            >= 0.5
            for prediction in predicted
        )
        split_count += int(fragments >= 2)
    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predicted),
        "matched_count_iou50": matches50,
        "matched_count_iou75": matches75,
        "unmatched_prediction_count_iou75": len(predicted) - matches75,
        "recall50": matches50 / len(ground_truth),
        "recall75": matches75 / len(ground_truth),
        "merge_count": merge_count,
        "split_count": split_count,
    }


def evaluate_synthetic_gate(
    *,
    baseline_by_stratum: Mapping[str, Sequence[Mapping[str, float | int]]],
    candidate_by_stratum: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> dict[str, object]:
    """Require general occlusion improvement without consulting replay frames."""

    if set(baseline_by_stratum) != set(SYNTHETIC_STRATA) or set(
        candidate_by_stratum
    ) != set(SYNTHETIC_STRATA):
        raise ContractError("A-R7 synthetic benchmark strata are incomplete")
    decisions: list[dict[str, object]] = []
    for stratum in SYNTHETIC_STRATA:
        baseline_rows = list(baseline_by_stratum[stratum])
        candidate_rows = list(candidate_by_stratum[stratum])
        if not baseline_rows or len(baseline_rows) != len(candidate_rows):
            raise ContractError("A-R7 paired synthetic row coverage differs")
        baseline_recall = sum(float(row["recall75"]) for row in baseline_rows) / len(
            baseline_rows
        )
        candidate_recall = sum(float(row["recall75"]) for row in candidate_rows) / len(
            candidate_rows
        )
        baseline_merges = sum(int(row["merge_count"]) for row in baseline_rows)
        candidate_merges = sum(int(row["merge_count"]) for row in candidate_rows)
        baseline_splits = sum(int(row["split_count"]) for row in baseline_rows)
        candidate_splits = sum(int(row["split_count"]) for row in candidate_rows)
        baseline_unmatched = sum(
            int(row["unmatched_prediction_count_iou75"]) for row in baseline_rows
        )
        candidate_unmatched = sum(
            int(row["unmatched_prediction_count_iou75"]) for row in candidate_rows
        )
        if stratum == "clean_single_instance":
            passed = (
                candidate_recall >= baseline_recall
                and candidate_merges == 0
                and candidate_splits <= baseline_splits
                and candidate_unmatched <= baseline_unmatched
            )
        else:
            passed = (
                candidate_recall > baseline_recall
                and candidate_merges < baseline_merges
                and candidate_splits <= baseline_splits
                and candidate_unmatched <= baseline_unmatched
            )
        decisions.append(
            {
                "stratum": stratum,
                "baseline_recall75": baseline_recall,
                "candidate_recall75": candidate_recall,
                "baseline_merge_count": baseline_merges,
                "candidate_merge_count": candidate_merges,
                "baseline_split_count": baseline_splits,
                "candidate_split_count": candidate_splits,
                "baseline_unmatched_prediction_count_iou75": baseline_unmatched,
                "candidate_unmatched_prediction_count_iou75": candidate_unmatched,
                "passed": passed,
            }
        )
    passed = all(bool(decision["passed"]) for decision in decisions)
    return {
        "status": "PASS_SYNTHETIC_DEBLEND_GATE" if passed else "NO_GO",
        "all_required_strata_passed": passed,
        "post_freeze_replay_frame_read_count": 0,
        "decisions": decisions,
    }


__all__ = [
    "DeblendCandidate",
    "evaluate_instance_set",
    "evaluate_synthetic_gate",
    "partition_prompted_candidates",
]
