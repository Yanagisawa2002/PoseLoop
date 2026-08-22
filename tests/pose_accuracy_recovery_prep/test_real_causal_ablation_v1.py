from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose_accuracy_recovery_prep.real_causal_ablation_v1 import runtime


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT / "protocols" / "poseloop_pose_accuracy_recovery_real_causal_ablation_v1.json"
)


def _mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    value = np.zeros((24, 32), dtype=bool)
    value[y0:y1, x0:x1] = True
    return value


def test_protocol_freezes_a_r7_and_real_non_scene9_frames() -> None:
    protocol = runtime.load_protocol(PROTOCOL)
    assert protocol["a_r7_freeze"] == {
        "commit": "1d9a2d650bada890db35d940aec7d63d57437f47",
        "status": "SYNTHETIC_POSITIVE_FORMAL_NO_GO_REAL_UNTESTED",
        "threshold_or_gate_mutation_permitted": False,
        "successor_threshold_tuning_permitted": False,
    }
    frames = runtime._planned_frames(protocol)
    assert len(frames) == 25
    assert {row["scene_id"] for row in frames} == {10, 25, 30, 40, 65}
    assert all(row["scene_id"] != 9 for row in frames)
    assert protocol["dataset"]["train_pbr_read_permitted"] is False
    assert protocol["dataset"]["scene9_replay_permitted"] is False


def test_frozen_status_is_explicit() -> None:
    text = (
        ROOT
        / "pose_accuracy_recovery_prep"
        / "instance_deblend_v1r7"
        / "A_R7_FROZEN.md"
    ).read_text(encoding="utf-8")
    assert "synthetic positive / formal NO-GO / real untested" in text
    assert "No A-R7.4" in text


def test_mask_bundle_round_trip_preserves_empty_variant(tmp_path: Path) -> None:
    mask = _mask(2, 3, 11, 15)
    candidate = runtime.Candidate(
        mask=mask,
        score=0.75,
        source="fixture",
        mask_stability=0.9,
    )
    path = tmp_path / "masks.npz"

    runtime._pack_masks(path, {"nonempty": [candidate], "empty": []})

    assert runtime._unpack_masks(path, "empty") == []
    restored = runtime._unpack_masks(path, "nonempty")
    assert len(restored) == 1
    assert np.array_equal(restored[0], mask)


def test_geometry_instances_separates_two_depth_components() -> None:
    left = _mask(4, 3, 20, 14)
    right = _mask(4, 18, 20, 29)
    support = left | right
    depth = np.zeros(support.shape, dtype=np.uint16)
    depth[left] = 900
    depth[right] = 1100
    candidates = runtime.geometry_instances(
        support,
        depth,
        source_masks=[support],
        source_scores=[0.8],
        minimum_pixels=9,
    )
    assert len(candidates) == 2
    assert all(candidate.score == 0.8 for candidate in candidates)
    assert np.logical_and(candidates[0].mask, candidates[1].mask).sum() == 0
    assert np.logical_or.reduce([row.mask for row in candidates]).sum() > 300


def test_instance_metrics_are_set_level_not_unmatched_gate() -> None:
    ground_truth = [_mask(2, 2, 10, 10), _mask(12, 18, 22, 29)]
    frames = [
        {
            "frame_id": "s000010-i000000",
            "gt_masks": ground_truth,
            "pred_masks": [mask.copy() for mask in ground_truth],
            "scores": [0.9, 0.8],
        }
    ]
    metrics = runtime.evaluate_predictions(frames)
    assert metrics["instance_precision_iou50"] == 1.0
    assert metrics["instance_recall_iou50"] == 1.0
    assert metrics["instance_f1_iou50"] == 1.0
    assert metrics["ap50"] == 1.0
    assert metrics["ap75"] == 1.0
    assert metrics["pq_iou50"] == 1.0

    duplicate = _mask(2, 2, 10, 10)
    duplicate_metrics = runtime.evaluate_predictions(
        [
            {
                **frames[0],
                "pred_masks": [*ground_truth, duplicate],
                "scores": [0.9, 0.8, 0.7],
            }
        ]
    )
    assert duplicate_metrics["instance_recall_iou50"] == 1.0
    assert duplicate_metrics["instance_precision_iou50"] == 2 / 3
    assert duplicate_metrics["instance_f1_iou50"] < 1.0


def test_cad_threshold_is_causal_keep_drop() -> None:
    scores = [0.95, 0.9, 0.4, 0.3]
    labels = [True, True, False, False]
    threshold = runtime._choose_cad_threshold(scores, labels)
    selected = [score >= threshold for score in scores]
    assert selected == [True, True, False, False]
    assert runtime._roc_auc(scores, labels) == 1.0


def test_native_a_r6_replays_normalized_policy_without_resize() -> None:
    first = runtime.Candidate(
        mask=_mask(4, 4, 12, 12),
        score=0.8,
        source="test",
        mask_stability=0.9,
        proposal_index=0,
        cad_ranking=(
            {"object_id": 1, "normalized_similarity": 0.9},
            {"object_id": 2, "normalized_similarity": 0.7},
        ),
    )
    second = runtime.Candidate(
        mask=_mask(13, 18, 21, 27),
        score=0.7,
        source="test",
        mask_stability=0.8,
        proposal_index=1,
        cad_ranking=(
            {"object_id": 2, "normalized_similarity": 0.75},
            {"object_id": 1, "normalized_similarity": 0.7},
        ),
    )
    selected, decision = runtime._native_a_r6_decision("native-frame", [first, second])
    assert selected is first
    assert decision["selected_proposal_index"] == 0
    assert decision["native_frame_height_width"] == [24, 32]
    assert all(row["eligible"] for row in decision["candidate_audits"])


def _metric(value: float) -> dict[str, object]:
    per_frame = []
    for scene_id in runtime.EXPECTED_SCENES:
        for image_id in runtime.EXPECTED_IMAGES:
            per_frame.append(
                {
                    "frame_id": f"s{scene_id:06d}-i{image_id:06d}",
                    "f1_iou50": value,
                }
            )
    return {
        "instance_f1_iou50": value,
        "ap50": value,
        "pq_iou50": value,
        "per_frame": per_frame,
    }


def test_decisions_drop_nonincremental_modules() -> None:
    metrics = {
        "a_r6_baseline": _metric(0.05),
        "raw_fastsam_eligible": _metric(0.10),
        "geometry_instances": _metric(0.40),
        "geometry_plus_sam2": _metric(0.40),
        "geometry_plus_sam2_plus_causal_cad": _metric(0.40),
    }
    decisions = runtime.make_decisions(
        metrics,
        cad_auc={"auroc": 0.5, "bootstrap_95_interval": [0.45, 0.55]},
        bootstrap_seed=7,
        bootstrap_draws=100,
    )
    assert decisions["sam2"]["decision"] == "DROP_SAM2"
    assert decisions["cad"]["decision"] == "DROP_CAD"
    assert decisions["selected_successor"] == "geometry_instances"
    assert (
        decisions["route_decision"]
        == "REAL_DEVELOPMENT_ROUTE_POSITIVE_BUILD_PHYSICAL_SYNTHETIC_NEXT"
    )
    assert decisions["scene9_replay_permitted"] is False


def test_protocol_is_plain_json_without_result_values() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    serialized = json.dumps(protocol).lower()
    assert 'train_pbr_read_permitted": true' not in serialized
    assert 'scene9_replay_permitted": true' not in serialized
    assert "absolute_unmatched_prediction_count" in serialized
