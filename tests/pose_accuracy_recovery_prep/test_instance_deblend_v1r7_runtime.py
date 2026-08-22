from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.core import read_json, sha256_file
from pose_accuracy_recovery_prep.instance_deblend_v1r7.contracts import (
    SYNTHETIC_STRATA,
)
from pose_accuracy_recovery_prep.instance_deblend_v1r7.execution_contract import (
    validate_execution_protocol,
)
from pose_accuracy_recovery_prep.instance_deblend_v1r7.synthetic_runtime import (
    DEVELOPMENT_FIRST_SEED,
    DEVELOPMENT_ROWS_PER_STRATUM,
    RuntimeProposal,
    SyntheticGateRuntime,
    _compose_depth_consistent_candidate_masks,
    _depth_instance_seeds,
    _evaluate_synthetic_gate_v1r7_3,
    _predict_prompt_exclusive_box_children,
    _seed_plan,
    generate_scene,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_PROTOCOL = (
    REPOSITORY_ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_instance_deblend_synthetic_v1r7_3.json"
)


def _template_fixture(root: Path) -> dict[str, object]:
    path = root / "assets" / "templates" / "shared.png"
    path.parent.mkdir(parents=True)
    image = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 20, 148, 100), radius=16, fill=(180, 90, 35, 255))
    image.save(path)
    asset = {
        "role": "cad_template_rgba",
        "relative_path": "assets/templates/shared.png",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return {
        "objects": [
            {
                "object_id": object_id,
                "views": [
                    {
                        "view_index": index,
                        "rgba": dict(asset),
                        "alpha_pixels": 1,
                        "rgb_nonzero_pixels": 1,
                    }
                    for index in range(42)
                ],
            }
            for object_id in (1, 2, 4, 5, 6)
        ]
    }


def test_execution_protocol_closes_synthetic_only_identity() -> None:
    protocol = read_json(EXECUTION_PROTOCOL)
    validated = validate_execution_protocol(protocol, repository_root=REPOSITORY_ROOT)
    assert validated["authorization"]["development_gate_permitted"] is True
    assert (
        validated["authorization"]["scene9_replay_permitted_before_gate_pass"] is False
    )


def test_development_seed_plan_is_fixed_and_disjoint() -> None:
    plan = _seed_plan("development", DEVELOPMENT_ROWS_PER_STRATUM)
    assert list(plan) == SYNTHETIC_STRATA
    flattened = [seed for rows in plan.values() for seed in rows]
    assert flattened == list(
        range(
            DEVELOPMENT_FIRST_SEED,
            DEVELOPMENT_FIRST_SEED
            + DEVELOPMENT_ROWS_PER_STRATUM * len(SYNTHETIC_STRATA),
        )
    )
    assert not set(flattened).intersection(
        seed for rows in _seed_plan("train-smoke", 8).values() for seed in rows
    )


def test_disconnected_same_depth_pieces_share_one_instance_seed() -> None:
    support = np.zeros((80, 100), dtype=bool)
    depth = np.zeros((80, 100), dtype=np.uint16)
    support[10:30, 10:30] = True
    support[42:62, 12:32] = True
    support[25:55, 60:90] = True
    depth[10:30, 10:30] = 1000
    depth[42:62, 12:32] = 1000
    depth[25:55, 60:90] = 800
    seeds = _depth_instance_seeds(support, depth)
    assert len(seeds) == 2
    assert {int(depth[y, x]) for x, y in seeds} == {800, 1000}


def test_box_prompt_preserves_rival_exclusion() -> None:
    class Predictor:
        def __init__(self) -> None:
            self.boxes: list[tuple[int, int, int, int]] = []

        def predict(
            self, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            box = tuple(int(value) for value in np.asarray(kwargs["box"]).tolist())
            self.boxes.append(box)
            x0, y0, x1, y1 = box
            logits = np.full((1, 60, 100), -2.0, dtype=np.float32)
            logits[0, y0:y1, x0:x1] = 2.0
            return logits, np.asarray([0.9]), logits.copy()

    predictor = Predictor()
    prompts = (
        ((15, 15), (10, 10, 30, 30)),
        ((65, 15), (50, 10, 80, 30)),
    )
    children, failed = _predict_prompt_exclusive_box_children(predictor, prompts)
    assert predictor.boxes == [(10, 10, 30, 30), (50, 10, 80, 30)]
    assert failed == []
    assert len(children) == 2
    assert not np.logical_and(children[0].mask, children[1].mask).any()


def test_box_prompt_keeps_two_valid_children_when_third_prompt_fails() -> None:
    class Predictor:
        def __init__(self) -> None:
            self.point_counts: list[int] = []

        def predict(
            self, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            points = np.asarray(kwargs["point_coords"])
            self.point_counts.append(len(points))
            box = tuple(int(value) for value in np.asarray(kwargs["box"]).tolist())
            logits = np.full((1, 70, 120), -2.0, dtype=np.float32)
            if box[0] < 80:
                x0, y0, x1, y1 = box
                logits[0, y0:y1, x0:x1] = 2.0
            return logits, np.asarray([0.9]), logits.copy()

    predictor = Predictor()
    prompts = (
        ((15, 15), (10, 10, 30, 30)),
        ((55, 15), (50, 10, 70, 30)),
        ((95, 15), (90, 10, 110, 30)),
    )
    children, failed = _predict_prompt_exclusive_box_children(predictor, prompts)
    assert predictor.point_counts == [3, 3, 3]
    assert failed == [2]
    assert len(children) == 2
    assert not np.logical_and(children[0].mask, children[1].mask).any()
    support = np.zeros((70, 120), dtype=bool)
    depth = np.zeros((70, 120), dtype=np.uint16)
    for index, (_, (x0, y0, x1, y1)) in enumerate(prompts):
        support[y0:y1, x0:x1] = True
        depth[y0:y1, x0:x1] = 1000 - index * 100
    parent_mask = np.zeros_like(support)
    parent_mask[10:30, 90:110] = True
    parent = RuntimeProposal(
        proposal_index=7,
        mask=parent_mask,
        proposal_score=0.8,
        mask_stability=0.7,
        cad_ranking=(),
    )
    masks, scores, stabilities, sources, region_pixels = (
        _compose_depth_consistent_candidate_masks(
            children=children,
            prompts=prompts,
            failed_prompt_indices=failed,
            parent=parent,
            support_mask=support,
            depth_mm=depth,
        )
    )
    assert len(masks) == 3
    assert [int(mask.sum()) for mask in masks] == [400, 400, 400]
    assert scores == [0.9, 0.9, 0.8]
    assert stabilities == [1.0, 1.0, 0.7]
    assert sources == [
        "sam2-union-depth-owner-seed-0000",
        "sam2-union-depth-owner-seed-0001",
        "a-r6-parent-authorized-depth-owner-seed-0002",
    ]
    assert region_pixels == [400, 400, 400]


def test_zero_baseline_merge_requires_recall_gain_not_impossible_reduction() -> None:
    baseline = {
        stratum: [
            {
                "recall75": 1.0 if stratum == "clean_single_instance" else 0.0,
                "merge_count": 0,
                "split_count": 0,
                "unmatched_prediction_count_iou75": 0,
            }
        ]
        for stratum in SYNTHETIC_STRATA
    }
    candidate = {
        stratum: [
            {
                "recall75": 1.0,
                "merge_count": 0,
                "split_count": 0,
                "unmatched_prediction_count_iou75": 0,
            }
        ]
        for stratum in SYNTHETIC_STRATA
    }
    gate = _evaluate_synthetic_gate_v1r7_3(
        baseline_by_stratum=baseline,
        candidate_by_stratum=candidate,
    )
    assert gate["all_required_strata_passed"] is True
    assert all(decision["merge_gate_passed"] for decision in gate["decisions"])


def test_synthetic_generator_produces_visible_instances(tmp_path: Path) -> None:
    manifest = _template_fixture(tmp_path)
    scene = generate_scene(
        stratum="same_cad_pile_three_plus",
        seed=4100,
        data_root=tmp_path,
        template_manifest=manifest,
    )
    assert scene.rgb.shape == (1080, 1440, 3)
    assert scene.depth_mm.shape == (1080, 1440)
    assert len(scene.visible_masks) == 3
    assert all(
        mask.dtype == np.bool_ and int(mask.sum()) >= 500
        for mask in scene.visible_masks
    )
    assert not any(
        np.logical_and(left, right).any()
        for index, left in enumerate(scene.visible_masks)
        for right in scene.visible_masks[index + 1 :]
    )
    assert np.array_equal(scene.depth_mm > 0, np.logical_or.reduce(scene.visible_masks))


def test_a_r6_parent_selection_uses_cad_scored_proposal() -> None:
    mask = np.zeros((1080, 1440), dtype=bool)
    mask[300:650, 470:820] = True
    ranking = tuple(
        {
            "object_id": object_id,
            "raw_cosine": raw,
            "normalized_similarity": (raw + 1.0) / 2.0,
            "top5_template_cosines": [raw] * 5,
            "top5_template_indices": [0, 1, 2, 3, 4],
        }
        for object_id, raw in zip(
            (1, 2, 4, 5, 6), (0.8, 0.5, 0.4, 0.3, 0.2), strict=True
        )
    )
    proposal = RuntimeProposal(
        proposal_index=7,
        mask=mask,
        proposal_score=0.9,
        mask_stability=0.95,
        cad_ranking=ranking,
    )
    selected, decision = SyntheticGateRuntime.select_a_r6_parent("fixture", [proposal])
    assert decision["decision_state"] == "SELECTED"
    assert decision["selected_object_id"] == 1
    assert np.array_equal(selected, mask)
