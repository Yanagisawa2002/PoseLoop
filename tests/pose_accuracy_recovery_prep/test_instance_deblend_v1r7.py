from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_deblend_v1r7 import (
    DeblendCandidate,
    derive_depth_discontinuity_seeds,
    evaluate_instance_set,
    evaluate_synthetic_gate,
    merge_seed_sources,
    partition_prompted_candidates,
    predict_prompt_exclusive_children,
)
from pose_accuracy_recovery_prep.instance_deblend_v1r7 import contracts
from pose_accuracy_recovery_prep.instance_deblend_v1r7.cli import main


def _rect(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = True
    return result


def _protocol(repository_root: Path) -> dict[str, object]:
    paths = [
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/__init__.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/__main__.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/adapter.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/cli.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/contracts.py",
        "pose_accuracy_recovery_prep/instance_deblend_v1r7/core.py",
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/instance_selection_v1r6.py",
    ]
    protocol: dict[str, object] = {
        "schema_version": contracts.PROTOCOL_SCHEMA,
        "protocol_id": contracts.PROTOCOL_ID,
        "role": contracts.PROTOCOL_ROLE,
        "predecessor_freeze": {
            "commit": contracts.A_R6_COMMIT,
            "tree": contracts.A_R6_TREE,
            "protocol_id": contracts.A_R6_PROTOCOL_ID,
            "protocol_lock_sha256": contracts.A_R6_PROTOCOL_LOCK,
            "selection_policy_sha256": contracts.A_R6_POLICY_SHA256,
            "selection_policy_mutation_permitted": False,
            "output_lock_sha256": contracts.A_R6_OUTPUT_LOCK,
            "output_sha256": contracts.A_R6_OUTPUT_SHA256,
            "receipt_lock_sha256": contracts.A_R6_RECEIPT_LOCK,
            "receipt_sha256": contracts.A_R6_RECEIPT_SHA256,
            "replay_frame_ids": contracts.REPLAY_FRAME_IDS,
            "historical_diagnostic_exposure": {
                "a_r6_output_bundle_opened": True,
                "human_content_review_completed": True,
                "identified_failure_frame_id": "scene-000009-image-000000",
                "identified_failure_taxonomy": "MERGED_ADJACENT_OCCLUDED_INSTANCES",
                "raw_rgb_or_depth_opened_for_a_r7_configuration": False,
                "numeric_threshold_or_weight_change_permitted": False,
            },
            "post_freeze_replay_role": "ONE_SHOT_POST_COMMIT_CONTENT_REPLAY_ONLY",
            "post_freeze_replay_may_influence_configuration": False,
        },
        "new_method": {
            "proposal_family": "SAM2_1_RGBD_NEGATIVE_PROMPT_INSTANCE_DEBLEND",
            "sam2_repository": "https://github.com/facebookresearch/sam2",
            "observed_main_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "source_tree_sha1": None,
            "source_archive_sha256": None,
            "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint_url": (
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
                "sam2.1_hiera_large.pt"
            ),
            "checkpoint_bytes": None,
            "checkpoint_sha256": None,
            "prompt_sources": [
                "RGB_IMAGE_EMBEDDING",
                "RAW_SENSOR_DEPTH_DISCONTINUITY_SEEDS",
                "RIVAL_SEEDS_AS_NEGATIVE_PROMPTS",
            ],
            "prompt_rule": "SYNTHETIC_CALIBRATION_ONLY_UNTIL_FROZEN",
            "negative_prompt_exclusion_required": True,
            "visible_masks_must_be_mutually_exclusive": True,
            "a_r6_policy_reused_byte_exact": True,
            "threshold_override_permitted": False,
        },
        "synthetic_development": {
            "label_source": "CAD_RENDERER_VISIBLE_INSTANCE_IDS_ONLY",
            "required_strata": contracts.SYNTHETIC_STRATA,
            "training_seed_interval": [0, 4095],
            "development_seed_interval": [4096, 5119],
            "seed_intervals_disjoint": True,
            "real_replay_frames_permitted": False,
            "gt_sensor_or_evaluator_assets_permitted": False,
            "result_driven_stratum_replacement_permitted": False,
        },
        "gates": {
            "matching_iou_thresholds": [0.5, 0.75],
            "occlusion_strata_require_strict_recall75_improvement": True,
            "occlusion_strata_require_strict_merge_reduction": True,
            "clean_singleton_recall75_may_decrease": False,
            "unmatched_prediction_count_may_increase": False,
            "split_count_may_increase": False,
            "every_required_stratum_must_pass": True,
            "scene9_pass_is_a_development_gate": False,
            "failure_action": "NO_GO_NO_THRESHOLD_TUNING_ON_REPLAY",
        },
        "access_boundary": {
            "post_freeze_replay_rgb_open_count": 0,
            "post_freeze_replay_depth_open_count": 0,
            "post_freeze_replay_prediction_open_count": 0,
            "gt_pose_open_count": 0,
            "gt_mask_open_count": 0,
            "evaluator_open_count": 0,
            "foundationpose_run_count": 0,
            "official_scorer_run_count": 0,
            "downstream_export_count": 0,
        },
        "readiness": {
            "prep_contract_ready": True,
            "formal_workload_permitted": False,
            "scene9_replay_permitted": False,
            "blockers": contracts.FORMAL_BLOCKERS,
        },
        "wrapper_files": [],
        "protocol_lock_sha256": "pending",
    }
    for relative in paths:
        path = repository_root / Path(*relative.split("/"))
        protocol["wrapper_files"].append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    protocol["protocol_lock_sha256"] = canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_lock_sha256"}
    )
    return protocol


def test_protocol_freezes_a_r6_and_keeps_replay_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = _protocol(root)
    committed = json.loads(
        (
            root
            / "protocols"
            / "poseloop_pose_accuracy_recovery_instance_deblend_v1r7.json"
        ).read_text(encoding="utf-8")
    )
    assert committed == protocol
    assert contracts.validate_protocol(protocol, repository_root=root)
    assert protocol["readiness"]["formal_workload_permitted"] is False
    assert protocol["readiness"]["scene9_replay_permitted"] is False


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("predecessor_freeze", "selection_policy_mutation_permitted", True, "frozen"),
        (
            "predecessor_freeze",
            "post_freeze_replay_may_influence_configuration",
            True,
            "frozen",
        ),
        ("new_method", "threshold_override_permitted", True, "no-retuning"),
        ("synthetic_development", "real_replay_frames_permitted", True, "synthetic"),
        ("gates", "scene9_pass_is_a_development_gate", True, "scientific"),
        ("readiness", "scene9_replay_permitted", True, "overstated"),
    ],
)
def test_protocol_rejects_replay_driven_retuning(
    section: str, field: str, value: object, match: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = _protocol(root)
    changed = copy.deepcopy(protocol)
    changed[section][field] = value
    changed["protocol_lock_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "protocol_lock_sha256"}
    )
    with pytest.raises(ContractError, match=match):
        contracts.validate_protocol(changed, repository_root=root)


def test_prompt_exclusion_partitions_visible_instances_without_iou_threshold() -> None:
    shape = (32, 48)
    left = _rect(shape, (4, 8, 27, 25))
    right = _rect(shape, (21, 8, 44, 25))
    left[16, 36] = False
    right[16, 12] = False
    candidates = [
        DeblendCandidate("left", left, (12, 16), ((36, 16),), 0.91, 0.96),
        DeblendCandidate("right", right, (36, 16), ((12, 16),), 0.90, 0.97),
    ]
    partitioned = partition_prompted_candidates(candidates)
    by_id = {candidate.candidate_id: candidate for candidate in partitioned}
    assert by_id["left"].mask[16, 12]
    assert by_id["right"].mask[16, 36]
    assert not np.logical_and(by_id["left"].mask, by_id["right"].mask).any()


def test_prompt_exclusion_rejects_merged_child() -> None:
    shape = (24, 32)
    merged = _rect(shape, (3, 4, 29, 20))
    rival = _rect(shape, (17, 4, 29, 20))
    candidates = [
        DeblendCandidate("merged", merged, (8, 12), ((24, 12),), 0.95, 0.98),
        DeblendCandidate("rival", rival, (24, 12), ((8, 12),), 0.90, 0.96),
    ]
    with pytest.raises(ContractError, match="rival negative seed"):
        partition_prompted_candidates(candidates)


def test_depth_discontinuity_seeds_are_sensor_adaptive_and_deterministic() -> None:
    shape = (48, 64)
    support = _rect(shape, (4, 8, 60, 40))
    depth = np.zeros(shape, dtype=np.uint16)
    depth[8:40, 4:32] = 900
    depth[8:40, 32:60] = 1050
    seeds = derive_depth_discontinuity_seeds(support, depth)
    assert len(seeds) == 2
    assert seeds[0][0] < 32 < seeds[1][0]
    assert seeds == derive_depth_discontinuity_seeds(support, depth)


def test_seed_sources_are_exact_union_and_do_not_use_distance_threshold() -> None:
    assert merge_seed_sources(((3, 4), (9, 4)), ((3, 4), (15, 4))) == (
        (3, 4),
        (9, 4),
        (15, 4),
    )
    with pytest.raises(ContractError, match="at least two"):
        merge_seed_sources(((3, 4),), ((3, 4),))


class _FakePredictor:
    def __init__(self, masks: dict[tuple[int, int], np.ndarray]) -> None:
        self.masks = masks
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(
        self,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
        return_logits: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert multimask_output is True and return_logits is True
        self.calls.append((point_coords.copy(), point_labels.copy()))
        seed = tuple(int(value) for value in point_coords[0])
        good = self.masks[seed]
        merged = np.logical_or.reduce(list(self.masks.values()))
        bad = np.zeros_like(good)
        bad[0, 0] = True
        masks = np.stack([merged, good, bad])
        full_res_logits = np.where(masks, 2.0, -2.0)
        low_res_logits = np.zeros((3, 8, 8), dtype=np.float32)
        return full_res_logits, np.asarray([0.99, 0.90, 0.10]), low_res_logits


def test_prompted_adapter_rejects_higher_scoring_merge_and_keeps_children() -> None:
    shape = (32, 48)
    seeds = ((10, 16), (36, 16))
    masks = {
        seeds[0]: _rect(shape, (3, 8, 20, 25)),
        seeds[1]: _rect(shape, (28, 8, 45, 25)),
    }
    predictor = _FakePredictor(masks)
    children = predict_prompt_exclusive_children(predictor, seeds)
    assert len(children) == 2
    assert all(child.predicted_iou == 0.90 for child in children)
    assert all(len(labels) == 2 for _, labels in predictor.calls)
    assert all(labels.tolist() == [1, 0] for _, labels in predictor.calls)


def test_instance_metrics_detect_merge_and_recovery() -> None:
    shape = (24, 32)
    left = _rect(shape, (2, 4, 14, 20))
    right = _rect(shape, (18, 4, 30, 20))
    baseline = evaluate_instance_set([left | right], [left, right])
    candidate = evaluate_instance_set([left, right], [left, right])
    assert baseline["merge_count"] == 1
    assert baseline["recall75"] == 0.0
    assert candidate["merge_count"] == 0
    assert candidate["recall75"] == 1.0


def test_instance_matching_handles_dense_proposal_sets_without_greedy_loss() -> None:
    shape = (64, 64)
    ground_truth = [_rect(shape, (2, 2, 14, 14)), _rect(shape, (48, 48, 60, 60))]
    predictions = [
        _rect(shape, (20 + index % 5, 20 + index // 5, 22 + index % 5, 22 + index // 5))
        for index in range(25)
    ]
    predictions.extend(ground_truth)
    metrics = evaluate_instance_set(predictions, ground_truth)
    assert metrics["matched_count_iou75"] == 2


def test_synthetic_gate_requires_every_occlusion_stratum_and_clean_guard() -> None:
    baseline: dict[str, list[dict[str, float | int]]] = {}
    candidate: dict[str, list[dict[str, float | int]]] = {}
    for stratum in contracts.SYNTHETIC_STRATA:
        if stratum == "clean_single_instance":
            baseline[stratum] = [
                {
                    "recall75": 1.0,
                    "merge_count": 0,
                    "split_count": 0,
                    "unmatched_prediction_count_iou75": 0,
                }
            ]
            candidate[stratum] = [
                {
                    "recall75": 1.0,
                    "merge_count": 0,
                    "split_count": 0,
                    "unmatched_prediction_count_iou75": 0,
                }
            ]
        else:
            baseline[stratum] = [
                {
                    "recall75": 0.0,
                    "merge_count": 1,
                    "split_count": 0,
                    "unmatched_prediction_count_iou75": 1,
                }
            ]
            candidate[stratum] = [
                {
                    "recall75": 1.0,
                    "merge_count": 0,
                    "split_count": 0,
                    "unmatched_prediction_count_iou75": 0,
                }
            ]
    result = evaluate_synthetic_gate(
        baseline_by_stratum=baseline, candidate_by_stratum=candidate
    )
    assert result["status"] == "PASS_SYNTHETIC_DEBLEND_GATE"
    assert result["post_freeze_replay_frame_read_count"] == 0
    failed = copy.deepcopy(candidate)
    failed["touching_pair_same_cad"] = copy.deepcopy(baseline["touching_pair_same_cad"])
    result = evaluate_synthetic_gate(
        baseline_by_stratum=baseline, candidate_by_stratum=failed
    )
    assert result["status"] == "NO_GO"
    flooded = copy.deepcopy(candidate)
    flooded["same_cad_pile_three_plus"][0]["unmatched_prediction_count_iou75"] = 2
    result = evaluate_synthetic_gate(
        baseline_by_stratum=baseline, candidate_by_stratum=flooded
    )
    assert result["status"] == "NO_GO"


def test_cli_reports_prep_only(capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = (
        root
        / "protocols"
        / "poseloop_pose_accuracy_recovery_instance_deblend_v1r7.json"
    )
    assert (
        main(
            [
                "contract-check",
                "--protocol",
                str(protocol_path),
                "--repository-root",
                str(root),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS_PREP_CONTRACT"
    assert output["formal_workload_permitted"] is False
    assert output["scene9_replay_permitted"] is False
