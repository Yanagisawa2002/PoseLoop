from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.real_instance_detector_v1 import runtime


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_real_instance_detector_v1.json"
)


def _mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    value = np.zeros((24, 32), dtype=bool)
    value[y0:y1, x0:x1] = True
    return value


def test_protocol_is_object_disjoint_and_scene9_closed() -> None:
    protocol = runtime.load_protocol(PROTOCOL)
    split = protocol["object_disjoint_split"]
    train = set(split["train_scenes"])
    internal = set(split["internal_validation_scenes"])
    evaluation = set(split["fixed_evaluation_scenes"])
    assert not train & internal
    assert not train & evaluation
    assert not internal & evaluation
    assert 9 not in train | internal | evaluation
    assert protocol["scene9_read_permitted"] is False
    assert protocol["sealed_claim_permitted"] is False
    assert protocol["model"]["trainable_backbone_layers"] == 3
    assert protocol["model"]["official_pretrained_weight"] == {
        "url": (
            "https://download.pytorch.org/models/"
            "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
        ),
        "size_bytes": 185828065,
        "sha256": "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
    }


def test_manual_weight_path_applies_official_backbone_freeze_policy() -> None:
    model = runtime._build_model(runtime.load_protocol(PROTOCOL), None)
    body_parameters = list(model.backbone.body.named_parameters())
    frozen = [
        name for name, parameter in body_parameters if not parameter.requires_grad
    ]
    trainable = [name for name, parameter in body_parameters if parameter.requires_grad]
    assert frozen
    assert trainable
    assert all(name.startswith(("conv1", "bn1", "layer1")) for name in frozen)
    assert all(name.startswith(("layer2", "layer3", "layer4")) for name in trainable)


def test_frame_coverage_is_exact() -> None:
    assert len(runtime._frame_specs("train")) == 400
    assert len(runtime._frame_specs("internal_validation")) == 100
    assert runtime._frame_specs("fixed_evaluation") == [
        (scene_id, image_id)
        for scene_id in runtime.EVALUATION_SCENES
        for image_id in runtime.EVALUATION_IMAGE_IDS
    ]


def test_raster_mask_bundle_preserves_empty_and_disconnected_instances(
    tmp_path: Path,
) -> None:
    left = _mask(2, 2, 9, 10)
    disconnected = _mask(12, 18, 18, 24) | _mask(19, 26, 23, 31)
    path = tmp_path / "masks.npz"
    runtime._pack_masks(path, [left, disconnected], left.shape)
    restored = runtime._unpack_masks(path)
    assert len(restored) == 2
    assert np.array_equal(restored[0], left)
    assert np.array_equal(restored[1], disconnected)

    empty_path = tmp_path / "empty.npz"
    runtime._pack_masks(empty_path, [], left.shape)
    assert runtime._unpack_masks(empty_path) == []


def test_labeled_frame_binds_raster_masks_without_polygon_conversion(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "val" / "000000"
    rgb_dir = scene / "rgb_realsense"
    mask_dir = scene / "mask_visib_realsense"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(rgb_dir / "000000.png")
    mask = _mask(3, 5, 15, 20) | _mask(18, 24, 22, 30)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / "000000_000000.png")

    record = runtime._audit_labeled_frame(
        tmp_path,
        scene_id=0,
        image_id=0,
        scene_gt={"0": [{"obj_id": 16}]},
    )

    assert record["frame_id"] == "s000000-i000000"
    assert record["instances"][0]["visible_pixels"] == int(mask.sum())
    assert record["instances"][0]["box_xyxy"] == [5, 3, 30, 22]
    assert record["instances"][0]["mask"]["sha256"]


def test_detector_metrics_use_fixed_operating_point_but_ranked_ap() -> None:
    gt = _mask(2, 2, 12, 12)
    false = _mask(13, 18, 22, 30)
    metrics = runtime._detector_metrics(
        [
            {
                "frame_id": "s000010-i000000",
                "gt_masks": [gt],
                "pred_masks": [gt.copy(), false],
                "scores": [0.9, 0.1],
            }
        ],
        operating_threshold=0.25,
    )
    assert metrics["tp_iou50"] == 1
    assert metrics["fp_iou50"] == 0
    assert metrics["prediction_count"] == 1
    assert metrics["ranked_prediction_count"] == 2
    assert metrics["ap50"] == 1.0


def test_paired_delta_is_frame_level() -> None:
    baseline = {
        "per_frame": [
            {"frame_id": "a", "f1_iou50": 0.1},
            {"frame_id": "b", "f1_iou50": 0.2},
        ]
    }
    candidate = {
        "per_frame": [
            {"frame_id": "a", "f1_iou50": 0.3},
            {"frame_id": "b", "f1_iou50": 0.5},
        ]
    }
    delta = runtime._paired_delta(baseline, candidate, seed=7, draws=200)
    assert delta["mean_frame_f1_delta"] == 0.25
    assert delta["positive_frame_count"] == 2
    assert delta["bootstrap_95_interval"][0] > 0


def test_mutated_split_is_rejected(tmp_path: Path) -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["object_disjoint_split"]["train_scenes"][0] = 10
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        runtime.load_protocol(path)
    except runtime.ContractError as error:
        assert "train scenes" in str(error)
    else:
        raise AssertionError("mutated split was accepted")


def test_status_refuses_sealed_or_scene9_claim() -> None:
    text = (
        ROOT
        / "pose_accuracy_recovery_prep"
        / "real_instance_detector_v1"
        / "PREP_STATUS.md"
    ).read_text(encoding="utf-8")
    assert "not sealed" in text
    assert "does not authorize scene 9" in text
