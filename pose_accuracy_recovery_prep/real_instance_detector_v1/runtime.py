"""A-R9 object-disjoint class-agnostic Mask R-CNN development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.real_causal_ablation_v1 import runtime as a8


PROTOCOL_ID = "poseloop.pose-accuracy-recovery.real-instance-detector.v1"
PROTOCOL_SCHEMA = "poseloop.a-r9.real-instance-detector-protocol.v1"
MANIFEST_SCHEMA = "poseloop.a-r9.object-disjoint-dataset-manifest.v1"
CHECKPOINT_SCHEMA = "poseloop.a-r9.mask-rcnn-checkpoint.v1"
PREDICTION_SCHEMA = "poseloop.a-r9.mask-rcnn-predictions.v1"
RESULT_SCHEMA = "poseloop.a-r9.real-instance-detector-result.v1"

TRAIN_SCENES = (0, 5, 15, 20, 35, 45, 50, 55)
VALIDATION_SCENES = (60, 70)
EVALUATION_SCENES = (10, 25, 30, 40, 65)
EVALUATION_IMAGE_IDS = (0, 10, 20, 30, 40)
EXPECTED_OBJECT_IDS = {
    0: 16,
    5: 17,
    10: 2,
    15: 13,
    20: 8,
    25: 1,
    30: 5,
    35: 12,
    40: 4,
    45: 9,
    50: 10,
    55: 15,
    60: 11,
    65: 6,
    70: 14,
}
A8_RESULT_SHA256 = "2adec52736b3010f6b7d0ebbb58fb7da0d0f9d32d7a82765f31f3f2b489b49cc"
A8_PREDICTION_MANIFEST_SHA256 = (
    "9680fdad88bc7d95c840337d9a9f086033e80ef5a8c37281c30e6a92f64b5079"
)


class ContractError(RuntimeError):
    """Raised when a frozen A-R9 boundary is violated."""


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


def _atomic_torch_save(path: Path, value: Mapping[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _exact_list(value: Any, expected: Sequence[int], label: str) -> None:
    if value != list(expected):
        raise ContractError(f"A-R9 {label} changed")


def load_protocol(path: Path) -> dict[str, Any]:
    """Load the exact A-R9 experiment contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("A-R9 protocol must be an object")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("A-R9 protocol schema changed")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("A-R9 protocol identity changed")
    if value.get("dataset_role") != "ALREADY_CONSUMED_REAL_DEVELOPMENT":
        raise ContractError("A-R9 dataset role changed")
    split = value.get("object_disjoint_split")
    if not isinstance(split, dict):
        raise ContractError("A-R9 split is missing")
    _exact_list(split.get("train_scenes"), TRAIN_SCENES, "train scenes")
    _exact_list(
        split.get("internal_validation_scenes"),
        VALIDATION_SCENES,
        "internal validation scenes",
    )
    _exact_list(
        split.get("fixed_evaluation_scenes"),
        EVALUATION_SCENES,
        "evaluation scenes",
    )
    _exact_list(
        split.get("fixed_evaluation_image_ids"),
        EVALUATION_IMAGE_IDS,
        "evaluation image IDs",
    )
    if split.get("train_and_internal_image_ids") != list(range(50)):
        raise ContractError("A-R9 train/internal image IDs changed")
    if split.get("scene_and_object_disjoint") is not True:
        raise ContractError("A-R9 split is not scene/object-disjoint")
    all_scenes = (*TRAIN_SCENES, *VALIDATION_SCENES, *EVALUATION_SCENES)
    if len(set(all_scenes)) != len(all_scenes) or 9 in all_scenes:
        raise ContractError("A-R9 split overlaps or touches scene 9")
    expected_ids = {f"{key:06d}": row for key, row in EXPECTED_OBJECT_IDS.items()}
    if split.get("expected_scene_object_ids") != expected_ids:
        raise ContractError("A-R9 object identities changed")
    if value.get("scene9_read_permitted") is not False:
        raise ContractError("A-R9 scene 9 boundary changed")
    if value.get("sealed_claim_permitted") is not False:
        raise ContractError("A-R9 cannot make a sealed claim")
    previous = value.get("frozen_a8_comparison")
    if previous != {
        "result_sha256": A8_RESULT_SHA256,
        "prediction_manifest_sha256": A8_PREDICTION_MANIFEST_SHA256,
        "comparison_variant": "raw_fastsam_eligible",
    }:
        raise ContractError("A-R9 A-R8 comparison identity changed")
    model = value.get("model")
    if not isinstance(model, dict):
        raise ContractError("A-R9 model contract is missing")
    required_model = {
        "architecture": "torchvision.maskrcnn_resnet50_fpn_v2",
        "foreground_classes": 1,
        "class_agnostic": True,
        "trainable_backbone_layers": 3,
        "minimum_image_side": 800,
        "maximum_image_side": 1333,
        "detections_per_image": 100,
        "prediction_score_floor": 0.01,
        "operating_score_threshold": 0.25,
        "mask_probability_threshold": 0.5,
    }
    for key, expected in required_model.items():
        if model.get(key) != expected:
            raise ContractError(f"A-R9 model field {key} changed")
    weight = model.get("official_pretrained_weight")
    if not isinstance(weight, dict):
        raise ContractError("A-R9 official weight is missing")
    if weight.get("url") != (
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
    ):
        raise ContractError("A-R9 official weight URL changed")
    if not isinstance(weight.get("size_bytes"), int) or weight["size_bytes"] <= 0:
        raise ContractError("A-R9 official weight size is not frozen")
    sha = weight.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or sha == "0" * 64:
        raise ContractError("A-R9 official weight SHA is not frozen")
    training = value.get("training")
    if training != {
        "epochs": 8,
        "physical_batch_size": 1,
        "gradient_accumulation": 2,
        "optimizer": "SGD",
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "scheduler": "StepLR(step_size=4,gamma=0.1)",
        "horizontal_flip_probability": 0.5,
        "internal_validation_health_image_ids": [0, 10, 20, 30, 40],
        "cublas_workspace_config": ":4096:8",
        "seed": 314159,
        "workers": 8,
    }:
        raise ContractError("A-R9 training contract changed")
    gate = value.get("promotion_gate")
    if gate != {
        "paired_frame_f1_bootstrap_lower_gt_zero": True,
        "positive_scene_count_minimum": 4,
        "recall_strictly_above_raw_fastsam": True,
        "ap50_strictly_above_raw_fastsam": True,
        "pq_strictly_above_raw_fastsam": True,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 314159,
    }:
        raise ContractError("A-R9 promotion gate changed")
    return value


def _frame_id(scene_id: int, image_id: int) -> str:
    return f"s{scene_id:06d}-i{image_id:06d}"


def _relative_sensor_path(scene_id: int, image_id: int) -> Path:
    return Path("val") / f"{scene_id:06d}" / "rgb_realsense" / f"{image_id:06d}.png"


def _frame_specs(split: str) -> list[tuple[int, int]]:
    if split == "train":
        scenes, image_ids = TRAIN_SCENES, range(50)
    elif split == "internal_validation":
        scenes, image_ids = VALIDATION_SCENES, range(50)
    elif split == "fixed_evaluation":
        scenes, image_ids = EVALUATION_SCENES, EVALUATION_IMAGE_IDS
    else:
        raise ContractError(f"A-R9 unknown split {split}")
    return [(scene_id, image_id) for scene_id in scenes for image_id in image_ids]


def _bound_file(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"A-R9 required file is missing: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ContractError("A-R9 asset escaped dataset root") from error
    return {
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_scene_gt(
    dataset_root: Path, scene_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = dataset_root / "val" / f"{scene_id:06d}" / "scene_gt_realsense.json"
    bound = _bound_file(path, dataset_root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("A-R9 development GT must be an object")
    return value, bound


def _audit_labeled_frame(
    dataset_root: Path,
    scene_id: int,
    image_id: int,
    scene_gt: Mapping[str, Any],
) -> dict[str, Any]:
    rgb_path = dataset_root / _relative_sensor_path(scene_id, image_id)
    with Image.open(rgb_path) as image:
        width, height = image.size
        if image.mode not in {"RGB", "RGBA"}:
            image.convert("RGB")
    rows = scene_gt.get(str(image_id))
    if not isinstance(rows, list) or not rows:
        raise ContractError("A-R9 labeled frame has no instances")
    expected_object_id = EXPECTED_OBJECT_IDS[scene_id]
    instances: list[dict[str, Any]] = []
    for gt_index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or int(row.get("obj_id", -1)) != expected_object_id
        ):
            raise ContractError("A-R9 scene/object identity changed")
        mask_path = (
            dataset_root
            / "val"
            / f"{scene_id:06d}"
            / "mask_visib_realsense"
            / f"{image_id:06d}_{gt_index:06d}.png"
        )
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != (height, width) or not mask.any():
            raise ContractError("A-R9 visible mask is empty or shape-mismatched")
        ys, xs = np.nonzero(mask)
        instances.append(
            {
                "gt_index": gt_index,
                "object_id": expected_object_id,
                "mask": _bound_file(mask_path, dataset_root),
                "visible_pixels": int(mask.sum()),
                "box_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
            }
        )
    return {
        "frame_id": _frame_id(scene_id, image_id),
        "scene_id": scene_id,
        "image_id": image_id,
        "height": height,
        "width": width,
        "rgb": _bound_file(rgb_path, dataset_root),
        "instances": instances,
    }


def _audit_unlabeled_frame(
    dataset_root: Path, scene_id: int, image_id: int
) -> dict[str, Any]:
    rgb_path = dataset_root / _relative_sensor_path(scene_id, image_id)
    with Image.open(rgb_path) as image:
        width, height = image.size
    return {
        "frame_id": _frame_id(scene_id, image_id),
        "scene_id": scene_id,
        "image_id": image_id,
        "height": height,
        "width": width,
        "rgb": _bound_file(rgb_path, dataset_root),
    }


def build_dataset_manifest(
    *, protocol_path: Path, dataset_root: Path, output_root: Path
) -> dict[str, Any]:
    """Freeze actual train/validation assets without opening evaluation GT."""

    protocol = load_protocol(protocol_path)
    if output_root.exists():
        raise ContractError("A-R9 dataset output root is create-only")
    output_root.mkdir(parents=True)
    frames: dict[str, list[dict[str, Any]]] = {}
    gt_assets: dict[str, dict[str, Any]] = {}
    development_gt_scene_count = 0
    for split, scenes in (
        ("train", TRAIN_SCENES),
        ("internal_validation", VALIDATION_SCENES),
    ):
        by_scene: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for scene_id in scenes:
            by_scene[scene_id] = _read_scene_gt(dataset_root, scene_id)
            gt_assets[f"{scene_id:06d}"] = by_scene[scene_id][1]
            development_gt_scene_count += 1
        frames[split] = [
            _audit_labeled_frame(
                dataset_root,
                scene_id,
                image_id,
                by_scene[scene_id][0],
            )
            for scene_id, image_id in _frame_specs(split)
        ]
    frames["fixed_evaluation"] = [
        _audit_unlabeled_frame(dataset_root, scene_id, image_id)
        for scene_id, image_id in _frame_specs("fixed_evaluation")
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_role": protocol["dataset_role"],
        "split_policy": "SCENE_AND_OBJECT_DISJOINT",
        "train_frame_count": len(frames["train"]),
        "internal_validation_frame_count": len(frames["internal_validation"]),
        "fixed_evaluation_frame_count": len(frames["fixed_evaluation"]),
        "development_gt_scene_count": development_gt_scene_count,
        "fixed_evaluation_gt_access_count": 0,
        "scene9_read_count": 0,
        "scene_gt_assets": gt_assets,
        "frames": frames,
    }
    _write_json(output_root / "dataset-manifest.json", manifest)
    return manifest


def _validate_bound_file(bound: Mapping[str, Any], root: Path) -> Path:
    path = root / str(bound["relative_path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(bound["size_bytes"])
        or _sha256_file(path) != bound["sha256"]
    ):
        raise ContractError("A-R9 hash-bound dataset asset changed")
    return path


class _ManifestDataset:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        dataset_root: Path,
        *,
        train_mode: bool,
        flip_probability: float,
        verify_assets: bool = False,
    ) -> None:
        self.records = list(records)
        self.dataset_root = dataset_root
        self.train_mode = train_mode
        self.flip_probability = flip_probability
        self.verify_assets = verify_assets

    def _asset_path(self, bound: Mapping[str, Any]) -> Path:
        if self.verify_assets:
            return _validate_bound_file(bound, self.dataset_root)
        path = self.dataset_root / str(bound["relative_path"])
        if not path.is_file():
            raise ContractError("A-R9 manifest asset disappeared")
        return path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        record = self.records[index]
        rgb_path = self._asset_path(record["rgb"])
        with Image.open(rgb_path) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        for instance in record["instances"]:
            mask_path = self._asset_path(instance["mask"])
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 0
            masks.append(mask)
            boxes.append([float(value) for value in instance["box_xyxy"]])
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if self.train_mode and random.random() < self.flip_probability:
            tensor = torch.flip(tensor, dims=(2,))
            mask_tensor = torch.flip(mask_tensor, dims=(2,))
            width = tensor.shape[2]
            old_left = box_tensor[:, 0].clone()
            old_right = box_tensor[:, 2].clone()
            box_tensor[:, 0] = width - old_right
            box_tensor[:, 2] = width - old_left
        target = {
            "boxes": box_tensor,
            "labels": torch.ones((len(masks),), dtype=torch.int64),
            "masks": mask_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": mask_tensor.flatten(1).sum(1).to(torch.float32),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        return tensor, target


def _collate(
    batch: Sequence[tuple[Any, dict[str, Any]]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _validate_labeled_records(
    records: Sequence[Mapping[str, Any]], dataset_root: Path
) -> None:
    for record in records:
        _validate_bound_file(record["rgb"], dataset_root)
        for instance in record["instances"]:
            _validate_bound_file(instance["mask"], dataset_root)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _freeze_resnet_backbone(model: Any, trainable_layers: int) -> None:
    """Mirror torchvision's pretrained ResNet-FPN backbone freeze policy.

    The official factory only applies ``trainable_backbone_layers`` when it is
    also responsible for loading the pretrained weights.  A-R9 loads a
    hash-bound local state dict instead, so the same policy must be applied
    explicitly after construction.
    """

    if not 0 <= trainable_layers <= 5:
        raise ContractError("A-R9 trainable backbone layer count changed")
    train_order = ["layer4", "layer3", "layer2", "layer1", "conv1"]
    layers_to_train = train_order[:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(
            any(name.startswith(layer) for layer in layers_to_train)
        )


def _build_model(protocol: Mapping[str, Any], weight_path: Path | None) -> Any:
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    config = protocol["model"]
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=int(config["minimum_image_side"]),
        max_size=int(config["maximum_image_side"]),
    )
    _freeze_resnet_backbone(model, int(config["trainable_backbone_layers"]))
    if weight_path is not None:
        weight = config["official_pretrained_weight"]
        if (
            not weight_path.is_file()
            or weight_path.stat().st_size != int(weight["size_bytes"])
            or _sha256_file(weight_path) != weight["sha256"]
        ):
            raise ContractError("A-R9 official pretrained weight changed")
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        box_features = model.roi_heads.box_predictor.cls_score.in_features
        mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, 2)
    model.roi_heads.detections_per_img = int(config["detections_per_image"])
    model.roi_heads.score_thresh = float(config["prediction_score_floor"])
    return model


def _load_manifest(path: Path, protocol_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("protocol_sha256") != _sha256_file(protocol_path)
        or value.get("train_frame_count") != 400
        or value.get("internal_validation_frame_count") != 100
        or value.get("fixed_evaluation_frame_count") != 25
        or value.get("fixed_evaluation_gt_access_count") != 0
        or value.get("scene9_read_count") != 0
    ):
        raise ContractError("A-R9 dataset manifest changed")
    frames = value.get("frames")
    if not isinstance(frames, dict):
        raise ContractError("A-R9 dataset frames are missing")
    for split, expected in (
        ("train", _frame_specs("train")),
        ("internal_validation", _frame_specs("internal_validation")),
        ("fixed_evaluation", _frame_specs("fixed_evaluation")),
    ):
        rows = frames.get(split)
        if (
            not isinstance(rows, list)
            or [(int(row["scene_id"]), int(row["image_id"])) for row in rows]
            != expected
        ):
            raise ContractError(f"A-R9 {split} coverage changed")
    return value


def _targets_to_cpu(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in row.items()
        }
        for row in targets
    ]


def _outputs_to_metric_frames(
    frame_ids: Sequence[str],
    outputs: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_id, output, target in zip(frame_ids, outputs, targets, strict=True):
        scores = output["scores"].detach().cpu().numpy()
        selected = scores >= score_threshold
        masks = output["masks"].detach().cpu().numpy()[:, 0] >= mask_threshold
        frames.append(
            {
                "frame_id": frame_id,
                "gt_masks": [row.astype(bool) for row in target["masks"].numpy()],
                "pred_masks": [row for row in masks[selected]],
                "scores": [float(value) for value in scores[selected]],
            }
        )
    return frames


def _evaluate_model_on_manifest(
    model: Any,
    records: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    protocol: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    dataset = _ManifestDataset(
        records,
        dataset_root,
        train_mode=False,
        flip_probability=0.0,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=_collate
    )
    model.eval()
    frames: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            outputs = model([image.to(device) for image in images])
            frames.extend(
                _outputs_to_metric_frames(
                    [str(records[index]["frame_id"])],
                    outputs,
                    _targets_to_cpu(targets),
                    score_threshold=float(
                        protocol["model"]["operating_score_threshold"]
                    ),
                    mask_threshold=float(
                        protocol["model"]["mask_probability_threshold"]
                    ),
                )
            )
    return a8.evaluate_predictions(frames)


def train(
    *,
    protocol_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    weight_path: Path,
    output_root: Path,
    device_name: str,
    smoke_batches: int | None = None,
) -> dict[str, Any]:
    """Fine-tune one class-agnostic Mask R-CNN exactly once."""

    import torch
    from torch.utils.data import DataLoader

    protocol = load_protocol(protocol_path)
    manifest = _load_manifest(dataset_manifest_path, protocol_path)
    if output_root.exists():
        raise ContractError("A-R9 training output root is create-only")
    output_root.mkdir(parents=True)
    training = protocol["training"]
    if device_name.startswith("cuda") and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != str(training["cublas_workspace_config"]):
        raise ContractError("A-R9 CUBLAS_WORKSPACE_CONFIG is not frozen")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    _validate_labeled_records(manifest["frames"]["train"], dataset_root)
    _validate_labeled_records(manifest["frames"]["internal_validation"], dataset_root)
    generator = torch.Generator().manual_seed(seed)
    dataset = _ManifestDataset(
        manifest["frames"]["train"],
        dataset_root,
        train_mode=True,
        flip_probability=float(training["horizontal_flip_probability"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["physical_batch_size"]),
        shuffle=True,
        num_workers=int(training["workers"]),
        collate_fn=_collate,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=True,
    )
    device = torch.device(device_name)
    model = _build_model(protocol, weight_path).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    accumulation = int(training["gradient_accumulation"])
    epochs = 1 if smoke_batches is not None else int(training["epochs"])
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        optimizer_steps = 0
        for batch_index, (images, targets) in enumerate(loader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in row.items()
                }
                for row in targets
            ]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise ContractError("A-R9 training loss became non-finite")
            scaler.scale(loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                loader
            )
            if boundary:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if smoke_batches is not None and batch_index + 1 >= smoke_batches:
                break
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "mean_total_loss": float(np.mean(losses)),
            "batch_count": len(losses),
            "optimizer_step_count": optimizer_steps,
            "learning_rate_after_epoch": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "protocol_sha256": _sha256_file(protocol_path),
            "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
            "official_weight_sha256": _sha256_file(weight_path),
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
        }
        _atomic_torch_save(
            output_root / f"checkpoint-epoch-{epoch + 1:02d}.pt", checkpoint, torch
        )
    final_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "official_weight_sha256": _sha256_file(weight_path),
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
    }
    final_path = output_root / "model-final.pt"
    _atomic_torch_save(final_path, final_checkpoint, torch)
    validation_metrics = None
    if smoke_batches is None:
        health_image_ids = set(training["internal_validation_health_image_ids"])
        health_records = [
            record
            for record in manifest["frames"]["internal_validation"]
            if int(record["image_id"]) in health_image_ids
        ]
        validation_metrics = _evaluate_model_on_manifest(
            model,
            health_records,
            dataset_root,
            protocol,
            device,
        )
    result = {
        "schema_version": "poseloop.a-r9.mask-rcnn-training-result.v1",
        "mode": "SMOKE_NOT_DECISION_ELIGIBLE"
        if smoke_batches is not None
        else "FORMAL_OBJECT_DISJOINT_TRAINING",
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "official_weight_sha256": _sha256_file(weight_path),
        "epochs_completed": epochs,
        "history": history,
        "internal_validation_metrics": validation_metrics,
        "checkpoint": {
            "relative_path": final_path.relative_to(output_root).as_posix(),
            "size_bytes": final_path.stat().st_size,
            "sha256": _sha256_file(final_path),
        },
        "wall_time_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "scene9_read_count": 0,
    }
    _write_json(output_root / "training-result.json", result)
    return result


def _pack_masks(
    path: Path, masks: Sequence[np.ndarray], shape: tuple[int, int]
) -> None:
    flat_pixels = int(shape[0]) * int(shape[1])
    if masks:
        stack = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
        if stack.shape[1:] != shape:
            raise ContractError("A-R9 prediction mask shape changed")
    else:
        stack = np.zeros((0, *shape), dtype=bool)
    packed = np.packbits(stack.reshape((len(stack), flat_pixels)), axis=1)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed=packed,
            count=np.asarray([len(stack)], dtype=np.int32),
            height_width=np.asarray(shape, dtype=np.int32),
        )
    os.replace(temporary, path)


def _unpack_masks(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        height, width = (int(value) for value in bundle["height_width"])
        count = int(bundle["count"][0])
        if count == 0:
            return []
        values = np.unpackbits(bundle["packed"], axis=1, count=height * width)
        return [row.reshape(height, width).astype(bool) for row in values[:count]]


def _load_checkpoint(
    checkpoint_path: Path,
    protocol_path: Path,
    dataset_manifest_path: Path,
    protocol: Mapping[str, Any],
    device: Any,
) -> Any:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("protocol_sha256") != _sha256_file(protocol_path)
        or checkpoint.get("dataset_manifest_sha256")
        != _sha256_file(dataset_manifest_path)
        or checkpoint.get("epoch") != int(protocol["training"]["epochs"])
    ):
        raise ContractError("A-R9 checkpoint identity changed")
    model = _build_model(protocol, None)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_channels, 256, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def predict(
    *,
    protocol_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run label-blind inference on the frozen 25 evaluation frames."""

    import torch
    from torchvision.transforms.functional import pil_to_tensor

    protocol = load_protocol(protocol_path)
    manifest = _load_manifest(dataset_manifest_path, protocol_path)
    if output_root.exists():
        raise ContractError("A-R9 prediction output root is create-only")
    output_root.mkdir(parents=True)
    frame_root = output_root / "frames"
    frame_root.mkdir()
    device = torch.device(device_name)
    model = _load_checkpoint(
        checkpoint_path,
        protocol_path,
        dataset_manifest_path,
        protocol,
        device,
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in manifest["frames"]["fixed_evaluation"]:
            rgb_path = _validate_bound_file(record["rgb"], dataset_root)
            with Image.open(rgb_path) as image:
                image_tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
            output = model([image_tensor.to(device)])[0]
            scores = output["scores"].detach().cpu().numpy()
            selected = scores >= float(protocol["model"]["prediction_score_floor"])
            masks = output["masks"].detach().cpu().numpy()[:, 0]
            masks = masks[selected] >= float(
                protocol["model"]["mask_probability_threshold"]
            )
            boxes = output["boxes"].detach().cpu().numpy()[selected]
            scores = scores[selected]
            frame_id = str(record["frame_id"])
            mask_path = frame_root / f"{frame_id}.npz"
            metadata_path = frame_root / f"{frame_id}.json"
            _pack_masks(
                mask_path, list(masks), (int(record["height"]), int(record["width"]))
            )
            _write_json(
                metadata_path,
                {
                    "frame_id": frame_id,
                    "scene_id": int(record["scene_id"]),
                    "image_id": int(record["image_id"]),
                    "scores": [float(value) for value in scores],
                    "boxes_xyxy": [[float(value) for value in row] for row in boxes],
                    "prediction_count": len(scores),
                },
            )
            rows.append(
                {
                    "frame_id": frame_id,
                    "scene_id": int(record["scene_id"]),
                    "image_id": int(record["image_id"]),
                    "metadata": {
                        "relative_path": metadata_path.relative_to(
                            output_root
                        ).as_posix(),
                        "size_bytes": metadata_path.stat().st_size,
                        "sha256": _sha256_file(metadata_path),
                    },
                    "masks": {
                        "relative_path": mask_path.relative_to(output_root).as_posix(),
                        "size_bytes": mask_path.stat().st_size,
                        "sha256": _sha256_file(mask_path),
                    },
                }
            )
            print(
                json.dumps({"frame_id": frame_id, "prediction_count": len(scores)}),
                flush=True,
            )
    prediction_manifest = {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "frame_count": len(rows),
        "fixed_evaluation_gt_access_count": 0,
        "scene9_read_count": 0,
        "frames": rows,
    }
    _write_json(output_root / "prediction-manifest.json", prediction_manifest)
    return prediction_manifest


def _load_gt(dataset_root: Path, scene_id: int, image_id: int) -> list[np.ndarray]:
    scene = dataset_root / "val" / f"{scene_id:06d}"
    gt = json.loads((scene / "scene_gt_realsense.json").read_text(encoding="utf-8"))
    rows = gt.get(str(image_id))
    if not isinstance(rows, list) or not rows:
        raise ContractError("A-R9 evaluation GT is missing")
    masks: list[np.ndarray] = []
    for gt_index, row in enumerate(rows):
        if int(row.get("obj_id", -1)) != EXPECTED_OBJECT_IDS[scene_id]:
            raise ContractError("A-R9 evaluation scene/object identity changed")
        path = scene / "mask_visib_realsense" / f"{image_id:06d}_{gt_index:06d}.png"
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.any():
            masks.append(mask)
    return masks


def _detector_metrics(
    frames: Sequence[Mapping[str, Any]], *, operating_threshold: float
) -> dict[str, Any]:
    operating = []
    for frame in frames:
        keep = [score >= operating_threshold for score in frame["scores"]]
        operating.append(
            {
                "frame_id": frame["frame_id"],
                "gt_masks": frame["gt_masks"],
                "pred_masks": [
                    mask
                    for mask, selected in zip(frame["pred_masks"], keep, strict=True)
                    if selected
                ],
                "scores": [
                    score
                    for score, selected in zip(frame["scores"], keep, strict=True)
                    if selected
                ],
            }
        )
    metrics = a8.evaluate_predictions(operating)
    ranked = a8.evaluate_predictions(frames)
    metrics["ap50"] = ranked["ap50"]
    metrics["ap75"] = ranked["ap75"]
    metrics["operating_score_threshold"] = operating_threshold
    metrics["ranked_prediction_count"] = ranked["prediction_count"]
    return metrics


def _paired_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, seed: int, draws: int
) -> dict[str, Any]:
    left = {row["frame_id"]: row for row in baseline["per_frame"]}
    right = {row["frame_id"]: row for row in candidate["per_frame"]}
    if set(left) != set(right):
        raise ContractError("A-R9 paired frames changed")
    deltas = [
        float(right[key]["f1_iou50"]) - float(left[key]["f1_iou50"])
        for key in sorted(left)
    ]
    return {
        "mean_frame_f1_delta": float(np.mean(deltas)),
        "bootstrap_95_interval": a8._bootstrap_interval(deltas, seed=seed, draws=draws),
        "positive_frame_count": sum(value > 0 for value in deltas),
    }


def _positive_scene_count(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> int:
    left = {row["frame_id"]: row for row in baseline["per_frame"]}
    right = {row["frame_id"]: row for row in candidate["per_frame"]}
    count = 0
    for scene_id in EVALUATION_SCENES:
        prefix = f"s{scene_id:06d}-"
        keys = [key for key in left if key.startswith(prefix)]
        count += int(
            np.mean([right[key]["f1_iou50"] for key in keys])
            > np.mean([left[key]["f1_iou50"] for key in keys])
        )
    return count


def _outline(
    image: np.ndarray, masks: Sequence[np.ndarray], color: tuple[int, int, int]
) -> np.ndarray:
    return a8._outline(image, masks, color)


def _write_visual(
    path: Path,
    rgb: np.ndarray,
    gt_masks: Sequence[np.ndarray],
    raw_masks: Sequence[np.ndarray],
    detector_masks: Sequence[np.ndarray],
) -> None:
    panels = [
        ("ground truth", _outline(rgb, gt_masks, (0, 255, 0))),
        ("raw FastSAM", _outline(rgb, raw_masks, (0, 180, 255))),
        ("trained Mask R-CNN", _outline(rgb, detector_masks, (255, 210, 0))),
    ]
    height, width = rgb.shape[:2]
    canvas = Image.new("RGB", (width * 3, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas.paste(Image.fromarray(panel), (x, 0))
        draw.rectangle((x, 0, x + 430, 34), fill=(0, 0, 0))
        draw.text((x + 8, 8), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(
    *,
    protocol_path: Path,
    dataset_root: Path,
    dataset_manifest_path: Path,
    predictions_root: Path,
    a8_result_path: Path,
    a8_predictions_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Open development labels after inference and make the route decision."""

    protocol = load_protocol(protocol_path)
    manifest = _load_manifest(dataset_manifest_path, protocol_path)
    if output_root.exists():
        raise ContractError("A-R9 evaluation output root is create-only")
    if _sha256_file(a8_result_path) != A8_RESULT_SHA256:
        raise ContractError("A-R9 frozen A-R8 result changed")
    a8_result = json.loads(a8_result_path.read_text(encoding="utf-8"))
    a8_manifest_path = a8_predictions_root / "prediction-manifest.json"
    if _sha256_file(a8_manifest_path) != A8_PREDICTION_MANIFEST_SHA256:
        raise ContractError("A-R9 frozen A-R8 predictions changed")
    a8_manifest = json.loads(a8_manifest_path.read_text(encoding="utf-8"))
    a8_rows = {row["frame_id"]: row for row in a8_manifest["frames"]}
    prediction_manifest_path = predictions_root / "prediction-manifest.json"
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    if (
        prediction_manifest.get("schema_version") != PREDICTION_SCHEMA
        or prediction_manifest.get("protocol_sha256") != _sha256_file(protocol_path)
        or prediction_manifest.get("dataset_manifest_sha256")
        != _sha256_file(dataset_manifest_path)
        or prediction_manifest.get("frame_count") != 25
        or prediction_manifest.get("fixed_evaluation_gt_access_count") != 0
        or prediction_manifest.get("scene9_read_count") != 0
    ):
        raise ContractError("A-R9 prediction manifest changed")
    predicted_rows = {row["frame_id"]: row for row in prediction_manifest["frames"]}
    output_root.mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    for source_record in manifest["frames"]["fixed_evaluation"]:
        frame_id = str(source_record["frame_id"])
        row = predicted_rows.get(frame_id)
        if row is None:
            raise ContractError("A-R9 prediction frame is missing")
        metadata_path = predictions_root / row["metadata"]["relative_path"]
        masks_path = predictions_root / row["masks"]["relative_path"]
        if (
            _sha256_file(metadata_path) != row["metadata"]["sha256"]
            or _sha256_file(masks_path) != row["masks"]["sha256"]
        ):
            raise ContractError("A-R9 prediction artifact changed")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pred_masks = _unpack_masks(masks_path)
        scores = [float(value) for value in metadata["scores"]]
        if len(pred_masks) != len(scores):
            raise ContractError("A-R9 prediction score/mask count differs")
        scene_id = int(source_record["scene_id"])
        image_id = int(source_record["image_id"])
        gt_masks = _load_gt(dataset_root, scene_id, image_id)
        frames.append(
            {
                "frame_id": frame_id,
                "gt_masks": gt_masks,
                "pred_masks": pred_masks,
                "scores": scores,
            }
        )
        a8_row = a8_rows[frame_id]
        a8_masks_path = a8_predictions_root / a8_row["mask_bundle"]["relative_path"]
        if _sha256_file(a8_masks_path) != a8_row["mask_bundle"]["sha256"]:
            raise ContractError("A-R9 frozen A-R8 mask bundle changed")
        raw_masks = a8._unpack_masks(a8_masks_path, "raw_fastsam_eligible")
        operating = [
            mask
            for mask, score in zip(pred_masks, scores, strict=True)
            if score >= float(protocol["model"]["operating_score_threshold"])
        ]
        rgb_path = _validate_bound_file(source_record["rgb"], dataset_root)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        _write_visual(
            output_root / "visualizations" / f"{frame_id}.png",
            rgb,
            gt_masks,
            raw_masks,
            operating,
        )
    candidate_metrics = _detector_metrics(
        frames,
        operating_threshold=float(protocol["model"]["operating_score_threshold"]),
    )
    raw_metrics = a8_result["metrics"]["raw_fastsam_eligible"]
    gate = protocol["promotion_gate"]
    paired = _paired_delta(
        raw_metrics,
        candidate_metrics,
        seed=int(gate["bootstrap_seed"]),
        draws=int(gate["bootstrap_draws"]),
    )
    positive_scene_count = _positive_scene_count(raw_metrics, candidate_metrics)
    checks = {
        "paired_frame_f1_bootstrap_lower_gt_zero": paired["bootstrap_95_interval"][0]
        > 0,
        "positive_scene_count_minimum": positive_scene_count
        >= int(gate["positive_scene_count_minimum"]),
        "recall_strictly_above_raw_fastsam": candidate_metrics["instance_recall_iou50"]
        > raw_metrics["instance_recall_iou50"],
        "ap50_strictly_above_raw_fastsam": candidate_metrics["ap50"]
        > raw_metrics["ap50"],
        "pq_strictly_above_raw_fastsam": candidate_metrics["pq_iou50"]
        > raw_metrics["pq_iou50"],
    }
    positive = all(checks.values())
    result = {
        "schema_version": RESULT_SCHEMA,
        "protocol_sha256": _sha256_file(protocol_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
        "split_policy": "SCENE_AND_OBJECT_DISJOINT",
        "frame_count": 25,
        "labels_opened_during_inference": False,
        "labels_opened_during_evaluation": True,
        "scene9_read_count": 0,
        "raw_fastsam_metrics": raw_metrics,
        "mask_rcnn_metrics": candidate_metrics,
        "paired_frame_f1_delta": paired,
        "positive_scene_count": positive_scene_count,
        "promotion_checks": checks,
        "route_decision": (
            "REAL_DEVELOPMENT_DETECTOR_POSITIVE_CONTINUE"
            if positive
            else "MASK_RCNN_NO_GO_MOVE_TO_3D_INSTANCE_PROPOSALS"
        ),
        "sealed_or_scene9_replay_permitted": False,
    }
    _write_json(output_root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("protocol-check")
    check.add_argument("--protocol", type=Path, required=True)
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--protocol", type=Path, required=True)
    training.add_argument("--dataset-root", type=Path, required=True)
    training.add_argument("--dataset-manifest", type=Path, required=True)
    training.add_argument("--weight", type=Path, required=True)
    training.add_argument("--output-root", type=Path, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--smoke-batches", type=int)
    inference = subparsers.add_parser("predict")
    inference.add_argument("--protocol", type=Path, required=True)
    inference.add_argument("--dataset-root", type=Path, required=True)
    inference.add_argument("--dataset-manifest", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output-root", type=Path, required=True)
    inference.add_argument("--device", default="cuda:0")
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--protocol", type=Path, required=True)
    evaluator.add_argument("--dataset-root", type=Path, required=True)
    evaluator.add_argument("--dataset-manifest", type=Path, required=True)
    evaluator.add_argument("--predictions-root", type=Path, required=True)
    evaluator.add_argument("--a8-result", type=Path, required=True)
    evaluator.add_argument("--a8-predictions-root", type=Path, required=True)
    evaluator.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "protocol-check":
        protocol = load_protocol(args.protocol)
        print(
            json.dumps(
                {
                    "status": "PASS_A_R9_PROTOCOL",
                    "protocol_id": protocol["protocol_id"],
                    "train_scene_count": len(TRAIN_SCENES),
                    "internal_validation_scene_count": len(VALIDATION_SCENES),
                    "fixed_evaluation_scene_count": len(EVALUATION_SCENES),
                    "scene9_read_permitted": False,
                },
                sort_keys=True,
            )
        )
    elif args.command == "build-dataset":
        result = build_dataset_manifest(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "status": "DATASET_FROZEN",
                    "train_frames": result["train_frame_count"],
                }
            )
        )
    elif args.command == "train":
        result = train(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            weight_path=args.weight,
            output_root=args.output_root,
            device_name=args.device,
            smoke_batches=args.smoke_batches,
        )
        print(
            json.dumps(
                {"status": result["mode"], "checkpoint": result["checkpoint"]},
                sort_keys=True,
            )
        )
    elif args.command == "predict":
        result = predict(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            checkpoint_path=args.checkpoint,
            output_root=args.output_root,
            device_name=args.device,
        )
        print(json.dumps({"status": "PREDICTED", "frame_count": result["frame_count"]}))
    elif args.command == "evaluate":
        result = evaluate(
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            predictions_root=args.predictions_root,
            a8_result_path=args.a8_result,
            a8_predictions_root=args.a8_predictions_root,
            output_root=args.output_root,
        )
        print(json.dumps({"status": result["route_decision"]}, sort_keys=True))
    else:
        raise AssertionError(args.command)
    return 0
