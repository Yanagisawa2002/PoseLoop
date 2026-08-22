"""Executable A-R7 synthetic occlusion gate.

This successor runtime deliberately has no argument for the ten frozen replay
frames.  It renders deterministic scenes from the already frozen A-R5 RGBA CAD
templates, runs the exact A-R5 FastSAM/DINOv2 assets, selects the A-R6 parent,
and then asks the pinned SAM2.1 image predictor to deblend that parent from raw
synthetic depth discontinuities.  Every SAM2 child is scored against the full
five-object CAD descriptor catalogue before the paired content gate is built.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.adapter import (
    OfficialCnosAdapter,
    mask_stability_from_probabilities,
    store_raw_cosine,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.adapter import (
    audit_runtime_module_origins,
    ensure_ultralytics_yolo_compat,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.contracts import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    load_source_execution_locks,
    mask_statistics,
)
from pose_accuracy_recovery_prep.instance_selection_v1r6 import (
    POLICY as A_R6_POLICY,
    select_frame,
)

from .adapter import (
    DEPTH_EDGE_NOISE_MULTIPLIER,
    derive_depth_discontinuity_seeds,
)
from .contracts import SYNTHETIC_STRATA
from .core import (
    DeblendCandidate,
    evaluate_instance_set,
    partition_prompted_candidates,
)
from .execution_contract import validate_execution_protocol


sys.dont_write_bytecode = True


SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_TREE = "64becbca23f880e0056449377496da248a74da43"
SAM2_ARCHIVE_SHA256 = "a9b182a4e502160a22226a32f1db4c31ea6a93000ca0f4502ddebdb7682d9209"
SAM2_SOURCE_MANIFEST_SHA256 = (
    "d7ffdda54476ef486083e4cd3e7ba89a39c4fabc030315f362da39dabbdd1d4f"
)
SAM2_CONFIG_RELATIVE = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CONFIG_BYTES = 3798
SAM2_CONFIG_SHA256 = "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
SAM2_CHECKPOINT_BYTES = 898_083_611
SAM2_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)

A_R5_REQUEST_RELATIVE = "contracts/runtime-request.json"
A_R5_REQUEST_BYTES = 11_033
A_R5_REQUEST_SHA256 = "2d1220987eee7cd32e7feb917363a4080116fa38853fc857d027e8d70ed30469"
A_R5_TEMPLATE_MANIFEST_RELATIVE = "contracts/rgba-template-manifest.json"
A_R5_TEMPLATE_MANIFEST_BYTES = 82_509
A_R5_TEMPLATE_MANIFEST_SHA256 = (
    "6cc2c958401a209ac3b69f3c93c730dc9c7d2b111a122d911bac9cdfc395b031"
)
A_R5_CONFIG_RELATIVE = "contracts/cnos_catalog_config_v1r5.json"
A_R5_CONFIG_BYTES = 1927
A_R5_CONFIG_SHA256 = "b1ee1d3638b3ce9f1659118c3b7f243fea8c35bfcd3640a4fc62f58c48207898"

DEVELOPMENT_ROWS_PER_STRATUM = 8
DEVELOPMENT_FIRST_SEED = 4096
BOUNDARY_ZERO = {
    "post_freeze_replay_rgb_open_count": 0,
    "post_freeze_replay_depth_open_count": 0,
    "post_freeze_replay_prediction_open_count": 0,
    "gt_pose_open_count": 0,
    "gt_mask_open_count": 0,
    "evaluator_open_count": 0,
    "foundationpose_run_count": 0,
    "official_scorer_run_count": 0,
    "downstream_export_count": 0,
}


@dataclass(frozen=True)
class SyntheticScene:
    item_id: str
    stratum: str
    seed: int
    rgb: np.ndarray
    depth_mm: np.ndarray
    visible_masks: tuple[np.ndarray, ...]
    object_ids: tuple[int, ...]
    template_views: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeProposal:
    proposal_index: int
    mask: np.ndarray
    proposal_score: float
    mask_stability: float
    cad_ranking: tuple[dict[str, Any], ...]


def _asset(path: Path, *, role: str, root: Path) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_file(
    path: Path, *, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha256
    ):
        raise ContractError(f"A-R7 frozen {label} changed")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_sam2_freeze(sam2_asset_root: Path) -> dict[str, Any]:
    checkout = (sam2_asset_root / "source").resolve()
    checkpoint = (sam2_asset_root / "sam2.1_hiera_large.pt").resolve()
    config = checkout / SAM2_CONFIG_RELATIVE
    manifest = sam2_asset_root / "source_files.sha256"
    if _git(checkout, "rev-parse", "HEAD") != SAM2_COMMIT:
        raise ContractError("A-R7 SAM2 commit changed")
    if _git(checkout, "rev-parse", "HEAD^{tree}") != SAM2_TREE:
        raise ContractError("A-R7 SAM2 tree changed")
    if _git(checkout, "status", "--porcelain"):
        raise ContractError("A-R7 SAM2 checkout is not all-clean")
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(archive).hexdigest() != SAM2_ARCHIVE_SHA256:
        raise ContractError("A-R7 SAM2 source archive identity changed")
    _assert_file(
        checkpoint,
        expected_bytes=SAM2_CHECKPOINT_BYTES,
        expected_sha256=SAM2_CHECKPOINT_SHA256,
        label="checkpoint",
    )
    _assert_file(
        config,
        expected_bytes=SAM2_CONFIG_BYTES,
        expected_sha256=SAM2_CONFIG_SHA256,
        label="model config",
    )
    if not manifest.is_file() or sha256_file(manifest) != SAM2_SOURCE_MANIFEST_SHA256:
        raise ContractError("A-R7 SAM2 source manifest changed")
    return {
        "repository": "https://github.com/facebookresearch/sam2",
        "commit": SAM2_COMMIT,
        "tree": SAM2_TREE,
        "source_archive_sha256": SAM2_ARCHIVE_SHA256,
        "source_manifest_sha256": SAM2_SOURCE_MANIFEST_SHA256,
        "config": {
            "relative_path": SAM2_CONFIG_RELATIVE,
            "bytes": SAM2_CONFIG_BYTES,
            "sha256": SAM2_CONFIG_SHA256,
        },
        "checkpoint": {
            "bytes": SAM2_CHECKPOINT_BYTES,
            "sha256": SAM2_CHECKPOINT_SHA256,
        },
    }


def _resolve_bound(root: Path, record: Mapping[str, Any], label: str) -> Path:
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ContractError(f"A-R7 {label} path is invalid")
    path = (root / Path(*relative.split("/"))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"A-R7 {label} escapes data root") from exc
    _assert_file(
        path,
        expected_bytes=int(record.get("bytes", -1)),
        expected_sha256=str(record.get("sha256", "")),
        label=label,
    )
    return path


def load_a_r5_assets(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = data_root / A_R5_REQUEST_RELATIVE
    manifest_path = data_root / A_R5_TEMPLATE_MANIFEST_RELATIVE
    config_path = data_root / A_R5_CONFIG_RELATIVE
    _assert_file(
        request_path,
        expected_bytes=A_R5_REQUEST_BYTES,
        expected_sha256=A_R5_REQUEST_SHA256,
        label="A-R5 runtime request",
    )
    _assert_file(
        manifest_path,
        expected_bytes=A_R5_TEMPLATE_MANIFEST_BYTES,
        expected_sha256=A_R5_TEMPLATE_MANIFEST_SHA256,
        label="A-R5 template manifest",
    )
    _assert_file(
        config_path,
        expected_bytes=A_R5_CONFIG_BYTES,
        expected_sha256=A_R5_CONFIG_SHA256,
        label="A-R5 adapter config",
    )
    request = read_json(request_path)
    manifest = read_json(manifest_path)
    if not isinstance(request, dict) or not isinstance(manifest, dict):
        raise ContractError("A-R7 A-R5 assets are not JSON objects")
    if request.get("adapter_config", {}).get("sha256") != A_R5_CONFIG_SHA256:
        raise ContractError("A-R7 A-R5 config cross-lock changed")
    catalog = request.get("catalog")
    if not isinstance(catalog, list) or [row.get("object_id") for row in catalog] != [
        1,
        2,
        4,
        5,
        6,
    ]:
        raise ContractError("A-R7 A-R5 CAD catalogue changed")
    for source_name in ("cnos_archive", "dinov2_archive"):
        _resolve_bound(data_root, request["source"][source_name], source_name)
    for prefix in ("cnos", "dinov2"):
        checkout = (
            data_root / request["source"][f"{prefix}_checkout_relative_path"]
        ).resolve()
        if (
            _git(checkout, "rev-parse", "HEAD") != request["source"][f"{prefix}_commit"]
            or _git(checkout, "rev-parse", "HEAD^{tree}")
            != request["source"][f"{prefix}_tree"]
            or _git(checkout, "status", "--porcelain")
        ):
            raise ContractError(f"A-R7 frozen {prefix} checkout changed")
    for model_name in ("fastsam_x_checkpoint", "dinov2_vitl14_checkpoint"):
        _resolve_bound(data_root, request["models"][model_name], model_name)
    for row in catalog:
        _resolve_bound(data_root, row["cad"], "catalog CAD")
        _resolve_bound(data_root, row["descriptor"], "catalog descriptor")
    return request, manifest


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:FRAME_HEIGHT, :FRAME_WIDTH]
    base = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
    base[..., 0] = 52 + 15 * x / FRAME_WIDTH + 7 * y / FRAME_HEIGHT
    base[..., 1] = 57 + 10 * x / FRAME_WIDTH + 9 * y / FRAME_HEIGHT
    base[..., 2] = 61 + 8 * x / FRAME_WIDTH + 12 * y / FRAME_HEIGHT
    base += rng.normal(0.0, 2.0, base.shape)
    for coordinate in range(0, FRAME_WIDTH, 180):
        base[:, coordinate : coordinate + 2] += 12
    for coordinate in range(0, FRAME_HEIGHT, 180):
        base[coordinate : coordinate + 2, :] += 12
    return np.clip(base, 0, 255).astype(np.uint8)


def _template_record(
    manifest: Mapping[str, Any], object_id: int, view_index: int
) -> Mapping[str, Any]:
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise ContractError("A-R7 template manifest objects changed")
    for row in objects:
        if row.get("object_id") == object_id:
            views = row.get("views")
            if not isinstance(views, list) or len(views) != 42:
                raise ContractError("A-R7 template view coverage changed")
            record = views[view_index]
            if record.get("view_index") != view_index:
                raise ContractError("A-R7 template view ordering changed")
            return record
    raise ContractError("A-R7 template object is absent")


def _scaled_patch(
    *, data_root: Path, record: Mapping[str, Any], long_edge: int
) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_bound(data_root, record["rgba"], "RGBA template")
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3] > 0
    if not alpha.any():
        raise ContractError("A-R7 RGBA template is empty")
    ys, xs = np.where(alpha)
    crop = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape[:2]
    scale = long_edge / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = np.asarray(
        Image.fromarray(crop, mode="RGBA").resize(size, Image.Resampling.BICUBIC)
    )
    return resized[..., :3], resized[..., 3] >= 128


def _paste_layer(
    *,
    canvas_shape: tuple[int, int],
    rgb: np.ndarray,
    alpha: np.ndarray,
    center: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = canvas_shape
    patch_height, patch_width = alpha.shape
    left = int(center[0] - patch_width // 2)
    top = int(center[1] - patch_height // 2)
    right = left + patch_width
    bottom = top + patch_height
    target_left = max(0, left)
    target_top = max(0, top)
    target_right = min(width, right)
    target_bottom = min(height, bottom)
    if target_left >= target_right or target_top >= target_bottom:
        raise ContractError("A-R7 synthetic object is outside the frame")
    source_left = target_left - left
    source_top = target_top - top
    source_right = source_left + target_right - target_left
    source_bottom = source_top + target_bottom - target_top
    full_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    full_mask = np.zeros((height, width), dtype=bool)
    full_rgb[target_top:target_bottom, target_left:target_right] = rgb[
        source_top:source_bottom, source_left:source_right
    ]
    full_mask[target_top:target_bottom, target_left:target_right] = alpha[
        source_top:source_bottom, source_left:source_right
    ]
    if int(full_mask.sum()) < 500:
        raise ContractError("A-R7 synthetic visible source patch is too small")
    return full_rgb, full_mask


def generate_scene(
    *,
    stratum: str,
    seed: int,
    data_root: Path,
    template_manifest: Mapping[str, Any],
) -> SyntheticScene:
    if stratum not in SYNTHETIC_STRATA:
        raise ContractError("A-R7 synthetic stratum is unknown")
    rng = np.random.default_rng(seed)
    catalog = (1, 2, 4, 5, 6)
    if stratum == "clean_single_instance":
        object_ids = (int(rng.choice(catalog)),)
        centers = ((FRAME_WIDTH // 2, FRAME_HEIGHT // 2),)
        edges = (360,)
    elif stratum == "touching_pair_different_cad":
        object_ids = tuple(
            int(value) for value in rng.choice(catalog, 2, replace=False)
        )
        centers = ((680, 535), (760, 550))
        edges = (430, 410)
    elif stratum == "touching_pair_same_cad":
        object_id = int(rng.choice(catalog))
        object_ids = (object_id, object_id)
        centers = ((680, 535), (760, 550))
        edges = (430, 410)
    elif stratum == "depth_ordered_occlusion_pair":
        object_ids = tuple(
            int(value) for value in rng.choice(catalog, 2, replace=False)
        )
        centers = ((680, 535), (760, 550))
        edges = (430, 410)
    elif stratum == "same_cad_pile_three_plus":
        object_id = int(rng.choice(catalog))
        object_ids = (object_id, object_id, object_id)
        centers = ((660, 520), (720, 560), (780, 520))
        edges = (420, 430, 420)
    else:
        object_ids = tuple(
            int(value) for value in rng.choice(catalog, 2, replace=False)
        )
        centers = ((680, 535), (760, 550))
        edges = (420, 410)

    jitter = rng.integers(-20, 21, size=(len(object_ids), 2))
    views = tuple(int(value) for value in rng.integers(0, 42, len(object_ids)))
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (object_id, view, center, long_edge) in enumerate(
        zip(object_ids, views, centers, edges, strict=True)
    ):
        record = _template_record(template_manifest, object_id, view)
        patch_rgb, patch_alpha = _scaled_patch(
            data_root=data_root, record=record, long_edge=long_edge
        )
        layers.append(
            _paste_layer(
                canvas_shape=(FRAME_HEIGHT, FRAME_WIDTH),
                rgb=patch_rgb,
                alpha=patch_alpha,
                center=(
                    center[0] + int(jitter[index, 0]),
                    center[1] + int(jitter[index, 1]),
                ),
            )
        )

    image = _background(seed)
    if stratum == "container_and_frame_boundary_distractor":
        drawn = Image.fromarray(image)
        draw = ImageDraw.Draw(drawn)
        draw.rectangle(
            (2, 170, FRAME_WIDTH - 3, 930), outline=(125, 130, 135), width=18
        )
        draw.rectangle((70, 245, FRAME_WIDTH - 70, 875), outline=(32, 35, 38), width=10)
        image = np.asarray(drawn).copy()

    owner = np.full((FRAME_HEIGHT, FRAME_WIDTH), -1, dtype=np.int16)
    depth = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16)
    # Input order is far-to-near; later instances overwrite earlier visibility.
    for index, (layer_rgb, layer_mask) in enumerate(layers):
        image[layer_mask] = layer_rgb[layer_mask]
        owner[layer_mask] = index
        depth[layer_mask] = np.uint16(1050 - index * 120)
    visible = tuple(owner == index for index in range(len(layers)))
    if any(int(mask.sum()) < 500 for mask in visible):
        raise ContractError("A-R7 synthetic layout hides an instance completely")
    return SyntheticScene(
        item_id=f"{stratum}-seed-{seed:06d}",
        stratum=stratum,
        seed=seed,
        rgb=image,
        depth_mm=depth,
        visible_masks=visible,
        object_ids=object_ids,
        template_views=views,
    )


class SyntheticGateRuntime:
    def __init__(
        self,
        *,
        data_root: Path,
        sam2_asset_root: Path,
        request: Mapping[str, Any],
        device: str,
    ) -> None:
        self.data_root = data_root.resolve()
        self.sam2_asset_root = sam2_asset_root.resolve()
        self.request = dict(request)
        self.device_name = device
        ensure_ultralytics_yolo_compat()
        self.cnos = OfficialCnosAdapter(
            deployment_root=self.data_root,
            deployment={
                "source": {
                    "cnos_checkout_relative_path": request["source"][
                        "cnos_checkout_relative_path"
                    ],
                    "dinov2_checkout_relative_path": request["source"][
                        "dinov2_checkout_relative_path"
                    ],
                },
                "models": {
                    "fastsam_checkpoint": {
                        "relative_path": request["models"]["fastsam_x_checkpoint"][
                            "relative_path"
                        ]
                    },
                    "dinov2_checkpoint": {
                        "relative_path": request["models"]["dinov2_vitl14_checkpoint"][
                            "relative_path"
                        ]
                    },
                },
                "adapter_config": {
                    "relative_path": request["adapter_config"]["relative_path"]
                },
            },
            device=device,
        )
        self.cnos.load()
        source_locks = load_source_execution_locks(request, data_root=self.data_root)
        audit_runtime_module_origins(source_locks)
        self.torch = self.cnos._torch
        build_module = importlib.import_module("sam2.build_sam")
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        model = build_module.build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(self.sam2_asset_root / "sam2.1_hiera_large.pt"),
            device=device,
        )
        self.sam2_predictor = predictor_module.SAM2ImagePredictor(model)
        sam2_checkout = (self.sam2_asset_root / "source").resolve()
        loaded_sam2 = [
            module
            for name, module in sys.modules.items()
            if name == "sam2" or name.startswith("sam2.")
        ]
        if not loaded_sam2:
            raise ContractError("A-R7 SAM2 runtime loaded no sam2 modules")
        for module in loaded_sam2:
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                Path(module_file).resolve().relative_to(sam2_checkout)
            except ValueError as exc:
                raise ContractError(
                    "A-R7 SAM2 module was imported outside the frozen checkout"
                ) from exc
        self.templates = {
            int(row["object_id"]): self.cnos._load_template_descriptors(
                row["descriptor"]["relative_path"]
            )
            for row in request["catalog"]
        }

    def infer_fastsam(self, image: np.ndarray) -> list[dict[str, Any]]:
        torch = self.torch
        raw = self.cnos._segmentor.model(image)[0]
        if raw.masks is None or raw.boxes is None:
            return []
        probability_masks = raw.masks.data.to(self.device_name)
        raw_scores = raw.boxes.data[:, 4].to(self.device_name)
        resized = self.cnos._segmentor.postprocess_resize(
            {"masks": probability_masks, "boxes": raw.boxes.data[:, :4]},
            image.shape[:2],
            update_boxes=False,
        )["masks"]
        config = self.cnos._config
        low = config["fastsam"]["mask_threshold"]
        min_box = config["postprocessing"]["min_box_side_relative"]
        min_area = config["postprocessing"]["min_mask_area_relative"]
        frame_area = image.shape[0] * image.shape[1]
        kept: list[dict[str, Any]] = []
        for index in range(len(resized)):
            probabilities = resized[index]
            binary = probabilities >= low
            locations = torch.nonzero(binary, as_tuple=False)
            if locations.numel() == 0:
                continue
            y_min, x_min = locations.min(dim=0).values
            y_max, x_max = locations.max(dim=0).values
            width = int(x_max.item() - x_min.item() + 1)
            height = int(y_max.item() - y_min.item() + 1)
            area = int(binary.sum().item())
            if (
                width * height
            ) / frame_area <= min_box**2 or area / frame_area <= min_area:
                continue
            mask = binary.detach().cpu().numpy().astype(bool)
            kept.append(
                {
                    "proposal_index": int(index),
                    "mask": mask,
                    "box": torch.tensor(
                        [x_min, y_min, x_max + 1, y_max + 1],
                        device=self.device_name,
                    ),
                    "proposal_score": float(raw_scores[index].item()),
                    "mask_stability": mask_stability_from_probabilities(
                        probabilities.detach().float().cpu().numpy(),
                        low_threshold=low,
                        high_threshold=config["fastsam"]["stability_high_threshold"],
                    ),
                }
            )
        return kept

    def score_masks(
        self,
        image: np.ndarray,
        masks: Sequence[np.ndarray],
        *,
        proposal_scores: Sequence[float],
        stabilities: Sequence[float],
        proposal_indices: Sequence[int],
    ) -> list[RuntimeProposal]:
        if not masks:
            return []
        if not (
            len(masks)
            == len(proposal_scores)
            == len(stabilities)
            == len(proposal_indices)
        ):
            raise ContractError("A-R7 CAD scorer inputs differ in coverage")
        torch = self.torch
        tensors = torch.stack(
            [
                torch.from_numpy(np.asarray(mask, dtype=bool))
                .to(self.device_name)
                .float()
                for mask in masks
            ]
        )
        boxes = []
        for mask in masks:
            locations = np.argwhere(mask)
            if not len(locations):
                raise ContractError("A-R7 CAD scorer received an empty mask")
            y_min, x_min = locations.min(axis=0)
            y_max, x_max = locations.max(axis=0)
            boxes.append(
                torch.tensor(
                    [x_min, y_min, x_max + 1, y_max + 1],
                    device=self.device_name,
                )
            )
        detections = self.cnos._detections_type(
            {"masks": tensors, "boxes": torch.stack(boxes)}
        )
        with torch.inference_mode():
            query = self.cnos._descriptor_model(image, detections)
            query = torch.nn.functional.normalize(query, dim=-1)
            by_object: dict[int, list[dict[str, Any]]] = {}
            for object_id, template in self.templates.items():
                reference = torch.nn.functional.normalize(template, dim=-1)
                top = torch.topk(query @ reference.transpose(0, 1), k=5, dim=-1)
                rows: list[dict[str, Any]] = []
                for values, indices in zip(
                    top.values.detach().cpu().tolist(),
                    top.indices.detach().cpu().tolist(),
                    strict=True,
                ):
                    raw = float(sum(values) / 5.0)
                    rows.append(
                        {
                            "object_id": object_id,
                            "raw_cosine": raw,
                            "normalized_similarity": store_raw_cosine(raw),
                            "top5_template_cosines": [float(value) for value in values],
                            "top5_template_indices": [int(value) for value in indices],
                        }
                    )
                by_object[object_id] = rows
        output: list[RuntimeProposal] = []
        for row_index, mask in enumerate(masks):
            ranking = tuple(
                sorted(
                    (by_object[object_id][row_index] for object_id in self.templates),
                    key=lambda row: (-row["raw_cosine"], row["object_id"]),
                )
            )
            output.append(
                RuntimeProposal(
                    proposal_index=int(proposal_indices[row_index]),
                    mask=np.asarray(mask, dtype=bool),
                    proposal_score=float(proposal_scores[row_index]),
                    mask_stability=float(stabilities[row_index]),
                    cad_ranking=ranking,
                )
            )
        return output

    @staticmethod
    def select_a_r6_parent(
        item_id: str, proposals: Sequence[RuntimeProposal]
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        records: list[dict[str, Any]] = []
        masks: list[np.ndarray] = []
        for proposal in proposals:
            statistics = mask_statistics(proposal.mask)
            ranking = [
                {
                    "rank": rank,
                    "object_id": row["object_id"],
                    "normalized_similarity": row["normalized_similarity"],
                }
                for rank, row in enumerate(proposal.cad_ranking, start=1)
            ]
            records.append(
                {
                    "proposal_index": proposal.proposal_index,
                    "mask": {"mask_pixels": statistics["mask_pixels"]},
                    "bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
                    "cad_ranking": ranking,
                    "selected_object_id": ranking[0]["object_id"],
                    "selected_cad_similarity": ranking[0]["normalized_similarity"],
                    "proposal_score": proposal.proposal_score,
                    "mask_stability": proposal.mask_stability,
                }
            )
            masks.append(proposal.mask)
        source_selected = None
        if records:
            source_selected = sorted(
                records,
                key=lambda row: (
                    -row["selected_cad_similarity"],
                    -row["proposal_score"],
                    -row["mask_stability"],
                    row["proposal_index"],
                ),
            )[0]["proposal_index"]
        decision = select_frame(
            {
                "item_id": item_id,
                "proposals": records,
                "selected_proposal_index": source_selected,
            },
            masks,
        )
        selected = decision["selected_proposal_index"]
        if selected is None:
            return None, decision
        for proposal in proposals:
            if proposal.proposal_index == selected:
                return proposal.mask, decision
        raise ContractError("A-R7 selected A-R6 parent is absent")

    def run_scene(self, scene: SyntheticScene) -> dict[str, Any]:
        fastsam = self.infer_fastsam(scene.rgb)
        baseline_proposals = self.score_masks(
            scene.rgb,
            [row["mask"] for row in fastsam],
            proposal_scores=[row["proposal_score"] for row in fastsam],
            stabilities=[row["mask_stability"] for row in fastsam],
            proposal_indices=[row["proposal_index"] for row in fastsam],
        )
        parent, decision = self.select_a_r6_parent(scene.item_id, baseline_proposals)
        baseline_masks = [] if parent is None else [parent]
        candidate_masks = list(baseline_masks)
        child_scores: list[RuntimeProposal] = []
        seed_trace: list[list[int]] = []
        box_trace: list[list[int]] = []
        failed_prompt_indices: list[int] = []
        depth_owner_region_pixels: list[int] = []
        candidate_mask_sources: list[str] = []
        support_mask_pixels = 0
        fallback_reason: str | None = None
        eligible_indices = {
            audit["proposal_index"]
            for audit in decision["candidate_audits"]
            if audit["eligible"]
        }
        support_masks = [
            proposal.mask
            for proposal in baseline_proposals
            if proposal.proposal_index in eligible_indices
        ]
        support = np.logical_or.reduce(support_masks) if support_masks else parent
        if support is not None:
            try:
                support_mask_pixels = int(support.sum())
                prompts = _depth_instance_prompts(support, scene.depth_mm)
                seeds = tuple(seed for seed, _ in prompts)
                seed_trace = [[x, y] for x, y in seeds]
                box_trace = [list(box) for _, box in prompts]
                if len(seeds) >= 2:
                    self.sam2_predictor.set_image(scene.rgb)
                    children, failed_prompt_indices = (
                        _predict_prompt_exclusive_box_children(
                            self.sam2_predictor, prompts
                        )
                    )
                    parent_proposal = next(
                        (
                            proposal
                            for proposal in baseline_proposals
                            if proposal.proposal_index
                            == decision["selected_proposal_index"]
                        ),
                        None,
                    )
                    (
                        candidate_masks,
                        proposal_scores,
                        stabilities,
                        candidate_mask_sources,
                        depth_owner_region_pixels,
                    ) = _compose_depth_consistent_candidate_masks(
                        children=children,
                        prompts=prompts,
                        failed_prompt_indices=failed_prompt_indices,
                        parent=parent_proposal,
                        support_mask=support,
                        depth_mm=scene.depth_mm,
                    )
                    child_scores = self.score_masks(
                        scene.rgb,
                        candidate_masks,
                        proposal_scores=proposal_scores,
                        stabilities=stabilities,
                        proposal_indices=list(range(len(candidate_masks))),
                    )
            except ContractError as exc:
                fallback_reason = str(exc)
        baseline_metrics = evaluate_instance_set(baseline_masks, scene.visible_masks)
        candidate_metrics = evaluate_instance_set(candidate_masks, scene.visible_masks)
        return {
            "item_id": scene.item_id,
            "stratum": scene.stratum,
            "seed": scene.seed,
            "object_ids": list(scene.object_ids),
            "template_views": list(scene.template_views),
            "a_r5_fastsam_proposal_count": len(baseline_proposals),
            "a_r6_decision": decision,
            "prompt_support_mask_pixels": support_mask_pixels,
            "depth_seed_xy": seed_trace,
            "depth_prompt_box_xyxy": box_trace,
            "failed_depth_prompt_indices": failed_prompt_indices,
            "depth_owner_region_pixels": depth_owner_region_pixels,
            "candidate_mask_sources": candidate_mask_sources,
            "deblend_fallback_reason": fallback_reason,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "candidate_cad_rankings": [
                {
                    "proposal_index": proposal.proposal_index,
                    "ranking": list(proposal.cad_ranking),
                }
                for proposal in child_scores
            ],
            "baseline_masks": baseline_masks,
            "candidate_masks": candidate_masks,
        }


def _outline(image: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    output = image.copy()
    colors = [(255, 70, 70), (70, 255, 100), (80, 160, 255), (255, 210, 70)]
    for index, mask in enumerate(masks):
        values = np.asarray(mask, dtype=bool)
        padded = np.pad(values, 1)
        eroded = np.logical_and.reduce(
            [
                padded[1 + dy : 1 + dy + FRAME_HEIGHT, 1 + dx : 1 + dx + FRAME_WIDTH]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
        )
        contour = values & ~eroded
        output[contour] = colors[index % len(colors)]
    return output


def _write_visual(
    path: Path,
    scene: SyntheticScene,
    baseline_masks: Sequence[np.ndarray],
    candidate_masks: Sequence[np.ndarray],
) -> None:
    panels = [
        scene.rgb,
        _outline(scene.rgb, scene.visible_masks),
        _outline(scene.rgb, baseline_masks),
        _outline(scene.rgb, candidate_masks),
    ]
    canvas = Image.new("RGB", (FRAME_WIDTH * 2, FRAME_HEIGHT * 2), (0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(
            Image.fromarray(panel),
            ((index % 2) * FRAME_WIDTH, (index // 2) * FRAME_HEIGHT),
        )
    draw = ImageDraw.Draw(canvas)
    labels = ("RGB", "VISIBLE GT", "A-R6 BASELINE", "SAM2 + CAD")
    for index, label in enumerate(labels):
        x = (index % 2) * FRAME_WIDTH + 18
        y = (index // 2) * FRAME_HEIGHT + 18
        draw.rectangle((x - 8, y - 8, x + 230, y + 30), fill=(0, 0, 0))
        draw.text((x, y), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        canvas.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS).save(
            stream, format="PNG", optimize=False
        )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _seed_plan(mode: str, rows_per_stratum: int) -> dict[str, list[int]]:
    if mode == "development":
        if rows_per_stratum != DEVELOPMENT_ROWS_PER_STRATUM:
            raise ContractError("A-R7 development coverage is fixed at 8 rows/stratum")
        first = DEVELOPMENT_FIRST_SEED
    elif mode == "train-smoke":
        if not 1 <= rows_per_stratum <= 8:
            raise ContractError("A-R7 train-smoke rows must be in [1,8]")
        first = 0
    else:
        raise ContractError("A-R7 synthetic mode is invalid")
    return {
        stratum: list(
            range(
                first + index * rows_per_stratum,
                first + (index + 1) * rows_per_stratum,
            )
        )
        for index, stratum in enumerate(SYNTHETIC_STRATA)
    }


def _evaluate_synthetic_gate_v1r7_3(
    *,
    baseline_by_stratum: Mapping[str, Sequence[Mapping[str, float | int]]],
    candidate_by_stratum: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> dict[str, Any]:
    if set(baseline_by_stratum) != set(SYNTHETIC_STRATA) or set(
        candidate_by_stratum
    ) != set(SYNTHETIC_STRATA):
        raise ContractError("A-R7.3 synthetic benchmark strata are incomplete")
    decisions: list[dict[str, Any]] = []
    for stratum in SYNTHETIC_STRATA:
        baseline_rows = list(baseline_by_stratum[stratum])
        candidate_rows = list(candidate_by_stratum[stratum])
        if not baseline_rows or len(baseline_rows) != len(candidate_rows):
            raise ContractError("A-R7.3 paired synthetic row coverage differs")
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
            recall_pass = candidate_recall >= baseline_recall
            merge_pass = candidate_merges == 0
        else:
            recall_pass = candidate_recall > baseline_recall
            merge_pass = (
                candidate_merges < baseline_merges
                if baseline_merges > 0
                else candidate_merges == 0
            )
        passed = (
            recall_pass
            and merge_pass
            and candidate_splits <= baseline_splits
            and candidate_unmatched <= baseline_unmatched
        )
        decisions.append(
            {
                "stratum": stratum,
                "baseline_recall75": baseline_recall,
                "candidate_recall75": candidate_recall,
                "strict_recall_improvement_required": stratum
                != "clean_single_instance",
                "recall_gate_passed": recall_pass,
                "baseline_merge_count": baseline_merges,
                "candidate_merge_count": candidate_merges,
                "strict_merge_reduction_applicable": baseline_merges > 0,
                "merge_gate_passed": merge_pass,
                "baseline_split_count": baseline_splits,
                "candidate_split_count": candidate_splits,
                "baseline_unmatched_prediction_count_iou75": baseline_unmatched,
                "candidate_unmatched_prediction_count_iou75": candidate_unmatched,
                "passed": passed,
            }
        )
    passed = all(bool(decision["passed"]) for decision in decisions)
    return {
        "status": "PASS_SYNTHETIC_DEBLEND_GATE_V1R7_3" if passed else "NO_GO",
        "all_required_strata_passed": passed,
        "post_freeze_replay_frame_read_count": 0,
        "decisions": decisions,
    }


def _depth_instance_prompts(
    support_mask: np.ndarray, depth_mm: np.ndarray
) -> tuple[tuple[tuple[int, int], tuple[int, int, int, int]], ...]:
    """Collapse same-depth pieces and bind each seed to its raw-depth box."""

    seeds = derive_depth_discontinuity_seeds(support_mask, depth_mm)
    support = np.asarray(support_mask, dtype=bool)
    depth = np.asarray(depth_mm)
    valid = support & (depth > 0)
    horizontal = np.abs(
        depth[:, 1:].astype(np.float64) - depth[:, :-1].astype(np.float64)
    )[valid[:, 1:] & valid[:, :-1]]
    vertical = np.abs(
        depth[1:, :].astype(np.float64) - depth[:-1, :].astype(np.float64)
    )[valid[1:, :] & valid[:-1, :]]
    differences = np.concatenate([horizontal, vertical])
    robust_noise = 0.0 if not differences.size else float(np.median(differences))
    grouping_threshold = max(1.0, DEPTH_EDGE_NOISE_MULTIPLIER * robust_noise)
    groups: list[tuple[float, tuple[int, int]]] = []
    for seed in seeds:
        value = float(depth[seed[1], seed[0]])
        if any(abs(value - previous) <= grouping_threshold for previous, _ in groups):
            continue
        groups.append((value, seed))
    prompts = []
    for value, seed in groups:
        region = valid & (
            np.abs(depth.astype(np.float64) - value) <= grouping_threshold
        )
        locations = np.argwhere(region)
        if not len(locations):
            raise ContractError("A-R7 depth prompt region is empty")
        y_min, x_min = locations.min(axis=0)
        y_max, x_max = locations.max(axis=0)
        prompts.append(
            (
                seed,
                (int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)),
            )
        )
    return tuple(prompts)


def _depth_instance_seeds(
    support_mask: np.ndarray, depth_mm: np.ndarray
) -> tuple[tuple[int, int], ...]:
    return tuple(seed for seed, _ in _depth_instance_prompts(support_mask, depth_mm))


def _logit_stability(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("A-R7 SAM2 box-prompt logits must be finite HxW")
    low = values > -1.0
    high = values > 1.0
    union = int(low.sum())
    return 0.0 if union == 0 else float(high.sum() / union)


def _predict_prompt_exclusive_box_children(
    predictor: Any,
    prompts: Sequence[tuple[tuple[int, int], tuple[int, int, int, int]]],
) -> tuple[list[DeblendCandidate], list[int]]:
    if len(prompts) < 2:
        raise ContractError("A-R7 box deblend requires at least two prompts")
    seeds = tuple(seed for seed, _ in prompts)
    if len(set(seeds)) != len(seeds):
        raise ContractError("A-R7 box deblend seeds are not unique")
    candidates: list[DeblendCandidate] = []
    failed_prompt_indices: list[int] = []
    for positive_index, (positive, box) in enumerate(prompts):
        rivals = tuple(seed for seed in seeds if seed != positive)
        coordinates = np.asarray([positive, *rivals], dtype=np.float32)
        labels = np.asarray([1, *([0] * len(rivals))], dtype=np.int32)
        result = predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
            return_logits=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise ContractError("A-R7 SAM2 box predictor returned an invalid result")
        logits, scores, low_resolution = (np.asarray(value) for value in result)
        if (
            logits.ndim != 3
            or scores.ndim != 1
            or low_resolution.ndim != 3
            or not len(logits)
            or len(logits) != len(scores)
            or len(logits) != len(low_resolution)
            or not np.isfinite(logits).all()
            or not np.isfinite(scores).all()
        ):
            raise ContractError("A-R7 SAM2 box result shapes changed")
        valid: list[tuple[float, float, int, np.ndarray]] = []
        for index, raw_logits in enumerate(logits):
            mask = raw_logits > 0
            px, py = positive
            if not mask[py, px] or any(mask[ny, nx] for nx, ny in rivals):
                continue
            score = float(scores[index])
            if not 0.0 <= score <= 1.0:
                raise ContractError("A-R7 SAM2 box predicted IoU is outside [0,1]")
            valid.append((score, _logit_stability(raw_logits), index, mask))
        if not valid:
            failed_prompt_indices.append(positive_index)
            continue
        score, stability, _, mask = sorted(
            valid, key=lambda row: (-row[0], -row[1], row[2])
        )[0]
        candidates.append(
            DeblendCandidate(
                candidate_id=f"seed-{positive_index:04d}",
                mask=mask,
                positive_seed_xy=positive,
                negative_seed_xys=rivals,
                predicted_iou=score,
                stability_score=stability,
            )
        )
    if len(candidates) < 2:
        raise ContractError(
            "A-R7 box deblend produced fewer than two exclusive children"
        )
    successful_seeds = tuple(candidate.positive_seed_xy for candidate in candidates)
    rebased = [
        replace(
            candidate,
            negative_seed_xys=tuple(
                seed for seed in successful_seeds if seed != candidate.positive_seed_xy
            ),
        )
        for candidate in candidates
    ]
    return partition_prompted_candidates(rebased), failed_prompt_indices


def _compose_depth_consistent_candidate_masks(
    *,
    children: Sequence[DeblendCandidate],
    prompts: Sequence[tuple[tuple[int, int], tuple[int, int, int, int]]],
    failed_prompt_indices: Sequence[int],
    parent: RuntimeProposal | None,
    support_mask: np.ndarray,
    depth_mm: np.ndarray,
) -> tuple[list[np.ndarray], list[float], list[float], list[str], list[int]]:
    """Clip SAM2 children by depth ownership and retain a failed-prompt parent slice."""

    support = np.asarray(support_mask)
    depth = np.asarray(depth_mm)
    if support.ndim != 2 or support.dtype != np.bool_ or not support.any():
        raise ContractError("A-R7.3 depth-owner support must be non-empty boolean HxW")
    if depth.shape != support.shape or not np.issubdtype(depth.dtype, np.number):
        raise ContractError("A-R7.3 depth-owner depth/support shapes differ")
    if np.issubdtype(depth.dtype, np.floating) and not np.isfinite(depth).all():
        raise ContractError("A-R7.3 depth-owner depth contains non-finite values")
    if len(prompts) < 2:
        raise ContractError("A-R7.3 depth-owner partition needs at least two prompts")
    seeds = tuple(seed for seed, _ in prompts)
    seed_depths = np.asarray([float(depth[y, x]) for x, y in seeds], dtype=np.float64)
    if (seed_depths <= 0).any() or len(set(seed_depths.tolist())) != len(seed_depths):
        raise ContractError(
            "A-R7.3 depth-owner seed depths must be positive and unique"
        )
    parent_mask = None if parent is None else np.asarray(parent.mask, dtype=bool)
    if parent_mask is not None and parent_mask.shape != support.shape:
        raise ContractError("A-R7.3 parent/depth-owner shapes differ")
    envelope_masks = [np.asarray(child.mask, dtype=bool) for child in children]
    if parent_mask is not None and failed_prompt_indices:
        envelope_masks.append(parent_mask)
    if not envelope_masks:
        raise ContractError("A-R7.3 depth-owner envelope is empty")
    envelope = np.logical_or.reduce(envelope_masks)
    valid = envelope & (depth > 0)
    distances = np.abs(
        depth.astype(np.float64)[None, :, :] - seed_depths[:, None, None]
    )
    owner = np.argmin(distances, axis=0)
    regions = [valid & (owner == index) for index in range(len(prompts))]
    region_pixels = [int(region.sum()) for region in regions]
    if any(not pixels for pixels in region_pixels):
        raise ContractError("A-R7.3 depth-owner partition contains an empty region")

    failed = tuple(int(index) for index in failed_prompt_indices)
    if len(set(failed)) != len(failed) or any(
        index < 0 or index >= len(prompts) for index in failed
    ):
        raise ContractError("A-R7.3 failed prompt indices are invalid")
    by_prompt: dict[int, DeblendCandidate] = {}
    for child in children:
        prefix = "seed-"
        if not child.candidate_id.startswith(prefix):
            raise ContractError("A-R7.3 child candidate ID lost its prompt index")
        try:
            prompt_index = int(child.candidate_id[len(prefix) :])
        except ValueError as exc:
            raise ContractError(
                "A-R7.3 child candidate ID has an invalid prompt index"
            ) from exc
        if (
            prompt_index in by_prompt
            or prompt_index < 0
            or prompt_index >= len(prompts)
        ):
            raise ContractError("A-R7.3 child prompt index coverage changed")
        by_prompt[prompt_index] = child
    if set(by_prompt).intersection(failed) or set(by_prompt).union(failed) != set(
        range(len(prompts))
    ):
        raise ContractError("A-R7.3 successful/failed prompt coverage differs")

    masks: list[np.ndarray] = []
    scores: list[float] = []
    stabilities: list[float] = []
    sources: list[str] = []
    for prompt_index in sorted(by_prompt):
        child = by_prompt[prompt_index]
        clipped = regions[prompt_index]
        x, y = seeds[prompt_index]
        if not clipped.any() or not clipped[y, x]:
            raise ContractError("A-R7.3 depth ownership removed a SAM2 positive seed")
        masks.append(clipped)
        scores.append(float(child.predicted_iou))
        stabilities.append(float(child.stability_score))
        sources.append(f"sam2-union-depth-owner-seed-{prompt_index:04d}")

    if parent is not None and parent_mask is not None:
        for prompt_index in sorted(failed):
            x, y = seeds[prompt_index]
            if not parent_mask[y, x]:
                continue
            residual = regions[prompt_index]
            masks.append(residual)
            scores.append(float(parent.proposal_score))
            stabilities.append(float(parent.mask_stability))
            sources.append(
                f"a-r6-parent-authorized-depth-owner-seed-{prompt_index:04d}"
            )
    if len(masks) < 2:
        raise ContractError(
            "A-R7.3 depth-owner composition produced fewer than two masks"
        )
    for left_index, left in enumerate(masks):
        for right in masks[left_index + 1 :]:
            if np.logical_and(left, right).any():
                raise ContractError("A-R7.3 depth-owner masks overlap")
    return masks, scores, stabilities, sources, region_pixels


def run_gate(
    *,
    protocol_path: Path,
    repository_root: Path,
    data_root: Path,
    sam2_asset_root: Path,
    output_root: Path,
    mode: str,
    rows_per_stratum: int,
    device: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise ContractError("A-R7 synthetic output root is create-only")
    stage = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if stage.exists():
        raise ContractError("A-R7 synthetic staging root already exists")
    protocol = read_json(protocol_path)
    if not isinstance(protocol, dict):
        raise ContractError("A-R7 execution protocol must be a JSON object")
    validate_execution_protocol(protocol, repository_root=repository_root)
    sam2_identity = verify_sam2_freeze(sam2_asset_root)
    request, template_manifest = load_a_r5_assets(data_root)
    seed_plan = _seed_plan(mode, rows_per_stratum)
    runtime = SyntheticGateRuntime(
        data_root=data_root,
        sam2_asset_root=sam2_asset_root,
        request=request,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    baseline_by_stratum: dict[str, list[Mapping[str, float | int]]] = {
        stratum: [] for stratum in SYNTHETIC_STRATA
    }
    candidate_by_stratum: dict[str, list[Mapping[str, float | int]]] = {
        stratum: [] for stratum in SYNTHETIC_STRATA
    }
    for stratum in SYNTHETIC_STRATA:
        for seed in seed_plan[stratum]:
            scene = generate_scene(
                stratum=stratum,
                seed=seed,
                data_root=data_root,
                template_manifest=template_manifest,
            )
            result = runtime.run_scene(scene)
            baseline_by_stratum[stratum].append(result["baseline_metrics"])
            candidate_by_stratum[stratum].append(result["candidate_metrics"])
            visual_path = stage / "visualizations" / f"{scene.item_id}.png"
            _write_visual(
                visual_path,
                scene,
                result.pop("baseline_masks"),
                result.pop("candidate_masks"),
            )
            result["visualization"] = _asset(
                visual_path, role="synthetic_before_after", root=stage
            )
            rows.append(result)
    gate = _evaluate_synthetic_gate_v1r7_3(
        baseline_by_stratum=baseline_by_stratum,
        candidate_by_stratum=candidate_by_stratum,
    )
    gate["scene9_replay_permitted"] = bool(
        mode == "development" and gate["all_required_strata_passed"]
    )
    manifest = {
        "schema_version": "poseloop.pose-accuracy-recovery.synthetic-deblend-result.v1r7.3",
        "protocol_id": protocol["protocol_id"],
        "execution_protocol_id": protocol["protocol_id"],
        "execution_protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "mode": mode,
        "seed_plan": seed_plan,
        "sam2_identity": sam2_identity,
        "a_r5_asset_identity": {
            "runtime_request_sha256": A_R5_REQUEST_SHA256,
            "template_manifest_sha256": A_R5_TEMPLATE_MANIFEST_SHA256,
            "adapter_config_sha256": A_R5_CONFIG_SHA256,
            "catalog_object_ids": [1, 2, 4, 5, 6],
        },
        "a_r6_policy": A_R6_POLICY,
        "rows": rows,
        "gate": gate,
        "boundary": dict(BOUNDARY_ZERO),
        "foundationpose_run_count": 0,
        "official_scorer_run_count": 0,
        "downstream_export_count": 0,
        "result_lock_sha256": "pending",
    }
    manifest["result_lock_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "result_lock_sha256"}
    )
    _write_json_create_only(stage / "synthetic-gate-result.json", manifest)
    os.replace(stage, output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.instance_deblend_v1r7.synthetic_runtime",
        description="Run the synthetic-only SAM2 + CAD deblending gate; no replay path is accepted.",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam2-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train-smoke", "development"), required=True)
    parser.add_argument(
        "--rows-per-stratum", type=int, default=DEVELOPMENT_ROWS_PER_STRATUM
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_gate(
            protocol_path=args.protocol,
            repository_root=args.repository_root,
            data_root=args.data_root,
            sam2_asset_root=args.sam2_asset_root,
            output_root=args.output_root,
            mode=args.mode,
            rows_per_stratum=args.rows_per_stratum,
            device=args.device,
        )
    except ContractError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result["gate"]["status"],
                "result_lock_sha256": result["result_lock_sha256"],
                "scene9_replay_permitted": result["gate"]["scene9_replay_permitted"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_ZERO",
    "DEVELOPMENT_ROWS_PER_STRATUM",
    "SyntheticGateRuntime",
    "SyntheticScene",
    "generate_scene",
    "load_a_r5_assets",
    "run_gate",
    "verify_sam2_freeze",
]
