"""Lazy official-CNOS CustomDINOv2 descriptor runtime for A-R5-P2."""

from __future__ import annotations

import importlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import ContractError

from . import FEATURE_DIMENSION, VIEW_COUNT
from .contracts import audit_module_origins


@dataclass(frozen=True)
class DescriptorRuntime:
    torch: Any
    descriptor_model: Any
    cropper: Any
    normalize: Any
    device: Any
    module_origin_audit: dict[str, dict[str, Any]]
    weight_strict_load: bool


def load_official_runtime(validation: Mapping[str, Any]) -> DescriptorRuntime:
    """Load only the request-bound CNOS/DINO source and strict official weight."""

    cnos_checkout = Path(validation["cnos_checkout"]).resolve()
    dinov2_checkout = Path(validation["dinov2_checkout"]).resolve()
    request = validation["request"]
    source_text = str(cnos_checkout)
    inserted = source_text not in sys.path
    sys.dont_write_bytecode = True
    if inserted:
        sys.path.insert(0, source_text)
    try:
        torch = importlib.import_module("torch")
        transforms = importlib.import_module("torchvision.transforms")
        dinov2_module = importlib.import_module("src.model.dinov2")
        importlib.import_module("src.model.utils")
        bbox_module = importlib.import_module("src.utils.bbox_utils")
        hub_model = torch.hub.load(
            str(dinov2_checkout),
            request["runtime"]["model_name"],
            source="local",
            pretrained=False,
        )
        state = torch.load(
            validation["weight_path"], map_location="cpu", weights_only=True
        )
        if not isinstance(state, dict):
            raise ContractError("Official DINOv2 checkpoint is not a state dictionary")
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        hub_model.load_state_dict(state, strict=True)
        descriptor = dinov2_module.CustomDINOv2(
            model_name=request["runtime"]["model_name"],
            model=hub_model,
            token_name=request["runtime"]["token_name"],
            image_size=request["runtime"]["proposal_image_size"],
            chunk_size=request["runtime"]["feature_chunk_size"],
            descriptor_width_size=request["runtime"]["descriptor_width_size"],
        )
        device = torch.device(request["runtime"]["device"])
        descriptor.model = descriptor.model.to(device)
        descriptor.model.eval()
        module_audit = audit_module_origins(
            sys.modules,
            cnos_checkout=cnos_checkout,
            cnos_files=validation["cnos_files"],
            dinov2_checkout=dinov2_checkout,
            dinov2_files=validation["dinov2_files"],
        )
        cropper = bbox_module.CropResizePad(request["runtime"]["proposal_image_size"])
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        return DescriptorRuntime(
            torch=torch,
            descriptor_model=descriptor,
            cropper=cropper,
            normalize=normalize,
            device=device,
            module_origin_audit=module_audit,
            weight_strict_load=True,
        )
    except ContractError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError(
            "Unable to load the bound official descriptor runtime"
        ) from exc
    finally:
        if inserted and source_text in sys.path:
            sys.path.remove(source_text)


def compute_object_descriptors(
    *,
    runtime: DescriptorRuntime,
    object_manifest: Mapping[str, Any],
    data_root: Path,
) -> Any:
    """Apply official CNOS PIL-bbox/CropResizePad/normalize/CLS semantics."""

    torch = runtime.torch
    images: list[Any] = []
    boxes: list[tuple[int, int, int, int]] = []
    views = object_manifest["views"]
    if not isinstance(views, list) or len(views) != VIEW_COUNT:
        raise ContractError("Descriptor input requires exactly 42 RGBA views")
    for view_index, view in enumerate(views):
        if view["view_index"] != view_index:
            raise ContractError("Descriptor input view order changed")
        relative = view["rgba"]["relative_path"]
        path = data_root.resolve() / Path(*relative.split("/"))
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGBA" or image.size != (640, 480):
                raise ContractError("Descriptor input must remain 640x480 RGBA")
            bbox = image.getbbox()
            if bbox is None or bbox != image.getchannel("A").getbbox():
                raise ContractError("Descriptor template bbox is not alpha-causal")
            rgb = np.asarray(image.convert("RGB")) / 255.0
        images.append(torch.from_numpy(rgb).float())
        boxes.append(bbox)
    templates = torch.stack(images).permute(0, 3, 1, 2)
    box_tensor = torch.tensor(np.asarray(boxes))
    processed = runtime.cropper(images=templates, boxes=box_tensor)
    processed = runtime.normalize(processed).to(runtime.device)
    with torch.no_grad():
        features = runtime.descriptor_model.compute_features(
            processed, token_name="x_norm_clstoken"
        )
    if (
        not torch.is_tensor(features)
        or list(features.shape) != [VIEW_COUNT, FEATURE_DIMENSION]
        or features.dtype != torch.float32
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ContractError("CustomDINOv2 must return finite float32 [42,1024]")
    return features.detach().to("cpu").contiguous()


def save_tensor_create_only(path: Path, tensor: Any, *, torch_module: Any) -> None:
    buffer = io.BytesIO()
    torch_module.save(tensor, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError(f"Descriptor tensor output is create-only: {path}") from exc


__all__ = [
    "DescriptorRuntime",
    "compute_object_descriptors",
    "load_official_runtime",
    "save_tensor_create_only",
]
