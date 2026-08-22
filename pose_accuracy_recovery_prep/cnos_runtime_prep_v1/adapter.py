"""Lazy adapter for official CNOS FastSAM + DINOv2 target-conditioned proposals."""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from pose_accuracy_recovery_prep.core import ContractError, read_json


@dataclass(frozen=True)
class Candidate:
    """One real FastSAM mask scored against the target CAD templates."""

    proposal_index: int
    mask: np.ndarray
    raw_cad_cosine: float
    cad_similarity: float
    top5_template_cosines: tuple[float, float, float, float, float]
    top5_template_indices: tuple[int, int, int, int, int]
    proposal_score: float
    mask_stability: float


def _unit_score(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ContractError(f"{label} must be finite in [0, 1]")
    return float(value)


def store_raw_cosine(raw_cosine: Any) -> float:
    """Store raw [-1, 1] cosine in the parent schema without changing ranking."""
    if (
        not isinstance(raw_cosine, (int, float))
        or isinstance(raw_cosine, bool)
        or not math.isfinite(raw_cosine)
        or not -1.0 <= float(raw_cosine) <= 1.0
    ):
        raise ContractError("Raw CNOS cosine must be finite in [-1, 1]")
    return (float(raw_cosine) + 1.0) / 2.0


def mask_stability_from_probabilities(
    probability_mask: np.ndarray,
    *,
    low_threshold: float = 0.50,
    high_threshold: float = 0.55,
) -> float:
    """Frozen third-order tie-break: IoU at the two declared thresholds."""
    values = np.asarray(probability_mask)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("FastSAM probability mask must be finite HxW")
    low = values >= low_threshold
    high = values >= high_threshold
    union = int(np.logical_or(low, high).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(low, high).sum() / union)


def rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    parsed = list(candidates)
    indices: set[int] = set()
    for candidate in parsed:
        if (
            not isinstance(candidate.proposal_index, int)
            or isinstance(candidate.proposal_index, bool)
            or candidate.proposal_index < 0
            or candidate.proposal_index in indices
        ):
            raise ContractError("CNOS proposal indices must be unique non-negative ints")
        indices.add(candidate.proposal_index)
        if (
            not isinstance(candidate.raw_cad_cosine, (int, float))
            or isinstance(candidate.raw_cad_cosine, bool)
            or not math.isfinite(candidate.raw_cad_cosine)
            or not -1.0 <= float(candidate.raw_cad_cosine) <= 1.0
            or len(candidate.top5_template_cosines) != 5
            or len(candidate.top5_template_indices) != 5
            or len(set(candidate.top5_template_indices)) != 5
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not -1.0 <= float(value) <= 1.0
                for value in candidate.top5_template_cosines
            )
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in candidate.top5_template_indices
            )
            or list(candidate.top5_template_cosines)
            != sorted(candidate.top5_template_cosines, reverse=True)
            or not math.isclose(
                float(candidate.raw_cad_cosine),
                sum(float(value) for value in candidate.top5_template_cosines) / 5.0,
                abs_tol=1e-7,
            )
            or not math.isclose(
                candidate.cad_similarity,
                store_raw_cosine(candidate.raw_cad_cosine),
                abs_tol=1e-7,
            )
        ):
            raise ContractError("CNOS candidate raw/top-5/normalized score trace changed")
        _unit_score(candidate.cad_similarity, "candidate.cad_similarity")
        _unit_score(candidate.proposal_score, "candidate.proposal_score")
        _unit_score(candidate.mask_stability, "candidate.mask_stability")
        mask = np.asarray(candidate.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
            raise ContractError("CNOS candidate mask must be non-empty boolean HxW")
    return sorted(
        parsed,
        key=lambda candidate: (
            -candidate.cad_similarity,
            -candidate.proposal_score,
            -candidate.mask_stability,
            candidate.proposal_index,
        ),
    )


def reduced_chunk_size(current: int, minimum: int) -> int:
    if (
        not isinstance(current, int)
        or isinstance(current, bool)
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum <= 0
        or current <= minimum
    ):
        raise ContractError("OOM chunk cannot be reduced further")
    return max(minimum, current // 2)


class OfficialCnosAdapter:
    """Load only the disk-frozen official CNOS/FastSAM/DINOv2 route."""

    def __init__(
        self,
        *,
        deployment_root: Path,
        deployment: Mapping[str, Any],
        device: str = "cuda:0",
    ) -> None:
        self.root = deployment_root.resolve()
        self.deployment = dict(deployment)
        self.device_name = device
        self._loaded = False
        self._torch: Any = None
        self._segmentor: Any = None
        self._descriptor_model: Any = None
        self._detections_type: Any = None
        self._config: dict[str, Any] = {}
        self._oom_events: list[dict[str, int]] = []

    def _path(self, relative: str) -> Path:
        candidate = (self.root / Path(*relative.split("/"))).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ContractError("CNOS runtime path escapes deployment root") from exc
        return candidate

    @staticmethod
    def _require_module_from(module: Any, root: Path, label: str) -> None:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ContractError(f"{label} does not expose a source file")
        try:
            Path(module_file).resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ContractError(f"{label} was imported outside the frozen CNOS checkout") from exc

    def load(self) -> None:
        if self._loaded:
            return
        source = self.deployment["source"]
        models = self.deployment["models"]
        cnos_checkout = self._path(source["cnos_checkout_relative_path"])
        dinov2_checkout = self._path(source["dinov2_checkout_relative_path"])
        fastsam_checkpoint = self._path(
            models["fastsam_checkpoint"]["relative_path"]
        )
        dinov2_checkpoint = self._path(
            models["dinov2_checkpoint"]["relative_path"]
        )
        config_path = self._path(self.deployment["adapter_config"]["relative_path"])
        config = read_json(config_path)
        if not isinstance(config, dict):
            raise ContractError("CNOS adapter config must be a JSON object")

        source_text = str(cnos_checkout)
        inserted = source_text not in sys.path
        if inserted:
            sys.path.insert(0, source_text)
        try:
            torch = importlib.import_module("torch")
            fast_sam_module = importlib.import_module("src.model.fast_sam")
            dinov2_module = importlib.import_module("src.model.dinov2")
            utils_module = importlib.import_module("src.model.utils")
            self._require_module_from(
                fast_sam_module, cnos_checkout, "CNOS FastSAM module"
            )
            self._require_module_from(
                dinov2_module, cnos_checkout, "CNOS DINOv2 adapter module"
            )
            self._require_module_from(
                utils_module, cnos_checkout, "CNOS detections module"
            )
            hub_model = torch.hub.load(
                str(dinov2_checkout),
                config["dinov2"]["model_name"],
                source="local",
                pretrained=False,
            )
            state = torch.load(
                dinov2_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict):
                raise ContractError("DINOv2 checkpoint is not a state dictionary")
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            hub_model.load_state_dict(state, strict=True)
            descriptor = dinov2_module.CustomDINOv2(
                model_name=config["dinov2"]["model_name"],
                model=hub_model,
                token_name=config["dinov2"]["token_name"],
                image_size=config["dinov2"]["proposal_image_size"],
                chunk_size=config["dinov2"]["feature_chunk_size"],
                descriptor_width_size=config["dinov2"]["descriptor_width_size"],
            )
            fastsam_config = SimpleNamespace(
                iou_threshold=config["fastsam"]["iou_threshold"],
                conf_threshold=config["fastsam"]["configured_conf_threshold"],
                max_det=config["fastsam"]["max_det"],
            )
            segmentor = fast_sam_module.FastSAM(
                checkpoint_path=fastsam_checkpoint,
                config=fastsam_config,
                segmentor_width_size=config["fastsam"]["segmentor_width_size"],
                device=torch.device(self.device_name),
            )
            device = torch.device(self.device_name)
            segmentor.model.setup_model(device=device, verbose=True)
            descriptor.model = descriptor.model.to(device)
            descriptor.model.eval()
            self._torch = torch
            self._segmentor = segmentor
            self._descriptor_model = descriptor
            self._detections_type = utils_module.Detections
            self._config = config
            self._loaded = True
        finally:
            if inserted and source_text in sys.path:
                sys.path.remove(source_text)

    def _load_template_descriptors(self, relative_path: str) -> Any:
        torch = self._torch
        value = torch.load(
            self._path(relative_path), map_location=self.device_name, weights_only=True
        )
        if isinstance(value, dict):
            if set(value) != {"descriptors"}:
                raise ContractError("CNOS descriptor file has ambiguous keys")
            value = value["descriptors"]
        if not torch.is_tensor(value) or value.ndim not in {2, 3}:
            raise ContractError("CNOS CAD descriptors must be a 2D/3D tensor")
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ContractError("Per-object CNOS descriptor file must contain one object")
            value = value[0]
        if value.shape[0] < 5 or not torch.isfinite(value).all():
            raise ContractError("CNOS CAD descriptors require at least five finite templates")
        return value.to(self.device_name)

    def _score_chunk(
        self,
        image_np: np.ndarray,
        masks: Any,
        boxes: Any,
        template_descriptors: Any,
    ) -> Any:
        torch = self._torch
        detections = self._detections_type({"masks": masks, "boxes": boxes})
        query = self._descriptor_model(image_np, detections)
        query = torch.nn.functional.normalize(query, dim=-1)
        reference = torch.nn.functional.normalize(template_descriptors, dim=-1)
        similarities = query @ reference.transpose(0, 1)
        top_five = torch.topk(similarities, k=5, dim=-1)
        return top_five.values, top_five.indices

    def infer(
        self,
        *,
        rgb_relative_path: str,
        descriptor_relative_path: str,
        target_object_id: int,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[Candidate], list[dict[str, int]]]:
        del target_object_id  # Identity is already bound by the descriptor manifest.
        self.load()
        torch = self._torch
        with Image.open(self._path(rgb_relative_path)) as image:
            image_np = np.asarray(image.convert("RGB"))
        if image_np.shape != (1080, 1440, 3):
            raise ContractError("CNOS producer RGB must remain 1440x1080 RGB")

        raw = self._segmentor.model(image_np)[0]
        if raw.masks is None or raw.boxes is None:
            raise ContractError("FastSAM returned no mask container")
        probability_masks = raw.masks.data.to(self.device_name)
        raw_scores = raw.boxes.data[:, 4].to(self.device_name)
        resized = self._segmentor.postprocess_resize(
            {"masks": probability_masks, "boxes": raw.boxes.data[:, :4]},
            image_np.shape[:2],
            update_boxes=False,
        )["masks"]
        low = self._config["fastsam"]["mask_threshold"]
        high = self._config["fastsam"]["stability_high_threshold"]
        min_box = self._config["postprocessing"]["min_box_side_relative"]
        min_area = self._config["postprocessing"]["min_mask_area_relative"]
        frame_area = image_np.shape[0] * image_np.shape[1]

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
            if (width * height) / frame_area <= min_box**2 or area / frame_area <= min_area:
                continue
            low_mask = binary.detach().cpu().numpy().astype(bool)
            high_mask = (probabilities >= high).detach().cpu().numpy().astype(bool)
            union = int(np.logical_or(low_mask, high_mask).sum())
            stability = (
                0.0
                if union == 0
                else float(np.logical_and(low_mask, high_mask).sum() / union)
            )
            kept.append(
                {
                    "proposal_index": index,
                    "probabilities": probabilities,
                    "mask": low_mask,
                    "box": torch.tensor(
                        [x_min, y_min, x_max + 1, y_max + 1],
                        device=self.device_name,
                    ),
                    "proposal_score": float(raw_scores[index].item()),
                    "mask_stability": stability,
                }
            )
        if not kept:
            raise ContractError("FastSAM produced no valid post-geometric-filter proposals")

        templates = self._load_template_descriptors(descriptor_relative_path)
        current_chunk = proposal_chunk_size
        cursor = 0
        raw_similarities: list[float] = []
        normalized_similarities: list[float] = []
        top5_values: list[tuple[float, float, float, float, float]] = []
        top5_indices: list[tuple[int, int, int, int, int]] = []
        oom_events: list[dict[str, int]] = []
        reductions = 0
        max_reductions = self._config["oom_policy"]["max_chunk_reductions"]
        while cursor < len(kept):
            stop = min(cursor + current_chunk, len(kept))
            try:
                masks = torch.stack(
                    [
                        torch.from_numpy(candidate["mask"])
                        .to(self.device_name)
                        .float()
                        for candidate in kept[cursor:stop]
                    ]
                )
                boxes = torch.stack(
                    [candidate["box"] for candidate in kept[cursor:stop]]
                )
                value_tensor, index_tensor = self._score_chunk(
                    image_np, masks, boxes, templates
                )
            except torch.cuda.OutOfMemoryError:
                if current_chunk <= minimum_chunk_size or reductions >= max_reductions:
                    raise
                next_chunk = reduced_chunk_size(current_chunk, minimum_chunk_size)
                oom_events.append(
                    {
                        "cursor": cursor,
                        "failed_chunk_size": current_chunk,
                        "retry_chunk_size": next_chunk,
                    }
                )
                current_chunk = next_chunk
                reductions += 1
                if "masks" in locals():
                    del masks
                if "boxes" in locals():
                    del boxes
                torch.cuda.empty_cache()
                continue
            value_rows = value_tensor.cpu().tolist()
            index_rows = index_tensor.cpu().tolist()
            for values, indices in zip(value_rows, index_rows, strict=True):
                component_values = tuple(float(value) for value in values)
                component_indices = tuple(int(value) for value in indices)
                if len(component_values) != 5 or len(component_indices) != 5:
                    raise ContractError("CNOS top-5 template evidence changed")
                raw = sum(component_values) / 5.0
                raw_similarities.append(raw)
                normalized_similarities.append(store_raw_cosine(raw))
                top5_values.append(component_values)
                top5_indices.append(component_indices)
            cursor = stop

        candidates = [
            Candidate(
                proposal_index=item["proposal_index"],
                mask=item["mask"],
                raw_cad_cosine=raw_similarity,
                cad_similarity=_unit_score(
                    normalized_similarity, "CNOS normalized CAD similarity"
                ),
                top5_template_cosines=component_values,
                top5_template_indices=component_indices,
                proposal_score=_unit_score(
                    item["proposal_score"], "FastSAM proposal confidence"
                ),
                mask_stability=_unit_score(
                    item["mask_stability"], "FastSAM mask stability"
                ),
            )
            for item, raw_similarity, normalized_similarity, component_values, component_indices in zip(
                kept,
                raw_similarities,
                normalized_similarities,
                top5_values,
                top5_indices,
                strict=True,
            )
        ]
        return rank_candidates(candidates), oom_events
