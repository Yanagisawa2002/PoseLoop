"""All-catalog wrapper around the pinned official CNOS runtime adapter.

The predecessor adapter accepts one descriptor bank at a time.  A-R5 calls it
for every catalogue object, proves that the underlying FastSAM proposal set is
identical across those calls, and only then constructs per-instance CAD ranks.
No per-frame target identity is accepted by this API.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.adapter import (
    Candidate,
    OfficialCnosAdapter,
)
from pose_accuracy_recovery_prep.core import ContractError, sha256_file

from .contracts import (
    CATALOG_OBJECT_IDS,
    load_source_execution_locks,
    normalized_cad_similarity,
)


def ensure_ultralytics_yolo_compat(
    *, import_module: Any = importlib.import_module
) -> str:
    """Expose the legacy CNOS ``ultralytics.yolo`` route on current releases.

    The frozen official CNOS source imports ``yolo.v8.segment`` while current
    Ultralytics exposes the same predictor at ``models.yolo.segment.predict``.
    Keep the CNOS bytes untouched and install only this narrow import alias.
    """

    ultralytics = import_module("ultralytics")
    legacy = getattr(ultralytics, "yolo", None)
    try:
        predictor = legacy.v8.segment.SegmentationPredictor
    except (AttributeError, TypeError):
        predictor_module = import_module("ultralytics.models.yolo.segment.predict")
        predictor = getattr(predictor_module, "SegmentationPredictor", None)
        if predictor is None:
            raise ContractError(
                "Ultralytics exposes neither the frozen CNOS legacy route nor "
                "the compatible current SegmentationPredictor"
            )
        ultralytics.yolo = SimpleNamespace(
            v8=SimpleNamespace(segment=SimpleNamespace(SegmentationPredictor=predictor))
        )
        return "CURRENT_MODELS_YOLO_ALIAS"
    if predictor is None:
        raise ContractError("Ultralytics legacy CNOS predictor route is empty")
    return "NATIVE_LEGACY_YOLO_ROUTE"


@dataclass(frozen=True)
class CatalogScore:
    """One proposal scored against one CAD descriptor bank."""

    object_id: int
    descriptor_sha256: str
    raw_cosine: float
    normalized_similarity: float
    top5_template_cosines: tuple[float, float, float, float, float]
    top5_template_indices: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class InstanceProposal:
    """One independent FastSAM instance mask with an all-object CAD ranking."""

    proposal_index: int
    mask: np.ndarray
    proposal_score: float
    mask_stability: float
    cad_ranking: tuple[CatalogScore, ...]


def rank_catalog(scores: Sequence[CatalogScore]) -> tuple[CatalogScore, ...]:
    if len(scores) != len(CATALOG_OBJECT_IDS):
        raise ContractError("Every A-R5 proposal must score all five catalogue objects")
    observed: set[int] = set()
    for score in scores:
        if score.object_id not in CATALOG_OBJECT_IDS or score.object_id in observed:
            raise ContractError(
                "A-R5 CAD score object coverage is duplicate/incomplete"
            )
        observed.add(score.object_id)
        if (
            not isinstance(score.raw_cosine, (int, float))
            or isinstance(score.raw_cosine, bool)
            or not math.isfinite(score.raw_cosine)
            or not -1.0 <= score.raw_cosine <= 1.0
            or not math.isclose(
                score.normalized_similarity,
                normalized_cad_similarity(score.raw_cosine),
                abs_tol=1e-7,
            )
            or len(score.top5_template_cosines) != 5
            or len(score.top5_template_indices) != 5
            or len(set(score.top5_template_indices)) != 5
            or tuple(sorted(score.top5_template_cosines, reverse=True))
            != score.top5_template_cosines
            or any(not 0 <= index < 42 for index in score.top5_template_indices)
            or not math.isclose(
                score.raw_cosine,
                sum(score.top5_template_cosines) / 5.0,
                abs_tol=1e-7,
            )
        ):
            raise ContractError("A-R5 CAD raw/top-5/normalized score trace changed")
    ranked = tuple(
        sorted(scores, key=lambda score: (-score.raw_cosine, score.object_id))
    )
    # The affine transform is strictly monotonic, including negative cosines.
    if [score.object_id for score in ranked] != [
        score.object_id
        for score in sorted(
            scores,
            key=lambda score: (-score.normalized_similarity, score.object_id),
        )
    ]:
        raise ContractError("A-R5 raw and normalized CAD rankings diverged")
    return ranked


def rank_proposals(proposals: Sequence[InstanceProposal]) -> list[InstanceProposal]:
    seen: set[int] = set()
    for proposal in proposals:
        if (
            not isinstance(proposal.proposal_index, int)
            or isinstance(proposal.proposal_index, bool)
            or proposal.proposal_index < 0
            or proposal.proposal_index in seen
        ):
            raise ContractError(
                "A-R5 proposal indices must be unique non-negative ints"
            )
        seen.add(proposal.proposal_index)
        mask = np.asarray(proposal.mask)
        if mask.shape != (1080, 1440) or mask.dtype != np.bool_ or not mask.any():
            raise ContractError(
                "A-R5 FastSAM proposal mask must be non-empty bool 1080x1440"
            )
        if (
            not 0.0 <= proposal.proposal_score <= 1.0
            or not 0.0 <= proposal.mask_stability <= 1.0
        ):
            raise ContractError("A-R5 proposal confidence/stability must be in [0,1]")
        if proposal.cad_ranking != rank_catalog(proposal.cad_ranking):
            raise ContractError("A-R5 proposal CAD ranking is not frozen order")
    return sorted(
        proposals,
        key=lambda proposal: (
            -proposal.proposal_score,
            -proposal.mask_stability,
            proposal.proposal_index,
        ),
    )


def audit_runtime_module_origins(
    source_locks: Mapping[str, Mapping[str, Any]],
    *,
    module_registry: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Prove loaded CNOS/DINO modules are exact files from the locked checkouts."""
    registry = sys.modules if module_registry is None else module_registry
    audited: dict[str, str] = {}

    def verify(name: str, kind: str, expected_relative: str) -> None:
        module = registry.get(name)
        if module is None:
            raise ContractError(f"A-R5 required runtime module was not loaded: {name}")
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise ContractError(f"A-R5 runtime module has no source origin: {name}")
        lock = source_locks[kind]
        record = lock["files"].get(expected_relative)
        if record is None:
            raise ContractError(
                f"A-R5 runtime module is absent from source lock: {name}"
            )
        expected = (
            Path(lock["checkout"]) / Path(*expected_relative.split("/"))
        ).resolve()
        actual = Path(module_file).resolve()
        if (
            actual != expected
            or not actual.is_file()
            or actual.stat().st_size != record["bytes"]
            or sha256_file(actual) != record["sha256"]
        ):
            raise ContractError(f"A-R5 runtime module origin/bytes changed: {name}")
        audited[name] = expected_relative

    for module_name, relative in {
        "src.model.fast_sam": "src/model/fast_sam.py",
        "src.model.dinov2": "src/model/dinov2.py",
        "src.model.utils": "src/model/utils.py",
    }.items():
        verify(module_name, "CNOS", relative)

    loaded_dinov2 = sorted(
        name for name in registry if name == "dinov2" or name.startswith("dinov2.")
    )
    if not loaded_dinov2:
        raise ContractError("A-R5 official DINOv2 runtime loaded no dinov2.* modules")
    checkout = Path(source_locks["DINOV2"]["checkout"]).resolve()
    for name in loaded_dinov2:
        module = registry[name]
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            spec = getattr(module, "__spec__", None)
            locations = getattr(spec, "submodule_search_locations", None)
            observed_locations = [] if locations is None else list(locations)
            expected_relative = Path(*name.split("."))
            if not observed_locations:
                raise ContractError(f"A-R5 DINOv2 module has no source origin: {name}")
            for location in observed_locations:
                if not isinstance(location, str) or not location:
                    raise ContractError(
                        f"A-R5 DINOv2 namespace has an invalid source root: {name}"
                    )
                actual_namespace = Path(location).resolve()
                expected_namespace = (checkout / expected_relative).resolve()
                if (
                    actual_namespace != expected_namespace
                    or not actual_namespace.is_dir()
                ):
                    raise ContractError(
                        f"A-R5 DINOv2 namespace was loaded outside locked checkout: {name}"
                    )
            # Namespace packages execute no bytes. Every loaded child module
            # with a source file is still checked by this same registry walk.
            continue
        actual = Path(module_file).resolve()
        try:
            relative = actual.relative_to(checkout).as_posix()
        except ValueError as exc:
            raise ContractError(
                f"A-R5 DINOv2 module was loaded outside locked checkout: {name}"
            ) from exc
        verify(name, "DINOV2", relative)
    return audited


class OfficialCnosCatalogAdapter:
    """Run pinned CNOS masks and score each instance against the full catalogue."""

    def __init__(
        self,
        *,
        data_root: Path,
        request: Mapping[str, Any],
        device: str,
        delegate: Any | None = None,
    ) -> None:
        self.root = data_root.resolve()
        self.request = dict(request)
        self._source_locks: dict[str, dict[str, Any]] | None = None
        self._module_audited = delegate is not None
        if delegate is not None:
            self._delegate = delegate
        else:
            # Source checkouts must remain byte-for-byte clean for the later
            # independent output validation; never create __pycache__ shadows.
            sys.dont_write_bytecode = True
            self._source_locks = load_source_execution_locks(
                request, data_root=self.root
            )
            self._delegate = OfficialCnosAdapter(
                deployment_root=self.root,
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
                            "relative_path": request["models"][
                                "dinov2_vitl14_checkpoint"
                            ]["relative_path"]
                        },
                    },
                    "adapter_config": {
                        "relative_path": request["adapter_config"]["relative_path"]
                    },
                },
                device=device,
            )

    def _audit_modules_once(self) -> None:
        if self._module_audited:
            return
        if self._source_locks is None:
            raise ContractError("A-R5 source execution locks were not loaded")
        audit_runtime_module_origins(self._source_locks)
        self._module_audited = True

    @staticmethod
    def _same_fastsam_proposal(left: Candidate, right: Candidate) -> bool:
        return (
            left.proposal_index == right.proposal_index
            and math.isclose(left.proposal_score, right.proposal_score, abs_tol=1e-8)
            and math.isclose(left.mask_stability, right.mask_stability, abs_tol=1e-8)
            and np.array_equal(left.mask, right.mask)
        )

    def infer_frame(
        self,
        *,
        rgb_relative_path: str,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[InstanceProposal], list[dict[str, Any]]]:
        """Infer frame proposals without accepting a target object argument."""
        if not self._module_audited:
            ensure_ultralytics_yolo_compat()
            load = getattr(self._delegate, "load", None)
            if not callable(load):
                raise ContractError(
                    "A-R5 official CNOS delegate has no load entrypoint"
                )
            load()
            self._audit_modules_once()
        by_object: dict[int, dict[int, Candidate]] = {}
        oom_evidence: list[dict[str, Any]] = []
        for catalog_item in self.request["catalog"]:
            object_id = catalog_item["object_id"]
            try:
                candidates, events = self._delegate.infer(
                    rgb_relative_path=rgb_relative_path,
                    descriptor_relative_path=catalog_item["descriptor"][
                        "relative_path"
                    ],
                    # The predecessor API discards this value; the descriptor bank
                    # is the sole object identity.  It is catalogue-wide, not a
                    # frame target or association.
                    target_object_id=object_id,
                    proposal_chunk_size=proposal_chunk_size,
                    minimum_chunk_size=minimum_chunk_size,
                )
            except ContractError as exc:
                if "no valid post-geometric-filter proposals" in str(exc):
                    self._audit_modules_once()
                    if by_object:
                        raise ContractError(
                            "FastSAM zero-proposal result changed across CAD catalogue calls"
                        ) from exc
                    return [], []
                raise
            self._audit_modules_once()
            by_object[object_id] = {
                candidate.proposal_index: candidate for candidate in candidates
            }
            for event in events:
                oom_evidence.append({"object_id": object_id, **event})

        if list(by_object) != list(CATALOG_OBJECT_IDS):
            raise ContractError("A-R5 runtime did not score the complete CAD catalogue")
        reference = by_object[CATALOG_OBJECT_IDS[0]]
        proposals: list[InstanceProposal] = []
        for proposal_index, base in reference.items():
            catalog_scores: list[CatalogScore] = []
            for catalog_item in self.request["catalog"]:
                object_id = catalog_item["object_id"]
                candidate = by_object[object_id].get(proposal_index)
                if candidate is None or not self._same_fastsam_proposal(
                    base, candidate
                ):
                    raise ContractError(
                        "FastSAM instance proposal set changed across CAD catalogue calls"
                    )
                catalog_scores.append(
                    CatalogScore(
                        object_id=object_id,
                        descriptor_sha256=catalog_item["descriptor"]["sha256"],
                        raw_cosine=float(candidate.raw_cad_cosine),
                        normalized_similarity=float(candidate.cad_similarity),
                        top5_template_cosines=tuple(
                            float(value) for value in candidate.top5_template_cosines
                        ),
                        top5_template_indices=tuple(candidate.top5_template_indices),
                    )
                )
            proposals.append(
                InstanceProposal(
                    proposal_index=proposal_index,
                    mask=np.asarray(base.mask, dtype=bool),
                    proposal_score=float(base.proposal_score),
                    mask_stability=float(base.mask_stability),
                    cad_ranking=rank_catalog(catalog_scores),
                )
            )
        expected_indices = set(reference)
        if any(
            set(candidates) != expected_indices for candidates in by_object.values()
        ):
            raise ContractError(
                "FastSAM proposal coverage changed across CAD catalogue calls"
            )
        return rank_proposals(proposals), oom_evidence


__all__ = [
    "CatalogScore",
    "InstanceProposal",
    "OfficialCnosCatalogAdapter",
    "audit_runtime_module_origins",
    "rank_catalog",
    "rank_proposals",
]
