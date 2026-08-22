"""A-R6 label-blind selection recalibration over frozen A-R5 proposals.

This revision deliberately does not rerun FastSAM or DINOv2.  It reopens the
exact, independently validated A-R5 candidate population and changes only the
post-proposal selection rule.  That makes every before/after content comparison
causal: masks and CAD scores are identical; only filtering, calibration and
abstention differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.contracts import (
    BOUNDARY_ZERO,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    mask_statistics,
)


PROTOCOL_SCHEMA = "poseloop.pose-accuracy-recovery.instance-selection-protocol.v1r6"
PROTOCOL_ID = "poseloop.pose-accuracy-recovery.development.instance-selection.v1r6"
OUTPUT_SCHEMA = "poseloop.pose-accuracy-recovery.instance-selection-output.v1r6"
RUN_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.instance-selection-receipt.v1r6"
SOURCE_PROTOCOL_ID = (
    "poseloop.pose-accuracy-recovery.development.instance-proposal.v1r5"
)
SOURCE_PROTOCOL_LOCK = (
    "90f23acd34021bd0fe58ff5da2b80e6748005cbe57e5fd2161477a26123d05e2"
)
SOURCE_OUTPUT_SCHEMA = "poseloop.pose-accuracy-recovery.instance-producer-output.v1r5"
SOURCE_VALIDATION_STATUS = "PASS"
SOURCE_OUTPUT_BUNDLE_LOCK = (
    "e35c414efed9cfeac08c50977bff1dd28e2d70e1ec83c34d43265737eab2f2a2"
)
SOURCE_RUN_RECEIPT_LOCK = (
    "468ebda07ccd8c0a0d653d918403432e08f5e3079e10b30a997e4764bf63812d"
)
SOURCE_FRAME_MANIFEST_LOCK = (
    "36afc3d29d7196e73bce274f1d7715b89d964824ab315cc6567beea1a5a1fc53"
)
SOURCE_RUNTIME_REQUEST_LOCK = (
    "32944d62f624da16f65cf89d7de47f10f94a5bab6f1e9ade3c075dec3bd97a45"
)
SOURCE_PRODUCER_MANIFEST_SHA256 = (
    "308c0f14391e330f1f84a1c58a77f1e3d430ee57d9bac0b404b2a311e7b5161e"
)
SOURCE_RUN_RECEIPT_SHA256 = (
    "837e1fe2bbeeafbccbc7c9a3d44891a29f2b6f4cbdd9d369616631f4ffbf5f93"
)
SOURCE_FRAME_MANIFEST_SHA256 = (
    "1597071a4ca49e763022f2fe20a3aa97e4e67d8e848b25abddb847cf8ec1d569"
)
SOURCE_VALIDATION_SHA256 = (
    "937de11962befb09617463a01b2debaee5ebbdbe7f2811a6fe13123c775798f4"
)
SOURCE_PROPOSAL_COUNT = 628

POLICY = {
    "maximum_mask_coverage": 0.25,
    "frame_edge_margin_px": 2,
    "minimum_selection_proposal_score": 0.30,
    "container_minimum_coverage": 0.10,
    "container_minimum_contained_children": 3,
    "container_child_max_area_ratio": 0.80,
    "container_containment_threshold": 0.95,
    "container_bbox_tolerance_px": 3,
    "rectangular_container_fill_threshold": 0.80,
    "rectangular_container_normalized_span_threshold": 0.65,
    "minimum_cad_margin_for_selection": 0.05,
    "calibration_weights": {
        "cad_similarity": 0.970,
        "proposal_score": 0.015,
        "cad_margin": 0.010,
        "mask_stability": 0.005,
    },
    "calibration_formula": (
        "0.970*cad_similarity+0.015*proposal_score+"
        "0.010*cad_margin+0.005*mask_stability"
    ),
    "eligible_tie_break": [
        "calibrated_score_desc",
        "cad_similarity_desc",
        "proposal_score_desc",
        "mask_stability_desc",
        "proposal_index_asc",
    ],
    "low_margin_action": "ABSTAIN_NO_DOWNSTREAM_EXPORT",
    "source_candidate_population_mutation_permitted": False,
}

FILTER_REASON_ORDER = (
    "MAX_COVERAGE",
    "FRAME_EDGE_TOUCH",
    "CONTAINS_MULTIPLE_CHILD_PROPOSALS",
    "RECTANGULAR_CONTAINER_SPAN",
    "LOW_PROPOSAL_SCORE",
)
DECISION_STATES = (
    "SELECTED",
    "ABSTAIN_LOW_CAD_MARGIN",
    "NO_ELIGIBLE_PROPOSAL",
)
VISUAL_ROLES = ("before_after_selection", "filter_audit_overlay")


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _self_lock(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContractError(f"{label}.{field} must be SHA-256")
    observed = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )
    if observed != expected:
        raise ContractError(f"{label}.{field} changed")


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _relative(value: Any, *, root: str, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    path = Path(*value.split("/"))
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != root
    ):
        raise ContractError(f"{label} escapes {root}")
    return path


def _resolve(root: Path, relative: Any, *, root_name: str, label: str) -> Path:
    path = (root / _relative(relative, root=root_name, label=label)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} escapes disk root") from exc
    return path


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ContractError(f"A-R6 output is create-only: {path}") from exc


def _write_image_create_only(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            image.save(stream, format="PNG", optimize=False)
    except FileExistsError as exc:
        raise ContractError(f"A-R6 image is create-only: {path}") from exc


def _asset(path: Path, output_root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "relative_path": path.resolve().relative_to(output_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_protocol(
    protocol: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    protocol = _exact(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "role",
            "source_contract",
            "selection_policy",
            "boundary",
            "scientific_boundary",
            "wrapper_files",
            "protocol_lock_sha256",
        },
        "A-R6 protocol",
    )
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["role"] != "DEVELOPMENT_ONLY_LABEL_BLIND_POST_SELECTION"
    ):
        raise ContractError("A-R6 protocol identity changed")
    source = _exact(
        protocol["source_contract"],
        {
            "protocol_id",
            "protocol_lock_sha256",
            "output_schema",
            "independent_disk_validation_required",
            "candidate_population_mutation_permitted",
            "output_bundle_lock_sha256",
            "run_receipt_lock_sha256",
            "frame_manifest_lock_sha256",
            "runtime_request_lock_sha256",
            "producer_manifest_sha256",
            "run_receipt_sha256",
            "frame_manifest_sha256",
            "independent_validation_sha256",
            "frame_count",
            "scene_count",
            "proposal_count",
        },
        "A-R6 source contract",
    )
    if source != {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "protocol_lock_sha256": SOURCE_PROTOCOL_LOCK,
        "output_schema": SOURCE_OUTPUT_SCHEMA,
        "independent_disk_validation_required": True,
        "candidate_population_mutation_permitted": False,
        "output_bundle_lock_sha256": SOURCE_OUTPUT_BUNDLE_LOCK,
        "run_receipt_lock_sha256": SOURCE_RUN_RECEIPT_LOCK,
        "frame_manifest_lock_sha256": SOURCE_FRAME_MANIFEST_LOCK,
        "runtime_request_lock_sha256": SOURCE_RUNTIME_REQUEST_LOCK,
        "producer_manifest_sha256": SOURCE_PRODUCER_MANIFEST_SHA256,
        "run_receipt_sha256": SOURCE_RUN_RECEIPT_SHA256,
        "frame_manifest_sha256": SOURCE_FRAME_MANIFEST_SHA256,
        "independent_validation_sha256": SOURCE_VALIDATION_SHA256,
        "frame_count": 10,
        "scene_count": 5,
        "proposal_count": SOURCE_PROPOSAL_COUNT,
    }:
        raise ContractError("A-R6 source A-R5 contract changed")
    if protocol["selection_policy"] != POLICY:
        raise ContractError("A-R6 frozen selection policy changed")
    if protocol["boundary"] != BOUNDARY_ZERO:
        raise ContractError("A-R6 zero-access boundary changed")
    scientific = _exact(
        protocol["scientific_boundary"],
        {
            "foundationpose_run_permitted",
            "official_scorer_run_permitted",
            "downstream_export_permitted",
            "human_content_acceptance_required",
            "accuracy_claim_permitted",
        },
        "A-R6 scientific boundary",
    )
    if scientific != {
        "foundationpose_run_permitted": False,
        "official_scorer_run_permitted": False,
        "downstream_export_permitted": False,
        "human_content_acceptance_required": True,
        "accuracy_claim_permitted": False,
    }:
        raise ContractError("A-R6 scientific boundary changed")
    files = protocol["wrapper_files"]
    if not isinstance(files, list) or not files:
        raise ContractError("A-R6 wrapper inventory is empty")
    seen: set[str] = set()
    for index, raw in enumerate(files):
        item = _exact(
            raw, {"relative_path", "bytes", "sha256"}, f"wrapper_files[{index}]"
        )
        relative = _relative(
            item["relative_path"],
            root="pose_accuracy_recovery_prep",
            label=f"wrapper_files[{index}].relative_path",
        )
        name = relative.as_posix()
        if name in seen:
            raise ContractError("A-R6 wrapper inventory contains duplicates")
        seen.add(name)
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ContractError("A-R6 wrapper bytes must be positive")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ContractError("A-R6 wrapper SHA must be SHA-256")
        if repository_root is not None:
            path = _resolve(
                repository_root,
                name,
                root_name="pose_accuracy_recovery_prep",
                label=f"wrapper_files[{index}]",
            )
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ContractError(f"A-R6 wrapper changed: {name}")
    required = {
        "pose_accuracy_recovery_prep/__init__.py",
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/__init__.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/contracts.py",
        "pose_accuracy_recovery_prep/instance_selection_v1r6.py",
    }
    if seen != required:
        raise ContractError("A-R6 wrapper inventory is not the exact execution closure")
    _self_lock(protocol, "protocol_lock_sha256", "A-R6 protocol")
    return dict(protocol)


def _verify_file_asset(
    asset: Any,
    *,
    disk_root: Path,
    root_name: str,
    role: str,
    label: str,
) -> Path:
    asset = _exact(asset, {"role", "relative_path", "bytes", "sha256"}, label)
    if asset["role"] != role:
        raise ContractError(f"{label}.role changed")
    path = _resolve(disk_root, asset["relative_path"], root_name=root_name, label=label)
    if (
        not path.is_file()
        or path.stat().st_size != asset["bytes"]
        or sha256_file(path) != asset["sha256"]
    ):
        raise ContractError(f"{label} changed on disk")
    return path


def _verify_frozen_source_files(
    *,
    source_manifest_path: Path,
    source_receipt_path: Path,
    source_validation_path: Path,
    frame_manifest_path: Path,
) -> None:
    expected = {
        "source A-R5 producer manifest": (
            source_manifest_path,
            SOURCE_PRODUCER_MANIFEST_SHA256,
        ),
        "source A-R5 run receipt": (
            source_receipt_path,
            SOURCE_RUN_RECEIPT_SHA256,
        ),
        "source A-R5 independent validation": (
            source_validation_path,
            SOURCE_VALIDATION_SHA256,
        ),
        "source A-R5 frame manifest": (
            frame_manifest_path,
            SOURCE_FRAME_MANIFEST_SHA256,
        ),
    }
    for label, (path, expected_sha256) in expected.items():
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ContractError(f"{label} is not the exact frozen A-R5 asset")


def _validate_source(
    *,
    source_manifest: Mapping[str, Any],
    source_run_receipt: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    source_output_root: Path,
    frame_manifest: Mapping[str, Any],
    data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    frames = source_manifest.get("frames")
    if not isinstance(frames, list) or any(
        not isinstance(frame, Mapping)
        or not isinstance(frame.get("proposal_count"), int)
        or isinstance(frame.get("proposal_count"), bool)
        or frame["proposal_count"] < 0
        for frame in frames
    ):
        raise ContractError("A-R6 source frames must be objects")
    if (
        source_manifest.get("schema_version") != SOURCE_OUTPUT_SCHEMA
        or source_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_manifest.get("protocol_lock_sha256") != SOURCE_PROTOCOL_LOCK
        or source_manifest.get("status") != "COMPLETE"
        or source_manifest.get("frame_count") != 10
        or source_manifest.get("scene_count") != 5
        or sum(frame.get("proposal_count", -1) for frame in frames)
        != SOURCE_PROPOSAL_COUNT
        or source_manifest.get("catalog_object_ids") != [1, 2, 4, 5, 6]
        or source_manifest.get("boundary") != BOUNDARY_ZERO
        or source_manifest.get("output_bundle_lock_sha256") != SOURCE_OUTPUT_BUNDLE_LOCK
        or source_manifest.get("frame_manifest_lock_sha256")
        != SOURCE_FRAME_MANIFEST_LOCK
        or source_manifest.get("runtime_request_lock_sha256")
        != SOURCE_RUNTIME_REQUEST_LOCK
    ):
        raise ContractError("A-R6 source A-R5 manifest identity/boundary changed")
    _self_lock(source_manifest, "output_bundle_lock_sha256", "source A-R5 output")
    if (
        source_run_receipt.get("status") != "COMPLETE"
        or source_run_receipt.get("frame_count") != 10
        or source_run_receipt.get("scene_count") != 5
        or source_run_receipt.get("boundary") != BOUNDARY_ZERO
        or source_run_receipt.get("run_receipt_lock_sha256") != SOURCE_RUN_RECEIPT_LOCK
    ):
        raise ContractError("A-R6 source A-R5 run receipt changed")
    _self_lock(source_run_receipt, "run_receipt_lock_sha256", "source A-R5 receipt")
    output_asset = _verify_file_asset(
        source_run_receipt["output_bundle"],
        disk_root=source_output_root,
        root_name="outputs",
        role="producer_output_manifest",
        label="source A-R5 manifest asset",
    )
    if read_json(output_asset) != source_manifest:
        raise ContractError("A-R6 source manifest asset content changed")
    if (
        source_validation.get("status") != SOURCE_VALIDATION_STATUS
        or source_validation.get("disk_assets_reverified") is not True
        or source_validation.get("frame_count") != 10
        or source_validation.get("scene_count") != 5
        or source_validation.get("boundary") != BOUNDARY_ZERO
        or source_validation.get("output_bundle_lock_sha256")
        != source_manifest["output_bundle_lock_sha256"]
        or source_validation.get("run_receipt_lock_sha256")
        != source_run_receipt["run_receipt_lock_sha256"]
    ):
        raise ContractError("A-R6 source independent validation receipt changed")
    _self_lock(frame_manifest, "frame_manifest_lock_sha256", "source frame manifest")
    if (
        frame_manifest.get("frame_count") != 10
        or frame_manifest.get("scene_count") != 5
        or source_manifest.get("frame_manifest_lock_sha256")
        != frame_manifest["frame_manifest_lock_sha256"]
        or frame_manifest.get("frame_manifest_lock_sha256")
        != SOURCE_FRAME_MANIFEST_LOCK
    ):
        raise ContractError("A-R6 source frame coverage/lock changed")
    manifest_frames = frame_manifest.get("frames")
    if not isinstance(frames, list) or not isinstance(manifest_frames, list):
        raise ContractError("A-R6 source frames must be lists")
    manifest_by_item = {frame.get("item_id"): frame for frame in manifest_frames}
    if len(frames) != 10 or len(manifest_by_item) != 10:
        raise ContractError("A-R6 source frame count/identity changed")
    validated: list[dict[str, Any]] = []
    by_item: dict[str, dict[str, Any]] = {}
    for frame in frames:
        item_id = frame.get("item_id")
        manifest_frame = manifest_by_item.get(item_id)
        if not isinstance(item_id, str) or not isinstance(manifest_frame, Mapping):
            raise ContractError("A-R6 source item is absent from frame manifest")
        inputs = manifest_frame.get("inputs")
        if not isinstance(inputs, Mapping) or "rgb" not in inputs:
            raise ContractError("A-R6 source frame lacks RGB binding")
        rgb_path = _verify_file_asset(
            inputs["rgb"],
            disk_root=data_root,
            root_name="inputs",
            role="rgb",
            label=f"source RGB {item_id}",
        )
        with Image.open(rgb_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ContractError("A-R6 source RGB decode changed")
        proposals = frame.get("proposals")
        if not isinstance(proposals, list) or len(proposals) != frame.get(
            "proposal_count"
        ):
            raise ContractError("A-R6 source proposal coverage changed")
        seen: set[int] = set()
        for proposal in proposals:
            index = proposal.get("proposal_index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in seen
            ):
                raise ContractError("A-R6 source proposal index changed")
            seen.add(index)
            mask_path = _verify_file_asset(
                {
                    key: proposal["mask"][key]
                    for key in ("role", "relative_path", "bytes", "sha256")
                },
                disk_root=source_output_root,
                root_name="outputs",
                role="independent_instance_mask",
                label=f"source mask {item_id}/{index}",
            )
            with Image.open(mask_path) as image:
                values = np.asarray(image.convert("L")) > 0
            statistics = mask_statistics(values)
            if (
                proposal["mask"].get("decoded_shape") != [FRAME_HEIGHT, FRAME_WIDTH]
                or proposal["mask"].get("decoded_mode") != "L"
                or proposal["mask"].get("foreground_pixels")
                != statistics["mask_pixels"]
                or proposal["mask"].get("mask_pixels") != statistics["mask_pixels"]
                or proposal["mask"].get("bbox_xyxy_half_open")
                not in (None, statistics["bbox_xyxy_half_open"])
                or proposal["mask"].get("support_bbox_xyxy_half_open")
                != statistics["bbox_xyxy_half_open"]
                or proposal.get("bbox_xyxy_half_open")
                != statistics["bbox_xyxy_half_open"]
                or not math.isclose(
                    proposal["mask"].get("coverage", -1),
                    statistics["coverage"],
                    abs_tol=1e-12,
                )
            ):
                raise ContractError("A-R6 source mask statistics changed")
            ranking = proposal.get("cad_ranking")
            if not isinstance(ranking, list) or len(ranking) != 5:
                raise ContractError("A-R6 source CAD ranking changed")
            for rank, score in enumerate(ranking, start=1):
                if score.get("rank") != rank:
                    raise ContractError("A-R6 source CAD rank order changed")
                _finite(score.get("normalized_similarity"), "source CAD similarity")
            if proposal.get("selected_object_id") != ranking[0].get(
                "object_id"
            ) or not math.isclose(
                _finite(proposal.get("selected_cad_similarity"), "selected CAD"),
                _finite(ranking[0].get("normalized_similarity"), "top CAD"),
                abs_tol=1e-12,
            ):
                raise ContractError("A-R6 source selected CAD identity changed")
            _finite(proposal.get("proposal_score"), "source proposal score")
            _finite(proposal.get("mask_stability"), "source mask stability")
        record = {
            "source_frame": dict(frame),
            "manifest_frame": dict(manifest_frame),
            "rgb_path": rgb_path,
        }
        validated.append(record)
        by_item[item_id] = record
    return validated, by_item


def _load_frame_masks(
    source_frame: Mapping[str, Any], source_output_root: Path
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for proposal in source_frame["proposals"]:
        path = _resolve(
            source_output_root,
            proposal["mask"]["relative_path"],
            root_name="outputs",
            label="source proposal mask",
        )
        with Image.open(path) as image:
            masks.append(np.asarray(image.convert("L")) > 0)
    return masks


def _contained_children(
    proposals: Sequence[Mapping[str, Any]], masks: Sequence[np.ndarray], index: int
) -> int:
    parent = proposals[index]
    parent_area = parent["mask"]["mask_pixels"]
    px0, py0, px1, py1 = parent["bbox_xyxy_half_open"]
    tolerance = POLICY["container_bbox_tolerance_px"]
    count = 0
    for child_index, child in enumerate(proposals):
        if child_index == index:
            continue
        child_area = child["mask"]["mask_pixels"]
        if child_area > parent_area * POLICY["container_child_max_area_ratio"]:
            continue
        cx0, cy0, cx1, cy1 = child["bbox_xyxy_half_open"]
        if (
            cx0 < px0 - tolerance
            or cy0 < py0 - tolerance
            or cx1 > px1 + tolerance
            or cy1 > py1 + tolerance
        ):
            continue
        child_crop = masks[child_index][cy0:cy1, cx0:cx1]
        parent_crop = masks[index][cy0:cy1, cx0:cx1]
        overlap = int(np.logical_and(parent_crop, child_crop).sum())
        if overlap / max(1, child_area) >= POLICY["container_containment_threshold"]:
            count += 1
    return count


def _calibrated_score(proposal: Mapping[str, Any], cad_margin: float) -> float:
    weights = POLICY["calibration_weights"]
    return (
        weights["cad_similarity"] * proposal["selected_cad_similarity"]
        + weights["proposal_score"] * proposal["proposal_score"]
        + weights["cad_margin"] * cad_margin
        + weights["mask_stability"] * proposal["mask_stability"]
    )


def select_frame(
    source_frame: Mapping[str, Any], masks: Sequence[np.ndarray]
) -> dict[str, Any]:
    proposals = source_frame["proposals"]
    if len(proposals) != len(masks):
        raise ContractError("A-R6 proposal/mask coverage differs")
    audits: list[dict[str, Any]] = []
    edge = POLICY["frame_edge_margin_px"]
    for index, (proposal, mask) in enumerate(zip(proposals, masks, strict=True)):
        statistics = mask_statistics(np.asarray(mask, dtype=bool))
        if statistics["mask_pixels"] != proposal["mask"]["mask_pixels"]:
            raise ContractError("A-R6 in-memory source mask changed")
        x0, y0, x1, y1 = statistics["bbox_xyxy_half_open"]
        bbox_fraction = ((x1 - x0) * (y1 - y0)) / (FRAME_WIDTH * FRAME_HEIGHT)
        bbox_fill = statistics["coverage"] / max(bbox_fraction, 1e-12)
        normalized_span = max((x1 - x0) / FRAME_WIDTH, (y1 - y0) / FRAME_HEIGHT)
        touches_edge = (
            x0 <= edge
            or y0 <= edge
            or x1 >= FRAME_WIDTH - edge
            or y1 >= FRAME_HEIGHT - edge
        )
        children = _contained_children(proposals, masks, index)
        margin = (
            proposal["cad_ranking"][0]["normalized_similarity"]
            - proposal["cad_ranking"][1]["normalized_similarity"]
        )
        reasons: list[str] = []
        if statistics["coverage"] > POLICY["maximum_mask_coverage"]:
            reasons.append("MAX_COVERAGE")
        if touches_edge:
            reasons.append("FRAME_EDGE_TOUCH")
        if (
            statistics["coverage"] >= POLICY["container_minimum_coverage"]
            and children >= POLICY["container_minimum_contained_children"]
        ):
            reasons.append("CONTAINS_MULTIPLE_CHILD_PROPOSALS")
        if (
            statistics["coverage"] >= POLICY["container_minimum_coverage"]
            and bbox_fill >= POLICY["rectangular_container_fill_threshold"]
            and normalized_span
            >= POLICY["rectangular_container_normalized_span_threshold"]
        ):
            reasons.append("RECTANGULAR_CONTAINER_SPAN")
        if proposal["proposal_score"] < POLICY["minimum_selection_proposal_score"]:
            reasons.append("LOW_PROPOSAL_SCORE")
        reasons = [reason for reason in FILTER_REASON_ORDER if reason in reasons]
        score = _calibrated_score(proposal, margin)
        audits.append(
            {
                "proposal_index": proposal["proposal_index"],
                "selected_object_id": proposal["selected_object_id"],
                "coverage": statistics["coverage"],
                "bbox_fill_ratio": bbox_fill,
                "maximum_normalized_bbox_span": normalized_span,
                "touches_frame_edge": touches_edge,
                "contained_child_count": children,
                "proposal_score": proposal["proposal_score"],
                "mask_stability": proposal["mask_stability"],
                "cad_similarity": proposal["selected_cad_similarity"],
                "cad_margin": margin,
                "calibrated_score": score,
                "filter_reasons": reasons,
                "eligible": not reasons,
            }
        )
    eligible = [audit for audit in audits if audit["eligible"]]
    best: dict[str, Any] | None = None
    if eligible:
        best = sorted(
            eligible,
            key=lambda audit: (
                -audit["calibrated_score"],
                -audit["cad_similarity"],
                -audit["proposal_score"],
                -audit["mask_stability"],
                audit["proposal_index"],
            ),
        )[0]
    if best is None:
        state = "NO_ELIGIBLE_PROPOSAL"
        selected_index = None
    elif best["cad_margin"] < POLICY["minimum_cad_margin_for_selection"]:
        state = "ABSTAIN_LOW_CAD_MARGIN"
        selected_index = None
    else:
        state = "SELECTED"
        selected_index = best["proposal_index"]
    if state not in DECISION_STATES:
        raise ContractError("A-R6 decision state changed")
    source_selected = source_frame["selected_proposal_index"]
    return {
        "item_id": source_frame["item_id"],
        "source_proposal_count": len(proposals),
        "eligible_proposal_count": len(eligible),
        "rejected_proposal_count": len(proposals) - len(eligible),
        "source_selected_proposal_index": source_selected,
        "best_candidate_proposal_index": None
        if best is None
        else best["proposal_index"],
        "selected_proposal_index": selected_index,
        "selected_object_id": None
        if state != "SELECTED"
        else best["selected_object_id"],
        "decision_state": state,
        "changed_from_a_r5": selected_index != source_selected,
        "candidate_audits": audits,
    }


def _contour(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values, 1)
    eroded = np.logical_and.reduce(
        [
            padded[1 + dy : 1 + dy + FRAME_HEIGHT, 1 + dx : 1 + dx + FRAME_WIDTH]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    return values & ~eroded


def _overlay(
    rgb: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int]
) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    if mask is not None:
        tint = np.asarray(color, dtype=np.float32)
        output[mask] = output[mask] * 0.45 + tint * 0.55
        output[_contour(mask)] = tint
    return np.clip(output, 0, 255).astype(np.uint8)


def _panel(image: np.ndarray, lines: Sequence[str]) -> Image.Image:
    result = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(result)
    height = 12 + 22 * len(lines)
    draw.rectangle((0, 0, min(FRAME_WIDTH, 1040), height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((8, 8 + 22 * index), line, fill=(255, 255, 255))
    return result


def _comparison_visual(
    rgb: np.ndarray,
    source_frame: Mapping[str, Any],
    decision: Mapping[str, Any],
    masks: Sequence[np.ndarray],
) -> Image.Image:
    index_by_proposal = {
        proposal["proposal_index"]: index
        for index, proposal in enumerate(source_frame["proposals"])
    }
    old_index = decision["source_selected_proposal_index"]
    old_mask = None if old_index is None else masks[index_by_proposal[old_index]]
    chosen_index = decision["selected_proposal_index"]
    display_index = (
        chosen_index
        if chosen_index is not None
        else decision["best_candidate_proposal_index"]
    )
    new_mask = (
        None if display_index is None else masks[index_by_proposal[display_index]]
    )
    left = _panel(
        _overlay(rgb, old_mask, (255, 48, 48)),
        [
            "A-R5 before: CAD-only selection",
            f"selected={old_index}",
        ],
    )
    color = (48, 255, 96) if chosen_index is not None else (255, 176, 32)
    right = _panel(
        _overlay(rgb, new_mask, color),
        [
            "A-R6 after: hard filters + calibrated rank",
            f"state={decision['decision_state']}",
            f"selected={chosen_index}; best_candidate={display_index}",
        ],
    )
    joined = Image.new("RGB", (FRAME_WIDTH * 2, FRAME_HEIGHT))
    joined.paste(left, (0, 0))
    joined.paste(right, (FRAME_WIDTH, 0))
    return joined


def _filter_visual(
    rgb: np.ndarray,
    source_frame: Mapping[str, Any],
    decision: Mapping[str, Any],
    masks: Sequence[np.ndarray],
) -> Image.Image:
    output = rgb.copy()
    best = decision["best_candidate_proposal_index"]
    for proposal, audit, mask in zip(
        source_frame["proposals"], decision["candidate_audits"], masks, strict=True
    ):
        if proposal["proposal_index"] == best:
            color = np.asarray((32, 240, 255), dtype=np.uint8)
        elif audit["eligible"]:
            color = np.asarray((64, 255, 96), dtype=np.uint8)
        else:
            color = np.asarray((255, 64, 64), dtype=np.uint8)
        output[_contour(mask)] = color
    counts = {
        reason: sum(
            reason in audit["filter_reasons"] for audit in decision["candidate_audits"]
        )
        for reason in FILTER_REASON_ORDER
    }
    return _panel(
        output,
        [
            "A-R6 filter audit: red=rejected, green=eligible, cyan=best",
            f"eligible={decision['eligible_proposal_count']} rejected={decision['rejected_proposal_count']}",
            " ".join(f"{reason}={counts[reason]}" for reason in FILTER_REASON_ORDER),
        ],
    )


def _source_identity(
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    source_validation_path: Path,
) -> dict[str, Any]:
    return {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "protocol_lock_sha256": SOURCE_PROTOCOL_LOCK,
        "output_bundle_lock_sha256": source_manifest["output_bundle_lock_sha256"],
        "producer_manifest_bytes": source_manifest_path.stat().st_size,
        "producer_manifest_sha256": sha256_file(source_manifest_path),
        "run_receipt_lock_sha256": source_receipt["run_receipt_lock_sha256"],
        "run_receipt_bytes": source_receipt_path.stat().st_size,
        "run_receipt_sha256": sha256_file(source_receipt_path),
        "independent_validation_bytes": source_validation_path.stat().st_size,
        "independent_validation_sha256": sha256_file(source_validation_path),
        "frame_manifest_lock_sha256": source_manifest["frame_manifest_lock_sha256"],
        "runtime_request_lock_sha256": source_manifest["runtime_request_lock_sha256"],
    }


def _aggregate(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = {
        reason: sum(
            reason in audit["filter_reasons"]
            for frame in frames
            for audit in frame["candidate_audits"]
        )
        for reason in FILTER_REASON_ORDER
    }
    return {
        "frame_count": len(frames),
        "source_proposal_count": sum(
            frame["source_proposal_count"] for frame in frames
        ),
        "eligible_proposal_count": sum(
            frame["eligible_proposal_count"] for frame in frames
        ),
        "rejected_proposal_count": sum(
            frame["rejected_proposal_count"] for frame in frames
        ),
        "changed_selection_frame_count": sum(
            frame["changed_from_a_r5"] for frame in frames
        ),
        "selected_frame_count": sum(
            frame["decision_state"] == "SELECTED" for frame in frames
        ),
        "abstained_frame_count": sum(
            frame["decision_state"] == "ABSTAIN_LOW_CAD_MARGIN" for frame in frames
        ),
        "no_eligible_frame_count": sum(
            frame["decision_state"] == "NO_ELIGIBLE_PROPOSAL" for frame in frames
        ),
        "filter_reason_counts": reasons,
    }


def _build_output(
    *,
    protocol: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    source_validation_path: Path,
    source_output_root: Path,
    output_root: Path,
    write_visuals: bool,
) -> tuple[dict[str, Any], dict[str, tuple[Image.Image, Image.Image]]]:
    frames: list[dict[str, Any]] = []
    expected_visuals: dict[str, tuple[Image.Image, Image.Image]] = {}
    for record in source_records:
        source_frame = record["source_frame"]
        masks = _load_frame_masks(source_frame, source_output_root)
        decision = select_frame(source_frame, masks)
        with Image.open(record["rgb_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        comparison = _comparison_visual(rgb, source_frame, decision, masks)
        audit_visual = _filter_visual(rgb, source_frame, decision, masks)
        item_id = source_frame["item_id"]
        expected_visuals[item_id] = (comparison, audit_visual)
        visual_assets: dict[str, Any] = {}
        if write_visuals:
            paths = {
                "before_after_selection": output_root
                / "outputs"
                / "visualizations"
                / "before_after_selection"
                / f"{item_id}.png",
                "filter_audit_overlay": output_root
                / "outputs"
                / "visualizations"
                / "filter_audit_overlay"
                / f"{item_id}.png",
            }
            for role, image in zip(
                VISUAL_ROLES, (comparison, audit_visual), strict=True
            ):
                _write_image_create_only(paths[role], image)
                visual_assets[role] = _asset(paths[role], output_root, role)
        frames.append({**decision, "visualizations": visual_assets})
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "role": "DEVELOPMENT_ONLY_LABEL_BLIND_POST_SELECTION",
        "source_identity": _source_identity(
            source_manifest_path,
            source_manifest,
            source_receipt_path,
            source_receipt,
            source_validation_path,
        ),
        "selection_policy": POLICY,
        "frames": frames,
        "aggregate": _aggregate(frames),
        "boundary": dict(BOUNDARY_ZERO),
        "human_content_acceptance": "UNASSESSED",
        "downstream_export_permitted": False,
        "foundationpose_run_count": 0,
        "official_scorer_run_count": 0,
        "output_lock_sha256": "pending",
    }
    output["output_lock_sha256"] = canonical_sha256(
        {key: value for key, value in output.items() if key != "output_lock_sha256"}
    )
    return output, expected_visuals


def run_selection(
    *,
    protocol: Mapping[str, Any],
    source_manifest_path: Path,
    source_receipt_path: Path,
    source_validation_path: Path,
    frame_manifest_path: Path,
    data_root: Path,
    source_output_root: Path,
    output_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_protocol(protocol, repository_root=repository_root)
    _verify_frozen_source_files(
        source_manifest_path=source_manifest_path,
        source_receipt_path=source_receipt_path,
        source_validation_path=source_validation_path,
        frame_manifest_path=frame_manifest_path,
    )
    if output_root.exists():
        raise ContractError("A-R6 output root is create-only")
    stage = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if stage.exists():
        raise ContractError("A-R6 staging root already exists")
    source_manifest = _load_object(source_manifest_path, "source manifest")
    source_receipt = _load_object(source_receipt_path, "source receipt")
    source_validation = _load_object(source_validation_path, "source validation")
    frame_manifest = _load_object(frame_manifest_path, "source frame manifest")
    records, _ = _validate_source(
        source_manifest=source_manifest,
        source_run_receipt=source_receipt,
        source_validation=source_validation,
        source_output_root=source_output_root,
        frame_manifest=frame_manifest,
        data_root=data_root,
    )
    output, _ = _build_output(
        protocol=protocol,
        source_records=records,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_receipt_path=source_receipt_path,
        source_receipt=source_receipt,
        source_validation_path=source_validation_path,
        source_output_root=source_output_root,
        output_root=stage,
        write_visuals=True,
    )
    manifest_path = stage / "outputs" / "selection-manifest.json"
    _write_create_only(manifest_path, _json_bytes(output))
    receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "source_output_bundle_lock_sha256": source_manifest[
            "output_bundle_lock_sha256"
        ],
        "selection_output": _asset(manifest_path, stage, "selection_manifest"),
        "aggregate": output["aggregate"],
        "boundary": dict(BOUNDARY_ZERO),
        "model_run_count": 0,
        "foundationpose_run_count": 0,
        "official_scorer_run_count": 0,
        "downstream_export_count": 0,
        "run_receipt_lock_sha256": "pending",
    }
    receipt["run_receipt_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "run_receipt_lock_sha256"
        }
    )
    _write_create_only(stage / "outputs" / "run-receipt.json", _json_bytes(receipt))
    os.replace(stage, output_root)
    return output, receipt


def validate_selection_output(
    *,
    protocol: Mapping[str, Any],
    source_manifest_path: Path,
    source_receipt_path: Path,
    source_validation_path: Path,
    frame_manifest_path: Path,
    data_root: Path,
    source_output_root: Path,
    output_root: Path,
    selection_manifest_path: Path,
    selection_receipt_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    validate_protocol(protocol, repository_root=repository_root)
    _verify_frozen_source_files(
        source_manifest_path=source_manifest_path,
        source_receipt_path=source_receipt_path,
        source_validation_path=source_validation_path,
        frame_manifest_path=frame_manifest_path,
    )
    source_manifest = _load_object(source_manifest_path, "source manifest")
    source_receipt = _load_object(source_receipt_path, "source receipt")
    source_validation = _load_object(source_validation_path, "source validation")
    frame_manifest = _load_object(frame_manifest_path, "source frame manifest")
    records, _ = _validate_source(
        source_manifest=source_manifest,
        source_run_receipt=source_receipt,
        source_validation=source_validation,
        source_output_root=source_output_root,
        frame_manifest=frame_manifest,
        data_root=data_root,
    )
    observed = _load_object(selection_manifest_path, "A-R6 selection manifest")
    _self_lock(observed, "output_lock_sha256", "A-R6 output")
    expected, visuals = _build_output(
        protocol=protocol,
        source_records=records,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_receipt_path=source_receipt_path,
        source_receipt=source_receipt,
        source_validation_path=source_validation_path,
        source_output_root=source_output_root,
        output_root=output_root,
        write_visuals=False,
    )
    expected_frames = {frame["item_id"]: frame for frame in expected["frames"]}
    observed_frames = observed.get("frames")
    if not isinstance(observed_frames, list) or len(observed_frames) != 10:
        raise ContractError("A-R6 output frame coverage changed")
    reconstructed_frames: list[dict[str, Any]] = []
    for observed_frame in observed_frames:
        item_id = observed_frame.get("item_id")
        expected_frame = expected_frames.get(item_id)
        if expected_frame is None:
            raise ContractError("A-R6 output contains unknown frame")
        visuals_record = observed_frame.get("visualizations")
        if not isinstance(visuals_record, Mapping) or set(visuals_record) != set(
            VISUAL_ROLES
        ):
            raise ContractError("A-R6 visualization coverage changed")
        stripped = {**observed_frame, "visualizations": {}}
        if stripped != expected_frame:
            raise ContractError("A-R6 selection decision/features changed")
        reconstructed_frames.append(
            {**expected_frame, "visualizations": dict(visuals_record)}
        )
        expected_images = visuals[item_id]
        for role, expected_image in zip(VISUAL_ROLES, expected_images, strict=True):
            path = _verify_file_asset(
                visuals_record[role],
                disk_root=output_root,
                root_name="outputs",
                role=role,
                label=f"A-R6 visualization {item_id}/{role}",
            )
            with Image.open(path) as image:
                image.load()
                if not np.array_equal(
                    np.asarray(image.convert("RGB")), np.asarray(expected_image)
                ):
                    raise ContractError("A-R6 visualization pixels changed")
        del expected_frames[item_id]
    if expected_frames:
        raise ContractError("A-R6 output misses frames")
    expected["frames"] = reconstructed_frames
    expected["output_lock_sha256"] = canonical_sha256(
        {key: value for key, value in expected.items() if key != "output_lock_sha256"}
    )
    if observed != expected:
        raise ContractError("A-R6 output aggregate/identity changed")
    receipt = _load_object(selection_receipt_path, "A-R6 run receipt")
    _self_lock(receipt, "run_receipt_lock_sha256", "A-R6 run receipt")
    if (
        receipt.get("schema_version") != RUN_RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE"
        or receipt.get("protocol_lock_sha256") != protocol["protocol_lock_sha256"]
        or receipt.get("source_output_bundle_lock_sha256")
        != source_manifest["output_bundle_lock_sha256"]
        or receipt.get("aggregate") != observed["aggregate"]
        or receipt.get("boundary") != BOUNDARY_ZERO
        or any(
            receipt.get(field) != 0
            for field in (
                "model_run_count",
                "foundationpose_run_count",
                "official_scorer_run_count",
                "downstream_export_count",
            )
        )
    ):
        raise ContractError("A-R6 run receipt changed")
    _verify_file_asset(
        receipt["selection_output"],
        disk_root=output_root,
        root_name="outputs",
        role="selection_manifest",
        label="A-R6 selection manifest asset",
    )
    return {
        "status": "PASS_INDEPENDENT_DISK_VALIDATION",
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "source_output_bundle_lock_sha256": source_manifest[
            "output_bundle_lock_sha256"
        ],
        "selection_output_lock_sha256": observed["output_lock_sha256"],
        "run_receipt_lock_sha256": receipt["run_receipt_lock_sha256"],
        "aggregate": observed["aggregate"],
        "boundary": dict(BOUNDARY_ZERO),
        "human_content_acceptance": "UNASSESSED",
        "downstream_export_permitted": False,
    }


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    _write_create_only(output, _json_bytes(value))


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-run-receipt", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.instance_selection_v1r6",
        description=(
            "Filter and recalibrate the exact frozen A-R5 proposal population; "
            "no model, labels, FoundationPose or official scorer are opened."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser("validate-protocol")
    protocol.add_argument("--protocol", type=Path, required=True)
    protocol.add_argument("--repository-root", type=Path)
    protocol.add_argument("--output", type=Path)
    preflight = commands.add_parser("preflight")
    _common_inputs(preflight)
    preflight.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    _common_inputs(run)
    run.add_argument("--output-root", type=Path, required=True)
    validate = commands.add_parser("validate-output")
    _common_inputs(validate)
    validate.add_argument("--output-root", type=Path, required=True)
    validate.add_argument("--selection-manifest", type=Path, required=True)
    validate.add_argument("--selection-receipt", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = _load_object(args.protocol, "A-R6 protocol")
        if args.command == "validate-protocol":
            validated = validate_protocol(
                protocol, repository_root=args.repository_root
            )
            _emit(
                {
                    "status": "PASS_PROTOCOL",
                    "protocol_id": validated["protocol_id"],
                    "protocol_lock_sha256": validated["protocol_lock_sha256"],
                    "model_run": False,
                    "server_connected": False,
                },
                args.output,
            )
            return 0
        if args.command == "run":
            output, receipt = run_selection(
                protocol=protocol,
                source_manifest_path=args.source_manifest,
                source_receipt_path=args.source_run_receipt,
                source_validation_path=args.source_validation,
                frame_manifest_path=args.frame_manifest,
                data_root=args.data_root,
                source_output_root=args.source_output_root,
                output_root=args.output_root,
                repository_root=args.repository_root,
            )
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "output_lock_sha256": output["output_lock_sha256"],
                        "run_receipt_lock_sha256": receipt["run_receipt_lock_sha256"],
                        "aggregate": output["aggregate"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "preflight":
            validate_protocol(protocol, repository_root=args.repository_root)
            _verify_frozen_source_files(
                source_manifest_path=args.source_manifest,
                source_receipt_path=args.source_run_receipt,
                source_validation_path=args.source_validation,
                frame_manifest_path=args.frame_manifest,
            )
            source_manifest = _load_object(args.source_manifest, "source manifest")
            source_receipt = _load_object(args.source_run_receipt, "source run receipt")
            source_validation = _load_object(
                args.source_validation, "source validation"
            )
            frame_manifest = _load_object(args.frame_manifest, "frame manifest")
            records, _ = _validate_source(
                source_manifest=source_manifest,
                source_run_receipt=source_receipt,
                source_validation=source_validation,
                source_output_root=args.source_output_root,
                frame_manifest=frame_manifest,
                data_root=args.data_root,
            )
            _emit(
                {
                    "status": "PASS_PREFLIGHT_EXACT_A_R5_SOURCE",
                    "protocol_lock_sha256": protocol["protocol_lock_sha256"],
                    "source_output_bundle_lock_sha256": source_manifest[
                        "output_bundle_lock_sha256"
                    ],
                    "frame_count": len(records),
                    "source_proposal_count": sum(
                        len(record["source_frame"]["proposals"]) for record in records
                    ),
                    "boundary": dict(BOUNDARY_ZERO),
                    "model_run": False,
                },
                args.output,
            )
            return 0
        result = validate_selection_output(
            protocol=protocol,
            source_manifest_path=args.source_manifest,
            source_receipt_path=args.source_run_receipt,
            source_validation_path=args.source_validation,
            frame_manifest_path=args.frame_manifest,
            data_root=args.data_root,
            source_output_root=args.source_output_root,
            output_root=args.output_root,
            selection_manifest_path=args.selection_manifest,
            selection_receipt_path=args.selection_receipt,
            repository_root=args.repository_root,
        )
        _emit(result, args.output)
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_STATES",
    "FILTER_REASON_ORDER",
    "OUTPUT_SCHEMA",
    "POLICY",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RUN_RECEIPT_SCHEMA",
    "select_frame",
    "validate_protocol",
    "validate_selection_output",
]
