"""Strict, label-blind contracts for an independent instance-proposal stage."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)

from . import (
    CONTENT_GATE_SCHEMA,
    INPUT_MANIFEST_SCHEMA,
    PROPOSAL_SCHEMA,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    RUNTIME_LOCK_SCHEMA,
)

ALLOWED_INPUT_ROLES = ["rgb", "camera", "cad", "cad_render_descriptors"]
VISUALIZATION_ROLES = ["rgb", "proposal_overview", "selected_mask", "contours"]
FRAME_SIZE = {"width": 1440, "height": 1080}
MASK_CONTRACT = {
    "format": "PNG",
    "mode": "L",
    "background_value": 0,
    "foreground_value": 255,
    "bbox_semantics": "tight_half_open_xyxy",
    "connected_components_connectivity": 8,
    "coverage_denominator": "frame_pixels",
}
RANKING_ALGORITHM_ID = "cad-similarity_then_proposal-score_then_stability_v1"
SELECTION_REASON = "frozen_rank_1_for_target_object"
PRIOR_NO_GO_SHA256 = (
    "ba744c3e62ede30356ad79dc4ccee710d2585a0cfcf1e65b3f50675ba8eebda5"
)

_MASK_METADATA_CACHE: dict[str, dict[str, Any]] = {}
_IMAGE_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_object(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ContractError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_object(value: Any, label: str) -> str:
    if not _is_git_object(value):
        raise ContractError(f"{label} must be a full lowercase 40-hex Git object")
    return value


def _require_finite_unit_score(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ContractError(f"{label} must be finite in [0, 1]")
    return float(value)


def _require_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    lock = _require_sha256(value.get(field), f"{label}.{field}")
    unlocked = dict(value)
    unlocked.pop(field, None)
    if lock != canonical_sha256(unlocked):
        raise ContractError(f"{label} canonical lock mismatch")
    return lock


def _relative_path(value: Any, *, root: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != root:
        raise ContractError(f"{label} must remain under {root}/")
    return path


def _resolved_asset_path(
    asset_root: Path, relative: PurePosixPath, *, label: str
) -> Path:
    resolved_root = asset_root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes the asset root") from exc
    return candidate


def _validate_asset(
    value: Any,
    *,
    root: str,
    label: str,
    asset_root: Path | None,
) -> dict[str, Any]:
    asset = _require_exact_keys(
        value, {"relative_path", "sha256", "bytes"}, label
    )
    relative = _relative_path(asset["relative_path"], root=root, label=label)
    _require_sha256(asset["sha256"], f"{label}.sha256")
    if (
        not isinstance(asset["bytes"], int)
        or isinstance(asset["bytes"], bool)
        or asset["bytes"] <= 0
    ):
        raise ContractError(f"{label}.bytes must be a positive integer")
    if asset_root is not None:
        candidate = _resolved_asset_path(asset_root, relative, label=label)
        if not candidate.is_file():
            raise ContractError(f"Missing asset: {relative.as_posix()}")
        if candidate.stat().st_size != asset["bytes"]:
            raise ContractError(f"Asset byte count mismatch: {relative.as_posix()}")
        if sha256_file(candidate) != asset["sha256"]:
            raise ContractError(f"Asset SHA-256 mismatch: {relative.as_posix()}")
    return dict(asset)


def _decode_image_metadata(
    asset: Mapping[str, Any],
    *,
    root: str,
    label: str,
    asset_root: Path,
    expected_size: tuple[int, int] | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    digest = asset["sha256"]
    metadata = _IMAGE_METADATA_CACHE.get(digest)
    if metadata is None:
        relative = _relative_path(asset["relative_path"], root=root, label=label)
        candidate = _resolved_asset_path(asset_root, relative, label=label)
        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(candidate) as image:
                image.load()
                metadata = {
                    "format": image.format,
                    "mode": image.mode,
                    "size": image.size,
                }
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"{label} is not a decodable image") from exc
        _IMAGE_METADATA_CACHE[digest] = metadata
    if expected_size is not None and metadata["size"] != expected_size:
        raise ContractError(
            f"{label} frame mismatch: expected={expected_size}, actual={metadata['size']}"
        )
    if expected_mode is not None and metadata["mode"] != expected_mode:
        raise ContractError(
            f"{label} mode mismatch: expected={expected_mode}, actual={metadata['mode']}"
        )
    return dict(metadata)


def _count_components_8(mask_bytes: bytes, width: int, height: int) -> int:
    """Count 8-connected foreground runs without a heavyweight vision runtime."""
    parents: list[int] = []

    def new_label() -> int:
        parents.append(len(parents))
        return len(parents) - 1

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left: int, right: int) -> int:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root
        return left_root

    previous: list[tuple[int, int, int]] = []
    foreground = b"\xff"
    background = b"\x00"
    for row in range(height):
        row_start = row * width
        row_end = row_start + width
        current: list[tuple[int, int, int]] = []
        offset = row_start
        while offset < row_end:
            start_absolute = mask_bytes.find(foreground, offset, row_end)
            if start_absolute < 0:
                break
            end_absolute = mask_bytes.find(background, start_absolute, row_end)
            if end_absolute < 0:
                end_absolute = row_end
            start = start_absolute - row_start
            end = end_absolute - row_start
            overlaps = [
                label
                for previous_start, previous_end, label in previous
                if previous_start <= end and previous_end >= start
            ]
            component = new_label() if not overlaps else find(overlaps[0])
            for overlap in overlaps[1:]:
                component = union(component, overlap)
            current.append((start, end, component))
            offset = end_absolute
        previous = current
    return len({find(label) for label in range(len(parents))})


def _decode_mask_metadata(
    asset: Mapping[str, Any], *, label: str, asset_root: Path
) -> dict[str, Any]:
    relative = _relative_path(asset["relative_path"], root="masks", label=label)
    if relative.suffix.lower() != ".png":
        raise ContractError(f"{label} must be a PNG mask")
    digest = asset["sha256"]
    metadata = _MASK_METADATA_CACHE.get(digest)
    if metadata is not None:
        return dict(metadata)
    candidate = _resolved_asset_path(asset_root, relative, label=label)
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(candidate) as image:
            image.load()
            if image.format != MASK_CONTRACT["format"]:
                raise ContractError(f"{label} must decode as PNG")
            if image.mode != MASK_CONTRACT["mode"]:
                raise ContractError(f"{label} must be a single-channel L mask")
            expected_size = (FRAME_SIZE["width"], FRAME_SIZE["height"])
            if image.size != expected_size:
                raise ContractError(
                    f"{label} frame mismatch: expected={expected_size}, actual={image.size}"
                )
            histogram = image.histogram()
            invalid_pixels = sum(
                count
                for value, count in enumerate(histogram)
                if value not in {
                    MASK_CONTRACT["background_value"],
                    MASK_CONTRACT["foreground_value"],
                }
            )
            if invalid_pixels:
                raise ContractError(f"{label} must be strictly binary 0/255")
            mask_pixels = histogram[MASK_CONTRACT["foreground_value"]]
            if mask_pixels <= 0:
                raise ContractError(f"{label} must contain non-empty foreground")
            bbox = image.getbbox()
            if bbox is None:
                raise ContractError(f"{label} has no tight foreground bounds")
            mask_bytes = image.tobytes()
    except ContractError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise ContractError(f"{label} is not a decodable PNG mask") from exc
    metadata = {
        "bbox_xyxy": list(bbox),
        "mask_pixels": mask_pixels,
        "coverage": mask_pixels / (FRAME_SIZE["width"] * FRAME_SIZE["height"]),
        "connected_components": _count_components_8(
            mask_bytes, FRAME_SIZE["width"], FRAME_SIZE["height"]
        ),
    }
    _MASK_METADATA_CACHE[digest] = metadata
    return dict(metadata)


def _validate_mask_asset(
    value: Any, *, label: str, asset_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = _require_exact_keys(
        value,
        {
            "relative_path",
            "sha256",
            "bytes",
            "mask_pixels",
            "coverage",
            "connected_components",
        },
        label,
    )
    asset = _validate_asset(
        {name: mask[name] for name in ("relative_path", "sha256", "bytes")},
        root="masks",
        label=label,
        asset_root=asset_root,
    )
    decoded = _decode_mask_metadata(asset, label=label, asset_root=asset_root)
    if (
        not isinstance(mask["mask_pixels"], int)
        or isinstance(mask["mask_pixels"], bool)
        or mask["mask_pixels"] <= 0
        or mask["mask_pixels"] != decoded["mask_pixels"]
    ):
        raise ContractError(f"{label}.mask_pixels does not match decoded mask")
    coverage = mask["coverage"]
    if (
        not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or not math.isfinite(coverage)
        or not math.isclose(coverage, decoded["coverage"], rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ContractError(f"{label}.coverage does not match decoded mask")
    if (
        not isinstance(mask["connected_components"], int)
        or isinstance(mask["connected_components"], bool)
        or mask["connected_components"] <= 0
        or mask["connected_components"] != decoded["connected_components"]
    ):
        raise ContractError(
            f"{label}.connected_components does not match decoded mask"
        )
    return dict(mask), decoded


def _sample_key(value: Any, label: str) -> tuple[int, int, int]:
    key = _require_exact_keys(value, {"scene_id", "image_id", "object_id"}, label)
    result: list[int] = []
    for name in ("scene_id", "image_id", "object_id"):
        item = key[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ContractError(f"{label}.{name} must be a non-negative integer")
        result.append(item)
    return result[0], result[1], result[2]


def _expected_samples(protocol: Mapping[str, Any]) -> dict[str, tuple[int, int, int]]:
    expected: dict[str, tuple[int, int, int]] = {}
    for index, item in enumerate(protocol["input_lock"]["samples"]):
        sample = _require_exact_keys(
            item, {"item_id", "scene_id", "image_id", "object_id"}, f"samples[{index}]"
        )
        item_id = sample["item_id"]
        if not isinstance(item_id, str) or not item_id:
            raise ContractError(f"samples[{index}].item_id must be non-empty")
        key = _sample_key(
            {name: sample[name] for name in ("scene_id", "image_id", "object_id")},
            f"samples[{index}]",
        )
        if item_id in expected or key in expected.values():
            raise ContractError("Protocol sample identities must be unique")
        expected[item_id] = key
    return expected


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable protocol and its prior-NO-GO/data boundaries."""
    _require_exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "namespace",
            "state",
            "role",
            "auto_deploy",
            "accuracy_claim_permitted",
            "base_commit",
            "prior_no_go",
            "input_lock",
            "runtime_requirements",
            "proposal_policy",
            "mask_contract",
            "content_gate",
            "boundary",
            "remote_audit",
        },
        "protocol",
    )
    if protocol["schema_version"] != PROTOCOL_SCHEMA or protocol["protocol_id"] != PROTOCOL_ID:
        raise ContractError("Instance-proposal protocol identity mismatch")
    if protocol["namespace"] != "pose_accuracy_recovery_prep.instance_proposal_v1":
        raise ContractError("Instance-proposal namespace changed")
    if (
        protocol["state"] != "PREFROZEN_RUNTIME_NOT_READY_AT_AUDIT"
        or protocol["role"] != "DEVELOPMENT_ONLY"
        or protocol["auto_deploy"] is not False
        or protocol["accuracy_claim_permitted"] is not False
    ):
        raise ContractError("Protocol must remain inert DEVELOPMENT_ONLY preparation")
    _require_git_object(protocol["base_commit"], "protocol.base_commit")

    prior = _require_exact_keys(
        protocol["prior_no_go"],
        {"receipt_sha256", "immutable", "rerun_permitted", "result_tuning_permitted"},
        "protocol.prior_no_go",
    )
    if (
        prior["receipt_sha256"] != PRIOR_NO_GO_SHA256
        or prior["immutable"] is not True
        or prior["rerun_permitted"] is not False
        or prior["result_tuning_permitted"] is not False
    ):
        raise ContractError("Prior SAM content NO_GO boundary changed")

    input_lock = _require_exact_keys(
        protocol["input_lock"],
        {
            "workload_sha256",
            "sample_count",
            "scene_ids",
            "object_ids",
            "samples",
            "cad_assets",
            "frame_size",
            "allowed_input_roles",
            "forbidden_proposal_cues",
        },
        "protocol.input_lock",
    )
    _require_sha256(input_lock["workload_sha256"], "input_lock.workload_sha256")
    expected = _expected_samples(protocol)
    if len(expected) != 10 or input_lock["sample_count"] != 10:
        raise ContractError("Protocol must freeze exactly 10 unique samples")
    scenes = sorted({key[0] for key in expected.values()})
    objects = sorted({key[2] for key in expected.values()})
    if scenes != [0, 3, 9, 12, 15] or input_lock["scene_ids"] != scenes:
        raise ContractError("Protocol must freeze exactly five declared scenes")
    if objects != [1, 2, 4, 5, 6] or input_lock["object_ids"] != objects:
        raise ContractError("Protocol must freeze exactly five declared objects")
    if input_lock["allowed_input_roles"] != ALLOWED_INPUT_ROLES:
        raise ContractError("Independent proposal input roles changed")
    if input_lock["frame_size"] != FRAME_SIZE:
        raise ContractError("Protocol frame size must remain 1440x1080")
    if input_lock["forbidden_proposal_cues"] != [
        "depth_component_mask",
        "depth_component_bbox",
        "bbox_mask",
        "legacy_predicted_mask",
        "gt_or_evaluator_asset",
    ]:
        raise ContractError("Forbidden proposal cue list changed")

    cad_assets = input_lock["cad_assets"]
    if not isinstance(cad_assets, list) or len(cad_assets) != 5:
        raise ContractError("Protocol requires five CAD asset locks")
    seen_cad: set[int] = set()
    for index, asset in enumerate(cad_assets):
        item = _require_exact_keys(
            asset, {"object_id", "sha256", "bytes"}, f"cad_assets[{index}]"
        )
        if item["object_id"] in seen_cad or item["object_id"] not in objects:
            raise ContractError("CAD object IDs must exactly match the five target objects")
        seen_cad.add(item["object_id"])
        _require_sha256(item["sha256"], f"cad_assets[{index}].sha256")
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ContractError(f"cad_assets[{index}].bytes must be positive")
    if seen_cad != set(objects):
        raise ContractError("CAD locks do not cover every target object")

    runtime = _require_exact_keys(
        protocol["runtime_requirements"],
        {
            "runtime_lock_schema",
            "backend_family",
            "execution_ready_at_audit",
            "source_commit_required",
            "checkpoint_sha256_required",
            "renderer_and_descriptor_locks_required",
        },
        "protocol.runtime_requirements",
    )
    if runtime != {
        "runtime_lock_schema": RUNTIME_LOCK_SCHEMA,
        "backend_family": "CAD_CONDITIONED_INSTANCE_PROPOSAL",
        "execution_ready_at_audit": False,
        "source_commit_required": True,
        "checkpoint_sha256_required": True,
        "renderer_and_descriptor_locks_required": True,
    }:
        raise ContractError("Runtime readiness requirements changed")

    policy = _require_exact_keys(
        protocol["proposal_policy"],
        {
            "proposal_source",
            "prompt_source",
            "ranking_algorithm_id",
            "ranking_order",
            "selection",
            "score_tuning_permitted",
        },
        "protocol.proposal_policy",
    )
    if (
        policy["proposal_source"] != "independent_rgb_instance_proposals"
        or policy["prompt_source"] is not None
        or policy["ranking_algorithm_id"] != RANKING_ALGORITHM_ID
        or policy["ranking_order"]
        != [
            "cad_similarity_desc",
            "proposal_score_desc",
            "mask_stability_desc",
            "proposal_index_asc",
        ]
        or policy["selection"] != SELECTION_REASON
        or policy["score_tuning_permitted"] is not False
    ):
        raise ContractError("Frozen proposal generation/ranking policy changed")

    if protocol["mask_contract"] != MASK_CONTRACT:
        raise ContractError("Frozen binary-mask geometry contract changed")

    gate = _require_exact_keys(
        protocol["content_gate"],
        {
            "required_pass_count",
            "human_visual_review_required",
            "machine_score_substitution_permitted",
            "single_target_instance_required",
            "tray_border_permitted",
            "multiple_objects_permitted",
            "visualization_roles",
        },
        "protocol.content_gate",
    )
    if gate != {
        "required_pass_count": 10,
        "human_visual_review_required": True,
        "machine_score_substitution_permitted": False,
        "single_target_instance_required": True,
        "tray_border_permitted": False,
        "multiple_objects_permitted": False,
        "visualization_roles": VISUALIZATION_ROLES,
    }:
        raise ContractError("Content gate changed")

    boundary = _require_exact_keys(
        protocol["boundary"],
        {
            "label_access_count",
            "gt_path_open_count",
            "evaluator_path_open_count",
            "sealed_split_access_count",
            "official_scorer_run",
            "server_shutdown_permitted",
            "auto_trigger_gpu_c",
        },
        "protocol.boundary",
    )
    if boundary != {
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "sealed_split_access_count": 0,
        "official_scorer_run": False,
        "server_shutdown_permitted": False,
        "auto_trigger_gpu_c": False,
    }:
        raise ContractError("Development/access/lifecycle boundary changed")

    audit = _require_exact_keys(
        protocol["remote_audit"],
        {
            "observed_at",
            "host_role",
            "gpu",
            "status",
            "available_runtime",
            "missing_runtime",
            "safe_cad_count",
        },
        "protocol.remote_audit",
    )
    if (
        audit["host_role"] != "GPU-A"
        or audit["status"] != "RUNTIME_NOT_READY"
        or audit["safe_cad_count"] != 5
        or "segment-anything-vit-b-excluded-prior-no-go" not in audit["available_runtime"]
    ):
        raise ContractError("Remote audit conclusion changed")

    protocol_lock = _require_lock(protocol, "protocol_lock_sha256", "protocol")
    return {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol_lock,
        "sample_count": 10,
        "scene_count": 5,
        "object_count": 5,
        "runtime_ready": False,
        "auto_deploy": False,
        "prior_no_go_immutable": True,
        "accuracy_claim_permitted": False,
    }


def load_and_validate_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("Protocol must be a JSON object")
    return value, validate_protocol(value)


def _validate_runtime_data(
    data_value: Any, protocol: Mapping[str, Any], *, input_root: Path
) -> dict[str, Any]:
    data = _require_exact_keys(
        data_value,
        {
            "schema_version",
            "input_manifest_lock_sha256",
            "source_manifest",
            "workload_manifest",
            "frame_size",
            "sample_count",
            "scene_ids",
            "object_ids",
            "items",
        },
        "runtime_lock.data",
    )
    if data["schema_version"] != INPUT_MANIFEST_SCHEMA:
        raise ContractError("Runtime input-manifest schema changed")
    expected_input = protocol["input_lock"]
    if data["frame_size"] != FRAME_SIZE:
        raise ContractError("Runtime input frame differs from 1440x1080 protocol")
    for name in ("sample_count", "scene_ids", "object_ids"):
        if data[name] != expected_input[name]:
            raise ContractError(f"Runtime data lock differs from protocol: {name}")

    source_manifest = _validate_asset(
        data["source_manifest"],
        root="inputs",
        label="runtime_lock.data.source_manifest",
        asset_root=input_root,
    )
    workload_manifest = _validate_asset(
        data["workload_manifest"],
        root="inputs",
        label="runtime_lock.data.workload_manifest",
        asset_root=input_root,
    )
    if workload_manifest["sha256"] != expected_input["workload_sha256"]:
        raise ContractError("Runtime workload manifest SHA differs from protocol")
    for label, asset in (
        ("source_manifest", source_manifest),
        ("workload_manifest", workload_manifest),
    ):
        relative = _relative_path(
            asset["relative_path"], root="inputs", label=f"runtime_lock.data.{label}"
        )
        candidate = _resolved_asset_path(
            input_root, relative, label=f"runtime_lock.data.{label}"
        )
        try:
            parsed = read_json(candidate)
        except ContractError as exc:
            raise ContractError(f"Runtime {label} must be valid JSON") from exc
        if not isinstance(parsed, (dict, list)):
            raise ContractError(f"Runtime {label} must contain a JSON object or list")

    expected_samples = _expected_samples(protocol)
    items = data["items"]
    if not isinstance(items, list) or len(items) != len(expected_samples):
        raise ContractError("Runtime input manifest must cover exact 10 protocol samples")
    expected_cad = {
        asset["object_id"]: {
            "bytes": asset["bytes"],
            "sha256": asset["sha256"],
        }
        for asset in expected_input["cad_assets"]
    }
    seen_items: set[str] = set()
    unique_paths: dict[str, set[str]] = {
        "rgb": set(),
        "camera": set(),
    }
    cad_by_object: dict[int, tuple[str, int, str]] = {}
    descriptors_by_object: dict[int, tuple[str, int, str]] = {}
    hashes_by_item: dict[str, dict[str, str]] = {}
    assets_verified = 2
    for index, item_value in enumerate(items):
        item = _require_exact_keys(
            item_value,
            {"item_id", "sample_key", "inputs"},
            f"runtime_lock.data.items[{index}]",
        )
        item_id = item["item_id"]
        if item_id not in expected_samples or item_id in seen_items:
            raise ContractError(f"Unexpected or duplicate runtime input item: {item_id}")
        seen_items.add(item_id)
        sample_key = _sample_key(
            item["sample_key"], f"runtime_lock.data.items[{index}].sample_key"
        )
        if sample_key != expected_samples[item_id]:
            raise ContractError(f"Runtime input sample key mismatch: {item_id}")
        object_id = sample_key[2]
        inputs = _require_exact_keys(
            item["inputs"], set(ALLOWED_INPUT_ROLES), f"runtime inputs for {item_id}"
        )
        item_hashes: dict[str, str] = {}
        for role in ("rgb", "camera"):
            asset = _validate_asset(
                inputs[role],
                root="inputs",
                label=f"runtime inputs for {item_id}.{role}",
                asset_root=input_root,
            )
            if asset["relative_path"] in unique_paths[role]:
                raise ContractError(f"Runtime {role} path is reused across samples")
            unique_paths[role].add(asset["relative_path"])
            if role == "rgb":
                _decode_image_metadata(
                    asset,
                    root="inputs",
                    label=f"runtime inputs for {item_id}.rgb",
                    asset_root=input_root,
                    expected_size=(FRAME_SIZE["width"], FRAME_SIZE["height"]),
                    expected_mode="RGB",
                )
            else:
                relative = _relative_path(
                    asset["relative_path"],
                    root="inputs",
                    label=f"runtime inputs for {item_id}.camera",
                )
                if relative.suffix.lower() != ".json":
                    raise ContractError(f"Runtime camera asset must be JSON: {item_id}")
                candidate = _resolved_asset_path(
                    input_root, relative, label=f"runtime inputs for {item_id}.camera"
                )
                if not isinstance(read_json(candidate), dict):
                    raise ContractError(f"Runtime camera JSON must be an object: {item_id}")
            item_hashes[role] = asset["sha256"]
            assets_verified += 1

        for role in ("cad", "cad_render_descriptors"):
            described = _require_exact_keys(
                inputs[role],
                {"object_id", "relative_path", "sha256", "bytes"},
                f"runtime inputs for {item_id}.{role}",
            )
            if (
                not isinstance(described["object_id"], int)
                or isinstance(described["object_id"], bool)
                or described["object_id"] != object_id
            ):
                raise ContractError(f"Runtime {role} target object mismatch: {item_id}")
            asset = _validate_asset(
                {
                    name: described[name]
                    for name in ("relative_path", "sha256", "bytes")
                },
                root="inputs",
                label=f"runtime inputs for {item_id}.{role}",
                asset_root=input_root,
            )
            identity = (asset["relative_path"], asset["bytes"], asset["sha256"])
            per_object = cad_by_object if role == "cad" else descriptors_by_object
            prior_identity = per_object.setdefault(object_id, identity)
            if prior_identity != identity:
                raise ContractError(
                    f"Runtime {role} asset differs within object {object_id}"
                )
            if role == "cad" and {
                "bytes": asset["bytes"],
                "sha256": asset["sha256"],
            } != expected_cad[object_id]:
                raise ContractError(f"Runtime CAD lock differs from protocol: {item_id}")
            item_hashes[role] = asset["sha256"]
            assets_verified += 1
        hashes_by_item[item_id] = item_hashes
    if seen_items != set(expected_samples):
        raise ContractError("Runtime input manifest sample coverage mismatch")
    if set(cad_by_object) != set(expected_cad) or set(descriptors_by_object) != set(
        expected_cad
    ):
        raise ContractError("Runtime CAD/descriptor assets do not cover all objects")
    input_lock = _require_lock(
        data, "input_manifest_lock_sha256", "runtime_lock.data"
    )
    return {
        "input_manifest_lock_sha256": input_lock,
        "source_manifest_sha256": source_manifest["sha256"],
        "workload_manifest_sha256": workload_manifest["sha256"],
        "hashes_by_item": hashes_by_item,
        "verified_input_assets": assets_verified,
    }


def validate_runtime_lock(
    runtime_lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    input_root: Path,
) -> dict[str, Any]:
    """Require complete source/model/render/CAD locks before any GPU execution."""
    protocol_validation = validate_protocol(protocol)
    _require_exact_keys(
        runtime_lock,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "runtime_lock_sha256",
            "execution_ready",
            "backend",
            "renderer",
            "data",
            "boundary",
        },
        "runtime_lock",
    )
    if (
        runtime_lock["schema_version"] != RUNTIME_LOCK_SCHEMA
        or runtime_lock["protocol_id"] != PROTOCOL_ID
        or runtime_lock["protocol_lock_sha256"]
        != protocol_validation["protocol_lock_sha256"]
        or runtime_lock["execution_ready"] is not True
    ):
        raise ContractError("Runtime lock is not execution-ready for this protocol")

    backend = _require_exact_keys(
        runtime_lock["backend"],
        {
            "family",
            "implementation_name",
            "source_repository",
            "source_commit",
            "source_tree",
            "source_archive_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "model_config_sha256",
        },
        "runtime_lock.backend",
    )
    if backend["family"] != "CAD_CONDITIONED_INSTANCE_PROPOSAL":
        raise ContractError("Backend must be CAD-conditioned instance proposal")
    if (
        not isinstance(backend["implementation_name"], str)
        or not backend["implementation_name"].strip()
        or not isinstance(backend["source_repository"], str)
        or not backend["source_repository"].startswith("https://")
    ):
        raise ContractError("Backend implementation/repository is incomplete")
    _require_git_object(backend["source_commit"], "backend.source_commit")
    _require_git_object(backend["source_tree"], "backend.source_tree")
    for name in ("source_archive_sha256", "checkpoint_sha256", "model_config_sha256"):
        _require_sha256(backend[name], f"backend.{name}")
    if not isinstance(backend["checkpoint_bytes"], int) or backend["checkpoint_bytes"] <= 0:
        raise ContractError("backend.checkpoint_bytes must be positive")

    renderer = _require_exact_keys(
        runtime_lock["renderer"],
        {
            "implementation_name",
            "source_commit",
            "source_tree",
            "source_archive_sha256",
            "renderer_config_sha256",
            "cad_render_manifest_sha256",
            "descriptor_model_sha256",
            "view_sampling_id",
            "render_view_count_per_object",
        },
        "runtime_lock.renderer",
    )
    if not isinstance(renderer["implementation_name"], str) or not renderer[
        "implementation_name"
    ].strip():
        raise ContractError("Renderer implementation is incomplete")
    _require_git_object(renderer["source_commit"], "renderer.source_commit")
    _require_git_object(renderer["source_tree"], "renderer.source_tree")
    for name in (
        "source_archive_sha256",
        "renderer_config_sha256",
        "cad_render_manifest_sha256",
        "descriptor_model_sha256",
    ):
        _require_sha256(renderer[name], f"renderer.{name}")
    if (
        not isinstance(renderer["view_sampling_id"], str)
        or not renderer["view_sampling_id"].strip()
        or not isinstance(renderer["render_view_count_per_object"], int)
        or renderer["render_view_count_per_object"] <= 0
    ):
        raise ContractError("Renderer view sampling lock is incomplete")

    data_validation = _validate_runtime_data(
        runtime_lock["data"], protocol, input_root=input_root
    )

    boundary = _require_exact_keys(
        runtime_lock["boundary"],
        {
            "label_access_count",
            "gt_path_open_count",
            "evaluator_path_open_count",
            "sealed_split_access_count",
            "official_scorer_run",
            "legacy_depth_prompt_read_count",
            "legacy_mask_read_count",
        },
        "runtime_lock.boundary",
    )
    if boundary != {
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "sealed_split_access_count": 0,
        "official_scorer_run": False,
        "legacy_depth_prompt_read_count": 0,
        "legacy_mask_read_count": 0,
    }:
        raise ContractError("Runtime boundary must remain zero-access and independent")
    runtime_sha = _require_lock(runtime_lock, "runtime_lock_sha256", "runtime_lock")
    return {
        "status": "valid",
        "execution_ready": True,
        "runtime_lock_sha256": runtime_sha,
        "input_manifest_lock_sha256": data_validation[
            "input_manifest_lock_sha256"
        ],
        "backend_family": backend["family"],
        "cad_count": 5,
        "verified_input_assets": data_validation["verified_input_assets"],
        "label_access_count": 0,
    }


def _validate_boundary(value: Any, label: str) -> None:
    boundary = _require_exact_keys(
        value,
        {
            "label_access_count",
            "gt_path_open_count",
            "evaluator_path_open_count",
            "sealed_split_access_count",
            "official_scorer_run",
            "legacy_depth_prompt_read_count",
            "legacy_mask_read_count",
        },
        label,
    )
    if boundary != {
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "sealed_split_access_count": 0,
        "official_scorer_run": False,
        "legacy_depth_prompt_read_count": 0,
        "legacy_mask_read_count": 0,
    }:
        raise ContractError(f"{label} must remain zero-access and independent")


def validate_proposal_bundle(
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    *,
    input_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    """Validate exact proposal coverage, ranking, provenance, and real assets."""
    protocol_validation = validate_protocol(protocol)
    runtime_validation = validate_runtime_lock(
        runtime_lock, protocol, input_root=input_root
    )
    runtime_data = _validate_runtime_data(
        runtime_lock["data"], protocol, input_root=input_root
    )
    _require_exact_keys(
        bundle,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "runtime_lock_sha256",
            "proposal_bundle_lock_sha256",
            "role",
            "input_roles",
            "ranking_algorithm_id",
            "boundary",
            "items",
        },
        "proposal_bundle",
    )
    if (
        bundle["schema_version"] != PROPOSAL_SCHEMA
        or bundle["protocol_id"] != PROTOCOL_ID
        or bundle["protocol_lock_sha256"]
        != protocol_validation["protocol_lock_sha256"]
        or bundle["runtime_lock_sha256"] != runtime_validation["runtime_lock_sha256"]
        or bundle["role"] != "DEVELOPMENT_ONLY_INDEPENDENT_INSTANCE_PROPOSAL"
        or bundle["input_roles"] != ALLOWED_INPUT_ROLES
        or bundle["ranking_algorithm_id"] != RANKING_ALGORITHM_ID
    ):
        raise ContractError("Proposal bundle identity/input/ranking contract mismatch")
    _validate_boundary(bundle["boundary"], "proposal_bundle.boundary")

    expected = _expected_samples(protocol)
    items = bundle["items"]
    if not isinstance(items, list) or len(items) != len(expected):
        raise ContractError("Proposal bundle must cover exact 10 protocol samples")
    seen: set[str] = set()
    proposal_count = 0
    asset_count = 0
    for item_index, item_value in enumerate(items):
        item = _require_exact_keys(
            item_value,
            {
                "item_id",
                "sample_key",
                "input_hashes",
                "proposals",
                "selected_proposal_index",
                "selected_mask",
                "selection_reason",
                "visualizations",
            },
            f"items[{item_index}]",
        )
        item_id = item["item_id"]
        if item_id not in expected or item_id in seen:
            raise ContractError(f"Unexpected or duplicate proposal item: {item_id}")
        seen.add(item_id)
        if _sample_key(item["sample_key"], f"items[{item_index}].sample_key") != expected[
            item_id
        ]:
            raise ContractError(f"Proposal sample key mismatch: {item_id}")
        input_hashes = _require_exact_keys(
            item["input_hashes"], set(ALLOWED_INPUT_ROLES), f"items[{item_index}].input_hashes"
        )
        for role, digest in input_hashes.items():
            _require_sha256(digest, f"items[{item_index}].input_hashes.{role}")
        if dict(input_hashes) != runtime_data["hashes_by_item"][item_id]:
            raise ContractError(
                f"{item_id} input hashes differ from the runtime input manifest"
            )
        object_id = expected[item_id][2]

        proposals = item["proposals"]
        if not isinstance(proposals, list) or not proposals:
            raise ContractError(f"{item_id} requires at least one independent proposal")
        parsed: list[tuple[tuple[float, float, float, int], Mapping[str, Any]]] = []
        indices: set[int] = set()
        for proposal_index, proposal_value in enumerate(proposals):
            proposal = _require_exact_keys(
                proposal_value,
                {
                    "proposal_index",
                    "rank",
                    "cad_object_id",
                    "cad_similarity",
                    "proposal_score",
                    "mask_stability",
                    "bbox_xyxy",
                    "mask",
                },
                f"items[{item_index}].proposals[{proposal_index}]",
            )
            index_value = proposal["proposal_index"]
            if (
                not isinstance(index_value, int)
                or isinstance(index_value, bool)
                or index_value < 0
                or index_value in indices
            ):
                raise ContractError(f"{item_id} proposal indices must be unique non-negative ints")
            indices.add(index_value)
            if proposal["cad_object_id"] != object_id:
                raise ContractError(f"{item_id} proposal CAD object mismatch")
            cad_similarity = _require_finite_unit_score(
                proposal["cad_similarity"], f"{item_id}.cad_similarity"
            )
            proposal_score = _require_finite_unit_score(
                proposal["proposal_score"], f"{item_id}.proposal_score"
            )
            stability = _require_finite_unit_score(
                proposal["mask_stability"], f"{item_id}.mask_stability"
            )
            bbox = proposal["bbox_xyxy"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ContractError(f"{item_id} proposal bbox must be valid half-open xyxy")
            mask, decoded_mask = _validate_mask_asset(
                proposal["mask"],
                label=f"items[{item_index}].proposals[{proposal_index}].mask",
                asset_root=asset_root,
            )
            if bbox != decoded_mask["bbox_xyxy"]:
                raise ContractError(
                    f"{item_id} proposal bbox is not the tight half-open mask bound"
                )
            asset_count += 1
            if proposal["mask"] != mask:
                raise ContractError(f"{item_id} proposal mask normalization changed")
            parsed.append(((-cad_similarity, -proposal_score, -stability, index_value), proposal))
        sorted_parsed = sorted(parsed, key=lambda value: value[0])
        if [proposal for _, proposal in sorted_parsed] != proposals:
            raise ContractError(f"{item_id} proposals do not follow frozen ranking")
        if [proposal["rank"] for proposal in proposals] != list(range(1, len(proposals) + 1)):
            raise ContractError(f"{item_id} proposal ranks are not contiguous from one")
        winner = proposals[0]
        if (
            item["selected_proposal_index"] != winner["proposal_index"]
            or item["selected_mask"] != winner["mask"]
            or item["selection_reason"] != SELECTION_REASON
        ):
            raise ContractError(f"{item_id} selection does not bind frozen rank one")
        visualizations = _require_exact_keys(
            item["visualizations"], set(VISUALIZATION_ROLES), f"items[{item_index}].visualizations"
        )
        for role in VISUALIZATION_ROLES:
            asset = _validate_asset(
                visualizations[role],
                root="visualizations",
                label=f"items[{item_index}].visualizations.{role}",
                asset_root=asset_root,
            )
            _decode_image_metadata(
                asset,
                root="visualizations",
                label=f"items[{item_index}].visualizations.{role}",
                asset_root=asset_root,
                expected_size=(FRAME_SIZE["width"], FRAME_SIZE["height"]),
            )
            asset_count += 1
        proposal_count += len(proposals)

    if seen != set(expected):
        raise ContractError("Proposal bundle does not cover the exact sample set")
    bundle_lock = _require_lock(bundle, "proposal_bundle_lock_sha256", "proposal_bundle")
    return {
        "status": "valid",
        "proposal_bundle_lock_sha256": bundle_lock,
        "item_count": 10,
        "scene_count": 5,
        "object_count": 5,
        "proposal_count": proposal_count,
        "asset_count": asset_count,
        "verified_assets": True,
        "input_manifest_lock_sha256": runtime_data[
            "input_manifest_lock_sha256"
        ],
        "legacy_depth_prompt_read_count": 0,
        "label_access_count": 0,
        "content_gate_status": "NOT_REVIEWED",
    }


def validate_content_gate(
    review: Mapping[str, Any],
    protocol: Mapping[str, Any],
    proposal_bundle: Mapping[str, Any],
    *,
    input_root: Path,
    proposal_root: Path,
    review_root: Path,
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the authoritative per-item human visual gate and decision."""
    proposal_validation = validate_proposal_bundle(
        proposal_bundle,
        protocol,
        runtime_lock,
        input_root=input_root,
        asset_root=proposal_root,
    )
    _require_exact_keys(
        review,
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "proposal_bundle_lock_sha256",
            "content_gate_lock_sha256",
            "review_method",
            "machine_score_substitution_permitted",
            "required_pass_count",
            "decision",
            "formal_five_variant_handoff_permitted",
            "summary",
            "items",
            "contact_sheet",
            "boundary",
        },
        "content_gate",
    )
    protocol_validation = validate_protocol(protocol)
    if (
        review["schema_version"] != CONTENT_GATE_SCHEMA
        or review["protocol_id"] != PROTOCOL_ID
        or review["protocol_lock_sha256"]
        != protocol_validation["protocol_lock_sha256"]
        or review["proposal_bundle_lock_sha256"]
        != proposal_validation["proposal_bundle_lock_sha256"]
        or review["review_method"] != "HUMAN_VISUAL_INSPECTION"
        or review["machine_score_substitution_permitted"] is not False
        or review["required_pass_count"] != 10
    ):
        raise ContractError("Content-gate identity/authority contract mismatch")

    _validate_boundary(review["boundary"], "content_gate.boundary")
    expected = _expected_samples(protocol)
    items = review["items"]
    if not isinstance(items, list) or len(items) != 10:
        raise ContractError("Content gate requires exactly 10 item verdicts")
    seen: set[str] = set()
    passed = 0
    failed = 0
    for index, value in enumerate(items):
        item = _require_exact_keys(
            value,
            {
                "item_id",
                "verdict",
                "single_target_instance",
                "includes_tray",
                "includes_border",
                "includes_multiple_objects",
                "reason",
                "panel",
            },
            f"content_gate.items[{index}]",
        )
        item_id = item["item_id"]
        if item_id not in expected or item_id in seen:
            raise ContractError(f"Unexpected or duplicate content verdict: {item_id}")
        seen.add(item_id)
        if item["verdict"] not in {"PASS", "FAIL"}:
            raise ContractError(f"Invalid content verdict: {item_id}")
        if any(
            not isinstance(item[name], bool)
            for name in (
                "single_target_instance",
                "includes_tray",
                "includes_border",
                "includes_multiple_objects",
            )
        ):
            raise ContractError(f"Content observations must be explicit booleans: {item_id}")
        observed_pass = (
            item["single_target_instance"]
            and not item["includes_tray"]
            and not item["includes_border"]
            and not item["includes_multiple_objects"]
        )
        if (item["verdict"] == "PASS") != observed_pass:
            raise ContractError(f"Content verdict contradicts observations: {item_id}")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ContractError(f"Content verdict requires a human reason: {item_id}")
        panel = _validate_asset(
            item["panel"],
            root="panels",
            label=f"content_gate.items[{index}].panel",
            asset_root=review_root,
        )
        _decode_image_metadata(
            panel,
            root="panels",
            label=f"content_gate.items[{index}].panel",
            asset_root=review_root,
        )
        if observed_pass:
            passed += 1
        else:
            failed += 1
    if seen != set(expected):
        raise ContractError("Content gate does not cover the exact protocol sample set")
    contact_sheet = _validate_asset(
        review["contact_sheet"],
        root="contact-sheet",
        label="content_gate.contact_sheet",
        asset_root=review_root,
    )
    _decode_image_metadata(
        contact_sheet,
        root="contact-sheet",
        label="content_gate.contact_sheet",
        asset_root=review_root,
    )
    summary = _require_exact_keys(
        review["summary"], {"pass_count", "fail_count"}, "content_gate.summary"
    )
    if summary != {"pass_count": passed, "fail_count": failed}:
        raise ContractError("Content-gate summary does not match item verdicts")
    expected_decision = "PASS" if passed == 10 else "NO_GO"
    if review["decision"] != expected_decision:
        raise ContractError("Content-gate decision does not match the 10/10 rule")
    if review["formal_five_variant_handoff_permitted"] is not (passed == 10):
        raise ContractError("A-to-C handoff permission does not match content gate")
    gate_lock = _require_lock(review, "content_gate_lock_sha256", "content_gate")
    return {
        "status": "valid",
        "decision": expected_decision,
        "content_gate_lock_sha256": gate_lock,
        "pass_count": passed,
        "fail_count": failed,
        "required_pass_count": 10,
        "formal_five_variant_handoff_permitted": passed == 10,
        "human_visual_reviewed": True,
        "machine_score_substitution_permitted": False,
        "label_access_count": 0,
    }


def load_mapping(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value
