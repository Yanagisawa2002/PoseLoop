"""Strict label-free manifest validator and mask-provider contract."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .common import (
    MANIFEST_SCHEMA,
    PREP_PROTOCOL_ID,
    RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2,
    PrepError,
    canonical_sha256,
    is_sha256,
    read_json,
    sha256_file,
)


ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
A_SOURCE_PROTOCOL_SHA256 = (
    "20380305a7fbd92b2c563baea7df1e4f82c39605379543cd8e9443925eae0b8b"
)
ASSET_KEYS = {"role", "relative_path", "sha256", "bytes"}
MANIFEST_KEYS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "source_contract",
    "mask_variants",
    "coverage",
    "items",
    "boundary",
}
ITEM_KEYS = {
    "item_id",
    "sample_key",
    "mask_variant_id",
    "frame_size",
    "inputs",
    "camera_intrinsics",
    "depth_scale",
    "mask_provenance",
}
INPUT_ROLES = {
    "rgb": "public_rgb",
    "depth": "public_depth",
    "mask": "input_mask",
    "camera": "public_camera",
    "cad": "public_cad",
}
MASK_INPUT_ROLES = {
    "sanity-input",
    "oracle-input",
    "predicted-segmentation",
    "predicted-detection",
}
MASK_PLUGIN_IDS = {
    "manifest-mask-file-v1",
    "coco-rle-inline-v1",
}
FORBIDDEN_PATH_TOKENS = {
    "scene_gt",
    "scene_gt_info",
    "mask_gt",
    "mask_visib",
    "gt_pose",
    "ground_truth",
    "evaluator",
    "evaluation",
    "official_score",
    "sealed",
    "oracle",
}
FORBIDDEN_KEYS = {
    "correct",
    "evaluator_pose_error",
    "gt_instance_index",
    "gt_model_to_camera_pose_m",
    "joint_success",
    "oracle_association",
    "px_count_visib",
    "raw_rotation_error_degrees",
    "translation_error_mm",
    "visib_fract",
    "visible_fraction",
    "visible_mask_pixel_count",
}
BOUNDARY = {
    "label_access_count": 0,
    "official_scorer_run": False,
    "contains_gt_pose": False,
    "contains_gt_visible_mask": False,
    "contains_evaluator_output": False,
    "contains_sealed_data": False,
    "uses_oracle_association": False,
    "gpu_c_reads_source_labels": False,
}
V2_VARIANT_IDS = (
    "official_known_sample_sanity",
    "oracle_mask_control",
    "predicted_mask",
    "depth_component_mask",
    "bbox_mask",
)
V2_VARIANT_INPUT_ROLES = {
    "official_known_sample_sanity": "development-control-input",
    "oracle_mask_control": "development-control-input",
    "predicted_mask": "predicted-segmentation",
    "depth_component_mask": "depth-component-segmentation",
    "bbox_mask": "bbox-segmentation",
}
V2_DERIVATION_CLASSES = {
    "official_known_sample_sanity": {"gt-derived-development-control"},
    "oracle_mask_control": {"gt-derived-development-control"},
    "predicted_mask": {"predicted-segmentation"},
    "depth_component_mask": {"depth-component"},
    "bbox_mask": {"bbox-from-predicted-detection"},
}
V2_BOUNDARY_FIXED = {
    "contains_raw_gt_paths": False,
    "gpu_c_resolves_derivation": False,
    "label_access_count_on_gpu_c": 0,
    "gt_path_open_count_on_gpu_c": 0,
    "evaluator_path_open_count_on_gpu_c": 0,
    "official_scorer_run": False,
    "contains_evaluator_output": False,
    "contains_sealed_data": False,
    "accuracy_claim_permitted": False,
}
V2_MANIFEST_KEYS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "input_kind",
    "source_contract",
    "producer_runtime_lock",
    "mask_variants",
    "coverage",
    "items",
    "boundary",
    "manifest_lock_sha256",
}
V2_VARIANT_KEYS = {
    "mask_variant_id",
    "input_role",
    "plugin_id",
    "source_boundary",
    "development_control",
}
V2_PROVENANCE_KEYS = {
    "plugin_id",
    "input_role",
    "source_artifact_sha256",
    "generator_id",
    "generator_version",
    "model_sha256",
    "score",
    "coverage",
    "derivation_class",
    "development_control",
    "source_path_disclosed_to_gpu_c",
    "gpu_c_resolves_derivation",
    "label_access_count_on_gpu_c",
}
V2_FORBIDDEN_ASSET_PATH_TOKENS = {
    "gt",
    "groundtruth",
    "ground_truth",
    "oracle",
    "sanity",
    "evaluator",
    "evaluation",
    "score",
    "sealed",
}


def _reject_forbidden_keys(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_KEYS & set(value))
        if forbidden:
            raise PrepError(f"Forbidden GT/evaluator fields at {location}: {forbidden}")
        for key, child in value.items():
            _reject_forbidden_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{location}[{index}]")


def _safe_relative_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PrepError(f"Invalid relative path for {role}: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PrepError(f"Unsafe relative path for {role}: {value}")
    lowered = [part.lower() for part in path.parts]
    if any(token in part for token in FORBIDDEN_PATH_TOKENS for part in lowered):
        raise PrepError(f"Forbidden GT/evaluator path for {role}: {value}")
    return path.as_posix()


def _validate_asset(value: Any, *, slot: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ASSET_KEYS:
        raise PrepError(f"Asset contract differs for {slot}")
    expected_role = INPUT_ROLES[slot]
    if value.get("role") != expected_role:
        raise PrepError(f"Asset role differs for {slot}")
    relative = _safe_relative_path(value.get("relative_path"), role=expected_role)
    if not is_sha256(value.get("sha256")):
        raise PrepError(f"Invalid SHA-256 for {slot}")
    byte_count = value.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise PrepError(f"Invalid byte count for {slot}")
    return {
        "role": expected_role,
        "relative_path": relative,
        "sha256": value["sha256"],
        "bytes": byte_count,
    }


def _validate_v2_asset(value: Any, *, slot: str) -> dict[str, Any]:
    asset = _validate_asset(value, slot=slot)
    relative = Path(asset["relative_path"])
    if not relative.parts or relative.parts[0].lower() != "assets":
        raise PrepError(f"V2 asset must remain below assets/: {asset['relative_path']}")
    tokens = {
        token
        for part in relative.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token
    }
    if tokens & V2_FORBIDDEN_ASSET_PATH_TOKENS:
        raise PrepError(
            f"V2 asset path exposes GT/evaluator derivation: {asset['relative_path']}"
        )
    return asset


def _validate_matrix(value: Any, *, rows: int, columns: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise PrepError(f"{name} must be {rows}x{columns}")
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise PrepError(f"{name} must be {rows}x{columns}")
        for scalar in row:
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                raise PrepError(f"{name} contains a non-numeric value")


def _validate_variant(value: Any) -> dict[str, Any]:
    required = {
        "mask_variant_id",
        "input_role",
        "plugin_id",
        "source_boundary",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PrepError("Mask variant contract differs")
    variant_id = value.get("mask_variant_id")
    if not isinstance(variant_id, str) or not ITEM_ID_PATTERN.fullmatch(variant_id):
        raise PrepError("Mask variant ID is invalid")
    if value.get("input_role") not in MASK_INPUT_ROLES:
        raise PrepError(f"Unsupported mask input role: {value.get('input_role')}")
    if value.get("plugin_id") not in MASK_PLUGIN_IDS:
        raise PrepError(f"Unsupported mask plugin: {value.get('plugin_id')}")
    if value.get("source_boundary") != "opaque-upstream-input-only":
        raise PrepError("Mask source boundary must remain opaque on GPU-C")
    return dict(value)


def _validate_mask_provenance(
    value: Any,
    *,
    variant: Mapping[str, Any],
    mask_asset: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "plugin_id",
        "input_role",
        "source_artifact_sha256",
        "generator_id",
        "generator_version",
        "model_sha256",
        "score",
        "coverage",
        "label_access_count_on_gpu_c",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PrepError("Mask provenance contract differs")
    if value.get("plugin_id") != variant["plugin_id"]:
        raise PrepError("Mask provenance plugin differs from variant")
    if value.get("input_role") != variant["input_role"]:
        raise PrepError("Mask provenance role differs from variant")
    if value.get("label_access_count_on_gpu_c") != 0:
        raise PrepError("Mask provenance declares GPU-C label access")
    if not is_sha256(value.get("source_artifact_sha256")):
        raise PrepError("Mask provenance source hash is invalid")
    model_sha = value.get("model_sha256")
    if model_sha is not None and not is_sha256(model_sha):
        raise PrepError("Mask provenance model hash is invalid")
    if not isinstance(value.get("generator_id"), str) or not value["generator_id"]:
        raise PrepError("Mask provenance generator ID is invalid")
    if not isinstance(value.get("generator_version"), str) or not value["generator_version"]:
        raise PrepError("Mask provenance generator version is invalid")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise PrepError("Mask provenance score is invalid")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "nonzero_pixels",
        "fraction",
    }:
        raise PrepError("Mask coverage contract differs")
    if not isinstance(coverage["nonzero_pixels"], int) or coverage["nonzero_pixels"] <= 0:
        raise PrepError("Mask nonzero coverage is invalid")
    fraction = coverage["fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise PrepError("Mask fractional coverage is invalid")
    if mask_asset["sha256"] != value["source_artifact_sha256"]:
        raise PrepError("Mask asset and provenance source hashes differ")
    return dict(value)


def _resolve_exact(asset_root: Path, asset: Mapping[str, Any]) -> Path:
    root = asset_root.resolve()
    candidate = (root / str(asset["relative_path"])).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PrepError(f"Asset escapes root: {asset['relative_path']}") from exc
    if candidate.is_symlink():
        raise PrepError(f"Symlink input is forbidden: {asset['relative_path']}")
    if not candidate.is_file():
        raise PrepError(f"Manifest-listed asset is missing: {asset['relative_path']}")
    if candidate.stat().st_size != asset["bytes"]:
        raise PrepError(f"Asset byte count differs: {asset['relative_path']}")
    if sha256_file(candidate) != asset["sha256"]:
        raise PrepError(f"Asset SHA-256 differs: {asset['relative_path']}")
    return candidate


def expected_coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
    variant_to_samples: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    execution_keys: list[tuple[int, int, int, str]] = []
    for item in items:
        key = item["sample_key"]
        sample = (key["scene_id"], key["image_id"], key["object_id"])
        variant = item["mask_variant_id"]
        variant_to_samples[variant].add(sample)
        execution_keys.append((*sample, variant))
    variants = sorted(variant_to_samples)
    reference = variant_to_samples[variants[0]]
    for variant in variants[1:]:
        if variant_to_samples[variant] != reference:
            raise PrepError("Mask variants do not cover the same base sample set")
    return {
        "required_percent": 100,
        "item_count": len(items),
        "unique_execution_key_count": len(set(execution_keys)),
        "base_sample_count": len(reference),
        "mask_variant_ids": variants,
        "per_variant_item_count": {
            variant: len(variant_to_samples[variant]) for variant in variants
        },
        "execution_keys_sha256": canonical_sha256(
            [list(key) for key in execution_keys]
        ),
    }


def expected_v2_coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the A-contract coverage lock while preserving frozen variant order."""

    base_samples = {
        (
            item["sample_key"]["scene_id"],
            item["sample_key"]["image_id"],
            item["sample_key"]["object_id"],
        )
        for item in items
    }
    execution_keys = [
        [
            item["sample_key"]["scene_id"],
            item["sample_key"]["image_id"],
            item["sample_key"]["object_id"],
            item["mask_variant_id"],
        ]
        for item in items
    ]
    return {
        "required_percent": 100,
        "item_count": len(items),
        "unique_execution_key_count": len({tuple(key) for key in execution_keys}),
        "base_sample_count": len(base_samples),
        "mask_variant_ids": list(V2_VARIANT_IDS),
        "per_variant_item_count": {
            variant_id: sum(
                item["mask_variant_id"] == variant_id for item in items
            )
            for variant_id in V2_VARIANT_IDS
        },
        "execution_keys_sha256": canonical_sha256(execution_keys),
    }


def _validate_manifest_v1(
    manifest_path: Path,
    *,
    protocol_path: Path,
    asset_root: Path | None = None,
    verify_assets: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path.resolve())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PrepError("PREP manifest schema is invalid")
    if set(manifest) != MANIFEST_KEYS:
        raise PrepError("PREP manifest fields differ from the frozen schema")
    if manifest.get("protocol_id") != PREP_PROTOCOL_ID:
        raise PrepError("PREP manifest protocol ID differs")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path.resolve()):
        raise PrepError("PREP manifest protocol SHA-256 differs")
    _reject_forbidden_keys(manifest)
    if manifest.get("boundary") != BOUNDARY:
        raise PrepError("PREP label/evaluator boundary differs")
    source = manifest.get("source_contract")
    if not isinstance(source, dict) or set(source) != {
        "protocol_id",
        "protocol_sha256",
        "bundle_sha256",
    }:
        raise PrepError("Upstream source contract differs")
    if not isinstance(source["protocol_id"], str) or not source["protocol_id"]:
        raise PrepError("Upstream protocol ID is invalid")
    for field in ("protocol_sha256", "bundle_sha256"):
        if not is_sha256(source[field]):
            raise PrepError(f"Upstream {field} is invalid")
    variants_raw = manifest.get("mask_variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise PrepError("PREP manifest has no mask variants")
    variants = [_validate_variant(value) for value in variants_raw]
    by_variant = {value["mask_variant_id"]: value for value in variants}
    if len(by_variant) != len(variants):
        raise PrepError("Duplicate mask variant ID")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise PrepError("PREP manifest has no items")
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[int, int, int, str]] = set()
    for ordinal, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise PrepError(f"PREP item is not an object: {ordinal}")
        if set(raw) != ITEM_KEYS:
            raise PrepError(f"PREP item fields differ: {ordinal}")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not ITEM_ID_PATTERN.fullmatch(item_id):
            raise PrepError(f"Invalid PREP item ID: {item_id!r}")
        if item_id in seen_ids:
            raise PrepError(f"Duplicate PREP item ID: {item_id}")
        seen_ids.add(item_id)
        sample = raw.get("sample_key")
        if not isinstance(sample, dict) or set(sample) != {
            "scene_id",
            "image_id",
            "object_id",
        }:
            raise PrepError(f"Invalid sample key: {item_id}")
        scene_id, image_id, object_id = (
            sample["scene_id"],
            sample["image_id"],
            sample["object_id"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (scene_id, image_id, object_id)):
            raise PrepError(f"Non-integer sample key: {item_id}")
        if scene_id < 0 or image_id < 0 or object_id <= 0:
            raise PrepError(f"Out-of-range sample key: {item_id}")
        variant_id = raw.get("mask_variant_id")
        if variant_id not in by_variant:
            raise PrepError(f"Unknown mask variant: {item_id}")
        execution_key = (scene_id, image_id, object_id, str(variant_id))
        if execution_key in seen_keys:
            raise PrepError(f"Duplicate PREP execution key: {execution_key}")
        seen_keys.add(execution_key)
        frame_size = raw.get("frame_size")
        if not isinstance(frame_size, dict) or set(frame_size) != {"width", "height"}:
            raise PrepError(f"Frame size contract differs: {item_id}")
        if any(not isinstance(frame_size[field], int) or frame_size[field] <= 0 for field in ("width", "height")):
            raise PrepError(f"Frame size is invalid: {item_id}")
        inputs = raw.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != set(INPUT_ROLES):
            raise PrepError(f"Input role coverage differs: {item_id}")
        assets = {
            slot: _validate_asset(inputs[slot], slot=slot) for slot in INPUT_ROLES
        }
        _validate_matrix(
            raw.get("camera_intrinsics"), rows=3, columns=3, name="camera_intrinsics"
        )
        depth_scale = raw.get("depth_scale")
        if isinstance(depth_scale, bool) or not isinstance(depth_scale, (int, float)) or depth_scale <= 0:
            raise PrepError(f"Depth scale is invalid: {item_id}")
        provenance = _validate_mask_provenance(
            raw.get("mask_provenance"),
            variant=by_variant[str(variant_id)],
            mask_asset=assets["mask"],
        )
        item = {
            "ordinal": ordinal,
            "item_id": item_id,
            "sample_key": dict(sample),
            "mask_variant_id": variant_id,
            "frame_size": dict(frame_size),
            "inputs": assets,
            "camera_intrinsics": raw["camera_intrinsics"],
            "depth_scale": depth_scale,
            "mask_provenance": provenance,
        }
        item["item_fingerprint"] = canonical_sha256(item)
        items.append(item)
    coverage = expected_coverage(items)
    if coverage["unique_execution_key_count"] != len(items):
        raise PrepError("PREP manifest execution keys are not unique")
    if manifest.get("coverage") != coverage:
        raise PrepError("PREP manifest coverage declaration differs")
    if verify_assets:
        if asset_root is None:
            raise PrepError("Asset root is required for file verification")
        for item in items:
            for asset in item["inputs"].values():
                _resolve_exact(asset_root, asset)
    return manifest, items


def _validate_v2_variant(value: Any, *, expected_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != V2_VARIANT_KEYS:
        raise PrepError("V2 mask variant contract differs")
    if value.get("mask_variant_id") != expected_id:
        raise PrepError("V2 mask variants differ from the frozen order")
    if value.get("input_role") != V2_VARIANT_INPUT_ROLES[expected_id]:
        raise PrepError(f"V2 mask input role differs: {expected_id}")
    if value.get("plugin_id") not in MASK_PLUGIN_IDS:
        raise PrepError(f"Unsupported V2 mask plugin: {expected_id}")
    if value.get("source_boundary") != "opaque-upstream-input-only":
        raise PrepError("V2 mask source boundary must remain opaque on GPU-C")
    if value.get("development_control") is not (
        expected_id in V2_VARIANT_IDS[:2]
    ):
        raise PrepError(f"V2 development-control marker differs: {expected_id}")
    return dict(value)


def _validate_v2_provenance(
    value: Any,
    *,
    variant: Mapping[str, Any],
    mask_asset: Mapping[str, Any],
    frame_size: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != V2_PROVENANCE_KEYS:
        raise PrepError("V2 mask provenance contract differs")
    variant_id = str(variant["mask_variant_id"])
    if value.get("plugin_id") != variant["plugin_id"]:
        raise PrepError("V2 mask provenance plugin differs")
    if value.get("input_role") != variant["input_role"]:
        raise PrepError("V2 mask provenance input role differs")
    if value.get("source_artifact_sha256") != mask_asset["sha256"]:
        raise PrepError("V2 mask asset and provenance source hashes differ")
    if value.get("derivation_class") not in V2_DERIVATION_CLASSES[variant_id]:
        raise PrepError(f"V2 mask derivation class differs: {variant_id}")
    if value.get("development_control") is not variant["development_control"]:
        raise PrepError("V2 mask development-control provenance differs")
    if value.get("source_path_disclosed_to_gpu_c") is not False:
        raise PrepError("V2 provenance discloses an upstream derivation path")
    if value.get("gpu_c_resolves_derivation") is not False:
        raise PrepError("V2 provenance lets GPU-C resolve upstream derivation")
    if value.get("label_access_count_on_gpu_c") != 0:
        raise PrepError("V2 mask provenance declares GPU-C label access")
    if value.get("generator_id") != "pose-accuracy-recovery-a-export":
        raise PrepError("V2 mask provenance generator ID differs")
    if value.get("generator_version") != "v2":
        raise PrepError("V2 mask provenance generator version differs")
    model_sha = value.get("model_sha256")
    if model_sha is not None and not is_sha256(model_sha):
        raise PrepError("V2 mask provenance model SHA-256 is invalid")
    score = value.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
    ):
        raise PrepError("V2 mask provenance score is invalid")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "nonzero_pixels",
        "fraction",
    }:
        raise PrepError("V2 mask coverage contract differs")
    nonzero = coverage.get("nonzero_pixels")
    fraction = coverage.get("fraction")
    total = int(frame_size["width"]) * int(frame_size["height"])
    if (
        isinstance(nonzero, bool)
        or not isinstance(nonzero, int)
        or not 0 < nonzero <= total
    ):
        raise PrepError("V2 mask nonzero coverage is invalid")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0 < float(fraction) <= 1
    ):
        raise PrepError("V2 mask fractional coverage is invalid")
    if not math.isclose(float(fraction), nonzero / total, abs_tol=1e-12):
        raise PrepError("V2 mask fractional coverage is inconsistent")
    return dict(value)


def _validate_v2_runtime_lock(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "implementation_commit",
        "implementation_sha256",
        "model_sha256",
        "checkpoint_sha256",
    }:
        raise PrepError("V2 producer runtime lock differs")
    commit = value.get("implementation_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PrepError("V2 implementation commit is invalid")
    for field in ("implementation_sha256", "model_sha256"):
        if not is_sha256(value.get(field)):
            raise PrepError(f"V2 {field} is invalid")
    checkpoints = value.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        "refiner",
        "scorer",
    }:
        raise PrepError("V2 checkpoint SHA contract differs")
    if any(not is_sha256(checkpoints[name]) for name in ("refiner", "scorer")):
        raise PrepError("V2 checkpoint SHA-256 is invalid")
    return dict(value)


def _validate_v2_intrinsics(value: Any, *, item_id: str) -> None:
    _validate_matrix(value, rows=3, columns=3, name="camera_intrinsics")
    flattened = [float(cell) for row in value for cell in row]
    if not all(math.isfinite(cell) for cell in flattened):
        raise PrepError(f"V2 camera intrinsics are non-finite: {item_id}")
    if float(value[0][0]) <= 0 or float(value[1][1]) <= 0:
        raise PrepError(f"V2 camera focal length is invalid: {item_id}")
    if [float(cell) for cell in value[2]] != [0.0, 0.0, 1.0]:
        raise PrepError(f"V2 camera intrinsics last row differs: {item_id}")


def _validate_manifest_v2(
    manifest_path: Path,
    *,
    protocol_path: Path,
    asset_root: Path | None = None,
    verify_assets: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path.resolve())
    if not isinstance(manifest, dict) or set(manifest) != V2_MANIFEST_KEYS:
        raise PrepError("V2 runtime-isolated manifest fields differ")
    if manifest.get("schema_version") != RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        raise PrepError("V2 runtime-isolated manifest schema is invalid")
    if manifest.get("protocol_id") != PREP_PROTOCOL_ID:
        raise PrepError("V2 runtime-isolated manifest protocol differs")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path.resolve()):
        raise PrepError("V2 runtime-isolated manifest protocol SHA-256 differs")
    if manifest.get("input_kind") not in {
        "SYNTHETIC_COMMITTED_FIXTURE",
        "DEVELOPMENT_DATA",
    }:
        raise PrepError("V2 input kind is invalid")
    unlocked = dict(manifest)
    declared_lock = unlocked.pop("manifest_lock_sha256")
    if not is_sha256(declared_lock) or declared_lock != canonical_sha256(unlocked):
        raise PrepError("V2 runtime-isolated manifest lock differs")
    source = manifest.get("source_contract")
    if not isinstance(source, dict) or set(source) != {
        "protocol_id",
        "protocol_sha256",
        "bundle_sha256",
        "a_manifest_lock_sha256",
    }:
        raise PrepError("V2 upstream source contract differs")
    if source.get("protocol_id") != "poseloop.pose-accuracy-recovery.development-prep.v1":
        raise PrepError("V2 upstream protocol ID differs")
    if source.get("protocol_sha256") != A_SOURCE_PROTOCOL_SHA256:
        raise PrepError("V2 upstream protocol SHA-256 differs from frozen A")
    for field in ("protocol_sha256", "bundle_sha256", "a_manifest_lock_sha256"):
        if not is_sha256(source.get(field)):
            raise PrepError(f"V2 upstream {field} is invalid")
    _validate_v2_runtime_lock(manifest.get("producer_runtime_lock"))
    variants_raw = manifest.get("mask_variants")
    if not isinstance(variants_raw, list) or len(variants_raw) != len(V2_VARIANT_IDS):
        raise PrepError("V2 manifest requires exactly five mask variants")
    variants = [
        _validate_v2_variant(value, expected_id=expected_id)
        for value, expected_id in zip(variants_raw, V2_VARIANT_IDS)
    ]
    by_variant = {value["mask_variant_id"]: value for value in variants}
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise PrepError("V2 manifest has no items")
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[int, int, int, str]] = set()
    gt_derived_controls = False
    for ordinal, raw in enumerate(raw_items):
        if not isinstance(raw, dict) or set(raw) != ITEM_KEYS:
            raise PrepError(f"V2 item fields differ: {ordinal}")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not ITEM_ID_PATTERN.fullmatch(item_id):
            raise PrepError(f"Invalid V2 item ID: {item_id!r}")
        if item_id in seen_ids:
            raise PrepError(f"Duplicate V2 item ID: {item_id}")
        seen_ids.add(item_id)
        sample = raw.get("sample_key")
        if not isinstance(sample, dict) or set(sample) != {
            "scene_id",
            "image_id",
            "object_id",
        }:
            raise PrepError(f"Invalid V2 sample key: {item_id}")
        scene_id, image_id, object_id = (
            sample["scene_id"],
            sample["image_id"],
            sample["object_id"],
        )
        if any(
            isinstance(number, bool) or not isinstance(number, int)
            for number in (scene_id, image_id, object_id)
        ) or scene_id < 0 or image_id < 0 or object_id <= 0:
            raise PrepError(f"Out-of-range V2 sample key: {item_id}")
        variant_id = raw.get("mask_variant_id")
        if variant_id not in by_variant:
            raise PrepError(f"Unknown V2 mask variant: {item_id}")
        execution_key = (scene_id, image_id, object_id, str(variant_id))
        if execution_key in seen_keys:
            raise PrepError(f"Duplicate V2 execution key: {execution_key}")
        seen_keys.add(execution_key)
        expected_item_id = (
            f"s{scene_id:06d}-i{image_id:06d}-o{object_id:06d}-{variant_id}"
        )
        if item_id != expected_item_id:
            raise PrepError(f"V2 item ID is not deterministic: {item_id}")
        frame_size = raw.get("frame_size")
        if not isinstance(frame_size, dict) or set(frame_size) != {"width", "height"}:
            raise PrepError(f"V2 frame size contract differs: {item_id}")
        if any(
            isinstance(frame_size[field], bool)
            or not isinstance(frame_size[field], int)
            or frame_size[field] <= 0
            for field in ("width", "height")
        ):
            raise PrepError(f"V2 frame size is invalid: {item_id}")
        inputs = raw.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != set(INPUT_ROLES):
            raise PrepError(f"V2 input role coverage differs: {item_id}")
        assets = {
            slot: _validate_v2_asset(inputs[slot], slot=slot)
            for slot in INPUT_ROLES
        }
        _validate_v2_intrinsics(raw.get("camera_intrinsics"), item_id=item_id)
        depth_scale = raw.get("depth_scale")
        if (
            isinstance(depth_scale, bool)
            or not isinstance(depth_scale, (int, float))
            or not math.isfinite(float(depth_scale))
            or depth_scale <= 0
        ):
            raise PrepError(f"V2 depth scale is invalid: {item_id}")
        provenance = _validate_v2_provenance(
            raw.get("mask_provenance"),
            variant=by_variant[str(variant_id)],
            mask_asset=assets["mask"],
            frame_size=frame_size,
        )
        gt_derived_controls = gt_derived_controls or (
            provenance["derivation_class"] == "gt-derived-development-control"
        )
        item = {
            "ordinal": ordinal,
            "item_id": item_id,
            "sample_key": dict(sample),
            "mask_variant_id": variant_id,
            "frame_size": dict(frame_size),
            "inputs": assets,
            "camera_intrinsics": raw["camera_intrinsics"],
            "depth_scale": depth_scale,
            "mask_provenance": provenance,
        }
        item["item_fingerprint"] = canonical_sha256(item)
        items.append(item)
    coverage = expected_v2_coverage(items)
    if coverage["unique_execution_key_count"] != len(items):
        raise PrepError("V2 execution keys are not unique")
    if coverage["mask_variant_ids"] != list(V2_VARIANT_IDS):
        raise PrepError("V2 coverage does not include exactly five variants")
    if manifest.get("coverage") != coverage:
        raise PrepError("V2 manifest coverage declaration differs")
    boundary = manifest.get("boundary")
    if not isinstance(boundary, dict) or set(boundary) != {
        "contains_gt_derived_control_inputs",
        *V2_BOUNDARY_FIXED,
    }:
        raise PrepError("V2 runtime-isolation boundary fields differ")
    if gt_derived_controls is not True:
        raise PrepError("V2 manifest is missing its GT-derived DEVELOPMENT controls")
    if boundary.get("contains_gt_derived_control_inputs") is not True:
        raise PrepError("V2 GT-derived control boundary is not truthful")
    for field, expected in V2_BOUNDARY_FIXED.items():
        if boundary.get(field) != expected:
            raise PrepError(f"V2 runtime-isolation boundary differs: {field}")
    _reject_forbidden_keys(manifest)
    if verify_assets:
        if asset_root is None:
            raise PrepError("Asset root is required for V2 file verification")
        for item in items:
            for asset in item["inputs"].values():
                _resolve_exact(asset_root, asset)
    return manifest, items


def validate_manifest(
    manifest_path: Path,
    *,
    protocol_path: Path,
    asset_root: Path | None = None,
    verify_assets: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate legacy v1 fixtures or the formal runtime-isolated v2 handoff."""

    value = read_json(manifest_path.resolve())
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema == MANIFEST_SCHEMA:
        return _validate_manifest_v1(
            manifest_path,
            protocol_path=protocol_path,
            asset_root=asset_root,
            verify_assets=verify_assets,
        )
    if schema == RUNTIME_ISOLATED_MANIFEST_SCHEMA_V2:
        return _validate_manifest_v2(
            manifest_path,
            protocol_path=protocol_path,
            asset_root=asset_root,
            verify_assets=verify_assets,
        )
    raise PrepError("Unsupported PREP manifest schema")


def asset_path(asset_root: Path, asset: Mapping[str, Any]) -> Path:
    """Resolve one already-validated manifest asset without scanning its root."""

    return _resolve_exact(asset_root, asset)
