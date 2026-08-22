"""Canonical runtime-isolated A-to-C development handoff contract.

This module is deliberately local and CPU-only.  GPU-A owns the unified
manifest and may read evaluator-only DEVELOPMENT_ONLY masks while exporting.
GPU-C receives only copied runtime inputs and never resolves their derivation.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .core import (
    ContractError,
    canonical_sha256,
    load_and_validate_manifest,
    sha256_file,
    write_json,
)

C_HANDOFF_SCHEMA = "poseloop.r4c.prep.runtime-isolated-manifest.v2"
C_HANDOFF_PROTOCOL_ID = "poseloop-r4c-foundationpose-runtime-prep-v1"
C_HANDOFF_RECEIPT_SCHEMA = "poseloop.pose-accuracy-recovery.c-handoff-receipt.v2"
C_HANDOFF_PIN_SCHEMA = "poseloop.pose-accuracy-recovery.c-runtime-protocol-pin.v1"
C_RUNTIME_PROTOCOL_SHA256 = (
    "d915325a5be8201a5b489c52c3c0720520bcfcebf9a3a06d7c5ec55c17efb0c9"
)
A_SOURCE_PROTOCOL_SHA256 = (
    "20380305a7fbd92b2c563baea7df1e4f82c39605379543cd8e9443925eae0b8b"
)

_ASSET_ROLES = {
    "rgb": "public_rgb",
    "depth": "public_depth",
    "mask": "input_mask",
    "camera": "public_camera",
    "cad": "public_cad",
}
_SHARED_INPUTS = ("rgb", "depth", "camera", "cad")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_FORBIDDEN_ASSET_PATH_TOKENS = {
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

# Stable order is part of the protocol and therefore part of the coverage lock.
C_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "mask_variant_id": "official_known_sample_sanity",
        "source_variant": "official_known_sample_sanity",
        "input_role": "development-control-input",
        "plugin_id": "manifest-mask-file-v1",
        "opaque_slot": "control-a",
        "development_control": True,
        "derivation_class": "gt-derived-development-control",
        "synthetic_score": 1.0,
    },
    {
        "mask_variant_id": "oracle_mask_control",
        "source_variant": "oracle_mask_control",
        "input_role": "development-control-input",
        "plugin_id": "manifest-mask-file-v1",
        "opaque_slot": "control-b",
        "development_control": True,
        "derivation_class": "gt-derived-development-control",
        "synthetic_score": 1.0,
    },
    {
        "mask_variant_id": "predicted_mask",
        "source_variant": "predicted_mask",
        "input_role": "predicted-segmentation",
        "plugin_id": "manifest-mask-file-v1",
        "opaque_slot": "variant-02",
        "development_control": False,
        "derivation_class": "predicted-segmentation",
        "synthetic_score": 0.9,
    },
    {
        "mask_variant_id": "depth_component_mask",
        "source_variant": "depth_component_mask",
        "input_role": "depth-component-segmentation",
        "plugin_id": "manifest-mask-file-v1",
        "opaque_slot": "variant-03",
        "development_control": False,
        "derivation_class": "depth-component",
        "synthetic_score": 0.8,
    },
    {
        "mask_variant_id": "bbox_mask",
        "source_variant": "bbox_mask",
        "input_role": "bbox-segmentation",
        "plugin_id": "manifest-mask-file-v1",
        "opaque_slot": "variant-04",
        "development_control": False,
        "derivation_class": "bbox-from-predicted-detection",
        "synthetic_score": 0.7,
    },
)

_BOUNDARY = {
    "contains_gt_derived_control_inputs": True,
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive " if positive else ""
        raise ContractError(f"{label} must be a finite {qualifier}number")
    return float(value)


def _load_json_asset(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} JSON must be an object")
    return value


def _resolve_source(asset: Mapping[str, Any], data_root: Path, label: str) -> Path:
    relative = asset.get("path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ContractError(f"{label}.path must be a non-empty POSIX path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise ContractError(f"{label}.path escapes the source data root")
    root = data_root.resolve()
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label}.path escapes the source data root") from exc
    if not candidate.is_file():
        raise ContractError(f"Missing {label}: {relative}")
    if candidate.stat().st_size <= 0:
        raise ContractError(f"Empty source asset for {label}: {relative}")
    if sha256_file(candidate) != asset.get("sha256"):
        raise ContractError(f"Source hash mismatch for {label}: {relative}")
    return candidate


def _extension(source: Path) -> str:
    suffix = "".join(source.suffixes)
    return suffix if suffix else ".bin"


def _copy_asset(
    source: Path,
    bundle_root: Path,
    relative: str,
    role: str,
) -> dict[str, Any]:
    destination = bundle_root / Path(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "role": role,
        "relative_path": relative,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _sample_key(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    key = sample["key"]
    return int(key["scene_id"]), int(key["image_id"]), int(key["object_id"])


def _sample_token(key: Sequence[int]) -> str:
    return f"s{key[0]:06d}-i{key[1]:06d}-o{key[2]:06d}"


def _item_id(key: Sequence[int], variant_id: str) -> str:
    return f"{_sample_token(key)}-{variant_id}"


def _read_image_frame_size(path: Path, label: str) -> dict[str, int]:
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ContractError(
            f"{label} must be JSON, PNG, JPEG, or JPG; got {path.suffix or '<none>'}"
        )
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise ContractError(
                f"Pillow or OpenCV is required to read real {label} images"
            ) from exc
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2:
            raise ContractError(f"Cannot decode {label} image {path}")
        height, width = image.shape[:2]
    else:
        try:
            with Image.open(path) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ContractError(
                        f"{label} image format must be PNG or JPEG, got {image.format}"
                    )
                width, height = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"Cannot decode {label} image {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ContractError(f"{label} image dimensions must be positive")
    return {"width": int(width), "height": int(height)}


def _read_frame_size(path: Path, label: str = "rgb") -> dict[str, int]:
    if path.suffix.lower() != ".json":
        return _read_image_frame_size(path, label)
    payload = _load_json_asset(path, label)
    width = payload.get("width")
    height = payload.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise ContractError(
            f"Synthetic/JSON {label} must declare positive width and height"
        )
    return {"width": width, "height": height}


def _read_camera(camera_path: Path) -> tuple[list[list[float]], float]:
    payload = _load_json_asset(camera_path, "camera")
    intrinsics = payload.get("camera_intrinsics")
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in intrinsics)
    ):
        raise ContractError("camera.camera_intrinsics must be a 3x3 matrix")
    normalized = [
        [_finite_number(cell, "camera_intrinsics") for cell in row]
        for row in intrinsics
    ]
    depth_scale = _finite_number(
        payload.get("depth_scale_to_m"), "camera.depth_scale_to_m", positive=True
    )
    return normalized, depth_scale


def _read_image_mask_coverage(
    mask_path: Path, frame_size: Mapping[str, int]
) -> dict[str, Any]:
    if mask_path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ContractError(
            "Mask must be JSON, PNG, JPEG, or JPG; "
            f"got {mask_path.suffix or '<none>'}"
        )
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise ContractError(
                "Pillow or OpenCV is required to read real mask images"
            ) from exc
        image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise ContractError(f"Mask image must be single-channel: {mask_path}")
        height, width = image.shape
        nonzero = int((image != 0).sum())
        values: set[int] = set()
        for value in image.flat:
            values.add(int(value))
            if len(values) > 2:
                break
    else:
        try:
            with Image.open(mask_path) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ContractError(
                        f"Mask image format must be PNG or JPEG, got {image.format}"
                    )
                if image.mode not in {"1", "L", "I", "I;16", "P"}:
                    raise ContractError(
                        f"Mask image must be single-channel, got mode {image.mode}"
                    )
                width, height = image.size
                grayscale = image.convert("L")
                histogram = grayscale.histogram()
                values = {index for index, count in enumerate(histogram) if count}
                nonzero = sum(histogram[1:])
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"Cannot decode mask image {mask_path}: {exc}") from exc
    if width != frame_size["width"] or height != frame_size["height"]:
        raise ContractError("Mask frame size differs from rgb frame size")
    if len(values) > 2:
        raise ContractError("Mask image must contain a binary background/foreground")
    if nonzero <= 0:
        raise ContractError("Mask image must contain at least one foreground pixel")
    pixel_count = width * height
    return {
        "nonzero_pixels": int(nonzero),
        "fraction": nonzero / pixel_count,
    }


def _read_mask_coverage(
    mask_path: Path, frame_size: Mapping[str, int]
) -> dict[str, Any]:
    if mask_path.suffix.lower() != ".json":
        return _read_image_mask_coverage(mask_path, frame_size)
    payload = _load_json_asset(mask_path, "mask")
    width = payload.get("width")
    height = payload.get("height")
    mask = payload.get("mask")
    if width != frame_size["width"] or height != frame_size["height"]:
        raise ContractError("Mask frame size differs from rgb frame size")
    pixel_count = width * height
    if (
        not isinstance(mask, list)
        or len(mask) != pixel_count
        or any(value not in (0, 1) or isinstance(value, bool) for value in mask)
    ):
        raise ContractError("Mask payload must contain exactly width*height binary pixels")
    nonzero = sum(mask)
    if nonzero <= 0:
        raise ContractError("Mask payload must contain at least one foreground pixel")
    return {
        "nonzero_pixels": nonzero,
        "fraction": nonzero / pixel_count,
    }


def _read_mask_score(
    mask_path: Path,
    *,
    input_kind: str,
    synthetic_score: float,
) -> float:
    if mask_path.suffix.lower() == ".json":
        payload = _load_json_asset(mask_path, "mask")
        score = payload.get("score")
    else:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise ContractError(
                "Pillow is required to read hash-bound mask score metadata"
            ) from exc
        try:
            with Image.open(mask_path) as image:
                score = image.info.get("score")
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"Cannot decode mask image {mask_path}: {exc}") from exc
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError as exc:
                raise ContractError("Mask score metadata must be numeric") from exc
    if score is None and input_kind == "SYNTHETIC_COMMITTED_FIXTURE":
        score = synthetic_score
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
    ):
        raise ContractError(
            "Non-synthetic/development mask payload must declare a finite score in [0,1]"
        )
    return float(score)


def _mask_source(sample: Mapping[str, Any], variant: Mapping[str, Any]) -> Mapping[str, Any]:
    name = variant["source_variant"]
    if variant["development_control"]:
        return sample["evaluator_only"]["masks"][name]
    return sample["producer_inputs"]["masks"][name]


def _load_handoff_protocol(path: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read C handoff protocol {path}: {exc}") from exc
    expected = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "handoff_schema_version",
        "source_protocol_id",
        "source_protocol_sha256",
        "mask_variant_ids",
        "boundary",
        "auto_deploy",
        "accuracy_claim_permitted",
    }
    protocol = _strict_keys(protocol, expected, "C handoff protocol")
    if protocol["schema_version"] != C_HANDOFF_PIN_SCHEMA:
        raise ContractError("C handoff protocol pin schema mismatch")
    if protocol["protocol_id"] != C_HANDOFF_PROTOCOL_ID:
        raise ContractError("C handoff protocol identity mismatch")
    if protocol["protocol_sha256"] != C_RUNTIME_PROTOCOL_SHA256:
        raise ContractError("C runtime protocol SHA-256 pin is invalid")
    if protocol["handoff_schema_version"] != C_HANDOFF_SCHEMA:
        raise ContractError("C handoff protocol schema mismatch")
    if protocol["source_protocol_id"] != (
        "poseloop.pose-accuracy-recovery.development-prep.v1"
    ):
        raise ContractError("C handoff source protocol mismatch")
    if protocol["source_protocol_sha256"] != A_SOURCE_PROTOCOL_SHA256:
        raise ContractError("C handoff source protocol SHA-256 pin is invalid")
    if protocol["mask_variant_ids"] != [
        variant["mask_variant_id"] for variant in C_VARIANTS
    ]:
        raise ContractError("C handoff protocol variant order mismatch")
    if protocol["boundary"] != _BOUNDARY:
        raise ContractError("C handoff protocol boundary mismatch")
    if (
        protocol["auto_deploy"] is not False
        or protocol["accuracy_claim_permitted"] is not False
    ):
        raise ContractError("C handoff protocol must remain inert and claim-free")
    return protocol


def _coverage(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_ids = [variant["mask_variant_id"] for variant in C_VARIANTS]
    base_keys = sorted(
        {
            (
                item["sample_key"]["scene_id"],
                item["sample_key"]["image_id"],
                item["sample_key"]["object_id"],
            )
            for item in items
        }
    )
    per_variant = {
        variant_id: sum(
            1 for item in items if item["mask_variant_id"] == variant_id
        )
        for variant_id in variant_ids
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
        "unique_execution_key_count": len(set(map(tuple, execution_keys))),
        "base_sample_count": len(base_keys),
        "mask_variant_ids": variant_ids,
        "per_variant_item_count": per_variant,
        "execution_keys_sha256": canonical_sha256(execution_keys),
    }


def export_c_handoff(
    *,
    manifest_path: Path,
    data_root: Path,
    protocol_path: Path,
    output_root: Path,
    implementation_commit: str,
    implementation_sha256: str,
    model_sha256: str,
    refiner_checkpoint_sha256: str,
    scorer_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Export a fresh, runtime-isolated v2 directory bundle.

    The exporter refuses to overwrite any existing output.  This makes the
    exact bytes used for a future producer run reviewable before deployment.
    """
    if not _is_commit(implementation_commit):
        raise ContractError("implementation_commit must be a full lowercase commit")
    runtime_hashes = {
        "implementation_sha256": implementation_sha256,
        "model_sha256": model_sha256,
        "refiner_checkpoint_sha256": refiner_checkpoint_sha256,
        "scorer_checkpoint_sha256": scorer_checkpoint_sha256,
    }
    for name, value in runtime_hashes.items():
        if not _is_sha256(value):
            raise ContractError(f"{name} must be a lowercase SHA-256")
    output_root = output_root.resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise ContractError(f"Refusing to overwrite non-empty handoff root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest, _ = load_and_validate_manifest(
        manifest_path.resolve(), data_root=data_root.resolve(), verify_hashes=True
    )
    protocol_pin = _load_handoff_protocol(protocol_path.resolve())
    source_manifest_sha = sha256_file(manifest_path.resolve())
    protocol_sha = protocol_pin["protocol_sha256"]
    items: list[dict[str, Any]] = []
    for sample in source_manifest["samples"]:
        key = _sample_key(sample)
        token = _sample_token(key)
        producer = sample["producer_inputs"]
        source_shared = {
            name: _resolve_source(producer[name], data_root, f"{token}.{name}")
            for name in _SHARED_INPUTS
        }
        frame_size = _read_frame_size(source_shared["rgb"], "rgb")
        depth_frame_size = _read_frame_size(source_shared["depth"], "depth")
        if depth_frame_size != frame_size:
            raise ContractError("Depth frame size differs from rgb frame size")
        camera_intrinsics, depth_scale = _read_camera(source_shared["camera"])
        shared_assets = {
            name: _copy_asset(
                source,
                output_root,
                f"assets/{name}/{token}{_extension(source)}",
                _ASSET_ROLES[name],
            )
            for name, source in source_shared.items()
        }
        for variant in C_VARIANTS:
            source_contract = _mask_source(sample, variant)
            source_mask = _resolve_source(
                source_contract,
                data_root,
                f"{token}.{variant['mask_variant_id']}",
            )
            coverage = _read_mask_coverage(source_mask, frame_size)
            score = _read_mask_score(
                source_mask,
                input_kind=source_manifest["input_kind"],
                synthetic_score=variant["synthetic_score"],
            )
            mask_relative = (
                f"assets/input_masks/{variant['opaque_slot']}/{token}"
                f"{_extension(source_mask)}"
            )
            mask_asset = _copy_asset(
                source_mask, output_root, mask_relative, _ASSET_ROLES["mask"]
            )
            item = {
                "item_id": _item_id(key, variant["mask_variant_id"]),
                "sample_key": {
                    "scene_id": key[0],
                    "image_id": key[1],
                    "object_id": key[2],
                },
                "mask_variant_id": variant["mask_variant_id"],
                "frame_size": dict(frame_size),
                "inputs": {
                    "rgb": shared_assets["rgb"],
                    "depth": shared_assets["depth"],
                    "mask": mask_asset,
                    "camera": shared_assets["camera"],
                    "cad": shared_assets["cad"],
                },
                "camera_intrinsics": camera_intrinsics,
                "depth_scale": depth_scale,
                "mask_provenance": {
                    "plugin_id": variant["plugin_id"],
                    "input_role": variant["input_role"],
                    "source_artifact_sha256": source_contract["sha256"],
                    "generator_id": "pose-accuracy-recovery-a-export",
                    "generator_version": "v2",
                    "model_sha256": None,
                    "score": score,
                    "coverage": coverage,
                    "derivation_class": variant["derivation_class"],
                    "development_control": variant["development_control"],
                    "source_path_disclosed_to_gpu_c": False,
                    "gpu_c_resolves_derivation": False,
                    "label_access_count_on_gpu_c": 0,
                },
            }
            items.append(item)
    variant_contracts = [
        {
            "mask_variant_id": variant["mask_variant_id"],
            "input_role": variant["input_role"],
            "plugin_id": variant["plugin_id"],
            "source_boundary": "opaque-upstream-input-only",
            "development_control": variant["development_control"],
        }
        for variant in C_VARIANTS
    ]
    handoff: dict[str, Any] = {
        "schema_version": C_HANDOFF_SCHEMA,
        "protocol_id": C_HANDOFF_PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "input_kind": source_manifest["input_kind"],
        "source_contract": {
            "protocol_id": source_manifest["protocol_id"],
            "protocol_sha256": protocol_pin["source_protocol_sha256"],
            "bundle_sha256": canonical_sha256(
                {
                    "unified_manifest_sha256": source_manifest_sha,
                    "input_sha256": sorted(
                        {
                            item["inputs"][name]["sha256"]
                            for item in items
                            for name in _ASSET_ROLES
                        }
                    ),
                }
            ),
            "a_manifest_lock_sha256": canonical_sha256(source_manifest),
        },
        "producer_runtime_lock": {
            "implementation_commit": implementation_commit,
            "implementation_sha256": implementation_sha256,
            "model_sha256": model_sha256,
            "checkpoint_sha256": {
                "refiner": refiner_checkpoint_sha256,
                "scorer": scorer_checkpoint_sha256,
            },
        },
        "mask_variants": variant_contracts,
        "coverage": _coverage(items),
        "items": items,
        "boundary": dict(_BOUNDARY),
    }
    handoff["manifest_lock_sha256"] = canonical_sha256(handoff)
    manifest_output = output_root / "manifest.json"
    write_json(manifest_output, handoff)
    validation = validate_c_handoff_manifest(
        handoff, bundle_root=output_root, verify_assets=True
    )
    members = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    )
    sums_lines = [
        f"{sha256_file(output_root / Path(*PurePosixPath(member).parts))}  {member}"
        for member in members
    ]
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("\n".join(sums_lines) + "\n", encoding="utf-8", newline="\n")
    receipt = {
        "schema_version": C_HANDOFF_RECEIPT_SCHEMA,
        "manifest_relative_path": "manifest.json",
        "manifest_sha256": sha256_file(manifest_output),
        "manifest_lock_sha256": handoff["manifest_lock_sha256"],
        "sha256sums_relative_path": "SHA256SUMS",
        "sha256sums_sha256": sha256_file(sums_path),
        "item_count": validation["item_count"],
        "base_sample_count": validation["base_sample_count"],
        "variant_count": validation["variant_count"],
        "coverage_percent": 100,
        "contains_gt_derived_control_inputs": True,
        "contains_raw_gt_paths": False,
        "label_access_count_on_gpu_c": 0,
        "dry_run_is_accuracy_result": False,
    }
    write_json(output_root / "handoff-receipt.json", receipt)
    return receipt


def _safe_bundle_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "assets":
        raise ContractError(f"{label} must remain below assets/")
    tokens = {
        token
        for part in path.parts
        for token in part.lower().replace("-", "_").replace(".", "_").split("_")
        if token
    }
    if tokens & _FORBIDDEN_ASSET_PATH_TOKENS:
        raise ContractError(f"{label} leaks a raw GT/evaluator path role")
    return path


def _validate_bundle_asset(
    value: Any,
    *,
    name: str,
    bundle_root: Path | None,
    verify_assets: bool,
) -> dict[str, Any]:
    asset = _strict_keys(
        value, {"role", "relative_path", "sha256", "bytes"}, f"inputs.{name}"
    )
    if asset["role"] != _ASSET_ROLES[name]:
        raise ContractError(f"inputs.{name}.role must be {_ASSET_ROLES[name]}")
    relative = _safe_bundle_path(asset["relative_path"], f"inputs.{name}.relative_path")
    if not _is_sha256(asset["sha256"]):
        raise ContractError(f"inputs.{name}.sha256 must be lowercase SHA-256")
    if (
        not isinstance(asset["bytes"], int)
        or isinstance(asset["bytes"], bool)
        or asset["bytes"] <= 0
    ):
        raise ContractError(f"inputs.{name}.bytes must be a positive integer")
    if bundle_root is not None:
        root = bundle_root.resolve()
        candidate = (root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"inputs.{name}.relative_path escapes bundle") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError(f"Missing or symlinked bundle asset: {relative.as_posix()}")
        if verify_assets:
            if candidate.stat().st_size != asset["bytes"]:
                raise ContractError(f"Bundle byte count mismatch: {relative.as_posix()}")
            if sha256_file(candidate) != asset["sha256"]:
                raise ContractError(f"Bundle hash mismatch: {relative.as_posix()}")
    return asset


def _validate_frame_size(value: Any) -> dict[str, int]:
    frame = _strict_keys(value, {"width", "height"}, "frame_size")
    for name in ("width", "height"):
        if (
            not isinstance(frame[name], int)
            or isinstance(frame[name], bool)
            or frame[name] <= 0
        ):
            raise ContractError(f"frame_size.{name} must be a positive integer")
    return frame


def _validate_intrinsics(value: Any) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in value)
    ):
        raise ContractError("camera_intrinsics must be a 3x3 matrix")
    for row in value:
        for cell in row:
            _finite_number(cell, "camera_intrinsics")
    if float(value[0][0]) <= 0 or float(value[1][1]) <= 0:
        raise ContractError("camera_intrinsics focal lengths must be positive")
    if [float(cell) for cell in value[2]] != [0.0, 0.0, 1.0]:
        raise ContractError("camera_intrinsics last row must be [0,0,1]")


def _validate_runtime_lock(value: Any) -> dict[str, Any]:
    runtime = _strict_keys(
        value,
        {
            "implementation_commit",
            "implementation_sha256",
            "model_sha256",
            "checkpoint_sha256",
        },
        "producer_runtime_lock",
    )
    if not _is_commit(runtime["implementation_commit"]):
        raise ContractError("producer_runtime_lock implementation commit is invalid")
    for name in ("implementation_sha256", "model_sha256"):
        if not _is_sha256(runtime[name]):
            raise ContractError(f"producer_runtime_lock {name} is invalid")
    checkpoints = _strict_keys(
        runtime["checkpoint_sha256"],
        {"refiner", "scorer"},
        "producer_runtime_lock.checkpoint_sha256",
    )
    if any(not _is_sha256(checkpoints[name]) for name in ("refiner", "scorer")):
        raise ContractError("producer_runtime_lock checkpoint SHA-256 is invalid")
    return runtime


def _validate_mask_provenance(
    value: Any,
    *,
    variant: Mapping[str, Any],
    mask_asset: Mapping[str, Any],
    frame_size: Mapping[str, int],
    mask_path: Path | None,
) -> None:
    provenance = _strict_keys(
        value,
        {
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
        },
        "mask_provenance",
    )
    if provenance["plugin_id"] != variant["plugin_id"]:
        raise ContractError("mask_provenance.plugin_id mismatch")
    if provenance["input_role"] != variant["input_role"]:
        raise ContractError("mask_provenance.input_role mismatch")
    if not _is_sha256(provenance["source_artifact_sha256"]):
        raise ContractError("mask_provenance.source_artifact_sha256 must be SHA-256")
    if provenance["source_artifact_sha256"] != mask_asset["sha256"]:
        raise ContractError("Mask source and copied bytes must have the same SHA-256")
    if provenance["generator_id"] != "pose-accuracy-recovery-a-export":
        raise ContractError("Unexpected mask provenance generator_id")
    if provenance["generator_version"] != "v2":
        raise ContractError("Unexpected mask provenance generator_version")
    if provenance["model_sha256"] is not None and not _is_sha256(
        provenance["model_sha256"]
    ):
        raise ContractError("mask_provenance.model_sha256 must be null or SHA-256")
    score = _finite_number(provenance["score"], "mask provenance score")
    if not 0 <= score <= 1:
        raise ContractError("mask_provenance.score must be in [0,1]")
    if provenance["derivation_class"] != variant["derivation_class"]:
        raise ContractError("Mask derivation class mismatch")
    if provenance["development_control"] is not variant["development_control"]:
        raise ContractError("Mask development-control declaration mismatch")
    if provenance["source_path_disclosed_to_gpu_c"] is not False:
        raise ContractError("GPU-C cannot receive an upstream source path")
    if provenance["gpu_c_resolves_derivation"] is not False:
        raise ContractError("GPU-C cannot resolve input derivation")
    if provenance["label_access_count_on_gpu_c"] != 0:
        raise ContractError("GPU-C label access count must remain zero")
    coverage = _strict_keys(
        provenance["coverage"],
        {"nonzero_pixels", "fraction"},
        "mask_provenance.coverage",
    )
    total = frame_size["width"] * frame_size["height"]
    if (
        not isinstance(coverage["nonzero_pixels"], int)
        or isinstance(coverage["nonzero_pixels"], bool)
        or not 0 < coverage["nonzero_pixels"] <= total
    ):
        raise ContractError("Mask nonzero pixel count is invalid")
    fraction = _finite_number(coverage["fraction"], "mask coverage fraction")
    if not math.isclose(fraction, coverage["nonzero_pixels"] / total, abs_tol=1e-12):
        raise ContractError("Mask coverage fraction is inconsistent")
    if mask_path is not None:
        observed = _read_mask_coverage(mask_path, frame_size)
        if observed != coverage:
            raise ContractError("Mask coverage does not match copied mask bytes")


def validate_c_handoff_manifest(
    manifest: Mapping[str, Any],
    *,
    bundle_root: Path | None = None,
    verify_assets: bool = True,
) -> dict[str, Any]:
    """Strictly validate a v2 handoff and its exact five-variant coverage."""
    top = _strict_keys(
        manifest,
        {
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
        },
        "handoff",
    )
    if top["schema_version"] != C_HANDOFF_SCHEMA:
        raise ContractError("C handoff schema mismatch")
    if top["protocol_id"] != C_HANDOFF_PROTOCOL_ID:
        raise ContractError("C handoff protocol id mismatch")
    if top["protocol_sha256"] != C_RUNTIME_PROTOCOL_SHA256:
        raise ContractError("C handoff protocol_sha256 differs from the pinned C protocol")
    if top["input_kind"] not in {
        "SYNTHETIC_COMMITTED_FIXTURE",
        "DEVELOPMENT_DATA",
    }:
        raise ContractError("C handoff input_kind is invalid")
    unlocked = dict(top)
    lock = unlocked.pop("manifest_lock_sha256")
    if not _is_sha256(lock) or lock != canonical_sha256(unlocked):
        raise ContractError("C handoff manifest lock mismatch")
    source = _strict_keys(
        top["source_contract"],
        {
            "protocol_id",
            "protocol_sha256",
            "bundle_sha256",
            "a_manifest_lock_sha256",
        },
        "source_contract",
    )
    if source["protocol_id"] != "poseloop.pose-accuracy-recovery.development-prep.v1":
        raise ContractError("Unexpected source protocol id")
    if source["protocol_sha256"] != A_SOURCE_PROTOCOL_SHA256:
        raise ContractError("Source protocol SHA-256 differs from the frozen A protocol")
    for name in ("protocol_sha256", "bundle_sha256", "a_manifest_lock_sha256"):
        if not _is_sha256(source[name]):
            raise ContractError(f"source_contract.{name} is invalid")
    _validate_runtime_lock(top["producer_runtime_lock"])
    expected_variants = [
        {
            "mask_variant_id": variant["mask_variant_id"],
            "input_role": variant["input_role"],
            "plugin_id": variant["plugin_id"],
            "source_boundary": "opaque-upstream-input-only",
            "development_control": variant["development_control"],
        }
        for variant in C_VARIANTS
    ]
    if top["mask_variants"] != expected_variants:
        raise ContractError("C handoff mask variant contracts changed")
    if top["boundary"] != _BOUNDARY:
        raise ContractError("C handoff boundary changed")
    items = top["items"]
    if not isinstance(items, list) or not items:
        raise ContractError("C handoff requires at least one item")
    expected_item_keys = {
        "item_id",
        "sample_key",
        "mask_variant_id",
        "frame_size",
        "inputs",
        "camera_intrinsics",
        "depth_scale",
        "mask_provenance",
    }
    seen_ids: set[str] = set()
    seen_execution: set[tuple[int, int, int, str]] = set()
    per_variant_samples: dict[str, set[tuple[int, int, int]]] = {
        variant["mask_variant_id"]: set() for variant in C_VARIANTS
    }
    variant_by_id = {
        variant["mask_variant_id"]: variant for variant in C_VARIANTS
    }
    ordered_execution: list[list[Any]] = []
    for index, raw_item in enumerate(items):
        item = _strict_keys(raw_item, expected_item_keys, f"items[{index}]")
        key_obj = _strict_keys(
            item["sample_key"], {"scene_id", "image_id", "object_id"}, "sample_key"
        )
        key_values: list[int] = []
        for name in ("scene_id", "image_id", "object_id"):
            value = key_obj[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (name == "object_id" and value == 0)
            ):
                raise ContractError(f"sample_key.{name} is invalid")
            key_values.append(value)
        variant_id = item["mask_variant_id"]
        if variant_id not in variant_by_id:
            raise ContractError(f"Unknown mask variant: {variant_id}")
        if item["item_id"] != _item_id(key_values, variant_id):
            raise ContractError(f"items[{index}].item_id is not deterministic")
        if item["item_id"] in seen_ids:
            raise ContractError(f"Duplicate item_id: {item['item_id']}")
        seen_ids.add(item["item_id"])
        execution = (*key_values, variant_id)
        if execution in seen_execution:
            raise ContractError(f"Duplicate execution key: {execution}")
        seen_execution.add(execution)
        base_key = tuple(key_values)
        per_variant_samples[variant_id].add(base_key)
        ordered_execution.append([*key_values, variant_id])
        frame_size = _validate_frame_size(item["frame_size"])
        inputs = _strict_keys(item["inputs"], set(_ASSET_ROLES), "inputs")
        validated_assets = {
            name: _validate_bundle_asset(
                inputs[name],
                name=name,
                bundle_root=bundle_root,
                verify_assets=verify_assets,
            )
            for name in _ASSET_ROLES
        }
        _validate_intrinsics(item["camera_intrinsics"])
        _finite_number(item["depth_scale"], "depth_scale", positive=True)
        mask_path = None
        if bundle_root is not None and verify_assets:
            mask_relative = PurePosixPath(validated_assets["mask"]["relative_path"])
            mask_path = bundle_root.resolve() / Path(*mask_relative.parts)
        _validate_mask_provenance(
            item["mask_provenance"],
            variant=variant_by_id[variant_id],
            mask_asset=validated_assets["mask"],
            frame_size=frame_size,
            mask_path=mask_path,
        )
    reference_samples = per_variant_samples[C_VARIANTS[0]["mask_variant_id"]]
    if not reference_samples or any(
        samples != reference_samples for samples in per_variant_samples.values()
    ):
        raise ContractError("Every variant must cover the exact same base sample set")
    expected_coverage = _coverage(items)
    if top["coverage"] != expected_coverage:
        raise ContractError("C handoff coverage declaration mismatch")
    if top["coverage"]["required_percent"] != 100:
        raise ContractError("C handoff coverage must be 100 percent")
    if top["coverage"]["execution_keys_sha256"] != canonical_sha256(
        ordered_execution
    ):
        raise ContractError("C handoff execution key order/hash mismatch")
    return {
        "schema_version": "poseloop.pose-accuracy-recovery.c-handoff-validation.v2",
        "status": "valid",
        "manifest_lock_sha256": lock,
        "item_count": len(items),
        "base_sample_count": len(reference_samples),
        "variant_count": len(C_VARIANTS),
        "per_variant_item_count": top["coverage"]["per_variant_item_count"],
        "coverage_percent": 100,
        "verified_assets": bool(bundle_root is not None and verify_assets),
        "contains_gt_derived_control_inputs": True,
        "contains_raw_gt_paths": False,
        "label_access_count_on_gpu_c": 0,
        "accuracy_claim_permitted": False,
    }


def load_and_validate_c_handoff(
    manifest_path: Path,
    *,
    bundle_root: Path,
    verify_assets: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read C handoff manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ContractError("C handoff manifest must be an object")
    return manifest, validate_c_handoff_manifest(
        manifest, bundle_root=bundle_root, verify_assets=verify_assets
    )
