"""Strict manifest validation and producer/evaluator boundary handling."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from . import MANIFEST_SCHEMA, PRODUCER_MANIFEST_SCHEMA, PROTOCOL_ID

PRODUCER_VARIANTS = ("predicted_mask", "depth_component_mask", "bbox_mask")
EVALUATOR_VARIANTS = (
    "official_known_sample_sanity",
    "oracle_mask_control",
    *PRODUCER_VARIANTS,
)
REQUIRED_PRODUCER_ASSETS = ("rgb", "depth", "camera", "cad")
FORBIDDEN_PRODUCER_TOKENS = {
    "gt",
    "ground_truth",
    "oracle",
    "evaluator",
    "evaluation",
    "score",
    "scores",
    "metric",
    "metrics",
    "sealed",
}


class ContractError(ValueError):
    """Raised when an input violates the frozen PREP data boundary."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tokens(value: str) -> set[str]:
    normalized = (
        value.lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return {token for token in normalized.split("_") if token}


def _assert_no_producer_leak(value: Any, trail: str = "producer_inputs") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _tokens(str(key)) & FORBIDDEN_PRODUCER_TOKENS:
                raise ContractError(f"GT/evaluator leak in producer key {trail}.{key}")
            _assert_no_producer_leak(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_producer_leak(child, f"{trail}[{index}]")
    elif isinstance(value, str) and _tokens(value) & FORBIDDEN_PRODUCER_TOKENS:
        raise ContractError(f"GT/evaluator leak in producer value {trail}")


def _relative_asset_path(value: Any, *, root: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}.path must be a non-empty POSIX-relative path")
    if "\\" in value:
        raise ContractError(f"{label}.path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != root:
        raise ContractError(f"{label}.path must remain under {root}/")
    return path


def _validate_asset(
    asset: Any,
    *,
    expected_role: str,
    root: str,
    label: str,
    data_root: Path | None,
    verify_hashes: bool,
) -> dict[str, Any]:
    if not isinstance(asset, dict):
        raise ContractError(f"{label} must be an asset object")
    if asset.get("role") != expected_role:
        raise ContractError(f"{label}.role must be {expected_role}")
    relative = _relative_asset_path(asset.get("path"), root=root, label=label)
    if not _is_sha256(asset.get("sha256")):
        raise ContractError(f"{label}.sha256 must be a lowercase SHA-256")
    if data_root is not None:
        resolved_root = data_root.resolve()
        candidate = (resolved_root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ContractError(f"{label}.path escapes data root") from exc
        if not candidate.is_file():
            raise ContractError(f"Missing asset: {relative.as_posix()}")
        if verify_hashes and sha256_file(candidate) != asset["sha256"]:
            raise ContractError(f"Asset SHA-256 mismatch: {relative.as_posix()}")
    return asset


def _sample_key(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    key = sample.get("key")
    if not isinstance(key, dict):
        raise ContractError("Each sample requires a key object")
    values: list[int] = []
    for name in ("scene_id", "image_id", "object_id"):
        value = key.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractError(f"sample.key.{name} must be a non-negative integer")
        values.append(value)
    return values[0], values[1], values[2]


def _validate_units_and_coordinates(manifest: Mapping[str, Any]) -> None:
    units = manifest.get("units")
    expected_units = {
        "pose_translation": "m",
        "depth_storage": "mm",
        "cad_vertices": "m",
        "metric_translation": "mm",
        "rotation_error": "degrees",
    }
    if not isinstance(units, dict):
        raise ContractError("units must be an object")
    for name, expected in expected_units.items():
        if units.get(name) != expected:
            raise ContractError(f"units.{name} must be {expected}")
    scale = units.get("depth_scale_to_m")
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ContractError("units.depth_scale_to_m must be finite and positive")
    coordinates = manifest.get("coordinate_conventions")
    expected_coordinates = {
        "pose_direction": "model_to_camera",
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "matrix_layout": "row_major_4x4",
        "rotation_handedness": "right_handed",
    }
    if not isinstance(coordinates, dict):
        raise ContractError("coordinate_conventions must be an object")
    for name, expected in expected_coordinates.items():
        if coordinates.get(name) != expected:
            raise ContractError(f"coordinate_conventions.{name} must be {expected}")
    if coordinates.get("homogeneous_last_row") != [0.0, 0.0, 0.0, 1.0]:
        raise ContractError("coordinate_conventions.homogeneous_last_row is incomplete")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate the unified evaluator-owned manifest without executing any model."""
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL_ID
    ):
        raise ContractError("Manifest identity does not match PREP v1")
    if (
        manifest.get("role") != "DEVELOPMENT_ONLY"
        or manifest.get("accuracy_claim_permitted") is not False
    ):
        raise ContractError(
            "Manifest must remain DEVELOPMENT_ONLY with accuracy claims disabled"
        )
    if manifest.get("auto_deploy") is not False:
        raise ContractError("AUTO_DEPLOY must remain false")
    if manifest.get("input_kind") not in {
        "SYNTHETIC_COMMITTED_FIXTURE",
        "DEVELOPMENT_DATA",
    }:
        raise ContractError("input_kind must be synthetic fixture or development data")
    _validate_units_and_coordinates(manifest)
    roots = manifest.get("roots")
    if roots != {"producer": "producer", "evaluator_only": "evaluator_only"}:
        raise ContractError(
            "Producer and evaluator-only roots must be exact and disjoint"
        )
    if manifest.get("mask_variants") != list(EVALUATOR_VARIANTS):
        raise ContractError("mask_variants must match the frozen order")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("Manifest requires at least one sample")
    seen: set[tuple[int, int, int]] = set()
    asset_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ContractError(f"samples[{index}] must be an object")
        key = _sample_key(sample)
        if key in seen:
            raise ContractError(f"Duplicate scene-image-object key: {key}")
        seen.add(key)
        producer = sample.get("producer_inputs")
        if not isinstance(producer, dict):
            raise ContractError(f"samples[{index}].producer_inputs must be an object")
        _assert_no_producer_leak(producer, f"samples[{index}].producer_inputs")
        for name in REQUIRED_PRODUCER_ASSETS:
            _validate_asset(
                producer.get(name),
                expected_role="producer",
                root=roots["producer"],
                label=f"samples[{index}].producer_inputs.{name}",
                data_root=data_root,
                verify_hashes=verify_hashes,
            )
            asset_count += 1
        masks = producer.get("masks")
        if not isinstance(masks, dict) or set(masks) != set(PRODUCER_VARIANTS):
            raise ContractError(
                f"samples[{index}] requires exactly the three producer mask variants"
            )
        for name in PRODUCER_VARIANTS:
            _validate_asset(
                masks[name],
                expected_role="producer",
                root=roots["producer"],
                label=f"samples[{index}].producer_inputs.masks.{name}",
                data_root=data_root,
                verify_hashes=verify_hashes,
            )
            asset_count += 1
        evaluator = sample.get("evaluator_only")
        if not isinstance(evaluator, dict):
            raise ContractError(f"samples[{index}].evaluator_only must be an object")
        _validate_asset(
            evaluator.get("gt_pose"),
            expected_role="evaluator_only",
            root=roots["evaluator_only"],
            label=f"samples[{index}].evaluator_only.gt_pose",
            data_root=data_root,
            verify_hashes=verify_hashes,
        )
        asset_count += 1
        evaluator_masks = evaluator.get("masks")
        if not isinstance(evaluator_masks, dict) or set(evaluator_masks) != {
            "official_known_sample_sanity",
            "oracle_mask_control",
        }:
            raise ContractError(f"samples[{index}] evaluator masks are incomplete")
        for name in ("official_known_sample_sanity", "oracle_mask_control"):
            _validate_asset(
                evaluator_masks[name],
                expected_role="evaluator_only",
                root=roots["evaluator_only"],
                label=f"samples[{index}].evaluator_only.masks.{name}",
                data_root=data_root,
                verify_hashes=verify_hashes,
            )
            asset_count += 1
        if evaluator.get("oracle_deployment_conclusion_permitted") is not False:
            raise ContractError(
                "Oracle mask is diagnostic-only and cannot support deployment conclusions"
            )
        if not isinstance(sample.get("symmetric_object"), bool):
            raise ContractError(f"samples[{index}].symmetric_object must be boolean")
    return {
        "schema_version": "poseloop.pose-accuracy-recovery.validation.v1",
        "status": "valid",
        "sample_count": len(samples),
        "unique_key_count": len(seen),
        "asset_reference_count": asset_count,
        "verified_hashes": bool(data_root is not None and verify_hashes),
        "producer_variant_count": len(PRODUCER_VARIANTS),
        "evaluator_variant_count": len(EVALUATOR_VARIANTS),
        "gt_leak_count": 0,
        "accuracy_claim_permitted": False,
    }


def load_and_validate_manifest(
    manifest_path: Path,
    *,
    data_root: Path | None = None,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ContractError("Manifest must be a JSON object")
    return manifest, validate_manifest(
        manifest, data_root=data_root, verify_hashes=verify_hashes
    )


def build_run_plan(
    manifest: Mapping[str, Any],
    *,
    namespace_role: str,
    variants: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic rows while keeping producer plans free of evaluator paths."""
    if namespace_role not in {"producer", "evaluator-only"}:
        raise ContractError("namespace_role must be producer or evaluator-only")
    allowed = PRODUCER_VARIANTS if namespace_role == "producer" else EVALUATOR_VARIANTS
    selected = tuple(variants) if variants is not None else allowed
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(variant not in allowed for variant in selected)
    ):
        raise ContractError(f"Invalid mask variants for {namespace_role}: {selected}")
    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        producer = sample["producer_inputs"]
        evaluator = sample["evaluator_only"]
        for variant in selected:
            mask = (
                producer["masks"][variant]
                if variant in PRODUCER_VARIANTS
                else evaluator["masks"][variant]
            )
            row: dict[str, Any] = {
                "scene_id": sample["key"]["scene_id"],
                "image_id": sample["key"]["image_id"],
                "object_id": sample["key"]["object_id"],
                "mask_variant": variant,
                "mask": mask,
                "rgb": producer["rgb"],
                "depth": producer["depth"],
                "camera": producer["camera"],
                "cad": producer["cad"],
                "namespace_role": namespace_role,
            }
            if namespace_role == "evaluator-only":
                row["gt_pose"] = evaluator["gt_pose"]
                row["symmetric_object"] = sample["symmetric_object"]
                row["oracle_diagnostic_only"] = variant in {
                    "official_known_sample_sanity",
                    "oracle_mask_control",
                }
            rows.append(row)
    if namespace_role == "producer":
        _assert_no_producer_leak(rows, "producer_plan")
    return rows


def export_producer_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create the only manifest permitted to leave GPU-A for a label-free producer."""
    samples = []
    for sample in manifest["samples"]:
        samples.append(
            {"key": sample["key"], "producer_inputs": sample["producer_inputs"]}
        )
    output = {
        "schema_version": PRODUCER_MANIFEST_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "role": "LABEL_FREE_PRODUCER_ONLY",
        "auto_deploy": False,
        "root": manifest["roots"]["producer"],
        "units": {
            "pose_translation": manifest["units"]["pose_translation"],
            "depth_storage": manifest["units"]["depth_storage"],
            "depth_scale_to_m": manifest["units"]["depth_scale_to_m"],
            "cad_vertices": manifest["units"]["cad_vertices"],
        },
        "coordinate_conventions": manifest["coordinate_conventions"],
        "mask_variants": list(PRODUCER_VARIANTS),
        "samples": samples,
        "label_access_count": 0,
        "accuracy_claim_permitted": False,
    }
    _assert_no_producer_leak(output, "producer_manifest")
    output["lock_sha256"] = canonical_sha256(output)
    return output


def validate_producer_manifest(
    manifest: Mapping[str, Any],
    *,
    data_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if manifest.get("schema_version") != PRODUCER_MANIFEST_SCHEMA:
        raise ContractError("Producer manifest schema mismatch")
    if (
        manifest.get("role") != "LABEL_FREE_PRODUCER_ONLY"
        or manifest.get("label_access_count") != 0
    ):
        raise ContractError("Producer manifest role boundary changed")
    expected_lock = manifest.get("lock_sha256")
    unlocked = dict(manifest)
    unlocked.pop("lock_sha256", None)
    if expected_lock != canonical_sha256(unlocked):
        raise ContractError("Producer manifest lock mismatch")
    _assert_no_producer_leak(unlocked, "producer_manifest")
    if (
        manifest.get("auto_deploy") is not False
        or manifest.get("accuracy_claim_permitted") is not False
    ):
        raise ContractError("Producer manifest must remain inert and claim-free")
    if manifest.get("root") != "producer":
        raise ContractError("Producer manifest root must be producer/")
    units = manifest.get("units")
    if not isinstance(units, dict) or set(units) != {
        "pose_translation",
        "depth_storage",
        "depth_scale_to_m",
        "cad_vertices",
    }:
        raise ContractError("Producer units are incomplete or changed")
    if (
        units.get("pose_translation") != "m"
        or units.get("depth_storage") != "mm"
        or units.get("cad_vertices") != "m"
    ):
        raise ContractError("Producer units are incomplete or changed")
    depth_scale = units.get("depth_scale_to_m")
    if (
        not isinstance(depth_scale, (int, float))
        or isinstance(depth_scale, bool)
        or not math.isfinite(depth_scale)
        or depth_scale <= 0
    ):
        raise ContractError("Producer depth scale must be finite and positive")
    coordinates = manifest.get("coordinate_conventions")
    if not isinstance(coordinates, dict):
        raise ContractError("Producer coordinate conventions are missing")
    for name, expected in {
        "pose_direction": "model_to_camera",
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "matrix_layout": "row_major_4x4",
        "rotation_handedness": "right_handed",
    }.items():
        if coordinates.get(name) != expected:
            raise ContractError(f"Producer coordinate convention changed: {name}")
    if coordinates.get("homogeneous_last_row") != [0.0, 0.0, 0.0, 1.0]:
        raise ContractError("Producer homogeneous convention is incomplete")
    if manifest.get("mask_variants") != list(PRODUCER_VARIANTS):
        raise ContractError("Producer mask variants changed")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("Producer manifest requires samples")
    seen: set[tuple[int, int, int]] = set()
    asset_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {"key", "producer_inputs"}:
            raise ContractError(f"Producer sample {index} contains unexpected roles")
        key = _sample_key(sample)
        if key in seen:
            raise ContractError(f"Duplicate producer scene-image-object key: {key}")
        seen.add(key)
        producer = sample["producer_inputs"]
        _assert_no_producer_leak(producer, f"producer.samples[{index}]")
        for name in REQUIRED_PRODUCER_ASSETS:
            _validate_asset(
                producer.get(name),
                expected_role="producer",
                root="producer",
                label=f"producer.samples[{index}].{name}",
                data_root=data_root,
                verify_hashes=verify_hashes,
            )
            asset_count += 1
        masks = producer.get("masks")
        if not isinstance(masks, dict) or set(masks) != set(PRODUCER_VARIANTS):
            raise ContractError(f"Producer sample {index} mask variants are incomplete")
        for name in PRODUCER_VARIANTS:
            _validate_asset(
                masks[name],
                expected_role="producer",
                root="producer",
                label=f"producer.samples[{index}].masks.{name}",
                data_root=data_root,
                verify_hashes=verify_hashes,
            )
            asset_count += 1
    return {
        "status": "valid",
        "label_access_count": 0,
        "sample_count": len(samples),
        "unique_key_count": len(seen),
        "asset_reference_count": asset_count,
        "verified_hashes": bool(data_root is not None and verify_hashes),
        "gt_leak_count": 0,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
