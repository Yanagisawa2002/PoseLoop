"""Fail-closed A-R2 camera provenance and derived-runtime-camera contracts.

The ten historical camera JSON files are opened only as opaque byte streams for
size/SHA verification.  Runtime camera JSON is deterministically derived from the
exact frozen JSONL workload's public intrinsics.  No source camera JSON is parsed,
and no source-camera path is admitted to the parent CNOS runtime lock.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import (
    DEPLOYMENT_REQUEST_SCHEMA as V1_DEPLOYMENT_REQUEST_SCHEMA,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import contracts as v1
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 import (
    ROUTE_ID as R1_ROUTE_ID,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 import contracts as r1
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    ALLOWED_INPUT_ROLES,
    validate_protocol,
)

from . import (
    CAMERA_DERIVATION_MANIFEST_SCHEMA,
    CAMERA_DERIVATION_RECEIPT_SCHEMA,
    CAMERA_PROVENANCE_MANIFEST_SCHEMA,
    DEPLOYMENT_REQUEST_SCHEMA,
    DEPLOYMENT_SCHEMA,
    DERIVED_CAMERA_SCHEMA,
    ROUTE_ID,
    ROUTE_SCHEMA,
    RUNTIME_LOCK_SCHEMA,
)

R1_IMPLEMENTATION_COMMIT = "7e15f06bc97c3466aa482b98b807e8f1cc211e2b"
R1_IMPLEMENTATION_TREE = "f57d944cc00acc3c72a13dc58f72d90a28aad3df"
R1_ROUTE_LOCK = "5fe80c54b38ba0d1cc6016207d8c5d69efe654eba056004f9e419779a3f4923a"
R1_ROUTE_RELATIVE_PATH = (
    "protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json"
)
R1_CAMERA_BLOCKER_CLOSEOUT_SHA256 = (
    "998436f319d56db3e3e22df587bd7b79251353f1c7d6d9057e10fdecc610aceb"
)
R1_CAMERA_AUDIT_SHA256 = (
    "b89744505c5a89c1e003139b1a59e5d3721e88759b1a4eec6f3773eca9d02867"
)
WORKLOAD_BYTES = r1.WORKLOAD_BYTES
WORKLOAD_SHA256 = r1.WORKLOAD_SHA256
WORKLOAD_LINE_COUNT = r1.WORKLOAD_LINE_COUNT
SOURCE_CAMERA_BYTES = 418
SOURCE_CAMERA_SHA256 = (
    "bf93b9cb8c2a94515b8cec410b0aa6da60ab637d5dd79354ea80083f63fb8430"
)
TRANSFORM_ID = "poseloop.cnos.public-intrinsics-runtime-camera"
TRANSFORM_VERSION = 1
CAMERA_PROVENANCE_MANIFEST_PATH = (
    "inputs/manifests/camera-provenance-v1r2.json"
)
CAMERA_DERIVATION_MANIFEST_PATH = (
    "inputs/manifests/derived-camera-manifest-v1r2.json"
)
DERIVED_CAMERA_ROOT = "inputs/runtime-camera"
SOURCE_CAMERA_ROOT = "inputs/provenance/camera"
FORBIDDEN_RUNTIME_TOKENS = (
    "depth",
    "depth_component",
    "depth_component_bbox",
    "bbox_mask",
    "legacy_mask",
    "legacy_predicted_mask",
    "oracle",
    "gt",
    "ground_truth",
    "evaluator",
    "sealed",
    "sam_vit_b",
    "sam_vit_h",
    "sam_vit_l",
    "sam6d_pose",
    "sam_6d_pose",
)

BOUNDARY_ZERO = copy.deepcopy(r1.BOUNDARY_ZERO)
SOURCE_PROVENANCE_BOUNDARY = {
    "access_mode": "HASH_ONLY_RAW_BYTES",
    "json_parse_count": 0,
    "field_access_count": 0,
    "runtime_inclusion_count": 0,
}
CAMERA_DERIVATION_BOUNDARY = {
    "source_camera_hash_verification_count": WORKLOAD_LINE_COUNT,
    "source_camera_json_parse_count": 0,
    "source_camera_field_access_count": 0,
    "source_camera_runtime_inclusion_count": 0,
    **BOUNDARY_ZERO,
}


@dataclass(frozen=True)
class CameraDerivationValidation:
    """Disk-verified camera derivation bindings keyed by item id."""

    manifest: dict[str, Any]
    provenance_manifest: dict[str, Any]
    items: dict[str, dict[str, Any]]


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"{label} must be lowercase SHA-256")
    return str(value)


def _require_lock(value: Mapping[str, Any], field: str, label: str) -> str:
    actual = _require_sha256(value.get(field), f"{label}.{field}")
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if actual != expected:
        raise ContractError(f"{label} self-lock mismatch")
    return actual


def _relative(value: Any, label: str, *, required_root: str | None = None) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label} must be a non-empty POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} must remain relative without traversal")
    if required_root is not None and path.parts[0] != required_root:
        raise ContractError(f"{label} must remain under {required_root}/")
    return path


def _resolve(root: Path, relative: PurePosixPath, label: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes deployment root") from exc
    return candidate


def _binding(
    root: Path,
    relative_value: Any,
    *,
    label: str,
    required_root: str | None = None,
) -> dict[str, Any]:
    relative = _relative(relative_value, label, required_root=required_root)
    candidate = _resolve(root, relative, label)
    if not candidate.is_file():
        raise ContractError(f"Missing required file: {relative.as_posix()}")
    size = candidate.stat().st_size
    if size <= 0:
        raise ContractError(f"Required file is empty: {relative.as_posix()}")
    return {
        "relative_path": relative.as_posix(),
        "bytes": size,
        "sha256": sha256_file(candidate),
    }


def _validate_bound_file(
    root: Path,
    value: Any,
    *,
    label: str,
    required_root: str | None = None,
) -> dict[str, Any]:
    asset = _exact(value, {"relative_path", "bytes", "sha256"}, label)
    _require_sha256(asset["sha256"], f"{label}.sha256")
    if (
        not isinstance(asset["bytes"], int)
        or isinstance(asset["bytes"], bool)
        or asset["bytes"] <= 0
    ):
        raise ContractError(f"{label}.bytes must be positive")
    actual = _binding(
        root,
        asset["relative_path"],
        label=label,
        required_root=required_root,
    )
    if actual != dict(asset):
        raise ContractError(f"{label} bytes/SHA differ from disk")
    return actual


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _sample_key(value: Any, label: str) -> dict[str, int]:
    item = _exact(value, {"scene_id", "image_id", "object_id"}, label)
    parsed: dict[str, int] = {}
    for key in ("scene_id", "image_id", "object_id"):
        component = item[key]
        if not isinstance(component, int) or isinstance(component, bool) or component < 0:
            raise ContractError(f"{label}.{key} must be a non-negative integer")
        parsed[key] = component
    return parsed


def _expected_samples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    return [
        {
            "item_id": sample["item_id"],
            "sample_key": {
                key: sample[key] for key in ("scene_id", "image_id", "object_id")
            },
        }
        for sample in protocol["input_lock"]["samples"]
    ]


def _expected_predecessor() -> dict[str, Any]:
    return {
        "implementation_commit": R1_IMPLEMENTATION_COMMIT,
        "implementation_tree": R1_IMPLEMENTATION_TREE,
        "r1_route_id": R1_ROUTE_ID,
        "r1_route_lock_sha256": R1_ROUTE_LOCK,
        "camera_blocker_closeout": {
            "status": "READY_FOR_R2_CODE_FIX",
            "sha256": R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
            "camera_audit_sha256": R1_CAMERA_AUDIT_SHA256,
            "immutable": True,
            "reinterpretation_permitted": False,
        },
    }


def _expected_camera_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_mode": "HASH_ONLY_PUBLIC_PROVENANCE",
        "source_json_parse_permitted": False,
        "source_field_access_permitted": False,
        "source_runtime_inclusion_permitted": False,
        "source_root": SOURCE_CAMERA_ROOT,
        "source_item_count": WORKLOAD_LINE_COUNT,
        "source_items": [
            {
                "item_id": item["item_id"],
                "sample_key": item["sample_key"],
                "bytes": SOURCE_CAMERA_BYTES,
                "sha256": SOURCE_CAMERA_SHA256,
            }
            for item in _expected_samples(protocol)
        ],
        "transform": {
            "id": TRANSFORM_ID,
            "version": TRANSFORM_VERSION,
            "workload_fields": ["camera_intrinsics"],
            "external_pose_included": False,
            "serialization": "utf8-json-indent2-sort-keys-lf",
            "derived_schema": DERIVED_CAMERA_SCHEMA,
            "derived_root": DERIVED_CAMERA_ROOT,
        },
        "provenance_manifest_path": CAMERA_PROVENANCE_MANIFEST_PATH,
        "derivation_manifest_path": CAMERA_DERIVATION_MANIFEST_PATH,
        "forbidden_runtime_tokens": list(FORBIDDEN_RUNTIME_TOKENS),
    }


def _expected_execution_identity() -> dict[str, Any]:
    return {
        "deployment_namespace": "poseloop_ga_cnos_v1r2",
        "screen_session_prefix": "poseloop_ga_cnos_v1r2",
        "output_namespace": "output-v1r2",
        "compatibility_payload_namespace": "compat-v1-payload",
    }


def _validate_dependencies(
    route: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    expected_paths = {
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/__init__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/__main__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/cli.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/contracts.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/producer.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1r1/SERVER_RUNBOOK_CNOS_RUNTIME_PREP_V1R1.md",
        R1_ROUTE_RELATIVE_PATH,
    }
    actual_paths: set[str] = set()
    for index, value in enumerate(route["r1_dependency_files"]):
        item = _exact(value, {"relative_path", "bytes", "sha256"}, f"dependency[{index}]")
        relative = _relative(item["relative_path"], f"dependency[{index}]")
        if relative.as_posix() in actual_paths:
            raise ContractError("CNOS R2 dependency paths must be unique")
        actual = _binding(repository_root, relative.as_posix(), label=f"dependency[{index}]")
        if actual != dict(item):
            raise ContractError(
                f"Frozen R1 dependency bytes changed: {relative.as_posix()}"
            )
        actual_paths.add(relative.as_posix())
    if actual_paths != expected_paths:
        raise ContractError("CNOS R2 frozen R1 dependency inventory changed")
    r1_route = read_json(
        repository_root / Path(*PurePosixPath(R1_ROUTE_RELATIVE_PATH).parts)
    )
    if not isinstance(r1_route, dict):
        raise ContractError("Frozen R1 route must be an object")
    validation = r1.validate_route(r1_route, repository_root=repository_root)
    if (
        validation["route_id"] != R1_ROUTE_ID
        or validation["route_lock_sha256"] != R1_ROUTE_LOCK
    ):
        raise ContractError("CNOS R2 predecessor route identity changed")
    return r1_route


def validate_route(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    _exact(
        route,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "auto_deploy",
            "role",
            "predecessor",
            "camera_contract",
            "execution_identity",
            "boundary",
            "r1_dependency_files",
        },
        "CNOS R2 route",
    )
    validate_protocol(protocol)
    if (
        route["schema_version"] != ROUTE_SCHEMA
        or route["route_id"] != ROUTE_ID
        or route["auto_deploy"] is not False
        or route["role"] != "DEVELOPMENT_ONLY"
    ):
        raise ContractError("CNOS R2 route identity/state mismatch")
    if route["predecessor"] != _expected_predecessor():
        raise ContractError("CNOS R2 predecessor/blocker binding changed")
    if route["camera_contract"] != _expected_camera_contract(protocol):
        raise ContractError("CNOS R2 camera isolation contract changed")
    if route["execution_identity"] != _expected_execution_identity():
        raise ContractError("CNOS R2 execution identity changed")
    if route["boundary"] != BOUNDARY_ZERO:
        raise ContractError("CNOS R2 zero-access boundary changed")
    dependencies = route["r1_dependency_files"]
    if not isinstance(dependencies, list) or len(dependencies) != 7:
        raise ContractError("CNOS R2 requires exactly seven frozen R1 dependencies")
    route_lock = _require_lock(route, "route_lock_sha256", "CNOS R2 route")
    if repository_root is not None:
        _validate_dependencies(route, repository_root.resolve())
    return {
        "status": "valid",
        "route_id": ROUTE_ID,
        "route_lock_sha256": route_lock,
        "predecessor_camera_blocker_closeout_sha256": (
            R1_CAMERA_BLOCKER_CLOSEOUT_SHA256
        ),
        "workload_sha256": WORKLOAD_SHA256,
        "execution_ready": False,
        "weights_downloaded": False,
    }


def _derived_camera_value(row: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = row["camera_intrinsics"]
    if not isinstance(intrinsics, list) or len(intrinsics) != 9:
        raise ContractError("workload camera_intrinsics must contain nine values")
    values = [_finite(value, "workload camera_intrinsics") for value in intrinsics]
    if values[0] <= 0 or values[4] <= 0 or values[6:] != [0.0, 0.0, 1.0]:
        raise ContractError("workload camera_intrinsics are not a valid OpenCV matrix")
    return {
        "schema_version": DERIVED_CAMERA_SCHEMA,
        "item_id": row["item_id"],
        "sample_key": {
            key: row[key] for key in ("scene_id", "image_id", "object_id")
        },
        "frame_size": {"height": 1080, "width": 1440},
        "camera_intrinsics": [values[0:3], values[3:6], values[6:9]],
        "coordinate_convention": "opencv-camera-x-right-y-down-z-forward",
        "derivation": {
            "transform_id": TRANSFORM_ID,
            "transform_version": TRANSFORM_VERSION,
            "source_workload_sha256": WORKLOAD_SHA256,
            "source_camera_sha256": row["camera_sha256"],
        },
    }


def derived_camera_bytes(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _derived_camera_value(row),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_derived_camera_value(
    value: Any,
    row: Mapping[str, Any],
    *,
    forbidden_tokens: Sequence[str] = FORBIDDEN_RUNTIME_TOKENS,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("derived runtime camera must be an object")
    v1._reject_forbidden_json(value, list(forbidden_tokens), "derived runtime camera")
    _exact(
        value,
        {
            "schema_version",
            "item_id",
            "sample_key",
            "frame_size",
            "camera_intrinsics",
            "coordinate_convention",
            "derivation",
        },
        "derived runtime camera",
    )
    _sample_key(value["sample_key"], "derived runtime camera.sample_key")
    _exact(value["frame_size"], {"height", "width"}, "derived runtime camera.frame_size")
    _exact(
        value["derivation"],
        {
            "transform_id",
            "transform_version",
            "source_workload_sha256",
            "source_camera_sha256",
        },
        "derived runtime camera.derivation",
    )
    expected = _derived_camera_value(row)
    if dict(value) != expected:
        raise ContractError("derived runtime camera differs from canonical workload transform")
    return expected


def _expected_provenance_asset(item_id: str) -> str:
    return f"{SOURCE_CAMERA_ROOT}/{item_id}.json"


def _expected_derived_asset(item_id: str) -> str:
    return f"{DERIVED_CAMERA_ROOT}/{item_id}.json"


def validate_camera_provenance_manifest(
    manifest: Mapping[str, Any],
    *,
    deployment_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _exact(
        manifest,
        {
            "schema_version",
            "role",
            "source_workload_sha256",
            "item_count",
            "items",
            "boundary",
            "camera_provenance_manifest_lock_sha256",
        },
        "camera provenance manifest",
    )
    if (
        manifest["schema_version"] != CAMERA_PROVENANCE_MANIFEST_SCHEMA
        or manifest["role"] != "PUBLIC_HASH_ONLY_CAMERA_PROVENANCE"
        or manifest["source_workload_sha256"] != WORKLOAD_SHA256
        or manifest["item_count"] != WORKLOAD_LINE_COUNT
        or manifest["boundary"] != SOURCE_PROVENANCE_BOUNDARY
    ):
        raise ContractError("camera provenance manifest identity/boundary changed")
    expected = _expected_samples(protocol)
    items = manifest["items"]
    if not isinstance(items, list) or len(items) != WORKLOAD_LINE_COUNT:
        raise ContractError("camera provenance manifest requires exact ten items")
    parsed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(items):
        item = _exact(value, {"item_id", "sample_key", "asset"}, f"camera provenance item[{index}]")
        if item["item_id"] != expected[index]["item_id"]:
            raise ContractError(f"camera provenance item/order mismatch at {index}")
        if _sample_key(item["sample_key"], f"camera provenance item[{index}].sample_key") != expected[index]["sample_key"]:
            raise ContractError(f"camera provenance sample swap at {index}")
        asset = _validate_bound_file(
            deployment_root,
            item["asset"],
            label=f"camera provenance {item['item_id']}",
            required_root="inputs",
        )
        expected_relative = _expected_provenance_asset(item["item_id"])
        if asset != {
            "relative_path": expected_relative,
            "bytes": SOURCE_CAMERA_BYTES,
            "sha256": SOURCE_CAMERA_SHA256,
        }:
            raise ContractError(
                f"source camera bytes/SHA differ from frozen workload evidence: {item['item_id']}"
            )
        parsed[item["item_id"]] = {
            "sample_key": expected[index]["sample_key"],
            "asset": asset,
        }
    _require_lock(
        manifest,
        "camera_provenance_manifest_lock_sha256",
        "camera provenance manifest",
    )
    return parsed


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise ContractError(f"Immutable CNOS R2 output already differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"Immutable CNOS R2 output raced with different data: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _path_relative_to_root(path: Path, root: Path, expected: str, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"{label} escapes deployment root") from exc
    if relative != expected:
        raise ContractError(f"{label} must be {expected}")
    return relative


def derive_cameras(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    deployment_root: Path,
    workload_path: Path,
    provenance_manifest_path: Path,
    manifest_output: Path,
    receipt_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash source cameras as opaque bytes and write canonical runtime cameras."""
    validate_route(route, protocol)
    _path_relative_to_root(
        workload_path,
        deployment_root,
        "inputs/manifests/workload.json",
        "workload path",
    )
    _path_relative_to_root(
        provenance_manifest_path,
        deployment_root,
        CAMERA_PROVENANCE_MANIFEST_PATH,
        "camera provenance manifest path",
    )
    _path_relative_to_root(
        manifest_output,
        deployment_root,
        CAMERA_DERIVATION_MANIFEST_PATH,
        "camera derivation manifest output",
    )
    workload = r1.audit_workload_jsonl(workload_path, protocol)
    provenance_value = read_json(provenance_manifest_path)
    if not isinstance(provenance_value, dict):
        raise ContractError("camera provenance manifest must be an object")
    provenance = validate_camera_provenance_manifest(
        provenance_value,
        deployment_root=deployment_root,
        protocol=protocol,
    )
    items: list[dict[str, Any]] = []
    for row in workload.rows:
        item_id = row["item_id"]
        relative = _expected_derived_asset(item_id)
        output = _resolve(
            deployment_root,
            _relative(relative, f"derived camera {item_id}", required_root="inputs"),
            f"derived camera {item_id}",
        )
        payload = derived_camera_bytes(row)
        _write_immutable(output, payload)
        derived_asset = {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    key: row[key] for key in ("scene_id", "image_id", "object_id")
                },
                "source_camera": copy.deepcopy(provenance[item_id]["asset"]),
                "derived_camera": derived_asset,
            }
        )
    workload_asset = _binding(
        deployment_root,
        "inputs/manifests/workload.json",
        label="workload manifest",
        required_root="inputs",
    )
    provenance_asset = _binding(
        deployment_root,
        CAMERA_PROVENANCE_MANIFEST_PATH,
        label="camera provenance manifest",
        required_root="inputs",
    )
    transform = copy.deepcopy(route["camera_contract"]["transform"])
    manifest: dict[str, Any] = {
        "schema_version": CAMERA_DERIVATION_MANIFEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY_DERIVED_RUNTIME_CAMERAS",
        "route_id": ROUTE_ID,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "source_workload": workload_asset,
        "source_camera_provenance_manifest": provenance_asset,
        "transform": transform,
        "item_count": WORKLOAD_LINE_COUNT,
        "items": items,
        "boundary": copy.deepcopy(CAMERA_DERIVATION_BOUNDARY),
        "camera_derivation_manifest_lock_sha256": "pending",
    }
    manifest["camera_derivation_manifest_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "camera_derivation_manifest_lock_sha256"
        }
    )
    _write_immutable(manifest_output, _json_bytes(manifest))
    validation = validate_camera_derivation(
        manifest,
        route=route,
        protocol=protocol,
        deployment_root=deployment_root,
        workload=workload,
    )
    manifest_asset = _binding(
        deployment_root,
        CAMERA_DERIVATION_MANIFEST_PATH,
        label="camera derivation manifest",
        required_root="inputs",
    )
    receipt: dict[str, Any] = {
        "schema_version": CAMERA_DERIVATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "r1_camera_blocker_closeout_sha256": R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
        "source_workload": workload_asset,
        "source_camera_provenance_manifest": provenance_asset,
        "camera_derivation_manifest": manifest_asset,
        "camera_derivation_manifest_lock_sha256": validation.manifest[
            "camera_derivation_manifest_lock_sha256"
        ],
        "transform": transform,
        "item_count": WORKLOAD_LINE_COUNT,
        "deterministic_generation": True,
        "execution_started": False,
        "model_imported": False,
        "boundary": copy.deepcopy(CAMERA_DERIVATION_BOUNDARY),
        "camera_derivation_receipt_lock_sha256": "pending",
    }
    receipt["camera_derivation_receipt_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "camera_derivation_receipt_lock_sha256"
        }
    )
    _write_immutable(receipt_output, _json_bytes(receipt))
    return manifest, receipt


def validate_camera_derivation(
    manifest: Mapping[str, Any],
    *,
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    deployment_root: Path,
    workload: r1.WorkloadAuditResult,
) -> CameraDerivationValidation:
    _exact(
        manifest,
        {
            "schema_version",
            "role",
            "route_id",
            "route_lock_sha256",
            "protocol_id",
            "protocol_lock_sha256",
            "source_workload",
            "source_camera_provenance_manifest",
            "transform",
            "item_count",
            "items",
            "boundary",
            "camera_derivation_manifest_lock_sha256",
        },
        "camera derivation manifest",
    )
    if (
        manifest["schema_version"] != CAMERA_DERIVATION_MANIFEST_SCHEMA
        or manifest["role"] != "DEVELOPMENT_ONLY_DERIVED_RUNTIME_CAMERAS"
        or manifest["route_id"] != ROUTE_ID
        or manifest["route_lock_sha256"] != route["route_lock_sha256"]
        or manifest["protocol_id"] != protocol["protocol_id"]
        or manifest["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or manifest["transform"] != route["camera_contract"]["transform"]
        or manifest["item_count"] != WORKLOAD_LINE_COUNT
        or manifest["boundary"] != CAMERA_DERIVATION_BOUNDARY
    ):
        raise ContractError("camera derivation manifest identity/boundary changed")
    workload_asset = _validate_bound_file(
        deployment_root,
        manifest["source_workload"],
        label="camera derivation source workload",
        required_root="inputs",
    )
    if workload_asset != {
        "relative_path": "inputs/manifests/workload.json",
        "bytes": WORKLOAD_BYTES,
        "sha256": WORKLOAD_SHA256,
    }:
        raise ContractError("camera derivation source workload changed")
    provenance_asset = _validate_bound_file(
        deployment_root,
        manifest["source_camera_provenance_manifest"],
        label="camera provenance manifest",
        required_root="inputs",
    )
    if provenance_asset["relative_path"] != CAMERA_PROVENANCE_MANIFEST_PATH:
        raise ContractError("camera provenance manifest path changed")
    provenance_path = _resolve(
        deployment_root,
        _relative(CAMERA_PROVENANCE_MANIFEST_PATH, "camera provenance manifest"),
        "camera provenance manifest",
    )
    provenance_value = read_json(provenance_path)
    if not isinstance(provenance_value, dict):
        raise ContractError("camera provenance manifest must be an object")
    provenance = validate_camera_provenance_manifest(
        provenance_value,
        deployment_root=deployment_root,
        protocol=protocol,
    )
    rows = {row["item_id"]: row for row in workload.rows}
    expected_order = [row["item_id"] for row in workload.rows]
    values = manifest["items"]
    if not isinstance(values, list) or len(values) != WORKLOAD_LINE_COUNT:
        raise ContractError("camera derivation manifest requires exact ten items")
    parsed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        item = _exact(
            value,
            {"item_id", "sample_key", "source_camera", "derived_camera"},
            f"camera derivation item[{index}]",
        )
        item_id = item["item_id"]
        if item_id != expected_order[index] or item_id in parsed:
            raise ContractError(f"camera derivation item/order mismatch at {index}")
        row = rows[item_id]
        expected_key = {
            key: row[key] for key in ("scene_id", "image_id", "object_id")
        }
        if _sample_key(item["sample_key"], f"camera derivation item[{index}].sample_key") != expected_key:
            raise ContractError(f"camera derivation sample swap at {index}")
        if item["source_camera"] != provenance[item_id]["asset"]:
            raise ContractError(f"camera derivation source camera swap: {item_id}")
        derived = _validate_bound_file(
            deployment_root,
            item["derived_camera"],
            label=f"derived runtime camera {item_id}",
            required_root="inputs",
        )
        if derived["relative_path"] != _expected_derived_asset(item_id):
            raise ContractError(f"derived runtime camera path changed: {item_id}")
        path = _resolve(
            deployment_root,
            _relative(derived["relative_path"], f"derived runtime camera {item_id}"),
            f"derived runtime camera {item_id}",
        )
        expected_payload = derived_camera_bytes(row)
        payload = path.read_bytes()
        if payload != expected_payload:
            raise ContractError(f"derived runtime camera canonical bytes differ: {item_id}")
        try:
            camera_value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot parse derived runtime camera {item_id}: {exc}") from exc
        validate_derived_camera_value(
            camera_value,
            row,
            forbidden_tokens=route["camera_contract"]["forbidden_runtime_tokens"],
        )
        parsed[item_id] = {
            "sample_key": expected_key,
            "source_camera": copy.deepcopy(provenance[item_id]["asset"]),
            "derived_camera": derived,
        }
    _require_lock(
        manifest,
        "camera_derivation_manifest_lock_sha256",
        "camera derivation manifest",
    )
    return CameraDerivationValidation(
        manifest=copy.deepcopy(dict(manifest)),
        provenance_manifest=copy.deepcopy(provenance_value),
        items=parsed,
    )


def _expected_request_paths(parent_route: Mapping[str, Any]) -> dict[str, str]:
    assets = parent_route["required_runtime_assets"]
    return {
        "implementation_source_archive": assets["implementation_source_archive"]["relative_path"],
        "implementation_source_checkout": assets["implementation_source_checkout"]["relative_path"],
        "cnos_source_archive": parent_route["source"]["archive"]["relative_path"],
        "cnos_source_checkout": parent_route["source"]["checkout_relative_path"],
        "fastsam_checkpoint": assets["fastsam_checkpoint"]["relative_path"],
        "dinov2_source_archive": assets["dinov2_source_archive"]["relative_path"],
        "dinov2_source_checkout": assets["dinov2_source_checkout"]["relative_path"],
        "dinov2_checkpoint": assets["dinov2_checkpoint"]["relative_path"],
        "adapter_config": "config/cnos_adapter_config_v1.json",
        "source_manifest": "inputs/manifests/source.json",
        "workload_manifest": "inputs/manifests/workload.json",
        "render_manifest": assets["render_manifest"]["relative_path"],
        "camera_provenance_manifest": CAMERA_PROVENANCE_MANIFEST_PATH,
        "camera_derivation_manifest": CAMERA_DERIVATION_MANIFEST_PATH,
    }


def _request_for_v1(
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_route: Mapping[str, Any],
    cameras: CameraDerivationValidation,
) -> dict[str, Any]:
    _exact(
        request,
        {
            "schema_version",
            "role",
            "route_lock_sha256",
            "paths",
            "implementation",
            "input_items",
            "runtime",
            "boundary",
        },
        "CNOS R2 deployment request",
    )
    if (
        request["schema_version"] != DEPLOYMENT_REQUEST_SCHEMA
        or request["role"] != "DEVELOPMENT_ONLY"
        or request["route_lock_sha256"] != route["route_lock_sha256"]
        or request["boundary"] != v1.DEPLOYMENT_BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R2 deployment request identity/boundary mismatch")
    paths = _exact(request["paths"], set(_expected_request_paths(parent_route)), "CNOS R2 request paths")
    if dict(paths) != _expected_request_paths(parent_route):
        raise ContractError("CNOS R2 deployment paths differ from frozen route")
    implementation = _exact(
        request["implementation"],
        {"approved_commit", "approved_tree"},
        "CNOS R2 implementation",
    )
    for field in ("approved_commit", "approved_tree"):
        value = implementation[field]
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"CNOS R2 {field} must be lowercase Git identity")
    if implementation["approved_commit"] == R1_IMPLEMENTATION_COMMIT:
        raise ContractError("CNOS R2 implementation must be a new reviewed commit")
    runtime = _exact(
        request["runtime"],
        {"device", "proposal_chunk_size", "minimum_proposal_chunk_size"},
        "CNOS R2 runtime",
    )
    if runtime != {
        "device": "cuda:0",
        "proposal_chunk_size": 16,
        "minimum_proposal_chunk_size": 1,
    }:
        raise ContractError("CNOS R2 runtime settings changed")
    expected = {item["item_id"]: item["sample_key"] for item in _expected_samples(protocol)}
    items = request["input_items"]
    if not isinstance(items, list) or len(items) != WORKLOAD_LINE_COUNT:
        raise ContractError("CNOS R2 request requires exact ten input items")
    seen: set[str] = set()
    for index, value in enumerate(items):
        item = _exact(
            value,
            {"item_id", "sample_key", "rgb_path", "camera_path", "cad_path", "descriptor_path"},
            f"CNOS R2 request item[{index}]",
        )
        item_id = item["item_id"]
        if item_id not in expected or item_id in seen:
            raise ContractError(f"unexpected or duplicate CNOS R2 item: {item_id}")
        if _sample_key(item["sample_key"], f"CNOS R2 item {item_id}.sample_key") != expected[item_id]:
            raise ContractError(f"CNOS R2 item sample swap: {item_id}")
        for name in ("rgb_path", "camera_path", "cad_path", "descriptor_path"):
            relative = v1._relative(item[name], f"CNOS R2 item {item_id}.{name}", required_root="inputs")
            v1._reject_forbidden_path(
                relative,
                parent_route["forbidden_inputs"],
                f"CNOS R2 item {item_id}.{name}",
            )
        if item["camera_path"] != cameras.items[item_id]["derived_camera"]["relative_path"]:
            raise ContractError(f"CNOS R2 runtime camera is not the derived asset: {item_id}")
        if item["camera_path"] == cameras.items[item_id]["source_camera"]["relative_path"]:
            raise ContractError(f"CNOS R2 source camera entered runtime: {item_id}")
        seen.add(item_id)
    if seen != set(expected):
        raise ContractError("CNOS R2 request item coverage changed")
    converted = copy.deepcopy(dict(request))
    converted["schema_version"] = V1_DEPLOYMENT_REQUEST_SCHEMA
    converted["route_lock_sha256"] = parent_route["route_lock_sha256"]
    converted["paths"].pop("camera_provenance_manifest")
    converted["paths"].pop("camera_derivation_manifest")
    return converted


def _workload_and_camera_manifest(
    deployment_root: Path,
    request: Mapping[str, Any],
    protocol: Mapping[str, Any],
    route: Mapping[str, Any],
) -> tuple[r1.WorkloadAuditResult, CameraDerivationValidation]:
    paths = request.get("paths")
    if not isinstance(paths, Mapping):
        raise ContractError("CNOS R2 request paths must be an object")
    if paths.get("workload_manifest") != "inputs/manifests/workload.json":
        raise ContractError("CNOS R2 workload path changed")
    if paths.get("camera_provenance_manifest") != CAMERA_PROVENANCE_MANIFEST_PATH:
        raise ContractError("CNOS R2 camera provenance path changed")
    if paths.get("camera_derivation_manifest") != CAMERA_DERIVATION_MANIFEST_PATH:
        raise ContractError("CNOS R2 camera derivation path changed")
    workload_path = _resolve(
        deployment_root,
        _relative("inputs/manifests/workload.json", "CNOS R2 workload", required_root="inputs"),
        "CNOS R2 workload",
    )
    workload = r1.audit_workload_jsonl(workload_path, protocol)
    manifest_path = _resolve(
        deployment_root,
        _relative(CAMERA_DERIVATION_MANIFEST_PATH, "camera derivation manifest", required_root="inputs"),
        "camera derivation manifest",
    )
    value = read_json(manifest_path)
    if not isinstance(value, dict):
        raise ContractError("camera derivation manifest must be an object")
    cameras = validate_camera_derivation(
        value,
        route=route,
        protocol=protocol,
        deployment_root=deployment_root,
        workload=workload,
    )
    return workload, cameras


def _validate_runtime_bindings(
    parent_runtime: Mapping[str, Any],
    workload: r1.WorkloadAuditResult,
    cameras: CameraDerivationValidation,
) -> None:
    rows = {row["item_id"]: row for row in workload.rows}
    provenance_paths = {
        item["source_camera"]["relative_path"] for item in cameras.items.values()
    }
    runtime_serialized = json.dumps(parent_runtime, sort_keys=True)
    leaked = sorted(path for path in provenance_paths if path in runtime_serialized)
    if leaked:
        raise ContractError(f"source camera provenance entered runtime: {leaked}")
    for item in parent_runtime["data"]["items"]:
        item_id = item["item_id"]
        inputs = item["inputs"]
        if set(inputs) != set(ALLOWED_INPUT_ROLES):
            raise ContractError(f"CNOS R2 runtime input roles changed: {item_id}")
        row = rows[item_id]
        if inputs["rgb"]["sha256"] != row["rgb_sha256"]:
            raise ContractError(f"CNOS R2 runtime RGB hash changed: {item_id}")
        if inputs["cad"]["sha256"] != row["model_sha256"]:
            raise ContractError(f"CNOS R2 runtime CAD hash changed: {item_id}")
        expected_camera = cameras.items[item_id]["derived_camera"]
        if inputs["camera"] != expected_camera:
            raise ContractError(f"CNOS R2 runtime camera binding changed: {item_id}")
        if inputs["camera"]["sha256"] == row["camera_sha256"]:
            raise ContractError(f"CNOS R2 runtime reused source camera hash: {item_id}")


def _asset_for_relative(root: Path, relative: str, label: str) -> dict[str, Any]:
    return _binding(root, relative, label=label, required_root="inputs")


def _runtime_lock(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_runtime: Mapping[str, Any],
    workload: r1.WorkloadAuditResult,
    cameras: CameraDerivationValidation,
    deployment_root: Path,
) -> dict[str, Any]:
    provenance_manifest = _asset_for_relative(
        deployment_root,
        CAMERA_PROVENANCE_MANIFEST_PATH,
        "camera provenance manifest",
    )
    derivation_manifest = _asset_for_relative(
        deployment_root,
        CAMERA_DERIVATION_MANIFEST_PATH,
        "camera derivation manifest",
    )
    value: dict[str, Any] = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "route_id": ROUTE_ID,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "runtime_lock_sha256": "pending",
        "state": "RGB_DERIVED_CAMERA_CAD_DESCRIPTORS_ONLY",
        "allowed_runtime_input_roles": list(ALLOWED_INPUT_ROLES),
        "parent_runtime_lock": copy.deepcopy(dict(parent_runtime)),
        "workload_provenance_audit": copy.deepcopy(workload.audit),
        "camera_provenance_audit": {
            "mode": "HASH_ONLY_RAW_BYTES",
            "manifest": provenance_manifest,
            "manifest_lock_sha256": cameras.provenance_manifest[
                "camera_provenance_manifest_lock_sha256"
            ],
            "item_count": WORKLOAD_LINE_COUNT,
            "hash_verification_count": WORKLOAD_LINE_COUNT,
            "json_parse_count": 0,
            "field_access_count": 0,
            "runtime_inclusion_count": 0,
        },
        "camera_derivation": {
            "manifest": derivation_manifest,
            "manifest_lock_sha256": cameras.manifest[
                "camera_derivation_manifest_lock_sha256"
            ],
            "transform": copy.deepcopy(route["camera_contract"]["transform"]),
            "item_count": WORKLOAD_LINE_COUNT,
        },
        "boundary": copy.deepcopy(BOUNDARY_ZERO),
    }
    value["runtime_lock_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "runtime_lock_sha256"}
    )
    return value


def _deployment(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    workload: r1.WorkloadAuditResult,
    cameras: CameraDerivationValidation,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "route_id": ROUTE_ID,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "deployment_lock_sha256": "pending",
        "role": "DEVELOPMENT_ONLY_CNOS_PRODUCER_V1R2",
        "state": "EXECUTION_ASSETS_HASHED_NOT_EXECUTED",
        "r1_camera_blocker_closeout_sha256": R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
        "execution_identity": copy.deepcopy(route["execution_identity"]),
        "parent_deployment": copy.deepcopy(dict(parent_deployment)),
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "workload_audit_lock_sha256": workload.audit["workload_audit_lock_sha256"],
        "camera_derivation_manifest_lock_sha256": cameras.manifest[
            "camera_derivation_manifest_lock_sha256"
        ],
        "boundary": copy.deepcopy(BOUNDARY_ZERO),
    }
    value["deployment_lock_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "deployment_lock_sha256"}
    )
    return value


def freeze_deployment(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    deployment_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_route(route, protocol)
    paths = request.get("paths")
    if not isinstance(paths, Mapping):
        raise ContractError("CNOS R2 request paths must be an object")
    implementation_relative = _relative(
        paths.get("implementation_source_checkout"),
        "CNOS R2 implementation checkout",
    )
    implementation_root = _resolve(
        deployment_root,
        implementation_relative,
        "CNOS R2 implementation checkout",
    )
    r1_route = _validate_dependencies(route, implementation_root)
    parent_route = read_json(
        implementation_root
        / Path(*PurePosixPath(r1.V1_ROUTE_RELATIVE_PATH).parts)
    )
    if not isinstance(parent_route, dict):
        raise ContractError("CNOS R2 frozen v1 route must be an object")
    workload, cameras = _workload_and_camera_manifest(
        deployment_root, request, protocol, route
    )
    parent_request = _request_for_v1(
        request, route, protocol, parent_route, cameras
    )
    implementation = parent_request["implementation"]
    if (
        implementation["approved_commit"] == R1_IMPLEMENTATION_COMMIT
        or not v1.git_is_ancestor(
            implementation_root,
            R1_IMPLEMENTATION_COMMIT,
            implementation["approved_commit"],
        )
    ):
        raise ContractError("CNOS R2 implementation must descend from clean R1")
    workload_path = _resolve(
        deployment_root,
        _relative("inputs/manifests/workload.json", "CNOS R2 workload"),
        "CNOS R2 workload",
    )
    with r1.v1_jsonl_compatibility(workload_path, workload):
        parent_deployment, parent_runtime = v1.freeze_deployment(
            parent_route,
            protocol,
            parent_request,
            deployment_root=deployment_root,
        )
    _validate_runtime_bindings(parent_runtime, workload, cameras)
    runtime_lock = _runtime_lock(
        route,
        protocol,
        parent_runtime,
        workload,
        cameras,
        deployment_root,
    )
    deployment = _deployment(
        route,
        protocol,
        parent_deployment,
        runtime_lock,
        workload,
        cameras,
    )
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    del r1_route  # Its validation is the transitive immutable-dependency gate.
    return deployment, runtime_lock


def validate_deployment(
    deployment: Mapping[str, Any],
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    *,
    deployment_root: Path,
) -> dict[str, Any]:
    validate_route(route, protocol)
    _exact(
        deployment,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "protocol_id",
            "protocol_lock_sha256",
            "deployment_lock_sha256",
            "role",
            "state",
            "r1_camera_blocker_closeout_sha256",
            "execution_identity",
            "parent_deployment",
            "runtime_lock_sha256",
            "workload_audit_lock_sha256",
            "camera_derivation_manifest_lock_sha256",
            "boundary",
        },
        "CNOS R2 deployment",
    )
    _exact(
        runtime_lock,
        {
            "schema_version",
            "route_id",
            "route_lock_sha256",
            "protocol_id",
            "protocol_lock_sha256",
            "runtime_lock_sha256",
            "state",
            "allowed_runtime_input_roles",
            "parent_runtime_lock",
            "workload_provenance_audit",
            "camera_provenance_audit",
            "camera_derivation",
            "boundary",
        },
        "CNOS R2 runtime lock",
    )
    if (
        deployment["schema_version"] != DEPLOYMENT_SCHEMA
        or deployment["route_id"] != ROUTE_ID
        or deployment["route_lock_sha256"] != route["route_lock_sha256"]
        or deployment["protocol_id"] != protocol["protocol_id"]
        or deployment["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or deployment["role"] != "DEVELOPMENT_ONLY_CNOS_PRODUCER_V1R2"
        or deployment["state"] != "EXECUTION_ASSETS_HASHED_NOT_EXECUTED"
        or deployment["r1_camera_blocker_closeout_sha256"]
        != R1_CAMERA_BLOCKER_CLOSEOUT_SHA256
        or deployment["execution_identity"] != _expected_execution_identity()
        or deployment["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R2 deployment identity/state changed")
    if (
        runtime_lock["schema_version"] != RUNTIME_LOCK_SCHEMA
        or runtime_lock["route_id"] != ROUTE_ID
        or runtime_lock["route_lock_sha256"] != route["route_lock_sha256"]
        or runtime_lock["protocol_id"] != protocol["protocol_id"]
        or runtime_lock["protocol_lock_sha256"] != protocol["protocol_lock_sha256"]
        or runtime_lock["state"] != "RGB_DERIVED_CAMERA_CAD_DESCRIPTORS_ONLY"
        or runtime_lock["allowed_runtime_input_roles"] != list(ALLOWED_INPUT_ROLES)
        or runtime_lock["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R2 runtime identity/roles changed")
    runtime_sha = _require_lock(runtime_lock, "runtime_lock_sha256", "CNOS R2 runtime lock")
    deployment_sha = _require_lock(
        deployment, "deployment_lock_sha256", "CNOS R2 deployment"
    )
    if deployment["runtime_lock_sha256"] != runtime_sha:
        raise ContractError("CNOS R2 deployment/runtime locks diverged")
    parent_deployment = deployment["parent_deployment"]
    parent_runtime = runtime_lock["parent_runtime_lock"]
    implementation_root = _resolve(
        deployment_root,
        _relative(
            parent_deployment["implementation"]["checkout_relative_path"],
            "CNOS R2 implementation checkout",
        ),
        "CNOS R2 implementation checkout",
    )
    _validate_dependencies(route, implementation_root)
    parent_route = read_json(
        implementation_root
        / Path(*PurePosixPath(r1.V1_ROUTE_RELATIVE_PATH).parts)
    )
    if not isinstance(parent_route, dict):
        raise ContractError("CNOS R2 parent route must be an object")
    workload_path = _resolve(
        deployment_root,
        _relative("inputs/manifests/workload.json", "CNOS R2 workload"),
        "CNOS R2 workload",
    )
    workload = r1.audit_workload_jsonl(workload_path, protocol)
    camera_manifest_path = _resolve(
        deployment_root,
        _relative(CAMERA_DERIVATION_MANIFEST_PATH, "camera derivation manifest"),
        "camera derivation manifest",
    )
    camera_value = read_json(camera_manifest_path)
    if not isinstance(camera_value, dict):
        raise ContractError("camera derivation manifest must be an object")
    cameras = validate_camera_derivation(
        camera_value,
        route=route,
        protocol=protocol,
        deployment_root=deployment_root,
        workload=workload,
    )
    with r1.v1_jsonl_compatibility(workload_path, workload):
        parent_validation = v1.validate_deployment(
            parent_deployment,
            parent_route,
            protocol,
            parent_runtime,
            deployment_root=deployment_root,
        )
    _validate_runtime_bindings(parent_runtime, workload, cameras)
    if (
        deployment["workload_audit_lock_sha256"]
        != workload.audit["workload_audit_lock_sha256"]
        or runtime_lock["workload_provenance_audit"] != workload.audit
        or deployment["camera_derivation_manifest_lock_sha256"]
        != cameras.manifest["camera_derivation_manifest_lock_sha256"]
    ):
        raise ContractError("CNOS R2 workload/camera derivation locks changed")
    provenance_audit = runtime_lock["camera_provenance_audit"]
    if provenance_audit != {
        "mode": "HASH_ONLY_RAW_BYTES",
        "manifest": _asset_for_relative(
            deployment_root,
            CAMERA_PROVENANCE_MANIFEST_PATH,
            "camera provenance manifest",
        ),
        "manifest_lock_sha256": cameras.provenance_manifest[
            "camera_provenance_manifest_lock_sha256"
        ],
        "item_count": WORKLOAD_LINE_COUNT,
        "hash_verification_count": WORKLOAD_LINE_COUNT,
        "json_parse_count": 0,
        "field_access_count": 0,
        "runtime_inclusion_count": 0,
    }:
        raise ContractError("CNOS R2 camera provenance audit changed")
    derivation = runtime_lock["camera_derivation"]
    if derivation != {
        "manifest": _asset_for_relative(
            deployment_root,
            CAMERA_DERIVATION_MANIFEST_PATH,
            "camera derivation manifest",
        ),
        "manifest_lock_sha256": cameras.manifest[
            "camera_derivation_manifest_lock_sha256"
        ],
        "transform": route["camera_contract"]["transform"],
        "item_count": WORKLOAD_LINE_COUNT,
    }:
        raise ContractError("CNOS R2 camera derivation audit changed")
    if (
        parent_deployment["implementation"]["commit"] == R1_IMPLEMENTATION_COMMIT
        or not v1.git_is_ancestor(
            implementation_root,
            R1_IMPLEMENTATION_COMMIT,
            parent_deployment["implementation"]["commit"],
        )
    ):
        raise ContractError("Frozen CNOS R2 implementation ancestry changed")
    return {
        "status": "valid",
        "execution_assets_ready": True,
        "deployment_lock_sha256": deployment_sha,
        "runtime_lock_sha256": runtime_sha,
        "parent_deployment_lock_sha256": parent_validation[
            "deployment_lock_sha256"
        ],
        "parent_runtime_lock_sha256": parent_validation["runtime_lock_sha256"],
        "workload_audit_lock_sha256": workload.audit["workload_audit_lock_sha256"],
        "camera_derivation_manifest_lock_sha256": cameras.manifest[
            "camera_derivation_manifest_lock_sha256"
        ],
        "item_count": WORKLOAD_LINE_COUNT,
        "runtime_input_roles": list(ALLOWED_INPUT_ROLES),
        "runtime_camera_mode": "DERIVED_PUBLIC_INTRINSICS_ONLY",
        "source_camera_json_parse_count": 0,
        "source_camera_runtime_inclusion_count": 0,
        "label_access_count": 0,
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value
