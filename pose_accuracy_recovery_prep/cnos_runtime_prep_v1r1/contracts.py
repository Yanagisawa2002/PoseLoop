"""Fail-closed R1 bridge for the exact historical JSONL workload.

The predecessor workload is provenance metadata. Only the workload file itself is
opened. Values which name depth or legacy segmentation artifacts are validated and
hashed in memory but are never resolved, opened, copied, or inserted into runtime
inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import (
    DEPLOYMENT_REQUEST_SCHEMA as V1_DEPLOYMENT_REQUEST_SCHEMA,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import (
    ROUTE_ID as V1_ROUTE_ID,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import contracts as v1
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import contracts as proposal_contracts
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    ALLOWED_INPUT_ROLES,
    validate_protocol,
)

from . import (
    DEPLOYMENT_REQUEST_SCHEMA,
    DEPLOYMENT_SCHEMA,
    ROUTE_ID,
    ROUTE_SCHEMA,
    RUNTIME_LOCK_SCHEMA,
    WORKLOAD_AUDIT_SCHEMA,
)

PREDECESSOR_IMPLEMENTATION_COMMIT = "0389cb3ac3f0b7854e9ca7c57eedda345aaa2fd6"
PREDECESSOR_IMPLEMENTATION_TREE = "e40a12ee4cbfc41052f4547ea1c2d90b9d46e021"
PREDECESSOR_ROUTE_LOCK = "f369893e4b4cb6e86ef92e6cf0a5e2e3711ba6fec1451d6c750f98fec84f21d6"
PREDECESSOR_BLOCKER_RECEIPT_SHA256 = (
    "68b112abe9c654440a1306749d93aa4e3bd27951b76f541ee81d45b85d856a67"
)
PREDECESSOR_BLOCKED_ROOT = (
    "/root/autodl-tmp/poseloop_ga_cnos_0389cb3_20260818_blocked_v1"
)
WORKLOAD_BYTES = 10_864
WORKLOAD_SHA256 = "dd9f9c4ce9b9ca380614064f58e332661aee9ce63c9602918d52ae391267dfb3"
WORKLOAD_LINE_COUNT = 10
V1_ROUTE_RELATIVE_PATH = (
    "protocols/poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json"
)

WORKLOAD_ROW_FIELDS = (
    "camera_intrinsics",
    "camera_relative_path",
    "camera_sha256",
    "camera_world_to_camera_pose_m",
    "category_id",
    "coco_prediction_sha256",
    "coco_score",
    "depth_relative_path",
    "depth_scale",
    "depth_sha256",
    "detection_index",
    "image_id",
    "item_id",
    "model_relative_path",
    "model_sha256",
    "object_id",
    "rgb_relative_path",
    "rgb_sha256",
    "scene_id",
    "segmentation_sha256",
    "sensor",
)
DECLARED_ONLY_FIELDS = (
    "camera_intrinsics",
    "camera_world_to_camera_pose_m",
    "category_id",
    "coco_prediction_sha256",
    "coco_score",
    "depth_relative_path",
    "depth_scale",
    "depth_sha256",
    "detection_index",
    "segmentation_sha256",
    "sensor",
)
SAFE_HASH_FIELDS = {
    "rgb": "rgb_sha256",
    "camera": "camera_sha256",
    "cad": "model_sha256",
}
SAFE_PATH_FIELDS = (
    "rgb_relative_path",
    "camera_relative_path",
    "model_relative_path",
)
DECLARED_PATH_FIELDS = ("depth_relative_path",)
SHA_FIELDS = (
    "camera_sha256",
    "coco_prediction_sha256",
    "depth_sha256",
    "model_sha256",
    "rgb_sha256",
    "segmentation_sha256",
)
STRICT_PATH_TOKENS = (
    "depth",
    "depth_component",
    "bbox_mask",
    "legacy_mask",
    "legacy_predicted_mask",
    "oracle",
    "ground_truth",
    "evaluator",
    "sealed",
    "sam_vit_b",
    "sam_vit_h",
    "sam_vit_l",
    "sam6d_pose",
    "sam_6d_pose",
)
BOUNDARY_ZERO = {
    "label_access_count": 0,
    "gt_path_open_count": 0,
    "evaluator_path_open_count": 0,
    "sealed_split_access_count": 0,
    "official_scorer_run": False,
    "depth_provenance_asset_open_count": 0,
    "legacy_mask_provenance_asset_open_count": 0,
    "provenance_asset_resolve_count": 0,
    "provenance_asset_copy_count": 0,
}

_V1_COMPATIBILITY_LOCK = threading.RLock()


@dataclass(frozen=True)
class WorkloadAuditResult:
    """Validated rows plus the public path-free provenance audit."""

    rows: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


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


def _binding(path: Path, *, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Missing required file: {relative_path}")
    size = path.stat().st_size
    if size <= 0:
        raise ContractError(f"Required file is empty: {relative_path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"relative_path": relative_path, "bytes": size, "sha256": digest}


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _lexical_path(value: Any, label: str, *, strict: bool) -> str:
    path = _relative(value, label)
    normalized = path.as_posix().lower().replace("-", "_").replace(".", "_")
    if strict:
        matches = [token for token in STRICT_PATH_TOKENS if token in normalized]
        if matches:
            raise ContractError(f"{label} contains forbidden runtime token(s): {matches}")
    return path.as_posix()


def _expected_samples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": sample["item_id"],
            "scene_id": sample["scene_id"],
            "image_id": sample["image_id"],
            "object_id": sample["object_id"],
        }
        for sample in protocol["input_lock"]["samples"]
    ]


def validate_workload_rows(
    rows: Sequence[Any], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Validate exact ordered identities without resolving any declared path."""
    validate_protocol(protocol)
    expected = _expected_samples(protocol)
    if len(rows) != WORKLOAD_LINE_COUNT:
        raise ContractError(
            f"CNOS R1 workload requires exactly {WORKLOAD_LINE_COUNT} JSONL rows"
        )
    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(_exact(raw, set(WORKLOAD_ROW_FIELDS), f"workload row[{index}]"))
        identity = {
            "item_id": row["item_id"],
            "scene_id": row["scene_id"],
            "image_id": row["image_id"],
            "object_id": row["object_id"],
        }
        if identity != expected[index]:
            raise ContractError(f"CNOS R1 workload sample/order mismatch at row {index}")
        if row["category_id"] != row["object_id"] or row["detection_index"] != index:
            raise ContractError(f"CNOS R1 workload category/detection mismatch at row {index}")
        if row["sensor"] != "synthetic_pbr":
            raise ContractError(f"CNOS R1 workload sensor changed at row {index}")
        for field in SHA_FIELDS:
            _require_sha256(row[field], f"workload row[{index}].{field}")
        for field in SAFE_PATH_FIELDS:
            _lexical_path(row[field], f"workload row[{index}].{field}", strict=True)
        for field in DECLARED_PATH_FIELDS:
            _lexical_path(row[field], f"workload row[{index}].{field}", strict=False)
        score = _finite(row["coco_score"], f"workload row[{index}].coco_score")
        if not 0.0 <= score <= 1.0:
            raise ContractError(f"CNOS R1 workload score changed at row {index}")
        if _finite(row["depth_scale"], f"workload row[{index}].depth_scale") <= 0:
            raise ContractError(f"CNOS R1 workload scale changed at row {index}")
        intrinsics = row["camera_intrinsics"]
        if not isinstance(intrinsics, list) or len(intrinsics) != 9:
            raise ContractError(f"CNOS R1 workload intrinsics changed at row {index}")
        for component in intrinsics:
            _finite(component, f"workload row[{index}].camera_intrinsics")
        pose = row["camera_world_to_camera_pose_m"]
        if (
            not isinstance(pose, list)
            or len(pose) != 4
            or any(not isinstance(line, list) or len(line) != 4 for line in pose)
        ):
            raise ContractError(f"CNOS R1 workload camera pose changed at row {index}")
        for line in pose:
            for component in line:
                _finite(component, f"workload row[{index}].camera pose")
        parsed.append(row)
    return tuple(parsed)


def audit_workload_jsonl(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    relative_path: str = "inputs/manifests/workload.json",
) -> WorkloadAuditResult:
    """Read only the manifest and emit a path-free declared-provenance audit."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"Cannot read CNOS R1 workload JSONL {path}: {exc}") from exc
    actual_sha = hashlib.sha256(payload).hexdigest()
    if len(payload) != WORKLOAD_BYTES or actual_sha != WORKLOAD_SHA256:
        raise ContractError("CNOS R1 workload bytes/SHA differ from the frozen JSONL")
    if (
        not payload.endswith(b"\n")
        or payload.count(b"\n") != WORKLOAD_LINE_COUNT
        or b"\r" in payload
    ):
        raise ContractError("CNOS R1 workload must remain exact ten-line LF JSONL")
    raw_lines = payload.splitlines()
    rows: list[Any] = []
    for index, line in enumerate(raw_lines):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"Cannot parse CNOS R1 workload JSONL row {index}: {exc}") from exc
        rows.append(value)
    parsed = validate_workload_rows(rows, protocol)
    identities = [
        {key: row[key] for key in ("item_id", "scene_id", "image_id", "object_id")}
        for row in parsed
    ]
    safe_bindings = [
        {
            "item_id": row["item_id"],
            "rgb_sha256": row["rgb_sha256"],
            "camera_sha256": row["camera_sha256"],
            "model_sha256": row["model_sha256"],
        }
        for row in parsed
    ]
    declared_values = [
        {field: row[field] for field in DECLARED_ONLY_FIELDS} for row in parsed
    ]
    audit: dict[str, Any] = {
        "schema_version": WORKLOAD_AUDIT_SCHEMA,
        "workload": {
            "relative_path": relative_path,
            "bytes": len(payload),
            "sha256": actual_sha,
            "jsonl_line_count": len(parsed),
        },
        "parse_mode": "exact_one_json_object_per_line_no_rewrite",
        "identity_inventory_sha256": canonical_sha256(identities),
        "safe_binding_inventory_sha256": canonical_sha256(safe_bindings),
        "declared_provenance": {
            "status": "DECLARED_BUT_NOT_OPENED",
            "field_names": list(DECLARED_ONLY_FIELDS),
            "value_inventory_sha256": canonical_sha256(declared_values),
            "declared_value_count": len(parsed) * len(DECLARED_ONLY_FIELDS),
            "declared_depth_reference_count": len(parsed) * 2,
            "declared_legacy_segmentation_reference_count": len(parsed) * 2,
            "resolved_path_count": 0,
            "asset_open_count": 0,
            "asset_copy_count": 0,
            "runtime_input_inclusion_count": 0,
        },
        "workload_audit_lock_sha256": "pending",
    }
    audit["workload_audit_lock_sha256"] = canonical_sha256(
        {key: value for key, value in audit.items() if key != "workload_audit_lock_sha256"}
    )
    return WorkloadAuditResult(rows=parsed, audit=audit)


def _expected_predecessor() -> dict[str, Any]:
    return {
        "implementation_commit": PREDECESSOR_IMPLEMENTATION_COMMIT,
        "implementation_tree": PREDECESSOR_IMPLEMENTATION_TREE,
        "v1_route_id": V1_ROUTE_ID,
        "v1_route_lock_sha256": PREDECESSOR_ROUTE_LOCK,
        "pre_model_blocker": {
            "status": "PRE_MODEL_SCHEMA_BLOCKER",
            "remote_root": PREDECESSOR_BLOCKED_ROOT,
            "receipt_sha256": PREDECESSOR_BLOCKER_RECEIPT_SHA256,
            "workload_bytes": WORKLOAD_BYTES,
            "workload_sha256": WORKLOAD_SHA256,
            "workload_jsonl_line_count": WORKLOAD_LINE_COUNT,
            "immutable": True,
            "reinterpretation_permitted": False,
        },
    }


def _expected_workload_contract() -> dict[str, Any]:
    return {
        "format": "JSONL",
        "bytes": WORKLOAD_BYTES,
        "sha256": WORKLOAD_SHA256,
        "line_count": WORKLOAD_LINE_COUNT,
        "row_fields": list(WORKLOAD_ROW_FIELDS),
        "identity_fields": ["item_id", "scene_id", "image_id", "object_id"],
        "declared_only_fields": list(DECLARED_ONLY_FIELDS),
        "runtime_hash_fields": dict(SAFE_HASH_FIELDS),
        "declared_path_resolution_permitted": False,
        "declared_asset_open_permitted": False,
        "declared_asset_copy_permitted": False,
        "declared_metadata_runtime_input_permitted": False,
    }


def _expected_execution_identity() -> dict[str, Any]:
    return {
        "deployment_namespace": "poseloop_ga_cnos_v1r1",
        "screen_session_prefix": "poseloop_ga_cnos_v1r1",
        "output_namespace": "output-v1r1",
        "compatibility_payload_namespace": "compat-v1-payload",
    }


def _validate_dependencies(
    route: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    dependency_paths: set[str] = set()
    bindings: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(route["v1_dependency_files"]):
        item = _exact(value, {"relative_path", "bytes", "sha256"}, f"dependency[{index}]")
        relative = _relative(item["relative_path"], f"dependency[{index}]")
        if relative.as_posix() in dependency_paths:
            raise ContractError("CNOS R1 dependency paths must be unique")
        _require_sha256(item["sha256"], f"dependency[{index}].sha256")
        actual = _binding(
            _resolve(repository_root, relative, f"dependency[{index}]"),
            relative_path=relative.as_posix(),
        )
        if actual != dict(item):
            raise ContractError(f"Frozen v1 dependency bytes changed: {relative.as_posix()}")
        dependency_paths.add(relative.as_posix())
        bindings[relative.as_posix()] = actual
    required = {
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/__init__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/__main__.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/adapter.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/cli.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/cnos_adapter_config_v1.json",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/contracts.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/producer.py",
        "pose_accuracy_recovery_prep/cnos_runtime_prep_v1/SERVER_RUNBOOK_CNOS_RUNTIME_PREP.md",
        "pose_accuracy_recovery_prep/instance_proposal_v1/__init__.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1/contracts.py",
        V1_ROUTE_RELATIVE_PATH,
        "protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1.json",
    }
    if dependency_paths != required:
        raise ContractError("CNOS R1 frozen v1 dependency inventory changed")
    v1_route = read_json(repository_root / Path(*PurePosixPath(V1_ROUTE_RELATIVE_PATH).parts))
    if not isinstance(v1_route, dict):
        raise ContractError("Frozen v1 route must be an object")
    validation = v1.validate_route(v1_route, repository_root=repository_root)
    if (
        validation["route_id"] != V1_ROUTE_ID
        or validation["route_lock_sha256"] != PREDECESSOR_ROUTE_LOCK
    ):
        raise ContractError("CNOS R1 predecessor route identity changed")
    return v1_route


def validate_route(
    route: Mapping[str, Any], *, repository_root: Path | None = None
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
            "workload_contract",
            "execution_identity",
            "boundary",
            "v1_dependency_files",
        },
        "CNOS R1 route",
    )
    if (
        route["schema_version"] != ROUTE_SCHEMA
        or route["route_id"] != ROUTE_ID
        or route["auto_deploy"] is not False
        or route["role"] != "DEVELOPMENT_ONLY"
    ):
        raise ContractError("CNOS R1 route identity/state mismatch")
    if route["predecessor"] != _expected_predecessor():
        raise ContractError("CNOS R1 predecessor/blocker binding changed")
    if route["workload_contract"] != _expected_workload_contract():
        raise ContractError("CNOS R1 workload contract changed")
    if route["execution_identity"] != _expected_execution_identity():
        raise ContractError("CNOS R1 execution identity changed")
    if route["boundary"] != BOUNDARY_ZERO:
        raise ContractError("CNOS R1 zero-access boundary changed")
    dependencies = route["v1_dependency_files"]
    if not isinstance(dependencies, list) or len(dependencies) != 13:
        raise ContractError("CNOS R1 requires exactly thirteen frozen dependency files")
    for index, value in enumerate(dependencies):
        item = _exact(value, {"relative_path", "bytes", "sha256"}, f"dependency[{index}]")
        _relative(item["relative_path"], f"dependency[{index}]")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or item["bytes"] <= 0:
            raise ContractError(f"dependency[{index}].bytes must be positive")
        _require_sha256(item["sha256"], f"dependency[{index}].sha256")
    route_lock = _require_lock(route, "route_lock_sha256", "CNOS R1 route")
    if repository_root is not None:
        _validate_dependencies(route, repository_root.resolve())
    return {
        "status": "valid",
        "route_id": ROUTE_ID,
        "route_lock_sha256": route_lock,
        "predecessor_blocker_receipt_sha256": PREDECESSOR_BLOCKER_RECEIPT_SHA256,
        "workload_sha256": WORKLOAD_SHA256,
        "execution_ready": False,
        "weights_downloaded": False,
    }


def _compatibility_view(result: WorkloadAuditResult) -> dict[str, Any]:
    return {
        "schema_version": "poseloop.cnos-workload-hash-view.v1r1",
        "workload_sha256": WORKLOAD_SHA256,
        "workload_audit_lock_sha256": result.audit["workload_audit_lock_sha256"],
        "identity_inventory_sha256": result.audit["identity_inventory_sha256"],
        "row_count": WORKLOAD_LINE_COUNT,
    }


@contextmanager
def v1_jsonl_compatibility(
    workload_path: Path, result: WorkloadAuditResult
) -> Iterator[None]:
    """Intercept only the exact workload-object read in frozen v1 code."""
    target = workload_path.resolve()
    view = _compatibility_view(result)
    with _V1_COMPATIBILITY_LOCK:
        original_load = v1._load_object
        original_runtime_read = proposal_contracts.read_json

        def load_object(path: Path, label: str) -> dict[str, Any]:
            if path.resolve() == target and label in {
                "workload manifest",
                "runtime workload manifest",
            }:
                return copy.deepcopy(view)
            return original_load(path, label)

        def runtime_read(path: Path) -> Any:
            if path.resolve() == target:
                return copy.deepcopy(view)
            return original_runtime_read(path)

        v1._load_object = load_object
        proposal_contracts.read_json = runtime_read
        try:
            yield
        finally:
            if (
                v1._load_object is not load_object
                or proposal_contracts.read_json is not runtime_read
            ):
                raise ContractError("CNOS R1 v1 compatibility hook was changed concurrently")
            v1._load_object = original_load
            proposal_contracts.read_json = original_runtime_read


def _request_for_v1(request: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
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
        "CNOS R1 deployment request",
    )
    if (
        request["schema_version"] != DEPLOYMENT_REQUEST_SCHEMA
        or request["role"] != "DEVELOPMENT_ONLY"
        or request["route_lock_sha256"] != route["route_lock_sha256"]
        or request["boundary"] != v1.DEPLOYMENT_BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R1 deployment request identity/boundary mismatch")
    converted = copy.deepcopy(dict(request))
    converted["schema_version"] = V1_DEPLOYMENT_REQUEST_SCHEMA
    converted["route_lock_sha256"] = PREDECESSOR_ROUTE_LOCK
    return converted


def _workload_path(
    deployment_root: Path, request: Mapping[str, Any]
) -> Path:
    paths = request.get("paths")
    if not isinstance(paths, Mapping) or paths.get("workload_manifest") != "inputs/manifests/workload.json":
        raise ContractError("CNOS R1 workload path changed")
    return _resolve(
        deployment_root,
        _relative(paths["workload_manifest"], "CNOS R1 workload", required_root="inputs"),
        "CNOS R1 workload",
    )


def _validate_safe_runtime_bindings(
    parent_runtime: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    by_item = {row["item_id"]: row for row in rows}
    items = parent_runtime["data"]["items"]
    for item in items:
        inputs = item["inputs"]
        if set(inputs) != set(ALLOWED_INPUT_ROLES):
            raise ContractError(f"CNOS R1 runtime input roles changed: {item['item_id']}")
        row = by_item[item["item_id"]]
        for role, workload_field in SAFE_HASH_FIELDS.items():
            if inputs[role]["sha256"] != row[workload_field]:
                raise ContractError(
                    f"CNOS R1 runtime {role} hash differs from workload: {item['item_id']}"
                )
        for role, asset in inputs.items():
            relative = _relative(
                asset["relative_path"],
                f"CNOS R1 runtime {item['item_id']}.{role}",
                required_root="inputs",
            )
            normalized = relative.as_posix().lower().replace("-", "_").replace(".", "_")
            matches = [token for token in STRICT_PATH_TOKENS if token in normalized]
            if matches:
                raise ContractError(
                    f"CNOS R1 runtime input path contains forbidden token(s): {matches}"
                )


def _runtime_lock(
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_runtime: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "route_id": ROUTE_ID,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "runtime_lock_sha256": "pending",
        "state": "RGB_CAMERA_CAD_DESCRIPTORS_ONLY",
        "allowed_runtime_input_roles": list(ALLOWED_INPUT_ROLES),
        "parent_runtime_lock": copy.deepcopy(dict(parent_runtime)),
        "workload_provenance_audit": copy.deepcopy(dict(audit)),
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
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "route_id": ROUTE_ID,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "deployment_lock_sha256": "pending",
        "role": "DEVELOPMENT_ONLY_CNOS_PRODUCER_V1R1",
        "state": "EXECUTION_ASSETS_HASHED_NOT_EXECUTED",
        "predecessor_blocker_receipt_sha256": PREDECESSOR_BLOCKER_RECEIPT_SHA256,
        "execution_identity": copy.deepcopy(route["execution_identity"]),
        "parent_deployment": copy.deepcopy(dict(parent_deployment)),
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "workload_audit_lock_sha256": audit["workload_audit_lock_sha256"],
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
    """Freeze an R1 envelope around the unchanged v1 runtime implementation."""
    validate_route(route)
    validate_protocol(protocol)
    parent_request = _request_for_v1(request, route)
    implementation_path = _resolve(
        deployment_root,
        _relative(
            parent_request["paths"]["implementation_source_checkout"],
            "CNOS R1 implementation checkout",
        ),
        "CNOS R1 implementation checkout",
    )
    v1_route = _validate_dependencies(route, implementation_path)
    implementation = parent_request["implementation"]
    if (
        implementation["approved_commit"] == PREDECESSOR_IMPLEMENTATION_COMMIT
        or not v1.git_is_ancestor(
            implementation_path,
            PREDECESSOR_IMPLEMENTATION_COMMIT,
            implementation["approved_commit"],
        )
    ):
        raise ContractError("CNOS R1 implementation must descend from commit 0389cb3")
    workload_path = _workload_path(deployment_root, request)
    result = audit_workload_jsonl(workload_path, protocol)
    with v1_jsonl_compatibility(workload_path, result):
        parent_deployment, parent_runtime = v1.freeze_deployment(
            v1_route,
            protocol,
            parent_request,
            deployment_root=deployment_root,
        )
    _validate_safe_runtime_bindings(parent_runtime, result.rows)
    runtime_lock = _runtime_lock(route, protocol, parent_runtime, result.audit)
    deployment = _deployment(
        route, protocol, parent_deployment, runtime_lock, result.audit
    )
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    return deployment, runtime_lock


def validate_deployment(
    deployment: Mapping[str, Any],
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    *,
    deployment_root: Path,
) -> dict[str, Any]:
    validate_route(route)
    protocol_validation = validate_protocol(protocol)
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
            "predecessor_blocker_receipt_sha256",
            "execution_identity",
            "parent_deployment",
            "runtime_lock_sha256",
            "workload_audit_lock_sha256",
            "boundary",
        },
        "CNOS R1 deployment",
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
            "boundary",
        },
        "CNOS R1 runtime lock",
    )
    if (
        deployment["schema_version"] != DEPLOYMENT_SCHEMA
        or deployment["route_id"] != ROUTE_ID
        or deployment["route_lock_sha256"] != route["route_lock_sha256"]
        or deployment["protocol_id"] != protocol["protocol_id"]
        or deployment["protocol_lock_sha256"] != protocol_validation["protocol_lock_sha256"]
        or deployment["role"] != "DEVELOPMENT_ONLY_CNOS_PRODUCER_V1R1"
        or deployment["state"] != "EXECUTION_ASSETS_HASHED_NOT_EXECUTED"
        or deployment["predecessor_blocker_receipt_sha256"]
        != PREDECESSOR_BLOCKER_RECEIPT_SHA256
        or deployment["execution_identity"] != _expected_execution_identity()
        or deployment["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R1 deployment identity/state changed")
    if (
        runtime_lock["schema_version"] != RUNTIME_LOCK_SCHEMA
        or runtime_lock["route_id"] != ROUTE_ID
        or runtime_lock["route_lock_sha256"] != route["route_lock_sha256"]
        or runtime_lock["protocol_id"] != protocol["protocol_id"]
        or runtime_lock["protocol_lock_sha256"] != protocol_validation["protocol_lock_sha256"]
        or runtime_lock["state"] != "RGB_CAMERA_CAD_DESCRIPTORS_ONLY"
        or runtime_lock["allowed_runtime_input_roles"] != list(ALLOWED_INPUT_ROLES)
        or runtime_lock["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError("CNOS R1 runtime identity/roles changed")
    runtime_lock_sha = _require_lock(runtime_lock, "runtime_lock_sha256", "CNOS R1 runtime lock")
    deployment_lock_sha = _require_lock(
        deployment, "deployment_lock_sha256", "CNOS R1 deployment"
    )
    if deployment["runtime_lock_sha256"] != runtime_lock_sha:
        raise ContractError("CNOS R1 deployment/runtime locks diverged")
    audit = runtime_lock["workload_provenance_audit"]
    _require_lock(audit, "workload_audit_lock_sha256", "CNOS R1 workload audit")
    if (
        audit["schema_version"] != WORKLOAD_AUDIT_SCHEMA
        or audit["declared_provenance"]["status"] != "DECLARED_BUT_NOT_OPENED"
        or any(
            audit["declared_provenance"][field] != 0
            for field in (
                "resolved_path_count",
                "asset_open_count",
                "asset_copy_count",
                "runtime_input_inclusion_count",
            )
        )
        or deployment["workload_audit_lock_sha256"]
        != audit["workload_audit_lock_sha256"]
    ):
        raise ContractError("CNOS R1 workload provenance boundary changed")
    parent_deployment = deployment["parent_deployment"]
    parent_runtime = runtime_lock["parent_runtime_lock"]
    implementation_path = _resolve(
        deployment_root,
        _relative(
            parent_deployment["implementation"]["checkout_relative_path"],
            "CNOS R1 implementation checkout",
        ),
        "CNOS R1 implementation checkout",
    )
    v1_route = _validate_dependencies(route, implementation_path)
    if (
        parent_deployment["implementation"]["commit"]
        == PREDECESSOR_IMPLEMENTATION_COMMIT
        or not v1.git_is_ancestor(
            implementation_path,
            PREDECESSOR_IMPLEMENTATION_COMMIT,
            parent_deployment["implementation"]["commit"],
        )
    ):
        raise ContractError("Frozen CNOS R1 implementation ancestry changed")
    workload_asset = parent_runtime["data"]["workload_manifest"]
    workload_path = _resolve(
        deployment_root,
        _relative(
            workload_asset["relative_path"],
            "CNOS R1 runtime workload",
            required_root="inputs",
        ),
        "CNOS R1 runtime workload",
    )
    result = audit_workload_jsonl(workload_path, protocol)
    if result.audit != audit:
        raise ContractError("CNOS R1 workload audit differs from disk")
    with v1_jsonl_compatibility(workload_path, result):
        parent_validation = v1.validate_deployment(
            parent_deployment,
            v1_route,
            protocol,
            parent_runtime,
            deployment_root=deployment_root,
        )
    _validate_safe_runtime_bindings(parent_runtime, result.rows)
    return {
        "status": "valid",
        "execution_assets_ready": True,
        "deployment_lock_sha256": deployment_lock_sha,
        "runtime_lock_sha256": runtime_lock_sha,
        "parent_deployment_lock_sha256": parent_validation["deployment_lock_sha256"],
        "parent_runtime_lock_sha256": parent_validation["runtime_lock_sha256"],
        "workload_audit_lock_sha256": audit["workload_audit_lock_sha256"],
        "item_count": WORKLOAD_LINE_COUNT,
        "runtime_input_roles": list(ALLOWED_INPUT_ROLES),
        "declared_provenance_status": "DECLARED_BUT_NOT_OPENED",
        "label_access_count": 0,
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value
