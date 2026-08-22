"""A-R2 output envelope around the unchanged official-CNOS v1 producer."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import producer as v1_producer
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 import contracts as r1
from pose_accuracy_recovery_prep.core import ContractError, canonical_sha256, read_json, sha256_file

from . import RUN_RECEIPT_SCHEMA
from .contracts import (
    BOUNDARY_ZERO,
    CAMERA_DERIVATION_MANIFEST_PATH,
    R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
    _json_bytes,
    _relative,
    _resolve,
    _validate_dependencies,
    _write_immutable,
    validate_deployment,
)

CnosBackend = v1_producer.CnosBackend
PlannedCrash = v1_producer.PlannedCrash


def _asset(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_producer(
    *,
    route: Mapping[str, Any],
    protocol: Mapping[str, Any],
    deployment: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    deployment_root: Path,
    output_root: Path,
    backend: CnosBackend,
    planned_crash_after_items: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute v1 with derived cameras and an immutable A-R2 receipt envelope."""
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    if output_root.name != route["execution_identity"]["output_namespace"]:
        raise ContractError("CNOS R2 output root must use the frozen namespace")
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
        implementation_root / Path(*PurePosixPath(r1.V1_ROUTE_RELATIVE_PATH).parts)
    )
    if not isinstance(parent_route, dict):
        raise ContractError("CNOS R2 frozen v1 route must be an object")
    workload_path = _resolve(
        deployment_root,
        _relative(
            parent_runtime["data"]["workload_manifest"]["relative_path"],
            "CNOS R2 workload",
            required_root="inputs",
        ),
        "CNOS R2 workload",
    )
    workload = r1.audit_workload_jsonl(workload_path, protocol)
    compatibility_root = (
        output_root / route["execution_identity"]["compatibility_payload_namespace"]
    )
    with r1.v1_jsonl_compatibility(workload_path, workload):
        bundle, parent_receipt = v1_producer.run_producer(
            route=parent_route,
            protocol=protocol,
            deployment=parent_deployment,
            runtime_lock=parent_runtime,
            deployment_root=deployment_root,
            output_root=compatibility_root,
            backend=backend,
            planned_crash_after_items=planned_crash_after_items,
        )
    parent_bundle_path = compatibility_root / "proposal-bundle.json"
    parent_receipt_path = compatibility_root / "run-receipt.json"
    camera_manifest_path = _resolve(
        deployment_root,
        _relative(CAMERA_DERIVATION_MANIFEST_PATH, "camera derivation manifest"),
        "camera derivation manifest",
    )
    receipt: dict[str, Any] = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": parent_receipt["status"],
        "route_id": route["route_id"],
        "route_lock_sha256": route["route_lock_sha256"],
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "r1_camera_blocker_closeout_sha256": R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
        "workload_audit_lock_sha256": workload.audit["workload_audit_lock_sha256"],
        "camera_derivation_manifest": _asset(camera_manifest_path, deployment_root),
        "camera_derivation_manifest_lock_sha256": deployment[
            "camera_derivation_manifest_lock_sha256"
        ],
        "parent_proposal_bundle": _asset(parent_bundle_path, output_root),
        "parent_run_receipt": _asset(parent_receipt_path, output_root),
        "parent_proposal_bundle_lock_sha256": bundle["proposal_bundle_lock_sha256"],
        "parent_run_receipt_lock_sha256": parent_receipt["run_receipt_lock_sha256"],
        "source_camera_json_parse_count": 0,
        "source_camera_runtime_inclusion_count": 0,
        "boundary": copy.deepcopy(BOUNDARY_ZERO),
        "run_receipt_lock_sha256": "pending",
    }
    receipt["run_receipt_lock_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "run_receipt_lock_sha256"}
    )
    receipt_path = output_root / "run-receipt-v1r2.json"
    payload = _json_bytes(receipt)
    _write_immutable(receipt_path, payload)
    if hashlib.sha256(payload).hexdigest() != sha256_file(receipt_path):
        raise ContractError("CNOS R2 run receipt disk verification failed")
    return bundle, receipt
