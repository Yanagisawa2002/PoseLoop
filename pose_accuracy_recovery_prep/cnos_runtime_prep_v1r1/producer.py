"""R1 output envelope around the frozen v1 CNOS producer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import producer as v1_producer
from pose_accuracy_recovery_prep.core import ContractError, canonical_sha256, sha256_file

from . import RUN_RECEIPT_SCHEMA
from .contracts import (
    BOUNDARY_ZERO,
    _resolve,
    _relative,
    _validate_dependencies,
    audit_workload_jsonl,
    validate_deployment,
    v1_jsonl_compatibility,
)

CnosBackend = v1_producer.CnosBackend
PlannedCrash = v1_producer.PlannedCrash


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise ContractError(f"Immutable CNOS R1 output already differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"Immutable CNOS R1 output raced with different data: {path}")
    finally:
        temporary.unlink(missing_ok=True)


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
    """Execute the unchanged v1 producer below a new, hash-bound R1 namespace."""
    validate_deployment(
        deployment,
        route,
        protocol,
        runtime_lock,
        deployment_root=deployment_root,
    )
    if output_root.name != route["execution_identity"]["output_namespace"]:
        raise ContractError("CNOS R1 output root must use the frozen output namespace")
    parent_deployment = deployment["parent_deployment"]
    parent_runtime = runtime_lock["parent_runtime_lock"]
    implementation_root = _resolve(
        deployment_root,
        _relative(
            parent_deployment["implementation"]["checkout_relative_path"],
            "CNOS R1 implementation checkout",
        ),
        "CNOS R1 implementation checkout",
    )
    v1_route = _validate_dependencies(route, implementation_root)
    workload_asset = parent_runtime["data"]["workload_manifest"]
    workload_path = _resolve(
        deployment_root,
        _relative(
            workload_asset["relative_path"],
            "CNOS R1 workload",
            required_root="inputs",
        ),
        "CNOS R1 workload",
    )
    workload = audit_workload_jsonl(workload_path, protocol)
    compatibility_root = (
        output_root / route["execution_identity"]["compatibility_payload_namespace"]
    )
    with v1_jsonl_compatibility(workload_path, workload):
        bundle, parent_receipt = v1_producer.run_producer(
            route=v1_route,
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
    receipt: dict[str, Any] = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": parent_receipt["status"],
        "route_id": route["route_id"],
        "route_lock_sha256": route["route_lock_sha256"],
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "workload_audit_lock_sha256": workload.audit["workload_audit_lock_sha256"],
        "parent_proposal_bundle": _asset(parent_bundle_path, output_root),
        "parent_run_receipt": _asset(parent_receipt_path, output_root),
        "parent_proposal_bundle_lock_sha256": bundle["proposal_bundle_lock_sha256"],
        "parent_run_receipt_lock_sha256": parent_receipt["run_receipt_lock_sha256"],
        "boundary": dict(BOUNDARY_ZERO),
        "run_receipt_lock_sha256": "pending",
    }
    receipt["run_receipt_lock_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "run_receipt_lock_sha256"}
    )
    receipt_path = output_root / "run-receipt-v1r1.json"
    _write_immutable(receipt_path, _json_bytes(receipt))
    if hashlib.sha256(_json_bytes(receipt)).hexdigest() != sha256_file(receipt_path):
        raise ContractError("CNOS R1 run receipt disk verification failed")
    return bundle, receipt
