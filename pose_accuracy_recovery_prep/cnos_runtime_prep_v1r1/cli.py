"""CLI for the CNOS runtime-prep v1r1 compatibility route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.adapter import OfficialCnosAdapter
from pose_accuracy_recovery_prep.core import ContractError, canonical_sha256, sha256_file
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import validate_protocol

from . import FREEZE_RECEIPT_SCHEMA
from .contracts import (
    BOUNDARY_ZERO,
    PREDECESSOR_BLOCKER_RECEIPT_SHA256,
    freeze_deployment,
    load_json_object,
    validate_deployment,
    validate_route,
)
from .producer import run_producer


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


def _asset(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _emit(value: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        _write_immutable(output, (text + "\n").encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1",
        description=(
            "Freeze the exact ten-line historical workload as declared-only provenance "
            "while retaining RGB/camera/CAD/descriptor-only CNOS runtime inputs."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("validate-route")
    route.add_argument("--route", type=Path, required=True)
    route.add_argument("--repository-root", type=Path)
    route.add_argument("--output", type=Path)
    freeze = commands.add_parser("freeze-deployment")
    freeze.add_argument("--route", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--request", type=Path, required=True)
    freeze.add_argument("--deployment-root", type=Path, required=True)
    freeze.add_argument("--deployment-output", type=Path, required=True)
    freeze.add_argument("--runtime-lock-output", type=Path, required=True)
    freeze.add_argument("--receipt-output", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--route", type=Path, required=True)
    preflight.add_argument("--protocol", type=Path, required=True)
    preflight.add_argument("--deployment", type=Path, required=True)
    preflight.add_argument("--runtime-lock", type=Path, required=True)
    preflight.add_argument("--deployment-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    run = commands.add_parser("run-producer")
    run.add_argument("--route", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--deployment", type=Path, required=True)
    run.add_argument("--runtime-lock", type=Path, required=True)
    run.add_argument("--deployment-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--planned-crash-after-items", type=int)
    return parser


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    route = load_json_object(args.route, "CNOS R1 route")
    protocol = load_json_object(args.protocol, "Parent protocol")
    request = load_json_object(args.request, "CNOS R1 deployment request")
    deployment, runtime_lock = freeze_deployment(
        route, protocol, request, deployment_root=args.deployment_root
    )
    deployment_payload = _json_bytes(deployment)
    runtime_payload = _json_bytes(runtime_lock)
    receipt: dict[str, Any] = {
        "schema_version": FREEZE_RECEIPT_SCHEMA,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "predecessor_blocker_receipt_sha256": PREDECESSOR_BLOCKER_RECEIPT_SHA256,
        "workload_audit_lock_sha256": deployment["workload_audit_lock_sha256"],
        "implementation_commit": deployment["parent_deployment"]["implementation"]["commit"],
        "implementation_tree": deployment["parent_deployment"]["implementation"]["tree"],
        "deployment": _asset(args.deployment_output, deployment_payload),
        "runtime_lock": _asset(args.runtime_lock_output, runtime_payload),
        "execution_started": False,
        "model_imported": False,
        "boundary": dict(BOUNDARY_ZERO),
        "freeze_receipt_lock_sha256": "pending",
    }
    receipt["freeze_receipt_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "freeze_receipt_lock_sha256"
        }
    )
    receipt_payload = _json_bytes(receipt)
    for path, payload in (
        (args.deployment_output, deployment_payload),
        (args.runtime_lock_output, runtime_payload),
        (args.receipt_output, receipt_payload),
    ):
        _write_immutable(path, payload)
    if (
        sha256_file(args.deployment_output) != receipt["deployment"]["sha256"]
        or sha256_file(args.runtime_lock_output) != receipt["runtime_lock"]["sha256"]
        or sha256_file(args.receipt_output)
        != hashlib.sha256(receipt_payload).hexdigest()
    ):
        raise ContractError("CNOS R1 immutable freeze output verification failed")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-route":
            route = load_json_object(args.route, "CNOS R1 route")
            _emit(validate_route(route, repository_root=args.repository_root), args.output)
            return 0
        if args.command == "freeze-deployment":
            print(json.dumps(_freeze(args), indent=2, sort_keys=True))
            return 0
        route = load_json_object(args.route, "CNOS R1 route")
        protocol = load_json_object(args.protocol, "Parent protocol")
        deployment = load_json_object(args.deployment, "CNOS R1 deployment")
        runtime_lock = load_json_object(args.runtime_lock, "CNOS R1 runtime lock")
        validate_protocol(protocol)
        if args.command == "preflight":
            _emit(
                validate_deployment(
                    deployment,
                    route,
                    protocol,
                    runtime_lock,
                    deployment_root=args.deployment_root,
                ),
                args.output,
            )
            return 0
        parent_deployment = deployment["parent_deployment"]
        backend = OfficialCnosAdapter(
            deployment_root=args.deployment_root,
            deployment=parent_deployment,
            device=parent_deployment["runtime"]["device"],
        )
        bundle, receipt = run_producer(
            route=route,
            protocol=protocol,
            deployment=deployment,
            runtime_lock=runtime_lock,
            deployment_root=args.deployment_root,
            output_root=args.output_root,
            backend=backend,
            planned_crash_after_items=args.planned_crash_after_items,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "proposal_bundle_lock_sha256": bundle["proposal_bundle_lock_sha256"],
                    "run_receipt_lock_sha256": receipt["run_receipt_lock_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2
