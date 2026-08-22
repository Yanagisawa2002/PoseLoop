"""CLI for frozen official-CNOS runtime preparation and future execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    validate_protocol,
)

from .adapter import OfficialCnosAdapter
from .contracts import (
    DEPLOYMENT_BOUNDARY_ZERO,
    freeze_deployment,
    load_json_object,
    validate_deployment,
    validate_route,
)
from .producer import run_producer

FREEZE_RECEIPT_SCHEMA = (
    "poseloop.pose-accuracy-recovery.cnos-deployment-freeze-receipt.v1"
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _ensure_immutable_target(path: Path, payload: bytes) -> None:
    if path.exists() and (not path.is_file() or path.read_bytes() != payload):
        raise ContractError(f"Immutable CNOS output already differs: {path}")


def _write_immutable(path: Path, payload: bytes) -> None:
    _ensure_immutable_target(path, payload)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"Immutable CNOS output raced with different data: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _payload_asset(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _emit(value: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        write_json(output, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1",
        description=(
            "Freeze and verify the DEVELOPMENT_ONLY official CNOS FastSAM + DINOv2 "
            "runtime. No evaluator or sealed input is permitted."
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
    route = load_json_object(args.route, "CNOS route")
    protocol = load_json_object(args.protocol, "Parent protocol")
    request = load_json_object(args.request, "CNOS deployment request")
    deployment, runtime_lock = freeze_deployment(
        route,
        protocol,
        request,
        deployment_root=args.deployment_root,
    )
    deployment_payload = _json_bytes(deployment)
    runtime_payload = _json_bytes(runtime_lock)
    receipt = {
        "schema_version": FREEZE_RECEIPT_SCHEMA,
        "route_lock_sha256": route["route_lock_sha256"],
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "deployment_lock_sha256": deployment["deployment_lock_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "implementation_commit": deployment["implementation"]["commit"],
        "implementation_tree": deployment["implementation"]["tree"],
        "implementation_source_archive_sha256": deployment["implementation"][
            "source_archive"
        ]["sha256"],
        "deployment": _payload_asset(args.deployment_output, deployment_payload),
        "runtime_lock": _payload_asset(args.runtime_lock_output, runtime_payload),
        "execution_started": False,
        "model_imported": False,
        "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
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
        _ensure_immutable_target(path, payload)
    _write_immutable(args.deployment_output, deployment_payload)
    _write_immutable(args.runtime_lock_output, runtime_payload)
    _write_immutable(args.receipt_output, receipt_payload)
    if (
        sha256_file(args.deployment_output) != receipt["deployment"]["sha256"]
        or sha256_file(args.runtime_lock_output) != receipt["runtime_lock"]["sha256"]
        or sha256_file(args.receipt_output)
        != hashlib.sha256(receipt_payload).hexdigest()
    ):
        raise ContractError("CNOS immutable freeze output verification failed")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-route":
            route = load_json_object(args.route, "CNOS route")
            result = validate_route(route, repository_root=args.repository_root)
            _emit(result, args.output)
            return 0
        if args.command == "freeze-deployment":
            result = _freeze(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        route = load_json_object(args.route, "CNOS route")
        protocol = load_json_object(args.protocol, "Parent protocol")
        deployment = load_json_object(args.deployment, "CNOS deployment")
        runtime_lock = load_json_object(args.runtime_lock, "Parent runtime lock")
        validate_protocol(protocol)
        if args.command == "preflight":
            result = validate_deployment(
                deployment,
                route,
                protocol,
                runtime_lock,
                deployment_root=args.deployment_root,
            )
            _emit(result, args.output)
            return 0

        backend = OfficialCnosAdapter(
            deployment_root=args.deployment_root,
            deployment=deployment,
            device=deployment["runtime"]["device"],
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
                    "proposal_bundle_lock_sha256": bundle[
                        "proposal_bundle_lock_sha256"
                    ],
                    "run_receipt_lock_sha256": receipt[
                        "run_receipt_lock_sha256"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2
