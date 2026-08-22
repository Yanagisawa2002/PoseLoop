"""CLI for inert A-R5-P2 prep and future reviewed descriptor execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.core import ContractError

from . import OBJECT_IDS
from .contracts import (
    build_runtime_request,
    create_only_json,
    read_json,
    repository_root_from_package,
    validate_protocol,
    validate_runtime_request,
)
from .producer import PlannedStop, run_producer, validate_success_from_disk
from .template_manifest import build_template_manifest


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _protocol_request(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, preflight, produce, and independently validate the exact "
            "five-object official-CNOS DINOv2 descriptor assets required by A-R5."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser("validate-protocol")
    protocol.add_argument("--protocol", type=Path, required=True)
    protocol.add_argument("--repository-root", type=Path)

    template = commands.add_parser("build-template-manifest")
    template.add_argument("--protocol", type=Path, required=True)
    template.add_argument("--data-root", type=Path, required=True)
    template.add_argument("--source-png-root", type=Path, required=True)
    template.add_argument("--authorization-receipt", type=Path, required=True)
    template.add_argument("--attempt-receipt", type=Path, required=True)
    template.add_argument("--content-audit", type=Path, required=True)
    template.add_argument("--safe-archive", type=Path, required=True)
    template.add_argument("--safe-member-inventory", type=Path, required=True)
    template.add_argument("--deployment-inventory", type=Path, required=True)
    template.add_argument("--cad-root", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--import-receipt-output", type=Path, required=True)

    request = commands.add_parser("build-request")
    request.add_argument("--protocol", type=Path, required=True)
    request.add_argument("--data-root", type=Path, required=True)
    request.add_argument("--implementation-archive", type=Path, required=True)
    request.add_argument("--template-manifest", type=Path, required=True)
    request.add_argument("--template-import-receipt", type=Path, required=True)
    request.add_argument("--cnos-checkout", type=Path, required=True)
    request.add_argument("--cnos-archive", type=Path, required=True)
    request.add_argument("--dinov2-checkout", type=Path, required=True)
    request.add_argument("--dinov2-archive", type=Path, required=True)
    request.add_argument("--dinov2-weights", type=Path, required=True)
    request.add_argument("--cad-root", type=Path, required=True)
    request.add_argument("--output", type=Path, required=True)

    preflight = commands.add_parser("preflight")
    _protocol_request(preflight)

    produce = commands.add_parser("produce")
    _protocol_request(produce)
    produce.add_argument("--planned-stop-after-objects", type=int)
    produce.add_argument("--resume", action="store_true")

    success = commands.add_parser("validate-success")
    _protocol_request(success)
    success.add_argument("--output", type=Path, required=True)
    return parser


def _cad_paths(root: Path) -> dict[int, Path]:
    return {object_id: root / f"obj_{object_id:06d}.ply" for object_id in OBJECT_IDS}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = read_json(args.protocol, "A-R5-P2 protocol")
        if args.command == "validate-protocol":
            value = validate_protocol(
                protocol,
                repository_root=args.repository_root or repository_root_from_package(),
            )
            _emit(
                {
                    "status": "PASS",
                    "protocol_id": value["protocol_id"],
                    "protocol_lock_sha256": value["protocol_lock_sha256"],
                    "server_connected": False,
                    "model_imported": False,
                }
            )
            return 0
        if args.command == "build-template-manifest":
            validate_protocol(protocol, repository_root=repository_root_from_package())
            manifest, receipt = build_template_manifest(
                data_root=args.data_root,
                source_png_root=args.source_png_root,
                authorization_receipt=args.authorization_receipt,
                attempt_receipt=args.attempt_receipt,
                content_audit=args.content_audit,
                safe_archive=args.safe_archive,
                safe_member_inventory=args.safe_member_inventory,
                deployment_inventory=args.deployment_inventory,
                cad_paths=_cad_paths(args.cad_root),
                manifest_output=args.output,
                import_receipt_output=args.import_receipt_output,
            )
            _emit(
                {
                    "status": receipt["status"],
                    "rgba_count": receipt["source_png_count"],
                    "template_manifest_lock_sha256": manifest[
                        "template_manifest_lock_sha256"
                    ],
                    "template_import_receipt_lock_sha256": receipt[
                        "template_import_receipt_lock_sha256"
                    ],
                }
            )
            return 0
        if args.command == "build-request":
            expected = (
                args.data_root.resolve()
                / "contracts"
                / "descriptor-assets-request.json"
            )
            if args.output.resolve() != expected:
                raise ContractError("A-R5-P2 request output path changed")
            request = build_runtime_request(
                protocol=protocol,
                data_root=args.data_root,
                implementation_archive=args.implementation_archive,
                template_manifest=args.template_manifest,
                template_import_receipt=args.template_import_receipt,
                cnos_checkout=args.cnos_checkout,
                cnos_archive=args.cnos_archive,
                dinov2_checkout=args.dinov2_checkout,
                dinov2_archive=args.dinov2_archive,
                dinov2_weights=args.dinov2_weights,
                cad_paths=_cad_paths(args.cad_root),
            )
            create_only_json(args.output, request)
            _emit(
                {
                    "status": "PASS_CREATE_ONLY_REQUEST",
                    "runtime_request_lock_sha256": request[
                        "runtime_request_lock_sha256"
                    ],
                }
            )
            return 0

        request = read_json(args.request, "A-R5-P2 request")
        if args.command == "preflight":
            value = validate_runtime_request(
                request, protocol, data_root=args.data_root
            )
            _emit(
                {
                    "status": "PASS_PREFLIGHT_NO_MODEL_IMPORT",
                    "object_ids": [item["object_id"] for item in value["catalog"]],
                    "rgba_count": 210,
                    "runtime_request_lock_sha256": request[
                        "runtime_request_lock_sha256"
                    ],
                    "model_imported": False,
                    "boundary": request["boundary"],
                }
            )
            return 0
        if args.command == "produce":
            result = run_producer(
                protocol=protocol,
                request=request,
                protocol_path=args.protocol,
                request_path=args.request,
                data_root=args.data_root,
                planned_stop_after_object_count=args.planned_stop_after_objects,
                resume=args.resume,
            )
            _emit(result)
            return 0
        result = validate_success_from_disk(
            protocol=protocol,
            request=request,
            protocol_path=args.protocol,
            request_path=args.request,
            data_root=args.data_root,
        )
        create_only_json(args.output, result)
        _emit(result)
        return 0
    except PlannedStop as exc:
        _emit({"status": "PLANNED_STOP", "error": str(exc)})
        return 75
    except ContractError as exc:
        _emit({"status": "FAIL_CLOSED", "error": str(exc)})
        return 2
    raise AssertionError("unreachable A-R5-P2 command")


__all__ = ["build_parser", "main"]
