"""CLI for local A-R5 contract checks and future reviewed producer execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.core import ContractError, canonical_sha256

from .contracts import (
    BOUNDARY_ZERO,
    load_json_object,
    validate_frame_manifest,
    validate_output_bundle,
    validate_protocol,
    validate_runtime_request,
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _emit(value: MappingLike, output: Path | None) -> None:
    if output is None:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    payload = _json_bytes(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ContractError(f"A-R5 CLI output is create-only: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise ContractError(f"A-R5 CLI output raced: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)


MappingLike = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.instance_proposal_v1r5",
        description=(
            "Validate or, after separate review/authorization, execute A-R5 "
            "label-blind FastSAM instance proposals with all-catalog DINOv2 matching."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser("validate-protocol")
    protocol.add_argument("--protocol", type=Path, required=True)
    protocol.add_argument("--repository-root", type=Path)
    protocol.add_argument("--output", type=Path)

    manifest = commands.add_parser("validate-manifest")
    manifest.add_argument("--protocol", type=Path, required=True)
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path)

    for name in ("validate-request", "preflight"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output", type=Path)

    run = commands.add_parser("run-producer")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--planned-crash-after-frames", type=int)
    run.add_argument("--resume", action="store_true")

    output = commands.add_parser("validate-output")
    output.add_argument("--protocol", type=Path, required=True)
    output.add_argument("--manifest", type=Path, required=True)
    output.add_argument("--request", type=Path, required=True)
    output.add_argument("--producer-manifest", type=Path, required=True)
    output.add_argument("--run-receipt", type=Path, required=True)
    output.add_argument("--data-root", type=Path, required=True)
    output.add_argument("--output-root", type=Path, required=True)
    output.add_argument("--output", type=Path)
    return parser


def _inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_json_object(args.protocol, "A-R5 protocol")
    manifest = load_json_object(args.manifest, "A-R5 frame manifest")
    request = load_json_object(args.request, "A-R5 runtime request")
    return protocol, manifest, request


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-protocol":
            protocol = load_json_object(args.protocol, "A-R5 protocol")
            validated = validate_protocol(
                protocol, repository_root=args.repository_root
            )
            _emit(
                {
                    "status": "PASS",
                    "protocol_id": validated["protocol_id"],
                    "protocol_lock_sha256": validated["protocol_lock_sha256"],
                    "server_connected": False,
                    "model_run": False,
                },
                args.output,
            )
            return 0
        if args.command == "validate-manifest":
            protocol = load_json_object(args.protocol, "A-R5 protocol")
            manifest = load_json_object(args.manifest, "A-R5 frame manifest")
            validated = validate_frame_manifest(
                manifest, protocol, data_root=args.data_root
            )
            _emit(
                {
                    "status": "PASS",
                    "frame_count": validated["frame_count"],
                    "scene_count": validated["scene_count"],
                    "frame_manifest_lock_sha256": validated[
                        "frame_manifest_lock_sha256"
                    ],
                },
                args.output,
            )
            return 0
        protocol, manifest, request = _inputs(args)
        if args.command in {"validate-request", "preflight"}:
            validated = validate_runtime_request(
                request,
                protocol,
                manifest,
                data_root=args.data_root,
            )
            result = {
                "schema_version": (
                    "poseloop.pose-accuracy-recovery.instance-preflight-receipt.v1r5"
                ),
                "status": "PASS",
                "protocol_lock_sha256": protocol["protocol_lock_sha256"],
                "frame_manifest_lock_sha256": manifest["frame_manifest_lock_sha256"],
                "runtime_request_lock_sha256": validated["runtime_request_lock_sha256"],
                "frame_count": 10,
                "scene_count": 5,
                "catalog_object_ids": [1, 2, 4, 5, 6],
                "boundary": dict(BOUNDARY_ZERO),
                "model_imported": False,
                "producer_started": False,
                "preflight_receipt_lock_sha256": "pending",
            }
            result["preflight_receipt_lock_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in result.items()
                    if key != "preflight_receipt_lock_sha256"
                }
            )
            _emit(result, args.output)
            return 0
        if args.command == "run-producer":
            # Importing the runtime adapter is deliberately delayed until every
            # request/input hash and boundary contract has passed.
            validate_runtime_request(
                request, protocol, manifest, data_root=args.data_root
            )
            from .adapter import OfficialCnosCatalogAdapter
            from .producer import run_producer

            backend = OfficialCnosCatalogAdapter(
                data_root=args.data_root,
                request=request,
                device=request["runtime"]["device"],
            )
            bundle, receipt = run_producer(
                protocol=protocol,
                manifest=manifest,
                request=request,
                data_root=args.data_root,
                output_root=args.output_root,
                backend=backend,
                planned_crash_after_frames=args.planned_crash_after_frames,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "output_bundle_lock_sha256": bundle[
                            "output_bundle_lock_sha256"
                        ],
                        "run_receipt_lock_sha256": receipt["run_receipt_lock_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        bundle = load_json_object(args.producer_manifest, "A-R5 producer manifest")
        receipt = load_json_object(args.run_receipt, "A-R5 run receipt")
        validate_output_bundle(
            bundle,
            protocol,
            manifest,
            request,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        from .producer import validate_run_receipt

        validate_run_receipt(receipt, bundle, output_root=args.output_root)
        _emit(
            {
                "status": "PASS",
                "frame_count": bundle["frame_count"],
                "scene_count": bundle["scene_count"],
                "output_bundle_lock_sha256": bundle["output_bundle_lock_sha256"],
                "run_receipt_lock_sha256": receipt["run_receipt_lock_sha256"],
                "disk_assets_reverified": True,
                "boundary": dict(BOUNDARY_ZERO),
            },
            args.output,
        )
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
