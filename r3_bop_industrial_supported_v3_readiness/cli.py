"""CLI for R3-v3 XYZ-IBD label readiness; intentionally no score command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    ContractError,
    create_readiness_receipt,
    freeze_readiness_lock,
    load_protocol,
    prepare_overlay,
    run_coco_generation,
    sha256_file,
    write_json_atomic,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocol_default() -> Path:
    return _repo_root() / "protocols" / "poseloop_r3_bop_industrial_xyzibd_label_readiness_v3_protocol.json"


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=_protocol_default())
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--v2-invocation-started", type=Path, required=True)
    parser.add_argument("--v2-sealed-receipt", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build overlay and derive COCO GT; never score.")
    _shared(prepare)
    prepare.add_argument("--receipt", type=Path, required=True)
    audit = commands.add_parser("audit", help="Read-only full label-readiness audit; never score.")
    _shared(audit)
    audit.add_argument("--receipt", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="Freeze a ready label-asset lock; never score.")
    _shared(freeze)
    freeze.add_argument("--lock", type=Path, required=True)
    return parser.parse_args(argv)


def _receipt(args: argparse.Namespace) -> dict:
    return create_readiness_receipt(
        protocol_path=args.protocol.resolve(),
        dataset_root=args.dataset_root.resolve(),
        archive=args.archive.resolve(),
        toolkit_root=args.toolkit_root.resolve(),
        input_root=args.input_root.resolve(),
        v2_invocation_started=args.v2_invocation_started.resolve(),
        v2_sealed_receipt=args.v2_sealed_receipt.resolve(),
        evaluator_root=args.evaluator_root.resolve(),
        python_executable=args.python_executable,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol_path = args.protocol.resolve()
        protocol = load_protocol(protocol_path)
        if args.command == "prepare":
            overlay = prepare_overlay(
                source_dataset_root=args.dataset_root.resolve(),
                evaluator_root=args.evaluator_root.resolve(),
                protocol=protocol,
                protocol_sha256=sha256_file(protocol_path),
            )
            generation = run_coco_generation(
                python_executable=args.python_executable,
                toolkit_root=args.toolkit_root.resolve(),
                evaluator_root=args.evaluator_root.resolve(),
                protocol=protocol,
            )
            receipt = _receipt(args)
            receipt["overlay_preparation"] = overlay
            receipt["coco_generation"] = generation
            if generation.get("status") == "failed":
                receipt["errors"].extend(generation.get("errors", ["COCO GT derivation failed"]))
                receipt["status"] = "blocked"
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"label readiness status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            print("scoring authorization created: false")
            if receipt["status"] != "ready":
                for error in receipt["errors"]:
                    print(f"blocker: {error}")
                raise SystemExit(2)
            return
        receipt = _receipt(args)
        if args.command == "audit":
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"label readiness status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return
        lock = freeze_readiness_lock(receipt)
        write_json_atomic(args.lock.resolve(), lock)
        print(f"label readiness lock: {args.lock.resolve()}")
        print(f"fingerprint: {lock['fingerprint_sha256']}")
        print("official evaluator entered: false")
    except (ContractError, OSError, ValueError) as exc:
        raise SystemExit(f"R3-v3 label-readiness contract failure: {exc}") from exc


if __name__ == "__main__":
    main()
