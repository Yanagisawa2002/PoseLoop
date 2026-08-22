"""CLI for the unique R3-v3 XYZ-IBD supported-metrics scoring batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    ContractError,
    create_environment_smoke_receipt,
    create_evaluator_authorization,
    create_final_prescore_receipt,
    create_preflight_receipt,
    execute_official_evaluation,
    freeze_input_lock,
    load_protocol,
    write_json_atomic,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocol_default() -> Path:
    return _repo_root() / "protocols" / "poseloop_r3_bop_industrial_xyzibd_supported_v3_protocol.json"


def _shared(parser: argparse.ArgumentParser) -> None:
    root = _repo_root()
    parser.add_argument("--protocol", type=Path, default=_protocol_default())
    parser.add_argument("--readiness-lock", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--v2-invocation-started", type=Path, required=True)
    parser.add_argument("--v2-sealed-receipt", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=root / "artifacts" / "r3_bop_industrial_supported_v3" / "official_once")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--python-executable", default=sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--protocol", type=Path, default=_protocol_default())
    smoke.add_argument("--python-executable", required=True)
    smoke.add_argument("--toolkit-root", type=Path, required=True)
    smoke.add_argument("--model", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--receipt", type=Path, required=True)
    for name in ("dry-run", "freeze", "verify", "evaluate"):
        child = commands.add_parser(name)
        _shared(child)
        if name == "dry-run":
            child.add_argument("--receipt", type=Path, required=True)
        else:
            child.add_argument("--input-lock", type=Path, required=True)
            if name in ("verify", "evaluate"):
                child.add_argument("--authorization-receipt", type=Path, required=True)
            if name == "verify":
                child.add_argument("--receipt", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--protocol", type=Path, default=_protocol_default())
    authorize.add_argument("--input-lock", type=Path, required=True)
    authorize.add_argument("--receipt", type=Path, required=True)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--authorization-reference", required=True)
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict:
    return create_preflight_receipt(
        protocol_path=args.protocol.resolve(),
        readiness_lock=args.readiness_lock.resolve(),
        dataset_root=args.dataset_root.resolve(),
        toolkit_root=args.toolkit_root.resolve(),
        input_root=args.input_root.resolve(),
        v2_invocation_started=args.v2_invocation_started.resolve(),
        v2_sealed_receipt=args.v2_sealed_receipt.resolve(),
        smoke_receipt=args.smoke_receipt.resolve(),
        output_root=args.output_root.resolve(),
        repo_root=args.repo_root.resolve(),
        python_executable=args.python_executable,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol.resolve())
        if args.command == "smoke":
            receipt = create_environment_smoke_receipt(
                protocol=protocol,
                python_executable=args.python_executable,
                toolkit_root=args.toolkit_root.resolve(),
                model_path=args.model.resolve(),
                output_root=args.output_root.resolve(),
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"environment smoke status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return
        if args.command == "authorize":
            receipt = create_evaluator_authorization(
                input_lock_path=args.input_lock.resolve(),
                protocol=protocol,
                approved_by=args.approved_by,
                authorization_reference=args.authorization_reference,
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"evaluator-only authorization: {args.receipt.resolve()}")
            return
        preflight = _preflight(args)
        if args.command == "dry-run":
            write_json_atomic(args.receipt.resolve(), preflight)
            print(f"supported-v3 dry-run status: {preflight['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            if preflight["status"] != "ready":
                for error in preflight["errors"]:
                    print(f"blocker: {error}")
                raise SystemExit(2)
            return
        if args.command == "freeze":
            lock = freeze_input_lock(preflight)
            write_json_atomic(args.input_lock.resolve(), lock)
            print(f"supported-v3 input lock: {args.input_lock.resolve()}")
            print(f"fingerprint: {lock['fingerprint_sha256']}")
            print(f"implementation commit: {lock['repository']['head']}")
            print("official evaluator entered: false")
            return
        if args.command == "verify":
            receipt = create_final_prescore_receipt(
                current_preflight=preflight,
                input_lock_path=args.input_lock.resolve(),
                authorization_path=args.authorization_receipt.resolve(),
                protocol=protocol,
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"final pre-score status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=args.input_lock.resolve(),
            authorization_path=args.authorization_receipt.resolve(),
            dataset_root=args.dataset_root.resolve(),
            toolkit_root=args.toolkit_root.resolve(),
            output_root=args.output_root.resolve(),
            protocol=protocol,
        )
        print(f"official score bundle: {args.output_root.resolve() / 'official_scores.json'}")
        print(f"result status: {result['status']}")
        print(f"metrics: {result['metrics']}")
        if result["status"] != "complete":
            raise SystemExit(3)
    except (ContractError, OSError, ValueError) as exc:
        raise SystemExit(f"supported-v3 R3 contract failure: {exc}") from exc


if __name__ == "__main__":
    main()
