"""Command-line entry point for the PoseLoop R3 BOP-Industrial scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    ContractError,
    assert_r3_output_scope,
    create_preflight_receipt,
    execute_official_evaluation,
    freeze_input_lock,
    load_protocol,
    write_json_atomic,
)


def _common(parser: argparse.ArgumentParser) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "protocols" / "poseloop_r3_bop_industrial_protocol.json",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo_root / "artifacts" / "r3_bop_industrial" / "eval",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python-executable", default=sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser(
        "dry-run",
        help="Audit public inputs and prediction requirements without opening evaluator labels.",
    )
    _common(dry)
    dry.add_argument("--receipt", type=Path, required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="Validate a complete prediction bundle and freeze its SHA-256 input lock.",
    )
    _common(freeze)
    freeze.add_argument("--input-lock", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run pinned official BOP evaluators after lock and label-access authorization.",
    )
    _common(evaluate)
    evaluate.add_argument("--input-lock", type=Path, required=True)
    evaluate.add_argument("--label-access-receipt", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict:
    eval_root = (
        args.output_root.resolve()
        if args.command == "evaluate"
        else args.eval_root.resolve()
    )
    return create_preflight_receipt(
        protocol_path=args.protocol.resolve(),
        dataset_root=args.dataset_root.resolve(),
        toolkit_root=args.toolkit_root.resolve(),
        input_root=args.input_root.resolve(),
        eval_root=eval_root,
        repo_root=args.repo_root.resolve(),
        python_executable=args.python_executable,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol.resolve())
        if args.command == "dry-run":
            assert_r3_output_scope(args.receipt, args.repo_root, protocol)
            receipt = _preflight(args)
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"R3 dry-run status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            if receipt["errors"]:
                for error in receipt["errors"]:
                    print(f"blocker: {error}")
            return

        if args.command == "freeze":
            assert_r3_output_scope(args.input_lock, args.repo_root, protocol)
            preflight = _preflight(args)
            lock = freeze_input_lock(preflight)
            write_json_atomic(args.input_lock.resolve(), lock)
            print(f"R3 input lock: {args.input_lock.resolve()}")
            print(f"fingerprint: {lock['fingerprint_sha256']}")
            return

        preflight = _preflight(args)
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=args.input_lock.resolve(),
            authorization_path=args.label_access_receipt.resolve(),
            dataset_root=args.dataset_root.resolve(),
            output_root=args.output_root.resolve(),
            repo_root=args.repo_root.resolve(),
            protocol=protocol,
        )
        print(f"R3 official score bundle: {args.output_root.resolve() / 'official_scores.json'}")
        print(f"single-vs-multiview delta: {result['single_vs_multiview_delta']}")
    except (ContractError, OSError, RuntimeError) as exc:
        raise SystemExit(f"R3 contract failure: {exc}") from exc


if __name__ == "__main__":
    main()
