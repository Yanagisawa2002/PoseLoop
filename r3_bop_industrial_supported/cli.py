"""CLI for the one-shot XYZ-IBD-supported PoseLoop R3 evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    ContractError,
    assert_namespaced_path,
    create_label_access_authorization,
    create_final_prescore_receipt,
    create_preflight_receipt,
    execute_official_evaluation,
    freeze_input_lock,
    load_protocol,
    write_json_atomic,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _preflight_arguments(parser: argparse.ArgumentParser) -> None:
    root = _repo_root()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root
        / "protocols"
        / "poseloop_r3_bop_industrial_xyzibd_supported_protocol.json",
    )
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=root / "protocols" / "poseloop_r3_bop_industrial_protocol.json",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prediction-archive", type=Path, required=True)
    parser.add_argument("--dataset-verification-receipt", type=Path, required=True)
    parser.add_argument("--prior-invocation-marker", type=Path, required=True)
    parser.add_argument("--prior-failure-receipt", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root
        / "artifacts"
        / "r3_bop_industrial_supported"
        / "official_once",
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--python-executable", default=sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dry = commands.add_parser(
        "dry-run",
        help="Audit and freeze the supported command view without entering a scorer.",
    )
    _preflight_arguments(dry)
    dry.add_argument("--receipt", type=Path, required=True)

    freeze = commands.add_parser(
        "freeze",
        help="Freeze current inputs, support matrix, hashes, commands, and output root.",
    )
    _preflight_arguments(freeze)
    freeze.add_argument("--input-lock", type=Path, required=True)

    authorize = commands.add_parser(
        "authorize",
        help="Bind the user's new one-shot authorization to the supported input lock.",
    )
    root = _repo_root()
    authorize.add_argument(
        "--protocol",
        type=Path,
        default=root
        / "protocols"
        / "poseloop_r3_bop_industrial_xyzibd_supported_protocol.json",
    )
    authorize.add_argument("--repo-root", type=Path, default=root)
    authorize.add_argument("--input-lock", type=Path, required=True)
    authorize.add_argument("--receipt", type=Path, required=True)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--authorization-reference", required=True)

    verify = commands.add_parser(
        "verify",
        help="Perform the final read-only lock/authorization check without entering a scorer.",
    )
    _preflight_arguments(verify)
    verify.add_argument("--input-lock", type=Path, required=True)
    verify.add_argument("--label-access-receipt", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate",
        help="Consume the single new evaluate authorization; no rerun is possible.",
    )
    _preflight_arguments(evaluate)
    evaluate.add_argument("--input-lock", type=Path, required=True)
    evaluate.add_argument("--label-access-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict:
    return create_preflight_receipt(
        protocol_path=args.protocol.resolve(),
        source_protocol_path=args.source_protocol.resolve(),
        dataset_root=args.dataset_root.resolve(),
        toolkit_root=args.toolkit_root.resolve(),
        input_root=args.input_root.resolve(),
        prediction_archive=args.prediction_archive.resolve(),
        dataset_verification_receipt=args.dataset_verification_receipt.resolve(),
        prior_invocation_marker=args.prior_invocation_marker.resolve(),
        prior_failure_receipt=args.prior_failure_receipt.resolve(),
        output_root=args.output_root.resolve(),
        repo_root=args.repo_root.resolve(),
        python_executable=args.python_executable,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol.resolve())
        if args.command == "authorize":
            assert_namespaced_path(
                args.input_lock, args.repo_root, protocol, kind="pre_score"
            )
            assert_namespaced_path(
                args.receipt, args.repo_root, protocol, kind="pre_score"
            )
            authorization = create_label_access_authorization(
                input_lock_path=args.input_lock.resolve(),
                approved_by=args.approved_by,
                authorization_reference=args.authorization_reference,
            )
            write_json_atomic(args.receipt.resolve(), authorization)
            print(f"authorization: {args.receipt.resolve()}")
            print("maximum evaluate invocations: 1")
            return

        if args.command == "dry-run":
            assert_namespaced_path(
                args.receipt, args.repo_root, protocol, kind="pre_score"
            )
            receipt = _preflight(args)
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"supported dry-run status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            for error in receipt["errors"]:
                print(f"blocker: {error}")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return

        if args.command == "verify":
            assert_namespaced_path(
                args.input_lock, args.repo_root, protocol, kind="pre_score"
            )
            assert_namespaced_path(
                args.label_access_receipt,
                args.repo_root,
                protocol,
                kind="pre_score",
            )
            assert_namespaced_path(
                args.receipt, args.repo_root, protocol, kind="pre_score"
            )
            preflight = _preflight(args)
            receipt = create_final_prescore_receipt(
                current_preflight=preflight,
                input_lock_path=args.input_lock.resolve(),
                authorization_path=args.label_access_receipt.resolve(),
                output_root=args.output_root.resolve(),
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"final pre-score status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            return

        if args.command == "freeze":
            assert_namespaced_path(
                args.input_lock, args.repo_root, protocol, kind="pre_score"
            )
            preflight = _preflight(args)
            lock = freeze_input_lock(preflight)
            write_json_atomic(args.input_lock.resolve(), lock)
            print(f"supported input lock: {args.input_lock.resolve()}")
            print(f"fingerprint: {lock['fingerprint_sha256']}")
            print("official evaluator entered: false")
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
        print(f"official score bundle: {args.output_root.resolve() / 'official_scores.json'}")
        print(f"metrics: {result['metrics']}")
    except (ContractError, OSError, RuntimeError) as exc:
        raise SystemExit(f"supported R3 contract failure: {exc}") from exc


if __name__ == "__main__":
    main()
