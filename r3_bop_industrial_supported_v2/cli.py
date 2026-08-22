"""CLI for the second one-shot XYZ-IBD-supported PoseLoop R3 evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import (
    ContractError,
    assert_namespaced_path,
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
    return (
        _repo_root()
        / "protocols"
        / "poseloop_r3_bop_industrial_xyzibd_supported_v2_protocol.json"
    )


def _preflight_arguments(parser: argparse.ArgumentParser) -> None:
    root = _repo_root()
    parser.add_argument("--protocol", type=Path, default=_protocol_default())
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=root / "protocols" / "poseloop_r3_bop_industrial_protocol.json",
    )
    parser.add_argument(
        "--prior-v1-protocol",
        type=Path,
        default=root
        / "protocols"
        / "poseloop_r3_bop_industrial_xyzibd_supported_protocol.json",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prediction-archive", type=Path, required=True)
    parser.add_argument("--dataset-verification-receipt", type=Path, required=True)
    parser.add_argument("--prior-invocation-marker", type=Path, required=True)
    parser.add_argument("--prior-failure-receipt", type=Path, required=True)
    parser.add_argument("--prior-v1-invocation-marker", type=Path, required=True)
    parser.add_argument("--prior-v1-sealed-receipt", type=Path, required=True)
    parser.add_argument("--prior-v1-raw-log", type=Path, required=True)
    parser.add_argument("--environment-smoke-receipt", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root
        / "artifacts"
        / "r3_bop_industrial_supported_v2"
        / "official_once",
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--python-executable", default=sys.executable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "smoke",
        help="Verify the exact evaluator Python/editable toolkit/renderer without entering a scorer.",
    )
    smoke.add_argument("--protocol", type=Path, default=_protocol_default())
    smoke.add_argument("--repo-root", type=Path, default=_repo_root())
    smoke.add_argument("--python-executable", required=True)
    smoke.add_argument("--toolkit-root", type=Path, required=True)
    smoke.add_argument("--model", type=Path, required=True)
    smoke.add_argument("--receipt", type=Path, required=True)

    for name, help_text in (
        ("dry-run", "Audit the frozen inputs and commands without entering a scorer."),
        ("freeze", "Freeze protocol, implementation, environment, inputs, and commands."),
        ("verify", "Perform the final read-only pre-score verification."),
        ("evaluate", "Consume the unique four-command evaluator-only authorization."),
    ):
        child = commands.add_parser(name, help=help_text)
        _preflight_arguments(child)
        if name == "dry-run":
            child.add_argument("--receipt", type=Path, required=True)
        elif name == "freeze":
            child.add_argument("--input-lock", type=Path, required=True)
        else:
            child.add_argument("--input-lock", type=Path, required=True)
            child.add_argument("--authorization-receipt", type=Path, required=True)
            if name == "verify":
                child.add_argument("--receipt", type=Path, required=True)

    authorize = commands.add_parser(
        "authorize",
        help="Bind the new user authorization to the frozen supported-v2 lock.",
    )
    authorize.add_argument("--protocol", type=Path, default=_protocol_default())
    authorize.add_argument("--repo-root", type=Path, default=_repo_root())
    authorize.add_argument("--input-lock", type=Path, required=True)
    authorize.add_argument("--receipt", type=Path, required=True)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--authorization-reference", required=True)
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict:
    return create_preflight_receipt(
        protocol_path=args.protocol.resolve(),
        source_protocol_path=args.source_protocol.resolve(),
        prior_v1_protocol_path=args.prior_v1_protocol.resolve(),
        dataset_root=args.dataset_root.resolve(),
        toolkit_root=args.toolkit_root.resolve(),
        input_root=args.input_root.resolve(),
        prediction_archive=args.prediction_archive.resolve(),
        dataset_verification_receipt=args.dataset_verification_receipt.resolve(),
        prior_invocation_marker=args.prior_invocation_marker.resolve(),
        prior_failure_receipt=args.prior_failure_receipt.resolve(),
        prior_v1_invocation_marker=args.prior_v1_invocation_marker.resolve(),
        prior_v1_sealed_receipt=args.prior_v1_sealed_receipt.resolve(),
        prior_v1_raw_log=args.prior_v1_raw_log.resolve(),
        environment_smoke_receipt=args.environment_smoke_receipt.resolve(),
        output_root=args.output_root.resolve(),
        repo_root=args.repo_root.resolve(),
        python_executable=args.python_executable,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol.resolve())
        if args.command == "smoke":
            assert_namespaced_path(args.receipt, args.repo_root, protocol, kind="pre_score")
            receipt = create_environment_smoke_receipt(
                protocol=protocol,
                python_executable=args.python_executable,
                toolkit_root=args.toolkit_root.resolve(),
                model_path=args.model.resolve(),
                repo_root=args.repo_root.resolve(),
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"environment smoke status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            print("label access count: 0")
            print("score file count: 0")
            for error in receipt["errors"]:
                print(f"blocker: {error}")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return

        if args.command == "authorize":
            assert_namespaced_path(args.input_lock, args.repo_root, protocol, kind="pre_score")
            assert_namespaced_path(args.receipt, args.repo_root, protocol, kind="pre_score")
            receipt = create_evaluator_authorization(
                input_lock_path=args.input_lock.resolve(),
                protocol=protocol,
                approved_by=args.approved_by,
                authorization_reference=args.authorization_reference,
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"evaluator-only authorization: {args.receipt.resolve()}")
            print("maximum evaluate invocations: 1")
            print("maximum invocations per command: 1")
            return

        if args.command == "dry-run":
            assert_namespaced_path(args.receipt, args.repo_root, protocol, kind="pre_score")
            receipt = _preflight(args)
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"supported-v2 dry-run status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            for error in receipt["errors"]:
                print(f"blocker: {error}")
            if receipt["status"] != "ready":
                raise SystemExit(2)
            return

        if args.command == "freeze":
            assert_namespaced_path(args.input_lock, args.repo_root, protocol, kind="pre_score")
            lock = freeze_input_lock(_preflight(args))
            write_json_atomic(args.input_lock.resolve(), lock)
            print(f"supported-v2 input lock: {args.input_lock.resolve()}")
            print(f"fingerprint: {lock['fingerprint_sha256']}")
            print(f"implementation commit: {lock['implementation_commit']}")
            print("official evaluator entered: false")
            return

        if args.command == "verify":
            for path in (args.input_lock, args.authorization_receipt, args.receipt):
                assert_namespaced_path(path, args.repo_root, protocol, kind="pre_score")
            receipt = create_final_prescore_receipt(
                current_preflight=_preflight(args),
                input_lock_path=args.input_lock.resolve(),
                authorization_path=args.authorization_receipt.resolve(),
                output_root=args.output_root.resolve(),
                protocol=protocol,
            )
            write_json_atomic(args.receipt.resolve(), receipt)
            print(f"final pre-score status: {receipt['status']}")
            print(f"receipt: {args.receipt.resolve()}")
            print("official evaluator entered: false")
            return

        preflight = _preflight(args)
        result = execute_official_evaluation(
            current_preflight=preflight,
            input_lock_path=args.input_lock.resolve(),
            authorization_path=args.authorization_receipt.resolve(),
            dataset_root=args.dataset_root.resolve(),
            toolkit_root=args.toolkit_root.resolve(),
            output_root=args.output_root.resolve(),
            repo_root=args.repo_root.resolve(),
            protocol=protocol,
        )
        print(f"official score bundle: {args.output_root.resolve() / 'official_scores.json'}")
        print(f"result status: {result['status']}")
        print(f"metrics: {result['metrics']}")
        if result["status"] != "complete":
            raise SystemExit(3)
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"supported-v2 R3 contract failure: {exc}") from exc


if __name__ == "__main__":
    main()
