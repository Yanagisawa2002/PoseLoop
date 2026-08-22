"""CLI for the inert independent instance-proposal preparation contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.core import ContractError, write_json

from .contracts import (
    load_and_validate_protocol,
    load_mapping,
    validate_content_gate,
    validate_proposal_bundle,
    validate_runtime_lock,
)


def _emit(value: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        write_json(output, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.instance_proposal_v1",
        description=(
            "Validate DEVELOPMENT_ONLY independent CAD-conditioned instance-proposal "
            "locks and content gates. This CLI never runs a model or evaluator."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("validate-protocol")
    protocol.add_argument("--protocol", type=Path, required=True)
    protocol.add_argument("--output", type=Path)

    runtime = subparsers.add_parser("validate-runtime-lock")
    runtime.add_argument("--protocol", type=Path, required=True)
    runtime.add_argument("--runtime-lock", type=Path, required=True)
    runtime.add_argument("--input-root", type=Path, required=True)
    runtime.add_argument("--output", type=Path)

    proposals = subparsers.add_parser("validate-proposals")
    proposals.add_argument("--protocol", type=Path, required=True)
    proposals.add_argument("--runtime-lock", type=Path, required=True)
    proposals.add_argument("--input-root", type=Path, required=True)
    proposals.add_argument("--proposal-bundle", type=Path, required=True)
    proposals.add_argument("--asset-root", type=Path, required=True)
    proposals.add_argument("--output", type=Path)

    content = subparsers.add_parser("validate-content-gate")
    content.add_argument("--protocol", type=Path, required=True)
    content.add_argument("--runtime-lock", type=Path, required=True)
    content.add_argument("--input-root", type=Path, required=True)
    content.add_argument("--proposal-bundle", type=Path, required=True)
    content.add_argument("--proposal-root", type=Path, required=True)
    content.add_argument("--review", type=Path, required=True)
    content.add_argument("--review-root", type=Path, required=True)
    content.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        protocol, protocol_validation = load_and_validate_protocol(args.protocol)
        if args.command == "validate-protocol":
            result = protocol_validation
        else:
            runtime_lock = load_mapping(args.runtime_lock, "Runtime lock")
            if args.command == "validate-runtime-lock":
                result = validate_runtime_lock(
                    runtime_lock, protocol, input_root=args.input_root
                )
            else:
                proposal_bundle = load_mapping(args.proposal_bundle, "Proposal bundle")
                if args.command == "validate-proposals":
                    result = validate_proposal_bundle(
                        proposal_bundle,
                        protocol,
                        runtime_lock,
                        input_root=args.input_root,
                        asset_root=args.asset_root,
                    )
                else:
                    review = load_mapping(args.review, "Content gate")
                    result = validate_content_gate(
                        review,
                        protocol,
                        proposal_bundle,
                        runtime_lock=runtime_lock,
                        input_root=args.input_root,
                        proposal_root=args.proposal_root,
                        review_root=args.review_root,
                    )
        _emit(result, args.output)
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2
