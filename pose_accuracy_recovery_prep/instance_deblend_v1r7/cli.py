"""Command line checks for the PREP-only A-R7 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pose_accuracy_recovery_prep.core import read_json

from .contracts import validate_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.instance_deblend_v1r7"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser(
        "contract-check",
        help="validate PREP boundary; does not authorize model or replay execution",
    )
    check.add_argument("--protocol", type=Path, required=True)
    check.add_argument("--repository-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "contract-check":
        protocol = read_json(args.protocol)
        validated = validate_protocol(
            protocol, repository_root=args.repository_root.resolve()
        )
        print(
            json.dumps(
                {
                    "status": "PASS_PREP_CONTRACT",
                    "protocol_id": validated["protocol_id"],
                    "formal_workload_permitted": False,
                    "scene9_replay_permitted": False,
                    "blockers": validated["readiness"]["blockers"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


__all__ = ["main"]
