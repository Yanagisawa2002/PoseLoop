"""Command-line interface for R4-A development-only preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .core import (
    audit_frozen_predictions,
    build_prefreeze_receipt,
    load_protocol,
    read_json,
    write_json_atomic,
)
from .splitzip import Part, RangeClient, SplitZip, write_catalog


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-frozen-predictions")
    audit.add_argument("--input-root", type=Path, required=True)
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-development")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    catalog = subparsers.add_parser("catalog-archive")
    catalog.add_argument("--protocol", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol.resolve())
    if args.command == "audit-frozen-predictions":
        names = protocol["frozen_v3_predictions"]["files"]
        root = args.input_root.resolve()
        result = audit_frozen_predictions(
            coco_path=root / names["coco"],
            association_path=root / names["association"],
            single_path=root / names["single"],
            multi_path=root / names["multi"],
            provenance_path=root / names["provenance"],
            expected_sha256=protocol["frozen_v3_predictions"]["sha256"],
            frozen_public_summary=protocol["frozen_v3_predictions"]["public_aggregate_domain"],
        )
        write_json_atomic(args.output.resolve(), result)
    elif args.command == "freeze-development":
        root = args.repo_root.resolve()
        build_prefreeze_receipt(
            protocol_path=args.protocol.resolve(),
            implementation_commit=_git(root, "rev-parse", "HEAD"),
            repository_tree=_git(root, "rev-parse", "HEAD^{tree}"),
            output_path=args.output.resolve(),
        )
    elif args.command == "catalog-archive":
        network = protocol["development_split"]["network_contract"]
        parts = [
            Part(
                name=row["filename"],
                url=row["url"],
                size=int(row["bytes"]),
                sha256=row["sha256"],
            )
            for row in protocol["development_split"]["archives"]
        ]
        client = RangeClient(int(network["central_directory_max_bytes"]))
        archive = SplitZip(parts, client)
        entries, metadata = archive.read_central_directory()
        write_catalog(args.output.resolve(), entries, metadata, client)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"command": args.command, "output": str(args.output.resolve())}, sort_keys=True))
    return 0
