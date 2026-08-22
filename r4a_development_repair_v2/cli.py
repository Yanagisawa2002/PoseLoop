"""CLI for the immutable R4-A v2 range-metadata repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from r4a_development_repair.splitzip import Part, SplitZip, write_catalog

from .core import build_prefreeze_receipt, load_protocol
from .splitzip import ChunkedRangeClient


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
    freeze = subparsers.add_parser("freeze-development")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--base-protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    catalog = subparsers.add_parser("catalog-archive")
    catalog.add_argument("--protocol", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_protocol(args.protocol.resolve())
    if args.command == "freeze-development":
        root = args.repo_root.resolve()
        build_prefreeze_receipt(
            protocol_path=args.protocol.resolve(),
            base_protocol_path=args.base_protocol.resolve(),
            implementation_commit=_git(root, "rev-parse", "HEAD"),
            repository_tree=_git(root, "rev-parse", "HEAD^{tree}"),
            output_path=args.output.resolve(),
        )
    elif args.command == "catalog-archive":
        network = protocol["development_split"]["network_contract"]
        parts = [
            Part(row["filename"], row["url"], int(row["bytes"]), row["sha256"])
            for row in protocol["development_split"]["archives"]
        ]
        client = ChunkedRangeClient(
            int(network["central_directory_max_bytes"]),
            int(network["range_chunk_bytes"]),
        )
        entries, metadata = SplitZip(parts, client).read_central_directory()
        write_catalog(args.output.resolve(), entries, metadata, client)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"command": args.command, "output": str(args.output.resolve())}, sort_keys=True))
    return 0
