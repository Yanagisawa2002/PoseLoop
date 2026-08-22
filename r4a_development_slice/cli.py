"""CLI for planning, pre-freezing, and extracting the R4-A slice."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .core import build_prefreeze_receipt
from .extractor import extract_plan
from .planner import build_plan


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--catalog", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--catalog", type=Path, required=True)
    freeze.add_argument("--v2-receipt", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--plan", type=Path, required=True)
    extract.add_argument("--prefreeze-receipt", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = build_plan(
            catalog_path=args.catalog.resolve(),
            output_path=args.output.resolve(),
            scene_ids=[0, 1, 2],
            image_ids=[0, 1, 2, 3],
            maximum_compressed_bytes=1610612736,
            maximum_target_count=128,
        )
        summary = {key: result[key] for key in ("entry_count", "target_count", "compressed_entry_bytes")}
    elif args.command == "freeze":
        root = args.repo_root.resolve()
        result = build_prefreeze_receipt(
            protocol_path=args.protocol.resolve(),
            plan_path=args.plan.resolve(),
            catalog_path=args.catalog.resolve(),
            v2_receipt_path=args.v2_receipt.resolve(),
            implementation_commit=_git(root, "rev-parse", "HEAD"),
            repository_tree=_git(root, "rev-parse", "HEAD^{tree}"),
            output_path=args.output.resolve(),
        )
        summary = {"lock_sha256": result["lock_sha256"]}
    elif args.command == "extract":
        result = extract_plan(
            protocol_path=args.protocol.resolve(),
            plan_path=args.plan.resolve(),
            output_root=args.output_root.resolve(),
            cache_root=args.cache_root.resolve(),
            receipt_path=args.prefreeze_receipt.resolve(),
            result_path=args.result.resolve(),
        )
        summary = {"entry_count": result["entry_count"], "network_bytes_downloaded": result["network_bytes_downloaded"]}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"command": args.command, **summary}, sort_keys=True))
    return 0
