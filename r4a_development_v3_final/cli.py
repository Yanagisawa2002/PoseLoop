"""Command line entrypoints for the frozen R4-A v3 final development evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .core import prepare, run_official_once, summarize, visualize


def _path(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    for name in (
        "protocol_path",
        "repo_root",
        "workload_path",
        "coco_path",
        "inference_root",
        "c_results_path",
        "c_output_archive_path",
        "c_output_descriptor_path",
        "evaluator_bundle_archive_path",
        "evaluator_stage_root",
        "models_eval_source",
        "toolkit_root",
        "venv_python",
        "output_root",
        "pre_score_path",
    ):
        _path(prepare_parser, name)

    once_parser = commands.add_parser("run-official-once")
    for name in ("protocol_path", "pre_score_path", "output_path", "log_root"):
        _path(once_parser, name)

    summary_parser = commands.add_parser("summarize")
    for name in ("protocol_path", "pre_score_path", "official_once_path", "output_path"):
        _path(summary_parser, name)

    visual_parser = commands.add_parser("visualize")
    for name in ("protocol_path", "pre_score_path", "output_root", "output_path"):
        _path(visual_parser, name)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {key: value for key, value in vars(args).items() if key != "command"}
    try:
        if args.command == "prepare":
            value = prepare(**kwargs)
        elif args.command == "run-official-once":
            value = run_official_once(**kwargs)
        elif args.command == "summarize":
            value = summarize(**kwargs)
        else:
            value = visualize(**kwargs)
    except Exception as error:  # noqa: BLE001 - CLI must seal every failure without a traceback.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": " ".join(str(error).split())[:1600],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    result = {
        "status": "complete",
        "command": args.command,
        "schema_version": value.get("schema_version"),
        "protocol_id": value.get("protocol_id"),
        "lock_sha256": value.get("lock_sha256"),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = ["main", "parse_args"]
