"""Freeze and run the R4-A development readiness v2 audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .core import audit_readiness, build_prefreeze_receipt


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extraction-result", type=Path, required=True)
    parser.add_argument("--slice-prefreeze", type=Path, required=True)
    parser.add_argument("--v1-protocol", type=Path, required=True)
    parser.add_argument("--v1-prefreeze", type=Path, required=True)
    parser.add_argument("--v1-job-log", type=Path, required=True)
    parser.add_argument("--v1-job-exit", type=Path, required=True)
    parser.add_argument("--v1-failure-receipt", type=Path, required=True)
    parser.add_argument("--schema-diagnosis", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    _common_inputs(freeze)
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--models-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    _common_inputs(audit)
    audit.add_argument("--repo-root", type=Path, required=True)
    audit.add_argument("--prefreeze-receipt", type=Path, required=True)
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--models-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--target-manifest", type=Path, required=True)
    audit.add_argument("--label-access", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "protocol_path": args.protocol.resolve(),
        "plan_path": args.plan.resolve(),
        "extraction_result_path": args.extraction_result.resolve(),
        "slice_prefreeze_path": args.slice_prefreeze.resolve(),
        "v1_protocol_path": args.v1_protocol.resolve(),
        "v1_prefreeze_path": args.v1_prefreeze.resolve(),
        "v1_job_log_path": args.v1_job_log.resolve(),
        "v1_job_exit_path": args.v1_job_exit.resolve(),
        "v1_failure_receipt_path": args.v1_failure_receipt.resolve(),
        "schema_diagnosis_path": args.schema_diagnosis.resolve(),
    }
    if args.command == "freeze":
        root = args.repo_root.resolve()
        result = build_prefreeze_receipt(
            **common,
            models_root=args.models_root.resolve(),
            implementation_commit=_git(root, "rev-parse", "HEAD"),
            repository_tree=_git(root, "rev-parse", "HEAD^{tree}"),
            repository_tracked_clean=not _git(root, "status", "--short", "--untracked-files=no"),
            output_path=args.output.resolve(),
        )
        summary = {"lock_sha256": result["lock_sha256"]}
    elif args.command == "audit":
        result = audit_readiness(
            **common,
            repo_root=args.repo_root.resolve(),
            prefreeze_receipt_path=args.prefreeze_receipt.resolve(),
            data_root=args.data_root.resolve(),
            models_root=args.models_root.resolve(),
            output_path=args.output.resolve(),
            target_manifest_path=args.target_manifest.resolve(),
            label_access_path=args.label_access.resolve(),
        )
        summary = {
            "status": result["status"],
            "gate_pass": result["gate_pass"],
            "target_count": result["target_count"],
        }
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"command": args.command, **summary}, sort_keys=True))
    return 0
