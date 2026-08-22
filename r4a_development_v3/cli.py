"""CLI for the frozen R4-A v3 multi-object development workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .assets import extract_assets
from .bundle import build_evaluator_bundle, build_inference_bundle
from .contract import (
    build_asset_plan,
    build_asset_prefreeze_receipt,
    build_id_plan,
    build_id_prefreeze_receipt,
)
from .id_job import discover_ids


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _repo_values(root: Path) -> dict[str, object]:
    return {
        "implementation_commit": _git(root, "rev-parse", "HEAD"),
        "repository_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "repository_tracked_clean": not bool(_git(root, "status", "--short", "--untracked-files=no")),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_id = commands.add_parser("plan-ids")
    plan_id.add_argument("--protocol", type=Path, required=True)
    plan_id.add_argument("--catalog", type=Path, required=True)
    plan_id.add_argument("--output", type=Path, required=True)

    freeze_id = commands.add_parser("freeze-ids")
    freeze_id.add_argument("--repo-root", type=Path, required=True)
    freeze_id.add_argument("--protocol", type=Path, required=True)
    freeze_id.add_argument("--catalog", type=Path, required=True)
    freeze_id.add_argument("--id-plan", type=Path, required=True)
    freeze_id.add_argument("--v2-protocol", type=Path, required=True)
    freeze_id.add_argument("--v2-prefreeze", type=Path, required=True)
    freeze_id.add_argument("--v2-job-log", type=Path, required=True)
    freeze_id.add_argument("--v2-job-exit", type=Path, required=True)
    freeze_id.add_argument("--v2-failure-receipt", type=Path, required=True)
    freeze_id.add_argument("--v2-object-diagnosis", type=Path, required=True)
    freeze_id.add_argument("--output", type=Path, required=True)

    discover = commands.add_parser("discover-ids")
    discover.add_argument("--protocol", type=Path, required=True)
    discover.add_argument("--id-plan", type=Path, required=True)
    discover.add_argument("--prefreeze", type=Path, required=True)
    discover.add_argument("--output-root", type=Path, required=True)
    discover.add_argument("--cache-root", type=Path, required=True)
    discover.add_argument("--exact-selection", type=Path, required=True)
    discover.add_argument("--result", type=Path, required=True)

    plan_assets = commands.add_parser("plan-assets")
    plan_assets.add_argument("--protocol", type=Path, required=True)
    plan_assets.add_argument("--catalog", type=Path, required=True)
    plan_assets.add_argument("--exact-selection", type=Path, required=True)
    plan_assets.add_argument("--id-result", type=Path, required=True)
    plan_assets.add_argument("--output", type=Path, required=True)

    freeze_assets = commands.add_parser("freeze-assets")
    freeze_assets.add_argument("--repo-root", type=Path, required=True)
    freeze_assets.add_argument("--protocol", type=Path, required=True)
    freeze_assets.add_argument("--catalog", type=Path, required=True)
    freeze_assets.add_argument("--exact-selection", type=Path, required=True)
    freeze_assets.add_argument("--id-result", type=Path, required=True)
    freeze_assets.add_argument("--asset-plan", type=Path, required=True)
    freeze_assets.add_argument("--id-prefreeze", type=Path, required=True)
    freeze_assets.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser("extract-assets")
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--asset-plan", type=Path, required=True)
    extract.add_argument("--prefreeze", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--result", type=Path, required=True)

    inference = commands.add_parser("bundle-inference")
    inference.add_argument("--repo-root", type=Path, required=True)
    inference.add_argument("--protocol", type=Path, required=True)
    inference.add_argument("--exact-selection", type=Path, required=True)
    inference.add_argument("--asset-root", type=Path, required=True)
    inference.add_argument("--models-root", type=Path, required=True)
    inference.add_argument("--stage-root", type=Path, required=True)
    inference.add_argument("--archive", type=Path, required=True)
    inference.add_argument("--descriptor", type=Path, required=True)

    evaluator = commands.add_parser("bundle-evaluator")
    evaluator.add_argument("--protocol", type=Path, required=True)
    evaluator.add_argument("--exact-selection", type=Path, required=True)
    evaluator.add_argument("--asset-root", type=Path, required=True)
    evaluator.add_argument("--models-root", type=Path, required=True)
    evaluator.add_argument("--models-info", type=Path, required=True)
    evaluator.add_argument("--stage-root", type=Path, required=True)
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan-ids":
        result = build_id_plan(protocol_path=args.protocol.resolve(), catalog_path=args.catalog.resolve(), output_path=args.output.resolve())
    elif args.command == "freeze-ids":
        result = build_id_prefreeze_receipt(
            protocol_path=args.protocol.resolve(),
            catalog_path=args.catalog.resolve(),
            id_plan_path=args.id_plan.resolve(),
            v2_protocol_path=args.v2_protocol.resolve(),
            v2_prefreeze_path=args.v2_prefreeze.resolve(),
            v2_job_log_path=args.v2_job_log.resolve(),
            v2_job_exit_path=args.v2_job_exit.resolve(),
            v2_failure_receipt_path=args.v2_failure_receipt.resolve(),
            v2_object_diagnosis_path=args.v2_object_diagnosis.resolve(),
            **_repo_values(args.repo_root.resolve()),
            output_path=args.output.resolve(),
        )
    elif args.command == "discover-ids":
        result = discover_ids(
            protocol_path=args.protocol.resolve(),
            id_plan_path=args.id_plan.resolve(),
            prefreeze_path=args.prefreeze.resolve(),
            output_root=args.output_root.resolve(),
            cache_root=args.cache_root.resolve(),
            exact_selection_path=args.exact_selection.resolve(),
            result_path=args.result.resolve(),
        )
    elif args.command == "plan-assets":
        result = build_asset_plan(
            protocol_path=args.protocol.resolve(),
            catalog_path=args.catalog.resolve(),
            exact_selection_path=args.exact_selection.resolve(),
            id_result_path=args.id_result.resolve(),
            output_path=args.output.resolve(),
        )
    elif args.command == "freeze-assets":
        result = build_asset_prefreeze_receipt(
            protocol_path=args.protocol.resolve(),
            catalog_path=args.catalog.resolve(),
            exact_selection_path=args.exact_selection.resolve(),
            id_result_path=args.id_result.resolve(),
            asset_plan_path=args.asset_plan.resolve(),
            id_prefreeze_path=args.id_prefreeze.resolve(),
            **_repo_values(args.repo_root.resolve()),
            output_path=args.output.resolve(),
        )
    elif args.command == "extract-assets":
        result = extract_assets(
            protocol_path=args.protocol.resolve(),
            asset_plan_path=args.asset_plan.resolve(),
            prefreeze_path=args.prefreeze.resolve(),
            output_root=args.output_root.resolve(),
            cache_root=args.cache_root.resolve(),
            result_path=args.result.resolve(),
        )
    elif args.command == "bundle-inference":
        result = build_inference_bundle(
            protocol_path=args.protocol.resolve(),
            exact_selection_path=args.exact_selection.resolve(),
            asset_root=args.asset_root.resolve(),
            models_root=args.models_root.resolve(),
            stage_root=args.stage_root.resolve(),
            archive_path=args.archive.resolve(),
            descriptor_path=args.descriptor.resolve(),
            implementation_commit=_git(args.repo_root.resolve(), "rev-parse", "HEAD"),
        )
    elif args.command == "bundle-evaluator":
        result = build_evaluator_bundle(
            protocol_path=args.protocol.resolve(),
            exact_selection_path=args.exact_selection.resolve(),
            asset_root=args.asset_root.resolve(),
            models_root=args.models_root.resolve(),
            models_info_path=args.models_info.resolve(),
            stage_root=args.stage_root.resolve(),
            archive_path=args.archive.resolve(),
            manifest_path=args.manifest.resolve(),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    summary = {key: result[key] for key in result if key in {"entry_count", "target_count", "gate_pass", "lock_sha256", "archive", "item_count"}}
    print(json.dumps({"command": args.command, **summary}, sort_keys=True))
    return 0


__all__ = ["main"]
