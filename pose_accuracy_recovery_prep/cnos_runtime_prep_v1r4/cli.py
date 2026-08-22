"""CLI for the inert A-R4 CNOS descriptor-renderer repair route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pose_accuracy_recovery_prep.core import ContractError

from .contracts import (
    load_json_object,
    validate_route,
    validate_runtime_request,
    write_create_only,
)
from .renderer import render_one_object, run_render_attempt


def _emit(value: dict[str, Any], output: Path | None = None) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output is None:
        print(payload.decode("utf-8"), end="")
    else:
        write_create_only(output, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4",
        description=(
            "Validate and run the A-R4 hash-locked scale-then-recenter thin "
            "wrapper around pinned official CNOS rendering. This CLI never "
            "runs FastSAM, DINOv2 matching, FoundationPose, or an evaluator."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    route = commands.add_parser("validate-route")
    route.add_argument("--route", type=Path, required=True)
    route.add_argument("--repository-root", type=Path, required=True)
    route.add_argument("--output", type=Path)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--route", type=Path, required=True)
    preflight.add_argument("--request", type=Path, required=True)
    preflight.add_argument("--deployment-root", type=Path, required=True)
    preflight.add_argument("--repository-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path)

    render_all = commands.add_parser("render-all")
    render_all.add_argument("--route", type=Path, required=True)
    render_all.add_argument("--request", type=Path, required=True)
    render_all.add_argument("--deployment-root", type=Path, required=True)
    render_all.add_argument("--repository-root", type=Path, required=True)
    render_all.add_argument("--output-root", type=Path, required=True)

    render_one = commands.add_parser("render-one")
    render_one.add_argument("--route", type=Path, required=True)
    render_one.add_argument("--request", type=Path, required=True)
    render_one.add_argument("--deployment-root", type=Path, required=True)
    render_one.add_argument("--repository-root", type=Path, required=True)
    render_one.add_argument("--object-id", type=int, required=True)
    render_one.add_argument("--output-dir", type=Path, required=True)
    render_one.add_argument("--gpu-override", type=str, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        route = load_json_object(args.route, "CNOS R4 route")
        if args.command == "validate-route":
            _emit(
                validate_route(route, repository_root=args.repository_root), args.output
            )
            return 0
        request = load_json_object(args.request, "CNOS R4 render request")
        if args.command == "preflight":
            validation = validate_runtime_request(
                route,
                request,
                deployment_root=args.deployment_root,
                repository_root=args.repository_root,
            )
            _emit(
                {
                    "status": "PASS",
                    "route_lock_sha256": route["route_lock_sha256"],
                    "request_lock_sha256": request["request_lock_sha256"],
                    "implementation": request["implementation"],
                    "source_commit": route["official_source"]["commit"],
                    "source_tree": route["official_source"]["tree"],
                    "object_ids": [item[0] for item in validation.objects],
                    "python_entry": request["venv"]["python_entry"],
                    "python_target": request["venv"]["python_target"],
                    "gpu_override": request["gpu_override"],
                    "geometry_repair": "scale_then_recompute_centroid_then_recenter",
                    "execution_started": False,
                    "model_imported": False,
                    "boundary": request["boundary"],
                },
                args.output,
            )
            return 0
        if args.command == "render-all":
            _emit(
                run_render_attempt(
                    route=route,
                    request=request,
                    route_path=args.route,
                    request_path=args.request,
                    deployment_root=args.deployment_root,
                    repository_root=args.repository_root,
                    output_root=args.output_root,
                )
            )
            return 0
        _emit(
            render_one_object(
                route=route,
                request=request,
                deployment_root=args.deployment_root,
                repository_root=args.repository_root,
                object_id=args.object_id,
                output_dir=args.output_dir,
                gpu_override=args.gpu_override,
            )
        )
        return 0
    except ContractError as exc:
        parser.error(str(exc))
    return 2
