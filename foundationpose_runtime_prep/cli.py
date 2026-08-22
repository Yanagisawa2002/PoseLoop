"""Command-line entry point for local-only FoundationPose runtime preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .a_output import export_a_live_results, export_a_results, validate_a_results
from .equivalence import compare_exact
from .common import PrepError, sha256_file, write_json_atomic
from .fixture import build_fixture
from .manifest import validate_manifest
from .live_contract import write_live_preflight
from .live_producer import run_live, run_live_smoke
from .preflight import static_preflight
from .producer import run_fixture
from .profile import fixture_samples, summarize_profile
from .visualization import build_visualization_plan, render_visualization


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--asset-root", type=Path)
    validate.add_argument("--verify-assets", action="store_true")

    fixture = sub.add_parser("build-fixture")
    fixture.add_argument("--protocol", type=Path, required=True)
    fixture.add_argument("--output-root", type=Path, required=True)

    produce = sub.add_parser("fixture-produce")
    produce.add_argument("--protocol", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--output-root", type=Path, required=True)
    produce.add_argument("--asset-root", type=Path)
    produce.add_argument("--verify-assets", action="store_true")
    produce.add_argument("--resume", action="store_true")
    produce.add_argument("--crash-after-n", type=int, default=0)

    live_preflight = sub.add_parser("live-preflight")
    live_preflight.add_argument("--protocol", type=Path, required=True)
    live_preflight.add_argument("--runtime-contract", type=Path, required=True)
    live_preflight.add_argument("--implementation-root", type=Path, required=True)
    live_preflight.add_argument("--implementation-commit", required=True)
    live_preflight.add_argument("--poseloop-runtime-root", type=Path, required=True)
    live_preflight.add_argument("--foundationpose-root", type=Path, required=True)
    live_preflight.add_argument("--gpu-index", type=int, default=0)
    live_preflight.add_argument("--output-root", type=Path, required=True)

    live_produce = sub.add_parser("live-produce")
    live_produce.add_argument("--protocol", type=Path, required=True)
    live_produce.add_argument("--runtime-contract", type=Path, required=True)
    live_produce.add_argument("--runtime-lock", type=Path, required=True)
    live_produce.add_argument("--backend-ack", type=Path, required=True)
    live_produce.add_argument("--manifest", type=Path, required=True)
    live_produce.add_argument("--asset-root", type=Path, required=True)
    live_produce.add_argument("--output-root", type=Path, required=True)
    live_produce.add_argument("--resume", action="store_true")
    live_produce.add_argument("--crash-after-n", type=int, default=0)

    live_smoke = sub.add_parser("live-backend-smoke")
    live_smoke.add_argument("--protocol", type=Path, required=True)
    live_smoke.add_argument("--runtime-contract", type=Path, required=True)
    live_smoke.add_argument("--runtime-lock", type=Path, required=True)
    live_smoke.add_argument("--backend-ack", type=Path, required=True)
    live_smoke.add_argument("--request", type=Path, required=True)
    live_smoke.add_argument("--source-manifest", type=Path, required=True)
    live_smoke.add_argument("--asset-root", type=Path, required=True)
    live_smoke.add_argument("--output-root", type=Path, required=True)
    live_smoke.add_argument("--resume", action="store_true")

    profile = sub.add_parser("fixture-profile")
    profile.add_argument("--protocol", type=Path, required=True)
    profile.add_argument("--results", type=Path, required=True)
    profile.add_argument("--output-root", type=Path, required=True)
    profile.add_argument("--warmup-repeats", type=int, default=2)
    profile.add_argument("--steady-repeats", type=int, default=5)

    equivalence = sub.add_parser("exact-equivalence")
    equivalence.add_argument("--protocol", type=Path, required=True)
    equivalence.add_argument("--manifest", type=Path, required=True)
    equivalence.add_argument("--baseline", type=Path, required=True)
    equivalence.add_argument("--candidate", type=Path, required=True)
    equivalence.add_argument("--output", type=Path, required=True)

    visualization_plan = sub.add_parser("visualization-plan")
    visualization_plan.add_argument("--protocol", type=Path, required=True)
    visualization_plan.add_argument("--manifest", type=Path, required=True)
    visualization_plan.add_argument("--baseline", type=Path, required=True)
    visualization_plan.add_argument("--improved", type=Path, required=True)
    visualization_plan.add_argument("--output", type=Path, required=True)

    visualization_render = sub.add_parser("render-visualization")
    visualization_render.add_argument("--protocol", type=Path, required=True)
    visualization_render.add_argument("--manifest", type=Path, required=True)
    visualization_render.add_argument("--asset-root", type=Path, required=True)
    visualization_render.add_argument("--baseline", type=Path, required=True)
    visualization_render.add_argument("--improved", type=Path, required=True)
    visualization_render.add_argument("--output-root", type=Path, required=True)

    export_a = sub.add_parser("export-a-results")
    export_a.add_argument("--protocol", type=Path, required=True)
    export_a.add_argument("--manifest", type=Path, required=True)
    export_a.add_argument("--producer-results", type=Path, required=True)
    export_a.add_argument("--output", type=Path, required=True)
    export_a.add_argument("--validation-output", type=Path, required=True)

    export_a_live = sub.add_parser("export-a-live-results")
    export_a_live.add_argument("--protocol", type=Path, required=True)
    export_a_live.add_argument("--manifest", type=Path, required=True)
    export_a_live.add_argument("--producer-results", type=Path, required=True)
    export_a_live.add_argument("--output", type=Path, required=True)
    export_a_live.add_argument("--validation-output", type=Path, required=True)

    validate_a = sub.add_parser("validate-a-results")
    validate_a.add_argument("--protocol", type=Path, required=True)
    validate_a.add_argument("--manifest", type=Path, required=True)
    validate_a.add_argument("--results", type=Path, required=True)
    validate_a.add_argument("--output", type=Path, required=True)

    preflight = sub.add_parser("static-preflight")
    preflight.add_argument("--protocol", type=Path, required=True)
    preflight.add_argument("--runtime-contract", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-manifest":
            manifest, items = validate_manifest(
                args.manifest,
                protocol_path=args.protocol,
                asset_root=args.asset_root,
                verify_assets=args.verify_assets,
            )
            result = {
                "status": "valid",
                "item_count": len(items),
                "mask_variant_ids": manifest["coverage"]["mask_variant_ids"],
                "manifest_sha256": sha256_file(args.manifest.resolve()),
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "build-fixture":
            result = build_fixture(
                protocol_path=args.protocol,
                output_root=args.output_root,
            )
            exit_code = 0
        elif args.command == "fixture-produce":
            exit_code, receipt = run_fixture(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                output_root=args.output_root,
                asset_root=args.asset_root,
                verify_assets=args.verify_assets,
                resume=args.resume,
                crash_after_new_successes=args.crash_after_n,
            )
            result = receipt
        elif args.command == "live-preflight":
            lock, ack = write_live_preflight(
                protocol_path=args.protocol,
                runtime_contract_path=args.runtime_contract,
                implementation_root=args.implementation_root,
                implementation_commit=args.implementation_commit,
                poseloop_runtime_root=args.poseloop_runtime_root,
                foundationpose_root=args.foundationpose_root,
                gpu_index=args.gpu_index,
                output_root=args.output_root,
            )
            result = {
                "status": "live-runtime-hash-locked",
                "runtime_lock": str(
                    (args.output_root / "live-runtime-lock.json").resolve()
                ),
                "runtime_lock_sha256": ack["runtime_lock_sha256"],
                "backend_transport_acknowledged": ack["acknowledged"],
                "implementation_sha256": lock["implementation_sha256"],
                "foundationpose_commit": lock["foundationpose_source"]["commit"],
                "gpu_uuid": lock["gpu"]["uuid"],
                "label_access_count": 0,
                "official_scorer_run": False,
            }
            exit_code = 0
        elif args.command == "live-produce":
            exit_code, receipt = run_live(
                protocol_path=args.protocol,
                runtime_contract_path=args.runtime_contract,
                manifest_path=args.manifest,
                asset_root=args.asset_root,
                output_root=args.output_root,
                runtime_lock_path=args.runtime_lock,
                backend_ack_path=args.backend_ack,
                resume=args.resume,
                crash_after_new_successes=args.crash_after_n,
            )
            result = receipt
        elif args.command == "live-backend-smoke":
            exit_code, receipt = run_live_smoke(
                protocol_path=args.protocol,
                runtime_contract_path=args.runtime_contract,
                request_path=args.request,
                source_manifest_path=args.source_manifest,
                asset_root=args.asset_root,
                output_root=args.output_root,
                runtime_lock_path=args.runtime_lock,
                backend_ack_path=args.backend_ack,
                resume=args.resume,
            )
            result = receipt
        elif args.command == "fixture-profile":
            root = args.output_root.resolve()
            samples_path = root / "profile-samples.jsonl"
            fixture_samples(
                results_path=args.results,
                output_path=samples_path,
                warmup_repeats=args.warmup_repeats,
                steady_repeats=args.steady_repeats,
            )
            summary = summarize_profile(
                protocol_path=args.protocol,
                samples_path=samples_path,
                output_json=root / "profile-summary.json",
                output_csv=root / "profile-samples.csv",
            )
            result = {
                "status": "profiled-fixture-only",
                "summary": str((root / "profile-summary.json").resolve()),
                "p50_ms": summary["steady"]["p50_ms"],
                "p95_ms": summary["steady"]["p95_ms"],
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "exact-equivalence":
            report = compare_exact(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                baseline_path=args.baseline,
                candidate_path=args.candidate,
                output_path=args.output,
            )
            result = {
                "status": (
                    "exactly-equivalent"
                    if report["passed_exact_guard"]
                    else "not-exactly-equivalent"
                ),
                "report": str(args.output.resolve()),
                "passed_exact_guard": report["passed_exact_guard"],
                "label_access_count": 0,
            }
            exit_code = 0 if report["passed_exact_guard"] else 3
        elif args.command == "visualization-plan":
            plan = build_visualization_plan(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                baseline_path=args.baseline,
                improved_path=args.improved,
                output_path=args.output,
            )
            result = {
                "status": "visualization-planned",
                "frame_count": plan["coverage"]["planned"],
                "output": str(args.output.resolve()),
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "render-visualization":
            receipt = render_visualization(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                asset_root=args.asset_root,
                baseline_path=args.baseline,
                improved_path=args.improved,
                output_root=args.output_root,
            )
            result = {
                "status": "visualization-rendered",
                "frame_count": receipt["coverage"]["rendered"],
                "receipt": str(
                    (args.output_root / "visualization-receipt.json").resolve()
                ),
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "export-a-results":
            rows, report = export_a_results(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                producer_results_path=args.producer_results,
                output_path=args.output,
                validation_output_path=args.validation_output,
            )
            result = {
                "status": "exported-and-self-validated",
                "row_count": len(rows),
                "output": str(args.output.resolve()),
                "results_sha256": report["results_sha256"],
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "export-a-live-results":
            rows, report = export_a_live_results(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                producer_results_path=args.producer_results,
                output_path=args.output,
                validation_output_path=args.validation_output,
            )
            result = {
                "status": "live-exported-and-self-validated",
                "row_count": len(rows),
                "output": str(args.output.resolve()),
                "results_sha256": report["results_sha256"],
                "synthetic_fixture_row_count": report["synthetic_fixture_row_count"],
                "label_access_count": 0,
            }
            exit_code = 0
        elif args.command == "validate-a-results":
            report = validate_a_results(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                results_path=args.results,
            )
            write_json_atomic(args.output, report)
            result = {
                "status": "valid-a-producer-output",
                "row_count": report["row_count"],
                "results_sha256": report["results_sha256"],
                "label_access_count": 0,
            }
            exit_code = 0
        else:
            report = static_preflight(
                protocol_path=args.protocol,
                runtime_contract_path=args.runtime_contract,
                output_path=args.output,
            )
            result = {
                "status": report["gpu_runtime_status"],
                "fixture_ready": report["fixture_ready"],
                "gpu_runtime_ready": report["gpu_runtime_ready"],
                "blocker_count": len(report["blockers"]),
                "label_access_count": 0,
            }
            exit_code = 0
    except PrepError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "auto_deploy": False,
                    "label_access_count": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
