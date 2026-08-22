"""CPU-only PREP CLI; no command connects to a server or invokes a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .c_handoff import export_c_handoff, load_and_validate_c_handoff
from .c_results import validate_c_results
from .core import (
    ContractError,
    build_run_plan,
    canonical_sha256,
    export_producer_manifest,
    load_and_validate_manifest,
    read_json,
    validate_producer_manifest,
    write_json,
    write_jsonl,
)
from .diagnostics import (
    evaluate_prepared_diagnostics,
    load_protocol,
    prepare_diagnostics,
    run_fixture_diagnostics,
    save_prepared_grid,
)
from .reporting import read_jsonl, synthetic_prediction_rows, write_report
from .visualization import build_visualization_plan


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--version", action="version", version="pose-accuracy-recovery-prep-v1"
    )
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-manifest", help="validate unified manifest and all local asset hashes"
    )
    _manifest_arguments(validate)
    validate.add_argument("--output", type=Path, required=True)

    validate_producer = commands.add_parser(
        "validate-producer",
        help="validate a label-free exported manifest and its local asset hashes",
    )
    _manifest_arguments(validate_producer)
    validate_producer.add_argument("--output", type=Path, required=True)

    export = commands.add_parser(
        "export-producer",
        help="write the legacy predicted-only label-free manifest",
    )
    _manifest_arguments(export)
    export.add_argument("--output", type=Path, required=True)

    export_c = commands.add_parser(
        "export-c-handoff",
        help=(
            "export canonical five-variant DEVELOPMENT_ONLY runtime bundle; "
            "contains opaque GT-derived controls but no raw GT paths"
        ),
    )
    _protocol_arguments(export_c)
    export_c.add_argument("--implementation-commit", required=True)
    export_c.add_argument("--implementation-sha256", required=True)
    export_c.add_argument("--model-sha256", required=True)
    export_c.add_argument("--refiner-checkpoint-sha256", required=True)
    export_c.add_argument("--scorer-checkpoint-sha256", required=True)
    export_c.add_argument("--output-root", type=Path, required=True)

    validate_c = commands.add_parser(
        "validate-c-handoff",
        help="strictly validate a canonical runtime-isolated C handoff",
    )
    validate_c.add_argument("--manifest", type=Path, required=True)
    validate_c.add_argument("--bundle-root", type=Path, required=True)
    validate_c.add_argument("--output", type=Path, required=True)

    validate_results = commands.add_parser(
        "validate-c-results",
        help="strictly validate exact C producer-output v1 rows and assets",
    )
    validate_results.add_argument("--handoff-manifest", type=Path, required=True)
    validate_results.add_argument("--handoff-root", type=Path, required=True)
    validate_results.add_argument("--results", type=Path, required=True)
    validate_results.add_argument("--result-root", type=Path, required=True)
    validate_results.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser(
        "plan", help="materialize deterministic per-mask run rows"
    )
    _manifest_arguments(plan)
    plan.add_argument(
        "--namespace-role", choices=("producer", "evaluator-only"), required=True
    )
    plan.add_argument("--variant", action="append", dest="variants")
    plan.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare-diagnostics", help="write frozen evaluator-only GT perturbations"
    )
    _protocol_arguments(prepare)
    prepare.add_argument("--namespace-role", choices=("EVALUATOR_ONLY",), required=True)
    prepare.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate-diagnostics",
        help="check real scorer scores and refiner traces against the frozen grid",
    )
    _protocol_arguments(evaluate)
    evaluate.add_argument(
        "--namespace-role", choices=("EVALUATOR_ONLY",), required=True
    )
    evaluate.add_argument("--grid", type=Path, required=True)
    evaluate.add_argument("--scorer-output", type=Path, required=True)
    evaluate.add_argument("--refiner-output", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)

    report = commands.add_parser(
        "report",
        help="evaluate prepared development rows and normalize official capabilities",
    )
    _protocol_arguments(report)
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--official-metrics", type=Path)
    report.add_argument("--output-root", type=Path, required=True)

    visualization = commands.add_parser(
        "visualization-plan", help="write separated producer/evaluator layer checklist"
    )
    _manifest_arguments(visualization)
    visualization.add_argument("--output", type=Path, required=True)

    dry_run = commands.add_parser(
        "dry-run",
        help="run contract, metric, and diagnostic code on committed synthetic fixture",
    )
    _protocol_arguments(dry_run)
    dry_run.add_argument("--output-root", type=Path, required=True)
    return root


def _manifest_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--data-root", type=Path, required=True)


def _protocol_arguments(command: argparse.ArgumentParser) -> None:
    _manifest_arguments(command)
    command.add_argument("--protocol", type=Path, required=True)


def _inputs(args: argparse.Namespace):
    manifest, validation = load_and_validate_manifest(
        args.manifest.resolve(), data_root=args.data_root.resolve(), verify_hashes=True
    )
    return manifest, validation


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-manifest":
        _, validation = _inputs(args)
        write_json(args.output.resolve(), validation)
        summary = validation
    elif args.command == "validate-producer":
        producer_manifest = read_json(args.manifest.resolve())
        if not isinstance(producer_manifest, dict):
            raise ContractError("Producer manifest must be a JSON object")
        summary = validate_producer_manifest(
            producer_manifest,
            data_root=args.data_root.resolve(),
            verify_hashes=True,
        )
        write_json(args.output.resolve(), summary)
    elif args.command == "export-producer":
        manifest, _ = _inputs(args)
        output = export_producer_manifest(manifest)
        producer_validation = validate_producer_manifest(
            output, data_root=args.data_root.resolve(), verify_hashes=True
        )
        write_json(args.output.resolve(), output)
        summary = producer_validation
    elif args.command == "export-c-handoff":
        summary = export_c_handoff(
            manifest_path=args.manifest.resolve(),
            data_root=args.data_root.resolve(),
            protocol_path=args.protocol.resolve(),
            output_root=args.output_root.resolve(),
            implementation_commit=args.implementation_commit,
            implementation_sha256=args.implementation_sha256,
            model_sha256=args.model_sha256,
            refiner_checkpoint_sha256=args.refiner_checkpoint_sha256,
            scorer_checkpoint_sha256=args.scorer_checkpoint_sha256,
        )
    elif args.command == "validate-c-handoff":
        _, summary = load_and_validate_c_handoff(
            args.manifest.resolve(),
            bundle_root=args.bundle_root.resolve(),
            verify_assets=True,
        )
        write_json(args.output.resolve(), summary)
    elif args.command == "validate-c-results":
        summary = validate_c_results(
            handoff_manifest_path=args.handoff_manifest.resolve(),
            handoff_bundle_root=args.handoff_root.resolve(),
            results_path=args.results.resolve(),
            result_root=args.result_root.resolve(),
            output_path=args.output.resolve(),
        )
    elif args.command == "plan":
        manifest, _ = _inputs(args)
        rows = build_run_plan(
            manifest, namespace_role=args.namespace_role, variants=args.variants
        )
        write_jsonl(args.output.resolve(), rows)
        summary = {"row_count": len(rows), "namespace_role": args.namespace_role}
    elif args.command == "prepare-diagnostics":
        protocol = load_protocol(args.protocol.resolve())
        manifest, _ = _inputs(args)
        rows = prepare_diagnostics(
            protocol,
            manifest,
            data_root=args.data_root.resolve(),
            namespace_role=args.namespace_role,
        )
        summary = save_prepared_grid(args.output.resolve(), rows)
    elif args.command == "evaluate-diagnostics":
        protocol = load_protocol(args.protocol.resolve())
        manifest, _ = _inputs(args)
        summary = evaluate_prepared_diagnostics(
            protocol,
            manifest,
            data_root=args.data_root.resolve(),
            grid_rows=read_jsonl(args.grid.resolve()),
            scorer_rows=read_jsonl(args.scorer_output.resolve()),
            refiner_traces=read_jsonl(args.refiner_output.resolve()),
            namespace_role=args.namespace_role,
            output_root=args.output_root.resolve(),
        )
    elif args.command == "report":
        protocol = load_protocol(args.protocol.resolve())
        manifest, _ = _inputs(args)
        official = (
            read_json(args.official_metrics.resolve())
            if args.official_metrics
            else None
        )
        summary = write_report(
            protocol,
            manifest,
            read_jsonl(args.predictions.resolve()),
            data_root=args.data_root.resolve(),
            output_root=args.output_root.resolve(),
            official_payload=official,
            fixture_mode=False,
        )
    elif args.command == "visualization-plan":
        manifest, _ = _inputs(args)
        summary = build_visualization_plan(manifest)
        write_json(args.output.resolve(), summary)
    elif args.command == "dry-run":
        summary = _dry_run(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"command": args.command, "summary": summary}, sort_keys=True))
    return 0


def _dry_run(args: argparse.Namespace) -> dict:
    protocol = load_protocol(args.protocol.resolve())
    manifest, validation = _inputs(args)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest-validation.json", validation)

    producer = export_producer_manifest(manifest)
    producer_validation = validate_producer_manifest(
        producer, data_root=args.data_root.resolve(), verify_hashes=True
    )
    write_json(root / "producer-manifest.json", producer)
    write_json(root / "producer-validation.json", producer_validation)
    producer_plan = build_run_plan(manifest, namespace_role="producer")
    evaluator_plan = build_run_plan(manifest, namespace_role="evaluator-only")
    write_jsonl(root / "producer-plan.jsonl", producer_plan)
    write_jsonl(root / "evaluator-plan.jsonl", evaluator_plan)

    diagnostics = run_fixture_diagnostics(
        protocol,
        manifest,
        data_root=args.data_root.resolve(),
        output_root=root / "diagnostics",
    )
    predictions = synthetic_prediction_rows(
        protocol, manifest, data_root=args.data_root.resolve()
    )
    write_jsonl(root / "synthetic-predictions.jsonl", predictions)
    metrics = write_report(
        protocol,
        manifest,
        predictions,
        data_root=args.data_root.resolve(),
        output_root=root / "metrics",
        official_payload=None,
        fixture_mode=True,
    )
    visualization = build_visualization_plan(manifest)
    write_json(root / "visualization-plan.json", visualization)
    summary = {
        "schema_version": "poseloop.pose-accuracy-recovery.dry-run.v1",
        "execution_mode": "CPU_SYNTHETIC_FIXTURE_DRY_RUN",
        "auto_deploy": False,
        "server_connection_count": 0,
        "download_count": 0,
        "model_invocation_count": 0,
        "sealed_split_access_count": 0,
        "sample_count": validation["sample_count"],
        "producer_plan_rows": len(producer_plan),
        "evaluator_plan_rows": len(evaluator_plan),
        "scorer_diagnostic_passed": diagnostics["scorer_passed"],
        "refiner_diagnostic_passed": diagnostics["refiner_passed"],
        "official_metric_status": {
            name: row["status"] for name, row in metrics["official_metrics"].items()
        },
        "accuracy_claim_permitted": False,
        "dry_run_is_result": False,
    }
    summary["lock_sha256"] = canonical_sha256(summary)
    write_json(root / "dry-run-summary.json", summary)
    return summary
