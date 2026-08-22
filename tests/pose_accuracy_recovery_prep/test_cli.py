from __future__ import annotations

import json
from pathlib import Path

from pose_accuracy_recovery_prep.cli import main, parser

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "fixtures" / "pose_accuracy_recovery_prep"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_pose_accuracy_recovery_prep_v1.json"


def test_cli_help_lists_all_preparation_commands(capsys) -> None:
    try:
        parser().parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for command in (
        "validate-manifest",
        "validate-producer",
        "export-producer",
        "export-c-handoff",
        "validate-c-handoff",
        "validate-c-results",
        "plan",
        "prepare-diagnostics",
        "evaluate-diagnostics",
        "report",
        "visualization-plan",
        "dry-run",
    ):
        assert command in output


def test_fixture_dry_run_is_cpu_only_and_writes_json_csv(
    tmp_path: Path, capsys
) -> None:
    assert (
        main(
            [
                "dry-run",
                "--protocol",
                str(PROTOCOL_PATH),
                "--manifest",
                str(MANIFEST_PATH),
                "--data-root",
                str(DATA_ROOT),
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    summary = json.loads(
        (tmp_path / "dry-run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["execution_mode"] == "CPU_SYNTHETIC_FIXTURE_DRY_RUN"
    assert summary["server_connection_count"] == 0
    assert summary["download_count"] == 0
    assert summary["model_invocation_count"] == 0
    assert summary["sealed_split_access_count"] == 0
    assert summary["accuracy_claim_permitted"] is False
    assert summary["dry_run_is_result"] is False
    assert summary["producer_plan_rows"] == 3
    assert summary["evaluator_plan_rows"] == 5
    assert all(
        status == "not_run_fixture"
        for status in summary["official_metric_status"].values()
    )
    expected = (
        "manifest-validation.json",
        "producer-manifest.json",
        "producer-plan.jsonl",
        "evaluator-plan.jsonl",
        "synthetic-predictions.jsonl",
        "visualization-plan.json",
        "diagnostics/diagnostic-summary.json",
        "diagnostics/diagnostic-summary.csv",
        "metrics/metrics.json",
        "metrics/internal-metrics.csv",
        "metrics/grouped-metrics.csv",
        "metrics/paired-deltas.csv",
        "metrics/official-metrics.csv",
    )
    assert all((tmp_path / name).is_file() for name in expected)
    assert '"command": "dry-run"' in capsys.readouterr().out


def test_export_and_evaluator_grid_cli(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    producer_validation = tmp_path / "producer-validation.json"
    grid = tmp_path / "grid.jsonl"
    assert (
        main(
            [
                "export-producer",
                "--manifest",
                str(MANIFEST_PATH),
                "--data-root",
                str(DATA_ROOT),
                "--output",
                str(producer),
            ]
        )
        == 0
    )
    exported = producer.read_text(encoding="utf-8").lower()
    assert "evaluator_only" not in exported
    assert "oracle" not in exported
    assert (
        main(
            [
                "validate-producer",
                "--manifest",
                str(producer),
                "--data-root",
                str(DATA_ROOT),
                "--output",
                str(producer_validation),
            ]
        )
        == 0
    )
    assert (
        json.loads(producer_validation.read_text(encoding="utf-8"))["verified_hashes"]
        is True
    )
    assert (
        main(
            [
                "prepare-diagnostics",
                "--protocol",
                str(PROTOCOL_PATH),
                "--manifest",
                str(MANIFEST_PATH),
                "--data-root",
                str(DATA_ROOT),
                "--namespace-role",
                "EVALUATOR_ONLY",
                "--output",
                str(grid),
            ]
        )
        == 0
    )
    assert len(grid.read_text(encoding="utf-8").splitlines()) == 169
    assert grid.with_suffix(".receipt.json").is_file()
