from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pose_accuracy_recovery_prep.c_handoff import (
    export_c_handoff,
    load_and_validate_c_handoff,
)
from pose_accuracy_recovery_prep.c_results import (
    read_c_result_rows,
    validate_c_result_rows,
    validate_c_results,
)
from pose_accuracy_recovery_prep.cli import main
from pose_accuracy_recovery_prep.core import ContractError

from fixture_support import write_valid_c_result_fixture

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "fixtures" / "pose_accuracy_recovery_prep"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_pose_accuracy_recovery_c_handoff_v2.json"
IMPLEMENTATION_COMMIT = "ad3affc9aed1f0ef9a12b3f0423703152e6ad481"
IMPLEMENTATION_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64
REFINER_SHA256 = "3" * 64
SCORER_SHA256 = "4" * 64
COMMITTED_HANDOFF_ROOT = DATA_ROOT / "c_handoff_v2"
COMMITTED_RESULT_ROOT = DATA_ROOT / "c_results_v1"


def _fixture(tmp_path: Path) -> tuple[Path, dict, Path, list[dict]]:
    handoff_root = tmp_path / "handoff"
    export_c_handoff(
        manifest_path=MANIFEST_PATH,
        data_root=DATA_ROOT,
        protocol_path=PROTOCOL_PATH,
        output_root=handoff_root,
        implementation_commit=IMPLEMENTATION_COMMIT,
        implementation_sha256=IMPLEMENTATION_SHA256,
        model_sha256=MODEL_SHA256,
        refiner_checkpoint_sha256=REFINER_SHA256,
        scorer_checkpoint_sha256=SCORER_SHA256,
    )
    handoff, _ = load_and_validate_c_handoff(
        handoff_root / "manifest.json", bundle_root=handoff_root
    )
    result_root = tmp_path / "c-result"
    results_path = write_valid_c_result_fixture(handoff, result_root)
    return handoff_root, handoff, results_path, read_c_result_rows(results_path)


def test_complete_c_result_fixture_validates_every_required_role(tmp_path: Path) -> None:
    handoff_root, handoff, results_path, rows = _fixture(tmp_path)
    summary = validate_c_result_rows(
        handoff, rows, result_root=results_path.parent, verify_assets=True
    )
    assert summary["status"] == "valid"
    assert summary["item_count"] == 5
    assert summary["variant_count"] == 5
    assert summary["status_counts"] == {"success": 5, "failed": 0, "oom": 0}
    assert summary["label_access_count_on_gpu_c"] == 0
    assert summary["scorer_path_open_count_on_gpu_c"] == 0
    assert summary["accuracy_claim_permitted"] is False
    assert handoff_root.is_dir()


def test_committed_c_result_fixture_matches_its_validation_receipt() -> None:
    expected = json.loads(
        (COMMITTED_RESULT_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    observed = validate_c_results(
        handoff_manifest_path=COMMITTED_HANDOFF_ROOT / "manifest.json",
        handoff_bundle_root=COMMITTED_HANDOFF_ROOT,
        results_path=COMMITTED_RESULT_ROOT / "results.jsonl",
        result_root=COMMITTED_RESULT_ROOT,
    )
    assert observed == expected
    assert observed["item_count"] == 5
    assert observed["status_counts"] == {"success": 5, "failed": 0, "oom": 0}
    assert observed["accuracy_claim_permitted"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].pop("top_k"), "keys mismatch"),
        (
            lambda rows: rows[0].pop("failure"),
            "keys mismatch",
        ),
        (
            lambda rows: rows[0]["access_counters"].pop(
                "scorer_path_open_count_on_gpu_c"
            ),
            "keys mismatch",
        ),
        (
            lambda rows: rows[0]["input_sha256"].update({"mask": "0" * 64}),
            "input_sha256 differs",
        ),
        (
            lambda rows: rows[0]["visualization_inventory"][0].pop("bytes"),
            "keys mismatch",
        ),
        (lambda rows: rows[0]["top_k"].pop(), "exactly five"),
        (
            lambda rows: rows[0]["refiner_trace"][0].pop("objective"),
            "keys mismatch",
        ),
    ],
)
def test_result_validator_never_defaults_missing_provenance_fields(
    tmp_path: Path, mutation, message: str
) -> None:
    _, handoff, results_path, rows = _fixture(tmp_path)
    changed = copy.deepcopy(rows)
    mutation(changed)
    with pytest.raises(ContractError, match=message):
        validate_c_result_rows(
            handoff, changed, result_root=results_path.parent, verify_assets=True
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "coverage mismatch"),
        (lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])), "item_id"),
        (
            lambda rows: rows[0].update(
                {"manifest_lock_sha256": "0" * 64}
            ),
            "manifest_lock_sha256 mismatch",
        ),
        (
            lambda rows: rows[0]["final_model_to_camera_pose_m"][0].__setitem__(0, 2.0),
            "not orthonormal",
        ),
        (
            lambda rows: rows[0]["access_counters"].update(
                {"label_access_count_on_gpu_c": 1}
            ),
            "reports label/GT/evaluator/scorer access",
        ),
    ],
)
def test_result_validator_rejects_coverage_lock_and_se3_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    _, handoff, results_path, rows = _fixture(tmp_path)
    changed = copy.deepcopy(rows)
    mutation(changed)
    with pytest.raises(ContractError, match=message):
        validate_c_result_rows(
            handoff, changed, result_root=results_path.parent, verify_assets=True
        )


def test_validate_c_results_cli_writes_hash_bound_receipt(
    tmp_path: Path, capsys
) -> None:
    handoff_root, _, results_path, _ = _fixture(tmp_path)
    output = tmp_path / "validation.json"
    assert (
        main(
            [
                "validate-c-results",
                "--handoff-manifest",
                str(handoff_root / "manifest.json"),
                "--handoff-root",
                str(handoff_root),
                "--results",
                str(results_path),
                "--result-root",
                str(results_path.parent),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "valid"
    assert len(summary["results_sha256"]) == 64
    assert len(summary["validation_lock_sha256"]) == 64
    assert '"command": "validate-c-results"' in capsys.readouterr().out


def test_result_jsonl_rejects_blank_lines_without_skipping(tmp_path: Path) -> None:
    _, _, results_path, _ = _fixture(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(results_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="blank"):
        read_c_result_rows(bad)
