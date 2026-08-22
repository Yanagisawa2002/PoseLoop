from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pose_accuracy_recovery_prep.instance_selection_v1r6 as selection
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)
from pose_accuracy_recovery_prep.instance_proposal_v1r5.contracts import (
    BOUNDARY_ZERO,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    mask_statistics,
)


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    result = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def _proposal(
    index: int,
    mask: np.ndarray,
    *,
    cad: float = 0.80,
    second: float = 0.70,
    proposal_score: float = 0.90,
    stability: float = 0.95,
) -> dict[str, object]:
    statistics = mask_statistics(mask)
    scores = [cad, second, second - 0.05, second - 0.10, second - 0.15]
    return {
        "proposal_index": index,
        "bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
        "mask": {
            "mask_pixels": statistics["mask_pixels"],
            "coverage": statistics["coverage"],
            "support_bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
        },
        "cad_ranking": [
            {"rank": rank, "object_id": object_id, "normalized_similarity": score}
            for rank, (object_id, score) in enumerate(
                zip([1, 2, 4, 5, 6], scores, strict=True), start=1
            )
        ],
        "selected_object_id": 1,
        "selected_cad_similarity": cad,
        "proposal_score": proposal_score,
        "mask_stability": stability,
    }


def _frame(proposals: list[dict[str, object]], selected: int = 0) -> dict[str, object]:
    return {
        "item_id": "synthetic-frame",
        "proposal_count": len(proposals),
        "selected_proposal_index": selected,
        "proposals": proposals,
    }


def _protocol(repository_root: Path) -> dict[str, object]:
    relative_paths = [
        "pose_accuracy_recovery_prep/__init__.py",
        "pose_accuracy_recovery_prep/core.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/__init__.py",
        "pose_accuracy_recovery_prep/instance_proposal_v1r5/contracts.py",
        "pose_accuracy_recovery_prep/instance_selection_v1r6.py",
    ]
    protocol: dict[str, object] = {
        "schema_version": selection.PROTOCOL_SCHEMA,
        "protocol_id": selection.PROTOCOL_ID,
        "role": "DEVELOPMENT_ONLY_LABEL_BLIND_POST_SELECTION",
        "source_contract": {
            "protocol_id": selection.SOURCE_PROTOCOL_ID,
            "protocol_lock_sha256": selection.SOURCE_PROTOCOL_LOCK,
            "output_schema": selection.SOURCE_OUTPUT_SCHEMA,
            "independent_disk_validation_required": True,
            "candidate_population_mutation_permitted": False,
            "output_bundle_lock_sha256": selection.SOURCE_OUTPUT_BUNDLE_LOCK,
            "run_receipt_lock_sha256": selection.SOURCE_RUN_RECEIPT_LOCK,
            "frame_manifest_lock_sha256": selection.SOURCE_FRAME_MANIFEST_LOCK,
            "runtime_request_lock_sha256": selection.SOURCE_RUNTIME_REQUEST_LOCK,
            "producer_manifest_sha256": selection.SOURCE_PRODUCER_MANIFEST_SHA256,
            "run_receipt_sha256": selection.SOURCE_RUN_RECEIPT_SHA256,
            "frame_manifest_sha256": selection.SOURCE_FRAME_MANIFEST_SHA256,
            "independent_validation_sha256": selection.SOURCE_VALIDATION_SHA256,
            "frame_count": 10,
            "scene_count": 5,
            "proposal_count": selection.SOURCE_PROPOSAL_COUNT,
        },
        "selection_policy": selection.POLICY,
        "boundary": BOUNDARY_ZERO,
        "scientific_boundary": {
            "foundationpose_run_permitted": False,
            "official_scorer_run_permitted": False,
            "downstream_export_permitted": False,
            "human_content_acceptance_required": True,
            "accuracy_claim_permitted": False,
        },
        "wrapper_files": [],
        "protocol_lock_sha256": "pending",
    }
    for relative in relative_paths:
        path = repository_root / Path(*relative.split("/"))
        protocol["wrapper_files"].append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    protocol["protocol_lock_sha256"] = canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_lock_sha256"}
    )
    return protocol


def test_calibrated_ranking_can_beat_cad_only() -> None:
    mask_a = _mask(100, 100, 260, 260)
    mask_b = _mask(400, 400, 560, 560)
    proposals = [
        _proposal(10, mask_a, cad=0.800, proposal_score=0.31),
        _proposal(20, mask_b, cad=0.799, proposal_score=0.99),
    ]
    decision = selection.select_frame(_frame(proposals, selected=10), [mask_a, mask_b])
    assert decision["decision_state"] == "SELECTED"
    assert decision["selected_proposal_index"] == 20
    assert decision["changed_from_a_r5"] is True


def test_maximum_coverage_and_frame_edge_are_hard_filters() -> None:
    giant = _mask(300, 200, 1100, 800)
    edge = _mask(1, 300, 101, 500)
    valid = _mask(600, 400, 760, 600)
    decision = selection.select_frame(
        _frame(
            [
                _proposal(1, giant, cad=0.99),
                _proposal(2, edge, cad=0.98),
                _proposal(3, valid, cad=0.75),
            ],
            selected=1,
        ),
        [giant, edge, valid],
    )
    audits = {item["proposal_index"]: item for item in decision["candidate_audits"]}
    assert "MAX_COVERAGE" in audits[1]["filter_reasons"]
    assert "FRAME_EDGE_TOUCH" in audits[2]["filter_reasons"]
    assert decision["selected_proposal_index"] == 3


def test_multi_child_container_is_rejected() -> None:
    parent = _mask(300, 300, 1100, 700)
    children = [
        _mask(350, 350, 450, 450),
        _mask(550, 350, 650, 450),
        _mask(750, 350, 850, 450),
    ]
    masks = [parent, *children]
    proposals = [_proposal(0, parent, cad=0.99)] + [
        _proposal(index, child, cad=0.75 + index / 100)
        for index, child in enumerate(children, start=1)
    ]
    decision = selection.select_frame(_frame(proposals), masks)
    parent_audit = decision["candidate_audits"][0]
    assert parent_audit["contained_child_count"] == 3
    assert "CONTAINS_MULTIPLE_CHILD_PROPOSALS" in parent_audit["filter_reasons"]
    assert decision["selected_proposal_index"] != 0


def test_rectangular_container_span_and_low_score_are_rejected() -> None:
    rectangle = _mask(200, 400, 1200, 600)
    low_score = _mask(500, 200, 650, 350)
    valid = _mask(800, 700, 950, 850)
    decision = selection.select_frame(
        _frame(
            [
                _proposal(4, rectangle, cad=0.99),
                _proposal(5, low_score, cad=0.98, proposal_score=0.299),
                _proposal(6, valid, cad=0.76),
            ],
            selected=4,
        ),
        [rectangle, low_score, valid],
    )
    audits = {item["proposal_index"]: item for item in decision["candidate_audits"]}
    assert "RECTANGULAR_CONTAINER_SPAN" in audits[4]["filter_reasons"]
    assert "LOW_PROPOSAL_SCORE" in audits[5]["filter_reasons"]
    assert decision["selected_proposal_index"] == 6


def test_low_cad_margin_abstains_and_forbids_selection() -> None:
    mask = _mask(500, 300, 700, 500)
    proposal = _proposal(9, mask, cad=0.80, second=0.77)
    decision = selection.select_frame(_frame([proposal], selected=9), [mask])
    assert decision["best_candidate_proposal_index"] == 9
    assert decision["selected_proposal_index"] is None
    assert decision["selected_object_id"] is None
    assert decision["decision_state"] == "ABSTAIN_LOW_CAD_MARGIN"


def test_tie_break_is_deterministic_and_input_is_not_mutated() -> None:
    mask_a = _mask(400, 300, 520, 420)
    mask_b = _mask(700, 300, 820, 420)
    proposals = [_proposal(8, mask_a), _proposal(2, mask_b)]
    frame = _frame(proposals, selected=8)
    before = copy.deepcopy(frame)
    decision = selection.select_frame(frame, [mask_a, mask_b])
    assert decision["selected_proposal_index"] == 2
    assert frame == before
    assert decision["source_proposal_count"] == 2


def test_no_eligible_proposal_is_explicit() -> None:
    edge = _mask(0, 0, 100, 100)
    decision = selection.select_frame(_frame([_proposal(0, edge)]), [edge])
    assert decision["decision_state"] == "NO_ELIGIBLE_PROPOSAL"
    assert decision["selected_proposal_index"] is None


def test_protocol_is_self_locked_and_hashes_execution_closure() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    protocol = _protocol(repository_root)
    committed = json.loads(
        (
            repository_root
            / "protocols"
            / "poseloop_pose_accuracy_recovery_instance_selection_v1r6.json"
        ).read_text(encoding="utf-8")
    )
    assert committed == protocol
    assert selection.validate_protocol(protocol, repository_root=repository_root)
    tampered = copy.deepcopy(protocol)
    tampered["selection_policy"]["maximum_mask_coverage"] = 0.30
    tampered["protocol_lock_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "protocol_lock_sha256"}
    )
    with pytest.raises(ContractError, match="selection policy"):
        selection.validate_protocol(tampered, repository_root=repository_root)


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_asset(path: Path, root: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _synthetic_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, Path]]:
    repository_root = Path(__file__).resolve().parents[2]
    data_root = tmp_path / "source-input"
    source_output_root = tmp_path / "source-output"
    manifest_frames: list[dict[str, object]] = []
    source_frames: list[dict[str, object]] = []
    for index in range(10):
        item_id = f"scene-{index // 2:06d}-image-{index % 2:06d}"
        rgb_path = data_root / "inputs" / "rgb" / f"{item_id}.png"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (20 + index, 30, 40)).save(
            rgb_path, format="PNG"
        )
        mask = _mask(500, 350, 700, 550)
        mask_path = (
            source_output_root / "outputs" / "masks" / item_id / "proposal-0000.png"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path, format="PNG")
        statistics = mask_statistics(mask)
        proposal = _proposal(0, mask)
        proposal["mask"] = {
            **_file_asset(mask_path, source_output_root, "independent_instance_mask"),
            "decoded_shape": [FRAME_HEIGHT, FRAME_WIDTH],
            "decoded_mode": "L",
            "foreground_pixels": statistics["mask_pixels"],
            "mask_pixels": statistics["mask_pixels"],
            "coverage": statistics["coverage"],
            "support_bbox_xyxy_half_open": statistics["bbox_xyxy_half_open"],
        }
        manifest_frames.append(
            {
                "item_id": item_id,
                "inputs": {"rgb": _file_asset(rgb_path, data_root, "rgb")},
            }
        )
        source_frames.append(
            {
                "item_id": item_id,
                "proposal_count": 1,
                "selected_proposal_index": 0,
                "proposals": [proposal],
            }
        )
    frame_manifest: dict[str, object] = {
        "frame_count": 10,
        "scene_count": 5,
        "frames": manifest_frames,
        "frame_manifest_lock_sha256": "pending",
    }
    frame_manifest["frame_manifest_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in frame_manifest.items()
            if key != "frame_manifest_lock_sha256"
        }
    )
    frame_manifest_path = data_root / "contracts" / "frame-manifest.json"
    _json_write(frame_manifest_path, frame_manifest)
    runtime_request_lock = "a" * 64
    source_manifest: dict[str, object] = {
        "schema_version": selection.SOURCE_OUTPUT_SCHEMA,
        "protocol_id": selection.SOURCE_PROTOCOL_ID,
        "protocol_lock_sha256": selection.SOURCE_PROTOCOL_LOCK,
        "status": "COMPLETE",
        "frame_count": 10,
        "scene_count": 5,
        "catalog_object_ids": [1, 2, 4, 5, 6],
        "frame_manifest_lock_sha256": frame_manifest["frame_manifest_lock_sha256"],
        "runtime_request_lock_sha256": runtime_request_lock,
        "frames": source_frames,
        "boundary": BOUNDARY_ZERO,
        "output_bundle_lock_sha256": "pending",
    }
    source_manifest["output_bundle_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in source_manifest.items()
            if key != "output_bundle_lock_sha256"
        }
    )
    source_manifest_path = source_output_root / "outputs" / "producer-manifest.json"
    _json_write(source_manifest_path, source_manifest)
    source_receipt: dict[str, object] = {
        "status": "COMPLETE",
        "frame_count": 10,
        "scene_count": 5,
        "boundary": BOUNDARY_ZERO,
        "output_bundle": _file_asset(
            source_manifest_path, source_output_root, "producer_output_manifest"
        ),
        "run_receipt_lock_sha256": "pending",
    }
    source_receipt["run_receipt_lock_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in source_receipt.items()
            if key != "run_receipt_lock_sha256"
        }
    )
    source_receipt_path = source_output_root / "outputs" / "run-receipt.json"
    _json_write(source_receipt_path, source_receipt)
    source_validation = {
        "status": "PASS",
        "disk_assets_reverified": True,
        "frame_count": 10,
        "scene_count": 5,
        "boundary": BOUNDARY_ZERO,
        "output_bundle_lock_sha256": source_manifest["output_bundle_lock_sha256"],
        "run_receipt_lock_sha256": source_receipt["run_receipt_lock_sha256"],
    }
    source_validation_path = tmp_path / "source-validation.json"
    _json_write(source_validation_path, source_validation)
    monkeypatch.setattr(selection, "SOURCE_PROPOSAL_COUNT", 10)
    monkeypatch.setattr(
        selection,
        "SOURCE_OUTPUT_BUNDLE_LOCK",
        source_manifest["output_bundle_lock_sha256"],
    )
    monkeypatch.setattr(
        selection, "SOURCE_RUN_RECEIPT_LOCK", source_receipt["run_receipt_lock_sha256"]
    )
    monkeypatch.setattr(
        selection,
        "SOURCE_FRAME_MANIFEST_LOCK",
        frame_manifest["frame_manifest_lock_sha256"],
    )
    monkeypatch.setattr(selection, "SOURCE_RUNTIME_REQUEST_LOCK", runtime_request_lock)
    monkeypatch.setattr(
        selection, "SOURCE_PRODUCER_MANIFEST_SHA256", sha256_file(source_manifest_path)
    )
    monkeypatch.setattr(
        selection, "SOURCE_RUN_RECEIPT_SHA256", sha256_file(source_receipt_path)
    )
    monkeypatch.setattr(
        selection, "SOURCE_FRAME_MANIFEST_SHA256", sha256_file(frame_manifest_path)
    )
    monkeypatch.setattr(
        selection, "SOURCE_VALIDATION_SHA256", sha256_file(source_validation_path)
    )
    return _protocol(repository_root), {
        "repository_root": repository_root,
        "data_root": data_root,
        "source_output_root": source_output_root,
        "source_manifest_path": source_manifest_path,
        "source_receipt_path": source_receipt_path,
        "source_validation_path": source_validation_path,
        "frame_manifest_path": frame_manifest_path,
        "output_root": tmp_path / "a-r6-output",
    }


def test_full_disk_run_validation_create_only_and_visual_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, paths = _synthetic_bundle(tmp_path, monkeypatch)
    output, receipt = selection.run_selection(protocol=protocol, **paths)
    assert output["aggregate"] == {
        "frame_count": 10,
        "source_proposal_count": 10,
        "eligible_proposal_count": 10,
        "rejected_proposal_count": 0,
        "changed_selection_frame_count": 0,
        "selected_frame_count": 10,
        "abstained_frame_count": 0,
        "no_eligible_frame_count": 0,
        "filter_reason_counts": {reason: 0 for reason in selection.FILTER_REASON_ORDER},
    }
    manifest_path = paths["output_root"] / "outputs" / "selection-manifest.json"
    receipt_path = paths["output_root"] / "outputs" / "run-receipt.json"
    validation = selection.validate_selection_output(
        protocol=protocol,
        selection_manifest_path=manifest_path,
        selection_receipt_path=receipt_path,
        **{key: value for key, value in paths.items() if key != "output_root"},
        output_root=paths["output_root"],
    )
    assert validation["status"] == "PASS_INDEPENDENT_DISK_VALIDATION"
    assert receipt["model_run_count"] == 0
    with pytest.raises(ContractError, match="create-only"):
        selection.run_selection(protocol=protocol, **paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visual_relative = manifest["frames"][0]["visualizations"]["before_after_selection"][
        "relative_path"
    ]
    visual_path = paths["output_root"] / Path(*visual_relative.split("/"))
    with Image.open(visual_path) as image:
        changed = image.convert("RGB")
    changed.putpixel((FRAME_WIDTH + 10, FRAME_HEIGHT - 10), (255, 0, 255))
    changed.save(visual_path, format="PNG")
    with pytest.raises(ContractError, match="changed on disk"):
        selection.validate_selection_output(
            protocol=protocol,
            selection_manifest_path=manifest_path,
            selection_receipt_path=receipt_path,
            **{key: value for key, value in paths.items() if key != "output_root"},
            output_root=paths["output_root"],
        )
