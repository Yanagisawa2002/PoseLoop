import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import interfaces as api
from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import runtime as v2
from pose_accuracy_recovery_prep.real_instance_detector_v1 import runtime as detector
from scripts.build_failure_taxonomy import reconcile_counts


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def frozen_predictions(tmp_path):
    frames = {split: [{"scene_id": s, "image_id": i, "frame_id": detector._frame_id(s, i),
                       "height": 2, "width": 2} for s, i in detector._frame_specs(split)]
              for split in ("train", "internal_validation", "fixed_evaluation")}
    dataset = dump(tmp_path / "dataset.json", {
        "schema_version": detector.MANIFEST_SCHEMA,
        "protocol_sha256": v2._sha256_file(api.DETECTOR_PROTOCOL),
        "train_frame_count": 400, "internal_validation_frame_count": 100,
        "fixed_evaluation_frame_count": 25, "fixed_evaluation_gt_access_count": 0,
        "scene9_read_count": 0, "frames": frames})
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = []
    for index, source in enumerate(frames["fixed_evaluation"]):
        masks = [np.ones((2, 2), dtype=bool)] if index < 3 else []
        metadata = dump(predictions / f"{index}.json", {
            "frame_id": source["frame_id"], "prediction_count": len(masks),
            "scores": [0.8] * len(masks)})
        mask_path = predictions / f"{index}.npz"
        detector._pack_masks(mask_path, masks, (2, 2))
        rows.append({**source, "metadata": v2._bound_file(metadata, root=predictions),
                     "masks": v2._bound_file(mask_path, root=predictions)})
    dump(predictions / "prediction-manifest.json", {
        "schema_version": detector.PREDICTION_SCHEMA,
        "protocol_sha256": v2._sha256_file(api.DETECTOR_PROTOCOL),
        "dataset_manifest_sha256": v2._sha256_file(dataset), "checkpoint_sha256": "c" * 64,
        "frame_count": 25, "fixed_evaluation_gt_access_count": 0, "scene9_read_count": 0,
        "frames": rows})
    protocol_path = tmp_path / "v2-protocol.json"
    protocol = api.generate_protocol(dataset, predictions, protocol_path)
    return dataset, predictions, protocol_path, protocol


def test_generated_protocol_binds_new_detector(frozen_predictions):
    dataset, predictions, path, protocol = frozen_predictions
    assert protocol["upstream_detector"]["checkpoint_sha256"] == "c" * 64
    assert v2.expected_count(protocol) == 3
    assert protocol["dataset"]["detector_prediction_manifest_sha256"] == v2._sha256_file(predictions / "prediction-manifest.json")
    assert protocol["dataset"]["detector_dataset_manifest_sha256"] == v2._sha256_file(dataset)
    assert protocol["upstream_detector"]["expected_ranked_prediction_count"] == 3
    with pytest.raises(v2.ContractError, match="create-only"):
        api.generate_protocol(dataset, predictions, path)


def test_dynamic_manifest_strict_coverage(frozen_predictions, tmp_path):
    _, _, path, protocol = frozen_predictions
    frame_ids = [detector._frame_id(s, i) for s, i in detector._frame_specs("fixed_evaluation")]
    items = [{"item_id": f"item-{n}", "frame_id": frame_ids[n], "scene_id": 10} for n in range(3)]
    manifest = {"schema_version": v2.INPUT_SCHEMA, "protocol_id": v2.PROTOCOL_ID,
                "protocol_sha256": v2._sha256_file(path), "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
                "frame_count": 25, "item_count": 3,
                "frames": [{"frame_id": f} for f in frame_ids], "items": items,
                "runtime_boundary": {k: 0 for k in ("label_access_count", "gt_path_open_count", "evaluator_path_open_count", "official_scorer_run_count", "scene9_read_count")}}
    def write():
        manifest["manifest_lock_sha256"] = v2._canonical_sha256(v2._without_lock(manifest, "manifest_lock_sha256"))
        return dump(tmp_path / "input.json", manifest)
    assert v2.validate_input_manifest(write(), path, verify_assets=False)["item_count"] == 3
    manifest["items"][2]["item_id"] = "item-0"
    with pytest.raises(v2.ContractError, match="duplicated"):
        v2.validate_input_manifest(write(), path, verify_assets=False)
    manifest["items"] = items[:2]
    with pytest.raises(v2.ContractError):
        v2.validate_input_manifest(write(), path, verify_assets=False)


@pytest.mark.parametrize("section,key,value", [("foundationpose", "iterations", 3),
    ("upstream_detector", "operating_score_threshold", 0.2),
    ("evaluation", "mask_iou_threshold", 0.2)])
def test_v2_rejects_algorithm_change(frozen_predictions, section, key, value):
    _, _, path, protocol = frozen_predictions
    protocol[section][key] = value
    dump(path, protocol)
    with pytest.raises(v2.ContractError):
        v2.load_protocol(path)


def test_tampered_masks_rejected_before_labels(frozen_predictions, tmp_path, monkeypatch):
    dataset, predictions, _, _ = frozen_predictions
    (predictions / "0.npz").write_bytes(b"changed")
    monkeypatch.setattr(detector, "_load_gt", lambda *args: pytest.fail("GT opened before verification"))
    with pytest.raises(v2.ContractError):
        api.evaluate_detector(tmp_path, dataset, predictions, tmp_path / "evaluation")


def test_standalone_metrics_need_no_a8_artifacts(frozen_predictions, tmp_path, monkeypatch):
    dataset, predictions, _, _ = frozen_predictions
    monkeypatch.setattr(detector, "_load_gt", lambda *args: [np.ones((2, 2), dtype=bool)])
    result = api.evaluate_detector(tmp_path, dataset, predictions, tmp_path / "evaluation")
    assert result["metrics"]["prediction_count"] == 3
    assert result["metrics"]["ground_truth_count"] == 25
    assert result["metrics"]["instance_recall_iou50"] == pytest.approx(3 / 25)
    assert len(result["per_scene"]) == 5


def test_taxonomy_counts_are_anchor_derived():
    anchor = {"ground_truth_instance_count": 7, "mask_iou50_match_count": 4, "joint_pose_success_count": 2}
    reconcile_counts(anchor, 7, 4, 2)
    with pytest.raises(SystemExit, match="reconciliation"):
        reconcile_counts(anchor, 7, 4, 3)


def test_external_protocol_has_path_independent_implementation_identity(frozen_predictions):
    _, _, path, _ = frozen_predictions
    identity = v2._implementation_identity(path)
    assert identity["files_sha256"]["generated-pose-protocol"] == v2._sha256_file(path)
    assert all(not Path(k).is_absolute() for k in identity["files_sha256"])


def test_data_selection_excludes_scene9_and_unneeded_sensors():
    from scripts.prepare_v12_data import selected_path
    assert selected_path("xyzibd_val/val/000009/rgb_realsense/000000.png") is None
    assert selected_path("xyzibd_val/val/000010/rgb_photoneo/000000.png") is None
    assert selected_path("xyzibd_val/val/000010/rgb_realsense/000001.png") is None
    assert selected_path("xyzibd_val/val/000010/rgb_realsense/000010.png") == Path("val/000010/rgb_realsense/000010.png")
    with pytest.raises(ValueError):
        selected_path("../escape")


def test_stage_resume_rejects_tampering(tmp_path):
    from scripts.run_v12_pipeline import checked_stage
    target = tmp_path / "result.json"
    checked_stage(tmp_path, "test", [target], lambda: target.write_text("original"))
    checked_stage(tmp_path, "test", [target], lambda: pytest.fail("Completed stage reran"))
    target.write_text("changed")
    with pytest.raises(v2.ContractError, match="changed"):
        checked_stage(tmp_path, "test", [target], lambda: None)
