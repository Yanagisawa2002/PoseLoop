"""Label-blind protocol generation and standalone development detector evaluation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from pose_accuracy_recovery_prep.real_instance_detector_v1 import runtime as detector
from . import runtime as pose

ROOT = Path(__file__).resolve().parents[2]
DETECTOR_PROTOCOL = ROOT / "protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json"
ANCESTOR = ROOT / "protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json"


def validate_successor_policy(protocol):
    """Only identities and label-blind population counts may differ from A9."""
    old = pose._read_json(ANCESTOR)
    for key in ("foundationpose", "runtime_boundary", "evaluation", "promotion_gate", "failure_policy"):
        if protocol.get(key) != old[key]:
            raise pose.ContractError(f"Frozen algorithm/boundary changed: {key}")
    if protocol.get("source_pose_protocol_sha256") != pose._sha256_file(ANCESTOR):
        raise pose.ContractError("A9 ancestry changed")
    if protocol["dataset"]["detector_protocol_sha256"] != pose._sha256_file(DETECTOR_PROTOCOL):
        raise pose.ContractError("Training protocol changed")
    if protocol.get("dataset_role") != "ALREADY_CONSUMED_REAL_DEVELOPMENT":
        raise pose.ContractError("Development role changed")


def validate_predictions(protocol_path, dataset_manifest_path, predictions_root):
    """Validate every prediction artifact before any evaluator may open GT."""
    protocol = detector.load_protocol(protocol_path)
    dataset = detector._load_manifest(dataset_manifest_path, protocol_path)
    path = predictions_root / "prediction-manifest.json"
    manifest = pose._read_json(path)
    if (manifest.get("schema_version") != detector.PREDICTION_SCHEMA
        or manifest.get("protocol_sha256") != pose._sha256_file(protocol_path)
        or manifest.get("dataset_manifest_sha256") != pose._sha256_file(dataset_manifest_path)
        or not pose._is_sha256(manifest.get("checkpoint_sha256"))
        or manifest.get("frame_count") != 25
        or manifest.get("fixed_evaluation_gt_access_count") != 0
        or manifest.get("scene9_read_count") != 0):
        raise pose.ContractError("Detector prediction identity/boundary changed")
    sources = {r["frame_id"]: r for r in dataset["frames"]["fixed_evaluation"]}
    rows = manifest.get("frames", [])
    if len(rows) != 25 or {r["frame_id"] for r in rows} != set(sources):
        raise pose.ContractError("Detector frame coverage changed")
    bundles = []
    ranked = operating = 0
    threshold = protocol["model"]["operating_score_threshold"]
    for row in rows:
        source = sources[row["frame_id"]]
        if (row["scene_id"], row["image_id"]) != (source["scene_id"], source["image_id"]):
            raise pose.ContractError("Prediction frame identity changed")
        metadata = pose._read_json(pose._resolve_bound(predictions_root, row["metadata"]))
        masks = detector._unpack_masks(pose._resolve_bound(predictions_root, row["masks"]))
        scores = np.asarray(metadata.get("scores", []), dtype=float)
        if (metadata.get("frame_id") != row["frame_id"]
            or metadata.get("prediction_count") != len(masks)
            or scores.shape != (len(masks),) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
            or any(m.shape != (source["height"], source["width"]) for m in masks)):
            raise pose.ContractError("Prediction score/mask bundle changed")
        ranked += len(masks)
        operating += int((scores >= threshold).sum())
        bundles.append((source, masks, scores.tolist()))
    return manifest, bundles, ranked, operating


def generate_protocol(dataset_manifest_path, predictions_root, output_path):
    if output_path.exists():
        raise pose.ContractError("Generated protocol is create-only")
    manifest, _, ranked, operating = validate_predictions(
        DETECTOR_PROTOCOL, dataset_manifest_path, predictions_root)
    protocol = copy.deepcopy(pose._read_json(ANCESTOR))
    protocol.update(schema_version=pose.SCHEMA, protocol_id=pose.PROTOCOL_ID,
                    experiment_version="v1.2",
                    source_pose_protocol_sha256=pose._sha256_file(ANCESTOR))
    dataset = protocol["dataset"]
    for key in list(dataset):
        if key.startswith("a9_"):
            del dataset[key]
    dataset.update(detector_dataset_manifest_sha256=pose._sha256_file(dataset_manifest_path),
                   detector_prediction_manifest_sha256=pose._sha256_file(predictions_root / "prediction-manifest.json"),
                   detector_protocol_sha256=pose._sha256_file(DETECTOR_PROTOCOL))
    upstream = protocol["upstream_detector"]
    upstream.pop("result_commit")
    upstream.update(implementation_commit=pose._git_head(ROOT),
                    checkpoint_sha256=manifest["checkpoint_sha256"],
                    expected_ranked_prediction_count=ranked,
                    expected_operating_prediction_count=operating)
    pose.expected_count(protocol)
    validate_successor_policy(protocol)
    pose._write_json_atomic(output_path, protocol)
    return pose.load_protocol(output_path)


def evaluate_detector(dataset_root, dataset_manifest_path, predictions_root, output_root):
    if output_root.exists():
        raise pose.ContractError("Detector evaluation is create-only")
    manifest, bundles, _, _ = validate_predictions(DETECTOR_PROTOCOL, dataset_manifest_path, predictions_root)
    prediction_sha = pose._sha256_file(predictions_root / "prediction-manifest.json")
    # The complete inventory is frozen and validated above, before the first GT open.
    frames = [{"frame_id": source["frame_id"], "pred_masks": masks, "scores": scores,
               "gt_masks": detector._load_gt(dataset_root, source["scene_id"], source["image_id"])}
              for source, masks, scores in bundles]
    metrics = detector._detector_metrics(frames, operating_threshold=0.25)
    per_scene = {str(scene): detector._detector_metrics(
        [f for f in frames if f["frame_id"].startswith(f"s{scene:06d}-")], operating_threshold=0.25)
        for scene in detector.EVALUATION_SCENES}
    if pose._sha256_file(predictions_root / "prediction-manifest.json") != prediction_sha:
        raise pose.ContractError("Predictions changed during evaluation")
    result = {"schema_version": "poseloop.v1.2.detector-evaluation.v1",
              "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT",
              "sealed_claim_permitted": False, "official_bop_leaderboard_result": False,
              "labels_opened_during_inference": False, "labels_opened_during_evaluation": True,
              "protocol_sha256": pose._sha256_file(DETECTOR_PROTOCOL),
              "dataset_manifest_sha256": pose._sha256_file(dataset_manifest_path),
              "prediction_manifest_sha256": prediction_sha,
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "metrics": metrics, "per_scene": per_scene, "scene9_read_count": 0}
    output_root.mkdir(parents=True)
    pose._write_json_atomic(output_root / "result.json", result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"generate-protocol", "evaluate-detector"}:
        return pose.main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate-protocol", "evaluate-detector"])
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate-protocol":
        result = generate_protocol(args.dataset_manifest, args.predictions_root, args.output)
    else:
        if args.dataset_root is None:
            parser.error("--dataset-root is required for evaluation")
        result = evaluate_detector(args.dataset_root, args.dataset_manifest, args.predictions_root, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
