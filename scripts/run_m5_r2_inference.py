#!/usr/bin/env python3
"""Run label-blind FoundationPose inference for the frozen M5-R2 replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m1_common import (
    INFERENCE_SEED,
    append_jsonl_durable,
    load_jsonl,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
SENSOR_MODALITY = "realsense"
WARP_BATCH_SIZE = 32
SCORE_DATA_BATCH_SIZE = 8
SCORE_FEATURE_BATCH_SIZE = 32
REFINE_FEATURE_BATCH_SIZE = 32
STRIP_EVALUATOR_FIELDS = {
    "gt_model_to_camera_pose_m",
    "translation_error_mm",
    "raw_rotation_error_degrees",
    "visible_fraction",
    "visibility_bin",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r2" / "m5_r2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "contract.json")
    parser.add_argument(
        "--manifest", type=Path, default=root / "inference_manifest.jsonl"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    parser.add_argument("--output", type=Path, default=root / "predictions.jsonl")
    parser.add_argument("--log", type=Path, default=root / "inference.log")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def effective_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("record_type") != "m5_r2_inference_sample":
        raise ValueError("Unexpected M5-R2 inference manifest row")
    if row.get("sensor_modality") != SENSOR_MODALITY or not row.get("input_available"):
        raise ValueError(f"Invalid M5-R2 inference sample: {row.get('sample_id')}")
    if any(field in row for field in STRIP_EVALUATOR_FIELDS):
        raise ValueError(f"Evaluator field leaked into M5-R2 inference: {row['sample_id']}")
    effective = dict(row)
    # The shared loader uses these values only for input/unit validation. They
    # are derived from the input depth and mask, not from evaluator labels.
    effective["gt_model_to_camera_pose_m"] = row["input_unit_check_pose_m"]
    effective["visible_fraction"] = float(row["input_mask_area_fraction"])
    effective["visibility_bin"] = "input_only"
    return effective


def label_blind_result(row: dict[str, Any], estimator: Any) -> dict[str, Any]:
    from run_xyzibd_batch import attempt_sample

    result = attempt_sample(effective_row(row), estimator)
    for field in STRIP_EVALUATOR_FIELDS:
        result.pop(field, None)
    result["record_type"] = "m5_r2_prediction"
    result["schema_version"] = SCHEMA_VERSION
    result["evaluator_label_read"] = False
    return result


def resource_contract() -> dict[str, Any]:
    return {
        "warp_batch_size": WARP_BATCH_SIZE,
        "score_data_batch_size": SCORE_DATA_BATCH_SIZE,
        "score_feature_batch_size": SCORE_FEATURE_BATCH_SIZE,
        "refine_feature_batch_size": REFINE_FEATURE_BATCH_SIZE,
        "candidate_attention_scope": "unchanged_full_candidate_set",
    }


def main() -> None:
    import torch
    import trimesh

    from run_r1_sealed_inference import (
        install_memory_bounded_refine_forward,
        install_memory_bounded_score_data,
        install_memory_bounded_score_forward,
        install_memory_bounded_warp,
    )
    from run_xyzibd_batch import (
        SystemicEnvironmentFailure,
        choose_warmup_row,
        clear_per_sample_estimator_state,
        configure_logging,
        load_mesh,
        run_warmup,
    )
    from run_xyzibd_smoke import (
        FOUNDATIONPOSE_COMMIT,
        git_commit,
        verify_checkpoints,
        verify_dataset_structure,
    )

    args = parse_args()
    contract_path = args.contract.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    log_path = args.log.resolve()
    data_root = args.dataset_root.resolve()
    foundationpose_root = args.foundationpose_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != "poseloop-m5-r2-v1"
        or contract.get("status") != "development_labels_unopened"
        or contract.get("labels_opened")
    ):
        raise RuntimeError("M5-R2 contract is not frozen and unopened")
    if sha256_file(Path(__file__).resolve()) != contract["code"]["inference"]["sha256"]:
        raise RuntimeError("M5-R2 inference code changed after bundle freeze")
    if sha256_file(manifest_path) != contract["files"]["inference_manifest"]["sha256"]:
        raise RuntimeError("M5-R2 inference manifest hash mismatch")
    rows = load_jsonl(manifest_path)
    expected_count = int(contract["selection_summary"]["inference_sample_count"])
    if len(rows) != expected_count:
        raise ValueError("M5-R2 inference sample count mismatch")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate M5-R2 inference sample ID")
    for row in rows:
        effective_row(row)
    if not foundationpose_root.joinpath("estimater.py").is_file():
        raise FileNotFoundError(foundationpose_root)
    verify_dataset_structure(data_root)
    if git_commit(foundationpose_root) != FOUNDATIONPOSE_COMMIT:
        raise SystemicEnvironmentFailure("FoundationPose commit changed")
    checkpoints = verify_checkpoints(foundationpose_root)
    if not torch.cuda.is_available():
        raise SystemicEnvironmentFailure("CUDA unavailable for M5-R2 inference")
    torch.cuda.set_device(0)

    completed: dict[str, dict[str, Any]] = {}
    manifest_sha = sha256_file(manifest_path)
    if output_path.exists():
        if not args.resume:
            raise RuntimeError("M5-R2 predictions exist; pass --resume only after interruption")
        existing = load_jsonl(output_path)
        if not existing or existing[0].get("record_type") != "m5_r2_inference_metadata":
            raise ValueError("Invalid M5-R2 prediction metadata")
        if existing[0].get("manifest_sha256") != manifest_sha:
            raise ValueError("Existing M5-R2 predictions use another manifest")
        for row in existing[1:]:
            sample_id = str(row["sample_id"])
            if sample_id not in ids or sample_id in completed:
                raise ValueError("Invalid existing M5-R2 prediction identity")
            completed[sample_id] = row
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "record_type": "m5_r2_inference_metadata",
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sensor_modality": SENSOR_MODALITY,
            "manifest_sha256": manifest_sha,
            "sample_count": len(rows),
            "evaluator_labels_path_read": False,
            "evaluator_pose_or_error_computed": False,
            "foundationpose_commit": FOUNDATIONPOSE_COMMIT,
            "resource_bounded_execution": resource_contract(),
            "checkpoint_sha256": {
                key: value["sha256"] for key, value in checkpoints["models"].items()
            },
        }
        write_jsonl_atomic(output_path, [metadata])

    pending = [row for row in rows if row["sample_id"] not in completed]
    if not pending:
        print(f"nothing pending: {len(completed)}/{len(rows)}")
        return
    configure_logging(log_path, append=args.resume and log_path.exists())
    os.chdir(foundationpose_root)
    if str(foundationpose_root) not in sys.path:
        sys.path.insert(0, str(foundationpose_root))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import set_seed

    install_memory_bounded_warp()
    configure_logging(log_path, append=True)
    models_info = read_json(data_root / "models" / "models_info.json")
    mesh_cache: dict[int, Any] = {}

    def object_mesh(row: dict[str, Any]) -> Any:
        object_id = int(row["object_id"])
        if object_id not in mesh_cache:
            mesh_cache[object_id] = load_mesh(effective_row(row), data_root, models_info)
        return mesh_cache[object_id]

    warmup_effective = choose_warmup_row([effective_row(row) for row in pending])
    warmup_original = next(
        row for row in pending if row["sample_id"] == warmup_effective["sample_id"]
    )
    warmup_mesh = object_mesh(warmup_original)
    set_seed(INFERENCE_SEED)
    scorer = ScorePredictor()
    install_memory_bounded_score_data(scorer)
    install_memory_bounded_score_forward(scorer)
    refiner = PoseRefinePredictor()
    install_memory_bounded_refine_forward(refiner)
    glctx = dr.RasterizeCudaContext(device=0)
    estimator = FoundationPose(
        model_pts=warmup_mesh.vertices.copy(),
        model_normals=warmup_mesh.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=warmup_mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(output_path.parent / "foundationpose_debug"),
    )
    current_object_id = int(warmup_original["object_id"])
    run_warmup(effective_row(warmup_original), estimator)
    total = len(rows)
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        object_id = int(row["object_id"])
        if object_id != current_object_id:
            mesh = object_mesh(row)
            set_seed(INFERENCE_SEED)
            estimator.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=None,
                mesh=mesh,
            )
            clear_per_sample_estimator_state(estimator)
            current_object_id = object_id
        result = label_blind_result(row, estimator)
        append_jsonl_durable(output_path, result)
        completed[sample_id] = result
        print(
            f"[{len(completed):04d}/{total:04d}] object={object_id:02d} "
            f"status={result['status']} "
            f"time={result.get('registration_seconds', float('nan')):.3f}s",
            flush=True,
        )
    counts = Counter(row["status"] for row in completed.values())
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M5-R2 label-blind RealSense inference",
        "status": "complete",
        "sample_count": len(completed),
        "status_counts": dict(sorted(counts.items())),
        "manifest_sha256": manifest_sha,
        "predictions": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "labels_opened": False,
        "evaluator_pose_or_error_computed": False,
        "resource_bounded_execution": resource_contract(),
    }
    write_json_atomic(output_path.parent / "inference_receipt.json", receipt)
    print(f"complete: {len(completed)}/{total}; statuses={dict(counts)}")
    print(output_path)


if __name__ == "__main__":
    main()
