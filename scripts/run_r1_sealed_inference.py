#!/usr/bin/env python3
"""Run label-blind FoundationPose inference for the sealed Photoneo manifest."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import trimesh

from m1_common import (
    INFERENCE_SEED,
    append_jsonl_durable,
    load_jsonl,
    read_json,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from run_xyzibd_batch import (
    SystemicEnvironmentFailure,
    attempt_sample,
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


SCHEMA_VERSION = 1
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


def install_memory_bounded_warp(batch_size: int = WARP_BATCH_SIZE) -> None:
    """Chunk image warps without changing their per-candidate computation."""
    import kornia

    original = kornia.geometry.transform.warp_perspective
    if getattr(original, "_poseloop_memory_bounded", False):
        return

    def chunked_warp(src: torch.Tensor, matrix: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        batch = int(src.shape[0]) if src.ndim == 4 else 0
        if batch <= batch_size:
            return original(src, matrix, *args, **kwargs)
        matrix_batch = int(matrix.shape[0]) if matrix.ndim == 3 else 0
        if matrix_batch not in (1, batch):
            raise ValueError(
                f"Cannot pair warp batches: source={batch}, matrix={matrix_batch}"
            )
        chunks: list[torch.Tensor] = []
        for start in range(0, batch, batch_size):
            stop = min(start + batch_size, batch)
            matrix_chunk = matrix if matrix_batch == 1 else matrix[start:stop]
            chunks.append(original(src[start:stop], matrix_chunk, *args, **kwargs))
        return torch.cat(chunks, dim=0)

    chunked_warp._poseloop_memory_bounded = True  # type: ignore[attr-defined]
    kornia.geometry.transform.warp_perspective = chunked_warp


def install_memory_bounded_score_forward(
    scorer: Any, batch_size: int = SCORE_FEATURE_BATCH_SIZE
) -> None:
    """Chunk independent CNN features, then retain the full candidate attention."""
    model = scorer.model

    def chunked_forward(
        score_model: Any, candidate_a: torch.Tensor, candidate_b: torch.Tensor, L: int
    ) -> dict[str, torch.Tensor]:
        if candidate_a.shape[0] != candidate_b.shape[0]:
            raise ValueError("Score candidate batch mismatch")
        if L <= 0 or candidate_a.shape[0] % L:
            raise ValueError("Invalid FoundationPose score grouping")
        features = torch.cat(
            [
                score_model.extract_feat(
                    candidate_a[start : start + batch_size],
                    candidate_b[start : start + batch_size],
                )
                for start in range(0, candidate_a.shape[0], batch_size)
            ],
            dim=0,
        )
        outer_batch = candidate_a.shape[0] // L
        attended = features.reshape(outer_batch, L, -1)
        attended, _ = score_model.att_cross(attended, attended, attended)
        return {"score_logit": score_model.linear(attended).reshape(outer_batch, L)}

    model.forward = types.MethodType(chunked_forward, model)


def install_memory_bounded_score_data(
    scorer: Any, batch_size: int = SCORE_DATA_BATCH_SIZE
) -> None:
    """Run the unchanged full-resolution XYZ transform on bounded candidate chunks."""
    dataset = scorer.dataset
    original = dataset.transform_batch

    def chunked_transform(
        _dataset: Any, batch: Any, H_ori: int, W_ori: int, bound: int = 1
    ) -> Any:
        total = int(batch.rgbAs.shape[0])
        if total <= batch_size:
            return original(batch, H_ori, W_ori, bound=bound)
        transformed: list[Any] = []
        chunk_lengths: list[int] = []
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            candidate_chunk = type(batch)()
            for key, value in batch.__dict__.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and int(value.shape[0]) == total
                ):
                    candidate_chunk.__dict__[key] = value[start:stop]
                else:
                    candidate_chunk.__dict__[key] = value
            transformed.append(
                original(candidate_chunk, H_ori, W_ori, bound=bound)
            )
            chunk_lengths.append(stop - start)
        combined = type(batch)()
        keys = set().union(*(item.__dict__.keys() for item in transformed))
        for key in keys:
            values = [item.__dict__.get(key) for item in transformed]
            if all(
                torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == length
                for value, length in zip(values, chunk_lengths)
            ):
                combined.__dict__[key] = torch.cat(values, dim=0)
            elif all(value is None for value in values):
                combined.__dict__[key] = None
            else:
                combined.__dict__[key] = values[0]
        return combined

    dataset.transform_batch = types.MethodType(chunked_transform, dataset)


def install_memory_bounded_refine_forward(
    refiner: Any, batch_size: int = REFINE_FEATURE_BATCH_SIZE
) -> None:
    """Chunk candidate-independent refiner CNN calls and preserve their order."""
    model = refiner.model
    original = model.forward

    def chunked_forward(
        _model: Any, candidate_a: torch.Tensor, candidate_b: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if candidate_a.shape[0] != candidate_b.shape[0]:
            raise ValueError("Refine candidate batch mismatch")
        outputs = [
            original(
                candidate_a[start : start + batch_size],
                candidate_b[start : start + batch_size],
            )
            for start in range(0, candidate_a.shape[0], batch_size)
        ]
        keys = set(outputs[0])
        if any(set(output) != keys for output in outputs):
            raise ValueError("Refine output keys changed across chunks")
        return {
            key: torch.cat([output[key] for output in outputs], dim=0)
            for key in sorted(keys)
        }

    model.forward = types.MethodType(chunked_forward, model)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "artifacts" / "r1" / "sealed_photoneo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "sealed_contract.json")
    parser.add_argument("--manifest", type=Path, default=root / "inference_manifest.jsonl")
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


def _effective_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("record_type") != "r1_sealed_inference_sample":
        raise ValueError("Unexpected sealed inference manifest row")
    if row.get("sensor_modality") != "photoneo" or not row.get("input_available"):
        raise ValueError(f"Invalid sealed inference sample: {row.get('sample_id')}")
    effective = dict(row)
    # The shared FoundationPose loader uses this field only for a unit sanity
    # check.  It is a synthetic depth-consistent pose created from input depth,
    # not the sealed evaluator pose.
    effective["gt_model_to_camera_pose_m"] = row["input_unit_check_pose_m"]
    effective["visible_fraction"] = float(row["input_mask_area_fraction"])
    effective["visibility_bin"] = "input_only"
    return effective


def _label_blind_result(row: dict[str, Any], estimator: Any) -> dict[str, Any]:
    result = attempt_sample(_effective_row(row), estimator)
    for field in STRIP_EVALUATOR_FIELDS:
        result.pop(field, None)
    result["record_type"] = "r1_sealed_prediction"
    result["schema_version"] = SCHEMA_VERSION
    result["evaluator_label_read"] = False
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = args.contract.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    log_path = args.log.resolve()
    data_root = args.dataset_root.resolve()
    foundationpose_root = args.foundationpose_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "sealed_unopened" or contract.get("labels_opened"):
        raise RuntimeError("Photoneo sealed contract is not unopened")
    expected_manifest = contract["files"]["inference_manifest"]
    if sha256_file(manifest_path) != expected_manifest["sha256"]:
        raise RuntimeError("Sealed inference manifest hash mismatch")
    rows = load_jsonl(manifest_path)
    if len(rows) != int(contract["inference_sample_count"]):
        raise ValueError("Sealed inference sample count mismatch")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sealed inference sample ID")
    if not foundationpose_root.joinpath("estimater.py").is_file():
        raise FileNotFoundError(foundationpose_root)
    verify_dataset_structure(data_root)
    if git_commit(foundationpose_root) != FOUNDATIONPOSE_COMMIT:
        raise SystemicEnvironmentFailure("FoundationPose commit changed")
    checkpoints = verify_checkpoints(foundationpose_root)
    if not torch.cuda.is_available():
        raise SystemicEnvironmentFailure("CUDA unavailable for sealed inference")
    torch.cuda.set_device(0)

    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        if not args.resume:
            raise RuntimeError("Sealed prediction output exists; pass --resume only after interruption")
        existing = load_jsonl(output_path)
        if not existing or existing[0].get("record_type") != "r1_sealed_inference_metadata":
            raise ValueError("Invalid sealed prediction metadata")
        if existing[0].get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("Existing sealed prediction stream has another manifest")
        for row in existing[1:]:
            sample_id = str(row["sample_id"])
            if sample_id not in ids or sample_id in completed:
                raise ValueError("Invalid existing sealed prediction identity")
            completed[sample_id] = row
    else:
        metadata = {
            "record_type": "r1_sealed_inference_metadata",
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sensor_modality": "photoneo",
            "manifest_sha256": sha256_file(manifest_path),
            "sample_count": len(rows),
            "evaluator_labels_path_read": False,
            "evaluator_pose_or_error_computed": False,
            "foundationpose_commit": FOUNDATIONPOSE_COMMIT,
            "resource_bounded_execution": {
                "warp_batch_size": WARP_BATCH_SIZE,
                "score_data_batch_size": SCORE_DATA_BATCH_SIZE,
                "score_feature_batch_size": SCORE_FEATURE_BATCH_SIZE,
                "refine_feature_batch_size": REFINE_FEATURE_BATCH_SIZE,
                "candidate_attention_scope": "unchanged_full_candidate_set",
            },
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
    mesh_cache: dict[int, trimesh.Trimesh] = {}

    def object_mesh(row: dict[str, Any]) -> trimesh.Trimesh:
        object_id = int(row["object_id"])
        if object_id not in mesh_cache:
            mesh_cache[object_id] = load_mesh(
                _effective_row(row), data_root, models_info
            )
        return mesh_cache[object_id]

    warmup_row = choose_warmup_row([_effective_row(row) for row in pending])
    warmup_original = next(row for row in pending if row["sample_id"] == warmup_row["sample_id"])
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
    run_warmup(_effective_row(warmup_original), estimator)
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
        result = _label_blind_result(row, estimator)
        append_jsonl_durable(output_path, result)
        completed[sample_id] = result
        print(
            f"[{len(completed):03d}/{total:03d}] object={object_id:02d} "
            f"status={result['status']} time={result.get('registration_seconds', float('nan')):.3f}s",
            flush=True,
        )
    counts = Counter(row["status"] for row in completed.values())
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": "R1 sealed Photoneo label-blind inference",
        "status": "complete",
        "sample_count": len(completed),
        "status_counts": dict(sorted(counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "labels_opened": False,
        "evaluator_pose_or_error_computed": False,
        "resource_bounded_execution": {
            "warp_batch_size": WARP_BATCH_SIZE,
            "score_data_batch_size": SCORE_DATA_BATCH_SIZE,
            "score_feature_batch_size": SCORE_FEATURE_BATCH_SIZE,
            "refine_feature_batch_size": REFINE_FEATURE_BATCH_SIZE,
            "candidate_attention_scope": "unchanged_full_candidate_set",
        },
    }
    write_json_atomic(output_path.parent / "inference_receipt.json", receipt)
    print(f"complete: {len(completed)}/{total}; statuses={dict(counts)}")
    print(output_path)


if __name__ == "__main__":
    main()
