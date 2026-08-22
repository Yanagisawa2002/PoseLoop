#!/usr/bin/env python3
"""Exercise the real FoundationPose model and CUDA initialization path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=repo_root / "third_party" / "FoundationPose",
    )
    return parser.parse_args()


def inspect_parameters(model: Any) -> dict[str, Any]:
    import torch

    total = 0
    tensors = 0
    nonfinite = 0
    non_cuda: list[str] = []
    for name, parameter in model.named_parameters():
        tensors += 1
        total += parameter.numel()
        if not parameter.is_cuda:
            non_cuda.append(name)
        nonfinite += int((~torch.isfinite(parameter)).sum().item())

    if total == 0:
        raise RuntimeError(f"{type(model).__name__} has no parameters")
    if non_cuda:
        preview = ", ".join(non_cuda[:5])
        raise RuntimeError(f"{type(model).__name__} has CPU parameters: {preview}")
    if nonfinite:
        raise RuntimeError(
            f"{type(model).__name__} has {nonfinite} non-finite parameter values"
        )

    return {
        "parameter_tensors": tensors,
        "parameter_values": total,
        "nonfinite_parameter_values": nonfinite,
        "all_parameters_cuda": True,
    }


def main() -> int:
    args = parse_args()
    foundationpose_root = args.foundationpose_root.resolve()
    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose checkout not found: {foundationpose_root}")

    os.chdir(foundationpose_root)
    sys.path.insert(0, str(foundationpose_root))

    import nvdiffrast.torch as dr
    import torch
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    scorer_started = time.perf_counter()
    scorer = ScorePredictor()
    torch.cuda.synchronize()
    scorer_seconds = time.perf_counter() - scorer_started

    refiner_started = time.perf_counter()
    refiner = PoseRefinePredictor()
    torch.cuda.synchronize()
    refiner_seconds = time.perf_counter() - refiner_started

    scorer_parameters = inspect_parameters(scorer.model)
    refiner_parameters = inspect_parameters(refiner.model)

    raster_started = time.perf_counter()
    glctx = dr.RasterizeCudaContext()
    positions = torch.tensor(
        [
            [-0.5, -0.5, 0.5, 1.0],
            [0.5, -0.5, 0.5, 1.0],
            [0.0, 0.5, 0.5, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    raster, _ = dr.rasterize(glctx, positions, triangles, resolution=[8, 8])
    torch.cuda.synchronize()
    raster_seconds = time.perf_counter() - raster_started
    if not bool(torch.isfinite(raster).all().item()):
        raise RuntimeError("nvdiffrast produced non-finite output")
    covered_pixels = int((raster[..., 3] > 0).sum().item())
    if covered_pixels == 0:
        raise RuntimeError("nvdiffrast CUDA kernel produced no covered pixels")

    result = {
        "status": "ok",
        "foundationpose_root": str(foundationpose_root),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "scorer": {
            **scorer_parameters,
            "initialization_seconds": scorer_seconds,
        },
        "refiner": {
            **refiner_parameters,
            "initialization_seconds": refiner_seconds,
        },
        "nvdiffrast": {
            "context": type(glctx).__name__,
            "covered_pixels": covered_pixels,
            "finite_output": True,
            "kernel_seconds": raster_seconds,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
