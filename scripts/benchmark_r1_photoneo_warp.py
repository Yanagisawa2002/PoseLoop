#!/usr/bin/env python3
"""Benchmark the label-blind Photoneo crop warp used by FoundationPose."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import kornia
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=252)
    parser.add_argument("--height", type=int, default=1544)
    parser.add_argument("--width", type=int, default=2064)
    parser.add_argument("--output-size", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.candidates <= 0:
        raise ValueError("batch sizes must be positive")
    device = torch.device("cuda:0")
    source = torch.zeros(
        (1, 3, args.height, args.width), dtype=torch.float32, device=device
    )
    matrices = torch.eye(3, dtype=torch.float32, device=device).repeat(
        args.candidates, 1, 1
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs: list[torch.Tensor] = []
    for start in range(0, args.candidates, args.batch_size):
        stop = min(start + args.batch_size, args.candidates)
        outputs.append(
            kornia.geometry.transform.warp_perspective(
                source.expand(stop - start, -1, -1, -1),
                matrices[start:stop],
                dsize=(args.output_size, args.output_size),
                mode="bilinear",
                align_corners=False,
            )
        )
    output = torch.cat(outputs, dim=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    receipt = {
        "batch_size": args.batch_size,
        "candidates": args.candidates,
        "source_shape": list(source.shape),
        "output_shape": list(output.shape),
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "device": torch.cuda.get_device_name(0),
        "labels_read": False,
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
