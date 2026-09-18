"""Instrument the unchanged A-R9 two-batch train path and verify checkpoint reload."""
from __future__ import annotations
import argparse
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import interfaces as api
from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import runtime as pose


def environment():
    import torch
    import torchvision
    return {"python": platform.python_version(), "pytorch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"], text=True).strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "dataset-manifest", "weight", "output-root"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--smoke-batches", type=int, choices=[2], default=2)
    args = parser.parse_args()
    if args.output_root.exists():
        raise pose.ContractError("Smoke output is create-only")
    if shutil.disk_usage(args.output_root.parent).free < 15 * 1024**3:
        raise pose.ContractError("Less than 15 GiB free before smoke")
    import torch
    detector = api.detector
    original_builder = detector._build_model
    original_step = torch.optim.SGD.step
    observed = {"actual_optimizer_steps": 0}
    holder = {}
    def builder(*arguments, **keywords):
        model = original_builder(*arguments, **keywords)
        holder["model"] = model
        return model
    def step(optimizer, *arguments, **keywords):
        result = original_step(optimizer, *arguments, **keywords)
        observed["actual_optimizer_steps"] += 1
        return result
    detector._build_model = builder
    torch.optim.SGD.step = step
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    identity = {"schema_version": "poseloop.v1.2.smoke-identity.v1", "experiment_version": "v1.2",
                "implementation_commit": pose._git_head(ROOT), **environment(),
                "source_detector_protocol_sha256": pose._sha256_file(api.DETECTOR_PROTOCOL),
                "dataset_manifest_sha256": pose._sha256_file(args.dataset_manifest),
                "official_weight_sha256": pose._sha256_file(args.weight)}
    try:
        result = detector.train(protocol_path=api.DETECTOR_PROTOCOL, dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest, weight_path=args.weight,
            output_root=args.output_root, device_name="cuda:0", smoke_batches=args.smoke_batches)
    finally:
        detector._build_model = original_builder
        torch.optim.SGD.step = original_step
    checkpoint = args.output_root / "model-final.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = holder.pop("model").cpu()
    # Strict reload plus bytewise tensor comparison proves the saved model is usable.
    identical = all(torch.equal(t, saved["model_state_dict"][key]) for key, t in model.state_dict().items())
    model.load_state_dict(saved["model_state_dict"], strict=True)
    tensors_valid = all(torch.isfinite(t).all().item() for t in saved["model_state_dict"].values())
    history = result["history"]
    checks = {"two_batches": history[0]["batch_count"] == 2,
              "one_real_optimizer_step": observed["actual_optimizer_steps"] == 1,
              "finite_loss": math.isfinite(history[0]["mean_total_loss"]),
              "finite_checkpoint": tensors_valid,
              "strict_checkpoint_reload": True,
              "checkpoint_matches_trained_model": identical,
              "checkpoint_hash_matches": pose._sha256_file(checkpoint) == result["checkpoint"]["sha256"],
              "training_protocol_bound": saved["protocol_sha256"] == identity["source_detector_protocol_sha256"],
              "smoke_only": result["mode"] == "SMOKE_NOT_DECISION_ELIGIBLE" and saved["epoch"] == 1}
    receipt = {**identity, "status": "DETECTOR_SMOKE_PASS" if all(checks.values()) else "DETECTOR_SMOKE_FAIL",
               "checks": checks, **observed, "amp": "torch.amp.autocast(cuda) + GradScaler in unchanged A-R9 train",
               "wall_time_seconds_including_reload": time.monotonic() - started,
               "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
               "mean_total_loss": history[0]["mean_total_loss"],
               "smoke_checkpoint": {"relative_path": "model-final.pt", "sha256": pose._sha256_file(checkpoint)},
               "training_result_sha256": pose._sha256_file(args.output_root / "training-result.json"),
               "remaining_disk_bytes": shutil.disk_usage(args.output_root).free,
               "formal_training_started": False}
    pose._write_json_atomic(args.output_root / "smoke-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
