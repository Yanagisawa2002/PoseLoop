#!/usr/bin/env python3
"""Check the release's external weights before provisioning or opening data.

Uses only the standard library unless --load-detector is explicitly requested.
No downloads, inference, training, label reads or existing-file writes occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "reproduction/frozen-assets.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(path: Path | None, expected: dict) -> dict:
    row = {"expected_sha256": expected["sha256"],
           "expected_size_bytes": expected["size_bytes"]}
    if path is None:
        return {**row, "status": "NOT_PROVIDED"}
    row["path"] = str(path)
    try:
        if not path.is_file():
            return {**row, "status": "MISSING_OR_NOT_FILE"}
        row["actual_size_bytes"] = path.stat().st_size
        if row["actual_size_bytes"] != expected["size_bytes"]:
            return {**row, "status": "SIZE_MISMATCH"}
        row["actual_sha256"] = sha256_file(path)
    except OSError as error:
        return {**row, "status": "UNREADABLE", "error": str(error)}
    return {**row, "status": "VERIFIED" if row["actual_sha256"] == expected["sha256"] else "SHA256_MISMATCH"}


def load_detector(checkpoint: Path, dataset_manifest: Path, catalog: dict) -> dict:
    """Use the existing release loader only after all identity checks pass."""
    detector = catalog["detector"]
    if inspect_file(checkpoint, detector)["status"] != "VERIFIED":
        raise ValueError("Frozen detector bytes must verify before model import")
    protocol_path = ROOT / detector["protocol_path"]
    if sha256_file(protocol_path) != detector["protocol_sha256"]:
        raise ValueError("Detector protocol changed")
    if sha256_file(dataset_manifest) != detector["dataset_manifest_sha256"]:
        raise ValueError("Historical dataset manifest identity changed")
    # _load_checkpoint checks schema, epoch, protocol and manifest identity,
    # constructs the frozen two-class head, and loads the state dict strictly.
    sys.path.insert(0, str(ROOT))
    import torch
    import torchvision
    from pose_accuracy_recovery_prep.real_instance_detector_v1.runtime import _load_checkpoint

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    model = _load_checkpoint(checkpoint, protocol_path, dataset_manifest, protocol, torch.device("cpu"))
    return {"status": "LOADED_STRICT_CPU", "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "inference_executed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--foundationpose-root", type=Path)
    parser.add_argument("--load-detector", action="store_true")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--receipt", type=Path, help="Create-only JSON receipt")
    args = parser.parse_args(argv)
    if args.load_detector and (args.detector_checkpoint is None or args.dataset_manifest is None):
        parser.error("--load-detector requires --detector-checkpoint and --dataset-manifest")
    if args.receipt is not None and args.receipt.exists():
        parser.error("Receipt already exists; refusing overwrite")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assets = {"detector": inspect_file(args.detector_checkpoint, catalog["detector"])}
    for name, expected in catalog["foundationpose"]["assets"].items():
        path = args.foundationpose_root / expected["relative_path"] if args.foundationpose_root else None
        assets[name] = inspect_file(path, expected)
    result = {"schema_version": 1, "catalog_sha256": sha256_file(CATALOG),
              "python": platform.python_version(), "platform": platform.platform(),
              "assets": assets, "model_load": {"status": "NOT_REQUESTED"},
              "inference_executed": False, "dataset_payload_opened": False}
    if args.load_detector:
        if assets["detector"]["status"] != "VERIFIED":
            result["model_load"] = {"status": "SKIPPED_UNVERIFIED_BYTES"}
        else:
            try:
                result["model_load"] = load_detector(args.detector_checkpoint, args.dataset_manifest, catalog)
            except Exception as error:
                result["model_load"] = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
    good_assets = all(row["status"] == "VERIFIED" for row in assets.values())
    good_load = not args.load_detector or result["model_load"]["status"] == "LOADED_STRICT_CPU"
    result["status"] = "ASSET_BYTES_VERIFIED" if good_assets and good_load else "BLOCKED"
    result["full_pipeline_reproduced"] = False
    result["exit_code"] = 0 if good_assets and good_load else 3
    encoded = json.dumps(result, indent=2) + "\n"
    if args.receipt is not None:
        with args.receipt.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
