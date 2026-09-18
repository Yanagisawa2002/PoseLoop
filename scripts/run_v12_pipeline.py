"""Explicit post-training pipeline, with hash-bound stage receipts and exact resume."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import interfaces as api
from pose_accuracy_recovery_prep.a10_foundationpose_e2e_v2 import runtime as pose
from scripts.run_v12_detector_smoke import environment


def file_inventory(root):
    return {p.relative_to(root).as_posix(): pose._sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


def checked_stage(root, name, outputs, action):
    receipt = root / "stages" / f"{name}.json"
    def inventory():
        records = {}
        for path in outputs:
            if path.is_dir():
                records.update({f"{path.relative_to(root).as_posix()}/{k}": v for k, v in file_inventory(path).items()})
            elif path.is_file():
                records[path.relative_to(root).as_posix()] = pose._sha256_file(path)
            else:
                raise pose.ContractError(f"Stage output missing: {path}")
        return records
    if receipt.exists():
        if pose._read_json(receipt)["files"] != inventory():
            raise pose.ContractError(f"Completed stage changed: {name}")
        return
    action()
    pose._write_json_atomic(receipt, {"stage": name, "files": inventory()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "dataset-manifest", "detector-checkpoint", "foundationpose-root", "toolkit-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise pose.ContractError("Commit implementation changes before pipeline execution")
    identity = {"schema_version": "poseloop.v1.2.run-identity.v1", "experiment_version": "v1.2",
                "dataset_role": "ALREADY_CONSUMED_REAL_DEVELOPMENT", "implementation_commit": pose._git_head(ROOT),
                "source_detector_protocol_sha256": pose._sha256_file(api.DETECTOR_PROTOCOL),
                "dataset_manifest_sha256": pose._sha256_file(args.dataset_manifest),
                "detector_checkpoint_sha256": pose._sha256_file(args.detector_checkpoint),
                "foundationpose_source_commit": pose._git_head(args.foundationpose_root),
                "foundationpose_compatibility_patch": None,
                "official_weight_sha256": api.detector.load_protocol(api.DETECTOR_PROTOCOL)["model"]["official_pretrained_weight"]["sha256"],
                **environment()}
    if root.exists():
        if not args.resume or pose._read_json(root / "run-identity.json") != identity:
            raise pose.ContractError("Output exists; exact identity and --resume required")
    else:
        if args.resume:
            raise pose.ContractError("Cannot resume a missing run")
        root.mkdir(parents=True)
        pose._write_json_atomic(root / "run-identity.json", identity)
    predictions = root / "detector-predictions"
    detector_eval = root / "detector-evaluation"
    protocol = root / "pose-protocol.json"
    freeze = root / "freeze"
    manifest = freeze / "input-manifest.json"
    primary = root / "foundationpose-primary"
    evaluation = root / "evaluation"
    taxonomy = root / "taxonomy"
    def persist_training_inputs():
        target = root / "training-inputs"
        target.mkdir()
        shutil.copyfile(args.dataset_manifest, target / "dataset-manifest.json")
        shutil.copyfile(args.detector_checkpoint, target / "model-final.pt")
    checked_stage(root, "training-inputs", [root / "training-inputs"], persist_training_inputs)
    checked_stage(root, "detector", [predictions], lambda: api.detector.predict(
        protocol_path=api.DETECTOR_PROTOCOL, dataset_root=args.dataset_root,
        dataset_manifest_path=args.dataset_manifest, checkpoint_path=args.detector_checkpoint,
        output_root=predictions, device_name="cuda:0"))
    # Protocol derives exclusively from validated label-blind predictions.
    checked_stage(root, "pose-protocol", [protocol], lambda: api.generate_protocol(args.dataset_manifest, predictions, protocol))
    checked_stage(root, "detector-evaluation", [detector_eval], lambda: api.evaluate_detector(
        args.dataset_root, args.dataset_manifest, predictions, detector_eval))
    checked_stage(root, "freeze", [freeze], lambda: pose.freeze_inputs(
        protocol_path=protocol, dataset_root=args.dataset_root,
        dataset_manifest_path=args.dataset_manifest, predictions_root=predictions, output_root=freeze))
    pose.validate_input_manifest(manifest, protocol, verify_assets=True)
    checked_stage(root, "primary", [primary], lambda: pose.run_primary(
        protocol_path=protocol, manifest_path=manifest, foundationpose_root=args.foundationpose_root,
        output_root=primary, implementation_commit=identity["implementation_commit"], resume=primary.exists()))
    checked_stage(root, "evaluation", [evaluation], lambda: pose.evaluate(
        protocol_path=protocol, manifest_path=manifest, primary_root=primary, dataset_root=args.dataset_root,
        toolkit_root=args.toolkit_root, output_root=evaluation))
    def taxonomy_action():
        from scripts.build_failure_taxonomy import extract_taxonomy
        extract_taxonomy(argparse.Namespace(protocol=protocol, manifest=manifest, primary_root=primary,
            dataset_root=args.dataset_root, toolkit_root=args.toolkit_root, output_root=taxonomy,
            result_anchor=evaluation / "evaluation-result.json"))
    checked_stage(root, "taxonomy", [taxonomy], taxonomy_action)
    complete_identity = {**identity,
        "detector_prediction_manifest_sha256": pose._sha256_file(predictions / "prediction-manifest.json"),
        "foundationpose_input_manifest_sha256": pose._sha256_file(manifest),
        "primary_prediction_sha256": pose._sha256_file(primary / "predictions.jsonl"),
        "evaluation_sha256": pose._sha256_file(evaluation / "evaluation-result.json")}
    complete_path = root / "completed-identity.json"
    if complete_path.exists():
        if pose._read_json(complete_path) != complete_identity:
            raise pose.ContractError("Completed identity changed")
    else:
        pose._write_json_atomic(complete_path, complete_identity)
    archive = root.parent / f"{root.name}-evidence.tar.gz"
    if archive.exists():
        raise pose.ContractError("Evidence archive is create-only")
    files = file_inventory(root)
    needed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if shutil.disk_usage(root).free < needed + 10 * 1024**3:
        raise pose.ContractError("Insufficient disk reserve for evidence archive")
    (root / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in files.items()), encoding="utf-8")
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(root, arcname=root.name)
    print(json.dumps({"status": "V1.2_EVIDENCE_PACKAGED", "archive": str(archive), "sha256": pose._sha256_file(archive)}))


if __name__ == "__main__":
    main()
