"""Build and audit the evaluator-only XYZ-IBD validation label overlay.

This module has no scoring entrypoint.  It hashes the frozen prediction bundle
without parsing predictions, reads public validation labels only inside the
evaluator-only readiness boundary, and invokes only the pinned label conversion
utility (never an ``eval_*`` script).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import PACKAGE_VERSION, PROTOCOL_ID


class ContractError(ValueError):
    """Raised when a frozen readiness boundary is violated."""


PROTOCOL_SCHEMA = "poseloop.r3.xyzibd-label-readiness.protocol.v3"
RECEIPT_SCHEMA = "poseloop.r3.xyzibd-label-readiness.receipt.v3"
LOCK_SCHEMA = "poseloop.r3.xyzibd-label-readiness.lock.v3"
SCORER_ENTRYPOINTS = ("eval_bop24_pose.py", "eval_bop22_coco.py", "eval_bop19_pose.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("Readiness protocol must be a JSON object")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("Readiness protocol schema mismatch")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Readiness protocol ID mismatch")
    if value.get("state") != "frozen_before_label_readiness_derivation":
        raise ContractError("Readiness protocol is not frozen before derivation")
    if value.get("scoring", {}).get("evaluator_invocations_allowed") != 0:
        raise ContractError("Readiness protocol must forbid scorer invocation")
    if value.get("label_boundary", {}).get("prediction_selection_label_access") != 0:
        raise ContractError("Prediction/selection label-access contract changed")

    hashes: list[Any] = [
        value.get("source_v2", {}).get("protocol_sha256"),
        value.get("source_v2", {}).get("invocation_started_sha256"),
        value.get("source_v2", {}).get("sealed_receipt_sha256"),
        value.get("dataset", {}).get("val_archive", {}).get("published_lfs_oid_sha256"),
        value.get("toolkit", {}).get("source_tree_sha256"),
    ]
    hashes.extend(value.get("frozen_prediction_sha256", {}).values())
    hashes.extend(value.get("toolkit", {}).get("required_file_sha256", {}).values())
    if not hashes or any(not _valid_sha256(digest) for digest in hashes):
        raise ContractError("Readiness protocol contains an invalid SHA-256")
    if value["dataset"]["val_archive"]["sha256"] != value["dataset"]["val_archive"]["published_lfs_oid_sha256"]:
        raise ContractError("Local archive contract differs from the published LFS OID")
    scenes = value["dataset"].get("expected_val_scene_ids")
    if scenes != list(range(0, 75, 5)):
        raise ContractError("XYZ-IBD validation scene contract changed")
    return value


def _file_contract(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("missing file")
        result["ready"] = False
        return result
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    if result["sha256"] != expected_sha256:
        result["errors"].append("SHA-256 mismatch")
    if expected_bytes is not None and result["bytes"] != expected_bytes:
        result["errors"].append("byte-count mismatch")
    result["ready"] = not result["errors"]
    return result


def scorer_process_audit() -> dict[str, Any]:
    if os.name != "posix":
        return {"supported": False, "count": 0, "matches": [], "errors": []}
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return {
            "supported": True,
            "count": None,
            "matches": [],
            "errors": [f"ps exited {completed.returncode}"],
        }
    matches = [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(entrypoint in line for entrypoint in SCORER_ENTRYPOINTS)
    ]
    return {"supported": True, "count": len(matches), "matches": matches, "errors": []}


def score_file_audit(root: Path) -> dict[str, Any]:
    files: list[str] = []
    if root.is_dir():
        for path in root.rglob("*.json"):
            if path.name.startswith("scores") or path.name == "official_scores.json":
                files.append(str(path.resolve()))
    return {"root": str(root.resolve()), "count": len(files), "files": sorted(files)}


def audit_archive(archive: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract = protocol["dataset"]["val_archive"]
    result = _file_contract(archive, contract["sha256"], int(contract["bytes"]))
    counts = Counter()
    scenes: set[int] = set()
    if result["ready"]:
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                parts = Path(name).parts
                if len(parts) < 4 or parts[0:2] != ("xyzibd_val", "val"):
                    continue
                try:
                    scene_id = int(parts[2])
                except ValueError:
                    continue
                scenes.add(scene_id)
                filename = parts[-1]
                for expected in (
                    "scene_camera_xyz.json",
                    "scene_gt_xyz.json",
                    "scene_gt_info_xyz.json",
                    "scene_gt_coco_xyz.json",
                ):
                    if filename == expected:
                        counts[expected] += 1
    expected_scenes = protocol["dataset"]["expected_val_scene_ids"]
    if sorted(scenes) != expected_scenes:
        result["errors"].append("archive validation scene set mismatch")
    for filename in ("scene_camera_xyz.json", "scene_gt_xyz.json", "scene_gt_info_xyz.json"):
        if counts[filename] != len(expected_scenes):
            result["errors"].append(f"archive {filename} count mismatch")
    if counts["scene_gt_coco_xyz.json"] != 0:
        result["errors"].append("distributed archive unexpectedly contains derived COCO GT")
    result["central_directory"] = {
        "scene_ids": sorted(scenes),
        "label_counts": dict(sorted(counts.items())),
        "scene_gt_coco_release_status": "not_distributed_generate_with_pinned_calc_gt_coco",
    }
    result["ready"] = not result["errors"]
    return result


def audit_toolkit(toolkit_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(toolkit_root.resolve()), "files": {}, "errors": []}
    for relative, expected in protocol["toolkit"]["required_file_sha256"].items():
        item = _file_contract(toolkit_root / relative, expected)
        result["files"][relative] = item
        result["errors"].extend(f"{relative}: {error}" for error in item["errors"])
    tree_contract = protocol["toolkit"]["source_tree_contract"]
    relative_paths: set[Path] = set()
    for relative_root in tree_contract["included_roots"]:
        candidate = toolkit_root / relative_root
        if candidate.is_file():
            relative_paths.add(candidate.relative_to(toolkit_root))
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc":
                    relative_paths.add(path.relative_to(toolkit_root))
    digest = hashlib.sha256()
    normalized_extensions = set(tree_contract["text_eol_normalized_extensions"])
    for relative in sorted(relative_paths, key=lambda value: value.as_posix()):
        data = (toolkit_root / relative).read_bytes()
        if relative.suffix.lower() in normalized_extensions:
            data = data.replace(b"\r\n", b"\n")
        file_sha = hashlib.sha256(data).hexdigest()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    result["source_tree"] = {
        "sha256": digest.hexdigest(),
        "file_count": len(relative_paths),
        "expected_sha256": tree_contract["sha256"],
        "expected_file_count": tree_contract["file_count"],
    }
    if result["source_tree"]["sha256"] != tree_contract["sha256"]:
        result["errors"].append("pinned toolkit source-tree SHA-256 mismatch")
    if result["source_tree"]["file_count"] != tree_contract["file_count"]:
        result["errors"].append("pinned toolkit source-tree file-count mismatch")
    result["ready"] = not result["errors"]
    return result


def audit_extracted_source_labels(dataset_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_scenes = protocol["dataset"]["expected_val_scene_ids"]
    val_root = dataset_root / "val"
    present_scenes = sorted(
        int(path.name)
        for path in val_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ) if val_root.is_dir() else []
    errors: list[str] = []
    if present_scenes != expected_scenes:
        errors.append("extracted validation scene set mismatch")
    files: dict[str, Any] = {}
    source_coco_count = 0
    for scene_id in expected_scenes:
        scene = val_root / f"{scene_id:06d}"
        for filename in ("scene_camera_xyz.json", "scene_gt_xyz.json", "scene_gt_info_xyz.json"):
            path = scene / filename
            key = f"{scene_id:06d}/{filename}"
            item = {"path": str(path.resolve()), "exists": path.is_file()}
            if path.is_file():
                item["sha256"] = sha256_file(path)
                try:
                    if not isinstance(read_json(path), dict):
                        errors.append(f"{key} is not a JSON object")
                except json.JSONDecodeError as exc:
                    errors.append(f"{key} JSON parse error: {exc}")
            else:
                errors.append(f"missing extracted label file: {key}")
            files[key] = item
        if (scene / "scene_gt_coco_xyz.json").exists():
            source_coco_count += 1
    manifest = [
        {"relative_path": key, "sha256": item.get("sha256")}
        for key, item in sorted(files.items())
    ]
    if source_coco_count != 0:
        errors.append("source dataset was modified with derived scene_gt_coco_xyz.json")
    return {
        "scene_ids": present_scenes,
        "required_json_file_count": sum(1 for item in files.values() if item["exists"]),
        "required_json_manifest_sha256": canonical_sha256(manifest),
        "files": files,
        "source_scene_gt_coco_count": source_coco_count,
        "errors": errors,
        "ready": not errors,
    }


def audit_frozen_predictions(input_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(input_root.resolve()),
        "files": {},
        "prediction_files_parsed": False,
        "prediction_selection_label_access_count": 0,
        "errors": [],
    }
    for filename, expected in protocol["frozen_prediction_sha256"].items():
        item = _file_contract(input_root / filename, expected)
        result["files"][filename] = item
        result["errors"].extend(f"{filename}: {error}" for error in item["errors"])
    result["ready"] = not result["errors"]
    return result


def audit_v2_failure_evidence(
    invocation_started: Path,
    sealed_receipt: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    contract = protocol["source_v2"]
    result = {
        "invocation_started": _file_contract(
            invocation_started, contract["invocation_started_sha256"]
        ),
        "sealed_receipt": _file_contract(
            sealed_receipt, contract["sealed_receipt_sha256"]
        ),
        "rerun_permitted": False,
        "immutable": True,
        "errors": [],
    }
    for role in ("invocation_started", "sealed_receipt"):
        result["errors"].extend(
            f"v2 {role}: {error}" for error in result[role]["errors"]
        )
    if result["invocation_started"]["ready"]:
        marker = read_json(invocation_started)
        if marker.get("protocol_id") != contract["protocol_id"]:
            result["errors"].append("v2 invocation protocol ID mismatch")
        if marker.get("evaluate_invocation_count") != 1:
            result["errors"].append("v2 invocation count is not one")
    if result["sealed_receipt"]["ready"]:
        sealed = read_json(sealed_receipt)
        if sealed.get("protocol_id") != contract["protocol_id"]:
            result["errors"].append("v2 sealed protocol ID mismatch")
        if sealed.get("rerun_permitted") is not False:
            result["errors"].append("v2 sealed receipt unexpectedly permits rerun")
        if sealed.get("all_exit_codes") != contract["all_four_exit_codes"]:
            result["errors"].append("v2 sealed exit-code vector changed")
    result["ready"] = not result["errors"]
    return result


def derive_validation_targets(
    dataset_root: Path,
    scene_ids: Sequence[int],
    *,
    images_per_scene: int,
    min_visibility: float,
) -> tuple[list[dict[str, int]], list[dict[str, int]], dict[str, Any]]:
    bop19: list[dict[str, int]] = []
    bop24: list[dict[str, int]] = []
    scene_summary: dict[str, Any] = {}
    mask_full_count = 0
    mask_visib_count = 0
    for scene_id in scene_ids:
        scene = dataset_root / "val" / f"{scene_id:06d}"
        gt = read_json(scene / "scene_gt_xyz.json")
        info = read_json(scene / "scene_gt_info_xyz.json")
        camera = read_json(scene / "scene_camera_xyz.json")
        if not all(isinstance(value, dict) for value in (gt, info, camera)):
            raise ContractError(f"scene {scene_id} GT/camera JSON must be objects")
        if set(gt) != set(info) or set(gt) != set(camera):
            raise ContractError(f"scene {scene_id} GT/info/camera image keys differ")
        image_ids = sorted(int(value) for value in gt)
        if image_ids != list(range(images_per_scene)):
            raise ContractError(f"scene {scene_id} image IDs differ from readiness contract")
        instance_count = 0
        for image_id in image_ids:
            key = str(image_id)
            instances = gt[key]
            metadata = info[key]
            if not isinstance(instances, list) or not isinstance(metadata, list) or len(instances) != len(metadata):
                raise ContractError(f"scene {scene_id} image {image_id} GT/info cardinality mismatch")
            bop24.append({"scene_id": scene_id, "im_id": image_id})
            visible_counts: Counter[int] = Counter()
            for gt_id, (instance, details) in enumerate(zip(instances, metadata)):
                if not isinstance(instance, dict) or not isinstance(details, dict):
                    raise ContractError("GT instance metadata must be JSON objects")
                object_id = instance.get("obj_id")
                visibility = details.get("visib_fract")
                if isinstance(object_id, bool) or not isinstance(object_id, int):
                    raise ContractError("GT object ID must be an integer")
                if isinstance(visibility, bool) or not isinstance(visibility, (int, float)):
                    raise ContractError("GT visibility must be numeric")
                if float(visibility) >= min_visibility:
                    visible_counts[object_id] += 1
                full = scene / "mask_xyz" / f"{image_id:06d}_{gt_id:06d}.png"
                visible = scene / "mask_visib_xyz" / f"{image_id:06d}_{gt_id:06d}.png"
                if not full.is_file() or not visible.is_file():
                    raise ContractError(f"scene {scene_id} image {image_id} mask pair is missing")
                mask_full_count += 1
                mask_visib_count += 1
                instance_count += 1
            for object_id in sorted(visible_counts):
                bop19.append(
                    {
                        "scene_id": scene_id,
                        "im_id": image_id,
                        "obj_id": object_id,
                        "inst_count": visible_counts[object_id],
                    }
                )
        scene_summary[f"{scene_id:06d}"] = {
            "image_count": len(image_ids),
            "instance_count": instance_count,
        }
    summary = {
        "scene_count": len(scene_ids),
        "image_count": len(bop24),
        "bop19_target_count": len(bop19),
        "bop24_target_count": len(bop24),
        "mask_full_count": mask_full_count,
        "mask_visib_count": mask_visib_count,
        "scenes": scene_summary,
    }
    return bop19, bop24, summary


def _safe_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ContractError(f"overlay symlink target mismatch: {destination}")
        return
    if destination.exists():
        raise ContractError(f"overlay destination already exists and is not a symlink: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_overlay(
    *,
    source_dataset_root: Path,
    evaluator_root: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    if os.name != "posix":
        raise ContractError("Evaluator-only overlay preparation is supported only on POSIX")
    expected_basename = protocol["label_boundary"]["evaluator_namespace_basename"]
    if evaluator_root.name != expected_basename:
        raise ContractError(f"evaluator root basename must be {expected_basename}")
    source_dataset_root = source_dataset_root.resolve()
    evaluator_root = evaluator_root.resolve()
    if evaluator_root == source_dataset_root or source_dataset_root in evaluator_root.parents:
        raise ContractError("evaluator-only root must not be inside the source dataset")

    dataset_overlay = evaluator_root / "datasets" / "xyzibd"
    dataset_overlay.mkdir(parents=True, exist_ok=True)
    marker_path = evaluator_root / "namespace-marker.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("protocol_id") != PROTOCOL_ID or marker.get("source_dataset_root") != str(source_dataset_root):
            raise ContractError("existing evaluator namespace marker differs from this protocol")
    else:
        write_json_atomic(
            marker_path,
            {
                "schema_version": "poseloop.r3.xyzibd-label-readiness.namespace.v3",
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": protocol_sha256,
                "source_dataset_root": str(source_dataset_root),
                "purpose": "evaluator_only_labels_never_prediction_or_selection",
            },
        )

    for name in ("camera_photoneo.json", "camera_realsense.json", "camera_xyz.json", "models", "models_eval"):
        _safe_symlink(source_dataset_root / name, dataset_overlay / name)
    val_overlay = dataset_overlay / "val"
    val_overlay.mkdir(exist_ok=True)
    for scene_id in protocol["dataset"]["expected_val_scene_ids"]:
        source_scene = source_dataset_root / "val" / f"{scene_id:06d}"
        overlay_scene = val_overlay / f"{scene_id:06d}"
        overlay_scene.mkdir(exist_ok=True)
        for child in source_scene.iterdir():
            if child.name == "scene_gt_coco_xyz.json":
                continue
            _safe_symlink(child, overlay_scene / child.name)

    bop19, bop24, target_summary = derive_validation_targets(
        source_dataset_root,
        protocol["dataset"]["expected_val_scene_ids"],
        images_per_scene=int(protocol["dataset"]["images_per_val_scene"]),
        min_visibility=float(protocol["target_derivation"]["min_visibility"]),
    )
    bop19_path = dataset_overlay / protocol["target_derivation"]["bop19_filename"]
    bop24_path = dataset_overlay / protocol["target_derivation"]["bop24_filename"]
    write_json_atomic(bop19_path, bop19)
    write_json_atomic(bop24_path, bop24)
    return {
        "evaluator_root": str(evaluator_root),
        "datasets_root": str((evaluator_root / "datasets").resolve()),
        "dataset_overlay": str(dataset_overlay.resolve()),
        "namespace_marker": str(marker_path.resolve()),
        "targets": {
            "bop19": {"path": str(bop19_path.resolve()), "sha256": sha256_file(bop19_path), "rows": len(bop19)},
            "bop24": {"path": str(bop24_path.resolve()), "sha256": sha256_file(bop24_path), "rows": len(bop24)},
        },
        "target_summary": target_summary,
    }


def build_coco_generation_command(
    *,
    python_executable: str,
    toolkit_root: Path,
    datasets_root: Path,
    protocol: Mapping[str, Any],
) -> list[str]:
    return [
        python_executable,
        str((toolkit_root / "scripts" / "calc_gt_coco.py").resolve()),
        "--dataset=xyzibd",
        "--dataset_split=val",
        f"--targets_filename={protocol['target_derivation']['bop19_filename']}",
        "--use_all_gt",
        "--bbox_type=amodal",
        f"--datasets_path={datasets_root.resolve()}",
    ]


def run_coco_generation(
    *,
    python_executable: str,
    toolkit_root: Path,
    evaluator_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_overlay = evaluator_root / "datasets" / "xyzibd"
    scene_ids = protocol["dataset"]["expected_val_scene_ids"]
    outputs = [dataset_overlay / "val" / f"{scene_id:06d}" / "scene_gt_coco_xyz.json" for scene_id in scene_ids]
    existing = [path for path in outputs if path.exists()]
    if existing and len(existing) != len(outputs):
        raise ContractError("partial COCO GT derivation exists; preserve it and use a fresh evaluator namespace")
    command = build_coco_generation_command(
        python_executable=python_executable,
        toolkit_root=toolkit_root,
        datasets_root=evaluator_root / "datasets",
        protocol=protocol,
    )
    before_process = scorer_process_audit()
    before_scores = score_file_audit(evaluator_root)
    if before_process["errors"] or before_process["count"] != 0 or before_scores["count"] != 0:
        raise ContractError("scorer process or score file exists before label derivation")
    if len(existing) == len(outputs):
        return {
            "status": "already_complete_not_rerun",
            "command": command,
            "exit_code": None,
            "output_count": len(outputs),
            "official_evaluator_entered": False,
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(toolkit_root.resolve())
    completed = subprocess.run(command, env=env, check=False)
    after_process = scorer_process_audit()
    after_scores = score_file_audit(evaluator_root)
    produced = [path for path in outputs if path.is_file()]
    result = {
        "status": "complete" if completed.returncode == 0 and len(produced) == len(outputs) else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "output_count": len(produced),
        "official_evaluator_entered": False,
        "scorer_process_before": before_process,
        "scorer_process_after": after_process,
        "score_files_before": before_scores,
        "score_files_after": after_scores,
        "errors": [],
    }
    if completed.returncode != 0:
        result["errors"].append(f"pinned calc_gt_coco.py exited {completed.returncode}")
    if len(produced) != len(outputs):
        result["errors"].append("pinned calc_gt_coco.py did not create all expected files")
    if after_process["errors"] or after_process["count"] != 0 or after_scores["count"] != 0:
        result["errors"].append("scorer process or score file appeared during label derivation")
    if result["errors"]:
        result["status"] = "failed"
    return result


_LOADER_SMOKE = r'''import json
import pathlib
import sys
from bop_toolkit_lib import dataset_params, inout

datasets_root = pathlib.Path(sys.argv[1]).resolve()
scene_ids = [int(value) for value in sys.argv[2].split(",")]
dp = dataset_params.get_split_params(str(datasets_root), "xyzibd", "val")
if list(dp["scene_ids"]) != scene_ids:
    raise RuntimeError("loader scene IDs differ from readiness protocol")
images = annotations = 0
for scene_id in scene_ids:
    keys = dataset_params.scene_tpaths_keys(dp["eval_modality"], dp["eval_sensor"], scene_id)
    gt = inout.load_scene_gt(dp[keys["scene_gt_tpath"]].format(scene_id=scene_id))
    info = inout.load_json(dp[keys["scene_gt_info_tpath"]].format(scene_id=scene_id), keys_to_int=True)
    coco = inout.load_json(dp[keys["scene_gt_coco_tpath"]].format(scene_id=scene_id), keys_to_int=True)
    if set(gt) != set(info):
        raise RuntimeError("loader GT/info keys differ")
    images += len(coco["images"])
    annotations += len(coco["annotations"])
print("POSELOOP_READINESS_LOADER=" + json.dumps({"scene_count": len(scene_ids), "image_count": images, "annotation_count": annotations}, sort_keys=True))
'''


def loader_smoke(
    *,
    python_executable: str,
    toolkit_root: Path,
    datasets_root: Path,
    scene_ids: Sequence[int],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(toolkit_root.resolve())
    completed = subprocess.run(
        [python_executable, "-c", _LOADER_SMOKE, str(datasets_root.resolve()), ",".join(map(str, scene_ids))],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    prefix = "POSELOOP_READINESS_LOADER="
    payloads = [line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)]
    payload = json.loads(payloads[0]) if completed.returncode == 0 and len(payloads) == 1 else None
    return {
        "command_exit_code": completed.returncode,
        "payload": payload,
        "stderr": completed.stderr,
        "ready": completed.returncode == 0 and payload is not None,
    }


def audit_derived_coco(dataset_overlay: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    files: dict[str, Any] = {}
    total_images = 0
    total_annotations = 0
    image_keys: set[tuple[int, int]] = set()
    annotation_keys: set[tuple[int, int, int]] = set()
    for scene_id in protocol["dataset"]["expected_val_scene_ids"]:
        path = dataset_overlay / "val" / f"{scene_id:06d}" / "scene_gt_coco_xyz.json"
        item: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing derived COCO GT for scene {scene_id}")
            files[f"{scene_id:06d}"] = item
            continue
        item["sha256"] = sha256_file(path)
        try:
            value = read_json(path)
            images = value["images"]
            annotations = value["annotations"]
            if len(images) != int(protocol["dataset"]["images_per_val_scene"]):
                errors.append(f"scene {scene_id} COCO image count mismatch")
            for image in images:
                image_keys.add((scene_id, int(image["id"])))
            for annotation in annotations:
                if not {"image_id", "category_id", "bbox", "segmentation"}.issubset(annotation):
                    raise ContractError("COCO annotation required fields are missing")
                annotation_keys.add((scene_id, int(annotation["image_id"]), int(annotation["category_id"])))
            item["image_count"] = len(images)
            item["annotation_count"] = len(annotations)
            total_images += len(images)
            total_annotations += len(annotations)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ContractError) as exc:
            errors.append(f"scene {scene_id} derived COCO parse error: {exc}")
        files[f"{scene_id:06d}"] = item
    return {
        "files": files,
        "file_count": sum(1 for item in files.values() if item.get("exists")),
        "image_count": total_images,
        "annotation_count": total_annotations,
        "image_keys": image_keys,
        "annotation_keys": annotation_keys,
        "errors": errors,
        "ready": not errors,
    }


def create_readiness_receipt(
    *,
    protocol_path: Path,
    dataset_root: Path,
    archive: Path,
    toolkit_root: Path,
    input_root: Path,
    v2_invocation_started: Path,
    v2_sealed_receipt: Path,
    evaluator_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    archive_audit = audit_archive(archive, protocol)
    toolkit_audit = audit_toolkit(toolkit_root, protocol)
    prediction_audit = audit_frozen_predictions(input_root, protocol)
    v2_audit = audit_v2_failure_evidence(
        v2_invocation_started, v2_sealed_receipt, protocol
    )
    source_label_audit = audit_extracted_source_labels(dataset_root, protocol)
    process_audit = scorer_process_audit()
    scores_audit = score_file_audit(evaluator_root)
    errors = [
        *archive_audit["errors"],
        *toolkit_audit["errors"],
        *prediction_audit["errors"],
        *v2_audit["errors"],
        *source_label_audit["errors"],
    ]
    if process_audit["errors"] or process_audit["count"] != 0:
        errors.append("official scorer process count is not zero")
    if scores_audit["count"] != 0:
        errors.append("score files exist in the readiness namespace")

    dataset_overlay = evaluator_root / "datasets" / "xyzibd"
    target_summary: dict[str, Any] | None = None
    target_files: dict[str, Any] = {}
    coco_public: dict[str, Any] = {"ready": False, "errors": ["overlay is missing"]}
    loader: dict[str, Any] = {"ready": False, "errors": ["overlay is missing"]}
    if dataset_overlay.is_dir():
        try:
            bop19, bop24, target_summary = derive_validation_targets(
                dataset_root,
                protocol["dataset"]["expected_val_scene_ids"],
                images_per_scene=int(protocol["dataset"]["images_per_val_scene"]),
                min_visibility=float(protocol["target_derivation"]["min_visibility"]),
            )
            for role, expected_rows, filename in (
                ("bop19", bop19, protocol["target_derivation"]["bop19_filename"]),
                ("bop24", bop24, protocol["target_derivation"]["bop24_filename"]),
            ):
                path = dataset_overlay / filename
                item = {"path": str(path.resolve()), "exists": path.is_file()}
                if path.is_file():
                    actual = read_json(path)
                    item.update(sha256=sha256_file(path), rows=len(actual), exact_content_match=actual == expected_rows)
                    if actual != expected_rows:
                        errors.append(f"derived {role} validation targets differ from frozen derivation")
                else:
                    errors.append(f"derived {role} validation targets are missing")
                target_files[role] = item
            coco_public = audit_derived_coco(dataset_overlay, protocol)
            errors.extend(coco_public["errors"])
            if coco_public["ready"]:
                bop24_keys = {(row["scene_id"], row["im_id"]) for row in bop24}
                bop19_keys = {(row["scene_id"], row["im_id"], row["obj_id"]) for row in bop19}
                missing_bop24 = bop24_keys - coco_public["image_keys"]
                missing_bop19 = bop19_keys - coco_public["annotation_keys"]
                target_summary["coco_target_coverage"] = {
                    "bop24_missing_keys": len(missing_bop24),
                    "bop19_missing_keys": len(missing_bop19),
                    "ready": not missing_bop24 and not missing_bop19,
                }
                if missing_bop24 or missing_bop19:
                    errors.append("derived COCO GT does not cover all validation target keys")
                loader = loader_smoke(
                    python_executable=python_executable,
                    toolkit_root=toolkit_root,
                    datasets_root=evaluator_root / "datasets",
                    scene_ids=protocol["dataset"]["expected_val_scene_ids"],
                )
                if not loader["ready"]:
                    errors.append("pinned toolkit loader smoke failed")
        except (ContractError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"validation label audit failed: {exc}")

    # Sets are useful internally but receipts must be canonical JSON.
    coco_receipt = {key: value for key, value in coco_public.items() if key not in {"image_keys", "annotation_keys"}}
    fingerprint = {
        "protocol_sha256": sha256_file(protocol_path),
        "archive_sha256": archive_audit.get("sha256"),
        "extracted_source_label_manifest_sha256": source_label_audit["required_json_manifest_sha256"],
        "toolkit_file_sha256": {name: item.get("sha256") for name, item in toolkit_audit["files"].items()},
        "frozen_prediction_sha256": {name: item.get("sha256") for name, item in prediction_audit["files"].items()},
        "v2_invocation_started_sha256": v2_audit["invocation_started"].get("sha256"),
        "v2_sealed_receipt_sha256": v2_audit["sealed_receipt"].get("sha256"),
        "target_sha256": {name: item.get("sha256") for name, item in target_files.items()},
        "derived_coco_sha256": {name: item.get("sha256") for name, item in coco_receipt.get("files", {}).items()},
    }
    return {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "package_version": PACKAGE_VERSION,
        "status": "ready" if not errors else "blocked",
        "protocol": {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)},
        "official_distribution": {
            "revision": protocol["dataset"]["source_revision"],
            "archive_url": protocol["dataset"]["val_archive"]["url"],
            "published_lfs_oid_sha256": protocol["dataset"]["val_archive"]["published_lfs_oid_sha256"],
            "scene_gt_coco_release_status": "not_distributed_official_toolkit_derivation_required",
        },
        "archive": archive_audit,
        "extracted_source_labels": source_label_audit,
        "toolkit": toolkit_audit,
        "immutable_v2_failure_evidence": v2_audit,
        "frozen_predictions": prediction_audit,
        "label_boundary": {
            "source_dataset_root": str(dataset_root.resolve()),
            "evaluator_root": str(evaluator_root.resolve()),
            "prediction_selection_label_access_count": 0,
            "evaluator_only_label_access": True,
            "original_dataset_modified": False,
        },
        "target_files": target_files,
        "target_summary": target_summary,
        "derived_coco": coco_receipt,
        "loader_smoke": loader,
        "official_scorer_process_audit": process_audit,
        "score_file_audit": scores_audit,
        "official_evaluator_executed": False,
        "scoring_authorization_created": False,
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "errors": errors,
    }


def freeze_readiness_lock(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != "ready":
        raise ContractError("Cannot freeze a blocked label-readiness receipt")
    value = dict(receipt)
    value["schema_version"] = LOCK_SCHEMA
    value["state"] = "label_assets_ready_frozen_before_v3_scoring_protocol"
    return value
