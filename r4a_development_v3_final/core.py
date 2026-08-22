"""Freeze, prepare, execute once, and summarize R4-A v3 development evaluation."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from r4a_development_repair.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)

from . import SUPPORTED_PROTOCOLS

PRE_SCORE_SCHEMA = "poseloop.r4a.development-final-pre-score.v1"
OFFICIAL_ONCE_SCHEMA = "poseloop.r4a.development-final-official-once.v1"
FINAL_REPORT_SCHEMA = "poseloop.r4a.development-final-report.v1"
VISUALIZATION_SCHEMA = "poseloop.r4a.development-final-visualization.v1"


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    inherited = value.get("inherits_protocol")
    if inherited is not None:
        relative = inherited.get("relative_path") if isinstance(inherited, dict) else None
        if relative not in {
            "poseloop_r4a_v3_development_final_evaluation_v2.json",
            "poseloop_r4a_v3_development_final_evaluation_v3.json",
        }:
            raise ContractError("R4-A final inherited protocol path changed")
        base_path = path.parent / relative
        if sha256_file(base_path) != inherited.get("sha256"):
            raise ContractError("R4-A final inherited protocol hash changed")
        base = load_protocol(base_path)

        def update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
            for key, item in source.items():
                if isinstance(item, dict) and isinstance(target.get(key), dict):
                    update(target[key], item)
                else:
                    target[key] = copy.deepcopy(item)

        merged = copy.deepcopy(base)
        update(merged, value)
        value = merged
    if SUPPORTED_PROTOCOLS.get(value.get("protocol_id")) != value.get("schema_version"):
        raise ContractError("R4-A final evaluation protocol identity mismatch")
    if value.get("state") not in {
        "frozen_before_gpu_c_prediction_bundle_open_and_before_any_official_development_evaluator_call",
        "frozen_after_label_free_gpu_c_return_hash_verification_before_any_official_development_evaluator_call",
        "frozen_after_v2_environment_failure_before_any_v3_official_development_evaluator_call",
        "frozen_after_v3_static_hash_rejection_before_any_v4_official_development_evaluator_call",
    }:
        raise ContractError("R4-A final evaluation protocol state changed")
    if value.get("role") != "DEVELOPMENT_ONLY":
        raise ContractError("R4-A final evaluation role changed")
    upstream = value.get("upstream_r4a", {})
    for key in (
        "protocol_sha256",
        "exact_selection_sha256",
        "no_gt_archive_sha256",
        "transfer_descriptor_sha256",
        "workload_sha256",
        "coco_sha256",
        "evaluator_only_archive_sha256",
    ):
        if not _valid_sha(upstream.get(key)):
            raise ContractError(f"R4-A final evaluation invalid upstream hash: {key}")
    if value["multi_view"]["score_or_gt_guided_selection"] is not False:
        raise ContractError("R4-A final multi-view selection is no longer label blind")
    upstream_r4c = value.get("upstream_r4c", {})
    for key in (
        "execution_protocol_sha256",
        "adapter_protocol_sha256",
        "adapter_lock_sha256",
        "flat_camera_intrinsics_amendment_protocol_sha256",
        "flat_camera_intrinsics_amendment_lock_sha256",
    ):
        if not _valid_sha(upstream_r4c.get(key)):
            raise ContractError(f"R4-A final evaluation invalid GPU-C hash: {key}")
    if value["bop24_development"]["variant_order"] != ["single_view", "multi_view"]:
        raise ContractError("R4-A final BOP24 command order changed")
    if value["bop24_development"]["maximum_calls_per_variant"] != 1:
        raise ContractError("R4-A final BOP24 call count changed")
    boundaries = value["immutable_boundaries"]
    for key in (
        "xyzibd_val_access_permitted",
        "xyzibd_val_evaluator_permitted",
        "prediction_checkpoint_candidate_or_input_change_permitted",
        "score_guided_selection_or_rerun_permitted",
        "official_evaluator_rerun_after_success_or_failure_permitted",
        "evaluator_bundle_transfer_to_gpu_c_permitted",
        "push_merge_tag_permitted",
        "release_or_delete_instance_or_disk_permitted",
    ):
        if boundaries.get(key) is not False:
            raise ContractError(f"R4-A final immutable boundary changed: {key}")
    return value


def _c_result_rows(protocol: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    contract = protocol.get("gpu_c_result_adapter")
    if contract is None:
        return rows
    if contract.get("input_schema") != "poseloop_gpu_merged_metadata_then_result_rows" or contract.get("metadata_rows") != 1:
        raise ContractError("R4-A final GPU-C result adapter contract changed")
    if not rows or rows[0].get("record_type") != "poseloop_gpu_merged_metadata":
        raise ContractError("R4-A final GPU-C merged metadata row is missing")
    metadata = rows[0]
    results = rows[1:]
    if int(metadata.get("item_count", -1)) != 10 or len(results) != 10:
        raise ContractError("R4-A final GPU-C merged metadata item count differs")
    if any(row.get("record_type") != "poseloop_gpu_result" for row in results):
        raise ContractError("R4-A final GPU-C result row type differs")
    return results


def _validate_c_return(
    protocol: Mapping[str, Any],
    archive_path: Path,
    descriptor_path: Path,
    results_path: Path,
) -> None:
    expected = protocol.get("upstream_r4c", {}).get("return_bundle")
    if expected is None:
        return
    descriptor = read_json(descriptor_path)
    checks = {
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "descriptor_sha256": sha256_file(descriptor_path),
        "merged_results_sha256": sha256_file(results_path),
    }
    for key, observed in checks.items():
        if observed != expected.get(key):
            raise ContractError(f"R4-A final GPU-C return binding differs: {key}")
    if (
        descriptor.get("schema_version") != "poseloop.r4c.multi-object-return-descriptor.v1"
        or descriptor.get("passed") is not True
        or descriptor.get("label_access_count") != 0
        or descriptor.get("official_scorer_run") is not False
        or descriptor.get("coverage", {}).get("expected") != 10
        or descriptor.get("coverage", {}).get("observed") != 10
        or descriptor.get("coverage", {}).get("missing") != []
        or descriptor.get("coverage", {}).get("extra") != []
        or descriptor.get("coverage", {}).get("duplicate") != []
        or descriptor.get("internal_sha256sums", {}).get("sha256") != expected.get("internal_sha256sums_sha256")
        or descriptor.get("internal_sha256sums", {}).get("entry_count") != 53
        or descriptor.get("internal_sha256sums", {}).get("verified") is not True
        or descriptor.get("primary_outputs", {}).get("execution/merged-results.jsonl", {}).get("sha256")
        != expected.get("merged_results_sha256")
        or descriptor.get("primary_outputs", {}).get("execution/execution-evidence.json", {}).get("sha256")
        != expected.get("execution_evidence_sha256")
    ):
        raise ContractError("R4-A final GPU-C return descriptor boundary differs")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"JSONL row {number} is not an object: {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _reject_val(path: Path) -> Path:
    resolved = path.resolve()
    lowered = {part.lower() for part in resolved.parts}
    if "val" in lowered or any("official_once" in part.lower() and "r4a" not in str(resolved).lower() for part in resolved.parts):
        raise ContractError(f"R4-A final path crosses a forbidden validation namespace: {resolved}")
    return resolved


def _pose(value: Any, *, context: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ContractError(f"{context} is not a finite 4x4 pose")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ContractError(f"{context} has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ContractError(f"{context} rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
        raise ContractError(f"{context} rotation determinant differs from one")
    return pose


def validate_results(workload: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = [str(row["item_id"]) for row in workload]
    by_id = {str(row.get("item_id", "")): dict(row) for row in results}
    if len(by_id) != len(results) or set(by_id) != set(expected):
        raise ContractError("GPU-C result item coverage differs from workload")
    ordered = []
    for payload in workload:
        item_id = str(payload["item_id"])
        row = by_id[item_id]
        if row.get("status") != "success":
            raise ContractError(f"GPU-C result is not success: {item_id}")
        for field in ("scene_id", "image_id", "object_id", "detection_index"):
            if int(row.get(field, -1)) != int(payload[field]):
                raise ContractError(f"GPU-C result identity differs for {item_id}: {field}")
        for flag in ("evaluator_label_read", "uses_gt_visible_mask", "uses_oracle_association", "result_selection_uses_evaluator_metrics"):
            if row.get(flag) is not False:
                raise ContractError(f"GPU-C label-blind flag changed for {item_id}: {flag}")
        if int(row.get("candidate_limit", -1)) != 252 or int(row.get("pose_hypothesis_count", -1)) != 252:
            raise ContractError(f"GPU-C result is not frozen c252: {item_id}")
        row["predicted_model_to_camera_pose_m"] = _pose(
            row.get("predicted_model_to_camera_pose_m"), context=f"GPU-C pose {item_id}"
        ).tolist()
        row["camera_world_to_camera_pose_m"] = _pose(
            row.get("camera_world_to_camera_pose_m"), context=f"GPU-C camera {item_id}"
        ).tolist()
        elapsed = float(row.get("registration_seconds", math.nan))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ContractError(f"GPU-C runtime is invalid: {item_id}")
        ordered.append(row)
    return ordered


def _rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def multi_view_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        camera = _pose(row["camera_world_to_camera_pose_m"], context="multi camera")
        model_camera = _pose(row["predicted_model_to_camera_pose_m"], context="multi prediction")
        row["world_model_pose"] = (np.linalg.inv(camera) @ model_camera).tolist()
        grouped[(int(row["scene_id"]), int(row["object_id"]))].append(row)
    medoid_by_key: dict[tuple[int, int], np.ndarray] = {}
    for key, group in sorted(grouped.items()):
        choices = []
        world = [_pose(row["world_model_pose"], context="world pose") for row in group]
        for index, left in enumerate(world):
            cost = 0.0
            for right in world:
                cost += float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
                cost += _rotation_angle(left[:3, :3], right[:3, :3]) * 0.001
            choices.append((cost, int(group[index]["detection_index"]), index))
        medoid_by_key[key] = world[min(choices)[2]]
    output = []
    for raw in rows:
        row = dict(raw)
        key = (int(row["scene_id"]), int(row["object_id"]))
        camera = _pose(row["camera_world_to_camera_pose_m"], context="target camera")
        pose = camera @ medoid_by_key[key]
        row["predicted_model_to_camera_pose_m"] = _pose(pose, context="multi projected pose").tolist()
        row["multi_view_group_size"] = len(grouped[key])
        row["multi_view_medoid_rule"] = "fixed_world_pose_medoid"
        output.append(row)
    return output


def _write_bop(path: Path, rows: Sequence[Mapping[str, Any]], *, multi: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        groups = Counter((int(row["scene_id"]), int(row["object_id"])) for row in rows)
        for row in rows:
            pose = _pose(row["predicted_model_to_camera_pose_m"], context="BOP pose")
            rotation = " ".join(f"{value:.12g}" for value in pose[:3, :3].reshape(-1))
            translation = " ".join(f"{value * 1000.0:.12g}" for value in pose[:3, 3])
            elapsed = float(row["registration_seconds"])
            if multi:
                elapsed *= groups[(int(row["scene_id"]), int(row["object_id"]))]
            writer.writerow([row["scene_id"], row["image_id"], row["object_id"], 1.0, rotation, translation, elapsed])
    temporary.replace(path)


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ContractError(f"R4-A final source file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _environment_python(path: Path) -> Path:
    """Return an absolute interpreter path without dereferencing its venv symlink."""
    value = Path(os.path.abspath(os.fspath(path)))
    if not value.is_file():
        raise ContractError(f"R4-A final Python interpreter is missing: {value}")
    return value


def _verify_sums(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise ContractError(f"R4-A final bundle has no SHA256SUMS: {root}")
    verified = {}
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ContractError(f"Malformed SHA256SUMS line {number}") from error
        path = root / relative
        if not _valid_sha(digest) or not path.is_file() or sha256_file(path) != digest:
            raise ContractError(f"R4-A final bundle member hash differs: {relative}")
        verified[relative] = digest
    return verified


def prepare(
    *,
    protocol_path: Path,
    repo_root: Path,
    workload_path: Path,
    coco_path: Path,
    inference_root: Path,
    c_results_path: Path,
    c_output_archive_path: Path,
    c_output_descriptor_path: Path,
    evaluator_bundle_archive_path: Path,
    evaluator_stage_root: Path,
    models_eval_source: Path,
    toolkit_root: Path,
    venv_python: Path,
    output_root: Path,
    pre_score_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    paths = [
        workload_path,
        coco_path,
        inference_root,
        c_results_path,
        c_output_archive_path,
        c_output_descriptor_path,
        evaluator_bundle_archive_path,
        evaluator_stage_root,
        models_eval_source,
        toolkit_root,
        venv_python,
        output_root,
        pre_score_path,
    ]
    for path in paths:
        _reject_val(path)
    upstream = protocol["upstream_r4a"]
    if sha256_file(workload_path) != upstream["workload_sha256"] or sha256_file(coco_path) != upstream["coco_sha256"]:
        raise ContractError("R4-A final no-GT contracts changed")
    if sha256_file(evaluator_bundle_archive_path) != upstream["evaluator_only_archive_sha256"]:
        raise ContractError("R4-A final evaluator-only archive changed")
    _validate_c_return(protocol, c_output_archive_path, c_output_descriptor_path, c_results_path)
    inference_members = _verify_sums(inference_root)
    evaluator_members = _verify_sums(evaluator_stage_root)
    workload = read_jsonl(workload_path)
    if len(workload) != 10 or len({(int(r["scene_id"]), int(r["image_id"]), int(r["object_id"])) for r in workload}) != 10:
        raise ContractError("R4-A final workload is not 10/10")
    results = validate_results(workload, _c_result_rows(protocol, c_results_path))
    multi = multi_view_predictions(results)
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ContractError(f"R4-A final pre-score output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    datasets = root / "datasets"
    results_root = root / "results"
    eval_root = root / "eval"
    raw_root = root / "predictions"
    source_split = evaluator_stage_root / "xyzibd" / "train_pbr_r4a_v3"
    target_source = evaluator_stage_root / "xyzibd" / "train_pbr_r4a_v3_targets_bop24.json"
    targets = read_json(target_source)
    selected_scenes = sorted({int(row["scene_id"]) for row in workload})
    for scene_id in selected_scenes:
        source_scene = source_split / f"{scene_id:06d}"
        target_scene = datasets / "xyzibd" / "train" / f"{scene_id:06d}"
        _copy(source_scene / "scene_gt.json", target_scene / "scene_gt_xyz.json")
        _copy(source_scene / "scene_gt_info.json", target_scene / "scene_gt_info_xyz.json")
        _copy(source_scene / "scene_camera.json", target_scene / "scene_camera_xyz.json")
    write_json_atomic(datasets / "xyzibd" / protocol["bop24_development"]["targets_filename"], targets)
    models_target = datasets / "xyzibd" / "models_eval"
    model_files = []
    for source in sorted(models_eval_source.glob("obj_*.ply")):
        target = models_target / source.name
        _copy(source, target)
        model_files.append({"name": source.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)})
    _copy(models_eval_source / "models_info.json", models_target / "models_info.json")
    if len(model_files) != 15:
        raise ContractError("R4-A final public model domain is not 15 objects")
    _write_jsonl(raw_root / "single_view.jsonl", results)
    _write_jsonl(raw_root / "multi_view.jsonl", multi)
    filenames = protocol["bop24_development"]["result_filenames"]
    single_csv = results_root / filenames["single_view"]
    multi_csv = results_root / filenames["multi_view"]
    _write_bop(single_csv, results, multi=False)
    _write_bop(multi_csv, multi, multi=True)
    toolkit_entry = toolkit_root / "scripts" / "eval_bop24_pose.py"
    if sha256_file(toolkit_entry) != protocol["toolkit"]["eval_bop24_pose_sha256"]:
        raise ContractError("R4-A final pinned BOP24 entrypoint changed")
    command_python = _environment_python(venv_python)
    commands = []
    for variant in protocol["bop24_development"]["variant_order"]:
        commands.append(
            {
                "variant": variant,
                "command": [
                    str(command_python),
                    str(toolkit_entry.resolve()),
                    "--renderer_type=python",
                    f"--result_filenames={filenames[variant]}",
                    f"--results_path={results_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['bop24_development']['targets_filename']}",
                    f"--num_workers={protocol['toolkit']['workers']}",
                ],
                "expected_score": str((eval_root / Path(filenames[variant]).stem / "scores_bop24.json").resolve()),
            }
        )
    if _git(repo_root, "status", "--short", "--untracked-files=no"):
        raise ContractError("R4-A final repository has tracked changes at pre-score freeze")
    receipt = {
        "schema_version": PRE_SCORE_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": _git(repo_root, "rev-parse", "HEAD"),
        "repository_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "repository_tracked_clean": True,
        "inputs": {
            "workload": {"path": str(workload_path.resolve()), "sha256": sha256_file(workload_path)},
            "coco": {"path": str(coco_path.resolve()), "sha256": sha256_file(coco_path)},
            "c_results": {"path": str(c_results_path.resolve()), "sha256": sha256_file(c_results_path)},
            "c_output_archive": {"path": str(c_output_archive_path.resolve()), "bytes": c_output_archive_path.stat().st_size, "sha256": sha256_file(c_output_archive_path)},
            "c_output_descriptor": {"path": str(c_output_descriptor_path.resolve()), "sha256": sha256_file(c_output_descriptor_path)},
            "evaluator_bundle": {"path": str(evaluator_bundle_archive_path.resolve()), "bytes": evaluator_bundle_archive_path.stat().st_size, "sha256": sha256_file(evaluator_bundle_archive_path)},
            "inference_member_count": len(inference_members),
            "evaluator_member_count": len(evaluator_members),
        },
        "coverage": {"expected": 10, "successful": len(results), "fraction": len(results) / 10},
        "se3": {"single_invalid": 0, "multi_invalid": 0},
        "paths": {
            "output_root": str(root),
            "datasets": str(datasets.resolve()),
            "results": str(results_root.resolve()),
            "eval": str(eval_root.resolve()),
            "inference_root": str(inference_root.resolve()),
            "models_eval": str(models_target.resolve()),
            "models_info": str((models_target / "models_info.json").resolve()),
            "targets": str((datasets / "xyzibd" / protocol["bop24_development"]["targets_filename"]).resolve()),
            "single_jsonl": str((raw_root / "single_view.jsonl").resolve()),
            "multi_jsonl": str((raw_root / "multi_view.jsonl").resolve()),
            "single_csv": str(single_csv.resolve()),
            "multi_csv": str(multi_csv.resolve()),
            "toolkit_root": str(toolkit_root.resolve()),
            "venv_python": str(command_python),
        },
        "hashes": {
            "single_jsonl": sha256_file(raw_root / "single_view.jsonl"),
            "multi_jsonl": sha256_file(raw_root / "multi_view.jsonl"),
            "single_csv": sha256_file(single_csv),
            "multi_csv": sha256_file(multi_csv),
            "targets": sha256_file(datasets / "xyzibd" / protocol["bop24_development"]["targets_filename"]),
            "models_info": sha256_file(models_target / "models_info.json"),
            "models_tree": canonical_sha256(model_files),
        },
        "model_files": model_files,
        "commands": commands,
        "official_evaluator_invocation_count_at_freeze": 0,
        "xyzibd_val_access_count": 0,
        "label_access_role": "GPU_A_EVALUATOR_ONLY_DEVELOPMENT",
        "result_or_score_guided_selection": False,
    }
    receipt["lock_sha256"] = canonical_sha256(receipt)
    write_json_atomic(pre_score_path, receipt)
    return receipt


def _verify_pre_score(path: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = read_json(path)
    if receipt.get("schema_version") != PRE_SCORE_SCHEMA or receipt.get("protocol_id") != protocol["protocol_id"]:
        raise ContractError("R4-A final pre-score receipt identity mismatch")
    unlocked = dict(receipt)
    lock = unlocked.pop("lock_sha256", None)
    if lock != canonical_sha256(unlocked):
        raise ContractError("R4-A final pre-score lock mismatch")
    if receipt.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("R4-A final protocol changed after pre-score freeze")
    for key in ("single_jsonl", "multi_jsonl", "single_csv", "multi_csv", "targets", "models_info"):
        if sha256_file(Path(receipt["paths"][key])) != receipt["hashes"][key]:
            raise ContractError(f"R4-A final pre-score output changed: {key}")
    model_files = []
    models_root = Path(receipt["paths"]["models_eval"])
    for record in receipt["model_files"]:
        path = models_root / record["name"]
        observed = {
            "name": record["name"],
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        if observed != record:
            raise ContractError(f"R4-A final model asset changed: {record['name']}")
        model_files.append(observed)
    if canonical_sha256(model_files) != receipt["hashes"]["models_tree"]:
        raise ContractError("R4-A final model tree lock changed")
    if receipt.get("official_evaluator_invocation_count_at_freeze") != 0:
        raise ContractError("R4-A final scorer ran before freeze")
    return receipt


def run_official_once(
    *,
    protocol_path: Path,
    pre_score_path: Path,
    output_path: Path,
    log_root: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = _verify_pre_score(pre_score_path, protocol_path)
    output_path = output_path.resolve()
    started_path = output_path.with_name("invocation-started.json")
    if output_path.exists() or started_path.exists():
        raise ContractError("R4-A final official development batch is already consumed")
    log_root.mkdir(parents=True, exist_ok=True)
    started = {
        "schema_version": "poseloop.r4a.development-final-invocation-started.v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "pre_score_sha256": sha256_file(pre_score_path),
        "commands": receipt["commands"],
        "maximum_calls_per_variant": 1,
        "rerun_permitted": False,
    }
    started["lock_sha256"] = canonical_sha256(started)
    write_json_atomic(started_path, started)
    environment = os.environ.copy()
    toolkit = Path(receipt["paths"]["toolkit_root"])
    venv_bin = Path(receipt["paths"]["venv_python"]).parent
    environment["BOP_PATH"] = receipt["paths"]["datasets"]
    environment["BOP_RESULTS_PATH"] = receipt["paths"]["results"]
    environment["BOP_EVAL_PATH"] = receipt["paths"]["eval"]
    environment["BOP_NUM_WORKERS"] = str(protocol["toolkit"]["workers"])
    environment["PYTHONPATH"] = str(toolkit) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PATH"] = str(venv_bin) + os.pathsep + environment.get("PATH", "")
    records = []
    for ordinal, command in enumerate(receipt["commands"]):
        variant = command["variant"]
        log_path = log_root / f"{ordinal:02d}-{variant}.log"
        with log_path.open("wb") as log:
            completed = subprocess.run(command["command"], stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
        score_path = Path(command["expected_score"])
        records.append(
            {
                "ordinal": ordinal,
                "variant": variant,
                "command": command["command"],
                "exit_code": int(completed.returncode),
                "log_path": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
                "score_path": str(score_path.resolve()),
                "score_exists": score_path.is_file(),
                "score_sha256": sha256_file(score_path) if score_path.is_file() else None,
            }
        )
    result = {
        "schema_version": OFFICIAL_ONCE_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "pre_score_sha256": sha256_file(pre_score_path),
        "invocation_started_sha256": sha256_file(started_path),
        "command_count": len(records),
        "records": records,
        "all_exit_codes": [row["exit_code"] for row in records],
        "rerun_permitted": False,
        "prediction_or_input_tuning_after_score": False,
        "xyzibd_val_access_count": 0,
    }
    result["lock_sha256"] = canonical_sha256(result)
    write_json_atomic(output_path, result)
    return result


def _csv_poses(path: Path) -> dict[tuple[int, int, int], tuple[np.ndarray, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]))
            rotation = np.asarray([float(value) for value in row["R"].split()], dtype=np.float64).reshape(3, 3)
            translation = np.asarray([float(value) for value in row["t"].split()], dtype=np.float64).reshape(3, 1)
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation[:, 0] * 0.001
            _pose(pose, context=f"CSV pose {key}")
            if key in rows:
                raise ContractError(f"Duplicate BOP CSV key: {key}")
            rows[key] = (pose, float(row["time"]))
    return rows


def _stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean_mm": statistics.fmean(finite) if finite else None,
        "median_mm": statistics.median(finite) if finite else None,
        "min_mm": min(finite) if finite else None,
        "max_mm": max(finite) if finite else None,
    }


def _metric_variant(
    *,
    poses: Mapping[tuple[int, int, int], tuple[np.ndarray, float]],
    targets: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    toolkit_root: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(toolkit_root))
    try:
        from bop_toolkit_lib import inout, pose_error
    finally:
        if sys.path[0] == str(toolkit_root):
            sys.path.pop(0)
    info = read_json(dataset_root / "xyzibd" / "models_eval" / "models_info.json")
    models = {}
    errors = []
    per_object: dict[int, list[float]] = defaultdict(list)
    per_scene: dict[int, list[float]] = defaultdict(list)
    correct = 0
    details = []
    for target in targets:
        scene = int(target["scene_id"])
        image = int(target["im_id"])
        obj = int(target["obj_id"])
        key = (scene, image, obj)
        gt_path = dataset_root / "xyzibd" / "train" / f"{scene:06d}" / "scene_gt_xyz.json"
        gt_rows = read_json(gt_path)[str(image)]
        matches = [row for row in gt_rows if int(row["obj_id"]) == obj]
        if len(matches) != 1:
            raise ContractError(f"Exact evaluator GT key is not unique: {key}")
        gt = matches[0]
        rotation_gt = np.asarray(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation_gt = np.asarray(gt["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
        if obj not in models:
            models[obj] = inout.load_ply(str(dataset_root / "xyzibd" / "models_eval" / f"obj_{obj:06d}.ply"))["pts"]
        symmetric = bool(info[str(obj)].get("symmetries_discrete") or info[str(obj)].get("symmetries_continuous"))
        if key not in poses:
            error_mm = math.inf
        else:
            pose = poses[key][0]
            translation_est = pose[:3, 3].reshape(3, 1) * 1000.0
            if symmetric:
                error_mm = float(pose_error.adi(pose[:3, :3], translation_est, rotation_gt, translation_gt, models[obj]))
            else:
                error_mm = float(pose_error.add(pose[:3, :3], translation_est, rotation_gt, translation_gt, models[obj]))
        diameter = float(info[str(obj)]["diameter"])
        passed = math.isfinite(error_mm) and error_mm < 0.1 * diameter
        correct += int(passed)
        errors.append(error_mm)
        per_object[obj].append(error_mm)
        per_scene[scene].append(error_mm)
        details.append({"scene_id": scene, "image_id": image, "object_id": obj, "metric": "ADI" if symmetric else "ADD", "error_mm": error_mm, "diameter_mm": diameter, "recall_0.1d": passed})
    return {
        **_stats(errors),
        "coverage": len(poses) / len(targets),
        "recall_0.1d": correct / len(targets),
        "per_object": {str(key): {**_stats(values), "recall_0.1d": sum(math.isfinite(v) and v < 0.1 * float(info[str(key)]["diameter"]) for v in values) / len(values)} for key, values in sorted(per_object.items())},
        "per_scene": {str(key): _stats(values) for key, values in sorted(per_scene.items())},
        "details": details,
    }


def summarize(
    *,
    protocol_path: Path,
    pre_score_path: Path,
    official_once_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = _verify_pre_score(pre_score_path, protocol_path)
    official = read_json(official_once_path)
    unlocked = dict(official)
    lock = unlocked.pop("lock_sha256", None)
    if official.get("schema_version") != OFFICIAL_ONCE_SCHEMA or lock != canonical_sha256(unlocked):
        raise ContractError("R4-A final official-once receipt is invalid")
    targets = read_json(Path(receipt["paths"]["targets"]))
    variants = {
        "single_view": _metric_variant(
            poses=_csv_poses(Path(receipt["paths"]["single_csv"])),
            targets=targets,
            dataset_root=Path(receipt["paths"]["datasets"]),
            toolkit_root=Path(receipt["paths"]["toolkit_root"]),
        ),
        "multi_view": _metric_variant(
            poses=_csv_poses(Path(receipt["paths"]["multi_csv"])),
            targets=targets,
            dataset_root=Path(receipt["paths"]["datasets"]),
            toolkit_root=Path(receipt["paths"]["toolkit_root"]),
        ),
    }
    bop24 = {}
    for record in official["records"]:
        value = None
        if record["exit_code"] == 0 and record["score_exists"]:
            scores = read_json(Path(record["score_path"]))
            value = scores.get("bop24_mAP")
        bop24[record["variant"]] = {
            "exit_code": record["exit_code"],
            "score_sha256": record["score_sha256"],
            "bop24_mAP": value,
        }
    single_map = bop24["single_view"]["bop24_mAP"]
    multi_map = bop24["multi_view"]["bop24_mAP"]
    map_delta = float(multi_map) - float(single_map) if single_map is not None and multi_map is not None else None
    add_delta = float(variants["multi_view"]["mean_mm"]) - float(variants["single_view"]["mean_mm"])
    gate = (
        variants["single_view"]["coverage"] == 1.0
        and variants["multi_view"]["coverage"] == 1.0
        and multi_map is not None
        and float(multi_map) > 0.0
        and map_delta is not None
        and map_delta > 0.0
        and add_delta < 0.0
    )
    runtimes = [float(row["registration_seconds"]) for row in read_jsonl(Path(receipt["paths"]["single_jsonl"]))]
    report = {
        "schema_version": FINAL_REPORT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "pre_score_sha256": sha256_file(pre_score_path),
        "official_once_sha256": sha256_file(official_once_path),
        "status": "GO" if gate else "NO_GO",
        "coverage": receipt["coverage"],
        "missing_count": 10 - receipt["coverage"]["successful"],
        "failure_count": 10 - receipt["coverage"]["successful"],
        "add_adi": variants,
        "bop24_development": bop24,
        "multi_minus_single": {"bop24_mAP": map_delta, "mean_ADD_or_ADI_mm": add_delta, "median_ADD_or_ADI_mm": float(variants["multi_view"]["median_mm"]) - float(variants["single_view"]["median_mm"])},
        "throughput": {
            "item_count": len(runtimes),
            "registration_seconds_total": sum(runtimes),
            "registration_seconds_p50": statistics.median(runtimes),
            "registration_seconds_p95": float(np.quantile(runtimes, 0.95)),
            "items_per_registration_second": len(runtimes) / sum(runtimes),
        },
        "gate": {
            "pass": gate,
            "failure_action": protocol["development_gate"]["failure_action"],
            "prediction_or_checkpoint_tuning_permitted": False,
        },
        "claim": protocol["bop24_development"]["claim"],
        "xyzibd_val_access_count": 0,
        "rerun_permitted": False,
    }
    report["lock_sha256"] = canonical_sha256(report)
    write_json_atomic(output_path, report)
    return report


def _decode_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
        or not isinstance(counts, list)
        or not all(isinstance(value, int) and value >= 0 for value in counts)
    ):
        raise ContractError("R4-A final predicted COCO RLE is invalid")
    flattened = np.zeros(int(size[0]) * int(size[1]), dtype=np.uint8)
    offset = 0
    value = 0
    for run in counts:
        end = offset + run
        if end > flattened.size:
            raise ContractError("R4-A final predicted COCO RLE exceeds its image")
        if value:
            flattened[offset:end] = 1
        offset = end
        value = 1 - value
    if offset != flattened.size:
        raise ContractError("R4-A final predicted COCO RLE does not cover its image")
    return flattened.reshape((int(size[0]), int(size[1])), order="F")


def _project_points(points_m: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (pose[:3, :3] @ points_m.T).T + pose[:3, 3]
    valid = camera[:, 2] > 1e-6
    projected = np.full((len(camera), 2), np.nan, dtype=np.float64)
    projected[valid] = (intrinsics @ camera[valid].T).T[:, :2] / camera[valid, 2, None]
    return projected, valid


def _render_pose_panel(
    *,
    cv2: Any,
    bgr: np.ndarray,
    points_m: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    title: str,
) -> np.ndarray:
    panel = bgr.copy()
    sampled = points_m[:: max(1, math.ceil(len(points_m) / 4000))]
    pixels, valid = _project_points(sampled, pose, intrinsics)
    height, width = panel.shape[:2]
    for x, y in np.rint(pixels[valid]).astype(np.int32):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(panel, (int(x), int(y)), 1, (0, 220, 255), -1, cv2.LINE_8)
    extent = float(np.max(np.ptp(points_m, axis=0))) if len(points_m) else 0.05
    axis = max(0.01, 0.5 * extent)
    axes = np.asarray([[0, 0, 0], [axis, 0, 0], [0, axis, 0], [0, 0, axis]], dtype=np.float64)
    axis_pixels, axis_valid = _project_points(axes, pose, intrinsics)
    if bool(np.all(axis_valid)):
        rounded = np.rint(axis_pixels).astype(np.int32)
        origin = tuple(int(value) for value in rounded[0])
        for endpoint, color in zip(rounded[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
            cv2.line(panel, origin, tuple(int(value) for value in endpoint), color, 4, cv2.LINE_8)
    cv2.putText(panel, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_8)
    cv2.putText(panel, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 1, cv2.LINE_8)
    return panel


def visualize(
    *,
    protocol_path: Path,
    pre_score_path: Path,
    output_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Render label-free input/single/multi panels without opening evaluator assets."""
    import cv2

    protocol = load_protocol(protocol_path)
    receipt = _verify_pre_score(pre_score_path, protocol_path)
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ContractError(f"R4-A final visualization output is not empty: {output_root}")
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    workload_path = Path(receipt["inputs"]["workload"]["path"])
    coco_path = Path(receipt["inputs"]["coco"]["path"])
    inference_root = Path(receipt["paths"]["inference_root"])
    workload = read_jsonl(workload_path)
    coco = read_json(coco_path)
    if not isinstance(coco, list):
        raise ContractError("R4-A final predicted COCO payload is not a list")
    single = {str(row["item_id"]): row for row in read_jsonl(Path(receipt["paths"]["single_jsonl"]))}
    multi = {str(row["item_id"]): row for row in read_jsonl(Path(receipt["paths"]["multi_jsonl"]))}
    frame_records = []
    video_frames = []
    model_cache: dict[str, np.ndarray] = {}
    sys.path.insert(0, str(Path(receipt["paths"]["toolkit_root"])))
    try:
        from bop_toolkit_lib import inout
    finally:
        if sys.path[0] == str(Path(receipt["paths"]["toolkit_root"])):
            sys.path.pop(0)
    for payload in workload:
        item_id = str(payload["item_id"])
        detection_index = int(payload["detection_index"])
        if not 0 <= detection_index < len(coco):
            raise ContractError(f"R4-A final detection index is invalid: {item_id}")
        detection = coco[detection_index]
        if canonical_sha256(detection["segmentation"]) != payload["segmentation_sha256"]:
            raise ContractError(f"R4-A final predicted segmentation changed: {item_id}")
        rgb_path = inference_root / str(payload["rgb_relative_path"])
        model_path = inference_root / str(payload["model_relative_path"])
        if sha256_file(rgb_path) != payload["rgb_sha256"] or sha256_file(model_path) != payload["model_sha256"]:
            raise ContractError(f"R4-A final visualization public asset changed: {item_id}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ContractError(f"R4-A final cannot decode public RGB: {item_id}")
        mask = _decode_rle(detection["segmentation"])
        if mask.shape != bgr.shape[:2]:
            raise ContractError(f"R4-A final mask/RGB shape differs: {item_id}")
        before = bgr.copy()
        green = np.zeros_like(before)
        green[:, :, 1] = 255
        selected = mask.astype(bool)
        before[selected] = ((before[selected].astype(np.uint16) * 2 + green[selected].astype(np.uint16)) // 3).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(before, contours, -1, (0, 255, 0), 2, cv2.LINE_8)
        cv2.putText(before, "INPUT: RGB + PREDICTED MASK", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_8)
        model_key = str(payload["model_sha256"])
        if model_key not in model_cache:
            model_cache[model_key] = np.asarray(inout.load_ply(str(model_path))["pts"], dtype=np.float64) * 0.001
        points_m = model_cache[model_key]
        intrinsics = np.asarray(payload["camera_intrinsics"], dtype=np.float64).reshape(3, 3)
        single_panel = _render_pose_panel(
            cv2=cv2,
            bgr=bgr,
            points_m=points_m,
            pose=_pose(single[item_id]["predicted_model_to_camera_pose_m"], context=f"single visualization {item_id}"),
            intrinsics=intrinsics,
            title="SINGLE: CAD + AXES",
        )
        multi_panel = _render_pose_panel(
            cv2=cv2,
            bgr=bgr,
            points_m=points_m,
            pose=_pose(multi[item_id]["predicted_model_to_camera_pose_m"], context=f"multi visualization {item_id}"),
            intrinsics=intrinsics,
            title="MULTI: CAD + AXES",
        )
        panel_height = min(720, bgr.shape[0])
        panel_width = max(1, round(bgr.shape[1] * panel_height / bgr.shape[0]))
        panels = [cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA) for panel in (before, single_panel, multi_panel)]
        frame = np.concatenate(panels, axis=1)
        frame_path = frames_root / f"{item_id}.png"
        if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ContractError(f"R4-A final cannot encode visualization PNG: {item_id}")
        frame_records.append(
            {
                "item_id": item_id,
                "scene_id": int(payload["scene_id"]),
                "image_id": int(payload["image_id"]),
                "object_id": int(payload["object_id"]),
                "relative_path": frame_path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(frame_path),
            }
        )
        video_frames.append(frame)
    video_path = output_root / "r4a-v3-input-single-multi.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (video_frames[0].shape[1], video_frames[0].shape[0]),
    )
    if not writer.isOpened():
        raise ContractError("R4-A final cannot open MP4 writer")
    try:
        for frame in video_frames:
            writer.write(frame)
    finally:
        writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ContractError("R4-A final visualization video is empty")
    value = {
        "schema_version": VISUALIZATION_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "pre_score_sha256": sha256_file(pre_score_path),
        "coverage": {"expected": len(workload), "rendered": len(frame_records), "fraction": len(frame_records) / len(workload)},
        "panel_order": ["input_rgb_plus_predicted_mask", "single_view_cad_plus_axes", "multi_view_cad_plus_axes"],
        "frames": frame_records,
        "video": {"relative_path": video_path.relative_to(output_root).as_posix(), "bytes": video_path.stat().st_size, "sha256": sha256_file(video_path), "fps": 2.0},
        "opened_roles": ["public_RGB", "predicted_input_mask_COCO_RLE", "public_camera_intrinsics", "public_CAD", "frozen_single_prediction", "fixed_multi_prediction"],
        "evaluator_label_files_opened": 0,
        "xyzibd_val_access_count": 0,
        "label_access_count": 0,
    }
    value["lock_sha256"] = canonical_sha256(value)
    write_json_atomic(output_path, value)
    return value


__all__ = [
    "load_protocol",
    "multi_view_predictions",
    "prepare",
    "run_official_once",
    "summarize",
    "validate_results",
    "visualize",
]
