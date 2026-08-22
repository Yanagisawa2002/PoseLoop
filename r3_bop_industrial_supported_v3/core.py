"""One-shot official supported-metrics orchestration for XYZ-IBD R3-v3."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from r3_bop_industrial_supported_v2.core import _SMOKE_PROGRAM
from r3_bop_industrial_supported_v3_readiness.core import (
    ContractError,
    audit_frozen_predictions,
    audit_toolkit,
    audit_v2_failure_evidence,
    canonical_sha256,
    read_json,
    scorer_process_audit,
    sha256_file,
    write_json_atomic,
)
from r3_bop_industrial.core import utc_now, write_text_atomic

from . import PACKAGE_VERSION, PROTOCOL_ID


PROTOCOL_SCHEMA = "poseloop.r3.xyzibd-supported.protocol.v3"
PREFLIGHT_SCHEMA = "poseloop.r3.xyzibd-supported.preflight.v3"
LOCK_SCHEMA = "poseloop.r3.xyzibd-supported.input-lock.v3"
AUTH_SCHEMA = "poseloop.r3.xyzibd-supported.evaluator-authorization.v3"
SMOKE_SCHEMA = "poseloop.r3.xyzibd-supported.environment-smoke.v3"
STARTED_SCHEMA = "poseloop.r3.xyzibd-supported.invocation-started.v3"
SEALED_SCHEMA = "poseloop.r3.xyzibd-supported.sealed-receipt.v3"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("Supported-v3 protocol must be a JSON object")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("Supported-v3 protocol schema mismatch")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported-v3 protocol ID mismatch")
    if value.get("state") != "frozen_before_supported_official_evaluation":
        raise ContractError("Supported-v3 protocol is not frozen before scoring")
    source_v2 = value.get("immutable_v2_failure", {})
    if source_v2.get("rerun_permitted") is not False or source_v2.get("immutable") is not True:
        raise ContractError("Supported-v2 failed protocol must remain immutable and non-rerunnable")
    readiness = value.get("label_readiness", {})
    if readiness.get("state") != "label_assets_ready_frozen_before_v3_scoring_protocol":
        raise ContractError("Label-readiness state is not frozen and ready")
    evaluation = value.get("evaluation", {})
    required = {
        "maximum_evaluate_invocations": 1,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "authorized_command_count": 4,
        "result_tuning_after_score": False,
        "rerun_after_score_or_error": False,
    }
    for key, expected in required.items():
        if evaluation.get(key) != expected:
            raise ContractError(f"Supported-v3 evaluation contract mismatch: {key}")
    expected_order = [
        "official_bop24_pose_map:single_view",
        "official_bop24_pose_map:multi_view",
        "official_bop22_bbox_ap:shared_predicted_input",
        "official_bop22_segm_ap:shared_predicted_input",
    ]
    if evaluation.get("command_order") != expected_order:
        raise ContractError("Supported-v3 command order mismatch")
    support = value.get("support_matrix", {})
    bop19 = support.get("official_bop19_average_recall", {})
    if bop19.get("status") != "unavailable" or bop19.get("must_not_invoke") is not True:
        raise ContractError("BOP19 AR must remain unavailable for XYZ-IBD")
    if bop19.get("missing_xyzibd_definition") != "vsd":
        raise ContractError("BOP19 unavailability reason changed")
    for metric in ("official_bop24_pose_map", "official_bop22_bbox_ap", "official_bop22_segm_ap"):
        if support.get(metric, {}).get("status") != "available":
            raise ContractError(f"Supported-v3 metric unexpectedly unavailable: {metric}")
    hashes: list[Any] = [
        source_v2.get("invocation_started_sha256"),
        source_v2.get("sealed_receipt_sha256"),
        readiness.get("protocol_sha256"),
        readiness.get("lock_sha256"),
        readiness.get("fingerprint_sha256"),
        value.get("dataset", {}).get("bop19_targets_sha256"),
        value.get("dataset", {}).get("bop24_targets_sha256"),
        value.get("toolkit", {}).get("source_tree_sha256"),
    ]
    hashes.extend(value.get("frozen_prediction_sha256", {}).values())
    hashes.extend(value.get("toolkit", {}).get("required_file_sha256", {}).values())
    if any(not _valid_sha256(digest) for digest in hashes):
        raise ContractError("Supported-v3 protocol contains an invalid SHA-256")
    if not _valid_git_commit(readiness.get("implementation_commit")):
        raise ContractError("Supported-v3 readiness implementation commit is invalid")
    return value


def _run_git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def repository_audit(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    head = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["branch", "--show-current"])
    status = _run_git(repo_root, ["status", "--short"])
    errors: list[str] = []
    if head.returncode != 0 or branch.returncode != 0 or status.returncode != 0:
        errors.append("Git repository audit failed")
    actual_branch = branch.stdout.strip()
    if actual_branch != protocol["required_execution_branch"]:
        errors.append("execution branch differs from frozen protocol")
    status_entries = [line for line in status.stdout.splitlines() if line]
    if status_entries:
        errors.append("repository is not clean")
    return {
        "root": str(repo_root.resolve()),
        "head": head.stdout.strip(),
        "branch": actual_branch,
        "status_entries": status_entries,
        "errors": errors,
        "ready": not errors,
    }


def implementation_audit() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files = []
    for name in ("__init__.py", "__main__.py", "cli.py", "core.py"):
        path = package_root / name
        if not path.is_file():
            raise ContractError(f"Supported-v3 implementation file is missing: {name}")
        files.append({"relative_path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "package_root": str(package_root.resolve()),
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }


def _file_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("missing file")
    else:
        result["sha256"] = sha256_file(path)
        result["bytes"] = path.stat().st_size
        if result["sha256"] != expected_sha256:
            result["errors"].append("SHA-256 mismatch")
    result["ready"] = not result["errors"]
    return result


def label_readiness_audit(
    lock_path: Path,
    dataset_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    contract = protocol["label_readiness"]
    result = _file_contract(lock_path, contract["lock_sha256"])
    result["errors"] = list(result["errors"])
    if not result["ready"]:
        return result
    lock = read_json(lock_path)
    required = {
        "schema_version": "poseloop.r3.xyzibd-label-readiness.lock.v3",
        "state": contract["state"],
        "status": "ready",
        "fingerprint_sha256": contract["fingerprint_sha256"],
        "official_evaluator_executed": False,
        "scoring_authorization_created": False,
    }
    for key, expected in required.items():
        if lock.get(key) != expected:
            result["errors"].append(f"label-readiness lock field mismatch: {key}")
    if lock.get("frozen_predictions", {}).get("prediction_selection_label_access_count") != 0:
        result["errors"].append("readiness lock prediction/selection label-access is not zero")
    if lock.get("immutable_v2_failure_evidence", {}).get("rerun_permitted") is not False:
        result["errors"].append("readiness lock no longer preserves v2 rerun prohibition")
    current_files: dict[str, Any] = {}
    target_contract = lock.get("fingerprint", {}).get("target_sha256", {})
    if target_contract.get("bop19") != protocol["dataset"]["bop19_targets_sha256"]:
        result["errors"].append("readiness-lock BOP19 validation-target hash differs from protocol")
    if target_contract.get("bop24") != protocol["dataset"]["bop24_targets_sha256"]:
        result["errors"].append("readiness-lock BOP24 validation-target hash differs from protocol")
    for role, filename in (
        ("bop19", protocol["dataset"]["bop19_targets_filename"]),
        ("bop24", protocol["dataset"]["bop24_targets_filename"]),
    ):
        item = _file_contract(dataset_root / filename, target_contract.get(role, ""))
        current_files[filename] = item
        result["errors"].extend(f"{filename}: {error}" for error in item["errors"])
    coco_contract = lock.get("fingerprint", {}).get("derived_coco_sha256", {})
    for scene_id in protocol["dataset"]["expected_val_scene_ids"]:
        key = f"{scene_id:06d}"
        filename = f"val/{key}/scene_gt_coco_xyz.json"
        item = _file_contract(dataset_root / filename, coco_contract.get(key, ""))
        current_files[filename] = item
        result["errors"].extend(f"{filename}: {error}" for error in item["errors"])
    result["current_files"] = current_files
    result["lock_fingerprint"] = lock.get("fingerprint")
    result["ready"] = not result["errors"]
    return result


def score_namespace_audit(output_root: Path) -> dict[str, Any]:
    entries: list[str] = []
    if output_root.exists():
        if not output_root.is_dir():
            entries.append(str(output_root.resolve()))
        else:
            entries.extend(str(path.resolve()) for path in output_root.rglob("*"))
    score_files = [path for path in entries if Path(path).name.startswith("scores") or Path(path).name == "official_scores.json"]
    return {
        "root": str(output_root.resolve()),
        "exists": output_root.exists(),
        "entry_count": len(entries),
        "score_file_count": len(score_files),
        "score_files": sorted(score_files),
    }


def create_environment_smoke_receipt(
    *,
    protocol: Mapping[str, Any],
    python_executable: str,
    toolkit_root: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    executable = Path(python_executable).absolute()
    before_process = scorer_process_audit()
    before_output = score_namespace_audit(output_root)
    errors: list[str] = []
    if not executable.is_file():
        errors.append("requested evaluator interpreter is missing")
    if not model_path.is_file():
        errors.append("public renderer-smoke model is missing")
    payload = None
    completed = None
    if not errors:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(toolkit_root.resolve())
        env["PYOPENGL_PLATFORM"] = "egl"
        completed = subprocess.run(
            [str(executable), "-c", _SMOKE_PROGRAM, str(toolkit_root.resolve()), str(model_path.resolve())],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            errors.append(f"environment smoke exited {completed.returncode}")
        prefix = "POSELOOP_SMOKE_JSON="
        payloads = [line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)]
        if len(payloads) == 1:
            payload = json.loads(payloads[0])
        else:
            errors.append("environment smoke emitted no unique payload")
    after_process = scorer_process_audit()
    after_output = score_namespace_audit(output_root)
    if before_process["errors"] or before_process["count"] != 0 or after_process["errors"] or after_process["count"] != 0:
        errors.append("official scorer process count is not zero during smoke")
    if before_output["entry_count"] != 0 or after_output["entry_count"] != 0:
        errors.append("supported-v3 official output namespace is not absent/empty during smoke")
    if payload is not None:
        toolkit_origin = Path(str(payload.get("bop_toolkit_lib_origin", ""))).resolve()
        if not str(toolkit_origin).startswith(str(toolkit_root.resolve())):
            errors.append("bop_toolkit_lib import origin is outside pinned toolkit")
        if payload.get("python_executable") != str(executable):
            errors.append("smoke interpreter differs from requested evaluator interpreter")
        if payload.get("venv_active") is not True:
            errors.append("smoke interpreter does not report an active venv")
        if payload.get("editable_match_count") != 1:
            errors.append("pinned editable toolkit distribution is not unique")
        if Path(str(payload.get("editable_project_root", ""))).resolve() != toolkit_root.resolve():
            errors.append("editable toolkit project root mismatch")
        if payload.get("renderer", {}).get("status") != "ready":
            errors.append("renderer smoke is not ready")
    return {
        "schema_version": SMOKE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "python_executable": str(executable),
        "toolkit_root": str(toolkit_root.resolve()),
        "model_path": str(model_path.resolve()),
        "payload": payload,
        "child_exit_code": completed.returncode if completed else None,
        "child_stdout": completed.stdout if completed else "",
        "child_stderr": completed.stderr if completed else "",
        "official_scorer_process_before": before_process,
        "official_scorer_process_after": after_process,
        "output_before": before_output,
        "output_after": after_output,
        "official_evaluator_executed": False,
        "prediction_selection_label_access_count": 0,
        "errors": errors,
    }


def smoke_receipt_audit(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    python_executable: str,
    toolkit_root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file(), "errors": []}
    if not path.is_file():
        result["errors"].append("environment-smoke receipt is missing")
    else:
        value = read_json(path)
        result.update(sha256=sha256_file(path), receipt=value)
        if value.get("schema_version") != SMOKE_SCHEMA or value.get("protocol_id") != PROTOCOL_ID:
            result["errors"].append("environment-smoke receipt schema/protocol mismatch")
        if value.get("status") != "ready" or value.get("official_evaluator_executed") is not False:
            result["errors"].append("environment-smoke receipt is not ready/non-scoring")
        if value.get("python_executable") != str(Path(python_executable).absolute()):
            result["errors"].append("environment-smoke interpreter changed")
        if value.get("toolkit_root") != str(toolkit_root.resolve()):
            result["errors"].append("environment-smoke toolkit root changed")
    result["ready"] = not result["errors"]
    return result


def build_official_commands(
    *,
    protocol: Mapping[str, Any],
    python_executable: str,
    toolkit_root: Path,
    input_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    eval_root = output_root / "eval"
    commands: list[dict[str, Any]] = []
    for variant, filename in (
        ("single_view", protocol["prediction_files"]["single_view_pose"]),
        ("multi_view", protocol["prediction_files"]["multi_view_pose"]),
    ):
        commands.append(
            {
                "command_id": f"official_bop24_pose_map:{variant}",
                "metric": "official_bop24_pose_map",
                "variant": variant,
                "argv": [
                    python_executable,
                    str((toolkit_root / "scripts" / "eval_bop24_pose.py").resolve()),
                    f"--result_filenames={filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['dataset']['bop24_targets_filename']}",
                    f"--num_workers={protocol['toolkit']['workers']}",
                ],
                "expected_score": str((eval_root / Path(filename).stem / "scores_bop24.json").resolve()),
            }
        )
    coco_filename = protocol["prediction_files"]["coco_predictions"]
    for annotation in ("bbox", "segm"):
        commands.append(
            {
                "command_id": f"official_bop22_{annotation}_ap:shared_predicted_input",
                "metric": f"official_bop22_{annotation}_ap",
                "variant": "shared_predicted_input",
                "argv": [
                    python_executable,
                    str((toolkit_root / "scripts" / "eval_bop22_coco.py").resolve()),
                    f"--result_filenames={coco_filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root.resolve()}",
                    f"--targets_filename={protocol['dataset']['bop19_targets_filename']}",
                    f"--ann_type={annotation}",
                    "--bbox_type=amodal",
                ],
                "expected_score": str((eval_root / Path(coco_filename).stem / f"scores_bop22_coco_{annotation}.json").resolve()),
            }
        )
    return commands


def create_preflight_receipt(
    *,
    protocol_path: Path,
    readiness_lock: Path,
    dataset_root: Path,
    toolkit_root: Path,
    input_root: Path,
    v2_invocation_started: Path,
    v2_sealed_receipt: Path,
    smoke_receipt: Path,
    output_root: Path,
    repo_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    repository = repository_audit(repo_root, protocol)
    implementation = implementation_audit()
    readiness = label_readiness_audit(readiness_lock, dataset_root, protocol)
    toolkit = audit_toolkit(toolkit_root, protocol)
    predictions = audit_frozen_predictions(input_root, protocol)
    v2_evidence = audit_v2_failure_evidence(v2_invocation_started, v2_sealed_receipt, {"source_v2": protocol["immutable_v2_failure"]})
    smoke = smoke_receipt_audit(
        smoke_receipt,
        protocol=protocol,
        python_executable=python_executable,
        toolkit_root=toolkit_root,
    )
    processes = scorer_process_audit()
    output = score_namespace_audit(output_root)
    commands = build_official_commands(
        protocol=protocol,
        python_executable=str(Path(python_executable).absolute()),
        toolkit_root=toolkit_root,
        input_root=input_root,
        output_root=output_root,
    )
    errors = [
        *repository["errors"],
        *readiness["errors"],
        *toolkit["errors"],
        *predictions["errors"],
        *v2_evidence["errors"],
        *smoke["errors"],
    ]
    if processes["errors"] or processes["count"] != 0:
        errors.append("official scorer process count is not zero")
    if output["exists"] or output["entry_count"] != 0 or output["score_file_count"] != 0:
        errors.append("supported-v3 official output root already exists")
    if [item["command_id"] for item in commands] != protocol["evaluation"]["command_order"]:
        errors.append("built command order differs from frozen protocol")
    command_manifest = [
        {"command_id": item["command_id"], "argv": item["argv"], "expected_score": item["expected_score"]}
        for item in commands
    ]
    fingerprint = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": repository["head"],
        "implementation_manifest_sha256": implementation["manifest_sha256"],
        "readiness_lock_sha256": readiness.get("sha256"),
        "readiness_fingerprint_sha256": protocol["label_readiness"]["fingerprint_sha256"],
        "toolkit_source_tree_sha256": toolkit.get("source_tree", {}).get("sha256"),
        "prediction_sha256": {name: item.get("sha256") for name, item in predictions["files"].items()},
        "v2_invocation_started_sha256": v2_evidence["invocation_started"].get("sha256"),
        "v2_sealed_receipt_sha256": v2_evidence["sealed_receipt"].get("sha256"),
        "smoke_receipt_sha256": smoke.get("sha256"),
        "command_manifest_sha256": canonical_sha256(command_manifest),
        "official_output_root": str(output_root.resolve()),
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "package_version": PACKAGE_VERSION,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "protocol": {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)},
        "repository": repository,
        "implementation": implementation,
        "label_readiness": readiness,
        "toolkit": toolkit,
        "frozen_predictions": predictions,
        "immutable_v2_failure_evidence": v2_evidence,
        "environment_smoke": smoke,
        "official_process_audit": processes,
        "official_output_audit": output,
        "official_commands": commands,
        "command_manifest": command_manifest,
        "command_manifest_sha256": canonical_sha256(command_manifest),
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "official_evaluation_executed": False,
        "prediction_selection_label_access_count": 0,
        "errors": errors,
    }


def freeze_input_lock(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if preflight.get("status") != "ready":
        raise ContractError("Cannot freeze a blocked supported-v3 preflight")
    value = dict(preflight)
    value["schema_version"] = LOCK_SCHEMA
    value["state"] = "frozen_before_unique_supported_v3_evaluation"
    value["frozen_utc"] = utc_now()
    return value


def validate_input_lock(path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(path)
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported-v3 input-lock schema/protocol mismatch")
    if lock.get("state") != "frozen_before_unique_supported_v3_evaluation":
        raise ContractError("Supported-v3 input lock is not frozen")
    if current.get("status") != "ready" or lock.get("fingerprint_sha256") != current.get("fingerprint_sha256"):
        raise ContractError("Current supported-v3 inputs differ from the frozen lock")
    return lock


def create_evaluator_authorization(
    *,
    input_lock_path: Path,
    protocol: Mapping[str, Any],
    approved_by: str,
    authorization_reference: str,
) -> dict[str, Any]:
    lock = read_json(input_lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Cannot authorize an invalid supported-v3 input lock")
    if approved_by != "user" or authorization_reference != protocol["authorization"]["reference"]:
        raise ContractError("Supported-v3 authorization identity/reference mismatch")
    return {
        "schema_version": AUTH_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "approved": True,
        "approved_by": approved_by,
        "authorization_reference": authorization_reference,
        "scope": protocol["authorization"]["scope"],
        "input_lock_sha256": sha256_file(input_lock_path),
        "input_fingerprint_sha256": lock["fingerprint_sha256"],
        "implementation_commit": lock["repository"]["head"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "readiness_lock_sha256": lock["label_readiness"]["sha256"],
        "command_manifest_sha256": lock["command_manifest_sha256"],
        "authorized_command_count": 4,
        "maximum_evaluate_invocations": 1,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "rerun_after_score_or_error": False,
    }


def validate_authorization(
    path: Path,
    input_lock_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != AUTH_SCHEMA or value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported-v3 evaluator authorization schema/protocol mismatch")
    if value.get("approved") is not True or value.get("scope") != protocol["authorization"]["scope"]:
        raise ContractError("Supported-v3 evaluator authorization is not approved/in scope")
    if value.get("input_lock_sha256") != sha256_file(input_lock_path):
        raise ContractError("Supported-v3 authorization does not bind current input lock")
    if value.get("maximum_evaluate_invocations") != 1 or value.get("maximum_invocations_per_command") != 1:
        raise ContractError("Supported-v3 authorization invocation bounds changed")
    if value.get("rerun_after_score_or_error") is not False:
        raise ContractError("Supported-v3 authorization unexpectedly permits rerun")
    return value


def create_final_prescore_receipt(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_authorization(authorization_path, input_lock_path, protocol)
    errors: list[str] = []
    if current_preflight["official_process_audit"]["count"] != 0:
        errors.append("official scorer process count is not zero")
    if current_preflight["official_output_audit"]["exists"]:
        errors.append("official output root exists before scoring")
    return {
        "schema_version": "poseloop.r3.xyzibd-supported.final-prescore.v3",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "input_lock": {"path": str(input_lock_path.resolve()), "sha256": sha256_file(input_lock_path), "fingerprint_sha256": lock["fingerprint_sha256"]},
        "authorization": {"path": str(authorization_path.resolve()), "sha256": sha256_file(authorization_path), "reference": authorization["authorization_reference"]},
        "implementation_commit": current_preflight["repository"]["head"],
        "command_manifest_sha256": current_preflight["command_manifest_sha256"],
        "label_readiness_lock_sha256": current_preflight["label_readiness"]["sha256"],
        "prediction_selection_label_access_count": 0,
        "official_evaluation_executed": False,
        "errors": errors,
    }


def _read_score(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        result["sha256"] = sha256_file(path)
        try:
            value = read_json(path)
            result["scores"] = value if isinstance(value, dict) else None
            if not isinstance(value, dict):
                result["error"] = "score JSON is not an object"
        except (OSError, json.JSONDecodeError) as exc:
            result["error"] = str(exc)
    return result


def execute_official_evaluation(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_authorization(authorization_path, input_lock_path, protocol)
    if output_root.exists():
        raise ContractError("Supported-v3 official output root already exists; rerun forbidden")
    output_root.mkdir(parents=True, exist_ok=False)
    started = {
        "schema_version": STARTED_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "evaluate_invocation_count": 1,
        "authorized_command_count": 4,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "input_lock_sha256": sha256_file(input_lock_path),
        "authorization_sha256": sha256_file(authorization_path),
        "command_manifest_sha256": current_preflight["command_manifest_sha256"],
        "label_readiness_lock_sha256": current_preflight["label_readiness"]["sha256"],
        "prediction_selection_label_access_count": 0,
        "prior_v2_rerun_permitted": False,
    }
    started_path = output_root / "invocation_started.json"
    write_json_atomic(started_path, started)
    env = os.environ.copy()
    env["BOP_PATH"] = str(dataset_root.resolve().parent)
    env["PYTHONPATH"] = str(toolkit_root.resolve())
    env["PYOPENGL_PLATFORM"] = "egl"
    evaluator_python = Path(current_preflight["official_commands"][0]["argv"][0]).absolute()
    env["PATH"] = str(evaluator_python.parent) + os.pathsep + env.get("PATH", "")
    executions: list[dict[str, Any]] = []
    for index, command in enumerate(current_preflight["official_commands"], start=1):
        launch_error = None
        try:
            completed = subprocess.run(
                command["argv"], capture_output=True, text=True, check=False, env=env
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except OSError as exc:
            exit_code = None
            stdout = ""
            stderr = ""
            launch_error = str(exc)
        log_path = output_root / "logs" / f"{index:02d}_{command['command_id'].replace(':', '_')}.log"
        write_text_atomic(
            log_path,
            f"command_id={command['command_id']}\nargv={json.dumps(command['argv'])}\n"
            f"exit_code={exit_code}\nlaunch_error={launch_error}\n\n[stdout]\n{stdout}\n[stderr]\n{stderr}",
        )
        executions.append(
            {
                "command_id": command["command_id"],
                "invocation_count": 1,
                "exit_code": exit_code,
                "launch_error": launch_error,
                "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
                "expected_score": command["expected_score"],
            }
        )
    score_records = {
        item["command_id"]: _read_score(Path(item["expected_score"]))
        for item in current_preflight["official_commands"]
    }
    all_success = all(item["exit_code"] == 0 for item in executions)
    all_scores = all(item["exists"] and "error" not in item and isinstance(item.get("scores"), dict) for item in score_records.values())
    status = "complete" if all_success and all_scores else "partial_or_failed"
    single = score_records["official_bop24_pose_map:single_view"].get("scores", {}).get("bop24_mAP")
    multi = score_records["official_bop24_pose_map:multi_view"].get("scores", {}).get("bop24_mAP")
    metrics = {
        "official_bop19_average_recall": {"status": "unavailable", "reason": "XYZ-IBD VSD is undefined; no BOP19 command was constructed"},
        "official_bop24_pose_map": {
            "single_view": single,
            "multi_view": multi,
            "multi_minus_single": float(multi) - float(single) if isinstance(single, (int, float)) and isinstance(multi, (int, float)) else None,
        },
        "official_bop22_bbox_ap": score_records["official_bop22_bbox_ap:shared_predicted_input"].get("scores", {}).get("AP"),
        "official_bop22_segm_ap": score_records["official_bop22_segm_ap:shared_predicted_input"].get("scores", {}).get("AP"),
    }
    result = {
        "schema_version": "poseloop.r3.xyzibd-supported.official-scores.v3",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": status,
        "evaluate_invocation_count": 1,
        "authorized_command_count": 4,
        "attempted_command_count": len(executions),
        "continued_after_failures": True,
        "input_lock": {"path": str(input_lock_path.resolve()), "sha256": sha256_file(input_lock_path), "fingerprint_sha256": lock["fingerprint_sha256"]},
        "authorization": {"path": str(authorization_path.resolve()), "sha256": sha256_file(authorization_path), "reference": authorization["authorization_reference"]},
        "label_readiness_lock_sha256": current_preflight["label_readiness"]["sha256"],
        "implementation_commit": current_preflight["repository"]["head"],
        "executions": executions,
        "raw_official_scores": score_records,
        "metrics": metrics,
        "prediction_selection_label_access_count": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output_root / "official_scores.json"
    write_json_atomic(result_path, result)
    output_manifest = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sealed_receipt.json":
            output_manifest.append({"relative_path": path.relative_to(output_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    sealed = {
        "schema_version": SEALED_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "complete" if status == "complete" else "partial_or_failed_after_evaluate_started",
        "result_status": status,
        "evaluate_invocation_count": 1,
        "authorized_command_count": 4,
        "attempted_command_count": len(executions),
        "command_invocation_counts": {item["command_id"]: 1 for item in executions},
        "all_exit_codes": [item["exit_code"] for item in executions],
        "continued_after_failures": True,
        "official_scores_sha256": sha256_file(result_path),
        "input_lock_sha256": sha256_file(input_lock_path),
        "authorization_sha256": sha256_file(authorization_path),
        "invocation_started_sha256": sha256_file(started_path),
        "label_readiness_lock_sha256": current_preflight["label_readiness"]["sha256"],
        "output_manifest": output_manifest,
        "output_manifest_sha256": canonical_sha256(output_manifest),
        "rerun_permitted": False,
    }
    write_json_atomic(output_root / "sealed_receipt.json", sealed)
    return result
