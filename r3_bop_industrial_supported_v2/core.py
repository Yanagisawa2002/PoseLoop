"""Audited second one-shot run of the XYZ-IBD-supported R3 metrics.

The original failed protocol and the supported-v1 failed protocol are immutable
inputs to this namespace.  This implementation fixes only the evaluator Python
environment and guarantees that each of the four frozen commands is attempted
at most once, even if an earlier command fails.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from r3_bop_industrial import core as source_core
from r3_bop_industrial_supported import core as v1_core

from . import (
    PACKAGE_VERSION,
    PRIOR_SUPPORTED_PROTOCOL_ID,
    PROTOCOL_ID,
    SOURCE_PROTOCOL_ID,
)


ContractError = source_core.ContractError
canonical_sha256 = source_core.canonical_sha256
read_json = source_core.read_json
sha256_file = source_core.sha256_file
utc_now = source_core.utc_now
write_json_atomic = source_core.write_json_atomic

PROTOCOL_SCHEMA = "poseloop.r3.xyzibd-supported.protocol.v2"
PREFLIGHT_SCHEMA = "poseloop.r3.xyzibd-supported.preflight.v2"
LOCK_SCHEMA = "poseloop.r3.xyzibd-supported.input-lock.v2"
AUTH_SCHEMA = "poseloop.r3.xyzibd-supported.evaluator-authorization.v2"
SMOKE_SCHEMA = "poseloop.r3.xyzibd-supported.environment-smoke.v2"


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ContractError("Supported-v2 protocol must be a JSON object")
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ContractError("Supported-v2 protocol schema is invalid")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported-v2 protocol ID mismatch")
    if value.get("state") != "frozen_before_supported_official_evaluation":
        raise ContractError("Supported-v2 protocol is not frozen before scoring")
    if value.get("source_evidence", {}).get("source_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ContractError("Immutable source protocol ID mismatch")

    source = value["source_evidence"]
    original = source.get("prior_failed_protocol_evidence", {})
    prior_v1 = source.get("prior_supported_v1_failure_evidence", {})
    if original.get("rerun_permitted") is not False:
        raise ContractError("Original failed protocol must remain non-rerunnable")
    if prior_v1.get("protocol_id") != PRIOR_SUPPORTED_PROTOCOL_ID:
        raise ContractError("Prior supported-v1 protocol ID mismatch")
    if prior_v1.get("rerun_permitted") is not False:
        raise ContractError("Prior supported-v1 failure must remain non-rerunnable")

    evaluation = value.get("evaluation", {})
    required_evaluation = {
        "maximum_evaluate_invocations": 1,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "authorized_command_count": 4,
        "result_tuning_after_score": False,
        "rerun_after_score_or_error": False,
    }
    for key, expected in required_evaluation.items():
        if evaluation.get(key) != expected:
            raise ContractError(f"Supported-v2 evaluation contract mismatch: {key}")

    expected_order = [
        "official_bop24_pose_map:single_view",
        "official_bop24_pose_map:multi_view",
        "official_bop22_bbox_ap:shared_predicted_input",
        "official_bop22_segm_ap:shared_predicted_input",
    ]
    if evaluation.get("command_order") != expected_order:
        raise ContractError("Supported-v2 official command order is not frozen")

    support = value.get("support_matrix", {})
    expected_metrics = {
        "official_bop19_average_recall",
        "official_bop24_pose_map",
        "official_bop22_bbox_ap",
        "official_bop22_segm_ap",
    }
    if set(support) != expected_metrics:
        raise ContractError("Supported-v2 metric matrix fields are not frozen")
    bop19 = support["official_bop19_average_recall"]
    if bop19.get("status") != "unavailable" or bop19.get("must_not_invoke") is not True:
        raise ContractError("Official BOP19 AR must remain unavailable for XYZ-IBD")
    if bop19.get("required_error_types") != ["vsd", "mssd", "mspd"]:
        raise ContractError("Pinned BOP19 AR composition has changed")
    if bop19.get("missing_xyzibd_definition") != "vsd":
        raise ContractError("XYZ-IBD BOP19 blocker must remain VSD")
    for metric in expected_metrics - {"official_bop19_average_recall"}:
        if support[metric].get("status") != "available":
            raise ContractError(f"Supported metric unexpectedly unavailable: {metric}")

    hashes: dict[str, Any] = {
        "source protocol": source.get("source_protocol_sha256"),
        "source fingerprint": source.get("source_preflight_fingerprint_sha256"),
        "prediction archive": source.get("prediction_archive", {}).get("sha256"),
        "original marker": original.get("invocation_marker_sha256"),
        "original failure": original.get("failure_receipt_sha256"),
        "prior-v1 protocol": prior_v1.get("protocol_sha256"),
        "prior-v1 marker": prior_v1.get("invocation_marker_sha256"),
        "prior-v1 receipt": prior_v1.get("sealed_receipt_sha256"),
        "prior-v1 log": prior_v1.get("raw_log_sha256"),
        "dataset archive": value.get("dataset", {}).get("val_archive", {}).get("sha256"),
        "dataset receipt": value.get("dataset", {}).get("verification_receipt_sha256"),
        "toolkit tree": value.get("toolkit", {}).get("source_tree_sha256"),
    }
    for group in (
        source.get("prediction_sha256", {}),
        source.get("source_artifact_sha256", {}),
        value.get("dataset", {}).get("public_input_sha256", {}),
        value.get("toolkit", {}).get("entrypoint_sha256", {}),
    ):
        hashes.update(group)
    for label, digest in hashes.items():
        if not _valid_sha256(digest):
            raise ContractError(f"Invalid SHA-256 contract for {label}")
    return value


def assert_namespaced_path(
    path: Path,
    repo_root: Path,
    protocol: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    key = {
        "pre_score": "pre_score_relative_path",
        "official_once": "official_once_relative_path",
        "report": "report_relative_path",
    }[kind]
    namespace = (repo_root.resolve() / protocol["evaluation"][key]).resolve()
    if not _is_relative_to(path.resolve(), namespace):
        raise ContractError(f"Path is outside the supported-v2 {kind} namespace")


def _file_audit(
    path: Path, *, expected_sha256: str, expected_bytes: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("file is missing")
        result["ready"] = False
        return result
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    if expected_bytes is not None and result["bytes"] != expected_bytes:
        result["errors"].append("byte count mismatch")
    if result["sha256"] != expected_sha256:
        result["errors"].append("SHA-256 mismatch")
    result["ready"] = not result["errors"]
    return result


def _audit_prior_supported_v1(
    *,
    protocol_path: Path,
    invocation_marker: Path,
    sealed_receipt: Path,
    raw_log: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    contract = protocol["source_evidence"]["prior_supported_v1_failure_evidence"]
    paths = {
        "protocol": (protocol_path, contract["protocol_sha256"]),
        "invocation_marker": (invocation_marker, contract["invocation_marker_sha256"]),
        "sealed_receipt": (sealed_receipt, contract["sealed_receipt_sha256"]),
        "raw_log": (raw_log, contract["raw_log_sha256"]),
    }
    audit: dict[str, Any] = {"rerun_permitted": False}
    errors: list[str] = []
    for role, (path, digest) in paths.items():
        item = _file_audit(path, expected_sha256=digest)
        audit[role] = item
        errors.extend(f"prior supported-v1 {role}: {error}" for error in item["errors"])

    if audit["protocol"]["ready"]:
        value = read_json(protocol_path)
        if value.get("protocol_id") != PRIOR_SUPPORTED_PROTOCOL_ID:
            errors.append("prior supported-v1 protocol content ID mismatch")
    if audit["invocation_marker"]["ready"]:
        value = read_json(invocation_marker)
        if value.get("evaluate_invocation_count") != 1:
            errors.append("prior supported-v1 invocation count is not one")
        if value.get("protocol_id") != PRIOR_SUPPORTED_PROTOCOL_ID:
            errors.append("prior supported-v1 invocation protocol ID mismatch")
    if audit["sealed_receipt"]["ready"]:
        value = read_json(sealed_receipt)
        if value.get("status") != "failed_after_evaluate_started":
            errors.append("prior supported-v1 failed status changed")
        if value.get("evaluate_invocation_count") != 1:
            errors.append("prior supported-v1 sealed invocation count is not one")
        if value.get("rerun_permitted") is not False:
            errors.append("prior supported-v1 receipt unexpectedly permits rerun")
    if audit["raw_log"]["ready"]:
        text = raw_log.read_text(encoding="utf-8", errors="replace")
        if "ModuleNotFoundError: No module named 'bop_toolkit_lib'" not in text:
            errors.append("prior supported-v1 raw failure signature changed")
    audit["errors"] = errors
    audit["ready"] = not errors
    return audit, errors


def audit_orchestrator_sources() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files = []
    for name in ("__init__.py", "__main__.py", "cli.py", "core.py"):
        path = package_root / name
        if not path.is_file():
            raise ContractError(f"Supported-v2 orchestrator source is missing: {path}")
        files.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "package_root": str(package_root),
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }


def _official_process_audit() -> dict[str, Any]:
    entrypoints = ("eval_bop24_pose.py", "eval_bop22_coco.py")
    if os.name != "posix":
        return {"supported": False, "count": 0, "matches": [], "errors": []}
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "supported": True,
            "count": None,
            "matches": [],
            "errors": [f"ps failed with exit code {completed.returncode}"],
        }
    matches = [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(entrypoint in line for entrypoint in entrypoints)
    ]
    return {"supported": True, "count": len(matches), "matches": matches, "errors": []}


def _score_file_audit(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    namespace = (
        repo_root.resolve()
        / Path(protocol["evaluation"]["official_once_relative_path"]).parent
    ).resolve()
    files: list[str] = []
    if namespace.is_dir():
        for path in namespace.rglob("*.json"):
            if path.name.startswith("scores") or path.name == "official_scores.json":
                files.append(str(path.resolve()))
    return {"namespace": str(namespace), "count": len(files), "files": sorted(files)}


_SMOKE_PROGRAM = r'''import importlib
import importlib.metadata as metadata
import json
import pathlib
import sys
import urllib.parse

import numpy as np

toolkit_root = pathlib.Path(sys.argv[1]).resolve()
model_path = pathlib.Path(sys.argv[2]).resolve()
module_names = [
    "bop_toolkit_lib", "numpy", "scipy", "cv2", "pycocotools",
    "vispy", "OpenGL", "imageio", "skimage", "PIL"
]
modules = {}
for name in module_names:
    module = importlib.import_module(name)
    modules[name] = {
        "origin": str(pathlib.Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None,
        "version": getattr(module, "__version__", None),
    }

distribution_candidates = []
for distribution in metadata.distributions(name="bop_toolkit_lib"):
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    editable_root = None
    if direct_url:
        parsed = urllib.parse.urlparse(direct_url["url"])
        editable_root = str(pathlib.Path(urllib.parse.unquote(parsed.path)).resolve())
    distribution_candidates.append({
        "version": distribution.version,
        "metadata_path": str(pathlib.Path(distribution._path).resolve()),
        "direct_url": direct_url,
        "editable_project_root": editable_root,
    })
editable_matches = [
    item for item in distribution_candidates
    if item["editable_project_root"] == str(toolkit_root)
    and (item["direct_url"] or {}).get("dir_info", {}).get("editable") is True
]
selected_distribution = editable_matches[0] if len(editable_matches) == 1 else None

from bop_toolkit_lib.rendering import renderer
renderer_instance = renderer.create_renderer(64, 48, renderer_type="vispy", mode="depth")
renderer_instance.add_object(1, str(model_path))
rendered = renderer_instance.render_object(
    1,
    np.eye(3, dtype=np.float64),
    np.asarray([[0.0], [0.0], [1000.0]], dtype=np.float64),
    600.0,
    600.0,
    32.0,
    24.0,
)
depth = np.asarray(rendered["depth"])
payload = {
    "python_executable": str(pathlib.Path(sys.executable).absolute()),
    "python_version": sys.version,
    "sys_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "sys_base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    "venv_active": sys.prefix != sys.base_prefix,
    "toolkit_root": str(toolkit_root),
    "bop_toolkit_lib_origin": modules["bop_toolkit_lib"]["origin"],
    "distribution_candidates": distribution_candidates,
    "editable_match_count": len(editable_matches),
    "distribution_version": selected_distribution["version"] if selected_distribution else None,
    "distribution_direct_url": selected_distribution["direct_url"] if selected_distribution else None,
    "editable_project_root": selected_distribution["editable_project_root"] if selected_distribution else None,
    "modules": modules,
    "renderer": {
        "status": "ready",
        "backend": "vispy-egl-depth",
        "depth_shape": list(depth.shape),
        "finite": bool(np.isfinite(depth).all()),
        "positive_pixel_count": int(np.count_nonzero(depth > 0)),
    },
}
if payload["renderer"]["depth_shape"] != [48, 64]:
    raise RuntimeError("renderer depth shape mismatch")
if not payload["renderer"]["finite"] or payload["renderer"]["positive_pixel_count"] <= 0:
    raise RuntimeError("renderer did not produce finite positive depth")
print("POSELOOP_SMOKE_JSON=" + json.dumps(payload, sort_keys=True))
'''


def _resolve_executable(value: str) -> Path | None:
    # Keep a venv launcher path intact.  Resolving its symlink would execute the
    # base interpreter directly and silently discard the venv's sys.prefix.
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.absolute()
    resolved = shutil.which(value)
    return Path(resolved).absolute() if resolved else None


def create_environment_smoke_receipt(
    *,
    protocol: Mapping[str, Any],
    python_executable: str,
    toolkit_root: Path,
    model_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    executable = _resolve_executable(python_executable)
    process_before = _official_process_audit()
    score_before = _score_file_audit(repo_root, protocol)
    errors: list[str] = []
    if executable is None:
        errors.append(f"Python executable is missing: {python_executable}")
    expected_model = protocol["dataset"]["renderer_smoke_public_model"]
    if model_path.as_posix().replace("\\", "/").split("/xyzibd/")[-1] != expected_model:
        errors.append("renderer smoke model is not the frozen public model path")
    if not model_path.is_file():
        errors.append("renderer smoke public model is missing")

    completed: subprocess.CompletedProcess[str] | None = None
    payload: dict[str, Any] | None = None
    env_contract = {
        "PYTHONPATH": str(toolkit_root.resolve()),
        "PYOPENGL_PLATFORM": "egl",
    }
    if not errors:
        env = os.environ.copy()
        env.update(env_contract)
        completed = subprocess.run(
            [str(executable), "-c", _SMOKE_PROGRAM, str(toolkit_root.resolve()), str(model_path.resolve())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            errors.append(f"environment smoke child exited {completed.returncode}")
        prefix = "POSELOOP_SMOKE_JSON="
        payload_lines = [line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)]
        if len(payload_lines) != 1:
            errors.append("environment smoke child did not emit exactly one payload")
        else:
            try:
                payload = json.loads(payload_lines[0])
            except json.JSONDecodeError as exc:
                errors.append(f"environment smoke payload is invalid JSON: {exc}")

    process_after = _official_process_audit()
    score_after = _score_file_audit(repo_root, protocol)
    for label, audit in (("before", process_before), ("after", process_after)):
        if audit["errors"] or audit["count"] != 0:
            errors.append(f"official evaluator process count is not zero {label}")
    if score_before["count"] != 0 or score_after["count"] != 0:
        errors.append("supported-v2 score files exist during environment smoke")

    if payload is not None and executable is not None:
        origin = Path(str(payload.get("bop_toolkit_lib_origin", ""))).resolve()
        editable_root = Path(str(payload.get("editable_project_root", ""))).resolve()
        if payload.get("python_executable") != str(executable):
            errors.append("smoke interpreter differs from requested interpreter")
        expected_prefix = executable.parent.parent.absolute()
        if payload.get("sys_prefix") != str(expected_prefix):
            errors.append("smoke sys.prefix differs from requested venv")
        if payload.get("venv_active") is not True:
            errors.append("smoke interpreter does not report an active venv")
        if payload.get("editable_match_count") != 1:
            errors.append("pinned editable bop_toolkit_lib distribution is not unique")
        if not _is_relative_to(origin, toolkit_root.resolve()):
            errors.append("bop_toolkit_lib import origin is outside pinned toolkit")
        if editable_root != toolkit_root.resolve():
            errors.append("editable bop_toolkit_lib project root differs from pinned toolkit")
        direct_url = payload.get("distribution_direct_url") or {}
        if direct_url.get("dir_info", {}).get("editable") is not True:
            errors.append("bop_toolkit_lib distribution is not an editable install")
        if payload.get("renderer", {}).get("status") != "ready":
            errors.append("renderer smoke is not ready")

    model_audit = {
        "path": str(model_path.resolve()),
        "exists": model_path.is_file(),
        "bytes": model_path.stat().st_size if model_path.is_file() else None,
        "sha256": sha256_file(model_path) if model_path.is_file() else None,
        "public_input_only": True,
    }
    return {
        "schema_version": SMOKE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "python_requested": python_executable,
        "python_resolved": str(executable) if executable else None,
        "toolkit_root": str(toolkit_root.resolve()),
        "model": model_audit,
        "environment_contract": env_contract,
        "child_exit_code": completed.returncode if completed else None,
        "child_payload": payload,
        "child_stdout": completed.stdout if completed else "",
        "child_stderr": completed.stderr if completed else "",
        "official_evaluator_invocation_count": 0,
        "official_process_audit_before": process_before,
        "official_process_audit_after": process_after,
        "score_file_audit_before": score_before,
        "score_file_audit_after": score_after,
        "prediction_and_selection_label_access_count": 0,
        "evaluator_only_label_paths_accessed": [],
        "errors": errors,
    }


def _audit_environment_smoke(
    path: Path,
    *,
    python_executable: str,
    toolkit_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        errors.append("environment smoke receipt is missing")
        result["errors"] = errors
        result["ready"] = False
        return result, errors
    result["sha256"] = sha256_file(path)
    value = read_json(path)
    result["receipt"] = value
    executable = _resolve_executable(python_executable)
    checks = {
        "schema": value.get("schema_version") == SMOKE_SCHEMA,
        "protocol": value.get("protocol_id") == PROTOCOL_ID,
        "status": value.get("status") == "ready",
        "interpreter": executable is not None and value.get("python_resolved") == str(executable),
        "toolkit": value.get("toolkit_root") == str(toolkit_root.resolve()),
        "pythopath": value.get("environment_contract", {}).get("PYTHONPATH") == str(toolkit_root.resolve()),
        "editable": value.get("child_payload", {}).get("distribution_direct_url", {}).get("dir_info", {}).get("editable") is True,
        "renderer": value.get("child_payload", {}).get("renderer", {}).get("status") == "ready",
        "no_evaluator": value.get("official_evaluator_invocation_count") == 0,
        "no_label": value.get("prediction_and_selection_label_access_count") == 0,
        "no_score_before": value.get("score_file_audit_before", {}).get("count") == 0,
        "no_score_after": value.get("score_file_audit_after", {}).get("count") == 0,
        "no_process_before": value.get("official_process_audit_before", {}).get("count") == 0,
        "no_process_after": value.get("official_process_audit_after", {}).get("count") == 0,
    }
    errors.extend(f"environment smoke check failed: {key}" for key, ok in checks.items() if not ok)
    result["checks"] = checks
    result["errors"] = errors
    result["ready"] = not errors
    return result, errors


def create_preflight_receipt(
    *,
    protocol_path: Path,
    source_protocol_path: Path,
    prior_v1_protocol_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    input_root: Path,
    prediction_archive: Path,
    dataset_verification_receipt: Path,
    prior_invocation_marker: Path,
    prior_failure_receipt: Path,
    prior_v1_invocation_marker: Path,
    prior_v1_sealed_receipt: Path,
    prior_v1_raw_log: Path,
    environment_smoke_receipt: Path,
    output_root: Path,
    repo_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    assert_namespaced_path(output_root, repo_root, protocol, kind="official_once")
    exact_output = (repo_root.resolve() / protocol["evaluation"]["official_once_relative_path"]).resolve()
    if output_root.resolve() != exact_output:
        raise ContractError("Official output root must equal the frozen supported-v2 path")

    source_protocol, source_audit, errors = v1_core._audit_source_inputs(
        source_protocol_path=source_protocol_path,
        dataset_root=dataset_root,
        toolkit_root=toolkit_root,
        input_root=input_root,
        prediction_archive=prediction_archive,
        dataset_verification_receipt=dataset_verification_receipt,
        prior_invocation_marker=prior_invocation_marker,
        prior_failure_receipt=prior_failure_receipt,
        repo_root=repo_root,
        protocol=protocol,
    )
    prior_v1, prior_errors = _audit_prior_supported_v1(
        protocol_path=prior_v1_protocol_path,
        invocation_marker=prior_v1_invocation_marker,
        sealed_receipt=prior_v1_sealed_receipt,
        raw_log=prior_v1_raw_log,
        protocol=protocol,
    )
    errors.extend(prior_errors)
    support = v1_core.audit_support_matrix(toolkit_root, protocol)
    errors.extend(support["errors"])
    orchestrator = audit_orchestrator_sources()
    environment, environment_errors = _audit_environment_smoke(
        environment_smoke_receipt,
        python_executable=python_executable,
        toolkit_root=toolkit_root,
        protocol=protocol,
    )
    errors.extend(environment_errors)

    repository = source_audit["repository"]
    if repository.get("branch") != protocol["required_execution_branch"]:
        errors.append("execution branch differs from frozen supported-v2 branch")
    if repository.get("status_entries") != []:
        errors.append("execution repository is not clean")
    if not repository.get("head"):
        errors.append("implementation commit is unavailable")

    process_audit = _official_process_audit()
    score_audit = _score_file_audit(repo_root, protocol)
    errors.extend(process_audit["errors"])
    if process_audit["count"] != 0:
        errors.append("official evaluator process count is not zero")
    if score_audit["count"] != 0:
        errors.append("supported-v2 score file count is not zero")
    if output_root.exists():
        errors.append("supported-v2 official output root already exists")

    commands = v1_core.build_supported_commands(
        input_root,
        toolkit_root,
        output_root,
        source_protocol,
        protocol,
        str(_resolve_executable(python_executable) or python_executable),
    )
    command_sha = canonical_sha256(commands)
    protocol_sha = sha256_file(protocol_path)
    fingerprint = {
        "protocol_sha256": protocol_sha,
        "implementation_commit": repository.get("head"),
        "execution_branch": repository.get("branch"),
        "orchestrator_source_manifest_sha256": orchestrator["manifest_sha256"],
        "source_preflight_fingerprint_sha256": source_audit["source_preflight_fingerprint_sha256"],
        "prediction_archive_sha256": source_audit["prediction_archive"].get("sha256"),
        "prediction_sha256": protocol["source_evidence"]["prediction_sha256"],
        "data_sha256": {
            "val_archive": protocol["dataset"]["val_archive"]["sha256"],
            **protocol["dataset"]["public_input_sha256"],
        },
        "model_sha256": protocol["source_evidence"]["source_artifact_sha256"]["model"],
        "toolkit_commit": protocol["toolkit"]["commit"],
        "toolkit_source_tree_sha256": protocol["toolkit"]["source_tree_sha256"],
        "original_failure_evidence": protocol["source_evidence"]["prior_failed_protocol_evidence"],
        "prior_v1_failure_evidence": protocol["source_evidence"]["prior_supported_v1_failure_evidence"],
        "environment_smoke_receipt_sha256": environment.get("sha256"),
        "environment_python": str(_resolve_executable(python_executable) or python_executable),
        "official_command_manifest_sha256": command_sha,
        "official_output_root": str(output_root.resolve()),
        "score_file_count": score_audit["count"],
        "official_process_count": process_audit["count"],
        "label_access_count": 0,
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "package_version": PACKAGE_VERSION,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "protocol": {"path": str(protocol_path.resolve()), "sha256": protocol_sha},
        "implementation_commit": repository.get("head"),
        "source_audit": source_audit,
        "prior_supported_v1_audit": prior_v1,
        "support_audit": support,
        "orchestrator_source_audit": orchestrator,
        "environment_smoke_audit": environment,
        "official_process_audit": process_audit,
        "score_file_audit": score_audit,
        "official_commands": commands,
        "official_command_manifest_sha256": command_sha,
        "official_output_root": str(output_root.resolve()),
        "official_evaluation_executed": False,
        "label_access_audit": {
            "prediction_and_selection_label_access_count": 0,
            "evaluator_only_label_paths_accessed": [],
            "official_evaluator_label_access_started": False,
        },
        "unavailable_metrics": {
            "official_bop19_average_recall": protocol["support_matrix"]["official_bop19_average_recall"]
        },
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "errors": errors,
    }


def freeze_input_lock(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if preflight.get("status") != "ready":
        raise ContractError("Cannot freeze a blocked supported-v2 preflight")
    if preflight.get("official_evaluation_executed") is not False:
        raise ContractError("Preflight unexpectedly entered official evaluation")
    if preflight.get("score_file_audit", {}).get("count") != 0:
        raise ContractError("Cannot freeze after a supported-v2 score file exists")
    if preflight.get("official_process_audit", {}).get("count") != 0:
        raise ContractError("Cannot freeze while an official evaluator process exists")
    lock = dict(preflight)
    lock["schema_version"] = LOCK_SCHEMA
    lock["state"] = "frozen_before_supported_official_evaluation"
    lock["frozen_utc"] = utc_now()
    return lock


def validate_input_lock(path: Path, current_preflight: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(path)
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA:
        raise ContractError("Supported-v2 input lock schema is invalid")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported-v2 input lock protocol ID mismatch")
    if lock.get("state") != "frozen_before_supported_official_evaluation":
        raise ContractError("Supported-v2 input lock state is invalid")
    if current_preflight.get("status") != "ready":
        raise ContractError("Current supported-v2 preflight is blocked")
    if lock.get("fingerprint_sha256") != current_preflight.get("fingerprint_sha256"):
        raise ContractError("Current inputs, implementation, environment, or commands differ from lock")
    return lock


def create_evaluator_authorization(
    *,
    input_lock_path: Path,
    protocol: Mapping[str, Any],
    approved_by: str,
    authorization_reference: str,
) -> dict[str, Any]:
    lock = read_json(input_lock_path)
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA:
        raise ContractError("Cannot authorize an invalid supported-v2 lock")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Cannot authorize another protocol")
    if authorization_reference != protocol["authorization"]["reference"]:
        raise ContractError("Authorization reference differs from the frozen user delegation")
    if not approved_by.strip():
        raise ContractError("Authorization identity is empty")
    return {
        "schema_version": AUTH_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "scope": protocol["authorization"]["scope"],
        "approved_by": approved_by,
        "approved_at_utc": utc_now(),
        "authorization_reference": authorization_reference,
        "input_lock_sha256": sha256_file(input_lock_path),
        "implementation_commit": lock["implementation_commit"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "environment_smoke_receipt_sha256": lock["fingerprint"]["environment_smoke_receipt_sha256"],
        "official_command_manifest_sha256": lock["official_command_manifest_sha256"],
        "maximum_evaluate_invocations": 1,
        "maximum_invocations_per_command": 1,
        "authorized_command_count": 4,
        "continue_after_command_failure": True,
        "prior_protocols_rerun_permitted": False,
        "prediction_and_selection_label_access_count_before_evaluate": 0,
    }


def validate_evaluator_authorization(
    path: Path,
    input_lock_path: Path,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != AUTH_SCHEMA:
        raise ContractError("Supported-v2 evaluator authorization schema is invalid")
    required = {
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "scope": protocol["authorization"]["scope"],
        "authorization_reference": protocol["authorization"]["reference"],
        "input_lock_sha256": sha256_file(input_lock_path),
        "implementation_commit": lock["implementation_commit"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "environment_smoke_receipt_sha256": lock["fingerprint"]["environment_smoke_receipt_sha256"],
        "official_command_manifest_sha256": lock["official_command_manifest_sha256"],
        "maximum_evaluate_invocations": 1,
        "maximum_invocations_per_command": 1,
        "authorized_command_count": 4,
        "continue_after_command_failure": True,
        "prior_protocols_rerun_permitted": False,
        "prediction_and_selection_label_access_count_before_evaluate": 0,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ContractError(f"Evaluator authorization mismatch: {key}")
    for key in ("approved_by", "approved_at_utc"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ContractError(f"Evaluator authorization field is empty: {key}")
    return value


def create_final_prescore_receipt(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_evaluator_authorization(
        authorization_path, input_lock_path, lock, protocol
    )
    if output_root.exists():
        raise ContractError("Supported-v2 official output root exists before evaluate")
    if current_preflight["score_file_audit"]["count"] != 0:
        raise ContractError("Supported-v2 score files exist before evaluate")
    if current_preflight["official_process_audit"]["count"] != 0:
        raise ContractError("Official evaluator process exists before evaluate")
    label_audit = current_preflight["label_access_audit"]
    if label_audit != {
        "prediction_and_selection_label_access_count": 0,
        "evaluator_only_label_paths_accessed": [],
        "official_evaluator_label_access_started": False,
    }:
        raise ContractError("Label-access audit is not zero at final pre-score")
    return {
        "schema_version": "poseloop.r3.xyzibd-supported.final-prescore.v2",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "ready",
        "protocol_sha256": current_preflight["protocol"]["sha256"],
        "implementation_commit": current_preflight["implementation_commit"],
        "preflight_fingerprint_sha256": current_preflight["fingerprint_sha256"],
        "orchestrator_source_manifest_sha256": current_preflight["orchestrator_source_audit"]["manifest_sha256"],
        "environment_smoke_receipt_sha256": current_preflight["environment_smoke_audit"]["sha256"],
        "official_command_manifest_sha256": current_preflight["official_command_manifest_sha256"],
        "input_lock": {"path": str(input_lock_path.resolve()), "sha256": sha256_file(input_lock_path)},
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "approved_by": authorization["approved_by"],
            "authorization_reference": authorization["authorization_reference"],
        },
        "official_output_root": str(output_root.resolve()),
        "official_output_root_absent": True,
        "score_file_count": 0,
        "official_process_count": 0,
        "authorized_command_count": 4,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "command_ids": [item["command_id"] for item in current_preflight["official_commands"]],
        "official_bop19_entrypoint_command_count": current_preflight["support_audit"]["official_bop19_entrypoint_command_count"],
        "unavailable_metrics": current_preflight["unavailable_metrics"],
        "label_access_audit": label_audit,
        "official_evaluation_executed": False,
        "prior_protocols_rerun_permitted": False,
    }


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _output_manifest(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "sealed_receipt.json":
            rows.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _read_score(command: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(command["expected_score"]))
    if execution["exit_code"] != 0:
        return {
            "status": "unavailable",
            "reason": f"official command exited {execution['exit_code']}",
            "path": str(path.resolve()),
        }
    if not path.is_file():
        return {
            "status": "unavailable",
            "reason": "official command exited zero but expected score file is missing",
            "path": str(path.resolve()),
        }
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "reason": f"invalid score JSON: {exc}", "path": str(path.resolve())}
    key = command["expected_score_key"]
    aggregate = value.get(key) if isinstance(value, dict) else None
    if not isinstance(aggregate, (int, float)) or not math.isfinite(float(aggregate)):
        return {"status": "unavailable", "reason": f"score lacks finite {key}", "path": str(path.resolve())}
    return {
        "status": "available",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "aggregate_key": key,
        "aggregate_value": float(aggregate),
        "scores": value,
    }


def _metrics(scores: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    single = scores["official_bop24_pose_map:single_view"]
    multi = scores["official_bop24_pose_map:multi_view"]
    bbox = scores["official_bop22_bbox_ap:shared_predicted_input"]
    segm = scores["official_bop22_segm_ap:shared_predicted_input"]
    single_value = single.get("aggregate_value") if single["status"] == "available" else None
    multi_value = multi.get("aggregate_value") if multi["status"] == "available" else None
    return {
        "official_bop19_average_recall": {
            "status": "unavailable",
            "value": None,
            "reason": protocol["support_matrix"]["official_bop19_average_recall"]["reason"],
        },
        "official_bop24_pose_map": {
            "status": "available" if single_value is not None and multi_value is not None else "partial_or_unavailable",
            "single_view": single_value,
            "single_view_status": single["status"],
            "multi_view": multi_value,
            "multi_view_status": multi["status"],
            "multi_minus_single": multi_value - single_value if single_value is not None and multi_value is not None else None,
        },
        "official_bop22_bbox_ap": {
            "status": bbox["status"],
            "shared_predicted_input": bbox.get("aggregate_value"),
        },
        "official_bop22_segm_ap": {
            "status": segm["status"],
            "shared_predicted_input": segm.get("aggregate_value"),
        },
    }


def execute_official_evaluation(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    output_root: Path,
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_evaluator_authorization(
        authorization_path, input_lock_path, lock, protocol
    )
    assert_namespaced_path(output_root, repo_root, protocol, kind="official_once")
    exact_output = (repo_root.resolve() / protocol["evaluation"]["official_once_relative_path"]).resolve()
    if output_root.resolve() != exact_output:
        raise ContractError("Supported-v2 official output root differs from lock")
    if output_root.exists():
        raise ContractError("Supported-v2 official output exists; rerun is permanently refused")

    commands = list(current_preflight["official_commands"])
    if canonical_sha256(commands) != current_preflight["official_command_manifest_sha256"]:
        raise ContractError("Official command manifest is internally inconsistent")
    if [item["command_id"] for item in commands] != protocol["evaluation"]["command_order"]:
        raise ContractError("Official command order differs from protocol")
    forbidden = protocol["support_matrix"]["official_bop19_average_recall"]["entrypoint"]
    if any(forbidden in argument for command in commands for argument in command["argv"]):
        raise ContractError("Forbidden BOP19 entrypoint reached execution boundary")

    output_root.mkdir(parents=True, exist_ok=False)
    started = {
        "schema_version": "poseloop.r3.xyzibd-supported.invocation-started.v2",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "evaluate_invocation_count": 1,
        "authorized_command_count": 4,
        "maximum_invocations_per_command": 1,
        "continue_after_command_failure": True,
        "official_command_manifest_sha256": current_preflight["official_command_manifest_sha256"],
        "input_lock_sha256": sha256_file(input_lock_path),
        "authorization_sha256": sha256_file(authorization_path),
        "prior_protocols_rerun_permitted": False,
        "prediction_and_selection_label_access_count": 0,
        "official_evaluator_label_access_started": True,
    }
    started_path = output_root / "invocation_started.json"
    _write_exclusive_json(started_path, started)

    input_root = Path(current_preflight["source_audit"]["prediction_bundle"]["root"])
    env = os.environ.copy()
    env.update(
        {
            "BOP_PATH": str(dataset_root.resolve().parent),
            "BOP_RESULTS_PATH": str(input_root.resolve()),
            "BOP_EVAL_PATH": str((output_root / "eval").resolve()),
            "PYTHONPATH": str(toolkit_root.resolve()),
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    python_path = Path(str(commands[0]["argv"][0])).absolute()
    env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")

    executions: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        log_path = output_root / "logs" / f"{index:02d}_{command['metric']}_{command['variant']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code = 127
        launch_error: dict[str, str] | None = None
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            log.write(f"command_id={command['command_id']}\n")
            log.write(f"argv={json.dumps(command['argv'])}\n")
            log.write(f"PYTHONPATH={env['PYTHONPATH']}\n")
            log.write("PYOPENGL_PLATFORM=egl\n")
            log.flush()
            try:
                completed = subprocess.run(
                    list(command["argv"]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                )
                exit_code = completed.returncode
            except BaseException as exc:
                launch_error = {"type": type(exc).__name__, "message": str(exc)}
                log.write(f"\nlaunch_error={json.dumps(launch_error)}\n")
            log.write(f"\nexit_code={exit_code}\n")
        executions.append(
            {
                "index": index,
                "command_id": command["command_id"],
                "metric": command["metric"],
                "variant": command["variant"],
                "attempt_count": 1,
                "exit_code": exit_code,
                "launch_error": launch_error,
                "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
            }
        )

    score_records = {
        command["command_id"]: _read_score(command, execution)
        for command, execution in zip(commands, executions, strict=True)
    }
    metrics = _metrics(score_records, protocol)
    all_available = all(item["status"] == "available" for item in score_records.values())
    any_available = any(item["status"] == "available" for item in score_records.values())
    result_status = "complete" if all_available else "partial" if any_available else "failed"
    result = {
        "schema_version": "poseloop.r3.xyzibd-supported.official-scores.v2",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": result_status,
        "evaluate_invocation_count": 1,
        "command_invocation_counts": {item["command_id"]: 1 for item in executions},
        "input_lock": {
            "path": str(input_lock_path.resolve()),
            "sha256": sha256_file(input_lock_path),
            "fingerprint_sha256": lock["fingerprint_sha256"],
        },
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "approved_by": authorization["approved_by"],
            "authorization_reference": authorization["authorization_reference"],
        },
        "environment": {
            "python": str(python_path),
            "PYTHONPATH": env["PYTHONPATH"],
            "PYOPENGL_PLATFORM": env["PYOPENGL_PLATFORM"],
            "smoke_receipt_sha256": current_preflight["environment_smoke_audit"]["sha256"],
        },
        "official_evaluation_executed": True,
        "prior_protocols_rerun_permitted": False,
        "prediction_and_selection_label_access_count": 0,
        "executions": executions,
        "raw_official_scores": score_records,
        "metrics": metrics,
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output_root / "official_scores.json"
    write_json_atomic(result_path, result)
    manifest = _output_manifest(output_root)
    sealed = {
        "schema_version": "poseloop.r3.xyzibd-supported.sealed-receipt.v2",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "complete" if result_status == "complete" else "partial_or_failed_after_evaluate_started",
        "result_status": result_status,
        "evaluate_invocation_count": 1,
        "authorized_command_count": 4,
        "attempted_command_count": len(executions),
        "command_invocation_counts": {item["command_id"]: 1 for item in executions},
        "all_exit_codes": [item["exit_code"] for item in executions],
        "continued_after_failures": True,
        "official_scores_path": str(result_path.resolve()),
        "official_scores_sha256": sha256_file(result_path),
        "input_lock_sha256": sha256_file(input_lock_path),
        "authorization_sha256": sha256_file(authorization_path),
        "invocation_started_sha256": sha256_file(started_path),
        "output_manifest": manifest,
        "output_manifest_sha256": canonical_sha256(manifest),
        "rerun_permitted": False,
    }
    write_json_atomic(output_root / "sealed_receipt.json", sealed)
    return result
