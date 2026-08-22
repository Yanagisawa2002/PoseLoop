"""One-shot official scoring for the XYZ-IBD-supported R3 metric subset.

The source R3 protocol, its failed invocation, and its prediction bundle are
immutable evidence.  This namespace only re-audits those prediction-side files,
freezes the subset that the pinned BOP Toolkit officially supports for XYZ-IBD,
and opens evaluator labels after a new, hash-bound authorization.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from r3_bop_industrial import core as source_core

from . import PACKAGE_VERSION, PROTOCOL_ID, SOURCE_PROTOCOL_ID


ContractError = source_core.ContractError
canonical_sha256 = source_core.canonical_sha256
read_json = source_core.read_json
sha256_file = source_core.sha256_file
utc_now = source_core.utc_now
write_json_atomic = source_core.write_json_atomic


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
        raise ContractError("Supported XYZ-IBD protocol must be a JSON object")
    if value.get("schema_version") != "poseloop.r3.xyzibd-supported.protocol.v1":
        raise ContractError("Supported XYZ-IBD protocol schema is invalid")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported XYZ-IBD protocol ID mismatch")
    if value.get("state") != "frozen_before_supported_official_evaluation":
        raise ContractError("Supported XYZ-IBD protocol is not frozen before scoring")

    source = value.get("source_evidence", {})
    if source.get("source_protocol_id") != SOURCE_PROTOCOL_ID:
        raise ContractError("Immutable source protocol ID mismatch")
    if source.get("prior_failed_protocol_evidence", {}).get("rerun_permitted") is not False:
        raise ContractError("The failed source protocol must remain non-rerunnable")

    evaluation = value.get("evaluation", {})
    if evaluation.get("maximum_evaluate_invocations") != 1:
        raise ContractError("Supported protocol must authorize exactly one evaluate invocation")
    if evaluation.get("authorized_command_count") != 4:
        raise ContractError("Supported protocol must freeze exactly four official commands")
    if evaluation.get("result_tuning_after_score") is not False:
        raise ContractError("Supported protocol must forbid tuning after scoring")
    if evaluation.get("rerun_after_score_or_error") is not False:
        raise ContractError("Supported protocol must forbid reruns after score or error")

    support = value.get("support_matrix", {})
    expected_metrics = {
        "official_bop19_average_recall",
        "official_bop24_pose_map",
        "official_bop22_bbox_ap",
        "official_bop22_segm_ap",
    }
    if set(support) != expected_metrics:
        raise ContractError("Supported metric matrix fields are not frozen")
    bop19 = support["official_bop19_average_recall"]
    if bop19.get("status") != "unavailable" or bop19.get("must_not_invoke") is not True:
        raise ContractError("Official BOP19 AR must be frozen as unavailable for XYZ-IBD")
    if bop19.get("required_error_types") != ["vsd", "mssd", "mspd"]:
        raise ContractError("Official BOP19 AR error composition differs from pinned code")
    if bop19.get("missing_xyzibd_definition") != "vsd":
        raise ContractError("XYZ-IBD BOP19 blocker must be VSD")
    for metric in expected_metrics - {"official_bop19_average_recall"}:
        if support[metric].get("status") != "available":
            raise ContractError(f"Supported metric unexpectedly unavailable: {metric}")

    hash_values = {
        "source protocol": source.get("source_protocol_sha256"),
        "source fingerprint": source.get("source_preflight_fingerprint_sha256"),
        "prediction archive": source.get("prediction_archive", {}).get("sha256"),
        "old invocation marker": source.get("prior_failed_protocol_evidence", {}).get(
            "invocation_marker_sha256"
        ),
        "old failure receipt": source.get("prior_failed_protocol_evidence", {}).get(
            "failure_receipt_sha256"
        ),
        "dataset archive": value.get("dataset", {}).get("val_archive", {}).get("sha256"),
        "dataset verification receipt": value.get("dataset", {}).get(
            "verification_receipt_sha256"
        ),
        "toolkit source tree": value.get("toolkit", {}).get("source_tree_sha256"),
    }
    hash_values.update(
        {f"prediction {key}": digest for key, digest in source.get("prediction_sha256", {}).items()}
    )
    hash_values.update(
        {
            f"source artifact {key}": digest
            for key, digest in source.get("source_artifact_sha256", {}).items()
        }
    )
    hash_values.update(
        {
            f"dataset public input {key}": digest
            for key, digest in value.get("dataset", {}).get("public_input_sha256", {}).items()
        }
    )
    hash_values.update(
        {
            f"toolkit source {key}": digest
            for key, digest in value.get("toolkit", {}).get("entrypoint_sha256", {}).items()
        }
    )
    for label, digest in hash_values.items():
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
    relative_key = {
        "pre_score": "pre_score_relative_path",
        "official_once": "official_once_relative_path",
        "report": "report_relative_path",
    }[kind]
    resolved = path.resolve()
    namespace = (repo_root.resolve() / protocol["evaluation"][relative_key]).resolve()
    if not _is_relative_to(resolved, namespace):
        raise ContractError(f"Path is outside the new {kind} namespace: {resolved}")


def _file_audit(path: Path, *, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
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
        result["errors"].append(
            f"byte count mismatch: {result['bytes']} != {expected_bytes}"
        )
    if result["sha256"] != expected_sha256:
        result["errors"].append("SHA-256 mismatch")
    result["ready"] = not result["errors"]
    return result


def _audit_dataset_verification_receipt(
    path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = protocol["dataset"]
    result = _file_audit(
        path,
        expected_sha256=dataset["verification_receipt_sha256"],
    )
    if not result["ready"]:
        return result
    value = read_json(path)
    errors: list[str] = result["errors"]
    archive = value.get("archives", {}).get("xyzibd_val.zip", {})
    if archive.get("sha256") != dataset["val_archive"]["sha256"]:
        errors.append("dataset verification receipt has a different val archive SHA-256")
    if archive.get("bytes") != dataset["val_archive"]["bytes"]:
        errors.append("dataset verification receipt has a different val archive byte count")
    if archive.get("verified") is not True:
        errors.append("dataset verification receipt does not mark val archive verified")
    if value.get("status") != "ready" or value.get("errors") != []:
        errors.append("dataset verification receipt is not ready")
    if value.get("official_evaluation_executed") is not False:
        errors.append("dataset verification receipt unexpectedly entered evaluation")
    if value.get("evaluator_only_label_paths_accessed") != []:
        errors.append("dataset verification receipt accessed evaluator-only labels")
    result["receipt_schema_version"] = value.get("schema_version")
    result["verified_val_archive"] = {
        "bytes": archive.get("bytes"),
        "sha256": archive.get("sha256"),
        "verified": archive.get("verified"),
    }
    result["ready"] = not errors
    return result


def _source_fingerprint(
    *,
    source_protocol_sha256: str,
    dataset: Mapping[str, Any],
    toolkit: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_hashes = {
        role: predictions[role]["sha256"]
        for role in (
            "coco_predictions",
            "predicted_association",
            "single_view_pose",
            "multi_view_pose",
            "provenance",
        )
        if role in predictions
    }
    return {
        "protocol_sha256": source_protocol_sha256,
        "dataset_public_input_sha256": {
            relative: item.get("sha256")
            for relative, item in dataset["required_public_inputs"].items()
            if "sha256" in item
        },
        "toolkit_commit": toolkit["actual_commit"],
        "toolkit_verified_revision": toolkit["actual_commit"] or toolkit["expected_commit"],
        "toolkit_pinned_source_tree_sha256": toolkit["expected_source_tree_sha256"],
        "toolkit_evaluator_sha256": {
            relative: item.get("sha256")
            for relative, item in toolkit["official_evaluators"].items()
        },
        "prediction_sha256": prediction_hashes,
    }


def audit_support_matrix(
    toolkit_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    expected = protocol["toolkit"]["entrypoint_sha256"]
    sources: dict[str, Any] = {}
    errors: list[str] = []
    for relative, expected_sha256 in expected.items():
        path = toolkit_root.resolve() / relative
        item = _file_audit(path, expected_sha256=expected_sha256)
        sources[relative] = item
        errors.extend(f"{relative}: {error}" for error in item["errors"])

    commandable = [
        metric
        for metric, row in protocol["support_matrix"].items()
        if row["status"] == "available"
    ]
    return {
        "status": "ready" if not errors else "blocked",
        "pinned_sources": sources,
        "xyzibd_supported_error_types": list(
            protocol["toolkit"]["xyzibd_supported_error_types"]
        ),
        "official_bop19_average_recall": dict(
            protocol["support_matrix"]["official_bop19_average_recall"]
        ),
        "commandable_metrics": commandable,
        "official_bop19_entrypoint_command_count": 0,
        "errors": errors,
    }


def audit_orchestrator_sources() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files = []
    for name in ("__init__.py", "__main__.py", "cli.py", "core.py"):
        path = package_root / name
        if not path.is_file():
            raise ContractError(f"Supported orchestrator source is missing: {path}")
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


def build_supported_commands(
    input_root: Path,
    toolkit_root: Path,
    output_root: Path,
    source_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    python_executable: str = sys.executable,
) -> list[dict[str, Any]]:
    files = source_protocol["prediction_bundle"]["files"]
    eval_root = output_root.resolve() / "eval"
    workers = int(protocol["toolkit"]["workers"])
    commands: list[dict[str, Any]] = []

    pose_entrypoint = toolkit_root.resolve() / protocol["support_matrix"][
        "official_bop24_pose_map"
    ]["entrypoint"]
    for variant, role in (
        ("single_view", "single_view_pose"),
        ("multi_view", "multi_view_pose"),
    ):
        filename = files[role]
        commands.append(
            {
                "command_id": f"official_bop24_pose_map:{variant}",
                "metric": "official_bop24_pose_map",
                "variant": variant,
                "argv": [
                    python_executable,
                    str(pose_entrypoint),
                    f"--result_filenames={filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root}",
                    "--targets_filename=test_targets_bop24.json",
                    f"--num_workers={workers}",
                ],
                "expected_score": str(
                    eval_root / Path(filename).stem / "scores_bop24.json"
                ),
                "expected_score_key": "bop24_mAP",
            }
        )

    coco_entrypoint = toolkit_root.resolve() / protocol["support_matrix"][
        "official_bop22_bbox_ap"
    ]["entrypoint"]
    coco_filename = files["coco_predictions"]
    coco_result_name = Path(coco_filename).stem
    for annotation_type in ("bbox", "segm"):
        metric = f"official_bop22_{annotation_type}_ap"
        commands.append(
            {
                "command_id": f"{metric}:shared_predicted_input",
                "metric": metric,
                "variant": "shared_predicted_input",
                "argv": [
                    python_executable,
                    str(coco_entrypoint),
                    f"--result_filenames={coco_filename}",
                    f"--results_path={input_root.resolve()}",
                    f"--eval_path={eval_root}",
                    "--targets_filename=test_targets_bop19.json",
                    f"--ann_type={annotation_type}",
                    "--bbox_type=amodal",
                ],
                "expected_score": str(
                    eval_root
                    / coco_result_name
                    / f"scores_bop22_coco_{annotation_type}.json"
                ),
                "expected_score_key": "AP",
            }
        )

    expected_order = list(protocol["evaluation"]["command_order"])
    actual_order = [command["command_id"] for command in commands]
    if actual_order != expected_order:
        raise ContractError("Supported official command order differs from protocol")
    if len(commands) != protocol["evaluation"]["authorized_command_count"]:
        raise ContractError("Supported official command count differs from protocol")
    forbidden = protocol["support_matrix"]["official_bop19_average_recall"]["entrypoint"]
    if any(forbidden in argument for command in commands for argument in command["argv"]):
        raise ContractError("Forbidden BOP19 entrypoint entered the supported command list")
    return commands


def _audit_source_inputs(
    *,
    source_protocol_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    input_root: Path,
    prediction_archive: Path,
    dataset_verification_receipt: Path,
    prior_invocation_marker: Path,
    prior_failure_receipt: Path,
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    source_contract = protocol["source_evidence"]
    source_protocol_file = _file_audit(
        source_protocol_path,
        expected_sha256=source_contract["source_protocol_sha256"],
    )
    errors.extend(f"source protocol: {error}" for error in source_protocol_file["errors"])
    source_protocol = source_core.load_protocol(source_protocol_path)
    if source_protocol["protocol_id"] != SOURCE_PROTOCOL_ID:
        errors.append("source protocol ID differs from immutable evidence")

    dataset = source_core.dataset_preflight(dataset_root, source_protocol)
    toolkit = source_core.toolkit_preflight(toolkit_root, source_protocol)
    predictions = source_core.validate_prediction_bundle(input_root, dataset_root, source_protocol)
    errors.extend(dataset["errors"])
    errors.extend(toolkit["errors"])
    errors.extend(predictions["errors"])

    for relative, expected_sha256 in protocol["dataset"]["public_input_sha256"].items():
        actual = dataset["required_public_inputs"].get(relative, {}).get("sha256")
        if actual != expected_sha256:
            errors.append(f"dataset public input SHA-256 mismatch: {relative}")
    for role, expected_sha256 in source_contract["prediction_sha256"].items():
        actual = predictions.get(role, {}).get("sha256")
        if actual != expected_sha256:
            errors.append(f"prediction SHA-256 mismatch: {role}")

    actual_artifacts = {
        artifact["role"]: artifact["sha256"]
        for artifact in predictions.get("provenance", {}).get("source_artifacts", [])
    }
    for role, expected_sha256 in source_contract["source_artifact_sha256"].items():
        if actual_artifacts.get(role) != expected_sha256:
            errors.append(f"prediction source artifact SHA-256 mismatch: {role}")

    archive_contract = source_contract["prediction_archive"]
    archive = _file_audit(
        prediction_archive,
        expected_sha256=archive_contract["sha256"],
        expected_bytes=archive_contract["bytes"],
    )
    if prediction_archive.name != archive_contract["filename"]:
        archive["errors"].append("prediction archive filename mismatch")
        archive["ready"] = False
    errors.extend(f"prediction archive: {error}" for error in archive["errors"])

    dataset_receipt = _audit_dataset_verification_receipt(
        dataset_verification_receipt, protocol
    )
    errors.extend(f"dataset receipt: {error}" for error in dataset_receipt["errors"])

    prior_contract = source_contract["prior_failed_protocol_evidence"]
    old_marker = _file_audit(
        prior_invocation_marker,
        expected_sha256=prior_contract["invocation_marker_sha256"],
    )
    old_failure = _file_audit(
        prior_failure_receipt,
        expected_sha256=prior_contract["failure_receipt_sha256"],
    )
    errors.extend(f"old invocation marker: {error}" for error in old_marker["errors"])
    errors.extend(f"old failure receipt: {error}" for error in old_failure["errors"])

    repository = source_core.repo_state(repo_root, source_contract["immutable_commit"])
    if repository["baseline_is_ancestor"] is not True:
        errors.append("immutable evidence commit 36a3aaf is not an ancestor of this checkout")

    source_fingerprint = _source_fingerprint(
        source_protocol_sha256=source_protocol_file.get("sha256", ""),
        dataset=dataset,
        toolkit=toolkit,
        predictions=predictions,
    )
    source_fingerprint_sha256 = canonical_sha256(source_fingerprint)
    if source_fingerprint_sha256 != source_contract["source_preflight_fingerprint_sha256"]:
        errors.append("current source inputs differ from frozen source preflight fingerprint")

    audit = {
        "source_protocol": source_protocol_file,
        "source_preflight_fingerprint": source_fingerprint,
        "source_preflight_fingerprint_sha256": source_fingerprint_sha256,
        "dataset": dataset,
        "dataset_verification_receipt": dataset_receipt,
        "toolkit": toolkit,
        "prediction_bundle": predictions,
        "prediction_archive": archive,
        "prior_failed_protocol_evidence": {
            "rerun_permitted": False,
            "invocation_marker": old_marker,
            "failure_receipt": old_failure,
        },
        "repository": repository,
        "prediction_and_selection_label_access_count": 0,
        "evaluator_only_label_paths_accessed": [],
    }
    return source_protocol, audit, errors


def create_preflight_receipt(
    *,
    protocol_path: Path,
    source_protocol_path: Path,
    dataset_root: Path,
    toolkit_root: Path,
    input_root: Path,
    prediction_archive: Path,
    dataset_verification_receipt: Path,
    prior_invocation_marker: Path,
    prior_failure_receipt: Path,
    output_root: Path,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    assert_namespaced_path(output_root, repo_root, protocol, kind="official_once")
    exact_output = (
        repo_root.resolve() / protocol["evaluation"]["official_once_relative_path"]
    ).resolve()
    if output_root.resolve() != exact_output:
        raise ContractError("Official output root must equal the frozen one-shot path")

    source_protocol, source_audit, errors = _audit_source_inputs(
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
    support = audit_support_matrix(toolkit_root, protocol)
    errors.extend(support["errors"])
    orchestrator = audit_orchestrator_sources()
    commands = build_supported_commands(
        input_root,
        toolkit_root,
        output_root,
        source_protocol,
        protocol,
        python_executable,
    )
    command_manifest_sha256 = canonical_sha256(commands)
    protocol_sha256 = sha256_file(protocol_path)

    fingerprint = {
        "protocol_sha256": protocol_sha256,
        "immutable_source_commit": protocol["source_evidence"]["immutable_commit"],
        "execution_checkout_head": source_audit["repository"]["head"],
        "orchestrator_source_manifest_sha256": orchestrator["manifest_sha256"],
        "source_preflight_fingerprint_sha256": source_audit[
            "source_preflight_fingerprint_sha256"
        ],
        "dataset_val_archive_sha256": source_audit["dataset_verification_receipt"]
        .get("verified_val_archive", {})
        .get("sha256"),
        "dataset_public_input_sha256": protocol["dataset"]["public_input_sha256"],
        "toolkit_commit": protocol["toolkit"]["commit"],
        "toolkit_source_tree_sha256": protocol["toolkit"]["source_tree_sha256"],
        "toolkit_entrypoint_sha256": protocol["toolkit"]["entrypoint_sha256"],
        "prediction_archive_sha256": source_audit["prediction_archive"].get("sha256"),
        "prediction_sha256": protocol["source_evidence"]["prediction_sha256"],
        "source_artifact_sha256": protocol["source_evidence"]["source_artifact_sha256"],
        "prior_failed_protocol_evidence": protocol["source_evidence"][
            "prior_failed_protocol_evidence"
        ],
        "support_matrix_sha256": canonical_sha256(protocol["support_matrix"]),
        "official_command_manifest_sha256": command_manifest_sha256,
        "official_output_root": str(output_root.resolve()),
    }
    receipt = {
        "schema_version": "poseloop.r3.xyzibd-supported.preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "package_version": PACKAGE_VERSION,
        "created_utc": utc_now(),
        "status": "ready" if not errors else "blocked",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": protocol_sha256,
            "state": protocol["state"],
        },
        "source_audit": source_audit,
        "support_audit": support,
        "orchestrator_source_audit": orchestrator,
        "official_commands": commands,
        "official_command_manifest_sha256": command_manifest_sha256,
        "official_output_root": str(output_root.resolve()),
        "official_evaluation_executed": False,
        "label_access_audit": {
            "prediction_and_selection_label_access_count": 0,
            "evaluator_only_label_paths_accessed": [],
            "official_evaluator_label_access_started": False,
        },
        "unavailable_metrics": {
            "official_bop19_average_recall": protocol["support_matrix"][
                "official_bop19_average_recall"
            ]
        },
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "errors": errors,
    }
    return receipt


def freeze_input_lock(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if preflight.get("status") != "ready":
        raise ContractError("Cannot freeze blocked supported-metric preflight")
    if preflight.get("official_evaluation_executed") is not False:
        raise ContractError("Preflight unexpectedly entered official evaluation")
    lock = dict(preflight)
    lock["schema_version"] = "poseloop.r3.xyzibd-supported.input-lock.v1"
    lock["state"] = "frozen_before_supported_official_evaluation"
    lock["frozen_utc"] = utc_now()
    return lock


def validate_input_lock(
    input_lock_path: Path,
    current_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    lock = read_json(input_lock_path)
    if not isinstance(lock, dict):
        raise ContractError("Supported input lock must be a JSON object")
    if lock.get("schema_version") != "poseloop.r3.xyzibd-supported.input-lock.v1":
        raise ContractError("Supported input lock schema is invalid")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Supported input lock protocol ID mismatch")
    if lock.get("state") != "frozen_before_supported_official_evaluation":
        raise ContractError("Supported input lock state is invalid")
    if current_preflight.get("status") != "ready":
        raise ContractError("Current supported-metric preflight is blocked")
    if lock.get("fingerprint_sha256") != current_preflight.get("fingerprint_sha256"):
        raise ContractError("Current inputs or commands differ from the supported input lock")
    return lock


def create_label_access_authorization(
    *,
    input_lock_path: Path,
    approved_by: str,
    authorization_reference: str,
) -> dict[str, Any]:
    lock = read_json(input_lock_path)
    if not isinstance(lock, dict) or lock.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("Cannot authorize an input lock for another protocol")
    if lock.get("schema_version") != "poseloop.r3.xyzibd-supported.input-lock.v1":
        raise ContractError("Cannot authorize an invalid supported input lock")
    if not approved_by.strip() or not authorization_reference.strip():
        raise ContractError("Authorization identity and reference must be non-empty")
    return {
        "schema_version": "poseloop.r3.xyzibd-supported.label-access.v1",
        "protocol_id": PROTOCOL_ID,
        "approved": True,
        "scope": "four_supported_official_bop_commands_evaluator_only",
        "approved_by": approved_by,
        "approved_at_utc": utc_now(),
        "authorization_reference": authorization_reference,
        "input_lock_sha256": sha256_file(input_lock_path),
        "official_command_manifest_sha256": lock[
            "official_command_manifest_sha256"
        ],
        "maximum_evaluate_invocations": 1,
        "authorized_command_count": 4,
        "source_protocol_rerun_permitted": False,
        "prediction_and_selection_label_access_count_before_evaluate": 0,
    }


def validate_label_access_authorization(
    authorization_path: Path,
    input_lock_path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(authorization_path)
    required = {
        "schema_version",
        "protocol_id",
        "approved",
        "scope",
        "approved_by",
        "approved_at_utc",
        "authorization_reference",
        "input_lock_sha256",
        "official_command_manifest_sha256",
        "maximum_evaluate_invocations",
        "authorized_command_count",
        "source_protocol_rerun_permitted",
        "prediction_and_selection_label_access_count_before_evaluate",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ContractError("Supported label-access authorization fields differ from schema")
    if value["schema_version"] != "poseloop.r3.xyzibd-supported.label-access.v1":
        raise ContractError("Supported label-access authorization schema is invalid")
    if value["protocol_id"] != PROTOCOL_ID or value["approved"] is not True:
        raise ContractError("New supported protocol is not explicitly authorized")
    if value["scope"] != "four_supported_official_bop_commands_evaluator_only":
        raise ContractError("Label access scope is broader or narrower than the frozen commands")
    if value["input_lock_sha256"] != sha256_file(input_lock_path):
        raise ContractError("Authorization does not match the supported input lock")
    if value["official_command_manifest_sha256"] != lock[
        "official_command_manifest_sha256"
    ]:
        raise ContractError("Authorization does not match the frozen command manifest")
    if value["maximum_evaluate_invocations"] != 1 or value["authorized_command_count"] != 4:
        raise ContractError("Authorization is not restricted to one four-command invocation")
    if value["source_protocol_rerun_permitted"] is not False:
        raise ContractError("Authorization must not permit the source protocol rerun")
    if value["prediction_and_selection_label_access_count_before_evaluate"] != 0:
        raise ContractError("Prediction/selection label access must remain zero")
    for key in ("approved_by", "approved_at_utc", "authorization_reference"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ContractError(f"Authorization field is empty: {key}")
    return value


def create_final_prescore_receipt(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_label_access_authorization(
        authorization_path, input_lock_path, lock
    )
    if output_root.exists():
        raise ContractError("Official output root exists before the authorized evaluate call")
    label_audit = current_preflight.get("label_access_audit", {})
    if label_audit.get("prediction_and_selection_label_access_count") != 0:
        raise ContractError("Prediction/selection label access is not zero at final pre-score")
    if label_audit.get("evaluator_only_label_paths_accessed") != []:
        raise ContractError("Evaluator-only labels were accessed before final pre-score")
    if label_audit.get("official_evaluator_label_access_started") is not False:
        raise ContractError("Official evaluator entered before final pre-score")
    return {
        "schema_version": "poseloop.r3.xyzibd-supported.final-prescore.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "status": "ready",
        "preflight_fingerprint_sha256": current_preflight["fingerprint_sha256"],
        "orchestrator_source_manifest_sha256": current_preflight[
            "orchestrator_source_audit"
        ]["manifest_sha256"],
        "official_command_manifest_sha256": current_preflight[
            "official_command_manifest_sha256"
        ],
        "input_lock": {
            "path": str(input_lock_path.resolve()),
            "sha256": sha256_file(input_lock_path),
        },
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
            "approved_by": authorization["approved_by"],
            "authorization_reference": authorization["authorization_reference"],
        },
        "official_output_root": str(output_root.resolve()),
        "official_output_root_absent": True,
        "authorized_command_count": len(current_preflight["official_commands"]),
        "command_ids": [
            command["command_id"] for command in current_preflight["official_commands"]
        ],
        "official_bop19_entrypoint_command_count": current_preflight["support_audit"][
            "official_bop19_entrypoint_command_count"
        ],
        "unavailable_metrics": current_preflight["unavailable_metrics"],
        "label_access_audit": label_audit,
        "official_evaluation_executed": False,
        "maximum_evaluate_invocations": 1,
        "source_protocol_rerun_permitted": False,
    }


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
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


def _load_scores(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for command in commands:
        path = Path(str(command["expected_score"]))
        if not path.is_file():
            raise ContractError(f"Official evaluator did not produce expected score: {path}")
        value = read_json(path)
        if not isinstance(value, dict):
            raise ContractError(f"Official score file is not a JSON object: {path}")
        aggregate = value.get(command["expected_score_key"])
        if not isinstance(aggregate, (int, float)) or not math.isfinite(float(aggregate)):
            raise ContractError(
                f"Official score lacks finite {command['expected_score_key']}: {path}"
            )
        scores[command["command_id"]] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "scores": value,
        }
    return scores


def _output_manifest(output_root: Path) -> list[dict[str, Any]]:
    excluded = {"sealed_receipt.json"}
    rows = []
    for path in sorted(output_root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _metrics(scores: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    single = float(
        scores["official_bop24_pose_map:single_view"]["scores"]["bop24_mAP"]
    )
    multi = float(
        scores["official_bop24_pose_map:multi_view"]["scores"]["bop24_mAP"]
    )
    bbox = float(
        scores["official_bop22_bbox_ap:shared_predicted_input"]["scores"]["AP"]
    )
    segm = float(
        scores["official_bop22_segm_ap:shared_predicted_input"]["scores"]["AP"]
    )
    return {
        "official_bop19_average_recall": {
            "status": "unavailable",
            "value": None,
            "reason": protocol["support_matrix"]["official_bop19_average_recall"][
                "reason"
            ],
        },
        "official_bop24_pose_map": {
            "status": "available",
            "single_view": single,
            "multi_view": multi,
            "multi_minus_single": multi - single,
        },
        "official_bop22_bbox_ap": {
            "status": "available",
            "shared_predicted_input": bbox,
        },
        "official_bop22_segm_ap": {
            "status": "available",
            "shared_predicted_input": segm,
        },
    }


def execute_official_evaluation(
    *,
    current_preflight: Mapping[str, Any],
    input_lock_path: Path,
    authorization_path: Path,
    dataset_root: Path,
    output_root: Path,
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    lock = validate_input_lock(input_lock_path, current_preflight)
    authorization = validate_label_access_authorization(
        authorization_path, input_lock_path, lock
    )
    assert_namespaced_path(output_root, repo_root, protocol, kind="official_once")
    exact_output = (
        repo_root.resolve() / protocol["evaluation"]["official_once_relative_path"]
    ).resolve()
    if output_root.resolve() != exact_output:
        raise ContractError("Official output root differs from frozen one-shot path")
    if output_root.exists():
        raise ContractError(
            "Supported official output root already exists; rerun is permanently refused"
        )

    commands = list(current_preflight["official_commands"])
    if canonical_sha256(commands) != current_preflight["official_command_manifest_sha256"]:
        raise ContractError("Current official command manifest is internally inconsistent")
    if [command["command_id"] for command in commands] != protocol["evaluation"][
        "command_order"
    ]:
        raise ContractError("Official command order differs from protocol")
    forbidden = protocol["support_matrix"]["official_bop19_average_recall"]["entrypoint"]
    if any(forbidden in argument for command in commands for argument in command["argv"]):
        raise ContractError("Forbidden BOP19 entrypoint reached execution boundary")

    output_root.mkdir(parents=True, exist_ok=False)
    started = {
        "schema_version": "poseloop.r3.xyzibd-supported.invocation-started.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "evaluate_invocation_count": 1,
        "authorized_command_count": len(commands),
        "official_command_manifest_sha256": current_preflight[
            "official_command_manifest_sha256"
        ],
        "input_lock_sha256": sha256_file(input_lock_path),
        "authorization_sha256": sha256_file(authorization_path),
        "source_protocol_rerun_permitted": False,
        "prediction_and_selection_label_access_count": 0,
        "official_evaluator_label_access_started": True,
    }
    started_path = output_root / "invocation_started.json"
    _write_exclusive_json(started_path, started)

    env = os.environ.copy()
    env["BOP_PATH"] = str(dataset_root.resolve().parent)
    input_root = Path(current_preflight["source_audit"]["prediction_bundle"]["root"])
    env["BOP_RESULTS_PATH"] = str(input_root.resolve())
    env["BOP_EVAL_PATH"] = str((output_root / "eval").resolve())
    python_command = str(commands[0]["argv"][0])
    python_path = shutil.which(python_command)
    if python_path is None and Path(python_command).is_file():
        python_path = str(Path(python_command).resolve())
    if python_path is not None:
        env["PATH"] = str(Path(python_path).parent) + os.pathsep + env.get("PATH", "")

    executions: list[dict[str, Any]] = []
    sealed_path = output_root / "sealed_receipt.json"
    try:
        for index, command in enumerate(commands, start=1):
            log_path = output_root / "logs" / f"{index:02d}_{command['metric']}_{command['variant']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                log.write(f"argv={json.dumps(command['argv'])}\n")
                log.flush()
                completed = subprocess.run(
                    list(command["argv"]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                )
                log.write(f"\nexit_code={completed.returncode}\n")
            execution = {
                "index": index,
                "command_id": command["command_id"],
                "metric": command["metric"],
                "variant": command["variant"],
                "exit_code": completed.returncode,
                "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
            }
            executions.append(execution)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Official evaluator failed at command {index}/4: {command['command_id']}"
                )

        scores = _load_scores(commands)
        metrics = _metrics(scores, protocol)
        result = {
            "schema_version": "poseloop.r3.xyzibd-supported.official-scores.v1",
            "protocol_id": PROTOCOL_ID,
            "created_utc": utc_now(),
            "status": "complete",
            "evaluate_invocation_count": 1,
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
            "official_evaluation_executed": True,
            "source_protocol_rerun_permitted": False,
            "prediction_and_selection_label_access_count": 0,
            "executions": executions,
            "raw_official_scores": scores,
            "metrics": metrics,
            "claim_boundary": protocol["claim_boundary"],
        }
        result_path = output_root / "official_scores.json"
        write_json_atomic(result_path, result)
        manifest = _output_manifest(output_root)
        sealed = {
            "schema_version": "poseloop.r3.xyzibd-supported.sealed-receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "created_utc": utc_now(),
            "status": "complete",
            "evaluate_invocation_count": 1,
            "authorized_command_count": 4,
            "completed_command_count": len(executions),
            "all_exit_codes": [row["exit_code"] for row in executions],
            "official_scores_path": str(result_path.resolve()),
            "official_scores_sha256": sha256_file(result_path),
            "input_lock_sha256": sha256_file(input_lock_path),
            "authorization_sha256": sha256_file(authorization_path),
            "invocation_started_sha256": sha256_file(started_path),
            "output_manifest": manifest,
            "output_manifest_sha256": canonical_sha256(manifest),
            "rerun_permitted": False,
        }
        write_json_atomic(sealed_path, sealed)
        return result
    except BaseException as exc:
        manifest = _output_manifest(output_root)
        failure = {
            "schema_version": "poseloop.r3.xyzibd-supported.sealed-receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "created_utc": utc_now(),
            "status": "failed_after_evaluate_started",
            "evaluate_invocation_count": 1,
            "authorized_command_count": 4,
            "completed_command_count": len(executions),
            "executions": executions,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "input_lock_sha256": sha256_file(input_lock_path),
            "authorization_sha256": sha256_file(authorization_path),
            "invocation_started_sha256": sha256_file(started_path),
            "output_manifest": manifest,
            "output_manifest_sha256": canonical_sha256(manifest),
            "rerun_permitted": False,
        }
        write_json_atomic(sealed_path, failure)
        raise
