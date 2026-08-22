"""Create-only, request-specific A-R5-P2 descriptor producer and validator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from pose_accuracy_recovery_prep.core import ContractError

from . import (
    COMPATIBILITY_BUNDLE_SCHEMA,
    FEATURE_DIMENSION,
    FINAL_RECEIPT_SCHEMA,
    OBJECT_IDS,
    OBJECT_SIDECAR_SCHEMA,
    PLANNED_STOP_SCHEMA,
    PROTOCOL_ID,
    VALIDATION_RECEIPT_SCHEMA,
    VIEW_COUNT,
)
from .contracts import (
    A_R5_PROTOCOL_ID,
    BOUNDARY_ZERO,
    DEFAULT_PROBES,
    RuntimeProbes,
    _asset_from_path,
    _exact,
    _self_lock,
    audit_descriptor_tensor,
    canonical_sha256,
    create_only_json,
    read_json,
    validate_recorded_module_audit,
    validate_runtime_request,
)
from .runtime import (
    DescriptorRuntime,
    compute_object_descriptors,
    load_official_runtime,
    save_tensor_create_only,
)

PLANNED_STOP_RELATIVE = "receipts/descriptor-assets-v1/planned-stop.json"
FINAL_RECEIPT_RELATIVE = "receipts/descriptor-generation-receipt.json"
COMPATIBILITY_BUNDLE_RELATIVE = "contracts/a-r5-descriptor-assets-bundle.json"


class PlannedStop(RuntimeError):
    """Expected exit boundary after a frozen completed object prefix."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(data_root: Path, relative: str) -> Path:
    return data_root.resolve() / Path(*relative.split("/"))


def _request_file_identity(request_path: Path) -> dict[str, Any]:
    if not request_path.is_file():
        raise ContractError("A-R5-P2 request file is missing")
    from .contracts import sha256_file

    return {
        "bytes": request_path.stat().st_size,
        "sha256": sha256_file(request_path),
    }


def _source_provenance(validation: Mapping[str, Any]) -> dict[str, Any]:
    request = validation["request"]
    return {
        "implementation": {
            "commit": request["implementation"]["commit"],
            "tree": request["implementation"]["tree"],
            "execution_inventory_sha256": request["implementation"][
                "execution_inventory_sha256"
            ],
        },
        "cnos": {
            "commit": validation["cnos"]["commit"],
            "tree": validation["cnos"]["tree"],
            "execution_inventory_sha256": validation["cnos"][
                "execution_inventory_sha256"
            ],
        },
        "dinov2": {
            "commit": validation["dinov2"]["commit"],
            "tree": validation["dinov2"]["tree"],
            "execution_inventory_sha256": validation["dinov2"][
                "execution_inventory_sha256"
            ],
        },
        "dinov2_vitl14_checkpoint": dict(request["model"]["dinov2_vitl14_checkpoint"]),
        "template_manifest_sha256": validation["template_manifest_asset"]["sha256"],
        "template_import_receipt_sha256": validation["template_import_receipt_asset"][
            "sha256"
        ],
    }


def _sidecar_path(data_root: Path, item: Mapping[str, Any]) -> Path:
    return _path(data_root, item["sidecar_relative_path"])


def _descriptor_path(data_root: Path, item: Mapping[str, Any]) -> Path:
    return _path(data_root, item["descriptor_relative_path"])


def _validate_sidecar(
    *,
    path: Path,
    item: Mapping[str, Any],
    validation: Mapping[str, Any],
    request_file_identity: Mapping[str, Any],
    data_root: Path,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    sidecar = _exact(
        read_json(path, "descriptor object sidecar"),
        {
            "schema_version",
            "protocol_id",
            "protocol_lock_sha256",
            "runtime_request_lock_sha256",
            "request_file",
            "object_id",
            "template_object_sha256",
            "source_provenance",
            "module_origin_audit",
            "weight_strict_load",
            "descriptor",
            "tensor",
            "boundary",
            "object_sidecar_lock_sha256",
        },
        "descriptor object sidecar",
    )
    request = validation["request"]
    object_id = item["object_id"]
    template_object = validation["template_objects"][object_id]
    if (
        sidecar["schema_version"] != OBJECT_SIDECAR_SCHEMA
        or sidecar["protocol_id"] != PROTOCOL_ID
        or sidecar["protocol_lock_sha256"]
        != validation["protocol"]["protocol_lock_sha256"]
        or sidecar["runtime_request_lock_sha256"]
        != request["runtime_request_lock_sha256"]
        or sidecar["request_file"] != dict(request_file_identity)
        or sidecar["object_id"] != object_id
        or sidecar["template_object_sha256"] != canonical_sha256(template_object)
        or sidecar["source_provenance"] != _source_provenance(validation)
        or sidecar["weight_strict_load"] is not True
        or sidecar["boundary"] != BOUNDARY_ZERO
    ):
        raise ContractError(
            "Descriptor sidecar is cross-request or provenance-mismatched"
        )
    validate_recorded_module_audit(
        sidecar["module_origin_audit"],
        cnos_files=validation["cnos_files"],
        dinov2_files=validation["dinov2_files"],
    )
    descriptor_path = _descriptor_path(data_root, item)
    audit = audit_descriptor_tensor(descriptor_path, torch_module=torch_module)
    descriptor = _exact(
        sidecar["descriptor"],
        {"role", "relative_path", "bytes", "sha256"},
        "sidecar descriptor",
    )
    if descriptor != {
        "role": "cad_template_descriptors",
        "relative_path": item["descriptor_relative_path"],
        "bytes": audit["bytes"],
        "sha256": audit["sha256"],
    } or sidecar["tensor"] != {
        "shape": [VIEW_COUNT, FEATURE_DIMENSION],
        "dtype": "float32",
        "finite": True,
    }:
        raise ContractError("Descriptor sidecar tensor bytes/schema changed")
    _self_lock(sidecar, "object_sidecar_lock_sha256", "descriptor object sidecar")
    return sidecar


def _completed_sidecars(
    *,
    validation: Mapping[str, Any],
    request_file_identity: Mapping[str, Any],
    data_root: Path,
    torch_module: Any | None = None,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    gap_seen = False
    for item in validation["catalog"]:
        tensor_path = _descriptor_path(data_root, item)
        sidecar_path = _sidecar_path(data_root, item)
        if tensor_path.exists() != sidecar_path.exists():
            raise ContractError(
                "Descriptor output exists without its request-specific sidecar"
            )
        if not tensor_path.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise ContractError(
                "Completed descriptor sidecars are not a frozen object prefix"
            )
        completed.append(
            _validate_sidecar(
                path=sidecar_path,
                item=item,
                validation=validation,
                request_file_identity=request_file_identity,
                data_root=data_root,
                torch_module=torch_module,
            )
        )
    return completed


def _planned_stop_value(
    *,
    validation: Mapping[str, Any],
    request_file_identity: Mapping[str, Any],
    completed: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": PLANNED_STOP_SCHEMA,
        "status": "PLANNED_STOP",
        "protocol_lock_sha256": validation["protocol"]["protocol_lock_sha256"],
        "runtime_request_lock_sha256": validation["request"][
            "runtime_request_lock_sha256"
        ],
        "request_file": dict(request_file_identity),
        "completed_object_ids": [sidecar["object_id"] for sidecar in completed],
        "completed_sidecars": [
            _asset_from_path(
                _sidecar_path(data_root, validation["catalog"][index]),
                data_root=data_root,
                role="descriptor_object_sidecar",
                probes=DEFAULT_PROBES,
            )
            for index in range(len(completed))
        ],
        "boundary": dict(BOUNDARY_ZERO),
        "planned_stop_lock_sha256": "pending",
    }
    value["planned_stop_lock_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "planned_stop_lock_sha256"}
    )
    return value


def _validate_planned_stop(
    *,
    path: Path,
    validation: Mapping[str, Any],
    request_file_identity: Mapping[str, Any],
    completed: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    observed = read_json(path, "planned stop receipt")
    expected = _planned_stop_value(
        validation=validation,
        request_file_identity=request_file_identity,
        completed=completed,
        data_root=data_root,
    )
    if observed != expected:
        raise ContractError("Resume planned-stop receipt is cross-request or tampered")
    return observed


def _final_receipt_value(
    *,
    validation: Mapping[str, Any],
    request_file_identity: Mapping[str, Any],
    completed: list[dict[str, Any]],
    data_root: Path,
    created_at: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": FINAL_RECEIPT_SCHEMA,
        "status": "PASS_EXACT_5_OBJECTS_X_42_X_1024_FLOAT32",
        "created_at": created_at,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": validation["protocol"]["protocol_lock_sha256"],
        "runtime_request_lock_sha256": validation["request"][
            "runtime_request_lock_sha256"
        ],
        "request_file": dict(request_file_identity),
        "objects": list(OBJECT_IDS),
        "descriptor_count": len(OBJECT_IDS),
        "shape_per_object": [VIEW_COUNT, FEATURE_DIMENSION],
        "dtype": "float32",
        "finite": True,
        "source_provenance": _source_provenance(validation),
        "module_origin_audit": completed[0]["module_origin_audit"],
        "weight_strict_load": True,
        "object_sidecars": [
            _asset_from_path(
                _sidecar_path(data_root, validation["catalog"][index]),
                data_root=data_root,
                role="descriptor_object_sidecar",
                probes=DEFAULT_PROBES,
            )
            for index in range(len(OBJECT_IDS))
        ],
        "descriptors": [sidecar["descriptor"] for sidecar in completed],
        "a_r5_compatibility": {
            "descriptor_generation_receipt_role": "descriptor_generation_receipt",
            "catalog_object_ids": list(OBJECT_IDS),
            "source_template_manifest_sha256": validation["template_manifest_asset"][
                "sha256"
            ],
        },
        "boundary": dict(BOUNDARY_ZERO),
        "accuracy_claim_permitted": False,
        "final_receipt_lock_sha256": "pending",
    }
    value["final_receipt_lock_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "final_receipt_lock_sha256"}
    )
    return value


def _compatibility_bundle_value(
    *,
    validation: Mapping[str, Any],
    final_receipt_asset: Mapping[str, Any],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = []
    for request_item, sidecar in zip(validation["catalog"], completed, strict=True):
        catalog.append(
            {
                "object_id": request_item["object_id"],
                "cad": request_item["cad"],
                "descriptor": sidecar["descriptor"],
                "descriptor_metadata": {
                    "shape": [VIEW_COUNT, FEATURE_DIMENSION],
                    "dtype": "float32",
                    "source_template_manifest_sha256": validation[
                        "template_manifest_asset"
                    ]["sha256"],
                    "generation_receipt_sha256": final_receipt_asset["sha256"],
                },
            }
        )
    value: dict[str, Any] = {
        "schema_version": COMPATIBILITY_BUNDLE_SCHEMA,
        "status": "PASS_A_R5_RUNTIME_REQUEST_COMPATIBLE_DESCRIPTOR_ASSETS",
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": validation["protocol"]["protocol_lock_sha256"],
        "a_r5_protocol_id": A_R5_PROTOCOL_ID,
        "runtime_request_lock_sha256": validation["request"][
            "runtime_request_lock_sha256"
        ],
        "template_manifest": validation["template_manifest_asset"],
        "descriptor_generation_receipt": dict(final_receipt_asset),
        "catalog_snippet": catalog,
        "boundary": dict(BOUNDARY_ZERO),
        "accuracy_claim_permitted": False,
        "compatibility_bundle_lock_sha256": "pending",
    }
    value["compatibility_bundle_lock_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "compatibility_bundle_lock_sha256"
        }
    )
    return value


def run_producer(
    *,
    protocol: Mapping[str, Any],
    request: Mapping[str, Any],
    protocol_path: Path,
    request_path: Path,
    data_root: Path,
    planned_stop_after_object_count: int | None = None,
    resume: bool = False,
    probes: RuntimeProbes = DEFAULT_PROBES,
    runtime_factory: Callable[
        [Mapping[str, Any]], DescriptorRuntime
    ] = load_official_runtime,
    compute_fn: Callable[..., Any] = compute_object_descriptors,
) -> dict[str, Any]:
    """Generate five tensors once, or resume only an exact planned-stop prefix."""

    if read_json(protocol_path, "A-R5-P2 protocol") != dict(protocol):
        raise ContractError("In-memory protocol differs from protocol file")
    validation = validate_runtime_request(
        request, protocol, data_root=data_root, probes=probes
    )
    if read_json(request_path, "A-R5-P2 request") != dict(request):
        raise ContractError("In-memory request differs from request file")
    request_identity = _request_file_identity(request_path)
    final_path = _path(data_root, FINAL_RECEIPT_RELATIVE)
    bundle_path = _path(data_root, COMPATIBILITY_BUNDLE_RELATIVE)
    stop_path = _path(data_root, PLANNED_STOP_RELATIVE)
    if final_path.exists() or bundle_path.exists():
        raise ContractError("A-R5-P2 final outputs are create-only and already exist")
    completed = _completed_sidecars(
        validation=validation,
        request_file_identity=request_identity,
        data_root=data_root,
    )
    if resume:
        if planned_stop_after_object_count is not None:
            raise ContractError("Resume cannot request a second planned stop")
        if (
            not completed
            or len(completed) >= len(OBJECT_IDS)
            or not stop_path.is_file()
        ):
            raise ContractError("Resume requires a non-final exact planned-stop prefix")
        _validate_planned_stop(
            path=stop_path,
            validation=validation,
            request_file_identity=request_identity,
            completed=completed,
            data_root=data_root,
        )
    else:
        if completed or stop_path.exists():
            raise ContractError(
                "Existing descriptor prefix requires explicit valid resume"
            )
        if planned_stop_after_object_count is not None and (
            not isinstance(planned_stop_after_object_count, int)
            or isinstance(planned_stop_after_object_count, bool)
            or not 1 <= planned_stop_after_object_count < len(OBJECT_IDS)
        ):
            raise ContractError("planned stop count must be in [1,4]")

    runtime = runtime_factory(validation)
    if runtime.weight_strict_load is not True:
        raise ContractError("Descriptor runtime did not prove strict weight loading")
    validate_recorded_module_audit(
        runtime.module_origin_audit,
        cnos_files=validation["cnos_files"],
        dinov2_files=validation["dinov2_files"],
    )
    for item in validation["catalog"][len(completed) :]:
        object_id = item["object_id"]
        tensor = compute_fn(
            runtime=runtime,
            object_manifest=validation["template_objects"][object_id],
            data_root=data_root,
        )
        descriptor_path = _descriptor_path(data_root, item)
        save_tensor_create_only(descriptor_path, tensor, torch_module=runtime.torch)
        audit = audit_descriptor_tensor(descriptor_path, torch_module=runtime.torch)
        sidecar: dict[str, Any] = {
            "schema_version": OBJECT_SIDECAR_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_lock_sha256": validation["protocol"]["protocol_lock_sha256"],
            "runtime_request_lock_sha256": request["runtime_request_lock_sha256"],
            "request_file": dict(request_identity),
            "object_id": object_id,
            "template_object_sha256": canonical_sha256(
                validation["template_objects"][object_id]
            ),
            "source_provenance": _source_provenance(validation),
            "module_origin_audit": runtime.module_origin_audit,
            "weight_strict_load": True,
            "descriptor": {
                "role": "cad_template_descriptors",
                "relative_path": item["descriptor_relative_path"],
                "bytes": audit["bytes"],
                "sha256": audit["sha256"],
            },
            "tensor": {
                "shape": [VIEW_COUNT, FEATURE_DIMENSION],
                "dtype": "float32",
                "finite": True,
            },
            "boundary": dict(BOUNDARY_ZERO),
            "object_sidecar_lock_sha256": "pending",
        }
        sidecar["object_sidecar_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in sidecar.items()
                if key != "object_sidecar_lock_sha256"
            }
        )
        create_only_json(_sidecar_path(data_root, item), sidecar)
        completed.append(sidecar)
        if (
            planned_stop_after_object_count is not None
            and len(completed) == planned_stop_after_object_count
        ):
            stop = _planned_stop_value(
                validation=validation,
                request_file_identity=request_identity,
                completed=completed,
                data_root=data_root,
            )
            create_only_json(stop_path, stop)
            raise PlannedStop(
                f"Planned stop after {len(completed)} request-bound objects"
            )
    if len(completed) != len(OBJECT_IDS):
        raise ContractError("A-R5-P2 descriptor completion count changed")
    final = _final_receipt_value(
        validation=validation,
        request_file_identity=request_identity,
        completed=completed,
        data_root=data_root,
        created_at=_utc_now(),
    )
    create_only_json(final_path, final)
    final_asset = _asset_from_path(
        final_path,
        data_root=data_root,
        role="descriptor_generation_receipt",
        probes=DEFAULT_PROBES,
    )
    bundle = _compatibility_bundle_value(
        validation=validation,
        final_receipt_asset=final_asset,
        completed=completed,
    )
    create_only_json(bundle_path, bundle)
    return {
        "status": final["status"],
        "descriptor_count": len(completed),
        "final_receipt_lock_sha256": final["final_receipt_lock_sha256"],
        "compatibility_bundle_lock_sha256": bundle["compatibility_bundle_lock_sha256"],
    }


def validate_success_from_disk(
    *,
    protocol: Mapping[str, Any],
    request: Mapping[str, Any],
    protocol_path: Path,
    request_path: Path,
    data_root: Path,
    probes: RuntimeProbes = DEFAULT_PROBES,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Reopen all 210 RGBA, five tensors, source locks, sidecars, and A-R5 bundle."""

    if read_json(protocol_path, "A-R5-P2 protocol") != dict(protocol):
        raise ContractError("In-memory protocol differs from protocol file")
    if read_json(request_path, "A-R5-P2 request") != dict(request):
        raise ContractError("In-memory request differs from request file")
    validation = validate_runtime_request(
        request, protocol, data_root=data_root, probes=probes
    )
    request_identity = _request_file_identity(request_path)
    completed = _completed_sidecars(
        validation=validation,
        request_file_identity=request_identity,
        data_root=data_root,
        torch_module=torch_module,
    )
    if len(completed) != len(OBJECT_IDS):
        raise ContractError(
            "A-R5-P2 independent validation found incomplete descriptors"
        )
    final_path = _path(data_root, FINAL_RECEIPT_RELATIVE)
    final = read_json(final_path, "descriptor generation receipt")
    if set(final) != {
        "schema_version",
        "status",
        "created_at",
        "protocol_id",
        "protocol_lock_sha256",
        "runtime_request_lock_sha256",
        "request_file",
        "objects",
        "descriptor_count",
        "shape_per_object",
        "dtype",
        "finite",
        "source_provenance",
        "module_origin_audit",
        "weight_strict_load",
        "object_sidecars",
        "descriptors",
        "a_r5_compatibility",
        "boundary",
        "accuracy_claim_permitted",
        "final_receipt_lock_sha256",
    }:
        raise ContractError("A-R5-P2 final receipt fields changed")
    try:
        created_at = datetime.fromisoformat(final["created_at"])
    except (TypeError, ValueError) as exc:
        raise ContractError("A-R5-P2 final receipt time is invalid") from exc
    if created_at.tzinfo is None:
        raise ContractError("A-R5-P2 final receipt time lacks timezone")
    expected_final = _final_receipt_value(
        validation=validation,
        request_file_identity=request_identity,
        completed=completed,
        data_root=data_root,
        created_at=final["created_at"],
    )
    if final != expected_final:
        raise ContractError("A-R5-P2 final receipt differs from current disk closure")
    final_asset = _asset_from_path(
        final_path,
        data_root=data_root,
        role="descriptor_generation_receipt",
        probes=DEFAULT_PROBES,
    )
    bundle_path = _path(data_root, COMPATIBILITY_BUNDLE_RELATIVE)
    bundle = read_json(bundle_path, "A-R5 compatibility bundle")
    expected_bundle = _compatibility_bundle_value(
        validation=validation,
        final_receipt_asset=final_asset,
        completed=completed,
    )
    if bundle != expected_bundle:
        raise ContractError("A-R5 descriptor compatibility bundle changed")
    return {
        "schema_version": VALIDATION_RECEIPT_SCHEMA,
        "status": "PASS_INDEPENDENT_DISK_VALIDATION",
        "rgba_count": len(OBJECT_IDS) * VIEW_COUNT,
        "descriptor_count": len(OBJECT_IDS),
        "objects": list(OBJECT_IDS),
        "runtime_request_lock_sha256": request["runtime_request_lock_sha256"],
        "descriptor_generation_receipt": final_asset,
        "compatibility_bundle": _asset_from_path(
            bundle_path,
            data_root=data_root,
            role="a_r5_descriptor_compatibility_bundle",
            probes=DEFAULT_PROBES,
        ),
        "boundary": dict(BOUNDARY_ZERO),
        "accuracy_claim_permitted": False,
    }


__all__ = [
    "COMPATIBILITY_BUNDLE_RELATIVE",
    "FINAL_RECEIPT_RELATIVE",
    "PLANNED_STOP_RELATIVE",
    "PlannedStop",
    "run_producer",
    "validate_success_from_disk",
]
