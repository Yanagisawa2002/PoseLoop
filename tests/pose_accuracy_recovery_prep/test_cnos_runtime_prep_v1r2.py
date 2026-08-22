from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import contracts as v1
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1 import contracts as r1
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2 import (
    CAMERA_PROVENANCE_MANIFEST_SCHEMA,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2.contracts import (
    CAMERA_DERIVATION_MANIFEST_PATH,
    CAMERA_PROVENANCE_MANIFEST_PATH,
    CAMERA_DERIVATION_BOUNDARY,
    R1_CAMERA_BLOCKER_CLOSEOUT_SHA256,
    SOURCE_CAMERA_BYTES,
    SOURCE_CAMERA_SHA256,
    SOURCE_PROVENANCE_BOUNDARY,
    WORKLOAD_BYTES,
    WORKLOAD_LINE_COUNT,
    WORKLOAD_SHA256,
    derive_cameras,
    freeze_deployment,
    validate_deployment,
    validate_camera_derivation,
    validate_derived_camera_value,
    validate_route,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import contracts as proposal_contracts

ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r2.json"
)
PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_instance_proposal_v1.json"
)
WORKLOAD_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "cnos_runtime_prep_v1r1"
    / "workload.jsonl"
)
SOURCE_CAMERA_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "cnos_runtime_prep_v1r2"
    / "source-camera.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _relock(value: dict[str, Any], field: str) -> None:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _camera_derivation_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "deployment"
    workload_path = root / "inputs" / "manifests" / "workload.json"
    workload_path.parent.mkdir(parents=True)
    shutil.copy2(WORKLOAD_FIXTURE, workload_path)
    route = _load(ROUTE_PATH)
    protocol = _load(PROTOCOL_PATH)
    rows = [json.loads(line) for line in WORKLOAD_FIXTURE.read_text().splitlines()]
    items: list[dict[str, Any]] = []
    for row in rows:
        item_id = row["item_id"]
        relative = f"inputs/provenance/camera/{item_id}.json"
        destination = root / Path(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_CAMERA_FIXTURE, destination)
        items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    key: row[key] for key in ("scene_id", "image_id", "object_id")
                },
                "asset": {
                    "relative_path": relative,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                },
            }
        )
    provenance: dict[str, Any] = {
        "schema_version": CAMERA_PROVENANCE_MANIFEST_SCHEMA,
        "role": "PUBLIC_HASH_ONLY_CAMERA_PROVENANCE",
        "source_workload_sha256": WORKLOAD_SHA256,
        "item_count": WORKLOAD_LINE_COUNT,
        "items": items,
        "boundary": copy.deepcopy(SOURCE_PROVENANCE_BOUNDARY),
        "camera_provenance_manifest_lock_sha256": "pending",
    }
    _relock(provenance, "camera_provenance_manifest_lock_sha256")
    provenance_path = root / Path(*CAMERA_PROVENANCE_MANIFEST_PATH.split("/"))
    write_json(provenance_path, provenance)
    manifest_path = root / Path(*CAMERA_DERIVATION_MANIFEST_PATH.split("/"))
    receipt_path = root / "receipts" / "camera-derivation-v1r2.json"
    manifest, receipt = derive_cameras(
        route,
        protocol,
        deployment_root=root,
        workload_path=workload_path,
        provenance_manifest_path=provenance_path,
        manifest_output=manifest_path,
        receipt_output=receipt_path,
    )
    return {
        "root": root,
        "route": route,
        "protocol": protocol,
        "rows": rows,
        "workload": r1.audit_workload_jsonl(workload_path, protocol),
        "workload_path": workload_path,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def test_route_source_fixture_and_r1_closeout_are_exact() -> None:
    route = _load(ROUTE_PATH)
    protocol = _load(PROTOCOL_PATH)
    result = validate_route(route, protocol, repository_root=ROOT)
    assert result["status"] == "valid"
    assert result["predecessor_camera_blocker_closeout_sha256"] == (
        R1_CAMERA_BLOCKER_CLOSEOUT_SHA256
    )
    assert len(WORKLOAD_FIXTURE.read_bytes()) == WORKLOAD_BYTES
    assert sha256_file(WORKLOAD_FIXTURE) == WORKLOAD_SHA256
    assert SOURCE_CAMERA_FIXTURE.stat().st_size == SOURCE_CAMERA_BYTES
    assert sha256_file(SOURCE_CAMERA_FIXTURE) == SOURCE_CAMERA_SHA256


def test_ten_cameras_are_deterministic_and_source_json_is_never_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _camera_derivation_fixture(tmp_path)
    source_paths = {
        (fixture["root"] / Path(*item["asset"]["relative_path"].split("/"))).resolve()
        for item in fixture["provenance"]["items"]
    }
    original_read_text = Path.read_text

    def reject_source_json_parse(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() in source_paths:
            raise AssertionError("hash-only source camera JSON was parsed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_source_json_parse)
    manifest2, receipt2 = derive_cameras(
        fixture["route"],
        fixture["protocol"],
        deployment_root=fixture["root"],
        workload_path=fixture["workload_path"],
        provenance_manifest_path=fixture["provenance_path"],
        manifest_output=fixture["manifest_path"],
        receipt_output=fixture["receipt_path"],
    )
    assert manifest2 == fixture["manifest"]
    assert receipt2 == fixture["receipt"]
    assert manifest2["item_count"] == 10
    assert manifest2["boundary"] == CAMERA_DERIVATION_BOUNDARY
    assert receipt2["boundary"]["source_camera_json_parse_count"] == 0
    assert receipt2["boundary"]["source_camera_field_access_count"] == 0
    assert receipt2["boundary"]["source_camera_runtime_inclusion_count"] == 0
    for item in manifest2["items"]:
        derived = fixture["root"] / Path(*item["derived_camera"]["relative_path"].split("/"))
        assert derived.read_bytes().endswith(b"\n")
        assert b"depth" not in derived.read_bytes().lower()
        assert item["derived_camera"]["sha256"] != item["source_camera"]["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.replace(b"\n", b"\r\n"),
        lambda payload: (
            json.dumps(
                dict(reversed(list(json.loads(payload).items()))),
                indent=2,
                sort_keys=False,
            )
            + "\n"
        ).encode(),
        lambda payload: _float_drift(payload),
    ],
    ids=["crlf", "key-order", "float-drift"],
)
def test_noncanonical_runtime_camera_bytes_fail_closed(
    tmp_path: Path, mutation: Callable[[bytes], bytes]
) -> None:
    fixture = _camera_derivation_fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest"])
    first = manifest["items"][0]
    path = fixture["root"] / Path(*first["derived_camera"]["relative_path"].split("/"))
    payload = mutation(path.read_bytes())
    path.write_bytes(payload)
    first["derived_camera"] = {
        "relative_path": first["derived_camera"]["relative_path"],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _relock(manifest, "camera_derivation_manifest_lock_sha256")
    with pytest.raises(ContractError, match="canonical bytes differ"):
        validate_camera_derivation(
            manifest,
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment_root=fixture["root"],
            workload=fixture["workload"],
        )


def _float_drift(payload: bytes) -> bytes:
    value = json.loads(payload)
    value["camera_intrinsics"][0][0] += 0.001
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_source_camera_hash_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _camera_derivation_fixture(tmp_path)
    first = fixture["provenance"]["items"][0]["asset"]
    path = fixture["root"] / Path(*first["relative_path"].split("/"))
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ContractError, match="bytes/SHA differ from disk"):
        validate_camera_derivation(
            fixture["manifest"],
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment_root=fixture["root"],
            workload=fixture["workload"],
        )


@pytest.mark.parametrize("swap_kind", ["sample", "asset-path"])
def test_source_camera_sample_and_asset_swap_fail_closed(
    tmp_path: Path, swap_kind: str
) -> None:
    fixture = _camera_derivation_fixture(tmp_path)
    provenance = copy.deepcopy(fixture["provenance"])
    first, second = provenance["items"][:2]
    if swap_kind == "sample":
        first["sample_key"], second["sample_key"] = (
            second["sample_key"],
            first["sample_key"],
        )
    else:
        first["asset"], second["asset"] = second["asset"], first["asset"]
    _relock(provenance, "camera_provenance_manifest_lock_sha256")
    write_json(fixture["provenance_path"], provenance)
    manifest = copy.deepcopy(fixture["manifest"])
    manifest["source_camera_provenance_manifest"] = {
        "relative_path": CAMERA_PROVENANCE_MANIFEST_PATH,
        "bytes": fixture["provenance_path"].stat().st_size,
        "sha256": sha256_file(fixture["provenance_path"]),
    }
    _relock(manifest, "camera_derivation_manifest_lock_sha256")
    with pytest.raises(ContractError, match="sample swap|frozen workload evidence"):
        validate_camera_derivation(
            manifest,
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment_root=fixture["root"],
            workload=fixture["workload"],
        )


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        (lambda value: value.update({"depth_scale": 0.1}), "forbidden"),
        (lambda value: value.update({"depth_path": "x"}), "forbidden"),
        (lambda value: value["derivation"].update({"extra": {"value": 1}}), "fields differ"),
        (lambda value: value.update({"extra": {"nested": 1}}), "fields differ"),
        (lambda value: value.update({"coordinate_convention": "GT camera"}), "forbidden"),
        (lambda value: value.update({"coordinate_convention": "evaluator camera"}), "forbidden"),
    ],
    ids=["depth-scale", "depth-path", "nested-extra", "top-extra", "gt-token", "evaluator-token"],
)
def test_runtime_camera_extra_and_forbidden_tokens_fail_closed(
    mutate: Callable[[dict[str, Any]], None], pattern: str
) -> None:
    row = json.loads(WORKLOAD_FIXTURE.read_text().splitlines()[0])
    fixture = _camera_derivation_fixture_value(row)
    mutate(fixture)
    with pytest.raises(ContractError, match=pattern):
        validate_derived_camera_value(fixture, row)


def _camera_derivation_fixture_value(row: dict[str, Any]) -> dict[str, Any]:
    from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r2.contracts import (
        derived_camera_bytes,
    )

    return json.loads(derived_camera_bytes(row))


@pytest.mark.parametrize("kind", ["crlf", "ordering", "float"])
def test_exact_workload_crlf_ordering_and_float_drift_fail_closed(
    tmp_path: Path, kind: str
) -> None:
    payload = WORKLOAD_FIXTURE.read_bytes()
    if kind == "crlf":
        changed = payload.replace(b"\n", b"\r\n")
    else:
        rows = [json.loads(line) for line in payload.splitlines()]
        if kind == "ordering":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[0]["camera_intrinsics"][0] += 0.001
        changed = b"\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            for row in rows
        ) + b"\n"
    path = tmp_path / "workload.jsonl"
    path.write_bytes(changed)
    with pytest.raises(ContractError, match="bytes/SHA|ten-line LF"):
        r1.audit_workload_jsonl(path, _load(PROTOCOL_PATH))


def test_exact_compatibility_hook_serializes_concurrent_users_and_restores(
    tmp_path: Path,
) -> None:
    protocol = _load(PROTOCOL_PATH)
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        shutil.copy2(WORKLOAD_FIXTURE, path)
    base = r1.audit_workload_jsonl(paths[0], protocol)
    results = []
    for digit in ("1", "2"):
        audit = copy.deepcopy(base.audit)
        audit["workload_audit_lock_sha256"] = digit * 64
        results.append(r1.WorkloadAuditResult(rows=base.rows, audit=audit))
    original_load = v1._load_object
    original_runtime = proposal_contracts.read_json
    gate = threading.Barrier(2)

    def worker(index: int) -> str:
        gate.wait(timeout=2)
        with r1.v1_jsonl_compatibility(paths[index], results[index]):
            time.sleep(0.01)
            return v1._load_object(paths[index], "workload manifest")[
                "workload_audit_lock_sha256"
            ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        observed = list(pool.map(worker, (0, 1)))
    assert observed == ["1" * 64, "2" * 64]
    assert v1._load_object is original_load
    assert proposal_contracts.read_json is original_runtime
    with pytest.raises(RuntimeError, match="planned"):
        with r1.v1_jsonl_compatibility(paths[0], results[0]):
            raise RuntimeError("planned")
    assert v1._load_object is original_load
    assert proposal_contracts.read_json is original_runtime


def _prepare_freeze_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Extend the frozen R1 fixture without modifying or reimplementing it."""
    r1_test = runpy.run_path(
        str(Path(__file__).with_name("test_cnos_runtime_prep_v1r1.py"))
    )
    fixture = r1_test["_prepare_deployment"](tmp_path, monkeypatch)
    route = _load(ROUTE_PATH)
    protocol = fixture["protocol"]
    root = fixture["root"]
    implementation = fixture["implementation"]

    for binding in route["r1_dependency_files"]:
        relative = Path(*binding["relative_path"].split("/"))
        destination = implementation / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    items: list[dict[str, Any]] = []
    for row in fixture["workload_rows"].values():
        relative = f"inputs/provenance/camera/{row['item_id']}.json"
        destination = root / Path(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_CAMERA_FIXTURE, destination)
        items.append(
            {
                "item_id": row["item_id"],
                "sample_key": {
                    key: row[key] for key in ("scene_id", "image_id", "object_id")
                },
                "asset": {
                    "relative_path": relative,
                    "bytes": SOURCE_CAMERA_BYTES,
                    "sha256": SOURCE_CAMERA_SHA256,
                },
            }
        )
    provenance: dict[str, Any] = {
        "schema_version": CAMERA_PROVENANCE_MANIFEST_SCHEMA,
        "role": "PUBLIC_HASH_ONLY_CAMERA_PROVENANCE",
        "source_workload_sha256": WORKLOAD_SHA256,
        "item_count": WORKLOAD_LINE_COUNT,
        "items": items,
        "boundary": copy.deepcopy(SOURCE_PROVENANCE_BOUNDARY),
        "camera_provenance_manifest_lock_sha256": "pending",
    }
    _relock(provenance, "camera_provenance_manifest_lock_sha256")
    provenance_path = root / Path(*CAMERA_PROVENANCE_MANIFEST_PATH.split("/"))
    write_json(provenance_path, provenance)
    manifest_path = root / Path(*CAMERA_DERIVATION_MANIFEST_PATH.split("/"))
    derive_cameras(
        route,
        protocol,
        deployment_root=root,
        workload_path=root / "inputs" / "manifests" / "workload.json",
        provenance_manifest_path=provenance_path,
        manifest_output=manifest_path,
        receipt_output=root / "receipts" / "camera-derivation-v1r2.json",
    )

    request = copy.deepcopy(fixture["request"])
    request["schema_version"] = (
        "poseloop.pose-accuracy-recovery.cnos-deployment-request.v1r2"
    )
    request["route_lock_sha256"] = route["route_lock_sha256"]
    request["paths"]["camera_provenance_manifest"] = (
        CAMERA_PROVENANCE_MANIFEST_PATH
    )
    request["paths"]["camera_derivation_manifest"] = (
        CAMERA_DERIVATION_MANIFEST_PATH
    )
    for item in request["input_items"]:
        item["camera_path"] = f"inputs/runtime-camera/{item['item_id']}.json"

    approved_commit = request["implementation"]["approved_commit"]
    allowed_ancestors = {
        v1.PINNED_BASE_COMMIT,
        r1_test["PREDECESSOR_IMPLEMENTATION_COMMIT"],
        "7e15f06bc97c3466aa482b98b807e8f1cc211e2b",
    }
    monkeypatch.setattr(
        v1,
        "git_is_ancestor",
        lambda checkout, ancestor, descendant: (
            checkout.name == "poseloop"
            and ancestor in allowed_ancestors
            and descendant == approved_commit
        ),
    )
    return {
        **fixture,
        "route": route,
        "request": request,
        "provenance": provenance,
    }


def test_freeze_binds_only_derived_cameras_and_preserves_r1_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_freeze_fixture(tmp_path, monkeypatch)
    deployment, runtime_lock = freeze_deployment(
        fixture["route"],
        fixture["protocol"],
        fixture["request"],
        deployment_root=fixture["root"],
    )
    result = validate_deployment(
        deployment,
        fixture["route"],
        fixture["protocol"],
        runtime_lock,
        deployment_root=fixture["root"],
    )
    assert result["status"] == "valid"
    assert result["item_count"] == WORKLOAD_LINE_COUNT
    assert result["runtime_camera_mode"] == "DERIVED_PUBLIC_INTRINSICS_ONLY"
    assert result["source_camera_json_parse_count"] == 0
    assert result["source_camera_runtime_inclusion_count"] == 0
    assert result["label_access_count"] == 0

    source_paths = {
        item["asset"]["relative_path"] for item in fixture["provenance"]["items"]
    }
    parent_runtime = runtime_lock["parent_runtime_lock"]
    serialized_parent = json.dumps(parent_runtime, sort_keys=True)
    assert all(path not in serialized_parent for path in source_paths)
    rows = fixture["workload_rows"]
    for item in parent_runtime["data"]["items"]:
        item_id = item["item_id"]
        inputs = item["inputs"]
        assert set(inputs) == {"rgb", "camera", "cad", "cad_render_descriptors"}
        assert inputs["rgb"]["sha256"] == rows[item_id]["rgb_sha256"]
        assert inputs["cad"]["sha256"] == rows[item_id]["model_sha256"]
        assert inputs["camera"]["relative_path"] == (
            f"inputs/runtime-camera/{item_id}.json"
        )
        assert inputs["camera"]["sha256"] != rows[item_id]["camera_sha256"]

    bad_request = copy.deepcopy(fixture["request"])
    first_id = bad_request["input_items"][0]["item_id"]
    bad_request["input_items"][0]["camera_path"] = (
        f"inputs/provenance/camera/{first_id}.json"
    )
    with pytest.raises(ContractError, match="forbidden|not the derived asset"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            bad_request,
            deployment_root=fixture["root"],
        )

    frozen_r1_dependency = (
        fixture["implementation"]
        / "pose_accuracy_recovery_prep"
        / "cnos_runtime_prep_v1r1"
        / "producer.py"
    )
    frozen_r1_dependency.write_bytes(frozen_r1_dependency.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="Frozen R1 dependency bytes changed"):
        validate_deployment(
            deployment,
            fixture["route"],
            fixture["protocol"],
            runtime_lock,
            deployment_root=fixture["root"],
        )
