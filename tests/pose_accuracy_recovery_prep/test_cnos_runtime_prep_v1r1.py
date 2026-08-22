from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import RENDER_MANIFEST_SCHEMA
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import contracts as v1
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts import (
    DEPLOYMENT_BOUNDARY_ZERO,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1.cli import main
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r1.contracts import (
    BOUNDARY_ZERO,
    PREDECESSOR_BLOCKER_RECEIPT_SHA256,
    PREDECESSOR_IMPLEMENTATION_COMMIT,
    WORKLOAD_BYTES,
    WORKLOAD_LINE_COUNT,
    WORKLOAD_SHA256,
    _validate_safe_runtime_bindings,
    audit_workload_jsonl,
    freeze_deployment,
    validate_deployment,
    validate_route,
    validate_workload_rows,
    v1_jsonl_compatibility,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import contracts as proposal_contracts

ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r1.json"
)
V1_ROUTE_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json"
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
ADAPTER_CONFIG_PATH = (
    ROOT
    / "pose_accuracy_recovery_prep"
    / "cnos_runtime_prep_v1"
    / "cnos_adapter_config_v1.json"
)
APPROVED_COMMIT = "a" * 40
APPROVED_TREE = "b" * 40


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _relock(value: dict[str, Any], field: str) -> None:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sized(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.seek(size - 1)
        stream.write(b"\0")
    return path


def _asset(
    root: Path,
    relative: str,
    *,
    forced_hashes: dict[Path, str],
    payload: bytes | None = None,
    size: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    path = root / Path(*relative.split("/"))
    if payload is not None:
        _write(path, payload)
    elif size is not None:
        _sized(path, size)
    else:
        raise AssertionError("asset payload or size required")
    if sha256 is not None:
        forced_hashes[path.resolve()] = sha256
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
    }


def _copy_frozen_dependencies(
    route: dict[str, Any], implementation_checkout: Path
) -> None:
    for binding in route["v1_dependency_files"]:
        relative = Path(*binding["relative_path"].split("/"))
        destination = implementation_checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def _prepare_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    route = _load(ROUTE_PATH)
    v1_route = _load(V1_ROUTE_PATH)
    protocol = _load(PROTOCOL_PATH)
    deployment_root = tmp_path / "deployment"
    implementation_checkout = deployment_root / "sources" / "poseloop"
    implementation_checkout.mkdir(parents=True)
    _copy_frozen_dependencies(route, implementation_checkout)

    forced_hashes: dict[Path, str] = {}
    _asset(
        deployment_root,
        v1_route["required_runtime_assets"]["implementation_source_archive"][
            "relative_path"
        ],
        forced_hashes=forced_hashes,
        payload=b"fixture-r1-implementation-archive\n",
    )

    cnos_checkout = deployment_root / "sources" / "cnos"
    cnos_checkout.mkdir(parents=True)
    _asset(
        deployment_root,
        v1_route["source"]["archive"]["relative_path"],
        forced_hashes=forced_hashes,
        size=v1_route["source"]["archive"]["bytes"],
        sha256=v1_route["source"]["archive"]["sha256"],
    )
    for binding in v1_route["source"]["interface_files"]:
        _asset(
            cnos_checkout,
            binding["relative_path"],
            forced_hashes=forced_hashes,
            size=binding["bytes"],
            sha256=binding["sha256"],
        )

    dinov2_checkout = deployment_root / "sources" / "dinov2"
    dinov2_checkout.mkdir(parents=True)
    _asset(
        deployment_root,
        v1_route["required_runtime_assets"]["dinov2_source_archive"][
            "relative_path"
        ],
        forced_hashes=forced_hashes,
        payload=b"fixture-official-dinov2-source\n",
    )
    _asset(
        deployment_root,
        v1_route["required_runtime_assets"]["fastsam_checkpoint"]["relative_path"],
        forced_hashes=forced_hashes,
        payload=b"fixture-fastsam-x-checkpoint\n",
    )
    _asset(
        deployment_root,
        v1_route["required_runtime_assets"]["dinov2_checkpoint"]["relative_path"],
        forced_hashes=forced_hashes,
        payload=b"fixture-dinov2-vitl14-checkpoint\n",
    )
    adapter = _asset(
        deployment_root,
        "config/cnos_adapter_config_v1.json",
        forced_hashes=forced_hashes,
        payload=ADAPTER_CONFIG_PATH.read_bytes(),
    )
    assert {
        "bytes": adapter["bytes"],
        "sha256": adapter["sha256"],
    } == {
        "bytes": v1_route["adapter_config"]["bytes"],
        "sha256": v1_route["adapter_config"]["sha256"],
    }

    source_manifest = _asset(
        deployment_root,
        "inputs/manifests/source.json",
        forced_hashes=forced_hashes,
        payload=b'{"role":"DEVELOPMENT_ONLY","source":"r1-fixture"}\n',
    )
    workload = _asset(
        deployment_root,
        "inputs/manifests/workload.json",
        forced_hashes=forced_hashes,
        payload=WORKLOAD_FIXTURE.read_bytes(),
    )
    assert workload == {
        "relative_path": "inputs/manifests/workload.json",
        "bytes": WORKLOAD_BYTES,
        "sha256": WORKLOAD_SHA256,
    }
    workload_rows = {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in WORKLOAD_FIXTURE.read_text(encoding="utf-8").splitlines()
        )
    }

    cad_by_object: dict[int, dict[str, Any]] = {}
    descriptors: dict[int, dict[str, Any]] = {}
    for lock in protocol["input_lock"]["cad_assets"]:
        object_id = lock["object_id"]
        cad_by_object[object_id] = _asset(
            deployment_root,
            f"inputs/cad/obj_{object_id:06d}.ply",
            forced_hashes=forced_hashes,
            size=lock["bytes"],
            sha256=lock["sha256"],
        )
        descriptors[object_id] = _asset(
            deployment_root,
            f"inputs/render/descriptors/obj_{object_id:06d}.pt",
            forced_hashes=forced_hashes,
            payload=f"fixture-descriptor-{object_id}\n".encode(),
        )

    renderer = {
        "backend": "pyrender",
        "view_sampling_id": "cnos_obj_poses_level0_all",
        "view_count_per_object": 42,
        "template_level": 0,
    }
    render_manifest: dict[str, Any] = {
        "schema_version": RENDER_MANIFEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY_CAD_RENDER_TEMPLATES",
        "render_manifest_lock_sha256": "pending",
        "adapter_config_sha256": adapter["sha256"],
        "renderer": renderer,
        "objects": [
            {
                "object_id": object_id,
                "cad": cad_by_object[object_id],
                "descriptor": descriptors[object_id],
                "template_inventory_sha256": canonical_sha256(
                    {
                        "object_id": object_id,
                        "cad": cad_by_object[object_id],
                        "descriptor": descriptors[object_id],
                        "adapter_config_sha256": adapter["sha256"],
                        "renderer": renderer,
                    }
                ),
            }
            for object_id in protocol["input_lock"]["object_ids"]
        ],
    }
    _relock(render_manifest, "render_manifest_lock_sha256")
    render_path = (
        deployment_root
        / Path(
            *v1_route["required_runtime_assets"]["render_manifest"][
                "relative_path"
            ].split("/")
        )
    )
    write_json(render_path, render_manifest)

    input_items = []
    for ordinal, sample in enumerate(protocol["input_lock"]["samples"]):
        item_id = sample["item_id"]
        row = workload_rows[item_id]
        rgb_relative = f"inputs/rgb/{item_id}.png"
        rgb_path = deployment_root / Path(*rgb_relative.split("/"))
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (1440, 1080),
            color=((ordinal * 23) % 255, (ordinal * 41) % 255, (ordinal * 61) % 255),
        ).save(rgb_path, format="PNG")
        forced_hashes[rgb_path.resolve()] = row["rgb_sha256"]

        camera_relative = f"inputs/camera/{item_id}.json"
        camera_path = deployment_root / Path(*camera_relative.split("/"))
        write_json(
            camera_path,
            {
                "item_id": item_id,
                "intrinsics": [
                    [1000.0, 0.0, 720.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        )
        forced_hashes[camera_path.resolve()] = row["camera_sha256"]
        assert row["model_sha256"] == cad_by_object[sample["object_id"]]["sha256"]
        input_items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    key: sample[key] for key in ("scene_id", "image_id", "object_id")
                },
                "rgb_path": rgb_relative,
                "camera_path": camera_relative,
                "cad_path": cad_by_object[sample["object_id"]]["relative_path"],
                "descriptor_path": descriptors[sample["object_id"]]["relative_path"],
            }
        )

    request = {
        "schema_version": (
            "poseloop.pose-accuracy-recovery.cnos-deployment-request.v1r1"
        ),
        "role": "DEVELOPMENT_ONLY",
        "route_lock_sha256": route["route_lock_sha256"],
        "paths": {
            "implementation_source_archive": v1_route["required_runtime_assets"][
                "implementation_source_archive"
            ]["relative_path"],
            "implementation_source_checkout": v1_route["required_runtime_assets"][
                "implementation_source_checkout"
            ]["relative_path"],
            "cnos_source_archive": v1_route["source"]["archive"]["relative_path"],
            "cnos_source_checkout": v1_route["source"]["checkout_relative_path"],
            "fastsam_checkpoint": v1_route["required_runtime_assets"][
                "fastsam_checkpoint"
            ]["relative_path"],
            "dinov2_source_archive": v1_route["required_runtime_assets"][
                "dinov2_source_archive"
            ]["relative_path"],
            "dinov2_source_checkout": v1_route["required_runtime_assets"][
                "dinov2_source_checkout"
            ]["relative_path"],
            "dinov2_checkpoint": v1_route["required_runtime_assets"][
                "dinov2_checkpoint"
            ]["relative_path"],
            "adapter_config": "config/cnos_adapter_config_v1.json",
            "source_manifest": source_manifest["relative_path"],
            "workload_manifest": workload["relative_path"],
            "render_manifest": v1_route["required_runtime_assets"]["render_manifest"][
                "relative_path"
            ],
        },
        "implementation": {
            "approved_commit": APPROVED_COMMIT,
            "approved_tree": APPROVED_TREE,
        },
        "input_items": input_items,
        "runtime": {
            "device": "cuda:0",
            "proposal_chunk_size": 16,
            "minimum_proposal_chunk_size": 1,
        },
        "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
    }

    real_sha256_file = sha256_file

    def fake_sha256_file(path: Path) -> str:
        candidate = Path(path).resolve()
        return forced_hashes.get(candidate, real_sha256_file(candidate))

    def fake_git_identity(path: Path) -> dict[str, Any]:
        if path.name == "cnos":
            return {
                "commit": v1.PINNED_CNOS_COMMIT,
                "tree": v1.PINNED_CNOS_TREE,
                "clean": True,
            }
        if path.name == "dinov2":
            return {"commit": "d" * 40, "tree": "e" * 40, "clean": True}
        if path.name == "poseloop":
            return {
                "commit": APPROVED_COMMIT,
                "tree": APPROVED_TREE,
                "clean": True,
            }
        raise AssertionError(f"Unexpected checkout: {path}")

    monkeypatch.setattr(v1, "sha256_file", fake_sha256_file)
    monkeypatch.setattr(proposal_contracts, "sha256_file", fake_sha256_file)
    monkeypatch.setattr(v1, "git_identity", fake_git_identity)
    monkeypatch.setattr(
        v1,
        "git_is_ancestor",
        lambda checkout, ancestor, descendant: (
            checkout.name == "poseloop"
            and ancestor in {v1.PINNED_BASE_COMMIT, PREDECESSOR_IMPLEMENTATION_COMMIT}
            and descendant == APPROVED_COMMIT
        ),
    )
    monkeypatch.setattr(
        v1,
        "git_remote_repository",
        lambda checkout: v1.PINNED_DINOV2_REPOSITORY,
    )
    return {
        "root": deployment_root,
        "implementation": implementation_checkout,
        "route": route,
        "protocol": protocol,
        "request": request,
        "workload_rows": workload_rows,
    }


def test_route_and_exact_workload_fixture_are_frozen() -> None:
    route = _load(ROUTE_PATH)
    result = validate_route(route, repository_root=ROOT)
    payload = WORKLOAD_FIXTURE.read_bytes()
    assert result["status"] == "valid"
    assert result["execution_ready"] is False
    assert result["weights_downloaded"] is False
    assert len(payload) == WORKLOAD_BYTES
    assert sha256_file(WORKLOAD_FIXTURE) == WORKLOAD_SHA256
    assert payload.count(b"\n") == WORKLOAD_LINE_COUNT
    assert b"\r" not in payload


def test_jsonl_audit_is_linewise_and_declared_only() -> None:
    protocol = _load(PROTOCOL_PATH)
    result = audit_workload_jsonl(WORKLOAD_FIXTURE, protocol)
    declared = result.audit["declared_provenance"]
    assert len(result.rows) == 10
    assert declared["status"] == "DECLARED_BUT_NOT_OPENED"
    assert declared["resolved_path_count"] == 0
    assert declared["asset_open_count"] == 0
    assert declared["asset_copy_count"] == 0
    assert declared["runtime_input_inclusion_count"] == 0
    assert all("depth_relative_path" not in binding for binding in result.audit.values())


def test_hash_view_removes_extra_data_without_rewriting_v1() -> None:
    protocol = _load(PROTOCOL_PATH)
    result = audit_workload_jsonl(WORKLOAD_FIXTURE, protocol)
    original_v1 = v1._load_object
    original_runtime = proposal_contracts.read_json
    with v1_jsonl_compatibility(WORKLOAD_FIXTURE, result):
        assert v1._load_object(WORKLOAD_FIXTURE, "workload manifest")["row_count"] == 10
        assert proposal_contracts.read_json(WORKLOAD_FIXTURE)["workload_sha256"] == (
            WORKLOAD_SHA256
        )
    assert v1._load_object is original_v1
    assert proposal_contracts.read_json is original_runtime


def test_jsonl_sha_line_count_and_sample_order_fail_closed(tmp_path: Path) -> None:
    protocol = _load(PROTOCOL_PATH)
    payload = WORKLOAD_FIXTURE.read_bytes()
    wrong_count = tmp_path / "wrong-count.jsonl"
    wrong_count.write_bytes(payload + b"{}\n")
    with pytest.raises(ContractError, match="bytes/SHA"):
        audit_workload_jsonl(wrong_count, protocol)
    rows = [json.loads(line) for line in payload.splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ContractError, match="sample/order mismatch"):
        validate_workload_rows(rows, protocol)


def test_freeze_emits_reviewable_receipt_and_never_opens_declared_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_deployment(tmp_path, monkeypatch)
    root = fixture["root"]
    first_row = next(iter(fixture["workload_rows"].values()))
    declared_depth = root / Path(*first_row["depth_relative_path"].split("/"))
    _write(declared_depth, b"must-never-open\n")
    guarded = declared_depth.absolute()
    original_open = Path.open

    def guard_declared_asset(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.absolute() == guarded:
            raise AssertionError("declared-only depth asset was opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_declared_asset)
    request_path = tmp_path / "request-v1r1.json"
    write_json(request_path, fixture["request"])
    deployment_path = tmp_path / "deployment-v1r1.json"
    runtime_path = tmp_path / "runtime-v1r1.json"
    receipt_path = tmp_path / "freeze-receipt-v1r1.json"
    assert (
        main(
            [
                "freeze-deployment",
                "--route",
                str(ROUTE_PATH),
                "--protocol",
                str(PROTOCOL_PATH),
                "--request",
                str(request_path),
                "--deployment-root",
                str(root),
                "--deployment-output",
                str(deployment_path),
                "--runtime-lock-output",
                str(runtime_path),
                "--receipt-output",
                str(receipt_path),
            ]
        )
        == 0
    )
    deployment = read_json(deployment_path)
    runtime_lock = read_json(runtime_path)
    receipt = read_json(receipt_path)
    assert isinstance(deployment, dict)
    assert isinstance(runtime_lock, dict)
    assert isinstance(receipt, dict)
    result = validate_deployment(
        deployment,
        fixture["route"],
        fixture["protocol"],
        runtime_lock,
        deployment_root=root,
    )
    assert result["status"] == "valid"
    assert result["runtime_input_roles"] == [
        "rgb",
        "camera",
        "cad",
        "cad_render_descriptors",
    ]
    assert receipt["predecessor_blocker_receipt_sha256"] == (
        PREDECESSOR_BLOCKER_RECEIPT_SHA256
    )
    assert receipt["execution_started"] is False
    assert receipt["model_imported"] is False
    assert receipt["boundary"] == BOUNDARY_ZERO
    assert runtime_lock["workload_provenance_audit"]["declared_provenance"][
        "status"
    ] == "DECLARED_BUT_NOT_OPENED"
    assert all(
        set(item["inputs"])
        == {"rgb", "camera", "cad", "cad_render_descriptors"}
        for item in runtime_lock["parent_runtime_lock"]["data"]["items"]
    )

    bad_request = copy.deepcopy(fixture["request"])
    bad_request["input_items"][0]["rgb_path"] = "inputs/depth/runtime.png"
    with pytest.raises(ContractError, match="forbidden"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            bad_request,
            deployment_root=root,
        )

    bad_parent = copy.deepcopy(runtime_lock["parent_runtime_lock"])
    bad_parent["data"]["items"][0]["inputs"]["depth"] = {
        "relative_path": "inputs/runtime/extra.bin",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(ContractError, match="runtime input roles changed"):
        _validate_safe_runtime_bindings(
            bad_parent,
            tuple(fixture["workload_rows"].values()),
        )

    frozen_dependency = (
        fixture["implementation"]
        / "pose_accuracy_recovery_prep"
        / "cnos_runtime_prep_v1"
        / "producer.py"
    )
    frozen_dependency.write_bytes(frozen_dependency.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="Frozen v1 dependency bytes changed"):
        validate_deployment(
            deployment,
            fixture["route"],
            fixture["protocol"],
            runtime_lock,
            deployment_root=root,
        )
