from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1 import (
    DEPLOYMENT_REQUEST_SCHEMA,
    RENDER_MANIFEST_SCHEMA,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.adapter import (
    Candidate,
    OfficialCnosAdapter,
    rank_candidates,
    reduced_chunk_size,
    store_raw_cosine,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.cli import main
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts import (
    BOUNDARY_ZERO,
    DEPLOYMENT_BOUNDARY_ZERO,
    PINNED_CNOS_COMMIT,
    PINNED_CNOS_TREE,
    freeze_deployment,
    validate_deployment,
    validate_route,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1.producer import (
    PlannedCrash,
    run_producer,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    read_json,
    sha256_file,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1.json"
)
PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "poseloop_pose_accuracy_recovery_instance_proposal_v1.json"
)
ADAPTER_CONFIG_PATH = (
    ROOT
    / "pose_accuracy_recovery_prep"
    / "cnos_runtime_prep_v1"
    / "cnos_adapter_config_v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _relock(value: dict[str, Any], field: str) -> None:
    unlocked = copy.deepcopy(value)
    unlocked.pop(field, None)
    value[field] = canonical_sha256(unlocked)


def _asset(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _image(root: Path, relative: str, ordinal: int) -> dict[str, Any]:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (1440, 1080),
        color=((ordinal * 17) % 255, (ordinal * 31) % 255, (ordinal * 47) % 255),
    ).save(path, format="PNG")
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fake_git_identity(path: Path) -> dict[str, Any]:
    if path.name == "cnos":
        return {
            "commit": PINNED_CNOS_COMMIT,
            "tree": PINNED_CNOS_TREE,
            "clean": True,
        }
    if path.name == "dinov2":
        return {"commit": "d" * 40, "tree": "e" * 40, "clean": True}
    if path.name == "poseloop":
        return {"commit": "a" * 40, "tree": "b" * 40, "clean": True}
    raise AssertionError(f"Unexpected checkout: {path}")


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    deployment_root = tmp_path / "deployment"
    route = _load(ROUTE_PATH)
    protocol = _load(PROTOCOL_PATH)

    implementation_checkout = deployment_root / "sources" / "poseloop"
    implementation_checkout.mkdir(parents=True)
    _asset(
        deployment_root,
        route["required_runtime_assets"]["implementation_source_archive"][
            "relative_path"
        ],
        b"fixture-approved-poseloop-implementation\n",
    )

    cnos_checkout = deployment_root / "sources" / "cnos"
    interfaces = []
    for ordinal, source_asset in enumerate(route["source"]["interface_files"]):
        interfaces.append(
            _asset(
                cnos_checkout,
                source_asset["relative_path"],
                f"fixture-cnos-interface-{ordinal}\n".encode(),
            )
        )
    route["source"]["interface_files"] = interfaces
    route["source"]["archive"] = _asset(
        deployment_root,
        route["source"]["archive"]["relative_path"],
        b"fixture-pinned-cnos-archive\n",
    )
    dinov2_checkout = deployment_root / "sources" / "dinov2"
    dinov2_checkout.mkdir(parents=True)
    _asset(
        deployment_root,
        route["required_runtime_assets"]["dinov2_source_archive"]["relative_path"],
        b"fixture-dinov2-source-archive\n",
    )
    _asset(
        deployment_root,
        route["required_runtime_assets"]["fastsam_checkpoint"]["relative_path"],
        b"fixture-fastsam-checkpoint\n",
    )
    _asset(
        deployment_root,
        route["required_runtime_assets"]["dinov2_checkpoint"]["relative_path"],
        b"fixture-dinov2-checkpoint\n",
    )
    adapter_config = _asset(
        deployment_root,
        "config/cnos_adapter_config_v1.json",
        ADAPTER_CONFIG_PATH.read_bytes(),
    )
    assert {
        "bytes": adapter_config["bytes"],
        "sha256": adapter_config["sha256"],
    } == {
        "bytes": route["adapter_config"]["bytes"],
        "sha256": route["adapter_config"]["sha256"],
    }

    source_manifest = _asset(
        deployment_root,
        "inputs/manifests/source.json",
        b'{"role":"DEVELOPMENT_ONLY","source":"fixture"}\n',
    )
    workload_payload = (
        json.dumps(
            {"samples": protocol["input_lock"]["samples"]}, sort_keys=True
        )
        + "\n"
    ).encode()
    workload_manifest = _asset(
        deployment_root,
        "inputs/manifests/workload.json",
        workload_payload,
    )
    protocol["input_lock"]["workload_sha256"] = workload_manifest["sha256"]

    cad_by_object: dict[int, dict[str, Any]] = {}
    descriptors: dict[int, dict[str, Any]] = {}
    for object_id in protocol["input_lock"]["object_ids"]:
        cad_by_object[object_id] = _asset(
            deployment_root,
            f"inputs/cad/obj_{object_id:06d}.ply",
            f"fixture-cad-{object_id}\n".encode(),
        )
        descriptors[object_id] = _asset(
            deployment_root,
            f"inputs/render/descriptors/obj_{object_id:06d}.pt",
            f"fixture-render-descriptors-{object_id}\n".encode(),
        )
    protocol["input_lock"]["cad_assets"] = [
        {
            "object_id": object_id,
            "bytes": cad_by_object[object_id]["bytes"],
            "sha256": cad_by_object[object_id]["sha256"],
        }
        for object_id in protocol["input_lock"]["object_ids"]
    ]
    _relock(protocol, "protocol_lock_sha256")

    render_manifest = {
        "schema_version": RENDER_MANIFEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY_CAD_RENDER_TEMPLATES",
        "render_manifest_lock_sha256": "pending",
        "adapter_config_sha256": adapter_config["sha256"],
        "renderer": {
            "backend": "pyrender",
            "view_sampling_id": "cnos_obj_poses_level0_all",
            "view_count_per_object": 42,
            "template_level": 0,
        },
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
                        "adapter_config_sha256": adapter_config["sha256"],
                        "renderer": {
                            "backend": "pyrender",
                            "view_sampling_id": "cnos_obj_poses_level0_all",
                            "view_count_per_object": 42,
                            "template_level": 0,
                        },
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
            *route["required_runtime_assets"]["render_manifest"][
                "relative_path"
            ].split("/")
        )
    )
    write_json(render_path, render_manifest)

    input_items = []
    for ordinal, sample in enumerate(protocol["input_lock"]["samples"]):
        item_id = sample["item_id"]
        rgb = _image(deployment_root, f"inputs/rgb/{item_id}.png", ordinal)
        camera = _asset(
            deployment_root,
            f"inputs/camera/{item_id}.json",
            (
                json.dumps(
                    {
                        "item_id": item_id,
                        "intrinsics": [
                            [1000.0, 0.0, 720.0],
                            [0.0, 1000.0, 540.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        object_id = sample["object_id"]
        input_items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    name: sample[name]
                    for name in ("scene_id", "image_id", "object_id")
                },
                "rgb_path": rgb["relative_path"],
                "camera_path": camera["relative_path"],
                "cad_path": cad_by_object[object_id]["relative_path"],
                "descriptor_path": descriptors[object_id]["relative_path"],
            }
        )

    route["parent_protocol"]["protocol_lock_sha256"] = protocol[
        "protocol_lock_sha256"
    ]
    route["parent_protocol"]["protocol_file_sha256"] = canonical_sha256(protocol)
    _relock(route, "route_lock_sha256")
    request = {
        "schema_version": DEPLOYMENT_REQUEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY",
        "route_lock_sha256": route["route_lock_sha256"],
        "paths": {
            "implementation_source_archive": route["required_runtime_assets"][
                "implementation_source_archive"
            ]["relative_path"],
            "implementation_source_checkout": route["required_runtime_assets"][
                "implementation_source_checkout"
            ]["relative_path"],
            "cnos_source_archive": route["source"]["archive"]["relative_path"],
            "cnos_source_checkout": "sources/cnos",
            "fastsam_checkpoint": "models/FastSAM-x.pt",
            "dinov2_source_archive": "sources/dinov2-source.tar.gz",
            "dinov2_source_checkout": "sources/dinov2",
            "dinov2_checkpoint": "models/dinov2_vitl14_pretrain.pth",
            "adapter_config": "config/cnos_adapter_config_v1.json",
            "source_manifest": source_manifest["relative_path"],
            "workload_manifest": workload_manifest["relative_path"],
            "render_manifest": route["required_runtime_assets"]["render_manifest"][
                "relative_path"
            ],
        },
        "implementation": {
            "approved_commit": "a" * 40,
            "approved_tree": "b" * 40,
        },
        "input_items": input_items,
        "runtime": {
            "device": "cuda:0",
            "proposal_chunk_size": 16,
            "minimum_proposal_chunk_size": 1,
        },
        "boundary": dict(DEPLOYMENT_BOUNDARY_ZERO),
    }
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts.git_identity",
        _fake_git_identity,
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts.git_is_ancestor",
        lambda checkout, ancestor, descendant: (
            checkout.name == "poseloop"
            and ancestor
            == "b5eea0522321721e08d3bef067f0f53e16b52b24"
            and descendant == "a" * 40
        ),
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts.git_remote_repository",
        lambda checkout: "https://github.com/facebookresearch/dinov2",
    )
    deployment, runtime_lock = freeze_deployment(
        route,
        protocol,
        request,
        deployment_root=deployment_root,
    )
    return {
        "root": deployment_root,
        "route": route,
        "protocol": protocol,
        "request": request,
        "deployment": deployment,
        "runtime_lock": runtime_lock,
    }


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def infer(
        self,
        *,
        rgb_relative_path: str,
        descriptor_relative_path: str,
        target_object_id: int,
        proposal_chunk_size: int,
        minimum_chunk_size: int,
    ) -> tuple[list[Candidate], list[dict[str, int]]]:
        assert "depth" not in rgb_relative_path.lower()
        assert "depth" not in descriptor_relative_path.lower()
        assert target_object_id > 0
        assert (proposal_chunk_size, minimum_chunk_size) == (16, 1)
        self.calls.append(rgb_relative_path)
        lower = np.zeros((1080, 1440), dtype=bool)
        lower[100:220, 200:360] = True
        higher = np.zeros((1080, 1440), dtype=bool)
        higher[300:460, 600:790] = True
        return [
            Candidate(
                proposal_index=9,
                mask=lower,
                raw_cad_cosine=-0.4,
                cad_similarity=store_raw_cosine(-0.4),
                top5_template_cosines=(-0.2, -0.3, -0.4, -0.5, -0.6),
                top5_template_indices=(1, 3, 5, 7, 9),
                proposal_score=0.95,
                mask_stability=0.80,
            ),
            Candidate(
                proposal_index=2,
                mask=higher,
                raw_cad_cosine=-0.2,
                cad_similarity=store_raw_cosine(-0.2),
                top5_template_cosines=(0.0, -0.1, -0.2, -0.3, -0.4),
                top5_template_indices=(0, 2, 4, 6, 8),
                proposal_score=0.50,
                mask_stability=0.70,
            ),
        ], []


def test_committed_route_is_inert_and_binds_official_cnos() -> None:
    route = _load(ROUTE_PATH)
    result = validate_route(route, repository_root=ROOT)
    assert result["status"] == "valid"
    assert result["execution_ready"] is False
    assert result["weights_downloaded"] is False
    assert result["source_commit"] == PINNED_CNOS_COMMIT
    assert route["formula"]["cad_similarity_stored"] == (
        "monotonic_affine_raw_cosine_plus_one_divide_two"
    )
    assert route["boundary"]["server_shutdown_permitted"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda route: route["formula"].update(
            {"cad_similarity_stored": "clamp_raw_cosine_to_unit"}
        ),
        lambda route: route["resume_policy"].update(
            {"formula_changes_on_resume_permitted": True}
        ),
        lambda route: route["forbidden_inputs"].remove("depth"),
        lambda route: route["boundary"].update({"depth_path_open_count": 1}),
    ],
)
def test_route_rejects_formula_resume_and_access_drift(mutation) -> None:
    route = _load(ROUTE_PATH)
    mutation(route)
    _relock(route, "route_lock_sha256")
    with pytest.raises(ContractError):
        validate_route(route)


def test_raw_cosine_affine_storage_is_strict_monotonic_without_clamp() -> None:
    raw = [-0.9, -0.4, -0.1, 0.2, 0.8]
    normalized = [store_raw_cosine(value) for value in raw]
    assert normalized == pytest.approx([0.05, 0.3, 0.45, 0.6, 0.9])
    assert len(set(normalized[:3])) == 3
    assert sorted(range(len(raw)), key=lambda index: -raw[index]) == sorted(
        range(len(raw)), key=lambda index: -normalized[index]
    )
    with pytest.raises(ContractError):
        store_raw_cosine(1.000001)


def test_candidate_ranking_matches_raw_cosine_ranking_for_negative_values() -> None:
    backend = _FakeBackend()
    candidates, _ = backend.infer(
        rgb_relative_path="inputs/rgb/a.png",
        descriptor_relative_path="inputs/render/descriptors/a.pt",
        target_object_id=1,
        proposal_chunk_size=16,
        minimum_chunk_size=1,
    )
    by_normalized = [candidate.proposal_index for candidate in rank_candidates(candidates)]
    by_raw = [
        candidate.proposal_index
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -item.raw_cad_cosine,
                -item.proposal_score,
                -item.mask_stability,
                item.proposal_index,
            ),
        )
    ]
    assert by_normalized == by_raw == [2, 9]
    assert reduced_chunk_size(16, 1) == 8


def test_freeze_and_preflight_recompute_all_actual_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = validate_deployment(
        fixture["deployment"],
        fixture["route"],
        fixture["protocol"],
        fixture["runtime_lock"],
        deployment_root=fixture["root"],
    )
    assert result["execution_assets_ready"] is True
    assert result["item_count"] == 10
    assert result["object_count"] == 5
    assert result["label_access_count"] == 0
    assert fixture["deployment"]["source"]["dinov2_commit"] == "d" * 40
    assert fixture["deployment"]["source"]["dinov2_repository"] == (
        "https://github.com/facebookresearch/dinov2"
    )
    assert fixture["deployment"]["implementation"]["commit"] == "a" * 40


def test_implementation_checkout_must_descend_from_frozen_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts.git_is_ancestor",
        lambda checkout, ancestor, descendant: False,
    )
    with pytest.raises(ContractError, match="does not descend"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            fixture["request"],
            deployment_root=fixture["root"],
        )


def test_dinov2_checkout_must_keep_official_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1.contracts.git_remote_repository",
        lambda checkout: "https://github.com/example/not-dinov2",
    )
    with pytest.raises(ContractError, match="Official DINOv2 repository"):
        validate_deployment(
            fixture["deployment"],
            fixture["route"],
            fixture["protocol"],
            fixture["runtime_lock"],
            deployment_root=fixture["root"],
        )


def test_runtime_import_guard_rejects_module_outside_frozen_cnos_checkout(
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "frozen-cnos"
    frozen.mkdir()
    outside = tmp_path / "other" / "fast_sam.py"
    outside.parent.mkdir()
    outside.write_text("# wrong source\n", encoding="utf-8")
    with pytest.raises(ContractError, match="outside the frozen CNOS checkout"):
        OfficialCnosAdapter._require_module_from(
            SimpleNamespace(__file__=str(outside)), frozen, "FastSAM"
        )


def test_render_derivation_inventory_cannot_be_an_arbitrary_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    render_path = fixture["root"] / "inputs" / "render" / "render-manifest.json"
    render_manifest = read_json(render_path)
    render_manifest["objects"][0]["template_inventory_sha256"] = "f" * 64
    _relock(render_manifest, "render_manifest_lock_sha256")
    write_json(render_path, render_manifest)
    with pytest.raises(ContractError, match="not disk-bound"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            fixture["request"],
            deployment_root=fixture["root"],
        )


def test_preflight_fails_closed_on_checkpoint_source_archive_interface_descriptor_and_cad_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runtime = fixture["runtime_lock"]
    deployment = fixture["deployment"]
    root = fixture["root"]
    paths = [
        root / Path(*deployment["models"]["fastsam_checkpoint"]["relative_path"].split("/")),
        root / Path(*deployment["models"]["dinov2_checkpoint"]["relative_path"].split("/")),
        root / Path(*deployment["source"]["cnos_archive"]["relative_path"].split("/")),
        root
        / Path(
            *deployment["implementation"]["source_archive"][
                "relative_path"
            ].split("/")
        ),
        root
        / "sources"
        / "cnos"
        / Path(*fixture["route"]["source"]["interface_files"][0]["relative_path"].split("/")),
        root
        / Path(
            *runtime["data"]["items"][0]["inputs"][
                "cad_render_descriptors"
            ]["relative_path"].split("/")
        ),
        root
        / Path(
            *runtime["data"]["items"][0]["inputs"]["cad"][
                "relative_path"
            ].split("/")
        ),
    ]
    for path in paths:
        original = path.read_bytes()
        path.write_bytes(original + b"swap")
        with pytest.raises(ContractError):
            validate_deployment(
                deployment,
                fixture["route"],
                fixture["protocol"],
                runtime,
                deployment_root=root,
            )
        path.write_bytes(original)
        assert validate_deployment(
            deployment,
            fixture["route"],
            fixture["protocol"],
            runtime,
            deployment_root=root,
        )["status"] == "valid"


def test_deployment_request_hard_rejects_old_depth_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    request = copy.deepcopy(fixture["request"])
    request["input_items"][0]["rgb_path"] = "inputs/depth_component/rgb.png"
    with pytest.raises(ContractError, match="forbidden input"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            request,
            deployment_root=fixture["root"],
        )


@pytest.mark.parametrize(
    ("target", "payload"),
    [
        ("camera", {"intrinsics": [1.0] * 9, "depth_path": "inputs/range.png"}),
        ("source", {"role": "DEVELOPMENT_ONLY", "sealed_split": "hidden"}),
    ],
)
def test_preflight_rejects_forbidden_tokens_inside_json_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    payload: dict[str, Any],
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "camera":
        relative = fixture["request"]["input_items"][0]["camera_path"]
    else:
        relative = fixture["request"]["paths"]["source_manifest"]
    write_json(fixture["root"] / Path(*relative.split("/")), payload)
    with pytest.raises(ContractError, match="forbidden JSON"):
        freeze_deployment(
            fixture["route"],
            fixture["protocol"],
            fixture["request"],
            deployment_root=fixture["root"],
        )


def test_producer_planned_crash_resumes_without_overwrite_and_binds_score_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "output"
    first = _FakeBackend()
    with pytest.raises(PlannedCrash):
        run_producer(
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment=fixture["deployment"],
            runtime_lock=fixture["runtime_lock"],
            deployment_root=fixture["root"],
            output_root=output_root,
            backend=first,
            planned_crash_after_items=3,
        )
    assert len(first.calls) == 3
    assert read_json(output_root / "run-state.json")["status"] == "PLANNED_CRASH"

    second = _FakeBackend()
    bundle, receipt = run_producer(
        route=fixture["route"],
        protocol=fixture["protocol"],
        deployment=fixture["deployment"],
        runtime_lock=fixture["runtime_lock"],
        deployment_root=fixture["root"],
        output_root=output_root,
        backend=second,
    )
    assert len(second.calls) == 7
    assert len(bundle["items"]) == 10
    assert receipt["status"] == "COMPLETE"
    assert receipt["attempt_count"] == 2
    assert len(receipt["candidate_score_trace_inventory"]) == 10
    trace = read_json(output_root / "score-traces" / f"{bundle['items'][0]['item_id']}.json")
    assert trace["formula"]["clamp_permitted"] is False
    assert [row["raw_cad_cosine"] for row in trace["candidates"]] == [-0.2, -0.4]
    assert [row["normalized_cad_similarity"] for row in trace["candidates"]] == [
        0.4,
        0.3,
    ]
    assert all(len(row["official_top5_template_cosines"]) == 5 for row in trace["candidates"])
    assert all(len(row["official_top5_template_indices"]) == 5 for row in trace["candidates"])

    third = _FakeBackend()
    repeated_bundle, repeated_receipt = run_producer(
        route=fixture["route"],
        protocol=fixture["protocol"],
        deployment=fixture["deployment"],
        runtime_lock=fixture["runtime_lock"],
        deployment_root=fixture["root"],
        output_root=output_root,
        backend=third,
    )
    assert third.calls == []
    assert repeated_bundle == bundle
    assert repeated_receipt == receipt


@pytest.mark.parametrize("asset_kind", ["mask", "score_trace"])
def test_resume_rejects_completed_asset_tamper_before_next_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset_kind: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "output"
    with pytest.raises(PlannedCrash):
        run_producer(
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment=fixture["deployment"],
            runtime_lock=fixture["runtime_lock"],
            deployment_root=fixture["root"],
            output_root=output_root,
            backend=_FakeBackend(),
            planned_crash_after_items=1,
        )
    item_id = fixture["protocol"]["input_lock"]["samples"][0]["item_id"]
    receipt = read_json(output_root / "receipts" / f"{item_id}.json")
    relative = (
        receipt["item"]["selected_mask"]["relative_path"]
        if asset_kind == "mask"
        else receipt["candidate_score_trace"]["relative_path"]
    )
    path = output_root / Path(*relative.split("/"))
    path.write_bytes(path.read_bytes() + b"tamper")
    resumed = _FakeBackend()
    with pytest.raises(ContractError, match="changed"):
        run_producer(
            route=fixture["route"],
            protocol=fixture["protocol"],
            deployment=fixture["deployment"],
            runtime_lock=fixture["runtime_lock"],
            deployment_root=fixture["root"],
            output_root=output_root,
            backend=resumed,
        )
    assert resumed.calls == []


def test_cli_freeze_is_immutable_and_preflight_is_model_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    route_path = tmp_path / "route.json"
    protocol_path = tmp_path / "protocol.json"
    request_path = tmp_path / "request.json"
    write_json(route_path, fixture["route"])
    write_json(protocol_path, fixture["protocol"])
    write_json(request_path, fixture["request"])
    deployment_path = tmp_path / "locks" / "deployment.json"
    runtime_path = tmp_path / "locks" / "runtime.json"
    receipt_path = tmp_path / "locks" / "freeze-receipt.json"
    freeze_args = [
        "freeze-deployment",
        "--route",
        str(route_path),
        "--protocol",
        str(protocol_path),
        "--request",
        str(request_path),
        "--deployment-root",
        str(fixture["root"]),
        "--deployment-output",
        str(deployment_path),
        "--runtime-lock-output",
        str(runtime_path),
        "--receipt-output",
        str(receipt_path),
    ]
    assert main(freeze_args) == 0
    first_hashes = [
        sha256_file(path) for path in (deployment_path, runtime_path, receipt_path)
    ]
    assert main(freeze_args) == 0
    assert first_hashes == [
        sha256_file(path) for path in (deployment_path, runtime_path, receipt_path)
    ]
    validation_path = tmp_path / "locks" / "preflight.json"
    assert (
        main(
            [
                "preflight",
                "--route",
                str(route_path),
                "--protocol",
                str(protocol_path),
                "--deployment",
                str(deployment_path),
                "--runtime-lock",
                str(runtime_path),
                "--deployment-root",
                str(fixture["root"]),
                "--output",
                str(validation_path),
            ]
        )
        == 0
    )
    assert read_json(validation_path)["execution_assets_ready"] is True
    assert "execution_started" in capsys.readouterr().out


def test_cli_help_lists_only_local_prep_and_future_producer(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "freeze-deployment" in output
    assert "preflight" in output
    assert "run-producer" in output
    assert "evaluator" in output


def test_runtime_boundaries_remain_zero_and_exclude_legacy_inputs() -> None:
    assert BOUNDARY_ZERO == {
        "label_access_count": 0,
        "gt_path_open_count": 0,
        "evaluator_path_open_count": 0,
        "sealed_split_access_count": 0,
        "official_scorer_run": False,
        "legacy_depth_prompt_read_count": 0,
        "legacy_mask_read_count": 0,
    }
    route = _load(ROUTE_PATH)
    assert {
        "depth",
        "depth_component_bbox",
        "legacy_mask",
        "sam_vit_b",
        "sam6d_pose",
    }.issubset(route["forbidden_inputs"])
