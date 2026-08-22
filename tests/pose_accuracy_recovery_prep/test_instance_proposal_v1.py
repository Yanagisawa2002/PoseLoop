from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
    write_json,
)
from pose_accuracy_recovery_prep.instance_proposal_v1 import (
    CONTENT_GATE_SCHEMA,
    INPUT_MANIFEST_SCHEMA,
    PROPOSAL_SCHEMA,
    PROTOCOL_ID,
    RUNTIME_LOCK_SCHEMA,
)
from pose_accuracy_recovery_prep.instance_proposal_v1.cli import main
from pose_accuracy_recovery_prep.instance_proposal_v1.contracts import (
    ALLOWED_INPUT_ROLES,
    FRAME_SIZE,
    MASK_CONTRACT,
    PRIOR_NO_GO_SHA256,
    RANKING_ALGORITHM_ID,
    SELECTION_REASON,
    validate_content_gate,
    validate_proposal_bundle,
    validate_protocol,
    validate_runtime_lock,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "protocols" / "poseloop_pose_accuracy_recovery_instance_proposal_v1.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _relock(value: dict, field: str) -> None:
    unlocked = copy.deepcopy(value)
    unlocked.pop(field, None)
    value[field] = canonical_sha256(unlocked)


def _asset(root: Path, relative: str, payload: bytes) -> dict:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _image_asset(
    root: Path,
    relative: str,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (1440, 1080),
    color: int | tuple[int, int, int] = (32, 64, 96),
) -> dict:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=color).save(path, format="PNG")
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _mask_asset(root: Path, relative: str, bbox: list[int]) -> dict:
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (FRAME_SIZE["width"], FRAME_SIZE["height"]), color=0)
    ImageDraw.Draw(image).rectangle(
        [bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1], fill=255
    )
    image.save(path, format="PNG")
    pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "mask_pixels": pixels,
        "coverage": pixels / (FRAME_SIZE["width"] * FRAME_SIZE["height"]),
        "connected_components": 1,
    }


def _runtime_lock(protocol: dict, input_root: Path) -> dict:
    source_manifest = _asset(
        input_root,
        "inputs/manifests/source.json",
        b'{"role":"DEVELOPMENT_ONLY","source":"fixture"}\n',
    )
    workload_manifest = _asset(
        input_root,
        "inputs/manifests/workload.json",
        (json.dumps({"samples": protocol["input_lock"]["samples"]}) + "\n").encode(),
    )
    protocol["input_lock"]["workload_sha256"] = workload_manifest["sha256"]

    cad_assets: dict[int, dict] = {}
    descriptor_assets: dict[int, dict] = {}
    for object_id in protocol["input_lock"]["object_ids"]:
        cad = _asset(
            input_root,
            f"inputs/cad/obj_{object_id:06d}.ply",
            f"fixture-cad-{object_id}\n".encode(),
        )
        descriptor = _asset(
            input_root,
            f"inputs/descriptors/obj_{object_id:06d}.bin",
            f"fixture-cad-render-descriptor-{object_id}\n".encode(),
        )
        cad_assets[object_id] = {"object_id": object_id, **cad}
        descriptor_assets[object_id] = {"object_id": object_id, **descriptor}
    protocol["input_lock"]["cad_assets"] = [
        {
            "object_id": object_id,
            "bytes": cad_assets[object_id]["bytes"],
            "sha256": cad_assets[object_id]["sha256"],
        }
        for object_id in protocol["input_lock"]["object_ids"]
    ]
    _relock(protocol, "protocol_lock_sha256")

    items = []
    for ordinal, sample in enumerate(protocol["input_lock"]["samples"]):
        item_id = sample["item_id"]
        rgb = _image_asset(
            input_root,
            f"inputs/rgb/{item_id}.png",
            color=((ordinal * 17) % 255, (ordinal * 29) % 255, (ordinal * 43) % 255),
        )
        camera = _asset(
            input_root,
            f"inputs/camera/{item_id}.json",
            (
                json.dumps(
                    {
                        "item_id": item_id,
                        "intrinsics": [1000.0, 0.0, 720.0, 0.0, 1000.0, 540.0],
                    }
                )
                + "\n"
            ).encode(),
        )
        object_id = sample["object_id"]
        items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    name: sample[name]
                    for name in ("scene_id", "image_id", "object_id")
                },
                "inputs": {
                    "rgb": rgb,
                    "camera": camera,
                    "cad": copy.deepcopy(cad_assets[object_id]),
                    "cad_render_descriptors": copy.deepcopy(
                        descriptor_assets[object_id]
                    ),
                },
            }
        )
    data = {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "input_manifest_lock_sha256": "pending",
        "source_manifest": source_manifest,
        "workload_manifest": workload_manifest,
        "frame_size": copy.deepcopy(FRAME_SIZE),
        "sample_count": protocol["input_lock"]["sample_count"],
        "scene_ids": copy.deepcopy(protocol["input_lock"]["scene_ids"]),
        "object_ids": copy.deepcopy(protocol["input_lock"]["object_ids"]),
        "items": items,
    }
    _relock(data, "input_manifest_lock_sha256")
    value = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "runtime_lock_sha256": "pending",
        "execution_ready": True,
        "backend": {
            "family": "CAD_CONDITIONED_INSTANCE_PROPOSAL",
            "implementation_name": "fixture-cad-instance-proposals",
            "source_repository": "https://example.invalid/official/source",
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "source_archive_sha256": "3" * 64,
            "checkpoint_sha256": "4" * 64,
            "checkpoint_bytes": 123456,
            "model_config_sha256": "5" * 64,
        },
        "renderer": {
            "implementation_name": "fixture-cad-renderer",
            "source_commit": "6" * 40,
            "source_tree": "7" * 40,
            "source_archive_sha256": "8" * 64,
            "renderer_config_sha256": "9" * 64,
            "cad_render_manifest_sha256": "a" * 64,
            "descriptor_model_sha256": "b" * 64,
            "view_sampling_id": "icosphere-42-fixed-v1",
            "render_view_count_per_object": 42,
        },
        "data": data,
        "boundary": {
            "label_access_count": 0,
            "gt_path_open_count": 0,
            "evaluator_path_open_count": 0,
            "sealed_split_access_count": 0,
            "official_scorer_run": False,
            "legacy_depth_prompt_read_count": 0,
            "legacy_mask_read_count": 0,
        },
    }
    _relock(value, "runtime_lock_sha256")
    return value


def _proposal_bundle(root: Path, protocol: dict, runtime: dict) -> dict:
    runtime_items = {item["item_id"]: item for item in runtime["data"]["items"]}
    items = []
    for sample in protocol["input_lock"]["samples"]:
        item_id = sample["item_id"]
        first_bbox = [10, 20, 50, 80]
        second_bbox = [100, 120, 130, 160]
        first_mask = _mask_asset(root, f"masks/{item_id}-p7.png", first_bbox)
        second_mask = _mask_asset(root, f"masks/{item_id}-p3.png", second_bbox)
        visualizations = {
            role: _image_asset(
                root,
                f"visualizations/{role}/{item_id}.png",
                color=(32, 64, 96),
            )
            for role in ("rgb", "proposal_overview", "selected_mask", "contours")
        }
        proposals = [
            {
                "proposal_index": 7,
                "rank": 1,
                "cad_object_id": sample["object_id"],
                "cad_similarity": 0.95,
                "proposal_score": 0.70,
                "mask_stability": 0.80,
                "bbox_xyxy": first_bbox,
                "mask": first_mask,
            },
            {
                "proposal_index": 3,
                "rank": 2,
                "cad_object_id": sample["object_id"],
                "cad_similarity": 0.80,
                "proposal_score": 0.99,
                "mask_stability": 0.99,
                "bbox_xyxy": second_bbox,
                "mask": second_mask,
            },
        ]
        items.append(
            {
                "item_id": item_id,
                "sample_key": {
                    name: sample[name] for name in ("scene_id", "image_id", "object_id")
                },
                "input_hashes": {
                    role: runtime_items[item_id]["inputs"][role]["sha256"]
                    for role in ALLOWED_INPUT_ROLES
                },
                "proposals": proposals,
                "selected_proposal_index": 7,
                "selected_mask": copy.deepcopy(first_mask),
                "selection_reason": SELECTION_REASON,
                "visualizations": visualizations,
            }
        )
    value = {
        "schema_version": PROPOSAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "runtime_lock_sha256": runtime["runtime_lock_sha256"],
        "proposal_bundle_lock_sha256": "pending",
        "role": "DEVELOPMENT_ONLY_INDEPENDENT_INSTANCE_PROPOSAL",
        "input_roles": list(ALLOWED_INPUT_ROLES),
        "ranking_algorithm_id": RANKING_ALGORITHM_ID,
        "boundary": copy.deepcopy(runtime["boundary"]),
        "items": items,
    }
    _relock(value, "proposal_bundle_lock_sha256")
    return value


def _content_gate(
    root: Path, protocol: dict, proposals: dict, runtime: dict
) -> dict:
    items = []
    for sample in protocol["input_lock"]["samples"]:
        item_id = sample["item_id"]
        items.append(
            {
                "item_id": item_id,
                "verdict": "PASS",
                "single_target_instance": True,
                "includes_tray": False,
                "includes_border": False,
                "includes_multiple_objects": False,
                "reason": "Fixture panel depicts exactly one target instance.",
                "panel": _image_asset(
                    root,
                    f"panels/{item_id}.png",
                    size=(640, 360),
                    color=(60, 80, 100),
                ),
            }
        )
    value = {
        "schema_version": CONTENT_GATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": protocol["protocol_lock_sha256"],
        "proposal_bundle_lock_sha256": proposals["proposal_bundle_lock_sha256"],
        "content_gate_lock_sha256": "pending",
        "review_method": "HUMAN_VISUAL_INSPECTION",
        "machine_score_substitution_permitted": False,
        "required_pass_count": 10,
        "decision": "PASS",
        "formal_five_variant_handoff_permitted": True,
        "summary": {"pass_count": 10, "fail_count": 0},
        "items": items,
        "contact_sheet": _image_asset(
            root,
            "contact-sheet/all-items.png",
            size=(1280, 720),
            color=(50, 70, 90),
        ),
        "boundary": copy.deepcopy(runtime["boundary"]),
    }
    _relock(value, "content_gate_lock_sha256")
    return value


def test_protocol_freezes_exact_development_slice_prior_no_go_and_runtime_blocker() -> None:
    protocol = _protocol()
    result = validate_protocol(protocol)
    assert result == {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": (
            "e50e61f6fea430a81e916a648fdd1bfa4e111a28f3be15287bd571d6d046fd81"
        ),
        "sample_count": 10,
        "scene_count": 5,
        "object_count": 5,
        "runtime_ready": False,
        "auto_deploy": False,
        "prior_no_go_immutable": True,
        "accuracy_claim_permitted": False,
    }
    assert protocol["prior_no_go"]["receipt_sha256"] == PRIOR_NO_GO_SHA256
    assert protocol["remote_audit"]["status"] == "RUNTIME_NOT_READY"
    assert protocol["input_lock"]["frame_size"] == FRAME_SIZE
    assert protocol["mask_contract"] == MASK_CONTRACT


def test_protocol_hard_forbids_old_depth_bbox_path() -> None:
    protocol = _protocol()
    assert protocol["input_lock"]["allowed_input_roles"] == ALLOWED_INPUT_ROLES
    assert "depth_component_bbox" in protocol["input_lock"]["forbidden_proposal_cues"]
    assert protocol["proposal_policy"]["prompt_source"] is None
    assert protocol["proposal_policy"]["proposal_source"] == (
        "independent_rgb_instance_proposals"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"auto_deploy": True}),
        lambda value: value["prior_no_go"].update({"rerun_permitted": True}),
        lambda value: value["input_lock"]["allowed_input_roles"].append("depth"),
        lambda value: value["content_gate"].update(
            {"machine_score_substitution_permitted": True}
        ),
    ],
)
def test_protocol_rejects_deploy_prior_result_input_and_gate_drift(mutation) -> None:
    protocol = _protocol()
    mutation(protocol)
    with pytest.raises(ContractError):
        validate_protocol(protocol)


def test_complete_runtime_lock_is_required_before_execution(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    result = validate_runtime_lock(runtime, protocol, input_root=input_root)
    assert result == {
        "status": "valid",
        "execution_ready": True,
        "runtime_lock_sha256": runtime["runtime_lock_sha256"],
        "input_manifest_lock_sha256": runtime["data"][
            "input_manifest_lock_sha256"
        ],
        "backend_family": "CAD_CONDITIONED_INSTANCE_PROPOSAL",
        "cad_count": 5,
        "verified_input_assets": 42,
        "label_access_count": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["backend"].update({"checkpoint_sha256": None}),
            "checkpoint_sha256",
        ),
        (
            lambda value: value["renderer"].update({"cad_render_manifest_sha256": None}),
            "cad_render_manifest_sha256",
        ),
        (
            lambda value: value["boundary"].update(
                {"legacy_depth_prompt_read_count": 1}
            ),
            "zero-access and independent",
        ),
        (
            lambda value: value["data"].update({"sample_count": 9}),
            "sample_count",
        ),
    ],
)
def test_runtime_rejects_incomplete_locks_old_prompt_access_and_data_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    mutation(runtime)
    _relock(runtime, "runtime_lock_sha256")
    with pytest.raises(ContractError, match=message):
        validate_runtime_lock(runtime, protocol, input_root=input_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda runtime: runtime["data"]["items"][0]["inputs"]["rgb"].update(
                {"sha256": "f" * 64}
            ),
            "SHA-256 mismatch",
        ),
        (
            lambda runtime: runtime["data"]["items"][0]["inputs"].update(
                {"cad": copy.deepcopy(runtime["data"]["items"][2]["inputs"]["cad"])}
            ),
            "cad target object mismatch",
        ),
        (
            lambda runtime: runtime["data"]["items"][0]["inputs"].update(
                {
                    "cad_render_descriptors": copy.deepcopy(
                        runtime["data"]["items"][2]["inputs"][
                            "cad_render_descriptors"
                        ]
                    )
                }
            ),
            "cad_render_descriptors target object mismatch",
        ),
        (
            lambda runtime: runtime["data"]["items"][0]["sample_key"].update(
                {"image_id": 1}
            ),
            "sample key mismatch",
        ),
        (
            lambda runtime: runtime["data"]["source_manifest"].update(
                {"sha256": "d" * 64}
            ),
            "SHA-256 mismatch",
        ),
        (
            lambda runtime: runtime["data"]["workload_manifest"].update(
                {"sha256": "e" * 64}
            ),
            "SHA-256 mismatch|workload manifest SHA",
        ),
    ],
)
def test_runtime_input_manifest_fails_closed_on_asset_and_identity_swaps(
    tmp_path: Path, mutation, message: str
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    mutation(runtime)
    _relock(runtime["data"], "input_manifest_lock_sha256")
    _relock(runtime, "runtime_lock_sha256")
    with pytest.raises(ContractError, match=message):
        validate_runtime_lock(runtime, protocol, input_root=input_root)


def test_runtime_cad_bytes_and_sha_must_equal_protocol_object_lock(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    first = runtime["data"]["items"][0]["inputs"]["cad"]
    path = input_root / Path(*first["relative_path"].split("/"))
    path.write_bytes(b"different-cad-content\n")
    changed = {
        "object_id": first["object_id"],
        "relative_path": first["relative_path"],
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    for item in runtime["data"]["items"]:
        if item["sample_key"]["object_id"] == first["object_id"]:
            item["inputs"]["cad"] = copy.deepcopy(changed)
    _relock(runtime["data"], "input_manifest_lock_sha256")
    _relock(runtime, "runtime_lock_sha256")
    with pytest.raises(ContractError, match="CAD lock differs from protocol"):
        validate_runtime_lock(runtime, protocol, input_root=input_root)


def test_proposal_bundle_validates_exact_coverage_assets_and_frozen_ranking(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    result = validate_proposal_bundle(
        bundle,
        protocol,
        runtime,
        input_root=input_root,
        asset_root=proposal_root,
    )
    assert result["item_count"] == 10
    assert result["scene_count"] == 5
    assert result["object_count"] == 5
    assert result["proposal_count"] == 20
    assert result["asset_count"] == 60
    assert result["verified_assets"] is True
    assert result["input_manifest_lock_sha256"] == runtime["data"][
        "input_manifest_lock_sha256"
    ]
    assert result["legacy_depth_prompt_read_count"] == 0
    assert result["content_gate_status"] == "NOT_REVIEWED"


def test_proposal_bundle_rejects_old_depth_input_and_ranking_drift(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    root = tmp_path / "proposal"
    bundle = _proposal_bundle(root, protocol, runtime)
    bundle["input_roles"].append("depth_component_bbox")
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="identity/input/ranking"):
        validate_proposal_bundle(
            bundle, protocol, runtime, input_root=input_root, asset_root=root
        )

    bundle = _proposal_bundle(tmp_path / "ranking", protocol, runtime)
    bundle["items"][0]["proposals"].reverse()
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="frozen ranking"):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=tmp_path / "ranking",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle["items"][0]["input_hashes"].update(
            {"rgb": "f" * 64}
        ),
        lambda bundle: bundle["items"][0].update(
            {"input_hashes": copy.deepcopy(bundle["items"][2]["input_hashes"])}
        ),
        lambda bundle: bundle["items"][0]["input_hashes"].update(
            {"cad": bundle["items"][2]["input_hashes"]["cad"]}
        ),
        lambda bundle: bundle["items"][0]["input_hashes"].update(
            {
                "cad_render_descriptors": bundle["items"][2]["input_hashes"][
                    "cad_render_descriptors"
                ]
            }
        ),
    ],
)
def test_proposal_input_hashes_must_equal_runtime_item_manifest(
    tmp_path: Path, mutation
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    mutation(bundle)
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="runtime input manifest"):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=proposal_root,
        )


def test_proposal_bundle_rejects_missing_coverage_and_asset_hash_drift(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    root = tmp_path / "proposal"
    bundle = _proposal_bundle(root, protocol, runtime)
    bundle["items"].pop()
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="exact 10"):
        validate_proposal_bundle(
            bundle, protocol, runtime, input_root=input_root, asset_root=root
        )

    root = tmp_path / "hash"
    bundle = _proposal_bundle(root, protocol, runtime)
    selected = bundle["items"][0]["selected_mask"]
    (root / Path(*selected["relative_path"].split("/"))).write_bytes(b"changed")
    with pytest.raises(ContractError, match="byte count mismatch|SHA-256 mismatch"):
        validate_proposal_bundle(
            bundle, protocol, runtime, input_root=input_root, asset_root=root
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("corrupt", "not a decodable PNG mask"),
        ("rgb", "single-channel L mask"),
        ("wrong_frame", "frame mismatch"),
        ("non_binary", "strictly binary"),
    ],
)
def test_proposal_masks_must_be_real_binary_full_frame_png(
    tmp_path: Path, kind: str, message: str
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    item = bundle["items"][0]
    mask = item["proposals"][0]["mask"]
    path = proposal_root / Path(*mask["relative_path"].split("/"))
    if kind == "corrupt":
        path.write_bytes(b"not-a-png")
    elif kind == "rgb":
        Image.new("RGB", (1440, 1080), color=(0, 0, 0)).save(path, format="PNG")
    elif kind == "wrong_frame":
        image = Image.new("L", (32, 32), color=0)
        ImageDraw.Draw(image).rectangle([2, 2, 9, 9], fill=255)
        image.save(path, format="PNG")
    else:
        image = Image.new("L", (1440, 1080), color=0)
        ImageDraw.Draw(image).rectangle([10, 20, 49, 79], fill=128)
        image.save(path, format="PNG")
    mask["sha256"] = sha256_file(path)
    mask["bytes"] = path.stat().st_size
    item["selected_mask"] = copy.deepcopy(mask)
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match=message):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=proposal_root,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["proposals"][0].update(
                {"bbox_xyxy": [9, 20, 50, 80]}
            ),
            "tight half-open",
        ),
        (
            lambda item: item["proposals"][0]["mask"].update(
                {"mask_pixels": 2399}
            ),
            "mask_pixels",
        ),
        (
            lambda item: item["proposals"][0]["mask"].update(
                {"connected_components": 2}
            ),
            "connected_components",
        ),
        (
            lambda item: item["proposals"][0]["mask"].update({"coverage": 0.5}),
            "coverage",
        ),
    ],
)
def test_proposal_mask_geometry_is_recomputed(
    tmp_path: Path, mutation, message: str
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    item = bundle["items"][0]
    mutation(item)
    item["selected_mask"] = copy.deepcopy(item["proposals"][0]["mask"])
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match=message):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=proposal_root,
        )


def test_selected_mask_must_equal_rank_one_asset_and_statistics(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    bundle["items"][0]["selected_mask"]["mask_pixels"] += 1
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="selection does not bind frozen rank one"):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=proposal_root,
        )


def test_proposal_visualizations_must_be_decodable_images(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    visualization = bundle["items"][0]["visualizations"]["contours"]
    path = proposal_root / Path(*visualization["relative_path"].split("/"))
    path.write_bytes(b"not-an-image")
    visualization["sha256"] = sha256_file(path)
    visualization["bytes"] = path.stat().st_size
    _relock(bundle, "proposal_bundle_lock_sha256")
    with pytest.raises(ContractError, match="not a decodable image"):
        validate_proposal_bundle(
            bundle,
            protocol,
            runtime,
            input_root=input_root,
            asset_root=proposal_root,
        )


def test_content_gate_requires_human_10_of_10_before_handoff(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    review_root = tmp_path / "review"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    review = _content_gate(review_root, protocol, bundle, runtime)
    result = validate_content_gate(
        review,
        protocol,
        bundle,
        runtime_lock=runtime,
        input_root=input_root,
        proposal_root=proposal_root,
        review_root=review_root,
    )
    assert result["decision"] == "PASS"
    assert result["pass_count"] == 10
    assert result["formal_five_variant_handoff_permitted"] is True
    assert result["human_visual_reviewed"] is True
    assert result["machine_score_substitution_permitted"] is False


def test_single_visual_failure_yields_valid_no_go_and_forbids_handoff(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    review_root = tmp_path / "review"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    review = _content_gate(review_root, protocol, bundle, runtime)
    first = review["items"][0]
    first.update(
        {
            "verdict": "FAIL",
            "single_target_instance": False,
            "includes_tray": True,
            "reason": "Panel includes the tray and more than one object.",
        }
    )
    review["summary"] = {"pass_count": 9, "fail_count": 1}
    review["decision"] = "NO_GO"
    review["formal_five_variant_handoff_permitted"] = False
    _relock(review, "content_gate_lock_sha256")
    result = validate_content_gate(
        review,
        protocol,
        bundle,
        runtime_lock=runtime,
        input_root=input_root,
        proposal_root=proposal_root,
        review_root=review_root,
    )
    assert result["decision"] == "NO_GO"
    assert result["pass_count"] == 9
    assert result["formal_five_variant_handoff_permitted"] is False


def test_content_gate_rejects_machine_substitution_and_contradictory_verdict(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    review_root = tmp_path / "review"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    review = _content_gate(review_root, protocol, bundle, runtime)
    review["machine_score_substitution_permitted"] = True
    _relock(review, "content_gate_lock_sha256")
    with pytest.raises(ContractError, match="identity/authority"):
        validate_content_gate(
            review,
            protocol,
            bundle,
            runtime_lock=runtime,
            input_root=input_root,
            proposal_root=proposal_root,
            review_root=review_root,
        )

    review = _content_gate(tmp_path / "contradiction", protocol, bundle, runtime)
    review["items"][0]["includes_multiple_objects"] = True
    _relock(review, "content_gate_lock_sha256")
    with pytest.raises(ContractError, match="contradicts observations"):
        validate_content_gate(
            review,
            protocol,
            bundle,
            runtime_lock=runtime,
            input_root=input_root,
            proposal_root=proposal_root,
            review_root=tmp_path / "contradiction",
        )


def test_cli_validates_protocol_runtime_proposals_and_content_gate(tmp_path: Path) -> None:
    protocol = _protocol()
    input_root = tmp_path / "inputs"
    runtime = _runtime_lock(protocol, input_root)
    proposal_root = tmp_path / "proposal"
    review_root = tmp_path / "review"
    bundle = _proposal_bundle(proposal_root, protocol, runtime)
    review = _content_gate(review_root, protocol, bundle, runtime)
    protocol_path = tmp_path / "protocol.json"
    runtime_path = tmp_path / "runtime.json"
    bundle_path = tmp_path / "proposals.json"
    review_path = tmp_path / "review.json"
    write_json(protocol_path, protocol)
    write_json(runtime_path, runtime)
    write_json(bundle_path, bundle)
    write_json(review_path, review)

    assert main(["validate-protocol", "--protocol", str(PROTOCOL_PATH)]) == 0
    assert (
        main(
            [
                "validate-runtime-lock",
                "--protocol",
                str(protocol_path),
                "--runtime-lock",
                str(runtime_path),
                "--input-root",
                str(input_root),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate-proposals",
                "--protocol",
                str(protocol_path),
                "--runtime-lock",
                str(runtime_path),
                "--input-root",
                str(input_root),
                "--proposal-bundle",
                str(bundle_path),
                "--asset-root",
                str(proposal_root),
            ]
        )
        == 0
    )
    output = tmp_path / "content-validation.json"
    assert (
        main(
            [
                "validate-content-gate",
                "--protocol",
                str(protocol_path),
                "--runtime-lock",
                str(runtime_path),
                "--input-root",
                str(input_root),
                "--proposal-bundle",
                str(bundle_path),
                "--proposal-root",
                str(proposal_root),
                "--review",
                str(review_path),
                "--review-root",
                str(review_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == "PASS"
