from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from pose_accuracy_recovery_prep.c_handoff import (
    C_HANDOFF_SCHEMA,
    C_VARIANTS,
    export_c_handoff,
    load_and_validate_c_handoff,
    validate_c_handoff_manifest,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "fixtures" / "pose_accuracy_recovery_prep"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
PROTOCOL_PATH = ROOT / "protocols" / "poseloop_pose_accuracy_recovery_c_handoff_v2.json"
IMPLEMENTATION_COMMIT = "ad3affc9aed1f0ef9a12b3f0423703152e6ad481"
IMPLEMENTATION_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64
REFINER_SHA256 = "3" * 64
SCORER_SHA256 = "4" * 64


def _export(root: Path) -> tuple[dict, dict]:
    receipt = export_c_handoff(
        manifest_path=MANIFEST_PATH,
        data_root=DATA_ROOT,
        protocol_path=PROTOCOL_PATH,
        output_root=root,
        implementation_commit=IMPLEMENTATION_COMMIT,
        implementation_sha256=IMPLEMENTATION_SHA256,
        model_sha256=MODEL_SHA256,
        refiner_checkpoint_sha256=REFINER_SHA256,
        scorer_checkpoint_sha256=SCORER_SHA256,
    )
    manifest, validation = load_and_validate_c_handoff(
        root / "manifest.json", bundle_root=root
    )
    assert receipt["manifest_lock_sha256"] == validation["manifest_lock_sha256"]
    return manifest, validation


def _relock(manifest: dict) -> None:
    unlocked = dict(manifest)
    unlocked.pop("manifest_lock_sha256", None)
    manifest["manifest_lock_sha256"] = canonical_sha256(unlocked)


def _real_image_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    data_root = tmp_path / "real-data"
    shutil.copytree(DATA_ROOT, data_root)
    manifest_path = data_root / "real-manifest.json"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["input_kind"] = "DEVELOPMENT_DATA"
    sample = manifest["samples"][0]

    rgb_path = data_root / "producer/rgb/sample-000.jpg"
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(rgb_path, quality=95)
    sample["producer_inputs"]["rgb"].update(
        {
            "path": "producer/rgb/sample-000.jpg",
            "sha256": sha256_file(rgb_path),
        }
    )

    depth_path = data_root / "producer/depth/sample-000.png"
    Image.new("I;16", (4, 4), color=750).save(depth_path)
    sample["producer_inputs"]["depth"].update(
        {
            "path": "producer/depth/sample-000.png",
            "sha256": sha256_file(depth_path),
        }
    )

    masks = {
        "predicted_mask": [0, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 0, 0, 0, 0, 0],
        "depth_component_mask": [0, 0, 255, 255, 0, 255, 255, 255, 0, 255, 0, 0, 0, 0, 0, 0],
        "bbox_mask": [0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0],
        "official_known_sample_sanity": [0, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 0, 0, 0, 0, 0],
        "oracle_mask_control": [0, 0, 255, 255, 0, 255, 255, 255, 0, 255, 255, 0, 0, 0, 0, 0],
    }
    for name, pixels in masks.items():
        evaluator = name in {
            "official_known_sample_sanity",
            "oracle_mask_control",
        }
        root = "evaluator_only" if evaluator else "producer"
        relative = f"{root}/masks/{name}.png"
        path = data_root / relative
        image = Image.new("L", (4, 4))
        image.putdata(pixels)
        metadata = PngInfo()
        metadata.add_text("score", "1.0" if evaluator else "0.8")
        image.save(path, pnginfo=metadata)
        contract = (
            sample["evaluator_only"]["masks"][name]
            if evaluator
            else sample["producer_inputs"]["masks"][name]
        )
        contract.update(
            {
                "path": relative,
                "sha256": sha256_file(path),
            }
        )
    write_json(manifest_path, manifest)
    return data_root, manifest_path, manifest


def test_export_c_handoff_has_exact_five_variant_coverage_and_honest_boundary(
    tmp_path: Path,
) -> None:
    manifest, validation = _export(tmp_path / "handoff")
    assert manifest["schema_version"] == C_HANDOFF_SCHEMA
    assert manifest["protocol_id"] == "poseloop-r4c-foundationpose-runtime-prep-v1"
    assert manifest["protocol_sha256"] == (
        "d915325a5be8201a5b489c52c3c0720520bcfcebf9a3a06d7c5ec55c17efb0c9"
    )
    assert [row["mask_variant_id"] for row in manifest["mask_variants"]] == [
        row["mask_variant_id"] for row in C_VARIANTS
    ]
    assert validation["item_count"] == 5
    assert validation["base_sample_count"] == 1
    assert validation["variant_count"] == 5
    assert set(validation["per_variant_item_count"].values()) == {1}
    assert manifest["boundary"]["contains_gt_derived_control_inputs"] is True
    assert manifest["boundary"]["contains_raw_gt_paths"] is False
    assert manifest["boundary"]["gpu_c_resolves_derivation"] is False
    assert manifest["boundary"]["label_access_count_on_gpu_c"] == 0
    assert manifest["boundary"]["official_scorer_run"] is False


def test_control_masks_are_copied_to_opaque_paths_with_complete_asset_contracts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "handoff"
    manifest, _ = _export(root)
    for item in manifest["items"]:
        assert set(item["inputs"]) == {"rgb", "depth", "mask", "camera", "cad"}
        for asset in item["inputs"].values():
            assert set(asset) == {"role", "relative_path", "sha256", "bytes"}
            assert asset["bytes"] > 0
            assert (root / Path(*asset["relative_path"].split("/"))).is_file()
        assert item["frame_size"] == {"width": 4, "height": 4}
        assert item["camera_intrinsics"] == [
            [400.0, 0.0, 1.5],
            [0.0, 400.0, 1.5],
            [0.0, 0.0, 1.0],
        ]
        assert item["depth_scale"] == pytest.approx(0.001)
        provenance = item["mask_provenance"]
        assert provenance["coverage"]["nonzero_pixels"] > 0
        assert provenance["label_access_count_on_gpu_c"] == 0
    for item in manifest["items"][:2]:
        relative = item["inputs"]["mask"]["relative_path"].lower()
        assert not any(
            token in relative
            for token in ("gt", "oracle", "sanity", "evaluator", "evaluation")
        )
        assert item["mask_provenance"]["development_control"] is True
        assert item["mask_provenance"]["derivation_class"] == (
            "gt-derived-development-control"
        )
    serialized_assets = json.dumps(
        [item["inputs"] for item in manifest["items"]], sort_keys=True
    ).lower()
    assert "evaluator_only" not in serialized_assets
    assert "gt_pose" not in serialized_assets


def test_export_accepts_hash_bound_real_png_jpeg_frames_and_binary_masks(
    tmp_path: Path,
) -> None:
    data_root, manifest_path, _ = _real_image_fixture(tmp_path)
    output_root = tmp_path / "real-handoff"
    export_c_handoff(
        manifest_path=manifest_path,
        data_root=data_root,
        protocol_path=PROTOCOL_PATH,
        output_root=output_root,
        implementation_commit=IMPLEMENTATION_COMMIT,
        implementation_sha256=IMPLEMENTATION_SHA256,
        model_sha256=MODEL_SHA256,
        refiner_checkpoint_sha256=REFINER_SHA256,
        scorer_checkpoint_sha256=SCORER_SHA256,
    )
    handoff, validation = load_and_validate_c_handoff(
        output_root / "manifest.json", bundle_root=output_root
    )
    assert validation["verified_assets"] is True
    assert validation["item_count"] == 5
    assert all(
        item["frame_size"] == {"width": 4, "height": 4}
        for item in handoff["items"]
    )
    assert all(
        item["inputs"]["rgb"]["relative_path"].endswith(".jpg")
        for item in handoff["items"]
    )
    assert all(
        item["inputs"]["depth"]["relative_path"].endswith(".png")
        for item in handoff["items"]
    )
    assert all(item["inputs"]["mask"]["bytes"] > 0 for item in handoff["items"])
    assert all(
        item["mask_provenance"]["coverage"]["nonzero_pixels"] > 0
        for item in handoff["items"]
    )


def test_real_image_export_rejects_depth_size_and_nonbinary_mask_drift(
    tmp_path: Path,
) -> None:
    data_root, manifest_path, manifest = _real_image_fixture(tmp_path)
    depth_path = data_root / "producer/depth/sample-000.png"
    Image.new("I;16", (3, 4), color=750).save(depth_path)
    manifest["samples"][0]["producer_inputs"]["depth"]["sha256"] = sha256_file(
        depth_path
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ContractError, match="Depth frame size differs"):
        export_c_handoff(
            manifest_path=manifest_path,
            data_root=data_root,
            protocol_path=PROTOCOL_PATH,
            output_root=tmp_path / "bad-depth",
            implementation_commit=IMPLEMENTATION_COMMIT,
            implementation_sha256=IMPLEMENTATION_SHA256,
            model_sha256=MODEL_SHA256,
            refiner_checkpoint_sha256=REFINER_SHA256,
            scorer_checkpoint_sha256=SCORER_SHA256,
        )

    data_root, manifest_path, manifest = _real_image_fixture(tmp_path / "mask")
    mask_path = data_root / "producer/masks/predicted_mask.png"
    image = Image.new("L", (4, 4))
    image.putdata([0, 64, 255, 255] * 4)
    image.save(mask_path)
    manifest["samples"][0]["producer_inputs"]["masks"]["predicted_mask"][
        "sha256"
    ] = sha256_file(mask_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ContractError, match="binary background/foreground"):
        export_c_handoff(
            manifest_path=manifest_path,
            data_root=data_root,
            protocol_path=PROTOCOL_PATH,
            output_root=tmp_path / "bad-mask",
            implementation_commit=IMPLEMENTATION_COMMIT,
            implementation_sha256=IMPLEMENTATION_SHA256,
            model_sha256=MODEL_SHA256,
            refiner_checkpoint_sha256=REFINER_SHA256,
            scorer_checkpoint_sha256=SCORER_SHA256,
        )


def test_real_mask_requires_explicit_auditable_input_score(tmp_path: Path) -> None:
    data_root, manifest_path, manifest = _real_image_fixture(tmp_path)
    mask_path = data_root / "producer/masks/predicted_mask.png"
    image = Image.new("L", (4, 4))
    image.putdata([0, 0, 255, 255] * 4)
    image.save(mask_path)
    manifest["samples"][0]["producer_inputs"]["masks"]["predicted_mask"][
        "sha256"
    ] = sha256_file(mask_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ContractError, match="must declare a finite score"):
        export_c_handoff(
            manifest_path=manifest_path,
            data_root=data_root,
            protocol_path=PROTOCOL_PATH,
            output_root=tmp_path / "missing-score",
            implementation_commit=IMPLEMENTATION_COMMIT,
            implementation_sha256=IMPLEMENTATION_SHA256,
            model_sha256=MODEL_SHA256,
            refiner_checkpoint_sha256=REFINER_SHA256,
            scorer_checkpoint_sha256=SCORER_SHA256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["items"][0]["inputs"]["mask"].pop("bytes"),
            "inputs.mask keys mismatch",
        ),
        (
            lambda value: value["items"].pop(),
            "exact same base sample set",
        ),
        (
            lambda value: value["boundary"].update(
                {"contains_gt_derived_control_inputs": False}
            ),
            "boundary changed",
        ),
    ],
)
def test_handoff_rejects_missing_fields_coverage_and_boundary_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest, _ = _export(tmp_path / "handoff")
    changed = copy.deepcopy(manifest)
    mutation(changed)
    _relock(changed)
    with pytest.raises(ContractError, match=message):
        validate_c_handoff_manifest(changed, bundle_root=tmp_path / "handoff")


def test_export_refuses_to_overwrite_a_frozen_bundle(tmp_path: Path) -> None:
    root = tmp_path / "handoff"
    _export(root)
    with pytest.raises(ContractError, match="Refusing to overwrite"):
        export_c_handoff(
            manifest_path=MANIFEST_PATH,
            data_root=DATA_ROOT,
            protocol_path=PROTOCOL_PATH,
            output_root=root,
            implementation_commit=IMPLEMENTATION_COMMIT,
            implementation_sha256=IMPLEMENTATION_SHA256,
            model_sha256=MODEL_SHA256,
            refiner_checkpoint_sha256=REFINER_SHA256,
            scorer_checkpoint_sha256=SCORER_SHA256,
        )
