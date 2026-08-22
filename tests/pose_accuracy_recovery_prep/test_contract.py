from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pose_accuracy_recovery_prep import PROTOCOL_ID
from pose_accuracy_recovery_prep.core import (
    ContractError,
    build_run_plan,
    export_producer_manifest,
    load_and_validate_manifest,
    sha256_file,
    validate_manifest,
    validate_producer_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "fixtures" / "pose_accuracy_recovery_prep"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
PREP_LOCK = ROOT / "pose_accuracy_recovery_prep" / "PREP_LOCK.json"


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_fixture_manifest_hashes_and_boundaries_validate() -> None:
    loaded, validation = load_and_validate_manifest(MANIFEST_PATH, data_root=DATA_ROOT)
    assert loaded["protocol_id"] == PROTOCOL_ID
    assert validation == {
        "schema_version": "poseloop.pose-accuracy-recovery.validation.v1",
        "status": "valid",
        "sample_count": 1,
        "unique_key_count": 1,
        "asset_reference_count": 10,
        "verified_hashes": True,
        "producer_variant_count": 3,
        "evaluator_variant_count": 5,
        "gt_leak_count": 0,
        "accuracy_claim_permitted": False,
    }


def test_prep_lock_binds_protocol_schema_fixture_and_uncommitted_state() -> None:
    lock = json.loads(PREP_LOCK.read_text(encoding="utf-8"))
    assert lock["state"] == "frozen_local_prep_awaiting_controller_commit"
    assert lock["auto_deploy"] is False
    assert lock["base_commit"] == "58939e69960f2227c4fa20be547e61df23fe391d"
    assert lock["implementation_commit"] is None
    for name in ("protocol", "manifest_schema", "fixture_manifest"):
        assert sha256_file(ROOT / lock[name]["path"]) == lock[name]["sha256"]
    assert lock["cpu_dry_run"]["tracked"] is False
    assert lock["cpu_dry_run"]["accuracy_claim_permitted"] is False


def test_producer_export_has_no_gt_or_evaluator_identity() -> None:
    loaded, _ = load_and_validate_manifest(MANIFEST_PATH, data_root=DATA_ROOT)
    exported = export_producer_manifest(loaded)
    assert validate_producer_manifest(exported, data_root=DATA_ROOT) == {
        "status": "valid",
        "label_access_count": 0,
        "sample_count": 1,
        "unique_key_count": 1,
        "asset_reference_count": 7,
        "verified_hashes": True,
        "gt_leak_count": 0,
    }
    text = json.dumps(exported, sort_keys=True).lower()
    for forbidden in (
        "gt_pose",
        "oracle",
        "evaluator_only",
        "official_known",
        "score",
        "sealed",
    ):
        assert forbidden not in text
    assert exported["mask_variants"] == [
        "predicted_mask",
        "depth_component_mask",
        "bbox_mask",
    ]


def test_run_plans_share_sample_key_but_preserve_roles() -> None:
    loaded = manifest()
    producer = build_run_plan(loaded, namespace_role="producer")
    evaluator = build_run_plan(loaded, namespace_role="evaluator-only")
    assert len(producer) == 3
    assert len(evaluator) == 5
    assert {
        (row["scene_id"], row["image_id"], row["object_id"])
        for row in producer + evaluator
    } == {(0, 0, 1)}
    assert all("gt_pose" not in row for row in producer)
    assert all("gt_pose" in row for row in evaluator)
    assert {
        row["mask_variant"] for row in evaluator if row["oracle_diagnostic_only"]
    } == {
        "official_known_sample_sanity",
        "oracle_mask_control",
    }
    with pytest.raises(ContractError, match="Invalid mask variants"):
        build_run_plan(
            loaded, namespace_role="producer", variants=["oracle_mask_control"]
        )


def test_duplicate_scene_image_object_key_is_rejected() -> None:
    value = manifest()
    value["samples"].append(copy.deepcopy(value["samples"][0]))
    with pytest.raises(ContractError, match="Duplicate scene-image-object"):
        validate_manifest(value)


def test_producer_gt_key_is_rejected() -> None:
    value = manifest()
    value["samples"][0]["producer_inputs"]["gt_pose"] = value["samples"][0][
        "evaluator_only"
    ]["gt_pose"]
    with pytest.raises(ContractError, match="GT/evaluator leak"):
        validate_manifest(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["samples"][0]["producer_inputs"].pop("cad"),
            "cad must be an asset",
        ),
        (
            lambda value: value["units"].update({"pose_translation": "mm"}),
            "pose_translation must be m",
        ),
        (
            lambda value: value["coordinate_conventions"].update(
                {"pose_direction": "camera_to_model"}
            ),
            "pose_direction must be model_to_camera",
        ),
        (
            lambda value: value.update({"auto_deploy": True}),
            "AUTO_DEPLOY must remain false",
        ),
    ],
)
def test_incomplete_assets_units_coordinates_and_deploy_are_rejected(
    mutation, message: str
) -> None:
    value = manifest()
    mutation(value)
    with pytest.raises(ContractError, match=message):
        validate_manifest(value)


def test_missing_asset_and_hash_mismatch_are_rejected(tmp_path: Path) -> None:
    value = manifest()
    with pytest.raises(ContractError, match="Missing asset"):
        validate_manifest(value, data_root=tmp_path)
    value["samples"][0]["producer_inputs"]["rgb"]["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        validate_manifest(value, data_root=DATA_ROOT)
