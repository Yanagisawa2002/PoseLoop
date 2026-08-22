from __future__ import annotations

import tarfile
from pathlib import Path

import cv2
import numpy as np

from r4a_development_repair.core import read_json, sha256_file, write_json_atomic
from r4a_development_v3.assets import bbox_from_mask, coco_rle, depth_component_mask
from r4a_development_v3.bundle import _deterministic_targz, build_inference_bundle


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "poseloop_r4a_xyzibd_train_pbr_development_multi_object_v3.json"


def test_depth_mask_and_rle_are_deterministic() -> None:
    depth = np.full((64, 64), 2000, dtype=np.uint16)
    depth[20:44, 22:42] = 500
    first = depth_component_mask(depth)
    second = depth_component_mask(depth.copy())
    assert np.array_equal(first, second)
    assert int(first.sum()) > 64
    rle = coco_rle(first)
    assert sum(rle["counts"]) == 64 * 64
    assert bbox_from_mask(first)[2] > 0


def test_deterministic_archive_has_zero_mtime_and_relative_members(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "a.json").write_text("{}\n", encoding="utf-8")
    first = _deterministic_targz(root, tmp_path / "one.tar.gz")
    second = _deterministic_targz(root, tmp_path / "two.tar.gz")
    assert first["sha256"] == second["sha256"]
    with tarfile.open(first["path"], "r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == ["contracts/a.json"]
    assert members[0].mtime == 0


def test_inference_bundle_contains_receiver_contract_and_no_gt(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    models = tmp_path / "models"
    scene = assets / "train_pbr" / "000000"
    (scene / "gray").mkdir(parents=True)
    (scene / "depth").mkdir(parents=True)
    models.mkdir()
    gray = np.full((64, 64), 127, dtype=np.uint8)
    depth = np.full((64, 64), 2000, dtype=np.uint16)
    depth[20:44, 22:42] = 500
    assert cv2.imwrite(str(scene / "gray" / "000001.png"), gray)
    assert cv2.imwrite(str(scene / "depth" / "000001.png"), depth)
    write_json_atomic(
        scene / "scene_camera.json",
        {"1": {"cam_K": [500, 0, 32, 0, 500, 32, 0, 0, 1], "depth_scale": 1.0}},
    )
    (models / "obj_000001.ply").write_text("ply\n", encoding="ascii")
    exact = tmp_path / "exact.json"
    write_json_atomic(
        exact,
        {
            "schema_version": "poseloop.r4a.development-exact-selection.v3",
            "protocol_id": "poseloop.r4a.xyzibd-train-pbr.development.multi-object.v3",
            "protocol_sha256": sha256_file(PROTOCOL),
            "gate_pass": True,
            "object_ids": [1],
            "targets": [{"scene_id": 0, "image_id": 1, "object_id": 1, "instance_ordinal": 0}],
        },
    )
    descriptor = build_inference_bundle(
        protocol_path=PROTOCOL,
        exact_selection_path=exact,
        asset_root=assets,
        models_root=models,
        stage_root=tmp_path / "stage",
        archive_path=tmp_path / "bundle.tar.gz",
        descriptor_path=tmp_path / "descriptor.json",
        implementation_commit="a" * 40,
    )
    assert descriptor["label_access_count"] == 0
    assert descriptor["coverage"] == 1.0
    stage = tmp_path / "stage"
    workload = (stage / "contracts" / "workload.jsonl").read_text(encoding="utf-8")
    assert "dataset/rgb/" in workload
    assert "cam_R_m2c" not in workload
    assert "mask_visib" not in "\n".join(path.as_posix() for path in stage.rglob("*"))
    handoff = read_json(stage / "contracts" / "handoff.json")
    assert handoff["workload_sha256"] == sha256_file(stage / "contracts" / "workload.jsonl")
    with tarfile.open(descriptor["archive"]["path"], "r:gz") as archive:
        names = archive.getnames()
    assert "contracts/workload.jsonl" in names
    assert "contracts/coco-predictions.json" in names
    assert "contracts/handoff.json" in names
    assert "SHA256SUMS" in names
