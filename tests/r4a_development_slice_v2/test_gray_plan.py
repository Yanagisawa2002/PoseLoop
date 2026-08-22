from __future__ import annotations

import json
from pathlib import Path

from r4a_development_slice_v2.planner import build_plan


def _entry(name: str) -> dict:
    return {"name": name, "disk_start": 0, "local_offset": 0, "compressed_size": 5, "uncompressed_size": 10, "compression": 8, "crc32": 1, "flags": 0}


def test_gray_three_scene_one_frame_plan(tmp_path: Path) -> None:
    entries = []
    for scene in range(3):
        prefix = f"train_pbr/{scene:06d}"
        entries.extend(_entry(f"{prefix}/{name}") for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json"))
        entries.extend((_entry(f"{prefix}/gray/000000.png"), _entry(f"{prefix}/depth/000000.png")))
        for instance in range(2):
            entries.append(_entry(f"{prefix}/mask_visib/000000_{instance:06d}.png"))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    plan = build_plan(
        catalog_path=catalog,
        output_path=tmp_path / "plan.json",
        scene_ids=[0, 1, 2],
        image_ids=[0],
        expected_target_count=6,
        maximum_compressed_bytes=1000,
        maximum_target_count=10,
    )
    assert plan["target_count"] == 6
    assert plan["role_counts"]["gray"] == 3
    assert plan["input_modality"] == "gray"
