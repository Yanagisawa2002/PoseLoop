from __future__ import annotations

import json
from pathlib import Path

from r4a_development_slice.planner import build_plan


def _entry(name: str, size: int = 10) -> dict:
    return {
        "name": name,
        "disk_start": 0,
        "local_offset": 0,
        "compressed_size": size,
        "uncompressed_size": size * 2,
        "compression": 8,
        "crc32": 1,
        "flags": 0,
    }


def test_plan_selects_exact_three_scene_four_frame_slice(tmp_path: Path) -> None:
    entries = []
    for scene in range(3):
        prefix = f"xyzibd/train_pbr/{scene:06d}"
        entries.extend(_entry(f"{prefix}/{name}") for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json"))
        for image in range(4):
            entries.append(_entry(f"{prefix}/rgb/{image:06d}.jpg"))
            entries.append(_entry(f"{prefix}/depth/{image:06d}.png"))
            entries.append(_entry(f"{prefix}/mask_visib/{image:06d}_000000.png"))
        entries.append(_entry(f"{prefix}/rgb/000004.jpg"))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    output = tmp_path / "plan.json"
    plan = build_plan(
        catalog_path=catalog,
        output_path=output,
        scene_ids=[0, 1, 2],
        image_ids=[0, 1, 2, 3],
        maximum_compressed_bytes=10000,
        maximum_target_count=128,
    )
    assert plan["entry_count"] == 45
    assert plan["target_count"] == 12
    assert plan["role_counts"] == {
        "depth": 12,
        "mask_visib": 12,
        "rgb": 12,
        "scene_camera": 3,
        "scene_gt": 3,
        "scene_gt_info": 3,
    }
    assert plan["development_label_bytes_opened"] == 0
