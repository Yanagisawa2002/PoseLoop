"""Build an exact, label-free entry plan from ZIP central-directory names."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from r4a_development_repair.core import ContractError, read_json, write_json_atomic


def _role_for_entry(name: str, scene_ids: set[int], image_ids: set[int]) -> tuple[str, int, int | None] | None:
    parts = PurePosixPath(name).parts
    try:
        train_index = parts.index("train_pbr")
    except ValueError:
        return None
    suffix = parts[train_index:]
    if len(suffix) < 3 or not suffix[1].isdigit():
        return None
    scene_id = int(suffix[1])
    if scene_id not in scene_ids:
        return None
    relative = suffix[2:]
    if len(relative) == 1 and relative[0] in {"scene_camera.json", "scene_gt.json", "scene_gt_info.json"}:
        return relative[0].removesuffix(".json"), scene_id, None
    if len(relative) != 2:
        return None
    directory, filename = relative
    stem = Path(filename).stem
    if directory in {"rgb", "depth"} and stem.isdigit() and int(stem) in image_ids:
        return directory, scene_id, int(stem)
    if directory == "mask_visib" and "_" in stem:
        image_text, instance_text = stem.split("_", 1)
        if image_text.isdigit() and instance_text.isdigit() and int(image_text) in image_ids:
            return "mask_visib", scene_id, int(image_text)
    return None


def build_plan(
    *,
    catalog_path: Path,
    output_path: Path,
    scene_ids: Sequence[int],
    image_ids: Sequence[int],
    maximum_compressed_bytes: int,
    maximum_target_count: int,
) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ContractError("Central-directory catalog has no entries array")
    selected: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    scene_role_counts: Counter[str] = Counter()
    frame_role_counts: Counter[str] = Counter()
    for entry in entries:
        role = _role_for_entry(str(entry.get("name", "")), set(scene_ids), set(image_ids))
        if role is None:
            continue
        role_name, scene_id, image_id = role
        row = dict(entry)
        row["role"] = role_name
        row["scene_id"] = scene_id
        row["image_id"] = image_id
        selected.append(row)
        role_counts[role_name] += 1
        scene_role_counts[f"{scene_id:06d}/{role_name}"] += 1
        if image_id is not None:
            frame_role_counts[f"{scene_id:06d}/{image_id:06d}/{role_name}"] += 1
    selected.sort(key=lambda row: str(row["name"]))
    errors: list[str] = []
    for scene_id in scene_ids:
        for role in ("scene_camera", "scene_gt", "scene_gt_info"):
            if scene_role_counts[f"{scene_id:06d}/{role}"] != 1:
                errors.append(f"expected one {role} for scene {scene_id}")
        for image_id in image_ids:
            for role in ("rgb", "depth"):
                if frame_role_counts[f"{scene_id:06d}/{image_id:06d}/{role}"] != 1:
                    errors.append(f"expected one {role} for scene {scene_id} image {image_id}")
            if frame_role_counts[f"{scene_id:06d}/{image_id:06d}/mask_visib"] < 1:
                errors.append(f"no mask_visib for scene {scene_id} image {image_id}")
    target_count = role_counts["mask_visib"]
    compressed_bytes = sum(int(row["compressed_size"]) for row in selected)
    uncompressed_bytes = sum(int(row["uncompressed_size"]) for row in selected)
    if target_count > maximum_target_count:
        errors.append(f"target stop gate exceeded: {target_count} > {maximum_target_count}")
    if compressed_bytes > maximum_compressed_bytes:
        errors.append(f"compressed-byte stop gate exceeded: {compressed_bytes} > {maximum_compressed_bytes}")
    if errors:
        raise ContractError("; ".join(errors))
    plan = {
        "schema_version": "poseloop.r4a.development-slice.plan.v1",
        "mode": "central_directory_names_and_sizes_only_no_label_bytes_opened",
        "catalog_path": str(catalog_path.resolve()),
        "scene_ids": list(scene_ids),
        "image_ids_per_scene": list(image_ids),
        "target_policy": "all_mask_visib_entries_for_selected_images",
        "entry_count": len(selected),
        "target_count": target_count,
        "role_counts": dict(sorted(role_counts.items())),
        "compressed_entry_bytes": compressed_bytes,
        "uncompressed_entry_bytes": uncompressed_bytes,
        "entries": selected,
        "development_label_bytes_opened": 0,
        "xyzibd_val_access_count": 0,
        "official_evaluator_invocation_count": 0,
    }
    write_json_atomic(output_path, plan)
    return plan
