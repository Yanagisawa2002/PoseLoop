"""Selective XYZ-IBD ``scene_gt.json`` parser for the frozen ID-only gate.

Only image keys, ``obj_id`` and array ordinals are converted to Python values.
Pose arrays are skipped lexically and are never decoded or exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from r4a_development_repair.core import ContractError


@dataclass(frozen=True)
class IdRecord:
    scene_id: int
    image_id: int
    object_id: int
    instance_ordinal: int


class _Cursor:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.index = 0

    def _space(self) -> None:
        while self.index < len(self.payload) and self.payload[self.index] in b" \t\r\n":
            self.index += 1

    def take(self, token: int) -> None:
        self._space()
        if self.index >= len(self.payload) or self.payload[self.index] != token:
            raise ContractError(f"ID-only JSON syntax mismatch at byte {self.index}")
        self.index += 1

    def string(self) -> str:
        self._space()
        if self.index >= len(self.payload) or self.payload[self.index] != ord('"'):
            raise ContractError(f"ID-only JSON string expected at byte {self.index}")
        self.index += 1
        raw = bytearray()
        while self.index < len(self.payload):
            value = self.payload[self.index]
            self.index += 1
            if value == ord('"'):
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ContractError("ID-only JSON key is not UTF-8") from error
            if value == ord("\\"):
                raise ContractError("Escaped strings are forbidden in ID-only scene_gt keys")
            raw.append(value)
        raise ContractError("Unterminated ID-only JSON string")

    def integer(self) -> int:
        self._space()
        start = self.index
        if self.index < len(self.payload) and self.payload[self.index] == ord("-"):
            self.index += 1
        while self.index < len(self.payload) and ord("0") <= self.payload[self.index] <= ord("9"):
            self.index += 1
        token = self.payload[start : self.index]
        if not token or token == b"-" or any(value in token for value in (b".", b"e", b"E")):
            raise ContractError("obj_id must be an integer")
        return int(token)

    def skip_value(self) -> None:
        """Skip one JSON value without converting its scalar contents."""

        self._space()
        start = self.index
        if start >= len(self.payload):
            raise ContractError("Missing JSON value")
        opening = self.payload[start]
        if opening == ord('"'):
            self.index += 1
            escaped = False
            while self.index < len(self.payload):
                value = self.payload[self.index]
                self.index += 1
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    return
            raise ContractError("Unterminated skipped JSON string")
        if opening in (ord("["), ord("{")):
            stack = [opening]
            self.index += 1
            quoted = False
            escaped = False
            while self.index < len(self.payload) and stack:
                value = self.payload[self.index]
                self.index += 1
                if quoted:
                    if escaped:
                        escaped = False
                    elif value == ord("\\"):
                        escaped = True
                    elif value == ord('"'):
                        quoted = False
                    continue
                if value == ord('"'):
                    quoted = True
                elif value in (ord("["), ord("{")):
                    stack.append(value)
                elif value in (ord("]"), ord("}")):
                    expected = ord("[") if value == ord("]") else ord("{")
                    if stack.pop() != expected:
                        raise ContractError("Mismatched JSON delimiter while skipping pose")
            if stack:
                raise ContractError("Unterminated JSON value while skipping pose")
            return
        while self.index < len(self.payload) and self.payload[self.index] not in b",]} \t\r\n":
            self.index += 1
        if self.index == start:
            raise ContractError("Invalid skipped JSON scalar")


def parse_scene_gt_ids(payload: bytes, *, scene_id: int) -> tuple[list[IdRecord], dict[str, int]]:
    cursor = _Cursor(payload)
    records: list[IdRecord] = []
    cursor.take(ord("{"))
    first_image = True
    while True:
        cursor._space()
        if cursor.index < len(payload) and payload[cursor.index] == ord("}"):
            cursor.index += 1
            break
        if not first_image:
            cursor.take(ord(","))
        first_image = False
        image_key = cursor.string()
        if not image_key.isdigit():
            raise ContractError("scene_gt top-level image key must be decimal")
        image_id = int(image_key)
        cursor.take(ord(":"))
        cursor.take(ord("["))
        first_instance = True
        instance_ordinal = 0
        while True:
            cursor._space()
            if cursor.index < len(payload) and payload[cursor.index] == ord("]"):
                cursor.index += 1
                break
            if not first_instance:
                cursor.take(ord(","))
            first_instance = False
            cursor.take(ord("{"))
            object_id: int | None = None
            first_field = True
            while True:
                cursor._space()
                if cursor.index < len(payload) and payload[cursor.index] == ord("}"):
                    cursor.index += 1
                    break
                if not first_field:
                    cursor.take(ord(","))
                first_field = False
                key = cursor.string()
                cursor.take(ord(":"))
                if key == "obj_id":
                    if object_id is not None:
                        raise ContractError("Duplicate obj_id in scene_gt row")
                    object_id = cursor.integer()
                elif key in {"cam_R_m2c", "cam_t_m2c"}:
                    cursor.skip_value()
                else:
                    raise ContractError(f"Forbidden or unknown scene_gt field in ID-only parser: {key}")
            if object_id is None or object_id <= 0:
                raise ContractError("scene_gt row is missing a positive obj_id")
            records.append(IdRecord(scene_id, image_id, object_id, instance_ordinal))
            instance_ordinal += 1
    cursor._space()
    if cursor.index != len(payload):
        raise ContractError("Trailing bytes after scene_gt JSON")
    return records, {
        "top_level_image_ids_decoded": len({row.image_id for row in records}),
        "obj_ids_decoded": len(records),
        "instance_ordinals_derived": len(records),
        "pose_values_decoded": 0,
        "visibility_values_decoded": 0,
        "score_values_decoded": 0,
    }


def select_multi_object_targets(
    records_by_scene: Sequence[Sequence[IdRecord]],
    *,
    required_objects: int = 5,
    required_scenes: int = 3,
    images_per_object: int = 2,
) -> dict[str, Any] | None:
    flattened = [record for rows in records_by_scene for record in rows]
    encounter: dict[int, int] = {}
    pairs: dict[int, list[tuple[int, int]]] = {}
    instances: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(flattened):
        encounter.setdefault(row.object_id, index)
        pair = (row.scene_id, row.image_id)
        values = pairs.setdefault(row.object_id, [])
        if pair not in values:
            values.append(pair)
        key = (row.scene_id, row.image_id, row.object_id)
        instances[key] = min(instances.get(key, row.instance_ordinal), row.instance_ordinal)
    eligible = [obj for obj in sorted(pairs, key=lambda obj: (encounter[obj], obj)) if len(pairs[obj]) >= images_per_object]
    import itertools

    for objects in itertools.combinations(eligible, required_objects):
        selected_pairs = {obj: pairs[obj][:images_per_object] for obj in objects}
        scenes = {scene for values in selected_pairs.values() for scene, _ in values}
        if len(scenes) < required_scenes:
            continue
        targets = []
        for object_id in objects:
            for scene_id, image_id in selected_pairs[object_id]:
                targets.append(
                    {
                        "item_id": f"s{scene_id:06d}-i{image_id:06d}-o{object_id:06d}",
                        "scene_id": scene_id,
                        "image_id": image_id,
                        "object_id": object_id,
                        "instance_ordinal": instances[(scene_id, image_id, object_id)],
                    }
                )
        return {
            "object_ids": list(objects),
            "scene_ids": sorted(scenes),
            "targets": targets,
            "target_count": len(targets),
            "unique_key_count": len({(row["scene_id"], row["image_id"], row["object_id"]) for row in targets}),
            "per_object_count": {str(obj): len(selected_pairs[obj]) for obj in objects},
        }
    return None


__all__ = ["IdRecord", "parse_scene_gt_ids", "select_multi_object_targets"]
