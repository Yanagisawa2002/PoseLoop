"""Common data loading and CAD-overlay helpers for PoseLoop R2 figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import trimesh

from paper_plot_style import COLORS


REPO_ROOT = Path(__file__).resolve().parents[3]
SEALED_ROOT = REPO_ROOT / "artifacts" / "r2" / "sealed_photoneo"
OUTPUT_DIR = REPO_ROOT / "reports" / "r2" / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_rows(
    rows: Iterable[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in output:
            raise ValueError(f"Duplicate {field}: {key}")
        output[key] = row
    return output


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {path}")


def read_rgb(path: str | Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 2:
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            raise ValueError(f"Empty image: {path}")
        if raw.dtype == np.uint8:
            normalized = raw
        else:
            low, high = np.percentile(finite, [0.5, 99.5])
            scale = max(float(high - low), 1.0)
            normalized = np.clip((raw.astype(np.float32) - low) * 255.0 / scale, 0, 255)
            normalized = normalized.astype(np.uint8)
        return np.repeat(normalized[:, :, None], 3, axis=2)
    if raw.shape[2] == 4:
        raw = raw[:, :, :3]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8) * 255


def load_mesh_m(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangular CAD mesh: {path}")
    vertices_m = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices_m, faces


def projected_silhouette(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    pose_m: np.ndarray,
    camera_matrix: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    rotation = np.asarray(pose_m, dtype=np.float64)[:3, :3]
    translation = np.asarray(pose_m, dtype=np.float64)[:3, 3]
    camera_points = vertices_m @ rotation.T + translation
    z = camera_points[:, 2]
    projected = camera_points @ np.asarray(camera_matrix, dtype=np.float64).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)

    valid_faces = faces[np.all(z[faces] > 1e-6, axis=1)]
    triangles = uv[valid_faces]
    finite = np.all(np.isfinite(triangles), axis=(1, 2))
    triangles = triangles[finite]
    if triangles.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    in_frame = (
        (np.max(triangles[:, :, 0], axis=1) >= 0)
        & (np.min(triangles[:, :, 0], axis=1) < width)
        & (np.max(triangles[:, :, 1], axis=1) >= 0)
        & (np.min(triangles[:, :, 1], axis=1) < height)
    )
    polygons = np.rint(triangles[in_frame]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
    return mask


def color_rgb(name: str) -> np.ndarray:
    value = COLORS[name].lstrip("#")
    return np.asarray([int(value[index : index + 2], 16) for index in (0, 2, 4)])


def overlay_pose_silhouettes(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    predicted_mask: np.ndarray,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = rgb.astype(np.float32)
    gt_color = color_rgb("green")
    pred_color = color_rgb("magenta")
    for mask, color in ((gt_mask, gt_color), (predicted_mask, pred_color)):
        selected = mask > 0
        canvas[selected] = 0.82 * canvas[selected] + 0.18 * color
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for mask, color, halo in (
        (gt_mask, gt_color, color_rgb("black")),
        (predicted_mask, pred_color, color_rgb("white")),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            canvas, contours, -1, tuple(int(v) for v in halo), 8, cv2.LINE_AA
        )
        cv2.drawContours(
            canvas, contours, -1, tuple(int(v) for v in color), 5, cv2.LINE_AA
        )
    crop = square_crop_box([visible_mask, gt_mask, predicted_mask], rgb.shape[:2])
    return crop_image(canvas, crop), crop


def acquisition_view(
    rgb: np.ndarray, visible_mask: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = rgb.copy()
    contours, _ = cv2.findContours(
        visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        canvas,
        contours,
        -1,
        tuple(int(v) for v in color_rgb("sky")),
        5,
        cv2.LINE_AA,
    )
    crop = square_crop_box([visible_mask], rgb.shape[:2])
    return crop_image(canvas, crop), crop


def square_crop_box(
    masks: Iterable[np.ndarray], image_shape: tuple[int, int], pad_fraction: float = 0.45
) -> tuple[int, int, int, int]:
    height, width = image_shape
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= np.asarray(mask) > 0
    ys, xs = np.nonzero(union)
    if not len(xs):
        return 0, 0, width, height
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    side = max(96, int(np.ceil(max(box_width, box_height) * (1.0 + 2.0 * pad_fraction))))
    side = min(side, height, width)
    center_x = 0.5 * (float(xs.min()) + float(xs.max()))
    center_y = 0.5 * (float(ys.min()) + float(ys.max()))
    x0 = int(round(center_x - side / 2))
    y0 = int(round(center_y - side / 2))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def crop_image(image: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    return image[y0:y1, x0:x1]


def transform_to_target(
    pose_source_camera_m: np.ndarray,
    source_world_to_camera_m: np.ndarray,
    target_world_to_camera_m: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(target_world_to_camera_m, dtype=np.float64)
        @ np.linalg.inv(np.asarray(source_world_to_camera_m, dtype=np.float64))
        @ np.asarray(pose_source_camera_m, dtype=np.float64)
    )
