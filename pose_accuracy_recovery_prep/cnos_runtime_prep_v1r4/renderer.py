"""Thin A-R4 supervisor around the pinned official CNOS PyRender function."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.renderer import (
    diameter_from_bound_cad_path,
    load_official_renderer_modules,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)

from . import (
    ATTEMPT_FAILURE_SCHEMA,
    ATTEMPT_RECEIPT_SCHEMA,
    ATTEMPT_START_SCHEMA,
    GPU_OVERRIDE,
    OBJECT_IDS,
    VIEW_COUNT,
)
from .contracts import (
    BOUNDARY_ZERO,
    RuntimeValidation,
    json_bytes,
    validate_runtime_request,
    write_create_only,
)


@dataclass(frozen=True)
class PreparedMesh:
    """Unit-consistent mesh and transform passed to official ``render()``."""

    render_mesh: Any
    recenter_transform: np.ndarray
    diameter: float
    applied_scale: float
    centroid_after_scale: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object_path(validation: RuntimeValidation, object_id: int) -> Path:
    for bound_object_id, path in validation.objects:
        if bound_object_id == object_id:
            return path
    raise ContractError(f"Object {object_id} is outside the frozen five-object lock")


def prepare_mesh_for_render(
    renderer_module: Any,
    helper_module: Any,
    cad_path: Path,
) -> PreparedMesh:
    """Apply the sole geometry repair before calling official ``render()``.

    The diameter helper still receives the hash-bound CAD path.  If the CAD is
    in millimetres, scaling occurs before the centroid is read, so the
    recenter translation has the same units as the scaled vertices.
    """
    if not cad_path.is_file():
        raise ContractError(f"Missing hash-bound CAD file: {cad_path}")
    mesh = renderer_module.trimesh.load_mesh(str(cad_path.resolve()))
    diameter = diameter_from_bound_cad_path(helper_module, cad_path)
    applied_scale = 1.0
    if diameter > 100.0:
        applied_scale = 0.001
        mesh.apply_scale(applied_scale)
    centroid = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ContractError("Scaled CNOS mesh centroid must be finite length-three")
    recenter_transform = np.eye(4, dtype=np.float64)
    recenter_transform[:3, 3] = -centroid
    render_mesh = renderer_module.pyrender.Mesh.from_trimesh(
        helper_module.as_mesh(mesh)
    )
    return PreparedMesh(
        render_mesh=render_mesh,
        recenter_transform=recenter_transform,
        diameter=diameter,
        applied_scale=applied_scale,
        centroid_after_scale=centroid.copy(),
    )


def expected_png_names() -> tuple[str, ...]:
    return tuple(f"{index:06d}.png" for index in range(VIEW_COUNT))


def validate_object_output(
    output_dir: Path, *, object_id: int, output_gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the exact 42-PNG RGB and alpha content gate from disk."""
    if not output_dir.is_dir():
        raise ContractError(
            f"CNOS child produced no output directory for object {object_id}"
        )
    expected_names = expected_png_names()
    entries = tuple(sorted(output_dir.iterdir(), key=lambda path: path.name))
    if tuple(path.name for path in entries) != expected_names or any(
        not path.is_file() for path in entries
    ):
        raise ContractError(
            f"CNOS object {object_id} output must contain exactly 42 canonical PNG files"
        )
    expected_size = (
        int(output_gate["image_size"]["width"]),
        int(output_gate["image_size"]["height"]),
    )
    allowed_modes = set(output_gate["allowed_modes"])
    frame_pixels = expected_size[0] * expected_size[1]
    inventory: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in entries:
        try:
            with Image.open(path) as image:
                image.load()
                image_format = image.format
                image_size = image.size
                image_mode = image.mode
                rgb = np.asarray(image.convert("RGB"))
                alpha = (
                    np.asarray(image.getchannel("A")) if image.mode == "RGBA" else None
                )
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"Cannot decode CNOS renderer PNG: {path}") from exc
        if image_format != "PNG" or image_size != expected_size:
            raise ContractError(
                f"CNOS object {object_id} image format/frame changed: {path.name}"
            )
        if image_mode not in allowed_modes:
            raise ContractError(
                f"CNOS object {object_id} image mode changed: {path.name}"
            )
        rgb_foreground = int(np.any(rgb != 0, axis=2).sum())
        if not 0 < rgb_foreground < frame_pixels:
            raise ContractError(
                f"CNOS object {object_id} PNG has empty/full RGB foreground: {path.name}"
            )
        alpha_foreground: int | None = None
        if alpha is not None:
            alpha_foreground = int((alpha != 0).sum())
            if not 0 < alpha_foreground < frame_pixels:
                raise ContractError(
                    f"CNOS object {object_id} PNG has empty/full alpha foreground: {path.name}"
                )
        digest = sha256_file(path)
        hashes.add(digest)
        inventory.append(
            {
                "relative_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "mode": image_mode,
                "rgb_foreground_pixels": rgb_foreground,
                "rgb_foreground_coverage": rgb_foreground / frame_pixels,
                "alpha_foreground_pixels": alpha_foreground,
                "alpha_foreground_coverage": (
                    None
                    if alpha_foreground is None
                    else alpha_foreground / frame_pixels
                ),
            }
        )
    if len(hashes) < output_gate["minimum_unique_png_sha256_per_object"]:
        raise ContractError(f"CNOS object {object_id} rendered views are not distinct")
    return {
        "object_id": object_id,
        "png_count": len(inventory),
        "unique_png_sha256_count": len(hashes),
        "inventory": inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }


def render_one_object(
    *,
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    deployment_root: Path,
    repository_root: Path,
    object_id: int,
    output_dir: Path,
    gpu_override: str,
    module_loader: Callable[[Path], tuple[Any, Any]] = load_official_renderer_modules,
) -> dict[str, Any]:
    """Prepare one bound CAD, then call the unchanged official render function."""
    validation = validate_runtime_request(
        route,
        request,
        deployment_root=deployment_root,
        repository_root=repository_root,
    )
    if gpu_override != GPU_OVERRIDE or not isinstance(gpu_override, str):
        raise ContractError("CNOS R4 child GPU override must remain string '0'")
    expected_output = (
        validation.deployment_root
        / "attempts"
        / str(request["attempt_id"])
        / "objects"
        / f"obj_{object_id:06d}"
    ).resolve()
    if output_dir.resolve() != expected_output:
        raise ContractError("CNOS R4 child output path differs from the frozen layout")
    if output_dir.exists():
        raise ContractError("CNOS R4 child output directory is create-only")
    if not output_dir.parent.is_dir():
        raise ContractError("CNOS R4 child object parent directory is missing")

    cad_path = _object_path(validation, object_id)
    renderer, helpers = module_loader(validation.cnos_checkout)
    poses = np.load(validation.poses_path, allow_pickle=False)
    if poses.shape != (VIEW_COUNT, 4, 4) or not np.isfinite(poses).all():
        raise ContractError("Pinned CNOS level-0 pose tensor shape/content changed")
    poses = np.asarray(poses, dtype=np.float64).copy()
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    prepared = prepare_mesh_for_render(renderer, helpers, cad_path)

    output_dir.mkdir(parents=False, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_override
    renderer_contract = route["repair_contract"]["renderer"]
    renderer.render(
        output_dir=str(output_dir),
        mesh=prepared.render_mesh,
        obj_poses=poses,
        intrinsic=np.asarray(renderer_contract["intrinsic"], dtype=np.float64),
        img_size=(
            renderer_contract["image_size"]["height"],
            renderer_contract["image_size"]["width"],
        ),
        light_itensity=renderer_contract["light_intensity"],
        re_center_transform=prepared.recenter_transform,
    )
    result = validate_object_output(
        output_dir, object_id=object_id, output_gate=route["output_gate"]
    )
    result["mesh_preparation"] = {
        "diameter": prepared.diameter,
        "applied_scale": prepared.applied_scale,
        "centroid_after_scale": prepared.centroid_after_scale.tolist(),
        "recenter_translation": prepared.recenter_transform[:3, 3].tolist(),
    }
    return result


def build_child_environment(
    validation: RuntimeValidation, base_environment: Mapping[str, str]
) -> dict[str, str]:
    """Build the exact venv-entry-first environment for five children."""
    environment = dict(base_environment)
    old_path = environment.get("PATH", "")
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PATH"] = str(validation.venv_bin) + (
        os.pathsep + old_path if old_path else ""
    )
    environment["PYTHONPATH"] = str(validation.implementation_checkout) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    environment["CUDA_VISIBLE_DEVICES"] = GPU_OVERRIDE
    environment["PYOPENGL_PLATFORM"] = "egl"
    return environment


def build_child_argv(
    validation: RuntimeValidation,
    *,
    route_path: Path,
    request_path: Path,
    repository_root: Path,
    object_id: int,
    output_dir: Path,
) -> list[str]:
    """Execute the lexical bound entry, never its resolved base target."""
    _object_path(validation, object_id)
    return [
        str(validation.python_entry),
        "-m",
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4",
        "render-one",
        "--route",
        str(route_path.resolve()),
        "--request",
        str(request_path.resolve()),
        "--deployment-root",
        str(validation.deployment_root),
        "--repository-root",
        str(repository_root.resolve()),
        "--object-id",
        str(object_id),
        "--output-dir",
        str(output_dir.resolve()),
        "--gpu-override",
        GPU_OVERRIDE,
    ]


def _write_text_log(path: Path, value: str) -> None:
    write_create_only(path, value.encode("utf-8", errors="replace"))


def run_render_attempt(
    *,
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    route_path: Path,
    request_path: Path,
    deployment_root: Path,
    repository_root: Path,
    output_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Launch five create-only children and fail on any output inconsistency."""
    validation = validate_runtime_request(
        route,
        request,
        deployment_root=deployment_root,
        repository_root=repository_root,
    )
    expected_root = (
        validation.deployment_root / "attempts" / str(request["attempt_id"])
    ).resolve()
    if output_root.resolve() != expected_root:
        raise ContractError("CNOS R4 attempt output root differs from request identity")
    if output_root.exists():
        raise ContractError("CNOS R4 attempt output root is create-only")
    output_root.mkdir(parents=True, exist_ok=False)
    object_root = output_root / "objects"
    logs_root = output_root / "logs"
    object_root.mkdir()
    logs_root.mkdir()

    environment = build_child_environment(validation, base_environment or os.environ)
    commands = [
        build_child_argv(
            validation,
            route_path=route_path,
            request_path=request_path,
            repository_root=repository_root,
            object_id=object_id,
            output_dir=object_root / f"obj_{object_id:06d}",
        )
        for object_id in OBJECT_IDS
    ]
    start = {
        "schema_version": ATTEMPT_START_SCHEMA,
        "created_at": _utc_now(),
        "route_id": route["route_id"],
        "route_lock_sha256": route["route_lock_sha256"],
        "request_lock_sha256": request["request_lock_sha256"],
        "attempt_id": request["attempt_id"],
        "implementation": dict(request["implementation"]),
        "source_commit": route["official_source"]["commit"],
        "source_tree": route["official_source"]["tree"],
        "venv_bin": str(validation.venv_bin),
        "venv_python_entry": request["venv"]["python_entry"],
        "venv_python_target": request["venv"]["python_target"],
        "gpu_override": GPU_OVERRIDE,
        "pyopengl_platform": "egl",
        "object_ids": list(OBJECT_IDS),
        "commands": commands,
        "geometry_repair": "scale_then_recompute_centroid_then_recenter",
        "execution_started": False,
        "boundary": dict(BOUNDARY_ZERO),
    }
    write_create_only(output_root / "attempt-start.json", json_bytes(start))

    inventories: list[dict[str, Any]] = []
    try:
        for object_id, command in zip(OBJECT_IDS, commands, strict=True):
            result = runner(
                command,
                cwd=str(validation.implementation_checkout),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            _write_text_log(
                logs_root / f"obj_{object_id:06d}.stdout.txt", result.stdout or ""
            )
            _write_text_log(
                logs_root / f"obj_{object_id:06d}.stderr.txt", result.stderr or ""
            )
            if result.returncode != 0:
                raise ContractError(
                    f"CNOS R4 renderer child failed for object {object_id}: "
                    f"exit {result.returncode}"
                )
            inventories.append(
                validate_object_output(
                    object_root / f"obj_{object_id:06d}",
                    object_id=object_id,
                    output_gate=route["output_gate"],
                )
            )
    except Exception as exc:
        failure = {
            "schema_version": ATTEMPT_FAILURE_SCHEMA,
            "created_at": _utc_now(),
            "status": "FAILED_CREATE_ONLY_ATTEMPT",
            "route_lock_sha256": route["route_lock_sha256"],
            "request_lock_sha256": request["request_lock_sha256"],
            "attempt_id": request["attempt_id"],
            "completed_object_ids": [item["object_id"] for item in inventories],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rerun_same_output_permitted": False,
            "boundary": dict(BOUNDARY_ZERO),
        }
        write_create_only(output_root / "attempt-failure.json", json_bytes(failure))
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"CNOS R4 renderer attempt failed: {exc}") from exc

    receipt = {
        "schema_version": ATTEMPT_RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "status": "PASS",
        "route_lock_sha256": route["route_lock_sha256"],
        "request_lock_sha256": request["request_lock_sha256"],
        "attempt_id": request["attempt_id"],
        "object_count": len(inventories),
        "png_count": sum(item["png_count"] for item in inventories),
        "objects": inventories,
        "object_inventory_sha256": canonical_sha256(inventories),
        "geometry_repair": "scale_then_recompute_centroid_then_recenter",
        "producer_started": False,
        "accuracy_claim_permitted": False,
        "boundary": dict(BOUNDARY_ZERO),
        "receipt_lock_sha256": "pending",
    }
    receipt["receipt_lock_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_lock_sha256"}
    )
    write_create_only(output_root / "attempt-receipt.json", json_bytes(receipt))
    return receipt
