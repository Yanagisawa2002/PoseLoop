"""Thin, create-only supervisor for the pinned official CNOS PyRender route."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from pose_accuracy_recovery_prep.core import ContractError, canonical_sha256, sha256_file

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _module_from_checkout(module: ModuleType, checkout: Path, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ContractError(f"{label} does not expose a source file")
    try:
        Path(module_file).resolve().relative_to(checkout.resolve())
    except ValueError as exc:
        raise ContractError(f"{label} imported outside the pinned CNOS checkout") from exc


def load_official_renderer_modules(
    checkout: Path,
) -> tuple[ModuleType, ModuleType]:
    """Import only the two already hash-verified official CNOS modules."""
    source_text = str(checkout.resolve())
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        renderer = importlib.import_module("src.poses.pyrender")
        helpers = importlib.import_module("src.utils.trimesh_utils")
        _module_from_checkout(renderer, checkout, "CNOS PyRender module")
        _module_from_checkout(helpers, checkout, "CNOS trimesh helper module")
        if not callable(getattr(renderer, "render", None)):
            raise ContractError("Pinned CNOS renderer no longer exposes render()")
        if not callable(getattr(helpers, "as_mesh", None)) or not callable(
            getattr(helpers, "get_obj_diameter", None)
        ):
            raise ContractError("Pinned CNOS trimesh helper interface changed")
        return renderer, helpers
    finally:
        if inserted and source_text in sys.path:
            sys.path.remove(source_text)


def diameter_from_bound_cad_path(helper_module: Any, cad_path: Path) -> float:
    """Call the pinned path-only helper with a path, never a loaded Trimesh."""
    if not cad_path.is_file():
        raise ContractError(f"Missing hash-bound CAD file: {cad_path}")
    value = helper_module.get_obj_diameter(str(cad_path.resolve()))
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or float(value) <= 0
    ):
        raise ContractError("Pinned CNOS path-based CAD diameter is not positive finite")
    return float(value)


def _object_path(validation: RuntimeValidation, object_id: int) -> Path:
    for bound_object_id, path in validation.objects:
        if bound_object_id == object_id:
            return path
    raise ContractError(f"Object {object_id} is outside the frozen five-object lock")


def expected_png_names() -> tuple[str, ...]:
    return tuple(f"{index:06d}.png" for index in range(VIEW_COUNT))


def validate_object_output(
    output_dir: Path, *, object_id: int, output_gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the exact 42-PNG content gate from disk."""
    if not output_dir.is_dir():
        raise ContractError(f"CNOS child produced no output directory for object {object_id}")
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
    inventory: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for path in entries:
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.size != expected_size:
                    raise ContractError(
                        f"CNOS object {object_id} image format/frame changed: {path.name}"
                    )
                if image.mode not in allowed_modes:
                    raise ContractError(
                        f"CNOS object {object_id} image mode changed: {path.name}"
                    )
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ContractError(f"Cannot decode CNOS renderer PNG: {path}") from exc
        foreground_pixels = int(np.any(rgb != 0, axis=2).sum())
        frame_pixels = expected_size[0] * expected_size[1]
        if not 0 < foreground_pixels < frame_pixels:
            raise ContractError(
                f"CNOS object {object_id} PNG has empty/full RGB foreground: {path.name}"
            )
        digest = sha256_file(path)
        hashes.add(digest)
        inventory.append(
            {
                "relative_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "foreground_pixels": foreground_pixels,
                "foreground_coverage": foreground_pixels / frame_pixels,
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
    """Render one bound CAD while preserving the official renderer implementation."""
    validation = validate_runtime_request(
        route,
        request,
        deployment_root=deployment_root,
        repository_root=repository_root,
    )
    if gpu_override != GPU_OVERRIDE or not isinstance(gpu_override, str):
        raise ContractError("CNOS R3 child GPU override must remain string '0'")
    expected_output = (
        validation.deployment_root
        / "attempts"
        / str(request["attempt_id"])
        / "objects"
        / f"obj_{object_id:06d}"
    ).resolve()
    if output_dir.resolve() != expected_output:
        raise ContractError("CNOS R3 child output path differs from the frozen layout")
    if output_dir.exists():
        raise ContractError("CNOS R3 child output directory is create-only")
    if not output_dir.parent.is_dir():
        raise ContractError("CNOS R3 child object parent directory is missing")

    cad_path = _object_path(validation, object_id)
    renderer, helpers = module_loader(validation.cnos_checkout)
    poses = np.load(validation.poses_path, allow_pickle=False)
    if poses.shape != (VIEW_COUNT, 4, 4) or not np.isfinite(poses).all():
        raise ContractError("Pinned CNOS level-0 pose tensor shape/content changed")
    poses = np.asarray(poses, dtype=np.float64).copy()
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0

    mesh = renderer.trimesh.load_mesh(str(cad_path.resolve()))
    re_center_transform = np.eye(4)
    re_center_transform[:3, 3] = -mesh.bounding_box.centroid
    diameter = diameter_from_bound_cad_path(helpers, cad_path)
    if diameter > 100:
        mesh.apply_scale(0.001)
    render_mesh = renderer.pyrender.Mesh.from_trimesh(helpers.as_mesh(mesh))

    output_dir.mkdir(parents=False, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_override
    renderer_contract = route["wrapper_contract"]["renderer"]
    renderer.render(
        output_dir=str(output_dir),
        mesh=render_mesh,
        obj_poses=poses,
        intrinsic=np.asarray(renderer_contract["intrinsic"], dtype=np.float64),
        img_size=(
            renderer_contract["image_size"]["height"],
            renderer_contract["image_size"]["width"],
        ),
        light_itensity=renderer_contract["light_intensity"],
        re_center_transform=re_center_transform,
    )
    return validate_object_output(
        output_dir, object_id=object_id, output_gate=route["output_gate"]
    )


def build_child_environment(
    validation: RuntimeValidation, base_environment: Mapping[str, str]
) -> dict[str, str]:
    """Build the exact venv-first environment used for all five children."""
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
    """Use the explicit bound interpreter; never invoke a bare ``python``."""
    _object_path(validation, object_id)
    return [
        str(validation.python_executable),
        "-m",
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3",
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
    """Launch five isolated children and fail on any exit/output inconsistency."""
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
        raise ContractError("CNOS R3 attempt output root differs from the request identity")
    if output_root.exists():
        raise ContractError("CNOS R3 attempt output root is create-only")
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
        "source_commit": route["source"]["commit"],
        "source_tree": route["source"]["tree"],
        "venv_bin": str(validation.venv_bin),
        "venv_python_sha256": request["venv"]["python"]["sha256"],
        "gpu_override": GPU_OVERRIDE,
        "object_ids": list(OBJECT_IDS),
        "commands": commands,
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
            _write_text_log(logs_root / f"obj_{object_id:06d}.stdout.txt", result.stdout or "")
            _write_text_log(logs_root / f"obj_{object_id:06d}.stderr.txt", result.stderr or "")
            if result.returncode != 0:
                raise ContractError(
                    f"CNOS R3 renderer child failed for object {object_id}: exit {result.returncode}"
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
        raise ContractError(f"CNOS R3 renderer attempt failed: {exc}") from exc

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
