from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4 import OBJECT_IDS
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.cli import main
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.contracts import (
    BASE_IMPLEMENTATION_COMMIT,
    BOUNDARY_ZERO,
    CAD_LOCKS,
    CNOS_ARCHIVE,
    CNOS_COMMIT,
    CNOS_REPOSITORY,
    CNOS_TREE,
    FINAL_SAFE_ARCHIVE,
    POSES_LOCK,
    REQUEST_SCHEMA,
    R3_DEPENDENCY_LOCKS,
    SECOND_BLOCKER_RECEIPT,
    SOURCE_INTERFACES,
    RuntimeValidation,
    python_entry_lock,
    python_target_lock,
    validate_route,
    validate_runtime_request,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer import (
    build_child_argv,
    build_child_environment,
    prepare_mesh_for_render,
    render_one_object,
    run_render_attempt,
    validate_object_output,
)
from pose_accuracy_recovery_prep.core import (
    ContractError,
    canonical_sha256,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    ROOT / "protocols" / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r4.json"
)


def _load_route() -> dict[str, Any]:
    return json.loads(ROUTE_PATH.read_text(encoding="utf-8"))


def _relock(value: dict[str, Any], field: str) -> None:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _request(tmp_path: Path, route: dict[str, Any]) -> dict[str, Any]:
    venv_bin = tmp_path / "runtime" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_bytes(b"fixture-bound-venv-python\n")
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY_DESCRIPTOR_RENDERER",
        "route_lock_sha256": route["route_lock_sha256"],
        "request_lock_sha256": "pending",
        "attempt_id": "poseloop_ga_cnos_v1r4_fixture",
        "implementation": {
            "checkout_relative_path": "sources/poseloop",
            "approved_commit": "a" * 40,
            "approved_tree": "b" * 40,
        },
        "paths": {
            "cnos_source_archive": CNOS_ARCHIVE["relative_path"],
            "cnos_source_checkout": "sources/cnos",
            "object_poses_level0": (
                "sources/cnos/src/poses/predefined_poses/obj_poses_level0.npy"
            ),
            "predecessor_second_blocker_receipt": SECOND_BLOCKER_RECEIPT[
                "relative_path"
            ],
            "predecessor_final_safe_archive": FINAL_SAFE_ARCHIVE["relative_path"],
        },
        "venv": {
            "bin_path": str(venv_bin),
            "python_entry": python_entry_lock(python),
            "python_target": python_target_lock(python),
        },
        "gpu_override": "0",
        "objects": [
            {
                "object_id": item["object_id"],
                "cad_relative_path": item["relative_path"],
            }
            for item in CAD_LOCKS
        ],
        "boundary": dict(BOUNDARY_ZERO),
    }
    _relock(value, "request_lock_sha256")
    return value


def _runtime_validation(
    tmp_path: Path, route: dict[str, Any], request: dict[str, Any]
) -> RuntimeValidation:
    root = tmp_path.resolve()
    implementation = root / "sources" / "poseloop"
    cnos = root / "sources" / "cnos"
    implementation.mkdir(parents=True, exist_ok=True)
    cnos.mkdir(parents=True, exist_ok=True)
    poses = cnos / "src" / "poses" / "predefined_poses" / "obj_poses_level0.npy"
    poses.parent.mkdir(parents=True, exist_ok=True)
    poses.write_bytes(b"fixture poses")
    objects: list[tuple[int, Path]] = []
    for item in CAD_LOCKS:
        path = root / Path(*str(item["relative_path"]).split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture cad {item['object_id']}".encode())
        objects.append((int(item["object_id"]), path))
    entry = Path(request["venv"]["python_entry"]["absolute_path"])
    return RuntimeValidation(
        route=route,
        request=request,
        deployment_root=root,
        implementation_checkout=implementation,
        cnos_checkout=cnos,
        poses_path=poses,
        venv_bin=Path(request["venv"]["bin_path"]),
        python_entry=entry,
        python_target=entry.resolve(),
        objects=tuple(objects),
    )


def _write_png_set(output_dir: Path, object_id: int) -> None:
    output_dir.mkdir(parents=False, exist_ok=True)
    for index in range(42):
        image = Image.new("RGBA", (640, 480), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        x = 20 + (index % 20)
        y = 30 + object_id
        draw.rectangle(
            (x, y, x + 40, y + 30),
            fill=(object_id * 20, 90, 180, 255),
        )
        image.save(output_dir / f"{index:06d}.png", format="PNG")


class _FakeBoundingBox:
    def __init__(self, mesh: _FakeMesh) -> None:
        self.mesh = mesh

    @property
    def centroid(self) -> np.ndarray:
        self.mesh.events.append("centroid")
        return self.mesh.centroid.copy()


class _FakeMesh:
    def __init__(self, centroid: list[float], events: list[str]) -> None:
        self.centroid = np.asarray(centroid, dtype=np.float64)
        self.events = events
        self.bounding_box = _FakeBoundingBox(self)

    def apply_scale(self, value: float) -> None:
        self.events.append(f"scale:{value}")
        self.centroid *= value


def _fake_modules(
    mesh: _FakeMesh, events: list[str], diameter: float
) -> tuple[Any, Any]:
    def load_mesh(path: str) -> _FakeMesh:
        events.append(f"load:{path}")
        return mesh

    def from_trimesh(value: Any) -> Any:
        events.append("from_trimesh")
        return value

    def get_obj_diameter(path: str) -> float:
        events.append(f"diameter:{path}")
        return diameter

    def as_mesh(value: Any) -> Any:
        events.append("as_mesh")
        return value

    renderer = SimpleNamespace(
        trimesh=SimpleNamespace(load_mesh=load_mesh),
        pyrender=SimpleNamespace(Mesh=SimpleNamespace(from_trimesh=from_trimesh)),
    )
    helpers = SimpleNamespace(
        get_obj_diameter=get_obj_diameter,
        as_mesh=as_mesh,
    )
    return renderer, helpers


def test_route_self_lock_binds_failures_source_and_all_r3_bytes() -> None:
    route = _load_route()
    result = validate_route(route, repository_root=ROOT)
    assert result["route_lock_sha256"] == route["route_lock_sha256"]
    assert result["object_count"] == 5
    assert result["expected_png_count"] == 210
    assert route["predecessor"]["attempt_002"]["observed"] == {
        "png_count": 42,
        "empty_or_transparent_count": 40,
        "visible_count": 2,
        "visible_pixels_each": [4, 4],
        "unique_png_sha256_count": 3,
    }
    assert route["r3_dependency_files"] == [dict(item) for item in R3_DEPENDENCY_LOCKS]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda route: route["predecessor"]["attempt_002"]["blocker_receipt"].update(
            {"sha256": "f" * 64}
        ),
        lambda route: route["predecessor"]["attempt_002"]["observed"].update(
            {"empty_or_transparent_count": 39}
        ),
        lambda route: route["repair_contract"]["mesh_units"].update(
            {"millimetre_scale_factor": 1.0}
        ),
        lambda route: route["official_source"]["interface_files"][0][
            "checkout_file"
        ].update({"sha256": "e" * 64}),
        lambda route: route["r3_dependency_files"][0].update({"bytes": 1}),
    ],
)
def test_route_rejects_relocked_evidence_repair_source_or_r3_tamper(
    mutation: Any,
) -> None:
    route = _load_route()
    mutation(route)
    _relock(route, "route_lock_sha256")
    with pytest.raises(ContractError):
        validate_route(route)


@pytest.mark.parametrize(
    ("diameter", "expected_scale", "expected_translation", "scale_event"),
    [
        (200.0, 0.001, [-0.8, 0.02, -0.03], "scale:0.001"),
        (99.0, 1.0, [-800.0, 20.0, -30.0], None),
    ],
)
def test_scale_then_recenter_is_exact_and_diameter_uses_bound_path(
    tmp_path: Path,
    diameter: float,
    expected_scale: float,
    expected_translation: list[float],
    scale_event: str | None,
) -> None:
    cad = tmp_path / "off_origin.ply"
    cad.write_bytes(b"bound cad fixture")
    events: list[str] = []
    mesh = _FakeMesh([800.0, -20.0, 30.0], events)
    renderer, helpers = _fake_modules(mesh, events, diameter)
    prepared = prepare_mesh_for_render(renderer, helpers, cad)
    assert prepared.applied_scale == expected_scale
    assert prepared.recenter_transform[:3, 3] == pytest.approx(expected_translation)
    diameter_event = f"diameter:{cad.resolve()}"
    assert diameter_event in events
    assert events.index(diameter_event) < events.index("centroid")
    if scale_event is None:
        assert not any(event.startswith("scale:") for event in events)
    else:
        assert events.index(scale_event) < events.index("centroid")
    assert events.index("centroid") < events.index("as_mesh")


def test_render_call_receives_scaled_recenter_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer.validate_runtime_request",
        lambda *args, **kwargs: validation,
    )
    poses = np.repeat(np.eye(4)[None, :, :], 42, axis=0)
    poses[:, 2, 3] = 1000.0
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer.np.load",
        lambda *args, **kwargs: poses,
    )
    events: list[str] = []
    mesh = _FakeMesh([800.0, -20.0, 30.0], events)
    renderer, helpers = _fake_modules(mesh, events, 200.0)

    def render(**kwargs: Any) -> None:
        events.append("render")
        assert kwargs["re_center_transform"][:3, 3] == pytest.approx(
            [-0.8, 0.02, -0.03]
        )
        _write_png_set(Path(kwargs["output_dir"]), 1)

    renderer.render = render
    output_parent = tmp_path / "attempts" / request["attempt_id"] / "objects"
    output_parent.mkdir(parents=True)
    result = render_one_object(
        route=route,
        request=request,
        deployment_root=tmp_path,
        repository_root=ROOT,
        object_id=1,
        output_dir=output_parent / "obj_000001",
        gpu_override="0",
        module_loader=lambda path: (renderer, helpers),
    )
    assert (
        events.index("scale:0.001") < events.index("centroid") < events.index("render")
    )
    assert result["png_count"] == 42
    assert result["mesh_preparation"]["recenter_translation"] == pytest.approx(
        [-0.8, 0.02, -0.03]
    )


def _projected_intersection_area(vertices: np.ndarray, recenter: np.ndarray) -> int:
    pose = np.eye(4)
    pose[2, 3] = 1.0
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    camera = (pose @ recenter @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > 1e-6
    if not valid.any():
        return 0
    camera = camera[valid]
    u = 572.4114 * camera[:, 0] / camera[:, 2] + 325.2611
    v = 573.57043 * camera[:, 1] / camera[:, 2] + 242.04899
    left = max(0.0, float(u.min()))
    right = min(640.0, float(u.max()))
    top = max(0.0, float(v.min()))
    bottom = min(480.0, float(v.max()))
    if right <= left or bottom <= top:
        return 0
    return int((right - left) * (bottom - top))


def test_off_origin_mm_projection_proves_old_empty_new_nonempty(
    tmp_path: Path,
) -> None:
    original = trimesh.creation.box(extents=[100.0, 80.0, 60.0])
    original.apply_translation([800.0, 20.0, 0.0])
    cad = tmp_path / "off_origin_mm.ply"
    original.export(cad)
    renderer = SimpleNamespace(
        trimesh=SimpleNamespace(load_mesh=trimesh.load_mesh),
        pyrender=SimpleNamespace(Mesh=SimpleNamespace(from_trimesh=lambda mesh: mesh)),
    )
    helpers = SimpleNamespace(
        get_obj_diameter=lambda path: 200.0,
        as_mesh=lambda mesh: mesh,
    )
    prepared = prepare_mesh_for_render(renderer, helpers, cad)
    vertices_scaled = np.asarray(original.vertices) * 0.001
    old = np.eye(4)
    old[:3, 3] = -np.asarray(original.bounding_box.centroid)
    frame_pixels = 640 * 480
    old_area = _projected_intersection_area(vertices_scaled, old)
    new_area = _projected_intersection_area(
        vertices_scaled, prepared.recenter_transform
    )
    assert old_area == 0
    assert 0 < new_area < frame_pixels


def test_png_gate_accepts_rgba_and_rejects_rgb_transparent_or_corrupt(
    tmp_path: Path,
) -> None:
    route = _load_route()
    output = tmp_path / "obj_000001"
    _write_png_set(output, 1)
    result = validate_object_output(
        output, object_id=1, output_gate=route["output_gate"]
    )
    assert result["png_count"] == 42
    assert result["inventory"][0]["alpha_foreground_pixels"] == 41 * 31
    with Image.open(output / "000000.png") as image:
        image.convert("RGB").save(output / "000000.png", format="PNG")
    with pytest.raises(ContractError, match="image mode changed"):
        validate_object_output(output, object_id=1, output_gate=route["output_gate"])
    _write_png_set(output, 1)
    transparent = Image.new("RGBA", (640, 480), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).rectangle((10, 10, 20, 20), fill=(50, 60, 70, 0))
    transparent.save(output / "000000.png")
    with pytest.raises(ContractError, match="alpha foreground"):
        validate_object_output(output, object_id=1, output_gate=route["output_gate"])
    (output / "000000.png").write_bytes(b"not a PNG")
    with pytest.raises(ContractError, match="decode"):
        validate_object_output(output, object_id=1, output_gate=route["output_gate"])


def test_bound_symlink_entry_is_executed_not_resolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "python-base-target"
    target.write_bytes(b"python binary fixture")
    bin_path = tmp_path / "venv" / "bin"
    bin_path.mkdir(parents=True)
    entry = bin_path / "python"
    entry.write_bytes(b"lexical entry fixture")
    real_is_symlink = Path.is_symlink
    real_readlink = os.readlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: True if path == entry else real_is_symlink(path),
    )
    monkeypatch.setattr(
        os,
        "readlink",
        lambda path: str(target) if Path(path) == entry else real_readlink(path),
    )
    request = {
        "venv": {
            "bin_path": str(bin_path),
            "python_entry": python_entry_lock(entry),
            "python_target": {
                "absolute_path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            },
        }
    }
    validation = RuntimeValidation(
        route={},
        request=request,
        deployment_root=tmp_path,
        implementation_checkout=tmp_path / "implementation",
        cnos_checkout=tmp_path / "cnos",
        poses_path=tmp_path / "poses.npy",
        venv_bin=bin_path,
        python_entry=entry,
        python_target=target.resolve(),
        objects=((1, tmp_path / "obj_000001.ply"),),
    )
    argv = build_child_argv(
        validation,
        route_path=ROUTE_PATH,
        request_path=tmp_path / "request.json",
        repository_root=ROOT,
        object_id=1,
        output_dir=tmp_path / "attempts" / "a" / "objects" / "obj_000001",
    )
    assert argv[0] == str(entry)
    assert argv[0] != str(target.resolve())
    assert request["venv"]["python_entry"]["kind"] == "symlink"
    assert request["venv"]["python_target"]["sha256"] == sha256_file(target)


def test_child_environment_pins_entry_bin_egl_gpu_and_pythonpath(
    tmp_path: Path,
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    environment = build_child_environment(
        validation,
        {
            "PATH": "system-path",
            "PYTHONPATH": "old-pythonpath",
            "PYOPENGL_PLATFORM": "wrong",
        },
    )
    assert environment["PATH"].split(os.pathsep)[0] == str(validation.venv_bin)
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
        validation.implementation_checkout
    )
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_render_attempt_covers_exact_five_objects_and_210_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer.validate_runtime_request",
        lambda *args, **kwargs: validation,
    )
    output_root = tmp_path / "attempts" / request["attempt_id"]
    observed: list[int] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert (output_root / "attempt-start.json").is_file()
        object_id = int(argv[argv.index("--object-id") + 1])
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        _write_png_set(output_dir, object_id)
        observed.append(object_id)
        return subprocess.CompletedProcess(argv, 0, "child-pass\n", "")

    receipt = run_render_attempt(
        route=route,
        request=request,
        route_path=ROUTE_PATH,
        request_path=tmp_path / "request.json",
        deployment_root=tmp_path,
        repository_root=ROOT,
        output_root=output_root,
        runner=runner,
        base_environment={"PATH": "system"},
    )
    assert observed == list(OBJECT_IDS)
    assert receipt["status"] == "PASS"
    assert receipt["object_count"] == 5
    assert receipt["png_count"] == 210
    assert receipt["producer_started"] is False
    assert receipt["boundary"] == BOUNDARY_ZERO
    assert (output_root / "attempt-receipt.json").is_file()


@pytest.mark.parametrize("return_code", [0, 9])
def test_parent_fails_closed_on_missing_pngs_or_nonzero_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer.validate_runtime_request",
        lambda *args, **kwargs: validation,
    )
    output_root = tmp_path / "attempts" / request["attempt_id"]

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, return_code, "", "failed")

    with pytest.raises(ContractError):
        run_render_attempt(
            route=route,
            request=request,
            route_path=ROUTE_PATH,
            request_path=tmp_path / "request.json",
            deployment_root=tmp_path,
            repository_root=ROOT,
            output_root=output_root,
            runner=runner,
        )
    failure = json.loads((output_root / "attempt-failure.json").read_text())
    assert failure["status"] == "FAILED_CREATE_ONLY_ATTEMPT"
    assert failure["rerun_same_output_permitted"] is False


def test_attempt_root_is_create_only_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.renderer.validate_runtime_request",
        lambda *args, **kwargs: validation,
    )
    output_root = tmp_path / "attempts" / request["attempt_id"]
    output_root.mkdir(parents=True)
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("preserve\n")
    with pytest.raises(ContractError, match="create-only"):
        run_render_attempt(
            route=route,
            request=request,
            route_path=ROUTE_PATH,
            request_path=tmp_path / "request.json",
            deployment_root=tmp_path,
            repository_root=ROOT,
            output_root=output_root,
        )
    assert sentinel.read_text() == "preserve\n"


@pytest.fixture
def runtime_disk_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    route = _load_route()
    request = _request(tmp_path, route)
    expected_hashes: dict[Path, str] = {}

    def materialize(relative: str, size: int, digest: str) -> Path:
        path = tmp_path / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(size)
        expected_hashes[path.resolve()] = digest
        return path

    materialize(
        str(CNOS_ARCHIVE["relative_path"]),
        int(CNOS_ARCHIVE["bytes"]),
        str(CNOS_ARCHIVE["sha256"]),
    )
    materialize(
        str(SECOND_BLOCKER_RECEIPT["relative_path"]),
        int(SECOND_BLOCKER_RECEIPT["bytes"]),
        str(SECOND_BLOCKER_RECEIPT["sha256"]),
    )
    materialize(
        str(FINAL_SAFE_ARCHIVE["relative_path"]),
        int(FINAL_SAFE_ARCHIVE["bytes"]),
        str(FINAL_SAFE_ARCHIVE["sha256"]),
    )
    for item in CAD_LOCKS:
        materialize(str(item["relative_path"]), int(item["bytes"]), str(item["sha256"]))
    cnos = tmp_path / "sources" / "cnos"
    for interface in SOURCE_INTERFACES:
        lock = interface["checkout_file"]
        relative = str(interface["relative_path"])
        path = cnos / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(int(lock["bytes"]))
        expected_hashes[path.resolve()] = str(lock["sha256"])
    poses = cnos / Path(*str(POSES_LOCK["relative_path"]).split("/"))
    poses.parent.mkdir(parents=True, exist_ok=True)
    with poses.open("wb") as stream:
        stream.truncate(int(POSES_LOCK["bytes"]))
    expected_hashes[poses.resolve()] = str(POSES_LOCK["sha256"])
    (tmp_path / "sources" / "poseloop").mkdir(parents=True)

    real_sha = sha256_file

    def fake_sha(path: Path) -> str:
        return expected_hashes.get(Path(path).resolve(), real_sha(Path(path)))

    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts.sha256_file",
        fake_sha,
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts._validate_source_archive_members",
        lambda archive: None,
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.contracts._validate_final_archive_members",
        lambda archive: None,
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.contracts.v1.git_identity",
        lambda path: (
            {"commit": CNOS_COMMIT, "tree": CNOS_TREE, "clean": True}
            if path.name == "cnos"
            else {"commit": "a" * 40, "tree": "b" * 40, "clean": True}
        ),
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.contracts.v1.git_is_ancestor",
        lambda checkout, ancestor, descendant: (
            ancestor == BASE_IMPLEMENTATION_COMMIT and descendant == "a" * 40
        ),
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r4.contracts.v1.git_remote_repository",
        lambda checkout: CNOS_REPOSITORY,
    )
    return route, request


def test_runtime_request_accepts_exact_assets_and_preserves_entry(
    tmp_path: Path,
    runtime_disk_fixture: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    route, request = runtime_disk_fixture
    validation = validate_runtime_request(route, request, deployment_root=tmp_path)
    assert [item[0] for item in validation.objects] == list(OBJECT_IDS)
    assert validation.python_entry == Path(
        request["venv"]["python_entry"]["absolute_path"]
    )
    assert validation.python_target == Path(
        request["venv"]["python_target"]["absolute_path"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update({"gpu_override": 0}),
        lambda request: request["objects"].__setitem__(0, request["objects"][1]),
        lambda request: request["paths"].update(
            {"cnos_source_checkout": "sources/depth/runtime"}
        ),
        lambda request: request["boundary"].update({"gt_path_open_count": 1}),
        lambda request: request["venv"]["python_entry"].update(
            {
                "absolute_path": str(
                    Path(request["venv"]["python_entry"]["absolute_path"]).with_name(
                        "python-swapped"
                    )
                )
            }
        ),
        lambda request: request["venv"]["python_target"].update({"sha256": "f" * 64}),
    ],
)
def test_runtime_request_rejects_type_path_boundary_entry_or_target_tamper(
    tmp_path: Path,
    runtime_disk_fixture: tuple[dict[str, Any], dict[str, Any]],
    mutation: Any,
) -> None:
    route, request = runtime_disk_fixture
    mutation(request)
    _relock(request, "request_lock_sha256")
    with pytest.raises(ContractError):
        validate_runtime_request(route, request, deployment_root=tmp_path)


def test_runtime_request_rejects_stale_self_lock_before_disk_access(
    tmp_path: Path,
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    request["attempt_id"] = "poseloop_ga_cnos_v1r4_changed"
    with pytest.raises(ContractError, match="self-lock"):
        validate_runtime_request(route, request, deployment_root=tmp_path)


def test_cli_help_exposes_only_review_preflight_and_create_only_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "validate-route" in output
    assert "preflight" in output
    assert "render-all" in output
    assert "render-one" in output
    assert "FastSAM" in output
