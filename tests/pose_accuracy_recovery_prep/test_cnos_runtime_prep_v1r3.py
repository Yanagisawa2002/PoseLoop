from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import trimesh
from PIL import Image, ImageDraw

from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3 import OBJECT_IDS
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.cli import main
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts import (
    BASE_IMPLEMENTATION_COMMIT,
    BLOCKER_RECEIPT,
    BOUNDARY_ZERO,
    CAD_LOCKS,
    CNOS_ARCHIVE,
    CNOS_COMMIT,
    CNOS_REPOSITORY,
    CNOS_TREE,
    POSES_LOCK,
    REQUEST_SCHEMA,
    SAFE_ARCHIVE,
    SOURCE_INTERFACES,
    RuntimeValidation,
    validate_route,
    validate_runtime_request,
)
from pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.renderer import (
    build_child_argv,
    build_child_environment,
    diameter_from_bound_cad_path,
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
    ROOT / "protocols" / "poseloop_pose_accuracy_recovery_cnos_runtime_prep_v1r3.json"
)
PINNED_HELPER = (
    Path(__file__).parent
    / "fixtures"
    / "cnos_runtime_prep_v1r3"
    / "trimesh_utils.py"
)


def _load_route() -> dict[str, Any]:
    return json.loads(ROUTE_PATH.read_text(encoding="utf-8"))


def _relock(value: dict[str, Any], field: str) -> None:
    value[field] = canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _load_pinned_helper() -> Any:
    spec = importlib.util.spec_from_file_location("pinned_cnos_trimesh_utils", PINNED_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(tmp_path: Path, route: dict[str, Any]) -> dict[str, Any]:
    venv_bin = (tmp_path / "runtime" / "venv" / "bin").resolve()
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_bytes(b"fixture-bound-venv-python\n")
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "role": "DEVELOPMENT_ONLY_DESCRIPTOR_RENDERER",
        "route_lock_sha256": route["route_lock_sha256"],
        "request_lock_sha256": "pending",
        "attempt_id": "poseloop_ga_cnos_v1r3_fixture",
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
            "predecessor_blocker_receipt": BLOCKER_RECEIPT["relative_path"],
            "predecessor_safe_archive": SAFE_ARCHIVE["relative_path"],
        },
        "venv": {
            "bin_path": str(venv_bin),
            "python": {
                "absolute_path": str(python),
                "bytes": python.stat().st_size,
                "sha256": sha256_file(python),
            },
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
    return RuntimeValidation(
        route=route,
        request=request,
        deployment_root=root,
        implementation_checkout=implementation,
        cnos_checkout=cnos,
        poses_path=poses,
        venv_bin=Path(request["venv"]["bin_path"]),
        python_executable=Path(request["venv"]["python"]["absolute_path"]),
        objects=tuple(objects),
    )


def _write_png_set(output_dir: Path, object_id: int) -> None:
    output_dir.mkdir(parents=False, exist_ok=False)
    for index in range(42):
        image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        x = 20 + (index % 20)
        y = 30 + object_id
        draw.rectangle((x, y, x + 40, y + 30), fill=(object_id * 20, 90, 180))
        image.save(output_dir / f"{index:06d}.png", format="PNG")


def test_route_self_lock_source_dual_bytes_and_frozen_dependencies() -> None:
    route = _load_route()
    validation = validate_route(route, repository_root=ROOT)
    assert validation["route_lock_sha256"] == route["route_lock_sha256"]
    assert validation["object_count"] == 5
    assert validation["expected_png_count"] == 210
    interfaces = {item["relative_path"]: item for item in route["source"]["interface_files"]}
    assert interfaces["src/poses/pyrender.py"]["archive_member"]["sha256"] == (
        "646a7c32ca31a855520fe97387c0ee544ce305a98b8e4740b60d4f79613d46a1"
    )
    assert interfaces["src/poses/pyrender.py"]["checkout_file"]["sha256"] == (
        "8dbe52bbb7c6d2fc419786ce1d8602fca4fe82ce6d80af708ff85feff31ed215"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda route: route["predecessor"]["blocker"]["receipt"].update(
            {"sha256": "f" * 64}
        ),
        lambda route: route["source"]["interface_files"][0]["checkout_file"].update(
            {"sha256": "e" * 64}
        ),
        lambda route: route["objects"].__setitem__(0, route["objects"][1]),
        lambda route: route["r2_dependency_files"][0].update({"bytes": 1}),
    ],
)
def test_route_rejects_relocked_predecessor_source_cad_or_dependency_tamper(
    mutation: Any,
) -> None:
    route = _load_route()
    mutation(route)
    _relock(route, "route_lock_sha256")
    with pytest.raises(ContractError):
        validate_route(route)


def test_pinned_helper_reproduces_mesh_failure_and_path_wrapper_succeeds(
    tmp_path: Path,
) -> None:
    assert PINNED_HELPER.stat().st_size == 1983
    assert sha256_file(PINNED_HELPER) == (
        "d2111f22b1266466e67b5917996b217d7a0923790a95a9ac83283841f1d07a6d"
    )
    helper = _load_pinned_helper()
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    with pytest.raises(NotImplementedError, match="file_type 'None' not supported"):
        helper.get_obj_diameter(mesh)
    cad_path = tmp_path / "box.ply"
    mesh.export(cad_path)
    assert diameter_from_bound_cad_path(helper, cad_path) == pytest.approx(
        helper.get_obj_diameter(str(cad_path.resolve()))
    )


def test_child_argv_and_environment_pin_venv_python_gpu_string_and_pythonpath(
    tmp_path: Path,
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    output = tmp_path / "attempts" / request["attempt_id"] / "objects" / "obj_000001"
    argv = build_child_argv(
        validation,
        route_path=ROUTE_PATH,
        request_path=tmp_path / "request.json",
        repository_root=ROOT,
        object_id=1,
        output_dir=output,
    )
    assert argv[0] == str(validation.python_executable)
    assert argv[0] != "python"
    assert argv[argv.index("--gpu-override") + 1] == "0"
    assert isinstance(argv[argv.index("--gpu-override") + 1], str)
    environment = build_child_environment(
        validation, {"PATH": "system-path", "PYTHONPATH": "old-pythonpath"}
    )
    assert environment["PATH"].split(os.pathsep)[0] == str(validation.venv_bin)
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
        validation.implementation_checkout
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_validate_object_output_accepts_exact_content_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    route = _load_route()
    output = tmp_path / "obj_000001"
    _write_png_set(output, 1)
    result = validate_object_output(
        output, object_id=1, output_gate=route["output_gate"]
    )
    assert result["png_count"] == 42
    assert result["unique_png_sha256_count"] >= 2
    (output / "000017.png").write_bytes(b"not a PNG")
    with pytest.raises(ContractError, match="decode"):
        validate_object_output(output, object_id=1, output_gate=route["output_gate"])


def test_render_attempt_covers_exact_five_objects_and_210_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.renderer.validate_runtime_request",
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
def test_parent_fails_closed_on_child_exit_zero_without_pngs_or_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
) -> None:
    route = _load_route()
    request = _request(tmp_path, route)
    validation = _runtime_validation(tmp_path, route, request)
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.renderer.validate_runtime_request",
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
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.renderer.validate_runtime_request",
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
        str(BLOCKER_RECEIPT["relative_path"]),
        int(BLOCKER_RECEIPT["bytes"]),
        str(BLOCKER_RECEIPT["sha256"]),
    )
    materialize(
        str(SAFE_ARCHIVE["relative_path"]),
        int(SAFE_ARCHIVE["bytes"]),
        str(SAFE_ARCHIVE["sha256"]),
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
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts.git_identity",
        lambda path: (
            {"commit": CNOS_COMMIT, "tree": CNOS_TREE, "clean": True}
            if path.name == "cnos"
            else {"commit": "a" * 40, "tree": "b" * 40, "clean": True}
        ),
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts.git_is_ancestor",
        lambda checkout, ancestor, descendant: (
            ancestor == BASE_IMPLEMENTATION_COMMIT and descendant == "a" * 40
        ),
    )
    monkeypatch.setattr(
        "pose_accuracy_recovery_prep.cnos_runtime_prep_v1r3.contracts.git_remote_repository",
        lambda checkout: CNOS_REPOSITORY,
    )
    return route, request


def test_runtime_request_accepts_only_exact_five_objects_and_bound_files(
    tmp_path: Path,
    runtime_disk_fixture: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    route, request = runtime_disk_fixture
    validation = validate_runtime_request(route, request, deployment_root=tmp_path)
    assert [item[0] for item in validation.objects] == list(OBJECT_IDS)
    assert validation.python_executable == Path(
        request["venv"]["python"]["absolute_path"]
    ).resolve()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update({"gpu_override": 0}),
        lambda request: request["objects"].__setitem__(0, request["objects"][1]),
        lambda request: request["paths"].update(
            {"cnos_source_checkout": "sources/depth/runtime"}
        ),
        lambda request: request["boundary"].update({"gt_path_open_count": 1}),
    ],
)
def test_runtime_request_rejects_gpu_type_swap_forbidden_path_or_boundary_tamper(
    tmp_path: Path,
    runtime_disk_fixture: tuple[dict[str, Any], dict[str, Any]],
    mutation: Any,
) -> None:
    route, request = runtime_disk_fixture
    mutation(request)
    _relock(request, "request_lock_sha256")
    with pytest.raises(ContractError):
        validate_runtime_request(route, request, deployment_root=tmp_path)


def test_cli_help_has_review_preflight_and_create_only_commands(
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
