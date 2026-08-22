"""Official FoundationPose Python adapter with non-synthetic evidence capture."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .a_output import VISUALIZATION_ROLES, validate_internal_live_success
from .common import (
    PREP_PROTOCOL_ID,
    RESULT_SCHEMA,
    PrepError,
    canonical_sha256,
    sha256_file,
    write_bytes_atomic,
)
from .live_contract import LIVE_BACKEND_ID
from .manifest import asset_path
from .producer import FIXED_INFERENCE
from .visualization import _decode_mask, _overlay


def _forbidden_camera_key(value: Any, location: str = "camera") -> None:
    forbidden = {
        "gt",
        "groundtruth",
        "ground_truth",
        "scene_gt",
        "evaluator",
        "evaluation",
        "score",
        "sealed",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_").replace(".", "_")
            tokens = {token for token in normalized.split("_") if token}
            if normalized in forbidden or tokens & forbidden:
                raise PrepError(
                    f"Public camera asset exposes a forbidden field: {location}.{key}"
                )
            _forbidden_camera_key(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_camera_key(child, f"{location}[{index}]")


def _camera_intrinsics(path: Path, expected: Any, image_id: int, np: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError(f"Cannot decode public camera JSON: {path.name}") from exc
    _forbidden_camera_key(value)
    candidate = value
    if isinstance(value, dict) and str(image_id) in value:
        candidate = value[str(image_id)]
    if not isinstance(candidate, dict):
        raise PrepError("Public camera JSON is not an object")
    known = [key for key in ("K", "cam_K", "camera_intrinsics") if key in candidate]
    if len(known) != 1:
        raise PrepError("Public camera JSON must expose exactly one intrinsics field")
    raw = candidate[known[0]]
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.size == 9:
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise PrepError("Public camera intrinsics are invalid")
    frozen = np.asarray(expected, dtype=np.float64)
    if not np.array_equal(matrix, frozen):
        raise PrepError("Public camera JSON intrinsics differ from the manifest")
    return matrix


def load_live_input(*, item: Mapping[str, Any], asset_root: Path) -> dict[str, Any]:
    """Read only the five hash-listed input roles for one normalized V2 item."""

    try:
        import cv2
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise PrepError("Live input decoding requires cv2, numpy, and trimesh") from exc

    paths = {
        slot: asset_path(asset_root, item["inputs"][slot])
        for slot in ("rgb", "depth", "mask", "camera", "cad")
    }
    required_suffixes = {
        "rgb": {".png"},
        "depth": {".png"},
        "mask": {".png", ".json"},
        "camera": {".json"},
        "cad": {".ply"},
    }
    for slot, allowed in required_suffixes.items():
        if paths[slot].suffix.lower() not in allowed:
            raise PrepError(f"Live {slot} asset type is not allowlisted")
    bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_raw is None:
        raise PrepError("OpenCV cannot decode public RGB/depth")
    height = int(item["frame_size"]["height"])
    width = int(item["frame_size"]["width"])
    if bgr.shape[:2] != (height, width) or depth_raw.shape[:2] != (height, width):
        raise PrepError("Public RGB/depth frame size differs from the manifest")
    if depth_raw.ndim != 2:
        raise PrepError("Public depth image must be single channel")
    depth_scale = float(item["depth_scale"])
    depth_m = np.ascontiguousarray(depth_raw.astype(np.float32) * depth_scale)
    if not np.isfinite(depth_m).all() or float(depth_m.max(initial=0.0)) <= 0:
        raise PrepError("Public depth contains no finite positive measurement")
    mask = np.ascontiguousarray(
        _decode_mask(paths["mask"], item["mask_provenance"]["plugin_id"], cv2, np),
        dtype=bool,
    )
    if mask.shape != (height, width):
        raise PrepError("Input mask frame size differs from the manifest")
    nonzero = int(mask.sum())
    coverage = item["mask_provenance"]["coverage"]
    if nonzero != int(coverage["nonzero_pixels"]) or not math.isclose(
        nonzero / float(mask.size),
        float(coverage["fraction"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PrepError("Input mask coverage differs from the manifest")
    valid = mask & np.isfinite(depth_m) & (depth_m >= 0.001)
    if int(valid.sum()) < 4:
        raise PrepError("Input mask contains fewer than four valid depth pixels")
    median_depth = float(np.median(depth_m[valid]))
    if not 0.05 < median_depth < 5.0:
        raise PrepError("Masked public depth is outside the metre sanity range")
    intrinsics = _camera_intrinsics(
        paths["camera"],
        item["camera_intrinsics"],
        int(item["sample_key"]["image_id"]),
        np,
    )
    mesh = trimesh.load(paths["cad"], force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise PrepError("Public CAD does not contain a triangle mesh")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if not np.isfinite(extents).all() or not np.all(extents > 1.0):
        raise PrepError("Public CAD is not in the frozen millimetre unit convention")
    mesh.apply_scale(0.001)
    if not 0.001 < float(np.max(mesh.extents)) < 2.0:
        raise PrepError("Scaled public CAD extents are implausible")
    _ = mesh.vertex_normals
    return {
        "bgr": np.ascontiguousarray(bgr),
        "rgb": np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
        "depth_m": depth_m,
        "mask": mask,
        "camera_intrinsics": intrinsics,
        "mesh": mesh,
        "paths": paths,
    }


def _numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "data") and hasattr(value.data, "cpu"):
        value = value.data.cpu().numpy()
    return np.asarray(value).copy()


def _pose_list(value: Any, np: Any) -> list[list[float]]:
    matrix = np.asarray(_numpy(value, np), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PrepError("FoundationPose adapter captured an invalid pose")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7, rtol=0.0):
        raise PrepError("FoundationPose adapter pose homogeneous row differs")
    return [[float(cell) for cell in row] for row in matrix]


def _mesh_render_data(mesh: Any, np: Any) -> tuple[Any, Any]:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise PrepError("Public CAD has no renderable edges")
    edges = np.sort(edges, axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    if len(edges) > 4000:
        edges = edges[np.linspace(0, len(edges) - 1, 4000, dtype=np.int64)]
    return np.asarray(mesh.vertices, dtype=np.float64), edges


def _png(path: Path, image: Any, cv2: Any) -> dict[str, Any]:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise PrepError(f"OpenCV cannot encode live visualization: {path.name}")
    write_bytes_atomic(path, bytes(encoded))
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_live_visualizations(
    *,
    item: Mapping[str, Any],
    inputs: Mapping[str, Any],
    initial_pose: Any,
    final_pose: Any,
    top_k: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise PrepError("Live visualization requires cv2 and numpy") from exc
    item_id = str(item["item_id"])
    root = output_root.resolve()
    vertices, edges = _mesh_render_data(inputs["mesh"], np)
    intrinsics = np.asarray(inputs["camera_intrinsics"], dtype=np.float64)
    initial = _overlay(
        bgr=inputs["bgr"],
        mask=inputs["mask"],
        vertices_m=vertices,
        edges=edges,
        pose=np.asarray(initial_pose, dtype=np.float64),
        intrinsics=intrinsics,
        cv2=cv2,
        np=np,
    )
    final = _overlay(
        bgr=inputs["bgr"],
        mask=inputs["mask"],
        vertices_m=vertices,
        edges=edges,
        pose=np.asarray(final_pose, dtype=np.float64),
        intrinsics=intrinsics,
        cv2=cv2,
        np=np,
    )
    top_overlays = [
        _overlay(
            bgr=inputs["bgr"],
            mask=inputs["mask"],
            vertices_m=vertices,
            edges=edges,
            pose=np.asarray(candidate["model_to_camera_pose_m"], dtype=np.float64),
            intrinsics=intrinsics,
            cv2=cv2,
            np=np,
        )
        for candidate in top_k
    ]
    top_composite = top_overlays[0]
    for overlay in top_overlays[1:]:
        top_composite = cv2.addWeighted(top_composite, 0.8, overlay, 0.2, 0.0)
    images = {
        "rgb": inputs["bgr"],
        "input_mask": (inputs["mask"].astype(np.uint8) * 255),
        "initial_pose_overlay": initial,
        "top_k_overlay": top_composite,
        "final_pose_overlay": final,
    }
    inventory: list[dict[str, Any]] = []
    for role in sorted(VISUALIZATION_ROLES):
        relative = PurePosixPath("visualizations", item_id, f"{role}.png")
        path = root / Path(*relative.parts)
        evidence = _png(path, images[role], cv2)
        inventory.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                **evidence,
            }
        )
    return inventory


class OfficialFoundationPoseBackend:
    """One process-local FoundationPose backend retaining the upstream algorithm."""

    def __init__(self, runtime_lock: Mapping[str, Any], *, output_root: Path):
        try:
            import numpy as np
            import torch
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise PrepError("Live FoundationPose CUDA imports are unavailable") from exc
        self.np = np
        self.torch = torch
        self.output_root = output_root.resolve()
        self.runtime_lock = dict(runtime_lock)
        self.fp_root = Path(runtime_lock["foundationpose_root"]).resolve()
        scripts = Path(runtime_lock["poseloop_runtime_root"]).resolve() / "scripts"
        for path in (self.fp_root, scripts):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        original_cwd = Path.cwd()
        os.chdir(self.fp_root)
        try:
            from estimater import FoundationPose
            from learning.training.predict_pose_refine import PoseRefinePredictor
            from learning.training.predict_score import ScorePredictor
            from run_r1_sealed_inference import (
                install_memory_bounded_refine_forward,
                install_memory_bounded_score_data,
                install_memory_bounded_score_forward,
                install_memory_bounded_warp,
            )
            from run_xyzibd_batch import clear_per_sample_estimator_state
            from Utils import set_seed
        except Exception as exc:
            raise PrepError(
                f"Cannot import the audited FoundationPose production path: {type(exc).__name__}"
            ) from exc
        finally:
            os.chdir(original_cwd)
        self.FoundationPose = FoundationPose
        self.clear_per_sample_estimator_state = clear_per_sample_estimator_state
        self.set_seed = set_seed
        gpu_index = int(runtime_lock["gpu"]["index"])
        torch.cuda.set_device(gpu_index)
        install_memory_bounded_warp(FIXED_INFERENCE["resource_batches"]["warp"])
        set_seed(int(FIXED_INFERENCE["seed"]))
        self.scorer = ScorePredictor()
        install_memory_bounded_score_data(
            self.scorer, FIXED_INFERENCE["resource_batches"]["score_data"]
        )
        install_memory_bounded_score_forward(
            self.scorer, FIXED_INFERENCE["resource_batches"]["score_feature"]
        )
        self.refiner = PoseRefinePredictor()
        install_memory_bounded_refine_forward(
            self.refiner, FIXED_INFERENCE["resource_batches"]["refine"]
        )
        self.glctx = dr.RasterizeCudaContext(device=gpu_index)
        self.estimator: Any | None = None
        self.estimator_key: tuple[int, str] | None = None

    def _estimator_for(self, item: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
        object_id = int(item["sample_key"]["object_id"])
        cad_sha = str(item["inputs"]["cad"]["sha256"])
        key = (object_id, cad_sha)
        if self.estimator is None or self.estimator_key != key:
            mesh = inputs["mesh"]
            self.estimator = self.FoundationPose(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=None,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
                debug=0,
                debug_dir=str(self.output_root / "foundationpose_debug"),
            )
            baseline = int(len(self.estimator.rot_grid))
            if baseline != 252:
                raise PrepError(f"Pinned upstream rotation grid changed: {baseline}")
            self.estimator.rot_grid = self.estimator.rot_grid[:252].clone()
            self.estimator_key = key
        return self.estimator

    def run_item(
        self, item: Mapping[str, Any], *, asset_root: Path, attempt: int
    ) -> dict[str, Any]:
        np, torch = self.np, self.torch
        item_started = time.perf_counter()
        decode_started = time.perf_counter()
        inputs = load_live_input(item=item, asset_root=asset_root)
        input_decode_ms = (time.perf_counter() - decode_started) * 1000.0
        estimator = self._estimator_for(item, inputs)
        self.clear_per_sample_estimator_state(estimator)
        self.set_seed(int(FIXED_INFERENCE["seed"]))

        captured: dict[str, Any] = {
            "refiner_calls": 0,
            "scorer_calls": 0,
        }
        original_refine = self.refiner.predict
        original_score = self.scorer.predict

        def capture_refine(*args: Any, **kwargs: Any) -> Any:
            captured["refiner_calls"] += 1
            if captured["refiner_calls"] != 1:
                raise PrepError(
                    "FoundationPose register made an unexpected refiner call"
                )
            captured["refine_kwargs"] = dict(kwargs)
            captured["initial_population"] = _numpy(kwargs.get("ob_in_cams"), np)
            torch.cuda.synchronize()
            captured["refine_start"] = time.perf_counter()
            result = original_refine(*args, **kwargs)
            torch.cuda.synchronize()
            captured["refine_end"] = time.perf_counter()
            captured["final_population_tensor"] = result[0].detach().clone()
            captured["final_population"] = _numpy(result[0], np)
            return result

        def capture_score(*args: Any, **kwargs: Any) -> Any:
            captured["scorer_calls"] += 1
            if captured["scorer_calls"] != 1:
                raise PrepError(
                    "FoundationPose register made an unexpected scorer call"
                )
            captured["score_kwargs"] = dict(kwargs)
            captured["scored_population"] = _numpy(kwargs.get("ob_in_cams"), np)
            torch.cuda.synchronize()
            captured["score_start"] = time.perf_counter()
            result = original_score(*args, **kwargs)
            torch.cuda.synchronize()
            captured["score_end"] = time.perf_counter()
            captured["raw_scores_tensor"] = result[0].detach().clone()
            captured["raw_scores"] = _numpy(result[0], np).reshape(-1)
            return result

        self.refiner.predict = capture_refine
        self.scorer.predict = capture_score
        official_started = 0.0
        official_ended = 0.0
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            official_started = time.perf_counter()
            pose = estimator.register(
                K=inputs["camera_intrinsics"],
                rgb=inputs["rgb"],
                depth=inputs["depth_m"],
                ob_mask=inputs["mask"],
                ob_id=int(item["sample_key"]["object_id"]),
                iteration=int(FIXED_INFERENCE["iterations"]),
            )
            torch.cuda.synchronize()
            official_ended = time.perf_counter()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        finally:
            self.refiner.predict = original_refine
            self.scorer.predict = original_score
        if captured["refiner_calls"] != 1 or captured["scorer_calls"] != 1:
            raise PrepError(
                "FoundationPose register did not expose one refine/score population"
            )
        initial_population = captured["initial_population"]
        final_population = captured["final_population"]
        scored_population = captured["scored_population"]
        raw_scores = captured["raw_scores"]
        if (
            initial_population.shape != (252, 4, 4)
            or final_population.shape != (252, 4, 4)
            or scored_population.shape != (252, 4, 4)
            or raw_scores.shape != (252,)
        ):
            raise PrepError("FoundationPose live capture pruned or reshaped c252")
        if not np.array_equal(final_population, scored_population):
            raise PrepError(
                "FoundationPose scorer input differs from the refined population"
            )

        # Preserve the exact CUDA argsort used by upstream register, including
        # its deterministic handling of equal score logits.
        sorted_ids = (
            captured["raw_scores_tensor"]
            .argsort(descending=True)
            .detach()
            .cpu()
            .numpy()
        )
        selected = (
            int(estimator.best_id.detach().cpu().item())
            if hasattr(estimator.best_id, "detach")
            else int(estimator.best_id)
        )
        if selected != int(sorted_ids[0]):
            raise PrepError(
                "FoundationPose selected candidate differs from raw score ordering"
            )
        center_tf_tensor = estimator.get_tf_to_centered_mesh()
        official_pose = _numpy(pose, np)
        selected_final = _numpy(
            captured["final_population_tensor"][selected] @ center_tf_tensor,
            np,
        )
        if not np.array_equal(official_pose, selected_final):
            raise PrepError(
                "FoundationPose returned pose differs from captured top candidate"
            )

        # The upstream refiner internally loops five times and exposes only its
        # endpoint.  Replay the exact frozen predictor as five iteration=1 calls
        # solely to capture each real state; require a bitwise-equal endpoint.
        evidence_started = time.perf_counter()
        state_populations = [initial_population]
        current = initial_population
        refine_kwargs = dict(captured["refine_kwargs"])
        for _ in range(int(FIXED_INFERENCE["iterations"])):
            refine_kwargs["ob_in_cams"] = current
            refine_kwargs["iteration"] = 1
            refine_kwargs["get_vis"] = False
            refined, _ = original_refine(**refine_kwargs)
            current = _numpy(refined, np)
            if current.shape != (252, 4, 4):
                raise PrepError("Trace replay changed the c252 refiner population")
            state_populations.append(current)
        if not np.array_equal(state_populations[-1], final_population):
            raise PrepError(
                "Five-state refiner replay is not bitwise-equal to register(5)"
            )

        objective_populations: list[Any] = []
        score_kwargs = dict(captured["score_kwargs"])
        torch.cuda.reset_peak_memory_stats()
        for state in state_populations:
            score_kwargs["ob_in_cams"] = state
            score_kwargs["get_vis"] = False
            scores, _ = original_score(**score_kwargs)
            values = _numpy(scores, np).reshape(-1)
            if values.shape != (252,):
                raise PrepError(
                    "Trace objective replay changed the c252 score population"
                )
            objective_populations.append(values)
        torch.cuda.synchronize()
        evidence_ended = time.perf_counter()
        evidence_peak_allocated = int(torch.cuda.max_memory_allocated())
        evidence_peak_reserved = int(torch.cuda.max_memory_reserved())
        if not np.array_equal(objective_populations[-1], raw_scores):
            raise PrepError(
                "Final trace objective is not bitwise-equal to production scoring"
            )

        top_k: list[dict[str, Any]] = []
        for rank, candidate_index in enumerate(sorted_ids[:5], 1):
            index = int(candidate_index)
            top_k.append(
                {
                    "rank": rank,
                    "candidate_id": f"foundationpose-c{index:03d}",
                    "score": float(raw_scores[index]),
                    "model_to_camera_pose_m": _pose_list(
                        captured["final_population_tensor"][index] @ center_tf_tensor,
                        np,
                    ),
                }
            )
        trace = [
            {
                "iteration": iteration,
                "model_to_camera_pose_m": _pose_list(
                    torch.as_tensor(state, device="cuda", dtype=torch.float)[selected]
                    @ center_tf_tensor,
                    np,
                ),
                "objective": float(objective_populations[iteration][selected]),
            }
            for iteration, state in enumerate(state_populations)
        ]
        initial_pose = trace[0]["model_to_camera_pose_m"]
        final_pose = trace[-1]["model_to_camera_pose_m"]
        # Bind the exported margin to the exact serialized top-two scores.  A
        # subtraction performed in the source tensor dtype could round to a
        # different value after both scores are promoted to JSON numbers.
        margin = float(top_k[0]["score"]) - float(top_k[1]["score"])
        if margin < 0:
            raise PrepError("FoundationPose top score margin is negative")

        hypothesis_ms = max(0.0, (captured["refine_start"] - official_started) * 1000.0)
        refine_ms = (captured["refine_end"] - captured["refine_start"]) * 1000.0
        # Include the small upstream handoff between refiner return and scorer
        # entry so the five declared stages sum to the full register wall time.
        score_ms = (captured["score_end"] - captured["refine_end"]) * 1000.0
        selection_ms = max(0.0, (official_ended - captured["score_end"]) * 1000.0)
        stages = {
            "input_decode_ms": input_decode_ms,
            "hypothesis_generation_ms": hypothesis_ms,
            "refine_ms": refine_ms,
            "score_ms": score_ms,
            "selection_ms": selection_ms,
        }
        total_latency = sum(stages.values())
        visualization_started = time.perf_counter()
        visualizations = write_live_visualizations(
            item=item,
            inputs=inputs,
            initial_pose=initial_pose,
            final_pose=final_pose,
            top_k=top_k,
            output_root=self.output_root,
        )
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0
        item_wall_time_ms = (time.perf_counter() - item_started) * 1000.0
        runtime_identity = self.runtime_lock["producer_runtime_lock"]
        result = {
            "schema_version": RESULT_SCHEMA,
            "record_type": "poseloop_r4c_prep_result",
            "protocol_id": PREP_PROTOCOL_ID,
            "backend_id": LIVE_BACKEND_ID,
            "item_id": item["item_id"],
            "sample_key": dict(item["sample_key"]),
            "mask_variant_id": item["mask_variant_id"],
            "status": "success",
            "predicted_model_to_camera_pose_m": final_pose,
            "initial_model_to_camera_pose_m": initial_pose,
            "final_model_to_camera_pose_m": final_pose,
            "top_k": top_k,
            "refiner_trace_model_to_camera_m": trace,
            "implementation_commit": runtime_identity["implementation_commit"],
            "implementation_sha256": runtime_identity["implementation_sha256"],
            "model_sha256": runtime_identity["model_sha256"],
            "checkpoint_sha256": canonical_sha256(
                runtime_identity["checkpoint_sha256"]
            ),
            "foundationpose_top_score": float(raw_scores[selected]),
            "foundationpose_top_score_margin": margin,
            "selected_candidate_index": selected,
            "candidate_limit": 252,
            "pose_hypothesis_count": 252,
            "resource_batches": dict(FIXED_INFERENCE["resource_batches"]),
            "stage_timings_ms": stages,
            "total_latency_ms": total_latency,
            "wall_time_ms": item_wall_time_ms,
            "evidence_capture_ms": (evidence_ended - evidence_started) * 1000.0,
            "visualization_ms": visualization_ms,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "evidence_peak_allocated_bytes": evidence_peak_allocated,
            "evidence_peak_reserved_bytes": evidence_peak_reserved,
            "input_mask_sha256": item["inputs"]["mask"]["sha256"],
            "mask_provenance": dict(item["mask_provenance"]),
            "evaluator_label_read": False,
            "contains_gt_derived_control_input": (
                item["mask_provenance"].get("derivation_class")
                == "gt-derived-development-control"
            ),
            "uses_oracle_association": False,
            "attempt": attempt,
            "failure_reason": None,
            "oom": False,
            "synthetic_backend": False,
            "producer_runtime_isolated": True,
            "label_access_count": 0,
            "gt_path_open_count": 0,
            "evaluator_path_open_count": 0,
            "scorer_path_open_count": 0,
            "official_scorer_run_count": 0,
            "official_scorer_run": False,
            "accuracy_claim": "unavailable-label-free-live-runtime",
            "visualization_inventory": visualizations,
            "backend_evidence": {
                "synthetic": False,
                "primary_register_iterations": 5,
                "primary_candidate_count": 252,
                "foundationpose_candidate_scorer_calls_primary": 1,
                "trace_capture_mode": "exact-refiner-replay-plus-full-c252-scorer",
                "trace_refiner_replay_calls": 5,
                "trace_objective_scorer_calls": 6,
                "trace_final_pose_population_bitwise_equal": True,
                "trace_final_score_population_bitwise_equal": True,
                "official_evaluator_scorer_run": False,
            },
        }
        result["prediction_fingerprint"] = canonical_sha256(
            {
                "pose": final_pose,
                "top_score": result["foundationpose_top_score"],
                "margin": margin,
                "selected_candidate_index": selected,
                "candidate_count": 252,
            }
        )
        validate_internal_live_success(result, visualization_root=self.output_root)
        return result


def assert_mock_live_result_structure(
    row: Mapping[str, Any], *, visualization_root: Path
) -> None:
    """Mock tests may validate shape, but cannot instantiate the live backend."""

    validate_internal_live_success(row, visualization_root=visualization_root)
    evidence = row.get("backend_evidence")
    if not isinstance(evidence, dict) or evidence.get("synthetic") is not False:
        raise PrepError("Mock live structure lacks explicit non-synthetic evidence")
    if row.get("synthetic_backend") is not False:
        raise PrepError("Mock live structure entered the synthetic producer path")
