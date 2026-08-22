"""Layer-separated visualization plans; this module does not render or read images."""

from __future__ import annotations

from typing import Any, Mapping

from .core import EVALUATOR_VARIANTS, PRODUCER_VARIANTS, canonical_sha256


FAILURE_TAXONOMY = (
    {
        "id": "F_INPUT_CONTRACT",
        "signal": "missing/hash-invalid CAD, camera, depth, RGB, mask, or coordinate metadata",
        "owner": "bundle-preflight",
    },
    {
        "id": "F_COVERAGE",
        "signal": "missing or duplicate scene-image-object-mask key",
        "owner": "orchestrator",
    },
    {
        "id": "F_MASK",
        "signal": "mask empty, implausible, or variant-specific pose degradation",
        "owner": "mask-producer",
    },
    {
        "id": "F_SCORER_RANK",
        "signal": "near-GT frozen perturbation is not ranked in configured top-k",
        "owner": "scorer",
    },
    {
        "id": "F_REFINER_NONMONOTONIC",
        "signal": "ADD(-S), rotation, or translation error increases across refiner steps",
        "owner": "refiner",
    },
    {
        "id": "F_SE3_GEOMETRY",
        "signal": "illegal rotation, non-finite pose, wrong units, or transform-direction mismatch",
        "owner": "pose-emitter",
    },
    {
        "id": "F_OFFICIAL_CAPABILITY",
        "signal": "required official error type or dataset asset is unavailable",
        "owner": "evaluator",
    },
    {
        "id": "F_RUNTIME",
        "signal": "producer/refiner failure, timeout, or OOM",
        "owner": "runtime",
    },
)


def build_visualization_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    producer_rows = []
    evaluator_rows = []
    for sample in manifest["samples"]:
        key = sample["key"]
        producer = sample["producer_inputs"]
        for variant in PRODUCER_VARIANTS:
            stem = f"s{key['scene_id']:06d}-i{key['image_id']:06d}-o{key['object_id']:06d}-{variant}"
            producer_rows.append(
                {
                    **key,
                    "mask_variant": variant,
                    "namespace_role": "LABEL_FREE_PRODUCER_ONLY",
                    "layers": {
                        "rgb": producer["rgb"]["path"],
                        "mask": producer["masks"][variant]["path"],
                        "initial_pose_overlay": f"producer_visuals/{stem}-initial-cad-axes.png",
                        "top_k_overlay": f"producer_visuals/{stem}-top-k-cad-axes.png",
                        "final_pose_overlay": f"producer_visuals/{stem}-final-cad-axes.png",
                    },
                    "forbidden_layers": [
                        "gt_pose",
                        "gt_overlay",
                        "score",
                        "evaluator_threshold",
                    ],
                }
            )
        for variant in EVALUATOR_VARIANTS:
            stem = f"s{key['scene_id']:06d}-i{key['image_id']:06d}-o{key['object_id']:06d}-{variant}"
            evaluator_rows.append(
                {
                    **key,
                    "mask_variant": variant,
                    "namespace_role": "EVALUATOR_ONLY",
                    "layers": {
                        "rgb": producer["rgb"]["path"],
                        "mask": (
                            producer["masks"][variant]["path"]
                            if variant in PRODUCER_VARIANTS
                            else sample["evaluator_only"]["masks"][variant]["path"]
                        ),
                        "initial_pose_overlay": f"evaluator_visuals/{stem}-initial-cad-axes.png",
                        "top_k_overlay": f"evaluator_visuals/{stem}-top-k-cad-axes.png",
                        "final_pose_overlay": f"evaluator_visuals/{stem}-final-cad-axes.png",
                        "gt_overlay": f"evaluator_visuals/{stem}-gt-cad-axes.png",
                    },
                    "export_to_producer_permitted": False,
                }
            )
    plan = {
        "schema_version": "poseloop.pose-accuracy-recovery.visualization-plan.v1",
        "execution_state": "PREP_ONLY_NOT_RENDERED",
        "producer": producer_rows,
        "evaluator_only": evaluator_rows,
        "comparison_video": {
            "entrypoint": "python -m pose_accuracy_recovery_prep visualization-plan",
            "layout": "baseline_left_improved_right",
            "required_common_layers": [
                "rgb",
                "mask",
                "initial_pose_overlay",
                "top_k_overlay",
                "final_pose_overlay",
            ],
            "gt_overlay_output_namespace": "evaluator_only",
            "producer_video_must_exclude_gt": True,
        },
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "accuracy_claim_permitted": False,
    }
    plan["lock_sha256"] = canonical_sha256(plan)
    return plan
