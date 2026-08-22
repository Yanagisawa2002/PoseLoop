# GPU-C label-isolated producer output contract

Schema ID: `poseloop.pose-accuracy-recovery.producer-output.v1`. One ordered JSONL row is required for every handoff `item_id`. The exact five `mask_variant_id` values are `official_known_sample_sanity`, `oracle_mask_control`, `predicted_mask`, `depth_component_mask`, and `bbox_mask`. The first two are opaque DEVELOPMENT_ONLY controls generated on GPU-A; therefore the exchange is not globally label-free even though GPU-C remains label-isolated.

Required on every row:

- exact `sample_key`, `mask_variant_id`, unique `item_id`, and `manifest_lock_sha256`;
- implementation commit, implementation/model SHA-256, the canonical checkpoint-map SHA-256, and exact RGB/depth/camera/CAD/mask input SHA-256 values;
- a strict `access_counters` object containing exactly four zero fields: `label_access_count_on_gpu_c`, `gt_path_open_count_on_gpu_c`, `evaluator_path_open_count_on_gpu_c`, and `scorer_path_open_count_on_gpu_c`;
- `status` (`success`, `failed`, or `oom`), positive attempt, finite non-negative latency, explicit `failure`/`oom` booleans, and `failure_reason`.

Successful rows additionally require:

- finite legal 4x4 `initial_model_to_camera_pose_m` and `final_model_to_camera_pose_m` in meters;
- exactly five ordered `top_k` entries, each with a stable `candidate_id`, finite descending producer score, and legal pose;
- exactly six ordered `refiner_trace` states (iterations 0 through 5), each with a legal pose and finite `objective`, whose endpoints equal the initial and final poses;
- exactly five real visualization files: RGB, selected input mask, initial-pose overlay, top-5 overlay, and final-pose overlay. Every member supplies an item-isolated path, positive byte count, and verified SHA-256.

Failed or OOM rows carry no fabricated pose and require a non-empty `failure_reason`. No row may contain raw GT/evaluator/metric/accuracy/threshold paths or values. GPU-A rejects duplicates, omissions, hash drift, illegal SE(3), forbidden roles, or any nonzero access counter before evaluator-only work. The implementation is `pose_accuracy_recovery_prep/c_results.py`; missing fields are never defaulted.
