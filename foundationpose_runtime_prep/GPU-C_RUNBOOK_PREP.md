# GPU-C FoundationPose PREP runbook

This namespace defaults to `AUTO_DEPLOY=false`. Local fixture commands never contact a server, install packages, download assets, pull images, compile a runtime, or run training/inference. A separately authorized live task may set `AUTO_DEPLOY=true` only for the explicit `live-preflight`, `live-backend-smoke`, and `live-produce` launcher commands below; the default refusal remains unchanged.

## Local fixture gate (current task, no GPU)

```bash
PREP_ROOT=artifacts/foundationpose_runtime_prep_dry_run
PROTOCOL=foundationpose_runtime_prep/contracts/producer_protocol_v1.json

python -m foundationpose_runtime_prep build-fixture \
  --protocol "$PROTOCOL" \
  --output-root "$PREP_ROOT/fixture"

python -m foundationpose_runtime_prep validate-manifest \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --asset-root "$PREP_ROOT/fixture" --verify-assets

python -m foundationpose_runtime_prep static-preflight \
  --protocol "$PROTOCOL" \
  --runtime-contract foundationpose_runtime_prep/contracts/python_runtime_contract_v1.json \
  --output "$PREP_ROOT/python-static-preflight.json"

python -m foundationpose_runtime_prep static-preflight \
  --protocol "$PROTOCOL" \
  --runtime-contract foundationpose_runtime_prep/contracts/isaac_ros_tensorrt_runtime_contract_v1.json \
  --output "$PREP_ROOT/isaac-static-preflight.json"

# Expected exit code is 75 after one atomically cached success.
set +e
python -m foundationpose_runtime_prep fixture-produce \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --asset-root "$PREP_ROOT/fixture" --verify-assets \
  --output-root "$PREP_ROOT/fixture-run" --crash-after-n 1
test "$?" -eq 75
set -e

python -m foundationpose_runtime_prep fixture-produce \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --asset-root "$PREP_ROOT/fixture" --verify-assets \
  --output-root "$PREP_ROOT/fixture-run" --resume

python -m foundationpose_runtime_prep export-a-results \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --producer-results "$PREP_ROOT/fixture-run/results.jsonl" \
  --output "$PREP_ROOT/a-producer-output.jsonl" \
  --validation-output "$PREP_ROOT/a-producer-output-validation.json"

python -m foundationpose_runtime_prep validate-a-results \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --results "$PREP_ROOT/a-producer-output.jsonl" \
  --output "$PREP_ROOT/a-producer-output-revalidation.json"

python -m foundationpose_runtime_prep fixture-profile \
  --protocol "$PROTOCOL" \
  --results "$PREP_ROOT/fixture-run/results.jsonl" \
  --output-root "$PREP_ROOT/fixture-profile" \
  --warmup-repeats 2 --steady-repeats 5

python -m foundationpose_runtime_prep exact-equivalence \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --baseline "$PREP_ROOT/fixture-run/results.jsonl" \
  --candidate "$PREP_ROOT/fixture-run/results.jsonl" \
  --output "$PREP_ROOT/exact-equivalence.json"

# This plan command is dependency-free and is the no-data/no-GPU contract gate.
python -m foundationpose_runtime_prep visualization-plan \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --baseline "$PREP_ROOT/fixture-run/results.jsonl" \
  --improved "$PREP_ROOT/fixture-run/results.jsonl" \
  --output "$PREP_ROOT/visualization-plan.json"

# Optional local render gate; it only uses already-installed cv2/numpy/trimesh.
python -m foundationpose_runtime_prep render-visualization \
  --protocol "$PROTOCOL" \
  --manifest "$PREP_ROOT/fixture/manifest.json" \
  --asset-root "$PREP_ROOT/fixture" \
  --baseline "$PREP_ROOT/fixture-run/results.jsonl" \
  --improved "$PREP_ROOT/fixture-run/results.jsonl" \
  --output-root "$PREP_ROOT/visualization"
```

The fixture timings are synthetic schema checks, not GPU measurements. The `89 ms` field is an external target only; no comparable speedup or performance claim is computed. Pixel equality is only a content no-regression signal and is never an accuracy result. MP4 is a lossy derived artifact; prediction and PNG/pixel hashes remain the canonical checks.

## Receiving a future A bundle

Use a fresh output namespace. The formal handoff schema is `poseloop.r4c.prep.runtime-isolated-manifest.v2`; v1 remains accepted only for legacy committed fixtures. Verify the archive SHA before extraction, reject symlink/path escape/unknown members, then run `validate-manifest --verify-assets` against exactly the extracted manifest and asset root. The v2 manifest must pin the A source protocol SHA-256 to `20380305a7fbd92b2c563baea7df1e4f82c39605379543cd8e9443925eae0b8b`, declare 100% coverage of the identical base sample set under all five frozen variants (`official_known_sample_sanity`, `oracle_mask_control`, `predicted_mask`, `depth_component_mask`, and `bbox_mask`), and contain only `public_rgb`, `public_depth`, `input_mask`, `public_camera`, and `public_cad` roles. Missing, duplicate, hash/byte-drifted, extra-schema, raw GT/evaluator/sealed paths, and nonzero access counters are hard failures.

`official_known_sample_sanity` and `oracle_mask_control` may be A-side GT-derived DEVELOPMENT controls, but they cross into GPU-C only as opaque, sanitized, hash-locked input masks. Their derivation must be disclosed truthfully with `contains_gt_derived_control_inputs=true`; the same manifest must require `contains_raw_gt_paths=false`, `gpu_c_resolves_derivation=false`, `label_access_count_on_gpu_c=0`, `gt_path_open_count_on_gpu_c=0`, `evaluator_path_open_count_on_gpu_c=0`, and `official_scorer_run=false`. GPU-C must never resolve either control back to source labels. The predicted/depth-component/bbox variants use their corresponding non-control provenance classes. Every variant records source artifact/model SHA, score, and nonzero/fractional coverage; this PREP package downloads no front-end model.

## Future authorized GPU-C gate

Before any future launch, receive a hash-frozen A bundle and verify its archive SHA, upstream protocol SHA, v2 manifest lock, 100% five-variant coverage, every role/path/byte count/SHA, provenance, and boundary declaration. Both Python and Isaac runtime contracts must also verify the exact SHA of `backend_transport_v1.json` (whose frozen schema is v2) before an adapter starts. GPU-C must never scan the dataset root, open `scene_gt*`, `mask_visib`, evaluator/score outputs, or run an official scorer.

The current Python static preflight deliberately reports three blockers:

1. Existing FoundationPose source commit has not been inspected in this local-only task.
2. Existing refiner/scorer checkpoint hashes have not been inspected in this local-only task.
3. CUDA/Python imports and GPU identity have not been inspected because no GPU session is authorized.

The Isaac ROS/TensorRT contract is more restrictive: image digest, Isaac ROS release/package identity, CUDA and TensorRT versions, refiner/scorer engine hashes, and adapter executable hash are intentionally `null`. `launch_isaac_ros_tensorrt.sh` refuses to launch until a future authorized live inspection freezes those values. `launch_official_python.sh` permits only `live-preflight`, `live-backend-smoke`, or `live-produce`, requires an explicit `AUTO_DEPLOY=true`, and never pulls, installs, compiles, or guesses a runtime.

When a future task explicitly enables deployment, resolve blockers with read-only checks against the exact contract before launch. Do not clone, install, pull, build, or change c252, score-data chunk 8, score-feature chunk 32, refiner chunk 32, warp chunk 32, five iterations, or seed 0 as an implicit fallback. A missing compatible backend remains a blocker rather than permission to edit the frozen core during a run.

The official Python live path uses three deliberately separate commands. First freeze the actual FoundationPose source, checkpoints/configs, reused chunking source, deployed adapter files, Python/Torch/CUDA, and GPU UUID, while acknowledging the exact backend transport contract:

```bash
export AUTO_DEPLOY=true
PROTOCOL=foundationpose_runtime_prep/contracts/producer_protocol_v1.json
RUNTIME=foundationpose_runtime_prep/contracts/python_runtime_contract_v1.json

foundationpose_runtime_prep/launch_official_python.sh live-preflight \
  --protocol "$PROTOCOL" --runtime-contract "$RUNTIME" \
  --implementation-root "$DEPLOYED_POSELOOP_ROOT" \
  --implementation-commit "$DEPLOYED_COMMIT" \
  --poseloop-runtime-root "$AUDITED_POSELOOP_RUNTIME_ROOT" \
  --foundationpose-root "$FOUNDATIONPOSE_ROOT" --gpu-index 0 \
  --output-root "$RUN_ROOT/preflight"
```

If the formal five-variant A bundle has not arrived, an authorized task may run exactly one separately locked `predicted_mask` smoke request built from a previously frozen no-GT R4-C input. Its purpose must be `one-item-no-gt-adapter-smoke-not-five-variant`; it cannot be exported to A or described as five-variant coverage:

```bash
foundationpose_runtime_prep/launch_official_python.sh live-backend-smoke \
  --protocol "$PROTOCOL" --runtime-contract "$RUNTIME" \
  --runtime-lock "$RUN_ROOT/preflight/live-runtime-lock.json" \
  --backend-ack "$RUN_ROOT/preflight/backend-transport-ack.json" \
  --request "$RUN_ROOT/smoke-input/live-smoke-request.json" \
  --source-manifest "$FROZEN_R4C_SOURCE_MANIFEST" \
  --asset-root "$RUN_ROOT/smoke-input" \
  --output-root "$RUN_ROOT/smoke-output"
```

`live-produce` accepts only `input_kind=DEVELOPMENT_DATA` under the frozen V2 manifest. It rejects the committed synthetic fixture, requires the manifest producer runtime lock to equal the live preflight identity, revalidates the environment before model construction, and retains failed attempts outside the success cache:

```bash
foundationpose_runtime_prep/launch_official_python.sh live-produce \
  --protocol "$PROTOCOL" --runtime-contract "$RUNTIME" \
  --runtime-lock "$RUN_ROOT/preflight/live-runtime-lock.json" \
  --backend-ack "$RUN_ROOT/preflight/backend-transport-ack.json" \
  --manifest "$A_BUNDLE_ROOT/manifest.json" \
  --asset-root "$A_BUNDLE_ROOT" --output-root "$RUN_ROOT/producer"

# Repeat the same command with --resume only after reviewing the retained
# planned-crash or failed-attempt receipt. Never delete or silently skip it.
python -m foundationpose_runtime_prep export-a-live-results \
  --protocol "$PROTOCOL" --manifest "$A_BUNDLE_ROOT/manifest.json" \
  --producer-results "$RUN_ROOT/producer/results.jsonl" \
  --output "$RUN_ROOT/a-return/a-producer-output.jsonl" \
  --validation-output "$RUN_ROOT/a-return/a-producer-output-validation.json"
```

The synthetic `export-a-results` command rejects live rows, and `export-a-live-results` rejects synthetic rows. The live adapter keeps upstream `register(iteration=5)` as the primary prediction. Because upstream exposes only its final population, trace evidence uses five exact `iteration=1` refiner replays from the captured initial 252 population and full-252 scorer objective calls for all six states. Both replay endpoints must be bitwise equal to the primary final pose and score populations; otherwise the item fails. Primary inference latency/VRAM and trace-evidence replay overhead/VRAM are recorded separately.

Expected future order is: archive/hash preflight → v2 manifest/role/provenance preflight → backend transport acknowledgement → one sanity-control item → complete sanity control → opaque oracle control → predicted/depth-component/bbox variants → warmup/steady 720p profile → exact-equivalence guard → runtime-isolated visualization → `export-a-live-results` strict self-validation → internal SHA256SUMS archive → local return/hash verification. The exported schema is exactly `poseloop.pose-accuracy-recovery.producer-output.v1`; every row carries explicit `failure` and `oom` booleans plus the exact four structured label/GT/evaluator/scorer access counters. Missing top-k or the six-state refiner trace with objective values, hash drift, missing coverage, missing visualization assets, or any nonzero access counter fails. Accuracy evaluation remains GPU-A's responsibility.

Budget the future run from observations, not a guessed FPS: allow up to five minutes for read-only hashes/import/GPU preflight, then estimate producer wall time as `measured steady p95 seconds × remaining items × 1.2`. Profile time is the declared warmup plus steady repeats; archiving/return is measured separately. If the single-item sanity run cannot produce a valid c252 result and atomic receipt within the authorized environment-preparation boundary, stop with its exact blocker.

Before return, include protocol/runtime/manifest/result/profile/equivalence/visualization receipts, PNGs and MP4, and a root `SHA256SUMS`. Re-verify the returned archive locally before asking GPU-A to accept it. GPU-C must not run an evaluator or official scorer at any point.
