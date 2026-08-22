# SERVER_RUNBOOK_PREP

Status: `AUTO_DEPLOY=false`. This is a future execution order, not an authorization or an execution log.

## 0. New execution authorization and freeze on GPU-A

1. Create a new execution protocol; do not reuse PREP or R4-A v3 as a scoring authorization.
2. Bind the implementation commit, repository tree, protocol SHA-256, unified manifest SHA-256, every input asset SHA-256, toolkit commit, model/checkpoint SHA-256, commands, output roots, and random seed.
3. Confirm development role, sealed-split access count zero, official process count zero, score-file count zero, and no mutation of commit `58939e69960f2227c4fa20be547e61df23fe391d` evidence.
4. Run `validate-manifest` on GPU-A. Stop on duplicate keys, missing assets, hash mismatch, unit/coordinate ambiguity, or GT-leak rejection.

Expected A preflight products: unified validation JSON, canonical v2 C handoff,
producer/evaluator run plans, SHA256SUMS, and a pre-execution receipt. No
prediction or score is expected yet.

## 1. GPU-A builds the runtime-isolated GPU-C bundle

1. In the GPU-A evaluator namespace, run the canonical command below. It reads
   the unified manifest and copies the known-sample and oracle controls into
   opaque asset slots alongside the three predicted/depth/bbox masks.
2. Validate `manifest_lock_sha256`, every member SHA/byte count, and exact five
   variants for every scene-image-object base key.
3. Confirm the boundary is exactly:
   `contains_gt_derived_control_inputs=true`,
   `contains_raw_gt_paths=false`, `gpu_c_resolves_derivation=false`, and
   `label_access_count_on_gpu_c=0`.
4. Inspect every exported asset path. No raw GT/evaluator path, GT pose,
   evaluator, score, threshold, metric, or sealed asset may leave A.
5. Record archive/directory absolute path, bytes, SHA-256, member count, exact
   coverage, protocol SHA, manifest lock, source manifest SHA, and implementation
   commit. Send only that frozen runtime bundle and descriptor to GPU-C.

Exact future command (substitute only frozen paths and the full commit):

```text
python -m pose_accuracy_recovery_prep export-c-handoff --protocol protocols/poseloop_pose_accuracy_recovery_c_handoff_v2.json --manifest <unified-manifest.json> --data-root <unified-data-root> --implementation-commit <40-hex-commit> --implementation-sha256 <implementation-sha256> --model-sha256 <model-sha256> --refiner-checkpoint-sha256 <refiner-sha256> --scorer-checkpoint-sha256 <scorer-sha256> --output-root <fresh-c-handoff-root>
```

Expected bundle rows: five mask variants per base sample and 100% per-variant
coverage. The bundle is not globally label-free because two opaque inputs are
GT-derived DEVELOPMENT_ONLY controls. GPU-C itself still reads no labels and
must report zero label/GT/evaluator/scorer access.

## 2. GPU-C runs the label-isolated producer

1. Verify archive/member/manifest hashes and exact coverage before allocating GPU work.
   Run `python -m pose_accuracy_recovery_prep validate-c-handoff --manifest <extracted-root>/manifest.json --bundle-root <extracted-root> --output <preflight>/c-handoff-validation.json` before any model process.
2. Record GPU/runtime/checkpoint hashes and verify the frozen producer command and output root.
3. Execute all five ordered mask variants for every base key. Treat every mask
   purely as an input role; never open labels or attempt to resolve whether/how
   a control was derived. Never synthesize missing rows.
4. Emit exactly `poseloop.pose-accuracy-recovery.producer-output.v1`; every row
   must bind the handoff lock, five input hashes, implementation/model/checkpoint
   hashes, initial/final legal SE(3), top-k, refiner trace, status/attempt/
   latency/failure/OOM, all four zero path-access counters, and the visualization
   inventory. `top_k` is exactly five entries and `refiner_trace` is exactly six
   states with an objective on every state.
5. Build five real, hash-bound `rgb`/`input_mask`/`initial_pose_overlay`/
   `top_k_overlay`/`final_pose_overlay` files. Do not include GT
   colors, GT overlay, error, metric, evaluator threshold, or score-file assets.
6. Freeze result JSONL, execution receipt, visualization receipt, SHA256SUMS,
   and return archive. Do not evaluate accuracy on C.

Expected C products per input row: exactly one complete status row. Missing
fields are not defaulted. Accuracy remains unavailable on C.

## 3. GPU-A accepts and freezes producer outputs

1. Before opening evaluator-only GT, run the strict acceptance command below.
   It verifies the return archive inputs, protocol/manifest lock, row ordering
   and uniqueness, exact five-variant coverage, implementation/model/checkpoint
   identities, finite legal SE(3), traces, attempts/latency/OOM, access counters,
   and visualization hashes.
2. Freeze accepted producer hashes before opening evaluator-only assets.
3. Keep the already-exported known-sample/oracle controls classified as opaque
   GT-derived DEVELOPMENT_ONLY inputs. Oracle remains fault localization only;
   neither control may be reported as deployment evidence.
4. Do not change the producer, checkpoint, association, candidate selection, or mask after any evaluator output.

```text
python -m pose_accuracy_recovery_prep validate-c-results --handoff-manifest <c-handoff-root>/manifest.json --handoff-root <c-handoff-root> --results <c-return-root>/results.jsonl --result-root <c-return-root> --output <acceptance-root>/c-result-validation.json
```

## 4. GPU-A runs evaluator-only diagnostics

1. Run `prepare-diagnostics --namespace-role EVALUATOR_ONLY` to create the frozen 169-pose grid per sample.
2. Feed those hypotheses through the real scorer without changing the grid. Save the exact score for every candidate.
3. Feed the frozen configured trace subset through the refiner. Save every pose step.
4. Use the diagnostic evaluator to check near-GT top-k rank and monotonic ADD(-S)/rotation/translation reduction.
5. Classify failures using `FAILURE_TAXONOMY.md`; do not tune from a sealed split.

Exact PREP entrypoints, with future frozen paths substituted:

```text
python -m pose_accuracy_recovery_prep prepare-diagnostics --protocol <protocol.json> --manifest <manifest.json> --data-root <data-root> --namespace-role EVALUATOR_ONLY --output <diag-root>/perturbation-grid.jsonl
python -m pose_accuracy_recovery_prep evaluate-diagnostics --protocol <protocol.json> --manifest <manifest.json> --data-root <data-root> --namespace-role EVALUATOR_ONLY --grid <diag-root>/perturbation-grid.jsonl --scorer-output <frozen-scorer.jsonl> --refiner-output <frozen-refiner.jsonl> --output-root <diag-root>/evaluated
```

Expected products: perturbation JSONL plus receipt, scorer/refiner JSONL, diagnostic JSON/CSV, and evaluator-only visual overlays.

## 5. GPU-A development evaluation and capability boundary

1. Produce complete baseline/final rows for all five evaluator variants.
2. Write internal JSON/CSV for ADD(-S), rotation, translation, missing/failure rates, object/scene/mask/stage groups, mask-minus-predicted paired deltas, and final-minus-baseline deltas.
3. Audit the pinned BOP toolkit and dataset assets. Import AR/MSSD/MSPD/VSD only when the official runtime output and capability receipt exist.
4. Mark unsupported metrics `unavailable` with the exact reason. If required VSD is unavailable, AR remains unavailable; never substitute a reduced error set.
5. Freeze reports and content visualizations before any later decision. Development measurements are not sealed or deployment conclusions.

## 6. Stop conditions

Stop before further model work on any contract/hash/coverage/SE(3) failure, scorer near-GT rank failure, refiner non-monotonicity, missing official capability, or GT boundary violation. A new protocol is required for any repaired attempt. Server shutdown, release, deletion, push, merge, and tag all require separate current authorization.
