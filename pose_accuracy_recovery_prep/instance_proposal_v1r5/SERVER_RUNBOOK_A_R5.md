# A-R5 CNOS Instance Proposals — PREP Runbook

## Current status and authority

This namespace is **PREP only**. `AUTO_DEPLOY=false`: creating these files and
running their fixture tests does not authorize SSH, model import, inference,
downstream export, FoundationPose, evaluator/scorer access, or shutdown. GPU-A
must remain on. A future execution needs a separately reviewed Git identity,
immutable deployment root, user authorization, and a create-only output root.

A-R3 and A-R4 are immutable predecessors. A-R5 freezes A-R4's successful
5-object × 42-view RGBA render provenance in the protocol; it does not reopen or
reinterpret either failed A-R3 attempt. The B2-P1 design at `5162dfa` was read
only as a design reference. Its per-frame known-object assumption is not reused.

## Scientific boundary

Each of the ten frozen development frames is identified only by
`(scene_id,image_id)`. Historical object IDs used to stratify that slice are
removed before the runtime manifest is written. Every frame uses the same full
catalogue `[1,2,4,5,6]`:

1. official CNOS FastSAM-x generates independent instance masks from RGB;
2. every mask is scored against all five DINOv2 ViT-L/14 CAD-template
   descriptor banks;
3. each per-mask CAD ranking is ordered by raw cosine descending then object ID;
4. stored similarity is `(raw_cosine + 1) / 2`, without clamp;
5. frame-level selected mask is ordered by rank-1 CAD similarity, proposal
   confidence, mask stability, then proposal index.

No target-object ID, GT association, whole-scene foreground fallback, or GT
visibility may enter that route. A selected proposal is only a label-blind CNOS
hypothesis; it is not proof of instance correctness or pose accuracy.

Raw sensor depth is present solely as a hash-locked B2-v2 exchange input. The
independent proposal backend is not passed depth. Depth-derived masks,
visibility, boxes, association, and fallback are forbidden. The independent
validator reopens depth only to verify image shape/mode/nonzero support and the
recorded byte/SHA binding.

## Frozen runtime inputs

Use a new immutable root with this layout (paths in JSON are POSIX-relative):

```text
ROOT/
  inputs/rgb/                 # ten 1440x1080 RGB development frames
  inputs/depth/               # ten raw 1440x1080 integer sensor-depth frames
  inputs/camera/              # ten depth-free public-camera JSON files
  assets/cad/                 # exact object 1,2,4,5,6 CAD files
  assets/templates/           # exact A-R4 5x42 RGBA views
  assets/descriptors/         # five 42x1024 float32 DINOv2 descriptor tensors
  models/FastSAM-x.pt
  models/dinov2_vitl14_pretrain.pth
  sources/cnos/               # pinned official commit 298d1f3...
  sources/dinov2/             # hash-locked official origin checkout
  sources/*.tar.gz            # source archives
  sources/*checkout-manifest.json
  contracts/frame-manifest.json
  contracts/rgba-template-manifest.json
  contracts/cnos_catalog_config_v1r5.json
  receipts/descriptor-generation-receipt.json
```

The frame manifest must cover exactly ten frames across scenes
`0,3,9,12,15`, two images per scene, and has no `object_id`. Camera JSON accepts
only frame size, intrinsics, public world-to-camera transform, and coordinate
convention. `depth_scale`, paths, nested extras, GT/evaluator tokens, and sealed
tokens fail closed.

The runtime request binds the implementation commit/tree/archive; official CNOS
and DINOv2 source archives plus checkout manifests; FastSAM-x and DINOv2
checkpoints; adapter config; five CADs; five descriptors; descriptor receipt;
the complete A-R4 RGBA manifest; and all zero-access counters. The pinned
FastSAM-x SHA-256 is
`752cadc2828edb1cd4bc4f9eb587100631af06ea2108f4c9ed56df4755701e76`.

The implementation archive is not accepted as a self-reported hash. It must be
reproducible from the clean checkout that is executing this package:

```text
git archive --format=tar.gz \
  --prefix=poseloop-${IMPLEMENTATION_COMMIT}/ ${IMPLEMENTATION_COMMIT}
```

Preflight, producer entry, and disk-output validation resolve the repository
root from `instance_proposal_v1r5/contracts.py`, compare clean `HEAD` and
`HEAD^{tree}` with the request, rebuild that archive, and revalidate every
protocol wrapper/dependency byte. The exact inventory includes the A-R5 files,
`pose_accuracy_recovery_prep/cnos_runtime_prep_v1/adapter.py`, and
`pose_accuracy_recovery_prep/core.py`.

Both checkout manifests use
`poseloop.pose-accuracy-recovery.source-checkout-manifest.v1r5` with the exact
fields `kind`, `repository`, `commit`, `tree`, `all_clean`, empty
`git_status_porcelain_v1_untracked_files_all`, the bound source archive,
`execution_files`, its canonical SHA, and the manifest self-lock. Preflight
reruns `git status --porcelain=v1 --untracked-files=all` and requires no output.
CNOS binds exactly `src/model/fast_sam.py`, `src/model/dinov2.py`, and
`src/model/utils.py`. DINOv2 binds `hubconf.py` and every tracked
`dinov2/**/*.py`; after local `torch.hub` loading, every actual `dinov2.*`
module and all three CNOS modules are checked against those exact paths,
sizes, and SHA-256 values. Source-free DINO namespace packages are accepted
only when every search root is the matching directory inside the frozen
checkout; any loaded source-bearing child remains byte-audited. Launch Python
with `-B` (the adapter also disables
bytecode writes) so no checkout-local `__pycache__` can invalidate clean status.

The DINOv2 source and weight identity is fixed, not merely strict-loadable:

- official repository `https://github.com/facebookresearch/dinov2`;
- commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, tree
  `2a27257b79b0633b027a21014bc9360e3c1b3f43`;
- official ViT-L/14 URL
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`;
- bytes `1217586395`, SHA-256
  `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`;
- source/weight identity receipt SHA-256
  `7f417bba08ea1dd8eb5d1481a06d76a45d32410bc7fcf495591dc196c8ef338e`.

## Future execution sequence (do not run in this PREP turn)

Use the reviewed commit's protocol path:

```powershell
$protocol = "protocols/poseloop_pose_accuracy_recovery_instance_proposal_v1r5.json"
$manifest = "D:/immutable-a-r5/contracts/frame-manifest.json"
$request = "D:/immutable-a-r5/contracts/runtime-request.json"
$dataRoot = "D:/immutable-a-r5"
$outputRoot = "D:/immutable-a-r5-output-attempt1"
```

Validate all reviewed wrapper bytes and input assets before model import:

```powershell
python -B -m pose_accuracy_recovery_prep.instance_proposal_v1r5 validate-protocol `
  --protocol $protocol --repository-root .
python -B -m pose_accuracy_recovery_prep.instance_proposal_v1r5 validate-manifest `
  --protocol $protocol --manifest $manifest --data-root $dataRoot
python -B -m pose_accuracy_recovery_prep.instance_proposal_v1r5 preflight `
  --protocol $protocol --manifest $manifest --request $request `
  --data-root $dataRoot --output "$outputRoot/preflight-receipt.json"
```

After a separate execution authorization, start exactly one detached job with
`screen` and `tee`; record its command, PID, GPU state, log, and exit code in an
external launch receipt. The Python output root is create-only:

```bash
screen -dmS poseloop_ga_a_r5 bash -lc 'set -o pipefail; \
python -B -m pose_accuracy_recovery_prep.instance_proposal_v1r5 run-producer \
  --protocol "$PROTOCOL" --manifest "$MANIFEST" --request "$REQUEST" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  2>&1 | tee "$OUTPUT_ROOT/producer.log"; \
rc=${PIPESTATUS[0]}; printf "%s\n" "$rc" > "$OUTPUT_ROOT/producer.exit"; exit "$rc"'
```

`--planned-crash-after-frames N` is a test/recovery mechanism. It seals a
completed prefix and exits via `PlannedCrash`; a later reviewed invocation uses
`--resume`. Resume is allowed only from `PLANNED_CRASH`, validates every item
receipt and asset SHA from disk, and never reruns completed frames. Ordinary
failures become `FAILED` and require a new output root/receipt.

## Disk-independent B2-v2 validation

Every output row records complete RGB/depth/camera bindings, producer source and
checkpoint identities, every proposal PNG (path/bytes/SHA/decoded mode and
shape/foreground/support), full five-object CAD ranks, OOM evidence, and one of
the explicit states `NO_PROPOSAL`, `ONE_PROPOSAL`,
`MULTIPLE_PROPOSALS_DISJOINT`, or `MULTIPLE_PROPOSALS_ADJACENT`.
`NO_PROPOSAL` is valid and never receives GT fill.

The four scene-content PNGs per frame are RGB with all instance contours,
selected-mask overlay, full CAD top-k text, and adjacency audit. A metric chart
does not satisfy this contract.

B2-v2 must independently reopen all input/output assets and run:

```powershell
python -B -m pose_accuracy_recovery_prep.instance_proposal_v1r5 validate-output `
  --protocol $protocol --manifest $manifest --request $request `
  --producer-manifest "$outputRoot/outputs/producer-manifest.json" `
  --run-receipt "$outputRoot/outputs/run-receipt.json" `
  --data-root $dataRoot --output-root $outputRoot `
  --output "$outputRoot/b2-v2-disk-validation.json"
```

The validator recomputes mask support/bounds/components, proposal adjacency,
all ranking orders, every source/model/config/input binding, visualization
decode, bundle lock, receipt lock, and all zero-access counters. A-R5 does not
export results or call B2-v2 itself.

## Explicit non-results

Fixture dry-runs are contract tests only. They make no accuracy, proposal
quality, FoundationPose, AR, AP, ADD(-S), or deployment claim. Formal human
content review and any downstream experiment remain future work.
