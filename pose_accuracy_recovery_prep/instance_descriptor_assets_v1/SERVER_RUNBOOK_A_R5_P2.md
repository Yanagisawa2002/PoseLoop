# A-R5-P2 descriptor assets — PREP runbook

## Authority and boundary

This namespace is local PREP only. `AUTO_DEPLOY=false`: the current review turn
must not connect to GPU-A, import a model, create remote outputs, run
FoundationPose, read GT/depth/evaluator/sealed assets, or shut down a server.
Deployment requires a later reviewed Git commit/tree and explicit execution
authorization. `instance_proposal_v1r5` and the frozen A-R4 evidence remain
byte-for-byte read only.

A future run produces only the five official-CNOS template descriptor banks for
objects `[1,2,4,5,6]`. It makes no proposal-quality, pose, AR, AP, or accuracy
claim.

The predecessor `A_R4_RENDER_PROVENANCE` object is preserved byte-for-byte,
including its historical 65-character attempt-lock transcription. P2 does not
correct or reinterpret that old field: its new protocol separately binds the
hash-verified content-audit receipt and that receipt's actual lowercase-64hex
internal attempt lock
`b36a9aacfe57f58292c0aaaaca2333fb763b4c38b50a3bdf723a60d9621fc570`.

## Frozen method

- Inputs: the exact A-R4 5-object × 42-view 640×480 RGBA renders and five CADs.
- Model source: official CNOS `298d1f3366171464ca271659f0e2f7a6eb8e39b4`
  and official DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Weight: official DINOv2 ViT-L/14, 1,217,586,395 bytes, SHA-256
  `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
- Semantics: PIL alpha-causal bbox, RGB conversion, official CNOS
  `CropResizePad(224)`, ImageNet normalization, and
  `CustomDINOv2.compute_features(..., token_name="x_norm_clstoken")`.
- Output: exactly five finite float32 tensors shaped `[42,1024]`.

The implementation, CNOS, and DINOv2 checkouts must all be clean. The request
binds every tracked Python source in each external checkout, all reviewed P2
wrapper files, request-bound source archives, the exact weight, every CAD, all
210 RGBA files, and zero-access counters. Runtime module origins are audited
after the local DINO hub load; source-bearing modules must match frozen bytes,
while source-free namespace packages must resolve to their exact directory
inside the bound checkout. Strict weight loading is mandatory.

## Immutable root layout

```text
ROOT/
  assets/cad/obj_00000{1,2,4,5,6}.ply
  assets/templates/obj_XXXXXX/000000.png ... 000041.png
  assets/descriptors/obj_XXXXXX.pth
  contracts/rgba-template-manifest.json
  contracts/descriptor-assets-request.json
  contracts/a-r5-descriptor-assets-bundle.json
  evidence/a-r4/safe-evidence.tar.gz
  evidence/a-r4/safe-evidence-members.json
  evidence/a-r4/deployment-final-inventory.json
  evidence/a-r4/{authorization,attempt,content_audit}.json
  models/dinov2_vitl14_pretrain.pth
  sources/poseloop-<commit>.tar.gz
  sources/cnos/
  sources/cnos-<commit>.tar.gz
  sources/dinov2/
  sources/dinov2-<commit>.tar.gz
  receipts/a-r4-template-import.json
  receipts/descriptor-assets-v1/obj_XXXXXX.json
  receipts/descriptor-assets-v1/planned-stop.json
  receipts/descriptor-generation-receipt.json
```

## Create the A-R4 manifest when the frozen root has only PNG/evidence

Do not modify the frozen A-R4 root. Copy its exact CADs into the new root first,
then run the create-only importer. The importer requires the frozen safe archive
(`1653423` bytes, SHA-256
`c997f4f6d65f2398527b55bcdffbb40cf4881783cbe50580e712697646f118bb`),
its external member inventory, and the deployment-final inventory. It recomputes
all 254 payload members inside the 256-member archive, compares the embedded and
external member inventories plus `SAFE_EVIDENCE_SHA256SUMS`, cross-pairs the
deployment inventory, and derives the canonical five CAD identities from those
three sources. Every caller-provided CAD must match its object-specific archive
member byte-for-byte; a fake, swapped, or cross-paired CAD/evidence set fails
closed. It also verifies the exact authorization, attempt, and content-audit
receipt SHA values; parses the frozen 210-row content audit; cross-checks every
source PNG's archive path, bytes, SHA, RGBA statistics, and object summary; then
decodes, hashes, and copies each PNG. It writes a self-locked A-R5-compatible
manifest plus an import receipt.

```bash
python -B -m pose_accuracy_recovery_prep.instance_descriptor_assets_v1 \
  build-template-manifest \
  --protocol protocols/poseloop_pose_accuracy_recovery_instance_descriptor_assets_v1.json \
  --data-root "$ROOT" \
  --source-png-root "$FROZEN_A_R4_ATTEMPT/objects" \
  --authorization-receipt "$FROZEN_A_R4_AUTHORIZATION" \
  --attempt-receipt "$FROZEN_A_R4_ATTEMPT_RECEIPT" \
  --content-audit "$FROZEN_A_R4_CONTENT_AUDIT" \
  --safe-archive "$FROZEN_A_R4_SAFE_ARCHIVE" \
  --safe-member-inventory "$FROZEN_A_R4_MEMBER_INVENTORY" \
  --deployment-inventory "$FROZEN_A_R4_DEPLOYMENT_INVENTORY" \
  --cad-root "$ROOT/assets/cad" \
  --output "$ROOT/contracts/rgba-template-manifest.json" \
  --import-receipt-output "$ROOT/receipts/a-r4-template-import.json"
```

Any partial importer failure leaves a create-only partial root. Preserve it as
evidence and use a new root; do not overwrite or hand-edit it.

## Freeze request and preflight before model import

Generate the implementation and DINO archives with the exact `git archive`
commands recorded in the request. Copy the already frozen CNOS transport
archive byte-for-byte: 17,114,735 bytes, SHA-256
`07c52c95f31ddae7fce14f5741c2fe5d7eab88369905237b2a52a5f4a8ebd6e5`.
That archive was produced by Git for Windows and is not byte-reconstructible by
Linux Git because export line endings and the gzip stream are platform/version
dependent. It remains independently protected by its pinned bytes/SHA, while
the live CNOS checkout is separately required to match the clean official
commit/tree and the complete tracked execution-file inventory. Do not replace
the pinned CNOS archive with a Linux-generated lookalike. Then build the
request:

```bash
python -B -m pose_accuracy_recovery_prep.instance_descriptor_assets_v1 \
  build-request \
  --protocol protocols/poseloop_pose_accuracy_recovery_instance_descriptor_assets_v1.json \
  --data-root "$ROOT" \
  --implementation-archive "$ROOT/sources/poseloop-${IMPLEMENTATION_COMMIT}.tar.gz" \
  --template-manifest "$ROOT/contracts/rgba-template-manifest.json" \
  --template-import-receipt "$ROOT/receipts/a-r4-template-import.json" \
  --cnos-checkout "$ROOT/sources/cnos" \
  --cnos-archive "$ROOT/sources/cnos-${CNOS_COMMIT}.tar.gz" \
  --dinov2-checkout "$ROOT/sources/dinov2" \
  --dinov2-archive "$ROOT/sources/dinov2-${DINOV2_COMMIT}.tar.gz" \
  --dinov2-weights "$ROOT/models/dinov2_vitl14_pretrain.pth" \
  --cad-root "$ROOT/assets/cad" \
  --output "$ROOT/contracts/descriptor-assets-request.json"

python -B -m pose_accuracy_recovery_prep.instance_descriptor_assets_v1 \
  preflight \
  --protocol protocols/poseloop_pose_accuracy_recovery_instance_descriptor_assets_v1.json \
  --request "$ROOT/contracts/descriptor-assets-request.json" \
  --data-root "$ROOT"
```

Preflight must say `PASS_PREFLIGHT_NO_MODEL_IMPORT`. A failure is a stop gate;
do not import the model or patch code remotely.

## Future authorized screen job

After a separately reviewed commit and execution authorization, use one detached
screen with tee and an external atomic exit receipt:

```bash
screen -dmS poseloop_ga_a_r5_p2 bash -lc 'set -o pipefail
python -B -m pose_accuracy_recovery_prep.instance_descriptor_assets_v1 produce \
  --protocol "$PROTOCOL" --request "$REQUEST" --data-root "$ROOT" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf "%s\n" "$rc" > "$EXIT_RECEIPT"
exit "$rc"'
```

`--planned-stop-after-objects N` accepts only `N=1..4` and exits 75 after a
create-only completed prefix. `--resume` is legal only when the exact planned
stop receipt, request file, tensor, and every completed sidecar still match.
Orphan tensors, missing sidecars, cross-request sidecars, ordinary failures, or
tampering require a new output root.

## Independent disk validation and A-R5 handoff

Run from a separate validation invocation after completion:

```bash
python -B -m pose_accuracy_recovery_prep.instance_descriptor_assets_v1 \
  validate-success \
  --protocol "$PROTOCOL" --request "$REQUEST" --data-root "$ROOT" \
  --output "$ROOT/receipts/independent-validation.json"
```

This reopens all 210 RGBA files, five tensors, source checkouts/manifests,
weight identity, module-origin evidence, request-specific sidecars, final
receipt, and compatibility bundle. The bundle supplies an exact A-R5
`descriptor_generation_receipt` asset and five-item `catalog_snippet` with
`descriptor_metadata`. It is an asset handoff only; it does not authorize the
A-R5 proposal producer or any evaluator.
