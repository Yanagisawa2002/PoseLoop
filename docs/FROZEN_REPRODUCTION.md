# Frozen weights and reproduction prerequisites

## Current outcome: reproduction SKIPPED; CPU batch FAILED

The 2026-09-16 bounded attempt started from main
`33311a52357c66819c9cb27bbd79e8035ec007c5`. Preflight could not obtain the
**frozen detector checkpoint**. No model was loaded, no dependency deployment
or GPU inference ran, and no new accuracy result was produced. The existing
25-frame results remain historical development results.

The single code candidate `e7b4a25a715cd35e6c492a80182a91079783d846` was checked
once in a fresh local Python 3.13.5 environment, without pip or inherited
site packages. All **4 asset-boundary tests passed**. The real missing-asset
invocation returned the expected exit 3, with detector loading skipped.
The subsequent historical portfolio-integrity check returned exit 1 because
`docs/media/poseloop-demo.mp4` is a **132-byte Git LFS pointer** in this checkout,
not its expected 1,846,462-byte media payload. The pointer's expected payload
hash agrees with the Release asset; the payload was not downloaded or retried.
The batch stopped there and is not an overall pass. Its final planned
`git diff --check` command was not reached (the pre-commit check had passed).

See the [validation receipt](../reproduction/evidence/single-attempt-20260916/validation.json)
and [raw test output](../reproduction/evidence/single-attempt-20260916/asset-boundary-tests.stderr.txt).
Actual model loading, CUDA compatibility and clean GPU pipeline reproduction
are unvalidated. The final PR is Draft; no runtime code changed after the
single validation batch.

[The asset catalog](../reproduction/frozen-assets.json) records exact file
sizes, hashes, provenance and the upstream source. It is not a weight delivery.
The detector's public download URL is explicitly `null`.

## What is missing

The historical training receipt identifies `model-final.pt` as 183,923,343
bytes with SHA-256
`a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`.
It belongs to epoch 8 of implementation
`958d1a66857e2ac928cac1eb34d825e65121c64d`. The training receipt itself hashes
to `d717840057d0b359a1720afef041479b7548bb29f874832435a7375719283991`.

Fourteen registered local PoseLoop worktrees were inspected by filename,
excluding sealed/test directories. No detector checkpoint was found. The two
known development evidence archives contain no checkpoint member; they store
receipts and outputs. The source checkout's two FoundationPose checkpoint
paths are unreadable on Windows. The public v1.1.0 asset list contains only
the poster, video, results JSON and SHA256SUMS. These checks do not establish
whether an owner backup exists elsewhere; no other server was searched.

The original COCO initialization weight is a different checkpoint and cannot
reproduce the trained detector. Recover the exact historical file from an
owner backup before any later attempt. Retraining is outside this attempt.

## Check assets before installing GPU dependencies

The new checker runs with Python's standard library. With no supplied files,
it reports `BLOCKED`, returns exit code **3**, and performs no model import or
data access:

```bash
python -B scripts/verify_frozen_assets.py --receipt missing-assets.json
```

After obtaining the exact detector and upstream FoundationPose files:

```bash
python -B scripts/verify_frozen_assets.py \
  --detector-checkpoint /models/model-final.pt \
  --foundationpose-root /runtime/sources/FoundationPose \
  --receipt asset-check.json
```

All five weight/config files must match their size and SHA-256. The receipt is
create-only. `ASSET_BYTES_VERIFIED` means file integrity only. The checker does
not certify GPU compatibility or pipeline accuracy.

The catalog pins NVIDIA's existing upstream repository, revision and URL
template. `scripts/fetch_foundationpose_weights.py` is the existing acquisition
entry. Its availability was not tested in this attempt because the detector
gate failed first. Its defaults include retries and a user cache; do not run
it unchanged under a single-attempt/no-global-cache campaign. A later approved
environment must bind all caches to its own runtime directory and use its
agreed timeout/retry limits.

## Strict detector loading

When the runtime dependencies and the historical, already-consumed dataset
**manifest** are available, check the actual release loader without inference:

```bash
python -B scripts/verify_frozen_assets.py \
  --detector-checkpoint /models/model-final.pt \
  --foundationpose-root /runtime/sources/FoundationPose \
  --dataset-manifest /evidence/data/dataset-manifest.json \
  --load-detector --receipt detector-load.json
```

Loading is gated on the detector bytes, protocol hash and manifest hash. The
existing loader then checks schema/epoch and strictly loads the two-class
Mask R-CNN state dictionary on CPU. It writes measured Python, torch and
torchvision versions. No raw image or label file is opened. Actual loading
remains **unvalidated** in this attempt because the checkpoint is absent.

## Clean environment and pipeline acceptance remain open

The historical pose receipt records Python 3.12.3, torch 2.8.0+cu128, CUDA
runtime 12.8 and RTX 5090. This is a partial historical environment record,
not a complete dependency lock. The old provisioning script uses
`--system-site-packages`, unpinned packages, `MAX_JOBS=16` and optional global
apt changes. It does not satisfy this campaign's clean environment,
four-job maximum and isolated-write requirements. It was not executed.

A later authorized attempt needs a fresh environment without inherited site
packages, pinned dependencies and upstream commits, task-local pip/Torch/XDG/
extension caches, verified compiler/CUDA support, and a real process lock and
hard timeout covering all children. Only then may the existing
`scripts/run_release_pipeline.sh` execute the complete frozen detector → pose
→ evaluation path. Its input is RGB-D, calibration and known CAD; its outputs
are predicted masks, per-instance poses, evaluation and a hashable archive.
The exact commands are in the [README](../README.md#run-the-frozen-pipeline).

No new independent samples were frozen: no verified unused development
manifest was supplied, and the weight gate already prevented inference.
The historical 25-frame set is not independent. Sealed/test data and retired
research routes remain outside scope. The bounded attempt has ended at this
prerequisite gate; there is no automatic retry.
