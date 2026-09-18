# V1.2 RTX 5090 detector smoke

V1.2 INTERFACES READY  
DETECTOR SMOKE PASS

Implementation executed: `bd5c5b55ad450d8670c1acc8e4c02a0532cffa90`. The final report commit
only adds evidence; it does not change the tested execution code.

| Measurement | Observed |
| --- | --- |
| Dataset | 400 train / 100 internal validation / 25 development evaluation frames |
| CUDA device | NVIDIA GeForce RTX 5090, sm_120 |
| Python / torch / torchvision | 3.12.3 / 2.8.0+cu128 / 0.23.0+cu128 |
| CUDA / driver | 12.8 / 595.71.05 |
| Training batches / actual SGD steps | 2 / 1 |
| AMP | autocast + GradScaler, unchanged A-R9 train function |
| Mean total loss | 8.5401177406 |
| Peak allocated GPU memory | 2454102016 bytes (2.286 GiB) |
| Training function wall time | 31.741 s |
| Smoke wall time including reload | 35.164 s |
| Post-run disk used / available | 9,719,230,464 / 43,967,860,736 bytes |
| GitHub CPU tests | 36 passed |
| Server CPU tests | 35 passed (GitHub additionally checks LFS release media) |

All nine smoke checks passed, including finite tensors, a real optimizer step,
strict checkpoint reload and exact equality with the in-memory trained model.
The process exited normally; no formal eight-epoch training was started.

## Artifact locations

Server checkout: `/root/autodl-tmp/PoseLoop-network`.
Run root: `/root/autodl-tmp/poseloop-v1.2-5090`.

- Smoke checkpoint: `detector-smoke/model-final.pt` (183923343 bytes).
- Smoke receipt: `detector-smoke/smoke-receipt.json`.
- Training result: `detector-smoke/training-result.json`.
- Dataset manifest: `dataset/dataset-manifest.json`.
- Data: `xyzibd/`; official pretrained weights: `weights/`.

Checkpoint SHA-256: `920ce406914808aa347d80ad4329010dce5507d83d2d3cfe78262f60a4419565`.
Dataset-manifest SHA-256: `b958091b602a3d9a62bc1b60ea28174d6f79ffcda5523c4e68bb33600f56810e`.
This dataset manifest matches the original fixed dataset identity; the smoke
checkpoint is a new artifact and is NOT the lost v1.1 detector or a formal v1.2
checkpoint. It is marked `SMOKE_NOT_DECISION_ELIGIBLE`.

The small JSON/log receipts are copied here as durable evidence. No dataset or
checkpoint bytes are committed. The checkpoint remains on the server; it is a
recreatable smoke artifact, not the future formal model requiring off-server backup.
All three pinned archive digests were verified; only required RealSense files
were extracted. Scene 9 was not extracted or used. Download HTTP 503/slow-range
recovery did not change the dataset or experiment contract.

## Validation and next boundary

[CPU workflow](https://github.com/Yanagisawa2002/PoseLoop/actions/runs/35331822723) passed all existing integrity steps and new tests.
No diff was introduced to the detector training protocol/runtime, A9 protocol/runtime,
old release runner or `release/v1.1.0`. Dynamic coverage was exercised with three
predictions through actual input freezing and asset validation, including tamper rejection.

READY FOR FORMAL TRAINING: **YES, for the unchanged detector training contract**.
This task stops here. FoundationPose provisioning, real registration smoke,
full primary inference, end-to-end scores, taxonomy and Nsight measurements are
not claimed complete. No FoundationPose compatibility patch or torch downgrade
was applied. The full-pipeline interfaces are CPU-tested, not GPU-end-to-end validated.
