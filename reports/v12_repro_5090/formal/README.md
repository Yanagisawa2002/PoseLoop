## Formal v1.2 detector training: PASS

Exactly one formal run completed 8/8 epochs using the unchanged A-R9 detector contract, from source commit `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41` on branch `run/v1.2-repro-5090`. PR #12 remains unmerged. Subsequent comparison with the frozen A9 protocol confirmed that this checkpoint exactly reconstructs the historical v1.1 checkpoint: SHA-256 `a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`.

The original receipt and export remain immutable records. Their `NEW_V1_2_DETECTOR_NOT_V1_1_RECONSTRUCTION` interpretation is superseded by this verified identity correction; the training measurements and artifact bytes are unchanged. See the [detector replay evidence](../detector-exact-replay/README.md) for the subsequent prediction identity and A9 handoff result.

- Runtime: 2117.200454 seconds (35m 17s); process exit code 0.
- Every epoch: 400 batches and 200 optimizer updates; 1,600 total updates.
- Peak allocated VRAM: 3,389,988,864 bytes (3.157173 GiB).
- Environment: RTX 5090; torch 2.8.0+cu128; torchvision 0.23.0+cu128; runtime CUDA 12.8; driver 595.71.05.
- No warnings/errors found in the persistent training or launcher logs; all epoch mean losses finite.
- Post-training data disk: 11,380,961,280 bytes used; 42,306,129,920 bytes free (39.400654 GiB).

### Frozen identity

- Protocol SHA-256: `c200f80cd47ef338341ba9e1ba941b0cd81ba9fc28f8b1dca4c05dfb33bcc5f1`
- Dataset manifest SHA-256: `b958091b602a3d9a62bc1b60ea28174d6f79ffcda5523c4e68bb33600f56810e`
- Official pretrained weight SHA-256: `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`
- Final checkpoint: `/root/autodl-tmp/poseloop-v1.2-5090/detector-formal/model-final.pt`
- Final size: 183,923,343 bytes.
- Final SHA-256: `a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`
- All eight epoch checkpoints exist; all hashes are preserved in `detector-formal/SHA256SUMS` and `receipts/formal-training-receipt.json`.

### Training curve

| Epoch | Mean total loss | Optimizer updates | LR after epoch |
| --- | ---: | ---: | ---: |
| 1 | 1.107295507863 | 200 | 0.005 |
| 2 | 0.600935297608 | 200 | 0.005 |
| 3 | 0.502788606808 | 200 | 0.005 |
| 4 | 0.443675146326 | 200 | 0.0005 |
| 5 | 0.387123356313 | 200 | 0.0005 |
| 6 | 0.372236086875 | 200 | 0.0005 |
| 7 | 0.368091716915 | 200 | 0.0005 |
| 8 | 0.359268779904 | 200 | 5e-05 |

### Internal validation (fixed 10-frame training-health subset only)

This is neither the entire 100-frame validation set nor the 25-frame development evaluation.

- Frames / GT instances / predictions: 10 / 225 / 216.
- AP50: 0.585987683127; AP75: 0.069059932121.
- Precision / recall / F1 at IoU 0.5: 0.671296296296 / 0.644444444444 / 0.657596371882.
- PQ at IoU 0.5: 0.451856846090.
- TP / FP / FN at IoU 0.5: 145 / 71 / 80.
- Merge / split counts: 10 / 19.
- Scene 9 reads: 0.

### Persistence and next boundary

The original server files remain unchanged. An off-server copy of `model-final.pt`, `training-result.json`, and `dataset-manifest.json` has been downloaded to `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/v12-formal-20260918/`; all three SHA-256 hashes were independently verified against the frozen receipt. Logs, all epoch hashes, and formal receipts are also saved locally. The eight intermediate epoch weight files remain on the server; download `detector-formal/checkpoint-epoch-01.pt` through `checkpoint-epoch-08.pt` before discarding server storage if those intermediate weights are needed. No model/dataset bytes are committed to Git. Compact evidence is preserved under `reports/v12_repro_5090/formal/` on the run branch.

At training closeout on 2026-09-18: READY FOR DETECTOR INFERENCE: YES. That task performed no development detector inference, FoundationPose work, additional training, PR merge, issue closure, or server shutdown. This records technical completion of training, not downstream accuracy or release acceptance. The subsequent detector replay is documented separately; issue #11 remains open.

## Verified export

Archive: `C:/Users/cgliu/Documents/Codex/poseloop-v12-5090/artifacts/v12-formal-20260918/poseloop-v1.2-detector-formal-export.zip`

Archive size: 190017558 bytes. SHA-256: `bae2b53ae1d433be2dd2da4ae53130c94319001ffd842fad92dabb840e52b124`.

The export contains the final checkpoint, training result, dataset manifest, training logs, launch/completion/formal receipts, all epoch checkpoint hashes, and a separate `BACKUP_SHA256SUMS` covering the files actually bundled. Intermediate epoch weight bytes are not bundled. ZIP CRC and the enclosed final checkpoint SHA-256 were verified.

Issue update: https://github.com/Yanagisawa2002/PoseLoop/issues/11#issuecomment-5729974351

The training source remains `3d5a33d64d9fbcdaecf97172fdea6f20ab647c41`; this evidence-only commit does not change or rerun training. The actual server checkout was `/root/autodl-tmp/PoseLoop-network`. The launch receipt records pre-training free disk; no reconstructed pre-training `du` snapshot is claimed.
