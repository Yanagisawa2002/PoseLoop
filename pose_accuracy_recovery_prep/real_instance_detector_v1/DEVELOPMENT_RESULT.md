# A-R9 real instance detector development result

Date: 2026-08-21

Status: **REAL_DEVELOPMENT_DETECTOR_POSITIVE_CONTINUE**.

A class-agnostic torchvision Mask R-CNN v2 replaced the failed geometry / SAM2
/ CAD proposal chain. It trained on complete scenes for eight object IDs,
used two different object scenes for internal validation, and ran one fixed
25-frame evaluation on five further object scenes. The final evaluation is
scene- and object-disjoint from training, but all data comes from the already
consumed XYZ-IBD Realsense development corpus. It is not sealed.

Evaluation inference was label-blind. The 25-frame development labels were
opened only after the prediction manifest was complete. Scene 9 was not read.

## Frozen comparison

| Proposal source | Predictions | TP / FP / FN at IoU 0.50 | Precision | Recall | F1 | AP50 | AP75 | PQ | Merge / split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-R8 raw FastSAM | 699 | 28 / 671 / 742 | 0.040057 | 0.036364 | 0.038121 | 0.002314 | 0 | 0.021011 | 17 / 70 |
| A-R9 Mask R-CNN | 820 | 577 / 243 / 193 | 0.703659 | 0.749351 | 0.725786 | 0.702393 | 0.101146 | 0.510410 | 40 / 84 |

Merge and split remain diagnostics rather than absolute stop gates. They rise
with the much larger number of correctly recovered instances and expose the
remaining boundary errors; they do not overturn the instance-level result.

## Promotion decision

All five pre-frozen AND conditions passed:

- paired mean frame-F1 delta: `+0.699376`;
- frame-bootstrap 95% interval: `[0.654824, 0.736649]`;
- positive frames: `25 / 25`;
- positive object scenes: `5 / 5` (minimum was four);
- recall, AP50, and PQ are each strictly above the frozen A-R8 raw FastSAM
  values.

The independent ten-frame internal health set also gave F1 `0.657596`, AP50
`0.585988`, and PQ `0.451857`. These values were not used to select a
checkpoint: the protocol fixed epoch 8 in advance.

The route decision is therefore to retain the true supervised instance
detector. Do not switch to 3D proposals merely to rescue this gate. The next
integration may use these masks as the upstream FoundationPose proposals, but
it still requires its own frozen handoff and real downstream evaluation.

## Qualitative result and limitation

All 25 GT / raw FastSAM / Mask R-CNN triptychs were reviewed. FastSAM usually
misses the dense objects or traces weak partial contours. Mask R-CNN instead
recovers separate object outlines across gears, rods, cups, brackets, and
washers. This is a visible instance-formation improvement, not only a metric
change.

The weakest scene is scene 40: reflective, heavily overlapping washers still
produce many false positives, splits, and unstable boundaries. Its frame F1
ranges from `0.411765` to `0.584615`. The overall AP75 of `0.101146` confirms
that high-IoU boundary quality remains much weaker than detection at IoU 0.50.
This should be addressed by boundary-aware refinement or calibration on a new
development protocol, not by reopening this evaluation split or changing its
operating threshold after seeing the result.

This result does not authorize scene-9 replay, a sealed claim, a new synthetic
gate, or official BOP scoring.

## Evidence identity

- implementation commit: `958d1a66857e2ac928cac1eb34d825e65121c64d`
- implementation tree: `96f24600bc2aa8be3905fd7b43f746f000375274`
- protocol SHA-256: `c200f80cd47ef338341ba9e1ba941b0cd81ba9fc28f8b1dca4c05dfb33bcc5f1`
- dataset manifest SHA-256: `b958091b602a3d9a62bc1b60ea28174d6f79ffcda5523c4e68bb33600f56810e`
- training result SHA-256: `d717840057d0b359a1720afef041479b7548bb29f874832435a7375719283991`
- final checkpoint SHA-256: `a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`
- prediction manifest SHA-256: `dded85d8d9bec68f3c6d80343183635bd80e738a28e9a93a87531d93fbe7441b`
- evaluation result SHA-256: `8678e34473ae46272292967d6a596173884f478f452f8787e11c800eab0ca0a1`
- safe evidence archive SHA-256: `ce15934bcb37cccaf677a2091ccf5df2583d711331746639aa0ef76647b784b8`
- remote run root: `/root/autodl-tmp/poseloop_a_r9_run_958d1a6`

The local safe archive contains all 25 predictions, result and lineage
manifests, training / prediction / evaluation logs and exit receipts, and all
25 downsampled qualitative triptychs. Its internal SHA manifest was verified
locally with zero mismatches. The full-resolution triptychs and model
checkpoint remain in the remote run root.
