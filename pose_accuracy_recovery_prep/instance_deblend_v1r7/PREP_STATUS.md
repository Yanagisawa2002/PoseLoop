# A-R7 PREP status

Status: **PREP only / formal workload NO-GO**.

A-R6 commit `b9e6e4b6e0172135142ace74718b3d65a635509e` and its
selection-policy hash are immutable. Historical A-R6 output and its scene-9
visualization were already opened to classify the failure as merged adjacent
occluded instances; that exposure is recorded rather than hidden. Raw replay
RGB/depth was not used. From the A-R7 freeze point, the ten frames are a
one-shot post-commit content replay and cannot choose prompts, thresholds,
weights, checkpoint, stopping criteria, or retry policy.

A-R7 changes the proposal family. It will use SAM2.1 RGB embeddings, raw-depth
discontinuity seeds, and every rival seed as a negative prompt. A child must
contain its own seed and exclude all rivals; visible child masks are then made
mutually exclusive without an IoU tuning knob. The exact A-R6 policy is reused
after new child proposals are produced.

Development selection is restricted to new CAD-rendered synthetic scenes with
disjoint training/development seeds and all six required clean/touching/
same-CAD/depth-occlusion/container strata. The synthetic gate requires strict
recall@0.75 improvement and strict merge reduction on every occlusion stratum,
with no clean-singleton regression.

Current blockers are intentionally fail-closed:

- SAM2 source tree/archive identity is not frozen;
- SAM2.1 Hiera-L checkpoint bytes/SHA are not frozen;
- the model-injected RGB-D prompt core is present, but the hash-locked SAM2
  runtime and post-deblend CAD scorer are not integrated;
- the synthetic CAD occlusion benchmark is not built.

Until a successor protocol closes all four, no download, GPU inference,
scene-9 replay, FoundationPose, scorer, or downstream export is authorized.
