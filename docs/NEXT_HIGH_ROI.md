# Next high-ROI work

This roadmap deliberately separates work that can improve the current public
engineering surface from work that requires new GPU runs or untouched data.
The frozen v1.1.0 development result should not be retuned in place.

## P0 — close public reproducibility gap

The exact v1.1.0 Mask R-CNN checkpoint is SHA-bound by the release runner but is
not currently redistributed. Resolve this before adding more provenance layers.

Acceptable outcomes:

1. If redistribution is permitted, publish the exact checkpoint through a stable
   release/model asset and verify size + SHA-256 before use.
2. If redistribution is not permitted, document that boundary explicitly and
   provide a deterministic retraining recipe whose output is treated as a new
   model/release rather than pretending it reproduces the unavailable frozen bytes.

Do not weaken the existing checkpoint identity gate to make reproduction appear
successful.

## P0 — failure taxonomy before another model change

The tracked v1.1.0 waterfall shows:

- 770 ground-truth instances;
- 577 detector-mask matches at IoU >= 0.50;
- 482 joint pose successes;
- 193 GT instances lost before the pose gate;
- 95 additional losses after a mask match.

The next analysis should split those two coarse buckets into actionable causes:

- outright detector miss;
- duplicate / false-positive competition;
- over-segmentation;
- under-segmentation / merged instances;
- boundary quality below the IoU50/IoU75 target;
- invalid or weak depth support;
- FoundationPose registration/refinement failure;
- symmetry-sensitive evaluation failure.

The output should be a per-instance machine-readable table plus a compact chart.
Do not tune thresholds against the already-consumed evaluation split after
reviewing this taxonomy.

## P1 — GPU performance case

Profile the supported end-to-end path with Nsight Systems first, then use Nsight
Compute only on kernels/ranges that the timeline identifies as material.

Record at minimum:

- detector latency and GPU occupancy/utilization context;
- FoundationPose per-instance registration latency distribution;
- CPU preprocessing and synchronization gaps;
- H2D/D2H transfer time where visible;
- render / refine / score range timing;
- peak VRAM and allocation behavior;
- throughput for the frozen 820 registrations;
- exact GPU, driver, CUDA, PyTorch, FoundationPose commit, and run configuration.

Only optimize after the baseline trace is frozen. Candidate experiments include
reducing avoidable synchronization, batching compatible work, overlapping CPU/GPU
stages, and eliminating repeated materialization/allocation. Every optimization
must report before/after timing and confirm that the frozen predictions/metrics are
unchanged or explain why a new evaluation boundary is required.

## P1 — untouched external evaluation

The current positive result is development-only. Add a genuinely untouched
external evaluation protocol rather than spending more iterations on the same 25
frames.

Requirements:

- choose the external dataset/capture and split before model inspection;
- freeze data identity and evaluation rules before inference;
- do not use the external labels for threshold/model selection;
- report negative results without reopening the protocol;
- distinguish official benchmark metrics from project-specific diagnostics.

A modest external result with a clean protocol is more valuable than a higher
number obtained by repeatedly tuning the existing development split.

## P1 — cleaner v2 engineering surface

Do not rewrite the frozen v1.1.0 evidence tree. Build a clean facade around the
supported path, for example:

```text
poseloop/
  detector.py
  pose.py
  evaluation.py
  contracts.py
  cli.py
```

The facade should call or migrate the supported implementations behind explicit
interfaces while historical experiment packages remain available for provenance.
The success criterion is reviewer and maintainer comprehension, not deleting old
research history.

## P2 — project-level license decision

PoseLoop's original source currently has no project-level license. Choose a license
only after confirming the intended reuse boundary and compatibility with the
third-party/non-commercial components documented in `LICENSES.md`. This is a
copyright-holder decision and should not be inferred from upstream licenses.
