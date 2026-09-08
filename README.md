# PoseLoop

**Turn crowded RGB-D scenes into separate object instances and 6D poses.**

In industrial bin-picking, touching and occluded parts make the detection-to-pose
handoff difficult. I built a pipeline that trains an instance detector, passes
its masks to FoundationPose, and evaluates the complete chain.

## Results

[![RGB-D, instance masks and projected 6D poses](docs/media/poseloop-demo-poster.jpg)](docs/media/poseloop-demo.mp4)

- **Instance F1: 0.038 → 0.726** at IoU 0.50, replacing the preceding generic
  proposal stack with a supervised class-agnostic Mask R-CNN on the same evaluation.
- **End-to-end joint pose F1: 0.606**, measuring successful detection and pose together.

Measured on a fixed XYZ-IBD development split: 25 frames, five scenes and 770
ground-truth instances. [Watch the 73-second walkthrough](docs/media/poseloop-demo.mp4).

## Visual walkthrough

[![Engineering overview and evidence](docs/portfolio/overview.svg)](docs/portfolio/overview.png)

The stage diagram complements the real-scene video above; the paired bars show how pose correctness changes the end-to-end metrics. [Sources and reproduction](docs/portfolio/README.md).

## Engineering challenges

1. **Separate heavily occluded instances.** A downstream pose model needs usable
   per-object masks; proposal quality determines how much of the scene reaches it.
2. **Evaluate the whole handoff.** Camera/CAD inputs, object symmetries and missed
   detections must remain consistent through pose execution and scoring.

## My contribution

I implemented detector preparation/training/inference, the FoundationPose
adapter and resumable per-mask execution, plus symmetry-aware evaluation and
release packaging. FoundationPose supplies the pose model and registration
algorithms; the implementation table below documents the upstream boundaries.

## Evidence and reproduction

[Detector evaluation](pose_accuracy_recovery_prep/real_instance_detector_v1/DEVELOPMENT_RESULT.md) ·
[End-to-end results](pose_accuracy_recovery_prep/a9_foundationpose_e2e/RESULT.md) ·
[Run the pipeline](#run-the-frozen-pipeline) · [Source navigation](docs/SOURCE_NAVIGATION.md).
The full metric table and development-evaluation context are available below.

## Pipeline

```mermaid
flowchart LR
    RGBD["RGB-D frame"] --> DET["Class-agnostic Mask R-CNN"]
    DET --> MASKS["Separate instance masks"]
    MASKS --> FP["FoundationPose registration"]
    CAD["Known CAD model + camera calibration"] --> FP
    FP --> POSES["Per-instance SE(3) poses"]
    POSES --> EVAL["Symmetry-aware development evaluation"]
```

The public entry point runs this exact inference path. It deliberately excludes
the retired exploratory branches and does not retune on the evaluation split.

## Implementation and upstream responsibilities

| Layer | Work in this repository | Upstream capability |
| --- | --- | --- |
| Instance detection | Dataset preparation, class-agnostic detector training/inference, mask handoff and evaluation. | Mask R-CNN architecture and its framework implementation. |
| 6D pose | FoundationPose adapter, frozen inputs, bounded execution/resume and per-mask orchestration. | FoundationPose's pose model, checkpoints and registration/refinement algorithms. |
| End-to-end evidence | Symmetry-aware evaluation, failure analysis, release checks and reproducible result packaging. | XYZ-IBD data, CAD models and BOP Toolkit utilities, under their respective terms. |

The engineering contribution is the measured detector-to-pose system and its
evaluation boundary. This project does not claim authorship of FoundationPose
or a new underlying pose network.

## Start with the release path

The supported entry point is `scripts/run_release_pipeline.sh`; the repository
map below identifies its implementation. Earlier experiment packages remain at
their original paths because archived tests and experiment modules import them.
Use the [source navigation and dependency audit](docs/SOURCE_NAVIGATION.md) to
separate the release path from historical exploration without breaking replay.

## Run the frozen pipeline

The full path requires Ubuntu, an NVIDIA GPU, the pinned XYZ-IBD development
data, the frozen detector checkpoint, FoundationPose, and BOP Toolkit. External
data, model weights, and third-party source are never stored in this repository.

```bash
git clone https://github.com/Yanagisawa2002/PoseLoop.git
cd PoseLoop

# Check the two frozen contracts without a GPU.
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 \
  protocol-check \
  --protocol protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e contract-check

# Run detector inference -> FoundationPose -> evaluation -> evidence package.
bash scripts/run_release_pipeline.sh \
  --dataset-root /datasets/xyzibd \
  --detector-checkpoint /models/poseloop-maskrcnn.pt \
  --foundationpose-root /opt/FoundationPose \
  --toolkit-root /opt/bop_toolkit \
  --output-root /runs/poseloop-v1.1.0
```

The runner is fail-fast and create-only. It verifies the frozen detector
checkpoint, reconstructs the dataset manifest, records the exact Git identity,
executes all 820 pose registrations, evaluates only after primary inference is
complete, and emits a hashable evidence archive.

For a CPU-only check of the tracked release summary and media:

```bash
python -B scripts/verify_portfolio.py
```

The verifier checks the original release manifest against the frozen README
snapshot and the unchanged result/media/script files. The editable project
overview is separate from that release snapshot.

## Evidence and design choices

- [End-to-end result and immutable evidence identity](pose_accuracy_recovery_prep/a9_foundationpose_e2e/RESULT.md)
- [Detector result](pose_accuracy_recovery_prep/real_instance_detector_v1/DEVELOPMENT_RESULT.md)
- [Release result bundle](release/v1.1.0/results.json)
- [Retired hypotheses and negative results](docs/archived-negative-results.md)
- [Third-party licenses and dataset attribution](LICENSES.md)

The release keeps prediction-time labels, evaluator inputs, and official scorer
access at zero until primary inference is frozen. Runtime inputs, upstream
commits, checkpoints, protocols, output manifests, and evidence members are
SHA-256 bound. Long FoundationPose scoring is chunked to keep memory bounded,
and the primary runner supports exact resume without changing candidate
attention or scoring semantics.

## Repository map

- `pose_accuracy_recovery_prep/real_instance_detector_v1/` — detector training,
  inference, and evaluation.
- `pose_accuracy_recovery_prep/a9_foundationpose_e2e/` — frozen detector-to-pose
  handoff, primary execution, evaluation, and packaging.
- `foundationpose_runtime_prep/` — audited FoundationPose adapter and
  memory-bounded execution.
- `protocols/` — immutable experiment contracts.
- `release/v1.1.0/` — compact public result bundle.
- `docs/media/` — release video, poster, and attribution.

<details>
<summary>Evaluation details, tradeoffs and supported scope</summary>

## Result

The detector was evaluated on a fixed 25-frame, five-scene split containing
770 ground-truth instances. The frozen masks were then passed through
FoundationPose without changing its model, checkpoints, candidate count,
refinement count, or promotion thresholds.

| Stage | Metric | Result |
| --- | --- | ---: |
| Instance segmentation | Precision / recall / F1 at IoU 0.50 | 0.704 / 0.749 / **0.726** |
| Instance segmentation | AP50 / AP75 / PQ | **0.702** / 0.101 / **0.510** |
| End-to-end pose | Runtime completion | **820 / 820** |
| End-to-end pose | Joint precision / recall / F1 | 0.588 / 0.626 / **0.606** |
| End-to-end pose | Joint AP | **0.532** |
| End-to-end pose | Combined AR MSSD/MSPD | **0.638** |

The preceding generic proposal stack reached only 0.038 instance F1 on the
same evaluation. Replacing that stack with a true supervised instance detector
produced a paired mean frame-F1 gain of +0.699 with a 95% bootstrap interval of
[0.655, 0.737], positive on all 25 frames and all five scenes.

## Evaluation boundary

This is a positive result on already-consumed XYZ-IBD RealSense development
data. Training and evaluation scenes and object identities are disjoint, but
they come from the same corpus. The reported pose AP and AR are custom frozen
metrics, not official BOP leaderboard scores. This release is not sealed, does
not claim state of the art, and does not claim production real-time behavior.

Scene 10 is the strongest representative example. Scene 25 remains the hardest:
its end-to-end joint recall is 0.437, exposing misses and pose ambiguity among
thin, heavily occluded parts. AP75 of 0.101 also shows that high-IoU mask
boundaries remain substantially weaker than IoU50 instance recovery.

</details>

## License boundary

No project-level license is granted for PoseLoop's original source at this
time. The demo media is an adaptation of XYZ-IBD and is separately distributed
under CC BY-NC-SA 4.0; see [the media notice](docs/media/README.md). FoundationPose
source and checkpoints remain subject to NVIDIA's upstream terms. See
[LICENSES.md](LICENSES.md) before reproducing or redistributing any component.
