# RTX 5090 reproduction and project freeze

The missing detector checkpoint was recovered by deterministic retraining. Its
SHA-256 and the detector prediction manifest match the frozen v1.1 identities
exactly. The original A9 route then completed **820/820 registrations** and
reproduced all recorded aggregate and per-scene evaluation metrics.

The sequence is: original result → unavailable checkpoint → deterministic
retraining → bit-exact checkpoint recovery → exact detector-output replay → exact
A9 recorded-metric replay → [770-instance failure analysis](FAILURE_TAXONOMY_RESULT.md)
→ [Nsight-driven experiment and rejected optimization](GPU_PERFORMANCE_CASE.md).

## What was reproduced

| Gate | Evidence | Interpretation |
| --- | --- | --- |
| Detector training | 8 epochs, 1,600 optimizer updates; 2,117.200454 s on RTX 5090 | One formal recovery run using the unchanged training contract |
| Checkpoint | `a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b` | Byte-identical to frozen v1.1 checkpoint |
| Detector manifest | `dded85d8d9bec68f3c6d80343183635bd80e738a28e9a93a87531d93fbe7441b` | Exact manifest identity; bound frame metadata and masks verified |
| Detector population | 997 ranked, 820 operating predictions | Original thresholds retained |
| A9 runtime | 820/820 successful registrations; 551.107429348 s | Original A9, seed 0, 5 iterations, 252 candidates, refine=32 |
| A9 evaluation | 577 mask matches, 482 joint successes; F1 0.6062893081761006 | All recorded aggregate and five-scene metrics exact |

Detector byte identity and A9 evaluation equivalence are different claims.
Historical raw A9 poses are unavailable for comparison, so per-instance historical
pose identity is **not established**. New timing, environment and run-lock fields
also mean that A9 output files are not byte-identical to the old artifacts.

The 25 frames / five scenes / 770 GT instances are already-consumed XYZ-IBD
development data. These are custom frozen metrics, not a fresh external evaluation
or an official BOP leaderboard result. Primary runtime counters recorded zero
label, GT-path, evaluator-path, scorer and scene-9 accesses before evaluation;
this is runtime instrumentation, not external filesystem tracing.

## Compact evidence and provenance

- [Summary, training/environment receipts, epoch hashes and GPU decision](../reports/reproduction_5090/summary.json)
- [Detector identity and metric comparison](../reports/reproduction_5090/detector-replay.json)
- [A9 comparison, primary receipt, run lock and runtime identity](../reports/reproduction_5090/a9-replay.json)
- [Taxonomy counts and artifact bindings](../reports/reproduction_5090/taxonomy-summary.json)

The compact JSON is a derived view, not a replacement for the original receipts.
`summary.json` binds its source receipts by commit, Git-blob SHA-256 and permalink.
The original training receipt's provisional “new v1.2 detector” interpretation
was superseded by verified historical checkpoint equality; original bytes remain
in history. Full reports, CSVs, inventories and logs are retained at
[the pre-pruning commit](https://github.com/Yanagisawa2002/PoseLoop/tree/4a96a7c102510abc43eee4b35e40f02cd932bee2/reports).
The performance evidence remains on the
[closed, unmerged PR #13 branch](https://github.com/Yanagisawa2002/PoseLoop/tree/1ce0d7a94d1fc08b21875b6dc7f18085d86eb36f/reports/gpu_perf_case).

Final checkpoint and experiment archives have verified off-server backups.
They are **not distributed by these JSON files**: a digest is an identity, not a
download. An independent GPU replay still requires the data, upstream weights,
pinned dependencies and a checkpoint with the expected hash.

## Reusable reproduction path

Use the existing detector CLI and original release runner; no A10 fallback,
dynamic pose protocol or temporary-server orchestration is needed. The recorded
environment was Python 3.12.3, torch 2.8.0+cu128, torchvision 0.23.0+cu128,
CUDA 12.8 and driver 595.71.05. Exact upstream revisions, weight hashes and
environment fixes are in `a9-replay.json`. Before downloads on the original
AutoDL host, run `source /etc/network_turbo`.

The following documents recovery commands; it is not a request for another run.
Paths are placeholders for separately acquired pinned assets; output roots must
be new. Use the recorded training environment, seed and worker configuration.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 build-dataset \
  --protocol protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json \
  --dataset-root /datasets/xyzibd --output-root /runs/recovery-dataset
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 train \
  --protocol protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json \
  --dataset-root /datasets/xyzibd \
  --dataset-manifest /runs/recovery-dataset/dataset-manifest.json \
  --weight /models/pinned-maskrcnn-pretrained.pth \
  --output-root /runs/recovery-training --device cuda:0
python -B scripts/verify_reproduction_5090.py \
  --checkpoint /runs/recovery-training/model-final.pt
bash scripts/run_release_pipeline.sh \
  --dataset-root /datasets/xyzibd \
  --detector-checkpoint /runs/recovery-training/model-final.pt \
  --foundationpose-root /opt/FoundationPose --toolkit-root /opt/bop_toolkit \
  --output-root /runs/replay-v1.1
```

Matching the recorded environment is necessary context, not a guarantee of
bit-exact recovery on arbitrary hardware/software. Stop if an identity gate fails.
CPU-only report verification is `python -B scripts/verify_reproduction_5090.py`;
it checks tracked evidence consistency and frozen source identities, not GPU execution.

## Freeze policy

PoseLoop is frozen for further experimental development. No Candidate #2, fallback
route or additional tuning on this development set is planned. Reopen work only
for a bug fix, README/portfolio presentation, or genuinely new external evaluation
data. The original v1.1 release, A9 implementation and protocols remain untouched.
This is a development freeze, not GitHub repository archival.
