# V1.2 execution surface

V1.2 retrains the existing A-R9 detector contract with a new checkpoint identity.
The original detector protocol, A9 runtime, release runner and `release/v1.1.0`
are unchanged. The 400 training, 100 internal-validation and 25 evaluation frames
are already-consumed development data, never a sealed or official BOP result.
Scene 9 is excluded. Training and evaluation thresholds remain unchanged.

## This delivery stops at detector smoke

Keep the verified torch 2.8.0+cu128 / torchvision 0.23.0+cu128 environment.
Use `/root/miniconda3/bin` in PATH. CPU contracts must pass before data download
and GPU work. `prepare_v12_data.py` sources `/etc/network_turbo` before every
download, verifies the pinned archive SHA/size, and extracts only required
RealSense frames, masks, calibration, and models. It neither downloads PBR/test
archives nor extracts scene 9. It retains the checked downloads until the
dataset manifest is successfully built.

```bash
export PATH=/root/miniconda3/bin:$PATH
export PIP_NO_CACHE_DIR=1
export TMPDIR=/root/autodl-tmp/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run=/root/autodl-tmp/poseloop-v1.2-5090
python -B scripts/prepare_v12_data.py --run-root "$run"
python -B -m pose_accuracy_recovery_prep.real_instance_detector_v1 build-dataset \
  --protocol protocols/poseloop_pose_accuracy_recovery_real_instance_detector_v1.json \
  --dataset-root "$run/xyzibd" --output-root "$run/dataset"
python -B scripts/run_v12_detector_smoke.py \
  --dataset-root "$run/xyzibd" --dataset-manifest "$run/dataset/dataset-manifest.json" \
  --weight "$run/weights/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth" \
  --output-root "$run/detector-smoke" --smoke-batches 2
```

The smoke wrapper calls the original training function with two batches. It
counts actual SGD steps (including AMP skip behavior), reloads the checkpoint
strictly, checks finite loss/tensors, and records CUDA identity, elapsed time,
peak allocated memory and free disk. Its one-epoch checkpoint is explicitly
not eligible for formal detector prediction, which still requires eight epochs.

Formal training is a separate explicit call to the original detector `train`
command without `--smoke-batches`; it is not part of this task or pipeline runner.

## Later post-training pipeline

```bash
bash scripts/run_v12_pipeline.sh \
  --dataset-root "$run/xyzibd" --dataset-manifest "$run/dataset/dataset-manifest.json" \
  --detector-checkpoint "$run/detector-formal/model-final.pt" \
  --foundationpose-root /root/autodl-tmp/FoundationPose \
  --toolkit-root /root/autodl-tmp/bop_toolkit \
  --output-root "$run/full-pipeline"
```

Add `--resume` only for the identical run. Completed stages are hash-verified;
partial FoundationPose inference uses its exact run-lock resume implementation.
Other interrupted stages fail closed rather than overwrite partial evidence.

`a10_foundationpose_e2e_v2` is a deliberate isolated successor of A9. It retains
the inference and metric algorithms but derives prediction coverage, runtime
denominators and result counts from its frozen population. The generated pose
protocol is frozen before detector labels open (stronger than required), and
before any FoundationPose inference. It binds the new checkpoint, prediction
manifest, dataset manifest and unchanged detector training protocol. Its own
identity is distinct from A9. Zero operating predictions stop before pose work.

The standalone detector evaluator reuses A-R9/A-R8 metric functions, but never
loads historical A-R8 prediction/result files. All mask artifacts are validated
before GT access. Taxonomy `extract --result-anchor <evaluation-result.json>`
uses the A10 runtime and binds the anchor to the exact protocol, input,
prediction and completion receipts. Without this option, the historical v1.1
status, self-test and extraction behavior remains anchored to its old results.

The new pipeline preserves negative results as evidence, without promoting
them. No GPU optimization, retuning, or FoundationPose compatibility change is
included. A10 currently requires the original pinned FoundationPose source;
if an actual compatibility failure occurs, freeze a minimal audited patch in a
separate change before running primary inference. FoundationPose provisioning
and a real registration smoke remain a later gate, not proven by detector smoke.
