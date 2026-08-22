# PoseLoop R3 GPU-A official-once execution receipt

Protocol: `poseloop.r3.bop-industrial.e2e.v1`

Baseline: `e5e14ab6cf5f0b4adf97ca161ec04f188f6ea7e5`

Execution branch: `codex/poseloop/r3-bop-official-run`

Execution date: 2026-08-16

## Frozen input readiness

- GPU-A staged the complete 7,710,607,370-byte XYZ-IBD validation archive.
  Its SHA-256 is
  `09c5639c6e55b8c9a0708a344037e98918c5935ddbd91b5db5a9b06f5484c659`.
- Public BOP19 and BOP24 target files contained 60 rows each and matched
  SHA-256 values `6ebafd20964d09f649a2e1acc98e5b8b1c39d89e746a0b8865c887f6d74fa011`
  and `7f620e5bdf8d6f11201d90464c5a93963ea37718ff662e5a392ca41eb10d6ebe`.
- The public `models_eval/models_info.json` described 15 objects and matched
  SHA-256
  `25b876a00ed7585f083f4b634c32fe13a3a548bee1f6fb033722932331e62374`.
- GPU-C produced the prediction bundle with FoundationPose using 252
  hypotheses. The single-shard run completed 8/8 items; the two-shard run
  completed 4/4 items per shard. Raw single- and two-shard pose matrices,
  scores, and margins were exactly equal.
- The frozen prediction archive is 195,822 bytes with SHA-256
  `9a4de5cd0b475fd3a930d596a95e90f75df1f82503d48681d11357c9e527f473`.
  It contains exactly five R3 inputs and 134 provenance/evidence payloads.
- The five input SHA-256 values are:

  - COCO detections/segmentations:
    `5ef98c1ea24a9ecc141d3f1af6a319dba1ce00f4cb21d522b99b4f1d3ab1a719`
  - predicted association:
    `6883a6ff9ea8fffaa835eb92e2f717a1fb6ff4eb6e5194924bd76becbb26d7b2`
  - single-view pose CSV:
    `459ca38d8f7d6d9d7ebafad6c7a65a3147b8776e79e03517f3e93c4d412eb017`
  - multi-view pose CSV:
    `145f19e1d545472a49e16a0954a291d734e3109a59e71f2ba533fcf576244066`
  - provenance:
    `ecc8dd05ae38bbd739a1330701746ef1ba7b5c2f4898282bd010efdf7f4759b1`

GPU-A independently verified every archive payload and all five input hashes,
then extracted only the five named inputs and made them read-only. The final
pre-score receipt was `ready`: bundle, dataset, toolkit, repository ancestry,
and six-command construction all passed; missing roles and errors were empty;
evaluator-only label paths accessed before scoring were empty.

The GPU-A input-lock fingerprint is
`d02659e9669d9aae2a56a9ec4bbafe042eb8df9e7a72123c6ca42fb0d2ee48d1`.
The input-lock file SHA-256 is
`1d67c92ed92124d65af51bbd835f53052a02e28ce06ef820fffc15ef50538a98`.
The evaluator-only authorization was bound to that exact lock; its SHA-256 is
`7daec264f6f514d1866f7986b36c729c4f83e23bb371119d17d625c1db8e0ffc`.

## Unique official evaluation outcome

Exactly one `evaluate` invocation was started in detached screen
`poseloop_r3_official_once`. A noclobber marker binds the invocation to the
input-lock and authorization hashes and prevents a second launch.

The invocation exited 1 in command 1 of 6, the single-view BOP19 localization
AR command. The pinned official `eval_bop19_pose.py` unconditionally requested
the VSD error, while the same pinned toolkit declares the supported XYZ-IBD
error types as `ad`, `add`, `adi`, `mssd`, and `mspd`. The exact failure was:

```text
ValueError: vsd error is not among xyzibd supported error types: ['ad', 'add', 'adi', 'mssd', 'mspd']
```

This is classified as
`frozen_protocol_metric_is_unsupported_by_pinned_toolkit_for_xyzibd`. The raw
official log is 4,234 bytes with SHA-256
`5b68885d0c64bcd35d0918a2188b519ecb05d8d6ffb8546f9ac2b9fa45897146`.

No official score file was produced. Commands 2 through 6 were not run. No
BOP19 AR, BOP24 mAP, BOP22 bbox AP, BOP22 segmentation AP, or
multi-minus-single delta is claimed. The frozen protocol was not changed and
the official evaluator was not retried.

## Runtime and evidence boundary

Before the unique invocation, GPU-A's Vispy/EGL public-model smoke exposed a
missing `libEGL.so.1`. The minimum Ubuntu `libegl1` dependency and its required
system packages were installed, after which the same smoke passed with EGL,
GL2, finite `(48, 64)` depth output, and exit 0. This smoke did not invoke an
official evaluator or access label files.

Primary retained evidence:

- local archive:
  `artifacts/r3_bop_industrial/gpu_a_official_once/r3_gpu_a_official_once_evidence.tar.gz`
  (9,560 bytes, SHA-256
  `a09b21ec3d81aa871b18b9e0e858bb43f9284bd0bfff554614781f798b552591`)
- local extracted failure receipt:
  `artifacts/r3_bop_industrial/gpu_a_official_once/extracted/evidence/r3_gpu_a_official_once_failure.json`
- local extracted raw official log:
  `artifacts/r3_bop_industrial/gpu_a_official_once/extracted/workspace/artifacts/r3_bop_industrial/official_once/logs/01_official_bop19_localization_ar_single_view.log`
- remote evidence archive:
  `/root/autodl-tmp/poseloop_r3_20260816_01a003cf/evidence/r3_gpu_a_official_once_evidence.tar.gz`
- remote failure receipt:
  `/root/autodl-tmp/poseloop_r3_20260816_01a003cf/evidence/r3_gpu_a_official_once_failure.json`

The remote failure receipt SHA-256 is
`37b44d5f869502be3ce8b5f3ee3e5814297961ae3b67ef374707692ecc87fc5fe`.
GPU-A and GPU-C were kept running. No shutdown, poweroff, release, disk
deletion, push, merge, or tag operation was performed.
