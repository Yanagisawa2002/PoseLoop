# R3 XYZ-IBD supported official metric audit

Protocol: `poseloop.r3.bop-industrial.xyzibd-supported.official.v1`

This audit is read-only with respect to commit `36a3aaf4698b5d3ef4440ffade731e4d1d903c8f`,
the source protocol, its unique failed invocation, and all frozen prediction
files. It does not reinterpret or rerun that protocol.

## Pinned-source findings

The pinned BOP Toolkit revision is
`cea62d651c7e395b2e1962b9749e4e89693c6ac4`, with normalized source-tree
SHA-256 `9726f6d1f189bd75e665205f62e9afcf021876270aadedcb41af8e8b1e8ff307`.

| Metric | XYZ-IBD boundary | New command count | Pinned evidence |
|---|---|---:|---|
| BOP19 average recall | **Unavailable** | 0 | `eval_bop19_pose.py` hard-codes VSD, MSSD, MSPD and averages all three. `dataset_params.py` supports only AD, ADD, ADI, MSSD, MSPD for XYZ-IBD, so VSD is not defined. Removing VSD would not be official BOP19 AR. |
| BOP24 pose mAP | Available for fixed single and multi pose CSVs | 2 | `eval_bop24_pose.py` computes the official aggregate from MSSD and MSPD, both supported for XYZ-IBD, and writes `bop24_mAP`. |
| BOP22 bbox AP | Available for the shared frozen COCO input | 1 | `eval_bop22_coco.py --ann_type=bbox --bbox_type=amodal` writes score key `AP`. |
| BOP22 segmentation AP | Available for the shared frozen COCO input | 1 | `eval_bop22_coco.py --ann_type=segm` writes score key `AP`. |

Raw pinned file SHA-256 values:

- `scripts/eval_bop19_pose.py`: `953a016e47a31ed05a5032302b8154722049d8dd953aeb7794ecedbf607bfcce`
- `scripts/eval_bop24_pose.py`: `d3aa46b4a67b563eb11772636e1532a206e1bbbce20d23b6c4d50c5d30c4a1d4`
- `scripts/eval_bop22_coco.py`: `4ef31672df3a1584cd01aca8a749e8437c1b57e9388d1e774e5df790ef578c9f`
- `bop_toolkit_lib/dataset_params.py`: `a935fc4f6fd42f367a0bbf816036f783fb4c5d87200a8615ce1e08e6d51af05e`

## Frozen new invocation

The new evaluate call has exactly four commands, in this order:

1. BOP24 pose mAP, single-view CSV.
2. BOP24 pose mAP, multi-view CSV.
3. BOP22 bbox AP, shared COCO predictions.
4. BOP22 segmentation AP, shared COCO predictions.

Before that call, the new namespace freezes the exact protocol, source commit,
dataset receipt and public-input hashes, toolkit code hashes, prediction archive,
five prediction hashes, producer-code/model hashes, command manifest, output
directory, and the new one-shot authorization. Prediction and selection label
access remains zero. The official evaluator is the only authorized label reader.

An output directory or invocation marker created by the new call permanently
blocks any retry, whether the call succeeds or fails. The failure receipt is
written before returning an error.
