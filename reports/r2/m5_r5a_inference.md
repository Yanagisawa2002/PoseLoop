# PoseLoop M5-R5A LM-O label-blind inference

Status: **COMPLETE — 1423 / 1423 SUCCESS**

This is an inference-integrity result, not a pose-effectiveness result. The run consumed only the frozen 1,423-row inference manifest. The evaluator-label path was not read and no pose error was computed during inference.

| Quantity | Value |
|---|---:|
| Unique inference samples | 1,423 |
| Successful registrations | 1,423 |
| Failed registrations | 0 |
| Mean registration time | 0.914 s |
| Median registration time | 0.910 s |
| P95 registration time | 0.995 s |
| Maximum registration time | 1.151 s |
| Total recorded registration time | 1,300.125 s |
| Maximum recorded per-sample CUDA allocation | 1,780,069,888 bytes |

| Object | Successful samples |
|---:|---:|
| 1 | 183 |
| 5 | 189 |
| 6 | 150 |
| 8 | 191 |
| 9 | 168 |
| 10 | 181 |
| 11 | 171 |
| 12 | 190 |

The resource-bounded execution retained all 252 pose candidates and unchanged candidate attention while using:

- warp batch size 32;
- scorer data batch size 8;
- scorer feature batch size 32;
- refine feature batch size 32.

Integrity receipts:

- protocol: `poseloop-m5-r5a-v1`;
- bundle contract SHA-256: `93d6a3a355d1132b3d8172d51517ca633fcf6feaeffb0cf8bf30de7bf4123512`;
- inference manifest SHA-256: `21389be67556c80bd5b9ce26f1ce46b8c533de71891271b4caa5b18c5748bd74`;
- predictions SHA-256: `e323478190013d91323c7274b8d459c2906d7776ad412c53b4a121b93dd9de26`;
- inference receipt SHA-256: `0b6c92a666571806f912c1458c8117b4a67da2104820fb8ad9f57fc72e5a497e`;
- all prediction IDs exactly matched the frozen manifest;
- inference metadata, receipt, and every prediction row declared no evaluator-label read;
- all predictions contained finite poses and a `success` status as enforced by the runner.

The run completed in one uninterrupted process without `--resume`. No development evaluator result existed when this receipt was recorded.
