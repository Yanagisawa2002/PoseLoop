# PoseLoop R1 sealed Photoneo validation

**INVALID_SEALED_RUN — evaluator aborted after the single label open; no stage metrics were produced.**

The label-blind preflight passed for 150 targets and 712 frozen FoundationPose
predictions. The frozen M3/M4 decisions, final pose selections, and M6 risks were
committed at `355d24e0013729c420a96422bd7e403ae28e3d94` before the evaluator was opened.
The project test split passed 127 policy/evaluation tests and 3 GPU-inference tests.

The evaluator committed its one-open receipt, read the evaluator package, and then
stopped in the inherited M2 cross-view ground-truth transform audit before computing
M3, M4, or M6 sealed metrics:

```text
Cross-view GT transformation residual exceeds the fixed M2 evaluation gate:
max normalized MSSD=6.294908689142092e-05,
max MSPD=0.013441919825472135px
```

The normalized-MSSD residual is below its fixed `5e-4` limit. The MSPD residual is
above its fixed `0.01 px` limit by approximately `0.00344 px`. This is an evaluator
validity failure, not evidence that any R1 model passed or failed on Photoneo.

The authoritative receipt is
`artifacts/r1/sealed_photoneo/sealed_open_receipt.json` with SHA-256
`ba9e212717a582b6978aab630e7834bcfef75ec4c835176578dd3d374ea68be0` and status
`evaluation_failed_after_open`. It records `evaluation_invocation_count=1` and
`labels_opened=true`. No `sealed_result.json`, evaluated-outcome stream, or opened
contract was written.

This split must not be rerun after changing the audit tolerance. A new validation
attempt requires a versioned protocol and a different untouched split. Until then,
only the already frozen development results may be claimed.
