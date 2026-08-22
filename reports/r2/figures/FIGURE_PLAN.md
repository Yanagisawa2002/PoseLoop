# PoseLoop R2 figure plan

| ID | Type | Description | Data source | Priority |
| --- | --- | --- | --- | --- |
| Fig. R2-1 | Multi-panel quantitative plot | Sealed M3/M4 gains, M6 ROC, and M6 risk-coverage | `artifacts/r2/sealed_photoneo/sealed_result.json`, `sealed_evaluated_outcomes.jsonl` | High |
| Fig. R2-2 | Qualitative real-observation comparison | Fixed-slot-2 acquisition and pose overlay versus the frozen R2 M4 selection; two rescues and the largest regression are selected deterministically after evaluation | Frozen R2 decisions, predictions, evaluator groups/labels, Photoneo images, official CAD | High |
| Fig. R2-3 | Qualitative risk examples | Two lowest-risk successful outputs and two highest-risk failed outputs, with predicted and GT CAD silhouettes on the target observation | Frozen R2 risks, sealed outcomes, Photoneo images, official CAD | High |
| Fig. M5-R3-1 | Sealed gate diagnostics | Failed all-frame, natural-missing, and bootstrap gates plus per-object heterogeneity | `artifacts/r2/m5_r3/result.json` | High |
| Fig. M5-R3-2 | Recorded sequence behavior | Within the post-evaluation visualization subset (at least 10 missing frames and baseline missing loss at most 2.0), show the greatest rescue and regression with baseline/proposed CAD overlays and full-track error replay | Frozen M5-R3 track results, Photoneo images, official CAD | High |
| Fig. M5-R6A-1 | Recorded before/after behavior | Deterministically show the weakest, global-typical, and strongest object outcomes with nearest/measurement-first CAD overlays and full 128-frame error replay | Frozen M5-R6A RU-APC development result and track results, RU-APC RGB, official CAD | High |
| Fig. M5-R6A-2 | Natural missing-confidence behavior | Show a 36-frame natural support-loss sequence at gaps 1/6/18/36, with exact nearest fallback, decaying confidence, growing uncertainty, and the first reacquired frame | Frozen M5-R6A RU-APC development track results, RU-APC RGB, official CAD | High |

The qualitative examples are post-evaluation visual evidence only. Their selection never changes a model, policy, threshold, metric, or sealed gate. These are recorded Photoneo observations, not a simulator replay.

The M5-R3 examples are likewise post-evaluation diagnostics. They visualize the
frozen failure result and must not be used to retune or reopen the consumed split.

The M5-R6A examples are post-evaluation development diagnostics selected only
after the RU-APC result was frozen. They do not alter the method or compensate
for the separate YCB-V sealed input-feasibility failure.
