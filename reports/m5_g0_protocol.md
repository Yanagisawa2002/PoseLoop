# PoseLoop M5-G0 frozen protocol

## Question and scope

This is a **Synthetic Mechanism Validation** of whether a symmetry-quotient
constant-body-twist estimator suppresses representation jumps and remains
useful through deterministic dropout/outliers without material lag or
asymmetric-object regression. It runs no FoundationPose inference, downloads,
training, real images, robot stack, active-view policy, VOI model, or factor
graph. It cannot support real-video, BOP benchmark, robot-control, or hardware
claims.

## Frozen universe

- 20 Hz, 120 frames, 6 seconds per trajectory.
- Symmetry: `ASYMMETRIC`, `C2`, `C4`, `CONTINUOUS_AXIAL`.
- Motion: `STATIC`, `CONSTANT_TWIST`, `CURVED_ACCELERATING`,
  `DIRECTION_REVERSAL` near frame 60.
- Stress: `NOMINAL`, `REPRESENTATION_SWITCH`, `DROPOUT_5`, `DROPOUT_10`,
  `OUTLIER_BURST`, `COMBINED`.
- Development: 4 x 4 x 6 x 12 = 1,152 trajectories.
- Sealed: 4 x 4 x 6 x 24 = 2,304 trajectories.
- Development and sealed trajectory/corruption seeds are disjoint.

Ground-truth generation is separate from corruption. Nominal measurement noise
is 0.002 m translation and 1 degree rotation. Finite representation switching
uses a seeded persistent Markov process (0.92 stay probability and at least
five frames of dwell); continuous measurements receive a seeded varying axial
gauge. Dropout starts at `38 + corruption_seed % 17`. The separate outlier
burst starts at `72 + corruption_seed % 13`, lasts three frames, and applies
approximately 0.030 m and 20 degrees of observable corruption. In `COMBINED`,
the ten-frame dropout starts at `40 + corruption_seed % 9`, and the disjoint
three-frame outlier starts at `76 + corruption_seed % 11`.

Estimator inputs contain only timestamp, optional pose measurement, missing
flag, and declared symmetry. Ground truth, stress labels, representative index,
intervals, future frames, and motion labels live in evaluator-only records.

## Compared methods

1. `RAW_HOLD`: raw measurement or last output during missing frames.
2. `STANDARD_SE3_CT`: full-rank, symmetry-unaware robust constant-twist filter.
3. `NEAREST_REPRESENTATIVE_CT`: selects finite/continuous gauge relative to
   the previous accepted output, then uses exactly the standard filter.
4. `SYMQUOT_CT`: selects finite representatives relative to the prediction and
   projects continuous axial innovation/twist out of the observable update.

All methods receive byte-identical measurement streams. Temporal methods use
the same diagonal uncertainty, process/measurement model, robust gate, and
four-consecutive-rejection reacquisition rule. That rule rejects the frozen
three-frame outlier burst while allowing recovery from persistent disagreement.

## Equal development grid

Measurement sigma is fixed at 0.002 m and 1 degree; `reacquire_after=4` for all
four configurations. Each temporal method receives exactly this same grid and
one configuration is selected across all symmetry classes.

| ID | Process t (m) | Process r (deg) | Velocity process | Gate (sigma) | Velocity gain |
|---|---:|---:|---:|---:|---:|
| `conservative` | 0.001 | 0.5 | 0.02 | 4.5 | 0.20 |
| `balanced` | 0.002 | 1.0 | 0.04 | 5.5 | 0.35 |
| `agile` | 0.003 | 1.5 | 0.08 | 6.5 | 0.50 |
| `robust_agile` | 0.003 | 1.5 | 0.08 | 4.5 | 0.50 |

For each method/configuration, minimize

\[
J = \frac{\operatorname{median}(e_q)}{1.0}
  + \frac{\operatorname{mean}(n_{jump})}{1.0}
  + \frac{\operatorname{median}(\max(l_t,l_R))}{4}
  + \frac{\operatorname{mean}(d_{outlier})}{3}
  + 100 f_{failure}.
\]

Here
`e_q = sqrt((translation_error_m/0.01)^2 + (quotient_rotation_error_deg/5)^2)`;
lag uses reversal trajectories, and contamination uses outlier/combined
trajectories. Ties within `1e-12` use lexicographic configuration ID. Selected
configurations and their hashes are written before sealed evaluation. Sealed
outcomes cannot change seeds, corruption timing, grid, objective, or thresholds.

## Frozen metrics

Results include per-frame, trajectory, condition, and aggregate translation,
quotient rotation, normalized quotient pose, synthetic finite-symmetry MSSD,
raw/quotient step, representation/gauge jumps, dropout final/return/recovery and
uncertainty growth, outlier accept/reject/recovery/contamination, reversal lag,
velocity attenuation, oversmoothing, asymmetric guardrails, CPU latency, and
finite/numerical status.

The authoritative frozen formulas are serialized in
`artifacts/m5_g0/frozen/metric_definitions.json` before development selection.
Intervals are half-open `[start, end)`. Scalar summaries retain total and
non-finite counts but compute mean, median, NumPy p95, maximum, and
`sqrt(mean(x^2))` RMSE from finite values only. Dropout reacquisition is the
first accepted update at or after `end`; recovery is the first output at or
after `end` within the nominal threshold. Outlier contamination is
`recovery_frame - end`, or the remaining post-burst horizon when censored by a
recovery failure. Reversal lag is the non-negative difference between the
first estimated and truth negative crossings projected on the mean observable
truth direction over the ten pre-reversal frames. Peak attenuation is
`max(0, 1 - peak_output/peak_truth)`, and oversmoothing is output/ground-truth
RMS observable normalized speed. Runtime summaries use finite non-negative
per-frame timings and carry exact pose, uncertainty, initialization, and
recovery failure counts.

A finite catastrophic jump is `raw_step > 90 deg` and
`quotient_step < 10 deg`. A continuous gauge jump is
`abs(axial_spin) > 45 deg` and observable quotient step `< 10 deg`. Recovery
means normalized quotient pose error `<= 1.0`. A requested relative reduction
cannot pass if the comparator count is zero; zero versus zero is not evidence
of reduction.

Representative timelines are fixed independently of outcomes: take the middle
seed in sorted sealed-seed order for the predeclared C2 switch, continuous
switch, C4 ten-frame dropout, C2 outlier, and C4 reversal conditions.

## Predeclared classification

`ALGORITHM GO` requires every threshold in the user-specified gate: at least
80% jump reduction versus standard and 30% versus nearest on every symmetric
switch/combined condition; at least 15% combined-stress median quotient-error
improvement versus standard and no more than 3% degradation versus nearest;
at least 20% ten-frame-dropout first-accepted improvement versus standard;
asymmetric translation/rotation degradation no more than 3% and lag worsening
no more than one frame; dynamic symmetric lag no more than one frame worse than
the best non-oracle temporal baseline; stable exact continuous observability
with at least 80% axial-velocity-RMS reduction; no NaN/Inf or hidden truth; and
p95 CPU latency below 10 ms/frame.

Use `MECHANISM PASS / REAL-DATA VALUE UNPROVEN` only if the exact jump,
continuous, asymmetric/dynamic, and numerical/runtime gates pass, at least one
of the combined-accuracy or dropout value gates passes, and the other prevents
`ALGORITHM GO`. If both value gates fail, use `NO-GO`; continuity alone is not
enough. Outlier comparisons against both temporal baselines are frozen as a
threshold-free diagnostic and cannot upgrade the classification. The
classification is recomputed mechanically from raw sealed trajectory rows and
negative results are retained.
