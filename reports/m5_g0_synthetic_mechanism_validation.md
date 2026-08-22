# PoseLoop M5-G0: Synthetic Mechanism Validation

## Claim boundary and decision

**NO-GO**

This report contains deterministic synthetic mechanism evidence only. It does
not claim real-video performance, FoundationPose improvement, official BOP
benchmark improvement, robot-control improvement, or hardware validation.

## Frozen benchmark

The sealed evaluation contains 2,304 trajectories (4 symmetries x 4 motions x
6 stresses x 24 seeds), each 120 frames at 20 Hz. All four methods saw the same
measurement hash per trajectory. Failed trajectories were retained; failures by
method are `{}`.

## Development selection

Each temporal method received the same 1,152 development trajectories and four
configuration candidates. The exact frozen normalized objective and
lexicographic tie rule selected:

| Method | Config | Objective |
|---|---|---:|
| `NEAREST_REPRESENTATIVE_CT` | `balanced` | 0.530349 |
| `STANDARD_SE3_CT` | `balanced` | 2.440443 |
| `SYMQUOT_CT` | `balanced` | 0.530349 |

## Sealed aggregate results

| Method | Median quotient error | Translation RMSE (mm) | Rotation RMSE (deg) | Symmetric switch jumps | p95 CPU ms | Failed trajectories |
|---|---:|---:|---:|---:|---:|---:|
| RAW_HOLD | 0.4601 | 4.621 | 2.547 | 4537 | 0.003 | 0 |
| STANDARD_SE3_CT | 0.2864 | 2.956 | 1.250 | 4291 | 0.284 | 0 |
| NEAREST_REPRESENTATIVE_CT | 0.2792 | 2.589 | 0.957 | 0 | 0.632 | 0 |
| SYMQUOT_CT | 0.2792 | 2.589 | 0.958 | 0 | 0.550 | 0 |

The formal symmetric combined-stress median errors are
`{'NEAREST_REPRESENTATIVE_CT': 0.291155223371368, 'STANDARD_SE3_CT': 0.3475354946353444, 'SYMQUOT_CT': 0.291155223371368}`. The relative comparison versus standard is
`{'comparator_zero': False, 'pass': True, 'threshold': 0.15, 'value': 0.16222881442119755}` and versus nearest is
`{'comparator_zero': False, 'pass': True, 'threshold': 0.03, 'value': 0.0}`.

## Representation stability against both baselines

| Symmetry | Stress | Standard jumps | Nearest jumps | SymQuot jumps | Gate |
|---|---|---:|---:|---:|---|
| C2 | REPRESENTATION_SWITCH | 703.0 | 0.0 | 0.0 | FAIL |
| C2 | COMBINED | 598.0 | 0.0 | 0.0 | FAIL |
| C4 | REPRESENTATION_SWITCH | 463.0 | 0.0 | 0.0 | FAIL |
| C4 | COMBINED | 402.0 | 0.0 | 0.0 | FAIL |
| CONTINUOUS_AXIAL | REPRESENTATION_SWITCH | 1150.0 | 0.0 | 0.0 | FAIL |
| CONTINUOUS_AXIAL | COMBINED | 975.0 | 0.0 | 0.0 | FAIL |

The jump criterion is **FAIL**.
The zero-comparator rule is enforced: zero-versus-zero cannot satisfy a requested
relative reduction.

## Ten-frame dropout

| Method | Final-missing quotient error | First-accepted quotient error | Recovery frames |
|---|---:|---:|---:|
| RAW_HOLD | 1.6597 | 0.4519 | 0.0 |
| STANDARD_SE3_CT | 1.0147 | 0.4617 | 0.0 |
| NEAREST_REPRESENTATIVE_CT | 1.0166 | 0.4634 | 0.0 |
| SYMQUOT_CT | 1.0167 | 0.4634 | 0.0 |

The formal first-accepted comparison is `{'comparator_zero': False, 'pass': False, 'threshold': 0.2, 'value': -0.003597456233740597}`.

## Outlier bursts

| Method | Accepted outlier frames | Rejected outlier frames | Median max error | Median post-burst recovery | Mean contamination frames |
|---|---:|---:|---:|---:|---:|
| RAW_HOLD | 2304 | 0 | 5.1641 | 0.0 | 0.00 |
| STANDARD_SE3_CT | 107 | 2197 | 0.5352 | 0.0 | 0.20 |
| NEAREST_REPRESENTATIVE_CT | 0 | 2304 | 0.4919 | 0.0 | 0.00 |
| SYMQUOT_CT | 0 | 2304 | 0.4919 | 0.0 | 0.00 |

## Quotient accuracy, dropout, dynamics, and asymmetry

- Symmetric combined-stress accuracy: **PASS**.
- Ten-frame symmetric dropout recovery: **FAIL**.
- Asymmetric translation/rotation/lag guardrail: **PASS**.
- Dynamic symmetric direction-change guardrail: **PASS**.
- Continuous-axis observability/gauge criterion: **PASS**.
- Numerical, contract, and p95 CPU latency criterion: **PASS**.

Exact guardrail inputs are:

- Asymmetric: `{'STANDARD_SE3_CT': {'median_max_direction_lag_frames': 1.0, 'rotation_rmse_deg': 0.9960405037686156, 'translation_rmse_m': 0.0025715976631855344}, 'SYMQUOT_CT': {'median_max_direction_lag_frames': 1.0, 'rotation_rmse_deg': 0.9960405037686156, 'translation_rmse_m': 0.0025715976631855344}}`; translation degradation
  `{'comparator_zero': False, 'pass': True, 'threshold': 0.03, 'value': 0.0}`, rotation degradation
  `{'comparator_zero': False, 'pass': True, 'threshold': 0.03, 'value': 0.0}`, lag worsening
  `0.0` frames.
- Dynamic symmetric median maximum lags: `{'NEAREST_REPRESENTATIVE_CT': 1.0, 'STANDARD_SE3_CT': 1.0, 'SYMQUOT_CT': 1.0}`;
  SymQuot worsening `0.0` frames.
- Continuous axial velocity RMS: `{'STANDARD_SE3_CT': 8.501954503352902, 'SYMQUOT_CT': 0.0005186830322874675}`;
  reduction `{'comparator_zero': False, 'pass': True, 'threshold': 0.8, 'value': 0.9999389924949512}`.
- SymQuot p95 CPU latency: `0.550` ms/frame;
  failed trajectories `0` and non-finite
  output values `0`.

Continuous axial yaw is unobservable. The estimator output gauge is a continuity
choice only; accuracy scores transformed-axis direction and never claims yaw
recovery.

## Frozen representative and aggregate figures

- `timeline_c2_representation_switch.png`
- `timeline_continuous_axial.png`
- `timeline_dropout_10.png`
- `timeline_outlier_burst.png`
- `timeline_direction_reversal.png`
- `aggregate_quotient_error_by_symmetry.png`
- `aggregate_jump_counts.png`
- `aggregate_dropout_recovery.png`
- `aggregate_asymmetric_guardrail.png`
- `aggregate_runtime_distribution.png`

Representative trajectories are the middle seed in sorted sealed-seed order for
five conditions declared before outcomes were inspected.

## Anti-cherry-picking answers

1. **Would nearest-equivalent representation achieve the same result?** The sealed per-condition comparison against NEAREST_REPRESENTATIVE_CT did not establish the required incremental benefit.

2. **Physical quotient accuracy or only coordinate continuity?** The combined-stress quotient-accuracy gate passed.

3. **Useful through dropout and outliers?** The formal ten-frame dropout gate failed. The threshold-free sealed outlier diagnostic (not a classification gate) is: NEAREST_REPRESENTATIVE_CT: rejection=1.000, mean contamination=0.000 frames, recovery failures=0 (0.000); RAW_HOLD: rejection=0.000, mean contamination=0.000 frames, recovery failures=0 (0.000); STANDARD_SE3_CT: rejection=0.954, mean contamination=0.195 frames, recovery failures=0 (0.000); SYMQUOT_CT: rejection=1.000, mean contamination=0.000 frames, recovery failures=0 (0.000).

4. **Material observable-motion lag?** The dynamic symmetric lag guardrail passed.

5. **Asymmetric objects preserved?** The asymmetric translation, rotation, and lag guardrails passed.

6. **Continuous yaw treated as unobservable?** Yes. Accuracy uses transformed-axis direction only; arbitrary axial gauge is diagnosed, never scored as physical yaw.

7. **Temporally structured representation switching?** Yes. The frozen corruption is a persistent seeded Markov process with minimum dwell, not independent labels.

8. **Survives a sealed seed set?** The mechanical sealed classification is NO-GO over the complete retained result set.

9. **Synthetic upper-bound evidence?** All evidence in M5-G0 is synthetic mechanism evidence; none is real-video, FoundationPose, BOP benchmark, robot, or hardware evidence.

10. **Evidence still needed for a real-video claim?** A frozen replay on real pose-predictor streams with timestamped dropouts/outliers, object symmetries, labeled pose truth, sealed thresholds, and latency measurements.

## Limitations and next evidence

The generator is bounded synthetic SE(3), noise and corruptions are declared,
and model-point normalized MSSD is diagnostic only. The result cannot establish
benefit on real predictor failure modes. A subsequent G1 would need frozen,
causal real-prediction replay and labeled symmetry-aware truth before any
real-video claim.
