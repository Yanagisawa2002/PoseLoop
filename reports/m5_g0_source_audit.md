# PoseLoop M5-G0 source audit

## Starting point

- Repository: `C:/Users/cgliu/OneDrive/Documents/PoseLoop`
- Branch: `master`
- Starting commit: `3b972845dc0818e08a23328baf769e72fa407b59`
  (`v1.0.0`, `docs: prepare PoseLoop v1.0.0 release`)
- Preceding commits: `11b6d6a` (M4) and `f04b17f` (M3)
- Initial worktree: clean (`git status --short --branch` returned `## master`)
- Upstream/remotes: none configured
- Repository policy files: no applicable `AGENTS.md`

The pre-change CPU baseline was checked with:

```powershell
python -B scripts/render_precomputed_report.py --check
```

It passed with `PASS: reports\release_summary.md matches results.json`.
The FoundationPose/GPU smoke was deliberately not run because M5-G0 is a
synthetic CPU mechanism gate and the lightweight release reconstruction is the
smallest applicable existing acceptance check.

## Existing evidence and immutable boundary

The repository confirms the historical orientation rather than relying on it
as an assumption:

- M2 uses 300 targets, 1,200 logical candidate rows, 169 physical instances,
  and 15 objects. Its positive diagnostic changes the target-only combined
  symmetry-aware score from 66.87% to 75.27% at five fixed-order views.
- M3 uses 230 M2-disjoint physical instances/targets and 1,150 saved view
  predictions. It supports a view-budget trade-off but not a learned allocation
  advantage.
- M4 uses 300/1,200/169 development targets/rows/instances and
  230/920/230 holdout targets/rows/instances. Learned VOI reaches 82.54%, is
  0.63 percentage points below target-only, and has no statistically resolved
  advantage over matched controls.

M5-G0 does not alter any M2, M3, or M4 script, report, figure, precomputed
release record, or raw artifact. Hashes of those protected files are frozen
before M5 evaluation and checked again by the M5 validator.

## Reused contracts

- `scripts/m1_common.py`: atomic sorted no-NaN JSON/JSONL writers, SHA-256
  helpers, finite homogeneous-pose validation, and the existing raw
  non-symmetry-aware pose error.
- `scripts/m2_common.py`: the calibrated column-vector transform convention,
  rigid inverse, and cross-camera composition.
- `scripts/evaluate_m1.py` and vendored BOP Toolkit: official MSSD/MSPD use
  model/object-to-camera poses and apply object-frame symmetries on the right.
- `scripts/evaluate_m2.py`: the existing bidirectional symmetry-aware MSSD
  medoid is retained unchanged. M5 does not substitute its Lie innovation for
  any existing evaluator result.

## New isolated surface

M5 adds local NumPy/SciPy SE(3) utilities, an exact SO(2) quotient residual,
deterministic synthetic generation/corruption, four causal estimators, temporal
metrics, frozen manifests, a CPU runner, focused tests, validation, plots, and a
synthetic-only report. Raw trajectory and frame evidence remains under ignored
`artifacts/m5_g0/`; compact results and publication artifacts are tracked.

## Pre-sealed protocol correction

An initial development-grid execution completed before any sealed evaluation,
then an independent review identified that the dedicated frozen metric manifest
did not spell out every already-implemented dropout, outlier, dynamic, runtime,
and aggregation formula. The same review requested an explicit uncertainty
finiteness count and a visible post-dropout acceptance/reacquisition trace.
That development output was invalidated and moved to the ignored
`artifacts/m5_g0_presealed_invalidated_20260731T172000Z/` archive. It was not
used for sealed evaluation.

The formulas, finiteness checks, plot semantics, outlier diagnostic, and
classification fallback were corrected; the focused test suite was rerun; and
a fresh run began from a new freeze. A final read-only audit then found that the
diagnostic helper excluded failed rows before computing outlier failure counts.
That run was stopped before configuration selection at 3,364/13,824 rows and
moved to the ignored
`artifacts/m5_g0_preselection_invalidated_20260731T173200Z/` archive.

Failure retention and failure-rate validation were fixed, 46 focused tests
passed again, and a new freeze was written at
`2026-07-31T17:33:25.379317+00:00`. Its development and all 9,216 raw sealed
method/trajectory rows completed, but post-run aggregation stopped before
acceptance because the flattened metric and metadata records both carried the
same `symmetry_class` field. No acceptance result or classification was
generated or inspected. That run was invalidated and moved to the ignored
`artifacts/m5_g0_preacceptance_invalidated_20260731T175300Z/` archive.

The aggregation path now removes only the redundant metric copy after checking
that it equals the authoritative trajectory metadata. A 47th regression test,
a full 13,824-row development aggregate smoke, all ten plot smokes, and an
independent aggregate/acceptance/report structural review passed before the
final authoritative freeze at `2026-07-31T17:57:52.454018+00:00`. The formal
development, selection, sealed chronology, results, acceptance, and report all
derive only from this last freeze.

After sealed acceptance, release review found that the valid all-zero dropout
recovery aggregate rendered as visually blank zero-height bars. The separate
`scripts/render_m5_g0_zero_aware_plot.py` presentation step now redraws that
same frozen metric with explicit `0.0` markers and an all-zero annotation, then
writes a source/hash receipt checked by formal validation. It does not change
the sealed rows, aggregate values, acceptance, or the frozen core, metrics, and
runner hashes.
