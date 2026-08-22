# GPU-A supported-metric official invocation blocker

Protocol: `poseloop.r3.bop-industrial.xyzibd-supported.official.v1`

Outcome: **failed after the one authorized evaluate invocation started; no
official score was produced and rerun is forbidden**.

## Frozen boundary before scoring

- Source evidence commit `36a3aaf4698b5d3ef4440ffade731e4d1d903c8f`
  and the prior protocol evidence were left unchanged.
- Execution commit: `6d9756e425c78b4e9a4bea2fcb38a79820387207`.
- Final pre-score status: `ready`; fingerprint
  `de151f6422b9ae8f40ab0c9bb407c2a6a78efec61c1025cf38ff116d37fb2d00`.
- Official command manifest:
  `d93036a141a8ea2f464c39c50f8c95b811c19c39882a506e08d406943d12b0c0`.
- Orchestrator source manifest:
  `027f51493e3f6f5c307e029b33880e0a366ee225a48727712eba3bfd0f1fa250`.
- Input lock SHA-256:
  `ebef9c2aa7c3b8e58c4c326000bfecfcfd005961f3e3b34f9e934bc6911234f1`.
- Authorization SHA-256:
  `e2216222e05df3b97557284f8e336c01dcee66530399c970e99e09df2ef18d17`.
- Command count was exactly four. The BOP19 entrypoint command count was zero.
- Prediction/selection label-access count was zero and no evaluator-only label
  path had been accessed before the invocation.

The support audit fixed official BOP19 AR as unavailable: the pinned official
aggregate requires VSD, MSSD, and MSPD, while the pinned XYZ-IBD parameters do
not support or define VSD. It was not replaced by a reduced-error surrogate.

## Unique invocation

Screen: `poseloop_r3_xyzibd_supported_official_once`.

The new invocation marker SHA-256 is
`254228e4a1ca105846750aa414bf0760630dced428944e343656cbdc2735c087`.
Command 1 of 4, BOP24 single-view pose mAP, exited 1 before calculating any
pose error or score:

```text
ModuleNotFoundError: No module named 'bop_toolkit_lib'
```

The invoked interpreter was `/root/miniconda3/bin/python`. Post-failure
diagnostics show `bop_toolkit_lib_spec=None`, `PYTHONPATH` unset, and the pinned
toolkit package present only in the source tree. Thus the exact blocker is the
official subprocess import environment: the toolkit root was not installed or
placed on that subprocess's import path. This is not an XYZ-IBD metric-support
failure and it does not authorize a retry.

The sealed failure receipt SHA-256 is
`ca4b040d3f7639d9dc52e4b0327d838b2ccad7ee9cafc2d32283526adaa92cb9`;
the raw command log SHA-256 is
`abeb31400727099b6d37e674ffdd9328e22d6c420ee1d3df18560f0040019f1e`.
It records `evaluate_invocation_count=1`, `exit_code=1`, and
`rerun_permitted=false`. There are zero `scores*.json` files, so BOP24 mAP,
BOP22 bbox AP, BOP22 segmentation AP, and the single/multi delta are all
unavailable for this invocation. No numeric AR/AP is claimed.

GPU-A remained running after collection (RTX 5090, 0 MiB at the diagnostic
snapshot). No shutdown, poweroff, release, delete, push, merge, or tag was
performed.

## Evidence

- Imported archive:
  `artifacts/r3_bop_industrial_supported/remote_gpu_a/r3_xyzibd_supported_once_evidence_6d9756e.tar.gz`
- Archive SHA-256:
  `2e2d63b589004406fdabf4988ff9ec353dbc44c37e77a4e31ff1bf381f0a51ce`
- Extracted final pre-score receipt:
  `artifacts/r3_bop_industrial_supported/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported/pre_score/final-prescore.json`
- Extracted invocation marker, raw log, and sealed receipt:
  `artifacts/r3_bop_industrial_supported/remote_gpu_a/extracted/workspace/artifacts/r3_bop_industrial_supported/official_once/`
- Extracted environment diagnostic:
  `artifacts/r3_bop_industrial_supported/remote_gpu_a/extracted/evidence/r3_xyzibd_supported/post_failure_diagnostic.txt`
- Remote canonical archive:
  `/root/autodl-tmp/poseloop_r3_20260816_01a003cf/exports/r3_xyzibd_supported_once_evidence_6d9756e.tar.gz`
