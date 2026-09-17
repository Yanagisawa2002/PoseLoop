# Reproduction status

PoseLoop v1.1.0 has two different reproducibility claims that should not be
conflated.

## What is independently checkable from the tracked repository

With Git LFS media materialized, the repository can verify the frozen public
release bundle against the preserved v1.1.0 snapshots:

```bash
git lfs pull
python -B scripts/verify_portfolio.py
python -B scripts/build_failure_waterfall.py --check docs/failure-waterfall.md
```

This checks tracked evidence identity and the generated stage-level accounting.
It does not execute detector or FoundationPose GPU inference.

## What is not currently turnkey

The supported GPU runner requires the exact detector checkpoint whose frozen
SHA-256 is:

`a90d4134cb36cb242e98481440cfdc782c15b65f8d1068c2964243d7152fec2b`

Those exact model bytes are not currently present in the repository or v1.1.0
Release assets, and there is no stable public download URL recorded for them.
The runner correctly rejects any checkpoint with a different identity. Therefore
an independent full v1.1.0 GPU replay is currently blocked at the detector asset
boundary rather than silently substituting another model.

The bounded investigation is recorded in PR #2. The preferred resolution is to
publish the exact checkpoint if redistribution is permitted. If it is not, keep
the limitation explicit and treat a deterministic retraining path as a new model
identity/release instead of weakening the frozen v1.1.0 claim.
