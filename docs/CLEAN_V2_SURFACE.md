# Clean v2 engineering surface

The frozen v1.1.0 evidence tree should remain intact. A maintainability cleanup
should create a small supported facade instead of rewriting historical research
packages in place.

## Proposed public surface

```text
poseloop/
  __init__.py
  detector.py
  pose.py
  evaluation.py
  contracts.py
  cli.py
```

## Interface goals

`detector.py`
: Build/load the supported detector, run inference, and return a documented
  instance-prediction structure without exposing experiment package names.

`pose.py`
: Accept an explicit RGB/depth/camera/CAD/mask request and execute the supported
  FoundationPose adapter with bounded/resumable execution semantics.

`evaluation.py`
: Contain or wrap supported instance and joint-pose metric entry points with clear
  distinction between custom project metrics and official upstream evaluators.

`contracts.py`
: Centralize reusable identity/hash/manifest validation primitives. Frozen protocol
  values remain in their historical modules/files; the facade must not silently
  reinterpret them.

`cli.py`
: Provide discoverable `check`, `predict`, `pose`, and `evaluate` entry points for
  current work. The frozen `scripts/run_release_pipeline.sh` remains the v1.1.0
  release reproduction entry point.

## Migration rules

1. Add facade tests before moving implementation.
2. Prefer thin wrappers first; do not mass-rename historical packages.
3. Preserve frozen snapshots and release hashes.
4. Mark historical package names as provenance paths rather than pretending they
   are current API.
5. Keep third-party/runtime boundaries explicit.
6. Measure success by code-reading path and interface clarity, not deletion count.

## Done criteria

A new reviewer should be able to understand the supported code path by reading no
more than the facade plus the release runner, while historical experiments remain
replayable and the v1.1.0 evidence verifier still passes.
