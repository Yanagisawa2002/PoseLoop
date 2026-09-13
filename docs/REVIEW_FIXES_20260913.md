# First-run repair and provenance correction — 2026-09-13

`scripts/run_release_pipeline.sh` is create-only. Its first primary invocation no longer passes `--resume`; the tracked-worktree cleanliness check now runs before detector inference. A direct primary resume validates existing output, implementation, manifest and run-lock identity before GPU model import/allocation. A completed matching primary returns its hash-checked receipt without constructing GPU models.

For recovery after freeze completed, retain the existing dataset/predictions/freeze directories and use the same clean implementation commit:

```bash
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e validate-inputs --protocol protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json --manifest /path/to/run/freeze/input-manifest.json --verify-assets
python -B -m pose_accuracy_recovery_prep.a9_foundationpose_e2e run-primary --protocol protocols/poseloop_pose_accuracy_recovery_a9_foundationpose_e2e_v1.json --manifest /path/to/run/freeze/input-manifest.json --foundationpose-root /path/to/FoundationPose --output-root /path/to/run/foundationpose-primary --implementation-commit "$(git rev-parse HEAD)" --resume
```

Use `--resume` only if the primary directory already exists with its matching run lock. If failure happened before primary creation, omit it. After primary completion, run the existing `evaluate` and `package` CLI stages with the same paths shown in the create-only runner. Do not rerun the outer create-only shell over an existing output. A run created by an older implementation must be resumed with that exact implementation; a code upgrade deliberately invalidates its lock.

Manifest checks now compare actual row counts, unique item/frame IDs and item-to-frame references. They reject duplicated rows even if someone recomputes the manifest hash.

For current main/review code use `python scripts/verify_portfolio.py`. Frozen `verify_release.py` remains for the original release bundle. The corrected runner's original bytes are preserved in `release/v1.1.0/run_release_pipeline.snapshot.sh`; original SHA256SUMS and release results are unchanged. See `docs/corrected-provenance.json` for the 65-character prediction-hash transcription erratum.

CPU tests cover first-run preflight, same-lock recovery, changed-lock rejection, completed read-only return and frozen bundle verification. No new detector/FoundationPose GPU run was performed. The 820 registrations / 25 frames, F1 0.726 and custom joint F1 0.606 remain historical results on an already-consumed development protocol, not official BOP scores or a production acceptance.
