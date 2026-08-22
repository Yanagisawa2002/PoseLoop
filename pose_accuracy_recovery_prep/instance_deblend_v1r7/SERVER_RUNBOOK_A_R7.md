# A-R7 server runbook (blocked PREP)

Do not deploy this PREP revision. `contract-check` is the only authorized CLI:

```powershell
python -B -m pose_accuracy_recovery_prep.instance_deblend_v1r7 contract-check `
  --protocol protocols/poseloop_pose_accuracy_recovery_instance_deblend_v1r7.json `
  --repository-root .
```

Before any GPU job, create a reviewed successor commit that:

1. pins the official SAM2 checkout commit, tree, clean source inventory, source
   archive bytes/SHA, exact imported module origins, config bytes/SHA, and the
   SAM2.1 Hiera-L checkpoint bytes/SHA;
2. binds a new CAD-rendered synthetic occlusion dataset with disjoint fixed
   train/development seed intervals and no dependency on the ten replay frames;
3. integrates an RGB-D prompt adapter whose seeds come from raw sensor depth,
   uses all rival seeds as negatives, and preserves prompt-exclusion evidence;
4. runs A-R6 and A-R7 on the same synthetic rows and passes every frozen gate;
5. commits code, protocol, dataset manifest, and synthetic result receipt before
   opening any of the ten replay RGB/depth assets.

Only then may a single immutable GPU job replay all ten frames. Scene-9 is an
outcome, not a tuning gate: PASS permits human content review; failure records
`NO_GO_NO_THRESHOLD_TUNING_ON_REPLAY`. No result-driven retry is allowed.

The eventual job must use `screen` plus `tee`, a create-only output root,
source/checkpoint preflight before model import, per-frame atomic receipts,
independent disk validation, and explicit zero counters for GT, FoundationPose,
official scorer, and downstream export. Server shutdown remains separately
authorized.
