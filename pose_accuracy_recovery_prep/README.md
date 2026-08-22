# Pose accuracy recovery PREP v1

This namespace prepares the next development-only pose-accuracy decomposition. It is intentionally local and inert:

- `AUTO_DEPLOY=false`;
- no server, network, model, dataset, training, inference, or sealed-split access;
- the committed fixture is synthetic and its dry-run is not an accuracy result;
- R4-A v3 and commit `58939e69960f2227c4fa20be547e61df23fe391d` remain immutable NO-GO evidence.

## Roles

The unified manifest is evaluator-owned. The canonical future A-to-C path is
`export-c-handoff`, which creates
`poseloop.r4c.prep.runtime-isolated-manifest.v2`. GPU-A reads the two
DEVELOPMENT_ONLY control masks and copies their bytes into opaque runtime input
paths. GPU-C sees five input variants but receives no GT pose, raw GT/evaluator
path, evaluator, threshold, or score asset. Its label/GT/evaluator/scorer access
counters must all remain zero.

This five-variant bundle is **not globally label-free**: known-sample sanity and
oracle-mask control are explicitly GT-derived controls. They are fault
localization only and cannot support a deployment conclusion. The boundary says
`contains_gt_derived_control_inputs=true`, `contains_raw_gt_paths=false`, and
`gpu_c_resolves_derivation=false`. Official and internal accuracy evaluation
remain GPU-A evaluator-only operations.

`export-producer` remains as a legacy predicted-only utility for the three
non-control variants. It is not the formal A-to-C recovery path.

## Local fixture commands

```text
python -m pose_accuracy_recovery_prep --help
python -m pose_accuracy_recovery_prep validate-manifest --manifest fixtures/pose_accuracy_recovery_prep/manifest.json --data-root fixtures/pose_accuracy_recovery_prep --output <output>/validation.json
python -m pose_accuracy_recovery_prep export-c-handoff --protocol protocols/poseloop_pose_accuracy_recovery_c_handoff_v2.json --manifest fixtures/pose_accuracy_recovery_prep/manifest.json --data-root fixtures/pose_accuracy_recovery_prep --implementation-commit <full-commit> --implementation-sha256 <implementation-sha256> --model-sha256 <model-sha256> --refiner-checkpoint-sha256 <refiner-sha256> --scorer-checkpoint-sha256 <scorer-sha256> --output-root <fresh-output>
python -m pose_accuracy_recovery_prep validate-c-results --handoff-manifest <handoff>/manifest.json --handoff-root <handoff> --results <c-return>/results.jsonl --result-root <c-return> --output <output>/c-result-validation.json
python -m pose_accuracy_recovery_prep dry-run --protocol protocols/poseloop_pose_accuracy_recovery_prep_v1.json --manifest fixtures/pose_accuracy_recovery_prep/manifest.json --data-root fixtures/pose_accuracy_recovery_prep --output-root <output>
```

The dry-run writes JSON, JSONL, and CSV for contract validation, run plans, SE(3) perturbations, scorer/refiner checks, grouped internal metrics, paired deltas, official capability status, and visualization layers. Every dry-run artifact states `accuracy_claim_permitted=false` and `dry_run_is_result=false`.

The committed schema fixtures live in
`fixtures/pose_accuracy_recovery_prep/c_handoff_v2/` and
`fixtures/pose_accuracy_recovery_prep/c_results_v1/`. They exercise byte/hash
contracts and complete result structure only; their synthetic poses, identities,
latencies, and visual placeholders are not measurements.

Future evaluator-only execution uses `prepare-diagnostics` to write the exact grid, then `evaluate-diagnostics` to ingest one finite score per frozen hypothesis plus the configured number of refiner traces. Both commands require `--namespace-role EVALUATOR_ONLY`; the latter recomputes and hash-compares the grid before reading scores.

## Official metrics

The PREP code does not call BOP tooling. AR, MSSD, MSPD, and VSD are accepted only from an explicit official-output payload after a future runtime capability audit. Frozen prior R3 evidence records XYZ-IBD support for `ad/add/adi/mssd/mspd`, not VSD; therefore BOP19 AR requiring VSD remains unavailable. VSD is never removed and a reduced error set is never relabeled as official AR. Fixture outputs use `not_run_fixture`, never a numeric zero.
