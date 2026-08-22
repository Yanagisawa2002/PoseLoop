# PoseLoop R3 validation receipt

Protocol: `poseloop.r3.bop-industrial.e2e.v1`

Baseline: `e5e14ab6cf5f0b4adf97ca161ec04f188f6ea7e5`

Validation date: 2026-08-16

## Local validation

- `python -B -m unittest discover -s tests/r3_bop_industrial -v`: 12/12 passed.
- `python -B -m compileall -q r3_bop_industrial tests/r3_bop_industrial`:
  passed.
- `python -B -m json.tool protocols/poseloop_r3_bop_industrial_protocol.json`:
  passed.
- WSL label-blind dry-run: `blocked` only by the five absent prediction roles.
  The XYZ-IBD dataset preflight and pinned clean BOP Toolkit checkout passed;
  six official evaluator commands were constructed; evaluator-only label paths
  accessed: 0; official evaluation executed: false. Fingerprint:
  `0c669ac053ffd845bc73c9a39da487ffdd1f82c7f976c1fbd27e1c014d9e6537`.
- Negative freeze gate with the incomplete bundle: exit 1 and no input-lock file
  created.

Primary local dry-run evidence:
`reports/r3_bop_industrial/dry_run_receipt.json`.

## Remote GPU-A validation

- R3 contract suite: 12/12 passed.
- BOP Toolkit/NumPy/SciPy/OpenCV/pycocotools imports passed; official BOP24 and
  BOP22 CLI probes exited 0.
- Pinned GitHub archive source-tree contract: 93 files, SHA-256
  `9726f6d1f189bd75e665205f62e9afcf021876270aadedcb41af8e8b1e8ff307`,
  accepted.
- Label-blind dry-run: six official commands constructed, evaluator-only label
  paths accessed: 0, official evaluation executed: false. Fingerprint:
  `f6558889f27086657073af58cb10930f9adefdc0f5617065f0903afd50741fae`.

The remote dry-run remains blocked by the absent full validation split and two
target files, the same five prediction roles, and unavailable Git ancestry in
the minimal code-only remote copy. See
`reports/r3_bop_industrial/remote_gpu_a/README.md` and the retained evidence
archive beneath that directory.

## Result boundary

No label-access authorization was created or used. No official AR/AP evaluation
ran, no score was opened, and no R3 numerical result is claimed. Passing tests,
imports, CLI probes, and dry-runs are structural evidence only.
