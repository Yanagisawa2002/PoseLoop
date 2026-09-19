# Reconstructed 770-instance failure taxonomy

The original A9 replay enabled a complete post-hoc diagnostic of the previously
consumed development population: **770 GT → 577 mask matches → 482 joint successes**.
The [compact summary](../reports/reproduction_5090/taxonomy-summary.json) records
artifact hashes, all five scene totals and the original classification output.

| Mutually exclusive bucket | GT instances |
| --- | ---: |
| Success | 482 |
| Detector miss | 1 |
| Mask boundary / IoU failure | 114 |
| Over-segmentation | 24 |
| Under-segmentation / merge | 54 |
| MSSD-only failure | 20 |
| MSPD-only failure | 2 |
| Both MSSD and MSPD failure | 73 |
| **Total** | **770** |

The upstream buckets sum to 193; pose-failure buckets sum to 95. There were no
runtime failures among the 820 registrations. The mask-match IoU bands were
193 below 0.50, 380 between 0.50 and 0.75, and 197 at or above 0.75.
Scene 25 contributes 166 of 288 failures (57.6%). These counts support focusing
the diagnosis on instance separation and boundary quality; they do not prove
that one model change would fix those cases.

Bucket labels follow the existing deterministic diagnostic rules in
[`build_failure_taxonomy.py`](../scripts/build_failure_taxonomy.py), not manual
causal ground truth. The aggregate chain matches the release, but historical
per-instance assignments cannot be compared because the old raw artifacts are
unavailable. This reconstruction is not a new benchmark and authorizes no tuning.

The generated [original recovery-status page](failure-taxonomy.md) describes what
the original frozen aggregate bundle alone can establish. This page records the
subsequent reconstruction. The complete
[770-row CSV and diagnostic report](https://github.com/Yanagisawa2002/PoseLoop/tree/4a96a7c102510abc43eee4b35e40f02cd932bee2/reports/v12_repro_5090/a9-v11-replay/failure-taxonomy)
remain in immutable commit history; they are omitted from the compact main diff.
