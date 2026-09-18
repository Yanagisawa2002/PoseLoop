# PoseLoop v1.1.0 per-instance failure taxonomy

Status: **COMPLETE_DIAGNOSTIC_RECONSTRUCTION**

This is a post-hoc diagnostic of the already-consumed v1.1.0 development split.
It is not a new untouched evaluation and must not be used to retune v1.1.0 and
then presented as if the split were unseen.

## Reconciliation

- GT instances: **770**
- IoU50 mask matches: **577**
- joint pose successes: **482**
- primary prediction SHA-256: 300deaca726983a5e15513e6ed49814527a752581c11bb957d0bde8135927b77
- input manifest SHA-256: e2d3bba5b743b214d2491ae6b6f3619d2849d861655a13b474fa170f4ed30a66

## Primary buckets

| Bucket | Count | Share of GT |
| --- | ---: | ---: |
| DETECTOR_MISS | 1 | 0.1% |
| OVER_SEGMENTATION | 24 | 3.1% |
| UNDER_SEGMENTATION_OR_MERGE | 54 | 7.0% |
| MASK_BOUNDARY_IOU_FAILURE | 114 | 14.8% |
| POSE_MSSD_ONLY_FAILURE | 20 | 2.6% |
| POSE_MSPD_ONLY_FAILURE | 2 | 0.3% |
| POSE_MSSD_AND_MSPD_FAILURE | 73 | 9.5% |
| SUCCESS | 482 | 62.6% |

## By scene

| Scene | GT | Success | Failure |
| ---: | ---: | ---: | ---: |
| 10 | 185 | 148 | 37 |
| 25 | 295 | 129 | 166 |
| 30 | 105 | 83 | 22 |
| 40 | 135 | 83 | 52 |
| 65 | 50 | 39 | 11 |

## Diagnostic IoU bands

For unmatched GT, the band uses maximum overlap with any frozen detector
prediction. For matched GT, it uses the assigned greedy-match IoU.

| IoU band | GT rows |
| --- | ---: |
| <0.50 | 193 |
| 0.50-0.75 | 380 |
| >=0.75 | 197 |

## Interpretation rule

Merge and split buckets use the same coverage definitions as the frozen detector
diagnostics: a merge prediction covers at least 50% of two or more GT masks; a
split GT contains at least two prediction fragments for which at least 50% of
the prediction lies inside that GT. IoU50 competition is checked before those
geometric buckets. Remaining nonzero sub-IoU50 overlap is conservatively labeled
MASK_BOUNDARY_IOU_FAILURE; zero overlap is DETECTOR_MISS.

Pose buckets use the frozen joint thresholds from the A9 protocol. A failed
primary registration is POSE_INPUT_INVALID_OR_WEAK_DEPTH only when its frozen
failure message explicitly references depth or point-cloud input; otherwise it
is POSE_REGISTRATION_FAILURE.
