# M5-R5 LM-O rotation adapter record

Status: **SOURCE ADAPTER REQUIRED BEFORE DEVELOPMENT SELECTION**

The M5-R5 protocol, method, candidates, evaluator, and initial builder were frozen in commit `0337285` before LM-O pose labels were opened. On the first label-open attempt, the builder stopped inside the shared BOP pose validator. It produced no selected bundle, inference, or performance result.

An exhaustive source-only scan of all 9,209 archived `cam_R_m2c` entries found:

| Raw archive property | Observed value |
|---|---:|
| Maximum infinity-norm orthogonality error | 0.009386876139 |
| Determinant range | 0.999989814436 to 1.013688776908 |
| Singular-value range | 0.999987465082 to 1.004793475723 |
| Non-positive determinants | 0 |
| Maximum closest-SO(3) elementwise correction | 0.004674010768 |
| Maximum closest-SO(3) Frobenius correction | 0.007879821712 |

These values show bounded, orientation-preserving scale/shear in a field that is semantically a rotation matrix. The LM-O-specific adapter therefore projects every source rotation to the closest proper rotation using SVD, while enforcing fixed source-integrity bounds:

- raw orthogonality infinity norm at most 0.02;
- every singular value between 0.99 and 1.01;
- maximum elementwise projection correction at most 0.01;
- Frobenius projection correction at most 0.02;
- positive raw determinant and a strict SO(3) output check.

Translation is copied unchanged except for the existing millimetre-to-metre conversion. Adapter magnitude is recorded but never used for window selection. The shared BOP parser is unchanged. M5-R5 candidate definitions, cross-fitting, metrics, and development gates remain exactly those frozen in `0337285`.

## Visible-mask metadata cross-check

After the rotation adapter was committed, the next source-build attempt stopped before window selection on a two-pixel difference between a decoded `mask_visib` PNG and its archived `px_count_visib`. A full scan found 9,131 exact matches, 76 one-pixel differences, and two two-pixel differences among 9,209 entries. The signed counts were `-2: 1`, `-1: 35`, `0: 9131`, `1: 41`, and `2: 1`.

The LM-O-specific source-integrity tolerance is therefore two pixels. The decoded PNG count, not archived metadata, is used by the already-frozen availability rule, so this adapter does not affect mask area, depth support, missing-frame labels, or window ranking. Any difference above two pixels remains a hard failure.
