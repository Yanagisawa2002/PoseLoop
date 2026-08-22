# A-R9 real instance detector status

A-R9 replaces the failed A-R6 / geometry / SAM2 / CAD proposal chain with one
class-agnostic Mask R-CNN v2 trained from raster instance masks.

The split is fixed by complete object scene: eight scenes train, two different
scenes provide internal validation, and the five A-R8 scenes remain the only
25-frame final development evaluation. No object ID occurs in more than one
split. Evaluation labels are not opened by dataset freezing, training, or
inference.

This is already-consumed XYZ-IBD Realsense development data. It is not sealed,
does not authorize scene 9, and cannot authorize a new synthetic or sealed run.
The only promotion condition is a stable gain over the hash-frozen A-R8 raw
FastSAM result. A failure moves the work to 3D instance proposals without
changing this protocol.

The official PyTorch weight is frozen at 185,828,065 bytes and SHA-256
`73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`.
The incorrect torchvision 0.23 URL spelling (`73ccbd019`) is not accepted.

The frozen run completed on 2026-08-21 with status
`REAL_DEVELOPMENT_DETECTOR_POSITIVE_CONTINUE`. See
`DEVELOPMENT_RESULT.md` for metrics, qualitative findings, identities, and the
remaining boundary-quality limitation. This positive development result does
not change the sealed or scene-9 restrictions above.
