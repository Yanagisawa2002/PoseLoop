# PoseLoop M5-R4A HB Primesense development input audit

Status: **DEVELOPMENT_BUNDLE_READY**

This development bundle was selected from new HomebrewedDB Primesense scenes 1-8 using only frozen mask/depth availability rules. The earlier M5-R4 builder opened development labels only for an input-feasibility scan and produced no predictions or pose-error results; M5-R4A was frozen before inference. No Kinect 2 sealed data was downloaded or read by this builder.

| Quantity | Value |
|---|---:|
| Selected tracks | 11 |
| Represented objects | 9 |
| Replay frames | 1408 |
| FoundationPose inference frames | 1279 |
| Natural missing frames | 129 |
| Sparse-dropout tracks | 5 |
| Stress-dropout tracks | 6 |

| Object | Tracks | Replay | Available | Missing |
|---:|---:|---:|---:|---:|
| 3 | 1 | 128 | 123 | 5 |
| 4 | 1 | 128 | 123 | 5 |
| 9 | 1 | 128 | 125 | 3 |
| 10 | 1 | 128 | 118 | 10 |
| 13 | 1 | 128 | 105 | 23 |
| 15 | 2 | 256 | 214 | 42 |
| 16 | 1 | 128 | 111 | 17 |
| 17 | 2 | 256 | 234 | 22 |
| 25 | 1 | 128 | 126 | 2 |

Source archives:

- `hb_base.zip`: `15de59e56148d7fa2bc137d1d52900f99e982f29f0b937a5447570749502055e`
- `hb_models.zip`: `026a9417387e7225463e63fd0f4ac9b1b15a961d85f3095f9dbf416f0b8bd9b8`
- `hb_val_primesense.zip`: `8dd87de6eed1067063081bd4aed3d32a9f0bb6e00887bde78b941eaefc6fec0f`

This audit is not an outcome and cannot justify advancement by itself.
