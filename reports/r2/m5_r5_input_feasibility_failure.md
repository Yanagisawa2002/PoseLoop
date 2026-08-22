# PoseLoop M5-R5 LM-O input feasibility result

Status: **FAIL_M5_R5_INPUT_FEASIBILITY**

M5-R5 stopped before bundle creation, FoundationPose inference, candidate evaluation, or any pose-error calculation. The frozen protocol required eight represented objects and required every 192-frame object window to contain at least two naturally unavailable frames. Only seven of the eight LM-O objects satisfy both conditions.

| Object | Eligible window start | Available | Natural missing | Result |
|---:|---:|---:|---:|---|
| 1 | 259 | 183 | 9 | eligible |
| 5 | 192 | 189 | 3 | eligible |
| 6 | 446 | 150 | 42 | eligible |
| 8 | 165 (best rejected window) | 191 | 1 | ineligible: below the frozen minimum of 2 |
| 9 | 987 | 168 | 24 | eligible |
| 10 | 436 | 181 | 11 | eligible |
| 11 | 196 | 171 | 21 | eligible |
| 12 | 800 | 190 | 2 | eligible |

The seven eligible windows would contain 1,344 replay frames and 112 natural missing frames. Their missing-frame strata are 42 short-gap frames, 61 medium-gap frames, and 9 long-gap frames, so the aggregate dropout quantity and strata gates are feasible. The sole failure is that object 8 has only one unavailable frame among all 1,214 GT-present source frames, and therefore no 192-frame window can reach the pre-frozen per-object minimum of two.

This is an outcome-blind source-structure failure. It does not justify changing M5-R5 in place. A successor protocol may reduce the per-object minimum to one while continuing to require all eight objects, the unchanged aggregate missing-frame count, and the unchanged gap strata before any inference result is produced.

Evidence boundary:

- initial method/protocol pre-freeze: `0337285`;
- bounded LM-O rotation adapter: `bf43408`;
- bounded visible-mask metadata audit: `bf6f185`;
- M5-R5 protocol SHA-256: `637262710e6377416050fab63ab38f6684201e3b3b9fa6748be633d00b347c64`;
- builder SHA-256 at the feasibility run: `bdf07a5ffe797761f1ab56b7953829fe787f247af0b81590d5aa6467069fef92`;
- `artifacts/r2/m5_r5` was not created;
- YCB-V archive and extracted-data roots remained absent.

This failure is not an M5 effectiveness result.
