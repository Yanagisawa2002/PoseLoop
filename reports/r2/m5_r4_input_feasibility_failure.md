# PoseLoop M5-R4 input-feasibility failure

Status: **FAIL_INPUT_FEASIBILITY**

The original M5-R4 protocol was frozen in commit `9e7cead` before HB development pose labels were opened. A one-pixel HB archive metadata tolerance was added in commit `e66898b` after a loader integrity check stopped before selection; across 12,240 development masks, 12,030 matched metadata exactly and 210 differed by only plus or minus one pixel. The actual stored masks remained the source of input support.

The second frozen builder run selected only 6 tracks under the rule requiring at least 8 natural missing frames per 128-frame window. This failed the predeclared minimum of 8 tracks, so no development bundle was written.

No FoundationPose inference ran. No candidate was selected. No pose error, relative improvement, confidence correlation, bootstrap result, or performance outcome was computed. The Kinect 2 archive was not downloaded or read.

Input-only feasibility statistics showed 11 tracks with at least 2 natural missing frames, spanning 9 objects and 129 total natural missing frames. Of those, 5 tracks had 2-7 missing frames and 6 tracks had at least 8. This audit motivates the separately versioned M5-R4A sparse/stress dropout strata; it does not change or erase the M5-R4 failure.
