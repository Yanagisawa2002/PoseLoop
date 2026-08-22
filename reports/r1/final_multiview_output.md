# PoseLoop R1 final multi-view development output

The frozen development output composes the nested-OOF M3-R1 continue decision with the nested-OOF M4-R1 CAD ranker. M3 stops retain the target pose; M3 continues acquire one M4-ranked candidate and use the frozen max-mask post-acquisition rule.

| Metric | Value |
| --- | ---: |
| Targets | 300 |
| Mean acquired views | 1.480 |
| Final macro combined | 72.67% |
| M3-R1 primary macro combined | 72.33% |
| Gain vs M3-R1 primary | +0.33 pp |
| Gain vs fixed-slot composition | +1.35 pp |
| Gain vs random composition | +2.18 pp |

This stream is the only pose-output contract permitted as M6-R1 input. M6-R1 may predict risk or abstain, but may not reselect a pose.
