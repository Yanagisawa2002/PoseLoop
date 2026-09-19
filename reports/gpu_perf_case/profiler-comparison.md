# Profiler comparison: not entered after evaluation rejection

The original Systems baseline used CLI 2026.5.1.161, 60 s delay and a 60.000211702 s window over approximately 81 registration equivalents (80 complete plus edge fragments). It recorded 20.573 s convolution/GEMM/attention time, 13.959 s for the dominant convolution, and approximately 1.605 million kernels. Source-supported render-marker inference attributed 26.268 s kernel time, including 16.252 s convolution/GEMM/attention, to refinement across 80 complete registrations. GPU activity union was 36.515 s; its complement was 23.485 s. These are recorded timeline values, not achieved-occupancy counters.

The original trace has capture-stop incompleteness warnings; 602 unmatched launch/kernel pairs were confined to its final second, with none in the interior 1–59 s. It samples scene 10/object 2, not every scene. Full baseline details remain in the [previous profiling package](https://github.com/Yanagisawa2002/PoseLoop/tree/4a96a7c102510abc43eee4b35e40f02cd932bee2/reports/gpu_profile_5090); only a compact stage summary was copied here.

| Requested comparison | Baseline | Refine64 / delta |
| --- | ---: | --- |
| Refine-range kernel time | 26.268 s / 80 complete registrations | unavailable |
| Conv/GEMM/attention | 20.573 s / captured window | unavailable |
| Dominant convolution | 13.959 s / captured window | unavailable |
| Kernel count | 1,605,437 | unavailable |
| Median kernel duration | 2.752 us | unavailable |
| StreamSynchronize calls / time | 102,017 / 18.634760 s | unavailable |
| Malloc / Free CPU time | 1.166143 / 4.207852 s | unavailable |
| GPU active / inactive recorded time | 36.515 / 23.485 s | unavailable |

**Mechanism supported by optimized profiler evidence: NO — not tested.** This is not evidence that the hypothesized mechanism is false. B1 lost three joint successes and changed four scene results; the required accuracy gate failed. Its single 0.7607% wall-time difference also does not establish reproducibility. The requested optimized-profile preconditions were therefore not met. No optimized `.nsys-rep` exists and no new trace SHA is claimed. No Nsight Compute pass was run.

Baseline trace path: `/root/autodl-tmp/poseloop-a9-perf/capture/baseline.nsys-rep`.
Baseline trace SHA-256: `20a72e4e34740a15a41af95fcaf4a52f0af88869ab1e70f5b99b9a73ac0deb2a`.

The branch changes the refiner chunk-forward count structurally (8→4 chunks per iteration, 40→20 per registration). Do not equate this with a measured halving of GPU launches, kernel time, or total latency. The experiment stops without implementing another mechanism.
