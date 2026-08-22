# PoseLoop v1.1.0

This release closes PoseLoop around one supported path: class-agnostic instance
segmentation followed by FoundationPose registration and symmetry-aware
development evaluation.

## Included

- Frozen Mask R-CNN detector with 0.726 instance F1 and 0.702 AP50.
- Complete 820-prediction FoundationPose run with 0.606 joint pose F1 and 0.638
  combined MSSD/MSPD AR.
- One fail-fast inference-to-evaluation runner.
- Compact machine-readable result bundle and validator.
- A 73-second visual walkthrough including a strong and a failure scene.
- Explicit archive of the rejected geometry, SAM2, CAD, missing-frame recovery,
  and confidence-modeling routes.

## Boundary

All metrics are from already-consumed XYZ-IBD RealSense development data. They
are custom frozen metrics, not sealed or official BOP leaderboard results. The
release makes no state-of-the-art, production, or real-time claim.
