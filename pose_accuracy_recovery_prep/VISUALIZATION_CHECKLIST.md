# Visualization checklist

## Label-free producer layers

- RGB and the exact selected predicted/depth-component/bbox mask.
- Initial pose CAD plus axes.
- Top-k CAD plus axes with scorer rank, without GT-derived color or error.
- Final pose CAD plus axes.
- Baseline-left / improved-right frame identity and timestamp.

The producer bundle and video must contain no GT pose, GT overlay, evaluator score, threshold, oracle mask, or sealed path.

## Evaluator-only layers

- The same RGB, selected mask, initial pose, top-k, and final pose layers.
- A separately named GT CAD/axes overlay.
- ADD(-S), rotation, and translation values generated only after producer outputs are frozen.
- Oracle and official-known-sample frames marked `DIAGNOSTIC_ONLY`.

GT overlays never enter the producer archive. The `visualization-plan` CLI writes the exact layer inventory and a future baseline/improved video entrypoint; PREP does not render images or videos.
