# Demo media notice

`poseloop-demo.mp4` and `poseloop-demo-poster.jpg` visualize frozen PoseLoop
outputs over RGB frames from the XYZ Industrial Bin-picking Dataset (XYZ-IBD),
revision `4fe4671783172622313ac0c7182012cee618f217`.

The two raw frames are exact members of `xyzibd_val.zip`:

- `xyzibd_val/val/000010/rgb_realsense/000000.png`
- `xyzibd_val/val/000025/rgb_realsense/000000.png`

The masks and CAD projections are PoseLoop-generated annotations. The GT CAD
projection is a post-inference development diagnostic and is visibly labeled as
such. No raw dataset archive, model checkpoint, or hidden evaluation label is
distributed here.

These media adaptations are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) to match
the dataset license. Attribution: XYZ-IBD contributors and BOP Benchmark,
[`bop-benchmark/xyzibd@4fe4671`](https://huggingface.co/datasets/bop-benchmark/xyzibd/tree/4fe4671783172622313ac0c7182012cee618f217).
