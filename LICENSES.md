# PoseLoop data, weights, and third-party notices

This document records the license boundaries of the external material used to
produce PoseLoop v1.0.0. It is informational and is not legal advice.

## PoseLoop original material

This repository currently has no top-level `LICENSE`. No copyright license for
PoseLoop's original scripts, documentation, aggregate result bundle, or figures
is granted by this notice. A project license must be selected separately by the
copyright holder.

The absence of a PoseLoop license does not replace or weaken any third-party
terms below. Likewise, an upstream license does not automatically license
PoseLoop's original material.

## External components

| Component | Exact source used by PoseLoop | License and redistribution boundary |
|---|---|---|
| XYZ Industrial Bin-picking Dataset (XYZ-IBD) | [`bop-benchmark/xyzibd@4fe4671783172622313ac0c7182012cee618f217`](https://huggingface.co/datasets/bop-benchmark/xyzibd/tree/4fe4671783172622313ac0c7182012cee618f217) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Use is non-commercial; sharing requires attribution, a license link, indication of changes, and share-alike treatment of adaptations. The dataset card says commercial users should request separate permission. |
| FoundationPose source | [`NVlabs/FoundationPose@a1b694b83e633c2cb6115b9063d940a687759392`](https://github.com/NVlabs/FoundationPose/tree/a1b694b83e633c2cb6115b9063d940a687759392) | [NVIDIA Source Code License](https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/LICENSE). Use and derivative works are limited to non-commercial research or evaluation. Redistribution must retain the license and notices and satisfy its terms. |
| FoundationPose refiner/scorer checkpoints | Experiments and the downloader use [`nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL@39a143ab8ff830a593f92b7ddf5d66c99e59bb44`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/39a143ab8ff830a593f92b7ddf5d66c99e59bb44/checkpoint/FoundationPose). The same model bytes were also verified at [`943946a972d5de2eb0d2ff214b236d0e43575fd7`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/943946a972d5de2eb0d2ff214b236d0e43575fd7/checkpoint/FoundationPose). | The later official GRAIL card explicitly assigns `checkpoint/FoundationPose/` to the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), rather than GRAIL's top-level Apache-2.0 dataset license. That component-specific statement is not present in the historical snapshot's card, so users relying on the historical distribution should confirm the applicable terms with NVIDIA. Review the agreement, including notice and acceptable-use requirements, before use or redistribution. |
| BOP Toolkit | [`thodan/bop_toolkit@cea62d651c7e395b2e1962b9749e4e89693c6ac4`](https://github.com/thodan/bop_toolkit/tree/cea62d651c7e395b2e1962b9749e4e89693c6ac4) | [MIT](https://github.com/thodan/bop_toolkit/blob/cea62d651c7e395b2e1962b9749e4e89693c6ac4/LICENSE). Redistribution of the toolkit or substantial portions must retain its copyright and permission notice. |
| `torch-ngp` grid encoder bundled by FoundationPose | Present inside the ignored FoundationPose checkout | MIT. Its license is retained at `third_party/FoundationPose/bundlesdf/mycuda/torch_ngp_grid_encoder/LICENSE` when the upstream checkout is present. |
| `nvdiffrast` | Installed separately by the FoundationPose environment | NVIDIA Source Code License 1-Way Commercial. Non-NVIDIA users are limited to non-commercial research/evaluation under the upstream terms; see the [upstream license](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt). |
| PyTorch3D | Installed separately by the FoundationPose environment | BSD 3-Clause; see the [upstream license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE). |
| Other Conda and pip dependencies | Installed separately; not vendored by PoseLoop | Each dependency remains under its own upstream license. PoseLoop v1.0.0 does not claim that this notice is a complete environment SBOM. |

BOP Toolkit also contains file-level notices, including Apache-2.0 terms for
the bundled Droid Sans Mono font and a BSD-style notice in
`bop_toolkit_lib/transform.py`. Those notices remain in the ignored pinned
checkout and must be retained if BOP files are redistributed.

## XYZ-IBD attribution

PoseLoop uses the XYZ Industrial Bin-picking Dataset (XYZ-IBD) by Junwen Huang,
Jiaqi Hu, Peter K. T. Yu, Martin Sundermeyer, Slobodan Ilic, and Benjamin
Busam, licensed under CC BY-NC-SA 4.0.

PoseLoop downloads the base, object-model, and validation archives and computes
derived evaluation metrics and figures. No raw XYZ-IBD images, depth maps,
masks, annotations, or meshes are redistributed, and no endorsement by the
dataset authors is implied.

The v1.1.0 demo video and poster are visibly annotated adaptations of two
validation RGB frames and the frozen PoseLoop output overlays. They are
distributed separately under CC BY-NC-SA 4.0 with attribution and an indication
of changes; see [`docs/media/README.md`](docs/media/README.md). They are not
licensed as PoseLoop original source code.

Project and license references:

- [XYZ-IBD project page](https://xyz-ibd.github.io/)
- [XYZ-IBD Hugging Face dataset card](https://huggingface.co/datasets/bop-benchmark/xyzibd)
- [BOP dataset catalogue](https://bop.felk.cvut.cz/datasets/)

## Weight identity

[`scripts/fetch_foundationpose_weights.py`](scripts/fetch_foundationpose_weights.py)
uses the historical official NVIDIA GRAIL experiment snapshot
`39a143ab8ff830a593f92b7ddf5d66c99e59bb44` and verifies both checkpoint
files. Their sizes and SHA-256 values also match the later official
license-labeled snapshot `943946a972d5de2eb0d2ff214b236d0e43575fd7`:

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| Refiner `2023-10-28-18-33-37/model_best.pth` | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| Scorer `2024-01-11-20-02-45/model_best.pth` | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |

The checkpoint files are not part of this repository or tag. The download
script verifies identity; it does not grant a license. The smoke runner retains
a legacy mirror identifier solely because that exact allowlist is hashed into
the frozen M1–M4 inference provenance. The release downloader exposes only the
official NVIDIA source, and the released results record that official source.

## What is and is not distributed

The root [`.gitignore`](.gitignore) excludes:

- `data/`, `archives/` — dataset files and download archives;
- `weights/` — model checkpoints;
- `third_party/` — FoundationPose, BOP Toolkit, and their bundled notices;
- `artifacts/` — raw manifests, predictions, metrics, fitted models, and logs.

The tracked `release/v1.1.0/results.json` and legacy
`precomputed/v1.0.0/results.json` files contain only compact aggregate numbers
and provenance references. Tracked reports and figures are derived research
outputs. Users remain responsible for determining whether upstream attribution,
non-commercial, or share-alike terms apply to their intended reuse.
