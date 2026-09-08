# Source navigation and dependency audit

## Supported public release

Start at [`run_release_pipeline.sh`](../scripts/run_release_pipeline.sh).
It uses the following source areas:

- [`real_instance_detector_v1`](../pose_accuracy_recovery_prep/real_instance_detector_v1/): detector preparation, training, inference and evaluation.
- [`a9_foundationpose_e2e`](../pose_accuracy_recovery_prep/a9_foundationpose_e2e/): frozen mask-to-pose handoff, execution and evidence packaging.
- [`foundationpose_runtime_prep`](../foundationpose_runtime_prep/): upstream runtime adaptation.
- [`protocols`](../protocols/), [`release/v1.1.0`](../release/v1.1.0/) and [`verify_release.py`](../scripts/verify_release.py): contracts and release checks.

## Historical exploration

The following packages are retained for historical imports and replay. They are
not alternative public release entry points. See [negative results](archived-negative-results.md).

- [`r3_bop_industrial`](../r3_bop_industrial/)
- [`r3_bop_industrial_supported`](../r3_bop_industrial_supported/)
- [`r3_bop_industrial_supported_v2`](../r3_bop_industrial_supported_v2/)
- [`r3_bop_industrial_supported_v3`](../r3_bop_industrial_supported_v3/)
- [`r3_bop_industrial_supported_v3_readiness`](../r3_bop_industrial_supported_v3_readiness/)
- [`r4a_development_readiness`](../r4a_development_readiness/)
- [`r4a_development_readiness_v2`](../r4a_development_readiness_v2/)
- [`r4a_development_repair`](../r4a_development_repair/)
- [`r4a_development_repair_v2`](../r4a_development_repair_v2/)
- [`r4a_development_repair_v3`](../r4a_development_repair_v3/)
- [`r4a_development_repair_v4`](../r4a_development_repair_v4/)
- [`r4a_development_slice`](../r4a_development_slice/)
- [`r4a_development_slice_v2`](../r4a_development_slice_v2/)
- [`r4a_development_v3`](../r4a_development_v3/)
- [`r4a_development_v3_final`](../r4a_development_v3_final/)

## Why source packages were not relocated

A static scan on 2026-09-08 found 79 top-level import statements
referencing these package families in 48 Python files.
Moving them into an archive folder would change Python module identities and
require a broader migration of historical tests, commands and protocol references.
This presentation cleanup therefore preserves source paths and frozen evidence.
The inventory below records direct textual imports, not a complete runtime call graph.

| Source | Import |
| --- | --- |
| [`r3_bop_industrial_supported/core.py:20`](../r3_bop_industrial_supported/core.py#L20) | `from r3_bop_industrial import core as source_core` |
| [`r3_bop_industrial_supported_v2/core.py:20`](../r3_bop_industrial_supported_v2/core.py#L20) | `from r3_bop_industrial import core as source_core` |
| [`r3_bop_industrial_supported_v2/core.py:21`](../r3_bop_industrial_supported_v2/core.py#L21) | `from r3_bop_industrial_supported import core as v1_core` |
| [`r3_bop_industrial_supported_v3/core.py:12`](../r3_bop_industrial_supported_v3/core.py#L12) | `from r3_bop_industrial_supported_v2.core import _SMOKE_PROGRAM` |
| [`r3_bop_industrial_supported_v3/core.py:13`](../r3_bop_industrial_supported_v3/core.py#L13) | `from r3_bop_industrial_supported_v3_readiness.core import (` |
| [`r3_bop_industrial_supported_v3/core.py:24`](../r3_bop_industrial_supported_v3/core.py#L24) | `from r3_bop_industrial.core import utc_now, write_text_atomic` |
| [`r4a_development_readiness/core.py:12`](../r4a_development_readiness/core.py#L12) | `from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_readiness_v2/core.py:13`](../r4a_development_readiness_v2/core.py#L13) | `from r4a_development_repair.core import (` |
| [`r4a_development_repair_v2/cli.py:11`](../r4a_development_repair_v2/cli.py#L11) | `from r4a_development_repair.splitzip import Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v2/core.py:8`](../r4a_development_repair_v2/core.py#L8) | `from r4a_development_repair.core import (` |
| [`r4a_development_repair_v2/splitzip.py:9`](../r4a_development_repair_v2/splitzip.py#L9) | `from r4a_development_repair.core import ContractError` |
| [`r4a_development_repair_v2/splitzip.py:10`](../r4a_development_repair_v2/splitzip.py#L10) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v3/cli.py:11`](../r4a_development_repair_v3/cli.py#L11) | `from r4a_development_repair.splitzip import Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v3/core.py:8`](../r4a_development_repair_v3/core.py#L8) | `from r4a_development_repair.core import (` |
| [`r4a_development_repair_v3/splitzip.py:14`](../r4a_development_repair_v3/splitzip.py#L14) | `from r4a_development_repair.core import ContractError` |
| [`r4a_development_repair_v3/splitzip.py:15`](../r4a_development_repair_v3/splitzip.py#L15) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v4/cli.py:11`](../r4a_development_repair_v4/cli.py#L11) | `from r4a_development_repair.splitzip import Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v4/core.py:8`](../r4a_development_repair_v4/core.py#L8) | `from r4a_development_repair.core import ContractError, canonical_sha256, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_repair_v4/splitzip.py:8`](../r4a_development_repair_v4/splitzip.py#L8) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip, write_catalog` |
| [`r4a_development_repair_v4/splitzip.py:9`](../r4a_development_repair_v4/splitzip.py#L9) | `from r4a_development_repair_v3.splitzip import CachedRetryRangeClient` |
| [`r4a_development_slice/core.py:8`](../r4a_development_slice/core.py#L8) | `from r4a_development_repair.core import (` |
| [`r4a_development_slice/extractor.py:9`](../r4a_development_slice/extractor.py#L9) | `from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_slice/extractor.py:10`](../r4a_development_slice/extractor.py#L10) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip` |
| [`r4a_development_slice/extractor.py:11`](../r4a_development_slice/extractor.py#L11) | `from r4a_development_repair_v3.splitzip import CachedRetryRangeClient` |
| [`r4a_development_slice/planner.py:9`](../r4a_development_slice/planner.py#L9) | `from r4a_development_repair.core import ContractError, read_json, write_json_atomic` |
| [`r4a_development_slice_v2/core.py:8`](../r4a_development_slice_v2/core.py#L8) | `from r4a_development_repair.core import ContractError, canonical_sha256, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_slice_v2/extractor.py:7`](../r4a_development_slice_v2/extractor.py#L7) | `from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_slice_v2/extractor.py:8`](../r4a_development_slice_v2/extractor.py#L8) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip` |
| [`r4a_development_slice_v2/extractor.py:9`](../r4a_development_slice_v2/extractor.py#L9) | `from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient` |
| [`r4a_development_slice_v2/extractor.py:10`](../r4a_development_slice_v2/extractor.py#L10) | `from r4a_development_slice.extractor import _crc32_file, _safe_relative` |
| [`r4a_development_slice_v2/planner.py:9`](../r4a_development_slice_v2/planner.py#L9) | `from r4a_development_repair.core import ContractError, read_json, write_json_atomic` |
| [`r4a_development_v3/assets.py:10`](../r4a_development_v3/assets.py#L10) | `from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_v3/assets.py:11`](../r4a_development_v3/assets.py#L11) | `from r4a_development_repair.development import camera_world_to_camera_pose_m` |
| [`r4a_development_v3/assets.py:12`](../r4a_development_v3/assets.py#L12) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip` |
| [`r4a_development_v3/assets.py:13`](../r4a_development_v3/assets.py#L13) | `from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient` |
| [`r4a_development_v3/bundle.py:13`](../r4a_development_v3/bundle.py#L13) | `from r4a_development_repair.core import ContractError, canonical_sha256, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_v3/contract.py:10`](../r4a_development_v3/contract.py#L10) | `from r4a_development_repair.core import (` |
| [`r4a_development_v3/id_job.py:9`](../r4a_development_v3/id_job.py#L9) | `from r4a_development_repair.core import ContractError, read_json, sha256_file, write_json_atomic` |
| [`r4a_development_v3/id_job.py:10`](../r4a_development_v3/id_job.py#L10) | `from r4a_development_repair.splitzip import Entry, Part, SplitZip` |
| [`r4a_development_v3/id_job.py:11`](../r4a_development_v3/id_job.py#L11) | `from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient` |
| [`r4a_development_v3/id_parser.py:12`](../r4a_development_v3/id_parser.py#L12) | `from r4a_development_repair.core import ContractError` |
| [`r4a_development_v3_final/core.py:21`](../r4a_development_v3_final/core.py#L21) | `from r4a_development_repair.core import (` |
| [`tests/r3_bop_industrial/test_contract.py:8`](../tests/r3_bop_industrial/test_contract.py#L8) | `from r3_bop_industrial.core import (` |
| [`tests/r3_bop_industrial/test_gpu_a_post_download.py:11`](../tests/r3_bop_industrial/test_gpu_a_post_download.py#L11) | `from r3_bop_industrial.gpu_a_post_download import (` |
| [`tests/r3_bop_industrial_supported/test_contract.py:9`](../tests/r3_bop_industrial_supported/test_contract.py#L9) | `from r3_bop_industrial import core as source_core` |
| [`tests/r3_bop_industrial_supported/test_contract.py:10`](../tests/r3_bop_industrial_supported/test_contract.py#L10) | `from r3_bop_industrial_supported.core import (` |
| [`tests/r3_bop_industrial_supported_v2/test_contract.py:10`](../tests/r3_bop_industrial_supported_v2/test_contract.py#L10) | `from r3_bop_industrial import core as source_core` |
| [`tests/r3_bop_industrial_supported_v2/test_contract.py:11`](../tests/r3_bop_industrial_supported_v2/test_contract.py#L11) | `from r3_bop_industrial_supported import core as v1_core` |
| [`tests/r3_bop_industrial_supported_v2/test_contract.py:12`](../tests/r3_bop_industrial_supported_v2/test_contract.py#L12) | `from r3_bop_industrial_supported_v2.core import (` |
| [`tests/r3_bop_industrial_supported_v3/test_contract.py:9`](../tests/r3_bop_industrial_supported_v3/test_contract.py#L9) | `from r3_bop_industrial_supported_v3 import core as v3_core` |
| [`tests/r3_bop_industrial_supported_v3/test_contract.py:10`](../tests/r3_bop_industrial_supported_v3/test_contract.py#L10) | `from r3_bop_industrial_supported_v3.core import (` |
| [`tests/r3_bop_industrial_supported_v3/test_contract.py:17`](../tests/r3_bop_industrial_supported_v3/test_contract.py#L17) | `from r3_bop_industrial_supported_v3_readiness.core import sha256_file, write_json_atomic` |
| [`tests/r3_bop_industrial_supported_v3_readiness/test_contract.py:9`](../tests/r3_bop_industrial_supported_v3_readiness/test_contract.py#L9) | `from r3_bop_industrial_supported_v3_readiness.cli import parse_args` |
| [`tests/r3_bop_industrial_supported_v3_readiness/test_contract.py:10`](../tests/r3_bop_industrial_supported_v3_readiness/test_contract.py#L10) | `from r3_bop_industrial_supported_v3_readiness.core import (` |
| [`tests/r4a_development_readiness/test_readiness_protocol.py:5`](../tests/r4a_development_readiness/test_readiness_protocol.py#L5) | `from r4a_development_readiness.core import load_protocol` |
| [`tests/r4a_development_readiness_v2/test_readiness_v2.py:5`](../tests/r4a_development_readiness_v2/test_readiness_v2.py#L5) | `from r4a_development_readiness_v2.core import _finite_matrix, _finite_vector, _loaded_pose_valid, load_protocol` |
| [`tests/r4a_development_repair/test_contract.py:8`](../tests/r4a_development_repair/test_contract.py#L8) | `from r4a_development_repair.core import ContractError, load_protocol` |
| [`tests/r4a_development_repair/test_contract.py:9`](../tests/r4a_development_repair/test_contract.py#L9) | `from r4a_development_repair.development import (` |
| [`tests/r4a_development_repair/test_splitzip.py:9`](../tests/r4a_development_repair/test_splitzip.py#L9) | `from r4a_development_repair.splitzip import Entry, Part, RangeClient, SplitZip` |
| [`tests/r4a_development_repair_v2/test_chunked_range.py:3`](../tests/r4a_development_repair_v2/test_chunked_range.py#L3) | `from r4a_development_repair_v2.splitzip import ChunkedRangeClient` |
| [`tests/r4a_development_repair_v2/test_v2_contract.py:5`](../tests/r4a_development_repair_v2/test_v2_contract.py#L5) | `from r4a_development_repair.core import sha256_file` |
| [`tests/r4a_development_repair_v2/test_v2_contract.py:6`](../tests/r4a_development_repair_v2/test_v2_contract.py#L6) | `from r4a_development_repair_v2.core import load_protocol` |
| [`tests/r4a_development_repair_v3/test_cached_retry.py:6`](../tests/r4a_development_repair_v3/test_cached_retry.py#L6) | `from r4a_development_repair_v3.splitzip import CachedRetryRangeClient` |
| [`tests/r4a_development_repair_v3/test_v3_contract.py:5`](../tests/r4a_development_repair_v3/test_v3_contract.py#L5) | `from r4a_development_repair_v3.core import load_protocol` |
| [`tests/r4a_development_repair_v4/test_short_body_retry.py:6`](../tests/r4a_development_repair_v4/test_short_body_retry.py#L6) | `from r4a_development_repair_v4.splitzip import ShortBodyRetryRangeClient` |
| [`tests/r4a_development_repair_v4/test_v4_contract.py:5`](../tests/r4a_development_repair_v4/test_v4_contract.py#L5) | `from r4a_development_repair_v4.core import load_protocol` |
| [`tests/r4a_development_slice/test_planner.py:6`](../tests/r4a_development_slice/test_planner.py#L6) | `from r4a_development_slice.planner import build_plan` |
| [`tests/r4a_development_slice_v2/test_gray_plan.py:6`](../tests/r4a_development_slice_v2/test_gray_plan.py#L6) | `from r4a_development_slice_v2.planner import build_plan` |
| [`tests/r4a_development_slice_v2/test_protocol.py:5`](../tests/r4a_development_slice_v2/test_protocol.py#L5) | `from r4a_development_slice_v2.core import load_protocol` |
| [`tests/r4a_development_v3/test_mask_and_archive.py:9`](../tests/r4a_development_v3/test_mask_and_archive.py#L9) | `from r4a_development_repair.core import read_json, sha256_file, write_json_atomic` |
| [`tests/r4a_development_v3/test_mask_and_archive.py:10`](../tests/r4a_development_v3/test_mask_and_archive.py#L10) | `from r4a_development_v3.assets import bbox_from_mask, coco_rle, depth_component_mask` |
| [`tests/r4a_development_v3/test_mask_and_archive.py:11`](../tests/r4a_development_v3/test_mask_and_archive.py#L11) | `from r4a_development_v3.bundle import _deterministic_targz, build_inference_bundle` |
| [`tests/r4a_development_v3/test_protocol_and_ids.py:8`](../tests/r4a_development_v3/test_protocol_and_ids.py#L8) | `from r4a_development_repair.core import ContractError` |
| [`tests/r4a_development_v3/test_protocol_and_ids.py:9`](../tests/r4a_development_v3/test_protocol_and_ids.py#L9) | `from r4a_development_v3.contract import load_protocol` |
| [`tests/r4a_development_v3/test_protocol_and_ids.py:10`](../tests/r4a_development_v3/test_protocol_and_ids.py#L10) | `from r4a_development_v3.id_parser import IdRecord, parse_scene_gt_ids, select_multi_object_targets` |
| [`tests/r4a_development_v3_final/test_contract.py:9`](../tests/r4a_development_v3_final/test_contract.py#L9) | `from r4a_development_repair.core import ContractError, sha256_file` |
| [`tests/r4a_development_v3_final/test_contract.py:10`](../tests/r4a_development_v3_final/test_contract.py#L10) | `from r4a_development_v3_final import (` |
| [`tests/r4a_development_v3_final/test_contract.py:16`](../tests/r4a_development_v3_final/test_contract.py#L16) | `from r4a_development_v3_final.cli import parse_args` |
| [`tests/r4a_development_v3_final/test_contract.py:17`](../tests/r4a_development_v3_final/test_contract.py#L17) | `from r4a_development_v3_final.core import (` |
