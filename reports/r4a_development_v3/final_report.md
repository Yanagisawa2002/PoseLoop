# PoseLoop R4-A v3 development result

Status: **NO-GO**. The deterministic multi-object development slice reached 100% prediction coverage and valid SE(3), but both single-view and fixed multi-view official BOP24 development mAP are 0.0. No prediction, checkpoint, association, candidate, or fusion tuning is permitted from this result.

## Frozen scope

- Role: `DEVELOPMENT_ONLY`; dataset: public XYZ-IBD `train_pbr`, never the consumed XYZ-IBD val split.
- Selection: 10 unique `(scene,image,object)` items, 5 scenes, 5 objects, 2 images per object.
- R4-A handoff protocol: SHA-256 `cd2cfb54486af547e942be633daa10d9233b3724be1edb2b439554767ee52bc6`; implementation commit `4eebb040cd047ab85783c06448bf82f23367e534`.
- Frozen no-GT archive: `806b4c3c7d297b533ce79e7db0efbccc409d93d917fbf5dae72b7a3a6935235a` (16,549,946 bytes); `label_access_count=0`.
- GPU-C return archive: `87d32fcdbb4be77fa63f284d11ddb1c59285411385c457cd25abd77094641136` (14,833,112 bytes); internal `SHA256SUMS` 53/53 verified.
- Final evaluator protocol v4: SHA-256 `b15359855924dcd7ed24bd5db42ee44d9a5e4b0d94b88ca06527997067fbea3a`; local implementation commit `38e963187c8182bbe713edba2076c63c30d0221c`; GPU-A implementation commit `17bddf0a0dcce8bb40e414c081600567858a5c6d`.
- Pinned BOP toolkit source commit `cea62d651c7e395b2e1962b9749e4e89693c6ac4`, 93-file tree SHA-256 `9726f6d1f189bd75e665205f62e9afcf021876270aadedcb41af8e8b1e8ff307`.

## GPU-C inference

- 10/10 success, failed/OOM 0/0, c252 only, attempt 1 exit 0, no resume or retry.
- Registration p50 825.323 ms, p95 1555.415 ms, 0.350233 item/s active-worker throughput.
- Peak allocated VRAM 7,240,450,048 bytes on RTX 5090.
- GT/oracle/evaluator/scorer paths opened: 0; official scorer runs on GPU-C: 0.

## Development evaluation

| Metric | Single | Multi | Multi - single |
|---|---:|---:|---:|
| Official BOP24 mAP | 0.000000 | 0.000000 | 0.000000 |
| BOP24 mAP MSSD | 0.000000 | 0.000000 | 0.000000 |
| BOP24 mAP MSPD | 0.000000 | 0.000000 | 0.000000 |
| Mean ADD/ADI (mm) | 125.721221 | 124.842171 | -0.879050 |
| Median ADD/ADI (mm) | 116.525322 | 118.939662 | +2.414340 |
| Recall at 0.1 diameter | 0.000000 | 0.000000 | 0.000000 |

Per-object mean ADD/ADI in millimetres:

| Object | Single | Multi | Delta |
|---:|---:|---:|---:|
| 1 | 116.525322 | 118.939662 | +2.414340 |
| 2 | 59.828980 | 64.795099 | +4.966119 |
| 4 | 180.603943 | 181.165860 | +0.561917 |
| 5 | 115.649793 | 108.556885 | -7.092907 |
| 6 | 155.998069 | 150.753351 | -5.244718 |

The official v4 batch ran exactly two fixed commands, both exit 0. The official-once receipt SHA-256 is `41518d5e79029396cf9aa72ad1e8f200b34007952dabea604d105a4c26d06b89`; single and multi score hashes are `c193b3a37c379a847c03d285e7b5841120a59cdbc8cc14b4db169c0e610645b1` and `eaa75ffc02515d24696d5c5724fb1afa329f066ef6531c00d14e684a0f59ca0f`.

## Preserved failures

- Evaluator v1 stopped before pre-score because the C merged JSONL contains one orchestration metadata row before its 10 prediction rows. No GT or scorer was used.
- Evaluator v2 normalized that exact hash-locked row, then its only two-command batch failed `[1,1]` before scoring because `Path.resolve()` dereferenced the venv symlink and the base interpreter lacked `imageio`. Receipt SHA-256 `844fed46ab9c97c21a70af27790d1bacd612be18fdfeeef68af370e58f6233fd`; score files 0; rerun forbidden.
- Evaluator v3 was rejected statically before pre-score because two copied v2 receipt hashes had invalid lengths. Evaluator calls 0.
- Evaluator v4 preserved the `.venv/bin/python` path, passed exact-interpreter imports and loader smoke, and produced the formal result above. It is sealed with `rerun_permitted=false`.

## Evidence

- Final GPU-A evidence archive: `artifacts/r4a_v3_multi_object/final_gpu_a/poseloop-r4a-v3-final-evidence.tar.gz`, SHA-256 `03779c41b4e4a0cd2aa377e483fe295b08a4ce7642d8c66985477bf7474dae21`, 8,559,564 bytes.
- Archive receipt: `artifacts/r4a_v3_multi_object/final_gpu_a/archive-receipt.txt`, SHA-256 `cb16bce400849d387b732467de1d94b02715449616f8903b1447b8f647111af9f`.
- Local independent archive verification: 53/53 internal hashes, no bad entries, no path escape/special member, raw GT path count 0.
- Final machine-readable report inside the archive: `gpu_a_v4/final-report.json`, SHA-256 `b6792b3efb25a42c203c18e85a2b1eb3dd4d1ab738dc811cc3e3bf3a5873fb881`.
- Label-free visualization: 10/10 PNG frames and `gpu_a_v4/visualization/r4a-v3-input-single-multi.mp4`, video SHA-256 `1e8ef5ec6e8a32402e8388974adfb65a1d25043fb71d8692aa30c347cd866849`.

Final boundary: `label_access_count=0` for prediction and visualization, `xyzibd_val_access_count=0`, no result-driven tuning, no rerun, no push/merge/tag. The next scientific evaluation must use a new sealed sensor/data split.

