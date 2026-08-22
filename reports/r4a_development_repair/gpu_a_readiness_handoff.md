# R4-A GPU-A development-readiness handoff

This handoff is development-only. The consumed XYZ-IBD validation evaluation,
its scores, receipts, and evidence remain immutable. No validation GT, score
JSON, or evaluator entrypoint was opened or invoked by R4-A.

## Label-free structural postmortem

The frozen five-file R3-v3 bundle covers 8 of 750 images (1.0667%), one of 15
public scenes, and one of 15 public objects. All emitted rotations are finite
legal SO(3), and the emitted translations are structurally consistent with BOP
millimetres. The primary defect is the global eight-image, fixed-object smoke
enumerator: an unenumerated target creates neither an inference work item nor an
explicit missing-result record. The detailed report is
`reports/r4a_development_repair/structural_postmortem.md`; the remote
machine-readable audit is
`r4a_development_coverage_geometry/structural_postmortem/prediction-audit.json`
(SHA-256
`95aea4beb1a0c16bd2add46c1ce3fb4dc0f741e1224e756bba3606b53e176d2a`).

## Frozen development input

- Role: `DEVELOPMENT_ONLY`; official XYZ-IBD `train_pbr`, never eligible as a
  sealed split.
- Official source revision: `4fe4671783172622313ac0c7182012cee618f217`.
- Fixed slice: scenes `000000`, `000001`, `000002`; image `000000`; `gray`,
  depth, all visible masks, scene camera, scene GT, and scene GT info.
- Exact extraction: 188 files and 173 declared instances; extraction result
  SHA-256
  `35a1a84f2ebb624feecf8a5c3757b06d1670dc0d764858fecd44938741297dda`.
- Slice plan SHA-256
  `b8ec0ce6b93064d689b10be656eb0961cc2ccb41a154b4798ec9a9f6861bc768`;
  slice prefreeze SHA-256
  `443fabb778813904097f220a5a8c7eb8507309a802abb45fd1ee5a5e5c49d875`.
- Public CAD domain: all 15 expected object files are present and hashed.

Remote evidence root:
`/root/autodl-tmp/poseloop_r3_20260816_01a003cf/r4a_development_coverage_geometry`.

## Readiness protocols and immutable attempts

Readiness v1 was frozen locally at commit
`b75c495423dbf1b936fc169bc271f97a66c5d0a5` and executed from the independent
GPU-A commit `3a33144dd950bbae83a26e85ba6d8edd16efacc8`. Protocol SHA-256 is
`5820f9f65cb06574173bcdc140e9868571e5fea48fedd6761cdf68f3646345f9`.
The screen `poseloop_r4a_development_readiness_v1` ended once with exit 1
because the official raw rotation is a nested 3x3 array rather than a flat
nine-value array. Its job log SHA-256 is
`720b15a8ce4736035aaaad905447e87e74ecb1807068cc77c9bce50468353084`,
and its sealed failure receipt SHA-256 is
`1c68d619f865ee598f748116c041e8f8d759fab428900b1153bcb338d580ca5f`.
`rerun_permitted=false`.

The schema-only diagnosis recorded no pose values. All 173 raw rotations have
shape 3x3 and translations have length 3; the pinned BOP loader returns shapes
3x3 and 3x1. Diagnosis SHA-256 is
`0c39209c4e59dc484ba86b7fcb95d46603901f4530dfd14fc820d1fd08fa60d4`.

Readiness v2 moved the repair into a new namespace and validates both official
raw and pinned-loader shapes. It was frozen locally at commit
`ff9e0e7eeda19281b65086eeed26b46e8414ce6e` and executed from GPU-A commit
`ed937dd0c0fef390fc6462c1bfcc9e5c3551e468`. Protocol SHA-256 is
`8f7553acfb4db10427c961b1fc4be0717d49776e2488c996242401958f143fa8`.
Its prefreeze lock is
`acbd2b19550423187cea54457745784c7c7786215394a6f84a57b2087ae2cbb9`;
at freeze, development-label file count, validation access count, and evaluator
invocation count were all zero.

The screen `poseloop_r4a_development_readiness_v2` also ended once with exit 1,
after all 173 target schemas passed, because the frozen minimum-two-object
grouping gate failed. Its job log SHA-256 is
`3bc9dce602d6fdfa8209f7833c9cddc09221b7bb9508896d3b0257d7a97eb280`,
and its sealed failure receipt SHA-256 is
`125a0aca441470d4377ca327181717a0454a633c399a70102689df700a2d5b8c`.
`rerun_permitted=false`.

The ID-only diagnosis contains no pose or visibility values. All 173 instances
are object 11: 58, 55, and 60 instances in scenes 0, 1, and 2 respectively.
Its SHA-256 is
`ffd58c01d5775802ab2547f889ce38c88709a51a95e94c8adff13aac77cc78c7`.

## Stop gate and exact blockers

The object-group readiness gate did not pass, so no target manifest was
promoted, no FoundationPose before/after inference was started, and no ADD(-S),
BOP24 development metric, overlay, axes projection, or multi-view strategy was
claimed. Relaxing the frozen gate to one object would not validate the object
mapping repair and is prohibited after observing the development labels.

GPU-A has the pinned BOP toolkit, a working venv, the official slice, and public
models, but it has no FoundationPose source/runtime or frozen FoundationPose
checkpoint bundle. The prior predictions were produced on GPU-C. Continuing
requires both:

1. a newly frozen, object-diverse `DEVELOPMENT_ONLY` slice selected without pose
   or score guidance; and
2. the GPU-C FoundationPose source/checkpoint/provenance bundle with matching
   hashes, or an equivalent immutable handoff to GPU-A.

No package was installed to paper over this blocker. GPU-A remains powered on;
the final observed GPU state was RTX 5090, 0 MiB of 32607 MiB used. No push,
merge, tag, shutdown, release, or disk deletion was performed.

## Verification

Local R4-A tests: 22 passed. `compileall` and `git diff --check` passed. The
GPU-A venv has no pytest, so no dependency was installed; remote `compileall`
passed and the pinned toolkit loader import smoke passed.
