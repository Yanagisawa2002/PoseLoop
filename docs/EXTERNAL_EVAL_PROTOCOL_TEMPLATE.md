# Untouched external evaluation protocol template

Use this template before running PoseLoop on a new external dataset or capture.
The point is to establish external validity without turning the new benchmark into
another development split.

## 1. Evaluation identity

- dataset / capture source:
- exact version, commit, archive hash, or acquisition date:
- camera / sensor:
- object population:
- selected scenes / sequences:
- exclusion rules fixed before inference:
- evidence that labels for this evaluation population have not been used for
  model, threshold, or route selection:

## 2. Frozen system identity

- PoseLoop commit:
- detector architecture:
- detector checkpoint SHA-256:
- FoundationPose commit:
- FoundationPose checkpoint identities:
- BOP Toolkit commit if applicable:
- CUDA / PyTorch / GPU environment:

## 3. Frozen inference configuration

Record every threshold and candidate/refinement setting that can change output.
Do not choose these values after inspecting external labels.

## 4. Metrics

Separate official benchmark metrics from project-specific diagnostics.
If an official evaluator exists, report its metric under its exact name and
version. Any custom joint metric must retain its explicit success definition.

## 5. Run boundary

Before opening evaluation labels:

- freeze the input manifest;
- run inference once under the declared configuration;
- freeze prediction identities / hashes;
- record completion and failure counts;
- only then run label-dependent evaluation.

If an environment-only failure occurs, document the repair and whether inference
was rerun. Do not change model, checkpoint, threshold, candidate count, refinement
count or promotion gates without declaring a new protocol.

## 6. Result reporting

Report:

- full aggregate metrics;
- per-scene/per-object breakdown where permitted;
- runtime completion;
- representative successes and failures;
- negative results;
- any mismatch between development and external behavior.

A lower external score is still useful evidence. Do not retune the external split
merely to recover the development number.
