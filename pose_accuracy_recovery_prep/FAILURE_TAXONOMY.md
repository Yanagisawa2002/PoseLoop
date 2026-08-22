# Failure taxonomy

| ID | Gate | Evidence | Owner |
| --- | --- | --- | --- |
| `F_INPUT_CONTRACT` | Missing or hash-invalid RGB/depth/camera/CAD/mask; incomplete units or coordinates | manifest validation JSON | bundle preflight |
| `F_COVERAGE` | Duplicate or missing scene-image-object-mask key | run-plan and prediction coverage | orchestrator |
| `F_MASK` | Empty/implausible mask or mask-variant-specific degradation | paired mask deltas and mask overlay | mask producer |
| `F_SCORER_RANK` | Near-GT frozen perturbation is not in configured top-k | scorer diagnostic JSON/CSV | scorer |
| `F_REFINER_NONMONOTONIC` | ADD(-S), rotation, or translation error increases by refiner step | refiner trace JSON/CSV | refiner |
| `F_SE3_GEOMETRY` | Illegal SO(3), non-finite pose, wrong unit, or reversed transform | contract and internal metrics | pose emitter |
| `F_OFFICIAL_CAPABILITY` | Dataset/toolkit lacks a required official error type or asset | capability receipt; unavailable reason | evaluator |
| `F_RUNTIME` | Producer/refiner failure, timeout, or OOM | producer status rows | runtime |

Categories locate a subsystem; they do not authorize hyperparameter selection or make an accuracy claim. Oracle-mask and GT-derived diagnoses remain evaluator-only.
