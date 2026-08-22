"""A-R7 instance-deblending preparation.

The package is deliberately PREP-only.  It freezes the scientific boundary,
synthetic occlusion gates, and prompt-exclusion invariants before any replay of
the ten A-R5/A-R6 development frames.
"""

from .contracts import PROTOCOL_ID, PROTOCOL_SCHEMA, validate_protocol
from .adapter import (
    derive_depth_discontinuity_seeds,
    merge_seed_sources,
    predict_prompt_exclusive_children,
)
from .core import (
    DeblendCandidate,
    evaluate_instance_set,
    evaluate_synthetic_gate,
    partition_prompted_candidates,
)

__all__ = [
    "DeblendCandidate",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "evaluate_instance_set",
    "evaluate_synthetic_gate",
    "derive_depth_discontinuity_seeds",
    "merge_seed_sources",
    "partition_prompted_candidates",
    "predict_prompt_exclusive_children",
    "validate_protocol",
]
