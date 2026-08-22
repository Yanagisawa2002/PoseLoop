"""A-R9 class-agnostic real instance detector development experiment."""

from .runtime import (
    ContractError,
    build_dataset_manifest,
    evaluate,
    load_protocol,
    predict,
    train,
)

__all__ = [
    "ContractError",
    "build_dataset_manifest",
    "evaluate",
    "load_protocol",
    "predict",
    "train",
]
