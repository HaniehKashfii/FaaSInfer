"""Batching and scheduling policies for FaaSInfer."""

from faasinfer.batching.policies import (
    BatchingPolicy,
    VLLMContinuousBatching,
    OrcaBatching,
    FIFOBatching,
)
from faasinfer.batching.registry import PolicyRegistry

__all__ = [
    "BatchingPolicy",
    "VLLMContinuousBatching", 
    "OrcaBatching",
    "FIFOBatching",
    "PolicyRegistry",
]