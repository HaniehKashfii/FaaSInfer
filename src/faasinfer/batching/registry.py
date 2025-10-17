"""Policy registry for dynamic policy switching."""

import logging
from typing import Dict, Type
from faasinfer.batching.policies import (
    BatchingPolicy,
    VLLMContinuousBatching,
    OrcaBatching,
    FIFOBatching,
)

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """Registry for batching policies."""
    
    def __init__(self):
        self._policies: Dict[str, Type[BatchingPolicy]] = {
            "vllm_continuous": VLLMContinuousBatching,
            "orca": OrcaBatching,
            "fifo": FIFOBatching,
        }
        logger.info("Initialized PolicyRegistry")
    
    def register(self, name: str, policy_class: Type[BatchingPolicy]):
        """Register a new policy."""
        self._policies[name] = policy_class
        logger.info(f"Registered policy: {name}")
    
    def get(self, name: str, **kwargs) -> BatchingPolicy:
        """Get policy instance by name."""
        if name not in self._policies:
            raise ValueError(f"Unknown policy: {name}")
        
        return self._policies[name](**kwargs)
    
    def list_policies(self) -> List[str]:
        """List available policies."""
        return list(self._policies.keys())