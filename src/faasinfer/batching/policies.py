"""
C7: Batching policies implementation.

Supports multiple scheduling strategies:
- vLLM Continuous Batching
- Orca
- FIFO
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BatchPlan:
    """Batch execution plan."""
    request_ids: List[str]
    batch_size: int
    estimated_latency_ms: float


class BatchingPolicy(ABC):
    """Base class for batching policies."""
    
    @abstractmethod
    def tick(self, queue_state: Dict) -> BatchPlan:
        """Generate batch plan from queue state."""
        pass


class VLLMContinuousBatching(BatchingPolicy):
    """
    vLLM-style continuous batching.
    Dynamically adds/removes requests as they complete.
    """
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        logger.info("Initialized VLLMContinuousBatching")
    
    def tick(self, queue_state: Dict) -> BatchPlan:
        """
        Continuous batching: pack as many as fit in batch.
        
        Args:
            queue_state: Current queue state with pending requests
            
        Returns:
            Batch plan
        """
        pending = queue_state.get("pending_requests", [])
        active = queue_state.get("active_requests", [])
        
        # Fill up to max batch size
        available_slots = self.max_batch_size - len(active)
        to_add = pending[:available_slots]
        
        all_requests = active + to_add
        
        return BatchPlan(
            request_ids=all_requests,
            batch_size=len(all_requests),
            estimated_latency_ms=10.0 * len(all_requests),
        )


class OrcaBatching(BatchingPolicy):
    """
    Orca-style iteration-level batching.
    Schedules based on iteration phases.
    """
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        logger.info("Initialized OrcaBatching")
    
    def tick(self, queue_state: Dict) -> BatchPlan:
        """Orca batching with selective admission."""
        pending = queue_state.get("pending_requests", [])
        
        # Orca: prioritize by arrival time and deadline
        sorted_pending = sorted(
            pending,
            key=lambda r: r.get("arrival_time", 0)
        )
        
        selected = sorted_pending[:self.max_batch_size]
        
        return BatchPlan(
            request_ids=[r["request_id"] for r in selected],
            batch_size=len(selected),
            estimated_latency_ms=15.0 * len(selected),
        )


class FIFOBatching(BatchingPolicy):
    """Simple FIFO batching."""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        logger.info("Initialized FIFOBatching")
    
    def tick(self, queue_state: Dict) -> BatchPlan:
        """FIFO: take first N requests."""
        pending = queue_state.get("pending_requests", [])
        selected = pending[:self.max_batch_size]
        
        return BatchPlan(
            request_ids=[r["request_id"] for r in selected],
            batch_size=len(selected),
            estimated_latency_ms=8.0 * len(selected),
        )

