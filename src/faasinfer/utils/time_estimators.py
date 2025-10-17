"""
Time estimation utilities for scheduling.

Implements q + n/b formula for load time estimation.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class LoadTimeEstimator:
    """
    Estimates model loading time: q + n/b
    where:
    - q = queue time
    - n = model size
    - b = bandwidth
    """
    
    def __init__(self):
        # Bandwidth estimates (GB/s)
        self.bandwidths = {
            "remote": 1.0,   # Network
            "ssd": 6.0,      # NVMe RAID
            "dram": 50.0,    # DDR4
            "gpu": 900.0,    # V100 HBM2
        }
    
    def estimate(
        self,
        model_size_gb: float,
        storage_tier: str,
        queue_time_s: float = 0.0
    ) -> float:
        """
        Estimate loading time.
        
        Args:
            model_size_gb: Model size in GB
            storage_tier: Storage tier
            queue_time_s: Current queue time
            
        Returns:
            Estimated time in seconds
        """
        if storage_tier == "gpu":
            return 0.0  # Already loaded
        
        bandwidth = self.bandwidths.get(storage_tier, 1.0)
        transfer_time = model_size_gb / bandwidth
        
        return queue_time_s + transfer_time
    
    def update_bandwidth(self, tier: str, observed_gbps: float):
        """Update bandwidth estimate based on observation."""
        if tier in self.bandwidths:
            # Exponential moving average
            alpha = 0.3
            self.bandwidths[tier] = (
                alpha * observed_gbps + 
                (1 - alpha) * self.bandwidths[tier]
            )
            logger.info(f"Updated {tier} bandwidth: {self.bandwidths[tier]:.2f} GB/s")


class MigrationTimeEstimator:
    """
    Estimates migration time: a*(t_in + t_out) + b
    """
    
    def __init__(self):
        self.a = 0.001  # seconds per token
        self.b = 0.5    # base overhead
    
    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate migration time in seconds."""
        return self.a * (input_tokens + output_tokens) + self.b
    
    def update_params(self, observed_time: float, num_tokens: int):
        """Update parameters based on observations."""
        if num_tokens > 0:
            # Simple linear regression update
            self.a = 0.9 * self.a + 0.1 * (observed_time - self.b) / num_tokens