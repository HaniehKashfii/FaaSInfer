"""
Bandwidth tracking for storage tiers.

Monitors actual bandwidth to improve scheduling estimates.
"""

import time
import logging
from typing import Dict, List
from collections import deque

logger = logging.getLogger(__name__)


class BandwidthTracker:
    """Tracks bandwidth for different storage tiers."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Recent measurements: tier -> deque of (bytes, duration)
        self.measurements: Dict[str, deque] = {
            "remote": deque(maxlen=window_size),
            "ssd": deque(maxlen=window_size),
            "dram": deque(maxlen=window_size),
        }
    
    def record(self, tier: str, bytes_transferred: int, duration_s: float):
        """Record a bandwidth measurement."""
        if tier in self.measurements:
            self.measurements[tier].append((bytes_transferred, duration_s))
    
    def get_bandwidth_gbps(self, tier: str) -> float:
        """Get average bandwidth in GB/s."""
        if tier not in self.measurements:
            return 1.0  # Default
        
        measurements = self.measurements[tier]
        if not measurements:
            return 1.0
        
        # Calculate average
        total_bytes = sum(m[0] for m in measurements)
        total_time = sum(m[1] for m in measurements)
        
        if total_time == 0:
            return 1.0
        
        # Convert to GB/s
        bandwidth = (total_bytes / 1e9) / total_time
        return bandwidth
    
    def get_p95_bandwidth_gbps(self, tier: str) -> float:
        """Get P95 bandwidth (conservative estimate)."""
        if tier not in self.measurements:
            return 1.0
        
        measurements = self.measurements[tier]
        if len(measurements) < 20:
            return self.get_bandwidth_gbps(tier)
        
        # Calculate bandwidth for each measurement
        bandwidths = [
            (m[0] / 1e9) / m[1] if m[1] > 0 else 0
            for m in measurements
        ]
        
        # Get P95 (conservative)
        sorted_bw = sorted(bandwidths)
        p95_idx = int(len(sorted_bw) * 0.05)  # Lower 5%
        
        return sorted_bw[p95_idx] if sorted_bw else 1.0