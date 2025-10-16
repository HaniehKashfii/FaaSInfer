"""
C12: Telemetry, Profiler & Monitoring

Responsibilities:
- Collect request and system metrics
- Export to monitoring systems
- Track TTFT, TBT, cache stats (FR 017)
"""

import time
import logging
from typing import Dict, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    model_id: str
    
    # Timing
    arrival_time: float
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Tokens
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    
    # Migration
    was_migrated: bool = False
    migration_count: int = 0
    
    # Server
    server_id: Optional[str] = None
    
    def get_ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time and self.arrival_time:
            return (self.first_token_time - self.arrival_time) * 1000
        return None
    
    def get_tbt_ms(self) -> Optional[float]:
        """Time between tokens in milliseconds."""
        if self.end_time and self.first_token_time and self.num_generated_tokens > 1:
            total_time = self.end_time - self.first_token_time
            return (total_time / (self.num_generated_tokens - 1)) * 1000
        return None
    
    def get_total_latency_ms(self) -> Optional[float]:
        """Total latency in milliseconds."""
        if self.end_time and self.arrival_time:
            return (self.end_time - self.arrival_time) * 1000
        return None


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    Supports FR 017: TTFT, TBT, batch mix, cache stats.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        export_interval_s: int = 10,
    ):
        self.window_size = window_size
        self.export_interval_s = export_interval_s
        
        # Recent requests (sliding window)
        self.recent_requests: Deque[RequestMetrics] = deque(maxlen=window_size)
        
        # Aggregated metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
        # Per-model metrics
        self.model_requests: Dict[str, int] = defaultdict(int)
        self.model_tokens: Dict[str, int] = defaultdict(int)
        
        # Migration metrics
        self.total_migrations = 0
        self.migration_pause_times: Deque[float] = deque(maxlen=100)
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Last export time
        self.last_export = time.time()
        
        logger.info("Initialized MetricsCollector")
    
    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a completed request."""
        self.recent_requests.append(metrics)
        self.completed_requests += 1
        
        # Update per-model metrics
        self.model_requests[metrics.model_id] += 1
        self.model_tokens[metrics.model_id] += metrics.num_generated_tokens
        
        # Update migration metrics
        if metrics.was_migrated:
            self.total_migrations += 1
    
    def record_migration(self, pause_time_ms: float):
        """Record migration pause time."""
        self.migration_pause_times.append(pause_time_ms)
    
    def record_cache_access(self, hit: bool):
        """Record cache access."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_ttft_stats(self) -> Dict[str, float]:
        """
        Get TTFT statistics (FR 017).
        
        Returns:
            Dict with p50, p90, p99, mean
        """
        ttfts = [
            m.get_ttft_ms()
            for m in self.recent_requests
            if m.get_ttft_ms() is not None
        ]
        
        if not ttfts:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
            }
        
        sorted_ttfts = sorted(ttfts)
        
        return {
            "mean": statistics.mean(ttfts),
            "p50": sorted_ttfts[int(len(sorted_ttfts) * 0.5)],
            "p90": sorted_ttfts[int(len(sorted_ttfts) * 0.9)],
            "p99": sorted_ttfts[int(len(sorted_ttfts) * 0.99)] if len(sorted_ttfts) > 1 else sorted_ttfts[0],
        }
    
    def get_tbt_stats(self) -> Dict[str, float]:
        """
        Get TBT (time between tokens) statistics (FR 017).
        
        Returns:
            Dict with p50, p99, mean
        """
        tbts = [
            m.get_tbt_ms()
            for m in self.recent_requests
            if m.get_tbt_ms() is not None
        ]
        
        if not tbts:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p99": 0.0,
            }
        
        sorted_tbts = sorted(tbts)
        
        return {
            "mean": statistics.mean(tbts),
            "p50": sorted_tbts[int(len(sorted_tbts) * 0.5)],
            "p99": sorted_tbts[int(len(sorted_tbts) * 0.99)] if len(sorted_tbts) > 1 else sorted_tbts[0],
        }
    
    def get_migration_stats(self) -> Dict[str, float]:
        """Get migration statistics."""
        if not self.migration_pause_times:
            return {
                "total_migrations": self.total_migrations,
                "avg_pause_ms": 0.0,
                "p50_pause_ms": 0.0,
                "p95_pause_ms": 0.0,
            }
        
        sorted_pauses = sorted(self.migration_pause_times)
        
        return {
            "total_migrations": self.total_migrations,
            "avg_pause_ms": statistics.mean(self.migration_pause_times),
            "p50_pause_ms": sorted_pauses[int(len(sorted_pauses) * 0.5)],
            "p95_pause_ms": sorted_pauses[int(len(sorted_pauses) * 0.95)] if len(sorted_pauses) > 1 else sorted_pauses[0],
        }
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics (FR 017)."""
        total_accesses = self.cache_hits + self.cache_misses
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / total_accesses if total_accesses > 0 else 0.0,
        }
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "ttft": self.get_ttft_stats(),
            "tbt": self.get_tbt_stats(),
            "migration": self.get_migration_stats(),
            "cache": self.get_cache_stats(),
            "model_requests": dict(self.model_requests),
            "model_tokens": dict(self.model_tokens),
        }
    
    def export_metrics(self) -> Dict:
        """
        Export metrics for external monitoring (FR 017).
        
        Returns:
            Metrics in export format
        """
        current_time = time.time()
        
        # Check if it's time to export
        if current_time - self.last_export < self.export_interval_s:
            return {}
        
        metrics = self.get_summary()
        metrics["timestamp"] = current_time
        
        self.last_export = current_time
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.recent_requests.clear()
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.model_requests.clear()
        self.model_tokens.clear()
        self.total_migrations = 0
        self.migration_pause_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("Reset all metrics")