"""
C3: Startup-Time-Optimized Scheduler - Intelligent placement decisions.

Responsibilities:
- Choose optimal server/GPU for model loading
- Estimate loading time from different storage tiers
- Estimate migration time
- Minimize startup latency (FR 002)
"""

import ray
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from collections import defaultdict

from faasinfer.core.types import (
    PlacementDecision,
    ServerInfo,
    LoadingTask,
    StorageTier,
)
from faasinfer.config import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class TimeEstimate:
    """Time estimation for loading/migration."""
    load_time_s: float
    migration_time_s: float
    total_time_s: float
    storage_tier: str


@ray.remote
class StartupTimeScheduler:
    """
    Scheduler that minimizes model startup time using:
    - Per-tier load time estimation: q + n/b
    - Migration time estimation: a*(tin + tout) + b
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        
        # Server registry: server_id -> ServerInfo
        self.servers: Dict[str, ServerInfo] = {}
        
        # Model locations: model_id -> {server_id -> storage_tier}
        self.model_locations: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Loading queues per server
        self.loading_queues: Dict[str, List[LoadingTask]] = defaultdict(list)
        
        # Bandwidth estimates (GB/s) - updated via monitoring
        self.bandwidth_estimates = {
            "network": config.network_bandwidth_gbps,
            "ssd": config.ssd_bandwidth_gbps,
            "dram": config.dram_bandwidth_gbps,
        }
        
        # Migration parameters (learned from monitoring)
        self.migration_params = {
            "a": 0.001,  # seconds per token
            "b": 0.5,    # base overhead in seconds
        }
        
        # Statistics
        self.total_placements = 0
        self.placement_errors = {}
        
        logger.info("Initialized StartupTimeScheduler")
    
    def register_server(self, server_info: ServerInfo):
        """Register a server in the cluster."""
        self.servers[server_info.server_id] = server_info
        logger.info(f"Registered server {server_info.server_id}")
    
    def update_model_location(
        self,
        model_id: str,
        server_id: str,
        storage_tier: str
    ):
        """Update model location information."""
        self.model_locations[model_id][server_id] = storage_tier
        logger.debug(f"Model {model_id} at {server_id}:{storage_tier}")
    
    def estimate_load_time(
        self,
        model_size_gb: float,
        storage_tier: str,
        server_id: str
    ) -> float:
        """
        Estimate model loading time: q + n/b
        
        Args:
            model_size_gb: Model size in GB
            storage_tier: Storage tier ("remote", "ssd", "dram", "gpu")
            server_id: Target server
            
        Returns:
            Estimated loading time in seconds
        """
        # Get queue time (q)
        queue_time = 0.0
        for task in self.loading_queues.get(server_id, []):
            queue_time += task.estimated_load_time_s
        
        # Get bandwidth (b)
        if storage_tier == "gpu":
            return 0.0  # Already loaded
        elif storage_tier == "dram":
            bandwidth = self.bandwidth_estimates["dram"]
        elif storage_tier == "ssd":
            bandwidth = self.bandwidth_estimates["ssd"]
        else:  # remote
            bandwidth = self.bandwidth_estimates["network"]
        
        # Calculate: q + n/b
        load_time = queue_time + (model_size_gb / bandwidth)
        
        return load_time
    
    def estimate_migration_time(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate migration time: a*(tin + tout) + b
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of generated tokens
            
        Returns:
            Estimated migration time in seconds
        """
        a = self.migration_params["a"]
        b = self.migration_params["b"]
        
        return a * (input_tokens + output_tokens) + b
    
    async def place(
        self,
        model_id: str,
        model_size_gb: float = 13.0,
        tenant_id: str = "default",
        deadline_ms: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> PlacementDecision:
        """
        Choose optimal server to minimize startup time (FR 002: ≤30ms decision).
        
        Args:
            model_id: Model to load
            model_size_gb: Model size
            tenant_id: Tenant ID
            deadline_ms: Optional deadline
            input_tokens: Number of input tokens (for migration estimation)
            output_tokens: Generated tokens (for migration estimation)
            
        Returns:
            Placement decision
        """
        start_time = time.time()
        self.total_placements += 1
        
        try:
            # Get servers with this model
            model_servers = self.model_locations.get(model_id, {})
            
            if not model_servers and not self.servers:
                raise RuntimeError("No servers available")
            
            best_server = None
            best_time = float('inf')
            best_requires_loading = True
            best_requires_migration = False
            alternatives = []
            
            # Evaluate each server
            for server_id, server_info in self.servers.items():
                # Check if server has available GPUs
                if server_info.available_gpus == 0:
                    # Consider migration
                    if not self.config.enable_live_migration:
                        continue
                    
                    migration_time = self.estimate_migration_time(
                        input_tokens,
                        output_tokens
                    )
                    
                    # Check if model is on this server
                    if server_id in model_servers:
                        storage_tier = model_servers[server_id]
                        load_time = self.estimate_load_time(
                            model_size_gb,
                            storage_tier,
                            server_id
                        )
                    else:
                        # Need to load from remote
                        load_time = self.estimate_load_time(
                            model_size_gb,
                            "remote",
                            server_id
                        )
                    
                    total_time = migration_time + load_time
                    
                    alternatives.append({
                        "server_id": server_id,
                        "time": total_time,
                        "requires_migration": True,
                    })
                    
                    if total_time < best_time:
                        best_time = total_time
                        best_server = server_id
                        best_requires_migration = True
                        best_requires_loading = True
                
                else:
                    # Server has available GPUs
                    if server_id in model_servers:
                        # Model already on server
                        storage_tier = model_servers[server_id]
                        load_time = self.estimate_load_time(
                            model_size_gb,
                            storage_tier,
                            server_id
                        )
                        requires_loading = (storage_tier != "gpu")
                    else:
                        # Need to load from remote
                        load_time = self.estimate_load_time(
                            model_size_gb,
                            "remote",
                            server_id
                        )
                        requires_loading = True
                    
                    alternatives.append({
                        "server_id": server_id,
                        "time": load_time,
                        "requires_migration": False,
                    })
                    
                    if load_time < best_time:
                        best_time = load_time
                        best_server = server_id
                        best_requires_migration = False
                        best_requires_loading = requires_loading
            
            if best_server is None:
                # Fallback: pick any server with model
                if model_servers:
                    best_server = list(model_servers.keys())[0]
                    best_time = 10.0  # Default estimate
                else:
                    # Pick first available server
                    best_server = list(self.servers.keys())[0]
                    best_time = 30.0
            
            # Create placement decision
            decision = PlacementDecision(
                request_id="",  # Will be filled by router
                model_id=model_id,
                server_id=best_server,
                gpu_ids=[0],  # Simplified
                estimated_startup_time_s=best_time,
                requires_loading=best_requires_loading,
                requires_migration=best_requires_migration,
                alternatives=alternatives,
            )
            
            # Add to loading queue if needed
            if best_requires_loading:
                task = LoadingTask(
                    model_id=model_id,
                    server_id=best_server,
                    estimated_load_time_s=best_time,
                )
                self.loading_queues[best_server].append(task)
            
            # Check decision time (FR 002: ≤30ms)
            decision_time_ms = (time.time() - start_time) * 1000
            if decision_time_ms > self.config.placement_decision_timeout_ms:
                logger.warning(
                    f"Placement decision took {decision_time_ms:.1f}ms "
                    f"(target: {self.config.placement_decision_timeout_ms}ms)"
                )
            
            logger.info(
                f"Placed model {model_id} on {best_server} "
                f"(estimated time: {best_time:.2f}s, "
                f"decision time: {decision_time_ms:.1f}ms)"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Placement failed: {e}")
            raise
    
    def learn_bandwidth(
        self,
        storage_tier: str,
        observed_bandwidth_gbps: float
    ):
        """
        Update bandwidth estimates based on observations.
        
        Args:
            storage_tier: Storage tier
            observed_bandwidth_gbps: Observed bandwidth
        """
        if storage_tier in self.bandwidth_estimates:
            # Exponential moving average
            alpha = 0.3
            old = self.bandwidth_estimates[storage_tier]
            self.bandwidth_estimates[storage_tier] = (
                alpha * observed_bandwidth_gbps + (1 - alpha) * old
            )
            logger.debug(
                f"Updated {storage_tier} bandwidth: "
                f"{old:.2f} -> {self.bandwidth_estimates[storage_tier]:.2f} GB/s"
            )
    
    async def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "total_placements": self.total_placements,
            "num_servers": len(self.servers),
            "bandwidth_estimates": self.bandwidth_estimates,
            "migration_params": self.migration_params,
            "queued_tasks": sum(len(q) for q in self.loading_queues.values()),
        }