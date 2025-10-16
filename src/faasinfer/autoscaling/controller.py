"""
C11: Autoscaler & Cost-Aware Placement

Responsibilities:
- Scale Ray nodes based on metrics (FR 011)
- Maintain SLO targets (TTFT P90, TBT P99)
- Optimize cost per QPS
"""

import ray
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from faasinfer.config import AutoscalingConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalingDecision:
    """Autoscaling decision."""
    action: str  # "scale_up", "scale_down", "no_change"
    target_nodes: int
    reason: str
    estimated_cost: float


@ray.remote
class AutoscalingController:
    """
    Autoscaling controller for Ray cluster.
    
    Implements FR 011: Scale based on:
    - Queue delay
    - TTFT/TBT SLOs
    - Cost per QPS
    """
    
    def __init__(self, config: AutoscalingConfig):
        self.config = config
        
        # Current cluster state
        self.current_nodes = 0
        self.target_nodes = 0
        
        # SLO targets (FR 011)
        self.ttft_p90_target = config.ttft_p90_target_s
        self.tbt_p99_target = config.tbt_p99_target_ms
        
        # Scaling thresholds
        self.scale_up_threshold = config.scale_up_threshold
        self.scale_down_threshold = config.scale_down_threshold
        
        # Cost tracking
        self.cost_per_node_hour = 10.0  # USD per node per hour
        self.total_cost = 0.0
        
        # Cooldown
        self.last_scale_time = 0.0
        self.cooldown_period = config.cooldown_period_s
        
        # References to other components
        self.metrics_collector = None
        
        logger.info("Initialized AutoscalingController")
    
    def set_metrics_collector(self, metrics_collector):
        """Set metrics collector reference."""
        self.metrics_collector = metrics_collector
    
    async def reconcile(self, cluster_metrics: Dict) -> ScalingDecision:
        """
        Reconcile desired state based on metrics.
        
        Implements FR 011: Keep TTFT P90 ≤ 2s, TBT P99 ≤ 200ms
        while minimizing $/QPS.
        
        Args:
            cluster_metrics: Current cluster metrics
            
        Returns:
            Scaling decision
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_time < self.cooldown_period:
            return ScalingDecision(
                action="no_change",
                target_nodes=self.target_nodes,
                reason="cooldown_period",
                estimated_cost=self.total_cost,
            )
        
        # Extract metrics
        gpu_utilization = cluster_metrics.get("gpu_utilization", 0.0)
        queue_length = cluster_metrics.get("queue_length", 0)
        ttft_p90 = cluster_metrics.get("ttft_p90_ms", 0.0) / 1000  # Convert to seconds
        tbt_p99 = cluster_metrics.get("tbt_p99_ms", 0.0)
        qps = cluster_metrics.get("requests_per_second", 0.0)
        
        # Check SLO violations
        ttft_violation = ttft_p90 > self.ttft_p90_target
        tbt_violation = tbt_p99 > self.tbt_p99_target
        
        # Determine scaling action
        decision = self._make_scaling_decision(
            gpu_utilization=gpu_utilization,
            queue_length=queue_length,
            ttft_violation=ttft_violation,
            tbt_violation=tbt_violation,
            qps=qps,
        )
        
        # Check budget limit
        if self.config.budget_limit_per_hour is not None:
            estimated_cost = decision.target_nodes * self.cost_per_node_hour
            if estimated_cost > self.config.budget_limit_per_hour:
                logger.warning(
                    f"Scaling would exceed budget: "
                    f"${estimated_cost:.2f} > ${self.config.budget_limit_per_hour:.2f}"
                )
                decision.action = "no_change"
                decision.reason = "budget_limit"
        
        # Execute scaling action
        if decision.action != "no_change":
            await self._execute_scaling(decision)
            self.last_scale_time = current_time
        
        return decision
    
    def _make_scaling_decision(
        self,
        gpu_utilization: float,
        queue_length: int,
        ttft_violation: bool,
        tbt_violation: bool,
        qps: float,
    ) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        
        # Scale up conditions
        if gpu_utilization > self.scale_up_threshold:
            return ScalingDecision(
                action="scale_up",
                target_nodes=self.current_nodes + 1,
                reason=f"gpu_utilization_high ({gpu_utilization:.2f})",
                estimated_cost=self.total_cost,
            )
        
        if ttft_violation:
            return ScalingDecision(
                action="scale_up",
                target_nodes=self.current_nodes + 1,
                reason="ttft_slo_violation",
                estimated_cost=self.total_cost,
            )
        
        if tbt_violation:
            return ScalingDecision(
                action="scale_up",
                target_nodes=self.current_nodes + 1,
                reason="tbt_slo_violation",
                estimated_cost=self.total_cost,
            )
        
        if queue_length > 100:
            return ScalingDecision(
                action="scale_up",
                target_nodes=self.current_nodes + 1,
                reason=f"queue_length_high ({queue_length})",
                estimated_cost=self.total_cost,
            )
        
        # Scale down conditions
        if (gpu_utilization < self.scale_down_threshold and
            self.current_nodes > 1 and
            not ttft_violation and
            not tbt_violation):
            return ScalingDecision(
                action="scale_down",
                target_nodes=self.current_nodes - 1,
                reason=f"gpu_utilization_low ({gpu_utilization:.2f})",
                estimated_cost=self.total_cost,
            )
        
        # No change
        return ScalingDecision(
            action="no_change",
            target_nodes=self.current_nodes,
            reason="within_thresholds",
            estimated_cost=self.total_cost,
        )
    
    async def _execute_scaling(self, decision: ScalingDecision):
        """
        Execute scaling action.
        
        In production, would use Ray autoscaler or cloud APIs.
        """
        logger.info(
            f"Scaling {decision.action}: "
            f"{self.current_nodes} -> {decision.target_nodes} nodes "
            f"(reason: {decision.reason})"
        )
        
        if decision.action == "scale_up":
            # In real implementation, would provision new nodes
            self.current_nodes = decision.target_nodes
            self.target_nodes = decision.target_nodes
            
        elif decision.action == "scale_down":
            # In real implementation, would terminate nodes gracefully
            self.current_nodes = decision.target_nodes
            self.target_nodes = decision.target_nodes
    
    async def get_status(self) -> Dict:
        """Get autoscaler status."""
        return {
            "enabled": self.config.enable_autoscaling,
            "current_nodes": self.current_nodes,
            "target_nodes": self.target_nodes,
            "last_scale_time": self.last_scale_time,
            "total_cost": self.total_cost,
            "slo_targets": {
                "ttft_p90_s": self.ttft_p90_target,
                "tbt_p99_ms": self.tbt_p99_target,
            },
        }
    
    def update_cost(self, duration_hours: float):
        """Update total cost."""
        self.total_cost += self.current_nodes * self.cost_per_node_hour * duration_hours