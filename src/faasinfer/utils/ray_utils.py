"""Ray cluster utilities."""

import ray
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_cluster_resources() -> Dict:
    """Get available cluster resources."""
    if not ray.is_initialized():
        return {}
    
    resources = ray.cluster_resources()
    return {
        "total_cpus": resources.get("CPU", 0),
        "total_gpus": resources.get("GPU", 0),
        "total_memory_gb": resources.get("memory", 0) / 1e9,
        "total_object_store_gb": resources.get("object_store_memory", 0) / 1e9,
    }


def get_available_nodes() -> List[Dict]:
    """Get list of available nodes."""
    if not ray.is_initialized():
        return []
    
    nodes = ray.nodes()
    return [
        {
            "node_id": node["NodeID"],
            "alive": node["Alive"],
            "resources": node.get("Resources", {}),
        }
        for node in nodes
        if node["Alive"]
    ]
