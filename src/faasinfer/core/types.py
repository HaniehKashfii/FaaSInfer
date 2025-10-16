"""
Shared type definitions for FaaSInfer core components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time
import uuid


class RequestStatus(Enum):
    """Status of an inference request."""
    PENDING = "pending"
    LOADING = "loading"
    RUNNING = "running"
    MIGRATING = "migrating"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ModelStatus(Enum):
    """Status of a model on a worker."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    EVICTING = "evicting"


class ServerStatus(Enum):
    """Status of a server/node."""
    IDLE = "idle"
    BUSY = "busy"
    FULL = "full"
    OFFLINE = "offline"


@dataclass
class InferenceRequest:
    """Inference request representation."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    
    # Input
    prompt: str = ""
    tokens: List[int] = field(default_factory=list)
    
    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    
    # Metadata
    tenant_id: str = "default"
    priority: int = 0
    deadline_ms: Optional[int] = None
    
    # Timestamps
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Status
    status: RequestStatus = RequestStatus.PENDING
    current_server: Optional[str] = None
    
    # Migration tracking
    migration_count: int = 0
    migration_history: List[str] = field(default_factory=list)
    
    def get_ttft(self) -> Optional[float]:
        """Time to first token."""
        if self.start_time:
            return self.start_time - self.arrival_time
        return None
    
    def get_latency(self) -> Optional[float]:
        """Total latency."""
        if self.end_time:
            return self.end_time - self.arrival_time
        return None


@dataclass
class InferenceResponse:
    """Inference response."""
    request_id: str
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    finish_reason: Optional[str] = None
    
    # Migration flag
    migrated: bool = False
    migration_destination: Optional[str] = None
    
    # Statistics
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    generation_time_ms: float = 0.0


@dataclass
class ServerInfo:
    """Information about a server node."""
    server_id: str
    node_id: str
    
    # GPU information
    num_gpus: int
    gpu_type: str
    gpu_memory_gb: float
    
    # Storage information
    dram_capacity_gb: float
    ssd_capacity_gb: float
    
    # Bandwidth information
    network_bandwidth_gbps: float
    ssd_bandwidth_gbps: float
    dram_bandwidth_gbps: float
    pcie_bandwidth_gbps: float
    
    # Status
    status: ServerStatus = ServerStatus.IDLE
    available_gpus: int = 0
    
    # Loaded models
    loaded_models: Dict[str, ModelStatus] = field(default_factory=dict)
    
    # Cache status
    dram_used_gb: float = 0.0
    ssd_used_gb: float = 0.0


@dataclass
class ModelPartition:
    """Model partition information for multi-GPU inference."""
    model_id: str
    partition_id: int
    gpu_id: int
    size_gb: float
    
    # Storage location
    storage_tier: str  # "remote", "ssd", "dram", "gpu"
    storage_path: str
    
    # Loading metadata
    tensor_index_path: str
    num_tensors: int


@dataclass
class LoadingTask:
    """Model loading task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    server_id: str = ""
    gpu_ids: List[int] = field(default_factory=list)
    
    # Timing estimates
    estimated_load_time_s: float = 0.0
    estimated_migration_time_s: float = 0.0
    
    # Status
    status: str = "pending"
    actual_load_time_s: Optional[float] = None
    
    # Priority
    priority: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class MigrationTask:
    """Live migration task."""
    migration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    
    # Source and destination
    source_server: str = ""
    dest_server: str = ""
    
    # Model information
    model_id: str = ""
    
    # Token state
    input_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    
    # Migration rounds
    current_round: int = 0
    max_rounds: int = 3
    
    # Status
    status: str = "pending"
    start_time: float = field(default_factory=time.time)
    pause_time_ms: Optional[float] = None
    
    # Failure handling
    failed: bool = False
    failure_reason: Optional[str] = None


@dataclass
class PlacementDecision:
    """Scheduler placement decision."""
    request_id: str
    model_id: str
    
    # Selected server
    server_id: str
    gpu_ids: List[int]
    
    # Decision rationale
    estimated_startup_time_s: float
    requires_loading: bool
    requires_migration: bool
    
    # Alternative considered
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ServerMetrics:
    """Real-time server metrics."""
    server_id: str
    timestamp: float = field(default_factory=time.time)
    
    # GPU utilization
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    
    # Queue status
    queue_length: int = 0
    
    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Latency statistics
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    tbt_p50_ms: float = 0.0
    tbt_p99_ms: float = 0.0
    
    # Cache statistics
    cache_hit_rate: float = 0.0
    cache_evictions: int = 0


@dataclass
class ClusterMetrics:
    """Cluster-wide metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # Server metrics
    num_servers: int = 0
    num_active_servers: int = 0
    total_gpus: int = 0
    available_gpus: int = 0
    
    # Request statistics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    # Performance metrics
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    
    avg_tbt_ms: float = 0.0
    p99_tbt_ms: float = 0.0
    
    # Migration statistics
    total_migrations: int = 0
    successful_migrations: int = 0
    avg_migration_pause_ms: float = 0.0
    
    # Cost metrics
    cost_per_hour: float = 0.0
    qps_per_dollar: float = 0.0