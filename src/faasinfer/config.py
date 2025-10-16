"""
Configuration dataclasses for FaaSInfer system.
Supports FR 001-020 configuration requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class StorageTier(Enum):
    """Storage tiers in the multi-tier hierarchy."""
    REMOTE = "remote"  # S3, GCS, Azure Blob
    SSD = "ssd"        # NVMe/SATA SSDs
    DRAM = "dram"      # Host memory
    GPU = "gpu"        # GPU memory


class BatchingPolicy(Enum):
    """Batching/scheduling policies (FR 006)."""
    VLLM_CONTINUOUS = "vllm_continuous"
    ORCA = "orca"
    SARATHI = "sarathi_serve"
    FIFO = "fifo"


class TransportType(Enum):
    """Transport types for API (FR 019)."""
    GRPC = "grpc"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str
    model_name: str  # HuggingFace model name
    model_size_gb: float
    num_parameters_b: float  # Billions
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_context_length: int = 2048
    dtype: str = "float16"
    
    # LoRA adapters support
    lora_adapters: List[str] = field(default_factory=list)
    
    # Batching configuration
    batching_policy: BatchingPolicy = BatchingPolicy.VLLM_CONTINUOUS
    max_batch_size: int = 32
    
    # KV cache configuration (FR 007)
    use_paged_kv: bool = True
    kv_quantization: Optional[str] = None  # None, "int8", "fp8"
    
    # Speculative decoding (FR 008)
    enable_speculative_decoding: bool = False
    draft_model: Optional[str] = None


@dataclass
class StorageConfig:
    """Storage system configuration (FR 004, FR 012, FR 019)."""
    # Remote storage
    remote_storage_type: str = "s3"  # s3, gcs, azure
    remote_storage_endpoint: Optional[str] = None
    remote_storage_bucket: str = "faasinfer-models"
    
    # Local storage paths
    ssd_cache_path: str = "/var/faasinfer/ssd_cache"
    dram_cache_path: str = "/dev/shm/faasinfer"
    
    # Cache sizes (GB)
    ssd_cache_size_gb: float = 500.0
    dram_cache_size_gb: float = 100.0
    
    # Cache policies
    cache_eviction_policy: str = "lru"
    cache_hit_target: float = 0.8  # FR 004: ≥80% cache hit rate
    
    # Loading optimization (FR 003)
    chunk_size_mb: int = 16
    num_io_threads: int = 4
    use_direct_io: bool = True
    use_pinned_memory: bool = True


@dataclass
class SchedulerConfig:
    """Scheduler configuration (C3, FR 002, FR 010)."""
    # Placement strategy
    locality_aware: bool = True
    startup_time_optimized: bool = True
    
    # Time estimation parameters
    placement_decision_timeout_ms: int = 30  # FR 002: ≤30ms
    
    # Bandwidth tracking (per tier)
    network_bandwidth_gbps: float = 10.0
    ssd_bandwidth_gbps: float = 6.0
    dram_bandwidth_gbps: float = 50.0
    
    # GPU heterogeneity support (FR 010)
    gpu_types: List[str] = field(default_factory=lambda: ["A10", "A100", "H100"])
    
    # Migration parameters (FR 005, FR 014)
    enable_live_migration: bool = True
    migration_pause_target_ms: int = 300  # FR 005: ≤300ms median
    max_concurrent_migrations: int = 100


@dataclass
class GatewayConfig:
    """API Gateway configuration (C1, FR 001)."""
    host: str = "0.0.0.0"
    port: int = 8000
    transport_type: TransportType = TransportType.SSE
    
    # Rate limiting (FR 015)
    enable_rate_limiting: bool = True
    default_rate_limit_per_min: int = 100
    
    # Authentication
    enable_auth: bool = True
    auth_type: str = "jwt"
    
    # Streaming configuration
    max_concurrent_streams: int = 1000  # FR 001: ≥1000
    stream_timeout_s: int = 300


@dataclass
class MigrationConfig:
    """Live migration configuration (C10, FR 005, FR 018)."""
    enable_migration: bool = True
    multi_round_migration: bool = True
    max_migration_rounds: int = 3
    
    # Token vs KV cache migration
    migrate_kv_cache: bool = False  # FR 005: token-only migration
    
    # Failure handling (FR 018)
    enable_failure_recovery: bool = True
    migration_timeout_s: int = 60


@dataclass
class AutoscalingConfig:
    """Autoscaling configuration (C11, FR 011)."""
    enable_autoscaling: bool = True
    
    # SLO targets (FR 011)
    ttft_p90_target_s: float = 2.0
    tbt_p99_target_ms: float = 200.0
    
    # Cost optimization
    cost_aware: bool = True
    budget_limit_per_hour: Optional[float] = None
    
    # Scaling parameters
    scale_up_threshold: float = 0.8  # GPU utilization
    scale_down_threshold: float = 0.3
    cooldown_period_s: int = 300


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration (C12, FR 017)."""
    enable_monitoring: bool = True
    enable_tracing: bool = True
    enable_profiling: bool = False
    
    # Vidur integration (FR 017)
    enable_vidur_planner: bool = True
    vidur_planning_interval_hours: int = 24
    
    # Metrics export
    metrics_export_interval_s: int = 10
    otel_endpoint: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration (FR 020)."""
    encrypt_at_rest: bool = True
    tls_enabled: bool = True
    
    # Ephemeral state
    ephemeral_kv_cache: bool = True
    retention_window_hours: int = 24
    
    # Key rotation
    key_rotation_days: int = 90


@dataclass
class FaaSInferConfig:
    """Main configuration for FaaSInfer system."""
    # Ray cluster configuration
    ray_address: Optional[str] = None  # None = start local cluster
    ray_namespace: str = "faasinfer"
    
    # Component configurations
    models: List[ModelConfig] = field(default_factory=list)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    autoscaling: AutoscalingConfig = field(default_factory=AutoscalingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System-wide settings
    log_level: str = "INFO"
    num_gpus_per_node: int = 4
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.models:
            raise ValueError("At least one model must be configured")
        
        # Validate TTFT targets (FR 013)
        for model in self.models:
            if model.num_parameters_b <= 8:
                assert model.max_context_length >= 2048
            elif model.num_parameters_b <= 13:
                assert model.max_context_length >= 2048
        
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FaaSInferConfig":
        """Load configuration from dictionary."""
        # Implementation for loading from dict
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Implementation for converting to dict
        return self.__dict__