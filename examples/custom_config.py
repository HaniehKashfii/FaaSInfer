"""
Custom configuration example.

Shows how to customize FaaSInfer for specific workloads.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig, StorageConfig
from faasinfer.config import (
    BatchingPolicy,
    SchedulerConfig,
    AutoscalingConfig,
    MonitoringConfig,
)
from faasinfer.utils.logging import setup_logging


async def main():
    """Run with custom configuration."""
    setup_logging(log_level="INFO")
    
    # Create custom config
    config = FaaSInferConfig()
    
    # Model configuration for 8x V100
    config.models = [
        ModelConfig(
            model_id="opt-6.7b",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            tensor_parallel_size=1,
            max_context_length=2048,
            batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
            max_batch_size=64,  # Larger batch
            use_paged_kv=True,
            kv_quantization="int8",  # Enable quantization
        ),
    ]
    
    # Storage: Use your 5.9TB SSD and 448GB RAM
    config.storage = StorageConfig(
        ssd_cache_size_gb=1000.0,  # 1TB of 5.9TB
        dram_cache_size_gb=256.0,   # 256GB of 448GB
        ssd_cache_path="/var/faasinfer/ssd_cache",
        dram_cache_path="/dev/shm/faasinfer",
        cache_eviction_policy="lru",
        cache_hit_target=0.85,  # Target 85% hit rate
        chunk_size_mb=32,  # Larger chunks for V100
        num_io_threads=8,
        use_direct_io=True,
        use_pinned_memory=True,
    )
    
    # Scheduler optimized for V100
    config.scheduler = SchedulerConfig(
        locality_aware=True,
        startup_time_optimized=True,
        network_bandwidth_gbps=25.0,  # V100 NVLink
        ssd_bandwidth_gbps=6.0,
        dram_bandwidth_gbps=50.0,
        enable_live_migration=True,
        migration_pause_target_ms=300,
    )
    
    # Autoscaling for burst workloads
    config.autoscaling = AutoscalingConfig(
        enable_autoscaling=True,
        ttft_p90_target_s=2.0,
        tbt_p99_target_ms=150.0,  # Tighter target
        cost_aware=True,
        scale_up_threshold=0.8,
        scale_down_threshold=0.2,
        cooldown_period_s=180,
    )
    
    # Monitoring
    config.monitoring = MonitoringConfig(
        enable_monitoring=True,
        enable_tracing=True,
        enable_profiling=True,
        metrics_export_interval_s=5,
    )
    
    print("="*60)
    print("Custom Configuration for 8x Tesla V100")
    print("="*60)
    print(f"Storage: {config.storage.ssd_cache_size_gb}GB SSD + "
          f"{config.storage.dram_cache_size_gb}GB DRAM")
    print(f"Batch size: {config.models[0].max_batch_size}")
    print(f"KV quantization: {config.models[0].kv_quantization}")
    print("="*60 + "\n")
    
    # Initialize
    system = FaaSInfer(config)
    await system.initialize()
    
    # Test
    prompt = "The benefits of serverless computing include"
    print(f"Prompt: {prompt}\n")
    
    async for response in system.generate(
        prompt=prompt,
        model_id="opt-6.7b",
        max_new_tokens=100,
    ):
        print(response.text, end="", flush=True)
        if response.finish_reason:
            break
    
    print("\n")
    
    # Show stats
    status = await system.get_status()
    print("\nSystem Status:")
    print(f"  Model Manager Stats:")
    for key, value in status.get('model_manager', {}).items():
        print(f"    {key}: {value}")
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())