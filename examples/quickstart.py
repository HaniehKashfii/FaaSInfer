"""
FaaSInfer Quickstart Example

This example demonstrates basic usage of FaaSInfer for LLM inference.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def main():
    """Main function demonstrating FaaSInfer usage."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create configuration
    config = FaaSInferConfig()
    
    # Configure models to serve
    # These models will be loaded from HuggingFace
    config.models = [
        ModelConfig(
            model_id="opt-6.7b",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            tensor_parallel_size=1,
            max_context_length=2048,
            batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
            use_paged_kv=True,
        ),
    ]
    
    # Configure storage
    config.storage.ssd_cache_path = "/tmp/faasinfer/ssd_cache"
    config.storage.dram_cache_path = "/tmp/faasinfer/dram_cache"
    config.storage.ssd_cache_size_gb = 100.0
    config.storage.dram_cache_size_gb = 20.0
    
    # Configure scheduler
    config.scheduler.locality_aware = True
    config.scheduler.enable_live_migration = True
    
    # Configure autoscaling
    config.autoscaling.enable_autoscaling = False  # Disabled for local testing
    
    print("=" * 60)
    print("FaaSInfer Quickstart Example")
    print("=" * 60)
    
    # Initialize FaaSInfer system
    print("\n[1/4] Initializing FaaSInfer system...")
    system = FaaSInfer(config)
    await system.initialize()
    print("✓ System initialized")
    
    # Check status
    print("\n[2/4] Getting system status...")
    status = await system.get_status()
    print(f"✓ Status: {status['num_workers']} workers, {status['num_servers']} servers")
    
    # Generate text
    print("\n[3/4] Generating text...")
    prompt = "Once upon a time in a land far away,"
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    full_text = ""
    async for response in system.generate(
        prompt=prompt,
        model_id="opt-6.7b",
        max_new_tokens=50,
        temperature=0.8,
    ):
        print(response.text, end="", flush=True)
        full_text = response.text
        
        if response.finish_reason:
            break
    
    print("\n\n✓ Generation complete")
    
    # Shutdown
    print("\n[4/4] Shutting down...")
    await system.shutdown()
    print("✓ Shutdown complete")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())