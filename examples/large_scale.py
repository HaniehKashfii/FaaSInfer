"""
Large-scale deployment example for 70B models.

Demonstrates deploying Falcon-40B across multiple GPUs.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def main():
    """Deploy large model."""
    setup_logging(log_level="INFO")
    
    config = FaaSInferConfig()
    
    # Falcon-40B requires 8 V100s with tensor parallelism
    config.models = [
        ModelConfig(
            model_id="falcon-40b",
            model_name="tiiuae/falcon-40b",
            model_size_gb=80.0,
            num_parameters_b=40.0,
            tensor_parallel_size=8,  # Use all 8 GPUs
            max_context_length=2048,
            batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
            use_paged_kv=True,
        ),
    ]
    
    # Increase cache sizes for large model
    config.storage.ssd_cache_size_gb = 1000.0
    config.storage.dram_cache_size_gb = 256.0
    
    print("Deploying Falcon-40B on 8x V100...")
    system = FaaSInfer(config)
    await system.initialize()
    
    # Test generation
    prompt = "Explain the concept of artificial general intelligence:"
    print(f"\nPrompt: {prompt}\n")
    print("Response: ", end="", flush=True)
    
    async for response in system.generate(
        prompt=prompt,
        model_id="falcon-40b",
        max_new_tokens=200,
        temperature=0.7,
    ):
        print(response.text, end="", flush=True)
        if response.finish_reason:
            break
    
    print("\n")
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())