"""
Multi-model serving example.

Demonstrates serving multiple models simultaneously with
different configurations for OPT, Llama, and Falcon.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def main():
    """Run multi-model server."""
    setup_logging(log_level="INFO")
    
    # Configure 3 models with 8 V100 GPUs
    config = FaaSInferConfig()
    config.models = [
        # OPT-6.7B - 2 replicas (GPUs 0-1)
        ModelConfig(
            model_id="opt-6.7b-1",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            tensor_parallel_size=1,
        ),
        ModelConfig(
            model_id="opt-6.7b-2",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            tensor_parallel_size=1,
        ),
        # Llama-2-13B (GPUs 2-3)
        ModelConfig(
            model_id="llama-2-13b",
            model_name="meta-llama/Llama-2-13b-hf",
            model_size_gb=26.0,
            num_parameters_b=13.0,
            tensor_parallel_size=2,
        ),
        # Falcon-30B (GPUs 4-7)
        ModelConfig(
            model_id="falcon-30b",
            model_name="tiiuae/falcon-30b",
            model_size_gb=60.0,
            num_parameters_b=30.0,
            tensor_parallel_size=4,
        ),
    ]
    
    print("Initializing multi-model system...")
    system = FaaSInfer(config)
    await system.initialize()
    
    print("\n" + "="*60)
    print("Multi-Model Server Ready")
    print("="*60)
    
    # Test each model
    test_prompts = {
        "opt-6.7b-1": "The future of AI is",
        "llama-2-13b": "Explain machine learning in simple terms:",
        "falcon-30b": "Write a short poem about technology:",
    }
    
    for model_id, prompt in test_prompts.items():
        print(f"\n[{model_id}] {prompt}")
        print("-" * 40)
        
        async for response in system.generate(
            prompt=prompt,
            model_id=model_id,
            max_new_tokens=50,
        ):
            print(response.text, end="", flush=True)
            if response.finish_reason:
                break
        print("\n")
    
    # Get status
    status = await system.get_status()
    print("\nSystem Status:")
    print(f"  Servers: {status['num_servers']}")
    print(f"  Workers: {status['num_workers']}")
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())