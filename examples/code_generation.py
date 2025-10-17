"""
Code generation example using FaaSInfer.

Demonstrates using LLMs for code completion and generation.
"""

import asyncio
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def main():
    """Run code generation example."""
    setup_logging(log_level="INFO")
    
    config = FaaSInferConfig()
    config.models = [
        ModelConfig(
            model_id="llama-2-13b",
            model_name="meta-llama/Llama-2-13b-hf",
            model_size_gb=26.0,
            num_parameters_b=13.0,
            tensor_parallel_size=2,
        ),
    ]
    
    system = FaaSInfer(config)
    await system.initialize()
    
    # Code generation tasks
    tasks = [
        "Write a Python function to sort a list using quicksort:",
        "Create a React component for a login form:",
        "Implement a binary search tree in C++:",
    ]
    
    for prompt in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {prompt}")
        print('='*60)
        print("\nGenerated Code:\n")
        
        async for response in system.generate(
            prompt=prompt,
            model_id="llama-2-13b",
            max_new_tokens=300,
            temperature=0.3,  # Lower temp for code
        ):
            print(response.text, end="", flush=True)
            if response.finish_reason:
                break
        print("\n")
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())