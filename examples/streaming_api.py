"""
Streaming API example.

Demonstrates real-time token streaming for interactive applications.
"""

import asyncio
import time
from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.utils.logging import setup_logging


async def stream_with_timing(system: FaaSInfer, prompt: str):
    """Stream generation with timing information."""
    start_time = time.time()
    first_token_time = None
    token_times = []
    
    print(f"Prompt: {prompt}\n")
    print("Streaming response:\n")
    
    async for response in system.generate(
        prompt=prompt,
        model_id="opt-6.7b",
        max_new_tokens=100,
        temperature=0.8,
    ):
        current_time = time.time()
        
        if first_token_time is None:
            first_token_time = current_time
            ttft = (first_token_time - start_time) * 1000
            print(f"[TTFT: {ttft:.1f}ms]")
        else:
            token_times.append(current_time)
        
        print(response.text, end="", flush=True)
        
        if response.finish_reason:
            break
    
    # Print timing stats
    end_time = time.time()
    total_time = end_time - start_time
    
    if token_times:
        inter_token_times = [
            (token_times[i] - token_times[i-1]) * 1000
            for i in range(1, len(token_times))
        ]
        avg_tbt = sum(inter_token_times) / len(inter_token_times) if inter_token_times else 0
    else:
        avg_tbt = 0
    
    print(f"\n\n--- Timing Stats ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"TTFT: {(first_token_time - start_time)*1000:.1f}ms")
    print(f"Avg TBT: {avg_tbt:.1f}ms")
    print(f"Tokens/sec: {len(token_times)/total_time:.1f}")


async def main():
    """Run streaming example."""
    setup_logging(log_level="INFO")
    
    config = FaaSInferConfig()
    config.models = [
        ModelConfig(
            model_id="opt-6.7b",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            tensor_parallel_size=1,
        ),
    ]
    
    system = FaaSInfer(config)
    await system.initialize()
    
    # Test different prompts
    prompts = [
        "Once upon a time in a distant galaxy,",
        "The key to success in machine learning is",
        "In the year 2050, technology will",
    ]
    
    for prompt in prompts:
        print("\n" + "="*60)
        await stream_with_timing(system, prompt)
        print("="*60)
        await asyncio.sleep(1)
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())