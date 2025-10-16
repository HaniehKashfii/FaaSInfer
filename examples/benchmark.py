"""
FaaSInfer Benchmark Script

Benchmarks FaaSInfer with different workloads:
- Azure serverless trace
- Bursty traffic (CV=8 Gamma)
- RPS scaling tests

Tests models: OPT-6.7B, LLaMA-2-13B, Falcon-30B
Datasets: GSM8K, ShareGPT, ContextLength
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, AsyncIterator
from dataclasses import dataclass
import json

from faasinfer import FaaSInfer, FaaSInferConfig, ModelConfig
from faasinfer.config import BatchingPolicy
from faasinfer.core.types import InferenceRequest
from faasinfer.utils.logging import setup_logging


@dataclass
class WorkloadConfig:
    """Workload configuration."""
    name: str
    rps: float
    duration_s: int
    arrival_pattern: str  # "poisson", "gamma", "trace"
    cv: float = 1.0  # Coefficient of variation


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    workload_name: str
    model_id: str
    dataset: str
    
    # Request statistics
    total_requests: int
    completed_requests: int
    failed_requests: int
    timeout_requests: int
    
    # Latency metrics (ms)
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    
    tbt_mean: float
    tbt_p50: float
    tbt_p99: float
    
    total_latency_mean: float
    total_latency_p50: float
    total_latency_p99: float
    
    # Throughput
    requests_per_second: float
    tokens_per_second: float
    
    # Migration statistics
    total_migrations: int
    migration_pause_mean_ms: float
    
    # Cache statistics
    cache_hit_rate: float


class WorkloadGenerator:
    """Generates workload patterns."""
    
    def __init__(self, config: WorkloadConfig):
        self.config = config
    
    async def generate_arrivals(self) -> AsyncIterator[float]:
        """
        Generate arrival times based on pattern.
        
        Yields:
            Arrival time (seconds from start)
        """
        if self.config.arrival_pattern == "poisson":
            # Poisson arrivals (exponential inter-arrival times)
            current_time = 0.0
            while current_time < self.config.duration_s:
                inter_arrival = random.expovariate(self.config.rps)
                current_time += inter_arrival
                if current_time < self.config.duration_s:
                    yield current_time
        
        elif self.config.arrival_pattern == "gamma":
            # Gamma distribution for bursty traffic (CV=8)
            # Shape parameter k = 1/CV^2, scale = CV^2 / lambda
            k = 1.0 / (self.config.cv ** 2)
            theta = (self.config.cv ** 2) / self.config.rps
            
            current_time = 0.0
            while current_time < self.config.duration_s:
                inter_arrival = np.random.gamma(k, theta)
                current_time += inter_arrival
                if current_time < self.config.duration_s:
                    yield current_time
        
        elif self.config.arrival_pattern == "trace":
            # Load from Azure trace (simplified)
            # In real implementation, would load actual trace data
            current_time = 0.0
            while current_time < self.config.duration_s:
                inter_arrival = random.expovariate(self.config.rps)
                current_time += inter_arrival
                if current_time < self.config.duration_s:
                    yield current_time


class DatasetLoader:
    """Loads datasets for benchmarking."""
    
    @staticmethod
    def load_gsm8k(num_samples: int = 100) -> List[str]:
        """Load GSM8K dataset."""
        # Mock implementation
        # In real system, would use datasets.load_dataset("gsm8k")
        prompts = [
            "What is 15 + 27?",
            "If John has 5 apples and buys 3 more, how many does he have?",
            "Calculate 123 * 45.",
            "A store sells pencils for $0.25 each. How much do 8 pencils cost?",
            "Mary has 3 times as many books as John. If John has 4 books, how many does Mary have?",
        ] * (num_samples // 5 + 1)
        return prompts[:num_samples]
    
    @staticmethod
    def load_sharegpt(num_samples: int = 100) -> List[str]:
        """Load ShareGPT dataset."""
        # Mock implementation
        prompts = [
            "Tell me a story about a brave knight.",
            "Explain quantum computing in simple terms.",
            "Write a poem about the ocean.",
            "What are the benefits of meditation?",
            "How do I learn to code?",
        ] * (num_samples // 5 + 1)
        return prompts[:num_samples]
    
    @staticmethod
    def load_long_context(num_samples: int = 100, context_length: int = 2048) -> List[str]:
        """Generate long context prompts."""
        # Generate prompts with specified context length
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        return [base_text[:context_length]] * num_samples


async def run_benchmark(
    system: FaaSInfer,
    workload: WorkloadConfig,
    model_id: str,
    dataset: str = "gsm8k",
) -> BenchmarkResult:
    """
    Run benchmark with specified workload.
    
    Args:
        system: FaaSInfer system
        workload: Workload configuration
        model_id: Model to test
        dataset: Dataset to use
        
    Returns:
        Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {workload.name}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset}")
    print(f"RPS: {workload.rps}, Duration: {workload.duration_s}s")
    print(f"{'='*60}\n")
    
    # Load dataset
    loader = DatasetLoader()
    if dataset == "gsm8k":
        prompts = loader.load_gsm8k(1000)
    elif dataset == "sharegpt":
        prompts = loader.load_sharegpt(1000)
    else:
        prompts = loader.load_long_context(1000)
    
    # Metrics collection
    results = []
    start_time = time.time()
    
    # Generate arrivals
    generator = WorkloadGenerator(workload)
    arrival_times = []
    
    async for arrival_time in generator.generate_arrivals():
        arrival_times.append(arrival_time)
    
    print(f"Generated {len(arrival_times)} arrivals")
    
    # Submit requests
    tasks = []
    for i, arrival_time in enumerate(arrival_times):
        # Wait until arrival time
        elapsed = time.time() - start_time
        wait_time = arrival_time - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        # Select prompt
        prompt = prompts[i % len(prompts)]
        
        # Submit request
        task = asyncio.create_task(
            process_request(system, model_id, prompt, i)
        )
        tasks.append(task)
        
        if (i + 1) % 10 == 0:
            print(f"Submitted {i+1}/{len(arrival_times)} requests")
    
    # Wait for completion
    print("\nWaiting for all requests to complete...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    valid_results = [r for r in results if isinstance(r, dict)]
    
    print(f"\nCompleted {len(valid_results)}/{len(arrival_times)} requests")
    
    # Calculate statistics
    if valid_results:
        ttfts = [r['ttft_ms'] for r in valid_results if 'ttft_ms' in r]
        tbts = [r['tbt_ms'] for r in valid_results if 'tbt_ms' in r]
        latencies = [r['total_latency_ms'] for r in valid_results if 'total_latency_ms' in r]
        
        ttfts_sorted = sorted(ttfts)
        tbts_sorted = sorted(tbts)
        latencies_sorted = sorted(latencies)
        
        benchmark_result = BenchmarkResult(
            workload_name=workload.name,
            model_id=model_id,
            dataset=dataset,
            total_requests=len(arrival_times),
            completed_requests=len(valid_results),
            failed_requests=len([r for r in results if isinstance(r, Exception)]),
            timeout_requests=0,
            ttft_mean=np.mean(ttfts) if ttfts else 0,
            ttft_p50=ttfts_sorted[int(len(ttfts_sorted)*0.5)] if ttfts_sorted else 0,
            ttft_p90=ttfts_sorted[int(len(ttfts_sorted)*0.9)] if ttfts_sorted else 0,
            ttft_p99=ttfts_sorted[int(len(ttfts_sorted)*0.99)] if ttfts_sorted else 0,
            tbt_mean=np.mean(tbts) if tbts else 0,
            tbt_p50=tbts_sorted[int(len(tbts_sorted)*0.5)] if tbts_sorted else 0,
            tbt_p99=tbts_sorted[int(len(tbts_sorted)*0.99)] if tbts_sorted else 0,
            total_latency_mean=np.mean(latencies) if latencies else 0,
            total_latency_p50=latencies_sorted[int(len(latencies_sorted)*0.5)] if latencies_sorted else 0,
            total_latency_p99=latencies_sorted[int(len(latencies_sorted)*0.99)] if latencies_sorted else 0,
            requests_per_second=len(valid_results) / workload.duration_s,
            tokens_per_second=sum(r.get('tokens', 0) for r in valid_results) / workload.duration_s,
            total_migrations=0,  # Would get from system
            migration_pause_mean_ms=0.0,
            cache_hit_rate=0.0,
        )
    else:
        benchmark_result = BenchmarkResult(
            workload_name=workload.name,
            model_id=model_id,
            dataset=dataset,
            total_requests=len(arrival_times),
            completed_requests=0,
            failed_requests=len(results),
            timeout_requests=0,
            ttft_mean=0, ttft_p50=0, ttft_p90=0, ttft_p99=0,
            tbt_mean=0, tbt_p50=0, tbt_p99=0,
            total_latency_mean=0, total_latency_p50=0, total_latency_p99=0,
            requests_per_second=0,
            tokens_per_second=0,
            total_migrations=0,
            migration_pause_mean_ms=0,
            cache_hit_rate=0,
        )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Completed: {benchmark_result.completed_requests}/{benchmark_result.total_requests}")
    print(f"\nTTFT (ms):")
    print(f"  Mean: {benchmark_result.ttft_mean:.1f}")
    print(f"  P50:  {benchmark_result.ttft_p50:.1f}")
    print(f"  P90:  {benchmark_result.ttft_p90:.1f}")
    print(f"  P99:  {benchmark_result.ttft_p99:.1f}")
    print(f"\nTBT (ms):")
    print(f"  Mean: {benchmark_result.tbt_mean:.1f}")
    print(f"  P50:  {benchmark_result.tbt_p50:.1f}")
    print(f"  P99:  {benchmark_result.tbt_p99:.1f}")
    print(f"\nThroughput:")
    print(f"  RPS: {benchmark_result.requests_per_second:.2f}")
    print(f"  Tokens/s: {benchmark_result.tokens_per_second:.1f}")
    print(f"{'='*60}\n")
    
    return benchmark_result


async def process_request(
    system: FaaSInfer,
    model_id: str,
    prompt: str,
    request_idx: int,
) -> Dict:
    """Process single request and collect metrics."""
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    
    try:
        async for response in system.generate(
            prompt=prompt,
            model_id=model_id,
            max_new_tokens=100,
            temperature=0.8,
        ):
            if first_token_time is None:
                first_token_time = time.time()
            
            total_tokens = response.num_generated_tokens
            
            if response.finish_reason:
                break
        
        end_time = time.time()
        
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        total_latency_ms = (end_time - start_time) * 1000
        tbt_ms = ((end_time - first_token_time) / max(total_tokens - 1, 1)) * 1000 if first_token_time and total_tokens > 1 else 0
        
        return {
            "request_idx": request_idx,
            "ttft_ms": ttft_ms,
            "tbt_ms": tbt_ms,
            "total_latency_ms": total_latency_ms,
            "tokens": total_tokens,
            "success": True,
        }
    
    except Exception as e:
        return {
            "request_idx": request_idx,
            "success": False,
            "error": str(e),
        }


async def main():
    """Main benchmark function."""
    setup_logging(log_level="INFO")
    
    # Create configuration
    config = FaaSInferConfig()
    config.models = [
        ModelConfig(
            model_id="opt-6.7b",
            model_name="facebook/opt-6.7b",
            model_size_gb=13.0,
            num_parameters_b=6.7,
            batching_policy=BatchingPolicy.VLLM_CONTINUOUS,
        ),
    ]
    
    # Initialize system
    print("Initializing FaaSInfer...")
    system = FaaSInfer(config)
    await system.initialize()
    print("✓ System ready\n")
    
    # Define workloads
    workloads = [
        WorkloadConfig(
            name="Low RPS",
            rps=0.2,
            duration_s=60,
            arrival_pattern="gamma",
            cv=8.0,
        ),
        WorkloadConfig(
            name="Medium RPS",
            rps=0.8,
            duration_s=60,
            arrival_pattern="gamma",
            cv=8.0,
        ),
        WorkloadConfig(
            name="High RPS",
            rps=1.4,
            duration_s=60,
            arrival_pattern="gamma",
            cv=8.0,
        ),
    ]
    
    # Run benchmarks
    all_results = []
    
    for workload in workloads:
        result = await run_benchmark(
            system=system,
            workload=workload,
            model_id="opt-6.7b",
            dataset="gsm8k",
        )
        all_results.append(result)
    
    # Save results
    results_file = f"benchmark_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump([vars(r) for r in all_results], f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Shutdown
    await system.shutdown()
    print("✓ Benchmark complete")


if __name__ == "__main__":
    asyncio.run(main())
```

## **24. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Logs
*.log
logs/

# FaaSInfer specific
/var/faasinfer/
/tmp/faasinfer/
*.ckpt
*.bin
*.safetensors
benchmark_results_*.json

# Ray
/tmp/ray/