"""
C6: Inference Worker - Ray Actor running vLLM for model inference.

Responsibilities:
- Load model shards onto GPUs
- Execute inference with batching
- Stream tokens back to router
- Support cancellation and migration
"""

import ray
import torch
import asyncio
import logging
from typing import List, Dict, Optional, AsyncIterator
from dataclasses import dataclass

from faasinfer.core.types import (
    InferenceRequest,
    InferenceResponse,
    ModelStatus,
)
from faasinfer.config import ModelConfig

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available, using mock inference")


logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for inference worker."""
    worker_id: str
    gpu_ids: List[int]
    model_config: ModelConfig
    max_concurrent_requests: int = 32
    enable_streaming: bool = True


@ray.remote(num_gpus=1)
class InferenceWorker:
    """
    Ray actor for LLM inference using vLLM.
    Handles model loading, batching, and streaming.
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config.worker_id
        self.model_config = config.model_config
        
        # Model state
        self.model_status = ModelStatus.UNLOADED
        self.engine: Optional[AsyncLLMEngine] = None
        
        # Request tracking
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.request_queues: Dict[str, asyncio.Queue] = {}
        
        # Statistics
        self.total_requests = 0
        self.completed_requests = 0
        self.total_tokens_generated = 0
        
        logger.info(f"Initialized InferenceWorker {self.worker_id}")
    
    async def load_model(
        self,
        model_path: str,
        gpu_memory_pointers: Optional[Dict[int, int]] = None
    ) -> bool:
        """
        Load model onto GPUs.
        
        Args:
            model_path: Path to model checkpoint
            gpu_memory_pointers: Pre-loaded GPU memory addresses (from Model Manager)
        
        Returns:
            Success status
        """
        try:
            self.model_status = ModelStatus.LOADING
            logger.info(f"Loading model {self.model_config.model_id} from {model_path}")
            
            if not VLLM_AVAILABLE:
                logger.warning("Using mock model loading")
                await asyncio.sleep(0.1)  # Simulate loading
                self.model_status = ModelStatus.LOADED
                return True
            
            # Configure vLLM engine
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                dtype=self.model_config.dtype,
                max_model_len=self.model_config.max_context_length,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                # Use paged attention (FR 007)
                enable_prefix_caching=True,
            )
            
            # Initialize async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.model_status = ModelStatus.LOADED
            logger.info(f"Successfully loaded model {self.model_config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_status = ModelStatus.UNLOADED
            return False
    
    async def unload_model(self) -> bool:
        """Unload model from GPU."""
        try:
            self.model_status = ModelStatus.EVICTING
            
            if self.engine:
                # Clean up vLLM engine
                del self.engine
                self.engine = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model_status = ModelStatus.UNLOADED
            logger.info(f"Unloaded model {self.model_config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False
    
    async def submit(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Submit inference request and stream tokens.
        
        Args:
            request: Inference request
            
        Yields:
            Inference responses with generated tokens
        """
        if self.model_status != ModelStatus.LOADED:
            raise RuntimeError(f"Model not loaded, status: {self.model_status}")
        
        self.total_requests += 1
        self.active_requests[request.request_id] = request
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_new_tokens,
            )
            
            if not VLLM_AVAILABLE:
                # Mock streaming response
                for i in range(min(10, request.max_new_tokens)):
                    await asyncio.sleep(0.01)
                    yield InferenceResponse(
                        request_id=request.request_id,
                        tokens=[i],
                        text=f"token_{i} ",
                        num_prompt_tokens=len(request.tokens),
                        num_generated_tokens=i+1,
                    )
                return
            
            # Stream tokens from vLLM
            request_id = request.request_id
            async for output in self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                # Extract generated tokens
                generated_text = output.outputs[0].text
                generated_tokens = output.outputs[0].token_ids
                finish_reason = output.outputs[0].finish_reason
                
                response = InferenceResponse(
                    request_id=request_id,
                    tokens=generated_tokens,
                    text=generated_text,
                    finish_reason=finish_reason,
                    num_prompt_tokens=len(request.tokens),
                    num_generated_tokens=len(generated_tokens),
                )
                
                self.total_tokens_generated += 1
                yield response
            
            self.completed_requests += 1
            
        except Exception as e:
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            yield InferenceResponse(
                request_id=request.request_id,
                finish_reason="error",
            )
        finally:
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def cancel(self, request_id: str) -> bool:
        """
        Cancel an ongoing inference request.
        
        Args:
            request_id: Request to cancel
            
        Returns:
            Success status
        """
        try:
            if request_id not in self.active_requests:
                return False
            
            if self.engine:
                await self.engine.abort(request_id)
            
            del self.active_requests[request_id]
            logger.info(f"Cancelled request {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel request {request_id}: {e}")
            return False
    
    async def migrate_out(self, request_id: str) -> Optional[Dict]:
        """
        Prepare request for migration (FR 005: token-only migration).
        
        Args:
            request_id: Request to migrate
            
        Returns:
            Migration state (tokens generated so far)
        """
        if request_id not in self.active_requests:
            return None
        
        try:
            request = self.active_requests[request_id]
            
            # Get current generated tokens
            # In real implementation, extract from vLLM engine state
            migration_state = {
                "request_id": request_id,
                "input_tokens": request.tokens,
                "generated_tokens": [],  # Would extract from engine
                "model_id": self.model_config.model_id,
            }
            
            # Cancel the request
            await self.cancel(request_id)
            
            logger.info(f"Migrating out request {request_id}")
            return migration_state
            
        except Exception as e:
            logger.error(f"Failed to migrate out request {request_id}: {e}")
            return None
    
    async def resume(
        self,
        migration_state: Dict
    ) -> AsyncIterator[InferenceResponse]:
        """
        Resume inference from migration state.
        Recomputes KV cache from tokens (FR 005).
        
        Args:
            migration_state: State from source worker
            
        Yields:
            Inference responses continuing from migration point
        """
        request_id = migration_state["request_id"]
        input_tokens = migration_state["input_tokens"]
        generated_tokens = migration_state["generated_tokens"]
        
        # Reconstruct request
        request = InferenceRequest(
            request_id=request_id,
            model_id=migration_state["model_id"],
            tokens=input_tokens + generated_tokens,
        )
        
        logger.info(f"Resuming request {request_id} after migration")
        
        # Continue inference (vLLM will recompute KV cache)
        async for response in self.submit(request):
            response.migrated = True
            yield response
    
    async def get_status(self) -> Dict:
        """Get worker status."""
        return {
            "worker_id": self.worker_id,
            "model_id": self.model_config.model_id,
            "model_status": self.model_status.value,
            "active_requests": len(self.active_requests),
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_tokens_generated": self.total_tokens_generated,
        }
    
    async def get_stats(self) -> Dict:
        """Get detailed statistics."""
        return {
            "worker_id": self.worker_id,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "active_requests": len(self.active_requests),
            "total_tokens_generated": self.total_tokens_generated,
        }