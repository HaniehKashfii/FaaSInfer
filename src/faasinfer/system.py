"""
Main FaaSInfer System - Orchestrates all components.

This is the main entry point for the FaaSInfer system.
"""

import ray
import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator
from pathlib import Path

from faasinfer.config import FaaSInferConfig, ModelConfig
from faasinfer.core.worker import InferenceWorker, WorkerConfig
from faasinfer.core.router import RequestRouter
from faasinfer.core.scheduler import StartupTimeScheduler
from faasinfer.core.model_manager import ModelManager
from faasinfer.core.migration import MigrationCoordinator
from faasinfer.core.types import (
    InferenceRequest,
    InferenceResponse,
    ServerInfo,
)

logger = logging.getLogger(__name__)


class FaaSInfer:
    """
    Main FaaSInfer system class.
    
    Coordinates all components:
    - C1: API Gateway (via gateway module)
    - C2: Request Router
    - C3: Scheduler
    - C5: Model Manager
    - C6: Inference Workers
    - C10: Migration Coordinator
    """
    
    def __init__(self, config: FaaSInferConfig):
        """
        Initialize FaaSInfer system.
        
        Args:
            config: System configuration
        """
        self.config = config
        config.validate()
        
        # Initialize Ray
        if config.ray_address is None:
            logger.info("Starting local Ray cluster")
            ray.init(
                namespace=config.ray_namespace,
                logging_level=logging.INFO,
            )
        else:
            logger.info(f"Connecting to Ray cluster at {config.ray_address}")
            ray.init(
                address=config.ray_address,
                namespace=config.ray_namespace,
            )
        
        # Core components (Ray actors)
        self.router: Optional[ray.ObjectRef] = None
        self.scheduler: Optional[ray.ObjectRef] = None
        self.migration_coordinator: Optional[ray.ObjectRef] = None
        
        # Model manager (local to each node)
        self.model_manager: Optional[ModelManager] = None
        
        # Workers: server_id -> worker_handle
        self.workers: Dict[str, ray.ObjectRef] = {}
        
        # Server registry
        self.servers: Dict[str, ServerInfo] = {}
        
        # System state
        self.initialized = False
        
        logger.info("FaaSInfer system created")
    
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        logger.info("Initializing FaaSInfer components...")
        
        try:
            # Initialize scheduler
            logger.info("Initializing scheduler...")
            self.scheduler = StartupTimeScheduler.remote(
                config=self.config.scheduler
            )
            
            # Initialize router
            logger.info("Initializing router...")
            self.router = RequestRouter.remote(
                max_queue_size=10000
            )
            
            # Connect router and scheduler
            await self.router.set_scheduler.remote(self.scheduler)
            
            # Initialize migration coordinator
            if self.config.migration.enable_migration:
                logger.info("Initializing migration coordinator...")
                self.migration_coordinator = MigrationCoordinator.remote(
                    max_concurrent_migrations=100,
                    max_migration_rounds=self.config.migration.max_migration_rounds,
                    target_pause_ms=self.config.scheduler.migration_pause_target_ms,
                )
                
                # Connect migration coordinator
                await self.migration_coordinator.set_router.remote(self.router)
                await self.migration_coordinator.set_scheduler.remote(self.scheduler)
            
            # Initialize model manager (local)
            logger.info("Initializing model manager...")
            self.model_manager = ModelManager(
                ssd_cache_path=self.config.storage.ssd_cache_path,
                dram_cache_path=self.config.storage.dram_cache_path,
                chunk_size_mb=self.config.storage.chunk_size_mb,
                num_io_threads=self.config.storage.num_io_threads,
            )
            
            if self.migration_coordinator:
                await self.migration_coordinator.set_model_manager.remote(
                    self.model_manager
                )
            
            # Register servers
            await self._discover_servers()
            
            # Pre-load models
            await self._preload_models()
            
            self.initialized = True
            logger.info("FaaSInfer initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def _discover_servers(self):
        """Discover available GPU servers in Ray cluster."""
        # Get available nodes
        nodes = ray.nodes()
        
        server_id = 0
        for node in nodes:
            if node['Alive'] and 'Resources' in node:
                resources = node['Resources']
                if 'GPU' in resources and resources['GPU'] > 0:
                    # Register server
                    num_gpus = int(resources['GPU'])
                    
                    server_info = ServerInfo(
                        server_id=f"server_{server_id}",
                        node_id=node['NodeID'],
                        num_gpus=num_gpus,
                        gpu_type="A100",  # Would detect actual GPU type
                        gpu_memory_gb=40.0,  # Would detect actual memory
                        dram_capacity_gb=512.0,
                        ssd_capacity_gb=2000.0,
                        network_bandwidth_gbps=10.0,
                        ssd_bandwidth_gbps=6.0,
                        dram_bandwidth_gbps=50.0,
                        pcie_bandwidth_gbps=32.0,
                        available_gpus=num_gpus,
                    )
                    
                    self.servers[server_info.server_id] = server_info
                    await self.scheduler.register_server.remote(server_info)
                    
                    logger.info(
                        f"Registered server {server_info.server_id} "
                        f"with {num_gpus} GPUs"
                    )
                    
                    server_id += 1
        
        if not self.servers:
            logger.warning("No GPU servers found in cluster")
    
    async def _preload_models(self):
        """Pre-load configured models."""
        for model_config in self.config.models:
            await self._load_model(model_config)
    
    async def _load_model(self, model_config: ModelConfig) -> bool:
        """
        Load a model and create workers.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Loading model {model_config.model_id}")
            
            # For each server, create a worker
            for server_id in list(self.servers.keys())[:1]:  # Start with first server
                # Create worker configuration
                worker_config = WorkerConfig(
                    worker_id=f"{server_id}_{model_config.model_id}",
                    gpu_ids=[0],  # Simplified
                    model_config=model_config,
                )
                
                # Create worker actor
                worker = InferenceWorker.remote(worker_config)
                
                # Load model
                # In real implementation, would use HuggingFace model hub
                model_path = f"facebook/{model_config.model_name}"
                
                success = await worker.load_model.remote(
                    model_path=model_path,
                    gpu_memory_pointers=None,
                )
                
                if not success:
                    logger.error(f"Failed to load model on {server_id}")
                    continue
                
                # Register worker
                self.workers[worker_config.worker_id] = worker
                await self.router.register_worker.remote(
                    server_id=server_id,
                    worker_handle=worker,
                    model_id=model_config.model_id,
                )
                
                # Update scheduler
                await self.scheduler.update_model_location.remote(
                    model_id=model_config.model_id,
                    server_id=server_id,
                    storage_tier="gpu",
                )
                
                logger.info(
                    f"Loaded {model_config.model_id} on {server_id}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_config.model_id}: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> AsyncIterator[InferenceResponse]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            model_id: Model to use (default: first model)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Yields:
            Inference responses with generated tokens
        """
        if not self.initialized:
            await self.initialize()
        
        # Use first model if not specified
        if model_id is None and self.config.models:
            model_id = self.config.models[0].model_id
        
        if model_id is None:
            raise ValueError("No models configured")
        
        # Create request
        request = InferenceRequest(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Stream from router
        async for response in self.router.stream_inference.remote(request):
            yield response
    
    async def get_status(self) -> Dict:
        """Get system status."""
        if not self.initialized:
            return {"initialized": False}
        
        router_status = await self.router.get_status.remote()
        scheduler_status = await self.scheduler.get_status.remote()
        
        migration_status = {}
        if self.migration_coordinator:
            migration_status = await self.migration_coordinator.get_stats.remote()
        
        model_manager_stats = {}
        if self.model_manager:
            model_manager_stats = self.model_manager.stats()
        
        return {
            "initialized": True,
            "num_servers": len(self.servers),
            "num_workers": len(self.workers),
            "router": router_status,
            "scheduler": scheduler_status,
            "migration": migration_status,
            "model_manager": model_manager_stats,
        }
    
    async def shutdown(self):
        """Shutdown the system."""
        logger.info("Shutting down FaaSInfer...")
        
        # Unload all models
        for worker in self.workers.values():
            try:
                await worker.unload_model.remote()
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
        
        # Shutdown Ray
        ray.shutdown()
        
        logger.info("FaaSInfer shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown())