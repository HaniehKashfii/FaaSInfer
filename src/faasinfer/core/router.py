"""
C2: Request Router - Ray Actor for routing and migration coordination.

Responsibilities:
- Route incoming requests to workers
- Coordinate live migrations
- Maintain routing table
- Handle backpressure
"""

import ray
import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator
from collections import defaultdict
import time

from faasinfer.core.types import (
    InferenceRequest,
    InferenceResponse,
    PlacementDecision,
    RequestStatus,
)

logger = logging.getLogger(__name__)


@ray.remote
class RequestRouter:
    """
    Request router for deferred binding and migration coordination.
    Single logical router per cluster with HA via leader election.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        
        # Routing table: request_id -> worker_handle
        self.routing_table: Dict[str, ray.ObjectRef] = {}
        
        # Request queue (for backpressure)
        self.pending_requests: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Worker registry: server_id -> worker_handle
        self.workers: Dict[str, ray.ObjectRef] = {}
        
        # Model locations: model_id -> List[server_id]
        self.model_locations: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.total_requests = 0
        self.routed_requests = 0
        self.failed_requests = 0
        self.total_migrations = 0
        
        # Scheduler reference (will be set externally)
        self.scheduler = None
        
        logger.info("Initialized RequestRouter")
    
    def set_scheduler(self, scheduler):
        """Set reference to scheduler."""
        self.scheduler = scheduler
    
    def register_worker(
        self,
        server_id: str,
        worker_handle: ray.ObjectRef,
        model_id: str
    ):
        """
        Register a worker with loaded model.
        
        Args:
            server_id: Server identifier
            worker_handle: Ray actor handle
            model_id: Loaded model ID
        """
        self.workers[server_id] = worker_handle
        if server_id not in self.model_locations[model_id]:
            self.model_locations[model_id].append(server_id)
        
        logger.info(f"Registered worker {server_id} with model {model_id}")
    
    def unregister_worker(self, server_id: str, model_id: str):
        """Unregister a worker."""
        if server_id in self.workers:
            del self.workers[server_id]
        
        if server_id in self.model_locations[model_id]:
            self.model_locations[model_id].remove(server_id)
        
        logger.info(f"Unregistered worker {server_id}")
    
    async def route(self, request: InferenceRequest) -> PlacementDecision:
        """
        Route request to appropriate worker.
        Implements deferred binding (late routing).
        
        Args:
            request: Inference request
            
        Returns:
            Placement decision from scheduler
        """
        self.total_requests += 1
        
        # Check backpressure
        if self.pending_requests.qsize() >= self.max_queue_size:
            logger.warning("Request queue full, applying backpressure")
            raise RuntimeError("Router backpressure - queue full")
        
        try:
            # Get placement decision from scheduler
            if self.scheduler is None:
                # Fallback: simple round-robin
                servers = self.model_locations.get(request.model_id, [])
                if not servers:
                    raise RuntimeError(f"No servers available for model {request.model_id}")
                
                server_id = servers[self.routed_requests % len(servers)]
                placement = PlacementDecision(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    server_id=server_id,
                    gpu_ids=[0],
                    estimated_startup_time_s=0.0,
                    requires_loading=False,
                    requires_migration=False,
                )
            else:
                # Get optimized placement from scheduler
                placement = await self.scheduler.place.remote(
                    request.model_id,
                    request.tenant_id,
                    request.deadline_ms,
                )
            
            # Update routing table
            self.routing_table[request.request_id] = placement.server_id
            request.current_server = placement.server_id
            
            self.routed_requests += 1
            logger.info(f"Routed request {request.request_id} to {placement.server_id}")
            
            return placement
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Failed to route request {request.request_id}: {e}")
            raise
    
    async def update_route(
        self,
        request_id: str,
        new_server_id: str
    ) -> bool:
        """
        Update routing for migrated request.
        
        Args:
            request_id: Request being migrated
            new_server_id: Destination server
            
        Returns:
            Success status
        """
        try:
            if request_id not in self.routing_table:
                logger.warning(f"Request {request_id} not in routing table")
                return False
            
            old_server = self.routing_table[request_id]
            self.routing_table[request_id] = new_server_id
            
            logger.info(f"Updated route for {request_id}: {old_server} -> {new_server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update route: {e}")
            return False
    
    async def complete(
        self,
        request_id: str,
        migrated: bool = False
    ) -> bool:
        """
        Mark request as complete.
        
        Args:
            request_id: Completed request
            migrated: Whether request was migrated
            
        Returns:
            Success status
        """
        try:
            if request_id in self.routing_table:
                del self.routing_table[request_id]
            
            if migrated:
                self.total_migrations += 1
            
            logger.info(f"Completed request {request_id} (migrated: {migrated})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete request: {e}")
            return False
    
    async def stream_inference(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Stream inference results, handling migrations transparently.
        
        Args:
            request: Inference request
            
        Yields:
            Inference responses
        """
        try:
            # Get placement
            placement = await self.route(request)
            
            # Get worker
            worker = self.workers.get(placement.server_id)
            if worker is None:
                raise RuntimeError(f"Worker {placement.server_id} not found")
            
            # Stream from worker
            async for response in worker.submit.remote(request):
                # Check if migrated
                if response.migrated and response.migration_destination:
                    # Update routing
                    await self.update_route(
                        request.request_id,
                        response.migration_destination
                    )
                    
                    # Get new worker
                    worker = self.workers.get(response.migration_destination)
                    if worker is None:
                        raise RuntimeError("Migration destination worker not found")
                
                yield response
                
                # Check if complete
                if response.finish_reason:
                    await self.complete(
                        request.request_id,
                        migrated=response.migrated
                    )
                    break
            
        except Exception as e:
            logger.error(f"Stream inference failed: {e}")
            # Return error response
            yield InferenceResponse(
                request_id=request.request_id,
                finish_reason="error",
            )
    
    async def get_status(self) -> Dict:
        """Get router status."""
        return {
            "total_requests": self.total_requests,
            "routed_requests": self.routed_requests,
            "failed_requests": self.failed_requests,
            "active_requests": len(self.routing_table),
            "pending_requests": self.pending_requests.qsize(),
            "total_migrations": self.total_migrations,
            "num_workers": len(self.workers),
        }
    
    async def get_worker_for_request(self, request_id: str) -> Optional[ray.ObjectRef]:
        """Get worker handling a request."""
        server_id = self.routing_table.get(request_id)
        if server_id:
            return self.workers.get(server_id)
        return None
    
    async def get_model_locations(self, model_id: str) -> List[str]:
        """Get list of servers with model loaded."""
        return self.model_locations.get(model_id, [])