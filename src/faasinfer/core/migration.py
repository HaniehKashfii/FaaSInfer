"""
C10: Live Migration Coordinator

Responsibilities:
- Coordinate token-only migration (FR 005)
- Multi-round migration process
- Failure handling (FR 018)
"""

import ray
import asyncio
import logging
from typing import Dict, Optional
import time

from faasinfer.core.types import MigrationTask, RequestStatus

logger = logging.getLogger(__name__)


@ray.remote
class MigrationCoordinator:
    """
    Coordinates live migration of LLM inference.
    Implements multi-round token-only migration (FR 005).
    """
    
    def __init__(
        self,
        max_concurrent_migrations: int = 100,
        max_migration_rounds: int = 3,
        target_pause_ms: int = 300,
    ):
        self.max_concurrent_migrations = max_concurrent_migrations
        self.max_migration_rounds = max_migration_rounds
        self.target_pause_ms = target_pause_ms
        
        # Active migrations: migration_id -> MigrationTask
        self.active_migrations: Dict[str, MigrationTask] = {}
        
        # Statistics
        self.total_migrations = 0
        self.successful_migrations = 0
        self.failed_migrations = 0
        self.total_pause_time_ms = 0.0
        
        # References to other components (set externally)
        self.router = None
        self.scheduler = None
        self.model_manager = None
        
        logger.info("Initialized MigrationCoordinator")
    
    def set_router(self, router):
        """Set router reference."""
        self.router = router
    
    def set_scheduler(self, scheduler):
        """Set scheduler reference."""
        self.scheduler = scheduler
    
    def set_model_manager(self, model_manager):
        """Set model manager reference."""
        self.model_manager = model_manager
    
    async def migrate(
        self,
        request_id: str,
        model_id: str,
        source_server: str,
        dest_server: str,
    ) -> bool:
        """
        Execute live migration.
        
        Implements multi-round token-only migration:
        1. Preload model at destination
        2. Send tokens and recompute KV
        3. Repeat until gap is small
        4. Stop source and flip route
        
        Args:
            request_id: Request to migrate
            model_id: Model ID
            source_server: Source server
            dest_server: Destination server
            
        Returns:
            Success status
        """
        self.total_migrations += 1
        
        if len(self.active_migrations) >= self.max_concurrent_migrations:
            logger.warning("Too many concurrent migrations")
            return False
        
        # Create migration task
        task = MigrationTask(
            request_id=request_id,
            source_server=source_server,
            dest_server=dest_server,
            model_id=model_id,
            max_rounds=self.max_migration_rounds,
        )
        
        self.active_migrations[task.migration_id] = task
        
        try:
            # Step 1: Ensure model loaded at destination
            logger.info(f"Migration {task.migration_id}: Preloading model at {dest_server}")
            
            # Get destination worker
            if self.router is None:
                raise RuntimeError("Router not set")
            
            dest_worker = await self.router.get_worker_for_request.remote(request_id)
            if dest_worker is None:
                # Need to load model
                if self.model_manager:
                    await self.model_manager.load(
                        model_id=model_id,
                        model_partitions=[],  # Would be filled in
                        gpu_ids=[0],
                    )
            
            # Step 2-4: Multi-round migration
            migration_start = time.time()
            
            for round_num in range(self.max_migration_rounds):
                task.current_round = round_num + 1
                
                logger.info(
                    f"Migration {task.migration_id}: Round {task.current_round}/{self.max_migration_rounds}"
                )
                
                # Get source worker
                source_worker = await self.router.get_worker_for_request.remote(request_id)
                if source_worker is None:
                    raise RuntimeError("Source worker not found")
                
                # Get intermediate tokens from source
                migration_state = await source_worker.migrate_out.remote(request_id)
                
                if migration_state is None:
                    # Inference completed during migration
                    logger.info(f"Migration {task.migration_id}: Inference completed at source")
                    await self.abort(task.migration_id)
                    return True
                
                task.input_tokens = migration_state["input_tokens"]
                task.generated_tokens = migration_state["generated_tokens"]
                
                # Send to destination for resume
                dest_worker = await self.router.get_worker_for_request.remote(request_id)
                
                # Start resume at destination
                resume_task = dest_worker.resume.remote(migration_state)
                
                # Check gap - if small enough, complete migration
                total_tokens = len(task.input_tokens) + len(task.generated_tokens)
                if total_tokens < 100 or round_num == self.max_migration_rounds - 1:
                    # Gap is small enough, complete migration
                    break
                
                # Wait for resume to catch up
                await asyncio.sleep(0.1)
            
            # Step 5: Flip route to destination
            pause_start = time.time()
            
            # Stop source
            await source_worker.cancel.remote(request_id)
            
            # Update route
            await self.router.update_route.remote(request_id, dest_server)
            
            pause_time_ms = (time.time() - pause_start) * 1000
            task.pause_time_ms = pause_time_ms
            
            # Complete migration
            migration_time = time.time() - migration_start
            
            self.successful_migrations += 1
            self.total_pause_time_ms += pause_time_ms
            
            logger.info(
                f"Migration {task.migration_id} completed: "
                f"total_time={migration_time:.2f}s, "
                f"pause_time={pause_time_ms:.1f}ms, "
                f"rounds={task.current_round}"
            )
            
            # Check pause time target (FR 005: ≤300ms median)
            if pause_time_ms > self.target_pause_ms:
                logger.warning(
                    f"Migration pause time {pause_time_ms:.1f}ms "
                    f"exceeded target {self.target_pause_ms}ms"
                )
            
            del self.active_migrations[task.migration_id]
            return True
            
        except Exception as e:
            logger.error(f"Migration {task.migration_id} failed: {e}")
            task.failed = True
            task.failure_reason = str(e)
            self.failed_migrations += 1
            
            # Clean up
            await self.abort(task.migration_id)
            return False
    
    async def abort(self, migration_id: str) -> bool:
        """
        Abort migration.
        
        Args:
            migration_id: Migration to abort
            
        Returns:
            Success status
        """
        if migration_id not in self.active_migrations:
            return False
        
        task = self.active_migrations[migration_id]
        
        try:
            # Clean up destination
            if self.router:
                dest_worker = await self.router.get_worker_for_request.remote(
                    task.request_id
                )
                if dest_worker:
                    await dest_worker.cancel.remote(task.request_id)
            
            # Remove from active migrations
            del self.active_migrations[migration_id]
            
            logger.info(f"Aborted migration {migration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to abort migration {migration_id}: {e}")
            return False
    
    async def status(self, migration_id: str) -> Optional[Dict]:
        """Get status of migration."""
        if migration_id not in self.active_migrations:
            return None
        
        task = self.active_migrations[migration_id]
        return {
            "migration_id": migration_id,
            "request_id": task.request_id,
            "source": task.source_server,
            "destination": task.dest_server,
            "current_round": task.current_round,
            "max_rounds": task.max_rounds,
            "status": task.status,
            "pause_time_ms": task.pause_time_ms,
        }
    
    async def get_stats(self) -> Dict:
        """Get migration statistics."""
        avg_pause_time = (
            self.total_pause_time_ms / self.successful_migrations
            if self.successful_migrations > 0
            else 0.0
        )
        
        return {
            "total_migrations": self.total_migrations,
            "successful_migrations": self.successful_migrations,
            "failed_migrations": self.failed_migrations,
            "active_migrations": len(self.active_migrations),
            "avg_pause_time_ms": avg_pause_time,
            "success_rate": (
                self.successful_migrations / self.total_migrations
                if self.total_migrations > 0
                else 0.0
            ),
        }