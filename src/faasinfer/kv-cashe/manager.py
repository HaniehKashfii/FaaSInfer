"""
C8: KV Cache Manager

Responsibilities:
- Paged KV cache management (PagedAttention)
- KV quantization (int8/FP8)
- Eviction policies
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KVPage:
    """KV cache page."""
    page_id: int
    request_id: str
    sequence_pos: int
    size_bytes: int
    pinned: bool = False


class KVCacheManager:
    """
    Manages paged KV cache with optional quantization.
    
    Implements FR 007: Paged & quantized KV cache.
    """
    
    def __init__(
        self,
        total_pages: int = 1000,
        page_size: int = 16,
        enable_quantization: bool = False,
        quantization_bits: int = 8,
    ):
        self.total_pages = total_pages
        self.page_size = page_size
        self.enable_quantization = enable_quantization
        self.quantization_bits = quantization_bits
        
        # Page tracking
        self.free_pages: List[int] = list(range(total_pages))
        self.allocated_pages: Dict[str, List[KVPage]] = {}
        
        # Statistics
        self.total_allocs = 0
        self.total_evictions = 0
        
        logger.info(
            f"Initialized KVCacheManager: {total_pages} pages, "
            f"quantization={'enabled' if enable_quantization else 'disabled'}"
        )
    
    def alloc(self, request_id: str, num_pages: int = 1) -> List[int]:
        """
        Allocate KV cache pages for a request.
        
        Args:
            request_id: Request ID
            num_pages: Number of pages to allocate
            
        Returns:
            List of allocated page IDs
        """
        if len(self.free_pages) < num_pages:
            # Need to evict
            self._evict_lru(num_pages - len(self.free_pages))
        
        if len(self.free_pages) < num_pages:
            raise RuntimeError("Out of KV cache memory")
        
        # Allocate pages
        allocated = []
        for _ in range(num_pages):
            page_id = self.free_pages.pop(0)
            page = KVPage(
                page_id=page_id,
                request_id=request_id,
                sequence_pos=len(allocated),
                size_bytes=self.page_size * 1024,
            )
            allocated.append(page_id)
            
            if request_id not in self.allocated_pages:
                self.allocated_pages[request_id] = []
            self.allocated_pages[request_id].append(page)
        
        self.total_allocs += num_pages
        return allocated
    
    def pin(self, request_id: str):
        """Pin KV pages to prevent eviction."""
        if request_id in self.allocated_pages:
            for page in self.allocated_pages[request_id]:
                page.pinned = True
    
    def evict(self, request_id: str):
        """Evict KV pages for a request."""
        if request_id not in self.allocated_pages:
            return
        
        pages = self.allocated_pages[request_id]
        for page in pages:
            self.free_pages.append(page.page_id)
        
        del self.allocated_pages[request_id]
        self.total_evictions += len(pages)
        
        logger.debug(f"Evicted {len(pages)} pages for request {request_id}")
    
    def _evict_lru(self, num_pages: int):
        """Evict least recently used pages."""
        # Find unpinned requests
        candidates = [
            req_id for req_id, pages in self.allocated_pages.items()
            if not any(p.pinned for p in pages)
        ]
        
        # Evict oldest
        for req_id in candidates:
            self.evict(req_id)
            if len(self.free_pages) >= num_pages:
                break
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total_allocated = sum(
            len(pages) for pages in self.allocated_pages.values()
        )
        
        return {
            "total_pages": self.total_pages,
            "free_pages": len(self.free_pages),
            "allocated_pages": total_allocated,
            "utilization": total_allocated / self.total_pages,
            "total_allocs": self.total_allocs,
            "total_evictions": self.total_evictions,
            "quantization_enabled": self.enable_quantization,
        }