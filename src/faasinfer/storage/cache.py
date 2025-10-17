"""
Multi-tier cache implementation.

Manages DRAM and SSD caches with LRU eviction.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    size_bytes: int
    tier: str  # "dram" or "ssd"
    last_access: float
    

class MultiTierCache:
    """
    Multi-tier cache with DRAM and SSD layers.
    
    Implements FR 004: Local checkpoint cache with LRU.
    """
    
    def __init__(
        self,
        dram_capacity_gb: float = 100.0,
        ssd_capacity_gb: float = 500.0,
        dram_path: str = "/dev/shm/faasinfer",
        ssd_path: str = "/var/faasinfer/ssd_cache",
    ):
        self.dram_capacity = int(dram_capacity_gb * 1e9)
        self.ssd_capacity = int(ssd_capacity_gb * 1e9)
        self.dram_path = Path(dram_path)
        self.ssd_path = Path(ssd_path)
        
        # Create directories
        self.dram_path.mkdir(parents=True, exist_ok=True)
        self.ssd_path.mkdir(parents=True, exist_ok=True)
        
        # Cache tracking
        self.dram_entries: Dict[str, CacheEntry] = {}
        self.ssd_entries: Dict[str, CacheEntry] = {}
        
        self.dram_used = 0
        self.ssd_used = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(
            f"Initialized MultiTierCache: "
            f"DRAM={dram_capacity_gb}GB, SSD={ssd_capacity_gb}GB"
        )
    
    def get(self, key: str) -> Optional[Path]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cached file or None
        """
        # Check DRAM first
        if key in self.dram_entries:
            self.hits += 1
            path = self.dram_path / key
            if path.exists():
                return path
        
        # Check SSD
        if key in self.ssd_entries:
            self.hits += 1
            path = self.ssd_path / key
            if path.exists():
                # Promote to DRAM if space available
                self._promote_to_dram(key)
                return path
        
        self.misses += 1
        return None
    
    def put(self, key: str, size_bytes: int, tier: str = "ssd"):
        """Add item to cache."""
        import time
        
        entry = CacheEntry(
            key=key,
            size_bytes=size_bytes,
            tier=tier,
            last_access=time.time(),
        )
        
        if tier == "dram":
            # Evict if needed
            while self.dram_used + size_bytes > self.dram_capacity:
                self._evict_dram_lru()
            
            self.dram_entries[key] = entry
            self.dram_used += size_bytes
        
        else:  # ssd
            while self.ssd_used + size_bytes > self.ssd_capacity:
                self._evict_ssd_lru()
            
            self.ssd_entries[key] = entry
            self.ssd_used += size_bytes
    
    def _promote_to_dram(self, key: str):
        """Promote SSD entry to DRAM."""
        if key not in self.ssd_entries:
            return
        
        entry = self.ssd_entries[key]
        
        # Check if fits in DRAM
        if self.dram_used + entry.size_bytes <= self.dram_capacity:
            self.dram_entries[key] = entry
            self.dram_used += entry.size_bytes
    
    def _evict_dram_lru(self):
        """Evict least recently used from DRAM."""
        if not self.dram_entries:
            return
        
        # Find LRU
        lru_key = min(
            self.dram_entries.keys(),
            key=lambda k: self.dram_entries[k].last_access
        )
        
        entry = self.dram_entries[lru_key]
        self.dram_used -= entry.size_bytes
        del self.dram_entries[lru_key]
    
    def _evict_ssd_lru(self):
        """Evict least recently used from SSD."""
        if not self.ssd_entries:
            return
        
        lru_key = min(
            self.ssd_entries.keys(),
            key=lambda k: self.ssd_entries[k].last_access
        )
        
        entry = self.ssd_entries[lru_key]
        self.ssd_used -= entry.size_bytes
        del self.ssd_entries[lru_key]
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        
        return {
            "dram_used_gb": self.dram_used / 1e9,
            "dram_capacity_gb": self.dram_capacity / 1e9,
            "ssd_used_gb": self.ssd_used / 1e9,
            "ssd_capacity_gb": self.ssd_capacity / 1e9,
            "dram_entries": len(self.dram_entries),
            "ssd_entries": len(self.ssd_entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }
