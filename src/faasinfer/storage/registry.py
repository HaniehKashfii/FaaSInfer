"""
C13: Model Registry & Storage Plugins

Responsibilities:
- Model metadata management
- Version control
- Storage plugin abstraction (S3/GCS/Azure)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata."""
    model_id: str
    model_name: str
    version: str
    size_gb: float
    num_parameters_b: float
    
    # Storage information
    storage_backend: str  # "s3", "gcs", "azure"
    storage_uri: str
    checksum: str
    
    # Model architecture
    architecture: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Status
    status: str = "available"  # "available", "loading", "deprecated"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Usage statistics
    download_count: int = 0
    inference_count: int = 0
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "size_gb": self.size_gb,
            "num_parameters_b": self.num_parameters_b,
            "storage_backend": self.storage_backend,
            "storage_uri": self.storage_uri,
            "checksum": self.checksum,
            "architecture": self.architecture,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "download_count": self.download_count,
            "inference_count": self.inference_count,
            "tags": self.tags,
        }


class ModelRegistry:
    """
    Model registry for metadata management.
    Supports versioning, tagging, and lifecycle management.
    """
    
    def __init__(self, registry_path: str = "/var/faasinfer/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Models: model_id -> ModelMetadata
        self.models: Dict[str, ModelMetadata] = {}
        
        # Version index: model_name -> List[version]
        self.versions: Dict[str, List[str]] = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Initialized ModelRegistry at {self.registry_path}")
    
    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_data in data.get("models", []):
                    # Convert datetime strings back
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    
                    metadata = ModelMetadata(**model_data)
                    self.models[metadata.model_id] = metadata
                    
                    if metadata.model_name not in self.versions:
                        self.versions[metadata.model_name] = []
                    if metadata.version not in self.versions[metadata.model_name]:
                        self.versions[metadata.model_name].append(metadata.version)
                
                logger.info(f"Loaded {len(self.models)} models from registry")
                
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        try:
            data = {
                "models": [m.to_dict() for m in self.models.values()],
                "updated_at": datetime.now().isoformat(),
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register(
        self,
        model_id: str,
        model_name: str,
        version: str,
        size_gb: float,
        num_parameters_b: float,
        storage_backend: str,
        storage_uri: str,
        checksum: str,
        architecture: str = "transformer",
        **kwargs
    ) -> bool:
        """
        Register a new model.
        
        Args:
            model_id: Unique model identifier
            model_name: Model name
            version: Model version
            size_gb: Model size in GB
            num_parameters_b: Number of parameters in billions
            storage_backend: Storage backend type
            storage_uri: Storage URI
            checksum: Model checksum
            architecture: Model architecture
            **kwargs: Additional metadata
            
        Returns:
            Success status
        """
        try:
            if model_id in self.models:
                logger.warning(f"Model {model_id} already registered")
                return False
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                size_gb=size_gb,
                num_parameters_b=num_parameters_b,
                storage_backend=storage_backend,
                storage_uri=storage_uri,
                checksum=checksum,
                architecture=architecture,
                **kwargs
            )
            
            self.models[model_id] = metadata
            
            if model_name not in self.versions:
                self.versions[model_name] = []
            if version not in self.versions[model_name]:
                self.versions[model_name].append(version)
            
            self._save_registry()
            
            logger.info(f"Registered model {model_id} ({model_name} v{version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self.models.get(model_id)
    
    def list_models(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            status: Filter by status
            tags: Filter by tags
            
        Returns:
            List of model metadata
        """
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tags:
            models = [
                m for m in models
                if any(tag in m.tags for tag in tags)
            ]
        
        return models
    
    def get_versions(self, model_name: str) -> List[str]:
        """Get all versions of a model."""
        return self.versions.get(model_name, [])
    
    def promote(self, model_id: str, tag: str = "production") -> bool:
        """
        Promote model to a tag (e.g., production).
        
        Args:
            model_id: Model to promote
            tag: Tag to add
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        metadata = self.models[model_id]
        if tag not in metadata.tags:
            metadata.tags.append(tag)
            metadata.updated_at = datetime.now()
            self._save_registry()
        
        logger.info(f"Promoted {model_id} to {tag}")
        return True
    
    def retire(self, model_id: str) -> bool:
        """
        Retire a model.
        
        Args:
            model_id: Model to retire
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        metadata = self.models[model_id]
        metadata.status = "deprecated"
        metadata.updated_at = datetime.now()
        self._save_registry()
        
        logger.info(f"Retired model {model_id}")
        return True
    
    def update_stats(
        self,
        model_id: str,
        downloads: int = 0,
        inferences: int = 0
    ):
        """Update model usage statistics."""
        if model_id not in self.models:
            return
        
        metadata = self.models[model_id]
        metadata.download_count += downloads
        metadata.inference_count += inferences
        metadata.updated_at = datetime.now()
        
        # Periodically save (not every time for performance)
        if (metadata.download_count + metadata.inference_count) % 100 == 0:
            self._save_registry()
    
    def place_hotset(
        self,
        max_models: int = 10
    ) -> List[str]:
        """
        Determine hotset of models to cache (FR 004).
        
        Uses inference count to determine popularity.
        
        Args:
            max_models: Maximum models in hotset
            
        Returns:
            List of model IDs in hotset
        """
        # Sort by inference count
        sorted_models = sorted(
            self.models.values(),
            key=lambda m: m.inference_count,
            reverse=True
        )
        
        hotset = [m.model_id for m in sorted_models[:max_models]]
        
        logger.info(f"Hotset: {hotset}")
        return hotset