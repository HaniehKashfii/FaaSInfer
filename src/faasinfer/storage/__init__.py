"""Storage module for FaaSInfer."""

from faasinfer.storage.registry import ModelRegistry
from faasinfer.storage.converter import CheckpointConverter

__all__ = ["ModelRegistry", "CheckpointConverter"]

