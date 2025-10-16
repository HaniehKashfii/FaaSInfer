"""Main entry point for FaaSInfer CLI."""

from faasinfer.cli import cli

if __name__ == '__main__':
    cli()
```

## **19. requirements.txt**
```
# FaaSInfer Production Dependencies

# Core dependencies
ray[default]>=2.9.0
torch>=2.1.0
transformers>=4.35.0
vllm>=0.2.7

# API and networking
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
httpx>=0.25.0

# Storage
safetensors>=0.4.0
boto3>=1.28.0  # AWS S3
google-cloud-storage>=2.10.0  # GCS
azure-storage-blob>=12.19.0  # Azure Blob

# Monitoring and telemetry
prometheus-client>=0.18.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0

# Utilities
click>=8.1.0
pyyaml>=6.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

## **20. requirements-dev.txt**
```
# Development Dependencies

-r requirements.txt

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Code quality
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
isort>=5.12.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.2.0
mkdocstrings[python]>=0.23.0