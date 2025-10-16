"""Setup script for FaaSInfer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version_file = Path(__file__).parent / "src" / "faasinfer" / "__version__.py"
version = {}
with open(version_file) as f:
    exec(f.read(), version)

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="faasinfer",
    version=version["__version__"],
    description="Low-Latency Serverless Inference for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FaaSInfer Team",
    author_email="team@faasinfer.ai",
    url="https://github.com/your-org/faasinfer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "ray[default]>=2.9.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "vllm>=0.2.7",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "safetensors>=0.4.0",
        "boto3>=1.28.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "gcs": ["google-cloud-storage>=2.10.0"],
        "azure": ["azure-storage-blob>=12.19.0"],
    },
    entry_points={
        "console_scripts": [
            "faasinfer=faasinfer.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)