FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install application
RUN pip install -e .

# Create directories
RUN mkdir -p /var/faasinfer/ssd_cache /dev/shm/faasinfer

# Expose port
EXPOSE 8000

# Run application
CMD ["faasinfer", "serve", "--host", "0.0.0.0", "--port", "8000"]