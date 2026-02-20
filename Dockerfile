# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Panoptic Segmentation App"
LABEL description="Real-time panoptic segmentation with YOLOv8-seg and FastAPI"

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/      app/
COPY frontend/ frontend/
COPY run.sh    run.sh
COPY tests/    tests/

RUN chmod +x run.sh

# Create directories used at runtime
RUN mkdir -p uploads outputs

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["bash", "run.sh"]
