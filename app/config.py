"""
Configuration module for the Panoptic Segmentation Web Application.

Reads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Application-wide configuration."""

    # Server settings
    host: str = os.environ.get("APP_HOST", "0.0.0.0")
    port: int = int(os.environ.get("APP_PORT", "8000"))
    debug: bool = os.environ.get("APP_DEBUG", "false").lower() == "true"

    # Model settings
    model_size: str = os.environ.get("MODEL_SIZE", "medium")  # small | medium | large
    model_device: str = os.environ.get("MODEL_DEVICE", "auto")  # auto | cpu | cuda

    # Allowed model sizes mapped to YOLOv8 seg variants
    model_map: dict = field(default_factory=lambda: {
        "small": "yolov8s-seg.pt",
        "medium": "yolov8m-seg.pt",
        "large": "yolov8x-seg.pt",
    })

    # Video processing
    max_upload_size_mb: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "500"))
    allowed_video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")
    upload_dir: str = os.environ.get("UPLOAD_DIR", "uploads")

    # Inference settings
    inference_input_size: int = int(os.environ.get("INFERENCE_INPUT_SIZE", "640"))
    confidence_threshold: float = float(os.environ.get("CONF_THRESHOLD", "0.35"))
    iou_threshold: float = float(os.environ.get("IOU_THRESHOLD", "0.45"))
    mask_alpha: float = float(os.environ.get("MASK_ALPHA", "0.45"))

    # Camera / streaming settings
    camera_index: int = int(os.environ.get("CAMERA_INDEX", "0"))
    stream_fps_target: int = int(os.environ.get("STREAM_FPS_TARGET", "25"))
    stream_width: int = int(os.environ.get("STREAM_WIDTH", "640"))
    stream_height: int = int(os.environ.get("STREAM_HEIGHT", "480"))
    jpeg_quality: int = int(os.environ.get("JPEG_QUALITY", "85"))

    # CORS origins (comma-separated)
    cors_origins: List[str] = field(
        default_factory=lambda: os.environ.get("CORS_ORIGINS", "*").split(",")
    )

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")

    # Thread pool for CPU-bound inference
    inference_workers: int = int(os.environ.get("INFERENCE_WORKERS", "2"))


# Singleton config instance used throughout the application
config = AppConfig()
