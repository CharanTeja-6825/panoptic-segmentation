"""
Configuration module for the Panoptic Segmentation Web Application.

Reads settings from environment variables with sensible defaults.
Optimized for low-resource hardware (8GB RAM, i3 CPU).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


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
    video_keyframe_interval_seconds: float = float(
        os.environ.get("VIDEO_KEYFRAME_INTERVAL_SECONDS", "2.0")
    )

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

    # Tracking settings
    tracker_max_age: int = int(os.environ.get("TRACKER_MAX_AGE", "30"))
    tracker_min_hits: int = int(os.environ.get("TRACKER_MIN_HITS", "3"))
    tracker_iou_threshold: float = float(os.environ.get("TRACKER_IOU_THRESHOLD", "0.3"))

    # Depth estimation (off by default)
    depth_enabled: bool = os.environ.get("DEPTH_ENABLED", "false").lower() == "true"
    depth_model: str = os.environ.get("DEPTH_MODEL", "MiDaS_small")

    # Performance
    use_fp16: bool = os.environ.get("USE_FP16", "false").lower() == "true"
    model_warmup_frames: int = int(os.environ.get("MODEL_WARMUP_FRAMES", "3"))

    # Analytics
    analytics_rolling_window: int = int(os.environ.get("ANALYTICS_ROLLING_WINDOW", "100"))
    heatmap_width: int = int(os.environ.get("HEATMAP_WIDTH", "640"))
    heatmap_height: int = int(os.environ.get("HEATMAP_HEIGHT", "480"))
    heatmap_decay: float = float(os.environ.get("HEATMAP_DECAY", "0.98"))

    # -------------------------------------------------------------------------
    # Chat frame throttling (optimized for low-resource hardware)
    # -------------------------------------------------------------------------
    chat_frame_interval_ms: int = int(os.environ.get("CHAT_FRAME_INTERVAL_MS", "2000"))
    chat_max_image_width: int = int(os.environ.get("CHAT_MAX_IMAGE_WIDTH", "512"))
    chat_jpeg_quality: int = int(os.environ.get("CHAT_JPEG_QUALITY", "60"))

    # -------------------------------------------------------------------------
    # LLM queue and concurrency settings
    # -------------------------------------------------------------------------
    llm_max_queue_size: int = int(os.environ.get("LLM_MAX_QUEUE_SIZE", "2"))
    llm_max_concurrent: int = int(os.environ.get("LLM_MAX_CONCURRENT", "1"))

    # -------------------------------------------------------------------------
    # Ollama LLM settings
    # -------------------------------------------------------------------------
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://100.91.144.84:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_vision_model: str = os.environ.get("OLLAMA_VISION_MODEL", "llava-phi3")
    ollama_timeout: float = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
    ollama_read_timeout: float = float(os.environ.get("OLLAMA_READ_TIMEOUT", "45"))
    ollama_connect_timeout: float = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5"))
    ollama_max_image_kb: int = int(os.environ.get("OLLAMA_MAX_IMAGE_KB", "350"))

    # -------------------------------------------------------------------------
    # LLM model configuration (primary and fallback)
    # -------------------------------------------------------------------------
    llm_primary_model: str = os.environ.get("LLM_PRIMARY_MODEL", "llava-phi3")
    llm_fallback_model: str = os.environ.get("LLM_FALLBACK_MODEL", "llava:7b")

    # -------------------------------------------------------------------------
    # Cloud fallback (feature-flagged, OFF by default)
    # -------------------------------------------------------------------------
    llm_enable_cloud_fallback: bool = (
        os.environ.get("LLM_ENABLE_CLOUD_FALLBACK", "false").lower() == "true"
    )
    openai_base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL") or None
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or None

    # Commentary engine
    commentary_enabled: bool = os.environ.get("COMMENTARY_ENABLED", "true").lower() == "true"
    commentary_interval: float = float(os.environ.get("COMMENTARY_INTERVAL", "30"))

    # Scene memory
    scene_memory_capacity: int = int(os.environ.get("SCENE_MEMORY_CAPACITY", "300"))
    scene_snapshot_interval: float = float(os.environ.get("SCENE_SNAPSHOT_INTERVAL", "5"))


# Singleton config instance used throughout the application
config = AppConfig()
