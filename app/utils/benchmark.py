"""
Benchmark utility module.

Provides system performance metrics (FPS, GPU memory, CPU usage) that can be
queried via the ``GET /benchmark`` API endpoint.
"""

import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BenchmarkUtil:
    """Collects and reports inference performance metrics.

    Args:
        history_size: Number of recent frame timings to keep for averaging.
    """

    def __init__(self, history_size: int = 200) -> None:
        self._history_size = history_size
        self._frame_times: list = []
        self._start_time: Optional[float] = None

    def record_frame_time(self, elapsed_seconds: float) -> None:
        """Record the time taken to process a single frame.

        Args:
            elapsed_seconds: Wall-clock seconds for one frame.
        """
        self._frame_times.append(elapsed_seconds)
        if len(self._frame_times) > self._history_size:
            self._frame_times = self._frame_times[-self._history_size :]

    def start_timer(self) -> None:
        """Start a frame processing timer."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop the timer and record the elapsed time."""
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.record_frame_time(elapsed)
            self._start_time = None

    def get_metrics(self) -> Dict:
        """Return current performance metrics.

        Returns:
            Dictionary with ``avg_fps``, ``gpu_memory_mb``,
            ``gpu_memory_total_mb``, ``cpu_percent``, and ``device``.
        """
        # Average FPS
        if self._frame_times:
            avg_time = sum(self._frame_times) / len(self._frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        else:
            avg_fps = 0.0

        # GPU memory
        gpu_memory_mb = 0.0
        gpu_memory_total_mb = 0.0
        gpu_name = None
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)

        # CPU usage – use os.getloadavg() as a lightweight proxy
        try:
            load_avg = os.getloadavg()
            cpu_load = load_avg[0]  # 1-minute load average
        except (OSError, AttributeError):
            cpu_load = 0.0

        device = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "avg_fps": round(avg_fps, 2),
            "gpu_memory_mb": round(gpu_memory_mb, 1),
            "gpu_memory_total_mb": round(gpu_memory_total_mb, 1),
            "gpu_name": gpu_name,
            "cpu_load_1m": round(cpu_load, 2),
            "device": device,
            "frames_measured": len(self._frame_times),
        }

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._frame_times.clear()
        self._start_time = None
        logger.info("Benchmark metrics reset")


def warmup_model(predictor, num_frames: int = 3, size: int = 640) -> None:
    """Run a few inference passes to warm up the model.

    Args:
        predictor: A :class:`~app.inference.panoptic_predictor.PanopticPredictor`.
        num_frames: Number of warm-up frames.
        size: Frame dimensions for synthetic input.
    """
    logger.info("Warming up model with %d synthetic frames…", num_frames)
    dummy = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(num_frames):
        predictor.predict(dummy)
    logger.info("Model warmup complete")
