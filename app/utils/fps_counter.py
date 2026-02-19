"""
FPS counter utility.

Provides a rolling-window FPS counter suitable for real-time video
processing pipelines.
"""

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Measures frames-per-second using a sliding window.

    Args:
        window: Number of recent frame timestamps to keep. Larger values
            produce smoother readings but are slower to adapt to changes.

    Example::

        counter = FPSCounter(window=30)
        while streaming:
            process_frame()
            counter.tick()
            print(f"FPS: {counter.fps:.1f}")
    """

    def __init__(self, window: int = 30) -> None:
        self._window = max(1, window)
        self._timestamps: Deque[float] = deque(maxlen=self._window)

    def tick(self) -> None:
        """Record a frame completion event (call once per processed frame)."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current FPS estimate based on the rolling window.

        Returns:
            Frames per second as a float; 0.0 if fewer than 2 ticks have
            been recorded.
        """
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
