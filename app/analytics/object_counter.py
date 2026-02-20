"""
Object counter module.

Maintains per-frame and rolling object counts by class, providing live
statistics for the analytics dashboard.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.inference.tracker import TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    """Per-frame counting snapshot."""

    timestamp: float = 0.0
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    total_objects: int = 0
    fps: float = 0.0


class ObjectCounter:
    """Counts objects per class across frames with rolling statistics.

    Args:
        rolling_window: Number of recent frames to include in rolling stats.
    """

    def __init__(self, rolling_window: int = 100) -> None:
        self._rolling_window = rolling_window
        self._lock = threading.Lock()
        self._history: List[FrameStats] = []
        self._total_frames: int = 0
        self._cumulative_counts: Dict[str, int] = defaultdict(int)

    def update(
        self,
        tracked_objects: List[TrackedObject],
        fps: float = 0.0,
    ) -> FrameStats:
        """Record counts for the current frame.

        Args:
            tracked_objects: List of tracked objects from the tracker.
            fps: Current FPS reading.

        Returns:
            :class:`FrameStats` snapshot for this frame.
        """
        counts: Dict[str, int] = defaultdict(int)
        for obj in tracked_objects:
            counts[obj.class_name] += 1

        stats = FrameStats(
            timestamp=time.time(),
            counts_by_class=dict(counts),
            total_objects=len(tracked_objects),
            fps=fps,
        )

        with self._lock:
            self._total_frames += 1
            self._history.append(stats)
            if len(self._history) > self._rolling_window:
                self._history = self._history[-self._rolling_window :]
            for cls_name, cnt in counts.items():
                self._cumulative_counts[cls_name] += cnt

        return stats

    def get_live_stats(self) -> Dict:
        """Return a snapshot of live analytics data.

        Returns:
            Dictionary with current counts, rolling averages, and totals.
        """
        with self._lock:
            if not self._history:
                return {
                    "current_counts": {},
                    "total_objects": 0,
                    "fps": 0.0,
                    "total_frames": 0,
                    "rolling_avg_counts": {},
                }

            latest = self._history[-1]

            # Compute rolling average counts
            rolling_totals: Dict[str, float] = defaultdict(float)
            for frame_stats in self._history:
                for cls_name, cnt in frame_stats.counts_by_class.items():
                    rolling_totals[cls_name] += cnt
            n = len(self._history)
            rolling_avg = {
                cls: round(total / n, 2)
                for cls, total in rolling_totals.items()
            }

            return {
                "current_counts": latest.counts_by_class,
                "total_objects": latest.total_objects,
                "fps": round(latest.fps, 1),
                "total_frames": self._total_frames,
                "rolling_avg_counts": rolling_avg,
            }

    def reset(self) -> None:
        """Clear all recorded statistics."""
        with self._lock:
            self._history.clear()
            self._total_frames = 0
            self._cumulative_counts.clear()
        logger.info("Object counter reset")
