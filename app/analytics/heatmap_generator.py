"""
Movement heatmap generator.

Accumulates object centroid positions over time and produces a colour heatmap
image that can be overlaid on the video feed.
"""

import logging
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.inference.tracker import TrackedObject

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Generates an activity heatmap from tracked object centroids.

    Args:
        width: Heatmap width in pixels.
        height: Heatmap height in pixels.
        decay: Per-frame exponential decay factor (0–1). Values closer to 1
            retain history longer.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        decay: float = 0.98,
    ) -> None:
        self._width = width
        self._height = height
        self._decay = decay
        self._lock = threading.Lock()
        self._accumulator = np.zeros((height, width), dtype=np.float64)

    def update(self, tracked_objects: List[TrackedObject]) -> None:
        """Add centroid positions from the current frame.

        Args:
            tracked_objects: List of currently tracked objects.
        """
        with self._lock:
            # Apply decay so older activity fades
            self._accumulator *= self._decay

            for obj in tracked_objects:
                cx, cy = obj.centroid
                ix = int(min(max(cx, 0), self._width - 1))
                iy = int(min(max(cy, 0), self._height - 1))
                # Gaussian-like stamp (small kernel)
                radius = 8
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        px = ix + dx
                        py = iy + dy
                        if 0 <= px < self._width and 0 <= py < self._height:
                            dist_sq = dx * dx + dy * dy
                            if dist_sq <= radius * radius:
                                weight = 1.0 - (dist_sq / (radius * radius))
                                self._accumulator[py, px] += weight

    def get_heatmap(
        self,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Return the heatmap as a BGR image.

        Args:
            target_size: Optional ``(width, height)`` to resize the heatmap to.

        Returns:
            BGR uint8 heatmap image.
        """
        with self._lock:
            acc = self._accumulator.copy()

        # Normalise to 0–255
        max_val = acc.max()
        if max_val > 0:
            normalised = (acc / max_val * 255).astype(np.uint8)
        else:
            normalised = np.zeros_like(acc, dtype=np.uint8)

        # Apply colour map
        heatmap = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

        if target_size is not None:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

        return heatmap

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Blend the heatmap onto a video frame.

        Args:
            frame: BGR uint8 frame to overlay onto.
            alpha: Blend factor for the heatmap (0 = invisible, 1 = opaque).

        Returns:
            Blended BGR frame.
        """
        h, w = frame.shape[:2]
        heatmap = self.get_heatmap(target_size=(w, h))
        return cv2.addWeighted(heatmap, alpha, frame, 1.0 - alpha, 0)

    def reset(self) -> None:
        """Clear the accumulated heatmap."""
        with self._lock:
            self._accumulator[:] = 0.0
        logger.info("Heatmap generator reset")
