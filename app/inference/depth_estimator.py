"""
Depth estimation module.

Integrates Intel MiDaS small model for monocular depth estimation. Provides
per-pixel depth maps that can be fused with segmentation outputs to estimate
object distances.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Monocular depth estimator using MiDaS (small variant).

    Args:
        device: Target device string ("auto", "cpu", or "cuda").
        use_fp16: Whether to use half-precision inference on CUDA.
    """

    def __init__(
        self,
        device: str = "auto",
        use_fp16: bool = False,
    ) -> None:
        self._device_pref = device
        self._use_fp16 = use_fp16
        self._model: Optional[torch.nn.Module] = None
        self._transform = None
        self._device: str = "cpu"
        self._enabled: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info("Depth estimation %s", "enabled" if value else "disabled")

    def load(self) -> None:
        """Download (if needed) and load the MiDaS small model."""
        if self._model is not None:
            logger.debug("Depth model already loaded")
            return

        self._device = self._resolve_device(self._device_pref)
        logger.info("Loading MiDaS small on '%s'…", self._device)

        self._model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
        )
        self._model.to(self._device)
        self._model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        self._transform = midas_transforms.small_transform

        if self._use_fp16 and self._device == "cuda":
            self._model = self._model.half()
            logger.info("Depth model using fp16")

        logger.info("MiDaS small loaded on %s", self._device)

    def unload(self) -> None:
        """Release the depth model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._transform = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Depth model unloaded")

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Produce a depth map for the given BGR frame.

        Args:
            frame: BGR uint8 NumPy array (H×W×3).

        Returns:
            Normalised depth map as float32 array (H×W) with values in
            ``[0, 1]`` where higher values indicate greater depth (farther),
            or *None* if depth estimation is not enabled / loaded.
        """
        if not self._enabled or self._model is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        if self._use_fp16 and self._device == "cuda":
            input_batch = input_batch.half()

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalise to [0, 1]
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def estimate_object_distance(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the distance of an object from the camera.

        Uses the median depth within the object's bounding box (or mask).
        Returns a value in arbitrary relative units (lower = closer).

        Args:
            depth_map: Normalised depth map from :meth:`estimate`.
            bbox: ``(x1, y1, x2, y2)`` bounding box.
            mask: Optional boolean mask for the object.

        Returns:
            Estimated relative distance (0–10 scale, approximate metres).
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if mask is not None:
            region = depth_map[mask]
        else:
            region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        # MiDaS outputs inverse depth; higher values = closer.
        # Invert and scale to approximate metres (rough heuristic).
        median_val = float(np.median(region))
        # Map [0,1] inverted → approximate distance in metres
        distance = max(0.1, (1.0 - median_val) * 10.0)
        return round(distance, 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_pref: str) -> str:
        if device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_pref == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available for depth – using CPU")
            return "cpu"
        return device_pref
