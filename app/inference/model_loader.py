"""
Model loader module.

Responsible for loading, caching and releasing the YOLOv8-seg model used
for panoptic segmentation inference.
"""

import logging
import os
from typing import Optional

import torch

from app.config import config

logger = logging.getLogger(__name__)


def _resolve_device(device_pref: str) -> str:
    """Resolve the target device string.

    Args:
        device_pref: "auto", "cpu", or "cuda".

    Returns:
        Resolved device string ("cuda" or "cpu").
    """
    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_pref == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available – falling back to CPU."
        )
        return "cpu"
    return device_pref


class ModelLoader:
    """Loads and holds a YOLOv8-seg model for inference.

    Usage::

        loader = ModelLoader()
        loader.load()
        model = loader.model
        loader.unload()
    """

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._device: str = _resolve_device(config.model_device)
        self._model_name: str = config.model_map.get(
            config.model_size, "yolov8m-seg.pt"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Return the loaded model (raises if not yet loaded)."""
        if self._model is None:
            raise RuntimeError(
                "Model has not been loaded yet. Call ModelLoader.load() first."
            )
        return self._model

    @property
    def device(self) -> str:
        """Return the active inference device string."""
        return self._device

    def load(self) -> None:
        """Download (if needed) and load the model into memory."""
        if self._model is not None:
            logger.debug("Model already loaded – skipping.")
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics package is required. "
                "Install it with: pip install ultralytics"
            ) from exc

        logger.info(
            "Loading model '%s' on device '%s'…",
            self._model_name,
            self._device,
        )

        # YOLO() will auto-download the weights on first call
        self._model = YOLO(self._model_name)

        # Move model to the target device
        if self._device == "cuda":
            self._model.to("cuda")
            logger.info(
                "Model loaded on GPU: %s",
                torch.cuda.get_device_name(0),
            )
        else:
            logger.info("Model loaded on CPU.")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory freed.")

    def reload(self, model_size: Optional[str] = None) -> None:
        """Reload the model, optionally switching size.

        Args:
            model_size: "small", "medium", or "large". If *None* the current
                size from config is used.
        """
        self.unload()
        if model_size is not None:
            new_name = config.model_map.get(model_size)
            if new_name is None:
                raise ValueError(
                    f"Unknown model_size '{model_size}'. "
                    f"Valid values: {list(config.model_map.keys())}"
                )
            self._model_name = new_name
        self.load()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ModelLoader":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
