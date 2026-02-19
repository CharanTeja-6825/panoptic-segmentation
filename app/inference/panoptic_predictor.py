"""
Panoptic predictor module.

Wraps the loaded YOLOv8-seg model and produces panoptic-style segmentation
results (instance + semantic) for a single BGR frame (NumPy array).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.config import config

logger = logging.getLogger(__name__)

# COCO 80-class palette (one colour per class index, BGR order)
_PALETTE: np.ndarray = np.array(
    [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 255],
        [255, 165, 0], [0, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        [0, 128, 128], [128, 0, 128], [255, 128, 0], [128, 255, 0],
        [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128],
        [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0],
        [0, 64, 64], [64, 0, 64], [128, 64, 0], [0, 128, 64],
        [0, 64, 128], [64, 0, 128], [128, 0, 64], [64, 128, 0],
        [32, 32, 0], [0, 32, 32], [32, 0, 32], [192, 192, 0],
        [0, 192, 192], [192, 0, 192], [96, 96, 0], [0, 96, 96],
        [96, 0, 96], [160, 96, 0], [0, 160, 96], [96, 0, 160],
        [64, 96, 0], [0, 64, 96], [96, 0, 64], [96, 64, 0],
        [48, 48, 0], [0, 48, 48], [48, 0, 48], [224, 128, 0],
        [0, 224, 128], [128, 0, 224], [32, 192, 0], [0, 32, 192],
        [192, 0, 32], [64, 160, 0], [0, 64, 160], [160, 0, 64],
        [96, 128, 0], [0, 96, 128], [128, 0, 96], [96, 64, 32],
    ],
    dtype=np.uint8,
)


@dataclass
class Detection:
    """Represents a single detected / segmented object."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    mask: Optional[np.ndarray] = None  # H×W bool mask, or None


@dataclass
class PanopticResult:
    """Holds the full panoptic segmentation result for one frame."""

    detections: List[Detection] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    semantic_map: Optional[np.ndarray] = None  # H×W int32 class map
    instance_map: Optional[np.ndarray] = None  # H×W int32 instance-id map
    fps: float = 0.0


class PanopticPredictor:
    """Performs panoptic segmentation using a pre-loaded YOLOv8-seg model.

    Args:
        model_loader: A *loaded* :class:`~app.inference.model_loader.ModelLoader`
            instance.
        mask_alpha: Transparency factor for drawn masks (0 = transparent,
            1 = opaque).
    """

    def __init__(self, model_loader, mask_alpha: float = config.mask_alpha) -> None:
        self._loader = model_loader
        self.mask_alpha = mask_alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> PanopticResult:
        """Run panoptic segmentation on a single BGR frame.

        Args:
            frame: A BGR uint8 NumPy array (H×W×3).

        Returns:
            :class:`PanopticResult` populated with detections and the
            annotated frame.
        """
        result = PanopticResult()
        model = self._loader.model

        # Resize for faster inference while preserving aspect ratio
        h, w = frame.shape[:2]
        resized, scale_x, scale_y = self._resize_frame(frame)

        # Run YOLOv8-seg inference (returns a list with one Results object)
        predictions = model(
            resized,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            device=self._loader.device,
            verbose=False,
        )

        if not predictions:
            result.annotated_frame = frame.copy()
            return result

        pred = predictions[0]

        # ---- Class names ------------------------------------------------
        names: Dict[int, str] = pred.names  # {class_id: name}

        # ---- Parse detections ------------------------------------------
        detections: List[Detection] = []
        semantic_map = np.full((h, w), -1, dtype=np.int32)
        instance_map = np.zeros((h, w), dtype=np.int32)

        boxes = pred.boxes
        masks_data = pred.masks  # may be None if no masks predicted

        if boxes is not None:
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Scale bbox back to original frame dimensions
                x1 = int(x1 / scale_x)
                y1 = int(y1 / scale_y)
                x2 = int(x2 / scale_x)
                y2 = int(y2 / scale_y)

                # Clamp to frame bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Extract and resize mask
                mask_arr: Optional[np.ndarray] = None
                if masks_data is not None and idx < len(masks_data.data):
                    raw_mask = masks_data.data[idx].cpu().numpy()  # float32
                    mask_arr = cv2.resize(
                        raw_mask,
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    ) > 0.5

                    # Update semantic / instance maps
                    semantic_map[mask_arr] = cls_id
                    instance_map[mask_arr] = idx + 1  # instance IDs start at 1

                det = Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    mask=mask_arr,
                )
                detections.append(det)

        result.detections = detections
        result.semantic_map = semantic_map
        result.instance_map = instance_map

        # ---- Draw annotations on a copy of the original frame ----------
        result.annotated_frame = self._draw_annotations(frame.copy(), detections)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_frame(
        frame: np.ndarray,
        target_size: int = config.inference_input_size,
    ) -> Tuple[np.ndarray, float, float]:
        """Resize *frame* so that the longer side equals *target_size*.

        Returns:
            Tuple of (resized_frame, scale_x, scale_y) where scale_x/y are
            the ratios resized_width/original_width etc., used to map
            coordinates back to the original resolution.
        """
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        if scale >= 1.0:
            return frame, 1.0, 1.0

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, new_w / w, new_h / h

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Overlay masks, bounding boxes and labels onto *frame* in-place.

        Args:
            frame: BGR uint8 array to draw on.
            detections: List of :class:`Detection` objects.

        Returns:
            The annotated frame (same array, modified in-place).
        """
        overlay = frame.copy()

        for det in detections:
            colour = _PALETTE[det.class_id % len(_PALETTE)].tolist()

            # Draw filled mask
            if det.mask is not None:
                overlay[det.mask] = colour

            # Blend mask overlay
        frame = cv2.addWeighted(overlay, self.mask_alpha, frame, 1 - self.mask_alpha, 0)

        for det in detections:
            colour = _PALETTE[det.class_id % len(_PALETTE)].tolist()
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background + text
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            lx1 = x1
            ly1 = max(y1 - th - baseline - 4, 0)
            lx2 = x1 + tw + 4
            ly2 = y1
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), colour, -1)
            cv2.putText(
                frame,
                label,
                (lx1 + 2, ly2 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
