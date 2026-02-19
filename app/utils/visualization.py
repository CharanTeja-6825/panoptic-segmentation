"""
Visualization utilities.

Helper functions for drawing panoptic segmentation overlays on frames.
These functions are thin wrappers kept separate so they can be reused
outside the predictor (e.g. for post-processing saved frames).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR NumPy frame to JPEG bytes.

    Args:
        frame: BGR uint8 array (H×W×3).
        quality: JPEG quality (1–100).

    Returns:
        JPEG-encoded bytes.
    """
    _, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return buf.tobytes()


def draw_fps(
    frame: np.ndarray,
    fps: float,
    pos: Tuple[int, int] = (10, 30),
    colour: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw an FPS counter on *frame*.

    Args:
        frame: BGR uint8 array to annotate (modified in-place).
        fps: Current FPS value.
        pos: Top-left pixel position of the text.
        colour: BGR colour tuple.

    Returns:
        The annotated frame (same object).
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_legend(
    frame: np.ndarray,
    class_names: List[str],
    palette: np.ndarray,
    max_items: int = 15,
) -> np.ndarray:
    """Draw a colour legend for detected classes in the top-right corner.

    Args:
        frame: BGR uint8 array (modified in-place).
        class_names: List of unique class name strings to display.
        palette: N×3 uint8 BGR colour palette array.
        max_items: Maximum number of legend entries to draw.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]
    item_height = 22
    pad = 8
    legend_w = 160

    visible = class_names[:max_items]
    legend_h = len(visible) * item_height + pad * 2

    # Semi-transparent background
    x1 = w - legend_w - pad
    y1 = pad
    x2 = w - pad
    y2 = y1 + legend_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, name in enumerate(visible):
        colour = palette[i % len(palette)].tolist()
        ty = y1 + pad + i * item_height + item_height // 2
        # Colour swatch
        cv2.rectangle(frame, (x1 + 4, ty - 7), (x1 + 16, ty + 5), colour, -1)
        # Label
        cv2.putText(
            frame,
            name,
            (x1 + 22, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def build_mjpeg_frame(jpeg_bytes: bytes) -> bytes:
    """Wrap JPEG bytes in an MJPEG multipart boundary.

    Args:
        jpeg_bytes: Raw JPEG-encoded image bytes.

    Returns:
        Byte string suitable for an MJPEG ``multipart/x-mixed-replace``
        streaming response.
    """
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpeg_bytes
        + b"\r\n"
    )
