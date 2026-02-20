"""
Multi-camera stream manager.

Supports concurrent camera feeds, each with a unique stream ID, independent
analytics, and lifecycle management.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.config import config
from app.inference.panoptic_predictor import PanopticPredictor
from app.inference.tracker import ObjectTracker, TrackingResult
from app.analytics.object_counter import ObjectCounter
from app.utils.fps_counter import FPSCounter

logger = logging.getLogger(__name__)


@dataclass
class CameraStream:
    """State for one camera stream."""

    stream_id: str
    camera_index: int
    label: str = ""
    running: bool = False
    fps: float = 0.0
    latest_frame: Optional[np.ndarray] = None
    latest_annotated: Optional[np.ndarray] = None
    object_count: int = 0


class CameraManager:
    """Manages multiple concurrent camera feeds.

    Each camera gets its own capture thread, predictor, tracker,
    and analytics counter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: Dict[str, _StreamContext] = {}

    def start_stream(
        self,
        camera_index: int,
        predictor: PanopticPredictor,
        label: str = "",
    ) -> str:
        """Start a new camera stream.

        Args:
            camera_index: OpenCV camera device index.
            predictor: Predictor instance for this stream.
            label: Optional human-readable label.

        Returns:
            Unique stream ID.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        stream_id = str(uuid.uuid4())[:8]

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.stream_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.stream_height)
        cap.set(cv2.CAP_PROP_FPS, config.stream_fps_target)

        ctx = _StreamContext(
            stream_id=stream_id,
            camera_index=camera_index,
            label=label or f"Camera {camera_index}",
            cap=cap,
            predictor=predictor,
            tracker=ObjectTracker(),
            counter=ObjectCounter(),
            fps_counter=FPSCounter(window=30),
        )

        with self._lock:
            self._streams[stream_id] = ctx

        ctx.running = True
        ctx.thread = threading.Thread(
            target=self._capture_loop, args=(stream_id,), daemon=True
        )
        ctx.thread.start()

        logger.info(
            "Started stream %s (camera %d, label='%s')",
            stream_id,
            camera_index,
            ctx.label,
        )
        return stream_id

    def stop_stream(self, stream_id: str) -> bool:
        """Stop a camera stream.

        Args:
            stream_id: ID of the stream to stop.

        Returns:
            True if the stream was found and stopped.
        """
        with self._lock:
            ctx = self._streams.get(stream_id)
        if ctx is None:
            return False

        ctx.running = False
        if ctx.thread is not None:
            ctx.thread.join(timeout=5)
        if ctx.cap is not None:
            ctx.cap.release()

        with self._lock:
            self._streams.pop(stream_id, None)

        logger.info("Stopped stream %s", stream_id)
        return True

    def list_streams(self) -> List[CameraStream]:
        """Return metadata for all active streams."""
        with self._lock:
            streams = list(self._streams.values())

        result = []
        for ctx in streams:
            result.append(
                CameraStream(
                    stream_id=ctx.stream_id,
                    camera_index=ctx.camera_index,
                    label=ctx.label,
                    running=ctx.running,
                    fps=round(ctx.fps_counter.fps, 1),
                    object_count=ctx.last_object_count,
                )
            )
        return result

    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get the latest annotated frame for a stream.

        Args:
            stream_id: Stream identifier.

        Returns:
            BGR annotated frame or None.
        """
        with self._lock:
            ctx = self._streams.get(stream_id)
        if ctx is None:
            return None
        with ctx.lock:
            if ctx.latest_annotated is None:
                return None
            return ctx.latest_annotated.copy()

    def get_stream_stats(self, stream_id: str) -> Optional[Dict]:
        """Get analytics stats for a specific stream."""
        with self._lock:
            ctx = self._streams.get(stream_id)
        if ctx is None:
            return None
        return ctx.counter.get_live_stats()

    def stop_all(self) -> None:
        """Stop all active camera streams."""
        with self._lock:
            ids = list(self._streams.keys())
        for sid in ids:
            self.stop_stream(sid)

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self, stream_id: str) -> None:
        with self._lock:
            ctx = self._streams.get(stream_id)
        if ctx is None:
            return

        delay = 1.0 / config.stream_fps_target

        while ctx.running:
            ret, frame = ctx.cap.read()
            if not ret:
                logger.warning("Stream %s: frame read failed", stream_id)
                time.sleep(0.05)
                continue

            try:
                result = ctx.predictor.predict(frame)
                annotated = (
                    result.annotated_frame
                    if result.annotated_frame is not None
                    else frame
                )

                tracking = ctx.tracker.update(result.detections)
                ctx.counter.update(tracking.tracked_objects, ctx.fps_counter.fps)
                ctx.last_object_count = len(tracking.tracked_objects)
            except Exception as exc:
                logger.error("Stream %s inference error: %s", stream_id, exc)
                annotated = frame

            ctx.fps_counter.tick()

            with ctx.lock:
                ctx.latest_annotated = annotated

            time.sleep(max(0, delay - 0.001))


@dataclass
class _StreamContext:
    """Internal mutable context for a single camera stream."""

    stream_id: str
    camera_index: int
    label: str
    cap: cv2.VideoCapture
    predictor: PanopticPredictor
    tracker: ObjectTracker
    counter: ObjectCounter
    fps_counter: FPSCounter
    running: bool = False
    thread: Optional[threading.Thread] = None
    latest_annotated: Optional[np.ndarray] = None
    last_object_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
