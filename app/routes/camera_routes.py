"""
Camera streaming API routes.

Endpoints:
    GET  /camera-stream        – MJPEG live camera feed with segmentation.
    POST /camera-stream/start  – Explicitly start the camera capture.
    POST /camera-stream/stop   – Stop the camera capture.
    GET  /camera-stream/status – Return camera and FPS status.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import config
from app.inference.panoptic_predictor import PanopticPredictor
from app.inference.tracker import ObjectTracker
from app.analytics.heatmap_generator import HeatmapGenerator
from app.routes.analytics_routes import get_event_logger, get_object_counter
from app.utils.fps_counter import FPSCounter
from app.utils.visualization import build_mjpeg_frame, draw_fps, encode_jpeg

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Shared camera state (single-camera, multi-consumer model)
# ---------------------------------------------------------------------------

class CameraState:
    """Thread-safe shared state for the background capture loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._running = False
        self._fps_counter = FPSCounter(window=30)
        self._thread: Optional[threading.Thread] = None
        self._predictor: Optional[PanopticPredictor] = None
        self._tracker: ObjectTracker = ObjectTracker(
            max_age=config.tracker_max_age,
            min_hits=config.tracker_min_hits,
            iou_threshold=config.tracker_iou_threshold,
        )
        self._heatmap: HeatmapGenerator = HeatmapGenerator(
            width=config.heatmap_width,
            height=config.heatmap_height,
            decay=config.heatmap_decay,
        )
        self._show_heatmap: bool = False
        self._depth_estimator = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self, predictor: PanopticPredictor, depth_estimator=None) -> None:
        """Start the background capture + inference loop."""
        with self._lock:
            if self._running:
                return
            self._predictor = predictor
            self._depth_estimator = depth_estimator
            self._cap = cv2.VideoCapture(config.camera_index)
            if not self._cap.isOpened():
                self._cap = None
                raise RuntimeError(
                    f"Cannot open camera at index {config.camera_index}."
                )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.stream_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.stream_height)
            self._cap.set(cv2.CAP_PROP_FPS, config.stream_fps_target)
            self._running = True

        self._fps_counter.reset()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera capture started (index=%d).", config.camera_index)

    def stop(self) -> None:
        """Stop the background capture loop and release the camera."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self._tracker.reset()
        self._heatmap.reset()
        logger.info("Camera capture stopped.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    @property
    def show_heatmap(self) -> bool:
        return self._show_heatmap

    @show_heatmap.setter
    def show_heatmap(self, value: bool) -> None:
        self._show_heatmap = value

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest annotated frame (thread-safe)."""
        with self._lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Continuously grab frames, run inference, update shared state."""
        delay = 1.0 / config.stream_fps_target
        counter = get_object_counter()
        event_logger = get_event_logger()

        while True:
            with self._lock:
                if not self._running or self._cap is None:
                    break

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed – retrying…")
                time.sleep(0.05)
                continue

            # Run inference
            try:
                result = self._predictor.predict(frame)
                annotated = (
                    result.annotated_frame
                    if result.annotated_frame is not None
                    else frame
                )

                # Tracking
                tracking_result = self._tracker.update(result.detections)
                tracked = tracking_result.tracked_objects

                # Analytics
                counter.update(tracked, self._fps_counter.fps)
                event_logger.update(tracked)
                self._heatmap.update(tracked)

                # Depth estimation (optional)
                depth_estimator = self._depth_estimator
                if depth_estimator is not None and depth_estimator.enabled:
                    depth_map = depth_estimator.estimate(frame)
                    if depth_map is not None:
                        for obj in tracked:
                            dist = depth_estimator.estimate_object_distance(
                                depth_map, obj.bbox
                            )
                            label = f"{obj.class_name} - {dist}m"
                            x1, y1, x2, y2 = obj.bbox
                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                # Draw tracking IDs on annotated frame
                for obj in tracked:
                    x1, y1, x2, y2 = obj.bbox
                    cv2.putText(
                        annotated,
                        f"ID:{obj.track_id}",
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        obj.color,
                        1,
                        cv2.LINE_AA,
                    )

                # Heatmap overlay
                if self._show_heatmap:
                    annotated = self._heatmap.overlay_on_frame(annotated, alpha=0.3)

            except Exception as exc:
                logger.error("Inference error: %s", exc)
                annotated = frame

            self._fps_counter.tick()
            draw_fps(annotated, self._fps_counter.fps)

            with self._lock:
                self._latest_frame = frame
                self._latest_annotated = annotated

            # Throttle to target FPS
            time.sleep(max(0, delay - 0.001))


# Module-level singleton
_camera_state = CameraState()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _get_predictor(request: Request) -> PanopticPredictor:
    loader = request.app.state.model_loader
    return PanopticPredictor(loader)


@router.post("/camera-stream/start", summary="Start live camera capture")
async def start_camera(request: Request) -> JSONResponse:
    """Open the camera and begin real-time segmentation."""
    if _camera_state.is_running:
        return JSONResponse({"status": "already_running"})

    predictor = _get_predictor(request)
    depth_estimator = getattr(request.app.state, "depth_estimator", None)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _camera_state.start, predictor, depth_estimator
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse({"status": "started"})


@router.post("/camera-stream/stop", summary="Stop live camera capture")
async def stop_camera() -> JSONResponse:
    """Stop the camera and release resources."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _camera_state.stop)
    return JSONResponse({"status": "stopped"})


@router.get("/camera-stream/status", summary="Camera status and FPS")
async def camera_status() -> JSONResponse:
    """Return whether the camera is running and the current FPS."""
    return JSONResponse(
        {
            "running": _camera_state.is_running,
            "fps": round(_camera_state.fps, 1),
            "camera_index": config.camera_index,
            "heatmap": _camera_state.show_heatmap,
        }
    )


@router.post("/camera-stream/toggle-heatmap", summary="Toggle heatmap overlay")
async def toggle_heatmap() -> JSONResponse:
    """Toggle the heatmap overlay on the camera stream."""
    _camera_state.show_heatmap = not _camera_state.show_heatmap
    return JSONResponse({"heatmap": _camera_state.show_heatmap})


@router.get("/camera-stream", summary="MJPEG live segmentation stream")
async def camera_stream(request: Request) -> StreamingResponse:
    """Stream annotated frames as ``multipart/x-mixed-replace`` (MJPEG).

    The stream continues until the client disconnects or the camera is
    stopped server-side.
    """
    # Auto-start the camera if it is not already running
    if not _camera_state.is_running:
        predictor = _get_predictor(request)
        depth_estimator = getattr(request.app.state, "depth_estimator", None)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, _camera_state.start, predictor, depth_estimator
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def _generate() -> AsyncGenerator[bytes, None]:
        while _camera_state.is_running:
            if await request.is_disconnected():
                break

            frame = _camera_state.get_annotated_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            jpeg = encode_jpeg(frame, config.jpeg_quality)
            yield build_mjpeg_frame(jpeg)

            # Throttle to avoid flooding the client
            await asyncio.sleep(1.0 / config.stream_fps_target)

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
