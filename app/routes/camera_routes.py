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

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self, predictor: PanopticPredictor) -> None:
        """Start the background capture + inference loop."""
        with self._lock:
            if self._running:
                return
            self._predictor = predictor
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
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _camera_state.start, predictor)
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
        }
    )


@router.get("/camera-stream", summary="MJPEG live segmentation stream")
async def camera_stream(request: Request) -> StreamingResponse:
    """Stream annotated frames as ``multipart/x-mixed-replace`` (MJPEG).

    The stream continues until the client disconnects or the camera is
    stopped server-side.
    """
    # Auto-start the camera if it is not already running
    if not _camera_state.is_running:
        predictor = _get_predictor(request)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _camera_state.start, predictor)
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
