"""
Multi-camera management API routes.

Endpoints:
    POST /camera/start  – Start a new camera stream.
    POST /camera/stop   – Stop a camera stream.
    GET  /camera/list   – List all active camera streams.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.inference.panoptic_predictor import PanopticPredictor
from app.streams.camera_manager import CameraManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singleton
_camera_manager = CameraManager()


def get_camera_manager() -> CameraManager:
    """Return the shared camera manager instance."""
    return _camera_manager


class StartCameraRequest(BaseModel):
    camera_index: int = Field(default=0, description="OpenCV camera device index")
    label: str = Field(default="", description="Optional label for the camera")


class StopCameraRequest(BaseModel):
    stream_id: str = Field(..., description="Stream ID to stop")


def _get_predictor(request: Request) -> PanopticPredictor:
    loader = request.app.state.model_loader
    return PanopticPredictor(loader)


@router.post("/camera/start", summary="Start a new camera stream")
async def start_camera_stream(
    body: StartCameraRequest,
    request: Request,
) -> JSONResponse:
    """Open a camera and begin real-time segmentation on a new stream."""
    predictor = _get_predictor(request)
    try:
        stream_id = _camera_manager.start_stream(
            camera_index=body.camera_index,
            predictor=predictor,
            label=body.label,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse({"stream_id": stream_id, "status": "started"})


@router.post("/camera/stop", summary="Stop a camera stream")
async def stop_camera_stream(body: StopCameraRequest) -> JSONResponse:
    """Stop an active camera stream."""
    stopped = _camera_manager.stop_stream(body.stream_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Stream not found")
    return JSONResponse({"stream_id": body.stream_id, "status": "stopped"})


@router.get("/camera/list", summary="List active camera streams")
async def list_camera_streams() -> JSONResponse:
    """Return metadata for all active camera streams."""
    streams = _camera_manager.list_streams()
    return JSONResponse({
        "streams": [
            {
                "stream_id": s.stream_id,
                "camera_index": s.camera_index,
                "label": s.label,
                "running": s.running,
                "fps": s.fps,
                "object_count": s.object_count,
            }
            for s in streams
        ]
    })
