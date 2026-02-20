"""
Video upload and processing API routes.

Endpoints:
    POST /upload-video  – Upload a video file, receive a job ID.
    POST /process-video – Trigger processing of an uploaded video.
    GET  /job-status/{job_id} – Poll processing progress.
    GET  /download/{job_id}   – Download the processed output video.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Dict

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import config
from app.inference.panoptic_predictor import PanopticPredictor
from app.inference.video_processor import VideoProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store (sufficient for a single-server deployment)
_jobs: Dict[str, dict] = {}


def _get_predictor(request: Request) -> PanopticPredictor:
    """Retrieve the shared predictor from app state."""
    loader = request.app.state.model_loader
    return PanopticPredictor(loader)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_filename(filename: str) -> str:
    """Return a sanitised base filename (no path components)."""
    return os.path.basename(filename).replace(" ", "_")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload-video", summary="Upload a video for segmentation")
async def upload_video(file: UploadFile) -> JSONResponse:
    """Accept a video upload and store it on disk.

    Returns a ``job_id`` that can be used with ``/process-video``.

    - Supported formats: ``.mp4``, ``.avi``, ``.mov``, ``.mkv``
    - Maximum size: ``MAX_UPLOAD_SIZE_MB`` (default 500 MB)
    """
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in config.allowed_video_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Allowed: {config.allowed_video_extensions}",
        )

    job_id = str(uuid.uuid4())
    safe_name = _safe_filename(file.filename or f"video{ext}")
    input_path = os.path.join(config.upload_dir, f"{job_id}_{safe_name}")

    # Stream to disk with a size limit
    max_bytes = config.max_upload_size_mb * 1024 * 1024
    written = 0
    try:
        async with aiofiles.open(input_path, "wb") as out_f:
            while chunk := await file.read(1024 * 256):  # 256 KB chunks
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {config.max_upload_size_mb} MB.",
                    )
                await out_f.write(chunk)
    except HTTPException:
        # Clean up partial upload
        if os.path.exists(input_path):
            os.remove(input_path)
        raise

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "uploaded",
        "input_path": input_path,
        "output_path": None,
        "progress": 0,
        "total_frames": 0,
        "fps": 0.0,
        "error": None,
        "created_at": time.time(),
    }

    logger.info("Uploaded '%s' → job %s", safe_name, job_id)
    return JSONResponse({"job_id": job_id, "filename": safe_name, "size_bytes": written})


@router.post("/process-video/{job_id}", summary="Start processing an uploaded video")
async def process_video(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
) -> JSONResponse:
    """Kick off background panoptic segmentation for a previously uploaded video.

    Returns immediately; poll ``/job-status/{job_id}`` for progress.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] not in ("uploaded", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already in state '{job['status']}'.",
        )

    predictor = _get_predictor(request)
    depth_estimator = getattr(request.app.state, "depth_estimator", None)
    processor = VideoProcessor(
        predictor,
        enable_tracking=True,
        depth_estimator=depth_estimator,
    )

    input_path = job["input_path"]
    output_filename = f"{job_id}_output.mp4"
    output_path = os.path.join(config.output_dir, output_filename)

    job["status"] = "processing"
    job["output_path"] = output_path

    def _progress_cb(done: int, total: int, fps: float) -> None:
        job["progress"] = done
        job["total_frames"] = total
        job["fps"] = round(fps, 1)

    async def _run() -> None:
        try:
            stats = await processor.process_video_async(
                input_path, output_path, _progress_cb
            )
            job["status"] = "done"
            job["stats"] = stats
            logger.info("Job %s complete: %s", job_id, stats)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            logger.exception("Job %s failed: %s", job_id, exc)

    background_tasks.add_task(_run)
    return JSONResponse({"job_id": job_id, "status": "processing"})


@router.get("/job-status/{job_id}", summary="Get processing job status")
async def job_status(job_id: str) -> JSONResponse:
    """Return current status and progress of a processing job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    payload = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total_frames": job["total_frames"],
        "fps": job["fps"],
        "error": job["error"],
    }
    if job["status"] == "done":
        payload["download_url"] = f"/api/download/{job_id}"
        payload["stats"] = job.get("stats", {})

    return JSONResponse(payload)


@router.get("/download/{job_id}", summary="Download the processed output video")
async def download_video(job_id: str) -> FileResponse:
    """Stream the processed video file to the client."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready yet (status: {job['status']}).",
        )

    output_path = job["output_path"]
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk.")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=os.path.basename(output_path),
    )
