"""
Video upload and processing API routes.

Endpoints:
    POST /upload-video  – Upload a video file, receive a job ID.
    POST /process-video – Trigger processing of an uploaded video.
    GET  /job-status/{job_id} – Poll processing progress.
    GET  /download/{job_id}   – Download the processed output video.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import config
from app.llm.prompt_templates import build_video_chat_messages
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


def _get_job_or_404(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _keyframe_public_url(path: str) -> str:
    rel = os.path.relpath(path, config.output_dir)
    rel = rel.replace(os.sep, "/")
    return f"/outputs/{rel}"


def _load_video_analysis(job: dict) -> dict:
    analysis_path = job.get("analysis_path")
    if not analysis_path:
        raise HTTPException(
            status_code=409,
            detail="Video analysis is not available for this job.",
        )
    if not os.path.isfile(analysis_path):
        raise HTTPException(status_code=404, detail="Video analysis artifact not found.")
    with open(analysis_path, "r", encoding="utf-8") as analysis_file:
        return json.load(analysis_file)


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
        "analysis_path": None,
        "keyframes_dir": None,
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
    analysis_filename = f"{job_id}_analysis.json"
    analysis_path = os.path.join(config.output_dir, analysis_filename)
    keyframes_dir = os.path.join(config.output_dir, f"{job_id}_keyframes")

    job["status"] = "processing"
    job["output_path"] = output_path
    job["analysis_path"] = analysis_path
    job["keyframes_dir"] = keyframes_dir

    def _progress_cb(done: int, total: int, fps: float) -> None:
        job["progress"] = done
        job["total_frames"] = total
        job["fps"] = round(fps, 1)

    async def _run() -> None:
        try:
            stats = await processor.process_video_async(
                input_path=input_path,
                output_path=output_path,
                progress_callback=_progress_cb,
                analysis_output_path=analysis_path,
                keyframes_dir=keyframes_dir,
                keyframe_interval_seconds=config.video_keyframe_interval_seconds,
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
    job = _get_job_or_404(job_id)

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
        if job.get("analysis_path"):
            payload["analysis_url"] = f"/api/video-analysis/{job_id}"

    return JSONResponse(payload)


@router.get("/download/{job_id}", summary="Download the processed output video")
async def download_video(job_id: str) -> FileResponse:
    """Stream the processed video file to the client."""
    job = _get_job_or_404(job_id)
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


@router.get("/video-analysis/{job_id}", summary="Get processed video analysis artifact")
async def video_analysis(job_id: str) -> JSONResponse:
    """Return structured analysis for a completed processed video job."""
    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video analysis is not ready yet (status: {job['status']}).",
        )

    analysis = _load_video_analysis(job)
    keyframes = analysis.get("keyframes", [])
    keyframes_payload: List[Dict[str, Any]] = []
    for item in keyframes:
        path = str(item.get("path", "")).strip()
        with_url = dict(item)
        if path:
            with_url["url"] = _keyframe_public_url(path)
        keyframes_payload.append(with_url)

    analysis["keyframes"] = keyframes_payload
    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "analysis": analysis,
        }
    )


class VideoChatRequest(BaseModel):
    """Request body for processed-video chat."""

    message: str
    model: str | None = None
    temperature: float = 0.5
    max_keyframes: int = 3


@router.post("/video-chat/{job_id}", summary="Ask a question about a processed video")
async def video_chat(job_id: str, req: VideoChatRequest, request: Request) -> JSONResponse:
    """Answer natural language questions using saved analysis and sampled keyframes."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    job = _get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video chat is only available after processing is done (status: {job['status']}).",
        )

    ollama_client = getattr(request.app.state, "ollama_client", None)
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="LLM service not initialised.")

    analysis = _load_video_analysis(job)
    messages = build_video_chat_messages(message, analysis)

    max_keyframes = max(0, min(req.max_keyframes, 6))
    encoded_keyframes: List[str] = []
    for item in analysis.get("keyframes", [])[:max_keyframes]:
        keyframe_path = str(item.get("path", "")).strip()
        if keyframe_path and os.path.isfile(keyframe_path):
            with open(keyframe_path, "rb") as keyframe_file:
                encoded_keyframes.append(
                    ollama_client.encode_image_bytes(keyframe_file.read())
                )

    if encoded_keyframes:
        enriched_messages = list(messages)
        final_msg = dict(enriched_messages[-1])
        final_msg["images"] = encoded_keyframes
        enriched_messages[-1] = final_msg
        messages = enriched_messages

    selected_model = req.model or (
        config.ollama_vision_model if encoded_keyframes else ollama_client.model
    )
    reply = await ollama_client.chat(
        messages=messages,
        model=selected_model,
        temperature=req.temperature,
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "reply": reply,
            "model": selected_model,
            "used_keyframes": len(encoded_keyframes),
        }
    )
