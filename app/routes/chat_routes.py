"""
Chat and LLM API routes.

Endpoints:
    POST /chat              – Send a chat message (non-streaming response).
    WS   /chat/stream       – WebSocket endpoint for streaming chat.
    GET  /chat/history      – Return recent commentary history.
    GET  /llm/status        – Check Ollama connection status.
    GET  /llm/models        – List available Ollama models.
    GET  /llm/vision-models – List available vision models.
    GET  /llm/metrics       – Get LLM performance metrics.

Optimized for low-resource hardware with:
- Bounded async queue for LLM tasks
- Rate limiting (HTTP 429 when busy)
- Image size validation (HTTP 413 if too large)
- Scene-memory-first query routing
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import config
from app.llm.ollama_client import LLMClient, LLMErrorType
from app.llm.prompt_templates import (
    build_chat_messages,
    build_vision_messages,
    classify_query,
    build_deterministic_response,
)
from app.memory.scene_memory import SceneMemory
from app.routes.camera_routes import get_latest_frame_jpeg

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialised by main.py lifespan)
_llm_client: Optional[LLMClient] = None
_scene_memory: Optional[SceneMemory] = None
_chat_history: List[Dict[str, str]] = []
_chat_lock = asyncio.Lock()

# Bounded queue for LLM tasks
_llm_queue: Optional[asyncio.Queue] = None
_queue_worker_task: Optional[asyncio.Task] = None

MAX_CHAT_HISTORY = 50


@dataclass
class LLMTask:
    """Task to be processed by the LLM worker."""
    task_id: str
    messages: List[Dict[str, Any]]
    model: str
    temperature: float
    result_future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class QueueMetrics:
    """Metrics for the LLM queue."""
    total_queued: int = 0
    total_processed: int = 0
    total_rejected: int = 0
    current_size: int = 0


_queue_metrics = QueueMetrics()


def init_chat_routes(
    llm_client: LLMClient,
    scene_memory: SceneMemory,
) -> None:
    """Initialise module references (called during app startup)."""
    global _llm_client, _scene_memory, _llm_queue, _queue_worker_task
    _llm_client = llm_client
    _scene_memory = scene_memory

    # Initialize bounded queue
    _llm_queue = asyncio.Queue(maxsize=config.llm_max_queue_size)

    # Start worker task
    _queue_worker_task = asyncio.create_task(_llm_queue_worker())
    logger.info(
        "Chat routes initialized with queue size=%d, max_concurrent=%d",
        config.llm_max_queue_size,
        config.llm_max_concurrent,
    )


async def shutdown_chat_routes() -> None:
    """Shutdown the queue worker (called during app shutdown)."""
    global _queue_worker_task
    if _queue_worker_task:
        _queue_worker_task.cancel()
        try:
            await _queue_worker_task
        except asyncio.CancelledError:
            pass
        _queue_worker_task = None
    logger.info("Chat routes shutdown complete")


async def _llm_queue_worker() -> None:
    """Worker coroutine that processes LLM tasks from the queue."""
    logger.info("LLM queue worker started")
    while True:
        try:
            task: LLMTask = await _llm_queue.get()
            _queue_metrics.current_size = _llm_queue.qsize()

            start_time = time.time()
            request_id = task.task_id

            try:
                if _llm_client is None:
                    task.result_future.set_exception(
                        RuntimeError("LLM client not initialized")
                    )
                    continue

                logger.info(
                    "[%s] Processing LLM task, queue_size=%d, model=%s",
                    request_id,
                    _queue_metrics.current_size,
                    task.model,
                )

                response = await _llm_client.chat(
                    messages=task.messages,
                    model=task.model,
                    temperature=task.temperature,
                )

                latency_ms = (time.time() - start_time) * 1000
                logger.info(
                    "[%s] LLM task completed, latency_ms=%.2f, model=%s, fallback=%s",
                    request_id,
                    latency_ms,
                    response.model,
                    response.used_fallback,
                )

                task.result_future.set_result(response)
                _queue_metrics.total_processed += 1

            except Exception as e:
                logger.error("[%s] LLM task failed: %s", request_id, e)
                task.result_future.set_exception(e)

            finally:
                _llm_queue.task_done()
                _queue_metrics.current_size = _llm_queue.qsize()

        except asyncio.CancelledError:
            logger.info("LLM queue worker cancelled")
            break
        except Exception as e:
            logger.error("LLM queue worker error: %s", e)
            await asyncio.sleep(1)


async def _submit_llm_task(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    request_id: Optional[str] = None,
) -> Any:
    """Submit a task to the LLM queue and wait for result."""
    if _llm_queue is None:
        raise RuntimeError("LLM queue not initialized")

    task_id = request_id or str(uuid.uuid4())[:8]
    task = LLMTask(
        task_id=task_id,
        messages=messages,
        model=model,
        temperature=temperature,
        result_future=asyncio.get_event_loop().create_future(),
    )

    try:
        _llm_queue.put_nowait(task)
        _queue_metrics.total_queued += 1
        _queue_metrics.current_size = _llm_queue.qsize()
        logger.debug("[%s] Task queued, queue_size=%d", task_id, _queue_metrics.current_size)
    except asyncio.QueueFull:
        _queue_metrics.total_rejected += 1
        logger.warning("[%s] Queue full, rejecting task", task_id)
        raise

    return await task.result_future


def _compress_and_validate_frame(frame_jpeg: bytes) -> tuple[bytes, bool, int]:
    """Compress and validate frame size."""
    if _llm_client is None:
        return frame_jpeg, True, len(frame_jpeg) // 1024

    # Compress frame
    compressed = _llm_client.compress_image(
        frame_jpeg,
        max_width=config.chat_max_image_width,
        quality=config.chat_jpeg_quality,
    )

    # Validate size
    valid, size_kb = _llm_client.validate_image_size(compressed)
    return compressed, valid, size_kb


def _with_latest_frame_context(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool, Optional[int]]:
    """Attach the latest camera frame to the last user message when available."""
    if _llm_client is None:
        return messages, False, None

    frame_jpeg = get_latest_frame_jpeg()
    if frame_jpeg is None or not messages:
        return messages, False, None

    # Compress and validate
    compressed, valid, size_kb = _compress_and_validate_frame(frame_jpeg)
    if not valid:
        logger.warning("Frame too large after compression: %dKB", size_kb)
        return messages, False, size_kb

    enriched = list(messages)
    last_msg = dict(enriched[-1])
    images = list(last_msg.get("images", []))
    images.append(_llm_client.encode_image_bytes(compressed))
    last_msg["images"] = images
    enriched[-1] = last_msg
    return enriched, True, size_kb


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    model: Optional[str] = None
    temperature: float = 0.7
    include_frame: bool = True


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""
    reply: str
    timestamp: float
    model: str
    used_fallback: bool = False
    from_memory: bool = False
    queue_size: int = 0
    latency_ms: float = 0


@router.post("/chat", summary="Send a chat message")
async def chat_message(req: ChatRequest) -> JSONResponse:
    """
    Send a message to the LLM with scene context.

    Returns:
        - 200: Successful response
        - 413: Image payload too large
        - 429: LLM queue full, try again
        - 503: LLM service not available
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if _llm_client is None or _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "LLM service not initialised",
                "error_type": "SERVICE_UNAVAILABLE",
            },
        )

    scene_summary = _scene_memory.get_scene_summary()

    # Check if query can be answered from scene memory directly
    query_type, needs_llm = classify_query(req.message)
    if not needs_llm:
        deterministic_response = build_deterministic_response(query_type, scene_summary)
        if deterministic_response:
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "[%s] Query answered from memory, type=%s, latency_ms=%.2f",
                request_id,
                query_type,
                latency_ms,
            )
            return JSONResponse({
                "reply": deterministic_response,
                "timestamp": time.time(),
                "model": "scene_memory",
                "used_fallback": False,
                "from_memory": True,
                "queue_size": _queue_metrics.current_size,
                "latency_ms": round(latency_ms, 2),
            })

    async with _chat_lock:
        history_snapshot = list(_chat_history)

    messages = build_chat_messages(
        user_query=req.message,
        scene_summary=scene_summary,
        chat_history=history_snapshot,
    )

    # Optionally include live frame
    image_size_kb = None
    has_live_frame = False
    if req.include_frame:
        messages, has_live_frame, image_size_kb = _with_latest_frame_context(messages)

        # Check if frame was rejected due to size
        if image_size_kb and image_size_kb > config.ollama_max_image_kb:
            return JSONResponse(
                status_code=413,
                content={
                    "error": f"Image too large: {image_size_kb}KB exceeds {config.ollama_max_image_kb}KB limit",
                    "error_type": LLMErrorType.IMAGE_TOO_LARGE.value,
                    "image_size_kb": image_size_kb,
                    "max_size_kb": config.ollama_max_image_kb,
                },
            )

    selected_model = req.model or (
        config.ollama_vision_model if has_live_frame else _llm_client.model
    )

    # Submit to queue
    try:
        response = await _submit_llm_task(
            messages=messages,
            model=selected_model,
            temperature=req.temperature,
            request_id=request_id,
        )
    except asyncio.QueueFull:
        return JSONResponse(
            status_code=429,
            content={
                "error": "LLM busy, try again shortly",
                "error_type": "QUEUE_FULL",
                "queue_size": config.llm_max_queue_size,
                "retry_after_ms": 2000,
            },
        )

    # Store in history
    reply_content = response.content if hasattr(response, 'content') else str(response)
    async with _chat_lock:
        _chat_history.append({"role": "user", "content": req.message})
        _chat_history.append({"role": "assistant", "content": reply_content})
        if len(_chat_history) > MAX_CHAT_HISTORY * 2:
            del _chat_history[: len(_chat_history) - MAX_CHAT_HISTORY * 2]

    latency_ms = (time.time() - start_time) * 1000
    logger.info(
        "[%s] Chat completed, model=%s, latency_ms=%.2f, image_kb=%s",
        request_id,
        response.model if hasattr(response, 'model') else selected_model,
        latency_ms,
        image_size_kb,
    )

    return JSONResponse({
        "reply": reply_content,
        "timestamp": time.time(),
        "model": response.model if hasattr(response, 'model') else selected_model,
        "used_fallback": response.used_fallback if hasattr(response, 'used_fallback') else False,
        "from_memory": False,
        "queue_size": _queue_metrics.current_size,
        "latency_ms": round(latency_ms, 2),
    })


@router.websocket("/chat/stream")
async def chat_stream_ws(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming LLM chat responses.

    Client sends JSON: {"message": "...", "model": "...", "temperature": 0.7}
    Server streams JSON events:
        - {"type": "chat_start", "id": "...", "model": "..."}
        - {"type": "chat_token", "id": "...", "token": "..."}
        - {"type": "chat_end", "id": "..."}
        - {"type": "chat_error", "error": "...", "error_type": "..."}
        - {"type": "queue_status", "queue_size": N, "status": "queued"|"processing"}
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "chat_error",
                    "error": "Invalid JSON",
                    "error_type": "INVALID_REQUEST",
                })
                continue

            message = str(payload.get("message", "")).strip()
            if not message:
                await websocket.send_json({
                    "type": "chat_error",
                    "error": "Empty message",
                    "error_type": "INVALID_REQUEST",
                })
                continue

            if _llm_client is None or _scene_memory is None:
                await websocket.send_json({
                    "type": "chat_error",
                    "error": "LLM service not available",
                    "error_type": "SERVICE_UNAVAILABLE",
                })
                continue

            # Check queue status
            if _llm_queue and _llm_queue.full():
                await websocket.send_json({
                    "type": "chat_error",
                    "error": "LLM busy, try again shortly",
                    "error_type": "QUEUE_FULL",
                    "queue_size": config.llm_max_queue_size,
                })
                continue

            scene_summary = _scene_memory.get_scene_summary()

            # Check if query can be answered from memory
            query_type, needs_llm = classify_query(message)
            if not needs_llm:
                deterministic_response = build_deterministic_response(query_type, scene_summary)
                if deterministic_response:
                    stream_id = f"memory-{int(time.time() * 1000)}"
                    await websocket.send_json({
                        "type": "chat_start",
                        "id": stream_id,
                        "model": "scene_memory",
                        "from_memory": True,
                    })
                    await websocket.send_json({
                        "type": "chat_token",
                        "id": stream_id,
                        "token": deterministic_response,
                    })
                    await websocket.send_json({
                        "type": "chat_end",
                        "id": stream_id,
                    })
                    continue

            async with _chat_lock:
                history_snapshot = list(_chat_history)

            messages = build_chat_messages(
                user_query=message,
                scene_summary=scene_summary,
                chat_history=history_snapshot,
            )
            messages, has_live_frame, image_size_kb = _with_latest_frame_context(messages)

            # Check image size
            if image_size_kb and image_size_kb > config.ollama_max_image_kb:
                await websocket.send_json({
                    "type": "chat_error",
                    "error": f"Image too large: {image_size_kb}KB",
                    "error_type": LLMErrorType.IMAGE_TOO_LARGE.value,
                })
                continue

            selected_model = payload.get("model") or (
                config.ollama_vision_model if has_live_frame else _llm_client.model
            )

            stream_id = f"chat-{int(time.time() * 1000)}"
            await websocket.send_json({
                "type": "chat_start",
                "id": stream_id,
                "model": selected_model,
                "queue_size": _queue_metrics.current_size,
            })

            full_reply = []
            async for token in _llm_client.chat_stream(
                messages=messages,
                model=selected_model,
                temperature=payload.get("temperature", 0.7),
            ):
                full_reply.append(token)
                await websocket.send_json({
                    "type": "chat_token",
                    "id": stream_id,
                    "token": token,
                })

            await websocket.send_json({
                "type": "chat_end",
                "id": stream_id,
            })

            # Store in history
            reply_text = "".join(full_reply)
            async with _chat_lock:
                _chat_history.append({"role": "user", "content": message})
                _chat_history.append({"role": "assistant", "content": reply_text})
                if len(_chat_history) > MAX_CHAT_HISTORY * 2:
                    del _chat_history[: len(_chat_history) - MAX_CHAT_HISTORY * 2]

    except WebSocketDisconnect:
        logger.debug("Chat WebSocket disconnected")
    except Exception as exc:
        logger.error("Chat WebSocket error: %s", exc)


@router.get("/llm/status", summary="LLM connection status")
async def llm_status() -> JSONResponse:
    """Check Ollama connection status and queue metrics."""
    if _llm_client is None:
        return JSONResponse({
            "available": False,
            "error": "Not initialised",
        })

    available = await _llm_client.check_health()
    metrics = _llm_client.get_metrics()

    return JSONResponse({
        "available": available,
        "base_url": _llm_client.base_url,
        "model": _llm_client.model,
        "vision_model": _llm_client.vision_model,
        "fallback_model": _llm_client.fallback_model,
        "queue": {
            "size": _queue_metrics.current_size,
            "max_size": config.llm_max_queue_size,
            "total_queued": _queue_metrics.total_queued,
            "total_processed": _queue_metrics.total_processed,
            "total_rejected": _queue_metrics.total_rejected,
        },
        "metrics": metrics,
    })


@router.get("/llm/models", summary="List available LLM models")
async def llm_models() -> JSONResponse:
    """List models available on the Ollama server."""
    if _llm_client is None:
        return JSONResponse({"models": []})

    models = await _llm_client.list_models()
    return JSONResponse({
        "models": [
            {"name": m.get("name", ""), "size": m.get("size", 0)}
            for m in models
        ]
    })


@router.get("/llm/vision-models", summary="List available vision models")
async def llm_vision_models() -> JSONResponse:
    """List vision-capable models available on the Ollama server."""
    if _llm_client is None:
        return JSONResponse({
            "models": [],
            "default": config.ollama_vision_model,
        })

    models = await _llm_client.list_vision_models()
    return JSONResponse({
        "models": [
            {"name": m.get("name", ""), "size": m.get("size", 0)}
            for m in models
        ],
        "default": config.ollama_vision_model,
        "fallback": config.llm_fallback_model,
    })


@router.get("/llm/metrics", summary="Get LLM performance metrics")
async def llm_metrics() -> JSONResponse:
    """Get detailed LLM performance metrics for observability."""
    if _llm_client is None:
        return JSONResponse({
            "available": False,
            "error": "Not initialised",
        })

    metrics = _llm_client.get_metrics()
    return JSONResponse({
        "llm": metrics,
        "queue": {
            "current_size": _queue_metrics.current_size,
            "max_size": config.llm_max_queue_size,
            "total_queued": _queue_metrics.total_queued,
            "total_processed": _queue_metrics.total_processed,
            "total_rejected": _queue_metrics.total_rejected,
        },
    })


# Backward compatibility alias
OllamaClient = LLMClient
