"""
Chat and LLM API routes.

Endpoints:
    POST /chat          – Send a chat message (non-streaming response).
    GET  /chat/stream   – WebSocket endpoint for streaming chat.
    GET  /chat/history  – Return recent commentary history.
    GET  /llm/status    – Check Ollama connection status.
    GET  /llm/models    – List available Ollama models.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import config
from app.llm.ollama_client import OllamaClient
from app.llm.prompt_templates import build_chat_messages
from app.memory.scene_memory import SceneMemory
from app.routes.camera_routes import get_latest_frame_jpeg

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialised by main.py lifespan)
_ollama_client: Optional[OllamaClient] = None
_scene_memory: Optional[SceneMemory] = None
_chat_history: List[Dict[str, str]] = []
_chat_lock = asyncio.Lock()

MAX_CHAT_HISTORY = 50


def init_chat_routes(
    ollama_client: OllamaClient,
    scene_memory: SceneMemory,
) -> None:
    """Initialise module references (called during app startup)."""
    global _ollama_client, _scene_memory
    _ollama_client = ollama_client
    _scene_memory = scene_memory


def _with_latest_frame_context(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
    """Attach the latest camera frame to the last user message when available."""
    if _ollama_client is None:
        return messages, False
    frame_jpeg = get_latest_frame_jpeg()
    if frame_jpeg is None or not messages:
        return messages, False

    enriched = list(messages)
    last_msg = dict(enriched[-1])
    images = list(last_msg.get("images", []))
    images.append(_ollama_client.encode_image_bytes(frame_jpeg))
    last_msg["images"] = images
    enriched[-1] = last_msg
    return enriched, True


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    model: Optional[str] = None
    temperature: float = 0.7


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""
    reply: str
    timestamp: float
    model: str


@router.post("/chat", summary="Send a chat message")
async def chat_message(req: ChatRequest) -> JSONResponse:
    """Send a message to the LLM with scene context."""
    if _ollama_client is None or _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM service not initialised"},
        )

    scene_summary = _scene_memory.get_scene_summary()

    async with _chat_lock:
        history_snapshot = list(_chat_history)

    messages = build_chat_messages(
        user_query=req.message,
        scene_summary=scene_summary,
        chat_history=history_snapshot,
    )
    messages, has_live_frame = _with_latest_frame_context(messages)
    selected_model = req.model or (
        config.ollama_vision_model if has_live_frame else _ollama_client.model
    )

    reply = await _ollama_client.chat(
        messages=messages,
        model=selected_model,
        temperature=req.temperature,
    )

    # Store in history
    async with _chat_lock:
        _chat_history.append({"role": "user", "content": req.message})
        _chat_history.append({"role": "assistant", "content": reply})
        if len(_chat_history) > MAX_CHAT_HISTORY * 2:
            del _chat_history[: len(_chat_history) - MAX_CHAT_HISTORY * 2]

    return JSONResponse({
        "reply": reply,
        "timestamp": time.time(),
        "model": selected_model,
    })


@router.websocket("/chat/stream")
async def chat_stream_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming LLM chat responses.

    Client sends JSON: {"message": "...", "model": "...", "temperature": 0.7}
    Server streams JSON: {"token": "...", "done": false} and {"done": true}
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            message = str(payload.get("message", "")).strip()
            if not message:
                await websocket.send_json({
                    "type": "chat_error",
                    "error": "Empty message",
                })
                continue

            if _ollama_client is None or _scene_memory is None:
                await websocket.send_json(
                    {"type": "chat_error", "error": "LLM service not available"}
                )
                continue

            scene_summary = _scene_memory.get_scene_summary()

            async with _chat_lock:
                history_snapshot = list(_chat_history)

            messages = build_chat_messages(
                user_query=message,
                scene_summary=scene_summary,
                chat_history=history_snapshot,
            )
            messages, has_live_frame = _with_latest_frame_context(messages)
            selected_model = payload.get("model") or (
                config.ollama_vision_model if has_live_frame else _ollama_client.model
            )

            stream_id = f"chat-{int(time.time() * 1000)}"
            await websocket.send_json({
                "type": "chat_start",
                "id": stream_id,
                "model": selected_model,
            })

            full_reply = []
            async for token in _ollama_client.chat_stream(
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

            await websocket.send_json({"type": "chat_end", "id": stream_id})

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
    """Check Ollama connection status."""
    if _ollama_client is None:
        return JSONResponse({"available": False, "error": "Not initialised"})

    available = await _ollama_client.check_health()
    return JSONResponse({
        "available": available,
        "base_url": _ollama_client.base_url,
        "model": _ollama_client.model,
    })


@router.get("/llm/models", summary="List available LLM models")
async def llm_models() -> JSONResponse:
    """List models available on the Ollama server."""
    if _ollama_client is None:
        return JSONResponse({"models": []})

    models = await _ollama_client.list_models()
    return JSONResponse({
        "models": [
            {"name": m.get("name", ""), "size": m.get("size", 0)}
            for m in models
        ]
    })
