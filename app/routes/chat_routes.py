"""
Chat and LLM API routes.

Endpoints:
    POST /chat          – Send a chat message (non-streaming response).
    GET  /chat/stream   – WebSocket endpoint for streaming chat.
    GET  /chat/history  – Return recent commentary history.
    GET  /llm/status    – Check Ollama connection status.
    GET  /llm/models    – List available Ollama models.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.llm.ollama_client import OllamaClient
from app.llm.prompt_templates import build_chat_messages
from app.memory.scene_memory import SceneMemory

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialised by main.py lifespan)
_ollama_client: Optional[OllamaClient] = None
_scene_memory: Optional[SceneMemory] = None
_chat_history: List[Dict[str, str]] = []

MAX_CHAT_HISTORY = 50


def init_chat_routes(
    ollama_client: OllamaClient,
    scene_memory: SceneMemory,
) -> None:
    """Initialise module references (called during app startup)."""
    global _ollama_client, _scene_memory
    _ollama_client = ollama_client
    _scene_memory = scene_memory


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
    messages = build_chat_messages(
        user_query=req.message,
        scene_summary=scene_summary,
        chat_history=_chat_history,
    )

    reply = await _ollama_client.chat(
        messages=messages,
        model=req.model,
        temperature=req.temperature,
    )

    # Store in history
    _chat_history.append({"role": "user", "content": req.message})
    _chat_history.append({"role": "assistant", "content": reply})
    if len(_chat_history) > MAX_CHAT_HISTORY * 2:
        del _chat_history[: len(_chat_history) - MAX_CHAT_HISTORY * 2]

    return JSONResponse({
        "reply": reply,
        "timestamp": time.time(),
        "model": req.model or _ollama_client.model,
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

            message = payload.get("message", "")
            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            if _ollama_client is None or _scene_memory is None:
                await websocket.send_json(
                    {"error": "LLM service not available"}
                )
                continue

            scene_summary = _scene_memory.get_scene_summary()
            messages = build_chat_messages(
                user_query=message,
                scene_summary=scene_summary,
                chat_history=_chat_history,
            )

            full_reply = []
            async for token in _ollama_client.chat_stream(
                messages=messages,
                model=payload.get("model"),
                temperature=payload.get("temperature", 0.7),
            ):
                full_reply.append(token)
                await websocket.send_json({"token": token, "done": False})

            await websocket.send_json({"done": True})

            # Store in history
            reply_text = "".join(full_reply)
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
