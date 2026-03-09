"""
Scene state and memory API routes.

Endpoints:
    GET  /scene/state       – Current scene state (active objects).
    GET  /scene/events      – Recent scene events.
    GET  /scene/summary     – Full scene summary for dashboards.
    GET  /scene/history     – Object history with optional filters.
    GET  /scene/time-summary – Activity summary for a time range.
    GET  /scene/commentary  – Recent auto-generated commentary.
    WS   /ws/events         – WebSocket for real-time event streaming.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.memory.scene_memory import SceneMemory
from app.services.commentary_engine import CommentaryEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons
_scene_memory: Optional[SceneMemory] = None
_commentary_engine: Optional[CommentaryEngine] = None

# Active WebSocket connections for event broadcasting
_event_subscribers: Set[WebSocket] = set()
_subscribers_lock = asyncio.Lock()


def init_scene_routes(
    scene_memory: SceneMemory,
    commentary_engine: Optional[CommentaryEngine] = None,
) -> None:
    """Initialise module references (called during app startup)."""
    global _scene_memory, _commentary_engine
    _scene_memory = scene_memory
    _commentary_engine = commentary_engine


async def broadcast_event(event: Dict[str, Any]) -> None:
    """Broadcast an event to all connected WebSocket subscribers."""
    async with _subscribers_lock:
        subscribers = set(_event_subscribers)
    if not subscribers:
        return
    message = json.dumps(event)
    disconnected: List[WebSocket] = []
    for ws in subscribers:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    if disconnected:
        async with _subscribers_lock:
            for ws in disconnected:
                _event_subscribers.discard(ws)


@router.get("/scene/state", summary="Current scene state")
async def scene_state() -> JSONResponse:
    """Return the current scene state with active objects."""
    if _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Scene memory not initialised"},
        )
    return JSONResponse(_scene_memory.get_scene_state())


@router.get("/scene/events", summary="Recent scene events")
async def scene_events(
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    """Return recent scene events."""
    if _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Scene memory not initialised"},
        )
    return JSONResponse(_scene_memory.get_recent_events(limit=limit))


@router.get("/scene/summary", summary="Scene summary")
async def scene_summary() -> JSONResponse:
    """Return a comprehensive scene summary."""
    if _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Scene memory not initialised"},
        )
    return JSONResponse(_scene_memory.get_scene_summary())


@router.get("/scene/history", summary="Object history")
async def scene_history(
    class_filter: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> JSONResponse:
    """Return object history with optional class filter."""
    if _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Scene memory not initialised"},
        )
    return JSONResponse(
        _scene_memory.get_object_history(
            class_filter=class_filter, limit=limit
        )
    )


@router.get("/scene/time-summary", summary="Time-based activity summary")
async def scene_time_summary(
    start: Optional[float] = Query(None, description="Unix timestamp start"),
    end: Optional[float] = Query(None, description="Unix timestamp end"),
    minutes: Optional[int] = Query(
        None, description="Minutes ago (alternative to start/end)"
    ),
) -> JSONResponse:
    """Return an activity summary for a time range."""
    if _scene_memory is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Scene memory not initialised"},
        )
    start_time = start
    end_time = end
    if minutes is not None:
        end_time = time.time()
        start_time = end_time - (minutes * 60)
    return JSONResponse(
        _scene_memory.get_time_summary(
            start_time=start_time, end_time=end_time
        )
    )


@router.get("/scene/commentary", summary="Recent commentary")
async def scene_commentary(
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    """Return recent auto-generated commentary."""
    if _commentary_engine is None:
        return JSONResponse({"commentary": []})
    return JSONResponse({
        "commentary": _commentary_engine.get_recent_commentary(limit=limit)
    })


@router.websocket("/ws/events")
async def events_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming.

    Connected clients receive JSON events as they occur.
    """
    await websocket.accept()
    async with _subscribers_lock:
        _event_subscribers.add(websocket)
        count = len(_event_subscribers)
    logger.debug("Event WebSocket connected (%d total)", count)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _subscribers_lock:
            _event_subscribers.discard(websocket)
            remaining = len(_event_subscribers)
        logger.debug(
            "Event WebSocket disconnected (%d remaining)",
            remaining,
        )
