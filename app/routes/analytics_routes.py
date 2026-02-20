"""
Analytics API routes.

Endpoints:
    GET  /analytics/live   – Return real-time object counts and stats.
    GET  /analytics/export – Export event log as CSV download.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.analytics.event_logger import EventLogger
from app.analytics.object_counter import ObjectCounter

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons shared with camera/video routes
_object_counter = ObjectCounter()
_event_logger = EventLogger()


def get_object_counter() -> ObjectCounter:
    """Return the shared object counter instance."""
    return _object_counter


def get_event_logger() -> EventLogger:
    """Return the shared event logger instance."""
    return _event_logger


@router.get("/analytics/live", summary="Live analytics snapshot")
async def analytics_live() -> JSONResponse:
    """Return current object counts, rolling averages, and FPS."""
    stats = _object_counter.get_live_stats()
    recent_events = _event_logger.get_events(limit=20)
    stats["recent_events"] = [
        {
            "timestamp": evt.timestamp,
            "event_type": evt.event_type,
            "track_id": evt.track_id,
            "class_name": evt.class_name,
        }
        for evt in recent_events
    ]
    return JSONResponse(stats)


@router.get("/analytics/export", summary="Export events as CSV")
async def analytics_export() -> Response:
    """Download all object entry/exit events as a CSV file."""
    csv_data = _event_logger.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )
