"""
Event logger module.

Records object entry/exit events and provides CSV export capability for
downstream analysis.
"""

import csv
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.inference.tracker import TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class ObjectEvent:
    """A single entry or exit event."""

    timestamp: float
    event_type: str  # "entry" or "exit"
    track_id: int
    class_name: str
    centroid_x: float
    centroid_y: float


class EventLogger:
    """Tracks object entry/exit events and supports CSV export.

    Detects new objects appearing (entry) and previously tracked objects
    disappearing (exit) between consecutive frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[ObjectEvent] = []
        self._active_ids: Set[int] = set()
        self._last_objects: Dict[int, TrackedObject] = {}

    def update(self, tracked_objects: List[TrackedObject]) -> List[ObjectEvent]:
        """Compare current tracked objects to previous frame and log events.

        Args:
            tracked_objects: Currently tracked objects.

        Returns:
            List of events generated in this frame.
        """
        current_ids = {obj.track_id for obj in tracked_objects}
        current_map = {obj.track_id: obj for obj in tracked_objects}
        now = time.time()
        new_events: List[ObjectEvent] = []

        with self._lock:
            # Entry events – IDs present now but not before
            for tid in current_ids - self._active_ids:
                obj = current_map[tid]
                event = ObjectEvent(
                    timestamp=now,
                    event_type="entry",
                    track_id=tid,
                    class_name=obj.class_name,
                    centroid_x=obj.centroid[0],
                    centroid_y=obj.centroid[1],
                )
                self._events.append(event)
                new_events.append(event)

            # Exit events – IDs that were active but are now gone
            for tid in self._active_ids - current_ids:
                prev = self._last_objects.get(tid)
                if prev is not None:
                    event = ObjectEvent(
                        timestamp=now,
                        event_type="exit",
                        track_id=tid,
                        class_name=prev.class_name,
                        centroid_x=prev.centroid[0],
                        centroid_y=prev.centroid[1],
                    )
                    self._events.append(event)
                    new_events.append(event)

            self._active_ids = current_ids
            self._last_objects = current_map

        if new_events:
            logger.debug("Logged %d events (frame)", len(new_events))

        return new_events

    def get_events(self, limit: Optional[int] = None) -> List[ObjectEvent]:
        """Return recorded events, newest first.

        Args:
            limit: Maximum number of events to return.
        """
        with self._lock:
            events = list(reversed(self._events))
        if limit is not None:
            events = events[:limit]
        return events

    def export_csv(self) -> str:
        """Export all recorded events as a CSV string.

        Returns:
            CSV-formatted string with headers.
        """
        with self._lock:
            events = list(self._events)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "event_type", "track_id",
            "class_name", "centroid_x", "centroid_y",
        ])
        for evt in events:
            writer.writerow([
                evt.timestamp,
                evt.event_type,
                evt.track_id,
                evt.class_name,
                round(evt.centroid_x, 1),
                round(evt.centroid_y, 1),
            ])
        return output.getvalue()

    def reset(self) -> None:
        """Clear all logged events."""
        with self._lock:
            self._events.clear()
            self._active_ids.clear()
            self._last_objects.clear()
        logger.info("Event logger reset")
