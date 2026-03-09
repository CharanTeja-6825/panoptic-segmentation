"""
Scene memory module.

Provides short-term buffer and long-term event storage for the vision
pipeline, enabling queryable scene state and time-based summaries.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SceneObject:
    """Represents a single detected object in the scene."""

    track_id: int
    class_name: str
    confidence: float
    bbox: tuple
    centroid: tuple
    first_seen: float
    last_seen: float
    status: str = "active"  # "active" or "exited"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneEvent:
    """A structured scene event for the memory store."""

    timestamp: float
    event_type: str  # "entry", "exit", "movement", "crowd", "idle"
    description: str
    objects_involved: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSnapshot:
    """Point-in-time scene summary."""

    timestamp: float
    total_objects: int
    counts_by_class: Dict[str, int]
    active_track_ids: List[int]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scene Memory
# ---------------------------------------------------------------------------


class SceneMemory:
    """Manages short-term and long-term scene memory.

    Short-term: rolling buffer of recent snapshots and events.
    Long-term: full event log and object history.
    """

    def __init__(
        self,
        short_term_capacity: int = 300,
        snapshot_interval: float = 5.0,
    ) -> None:
        self._lock = threading.Lock()

        # Short-term memory (rolling buffer)
        self._short_term_events: Deque[SceneEvent] = deque(
            maxlen=short_term_capacity
        )
        self._short_term_snapshots: Deque[SceneSnapshot] = deque(
            maxlen=short_term_capacity
        )

        # Long-term memory
        self._all_events: List[SceneEvent] = []
        self._object_history: Dict[int, SceneObject] = {}
        self._class_totals: Dict[str, int] = defaultdict(int)

        # Active scene state
        self._active_objects: Dict[int, SceneObject] = {}

        # Snapshot timing
        self._snapshot_interval = snapshot_interval
        self._last_snapshot_time: float = 0.0

    # ---- Scene state updates ----

    def update_scene(
        self,
        tracked_objects: list,
        fps: float = 0.0,
    ) -> List[SceneEvent]:
        """Update scene memory with current tracked objects.

        Args:
            tracked_objects: List of TrackedObject from the tracker.
            fps: Current processing FPS.

        Returns:
            List of new events generated.
        """
        now = time.time()
        new_events: List[SceneEvent] = []
        current_ids = set()

        with self._lock:
            for obj in tracked_objects:
                tid = obj.track_id
                current_ids.add(tid)

                if tid not in self._active_objects:
                    # New object entry
                    scene_obj = SceneObject(
                        track_id=tid,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=obj.bbox,
                        centroid=obj.centroid,
                        first_seen=now,
                        last_seen=now,
                    )
                    self._active_objects[tid] = scene_obj
                    self._object_history[tid] = scene_obj
                    self._class_totals[obj.class_name] += 1

                    event = SceneEvent(
                        timestamp=now,
                        event_type="entry",
                        description=f"{obj.class_name} entered the scene (track {tid})",
                        objects_involved=[tid],
                        metadata={
                            "class_name": obj.class_name,
                            "confidence": round(obj.confidence, 2),
                        },
                    )
                    new_events.append(event)
                else:
                    # Update existing object
                    existing = self._active_objects[tid]
                    existing.bbox = obj.bbox
                    existing.centroid = obj.centroid
                    existing.confidence = obj.confidence
                    existing.last_seen = now

            # Detect exits
            exited_ids = set(self._active_objects.keys()) - current_ids
            for tid in exited_ids:
                exited_obj = self._active_objects.pop(tid)
                exited_obj.status = "exited"
                exited_obj.last_seen = now
                if tid in self._object_history:
                    self._object_history[tid].status = "exited"
                    self._object_history[tid].last_seen = now

                duration = round(now - exited_obj.first_seen, 1)
                event = SceneEvent(
                    timestamp=now,
                    event_type="exit",
                    description=(
                        f"{exited_obj.class_name} exited the scene "
                        f"(track {tid}, was present {duration}s)"
                    ),
                    objects_involved=[tid],
                    metadata={
                        "class_name": exited_obj.class_name,
                        "duration": duration,
                    },
                )
                new_events.append(event)

            # Store events
            for evt in new_events:
                self._short_term_events.append(evt)
                self._all_events.append(evt)

            # Periodic snapshot
            if now - self._last_snapshot_time >= self._snapshot_interval:
                self._take_snapshot(now, tracked_objects)
                self._last_snapshot_time = now

        return new_events

    def _take_snapshot(self, now: float, tracked_objects: list) -> None:
        """Create a point-in-time scene snapshot (must hold lock)."""
        counts: Dict[str, int] = defaultdict(int)
        for obj in tracked_objects:
            counts[obj.class_name] += 1

        parts = [f"{cnt} {cls}" for cls, cnt in sorted(counts.items())]
        summary = f"Scene contains {', '.join(parts) or 'no objects'}"

        snapshot = SceneSnapshot(
            timestamp=now,
            total_objects=len(tracked_objects),
            counts_by_class=dict(counts),
            active_track_ids=[o.track_id for o in tracked_objects],
            summary_text=summary,
        )
        self._short_term_snapshots.append(snapshot)

    # ---- Query methods ----

    def get_scene_state(self) -> Dict[str, Any]:
        """Return the current scene state."""
        with self._lock:
            active = [obj.to_dict() for obj in self._active_objects.values()]
            counts: Dict[str, int] = defaultdict(int)
            for obj in self._active_objects.values():
                counts[obj.class_name] += 1
            return {
                "active_objects": active,
                "active_count": len(active),
                "counts_by_class": dict(counts),
                "total_unique_objects": len(self._object_history),
                "class_totals": dict(self._class_totals),
            }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent events from short-term memory."""
        with self._lock:
            events = list(self._short_term_events)
        events.reverse()
        return [e.to_dict() for e in events[:limit]]

    def get_events_since(self, since: float) -> List[Dict[str, Any]]:
        """Return all events since a given timestamp."""
        with self._lock:
            events = [
                e.to_dict()
                for e in self._all_events
                if e.timestamp >= since
            ]
        return events

    def get_scene_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive scene summary for LLM context."""
        with self._lock:
            active = list(self._active_objects.values())
            recent_events = list(self._short_term_events)[-20:]
            latest_snapshot = (
                self._short_term_snapshots[-1].to_dict()
                if self._short_term_snapshots
                else None
            )

        counts: Dict[str, int] = defaultdict(int)
        for obj in active:
            counts[obj.class_name] += 1

        return {
            "timestamp": time.time(),
            "active_objects": len(active),
            "counts_by_class": dict(counts),
            "total_unique_objects_seen": len(self._object_history),
            "recent_events": [e.to_dict() for e in recent_events],
            "latest_snapshot": latest_snapshot,
            "class_totals": dict(self._class_totals),
        }

    def get_object_history(
        self,
        class_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return object history, optionally filtered by class."""
        with self._lock:
            objects = list(self._object_history.values())
        if class_filter:
            objects = [o for o in objects if o.class_name == class_filter]
        objects.sort(key=lambda o: o.last_seen, reverse=True)
        return [o.to_dict() for o in objects[:limit]]

    def get_time_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get activity summary for a time range."""
        now = time.time()
        if start_time is None:
            start_time = now - 3600  # last hour
        if end_time is None:
            end_time = now

        with self._lock:
            events_in_range = [
                e for e in self._all_events
                if start_time <= e.timestamp <= end_time
            ]

        entries = [e for e in events_in_range if e.event_type == "entry"]
        exits = [e for e in events_in_range if e.event_type == "exit"]

        entry_classes: Dict[str, int] = defaultdict(int)
        for e in entries:
            cls = e.metadata.get("class_name", "unknown")
            entry_classes[cls] += 1

        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_events": len(events_in_range),
            "entries": len(entries),
            "exits": len(exits),
            "entry_classes": dict(entry_classes),
        }

    def reset(self) -> None:
        """Clear all memory."""
        with self._lock:
            self._short_term_events.clear()
            self._short_term_snapshots.clear()
            self._all_events.clear()
            self._object_history.clear()
            self._active_objects.clear()
            self._class_totals.clear()
            self._last_snapshot_time = 0.0
        logger.info("Scene memory reset")
