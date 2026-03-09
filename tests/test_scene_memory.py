"""
Tests for the scene memory module.
"""

import time

import pytest

from app.memory.scene_memory import SceneMemory, SceneObject, SceneEvent, SceneSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTrackedObject:
    """Minimal stand-in for TrackedObject to avoid importing full inference."""

    def __init__(
        self,
        track_id: int = 1,
        class_name: str = "person",
        confidence: float = 0.9,
        bbox: tuple = (100, 100, 200, 200),
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.centroid = (cx, cy)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSceneMemory:

    def test_initial_state_is_empty(self):
        mem = SceneMemory()
        state = mem.get_scene_state()
        assert state["active_count"] == 0
        assert state["active_objects"] == []
        assert state["total_unique_objects"] == 0

    def test_entry_event_on_new_object(self):
        mem = SceneMemory()
        obj = FakeTrackedObject(track_id=1, class_name="person")
        events = mem.update_scene([obj])
        assert len(events) == 1
        assert events[0].event_type == "entry"
        assert events[0].objects_involved == [1]

    def test_no_events_when_stable(self):
        mem = SceneMemory()
        obj = FakeTrackedObject(track_id=1)
        mem.update_scene([obj])
        events = mem.update_scene([obj])
        assert len(events) == 0

    def test_exit_event_on_disappearance(self):
        mem = SceneMemory()
        obj = FakeTrackedObject(track_id=1, class_name="car")
        mem.update_scene([obj])
        events = mem.update_scene([])
        assert len(events) == 1
        assert events[0].event_type == "exit"
        assert "car" in events[0].description

    def test_multiple_entries(self):
        mem = SceneMemory()
        objects = [
            FakeTrackedObject(track_id=1, class_name="person"),
            FakeTrackedObject(track_id=2, class_name="car"),
        ]
        events = mem.update_scene(objects)
        assert len(events) == 2
        entry_types = {e.event_type for e in events}
        assert entry_types == {"entry"}

    def test_scene_state_reflects_active_objects(self):
        mem = SceneMemory()
        objects = [
            FakeTrackedObject(track_id=1, class_name="person"),
            FakeTrackedObject(track_id=2, class_name="person"),
            FakeTrackedObject(track_id=3, class_name="car"),
        ]
        mem.update_scene(objects)
        state = mem.get_scene_state()
        assert state["active_count"] == 3
        assert state["counts_by_class"]["person"] == 2
        assert state["counts_by_class"]["car"] == 1

    def test_recent_events_newest_first(self):
        mem = SceneMemory()
        for i in range(5):
            mem.update_scene([FakeTrackedObject(track_id=i)])
        events = mem.get_recent_events(limit=3)
        assert len(events) == 3
        # Events should be newest first
        assert events[0]["timestamp"] >= events[1]["timestamp"]

    def test_scene_summary_contains_expected_fields(self):
        mem = SceneMemory()
        mem.update_scene([FakeTrackedObject(track_id=1, class_name="dog")])
        summary = mem.get_scene_summary()
        assert "active_objects" in summary
        assert "counts_by_class" in summary
        assert "total_unique_objects_seen" in summary
        assert "recent_events" in summary
        assert summary["counts_by_class"]["dog"] == 1

    def test_object_history_tracks_all(self):
        mem = SceneMemory()
        mem.update_scene([FakeTrackedObject(track_id=1, class_name="person")])
        mem.update_scene([FakeTrackedObject(track_id=2, class_name="car")])
        history = mem.get_object_history()
        assert len(history) == 2

    def test_object_history_class_filter(self):
        mem = SceneMemory()
        mem.update_scene([
            FakeTrackedObject(track_id=1, class_name="person"),
            FakeTrackedObject(track_id=2, class_name="car"),
        ])
        history = mem.get_object_history(class_filter="car")
        assert len(history) == 1
        assert history[0]["class_name"] == "car"

    def test_time_summary(self):
        mem = SceneMemory()
        mem.update_scene([FakeTrackedObject(track_id=1)])
        mem.update_scene([])
        summary = mem.get_time_summary()
        assert summary["entries"] == 1
        assert summary["exits"] == 1
        assert summary["total_events"] == 2

    def test_events_since(self):
        mem = SceneMemory()
        before = time.time()
        mem.update_scene([FakeTrackedObject(track_id=1)])
        events = mem.get_events_since(before)
        assert len(events) >= 1

    def test_reset_clears_everything(self):
        mem = SceneMemory()
        mem.update_scene([FakeTrackedObject(track_id=1)])
        mem.reset()
        state = mem.get_scene_state()
        assert state["active_count"] == 0
        assert state["total_unique_objects"] == 0

    def test_class_totals_accumulate(self):
        mem = SceneMemory()
        mem.update_scene([FakeTrackedObject(track_id=1, class_name="person")])
        mem.update_scene([FakeTrackedObject(track_id=2, class_name="person")])
        state = mem.get_scene_state()
        assert state["class_totals"]["person"] == 2


class TestSceneEvent:

    def test_to_dict(self):
        evt = SceneEvent(
            timestamp=1234567890.0,
            event_type="entry",
            description="person entered",
            objects_involved=[1],
            metadata={"class_name": "person"},
        )
        d = evt.to_dict()
        assert d["event_type"] == "entry"
        assert d["description"] == "person entered"
        assert d["objects_involved"] == [1]


class TestSceneSnapshot:

    def test_to_dict(self):
        snap = SceneSnapshot(
            timestamp=1234567890.0,
            total_objects=3,
            counts_by_class={"person": 2, "car": 1},
            active_track_ids=[1, 2, 3],
            summary_text="Scene contains 1 car, 2 person",
        )
        d = snap.to_dict()
        assert d["total_objects"] == 3
        assert d["counts_by_class"]["person"] == 2
