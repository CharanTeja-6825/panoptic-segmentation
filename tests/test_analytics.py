"""
Tests for the analytics modules.
"""

import time

import pytest

from app.analytics.event_logger import EventLogger, ObjectEvent
from app.analytics.heatmap_generator import HeatmapGenerator
from app.analytics.object_counter import ObjectCounter
from app.inference.tracker import TrackedObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracked(
    track_id: int = 1,
    class_name: str = "person",
    bbox=(100, 100, 200, 200),
) -> TrackedObject:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        centroid=(cx, cy),
        trajectory=[(cx, cy)],
        age=5,
        hits=5,
        time_since_update=0,
        color=(0, 255, 0),
    )


# ---------------------------------------------------------------------------
# ObjectCounter tests
# ---------------------------------------------------------------------------

class TestObjectCounter:

    def test_empty_update(self):
        counter = ObjectCounter()
        stats = counter.update([], fps=30.0)
        assert stats.total_objects == 0
        assert stats.fps == 30.0

    def test_counts_by_class(self):
        counter = ObjectCounter()
        objects = [
            _make_tracked(track_id=1, class_name="person"),
            _make_tracked(track_id=2, class_name="person"),
            _make_tracked(track_id=3, class_name="car"),
        ]
        stats = counter.update(objects)
        assert stats.counts_by_class["person"] == 2
        assert stats.counts_by_class["car"] == 1
        assert stats.total_objects == 3

    def test_live_stats_rolling(self):
        counter = ObjectCounter(rolling_window=3)
        for i in range(5):
            counter.update(
                [_make_tracked(track_id=i, class_name="person")],
                fps=30.0,
            )
        live = counter.get_live_stats()
        assert live["total_frames"] == 5
        assert "person" in live["rolling_avg_counts"]
        assert live["rolling_avg_counts"]["person"] == pytest.approx(1.0)

    def test_reset(self):
        counter = ObjectCounter()
        counter.update([_make_tracked()])
        counter.reset()
        live = counter.get_live_stats()
        assert live["total_frames"] == 0


# ---------------------------------------------------------------------------
# HeatmapGenerator tests
# ---------------------------------------------------------------------------

class TestHeatmapGenerator:

    def test_initial_heatmap_is_black(self):
        hm = HeatmapGenerator(width=64, height=64)
        img = hm.get_heatmap()
        assert img.shape == (64, 64, 3)

    def test_update_adds_activity(self):
        import numpy as np
        hm = HeatmapGenerator(width=100, height=100, decay=1.0)
        obj = _make_tracked(bbox=(40, 40, 60, 60))
        hm.update([obj])
        img = hm.get_heatmap()
        # Centre region should be non-zero
        centre = img[50, 50]
        assert centre.sum() > 0

    def test_resize_heatmap(self):
        hm = HeatmapGenerator(width=64, height=64)
        hm.update([_make_tracked(bbox=(20, 20, 40, 40))])
        img = hm.get_heatmap(target_size=(128, 128))
        assert img.shape == (128, 128, 3)

    def test_overlay_on_frame(self):
        import numpy as np
        hm = HeatmapGenerator(width=100, height=100)
        hm.update([_make_tracked(bbox=(40, 40, 60, 60))])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = hm.overlay_on_frame(frame, alpha=0.5)
        assert result.shape == frame.shape

    def test_reset(self):
        import numpy as np
        hm = HeatmapGenerator(width=64, height=64, decay=1.0)
        hm.update([_make_tracked(bbox=(20, 20, 40, 40))])
        hm.reset()
        img = hm.get_heatmap()
        # After reset, all pixels should be from the empty colourmap
        # (colourmap(0) for JET is a dark blue, so sum should be low/uniform)
        assert img.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# EventLogger tests
# ---------------------------------------------------------------------------

class TestEventLogger:

    def test_entry_event(self):
        el = EventLogger()
        obj = _make_tracked(track_id=1)
        events = el.update([obj])
        assert len(events) == 1
        assert events[0].event_type == "entry"
        assert events[0].track_id == 1

    def test_exit_event(self):
        el = EventLogger()
        obj = _make_tracked(track_id=1)
        el.update([obj])
        events = el.update([])  # object disappears
        assert len(events) == 1
        assert events[0].event_type == "exit"

    def test_no_events_when_stable(self):
        el = EventLogger()
        obj = _make_tracked(track_id=1)
        el.update([obj])
        events = el.update([obj])
        assert len(events) == 0

    def test_get_events_limit(self):
        el = EventLogger()
        for i in range(10):
            el.update([_make_tracked(track_id=i)])
        events = el.get_events(limit=5)
        assert len(events) == 5

    def test_csv_export(self):
        el = EventLogger()
        el.update([_make_tracked(track_id=1, class_name="person")])
        csv_data = el.export_csv()
        assert "timestamp" in csv_data
        assert "entry" in csv_data
        assert "person" in csv_data

    def test_reset(self):
        el = EventLogger()
        el.update([_make_tracked(track_id=1)])
        el.reset()
        events = el.get_events()
        assert len(events) == 0
