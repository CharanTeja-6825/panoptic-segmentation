"""
Tests for the object tracker module.
"""

import numpy as np
import pytest

from app.inference.panoptic_predictor import Detection
from app.inference.tracker import ObjectTracker, TrackedObject, TrackingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    bbox=(100, 100, 200, 200),
    mask=None,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObjectTracker:
    """Tests for ObjectTracker."""

    def test_create_tracker(self):
        tracker = ObjectTracker()
        result = tracker.update([])
        assert isinstance(result, TrackingResult)
        assert result.tracked_objects == []
        assert result.frame_count == 1

    def test_single_detection_becomes_track(self):
        tracker = ObjectTracker(min_hits=1)
        det = _make_detection()
        result = tracker.update([det])
        assert len(result.tracked_objects) == 1
        obj = result.tracked_objects[0]
        assert obj.class_name == "person"
        assert obj.track_id == 1
        assert obj.hits == 1

    def test_persistent_id_across_frames(self):
        tracker = ObjectTracker(min_hits=1, iou_threshold=0.1)
        det1 = _make_detection(bbox=(100, 100, 200, 200))
        result1 = tracker.update([det1])
        tid = result1.tracked_objects[0].track_id

        # Same location → same ID
        det2 = _make_detection(bbox=(105, 105, 205, 205))
        result2 = tracker.update([det2])
        assert result2.tracked_objects[0].track_id == tid

    def test_new_detection_gets_new_id(self):
        tracker = ObjectTracker(min_hits=1, iou_threshold=0.3)
        det1 = _make_detection(bbox=(10, 10, 50, 50))
        det2 = _make_detection(bbox=(500, 500, 600, 600), class_name="car")
        result = tracker.update([det1, det2])
        ids = {obj.track_id for obj in result.tracked_objects}
        assert len(ids) == 2

    def test_track_removal_after_max_age(self):
        tracker = ObjectTracker(max_age=2, min_hits=1)
        det = _make_detection()
        tracker.update([det])

        # No detections for 3 frames → track should be pruned
        for _ in range(4):
            tracker.update([])

        result = tracker.update([])
        assert len(result.tracked_objects) == 0

    def test_min_hits_filter(self):
        tracker = ObjectTracker(min_hits=3)
        det = _make_detection()
        # First update: 1 hit → not yet confirmed
        result = tracker.update([det])
        assert len(result.tracked_objects) == 0

        # Second: 2 hits → still not confirmed
        result = tracker.update([det])
        assert len(result.tracked_objects) == 0

        # Third: 3 hits → confirmed
        result = tracker.update([det])
        assert len(result.tracked_objects) == 1

    def test_trajectory_stored(self):
        tracker = ObjectTracker(min_hits=1, max_trajectory=5)
        for i in range(7):
            det = _make_detection(bbox=(i * 10, 0, i * 10 + 50, 50))
            result = tracker.update([det])

        obj = result.tracked_objects[0]
        assert len(obj.trajectory) <= 5

    def test_reset_clears_state(self):
        tracker = ObjectTracker(min_hits=1)
        tracker.update([_make_detection()])
        tracker.reset()
        result = tracker.update([])
        assert len(result.tracked_objects) == 0
        assert result.frame_count == 1

    def test_color_deterministic(self):
        tracker = ObjectTracker(min_hits=1)
        det = _make_detection()
        r1 = tracker.update([det])
        color1 = r1.tracked_objects[0].color

        tracker.reset()
        r2 = tracker.update([det])
        color2 = r2.tracked_objects[0].color

        assert color1 == color2

    def test_iou_computation(self):
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        boxes_b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        iou = ObjectTracker._compute_iou_matrix(boxes_a, boxes_b)
        assert iou.shape == (1, 2)
        assert iou[0, 0] == pytest.approx(1.0)
        assert iou[0, 1] == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        boxes_b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        iou = ObjectTracker._compute_iou_matrix(boxes_a, boxes_b)
        # Overlap area = 5*5 = 25, union = 100+100-25 = 175
        assert iou[0, 0] == pytest.approx(25.0 / 175.0, abs=1e-6)
