"""
ByteTrack-style object tracker module.

Associates detections across frames using IoU-based matching with the
Hungarian algorithm, maintaining persistent track IDs, per-object colours,
and centroid trajectory histories.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.inference.panoptic_predictor import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """State of a single tracked object."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[float, float]  # (cx, cy)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class TrackingResult:
    """Aggregated tracking output for one frame."""

    tracked_objects: List[TrackedObject] = field(default_factory=list)
    frame_count: int = 0


class ObjectTracker:
    """IoU-based multi-object tracker inspired by ByteTrack.

    Args:
        max_age: Number of consecutive frames a track is kept alive without
            a matching detection before it is removed.
        min_hits: Minimum number of associated detections before a track is
            reported in results.
        iou_threshold: Minimum IoU required to associate a detection with an
            existing track.
        max_trajectory: Maximum number of centroid positions stored in each
            track's trajectory history.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_trajectory: int = 50,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._max_trajectory = max_trajectory

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> TrackingResult:
        """Update tracker state with new detections and return results.

        Args:
            detections: Detections produced by the panoptic predictor for the
                current frame.

        Returns:
            :class:`TrackingResult` containing tracked objects that have been
            confirmed (i.e. ``hits >= min_hits``).
        """
        self._frame_count += 1

        if not self._tracks:
            # First frame or after reset – initialise tracks directly.
            for det in detections:
                self._create_track(det)
            return self._build_result()

        if not detections:
            # No detections – age all existing tracks.
            self._age_tracks()
            self._prune_tracks()
            return self._build_result()

        # ----- Match existing tracks to new detections ------------------
        track_ids = list(self._tracks.keys())
        track_bboxes = np.array(
            [self._tracks[tid].bbox for tid in track_ids], dtype=np.float64
        )
        det_bboxes = np.array(
            [d.bbox for d in detections], dtype=np.float64
        )

        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

        match_result = self._hungarian_match(iou_matrix)
        matched_track_indices = match_result[0]
        matched_det_indices = match_result[1]
        unmatched_track_indices = match_result[2]
        unmatched_det_indices = match_result[3]

        # ----- Update matched tracks ------------------------------------
        for t_idx, d_idx in zip(matched_track_indices, matched_det_indices):
            tid = track_ids[t_idx]
            self._update_track(tid, detections[d_idx])

        # ----- Age unmatched tracks -------------------------------------
        for t_idx in unmatched_track_indices:
            tid = track_ids[t_idx]
            self._tracks[tid].time_since_update += 1
            self._tracks[tid].age += 1

        # ----- Create tracks for unmatched detections -------------------
        for d_idx in unmatched_det_indices:
            self._create_track(detections[d_idx])

        self._prune_tracks()
        return self._build_result()

    def reset(self) -> None:
        """Clear all tracks and reset internal counters."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("Tracker state reset")

    # ------------------------------------------------------------------
    # IoU computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Compute pair-wise IoU between two sets of bounding boxes.

        Args:
            boxes_a: Array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)`` rows.
            boxes_b: Array of shape ``(M, 4)`` with ``(x1, y1, x2, y2)`` rows.

        Returns:
            IoU matrix of shape ``(N, M)``.
        """
        n = boxes_a.shape[0]
        m = boxes_b.shape[0]
        iou = np.zeros((n, m), dtype=np.float64)

        for i in range(n):
            xa1, ya1, xa2, ya2 = boxes_a[i]
            area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
            for j in range(m):
                xb1, yb1, xb2, yb2 = boxes_b[j]
                area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

                inter_x1 = max(xa1, xb1)
                inter_y1 = max(ya1, yb1)
                inter_x2 = min(xa2, xb2)
                inter_y2 = min(ya2, yb2)
                inter_area = max(0.0, inter_x2 - inter_x1) * max(
                    0.0, inter_y2 - inter_y1
                )

                union = area_a + area_b - inter_area
                iou[i, j] = inter_area / union if union > 0 else 0.0

        return iou

    # ------------------------------------------------------------------
    # Hungarian matching
    # ------------------------------------------------------------------

    def _hungarian_match(
        self, iou_matrix: np.ndarray
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Apply the Hungarian algorithm on a cost matrix derived from IoU.

        Returns:
            Tuple of (matched_track_indices, matched_det_indices,
            unmatched_track_indices, unmatched_det_indices).
        """
        n_tracks, n_dets = iou_matrix.shape

        if n_tracks == 0 or n_dets == 0:
            return (
                [],
                [],
                list(range(n_tracks)),
                list(range(n_dets)),
            )

        # Convert IoU (similarity) to cost (minimisation problem).
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks: List[int] = []
        matched_dets: List[int] = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self._iou_threshold:
                matched_tracks.append(r)
                matched_dets.append(c)
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return (
            matched_tracks,
            matched_dets,
            sorted(unmatched_tracks),
            sorted(unmatched_dets),
        )

    # ------------------------------------------------------------------
    # Track lifecycle helpers
    # ------------------------------------------------------------------

    def _create_track(self, detection: Detection) -> None:
        """Initialise a new track from a detection."""
        tid = self._next_id
        self._next_id += 1

        centroid = self._bbox_centroid(detection.bbox)
        self._tracks[tid] = TrackedObject(
            track_id=tid,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            centroid=centroid,
            trajectory=[centroid],
            age=1,
            hits=1,
            time_since_update=0,
            color=self._id_to_color(tid),
        )
        logger.debug("Created track %d for '%s'", tid, detection.class_name)

    def _update_track(self, track_id: int, detection: Detection) -> None:
        """Update an existing track with a matched detection."""
        track = self._tracks[track_id]
        centroid = self._bbox_centroid(detection.bbox)

        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.bbox = detection.bbox
        track.centroid = centroid
        track.age += 1
        track.hits += 1
        track.time_since_update = 0

        track.trajectory.append(centroid)
        if len(track.trajectory) > self._max_trajectory:
            track.trajectory = track.trajectory[-self._max_trajectory :]

    def _age_tracks(self) -> None:
        """Increment age and time_since_update for every track."""
        for track in self._tracks.values():
            track.age += 1
            track.time_since_update += 1

    def _prune_tracks(self) -> None:
        """Remove tracks that have not been updated for *max_age* frames."""
        stale_ids = [
            tid
            for tid, t in self._tracks.items()
            if t.time_since_update > self._max_age
        ]
        for tid in stale_ids:
            logger.debug(
                "Removing stale track %d ('%s')",
                tid,
                self._tracks[tid].class_name,
            )
            del self._tracks[tid]

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(self) -> TrackingResult:
        """Return a :class:`TrackingResult` with confirmed tracks only."""
        confirmed = [
            t for t in self._tracks.values() if t.hits >= self._min_hits
        ]
        return TrackingResult(
            tracked_objects=confirmed,
            frame_count=self._frame_count,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Return the centre point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _id_to_color(track_id: int) -> Tuple[int, int, int]:
        """Generate a deterministic BGR colour from a track ID.

        Uses a simple hash-based mapping to spread colours across the
        spectrum so that nearby IDs get visually distinct colours.
        """
        # Golden-ratio hue spacing gives good perceptual separation.
        golden_ratio_conjugate = 0.6180339887498949
        hue = int(((track_id * golden_ratio_conjugate) % 1.0) * 179)  # OpenCV hue range 0-179
        hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
