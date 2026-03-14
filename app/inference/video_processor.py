"""
Video processor module.

Handles reading an uploaded video file, running panoptic segmentation
frame-by-frame, writing the annotated output, and reporting progress
via an async callback.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.config import config
from app.inference.panoptic_predictor import PanopticPredictor
from app.inference.tracker import ObjectTracker
from app.utils.fps_counter import FPSCounter

logger = logging.getLogger(__name__)

# Callback type: receives (frames_done, total_frames, current_fps)
ProgressCallback = Callable[[int, int, float], None]


class VideoProcessor:
    """Processes a video file with panoptic segmentation.

    Args:
        predictor: A :class:`~app.inference.panoptic_predictor.PanopticPredictor`
            instance.
        enable_tracking: Whether to run object tracking on detections.
        depth_estimator: Optional depth estimator for distance labelling.
    """

    def __init__(
        self,
        predictor: PanopticPredictor,
        enable_tracking: bool = True,
        depth_estimator=None,
    ) -> None:
        self._predictor = predictor
        self._enable_tracking = enable_tracking
        self._depth_estimator = depth_estimator
        self._tracker: Optional[ObjectTracker] = (
            ObjectTracker(
                max_age=config.tracker_max_age,
                min_hits=config.tracker_min_hits,
                iou_threshold=config.tracker_iou_threshold,
            )
            if enable_tracking
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        analysis_output_path: Optional[str] = None,
        keyframes_dir: Optional[str] = None,
        keyframe_interval_seconds: float = 2.0,
    ) -> dict:
        """Process *input_path* and write segmented frames to *output_path*.

        Args:
            input_path: Absolute path to the source video.
            output_path: Absolute path where the output video is saved.
            progress_callback: Optional callable called after every frame with
                ``(frames_done, total_frames, current_fps)``.

        Returns:
            Dictionary with processing statistics.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            RuntimeError: If the video cannot be opened.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                "Processing '%s': %dx%d @ %.1f fps, %d frames",
                os.path.basename(input_path),
                width,
                height,
                fps,
                total_frames,
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # Use MP4V codec for broad compatibility
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

            fps_counter = FPSCounter(window=30)
            frame_idx = 0
            start_time = time.time()
            safe_fps = fps if fps > 0 else 25.0

            analysis_enabled = bool(analysis_output_path)
            timeline_stride = max(int(round(safe_fps)), 1)
            keyframe_stride = max(
                int(round(safe_fps * max(keyframe_interval_seconds, 0.1))),
                1,
            )

            active_track_ids: set[int] = set()
            first_seen_frame: Dict[int, int] = {}
            first_seen_time: Dict[int, float] = {}
            track_class: Dict[int, str] = {}
            unique_tracks_by_class: Dict[str, int] = defaultdict(int)
            frame_detections_by_class: Dict[str, int] = defaultdict(int)
            timeline_counts: List[Dict[str, Any]] = []
            events: List[Dict[str, Any]] = []
            keyframes: List[Dict[str, Any]] = []

            if keyframes_dir:
                os.makedirs(keyframes_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                fps_counter.tick()
                result = self._predictor.predict(frame)
                annotated = result.annotated_frame if result.annotated_frame is not None else frame

                # Tracking overlay
                tracked_objects = []
                if self._tracker is not None:
                    tracking_result = self._tracker.update(result.detections)
                    tracked_objects = tracking_result.tracked_objects
                    for obj in tracking_result.tracked_objects:
                        x1, y1, x2, y2 = obj.bbox
                        cv2.putText(
                            annotated,
                            f"ID:{obj.track_id}",
                            (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            obj.color,
                            1,
                            cv2.LINE_AA,
                        )

                    # Depth estimation overlay
                    depth_est = self._depth_estimator
                    if depth_est is not None and depth_est.enabled:
                        depth_map = depth_est.estimate(frame)
                        if depth_map is not None:
                            for obj in tracking_result.tracked_objects:
                                dist = depth_est.estimate_object_distance(
                                    depth_map, obj.bbox
                                )
                                label = f"{obj.class_name} - {dist}m"
                                bx1, by1, _, _ = obj.bbox
                                cv2.putText(
                                    annotated,
                                    label,
                                    (bx1, max(by1 - 10, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

                if analysis_enabled:
                    frame_time_s = round(frame_idx / safe_fps, 3)
                    frame_counts: Dict[str, int] = defaultdict(int)
                    current_track_ids: set[int] = set()

                    for obj in tracked_objects:
                        frame_counts[obj.class_name] += 1
                        frame_detections_by_class[obj.class_name] += 1
                        current_track_ids.add(obj.track_id)
                        if obj.track_id not in active_track_ids:
                            active_track_ids.add(obj.track_id)
                            first_seen_frame[obj.track_id] = frame_idx
                            first_seen_time[obj.track_id] = frame_time_s
                            track_class[obj.track_id] = obj.class_name
                            unique_tracks_by_class[obj.class_name] += 1
                            events.append(
                                {
                                    "frame_index": frame_idx,
                                    "timestamp_seconds": frame_time_s,
                                    "event_type": "entry",
                                    "track_id": obj.track_id,
                                    "class_name": obj.class_name,
                                    "confidence": round(float(obj.confidence), 3),
                                }
                            )

                    exited_ids = sorted(active_track_ids - current_track_ids)
                    for track_id in exited_ids:
                        class_name = track_class.get(track_id, "unknown")
                        start_frame = first_seen_frame.get(track_id, frame_idx)
                        start_time_s = first_seen_time.get(track_id, frame_time_s)
                        duration_seconds = max(frame_time_s - start_time_s, 0.0)
                        events.append(
                            {
                                "frame_index": frame_idx,
                                "timestamp_seconds": frame_time_s,
                                "event_type": "exit",
                                "track_id": track_id,
                                "class_name": class_name,
                                "duration_seconds": round(duration_seconds, 3),
                                "duration_frames": max(frame_idx - start_frame, 0),
                            }
                        )
                        active_track_ids.discard(track_id)

                    if frame_idx % timeline_stride == 0:
                        timeline_counts.append(
                            {
                                "frame_index": frame_idx,
                                "timestamp_seconds": frame_time_s,
                                "total_objects": len(tracked_objects),
                                "counts_by_class": dict(frame_counts),
                            }
                        )

                    if (
                        keyframes_dir
                        and frame_idx % keyframe_stride == 0
                    ):
                        keyframe_name = f"frame_{frame_idx:06d}.jpg"
                        keyframe_path = os.path.join(keyframes_dir, keyframe_name)
                        if cv2.imwrite(keyframe_path, annotated):
                            keyframes.append(
                                {
                                    "frame_index": frame_idx,
                                    "timestamp_seconds": frame_time_s,
                                    "path": keyframe_path,
                                    "file_name": keyframe_name,
                                }
                            )

                writer.write(annotated)
                frame_idx += 1

                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames, fps_counter.fps)

            elapsed = time.time() - start_time
            writer.release()

            if analysis_enabled and active_track_ids:
                final_time_s = round(frame_idx / safe_fps, 3)
                for track_id in sorted(active_track_ids):
                    class_name = track_class.get(track_id, "unknown")
                    start_frame = first_seen_frame.get(track_id, frame_idx)
                    start_time_s = first_seen_time.get(track_id, final_time_s)
                    duration_seconds = max(final_time_s - start_time_s, 0.0)
                    events.append(
                        {
                            "frame_index": frame_idx,
                            "timestamp_seconds": final_time_s,
                            "event_type": "exit",
                            "track_id": track_id,
                            "class_name": class_name,
                            "duration_seconds": round(duration_seconds, 3),
                            "duration_frames": max(frame_idx - start_frame, 0),
                            "reason": "video_end",
                        }
                    )
                active_track_ids.clear()

        finally:
            cap.release()

        stats = {
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
            "output_path": output_path,
        }

        if analysis_enabled and analysis_output_path:
            duration_seconds = round(frame_idx / safe_fps, 3) if safe_fps > 0 else 0.0
            analysis_data: Dict[str, Any] = {
                "video": {
                    "input_path": input_path,
                    "output_path": output_path,
                    "width": width,
                    "height": height,
                    "fps": round(safe_fps, 3),
                    "total_frames": total_frames,
                    "frames_processed": frame_idx,
                    "duration_seconds": duration_seconds,
                },
                "summary": {
                    "total_unique_tracks": len(first_seen_frame),
                    "unique_tracks_by_class": dict(unique_tracks_by_class),
                    "frame_detections_by_class": dict(frame_detections_by_class),
                    "events_count": len(events),
                    "keyframes_count": len(keyframes),
                },
                "timeline": {
                    "sample_interval_frames": timeline_stride,
                    "counts": timeline_counts,
                },
                "events": events,
                "keyframes": keyframes,
            }
            os.makedirs(os.path.dirname(analysis_output_path) or ".", exist_ok=True)
            with open(analysis_output_path, "w", encoding="utf-8") as analysis_file:
                json.dump(analysis_data, analysis_file, indent=2)

            stats["analysis_path"] = analysis_output_path
            stats["analysis_summary"] = analysis_data["summary"]

        logger.info("Video processing complete: %s", stats)
        return stats

    async def process_video_async(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        analysis_output_path: Optional[str] = None,
        keyframes_dir: Optional[str] = None,
        keyframe_interval_seconds: float = 2.0,
    ) -> dict:
        """Async wrapper around :meth:`process_video`.

        Offloads the blocking I/O + inference to the default thread pool so
        the FastAPI event loop is not blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_video,
            input_path,
            output_path,
            progress_callback,
            analysis_output_path,
            keyframes_dir,
            keyframe_interval_seconds,
        )

    async def frame_generator(
        self, input_path: str
    ) -> AsyncGenerator[np.ndarray, None]:
        """Async generator that yields annotated frames one by one.

        Useful for streaming responses.

        Args:
            input_path: Path to source video.

        Yields:
            BGR annotated frame arrays.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        loop = asyncio.get_event_loop()
        try:
            while True:
                # Read frame in thread to avoid blocking event loop
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    break
                result = await loop.run_in_executor(
                    None, self._predictor.predict, frame
                )
                annotated = result.annotated_frame if result.annotated_frame is not None else frame
                yield annotated
        finally:
            cap.release()
