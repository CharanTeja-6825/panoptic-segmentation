"""
Video processor module.

Handles reading an uploaded video file, running panoptic segmentation
frame-by-frame, writing the annotated output, and reporting progress
via an async callback.
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator, Callable, Optional

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

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                fps_counter.tick()
                result = self._predictor.predict(frame)
                annotated = result.annotated_frame if result.annotated_frame is not None else frame

                # Tracking overlay
                if self._tracker is not None:
                    tracking_result = self._tracker.update(result.detections)
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

                writer.write(annotated)
                frame_idx += 1

                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames, fps_counter.fps)

            elapsed = time.time() - start_time
            writer.release()

        finally:
            cap.release()

        stats = {
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
            "output_path": output_path,
        }
        logger.info("Video processing complete: %s", stats)
        return stats

    async def process_video_async(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None,
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
