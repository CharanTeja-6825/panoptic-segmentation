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
from app.utils.fps_counter import FPSCounter

logger = logging.getLogger(__name__)

# Callback type: receives (frames_done, total_frames, current_fps)
ProgressCallback = Callable[[int, int, float], None]


class VideoProcessor:
    """Processes a video file with panoptic segmentation.

    Args:
        predictor: A :class:`~app.inference.panoptic_predictor.PanopticPredictor`
            instance.
    """

    def __init__(self, predictor: PanopticPredictor) -> None:
        self._predictor = predictor

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
