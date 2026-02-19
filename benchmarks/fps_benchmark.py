"""
FPS benchmarking script.

Measures raw inference throughput (frames per second) of the panoptic
segmentation model on synthetic or real input data.

Usage::

    python benchmarks/fps_benchmark.py [--frames 200] [--size 640] [--model medium]
"""

import argparse
import sys
import time
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8-seg inference speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    parser.add_argument("--size",   type=int, default=640, help="Frame width/height in pixels")
    parser.add_argument(
        "--model",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up frames (not counted in FPS)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file to use as input instead of synthetic frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Override model size in config before loading ----
    import app.config as cfg_module
    cfg_module.config.model_size = args.model
    cfg_module.config.model_device = "auto"
    cfg_module.config.inference_input_size = args.size

    from app.inference.model_loader import ModelLoader
    from app.inference.panoptic_predictor import PanopticPredictor

    print(f"\n{'='*60}")
    print(f"  Panoptic Segmentation FPS Benchmark")
    print(f"{'='*60}")
    print(f"  Model size  : {args.model}")
    print(f"  Frame size  : {args.size}x{args.size}")
    print(f"  Frames      : {args.frames}  (+ {args.warmup} warm-up)")
    if args.video:
        print(f"  Input video : {args.video}")
    else:
        print(f"  Input       : synthetic (random RGB frames)")
    print()

    loader = ModelLoader()
    loader.load()
    predictor = PanopticPredictor(loader)

    print(f"  Device      : {loader.device}\n")

    # ---- Prepare frames ----
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video '{args.video}'")
            sys.exit(1)
        frames = []
        while len(frames) < args.frames + args.warmup:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames.append(frame)
        cap.release()
    else:
        rng = np.random.default_rng(42)
        frames = [
            (rng.integers(0, 255, (args.size, args.size, 3), dtype=np.uint8))
            for _ in range(args.frames + args.warmup)
        ]

    # ---- Warm-up ----
    print(f"  Warming up ({args.warmup} frames)…", end="", flush=True)
    for frame in frames[: args.warmup]:
        predictor.predict(frame)
    print(" done.\n")

    # ---- Benchmark ----
    bench_frames = frames[args.warmup :]
    start = time.perf_counter()
    for frame in bench_frames:
        predictor.predict(frame)
    elapsed = time.perf_counter() - start

    fps = len(bench_frames) / elapsed if elapsed > 0 else 0.0
    ms_per_frame = (elapsed / len(bench_frames) * 1000) if bench_frames else 0.0

    print(f"  Results:")
    print(f"    Frames processed : {len(bench_frames)}")
    print(f"    Total time       : {elapsed:.2f} s")
    print(f"    Average FPS      : {fps:.2f}")
    print(f"    ms / frame       : {ms_per_frame:.1f}")
    print(f"\n{'='*60}\n")

    loader.unload()


if __name__ == "__main__":
    main()
