# 🔬 Real-Time Intelligent Scene Understanding & Analytics Platform

A **production-ready** real-time panoptic segmentation system with **object tracking**, **analytics**, **depth estimation**, and **multi-camera support**, built with:

- **YOLOv8-seg** (Ultralytics) – instance + semantic fusion
- **FastAPI** – async REST API backend
- **OpenCV** – video I/O and MJPEG streaming
- **PyTorch** – CUDA-accelerated inference
- **MiDaS** – monocular depth estimation
- **SciPy** – Hungarian algorithm for object tracking

The platform goes beyond basic segmentation through five integrated stages:

1. **Object Tracking** – persistent IDs across frames via IoU-based Hungarian matching
2. **Analytics Engine** – live object counting, spatial heatmaps, and event logging
3. **Depth Estimation** – optional MiDaS-powered monocular depth overlays
4. **Performance Optimization** – FP16 inference, model warm-up, and built-in benchmarking
5. **Multi-Camera Support** – manage and stream from multiple camera sources simultaneously

---

## 📸 Features

| Feature | Detail |
|---|---|
| Video upload processing | .mp4 .avi .mov .mkv, up to 500 MB |
| Live camera segmentation | MJPEG stream via `/api/camera-stream` |
| GPU / CUDA support | Auto-detects and uses CUDA when available |
| Instance segmentation | Per-object masks and bounding boxes |
| Semantic segmentation | Per-pixel class labels (via instance-to-class mapping) |
| Object tracking | Persistent track IDs across frames (IoU + Hungarian algorithm) |
| Analytics dashboard | Live object counts, class breakdowns, and event logs |
| Spatial heatmaps | Accumulative motion / presence heatmap overlay |
| Depth estimation | Optional MiDaS monocular depth map overlay |
| Multi-camera management | Start, stop, and list multiple camera streams |
| Progress tracking | Real-time frame progress + FPS during video processing |
| Model size selector | small / medium / large (via `MODEL_SIZE` env var) |
| FP16 inference | Half-precision mode for faster GPU throughput |
| Async processing | Background task queue; non-blocking API responses |
| FPS benchmarking | Built-in benchmark script + `/api/benchmark` endpoint |
| Docker support | Dockerfile + docker-compose.yml |

---

## 🗂 Project Structure

```
panoptic-segmentation-app/
│
├── app/
│   ├── main.py                  # FastAPI application factory
│   ├── config.py                # All settings (env-var driven)
│   ├── inference/
│   │   ├── model_loader.py      # YOLOv8-seg model loading + CUDA
│   │   ├── panoptic_predictor.py# Per-frame segmentation logic
│   │   ├── video_processor.py   # Frame-by-frame video pipeline
│   │   ├── tracker.py           # IoU-based multi-object tracker
│   │   └── depth_estimator.py   # MiDaS monocular depth estimation
│   ├── analytics/
│   │   ├── object_counter.py    # Per-class rolling object counter
│   │   ├── heatmap_generator.py # Spatial presence heatmap
│   │   └── event_logger.py      # Timestamped analytics event log
│   ├── routes/
│   │   ├── video_routes.py      # /upload-video, /process-video, etc.
│   │   ├── camera_routes.py     # /camera-stream (MJPEG)
│   │   ├── analytics_routes.py  # /api/analytics/* endpoints
│   │   └── multicam_routes.py   # /api/camera/* multi-cam endpoints
│   ├── streams/
│   │   └── camera_manager.py    # Multi-camera lifecycle manager
│   └── utils/
│       ├── visualization.py     # JPEG encode, FPS overlay, legend
│       ├── fps_counter.py       # Rolling-window FPS counter
│       └── benchmark.py         # Inference performance benchmarking
│
├── frontend/
│   ├── index.html               # Web UI
│   ├── script.js                # UI logic (upload, polling, camera)
│   └── styles.css               # Dark-theme CSS
│
├── benchmarks/
│   └── fps_benchmark.py         # Inference FPS benchmark
│
├── tests/
│   ├── test_tracker.py          # Object tracker unit tests
│   └── test_analytics.py        # Analytics engine unit tests
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── README.md
```

---

## 🏗 Architecture

### Pipeline Flows

**Video upload flow**

```
Upload → Frame Extraction → Segmentation → Tracking → Depth (optional) → Analytics → Overlay → Output Video
```

**Live camera flow**

```
Camera Frame → Segmentation → Tracking → Depth → Analytics Update → Overlay → Stream Back
```

### Component Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌────────────┐
│  Video File  │────▶│  Frame Extractor  │────▶│            │
└─────────────┘     └──────────────────┘     │            │
                                              │  YOLOv8    │     ┌───────────┐     ┌────────────┐
┌─────────────┐                               │  Panoptic  │────▶│  Tracker   │────▶│  Analytics  │
│  Camera(s)  │────▶─────────────────────────▶│  Predictor │     │ (Hungarian)│     │  Engine     │
└─────────────┘                               │            │     └─────┬─────┘     └─────┬──────┘
                                              └─────┬──────┘           │                 │
                                                    │            ┌─────▼─────┐     ┌─────▼──────┐
                                                    │            │   Depth    │     │  Heatmap / │
                                                    │            │ Estimator  │     │  Counter / │
                                                    │            │  (MiDaS)   │     │   Logger   │
                                                    │            └─────┬─────┘     └─────┬──────┘
                                                    │                  │                 │
                                                    ▼                  ▼                 ▼
                                              ┌──────────────────────────────────────────┐
                                              │          Visualization & Overlay          │
                                              └──────────────────────────────────────────┘
                                                                   │
                                              ┌────────────────────┴────────────────────┐
                                              ▼                                         ▼
                                        Output Video                              MJPEG Stream
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10+**
- (Optional) NVIDIA GPU with CUDA 11.8+ and compatible drivers

### 1 – Clone the repository

```bash
git clone https://github.com/CharanTeja-6825/panoptic-segmentation.git
cd panoptic-segmentation
```

### 2 – Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 – Install Python dependencies

#### CPU-only

```bash
pip install -r requirements.txt
```

#### GPU (CUDA 12.1)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note:** YOLOv8-seg weights (`yolov8m-seg.pt` etc.) are downloaded automatically
> from Ultralytics on first run. MiDaS weights are downloaded on first use of
> depth estimation.

---

## 🚀 Running the Application

### Option A – Shell script

```bash
chmod +x run.sh
./run.sh
```

### Option B – Direct uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option C – Python entry point

```bash
python -m app.main
```

Once started, open your browser at: **http://localhost:8000**

---

## 🐳 Docker

### Build and run (CPU)

```bash
docker compose up --build
```

### GPU support

Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed, then:

```bash
docker compose up --build
```

The `docker-compose.yml` already contains the GPU reservation block.

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serve the web UI |
| `GET`  | `/api/health` | Device, GPU name, model info |
| `POST` | `/api/upload-video` | Upload a video file |
| `POST` | `/api/process-video/{job_id}` | Start background processing |
| `GET`  | `/api/job-status/{job_id}` | Poll processing progress |
| `GET`  | `/api/download/{job_id}` | Download the output video |
| `POST` | `/api/camera-stream/start` | Open the camera |
| `POST` | `/api/camera-stream/stop` | Release the camera |
| `GET`  | `/api/camera-stream/status` | Camera FPS and running state |
| `GET`  | `/api/camera-stream` | MJPEG live segmentation stream |
| `POST` | `/api/camera-stream/toggle-heatmap` | Toggle heatmap overlay on stream |
| `GET`  | `/api/analytics/live` | Live analytics snapshot (counts, events) |
| `GET`  | `/api/analytics/export` | Export analytics data as JSON |
| `POST` | `/api/toggle-depth` | Enable / disable depth estimation |
| `GET`  | `/api/benchmark` | Run and return inference benchmark results |
| `POST` | `/api/camera/start` | Start a new camera stream (multi-cam) |
| `POST` | `/api/camera/stop` | Stop a camera stream (multi-cam) |
| `GET`  | `/api/camera/list` | List all active camera streams |

Interactive API docs are available at **http://localhost:8000/docs**.

---

## ⚡ Environment Variables

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | HTTP port |
| `APP_DEBUG` | `false` | Enable uvicorn reload |
| `MODEL_SIZE` | `medium` | `small` \| `medium` \| `large` |
| `MODEL_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max uploaded file size |
| `INFERENCE_INPUT_SIZE` | `640` | Frame resize target for inference |
| `CONF_THRESHOLD` | `0.35` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `MASK_ALPHA` | `0.45` | Mask overlay transparency (0–1) |
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `STREAM_FPS_TARGET` | `25` | Target camera stream FPS |
| `JPEG_QUALITY` | `85` | MJPEG stream JPEG quality |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `TRACKER_MAX_AGE` | `30` | Frames before a lost track is removed |
| `TRACKER_MIN_HITS` | `3` | Minimum detections before track is confirmed |
| `TRACKER_IOU_THRESHOLD` | `0.3` | IoU threshold for track association |
| `DEPTH_ENABLED` | `false` | Enable MiDaS depth estimation |
| `DEPTH_MODEL` | `MiDaS_small` | MiDaS model variant |
| `USE_FP16` | `false` | Enable FP16 half-precision inference |
| `MODEL_WARMUP_FRAMES` | `3` | Warm-up frames at startup |
| `ANALYTICS_ROLLING_WINDOW` | `100` | Rolling window size for analytics |
| `HEATMAP_WIDTH` | `640` | Heatmap grid width |
| `HEATMAP_HEIGHT` | `480` | Heatmap grid height |
| `HEATMAP_DECAY` | `0.98` | Heatmap exponential decay factor |

Example:

```bash
MODEL_SIZE=small MODEL_DEVICE=cuda DEPTH_ENABLED=true USE_FP16=true ./run.sh
```

---

## 📊 FPS Benchmark

```bash
# Synthetic input, medium model
python benchmarks/fps_benchmark.py --frames 100 --model medium

# Real video input, small model
python benchmarks/fps_benchmark.py --video myvideo.mp4 --model small --frames 200
```

Expected throughput (approximate):

| Model | GPU (RTX 3080) | CPU (i7-12th gen) |
|-------|---------------|-------------------|
| small  | ~95 FPS | ~12 FPS |
| medium | ~55 FPS | ~7 FPS  |
| large  | ~25 FPS | ~3 FPS  |

> **Tip:** Enable `USE_FP16=true` on supported GPUs for up to ~30 % faster inference.
> Use the `/api/benchmark` endpoint to measure throughput on your hardware.

---

## 🔧 Troubleshooting

### `CUDA not available`
- Verify `nvidia-smi` works.
- Reinstall PyTorch with the correct CUDA version (see Installation section).
- Set `MODEL_DEVICE=cpu` to force CPU inference.

### `Cannot open camera at index 0`
- Check no other application is using the camera.
- Try a different index via `CAMERA_INDEX=1`.
- On Linux, ensure your user is in the `video` group: `sudo usermod -aG video $USER`.

### `No module named 'ultralytics'`
```bash
pip install ultralytics
```

### `No module named 'scipy'`
```bash
pip install scipy
```

### Slow inference
- Use `MODEL_SIZE=small` for faster (but less accurate) predictions.
- Reduce `INFERENCE_INPUT_SIZE` (e.g. `480`).
- Enable GPU if available.
- Set `USE_FP16=true` for half-precision on CUDA devices.

### Upload fails with 413
- Increase `MAX_UPLOAD_SIZE_MB` environment variable.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
