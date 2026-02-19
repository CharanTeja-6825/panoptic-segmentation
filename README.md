# 🔬 Panoptic Segmentation Web Application

A **production-ready** real-time panoptic segmentation system built with:

- **YOLOv8-seg** (Ultralytics) – instance + semantic fusion
- **FastAPI** – async REST API backend
- **OpenCV** – video I/O and MJPEG streaming
- **PyTorch** – CUDA-accelerated inference

The application accepts uploaded video files **and** streams a live camera feed through a modern web UI, overlaying colour-coded masks, bounding boxes, and class labels in real time.

---

## 📸 Features

| Feature | Detail |
|---|---|
| Video upload processing | .mp4 .avi .mov .mkv, up to 500 MB |
| Live camera segmentation | MJPEG stream via `/api/camera-stream` |
| GPU / CUDA support | Auto-detects and uses CUDA when available |
| Instance segmentation | Per-object masks and bounding boxes |
| Semantic segmentation | Per-pixel class labels (via instance-to-class mapping) |
| Progress tracking | Real-time frame progress + FPS during video processing |
| Model size selector | small / medium / large (via `MODEL_SIZE` env var) |
| Async processing | Background task queue; non-blocking API responses |
| FPS benchmarking | Built-in benchmark script |
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
│   │   └── video_processor.py   # Frame-by-frame video pipeline
│   ├── routes/
│   │   ├── video_routes.py      # /upload-video, /process-video, etc.
│   │   └── camera_routes.py     # /camera-stream (MJPEG)
│   └── utils/
│       ├── visualization.py     # JPEG encode, FPS overlay, legend
│       └── fps_counter.py       # Rolling-window FPS counter
│
├── frontend/
│   ├── index.html               # Web UI
│   ├── script.js                # UI logic (upload, polling, camera)
│   └── styles.css               # Dark-theme CSS
│
├── benchmarks/
│   └── fps_benchmark.py         # Inference FPS benchmark
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── README.md
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
> from Ultralytics on first run.

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

Example:

```bash
MODEL_SIZE=small MODEL_DEVICE=cuda ./run.sh
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

### Slow inference
- Use `MODEL_SIZE=small` for faster (but less accurate) predictions.
- Reduce `INFERENCE_INPUT_SIZE` (e.g. `480`).
- Enable GPU if available.

### Upload fails with 413
- Increase `MAX_UPLOAD_SIZE_MB` environment variable.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
