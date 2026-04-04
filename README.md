# 🔬 Real-Time Intelligent Scene Understanding & Analytics Platform

A **production-ready** multimodal AI vision system with **real-time object detection**, **LLM-powered reasoning**, **scene memory**, and a **professional React dashboard**, built with:

- **YOLOv8-seg** (Ultralytics) – instance + semantic fusion
- **FastAPI** – async REST API + WebSocket backend
- **Ollama** – local LLM integration for natural language scene understanding
- **React + TypeScript + Tailwind CSS** – professional SaaS-grade frontend with shadcn-style components
- **OpenCV** – video I/O and MJPEG streaming
- **PyTorch** – CUDA-accelerated inference
- **MiDaS** – monocular depth estimation
- **SciPy** – Hungarian algorithm for object tracking

The platform integrates six major subsystems:

1. **Vision Module** – YOLOv8-seg panoptic segmentation with persistent object tracking
2. **Scene Memory** – short-term buffer and long-term event database for queryable scene state
3. **Multimodal Reasoning** – Ollama LLM integration with structured prompts, streaming responses, and vision model support
4. **Live Commentary** – auto-generated scene descriptions and priority alerts
5. **Analytics Engine** – live object counting, spatial heatmaps, entry/exit logging, and CSV export
6. **Professional Dashboard** – React + TypeScript SaaS-quality UI with real-time WebSocket updates

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
| **Scene memory** | Short-term buffer + long-term event store with queryable state |
| **AI chat interface** | Natural language queries with streaming LLM responses |
| **Vision model selector** | Choose vision models (llava-phi3, llava:7b, etc.) for image analysis |
| **Processed-video Q&A** | Ask questions grounded in full-video analysis and sampled keyframes |
| **Live commentary** | Auto-generated scene descriptions and alerts |
| **Scene insights panel** | Real-time AI-powered analysis of camera feed |
| Analytics dashboard | Live object counts, class breakdowns, and event logs |
| Spatial heatmaps | Accumulative motion / presence heatmap overlay |
| Depth estimation | Optional MiDaS monocular depth map overlay |
| Multi-camera management | Start, stop, and list multiple camera streams |
| **Real-time WebSocket events** | Live event broadcasting to connected clients |
| **Professional React UI** | Dark-themed SaaS dashboard with shadcn-style components |
| **Rate limiting & queue** | Bounded LLM task queue to prevent overload on low-resource hardware |
| Progress tracking | Real-time frame progress + FPS during video processing |
| Model size selector | small / medium / large (via `MODEL_SIZE` env var) |
| FP16 inference | Half-precision mode for faster GPU throughput |
| Async processing | Background task queue; non-blocking API responses |
| FPS benchmarking | Built-in benchmark script + `/api/benchmark` endpoint |
| Docker support | Dockerfile + docker-compose.yml with Ollama service |

---

## 🗂 Project Structure

```
panoptic-segmentation/
│
├── app/
│   ├── main.py                    # FastAPI application factory
│   ├── config.py                  # All settings (env-var driven)
│   ├── inference/
│   │   ├── model_loader.py        # YOLOv8-seg model loading + CUDA
│   │   ├── panoptic_predictor.py  # Per-frame segmentation logic
│   │   ├── video_processor.py     # Frame-by-frame video pipeline
│   │   ├── tracker.py             # IoU-based multi-object tracker
│   │   └── depth_estimator.py     # MiDaS monocular depth estimation
│   ├── analytics/
│   │   ├── object_counter.py      # Per-class rolling object counter
│   │   ├── heatmap_generator.py   # Spatial presence heatmap
│   │   └── event_logger.py        # Timestamped analytics event log
│   ├── memory/                    # Scene memory module
│   │   └── scene_memory.py        # Short/long-term memory with queryable state
│   ├── llm/                       # LLM integration module
│   │   ├── ollama_client.py       # Enhanced LLM client with timeouts, fallback, and structured errors
│   │   └── prompt_templates.py    # Structured prompt builders with context compaction
│   ├── services/                  # Service layer
│   │   └── commentary_engine.py   # Auto-commentary and alert generation
│   ├── routes/
│   │   ├── video_routes.py        # /upload-video, /process-video, etc.
│   │   ├── camera_routes.py       # /camera-stream (MJPEG)
│   │   ├── analytics_routes.py    # /api/analytics/* endpoints
│   │   ├── multicam_routes.py     # /api/camera/* multi-cam endpoints
│   │   ├── chat_routes.py         # /api/chat, /api/chat/stream, queue-based rate limiting
│   │   └── scene_routes.py        # /api/scene/*, /api/ws/events (WebSocket)
│   ├── streams/
│   │   └── camera_manager.py      # Multi-camera lifecycle manager
│   └── utils/
│       ├── visualization.py       # JPEG encode, FPS overlay, legend
│       ├── fps_counter.py         # Rolling-window FPS counter
│       └── benchmark.py           # Inference performance benchmarking
│
├── frontend/                      # React + TypeScript professional UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/            # Reusable UI components (Button, Card, Select, Badge, Alert, Tooltip)
│   │   │   ├── features/          # Feature-specific components
│   │   │   └── layout/            # Layout components (TopNav, Sidebar)
│   │   ├── features/
│   │   │   ├── camera/            # CameraPage with scene insights panel
│   │   │   ├── chat/              # ChatPage with vision model selector
│   │   │   ├── analytics/         # Analytics dashboard
│   │   │   └── video/             # Video processing pages
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── store/                 # Zustand state management
│   │   ├── services/              # API client and WebSocket manager
│   │   └── styles/                # Global CSS with Tailwind and shadcn aesthetics
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── frontend-legacy/               # Original HTML/JS/CSS UI
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── benchmarks/
│   └── fps_benchmark.py           # Inference FPS benchmark
│
├── tests/
│   ├── test_tracker.py            # Object tracker unit tests
│   ├── test_analytics.py          # Analytics engine unit tests
│   ├── test_scene_memory.py       # Scene memory tests
│   ├── test_llm.py                # LLM module tests (prompts, client, structured errors)
│   └── test_commentary.py         # Commentary engine tests
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── README.md
```

---

## 🏗 Architecture

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Professional React Dashboard                           │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│  │  Live Feed +  │ │  AI Chat +    │ │  Event        │ │  Stats &      │    │
│  │  Scene Insights│ │  Model Select │ │  Timeline     │ │  Analytics    │    │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘    │
└──────────┼─────────────────┼─────────────────┼─────────────────┼────────────┘
           │ MJPEG           │ WebSocket       │ WebSocket       │ REST
           ▼                 ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                                      │
│                                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐   │
│  │ Vision      │  │ Scene Memory │  │ LLM Client     │  │ Commentary   │   │
│  │ Module      │──│ Module       │──│ (Queue + Rate  │──│ Engine       │   │
│  │ (YOLOv8)    │  │ (Buffer+DB)  │  │  Limiting)     │  │ (Auto-gen)   │   │
│  └──────┬──────┘  └──────────────┘  └───────┬────────┘  └──────────────┘   │
│         │                                    │                               │
│  ┌──────▼──────┐                    ┌───────▼─────────┐                     │
│  │ Tracker +   │                    │ Ollama Server    │                     │
│  │ Analytics   │                    │ (llava-phi3,     │                     │
│  └─────────────┘                    │  llama3.2, etc.) │                     │
│                                     └─────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### LLM Client Architecture (Optimized for Low-Resource Hardware)

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Client (ollama_client.py)               │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Structured   │    │ Separate     │    │ Fallback     │      │
│  │ Error Types  │    │ Timeouts     │    │ Model        │      │
│  │ (LLMError)   │    │ (5s/45s)     │    │ Support      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Image        │    │ Metrics      │    │ Vision       │      │
│  │ Compression  │    │ Tracking     │    │ Chat         │      │
│  │ (512px/60%)  │    │ (latency,    │    │ Support      │      │
│  └──────────────┘    │ fallbacks)   │    └──────────────┘      │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Chat Routes (chat_routes.py)                │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Bounded Queue    │    │ Scene-Memory-First│                  │
│  │ (maxsize=2)      │    │ Query Routing     │                  │
│  │ → HTTP 429 when  │    │ (count/status →   │                  │
│  │   queue full     │    │  memory; analyze  │                  │
│  └──────────────────┘    │  → LLM)           │                  │
│                          └──────────────────┘                   │
│                                                                  │
│  Endpoints:                                                      │
│  • POST /api/chat              (scene-aware chat)               │
│  • WS   /api/chat/stream       (streaming responses)            │
│  • GET  /api/llm/vision-models (available vision models)        │
│  • GET  /api/llm/metrics       (LLM performance metrics)        │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Flows

**Video upload flow**

```
Upload → Frame Extraction → Segmentation → Tracking → Depth (optional) → Analytics → Overlay → Output Video
```

**Live camera flow**

```
Camera Frame → Segmentation → Tracking → Scene Memory → Commentary → Analytics → Overlay → Stream Back
                                              │
                                              └── Scene Insights Panel (real-time AI analysis)
```

**Chat flow (with queue-based rate limiting)**

```
User Query → Query Classification → [Memory Query?] → Scene Memory → Response
                                  │
                                  └── [Analysis Query?] → Queue Check → LLM (with fallback) → Response
                                                              │
                                                              └── Queue Full → HTTP 429
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.10+**
- Node.js **18+** (for frontend build)
- (Optional) NVIDIA GPU with CUDA 11.8+ and compatible drivers
- (Optional) [Ollama](https://ollama.ai) for LLM features

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

### 4 – Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5 – Install Ollama (optional, for AI chat)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision model (default for image analysis)
ollama pull llava-phi3

# Pull text model (for general chat)
ollama pull llama3.2

# Start the server (runs on port 11434)
ollama serve
```

> **Note:** YOLOv8-seg weights are downloaded automatically on first run.
> The system works without Ollama – chat features will simply be unavailable.

---

## 🚀 Running the Application

### Option A – Docker Compose (recommended)

```bash
docker compose up --build
```

This starts both the main application and Ollama LLM server. Open **http://localhost:8000** in your browser.

### Option B – Shell script

```bash
chmod +x run.sh
./run.sh
```

### Option C – Direct uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option D – Python entry point

```bash
python -m app.main
```

For frontend development with hot reload:

```bash
cd frontend && npm run dev
```

This starts a Vite dev server at **http://localhost:3000** with API proxy to port 8000.

---

## 🐳 Docker

### Build and run (with Ollama)

```bash
docker compose up --build
```

### GPU support

Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.
The `docker-compose.yml` already includes GPU reservation for both the app and Ollama.

### Without GPU

Remove the `deploy.resources.reservations` blocks from `docker-compose.yml` and set `MODEL_DEVICE=cpu`.

---

## 🌐 API Endpoints

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serve the React dashboard |
| `GET`  | `/api/health` | Device, GPU, model, LLM queue metrics, and service status |

### Video Processing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload-video` | Upload a video file |
| `POST` | `/api/process-video/{job_id}` | Start background processing |
| `GET`  | `/api/job-status/{job_id}` | Poll processing progress |
| `GET`  | `/api/download/{job_id}` | Download the output video |
| `GET`  | `/api/video-analysis/{job_id}` | Fetch structured analysis for a processed video |
| `POST` | `/api/video-chat/{job_id}` | Ask questions about a processed video |

### Live Camera

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/camera-stream/start` | Open the camera |
| `POST` | `/api/camera-stream/stop` | Release the camera |
| `GET`  | `/api/camera-stream` | MJPEG live segmentation stream |
| `GET`  | `/api/camera-stream/status` | Camera FPS and running state |
| `POST` | `/api/camera-stream/toggle-heatmap` | Toggle heatmap overlay |

### AI Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send a chat message with scene context (rate-limited) |
| `WS`   | `/api/chat/stream` | WebSocket for streaming LLM responses |
| `GET`  | `/api/llm/status` | Check Ollama connection status and queue metrics |
| `GET`  | `/api/llm/models` | List available LLM models |
| `GET`  | `/api/llm/vision-models` | List available vision models with default selection |
| `GET`  | `/api/llm/metrics` | LLM performance metrics (latency, fallbacks, timeouts) |

### Scene State

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/scene/state` | Current scene state (active objects) |
| `GET`  | `/api/scene/events` | Recent scene events |
| `GET`  | `/api/scene/summary` | Comprehensive scene summary |
| `GET`  | `/api/scene/history` | Object history with class filter |
| `GET`  | `/api/scene/time-summary` | Activity summary for time range |
| `GET`  | `/api/scene/commentary` | Recent auto-generated commentary |
| `WS`   | `/api/ws/events` | Real-time event WebSocket |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/analytics/live` | Live analytics snapshot |
| `GET`  | `/api/analytics/export` | Export events as CSV |
| `POST` | `/api/toggle-depth` | Enable/disable depth estimation |
| `GET`  | `/api/benchmark` | Performance metrics |

### Multi-Camera

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/camera/start` | Start a new camera stream |
| `POST` | `/api/camera/stop` | Stop a camera stream |
| `GET`  | `/api/camera/list` | List all active camera streams |

Interactive API docs: **http://localhost:8000/docs**

---

## ⚡ Environment Variables

All settings can be overridden via environment variables:

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | HTTP port |
| `APP_DEBUG` | `false` | Enable uvicorn reload |

### Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SIZE` | `medium` | `small` \| `medium` \| `large` |
| `MODEL_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `INFERENCE_INPUT_SIZE` | `640` | Frame resize target for inference |
| `CONF_THRESHOLD` | `0.35` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `USE_FP16` | `false` | Enable FP16 half-precision inference |

### Video Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | `500` | Max uploaded file size |
| `VIDEO_KEYFRAME_INTERVAL_SECONDS` | `2.0` | Keyframe sampling interval for processed-video Q&A |
| `MASK_ALPHA` | `0.45` | Mask overlay transparency (0–1) |

### Camera Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `STREAM_FPS_TARGET` | `25` | Target camera stream FPS |
| `JPEG_QUALITY` | `85` | MJPEG stream JPEG quality |

### Tracking & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACKER_MAX_AGE` | `30` | Frames before a lost track is removed |
| `TRACKER_MIN_HITS` | `3` | Minimum detections before track is confirmed |
| `SCENE_MEMORY_CAPACITY` | `300` | Short-term memory buffer size |
| `DEPTH_ENABLED` | `false` | Enable MiDaS depth estimation |

### LLM Settings (Optimized for Low-Resource Hardware)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default text LLM model |
| `OLLAMA_VISION_MODEL` | `llava-phi3` | Default vision model for image analysis |
| `OLLAMA_CONNECT_TIMEOUT` | `5` | Connection timeout (seconds) |
| `OLLAMA_READ_TIMEOUT` | `45` | Read timeout (seconds) |
| `OLLAMA_MAX_IMAGE_KB` | `350` | Max image payload size (KB) |
| `LLM_PRIMARY_MODEL` | `llava-phi3` | Primary vision model |
| `LLM_FALLBACK_MODEL` | `llava:7b` | Fallback model when primary fails |
| `LLM_MAX_QUEUE_SIZE` | `2` | Max LLM tasks in queue (prevents overload) |
| `LLM_MAX_CONCURRENT` | `1` | Max concurrent LLM requests |
| `LLM_ENABLE_CLOUD_FALLBACK` | `false` | Enable cloud API fallback |

### Chat Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_FRAME_INTERVAL_MS` | `2000` | Frame send interval for vision chat (ms) |
| `CHAT_MAX_IMAGE_WIDTH` | `512` | Max image width for compression |
| `CHAT_JPEG_QUALITY` | `60` | JPEG quality for chat images |

### Commentary Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `COMMENTARY_ENABLED` | `true` | Enable auto-commentary generation |
| `COMMENTARY_INTERVAL` | `30` | Seconds between auto-commentaries |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Python logging level |

Example (low-resource hardware optimization):

```bash
MODEL_SIZE=small \
OLLAMA_VISION_MODEL=llava-phi3 \
LLM_MAX_QUEUE_SIZE=2 \
CHAT_FRAME_INTERVAL_MS=2000 \
./run.sh
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_scene_memory.py -v
python -m pytest tests/test_llm.py -v
python -m pytest tests/test_commentary.py -v
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

---

## 💬 Using the AI Chat

Once Ollama is running with models loaded, you can ask natural language questions about the scene:

### Scene-Memory-First Queries (Fast, No LLM Required)

These queries are answered directly from scene memory:

- "How many people are in the scene?"
- "What is the current count of cars?"
- "Show me the object status"
- "What happened in the last 5 minutes?"

### LLM-Powered Analysis (Vision + Reasoning)

These queries use the vision model for image analysis:

- "Describe what you see in the camera"
- "Analyze the current scene for safety concerns"
- "What activities are happening right now?"
- "Alert me if a crowd forms"

### Vision Model Selection

Use the dropdown on the Camera or Chat page to select different vision models:

- **llava-phi3** (default) – Compact, fast, good for low-resource hardware
- **llava:7b** – Larger model with better reasoning
- **llava:13b** – High-quality analysis (requires more RAM)

The system responds using real-time scene data, combining vision detections with LLM reasoning.

---

## 🔧 Troubleshooting

### `CUDA not available`
- Verify `nvidia-smi` works.
- Reinstall PyTorch with the correct CUDA version.
- Set `MODEL_DEVICE=cpu` to force CPU inference.

### `Cannot open camera at index 0`
- Check no other application is using the camera.
- Try a different index via `CAMERA_INDEX=1`.

### Ollama not connecting
- Ensure Ollama is running: `ollama serve`
- Check the URL: `curl http://localhost:11434/api/tags`
- When using Docker, ensure the services can communicate (default config handles this).

### LLM queue full (HTTP 429)
- The system is rate-limited to prevent overload on low-resource hardware.
- Wait a moment and retry, or reduce `CHAT_FRAME_INTERVAL_MS`.
- Increase `LLM_MAX_QUEUE_SIZE` if you have more resources.

### LLM timeouts
- Check Ollama is running and responsive.
- Increase `OLLAMA_READ_TIMEOUT` for larger models.
- Try a smaller model like `llava-phi3`.

### Image too large (HTTP 413)
- Reduce image resolution or quality before sending.
- The system compresses images to max 512px width by default.

### Frontend build fails
- Ensure Node.js 18+ is installed.
- Run `cd frontend && npm install && npm run build`.

### Slow inference
- Use `MODEL_SIZE=small` for faster predictions.
- Reduce `INFERENCE_INPUT_SIZE` (e.g. `480`).
- Enable GPU if available.
- Set `USE_FP16=true` for half-precision on CUDA devices.

---

## 🎨 Frontend Design System

The frontend uses a shadcn-inspired design system with:

### Components

- **Button** – Variants: default, destructive, outline, secondary, ghost, link
- **Card** – With Header, Content, Footer, Title, Description subcomponents
- **Badge** – Variants: default, secondary, success, warning, destructive, outline, info
- **Select** – With label and error support
- **Input** – With label and error support
- **Alert** – Variants: default, info, success, warning, destructive
- **Tooltip** – With configurable positioning

### CSS Custom Properties

The theme uses CSS custom properties for easy customization:

```css
:root {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --primary: 217.2 91.2% 59.8%;
  --destructive: 0 62.8% 30.6%;
  /* ... */
}
```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
