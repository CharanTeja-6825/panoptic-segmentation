"""
FastAPI application entry point.

Wires together routes, middleware, CORS, static files and startup / shutdown
lifecycle events for the Panoptic Segmentation Web Application.
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import config
from app.inference.model_loader import ModelLoader
from app.inference.depth_estimator import DepthEstimator
from app.inference.panoptic_predictor import PanopticPredictor
from app.routes.camera_routes import router as camera_router
from app.routes.video_routes import router as video_router
from app.routes.analytics_routes import router as analytics_router
from app.routes.multicam_routes import router as multicam_router
from app.utils.benchmark import BenchmarkUtil, warmup_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

_model_loader: ModelLoader | None = None
_depth_estimator: DepthEstimator | None = None
_benchmark: BenchmarkUtil | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    global _model_loader, _depth_estimator, _benchmark

    # ---- startup ----
    logger.info("Starting Real-Time Intelligent Scene Understanding Platform…")

    # Ensure required directories exist
    os.makedirs(config.upload_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Pre-load the model so first requests are fast
    _model_loader = ModelLoader()
    _model_loader.load()
    app.state.model_loader = _model_loader

    # Depth estimator (optional, toggled at runtime)
    _depth_estimator = DepthEstimator(
        device=config.model_device,
        use_fp16=config.use_fp16,
    )
    if config.depth_enabled:
        _depth_estimator.load()
        _depth_estimator.enabled = True
    app.state.depth_estimator = _depth_estimator

    # Benchmark utility
    _benchmark = BenchmarkUtil()
    app.state.benchmark = _benchmark

    # Model warmup
    if config.model_warmup_frames > 0:
        predictor = PanopticPredictor(_model_loader)
        warmup_model(predictor, num_frames=config.model_warmup_frames)

    logger.info("Model loaded. Application ready.")
    yield

    # ---- shutdown ----
    logger.info("Shutting down – releasing resources…")
    # Stop multi-camera streams
    from app.routes.multicam_routes import get_camera_manager
    get_camera_manager().stop_all()

    if _depth_estimator is not None:
        _depth_estimator.unload()
    if _model_loader is not None:
        _model_loader.unload()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Real-Time Intelligent Scene Understanding Platform",
        description=(
            "Real-time panoptic segmentation with object tracking, depth estimation, "
            "analytics, and multi-camera support. Powered by YOLOv8-seg and FastAPI."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(video_router, prefix="/api", tags=["video"])
    app.include_router(camera_router, prefix="/api", tags=["camera"])
    app.include_router(analytics_router, prefix="/api", tags=["analytics"])
    app.include_router(multicam_router, prefix="/api", tags=["multi-camera"])

    # Serve frontend static files
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            """Serve the main web UI."""
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    # Serve processed output videos
    if os.path.isdir(config.output_dir):
        app.mount(
            "/outputs",
            StaticFiles(directory=config.output_dir),
            name="outputs",
        )

    @app.get("/api/health", tags=["health"])
    async def health_check():
        """Return application health status."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        depth_status = "disabled"
        if hasattr(app.state, "depth_estimator") and app.state.depth_estimator is not None:
            depth_status = "enabled" if app.state.depth_estimator.enabled else "disabled"
        return {
            "status": "ok",
            "device": device,
            "gpu": gpu_name,
            "model_size": config.model_size,
            "depth_estimation": depth_status,
            "tracking": True,
        }

    @app.post("/api/toggle-depth", tags=["depth"])
    async def toggle_depth():
        """Toggle depth estimation on or off."""
        estimator = app.state.depth_estimator
        if estimator is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="Depth estimator not initialised")
        if not estimator.enabled:
            if estimator._model is None:
                estimator.load()
            estimator.enabled = True
        else:
            estimator.enabled = False
        return {"depth_enabled": estimator.enabled}

    @app.get("/api/benchmark", tags=["benchmark"])
    async def benchmark_endpoint():
        """Return performance metrics."""
        bench = getattr(app.state, "benchmark", None)
        if bench is None:
            return {"error": "Benchmark not initialised"}
        return bench.get_metrics()

    return app


app = create_app()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )
