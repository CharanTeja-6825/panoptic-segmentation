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
from app.routes.camera_routes import router as camera_router
from app.routes.video_routes import router as video_router

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    global _model_loader

    # ---- startup ----
    logger.info("Starting Panoptic Segmentation Web App…")

    # Ensure required directories exist
    os.makedirs(config.upload_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Pre-load the model so first requests are fast
    _model_loader = ModelLoader()
    _model_loader.load()
    app.state.model_loader = _model_loader

    logger.info("Model loaded. Application ready.")
    yield

    # ---- shutdown ----
    logger.info("Shutting down – releasing resources…")
    if _model_loader is not None:
        _model_loader.unload()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Panoptic Segmentation API",
        description=(
            "Real-time panoptic segmentation via uploaded videos and live camera feed. "
            "Powered by YOLOv8-seg and FastAPI."
        ),
        version="1.0.0",
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
        return {
            "status": "ok",
            "device": device,
            "gpu": gpu_name,
            "model_size": config.model_size,
        }

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
