"""
Project N: Local FastAPI Daemon.
Main application setup, SSE streaming bus, CORS, and modular router mounting.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.deps import get_analysis_service, get_sse_bus
from server.routes import auth, clips, episodes, facts, models
from server.services.analysis_service import AnalysisService
from server.sse_bus import SSEBus


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for background workers and resource initialization."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Project N Local Assistant Daemon",
    description="Offline behavioral insight and co-regulatory assistant running locally on Apple Silicon.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS restricted strictly to localhost origins (Invariant 1: Offline local only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount domain routers
app.include_router(auth.router)
app.include_router(clips.router)
app.include_router(episodes.router)
app.include_router(facts.router)
app.include_router(models.router)


@app.get("/api/v1/health")
def health_check(
    analysis: AnalysisService = Depends(get_analysis_service),
) -> dict[str, Any]:
    """System health check returning active Metal memory, engine status, and loaded checkpoint."""
    active_gb = mx.get_active_memory() / 1e9
    peak_gb = mx.get_peak_memory() / 1e9
    ckpt_id = (
        f"ckpt_{analysis.projection_head.checkpoint_hash[:8]}"
        if analysis.projection_head.checkpoint_hash
        else "uncalibrated_v0"
    )
    return {
        "status": "healthy",
        "engine": "Apple MLX",
        "active_metal_memory_gb": round(active_gb, 3),
        "peak_metal_memory_gb": round(peak_gb, 3),
        "checkpoint_id": ckpt_id,
        "is_trained": analysis.projection_head.is_trained,
    }


@app.get("/api/v1/events/stream")
async def event_stream(
    last_event_id: str | None = Header(default=None),
    sse: SSEBus = Depends(get_sse_bus),
) -> StreamingResponse:
    """Server-Sent Events (SSE) stream delivering real-time pipeline events and telemetry."""
    eid_int = int(last_event_id) if last_event_id and last_event_id.isdigit() else None
    return StreamingResponse(
        sse.subscribe(last_event_id=eid_int),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Mount local Zero-Cloud React SPA Dashboard if built (Invariant 1)
ui_dist_dir = Path(__file__).resolve().parent.parent / "ui" / "dist"
if (ui_dist_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(ui_dist_dir / "assets")), name="assets")

if ui_dist_dir.exists():

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Serves static files or falls back to index.html for client-side routing."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        candidate = ui_dist_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(ui_dist_dir / "index.html")
