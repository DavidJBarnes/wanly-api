import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.heartbeat_monitor import heartbeat_monitor
from app.limiter import limiter
from app.routes import app_settings, auth, faceswap, files, images, jobs, loras, prompt_presets, segments, tags, wildcards, workers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(heartbeat_monitor())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="wanly-api", lifespan=lifespan)

# --- Rate limiting -----------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS --------------------------------------------------------------------
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --- Health check ------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Liveness/readiness probe for deployment health checks."""
    return {"status": "ok"}


app.include_router(app_settings.router)
app.include_router(auth.router)
app.include_router(images.router)
app.include_router(jobs.router)
app.include_router(segments.router)
app.include_router(faceswap.router)
app.include_router(files.router)
app.include_router(loras.router)
app.include_router(tags.router)
app.include_router(wildcards.router)
app.include_router(prompt_presets.router)
app.include_router(workers.router)
