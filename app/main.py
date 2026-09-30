import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import TimeoutError as PoolTimeout

from app.caption_hold import caption_hold_monitor
from app.config import settings
from app.heartbeat_monitor import heartbeat_monitor
from app.reservation_monitor import reservation_monitor
from app.limiter import limiter
from app.routes import app_settings, auth, captions, datasets, favorites, files, image_edit, images, jobs, ltx_recipes, runpod, segments, stats, tags, training, videos, wildcards, workers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(heartbeat_monitor()),
        asyncio.create_task(reservation_monitor()),
        # Segments held on a caption (console#562): resumes their waiters after a restart
        # and releases any whose words were saved while nobody was watching.
        asyncio.create_task(caption_hold_monitor()),
    ]
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="wanly-api", lifespan=lifespan)

# --- Rate limiting -----------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- An empty connection pool is a 503 with a reason, not a bare 500 (console#559) ----------
@app.exception_handler(PoolTimeout)
async def _pool_exhausted(request: Request, exc: PoolTimeout):
    """Every pooled connection stayed checked out for POOL_TIMEOUT_S.

    Temporary and on our side, so 503 -- and a message the console can show, instead of the
    unhandled-exception 500 that reads as "the delete is broken" when it is "the API is
    saturated, try again".
    """
    logger.error("connection pool exhausted on %s %s: %s",
                 request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The API is out of database connections right now "
                           "(too much in flight at once). Try again in a moment."},
    )

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
app.include_router(favorites.router)
app.include_router(images.router)
app.include_router(image_edit.router)
app.include_router(jobs.router)
app.include_router(segments.router)
app.include_router(files.router)
app.include_router(captions.router)
app.include_router(ltx_recipes.router)
app.include_router(tags.router)
app.include_router(videos.router)
app.include_router(wildcards.router)
app.include_router(workers.router)
app.include_router(runpod.router)
app.include_router(stats.router)
app.include_router(datasets.router)
app.include_router(training.router)
