from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.limiter import limiter
from app.routes import auth, faceswap, files, images, jobs, loras, prompt_presets, segments, tags, wildcards

app = FastAPI(title="wanly-api")

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
