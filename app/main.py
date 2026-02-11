from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import auth, jobs, segments

app = FastAPI(title="wanly-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(segments.router)
