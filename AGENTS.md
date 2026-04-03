# Wanly API

FastAPI backend for the Wanly video generation system.

## Purpose

- Job and segment management
- S3 file storage for videos/images
- Video stitching (ffmpeg)
- PostgreSQL database with Alembic migrations
- REST API for daemon workers and console frontend

## Key Models

- **Job**: Top-level video generation request
- **Segment**: Individual ~5s video segment (chained via last frame)
- **Video**: Final stitched output
- **Lora**: LoRA library entries
- **PromptPreset**: Saved prompt templates

## Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /jobs` | Create new job |
| `POST /segments` | Add segment to job |
| `POST /segments/{id}/claim` | Daemon claims segment |
| `PATCH /segments/{id}` | Update segment status |
| `POST /segments/{id}/upload` | Upload segment output |
| `POST /videos/{id}/stitch` | Stitch all segments |

## Quality Enhancement Features

### Motion Keywords
Segments extract and propagate motion keywords (walking, running, standing, etc.) to improve continuity.

### Reference Frames
Segments track up to 3 previous output frames for multi-frame identity anchoring in PainterLongVideo.

## Database

- PostgreSQL with Alembic migrations
- Key tables: `jobs`, `segments`, `videos`, `loras`, `app_settings`
- Migrations in `alembic/versions/`

## Deployment

- Docker container on EC2
- GitHub Actions workflow: `.github/workflows/deploy.yml`
- Migrations run automatically during deploy

## Related Projects

- `wanly-gpu-daemon`: Worker daemon that runs ComfyUI
- `wanly-console`: React frontend
