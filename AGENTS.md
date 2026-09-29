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
| `POST /images/edit` (+ `/preview`, `/faces`, `GET /presets`) | Image Edit tool: face mode via the face-edit service (`face_edit_url`); saves a NEW image, refuses locked datasets (console#547). Takes a preset, slider values and/or a `prompt` ("describe the change", console#550 -- read by the service's lexicon; answers with the resolved `expression` and `matched_terms`, 422 "nothing to apply" when no term is known). `/faces` lists the faces (source pixels, left to right, `default_index`); preview/save take `face_box` or `face_index` to edit one that is not the centre-most (console#553) |
| `GET/PUT /settings`, `POST /images/scene/try` | Settings returns the default text of every caption style and the default motion template (`{scene}`, `{style}`, `{#scene}...{/scene}`; see `app/joycaption.py`), and refuses a template with an unknown placeholder (422). `/images/scene/try` runs the saved or unsaved prompts on one image through the caption queue and stores nothing (console#555) |

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
