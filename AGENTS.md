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
| `POST /images/edit` `mode:"full"`, `GET /images/edit/jobs/{id}`, `POST /images/edit/jobs/{id}/save` | Image Edit full mode (console#548): Qwen-Image-Edit on the 3090's image-edit service (`image_edit_url`), an `instruction` or a head angle (`head_preset` / `angle` {yaw, pitch}, image-space: yaw<0 = toward the image's left, pitch>0 = chin up). Returns 202 with a job; `app/full_edit.py` switches the box into edit mode through its control API (waits for the render segment, never interrupts training), runs the queue FIFO, and hands the card back after `image_edit_return_grace_s`. Results are held in memory (with an AuraFace score vs the source) until saved as a NEW image. `GET /images/edit/presets` carries `head_angles` with their route (±20° = face mode) |

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
