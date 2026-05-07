# wanly-api — Codebase Audit

**Audit Date:** 2026-03-12T15:58:56Z
**Branch:** main
**Commit:** 26849be1298459ddfebe25f12545e8f1fd4192ca Merge pull request #34 from DavidJBarnes/feature/reduce-duplicate-starting-images
**Auditor:** Claude Code (Automated)
**Purpose:** Zero-context reference for AI-assisted development
**Stack:** Python/FastAPI backend
**Audit File:** wanly-api-Audit.md
**Scorecard:** wanly-api-Scorecard.md
**OpenAPI Spec:** wanly-api-OpenAPI.yaml (generated separately)

> This audit is the source of truth for the wanly-api codebase structure, models, services, components, and configuration.
> The OpenAPI spec (wanly-api-OpenAPI.yaml) is the source of truth for all endpoints, Pydantic schemas, and API contracts.
> An AI reading this audit + the OpenAPI spec should be able to generate accurate code changes, new features, tests, and fixes without filesystem access.

---

## Section 1: Project Identity

| Field | Value |
|-------|-------|
| Repo Name | wanly-api |
| Language | Python 3.12 (Docker), dev machine Python 3.14.2 |
| Framework | FastAPI |
| ORM | SQLAlchemy (async, asyncpg) |
| Migrations | Alembic (14 migrations) |
| Database | PostgreSQL (async via asyncpg) |
| Auth | JWT (HS256) + bcrypt password hashing |
| Storage | AWS S3 (boto3) |
| Rate Limiting | slowapi |
| CI/CD | GitHub Actions -> ECR -> SSM deploy to EC2 |
| Port | 8001 |

---

## Section 3: Build & Dependency Manifest

**File:** `/home/david/projects/wanly/wanly-api/requirements.txt`

| Dependency | Pinned? | Purpose |
|------------|---------|---------|
| fastapi | No | Web framework |
| uvicorn[standard] | No | ASGI server |
| sqlalchemy[asyncio] | No | ORM |
| asyncpg | No | PostgreSQL async driver |
| alembic | No | Database migrations |
| pydantic-settings | No | Env-based config |
| python-dotenv | No | .env file loading |
| pyjwt | No | JWT token encoding/decoding |
| bcrypt | No | Password hashing |
| boto3 | No | AWS S3 client |
| python-multipart | No | File upload support |
| httpx | No | Async HTTP client (CivitAI downloads, tests) |
| slowapi | No | Rate limiting |

**CRITICAL: Zero dependencies are pinned.** Builds are not reproducible.

**Dockerfile:** `/home/david/projects/wanly/wanly-api/Dockerfile`
- Multi-stage build: `python:3.12-slim` builder + runtime
- Installs `curl` and `ffmpeg` (for video stitching)
- Runs as non-root `appuser` (uid 1000)
- CMD: `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8001`
- No HEALTHCHECK instruction

---

## Section 4: Configuration & Infrastructure Summary

**File:** `/home/david/projects/wanly/wanly-api/app/config.py`

```python
class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    jwt_expiry_hours: int = 24
    s3_jobs_bucket: str = "wanly-jobs"
    s3_loras_bucket: str = "wanly-loras"
    s3_faces_bucket: str = "wanly-faces"
    s3_images_bucket: str = "wanly-images"
    aws_region: str = "us-west-2"
    api_key: str = ""
    civitai_api_token: str = ""
    cors_origins: str = ""
    login_rate_limit: str = "5/minute"

    model_config = {"env_file": ".env"}
```

No env prefix is used (reads DATABASE_URL, JWT_SECRET directly).

**File:** `/home/david/projects/wanly/wanly-api/.env.example`
- DATABASE_URL, JWT_SECRET, CORS_ORIGINS, LOGIN_RATE_LIMIT

**File:** `/home/david/projects/wanly/wanly-api/alembic.ini`
- `script_location = alembic`; placeholder URL overridden by `alembic/env.py` which uses `settings.database_url`

**CI/CD:** `/home/david/projects/wanly/wanly-api/.github/workflows/deploy.yml`
- Triggers on push to `main`
- OIDC auth to AWS, ECR build/push, SSM deploy to EC2
- Runs migrations in a one-off container before swapping
- Health check polls `http://localhost:8001/docs` (Swagger UI, not a dedicated health endpoint)

---

## Section 5: Startup & Runtime Behavior

**File:** `/home/david/projects/wanly/wanly-api/app/main.py`

1. Creates `FastAPI(title="wanly-api")`
2. Attaches slowapi rate limiter to app state
3. Configures CORS from comma-separated `settings.cors_origins`
4. Includes 10 routers: auth, images, jobs, segments, faceswap, files, loras, tags, wildcards, prompt_presets

No startup/shutdown events. No middleware beyond CORS. No structured logging configuration.

**File:** `/home/david/projects/wanly/wanly-api/app/database.py`

- Creates async engine from `settings.database_url`
- `async_session` factory with `expire_on_commit=False`
- `get_db()` generator dependency yields session

---

## Section 6: SQLAlchemy Models

**File:** `/home/david/projects/wanly/wanly-api/app/models.py`

### User
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| username | String(255) | unique, not null |
| password_hash | String(255) | not null |
| created_at | DateTime(tz) | default now() |

**Relationships:** `jobs` -> Job (back_populates="user")

### Job
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| user_id | UUID | FK users.id, not null, indexed |
| name | String(255) | not null |
| width | Integer | not null |
| height | Integer | not null |
| fps | Integer | not null |
| seed | BigInteger | not null |
| starting_image | Text | nullable |
| starting_image_hash | String(64) | nullable, indexed |
| lightx2v_strength_high | Float | nullable |
| lightx2v_strength_low | Float | nullable |
| priority | Integer | not null, default 0, indexed |
| status | String(20) | not null, default "pending", indexed |
| created_at | DateTime(tz) | default now() |
| updated_at | DateTime(tz) | default now(), onupdate now() |

**Relationships:** `user` -> User, `segments` -> Segment (ordered by index), `videos` -> Video

### Segment
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| job_id | UUID | FK jobs.id, not null, indexed |
| index | Integer | not null (unique with job_id) |
| prompt | Text | not null |
| prompt_template | Text | nullable |
| duration_seconds | Float | not null, default 5.0 |
| speed | Float | not null, default 1.0 |
| start_image | Text | nullable |
| loras | JSON | nullable |
| faceswap_enabled | Boolean | not null, default False |
| faceswap_method | String(20) | nullable |
| faceswap_source_type | String(20) | nullable |
| faceswap_image | Text | nullable |
| faceswap_faces_order | Text | nullable |
| faceswap_faces_index | Text | nullable |
| auto_finalize | Boolean | not null, default False |
| status | String(20) | not null, default "pending" |
| worker_id | UUID | nullable |
| worker_name | String(255) | nullable |
| output_path | Text | nullable |
| last_frame_path | Text | nullable |
| created_at | DateTime(tz) | default now() |
| claimed_at | DateTime(tz) | nullable |
| completed_at | DateTime(tz) | nullable |
| error_message | Text | nullable |
| progress_log | Text | nullable |

**Constraints:** `UniqueConstraint("job_id", "index", name="uq_segments_job_index")`
**Relationships:** `job` -> Job

### Video
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| job_id | UUID | FK jobs.id, not null, indexed |
| output_path | Text | nullable |
| duration_seconds | Float | nullable |
| status | String(20) | not null, default "pending" |
| error_message | Text | nullable |
| created_at | DateTime(tz) | default now() |
| completed_at | DateTime(tz) | nullable |

**Relationships:** `job` -> Job

### Lora
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| name | String(255) | not null |
| description | Text | nullable |
| trigger_words | Text | nullable |
| default_prompt | Text | nullable |
| source_url | Text | nullable |
| preview_image | Text | nullable |
| high_file | String(255) | nullable |
| high_s3_uri | Text | nullable |
| low_file | String(255) | nullable |
| low_s3_uri | Text | nullable |
| default_high_weight | Float | not null, default 1.0 |
| default_low_weight | Float | not null, default 1.0 |
| created_at | DateTime(tz) | default now() |
| updated_at | DateTime(tz) | default now(), onupdate now() |

### TitleTag
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| name | String(255) | not null |
| group | Integer | not null, indexed |
| created_at | DateTime(tz) | default now() |
| updated_at | DateTime(tz) | default now(), onupdate now() |

**Constraints:** `UniqueConstraint("name", "group", name="uq_title_tags_name_group")`

### Wildcard
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| name | String(255) | unique, not null |
| options | JSON | not null, default list |
| created_at | DateTime(tz) | default now() |
| updated_at | DateTime(tz) | default now(), onupdate now() |

### PromptPreset
| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, default uuid4 |
| name | String(255) | unique, not null |
| prompt | Text | not null |
| loras | JSON | nullable |
| created_at | DateTime(tz) | default now() |
| updated_at | DateTime(tz) | default now(), onupdate now() |

---

## Section 7: Enum Inventory

No Python enums are defined. Status values are implicit strings stored in String(20) columns.

**Job statuses (inferred from code):** `pending`, `processing`, `awaiting`, `failed`, `paused`, `finalized`, `finalizing`, `archived`

**Segment statuses (inferred from code):** `pending`, `claimed`, `processing`, `completed`, `failed`

**Video statuses (inferred from code):** `pending`, `completed`, `failed`

**Note:** These are not enforced at the database level (no CHECK constraints or PostgreSQL ENUMs).

---

## Section 8: Repository/DAO Layer

No dedicated repository/DAO layer exists. All database queries are inline in route handlers.

---

## Section 9: Service Layer

No dedicated service layer exists. Business logic is embedded in route handlers and two standalone modules:

### `app/stitch.py`
```python
async def stitch_video(video_id: UUID, job_id: UUID) -> None
```
Background task: downloads segment videos from S3, concatenates with ffmpeg (`-c copy`), uploads result. Creates its own `async_session` (not dependency-injected).

### `app/estimation.py`
```python
async def get_estimation_rates(db: AsyncSession, user_id: UUID) -> dict
def estimate_segment_time(rates: dict, width: int, height: int, fps: int, duration_seconds: float, worker_name: str | None = None) -> float | None
```
Computes average run-time-per-second rates from completed segments. Returns rates at config-level, worker+config level, and global fallback.

### `app/routes/segments.py` (helper functions)
```python
async def _resolve_loras(db: AsyncSession, loras_input: list | None) -> list | None
async def _resolve_wildcards(db: AsyncSession, prompt: str) -> tuple[str, str | None]
```
- `_resolve_loras`: Expands `{"lora_id": "<uuid>"}` entries to full S3 metadata for daemon consumption.
- `_resolve_wildcards`: Substitutes `<name>` placeholders in prompts with random choices from Wildcard table.

---

## Section 10: Router/API Layer

### Auth (`app/routes/auth.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| POST | `/login` | None (rate limited) | `async def login(request: Request, body: LoginRequest, db)` |

### Jobs (`app/routes/jobs.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| POST | `/jobs` | JWT | `async def create_job(data: str = Form, starting_image: UploadFile, faceswap_image: UploadFile, user, db)` |
| GET | `/jobs` | JWT | `async def list_jobs(limit, offset, status_filter, sort, user, db)` |
| PUT | `/jobs/reorder` | JWT | `async def reorder_jobs(body: JobReorderRequest, user, db)` |
| GET | `/jobs/{job_id}` | JWT | `async def get_job(job_id, user, db)` |
| PATCH | `/jobs/{job_id}` | JWT | `async def update_job(job_id, body: JobUpdate, background_tasks, user, db)` |
| POST | `/jobs/{job_id}/reopen` | JWT | `async def reopen_job(job_id, user, db)` |
| DELETE | `/jobs/{job_id}` | JWT | `async def delete_job(job_id, user, db)` |
| GET | `/stats` | JWT | `async def get_stats(user, db)` |

### Segments (`app/routes/segments.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/segments` | **None** | `async def list_segments(worker_id, limit, db)` |
| POST | `/jobs/{job_id}/segments` | JWT | `async def add_segment(job_id, body: SegmentCreate, user, db)` |
| GET | `/segments/next` | **None** | `async def claim_next_segment(worker_id, worker_name, db)` |
| PATCH | `/segments/{segment_id}` | **None** | `async def update_segment(segment_id, body: SegmentStatusUpdate, background_tasks, db)` |
| POST | `/segments/{segment_id}/retry` | JWT | `async def retry_segment(segment_id, user, db)` |
| POST | `/segments/{segment_id}/cancel` | JWT | `async def cancel_segment(segment_id, user, db)` |
| DELETE | `/segments/{segment_id}` | JWT | `async def delete_segment(segment_id, user, db)` |

### Files (`app/routes/files.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| POST | `/upload` | JWT | `async def upload_file(file: UploadFile, job_id, filename, _user)` |
| GET | `/files` | **None** | `async def download_file(path, request)` |
| POST | `/segments/{segment_id}/upload` | **None** | `async def upload_segment_output(segment_id, background_tasks, video: UploadFile, last_frame: UploadFile, db)` |

### Faceswap (`app/routes/faceswap.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/faceswap/presets` | JWT | `async def list_faceswap_presets(request, _user)` |

### Images (`app/routes/images.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| POST | `/images/upload` | API Key | `async def upload_image(file: UploadFile, filename)` |
| GET | `/images/folders` | JWT | `async def list_folders()` |
| GET | `/images/folder/{date}` | JWT | `async def list_folder_images(date)` |
| DELETE | `/images` | JWT | `async def delete_image(path)` |

### LoRAs (`app/routes/loras.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/loras` | JWT | `async def list_loras(_user, db)` |
| GET | `/loras/{lora_id}` | JWT | `async def get_lora(lora_id, _user, db)` |
| POST | `/loras` | JWT | `async def create_lora(body: LoraCreate, _user, db)` |
| POST | `/loras/upload` | JWT | `async def upload_lora(data: str = Form, high_file, low_file, preview_image, _user, db)` |
| PATCH | `/loras/{lora_id}` | JWT | `async def update_lora(lora_id, body: LoraUpdate, _user, db)` |
| DELETE | `/loras/{lora_id}` | JWT | `async def delete_lora(lora_id, _user, db)` |

### Tags (`app/routes/tags.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/tags` | JWT | `async def list_tags(group, _user, db)` |
| POST | `/tags` | JWT | `async def create_tag(body: TitleTagCreate, _user, db)` |
| DELETE | `/tags/{tag_id}` | JWT | `async def delete_tag(tag_id, _user, db)` |

### Wildcards (`app/routes/wildcards.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/wildcards` | JWT | `async def list_wildcards(user, db)` |
| POST | `/wildcards` | JWT | `async def create_wildcard(body: WildcardCreate, user, db)` |
| GET | `/wildcards/{wildcard_id}` | JWT | `async def get_wildcard(wildcard_id, user, db)` |
| PATCH | `/wildcards/{wildcard_id}` | JWT | `async def update_wildcard(wildcard_id, body: WildcardUpdate, user, db)` |
| DELETE | `/wildcards/{wildcard_id}` | JWT | `async def delete_wildcard(wildcard_id, user, db)` |

### Prompt Presets (`app/routes/prompt_presets.py`)
| Method | Path | Auth | Handler |
|--------|------|------|---------|
| GET | `/prompt-presets` | JWT | `async def list_prompt_presets(user, db)` |
| POST | `/prompt-presets` | JWT | `async def create_prompt_preset(body: PromptPresetCreate, user, db)` |
| GET | `/prompt-presets/{preset_id}` | JWT | `async def get_prompt_preset(preset_id, user, db)` |
| PATCH | `/prompt-presets/{preset_id}` | JWT | `async def update_prompt_preset(preset_id, body: PromptPresetUpdate, user, db)` |
| DELETE | `/prompt-presets/{preset_id}` | JWT | `async def delete_prompt_preset(preset_id, user, db)` |

**Total routes: 43** (37 JWT-protected, 1 API-key-protected, 5 unauthenticated)

---

## Section 11: Security Configuration

### Authentication
- **JWT:** HS256, `settings.jwt_secret`, 24-hour expiry (configurable)
- **API Key:** `X-API-Key` header for `/images/upload` only
- **Password hashing:** bcrypt via `bcrypt.hashpw`

### CORS
- Origins parsed from comma-separated `settings.cors_origins`
- `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`

### Rate Limiting
- Login endpoint: `5/minute` (configurable via `LOGIN_RATE_LIMIT`)
- Key function: `get_remote_address` (IP-based)
- No rate limiting on other endpoints

---

## Section 12: Custom Security Components

**File:** `/home/david/projects/wanly/wanly-api/app/auth.py`

```python
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(key: str = Depends(api_key_header))
def hash_password(password: str) -> str
def verify_password(password: str, password_hash: str) -> bool
def create_access_token(user_id: UUID) -> str
def decode_access_token(token: str) -> UUID
async def get_current_user(credentials, db) -> User
```

**Intentionally unauthenticated routes (daemon-facing):**
- `GET /segments/next` (claim next segment)
- `PATCH /segments/{segment_id}` (update segment status)
- `POST /segments/{segment_id}/upload` (upload segment output)
- `GET /segments` (list worker's segments)
- `GET /files` (download S3 file by URI)

These are used by GPU worker daemons which do not hold user credentials. Any client with network access to the API can call these endpoints.

---

## Section 13: Exception Handling

No global exception handler beyond the default FastAPI 422 validation and the slowapi `_rate_limit_exceeded_handler`.

Route-level pattern: `HTTPException` raised for 400, 401, 404, 422, 502 cases. Bare `except Exception` used in 13 places for best-effort S3 cleanup and background task error handling.

The `stitch_video` background task has its own try/except that records errors back to the Video/Job records.

---

## Section 14: Pydantic Schemas/Mappers

**File locations:** `/home/david/projects/wanly/wanly-api/app/schemas/`

| Schema | Mapped Model | Direction |
|--------|-------------|-----------|
| `LoginRequest` | - | Request |
| `TokenResponse` | - | Response |
| `JobCreate` | Job | Request (contains `SegmentCreate`) |
| `JobResponse` | Job | Response |
| `JobListResponse` | - | Response (paginated wrapper) |
| `JobDetailResponse` | Job | Response (extends JobResponse) |
| `JobUpdate` | Job | Request (partial) |
| `JobReorderRequest` | - | Request |
| `StatsResponse` | - | Response (aggregated) |
| `WorkerStatsItem` | - | Response (embedded) |
| `SegmentCreate` | Segment | Request |
| `SegmentResponse` | Segment | Response |
| `SegmentClaimResponse` | Segment+Job | Response (denormalized for daemon) |
| `SegmentStatusUpdate` | Segment | Request (partial) |
| `WorkerSegmentResponse` | Segment+Job | Response |
| `VideoResponse` | Video | Response |
| `LoraCreate` | Lora | Request |
| `LoraUpdate` | Lora | Request (partial) |
| `LoraResponse` | Lora | Response |
| `LoraListItem` | Lora | Response (summary) |
| `TitleTagCreate` | TitleTag | Request |
| `TitleTagResponse` | TitleTag | Response |
| `WildcardCreate` | Wildcard | Request |
| `WildcardUpdate` | Wildcard | Request (partial) |
| `WildcardResponse` | Wildcard | Response |
| `PromptPresetCreate` | PromptPreset | Request |
| `PromptPresetUpdate` | PromptPreset | Request (partial) |
| `PromptPresetResponse` | PromptPreset | Response |
| `LoraSlot` | - | Embedded in PromptPreset schemas |

All response schemas use `model_config = {"from_attributes": True}` for ORM mode.

---

## Section 15: Utility Modules

### `app/s3.py`
S3 client wrapper with lazy singleton boto3 client.

```python
def _get_client()
def upload_bytes(data: bytes, key: str, bucket: str) -> str
def upload_file(path: str, key: str, bucket: str) -> str
def download_bytes(uri: str) -> bytes
def delete_prefix(prefix: str, bucket: str) -> int
def delete_object(uri: str) -> None
def download_file(uri: str, local_path: str) -> None
def list_common_prefixes(bucket: str, prefix: str = "", delimiter: str = "/") -> list[str]
def list_objects(bucket: str, prefix: str) -> list[dict]
def get_first_object_key(bucket: str, prefix: str) -> str | None
def parse_s3_uri(uri: str) -> tuple[str, str]
```

All functions are synchronous (called via `asyncio.to_thread` from route handlers).

### `app/limiter.py`
```python
limiter = Limiter(key_func=get_remote_address)
```

### `app/estimation.py`
Segment run-time estimation (documented in Section 9).

### `app/stitch.py`
Video stitching background task (documented in Section 9).

---

## Section 16: Database Schema (live)

Skipped (no local database connection).

---

## Section 17: Message Broker

No message broker (RabbitMQ, Kafka, SQS, etc.) is used. Job dispatch is pull-based: daemons poll `GET /segments/next`.

---

## Section 18: Cache Layer

No cache layer (Redis, Memcached, etc.) is used in this service.

ETag-based caching is implemented for `GET /files` responses on cacheable file extensions (images/video), with `Cache-Control: public, max-age=86400, immutable`.

---

## Section 19: Environment Variable Inventory

| Variable | Required | Default | Source |
|----------|----------|---------|--------|
| DATABASE_URL | Yes | - | .env |
| JWT_SECRET | Yes | - | .env |
| JWT_EXPIRY_HOURS | No | 24 | .env |
| S3_JOBS_BUCKET | No | "wanly-jobs" | .env |
| S3_LORAS_BUCKET | No | "wanly-loras" | .env |
| S3_FACES_BUCKET | No | "wanly-faces" | .env |
| S3_IMAGES_BUCKET | No | "wanly-images" | .env |
| AWS_REGION | No | "us-west-2" | .env |
| API_KEY | No | "" | .env |
| CIVITAI_API_TOKEN | No | "" | .env |
| CORS_ORIGINS | No | "" | .env |
| LOGIN_RATE_LIMIT | No | "5/minute" | .env |

No env prefix is used (Settings reads bare variable names).

---

## Section 20: Service Dependency Map

```
wanly-api
├── PostgreSQL (DATABASE_URL, asyncpg)
├── AWS S3 (4 buckets: wanly-jobs, wanly-loras, wanly-faces, wanly-images)
├── CivitAI API (optional, for LoRA preview/download)
└── ffmpeg (subprocess, for video stitching)
```

**Inbound consumers:**
- wanly-console (React frontend, JWT-authenticated)
- wanly-gpu-daemon (GPU workers, unauthenticated daemon endpoints)
- External tools via API key (`/images/upload`)

---

## Section 21: Known Technical Debt

### No TODO/FIXME/HACK markers found in source code.

### Architectural Debt

1. **No service/repository layer** — All business logic is inline in route handlers. Route files are 400+ lines. Code duplication exists between `update_segment`, `upload_segment_output`, and `cancel_segment` for job status transition logic.

2. **No CASCADE deletes on foreign keys** — Job deletion manually deletes segments and videos in Python. A missed FK relationship would leave orphan rows.

3. **Status values are not enforced** — No Python enums, no DB CHECK constraints. A typo in a status string would silently corrupt data.

4. **Daemon endpoints are fully unauthenticated** — 5 routes (segment claim, update, upload, list; file download) have no auth. Any network-reachable client can claim jobs, update statuses, or download S3 content.

5. **No health check endpoint** — Deployment health check hits `/docs` (Swagger UI). No readiness/liveness probes for proper orchestration.

6. **Unpinned dependencies** — All 13 requirements are unpinned. Docker builds are non-reproducible.

7. **Background task session management** — `stitch_video` creates its own `async_session()` outside the dependency injection system, which could mask connection pool issues.

8. **`delete_prefix` S3 cleanup is not paginated** — `list_objects_v2` returns max 1000 objects; if a job has >1000 objects, not all will be deleted.

9. **Seeded user with hardcoded bcrypt hash** — Migration `001` inserts a `dbarnes` user with a static password hash.

10. **Job status transition map inconsistency** — The test file (`test_job_status_transitions.py`) has a different transition map than `app/routes/jobs.py` (test omits `archived` transitions).

---

## Appendix: File Index

```
app/
├── __init__.py              (empty)
├── auth.py                  (JWT + password auth, 58 lines)
├── config.py                (Settings via pydantic-settings, 22 lines)
├── database.py              (async engine + session factory, 12 lines)
├── estimation.py            (segment run-time estimation, 110 lines)
├── limiter.py               (slowapi limiter instance, 4 lines)
├── main.py                  (FastAPI app setup, 37 lines)
├── models.py                (8 SQLAlchemy models, 162 lines)
├── s3.py                    (S3 client wrapper, 121 lines)
├── stitch.py                (video stitching background task, 112 lines)
├── routes/
│   ├── __init__.py          (empty)
│   ├── auth.py              (login, 24 lines)
│   ├── faceswap.py          (presets from S3, 54 lines)
│   ├── files.py             (upload/download + segment upload, 163 lines)
│   ├── images.py            (image CRUD in S3, 82 lines)
│   ├── jobs.py              (job CRUD + stats, 564 lines)
│   ├── loras.py             (LoRA management + CivitAI, 403 lines)
│   ├── prompt_presets.py    (preset CRUD, 91 lines)
│   ├── segments.py          (segment CRUD + claim, 450 lines)
│   ├── tags.py              (tag CRUD, 68 lines)
│   └── wildcards.py         (wildcard CRUD, 85 lines)
├── schemas/
│   ├── __init__.py          (re-exports, 24 lines)
│   ├── auth.py              (LoginRequest, TokenResponse)
│   ├── jobs.py              (JobCreate, JobResponse, etc.)
│   ├── loras.py             (LoraCreate, LoraResponse, etc.)
│   ├── prompt_presets.py    (PromptPresetCreate, etc.)
│   ├── segments.py          (SegmentCreate, SegmentClaimResponse, etc.)
│   ├── tags.py              (TitleTagCreate, TitleTagResponse)
│   ├── videos.py            (VideoResponse)
│   └── wildcards.py         (WildcardCreate, WildcardResponse)
alembic/
├── env.py                   (async migration runner)
└── versions/                (14 migration files, 001-014)
tests/
├── __init__.py              (empty)
├── conftest.py              (env var defaults)
├── test_auth.py             (password + JWT + upload auth tests)
├── test_cors.py             (CORS policy test)
├── test_faceswap_presets.py (S3 mock + auth tests)
├── test_job_status_transitions.py (state machine tests)
├── test_rate_limit.py       (login rate limiting test)
├── test_resolve_loras.py    (LoRA resolution unit tests)
└── test_resolve_wildcards.py (wildcard resolution unit tests)
```
