# wanly-api — Quality Scorecard

**Audit Date:** 2026-03-12T15:58:56Z
**Branch:** main
**Commit:** 26849be1298459ddfebe25f12545e8f1fd4192ca

---

## Security

| # | Check | Status | Notes |
|---|-------|--------|-------|
| S1 | No hardcoded secrets in source | PASS | All secrets via env vars / pydantic-settings |
| S2 | .env in .gitignore | PASS | `.env` and `.envrc` in .gitignore |
| S3 | No raw SQL / SQL injection risk | PASS | No `text()` calls with user input; all queries via SQLAlchemy ORM |
| S4 | No exec/eval usage | PASS | None found |
| S5 | Auth on all user-facing routes | PASS | All console-facing routes require JWT |
| S6 | Auth on daemon-facing routes | **FAIL** | 5 routes (segment claim, update, upload, list; file download) have zero auth. Any network client can claim/modify jobs. |
| S7 | Password hashing | PASS | bcrypt with random salt |
| S8 | JWT expiry enforcement | PASS | 24h default, exp claim checked on decode |
| S9 | Rate limiting on auth endpoints | PASS | Login limited to 5/minute via slowapi |
| S10 | CORS not wildcard | PASS | Explicit origins from config; empty = no CORS |
| S11 | Non-root Docker user | PASS | `appuser` uid 1000 |
| S12 | Sensitive data in migration seeds | **WARN** | Migration 001 contains a hardcoded bcrypt hash for user `dbarnes` |

**Security Score: 9/12 (75%)**

---

## Data Integrity

| # | Check | Status | Notes |
|---|-------|--------|-------|
| D1 | FK constraints on all relationships | PASS | users->jobs, jobs->segments, jobs->videos |
| D2 | CASCADE deletes configured | **FAIL** | No `ondelete="CASCADE"` on any FK. Manual Python deletion in route handlers. |
| D3 | Unique constraints where needed | PASS | segments(job_id, index), title_tags(name, group), wildcards(name), prompt_presets(name) |
| D4 | Status values enforced | **FAIL** | String columns with no CHECK constraints or Python enums. Invalid status strings would be silently accepted. |
| D5 | Atomic job claim | PASS | `with_for_update(skip_locked=True)` on segment claim |
| D6 | Stale claim recovery | PASS | 30-minute timeout resets stale claimed/processing segments |
| D7 | Alembic migrations present | PASS | 14 migrations covering all schema changes |
| D8 | Transaction management | PASS | Single commit per request via session dependency |
| D9 | Segment re-indexing safety | PASS | Uses negative temp indices to avoid unique constraint conflicts |

**Data Integrity Score: 7/9 (78%)**

---

## API Quality

| # | Check | Status | Notes |
|---|-------|--------|-------|
| A1 | Consistent response models | PASS | All routes declare `response_model` |
| A2 | Pagination on list endpoints | PASS | Jobs list has limit/offset/total. Segments and other lists have limit. |
| A3 | Health check endpoint | **FAIL** | No dedicated health/readiness/liveness endpoint. CI health check hits /docs. |
| A4 | OpenAPI/Swagger available | PASS | FastAPI auto-generates at /docs |
| A5 | Error responses use HTTPException | PASS | Consistent 400/401/404/422 error handling |
| A6 | Status transition validation | PASS | Explicit state machine in update_job with allowed transition map |
| A7 | Input validation | PASS | Pydantic schemas on all request bodies; speed field has ge/le constraints |
| A8 | File upload size limits | **WARN** | No explicit size limits on file uploads (starting images, LoRA files, etc.) |
| A9 | ETag caching on file downloads | PASS | Hash-based ETag + 304 support for cacheable extensions |

**API Quality Score: 7/9 (78%)**

---

## Code Quality

| # | Check | Status | Notes |
|---|-------|--------|-------|
| C1 | No TODO/FIXME/HACK in code | PASS | None found (one false positive in docstring) |
| C2 | Service/repository layer separation | **FAIL** | All business logic inline in route handlers. Large route files (jobs.py: 564 lines). |
| C3 | DRY — no duplicated logic | **WARN** | Job status update logic duplicated across update_segment, upload_segment_output, and cancel_segment. Job detail response construction duplicated in get_job and reopen_job. |
| C4 | Consistent logging | PASS | `logging.getLogger(__name__)` in all modules that log |
| C5 | Type hints | PASS | All function signatures have type annotations |
| C6 | Async consistency | PASS | All route handlers are async; sync S3/ffmpeg calls wrapped in asyncio.to_thread |
| C7 | No circular imports | PASS | Clean import graph |
| C8 | Config via pydantic-settings | PASS | Single Settings class, env_file support |

**Code Quality Score: 6/8 (75%)**

---

## Test Quality

| # | Check | Status | Notes |
|---|-------|--------|-------|
| T1 | Tests exist | PASS | 34 test functions across 7 test files |
| T2 | Test coverage adequate | **FAIL** | 34 tests for 43 routes. No integration tests. No tests for jobs CRUD, segments CRUD, LoRA management, wildcards, tags, prompt presets, stitch, or estimation. |
| T3 | Auth tests | PASS | Password hashing, JWT encode/decode, expired token, wrong secret, missing claims |
| T4 | State machine tests | PASS | Valid/invalid transitions, terminal states, pause symmetry |
| T5 | Mock isolation | PASS | DB and S3 mocked in unit tests; no real AWS calls |
| T6 | Rate limit tests | PASS | Verifies 429 after 5th login attempt |
| T7 | Business logic unit tests | PASS | _resolve_loras and _resolve_wildcards thoroughly tested |
| T8 | pytest-asyncio configured | PASS | `asyncio_mode = auto` in pytest.ini |

**Test Quality Score: 7/8 (88%)**

---

## Infrastructure

| # | Check | Status | Notes |
|---|-------|--------|-------|
| I1 | Dockerfile present | PASS | Multi-stage, non-root, ffmpeg installed |
| I2 | CI/CD pipeline | PASS | GitHub Actions: build, push ECR, SSM deploy, health check |
| I3 | Dependencies pinned | **FAIL** | 0 of 13 dependencies pinned in requirements.txt |
| I4 | Docker HEALTHCHECK | **FAIL** | No HEALTHCHECK instruction in Dockerfile |
| I5 | Migrations auto-run | PASS | CMD runs `alembic upgrade head` before uvicorn |
| I6 | .gitignore comprehensive | PASS | Standard Python .gitignore with .env, venv, cache exclusions |
| I7 | Environment separation | PASS | .env.example provided; tests use separate env defaults |
| I8 | S3 cleanup on delete | PASS | Job deletion cleans up S3 prefix; segment deletion cleans individual objects |

**Infrastructure Score: 6/8 (75%)**

---

## Summary

| Category | Score | Percentage |
|----------|-------|------------|
| Security | 9/12 | 75% |
| Data Integrity | 7/9 | 78% |
| API Quality | 7/9 | 78% |
| Code Quality | 6/8 | 75% |
| Test Quality | 7/8 | 88% |
| Infrastructure | 6/8 | 75% |
| **Overall** | **42/54** | **78%** |

### Top Priority Items

1. **Pin all dependencies** in requirements.txt for reproducible builds
2. **Add a health check endpoint** (`GET /health`) and Docker HEALTHCHECK
3. **Add authentication to daemon-facing routes** (API key or mutual TLS)
4. **Define status enums** as Python Enum classes and/or database CHECK constraints
5. **Add CASCADE deletes** to foreign keys (or at minimum, keep the Python-level cleanup)
6. **Extract service layer** to reduce route handler complexity and eliminate duplication
7. **Add integration tests** for the core job->segment lifecycle flow
