# Plan: Fix CORS + Add Login Rate Limiting

## Issue #2 — CORS allows all origins

**Problem:** `app/main.py:10` has `allow_origins=["*"]` with `allow_credentials=True`. Any website can make authenticated API requests on behalf of a logged-in user's browser.

**Approach:** Make the allowed origins configurable via an environment variable with a safe default.

### Steps

1. **`app/config.py`** — Add a `cors_origins: str = ""` field to `Settings`
   - Accepts a comma-separated string (e.g. `CORS_ORIGINS=https://my-app.com,https://staging.my-app.com`)
   - Empty string = no origins allowed (secure default)

2. **`app/main.py`** — Parse `settings.cors_origins` into a list and pass it to `CORSMiddleware`
   - Split on commas, strip whitespace, filter empty strings
   - Only add the middleware if at least one origin is configured
   - Remove the hardcoded `["*"]`

3. **`.env.example`** — Add `CORS_ORIGINS=` with a comment showing the format

4. **`tests/test_auth.py`** — Add a test that verifies a request from a non-allowed origin does NOT receive permissive CORS headers

### Files changed
- `app/config.py` (add field)
- `app/main.py` (read from settings, conditional middleware)
- `.env.example` (document new var)
- `tests/test_auth.py` (regression test)

---

## Issue #3 — No rate limiting on `/login`

**Problem:** `app/routes/auth.py` accepts unlimited login attempts. An attacker can brute-force passwords at thousands of requests per second.

**Approach:** Use `slowapi` (the standard FastAPI rate-limiting library, built on `limits`). Apply a strict per-IP limit to `/login` only — no global middleware that could affect daemon endpoints.

### Steps

1. **`requirements.txt`** — Add `slowapi`

2. **`app/main.py`** — Wire up slowapi's exception handler
   - Import `SlowAPIMiddleware` or the `_rate_limit_exceeded_handler`
   - Register the 429 exception handler on the app
   - Create a shared `Limiter` instance (key_func = client IP from `Request`)

3. **`app/routes/auth.py`** — Decorate `POST /login` with a rate limit
   - `@limiter.limit("5/minute")` — 5 attempts per IP per minute
   - Add `request: Request` parameter to the route (required by slowapi)

4. **`tests/test_auth.py`** — Add a test that verifies the 6th login attempt within a minute returns 429

### Files changed
- `requirements.txt` (add slowapi)
- `app/main.py` (register exception handler)
- `app/routes/auth.py` (apply rate limit decorator)
- `tests/test_auth.py` (regression test)

---

## Order of operations

1. CORS fix first (simpler, no new dependencies beyond what we have)
2. Rate limiting second (requires installing `slowapi`)
3. Run full test suite after each change
4. Single commit per fix, then push
