# Plan: Add Cache-Control Headers to File Responses

## Context

The `/files` endpoint (`app/routes/files.py:47-82`) proxies S3 objects through the API server with **zero caching headers**. Every browser navigation or re-render triggers a full S3 round-trip — even for identical, immutable content like completed segment thumbnails.

Segment outputs (`last_frame_path`, `output_path`) and uploaded images (`starting_image`, faceswap faces) are **write-once, never modified**. This makes them ideal candidates for aggressive caching.

## Changes

### Step 1: Add Cache-Control headers to `GET /files` response

**File:** `app/routes/files.py` — `download_file()` function (line 47-82)

Add `Cache-Control` and `ETag` headers to the `Response` based on content type:

- **Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.avif`, `.gif`): `Cache-Control: public, max-age=86400, immutable` — browsers cache for 24 hours and never revalidate. These are segment last-frames and uploaded images that never change.
- **Videos** (`.mp4`): `Cache-Control: public, max-age=86400, immutable` — same rationale, segment videos and final stitched videos are immutable once created.
- **Other** (`.safetensors`, unknown): `Cache-Control: no-store` — large binary blobs served to daemons, no browser caching benefit.

Add an `ETag` header derived from the S3 path itself (since the content at a given path is immutable, the path is a stable identifier). This enables `304 Not Modified` responses if we add conditional-GET support later.

### Step 2: Handle `If-None-Match` for conditional requests

Check the incoming `If-None-Match` request header. If it matches our ETag, return `304 Not Modified` immediately **without downloading from S3**. This is the biggest performance win — repeat thumbnail loads skip the S3 download entirely.

**File:** `app/routes/files.py` — `download_file()` function

- Accept `request: Request` parameter to read headers
- Compute ETag from the S3 path (hash)
- If `If-None-Match` matches, return `Response(status_code=304)` before the S3 call
- Otherwise proceed as normal and include the ETag in the response

### Summary of changes

Only **one file** is modified: `app/routes/files.py`

Changes:
1. Import `hashlib` and `Request` from FastAPI
2. Add `request: Request` parameter to `download_file()`
3. Compute ETag from path, check `If-None-Match`, short-circuit with 304
4. Set `Cache-Control` header (aggressive for images/video, `no-store` for other)
5. Set `ETag` header on all cacheable responses
