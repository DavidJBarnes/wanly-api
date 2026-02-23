import asyncio
import logging
import os
import re
import subprocess
import tempfile
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Lora, User
from app.s3 import delete_prefix, upload_bytes, upload_file
LORAS_BUCKET = settings.s3_loras_bucket
from app.schemas.loras import LoraCreate, LoraListItem, LoraResponse, LoraUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filename_from_url(url: str) -> str:
    """Extract filename from a URL path, stripping query params."""
    path = url.split("?")[0].split("#")[0]
    name = path.rsplit("/", 1)[-1]
    return name if name else "model.safetensors"


def _filename_from_response(resp: httpx.Response, url: str) -> str:
    """Get filename from Content-Disposition header or fall back to URL."""
    cd = resp.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', cd)
    if match:
        return match.group(1).strip()
    return _filename_from_url(url)


def _civitai_auth_url(url: str) -> str:
    """Append CivitAI API token to download URLs if configured."""
    if "civitai.com" not in url:
        return url
    token = settings.civitai_api_token
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={token}"


async def _download_to_temp(url: str) -> tuple[str, str]:
    """Stream-download a file from a URL to a temp file. Returns (temp_path, filename).

    Streams in chunks to avoid loading the entire file into memory (critical
    for the t3.micro with 1GB RAM handling 50-500MB .safetensors files).
    """
    download_url = _civitai_auth_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=600, headers=headers) as client:
        async with client.stream("GET", download_url) as resp:
            resp.raise_for_status()
            # Detect auth-redirect: CivitAI returns HTML login page instead of file
            ct = resp.headers.get("content-type", "")
            if "text/html" in ct:
                raise RuntimeError(
                    "CivitAI returned HTML instead of a file — "
                    "this model likely requires authentication. "
                    "Set CIVITAI_API_TOKEN in .env."
                )
            filename = _filename_from_response(resp, url)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
            try:
                async for chunk in resp.aiter_bytes(chunk_size=8 * 1024 * 1024):
                    tmp.write(chunk)
                tmp.close()
                return tmp.name, filename
            except Exception:
                tmp.close()
                os.unlink(tmp.name)
                raise


def _parse_civitai_model_id(url: str) -> int | None:
    """Extract model ID from a CivitAI URL like https://civitai.com/models/12345/..."""
    match = re.search(r"civitai\.com/models/(\d+)", url)
    return int(match.group(1)) if match else None


def _ext_from_content_type(content_type: str) -> str:
    """Map Content-Type to file extension."""
    ct = content_type.split(";")[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/avif": ".avif",
        "image/gif": ".gif",
    }.get(ct, ".jpg")


def _extract_first_frame_from_file(video_path: str) -> bytes | None:
    """Extract first frame from a video file as WebP using ffmpeg."""
    out_path = video_path + ".webp"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vframes", "1", "-q:v", "80", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            logger.warning("ffmpeg failed: %s", result.stderr.decode())
            return None
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


async def _fetch_civitai_preview(source_url: str) -> tuple[bytes, str] | None:
    """Fetch preview from CivitAI. Handles both static images and videos.

    For videos, extracts the first frame as WebP using ffmpeg (matching v1).
    Returns (image_bytes, extension) or None.
    """
    model_id = _parse_civitai_model_id(source_url)
    if model_id is None:
        return None
    try:
        # User-Agent required — CloudFlare blocks requests without one
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with httpx.AsyncClient(follow_redirects=True, timeout=120, headers=headers) as client:
            resp = await client.get(f"https://civitai.com/api/v1/models/{model_id}")
            resp.raise_for_status()
            data = resp.json()

            versions = data.get("modelVersions", [])
            if not versions:
                return None
            images = versions[0].get("images", [])
            if not images:
                return None

            # Prefer static images over videos
            target = next((img for img in images if img.get("type") == "image"), None)
            is_video = target is None
            if not target:
                target = images[0]
            img_url = target.get("url", "")
            if not img_url:
                return None

            # Resize static images only — videos return 500 with width param
            if "original=true" in img_url and not is_video:
                img_url = img_url.replace("original=true", "width=512")

            if is_video:
                # Stream video to temp file to avoid OOM, then extract frame
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                try:
                    async with client.stream("GET", img_url) as stream:
                        stream.raise_for_status()
                        async for chunk in stream.aiter_bytes(chunk_size=1024 * 1024):
                            tmp.write(chunk)
                    tmp.close()
                    frame_data = await asyncio.to_thread(_extract_first_frame_from_file, tmp.name)
                    if frame_data:
                        return frame_data, ".webp"
                    return None
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
            else:
                img_resp = await client.get(img_url)
                img_resp.raise_for_status()
                ext = _ext_from_content_type(img_resp.headers.get("content-type", ""))
                return img_resp.content, ext
    except Exception:
        logger.warning("Failed to fetch CivitAI preview for %s", source_url, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/loras", response_model=list[LoraListItem])
async def list_loras(
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Lora).order_by(Lora.created_at.desc()))
    return result.scalars().all()


@router.get("/loras/{lora_id}", response_model=LoraResponse)
async def get_lora(
    lora_id: UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await db.get(Lora, lora_id)
    if lora is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LoRA not found")
    return lora


@router.post("/loras", response_model=LoraResponse, status_code=status.HTTP_201_CREATED)
async def create_lora(
    body: LoraCreate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a LoRA by providing metadata and optional download URLs.

    The API server downloads .safetensors files from the provided URLs and
    stores them in S3. If source_url is a CivitAI link, the preview image
    is automatically fetched.
    """
    lora = Lora(
        name=body.name,
        description=body.description,
        trigger_words=body.trigger_words,
        default_prompt=body.default_prompt,
        source_url=body.source_url,
        default_high_weight=body.default_high_weight,
        default_low_weight=body.default_low_weight,
    )
    db.add(lora)
    await db.flush()  # get lora.id

    prefix = f"loras/{lora.id}"

    # Download high-noise file (streamed to temp file to avoid OOM)
    if body.high_url:
        tmp_path = None
        try:
            tmp_path, filename = await _download_to_temp(body.high_url)
            key = f"{prefix}/{filename}"
            uri = await asyncio.to_thread(upload_file, tmp_path, key, LORAS_BUCKET)
            lora.high_file = filename
            lora.high_s3_uri = uri
        except Exception:
            logger.error("Failed to download high-noise LoRA from %s", body.high_url, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download high-noise file from {body.high_url}",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Download low-noise file (streamed to temp file to avoid OOM)
    if body.low_url:
        tmp_path = None
        try:
            tmp_path, filename = await _download_to_temp(body.low_url)
            key = f"{prefix}/{filename}"
            uri = await asyncio.to_thread(upload_file, tmp_path, key, LORAS_BUCKET)
            lora.low_file = filename
            lora.low_s3_uri = uri
        except Exception:
            logger.error("Failed to download low-noise LoRA from %s", body.low_url, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download low-noise file from {body.low_url}",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Auto-fetch CivitAI preview
    if body.source_url and _parse_civitai_model_id(body.source_url):
        result = await _fetch_civitai_preview(body.source_url)
        if result:
            preview_data, ext = result
            key = f"{prefix}/preview{ext}"
            uri = await asyncio.to_thread(upload_bytes, preview_data, key, LORAS_BUCKET)
            lora.preview_image = uri

    await db.commit()
    await db.refresh(lora)
    return lora


@router.post("/loras/upload", response_model=LoraResponse, status_code=status.HTTP_201_CREATED)
async def upload_lora(
    data: str = Form(...),
    high_file: UploadFile | None = File(None),
    low_file: UploadFile | None = File(None),
    preview_image: UploadFile | None = File(None),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a LoRA by uploading files directly (fallback for non-URL sources)."""
    try:
        body = LoraCreate.model_validate_json(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in data field: {e}",
        )

    lora = Lora(
        name=body.name,
        description=body.description,
        trigger_words=body.trigger_words,
        default_prompt=body.default_prompt,
        source_url=body.source_url,
        default_high_weight=body.default_high_weight,
        default_low_weight=body.default_low_weight,
    )
    db.add(lora)
    await db.flush()

    prefix = f"loras/{lora.id}"

    if high_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            while chunk := await high_file.read(8 * 1024 * 1024):
                tmp.write(chunk)
            tmp.close()
            filename = high_file.filename or "high.safetensors"
            key = f"{prefix}/{filename}"
            uri = await asyncio.to_thread(upload_file, tmp.name, key, LORAS_BUCKET)
            lora.high_file = filename
            lora.high_s3_uri = uri
        finally:
            tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    if low_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            while chunk := await low_file.read(8 * 1024 * 1024):
                tmp.write(chunk)
            tmp.close()
            filename = low_file.filename or "low.safetensors"
            key = f"{prefix}/{filename}"
            uri = await asyncio.to_thread(upload_file, tmp.name, key, LORAS_BUCKET)
            lora.low_file = filename
            lora.low_s3_uri = uri
        finally:
            tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    if preview_image is not None:
        img_data = await preview_image.read()
        ext = os.path.splitext(preview_image.filename or "preview.jpg")[1] or ".jpg"
        key = f"{prefix}/preview{ext}"
        uri = await asyncio.to_thread(upload_bytes, img_data, key, LORAS_BUCKET)
        lora.preview_image = uri

    await db.commit()
    await db.refresh(lora)
    return lora


@router.patch("/loras/{lora_id}", response_model=LoraResponse)
async def update_lora(
    lora_id: UUID,
    body: LoraUpdate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await db.get(Lora, lora_id)
    if lora is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LoRA not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(lora, field, value)

    await db.commit()
    await db.refresh(lora)
    return lora


@router.delete("/loras/{lora_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lora(
    lora_id: UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    lora = await db.get(Lora, lora_id)
    if lora is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LoRA not found")

    # Delete S3 files
    prefix = f"loras/{lora_id}"
    await asyncio.to_thread(delete_prefix, prefix, LORAS_BUCKET)

    await db.delete(lora)
    await db.commit()
