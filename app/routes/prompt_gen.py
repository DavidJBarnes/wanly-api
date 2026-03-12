import asyncio
import base64
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import get_current_user
from app.config import settings
from app.models import User
from app.s3 import download_bytes
from app.schemas.prompt_gen import (
    PromptGenRequest,
    PromptGenResponse,
    PromptTemplatesResponse,
    PromptTemplatesUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory template overrides (reset on restart)
_template_overrides: dict[str, str] = {}


def _get_template(key: str) -> str:
    return _template_overrides.get(key, getattr(settings, f"ollama_{key}"))


@router.post("/prompt/generate", response_model=PromptGenResponse)
async def generate_prompt(
    body: PromptGenRequest,
    user: User = Depends(get_current_user),
):
    # Step 1: Resolve image to base64
    if body.image_s3_uri:
        try:
            raw = await asyncio.to_thread(download_bytes, body.image_s3_uri)
            image_b64 = base64.b64encode(raw).decode()
        except Exception:
            logger.exception("Failed to download image from S3: %s", body.image_s3_uri)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to download image from S3",
            )
    else:
        image_b64 = body.image_base64

    ollama_url = settings.ollama_url

    # Step 2: Vision model — describe the image
    vision_payload = {
        "model": _get_template("vision_model"),
        "prompt": _get_template("vision_prompt"),
        "images": [image_b64],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"{ollama_url}/api/generate", json=vision_payload)
            resp.raise_for_status()
            description = resp.json()["response"]
        except Exception:
            logger.exception("Ollama vision call failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama vision model failed — is Ollama running?",
            )

    # Step 3: Text model — generate video prompt from description
    prefix = body.prompt_prefix or ""
    template = _get_template("text_prompt")
    # Strip " of {prefix}" when no prefix provided
    if not prefix and " of {prefix}" in template:
        template = template.replace(" of {prefix}", "")
    text_prompt = template.replace("{prefix}", prefix).replace("{description}", description)

    text_payload = {
        "model": _get_template("text_model"),
        "prompt": text_prompt,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"{ollama_url}/api/generate", json=text_payload)
            resp.raise_for_status()
            generated_prompt = resp.json()["response"]
        except Exception:
            logger.exception("Ollama text call failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama text model failed — is Ollama running?",
            )

    return PromptGenResponse(prompt=generated_prompt.strip(), description=description.strip())


@router.get("/prompt/templates", response_model=PromptTemplatesResponse)
async def get_templates(user: User = Depends(get_current_user)):
    return PromptTemplatesResponse(
        vision_prompt=_get_template("vision_prompt"),
        text_prompt=_get_template("text_prompt"),
        vision_model=_get_template("vision_model"),
        text_model=_get_template("text_model"),
    )


@router.put("/prompt/templates", response_model=PromptTemplatesResponse)
async def update_templates(
    body: PromptTemplatesUpdate,
    user: User = Depends(get_current_user),
):
    if body.vision_prompt is not None:
        _template_overrides["vision_prompt"] = body.vision_prompt
    if body.text_prompt is not None:
        _template_overrides["text_prompt"] = body.text_prompt
    if body.vision_model is not None:
        _template_overrides["vision_model"] = body.vision_model
    if body.text_model is not None:
        _template_overrides["text_model"] = body.text_model

    return PromptTemplatesResponse(
        vision_prompt=_get_template("vision_prompt"),
        text_prompt=_get_template("text_prompt"),
        vision_model=_get_template("vision_model"),
        text_model=_get_template("text_model"),
    )
