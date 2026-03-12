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

RUNPOD_TIMEOUT = 300.0


def _get_template(key: str) -> str:
    return _template_overrides.get(key, getattr(settings, f"ollama_{key}"))


def _runpod_url() -> str:
    eid = settings.runpod_ollama_endpoint_id
    return f"https://api.runpod.ai/v2/{eid}/runsync"


def _runpod_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.runpod_api_key}",
        "Content-Type": "application/json",
    }


async def _runpod_vision(image_b64: str, prompt: str, model: str) -> str:
    """Call RunPod serverless Ollama with a vision model (OpenAI chat format)."""
    payload = {
        "input": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }
    }
    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT) as client:
        resp = await client.post(_runpod_url(), headers=_runpod_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") == "FAILED":
        raise RuntimeError(data.get("error", "RunPod vision job failed"))

    output = data.get("output")
    logger.info("RunPod vision response structure: %s", type(output))

    # RunPod may wrap the OpenAI response differently depending on worker version
    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
    elif isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict) and "message" in first:
            return first["message"]["content"]
        if isinstance(first, dict) and "choices" in first:
            return first["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unexpected RunPod response: {data}")


async def _runpod_text(prompt: str, model: str) -> str:
    """Call RunPod serverless Ollama with a text model (OpenAI chat format)."""
    payload = {
        "input": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        }
    }
    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT) as client:
        resp = await client.post(_runpod_url(), headers=_runpod_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") == "FAILED":
        raise RuntimeError(data.get("error", "RunPod text job failed"))

    output = data.get("output")
    logger.info("RunPod text response structure: %s", type(output))

    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
    elif isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict) and "message" in first:
            return first["message"]["content"]
        if isinstance(first, dict) and "choices" in first:
            return first["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unexpected RunPod response: {data}")


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

    # Step 2: Vision model — describe the image
    try:
        description = await _runpod_vision(
            image_b64,
            _get_template("vision_prompt"),
            _get_template("vision_model"),
        )
    except Exception:
        logger.exception("RunPod vision call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vision model failed — check RunPod endpoint",
        )

    # Step 3: Text model — generate video prompt from description
    prefix = body.prompt_prefix or ""
    template = _get_template("text_prompt")
    if not prefix and " of {prefix}" in template:
        template = template.replace(" of {prefix}", "")
    text_prompt = template.replace("{prefix}", prefix).replace("{description}", description)

    try:
        generated_prompt = await _runpod_text(
            text_prompt,
            _get_template("text_model"),
        )
    except Exception:
        logger.exception("RunPod text call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Text model failed — check RunPod endpoint",
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
