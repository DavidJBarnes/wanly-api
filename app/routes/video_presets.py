from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User, VideoSettingsPreset
from app.schemas.video_presets import VideoPresetCreate, VideoPresetResponse, VideoPresetUpdate

router = APIRouter()

_PARAMS = (
    "lightx2v_strength_high",
    "lightx2v_strength_low",
    "cfg_high",
    "cfg_low",
    "steps_total",
    "high_noise_steps",
    "flow_shift",
)


@router.get("/video-presets", response_model=list[VideoPresetResponse])
async def list_video_presets(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(VideoSettingsPreset).order_by(VideoSettingsPreset.name))
    return result.scalars().all()


@router.post("/video-presets", response_model=VideoPresetResponse, status_code=status.HTTP_201_CREATED)
async def create_video_preset(
    body: VideoPresetCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(VideoSettingsPreset).where(VideoSettingsPreset.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preset '{body.name}' already exists",
        )
    preset = VideoSettingsPreset(
        name=body.name,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        **{p: getattr(body, p) for p in _PARAMS},
    )
    if body.loras is not None:
        preset.loras = [slot.model_dump() for slot in body.loras]
    db.add(preset)
    await db.commit()
    await db.refresh(preset)
    return preset


@router.patch("/video-presets/{preset_id}", response_model=VideoPresetResponse)
async def update_video_preset(
    preset_id: UUID,
    body: VideoPresetUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    preset = await db.get(VideoSettingsPreset, preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    if body.name is not None:
        preset.name = body.name
    for p in _PARAMS:
        v = getattr(body, p)
        if v is not None:
            setattr(preset, p, v)
    if body.sampler_name is not None:
        preset.sampler_name = body.sampler_name
    if body.scheduler is not None:
        preset.scheduler = body.scheduler
    if body.loras is not None:
        preset.loras = [slot.model_dump() for slot in body.loras]
    await db.commit()
    await db.refresh(preset)
    return preset


@router.delete("/video-presets/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_preset(
    preset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    preset = await db.get(VideoSettingsPreset, preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    await db.delete(preset)
    await db.commit()
