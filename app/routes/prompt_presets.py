from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import PromptPreset, User
from app.schemas.prompt_presets import PromptPresetCreate, PromptPresetResponse, PromptPresetUpdate

router = APIRouter()


@router.get("/prompt-presets", response_model=list[PromptPresetResponse])
async def list_prompt_presets(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(PromptPreset).order_by(PromptPreset.name))
    return result.scalars().all()


@router.post("/prompt-presets", response_model=PromptPresetResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt_preset(
    body: PromptPresetCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(PromptPreset).where(PromptPreset.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preset '{body.name}' already exists",
        )
    preset = PromptPreset(
        name=body.name,
        prompt=body.prompt,
        loras=[l.model_dump() for l in body.loras] if body.loras else None,
    )
    db.add(preset)
    await db.commit()
    await db.refresh(preset)
    return preset


@router.get("/prompt-presets/{preset_id}", response_model=PromptPresetResponse)
async def get_prompt_preset(
    preset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    preset = await db.get(PromptPreset, preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    return preset


@router.patch("/prompt-presets/{preset_id}", response_model=PromptPresetResponse)
async def update_prompt_preset(
    preset_id: UUID,
    body: PromptPresetUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    preset = await db.get(PromptPreset, preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    if body.name is not None:
        preset.name = body.name
    if body.prompt is not None:
        preset.prompt = body.prompt
    if body.loras is not None:
        preset.loras = [l.model_dump() for l in body.loras]
    await db.commit()
    await db.refresh(preset)
    return preset


@router.delete("/prompt-presets/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt_preset(
    preset_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    preset = await db.get(PromptPreset, preset_id)
    if preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    await db.delete(preset)
    await db.commit()
