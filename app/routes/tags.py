from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import ACTION_GROUP, TitleTag, User
from app.schemas.tags import TitleTagCreate, TitleTagResponse, TitleTagUpdate

router = APIRouter()

def _description(group: int, text: str | None) -> str | None:
    """The description to store: None when blank; a 400 when a Person tag is given one."""
    text = (text or "").strip() or None
    if text is not None and group != ACTION_GROUP:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Only Action tags have a description")
    return text


@router.get("/tags", response_model=list[TitleTagResponse])
async def list_tags(
    group: Optional[int] = Query(None),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(TitleTag)
    if group is not None:
        stmt = stmt.where(TitleTag.group == group)
    stmt = stmt.order_by(TitleTag.name)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/tags", response_model=TitleTagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    body: TitleTagCreate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tag name cannot be empty")

    # Case-insensitive duplicate check
    existing = await db.execute(
        select(TitleTag).where(
            func.lower(TitleTag.name) == name.lower(),
            TitleTag.group == body.group,
        )
    )
    if existing.scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tag already exists in this group")

    tag = TitleTag(name=name, group=body.group, description=_description(body.group, body.description))
    db.add(tag)
    await db.commit()
    await db.refresh(tag)
    return tag


@router.patch("/tags/{tag_id}", response_model=TitleTagResponse)
async def update_tag(
    tag_id: UUID,
    body: TitleTagUpdate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tag = await db.get(TitleTag, tag_id)
    if tag is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found")
    tag.description = _description(tag.group, body.description)
    await db.commit()
    await db.refresh(tag)
    return tag


@router.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tag(
    tag_id: UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tag = await db.get(TitleTag, tag_id)
    if tag is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found")
    await db.delete(tag)
    await db.commit()
