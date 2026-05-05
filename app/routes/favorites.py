from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Favorite, User

router = APIRouter(tags=["favorites"])

VALID_TYPES = {"video", "image"}


@router.post("/favorites/toggle", dependencies=[Depends(get_current_user)])
async def toggle_favorite(
    body: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Toggle a favorite on or off. Returns { favorited: bool, item_ref: str }."""
    item_type = body.get("item_type", "").strip()
    item_ref = body.get("item_ref", "").strip()

    if item_type not in VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"item_type must be one of: {', '.join(sorted(VALID_TYPES))}")
    if not item_ref:
        raise HTTPException(status_code=400, detail="item_ref is required")

    # Check if already favorited
    existing = await db.execute(
        select(Favorite).where(
            Favorite.user_id == user.id,
            Favorite.item_type == item_type,
            Favorite.item_ref == item_ref,
        )
    )
    fav = existing.scalar_one_or_none()

    if fav:
        await db.delete(fav)
        await db.commit()
        return {"favorited": False, "item_ref": item_ref}
    else:
        new_fav = Favorite(user_id=user.id, item_type=item_type, item_ref=item_ref)
        db.add(new_fav)
        await db.commit()
        return {"favorited": True, "item_ref": item_ref}


@router.get("/favorites", dependencies=[Depends(get_current_user)])
async def list_favorites(
    item_type: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all favorited item_refs for the current user, optionally filtered by type."""
    query = select(Favorite.item_ref).where(Favorite.user_id == user.id)
    if item_type:
        if item_type not in VALID_TYPES:
            raise HTTPException(status_code=400, detail=f"item_type must be one of: {', '.join(sorted(VALID_TYPES))}")
        query = query.where(Favorite.item_type == item_type)
    query = query.order_by(Favorite.created_at.desc())

    result = await db.execute(query)
    refs = [row[0] for row in result.all()]
    return {"item_refs": refs}
