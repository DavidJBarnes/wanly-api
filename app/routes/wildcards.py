from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User, Wildcard
from app.schemas.wildcards import WildcardCreate, WildcardResponse, WildcardUpdate

router = APIRouter()

# LTX poses carry <TRIGGER>, which the character's trigger word fills. It shares syntax with
# wildcards, so a wildcard by this name would make the resolver substitute a RANDOM option --
# the render would name the wrong character, or none, and nothing would report a problem.
#
# The trigger is already substituted before wildcard resolution runs, so this is the second of
# two guards rather than the only one. Reserving the name means the collision cannot be created
# by hand in the first place, which is cheaper than reasoning about ordering later.
RESERVED_WILDCARD_NAMES = {"TRIGGER"}


@router.get("/wildcards", response_model=list[WildcardResponse])
async def list_wildcards(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Wildcard).order_by(Wildcard.name))
    return result.scalars().all()


@router.post("/wildcards", response_model=WildcardResponse, status_code=status.HTTP_201_CREATED)
async def create_wildcard(
    body: WildcardCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.name.strip().upper() in RESERVED_WILDCARD_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{body.name}' is reserved: LTX poses use <TRIGGER> for the character "
                   "trigger word, and a wildcard by that name would randomly replace it.",
        )
    existing = await db.execute(select(Wildcard).where(Wildcard.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Wildcard '{body.name}' already exists",
        )
    wildcard = Wildcard(name=body.name, options=body.options)
    db.add(wildcard)
    await db.commit()
    await db.refresh(wildcard)
    return wildcard


@router.get("/wildcards/{wildcard_id}", response_model=WildcardResponse)
async def get_wildcard(
    wildcard_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wildcard = await db.get(Wildcard, wildcard_id)
    if wildcard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wildcard not found")
    return wildcard


@router.patch("/wildcards/{wildcard_id}", response_model=WildcardResponse)
async def update_wildcard(
    wildcard_id: UUID,
    body: WildcardUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wildcard = await db.get(Wildcard, wildcard_id)
    if wildcard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wildcard not found")
    if body.name is not None:
        # Renaming into the reserved name is the same collision by another route.
        if body.name.strip().upper() in RESERVED_WILDCARD_NAMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{body.name}' is reserved: LTX poses use <TRIGGER> for the character "
                       "trigger word, and a wildcard by that name would randomly replace it.",
            )
        wildcard.name = body.name
    if body.options is not None:
        wildcard.options = body.options
    await db.commit()
    await db.refresh(wildcard)
    return wildcard


@router.delete("/wildcards/{wildcard_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_wildcard(
    wildcard_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wildcard = await db.get(Wildcard, wildcard_id)
    if wildcard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wildcard not found")
    await db.delete(wildcard)
    await db.commit()
