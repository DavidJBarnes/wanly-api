"""The LTX recipe book: import the sheet, serve the book.

wanly-api owns the book (migration 070). The console reads recipes from here, and a claim
carries the RESOLVED recipe rather than a name for the engine to look up — an engine that
cannot look a recipe up cannot look up a stale one.
"""

import hashlib
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.ltx_sheet import SheetError, book_sha256, parse_sheet
from app.models import LtxRecipeBook, User

logger = logging.getLogger(__name__)
router = APIRouter()

# The sheet is a few hundred KB of XML in a zip. Anything far larger is not a recipe sheet,
# and refusing early beats unzipping it to find out.
MAX_SHEET_BYTES = 8 * 1024 * 1024


@router.get("/recipes")
async def get_recipes(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The recipe book, in the shape the console already reads.

    404 rather than an empty book when nothing has been imported: an empty book and a missing
    one look identical downstream, and "the dropdowns are empty" should not be the first sign
    that no sheet was ever uploaded.
    """
    row = (await db.execute(select(LtxRecipeBook).where(LtxRecipeBook.id == 1))).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recipe book has been imported. POST the sheet to /recipes/import.",
        )
    book = dict(row.book)
    # Provenance travels with the book so a render can be tied back to the exact sheet that
    # produced it, without a second call.
    book["book_sha256"] = row.book_sha256
    book["imported_at"] = row.imported_at.isoformat() if row.imported_at else None
    return book


@router.post("/recipes/import")
async def import_recipes(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload the hand-authored .ods and replace the book.

    Parsing happens HERE rather than in a script that writes a file, so there is exactly one
    parser and exactly one copy. Two copies of recipes.json once shipped a 16-render batch
    against stale prompts.

    The sheet itself is never written back. It is authored by hand, and regenerating it from
    code once silently overwrote a hand-edited prompt.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_SHEET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"sheet is {len(raw)} bytes, over the {MAX_SHEET_BYTES} limit",
        )
    try:
        book = parse_sheet(raw)
    except SheetError as e:
        # The sheet is hand-authored, so a parse failure is usually a human edit rather than a
        # corrupt file. Say what was wrong, not just that something was.
        raise HTTPException(status_code=422, detail=str(e)) from e

    new_hash = book_sha256(book)
    row = (await db.execute(select(LtxRecipeBook).where(LtxRecipeBook.id == 1))).scalar_one_or_none()
    previous = row.book_sha256 if row else None

    if row is None:
        row = LtxRecipeBook(id=1)
        db.add(row)
    row.book = book
    row.book_sha256 = new_hash
    row.source_sha256 = hashlib.sha256(raw).hexdigest()
    row.source_filename = file.filename
    await db.commit()
    await db.refresh(row)

    characters = {c: len(v.get("recipes", {})) for c, v in book.get("characters", {}).items()}
    logger.info(
        "recipe book imported from %s: %s, book_sha256 %s (was %s)",
        file.filename, characters, new_hash[:16], (previous or "none")[:16],
    )
    # `changed` is the useful bit: re-saving the sheet with no edits changes the file bytes
    # but not the book, and the caller should be able to tell those apart.
    return {
        "characters": characters,
        "definitions": len(book.get("definitions", {})),
        "book_sha256": new_hash,
        "previous_book_sha256": previous,
        "changed": previous != new_hash,
        "imported_at": row.imported_at.isoformat() if row.imported_at else None,
    }
