"""A character with everything that belongs to it (wanly-api#452): Characters and Datasets
collapsed -- a character HAS its dataset, its runs and its archived version sets."""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel

from app.schemas.datasets import DatasetResponse, DatasetRun
from app.schemas.ltx import LtxCharacterResponse


class CharacterRef(BaseModel):
    """A member of a pair, or a pair this character is in: enough to link and badge it."""
    id: uuid.UUID
    name: str
    kind: str = "solo"
    hidden: bool = False


class CharacterFull(BaseModel):
    character: LtxCharacterResponse
    #: The living set: the character set of a solo, the composition set of a pair. None until
    #: one is created (a character registered before its images).
    dataset: Optional[DatasetResponse] = None
    #: Archived version sets (Joana v1..v4), newest first -- History, read-only.
    archived: list[DatasetResponse] = []
    #: Every run this character's page lists, newest first, with its role (DatasetRun).
    runs: list[DatasetRun] = []
    #: A pair's members, in order; a solo's pairs (DavidJoana on Joana's page).
    members: list[CharacterRef] = []
    pairs: list[CharacterRef] = []
