"""wanly-api#452 data: DavidPayton becomes a pair; a subject with only archived sets gets its
living set back. Dry run writes nothing; apply is idempotent."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app import characters_backfill as cb
from app.models import Dataset, LtxCharacter


@pytest.fixture(autouse=True)
def _no_s3(monkeypatch):
    monkeypatch.setattr(cb, "_head_all", lambda uris: {u: {"Size": 1} for u in uris})


@pytest.mark.asyncio
async def test_pair_and_rebuild(db, monkeypatch):
    sub = f"Sub{uuid.uuid4().hex[:5]}"
    me, pay = f"Me{uuid.uuid4().hex[:4]}", f"Pay{uuid.uuid4().hex[:4]}"
    pair = f"Pair{uuid.uuid4().hex[:4]}"
    monkeypatch.setattr(cb, "PAIRS", {pair: [me, pay]})
    db.add_all([LtxCharacter(name=me, trigger="d@vid", gender="man"),
                LtxCharacter(name=pay, trigger="p@yton", gender="woman"),
                LtxCharacter(name=sub, trigger="s", gender="woman")])
    now = datetime.now(timezone.utc)
    db.add_all([
        Dataset(id=uuid.uuid4(), name=f"{sub} v1", kind="character", character=sub,
                images=["s3://b/1.png", "s3://b/2.png"], prefix="a", captions={"s3://b/1.png": "old"},
                scores={}, faces={}, archived_at=now),
        Dataset(id=uuid.uuid4(), name=f"{sub} v2", kind="character", character=sub,
                images=["s3://b/2.png", "s3://b/3.png"], prefix="b", captions={"s3://b/1.png": "new"},
                scores={}, faces={}, archived_at=now, anchor_uri="s3://b/3.png"),
    ])
    await db.flush()
    await cb.run(False, db)
    assert (await db.execute(select(LtxCharacter).where(LtxCharacter.name == pair))).first() is None
    lines = await cb.run(True, db)
    assert any(sub in line for line in lines)
    p = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == pair))).scalar_one()
    assert p.kind == "pair" and p.char_lora is None
    assert p.trigger == "d@vid, man and p@yton, woman"
    living = (await db.execute(select(Dataset).where(
        Dataset.character == sub, Dataset.archived_at.is_(None)))).scalar_one()
    assert living.images == ["s3://b/1.png", "s3://b/2.png", "s3://b/3.png"] or \
        set(living.images) == {"s3://b/1.png", "s3://b/2.png", "s3://b/3.png"}
    assert living.anchor_uri == "s3://b/3.png"
    again = await cb.run(True, db)
    assert not any("created" in line or "living set" in line for line in again
                   if pair in line or sub in line)
