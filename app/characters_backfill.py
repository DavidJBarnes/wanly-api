"""Characters own their dataset: the data half of wanly-api#452.

    python -m app.characters_backfill            # dry run (default): says what it would do
    python -m app.characters_backfill --apply    # does it

Idempotent; only CREATES and LINKS -- nothing is deleted.

1. DavidPayton is its own PAIR character (David, 2026-10-09): members Me + Payton, the
   members' trained phrases joined ("d@vid, man and p@yton, woman"), no LoRA yet (a draft),
   with an empty composition set of its own. (The 2-image "DavidPayton v1" set it would have
   owned was deleted before this ran.)
2. A character whose living set is gone but whose archived version sets remain gets its
   living set rebuilt from them, the way the #424 backfill merged subjects (union, dedupe,
   newest non-empty caption, newest anchor) -- Payton, whose living set was deleted. Link rows
   pointing at those archived sets follow to the new living set. Run app.backfill_lineage
   afterwards for its lineage.
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select

from app.backfill_living_datasets import _head_all, _merge
from app.character_registry import pair_phrase
from app.database import async_session
from app.models import Dataset, LtxCharacter, TrainingRunDataset

PAIRS = {"DavidPayton": ["Me", "Payton"]}


async def run(apply: bool, db=None) -> list[str]:
    if db is None:
        async with async_session() as own:
            return await _run(apply, own)
    return await _run(apply, db)


async def _run(apply: bool, db) -> list[str]:
    nested = await db.begin_nested()
    out: list[str] = []
    chars = {c.name: c for c in (await db.execute(select(LtxCharacter))).scalars().all()}
    sets = list((await db.execute(select(Dataset))).scalars().all())

    # ---- 1. pair characters
    for name, members in PAIRS.items():
        rows = [chars.get(m) for m in members]
        if any(r is None for r in rows):
            out.append(f"pair {name}: a member is not registered ({members}); skipped")
            continue
        phrase = pair_phrase(rows)
        c = chars.get(name)
        if c is None:
            c = LtxCharacter(name=name, kind="pair", members=members, trigger=phrase,
                             gender=None, char_lora=None)
            db.add(c)
            out.append(f"pair {name}: created (members {members}, trigger {phrase!r}, "
                       f"no LoRA yet: a draft)")
        else:
            out.append(f"pair {name}: already registered ({c.kind}, {c.members})")
        if not any(d.kind == "composition" and d.character == name and d.archived_at is None
                   for d in sets):
            new_id = uuid.uuid4()
            db.add(Dataset(id=new_id, name=name, kind="composition", character=name,
                           prefix=f"datasets/{new_id}", images=[], captions={}, scores={},
                           faces={}, notes="The together set of the pair (wanly-api#452)."))
            out.append(f"pair {name}: empty composition set {name!r} created")

    # ---- 2. living sets rebuilt from archived version sets
    by_subject: dict[str, list[Dataset]] = defaultdict(list)
    for d in sets:
        if d.kind == "character" and d.character:
            by_subject[d.character].append(d)
    for subject, group in sorted(by_subject.items()):
        if any(d.archived_at is None for d in group):
            continue
        heads = await asyncio.to_thread(
            _head_all, sorted({u for d in group for u in (d.images or [])}))
        plan = _merge(subject, group, heads)
        if any(d.name == subject for d in sets):
            plan.target_name = f"{subject} (living)"
        new_id = uuid.uuid4()
        living = Dataset(id=new_id, name=plan.target_name, kind="character", character=subject,
                         prefix=f"datasets/{new_id}", images=plan.images,
                         captions=plan.captions, scores=plan.scores, faces={},
                         anchor_uri=plan.anchor,
                         notes=("Living set rebuilt from " +
                                ", ".join(f"{n} ({k})" for n, k in plan.source_labels) +
                                " (wanly-api#452)."))
        db.add(living)
        await db.flush()
        ids = [d.id for d in group]
        moved = 0
        for row in (await db.execute(select(TrainingRunDataset).where(
                TrainingRunDataset.dataset_id.in_(ids)))).scalars().all():
            row.dataset_id = new_id
            moved += 1
        out.append(f"living set {plan.target_name!r} for {subject}: {len(plan.images)} images "
                   f"from {plan.source_labels}, {plan.duplicates} duplicates collapsed, "
                   f"{len(plan.caption_conflicts)} caption conflicts, anchor "
                   f"{'kept' if plan.anchor else 'none'}, {len(plan.missing)} files missing "
                   f"from S3, {moved} run link(s) moved to it")

    if apply:
        await db.flush()
        await nested.commit()
        await db.commit()
    else:
        await nested.rollback()
    return out or ["nothing to do"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()
    lines = asyncio.run(run(args.apply))
    print(("APPLIED" if args.apply else "DRY RUN") + f" ({datetime.now(timezone.utc):%F %T}Z)")
    for line in lines:
        print(" ", line)


if __name__ == "__main__":
    main()
