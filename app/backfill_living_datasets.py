"""Backfill: one living dataset per subject, archive the version sets, link every run (#424).

    python -m app.backfill_living_datasets            # dry run: report only, writes nothing
    python -m app.backfill_living_datasets --apply    # do it

Part of wanly-api#419. Until #420 a set locked once it trained, so every change meant a clone
("Joana v1..v4"). Sets are living now and each run is the record of what it trained on
(training_run_datasets). This folds the history into that shape:

  1. UNOWNED SETS are assigned where the owner is clear (ASSIGN below); the rest are listed.
  2. Each subject's character sets are MERGED into one living set named after the subject:
     the union of their images in order (oldest set first), exact-URI duplicates collapsed.
     Captions: the newest non-empty one wins; every disagreement is reported. Likely
     near-duplicates -- a face crop beside its original, or identical bytes under two keys
     (same S3 ETag) -- are FLAGGED, never removed. The anchor and its scores come from the
     newest set that has an anchor.
  3. The merged version sets are ARCHIVED (hidden, read-only), never deleted.
  4. A subject with a single set keeps it; it is renamed to the subject only when its name is
     exactly "<subject> vN". Every rename is reported.
  5. Every run gets training_run_datasets rows from its own snapshot, pointing at the LIVING
     set (the version set's name is kept as `dataset_name`, as trained). A run with no
     recorded dataset links by image overlap. Pre-#352 runs (no caption snapshot) record the
     caption they trained under. Runs that already have rows are left alone.
  6. Before anything is written, every file any run's snapshot names is checked in S3.

IDEMPOTENT: a second --apply finds every subject already merged and every run linked.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from app import s3
from app.character_registry import identity_phrase
from app.database import async_session
from app.models import Dataset, LtxCharacter, TrainingJob, TrainingRunDataset
from app.run_datasets import job_groups

#: Unowned sets whose owner is clear from the name, as David confirmed (2026-10-08).
ASSIGN = {"Kelly 2000 v4": "Kelly-2000", "Payton v2": "Payton"}

#: A version suffix: "Joana v3", "Kelly 2000 v5".
_VERSION = re.compile(r"\s+v\d+$", re.IGNORECASE)
#: A 32-hex token in a filename -- the source image's id that a face crop keeps in its name
#: ("001_2ab81aae..._f0.jpg" is a crop of ".../2ab81aae....jpg").
_HEX = re.compile(r"[0-9a-f]{32}")


def _basename(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _created(d: Dataset) -> datetime:
    return d.created_at or datetime.min.replace(tzinfo=timezone.utc)


@dataclass
class SubjectPlan:
    subject: str
    target_name: str
    target_existing: Dataset | None
    sources: list[Dataset]
    images: list[str] = field(default_factory=list)
    captions: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float | None] = field(default_factory=dict)
    anchor: str | None = None
    duplicates: int = 0
    caption_conflicts: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)
    near_dups: list[tuple[str, str, str]] = field(default_factory=list)
    #: Plain copies for the report: after a dry run's rollback the ORM rows are expired.
    source_labels: list[tuple[str, int]] = field(default_factory=list)
    #: Images in the living set whose file is gone from S3 -- listed for David, kept in the set.
    missing: list[str] = field(default_factory=list)
    existing_name: str | None = None


@dataclass
class Report:
    missing: list[tuple[str, str]] = field(default_factory=list)
    checked: int = 0
    assigned: list[tuple[str, str]] = field(default_factory=list)
    unowned_left: list[str] = field(default_factory=list)
    merges: list[SubjectPlan] = field(default_factory=list)
    renames: list[tuple[str, str]] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)
    already_linked: int = 0
    repointed: int = 0


def _head_all(uris: list[str]) -> dict[str, dict | None]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(zip(uris, pool.map(s3.head_object, uris)))


def _near_duplicates(images: list[str], heads: dict[str, dict | None]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    by_hex: dict[str, list[str]] = defaultdict(list)
    for u in images:
        for h in set(_HEX.findall(_basename(u))):
            by_hex[h].append(u)
    for h, group in by_hex.items():
        if len(group) > 1:
            for other in group[1:]:
                out.append((group[0], other, "crop/original (same source id in the name)"))
    by_etag: dict[str, list[str]] = defaultdict(list)
    for u in images:
        etag = ((heads.get(u) or {}).get("ETag") or "").strip('"')
        if etag and "-" not in etag:  # multipart ETags are not content hashes
            by_etag[etag].append(u)
    for group in by_etag.values():
        for other in group[1:]:
            out.append((group[0], other, "identical bytes (same ETag)"))
    return out


def _merge(subject: str, sets: list[Dataset], heads: dict[str, dict | None]) -> SubjectPlan:
    existing = next((d for d in sets if d.name == subject), None)
    sources = sorted([d for d in sets if d is not existing], key=_created)
    plan = SubjectPlan(subject=subject, target_name=subject, target_existing=existing,
                       sources=sources,
                       source_labels=[(d.name, len(d.images or [])) for d in sources],
                       existing_name=existing.name if existing else None)
    ordered = ([existing] if existing else []) + sources
    seen: set[str] = set()
    for d in ordered:
        for u in d.images or []:
            if u in seen:
                plan.duplicates += 1
                continue
            seen.add(u)
            plan.images.append(u)
    # Newest non-empty caption wins; disagreements are reported.
    by_uri: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d in sorted(ordered, key=_created):
        for u, c in (d.captions or {}).items():
            if u in seen and (c or "").strip():
                by_uri[u].append((d.name, c.strip()))
    for u, entries in by_uri.items():
        plan.captions[u] = entries[-1][1]
        if len({c for _, c in entries}) > 1:
            plan.caption_conflicts.append((u, entries))
    with_anchor = [d for d in sorted(ordered, key=_created) if d.anchor_uri in seen]
    if with_anchor:
        newest = with_anchor[-1]
        plan.anchor = newest.anchor_uri
        plan.scores = {u: v for u, v in (newest.scores or {}).items() if u in seen}
    plan.near_dups = _near_duplicates(plan.images, heads)
    plan.missing = [u for u in plan.images if heads.get(u) is None]
    return plan


def _legacy_captions(job: TrainingJob, g: dict, chars: dict[str, LtxCharacter]) -> list[str]:
    """What a pre-#352 run trained every image under: its recorded single caption, else the
    bare "<trigger>, <gender>" phrase (the standard since console#487), else the trigger."""
    config = job.config or {}
    one = config.get("caption") if g["index"] == 0 else None
    if not one:
        c = chars.get(g["character"] or job.character)
        trigger = (job.trigger if g["index"] == 0 else None) or (c.trigger if c else None)
        gender = config.get("gender") if g["index"] == 0 else None
        gender = gender or (c.gender if c else None)
        one = (identity_phrase(trigger, gender) or trigger) if trigger else None
    return [one] * len(g["images"]) if one else []


async def run(apply: bool, db=None) -> Report:
    """Plan (and with `apply`, write) the backfill. `db` is for tests; the CLI opens its own."""
    if db is None:
        async with async_session() as own:
            return await _run(apply, own)
    return await _run(apply, db)


async def _run(apply: bool, db) -> Report:
    # Everything happens inside a SAVEPOINT: a dry run rolls back exactly that, never more --
    # under a test's outer transaction as much as in the CLI's own session.
    nested = await db.begin_nested()
    rep = Report()
    datasets = list((await db.execute(select(Dataset))).scalars().all())
    jobs = list((await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.asc()))).scalars().all())
    chars = {c.name: c for c in (await db.execute(select(LtxCharacter))).scalars().all()}
    linked_jobs = {jid for (jid,) in (await db.execute(
        select(TrainingRunDataset.training_job_id).distinct())).all()}

    # ---- 6. every file any run names, checked first; dataset images too, for ETags
    run_uris = sorted({u for j in jobs for g in job_groups(j) for u in g["images"]})
    ds_uris = sorted({u for d in datasets for u in (d.images or [])})
    heads = await asyncio.to_thread(_head_all, sorted(set(run_uris) | set(ds_uris)))
    rep.checked = len(run_uris)
    for j in jobs:
        for g in job_groups(j):
            for u in g["images"]:
                if heads.get(u) is None:
                    rep.missing.append((f"{j.character} v{j.version} "
                                        f"{(j.config or {}).get('arch') or 'ltx'}", u))

    # ---- 1. unowned sets
    for d in datasets:
        if d.kind is None and d.archived_at is None:
            owner = ASSIGN.get(d.name)
            if owner:
                rep.assigned.append((d.name, owner))
                d.kind, d.character = "character", owner
            else:
                rep.unowned_left.append(f"{d.name} ({len(d.images or [])} images)")

    # ---- 2-4. one living set per subject
    live = [d for d in datasets if d.archived_at is None]
    by_subject: dict[str, list[Dataset]] = defaultdict(list)
    for d in live:
        if d.kind == "character" and d.character:
            by_subject[d.character].append(d)
    living_id: dict[uuid.UUID, uuid.UUID] = {}  # any set id -> its subject's living set id
    # The report names each link by the set it lands in -- new living sets included, which
    # are not in `datasets`.
    living_names: dict[uuid.UUID, str] = {}
    for subject, sets in sorted(by_subject.items()):
        if len(sets) == 1:
            d = sets[0]
            if d.name != subject and _VERSION.sub("", d.name) == subject \
                    and not any(o.name == subject for o in datasets):
                rep.renames.append((d.name, subject))
                d.name = subject
            else:
                rep.kept.append(d.name)
            living_id[d.id] = d.id
            continue
        plan = _merge(subject, sets, heads)
        if any(o.name == subject and o not in sets for o in datasets):
            plan.target_name = f"{subject} (living)"
        rep.merges.append(plan)
        target = plan.target_existing
        if target is None:
            new_id = uuid.uuid4()
            target = Dataset(id=new_id, name=plan.target_name, kind="character",
                             character=subject, prefix=f"datasets/{new_id}",
                             images=[], captions={}, scores={},
                             user_id=plan.sources[-1].user_id,
                             notes=("Living set (#424), merged from "
                                    + ", ".join(s.name for s in plan.sources) + "."))
            db.add(target)
            living_names[new_id] = plan.target_name
        target.images = plan.images
        target.captions = plan.captions
        target.scores = plan.scores
        target.anchor_uri = plan.anchor
        now = datetime.now(timezone.utc)
        for src in plan.sources:
            src.archived_at = now
            living_id[src.id] = target.id
        living_id[target.id] = target.id
    for d in live:
        if d.kind in ("composition", "regularization"):
            living_id.setdefault(d.id, d.id)
    # Sets already archived by an earlier --apply point at their subject's living set.
    for d in datasets:
        if d.archived_at is not None and d.kind == "character" and d.character:
            tgt = next((x for x in live if x.kind == "character" and
                        x.character == d.character and x.archived_at is None), None)
            if tgt is not None:
                living_id.setdefault(d.id, tgt.id)

    # ---- 5a. link rows written since #422 shipped (run created before this --apply) that
    # point at a set just archived follow it to the living set; their name-as-trained stays.
    moved = {src: tgt for src, tgt in living_id.items() if src != tgt}
    if moved:
        for row in (await db.execute(select(TrainingRunDataset).where(
                TrainingRunDataset.dataset_id.in_(list(moved))))).scalars().all():
            row.dataset_id = moved[row.dataset_id]
            rep.repointed += 1

    # ---- 5. link every run
    names = {d.id: d.name for d in datasets}

    for j in jobs:
        label = f"{j.character} v{j.version} {(j.config or {}).get('arch') or 'ltx'} [{j.status}]"
        if j.id in linked_jobs:
            rep.already_linked += 1
            continue
        parts = []
        for g in job_groups(j):
            ds_id = g["dataset_id"] if g["dataset_id"] in names else None
            how = "recorded"
            if ds_id is None and g["images"]:
                imgs = set(g["images"])
                best = max(datasets, key=lambda d: len(imgs & set(d.images or [])),
                           default=None)
                overlap = len(imgs & set(best.images or [])) if best else 0
                if best is not None and overlap * 2 >= len(imgs):
                    ds_id, how = best.id, f"by overlap {overlap}/{len(imgs)}"
            target = living_id.get(ds_id, ds_id) if ds_id else None
            captions = g["captions"]
            if captions is None:
                captions = _legacy_captions(j, g, chars) or None
                how += ", caption = bare trigger phrase" if captions else ", no caption"
            as_trained = g["dataset_name"] or (names.get(ds_id) if ds_id else None)
            if target is None:
                rep.unlinked.append(f"{label} group {g['index']} ({g['kind']}, "
                                    f"{len(g['images'])} images, recorded as "
                                    f"{g['dataset_name']!r}): no dataset found")
            db.add(TrainingRunDataset(
                training_job_id=j.id, group_index=g["index"], dataset_id=target,
                dataset_name=as_trained, kind=g["kind"], character=g["character"],
                images=g["images"], captions=captions, num_repeats=g["num_repeats"],
                windows=g["windows"], source="backfill"))
            tname = (living_names.get(target) or names.get(target)) if target else None
            parts.append(f"g{g['index']} {g['kind']} {len(g['images'])} img "
                         f"{as_trained!r} -> {tname!r} ({how})")
        rep.links.append(f"{label}: " + "; ".join(parts))

    if apply:
        await db.flush()
        await nested.commit()
        await db.commit()
    else:
        await nested.rollback()
    return rep


def render(rep: Report, apply: bool) -> str:
    out = [f"== living-datasets backfill ({'APPLIED' if apply else 'DRY RUN: nothing written'})"]
    out.append(f"\n-- S3: {rep.checked} files named by run snapshots checked; "
               f"{len(rep.missing)} missing")
    for who, u in rep.missing:
        out.append(f"   MISSING {u}  (used by {who})")
    out.append("\n-- unowned sets")
    for name, owner in rep.assigned:
        out.append(f"   assign {name!r} -> {owner}")
    for name in rep.unowned_left:
        out.append(f"   left alone for David: {name}")
    out.append("\n-- merges (one living set per subject)")
    for p in rep.merges:
        into = (f"existing {p.existing_name!r}" if p.existing_name
                else f"NEW {p.target_name!r}")
        out.append(f"   {p.subject}: {', '.join(f'{n} ({c})' for n, c in p.source_labels)}"
                   f" -> {into}: {len(p.images)} images ({p.duplicates} exact duplicates "
                   f"collapsed), {len(p.captions)} captioned, anchor "
                   f"{'kept' if p.anchor else 'none'}; archive {len(p.source_labels)}")
        for u in p.missing:
            out.append(f"      file missing from S3 (kept in the set): {u}")
        for u, entries in p.caption_conflicts:
            out.append(f"      caption conflict {_basename(u)}: "
                       + " | ".join(f"{n}: {c[:50]!r}" for n, c in entries))
        for a, b, why in p.near_dups:
            out.append(f"      near-duplicate: {_basename(a)} ~ {_basename(b)} ({why})")
    out.append("\n-- single-set subjects")
    for a, b in rep.renames:
        out.append(f"   rename {a!r} -> {b!r}")
    for name in rep.kept:
        out.append(f"   keep {name!r}")
    out.append(f"\n-- run links ({len(rep.links)} to write, {rep.already_linked} already linked, "
               f"{rep.repointed} existing row(s) re-pointed to a living set)")
    for line in rep.links:
        out.append(f"   {line}")
    out.append(f"\n-- groups with no dataset: {len(rep.unlinked)}")
    for line in rep.unlinked:
        out.append(f"   {line}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()
    rep = asyncio.run(run(args.apply))
    print(render(rep, args.apply))


if __name__ == "__main__":
    main()
