"""Backfill dataset lineage (wanly-api#445): which image in each set was made from which.

    python -m app.backfill_lineage            # dry run: what would be recorded, per set
    python -m app.backfill_lineage --apply    # record it

ADDS METADATA ONLY. Nothing is removed from any set and no file is touched; an entry already
in `datasets.derived` is never overwritten (what a path recorded beats what a name suggests).
IDEMPOTENT: a second --apply records nothing new. Only within one set: lineage is "this image
in THIS set came from that one", and a source outside the set cannot be named from here.

Evidence, strongest first:
  1. "Fix small faces" bookkeeping in `faces`: `crop_uri` on a photograph (-> fix_crop) and
     `upscaled_from` on an upscale (-> upscale).
  2. The names every path writes, each of which carries its source's stem:
       crop faces   <batch>/faces-x|portraits-x/NNN_<stem>_fN.ext      -> crop
       fix crop     <prefix>/portraits-x|pairs-x/NNN_<stem>.ext        -> fix_crop
       fix upscale  <prefix>/upscaled-x/NNN_<stem>.jpg                 -> upscale
       image edit   .../<stem>_edit-<tag>_<hex6>.png                   -> edit
     The source is the set's ONE image with that stem; none or several -> not recorded.
  3. Identical bytes (same single-part ETag) under different URIs -> duplicate of the first.
  4. Shared 32-hex id (the living-datasets near-duplicate heuristic): within a group, the one
     member that is not itself recognisably derived is the original; the rest that nothing
     above explained -> duplicate when the bytes match it, else crop.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from app import clips, lineage, s3
from app.database import async_session
from app.models import Dataset

_HEX = re.compile(r"[0-9a-f]{32}")
_CROP = re.compile(r"/(?:faces|portraits)-[^/]+/\d{3}_(?P<stem>.+)_f\d+\.[A-Za-z0-9]+$")
_FIX_CROP = re.compile(r"/(?:portraits|pairs)-[^/]+/\d{3}_(?P<stem>.+)\.[A-Za-z0-9]+$")
_UPSCALE = re.compile(r"/upscaled-[^/]+/\d{3}_(?P<stem>.+)\.[A-Za-z0-9]+$")
_EDIT = re.compile(r"/(?P<stem>[^/]+)_edit-[^/]+_[0-9a-f]{6}\.[A-Za-z0-9]+$")


def _stem(uri: str) -> str:
    name = uri.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def by_name(uri: str) -> tuple[str, str] | None:
    """(source stem, how) when `uri`'s name says what it was made from, else None. A crop
    batch's name also fits the fix-crop pattern; it is tried first, as it is the narrower."""
    for pat, how in ((_CROP, "crop"), (_UPSCALE, "upscale"), (_EDIT, "edit"),
                     (_FIX_CROP, "fix_crop")):
        m = pat.search(uri)
        if m:
            return m.group("stem"), how
    return None


def plan(images: list[str], faces: dict, existing: dict,
         etags: dict[str, str]) -> dict[str, tuple[str, str]]:
    """{derived_uri: (source_uri, how)} to ADD for one set. Pure, for tests."""
    present = list(dict.fromkeys(images))
    inset = set(present)
    out: dict[str, tuple[str, str]] = {}

    def add(derived: str, source: str, how: str) -> None:
        if derived in inset and derived != source and derived not in existing \
                and derived not in out:
            out[derived] = (source, how)

    # 1. Fix small faces' own bookkeeping.
    for orig, e in (faces or {}).items():
        if e and e.get("crop_uri"):
            add(e["crop_uri"], orig, "fix_crop")
        if e and e.get("upscaled_from"):
            add(orig, e["upscaled_from"], "upscale")
    # 2. Names that carry their source's stem.
    by_stem: dict[str, list[str]] = defaultdict(list)
    for u in present:
        by_stem[_stem(u)].append(u)
    for u in present:
        hit = by_name(u)
        if not hit:
            continue
        stem, how = hit
        sources = [s for s in by_stem.get(stem, []) if s != u]
        if len(sources) == 1:
            add(u, sources[0], how)
    # 3. Identical bytes.
    by_etag: dict[str, list[str]] = defaultdict(list)
    for u in present:
        tag = etags.get(u, "")
        if tag and "-" not in tag:
            by_etag[tag].append(u)
    for group in by_etag.values():
        for other in group[1:]:
            add(other, group[0], "duplicate")
    # 4. Shared hex id: one plainly-original member, the rest derived from it.
    by_hex: dict[str, list[str]] = defaultdict(list)
    for u in present:
        for h in set(_HEX.findall(u.rsplit("/", 1)[-1])):
            by_hex[h].append(u)
    for group in by_hex.values():
        if len(group) < 2:
            continue
        originals = [u for u in group if by_name(u) is None and u not in existing
                     and u not in out]
        if len(originals) != 1:
            continue
        orig = originals[0]
        for u in group:
            if u == orig:
                continue
            same = etags.get(u) and etags.get(u) == etags.get(orig)
            add(u, orig, "duplicate" if same else (by_name(u) or ("", "crop"))[1])
    return out


def _etags(uris: list[str]) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        heads = dict(zip(uris, pool.map(s3.head_object, uris)))
    return {u: ((h or {}).get("ETag") or "").strip('"') for u, h in heads.items()}


async def run(apply: bool) -> list[tuple[str, Counter, int]]:
    report: list[tuple[str, Counter, int]] = []
    async with async_session() as db:
        sets = (await db.execute(select(Dataset).order_by(Dataset.name))).scalars().all()
        for ds in sets:
            stills = [u for u in (ds.images or []) if not clips.is_clip(u)]
            if len(stills) < 2:
                continue
            etags = await asyncio.to_thread(_etags, stills)
            add = plan(stills, ds.faces or {}, ds.derived or {}, etags)
            if not add:
                continue
            report.append((ds.name, Counter(how for _, how in add.values()), len(stills)))
            if apply:
                for derived, (source, how) in add.items():
                    lineage.record(ds, derived, source, how)
        if apply:
            await db.commit()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="record it (default: dry run)")
    args = ap.parse_args()
    rep = asyncio.run(run(args.apply))
    total = Counter()
    for name, counts, n in rep:
        total.update(counts)
        print(f"  {name:28} {sum(counts.values()):4} of {n:4} stills: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(("APPLIED" if args.apply else "DRY RUN") + f": {sum(total.values())} lineage entries "
          f"across {len(rep)} sets ({', '.join(f'{k} {v}' for k, v in sorted(total.items()))})")


if __name__ == "__main__":
    main()
