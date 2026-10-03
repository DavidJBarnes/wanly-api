"""Where a character LoRA's trigger and gender come from (wanly-console#596).

The character editor used to make you type the trigger and gender a LoRA was trained on, and
a typo there is silent: the render fills <TRIGGER> with words the weights never learned and
quietly stops being that person. Several characters ended up with the wrong pair that way.
This answers "what did THIS file train on", from two sources, in order:

1. THE WANLY TRAINING RUN that wrote the file. `training_jobs` snapshots the trigger, the
   gender and the per-image captions ("<trigger>, <gender>, ..."), and the file's name is one
   `_artifact_key` produced, so the run is found by name. A pair's trigger is the joined
   phrase its composition captions carried, gender None -- exactly what publish writes.

2. THE FILE'S OWN safetensors METADATA, for LoRAs trained outside wanly. kohya writes
   `ss_tag_frequency` (caption tags per dataset dir) and `ss_dataset_dirs` (the DreamBooth
   "<repeats>_<trigger> <class>" folder names), and `ss_datasets` subsets may carry
   `class_tokens`. Read with two small ranged GETs -- eight bytes of header length, then the
   header -- never the 650 MB file.

   CHECKED AGAINST REAL FILES (2026-10-03): musubi-tuner's LTX-2 trainer, which wrote every
   character LoRA in the bucket (CLI `k3lly2026_v2` and wanly's own runs alike), writes NONE
   of those keys. Its `ss_datasets` holds only `image_directory: "data"`, and there is no
   `ss_tag_frequency` or `ss_dataset_dirs`. For those files the metadata source finds
   nothing, and the answer is honestly "none" rather than a guess from the file name. What
   the header does give is `ss_output_name`, which links a RENAMED wanly file back to its run.

Results from metadata are cached per (S3 key, etag): the etag changes when the file does, so
a retrained LoRA republished under the same name is read again. Run lookups are not cached --
they are one indexed query and a run can still be finishing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import TrainingJob

logger = logging.getLogger(__name__)

#: The gender words a character row may store (schemas.ltx.Gender).
GENDERS = ("woman", "man", "person")
#: A header bigger than this is not a LoRA header; refuse rather than read it.
MAX_HEADER_BYTES = 16 * 1024 * 1024


@dataclass
class Provenance:
    name: str
    trigger: str | None = None
    gender: str | None = None
    #: "training_run" | "lora_metadata" | "none"
    source: str = "none"
    run_id: str | None = None
    run_character: str | None = None
    run_version: int | None = None
    #: ISO date the run finished (or was created), for the console's hint.
    run_date: str | None = None
    #: Which metadata key answered, or why nothing did.
    detail: str | None = None
    #: A pair run: the trigger is the joined phrase and there is no single gender.
    pair: bool = False

    def public(self) -> dict:
        return asdict(self)


def stem(name: str) -> str:
    """`x.safetensors`, `character/x.safetensors` and `x` are all the LoRA `x`."""
    return name.strip().rsplit("/", 1)[-1].removesuffix(".safetensors")


# ---------------------------------------------------------------- 1. the training run

def _run_stems(job: TrainingJob) -> set[str]:
    """Every file stem this run wrote or could write (see training._artifact_key)."""
    out = {stem(u) for u in (job.checkpoints or []) if isinstance(u, str)}
    if job.output_lora_path:
        out.add(stem(job.output_lora_path))
    return out


def _run_could_have_written(job: TrainingJob, s: str) -> bool:
    lora_name = (job.config or {}).get("lora_name")
    if not lora_name:
        return False
    base = f"{lora_name}_v{job.version}"
    return s == base or re.fullmatch(re.escape(base) + r"(_e\d{2}|_final)", s) is not None


def find_run(jobs: list[TrainingJob], s: str) -> TrainingJob | None:
    """The run that produced the LoRA stem `s`: a recorded checkpoint first, then a name its
    artifact key could produce. Newest wins (a retry reuses the name)."""
    def newest(cands):
        return max(cands, key=lambda j: (j.completed_at or j.created_at)) if cands else None
    return (newest([j for j in jobs if s in _run_stems(j)])
            or newest([j for j in jobs if _run_could_have_written(j, s)]))


def _caption_prefix(captions: list) -> tuple[str | None, str | None]:
    """The most common "<trigger>, <gender>" prefix of a list of captions."""
    c = Counter()
    for cap in captions or []:
        if not isinstance(cap, str):
            continue
        parts = [p.strip() for p in cap.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1] in GENDERS:
            c[(parts[0], parts[1])] += 1
    if not c:
        return None, None
    return c.most_common(1)[0][0]


def run_identity(job: TrainingJob) -> tuple[str | None, str | None, bool]:
    """(trigger, gender, is_pair) as the run's publish writes them to its character row.

    Mirrors training._publish_registered (solo/pair runs) and _publish_character (older
    runs): a pair is the identity groups' "<trigger>, <gender>" phrases joined with " and ",
    gender None; a solo is the run's trigger and gender, with the captions' prefix as the
    fallback for a run that recorded no gender.
    """
    cfg = job.config or {}
    groups = [g for g in (job.identities or []) if isinstance(g, dict)]
    if cfg.get("mode") == "pair":
        phrases = [f"{t}, {g}" for t, g in
                   [(job.trigger, cfg.get("gender"))]
                   + [(g.get("trigger"), g.get("gender")) for g in groups
                      if g.get("kind") == "identity"]
                   if t and g]
        return " and ".join(phrases) or None, None, True
    if cfg.get("mode") != "solo":
        # Pre-#352 joint runs published every identity's pair (bare trigger allowed).
        pairs: list[str] = []
        for t, g in [(job.trigger, cfg.get("gender"))] + [(g.get("trigger"), g.get("gender"))
                                                          for g in groups]:
            if t:
                p = f"{t}, {g}" if g else t
                if p not in pairs:
                    pairs.append(p)
        if len(pairs) > 1:
            return " and ".join(pairs), None, True
    trigger, gender = job.trigger, cfg.get("gender")
    if not gender:
        ct, cg = _caption_prefix(cfg.get("captions") or [cfg.get("caption")])
        if ct == trigger or not trigger:
            trigger, gender = trigger or ct, cg
    return trigger or None, gender if gender in GENDERS else None, False


def from_run(name: str, job: TrainingJob) -> Provenance:
    trigger, gender, pair = run_identity(job)
    when = job.completed_at or job.created_at
    return Provenance(name=name, trigger=trigger, gender=gender, source="training_run",
                      run_id=str(job.id), run_character=job.character,
                      run_version=job.version, run_date=when.date().isoformat() if when else None,
                      detail="training_jobs", pair=pair)


# ---------------------------------------------------------------- 2. the file's metadata

def _split_class(token: str) -> tuple[str, str | None]:
    """"ohwx woman" -> ("ohwx", "woman"); "p@y, woman" -> ("p@y", "woman")."""
    t = token.strip().strip(",").strip()
    for sep in (",", " "):
        head, _, tail = t.rpartition(sep)
        if head.strip() and tail.strip() in GENDERS:
            return head.strip().rstrip(",").strip(), tail.strip()
    return t, None


def metadata_identity(meta: dict) -> tuple[str | None, str | None, str | None]:
    """(trigger, gender, key) from a safetensors `__metadata__`, or (None, None, why).

    Every source votes, weighted by how many captions/images it speaks for, and the most
    common "<trigger>, <gender>" wins.
    """
    votes: Counter = Counter()
    used: list[str] = []

    # kohya: {"<repeats>_<trigger> <class>": {"n_repeats": n, "img_count": n}}
    dirs = _json(meta.get("ss_dataset_dirs"))
    if isinstance(dirs, dict):
        for d, info in dirs.items():
            name = re.sub(r"^\d+_", "", str(d).rsplit("/", 1)[-1])
            trig, gender = _split_class(name)
            n = (info or {}).get("img_count", 1) if isinstance(info, dict) else 1
            if trig and gender:
                votes[(trig, gender)] += max(int(n or 1), 1)
                used.append("ss_dataset_dirs")

    # kohya: ss_datasets[].subsets[].class_tokens = "ohwx woman" (DreamBooth subsets)
    datasets = _json(meta.get("ss_datasets"))
    if isinstance(datasets, list):
        for ds in datasets:
            for sub in (ds or {}).get("subsets") or [] if isinstance(ds, dict) else []:
                ct = sub.get("class_tokens") if isinstance(sub, dict) else None
                if isinstance(ct, str) and ct.strip():
                    trig, gender = _split_class(ct)
                    if trig and gender:
                        votes[(trig, gender)] += max(int(sub.get("img_count") or 1), 1)
                        used.append("ss_datasets")

    # kohya: {"<dir>": {"<tag>": count}} -- captions split on commas, so the trigger and
    # the gender are separate tags. The trigger is the most frequent tag that is not a
    # gender word and is in most captions; the gender is the most frequent gender word.
    freq = _json(meta.get("ss_tag_frequency"))
    if isinstance(freq, dict) and not votes:
        tags: Counter = Counter()
        for per_dir in freq.values():
            if isinstance(per_dir, dict):
                for tag, n in per_dir.items():
                    try:
                        tags[str(tag).strip()] += int(n)
                    except (TypeError, ValueError):
                        continue
        if tags:
            top = tags.most_common(1)[0][1]
            genders = [(t, n) for t, n in tags.most_common() if t in GENDERS]
            others = [(t, n) for t, n in tags.most_common() if t not in GENDERS and t]
            if others and others[0][1] >= top * 0.5:
                trig = others[0][0]
                gender = genders[0][0] if genders and genders[0][1] >= top * 0.5 else None
                votes[(trig, gender)] += others[0][1]
                used.append("ss_tag_frequency")

    if not votes:
        present = [k for k in ("ss_tag_frequency", "ss_dataset_dirs", "ss_datasets") if k in meta]
        return None, None, ("no caption or dataset-dir keys in the header" if not present
                            else f"{', '.join(present)} present but name no trigger")
    (trig, gender), _ = votes.most_common(1)[0]
    return trig, gender, ", ".join(dict.fromkeys(used))


def _json(v):
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        return json.loads(v)
    except ValueError:
        return None


def read_header_metadata(bucket: str, key: str) -> dict | None:
    """The `__metadata__` of a safetensors object, by two ranged GETs. None if unreadable."""
    from app.s3 import _client_for_bucket
    client = _client_for_bucket(bucket)
    try:
        first = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-7")["Body"].read()
        n = int.from_bytes(first, "little")
        if len(first) != 8 or not 0 < n <= MAX_HEADER_BYTES:
            return None
        raw = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=8-{7 + n}")["Body"].read()
        meta = json.loads(raw).get("__metadata__")
        return meta if isinstance(meta, dict) else {}
    except Exception as e:  # noqa: BLE001 - an unreadable header is "no metadata", logged
        logger.warning("could not read safetensors header of %s/%s: %s", bucket, key, e)
        return None


#: (key, etag) -> (meta-derived trigger, gender, detail, ss_output_name)
_meta_cache: dict[tuple[str, str], tuple] = {}


async def _metadata_for(obj: dict) -> tuple:
    ck = (obj["key"], obj.get("etag") or "")
    if ck not in _meta_cache:
        meta = await asyncio.to_thread(read_header_metadata, settings.s3_loras_bucket, obj["key"])
        if meta is None:
            # Not cached: a transient S3 failure must not stick until the next deploy.
            return None, None, "the safetensors header could not be read", None
        trig, gender, detail = metadata_identity(meta)
        out_name = meta.get("ss_output_name") or meta.get("modelspec.title")
        _meta_cache[ck] = (trig, gender, detail, out_name if isinstance(out_name, str) else None)
    return _meta_cache[ck]


def clear_cache() -> None:
    _meta_cache.clear()


# ---------------------------------------------------------------- together

def lora_objects(listing: list[dict]) -> dict[str, dict]:
    """stem -> bucket object, from a list_bucket() listing. character/ wins a name clash."""
    out: dict[str, dict] = {}
    for o in sorted(listing, key=lambda o: not o["name"].startswith("character/")):
        if o["name"].endswith(".safetensors"):
            out.setdefault(stem(o["name"]), {"key": o["name"], "etag": o.get("etag")})
    return out


async def load_runs(db: AsyncSession) -> list[TrainingJob]:
    return list((await db.execute(select(TrainingJob))).scalars().all())


async def provenance(name: str, runs: list[TrainingJob], objects: dict[str, dict]) -> Provenance:
    """What the LoRA `name` trained on: its run, else its header, else nothing."""
    s = stem(name)
    job = find_run(runs, s)
    if job is not None:
        return from_run(s, job)
    obj = objects.get(s)
    if obj is None:
        return Provenance(name=s, detail="no training run, and no such file in the LoRA bucket")
    trig, gender, detail, out_name = await _metadata_for(obj)
    if trig:
        return Provenance(name=s, trigger=trig, gender=gender, source="lora_metadata",
                          detail=detail)
    # A renamed wanly file: its header still carries the name the run wrote it under.
    if out_name and stem(out_name) != s:
        job = find_run(runs, stem(out_name))
        if job is not None:
            p = from_run(s, job)
            p.detail = f"training_jobs, via ss_output_name {out_name!r}"
            return p
    return Provenance(name=s, detail=f"no training run; LoRA metadata: {detail}")


def trained_values(p: Provenance) -> dict:
    """The character fields this provenance vouches for. Gender only when it is known -- or
    a pair, whose right gender IS none."""
    if p.source == "none" or not p.trigger:
        return {}
    out = {"trigger": p.trigger}
    if p.gender or p.pair:
        out["gender"] = p.gender
    return out


def mismatches(stored: dict, p: Provenance) -> list[dict]:
    """[{field, stored, trained}] where the character row disagrees with its LoRA."""
    out = []
    for field, trained in trained_values(p).items():
        have = stored.get(field)
        have = have.strip() if isinstance(have, str) else have
        if (have or None) != (trained or None):
            out.append({"field": field, "stored": have or None, "trained": trained})
    return out
