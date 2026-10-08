"""What each training run trained on, per dataset -- the run is the record (wanly-api#419).

A dataset no longer locks once it trains (#420): it is the subject's living set, edited
freely. So everything that used to be read off a frozen set is read off the RUN instead:

  rows_for_job(job)        the run's groups as `training_run_datasets` rows, built from the
                           job's own snapshot columns -- the same data the trainer reads
  trained_uris(db)         every file any run's snapshot names, whatever its status: what a
                           delete or an overwrite must never touch (#421)
  used_in(db)              {uri: [the runs that trained on it]}, for the dataset page's badges
  trained_on(db, job)      per dataset, what the run trained on and how the set differs now

READ FROM THE JOB SNAPSHOT, NOT ONLY THE LINK TABLE. Runs created before migration 114 have no
link rows until the backfill (#424) writes them, and the guarantees above must hold for them on
deploy day. Every run since #352 carries its images per group in dataset_images + identities,
and every run ever carries dataset_images -- that is the whole of what was trained.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enums import TrainingStatus
from app.models import Dataset, TrainingJob, TrainingRunDataset

#: Runs that produced (or are producing) a LoRA. A failed or cancelled run trained nothing a
#: badge should claim; its files are still protected by trained_uris, which reads every run.
_NO_LORA = (TrainingStatus.FAILED, TrainingStatus.CANCELLED)


def _arch(job: TrainingJob) -> str:
    return ((job.config or {}).get("arch") or "ltx")


def _as_uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def job_groups(job: TrainingJob) -> list[dict]:
    """The run's groups, group 0 first, as plain dicts read from its snapshot columns.

    Group 0 is flat on the job (dataset_images + config); 1.. are `identities`. Keys:
    index, kind, character, dataset_id, dataset_name, images, captions, num_repeats, windows.
    """
    config = job.config or {}
    prov0 = config.get("dataset") or {}
    out = [{
        "index": 0,
        "kind": config.get("kind") or "identity",
        "character": (config.get("members") or [job.character])[0] or job.character,
        "dataset_id": _as_uuid(prov0.get("id")),
        "dataset_name": prov0.get("name"),
        "images": list(job.dataset_images or []),
        "captions": config.get("captions"),
        "num_repeats": config.get("num_repeats"),
        "windows": 1,
    }]
    for i, g in enumerate(job.identities if isinstance(job.identities, list) else [], start=1):
        if not isinstance(g, dict):
            continue
        prov = g.get("dataset") or {}
        out.append({
            "index": i,
            "kind": g.get("kind") or ("identity" if g.get("trigger") else "composition"),
            "character": g.get("character"),
            "dataset_id": _as_uuid(prov.get("id")),
            "dataset_name": prov.get("name"),
            "images": list(g.get("images") or []),
            "captions": g.get("captions"),
            "num_repeats": g.get("num_repeats"),
            "windows": g.get("windows") or 1,
        })
    return out


def rows_for_job(job: TrainingJob, source: str = "created") -> list[TrainingRunDataset]:
    """One link row per group of the run, exactly as its snapshot says it trained."""
    return [TrainingRunDataset(
        training_job_id=job.id, group_index=g["index"], dataset_id=g["dataset_id"],
        dataset_name=g["dataset_name"], kind=g["kind"], character=g["character"],
        images=g["images"], captions=g["captions"], num_repeats=g["num_repeats"],
        windows=g["windows"], source=source,
    ) for g in job_groups(job)]


async def trained_uris(db: AsyncSession) -> set[str]:
    """Every file any training run's snapshot names, in any status (#421).

    Failed and cancelled runs count too: their record says what they tried to train, and a
    retry trains exactly that snapshot (#423). One row per run ever, matched in Python, the
    same trade find_image_references makes.
    """
    out: set[str] = set()
    rows = (await db.execute(select(TrainingJob.dataset_images, TrainingJob.identities))).all()
    for images, identities in rows:
        out.update(u for u in (images or []) if isinstance(u, str))
        for g in identities if isinstance(identities, list) else []:
            if isinstance(g, dict):
                out.update(u for u in (g.get("images") or []) if isinstance(u, str))
    for (images,) in (await db.execute(select(TrainingRunDataset.images))).all():
        out.update(u for u in (images or []) if isinstance(u, str))
    return out


def _run_ref(job: TrainingJob) -> dict:
    return {"job_id": str(job.id), "character": job.character, "version": job.version,
            "arch": _arch(job), "status": job.status, "created_at": job.created_at}


async def used_in(db: AsyncSession) -> dict[str, list[dict]]:
    """{uri: every run with a LoRA that trained on it, oldest first} -- the "used in" badges.

    Regularization groups are left out: a pool image "used in Kelly v2" says nothing useful
    about the pool, and the badge is for a subject's own photographs.
    """
    jobs = (await db.execute(
        select(TrainingJob).where(TrainingJob.status.not_in(list(_NO_LORA)))
        .order_by(TrainingJob.created_at.asc(), TrainingJob.id.asc()))).scalars().all()
    out: dict[str, list[dict]] = {}
    for job in jobs:
        ref = _run_ref(job)
        seen: set[str] = set()
        for g in job_groups(job):
            if g["kind"] == "regularization":
                continue
            for u in g["images"]:
                if isinstance(u, str) and u not in seen:
                    seen.add(u)
                    out.setdefault(u, []).append(ref)
    return out


async def link_rows(db: AsyncSession, job: TrainingJob) -> list[TrainingRunDataset]:
    """The run's link rows, or -- for a run the backfill has not reached -- unsaved rows built
    from its snapshot, so every run answers the same way."""
    rows = (await db.execute(
        select(TrainingRunDataset).where(TrainingRunDataset.training_job_id == job.id)
        .order_by(TrainingRunDataset.group_index))).scalars().all()
    return list(rows) or rows_for_job(job, source="snapshot")


async def trained_on(db: AsyncSession, job: TrainingJob) -> list[dict]:
    """Per group: the dataset, what trained (images + captions as trained), and the diff
    against what that dataset holds NOW -- added since, removed since."""
    out = []
    for row in await link_rows(db, job):
        ds = await db.get(Dataset, row.dataset_id) if row.dataset_id else None
        trained = list(row.images or [])
        captions = row.captions if isinstance(row.captions, list) else None
        now = list(ds.images or []) if ds else None
        trained_set = set(trained)
        out.append({
            "group_index": row.group_index,
            "kind": row.kind,
            "character": row.character,
            "dataset_id": str(ds.id) if ds else (str(row.dataset_id) if row.dataset_id else None),
            "dataset_name": ds.name if ds else row.dataset_name,
            "dataset_exists": ds is not None,
            "dataset_name_as_trained": row.dataset_name,
            "num_repeats": row.num_repeats,
            "windows": row.windows or 1,
            "images": [{"uri": u,
                        "caption": captions[i] if captions and i < len(captions) else None,
                        "still_in_dataset": (u in set(now)) if now is not None else None}
                       for i, u in enumerate(trained)],
            "added_since": [u for u in now if u not in trained_set] if now is not None else [],
            "removed_since": ([u for u in trained if u not in set(now)]
                              if now is not None else []),
        })
    return out
