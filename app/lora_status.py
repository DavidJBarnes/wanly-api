"""Has the character's latest LoRA been tried? (the "tested" chip on the Characters grid)

Two queries for every character at once -- never one per character:

  * the newest COMPLETED LTX run of each character (pairs: the pair's own runs), with the
    file names its checkpoints were published under -- `_artifact_key`'s rule,
    `{lora_name}_v{version}_{eNN|final}`, which is also what `char_lora` and every render
    recipe store;
  * one aggregate over completed render segments: how many named each LoRA, and when last.

TESTED = at least one completed render named one of the latest run's checkpoints. There is
no per-checkpoint timestamp (a run records epochs by step, not time), so the latest LoRA's
`at` is its run's completed_at.
"""
from __future__ import annotations

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import TrainingJob

_RENDERS = text("""
    SELECT name, count(DISTINCT id) AS renders, max(completed_at) AS last_at
    FROM (
        SELECT s.id, s.completed_at, s.ltx_recipe->>'char_lora' AS name
          FROM segments s
         WHERE s.status = 'completed' AND jsonb_typeof(s.ltx_recipe) = 'object'
        UNION ALL
        SELECT s.id, s.completed_at, c->>'char_lora' AS name
          FROM segments s,
               jsonb_array_elements(CASE WHEN jsonb_typeof(s.ltx_recipe->'characters') = 'array'
                                         THEN s.ltx_recipe->'characters'
                                         ELSE '[]'::jsonb END) c
         WHERE s.status = 'completed' AND jsonb_typeof(s.ltx_recipe) = 'object'
    ) x
    WHERE name IS NOT NULL AND name <> '' AND lower(name) <> 'none'
    GROUP BY name
""")


def _stem(job: TrainingJob) -> str:
    from app.routes.training import _default_lora_name
    return (job.config or {}).get("lora_name") or _default_lora_name(job.character)


def _label_order(label: str) -> int:
    """final last; e01 < e02 < ..."""
    if label == "final":
        return 10_000
    try:
        return int(label.lstrip("e"))
    except ValueError:
        return -1


def checkpoint_names(job: TrainingJob) -> list[tuple[str, str]]:
    """[(label, basename)] for an LTX run, in epoch order -- the names a render recipe uses."""
    stem = _stem(job)
    labels = [e.get("label") for e in (job.epochs or []) if isinstance(e, dict) and e.get("label")]
    labels = sorted(set(labels), key=_label_order)
    return [(lb, f"{stem}_v{job.version}_{lb}") for lb in labels]


def _base(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].removesuffix(".safetensors")


async def render_counts(db: AsyncSession) -> dict[str, tuple[int, object]]:
    """{lora basename: (completed renders, last completed_at)} over every segment."""
    rows = (await db.execute(_RENDERS)).all()
    out: dict[str, tuple[int, object]] = {}
    for name, n, last in rows:
        name = name.removesuffix(".safetensors")
        prev = out.get(name)
        out[name] = (n + (prev[0] if prev else 0),
                     max(filter(None, [last, prev[1] if prev else None]), default=None))
    return out


async def lora_status(db: AsyncSession, characters) -> dict[str, dict]:
    """{character name: {"latest_lora": {...} | None, "starred_lora_renders": {...} | None}}"""
    names = [c.name for c in characters]
    if not names:
        return {}
    runs = (await db.execute(
        select(TrainingJob)
        .where(TrainingJob.character.in_(names), TrainingJob.status == "completed",
               or_(TrainingJob.config["arch"].astext.is_(None),
                   TrainingJob.config["arch"].astext == "ltx"))
        .order_by(TrainingJob.completed_at.desc().nulls_last()))).scalars().all()
    latest: dict[str, TrainingJob] = {}
    for j in runs:
        latest.setdefault(j.character, j)
    counts = await render_counts(db)
    out: dict[str, dict] = {}
    for c in characters:
        star = (c.char_lora or "").removesuffix(".safetensors") or None
        starred = None
        if star and star.lower() != "none":
            n, last = counts.get(star, (0, None))
            starred = {"name": star, "renders": n, "last_rendered_at": last}
        job = latest.get(c.name)
        info = None
        if job is not None:
            ckpts = checkpoint_names(job)
            uploaded = {_base(u) for u in (job.checkpoints or []) if isinstance(u, str)}
            per = [{"label": lb, "name": nm, "uploaded": nm in uploaded,
                    "renders": counts.get(nm, (0, None))[0],
                    "last_rendered_at": counts.get(nm, (0, None))[1]} for lb, nm in ckpts]
            pick = (next((p for p in per if p["name"] == star), None)
                    or max((p for p in per if p["renders"]), key=lambda p: p["renders"], default=None)
                    or next((p for p in reversed(per) if p["uploaded"]), None)
                    or (per[-1] if per else None))
            total = sum(p["renders"] for p in per)
            lasts = [p["last_rendered_at"] for p in per if p["last_rendered_at"]]
            info = {
                "run_id": str(job.id), "run_version": job.version, "at": job.completed_at,
                "name": pick["name"] if pick else None, "label": pick["label"] if pick else None,
                "uploaded": bool(pick and pick["uploaded"]),
                "is_starred": bool(pick and pick["name"] == star),
                "renders": total, "tested": total > 0,
                "last_rendered_at": max(lasts) if lasts else None,
                "checkpoints": per,
            }
        out[c.name] = {"latest_lora": info, "starred_lora_renders": starred}
    return out
