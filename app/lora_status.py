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

SDXL (#458): SDXL LoRAs are never used by a Wanly render -- they are used by hand in A1111 --
so their "tested" comes from `lora_usage`, which the A1111 box's reporter fills from A1111's
saved PNGs (wanly-gpu-docker#211). Same shape, `{stem}_sdxl_v{N}_{label}` per _artifact_key,
matched case-insensitively against the names A1111 wrote.
"""
from __future__ import annotations

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LoraUsage, TrainingJob

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


def _is_sdxl(job: TrainingJob) -> bool:
    return (job.config or {}).get("arch") == "sdxl"


def checkpoint_names(job: TrainingJob) -> list[tuple[str, str]]:
    """[(label, basename)] in epoch order -- the names a render recipe (LTX) or A1111 (SDXL,
    with `_sdxl` in the name) uses. Mirrors training._artifact_key."""
    stem = _stem(job)
    infix = "_sdxl" if _is_sdxl(job) else ""
    labels = [e.get("label") for e in (job.epochs or []) if isinstance(e, dict) and e.get("label")]
    labels = sorted(set(labels), key=_label_order)
    return [(lb, f"{stem}{infix}_v{job.version}_{lb}") for lb in labels]


async def a1111_counts(db: AsyncSession) -> dict[str, tuple[int, object]]:
    """{lower-cased lora name: (images, last_used_at)} from the A1111 reporter (#458)."""
    rows = (await db.execute(select(LoraUsage.name, LoraUsage.images, LoraUsage.last_used_at)
                             .where(LoraUsage.source == "a1111"))).all()
    out: dict[str, tuple[int, object]] = {}
    for name, n, last in rows:
        k = name.removesuffix(".safetensors").lower()
        prev = out.get(k)
        out[k] = (n + (prev[0] if prev else 0),
                  max(filter(None, [last, prev[1] if prev else None]), default=None))
    return out


def _sdxl_info(job: TrainingJob, usage: dict[str, tuple[int, object]]) -> dict:
    ckpts = checkpoint_names(job)
    uploaded = {_base(u) for u in (job.checkpoints or []) if isinstance(u, str)}
    per = [{"label": lb, "name": nm, "uploaded": nm in uploaded,
            "a1111_images": usage.get(nm.lower(), (0, None))[0],
            "a1111_last_used_at": usage.get(nm.lower(), (0, None))[1]} for lb, nm in ckpts]
    # The checkpoint shown: the one used most in A1111, else the newest uploaded, else the last.
    pick = (max((p for p in per if p["a1111_images"]), key=lambda p: p["a1111_images"], default=None)
            or next((p for p in reversed(per) if p["uploaded"]), None)
            or (per[-1] if per else None))
    total = sum(p["a1111_images"] for p in per)
    lasts = [p["a1111_last_used_at"] for p in per if p["a1111_last_used_at"]]
    return {
        "run_id": str(job.id), "run_version": job.version, "at": job.completed_at,
        "name": pick["name"] if pick else None, "label": pick["label"] if pick else None,
        "uploaded": bool(pick and pick["uploaded"]),
        "a1111_images": total, "tested": total > 0,
        "a1111_last_used_at": max(lasts) if lasts else None,
        "checkpoints": per,
    }


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
    """{character name: {"latest_lora": {...} | None, "starred_lora_renders": {...} | None,
    "latest_sdxl_lora": {...} | None}}"""
    names = [c.name for c in characters]
    if not names:
        return {}
    runs = (await db.execute(
        select(TrainingJob)
        .where(TrainingJob.character.in_(names), TrainingJob.status == "completed",
               or_(TrainingJob.config["arch"].astext.is_(None),
                   TrainingJob.config["arch"].astext.in_(["ltx", "sdxl"])))
        .order_by(TrainingJob.completed_at.desc().nulls_last()))).scalars().all()
    latest: dict[str, TrainingJob] = {}
    latest_sdxl: dict[str, TrainingJob] = {}
    for j in runs:
        (latest_sdxl if _is_sdxl(j) else latest).setdefault(j.character, j)
    counts = await render_counts(db)
    usage = await a1111_counts(db) if latest_sdxl else {}
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
        sx = latest_sdxl.get(c.name)
        out[c.name] = {"latest_lora": info, "starred_lora_renders": starred,
                       "latest_sdxl_lora": _sdxl_info(sx, usage) if sx is not None else None}
    return out
