"""Has a character's newest LTX LoRA been tried? (app/lora_status.py)

TESTED = a COMPLETED render segment whose recipe names one of the newest completed LTX run's
checkpoints, by the published basename `{lora_name}_v{N}_{label}`. One aggregate, every
character at once.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app import lora_status as ls
from app.enums import JobStatus, SegmentStatus, TrainingStatus
from app.models import Job, LtxCharacter, Segment, TrainingJob, User


async def _char(db, name, char_lora=None):
    c = LtxCharacter(name=name, char_lora=char_lora, trigger="t@g", gender="woman",
                     strength_stage_1=0.8, strength_stage_2=1.0)
    db.add(c)
    await db.flush()
    return c


async def _run(db, character, version, *, done_at, arch="ltx", labels=("e01", "e02", "final"),
               uploaded=(), status=TrainingStatus.COMPLETED, lora_name=None):
    cfg = {"arch": arch, "mode": "solo"}
    if lora_name:
        cfg["lora_name"] = lora_name
    stem = lora_name or character
    j = TrainingJob(id=uuid.uuid4(), character=character, trigger="t@g", version=version,
                    dataset_images=["s3://b/1.png"], status=status, config=cfg,
                    epochs=[{"label": lb, "step": 1, "loss": 0.5} for lb in labels],
                    checkpoints=[f"s3://ltx-loras/character/{stem}_v{version}_{lb}.safetensors"
                                 for lb in uploaded] or None,
                    created_at=done_at - timedelta(hours=1), completed_at=done_at)
    db.add(j)
    await db.flush()
    return j


async def _render(db, name, lora, *, status=SegmentStatus.COMPLETED, as_list=True, at=None):
    u = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(u)
    await db.flush()
    job = Job(user_id=u.id, name="j", width=8, height=8, fps=24, seed=1,
              status=JobStatus.PENDING)
    db.add(job)
    await db.flush()
    recipe = {"character": name, "char_lora": lora}
    if as_list:
        recipe["characters"] = [{"name": name, "char_lora": lora, "s1": 0.8, "s2": 1.0}]
    db.add(Segment(job_id=job.id, index=0, prompt="p", status=status, ltx_recipe=recipe,
                   completed_at=at or datetime.now(timezone.utc)))
    await db.flush()


def _n(p="c"):
    return f"{p}{uuid.uuid4().hex[:6]}"


@pytest.mark.asyncio
async def test_the_newest_run_is_tested_once_a_render_names_its_checkpoint(db):
    now = datetime.now(timezone.utc)
    name = _n()
    c = await _char(db, name, char_lora=f"{name}_v1_final")
    await _run(db, name, 1, done_at=now - timedelta(days=3), uploaded=("final",))
    await _run(db, name, 2, done_at=now - timedelta(days=1), uploaded=("e02",))
    await _run(db, name, 3, done_at=now, arch="sdxl")          # SDXL never counts
    st = (await ls.lora_status(db, [c]))[name]["latest_lora"]
    assert st["run_version"] == 2 and not st["tested"] and st["renders"] == 0
    assert st["name"] == f"{name}_v2_e02" and st["uploaded"]   # the uploaded one is shown

    await _render(db, name, f"{name}_v2_e02")
    await _render(db, name, f"{name}_v2_e02", as_list=False)
    await _render(db, name, f"{name}_v2_e02", status=SegmentStatus.FAILED)  # not counted
    st = (await ls.lora_status(db, [c]))[name]
    assert st["latest_lora"]["tested"] and st["latest_lora"]["renders"] == 2
    assert st["latest_lora"]["last_rendered_at"] is not None
    # The star (v1 final) is not the latest -- and has no renders of its own.
    assert st["latest_lora"]["is_starred"] is False
    assert st["starred_lora_renders"] == {"name": f"{name}_v1_final", "renders": 0,
                                          "last_rendered_at": None}


@pytest.mark.asyncio
async def test_lora_name_is_the_stem_not_the_character(db):
    """`pay_v2_e03`, not `p@yton_...`: the stem the run was created with (_artifact_key)."""
    now = datetime.now(timezone.utc)
    name = _n("p@y")
    c = await _char(db, name)
    await _run(db, name, 2, done_at=now, labels=("e03",), uploaded=("e03",), lora_name="pay")
    await _render(db, name, "pay_v2_e03.safetensors")
    st = (await ls.lora_status(db, [c]))[name]["latest_lora"]
    assert st["name"] == "pay_v2_e03" and st["tested"] and st["renders"] == 1


@pytest.mark.asyncio
async def test_no_ltx_run_means_no_status(db):
    name = _n()
    c = await _char(db, name)
    await _run(db, name, 1, done_at=datetime.now(timezone.utc), status=TrainingStatus.FAILED)
    assert (await ls.lora_status(db, [c]))[name]["latest_lora"] is None


@pytest.mark.asyncio
async def test_the_grid_and_list_payloads_carry_it(db):
    """Both are hand-built or re-validated; a field missing from either is silently dropped."""
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app
    name = _n()
    await _char(db, name)
    await _run(db, name, 1, done_at=datetime.now(timezone.utc), uploaded=("final",))
    await _render(db, name, f"{name}_v1_final")
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
            book = (await cl.get("/recipes")).json()
            listed = (await cl.get("/ltx/characters")).json()
    finally:
        app.dependency_overrides.clear()
    for rows in (book["characters"], listed):
        row = next(r for r in rows if r["name"] == name)
        assert row["latest_lora"]["tested"] is True and row["latest_lora"]["renders"] == 1
        assert "starred_lora_renders" in row
