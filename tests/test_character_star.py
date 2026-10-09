"""wanly-api#452: a character HAS its dataset, and its render LoRA is a STAR.

David: "A character has a dataset, that dataset is used to build loras for sdxl and ltx. Let the
user star the default LTX version safetensor." A run finishing no longer repoints char_lora;
the star does, and a checkpoint still on the trainer is uploaded first.
"""
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException, Response

from app.enums import TrainingStatus
from app.models import Dataset, LtxCharacter, TrainingJob
from app.routes import characters as ch
from app.routes.datasets import _refuse_second_living
from app.routes.training import apply_pending_stars, update_training_job
from app.schemas.ltx import CharacterStar
from app.schemas.training import TrainingProgress

BUCKET = "s3://ltx-loras/character"


def _n(p="c"):
    return f"{p}{uuid.uuid4().hex[:6]}"


async def _char(db, **kw):
    kw.setdefault("name", _n("Char"))
    kw.setdefault("trigger", "t@g")
    kw.setdefault("gender", "woman")
    kw.setdefault("char_lora", "old_v1_final")
    c = LtxCharacter(**kw)
    db.add(c)
    await db.flush()
    return c


async def _run(db, character, *, arch="ltx", mode="solo", checkpoints=(), labels=("e01", "final"),
               status=TrainingStatus.COMPLETED, **cfg):
    j = TrainingJob(id=uuid.uuid4(), character=character, trigger="t@g", version=2,
                    dataset_images=["s3://b/1.png"], status=status,
                    config={"arch": arch, "mode": mode, "gender": "woman",
                            "base_checkpoint": "10Eros_v1.5_bf16", **cfg},
                    epochs=[{"label": l, "step": 1, "loss": 0.5} for l in labels],
                    checkpoints=list(checkpoints) or None,
                    created_at=datetime.now(timezone.utc))
    db.add(j)
    await db.flush()
    return j


class _U:
    id = None


@pytest.mark.asyncio
class TestFinishingARunDoesNotRepoint:
    async def test_a_completed_run_leaves_the_render_lora_alone(self, db):
        c = await _char(db)
        j = await _run(db, c.name, status=TrainingStatus.RUNNING,
                       checkpoints=[f"{BUCKET}/new_v2_final.safetensors"])
        j.output_lora_path = f"{BUCKET}/new_v2_final.safetensors"
        await update_training_job(j.id, TrainingProgress(status=TrainingStatus.COMPLETED), db=db)
        await db.refresh(c)
        assert c.char_lora == "old_v1_final"

    async def test_a_pairs_first_run_creates_its_row_as_a_draft(self, db):
        name = _n("Pair")
        j = await _run(db, name, mode="pair", status=TrainingStatus.RUNNING,
                       members=["Me", "Joana"])
        j.identities = [{"kind": "identity", "trigger": "jo@na", "gender": "woman"}]
        j.trigger, j.config = "d@vid", {**j.config, "gender": "man"}
        await db.flush()
        await update_training_job(j.id, TrainingProgress(status=TrainingStatus.COMPLETED), db=db)
        row = (await db.execute(__import__("sqlalchemy").select(LtxCharacter).where(
            LtxCharacter.name == name))).scalar_one()
        assert row.kind == "pair" and row.char_lora is None
        assert row.trigger == "d@vid, man and jo@na, woman"


@pytest.mark.asyncio
class TestTheStar:
    async def test_an_uploaded_checkpoint_is_starred_now(self, db):
        c = await _char(db)
        j = await _run(db, c.name, checkpoints=[f"{BUCKET}/new_v2_e01.safetensors",
                                                f"{BUCKET}/new_v2_final.safetensors"])
        r = Response()
        out = await ch.star(c.id, CharacterStar(training_job_id=j.id, label="e01"), r,
                            _user=_U(), db=db)
        assert out.char_lora == "new_v2_e01" and out.star_pending is None
        assert out.base_checkpoint == "10Eros_v1.5_bf16"
        assert r.status_code in (None, 200)

    async def test_one_still_on_the_trainer_is_uploaded_first(self, db):
        c = await _char(db)
        j = await _run(db, c.name, checkpoints=[f"{BUCKET}/new_v2_final.safetensors"])
        r = Response()
        out = await ch.star(c.id, CharacterStar(training_job_id=j.id, label="e01"), r,
                            _user=_U(), db=db)
        assert r.status_code == 202
        assert out.char_lora == "old_v1_final", "nothing changes until it lands"
        assert out.star_pending == {"training_job_id": str(j.id), "label": "e01"}
        await db.refresh(j)
        assert "e01" in (j.publish_requests or [])
        # it lands
        uri = f"{BUCKET}/new_v2_e01.safetensors"
        j.checkpoints = (j.checkpoints or []) + [uri]
        await apply_pending_stars(db, j, uri)
        await db.flush()  # the route commits right after
        await db.refresh(c)
        assert c.char_lora == "new_v2_e01" and c.star_pending is None

    async def test_sdxl_is_for_a1111_and_cannot_be_starred(self, db):
        c = await _char(db)
        j = await _run(db, c.name, arch="sdxl",
                       checkpoints=[f"{BUCKET}/sdxl/new_sdxl_v2_final.safetensors"])
        with pytest.raises(HTTPException) as e:
            await ch.star(c.id, CharacterStar(training_job_id=j.id, label="final"), Response(),
                          _user=_U(), db=db)
        assert e.value.status_code == 422

    async def test_another_characters_run_is_refused(self, db):
        c = await _char(db)
        j = await _run(db, _n("Other"), checkpoints=[f"{BUCKET}/x_v2_final.safetensors"])
        with pytest.raises(HTTPException) as e:
            await ch.star(c.id, CharacterStar(training_job_id=j.id, label="final"), Response(),
                          _user=_U(), db=db)
        assert e.value.status_code == 422

    async def test_a_label_the_run_never_made_is_404(self, db):
        c = await _char(db)
        j = await _run(db, c.name)
        with pytest.raises(HTTPException) as e:
            await ch.star(c.id, CharacterStar(training_job_id=j.id, label="e09"), Response(),
                          _user=_U(), db=db)
        assert e.value.status_code == 404


@pytest.mark.asyncio
class TestOneLivingSet:
    async def test_a_second_living_set_is_refused_and_an_archived_one_is_not_counted(self, db):
        c = await _char(db)
        d = Dataset(id=uuid.uuid4(), name=_n("set"), kind="character", character=c.name,
                    images=[], prefix="p", captions={}, scores={}, faces={})
        db.add(d)
        await db.flush()
        with pytest.raises(HTTPException) as e:
            await _refuse_second_living(db, "character", c.name)
        assert e.value.status_code == 409 and d.name in e.value.detail
        await _refuse_second_living(db, "character", c.name, exclude=d.id)
        d.archived_at = datetime.now(timezone.utc)
        await db.flush()
        await _refuse_second_living(db, "character", c.name)


@pytest.mark.asyncio
class TestFull:
    async def test_the_page_in_one_call(self, db):
        c = await _char(db)
        live = Dataset(id=uuid.uuid4(), name=c.name, kind="character", character=c.name,
                       images=["s3://b/a.png"], prefix="p", captions={}, scores={}, faces={})
        old = Dataset(id=uuid.uuid4(), name=f"{c.name} v1", kind="character", character=c.name,
                      images=["s3://b/a.png"], prefix="q", captions={}, scores={}, faces={},
                      archived_at=datetime.now(timezone.utc))
        db.add_all([live, old])
        await db.flush()
        out = await ch.character_full(c.name.upper(), db=db)
        assert out.character.name == c.name
        assert out.dataset.id == live.id
        assert [a.id for a in out.archived] == [old.id]

    async def test_unknown_is_404(self, db):
        with pytest.raises(HTTPException) as e:
            await ch.character_full(_n("nobody"), db=db)
        assert e.value.status_code == 404


@pytest.mark.asyncio
class TestRunsHomedOnArchivedSets:
    async def test_listed_on_the_page_and_starrable(self, db):
        """Payton renders with Payton-Synthetic's checkpoint, whose home is the archived
        "Payton v1": the page lists it and the star accepts it."""
        from app.models import TrainingRunDataset
        c = await _char(db)
        live = Dataset(id=uuid.uuid4(), name=c.name, kind="character", character=c.name,
                       images=["s3://b/a.png"], prefix="p", captions={}, scores={}, faces={})
        old = Dataset(id=uuid.uuid4(), name=f"{c.name} v1", kind="character", character=c.name,
                      images=["s3://b/a.png"], prefix="q", captions={}, scores={}, faces={},
                      archived_at=datetime.now(timezone.utc))
        db.add_all([live, old])
        await db.flush()
        other = _n("Synth")
        j = await _run(db, other, mode=None, checkpoints=[f"{BUCKET}/{other}_v1_final.safetensors"])
        db.add(TrainingRunDataset(training_job_id=j.id, group_index=0, dataset_id=old.id,
                                  dataset_name=old.name, kind="identity", character=other,
                                  images=["s3://b/a.png"], captions=None, num_repeats=10,
                                  source="backfill"))
        await db.flush()
        full = await ch.character_full(c.name, db=db)
        assert str(j.id) in {r.job_id for r in full.runs}
        out = await ch.star(c.id, CharacterStar(training_job_id=j.id, label="final"), Response(),
                            _user=_U(), db=db)
        assert out.char_lora == f"{other}_v1_final"
