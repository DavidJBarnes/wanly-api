"""Living datasets: the run is the record (wanly-api#419).

  #422  every run writes training_run_datasets rows; GET /training/{id}/trained-on answers
        "what did vN train on" with a diff against the set now
  #421  files a run trained on survive: the Image Repo's delete gate names every run that
        used them, and a move copies them rather than taking them away
  #419  archived sets: hidden, read-only, never trained from
  #424  the backfill: one living set per subject, version sets archived, every run linked,
        dry run by default, idempotent
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.enums import TrainingStatus
from app.schemas.training import TrainingCreate
from tests.test_dataset_lock import _ds, _patch, _run, _Usr
from tests.test_training_jobs import SOLO, _U, _world

pytestmark = pytest.mark.asyncio


async def _http(db, method, url, **kw):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user, verify_api_key_or_bearer
    from app.database import get_db
    from app.main import app
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: _Usr()
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await getattr(c, method)(url, **kw)
    finally:
        app.dependency_overrides.clear()


class TestTrainedOn:
    async def test_a_new_run_records_each_group_and_the_diff_follows_the_set(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        trained = list(job.dataset_images)
        david = w["david"]
        gone = trained[0]
        david.images = [u for u in david.images if u != gone] + ["s3://wanly-images/new.jpg"]
        await db.commit()

        r = await _http(db, "get", f"/training/{job.id}/trained-on")
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["character"], body["version"], body["arch"]) == ("David", 1, "ltx")
        g0 = body["groups"][0]
        assert g0["dataset_id"] == str(david.id) and g0["dataset_name"] == "David"
        assert [i["uri"] for i in g0["images"]] == trained
        assert g0["images"][1]["caption"] == job.config["captions"][1]
        assert g0["images"][0]["still_in_dataset"] is False
        assert g0["added_since"] == ["s3://wanly-images/new.jpg"]
        assert g0["removed_since"] == [gone]

    async def test_a_run_from_before_the_table_answers_from_its_snapshot(self, db):
        from app.models import TrainingJob
        ds = await _ds(db)
        job = TrainingJob(character="Old", trigger="o", version=1, status="completed",
                          dataset_images=list(ds.images),
                          config={"dataset": {"id": str(ds.id), "name": ds.name}})
        db.add(job)
        await db.commit()
        r = await _http(db, "get", f"/training/{job.id}/trained-on")
        g0 = r.json()["groups"][0]
        assert g0["dataset_name"] == ds.name and len(g0["images"]) == 4
        assert g0["added_since"] == [] and g0["removed_since"] == []

    async def test_a_deleted_set_keeps_its_name_as_trained(self, db):
        from app.routes.datasets import delete_dataset
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        await delete_dataset(w["david"].id, purge=False, _user=None, db=db)
        r = await _http(db, "get", f"/training/{job.id}/trained-on")
        g0 = r.json()["groups"][0]
        assert g0["dataset_exists"] is False and g0["dataset_name"] == "David"
        assert len(g0["images"]) == len(job.dataset_images)

    async def test_a_sets_other_group_is_not_added_since(self, db):
        """Joana v3 trained a set's stills and its clips as two groups: neither half is
        'added since' the other."""
        from app.models import TrainingJob
        ds = await _ds(db, images=[f"s3://wanly-images/j/{i}.jpg" for i in range(3)]
                       + ["s3://wanly-images/j/c.mp4"])
        prov = {"id": str(ds.id), "name": ds.name}
        job = TrainingJob(character="J", trigger="j", version=1, status="completed",
                          dataset_images=ds.images[:3], config={"dataset": prov},
                          identities=[{"kind": "clip", "images": ds.images[3:],
                                       "dataset": prov}])
        db.add(job)
        await db.commit()
        body = (await _http(db, "get", f"/training/{job.id}/trained-on")).json()
        assert [g["added_since"] for g in body["groups"]] == [[], []]

    async def test_unknown_run_is_404(self, db):
        r = await _http(db, "get", f"/training/{uuid.uuid4()}/trained-on")
        assert r.status_code == 404


class TestArchive:
    async def test_archived_sets_are_hidden_read_only_and_reversible(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import archive_dataset, list_datasets, unarchive_dataset
        ds = await _ds(db)
        out = await archive_dataset(ds.id, _user=None, db=db)
        assert out.archived_at is not None and out.locked is True
        assert ds.id not in {d.id for d in await list_datasets(include_archived=False, db=db)}
        assert ds.id in {d.id for d in await list_datasets(include_archived=True, db=db)}
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[:1])
        assert e.value.status_code == 409 and "archived" in e.value.detail
        back = await unarchive_dataset(ds.id, _user=None, db=db)
        assert back.archived_at is None and back.locked is False

    async def test_over_http(self, db):
        ds = await _ds(db)
        r = await _http(db, "post", f"/datasets/{ds.id}/archive")
        assert r.status_code == 200 and r.json()["archived_at"]
        lst = await _http(db, "get", "/datasets")
        assert str(ds.id) not in {d["id"] for d in lst.json()}
        lst = await _http(db, "get", "/datasets?include_archived=true")
        assert str(ds.id) in {d["id"] for d in lst.json()}

    async def test_an_archived_set_never_trains(self, db):
        from app.routes.datasets import archive_dataset
        from app.training_plan import plan_training
        w = await _world(db)
        await archive_dataset(w["david"].id, _user=None, db=db)
        plan = await plan_training(db, TrainingCreate(**SOLO))
        assert "dataset_missing" in {p["code"] for p in plan.problems}
        chosen = TrainingCreate(**SOLO, datasets={"David": w["david"].id})
        plan = await plan_training(db, chosen)
        assert "dataset_archived" in {p["code"] for p in plan.problems}

    async def test_the_living_set_is_picked_beside_archived_versions(self, db):
        """After the backfill a subject owns one live set and its archived versions: the
        plan must pick the live one without asking."""
        from app.models import Dataset
        from app.training_plan import plan_training
        w = await _world(db)
        old = Dataset(id=uuid.uuid4(), name="David v0", prefix="datasets/v0",
                      kind="character", character="David", images=list(w["david"].images),
                      archived_at=datetime.now(timezone.utc))
        db.add(old)
        await db.flush()
        plan = await plan_training(db, TrainingCreate(**SOLO))
        assert plan.ok, plan.problems
        assert plan.groups[0].dataset.id == w["david"].id


class TestTrainedFilesSurvive:
    async def test_the_delete_gate_names_a_finished_runs_every_group(self, db):
        from app.routes.images import find_image_references
        g0, comp = await _ds(db), await _ds(db)
        job = await _run(db, g0, identities_ds=[comp], status=TrainingStatus.COMPLETED)
        refs = await find_image_references(db, [g0.images[0], comp.images[1]])
        assert refs[g0.images[0]]["training_ids"] == [str(job.id)]
        assert refs[comp.images[1]]["training_ids"] == [str(job.id)]

    async def test_a_move_copies_a_trained_file_and_the_set_follows(self, db, monkeypatch):
        from app.config import settings
        from app.models import Dataset
        from app.routes import images as mod
        monkeypatch.setattr(settings, "s3_images_bucket", "wanly-images")
        copied, moved = [], []
        monkeypatch.setattr(mod, "copy_object", lambda b, s, d: copied.append((s, d)))
        monkeypatch.setattr(mod, "move_object", lambda b, s, d: moved.append((s, d)))
        trained_uri = "s3://wanly-images/Zed/trained.jpg"
        free_uri = "s3://wanly-images/Zed/free.jpg"
        ds = await _ds(db, images=[trained_uri, free_uri], anchor_uri=trained_uri,
                       captions={trained_uri: "c"})
        from app.models import TrainingJob
        db.add(TrainingJob(character="Zed", trigger="z", version=1, status="completed",
                           dataset_images=[trained_uri], config={}))
        await db.commit()
        await mod.move_images({"keys": ["Zed/trained.jpg", "Zed/free.jpg"],
                               "target_folder": "Moved"}, db=db)
        assert copied == [("Zed/trained.jpg", "Moved/trained.jpg")]
        assert moved == [("Zed/free.jpg", "Moved/free.jpg")]
        row = await db.get(Dataset, ds.id)
        assert row.images == ["s3://wanly-images/Moved/trained.jpg",
                              "s3://wanly-images/Moved/free.jpg"]
        assert row.anchor_uri == "s3://wanly-images/Moved/trained.jpg"
        assert row.captions == {"s3://wanly-images/Moved/trained.jpg": "c"}


# ---------------------------------------------------------------------------------------
# The backfill (#424)
# ---------------------------------------------------------------------------------------


async def _subject(db, name, trigger, gender="woman"):
    from app.models import LtxCharacter
    db.add(LtxCharacter(name=name, trigger=trigger, gender=gender, char_lora="none"))
    await db.flush()


async def _set(db, name, owner, images, captions=None, kind="character", age_days=0, **kw):
    from app.models import Dataset
    d = Dataset(id=uuid.uuid4(), name=name, prefix=f"datasets/{name}", kind=kind,
                character=owner, images=images, captions=captions or {}, scores={},
                created_at=datetime.now(timezone.utc) - timedelta(days=age_days), **kw)
    db.add(d)
    await db.flush()
    return d


@pytest.fixture
def heads(monkeypatch):
    from app import s3
    missing: set[str] = set()
    etags: dict[str, str] = {}
    monkeypatch.setattr(s3, "head_object",
                        lambda u: None if u in missing else {"Key": u, "ETag": etags.get(u)})
    return missing, etags


class TestBackfill:
    async def _world(self, db):
        from app.models import TrainingJob
        tag = uuid.uuid4().hex[:6]
        subj = f"Zed{tag}"
        await _subject(db, subj, f"z{tag}")
        u = [f"s3://wanly-images/{subj}/{i:032x}.jpg" for i in range(5)]
        crop = f"s3://wanly-images/datasets/x/001_{1:032x}_f0.jpg"  # a crop of u[1]
        v1 = await _set(db, f"{subj} v1", subj, u[:3], {u[0]: "a", u[1]: "old"}, age_days=9)
        v2 = await _set(db, f"{subj} v2", subj, u[1:] + [crop], {u[1]: "new", u[4]: "d"},
                        age_days=1)
        solo = f"Solo{tag}"
        await _subject(db, solo, f"s{tag}")
        single = await _set(db, f"{solo} v1", solo, [f"s3://wanly-images/{solo}/a.jpg"])
        recorded = TrainingJob(character=subj, trigger=f"z{tag}", version=1,
                               status="completed", dataset_images=u[:3],
                               config={"dataset": {"id": str(v1.id), "name": v1.name},
                                       "captions": ["c0", "c1", "c2"]})
        early = TrainingJob(character=subj, trigger=f"z{tag}", version=2, status="completed",
                            dataset_images=u[1:4], config={"gender": "woman"})
        db.add_all([recorded, early])
        await db.commit()
        return dict(subj=subj, tag=tag, u=u, crop=crop, v1=v1, v2=v2, single=single,
                    solo=solo, recorded=recorded, early=early)

    async def test_a_dry_run_writes_nothing(self, db, heads):
        from sqlalchemy import func, select
        from app.backfill_living_datasets import render, run
        from app.models import Dataset, TrainingRunDataset
        w = await self._world(db)
        v1_id, subj = w["v1"].id, w["subj"]
        n_sets = (await db.execute(select(func.count()).select_from(Dataset))).scalar_one()
        rep = await run(apply=False, db=db)
        text = render(rep, apply=False)
        assert "DRY RUN" in text and subj in text
        assert (await db.execute(select(func.count()).select_from(Dataset))).scalar_one() \
            == n_sets
        assert (await db.execute(select(func.count()).select_from(TrainingRunDataset))
                ).scalar_one() == 0
        assert (await db.get(Dataset, v1_id)).archived_at is None

    async def test_apply_merges_archives_links_and_is_idempotent(self, db, heads):
        from sqlalchemy import select
        from app.backfill_living_datasets import run
        from app.models import Dataset, TrainingRunDataset
        w = await self._world(db)
        missing, _ = heads
        missing.add(w["u"][3])
        rep = await run(apply=True, db=db)

        living = (await db.execute(select(Dataset).where(Dataset.name == w["subj"])
                                   )).scalar_one()
        u = w["u"]
        assert living.images == u[:3] + u[3:] + [w["crop"]]
        assert living.archived_at is None and living.character == w["subj"]
        # newest non-empty caption wins, and the disagreement is reported
        assert living.captions[u[1]] == "new" and living.captions[u[0]] == "a"
        plan = next(p for p in rep.merges if p.subject == w["subj"])
        assert plan.duplicates == 2
        assert [c[0] for c in plan.caption_conflicts] == [u[1]]
        assert any(w["crop"] in pair for pair in plan.near_dups)
        for src in (w["v1"], w["v2"]):
            assert (await db.get(Dataset, src.id)).archived_at is not None
        # the single set is renamed off its version suffix
        assert (await db.get(Dataset, w["single"].id)).name == w["solo"]
        # the missing file is reported, not fatal
        assert any(m[1] == u[3] for m in rep.missing)

        rows = {r.training_job_id: r for r in (await db.execute(select(TrainingRunDataset)
                                                                )).scalars().all()}
        rec = rows[w["recorded"].id]
        assert rec.dataset_id == living.id and rec.dataset_name == w["v1"].name
        assert rec.captions == ["c0", "c1", "c2"] and rec.source == "backfill"
        early = rows[w["early"].id]
        assert early.dataset_id == living.id  # by image overlap, then to the living set
        assert early.captions == [f"z{w['tag']}, woman"] * 3

        again = await run(apply=True, db=db)
        assert not any(p.subject == w["subj"] for p in again.merges)
        assert again.links == [] and again.already_linked >= 2


async def test_migration_114_up_and_down(db_engine):
    """Down drops the table and the column, up puts them back. Rolled back afterwards."""
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect

    spec = importlib.util.spec_from_file_location(
        "m114", Path("alembic/versions/114_living_datasets.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.revision, m.down_revision) == ("114", "113")

    def _state(conn):
        i = inspect(conn)
        return ("training_run_datasets" in i.get_table_names(),
                "archived_at" in {c["name"] for c in i.get_columns("datasets")})

    def _run_both(conn):
        with Operations.context(MigrationContext.configure(conn)):
            m.downgrade()
            gone = _state(conn)
            m.upgrade()
            back = _state(conn)
        return gone, back

    async with db_engine.connect() as conn:
        trans = await conn.begin()
        try:
            gone, back = await conn.run_sync(_run_both)
        finally:
            await trans.rollback()
    assert gone == (False, False) and back == (True, True)


async def test_the_report_names_each_links_own_living_set(db, heads):
    """Two subjects merged in one pass: each run's link line names its own subject's set."""
    from app.backfill_living_datasets import render, run
    from app.models import TrainingJob
    tag = uuid.uuid4().hex[:6]
    lines = []
    for who in (f"Aa{tag}", f"Bb{tag}"):
        await _subject(db, who, who.lower())
        v1 = await _set(db, f"{who} v1", who, [f"s3://wanly-images/{who}/1.jpg"], age_days=2)
        await _set(db, f"{who} v2", who, [f"s3://wanly-images/{who}/2.jpg"])
        db.add(TrainingJob(character=who, trigger=who.lower(), version=1, status="completed",
                           dataset_images=list(v1.images),
                           config={"dataset": {"id": str(v1.id), "name": v1.name},
                                   "captions": ["c"]}))
        lines.append(who)
    await db.commit()
    text = render(await run(apply=False, db=db), apply=False)
    for who in lines:
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{who} v1 ltx"))
        assert f"-> '{who}'" in line, line
