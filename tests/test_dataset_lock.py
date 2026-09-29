"""A dataset that trained a LoRA is locked; a clone is how it changes (wanly-api#356).

A set edited after its LoRA trained leaves the LoRA's record pointing at a dataset that no
longer holds what it learned from. So once a run that is not failed or cancelled has used a
set -- group 0's `config.dataset.id`, or any `identities[].dataset.id` -- every mutation of
its training data is a 409 naming the run, and POST /datasets/{id}/clone makes an unlocked
copy that shares the same image URIs.

Sharing URIs is only safe if nothing that edits one set can delete an object the other
still lists. The S3 tests below run the real app.s3 functions against an in-memory bucket,
so they prove what reaches the bucket, not what a route meant to send.
"""
import io
import uuid

import pytest

from app.enums import TrainingStatus

BUCKET = "wanly-images"


def _imgs(prefix: str, n: int = 4) -> list[str]:
    return [f"s3://{BUCKET}/{prefix}/{i}.jpg" for i in range(n)]


async def _ds(db, **kw):
    from app.models import Dataset
    ds_id = uuid.uuid4()
    prefix = f"datasets/{ds_id}"
    base = dict(id=ds_id, name=f"lock-{uuid.uuid4().hex[:6]}", images=_imgs(prefix),
                prefix=prefix, captions={}, scores={})
    base.update(kw)
    d = Dataset(**base)
    db.add(d)
    await db.commit()
    return d


async def _run(db, ds=None, *, status=TrainingStatus.COMPLETED, identities_ds=(),
               character="Kelly-2000", version=None):
    """A training job that used `ds` as group 0 and/or `identities_ds` as extra groups."""
    from app.models import TrainingJob
    job = TrainingJob(
        character=character, trigger="k3lly", version=version or _run.v,
        dataset_images=list(ds.images) if ds else [],
        status=status,
        config={"dataset": {"id": str(ds.id) if ds else None,
                            "name": ds.name if ds else None, "count": 4}},
        identities=[{"kind": "composition", "images": list(g.images), "captions": [],
                     "dataset": {"id": str(g.id), "name": g.name, "count": len(g.images)}}
                    for g in identities_ds] or None,
    )
    _run.v += 1
    db.add(job)
    await db.commit()
    return job


_run.v = 1000


async def _patch(db, ds, **fields):
    from app.routes.datasets import update_dataset
    from app.schemas.datasets import DatasetUpdate
    return await update_dataset(ds.id, DatasetUpdate(**fields), _user=None, db=db)


class _Usr:
    id = None


class _Upload:
    filename = "new.jpg"

    async def read(self):
        return b"new bytes"


class _FakeBucket:
    """A boto3 client over a dict, installed as app.s3's cached client -- so the real
    upload/download/delete_prefix_except code runs, and every delete is recorded."""

    def __init__(self, uris=()):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[str] = []
        for u in uris:
            b, k = u[len("s3://"):].split("/", 1)
            self.objects[(b, k)] = b"x"

    def uris(self) -> set[str]:
        return {f"s3://{b}/{k}" for b, k in self.objects}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix, **_):
        return {"Contents": [{"Key": k} for b, k in sorted(self.objects)
                             if b == Bucket and k.startswith(Prefix)],
                "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop((Bucket, o["Key"]), None)
            self.deleted.append(f"s3://{Bucket}/{o['Key']}")

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        self.deleted.append(f"s3://{Bucket}/{Key}")


@pytest.fixture
def bucket(monkeypatch):
    from app import s3
    from app.config import settings
    monkeypatch.setattr(settings, "s3_images_bucket", BUCKET)
    fake = _FakeBucket()
    monkeypatch.setattr(s3, "_client", fake)
    return fake


@pytest.mark.asyncio
class TestWhatLocks:
    async def test_a_set_nobody_trained_on_is_unlocked(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is False and out.trained_by == []

    @pytest.mark.parametrize("status", [TrainingStatus.PENDING, TrainingStatus.CLAIMED,
                                        TrainingStatus.RUNNING, TrainingStatus.COMPLETED])
    async def test_a_queued_running_or_finished_run_locks_its_set(self, db, status):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        job = await _run(db, ds, status=status, version=5)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is True
        assert [t.model_dump() for t in out.trained_by] == [
            {"job_id": str(job.id), "character": "Kelly-2000", "version": 5,
             "status": str(status), "created_at": job.created_at}]

    @pytest.mark.parametrize("status", [TrainingStatus.FAILED, TrainingStatus.CANCELLED])
    async def test_a_failed_or_cancelled_run_leaves_it_editable(self, db, status):
        """No LoRA came of it, so there is no record for an edit to falsify."""
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds, status=status)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is False and out.trained_by == []
        # ...and the mutations the lock refuses go through.
        assert (await _patch(db, ds, images=ds.images[:2])).images == ds.images[:2]

    async def test_a_failing_run_unlocks_the_set(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        job = await _run(db, ds, status=TrainingStatus.RUNNING)
        assert (await get_dataset(ds.id, db=db)).locked
        job.status = TrainingStatus.FAILED
        await db.commit()
        assert not (await get_dataset(ds.id, db=db)).locked

    async def test_an_identity_group_locks_its_set_too(self, db):
        """A joint run's second member, composition set or regularization pool lives in
        `identities`, not group 0 -- and trained the LoRA just the same."""
        from app.routes.datasets import get_dataset
        g0, comp, other = await _ds(db), await _ds(db), await _ds(db)
        job = await _run(db, g0, identities_ds=[comp])
        assert (await get_dataset(comp.id, db=db)).trained_by[0].job_id == str(job.id)
        assert (await get_dataset(g0.id, db=db)).locked
        assert not (await get_dataset(other.id, db=db)).locked

    async def test_trained_by_lists_every_locking_run_and_only_those(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        a = await _run(db, ds, version=4)
        await _run(db, ds, version=5, status=TrainingStatus.FAILED)
        b = await _run(db, None, identities_ds=[ds], version=6, status=TrainingStatus.PENDING,
                       character="DavidKelly-2026")
        out = await get_dataset(ds.id, db=db)
        assert [(t.job_id, t.character, t.version, t.status) for t in out.trained_by] == [
            (str(a.id), "Kelly-2000", 4, "completed"),
            (str(b.id), "DavidKelly-2026", 6, "pending")]

    async def test_a_clone_does_not_inherit_the_lock_by_sharing_uris(self, db):
        """Locking reads recorded provenance, never URI overlap."""
        from app.routes.datasets import clone_dataset, get_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db)
        await _run(db, src)
        c = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                user=_Usr(), db=db)
        assert c.locked is False and c.images == src.images
        assert (await get_dataset(c.id, db=db)).locked is False

    async def test_the_list_fills_the_lock_for_every_set(self, db):
        from sqlalchemy import event
        from app.routes.datasets import list_datasets
        locked, free = await _ds(db), await _ds(db)
        await _run(db, locked)
        await _run(db, None, identities_ds=[locked])

        statements = []
        sync_engine = db.bind.sync_engine if hasattr(db.bind, "sync_engine") else db.bind
        listen = lambda *a: statements.append(a[2])  # noqa: E731
        event.listen(sync_engine, "before_cursor_execute", listen)
        try:
            rows = await list_datasets(db=db)
        finally:
            event.remove(sync_engine, "before_cursor_execute", listen)
        by_id = {r.id: r for r in rows}
        assert by_id[locked.id].locked and len(by_id[locked.id].trained_by) == 2
        assert not by_id[free.id].locked and by_id[free.id].trained_by == []
        assert len(statements) == 2, "one query for the sets, one for the runs -- not N+1"


@pytest.mark.asyncio
class TestRefusedWhileLocked:
    """Everything that changes what a run would train on is a 409 naming the run."""

    async def _locked(self, db, **kw):
        ds = await _ds(db, name=f"Kelly 2000 v5 {uuid.uuid4().hex[:4]}", **kw)
        await _run(db, ds, version=5)
        return ds

    @staticmethod
    def _is_lock(e, ds):
        assert e.value.status_code == 409
        assert e.value.detail.startswith(f"{ds.name!r} trained Kelly-2000 v5 and is locked")
        assert "clone it" in e.value.detail

    async def test_removing_or_reordering_images(self, db):
        from fastapi import HTTPException
        ds = await self._locked(db)
        for images in (ds.images[1:], list(reversed(ds.images)), ds.images + ["s3://x/y.jpg"]):
            with pytest.raises(HTTPException) as e:
                await _patch(db, ds, images=images)
            self._is_lock(e, ds)

    async def test_a_patch_that_also_renames_is_refused_whole(self, db):
        from fastapi import HTTPException
        ds = await self._locked(db)
        name = ds.name
        with pytest.raises(HTTPException):
            await _patch(db, ds, name="renamed-" + uuid.uuid4().hex[:6], images=ds.images[1:])
        await db.refresh(ds)
        assert ds.name == name and len(ds.images) == 4

    async def test_changing_a_pools_class(self, db):
        from fastapi import HTTPException
        ds = await self._locked(db, kind="regularization", reg_class="woman")
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, reg_class="man")
        self._is_lock(e, ds)

    async def test_changing_the_owner_or_kind(self, db):
        """Composition, because its owner need not be registered -- so what refuses these is
        the lock, not ownership validation."""
        from fastapi import HTTPException
        ds = await self._locked(db, kind="composition", character="KellyPair-2000")
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, character="OtherPair-2000")
        self._is_lock(e, ds)
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, kind=None)
        self._is_lock(e, ds)

    async def test_uploading_images(self, db, bucket):
        from fastapi import HTTPException
        from app.routes.datasets import add_images
        ds = await self._locked(db)
        with pytest.raises(HTTPException) as e:
            await add_images(ds.id, files=[_Upload()], _user=None, db=db)
        self._is_lock(e, ds)
        assert bucket.objects == {}, "refused before anything was uploaded"

    async def test_cropping(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import crop_faces
        ds = await self._locked(db)
        for save_as in (False, True):
            with pytest.raises(HTTPException) as e:
                await crop_faces(ds.id, largest_only=False, uris=None, save_as=save_as,
                                 _user=None, db=db)
            self._is_lock(e, ds)

    async def test_generating_captions(self, db):
        from fastapi import BackgroundTasks, HTTPException
        from app.routes.datasets import caption_dataset
        ds = await self._locked(db)
        tasks = BackgroundTasks()
        with pytest.raises(HTTPException) as e:
            await caption_dataset(ds.id, tasks, body=None, _user=None, db=db)
        self._is_lock(e, ds)
        assert tasks.tasks == []

    async def test_editing_a_caption(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import edit_dataset_caption
        from app.schemas.datasets import DatasetCaptionEdit
        ds = await self._locked(db)
        with pytest.raises(HTTPException) as e:
            await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri=ds.images[0], caption="x"),
                                       _user=None, db=db)
        self._is_lock(e, ds)

    async def test_a_caption_run_stops_when_the_set_locks_under_it(self, db, monkeypatch):
        """The Train button does not wait for captioning to finish."""
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        ds = await _ds(db)
        calls = []

        async def _caption(db_, image, style=None, instruction=None, interactive=True):
            calls.append(1)
            if len(calls) == 2:
                await _run(db, ds, status=TrainingStatus.PENDING)
            return f"caption {len(calls)}", instruction

        class _S3:
            def download_bytes(self, uri):
                return b"x"
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _S3())
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 1 and len(ds.captions) == 1
        assert "is locked" in mod._CAPTION_RUNS.pop(ds.id)["error"]

    async def test_regularizing(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import regularize_dataset
        from app.schemas.datasets import DatasetRegularize
        ds = await self._locked(db, kind="regularization", reg_class="woman")
        with pytest.raises(HTTPException) as e:
            await regularize_dataset(ds.id, DatasetRegularize(count=3), user=_Usr(), db=db)
        self._is_lock(e, ds)

    async def test_a_locked_pool_does_not_collect_late_renders(self, db, bucket):
        """Renders queued before the pool trained can finish after it."""
        from app.enums import JobStatus, SegmentStatus
        from app.models import Job, Segment, User
        from app.regularization import reg_tag
        from app.routes.datasets import regularize_status
        user = User(username=f"u{uuid.uuid4().hex[:6]}", password_hash="x")
        db.add(user)
        await db.flush()
        ds = await _ds(db, images=[], kind="regularization", reg_class="man")
        j = Job(user_id=user.id, name="r", width=832, height=1216, fps=24, seed=1,
                tags=f"regularization, {reg_tag(ds.id)}", status=JobStatus.PROCESSING)
        db.add(j)
        await db.flush()
        frame = f"s3://wanly-jobs/{j.id}/last_frame.png"
        bucket.objects[("wanly-jobs", f"{j.id}/last_frame.png")] = b"png"
        db.add(Segment(job_id=j.id, index=0, prompt="p", status=SegmentStatus.COMPLETED,
                       last_frame_path=frame))
        await db.commit()
        run = await _run(db, None, identities_ds=[ds], status=TrainingStatus.RUNNING)

        held = await regularize_status(ds.id, _user=None, db=db)
        await db.refresh(ds)
        assert (held.collected_now, held.running, ds.images) == (0, 1, [])
        assert j.status == JobStatus.PROCESSING

        run.status = TrainingStatus.CANCELLED
        await db.commit()
        got = await regularize_status(ds.id, _user=None, db=db)
        assert got.collected_now == 1

    async def test_deleting_with_or_without_purge(self, db, bucket):
        from fastapi import HTTPException
        from app.models import Dataset
        from app.routes.datasets import delete_dataset
        ds = await self._locked(db)
        for purge in (False, True):
            with pytest.raises(HTTPException) as e:
                await delete_dataset(ds.id, purge=purge, _user=None, db=db)
            self._is_lock(e, ds)
        assert await db.get(Dataset, ds.id) is not None
        assert bucket.deleted == []

    async def test_the_refusal_over_http(self, db):
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user, verify_api_key_or_bearer
        from app.database import get_db
        from app.main import app
        ds = await self._locked(db)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _Usr()
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.patch(f"/datasets/{ds.id}", json={"images": ds.images[:1]})
                g = await c.get(f"/datasets/{ds.id}")
                lst = await c.get("/datasets")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 409 and "trained Kelly-2000 v5" in r.json()["detail"]
        body = g.json()
        assert body["locked"] is True
        assert body["trained_by"][0]["character"] == "Kelly-2000"
        assert set(body["trained_by"][0]) == {"job_id", "character", "version", "status",
                                            "created_at"}
        assert next(d for d in lst.json() if d["id"] == str(ds.id))["locked"] is True


@pytest.mark.asyncio
class TestAllowedWhileLocked:
    """Rename, notes, tags, anchor and scoring do not change the training data."""

    async def _locked(self, db):
        ds = await _ds(db)
        await _run(db, ds)
        return ds

    async def test_rename_notes_and_tags(self, db):
        ds = await self._locked(db)
        name = f"renamed-{uuid.uuid4().hex[:6]}"
        out = await _patch(db, ds, name=name, notes="kept for v5", tags="a, b")
        assert (out.name, out.notes, out.tags) == (name, "kept for v5", "a, b")
        assert out.locked is True and out.trained_by

    async def test_sending_the_same_list_back_is_not_a_change(self, db):
        """A form that PATCHes every field with a new name must still work."""
        ds = await self._locked(db)
        out = await _patch(db, ds, name=f"n-{uuid.uuid4().hex[:6]}", images=list(ds.images))
        assert out.images == ds.images

    async def test_the_same_owner_resent_is_not_a_change(self, db):
        from app.models import LtxCharacter
        who = f"Kel-{uuid.uuid4().hex[:6]}"
        db.add(LtxCharacter(name=who, trigger="k", gender="woman", char_lora="none"))
        ds = await _ds(db, kind="character", character=who)
        await _run(db, ds)
        out = await _patch(db, ds, kind="character", character=who.lower())
        assert out.character == who

    async def test_the_anchor(self, db):
        ds = await self._locked(db)
        out = await _patch(db, ds, anchor_uri=ds.images[1])
        assert out.anchor_uri == ds.images[1]
        assert (await _patch(db, ds, anchor_uri="")).anchor_uri is None

    async def test_scoring(self, db, monkeypatch):
        from app.config import settings
        from app.routes import datasets as mod
        ds = await self._locked(db)
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")

        async def _embed(uris):
            return [[1.0, 0.0] for _ in uris]
        monkeypatch.setattr(mod, "_embed_all", _embed)
        out = await mod.score_against_anchor(ds.id, anchor_uri=ds.images[0], _user=None, db=db)
        assert len(out.scores) == 4
        await db.refresh(ds)
        assert ds.anchor_uri == ds.images[0] and len(ds.scores) == 4

    async def test_cloning(self, db):
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        ds = await self._locked(db)
        out = await clone_dataset(ds.id, DatasetClone(name=f"v6-{uuid.uuid4().hex[:6]}"),
                                  user=_Usr(), db=db)
        assert out.locked is False


@pytest.mark.asyncio
class TestClone:
    async def test_it_copies_everything_but_identity_and_prefix(self, db):
        from app.models import Dataset
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db, tags="kelly, v5", notes="culled twice",
                        kind="composition", character="KellyPair-2000")
        src.captions = {src.images[0]: "close-up", src.images[2]: "outdoors"}
        src.scores = {u: 0.5 for u in src.images}
        src.anchor_uri = src.images[1]
        await db.commit()
        await _run(db, src)

        name = f"Kelly 2000 v6 {uuid.uuid4().hex[:4]}"
        out = await clone_dataset(src.id, DatasetClone(name=f"  {name} "), user=_Usr(), db=db)
        row = await db.get(Dataset, out.id)
        assert out.id != src.id and out.name == name
        assert row.prefix == f"datasets/{out.id}" != src.prefix
        assert row.images == src.images, "the same URIs, not copies"
        assert (row.captions, row.scores, row.anchor_uri) == (
            src.captions, src.scores, src.anchor_uri)
        assert (row.kind, row.character, row.reg_class, row.tags) == (
            "composition", "KellyPair-2000", None, "kelly, v5")
        assert row.notes == f"Cloned from {src.name!r}. culled twice"
        assert out.locked is False and out.trained_by == []

    async def test_a_source_without_notes(self, db):
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db)
        out = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                  user=_Usr(), db=db)
        assert out.notes == f"Cloned from {src.name!r}."

    async def test_the_clone_is_editable_and_the_source_is_untouched(self, db):
        from app.models import Dataset
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db, captions={})
        await _run(db, src)
        c = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                user=_Usr(), db=db)
        clone = await db.get(Dataset, c.id)
        out = await _patch(db, clone, images=clone.images[:2])
        assert out.images == src.images[:2]
        await db.refresh(src)
        assert len(src.images) == 4

    async def test_the_name_must_be_unique(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src, other = await _ds(db), await _ds(db)
        for taken in (src.name, other.name):
            with pytest.raises(HTTPException) as e:
                await clone_dataset(src.id, DatasetClone(name=taken), user=_Usr(), db=db)
            assert e.value.status_code == 409

    def test_the_name_is_validated_like_a_new_sets(self):
        from pydantic import ValidationError
        from app.schemas.datasets import DatasetClone
        for bad in ("", "a/b", "../x"):
            with pytest.raises(ValidationError):
                DatasetClone(name=bad)

    async def test_a_missing_source_is_404(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        with pytest.raises(HTTPException) as e:
            await clone_dataset(uuid.uuid4(), DatasetClone(name="x"), user=_Usr(), db=db)
        assert e.value.status_code == 404

    async def test_over_http(self, db):
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user
        from app.database import get_db
        from app.main import app
        src = await _ds(db)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _Usr()
        name = f"c-{uuid.uuid4().hex[:6]}"
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(f"/datasets/{src.id}/clone", json={"name": name})
                dup = await c.post(f"/datasets/{src.id}/clone", json={"name": name})
                bad = await c.post(f"/datasets/{src.id}/clone", json={"name": "a/b"})
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 201, r.text
        assert r.json()["name"] == name and r.json()["locked"] is False
        assert r.json()["trained_by"] == [] and r.json()["images"] == src.images
        assert dup.status_code == 409
        assert bad.status_code == 422


@pytest.mark.asyncio
class TestSharedUrisAreSafe:
    """A clone lists objects under its source's prefix. Nothing either side does may delete
    an object the other still lists."""

    async def _pair(self, db, bucket):
        from app.models import Dataset
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db)
        # One object under the source's prefix that only the source ever listed, and it is
        # not even in its list any more -- a purge is for exactly this.
        orphan = f"s3://{BUCKET}/{src.prefix}/removed-long-ago.jpg"
        for u in src.images + [orphan]:
            b, k = u[len("s3://"):].split("/", 1)
            bucket.objects[(b, k)] = b"x"
        c = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                user=_Usr(), db=db)
        return src, await db.get(Dataset, c.id), orphan

    async def test_removing_an_image_never_deletes_the_object(self, db, bucket):
        src, clone, _ = await self._pair(db, bucket)
        await _patch(db, clone, images=[])
        await _patch(db, src, images=src.images[:1])
        assert bucket.deleted == []

    async def test_cropping_in_place_never_deletes_the_photographs(self, db, bucket, monkeypatch):
        """The crops land under the CLONE's prefix; the photographs, under the source's, are
        only dropped from the clone's list."""
        from app.config import settings
        from app.routes import datasets as mod
        from tests.test_datasets import _HttpxShim, _crop_client
        src, clone, _ = await self._pair(db, bucket)
        before = bucket.uris()
        monkeypatch.setattr(mod, "httpx", _HttpxShim(_crop_client(None)))
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")
        out = await mod.crop_faces(clone.id, largest_only=False, uris=None, save_as=False,
                                   _user=None, db=db)
        assert bucket.deleted == []
        assert before <= bucket.uris()
        assert all(u.startswith(f"s3://{BUCKET}/{clone.prefix}/faces-") for u in out.images)

    async def test_purging_the_clone_leaves_the_source_whole(self, db, bucket):
        from app.models import Dataset
        from app.routes.datasets import add_images, delete_dataset
        src, clone, orphan = await self._pair(db, bucket)
        await add_images(clone.id, files=[_Upload()], _user=None, db=db)
        own = f"s3://{BUCKET}/{clone.prefix}/new.jpg"
        assert own in bucket.uris()

        await delete_dataset(clone.id, purge=True, _user=None, db=db)
        assert bucket.deleted == [own], "only the clone's own prefix"
        assert set(src.images) | {orphan} <= bucket.uris()
        assert await db.get(Dataset, src.id) is not None

    async def test_purging_the_source_keeps_what_the_clone_lists(self, db, bucket):
        from app.routes.datasets import delete_dataset
        src, clone, orphan = await self._pair(db, bucket)
        await _patch(db, clone, images=clone.images[:3])
        shared, dropped = clone.images[:3], src.images[3]

        await delete_dataset(src.id, purge=True, _user=None, db=db)
        assert set(shared) <= bucket.uris(), "the clone still lists these"
        assert sorted(bucket.deleted) == sorted([dropped, orphan])

    async def test_a_purge_stays_inside_its_prefix(self, db, bucket):
        """`datasets/<id>/`, with the slash: a sibling whose key merely starts with the same
        characters is somebody else's."""
        from app.routes.datasets import delete_dataset
        ds = await _ds(db, images=[])
        sibling = f"s3://{BUCKET}/{ds.prefix}-other/a.jpg"
        mine = f"s3://{BUCKET}/{ds.prefix}/a.jpg"
        for u in (sibling, mine):
            b, k = u[len("s3://"):].split("/", 1)
            bucket.objects[(b, k)] = b"x"
        await delete_dataset(ds.id, purge=True, _user=None, db=db)
        assert bucket.deleted == [mine]
