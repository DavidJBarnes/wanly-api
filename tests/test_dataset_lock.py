"""A set that trained a LoRA stays living (wanly-api#420, retiring #356's lock).

Each run records what it trained on (training_run_datasets, #422), so the dataset is the
subject's living set: edits go through, `trained_by` and `used_in` only inform, and no file a
run trained on is ever deleted or overwritten (#421). Clone stays, for a separate set.

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
class TestATrainedSetStaysLiving:
    """#420: training no longer locks a set. Every run records what it trained on, so the
    set is the subject's living set -- each of these was a 409 under #356."""

    async def _trained(self, db, **kw):
        ds = await _ds(db, name=f"Kelly 2000 v5 {uuid.uuid4().hex[:4]}", **kw)
        await _run(db, ds, version=5)
        return ds

    @pytest.mark.parametrize("status", [TrainingStatus.PENDING, TrainingStatus.CLAIMED,
                                        TrainingStatus.RUNNING, TrainingStatus.COMPLETED])
    async def test_any_run_is_listed_but_locks_nothing(self, db, status):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        job = await _run(db, ds, status=status, version=5)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is False
        assert [t.model_dump() for t in out.trained_by] == [
            {"job_id": str(job.id), "character": "Kelly-2000", "version": 5,
             "status": str(status), "arch": "ltx", "created_at": job.created_at}]

    @pytest.mark.parametrize("status", [TrainingStatus.FAILED, TrainingStatus.CANCELLED])
    async def test_a_failed_or_cancelled_run_is_not_listed(self, db, status):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds, status=status)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is False and out.trained_by == []

    async def test_an_identity_group_lists_its_set_too(self, db):
        from app.routes.datasets import get_dataset
        g0, comp, other = await _ds(db), await _ds(db), await _ds(db)
        job = await _run(db, g0, identities_ds=[comp])
        assert (await get_dataset(comp.id, db=db)).trained_by[0].job_id == str(job.id)
        assert (await get_dataset(other.id, db=db)).trained_by == []

    async def test_used_in_badges_each_image_its_runs(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        a = await _run(db, ds, version=4)
        await _patch(db, ds, images=ds.images[:2] + [ds.images[3]])
        out = await get_dataset(ds.id, db=db)
        assert set(out.used_in) == set(out.images)
        assert [r.job_id for r in out.used_in[ds.images[0]]] == [str(a.id)]

    async def test_removing_reordering_and_renaming_go_through(self, db):
        ds = await self._trained(db)
        want = list(reversed(ds.images[1:]))
        out = await _patch(db, ds, name=f"n-{uuid.uuid4().hex[:6]}", images=want)
        assert out.images == want and out.locked is False

    async def test_changing_the_owner_or_class(self, db):
        ds = await self._trained(db, kind="regularization", reg_class="woman")
        assert (await _patch(db, ds, reg_class="man")).reg_class == "man"

    async def test_uploading_into_a_trained_set(self, db, bucket):
        from app.routes.datasets import add_images
        ds = await self._trained(db)
        out = await add_images(ds.id, files=[_Upload()], _user=None, db=db)
        assert len(out.images) == 5

    async def test_reuploading_a_trained_file_never_overwrites_it(self, db, bucket):
        """#421: the bytes under a run's recorded URI stay the bytes it learned from."""
        from app.routes.datasets import add_images

        class _Same(_Upload):
            filename = "0.jpg"
        ds = await self._trained(db)
        before = list(ds.images)
        trained = ds.images[0]
        b, k = trained[len("s3://"):].split("/", 1)
        bucket.objects[(b, k)] = b"original"
        out = await add_images(ds.id, files=[_Same()], _user=None, db=db)
        assert bucket.objects[(b, k)] == b"original"
        fresh = [u for u in out.images if u not in before]
        assert len(fresh) == 1 and fresh[0].rsplit("/", 1)[1].startswith("0-")

    async def test_captions_edit(self, db):
        from app.routes.datasets import edit_dataset_caption
        from app.schemas.datasets import DatasetCaptionEdit
        ds = await self._trained(db)
        out = await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri=ds.images[0], caption="x"),
                                         _user=None, db=db)
        assert out.captions[ds.images[0]] == "x"

    async def test_a_caption_run_does_not_stop_when_a_run_is_queued(self, db, monkeypatch):
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

            def head_object(self, uri):
                return {"Key": uri}
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _S3())
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        assert n == 4
        mod._CAPTION_RUNS.pop(ds.id, None)

    async def test_deleting_a_trained_set_keeps_every_trained_file(self, db, bucket):
        """#421: the row may go -- the run keeps its own record -- but a purge never deletes
        a file any run trained on."""
        from app.models import Dataset
        from app.routes.datasets import delete_dataset
        ds = await self._trained(db)
        for u in ds.images:
            b, k = u[len("s3://"):].split("/", 1)
            bucket.objects[(b, k)] = b"x"
        k = f"{ds.prefix}/untrained.jpg"
        bucket.objects[(BUCKET, k)] = b"x"
        ds_id = ds.id
        await delete_dataset(ds_id, purge=True, _user=None, db=db)
        assert await db.get(Dataset, ds_id) is None
        assert bucket.deleted == [f"s3://{BUCKET}/{k}"]

    async def test_over_http(self, db):
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user, verify_api_key_or_bearer
        from app.database import get_db
        from app.main import app
        ds = await self._trained(db)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _Usr()
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.patch(f"/datasets/{ds.id}", json={"images": ds.images[:1]})
                g = await c.get(f"/datasets/{ds.id}")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200
        body = g.json()
        assert body["locked"] is False and body["trained_by"][0]["arch"] == "ltx"
        assert body["used_in"][ds.images[0]][0]["version"] == 5
        assert set(body["trained_by"][0]) == {"job_id", "character", "version", "status",
                                            "arch", "created_at"}

    async def test_the_list_is_not_n_plus_one(self, db):
        from sqlalchemy import event
        from app.routes.datasets import list_datasets

        async def count():
            statements = []
            sync_engine = db.bind.sync_engine if hasattr(db.bind, "sync_engine") else db.bind
            listen = lambda *a: statements.append(a[2])  # noqa: E731
            event.listen(sync_engine, "before_cursor_execute", listen)
            try:
                await list_datasets(include_archived=False, db=db)
            finally:
                event.remove(sync_engine, "before_cursor_execute", listen)
            return len(statements)
        await _run(db, await _ds(db))
        before = await count()
        for _ in range(3):
            await _run(db, await _ds(db))
        assert await count() == before, "the query count must not grow with sets or runs"


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
