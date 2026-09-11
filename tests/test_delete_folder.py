"""Deleting an image folder warns that everything in it is unrecoverable (wanly-api#311).

The gate is the same one DELETE /images uses — jobs, segments, archived jobs, datasets —
applied to every image in the folder at once: one reference check and one batch S3 delete
beat N x per-image calls on a folder of hundreds. The datasets/ prefix is not deletable
here; dataset removal is the Datasets page's job (wanly-console#464's split).
"""

from unittest.mock import patch

import pytest

from app.enums import JobStatus
from app.models import Job

BUCKET = "wanly-images"
FOLDER = "2026-09-05"
PREFIX = f"{FOLDER}/"
IMG_A = f"s3://{BUCKET}/{FOLDER}/00001.png"
IMG_B = f"s3://{BUCKET}/{FOLDER}/00002.png"
ELSEWHERE = f"s3://{BUCKET}/2026-09-06/00003.png"
DATASET_FOLDER = "datasets/faces-abc"

_KEYS = {IMG_A, IMG_B, ELSEWHERE, f"s3://{BUCKET}/datasets/faces-abc/staging.png"}


def _fake_s3():
    def list_objects(bucket, prefix):
        return [
            {"Key": k[len(f"s3://{bucket}/"):], "Size": 1024, "LastModified": "2026-09-05T00:00:00Z"}
            for k in sorted(_KEYS) if k.startswith(f"s3://{bucket}/{prefix}")
        ]

    def delete_prefix(prefix, bucket):
        return sum(1 for k in _KEYS if k.startswith(f"s3://{bucket}/{prefix}"))

    return list_objects, delete_prefix


class _FakeSession:
    """Same shape as the delete-refs fake: answers find_image_references' SELECTs."""

    def __init__(self, jobs=(), segments=(), datasets=()):
        self._rows = {"jobs": list(jobs), "segments": list(segments),
                      "datasets": list(datasets)}

    async def execute(self, query):
        table = query.get_final_froms()[0].name
        columns = list(query.selected_columns.keys())
        wanted: set[str] = set()
        for value in query.compile().params.values():
            if isinstance(value, (list, tuple)):
                wanted.update(v for v in value if isinstance(v, str))
            elif isinstance(value, str):
                wanted.add(value)
        rows = []
        for obj in self._rows.get(table, []):
            values = tuple(getattr(obj, name, None) for name in columns)
            if table == "datasets" or any(
                    isinstance(v, str) and v in wanted for v in values):
                rows.append(values)
        return _FakeResult(rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


def _override(session):
    from app.auth import get_current_user, verify_api_key_or_bearer
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: type("Me", (), {"id": "u"})()
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    app.dependency_overrides[get_db] = lambda: session


async def _delete_folder(name, force=None):
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    list_objects, delete_prefix = _fake_s3()
    params = {"name": name}
    if force is not None:
        params["force"] = force
    with patch("app.routes.images.list_objects", list_objects), \
         patch("app.routes.images.delete_prefix", delete_prefix):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/images/folder", params=params)
    return resp


class TestFolderDelete:
    def teardown_method(self):
        from app.main import app
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_an_unreferenced_folder_deletes_everything(self):
        _override(_FakeSession())
        resp = await _delete_folder(FOLDER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["deleted"] == 2, body  # both folder images, nothing outside

    @pytest.mark.asyncio
    async def test_a_referenced_image_refuses_the_folder_and_names_what_dangles(self):
        """The warning the console pops has to say exactly what will dangle: one batch check
        names every holder, the same gate DELETE /images uses."""
        job = Job(id=__import__("uuid").uuid4(), name="j", width=832, height=480, fps=16,
                  seed=1, starting_image=IMG_A, status=JobStatus.PENDING)
        _override(_FakeSession(jobs=[job]))
        resp = await _delete_folder(FOLDER)
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["image_count"] == 2
        assert detail["referenced_count"] == 1
        assert str(job.id) in detail["paths"][IMG_A]["job_ids"]

    @pytest.mark.asyncio
    async def test_force_deletes_a_referenced_folder(self):
        """Everything in the directory is deleted and unrecoverable — that is what force
        confirms, same escape as DELETE /images."""
        import uuid
        job = Job(id=uuid.uuid4(), name="j", width=832, height=480, fps=16,
                  seed=1, starting_image=IMG_A, status=JobStatus.PENDING)
        _override(_FakeSession(jobs=[job]))
        resp = await _delete_folder(FOLDER, force="true")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2

    @pytest.mark.asyncio
    async def test_a_dataset_folder_is_not_deletable_here(self):
        """Dataset removal is the Datasets page's job; a folder delete that could reach
        training input would make the repo split (wanly-console#464) lie."""
        _override(_FakeSession())
        resp = await _delete_folder(DATASET_FOLDER)
        assert resp.status_code == 400
        assert "Datasets" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_an_empty_folder_is_a_404(self):
        """An unfiltered delete of nothing is a miss, not a success."""
        _override(_FakeSession())
        resp = await _delete_folder("2026-01-01")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_images_outside_the_folder_survive(self):
        """The batch delete is prefix-scoped: a fire-the-admin moment would be unrecoverable
        in the wrong way.

        _delete_folder supplies the delete_prefix fake; the assertion is via the fake's own
        return (the count is per-prefix), plus the fact that no other prefix's images could
        be in a prefix-scoped listing.
        """
        _override(_FakeSession())
        # The fake's keys span two folders; the response's count says what the prefix
        # delete claimed.
        resp = await _delete_folder(FOLDER)
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2  # the folder's two images, not the other folder's
