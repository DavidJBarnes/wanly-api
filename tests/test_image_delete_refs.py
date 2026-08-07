"""Tests for refusing to delete an image that is still referenced.

DELETE /images used to check only that the path was in the images bucket. Deleting a
referenced image looked fine and stayed fine until a worker claimed the segment — often weeks
later — and S3 answered 404, burning a pickup and surfacing as a red segment nowhere near the
cause. The advisory in_use flag on the listing endpoint did not help: it looked at
Job.starting_image only and skipped ARCHIVED jobs, so it reported "unused" for images that
were very much in use, and the UI is the thing that gets believed.

Both those blind spots get a test here, plus the force escape hatch and the case that must
still delete. Unit-level (no live DB), matching the rest of this suite: _FakeSession answers
the helper's SELECTs from in-memory ORM objects.
"""

import uuid
from unittest.mock import patch

import pytest

from app.auth import get_current_user
from app.database import get_db
from app.enums import JobStatus
from app.main import app
from app.models import Job, Segment, User

_fake_user = User(id=uuid.uuid4(), username="testuser", password_hash="x")

BUCKET = "wanly-images"
FACE = f"s3://{BUCKET}/2026-07-09/00054-swapped.png"
START = f"s3://{BUCKET}/2026-07-09/00055-start.png"
LOOSE = f"s3://{BUCKET}/2026-07-09/00056-unused.png"


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    """Stands in for AsyncSession, answering find_image_references' SELECTs from ORM objects.

    It reads the selected columns and the bound IN values off the statement rather than
    matching a hardcoded query, so it does not have to be rewritten every time the helper's
    column list widens — which is exactly the change this ticket is about.
    """

    def __init__(self, jobs=(), segments=()):
        self._rows = {"jobs": list(jobs), "segments": list(segments)}

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
            if any(isinstance(v, str) and v in wanted for v in values):
                rows.append(values)
        return _FakeResult(rows)


def _override(session):
    app.dependency_overrides[get_current_user] = lambda: _fake_user
    app.dependency_overrides[get_db] = lambda: session


async def _delete(path, force=None):
    from httpx import ASGITransport, AsyncClient

    params = {"path": path}
    if force is not None:
        params["force"] = force
    with patch("app.routes.images.delete_object") as deleter:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/images", params=params)
    return resp, deleter


class TestDeleteRefusesReferencedImages:
    def teardown_method(self):
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_segment_faceswap_reference_returns_409(self):
        """The identity recipe puts a face on every job's segments, and no job row mentions it —
        this is the reference the old Job.starting_image-only check could never see."""
        segment = Segment(id=uuid.uuid4(), prompt="x", faceswap_image=FACE)
        _override(_FakeSession(segments=[segment]))

        resp, deleter = await _delete(FACE)

        assert resp.status_code == 409
        assert str(segment.id) in resp.json()["detail"]["segment_ids"]
        deleter.assert_not_called()

    @pytest.mark.asyncio
    async def test_archived_job_reference_returns_409(self):
        """Archiving hides a job; it does not stop its segments being re-run, so its images are
        still live references. The old query excluded archived jobs outright."""
        job = Job(id=uuid.uuid4(), name="j", width=832, height=480, fps=16, seed=1,
                  starting_image=START, status=JobStatus.ARCHIVED)
        _override(_FakeSession(jobs=[job]))

        resp, deleter = await _delete(START)

        assert resp.status_code == 409
        assert str(job.id) in resp.json()["detail"]["job_ids"]
        deleter.assert_not_called()

    @pytest.mark.asyncio
    async def test_segment_start_image_reference_returns_409(self):
        segment = Segment(id=uuid.uuid4(), prompt="x", start_image=START)
        _override(_FakeSession(segments=[segment]))

        resp, deleter = await _delete(START)

        assert resp.status_code == 409
        deleter.assert_not_called()

    @pytest.mark.asyncio
    async def test_409_names_every_holder(self):
        """The point of the 409 body is to say what to fix; a bare "in use" leaves you hunting."""
        job = Job(id=uuid.uuid4(), name="j", width=832, height=480, fps=16, seed=1,
                  starting_image=FACE, status=JobStatus.PENDING)
        segment = Segment(id=uuid.uuid4(), prompt="x", faceswap_image=FACE)
        _override(_FakeSession(jobs=[job], segments=[segment]))

        resp, _ = await _delete(FACE)

        detail = resp.json()["detail"]
        assert detail["job_ids"] == [str(job.id)]
        assert detail["segment_ids"] == [str(segment.id)]

    @pytest.mark.asyncio
    async def test_force_deletes_a_referenced_image(self):
        segment = Segment(id=uuid.uuid4(), prompt="x", faceswap_image=FACE)
        _override(_FakeSession(segments=[segment]))

        resp, deleter = await _delete(FACE, force="true")

        assert resp.status_code == 200
        deleter.assert_called_once_with(FACE)

    @pytest.mark.asyncio
    async def test_unreferenced_image_is_deleted(self):
        """The check must not turn into "nothing is deletable" — rows that point elsewhere
        are not holders."""
        job = Job(id=uuid.uuid4(), name="j", width=832, height=480, fps=16, seed=1,
                  starting_image=START, status=JobStatus.PENDING)
        segment = Segment(id=uuid.uuid4(), prompt="x", faceswap_image=FACE)
        _override(_FakeSession(jobs=[job], segments=[segment]))

        resp, deleter = await _delete(LOOSE)

        assert resp.status_code == 200
        deleter.assert_called_once_with(LOOSE)

    @pytest.mark.asyncio
    async def test_wrong_bucket_still_rejected_before_any_db_work(self):
        _override(_FakeSession())

        resp, deleter = await _delete("s3://wanly-jobs/abc/last_frame.png")

        assert resp.status_code == 400
        deleter.assert_not_called()


class TestListingAgreesWithDelete:
    """The listing flag and the delete gate must come from the same query — the divergence is
    what made this invisible: the UI said unused, the delete obeyed, the segment died later."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_faceswap_only_reference_reads_as_in_use(self):
        from httpx import ASGITransport, AsyncClient

        segment = Segment(id=uuid.uuid4(), prompt="x", faceswap_image=FACE)
        _override(_FakeSession(segments=[segment]))

        objects = [
            {"Key": FACE.split(f"{BUCKET}/")[1], "Size": 10, "LastModified": "2026-07-09T00:00:00"},
            {"Key": LOOSE.split(f"{BUCKET}/")[1], "Size": 10, "LastModified": "2026-07-09T00:00:00"},
        ]
        with patch("app.routes.images.list_objects", return_value=objects):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/images/folder/2026-07-09")

        assert resp.status_code == 200
        by_path = {item["path"]: item["in_use"] for item in resp.json()}
        assert by_path[FACE] is True
        assert by_path[LOOSE] is False
