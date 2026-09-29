"""The Image Edit tool, phase 1: face mode (wanly-console#547).

What these pin down:
  * validation happens before anything is fetched or run -- ranges, unknown axes, unknown
    presets, "nothing to apply", a source outside the images bucket, a mode phase 1 lacks
  * a locked dataset is refused with 409 BEFORE the service is called or anything is written
  * the service being down, slow or busy is a 503 that says which, and writes nothing
  * a save is always a NEW object; the source is never overwritten
  * a preview stores nothing
"""
import base64
import uuid

import httpx
import pytest

from app import face_edit
from app.config import settings
from tests.test_dataset_lock import _Usr, _ds, _run

BUCKET = "wanly-images"
SRC = f"s3://{BUCKET}/2026-09-01/sel_008.jpg"


# ------------------------------------------------------------------------ resolve

class TestResolve:
    def test_a_preset_alone(self):
        assert face_edit.resolve("smile", None) == {"smile": 0.5}

    def test_explicit_values_lay_over_the_preset(self):
        """The editor's model: a preset moves the sliders, a drag changes one axis."""
        got = face_edit.resolve("big_laugh", {"aaa": 20, "blink": None})
        assert got == {"smile": 1.3, "aaa": 20}

    def test_zeros_are_dropped_from_the_record(self):
        assert face_edit.resolve(None, {"smile": 0.4, "aaa": 0}) == {"smile": 0.4}

    def test_an_explicit_zero_can_cancel_a_presets_axis(self):
        assert face_edit.resolve("surprised", {"aaa": 0}) == {"eyebrow": 8, "blink": 4}

    def test_an_unknown_preset_names_the_real_ones(self):
        with pytest.raises(face_edit.FaceEditError) as e:
            face_edit.resolve("grimace", None)
        assert e.value.status_code == 422 and "eyes_closed" in e.value.detail

    def test_nothing_to_apply_is_refused(self):
        """Saving an unchanged copy of the original is the worst possible outcome."""
        for preset, expr in ((None, None), (None, {"smile": 0}), (None, {})):
            with pytest.raises(face_edit.FaceEditError) as e:
                face_edit.resolve(preset, expr)
            assert e.value.status_code == 422

    def test_every_preset_the_issue_asks_for_exists(self):
        wanted = {"smile", "big_laugh", "eyes_closed", "surprised", "serious", "speaking",
                  "look_left", "look_right", "look_up", "look_down",
                  "turn_head_left", "turn_head_right"}
        assert wanted == set(face_edit.PRESETS)

    def test_every_preset_is_inside_the_nodes_ranges(self):
        bounds = {a["key"]: (a["min"], a["max"]) for a in face_edit.AXES}
        for name, (_label, exp) in face_edit.PRESETS.items():
            for k, v in exp.items():
                lo, hi = bounds[k]
                assert lo <= v <= hi, f"{name}.{k}={v} outside {lo}..{hi}"

    def test_the_seven_main_sliders(self):
        main = {a["key"] for a in face_edit.AXES if a["group"] == "main"}
        assert main == {"rotate_yaw", "rotate_pitch", "rotate_roll", "blink", "smile", "aaa",
                        "eyebrow"}


# ---------------------------------------------------------------------- the client

class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text
        self.reason_phrase = "x"

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _client(resp=None, exc=None, seen=None):
    class C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            if seen is not None:
                seen.append((url, json))
            if exc is not None:
                raise exc
            return resp

    return C


class _Httpx:
    """Replaces face_edit.httpx for one test, keeping the real exception classes."""

    def __init__(self, client):
        self.AsyncClient = client
        self.HTTPError = httpx.HTTPError
        self.TimeoutException = httpx.TimeoutException


def _ok_body(fmt="png"):
    return {"image": base64.b64encode(b"\x89PNG fake").decode(), "format": fmt,
            "width": 10, "height": 20, "device": "cuda", "device_reason": "free",
            "timings_ms": {"total": 900}}


@pytest.mark.asyncio
class TestTheClient:
    @pytest.fixture(autouse=True)
    def _url(self, monkeypatch):
        monkeypatch.setattr(settings, "face_edit_url", "http://edit.test:8085")

    async def _edit(self, monkeypatch, preview=False, **kw):
        monkeypatch.setattr(face_edit, "httpx", _Httpx(_client(**kw)))
        return await face_edit.edit(b"src", {"smile": 0.5}, preview=preview)

    async def test_ok_decodes_the_image(self, monkeypatch):
        seen = []
        out = await self._edit(monkeypatch, resp=_Resp(200, _ok_body()), seen=seen)
        assert out["image"] == b"\x89PNG fake" and out["device"] == "cuda"
        url, body = seen[0]
        assert url == "http://edit.test:8085/edit"
        assert body["expression"] == {"smile": 0.5} and "format" not in body

    async def test_a_preview_asks_for_a_small_jpeg(self, monkeypatch):
        seen = []
        await self._edit(monkeypatch, preview=True, resp=_Resp(200, _ok_body("jpeg")), seen=seen)
        assert seen[0][1]["format"] == "jpeg"
        assert seen[0][1]["max_edge"] == face_edit.PREVIEW_MAX_EDGE

    async def test_not_configured_is_503(self, monkeypatch):
        monkeypatch.setattr(settings, "face_edit_url", "")
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.edit(b"x", {"smile": 1})
        assert e.value.status_code == 503 and "face_edit_url is empty" in e.value.detail

    async def test_unreachable_is_503(self, monkeypatch):
        with pytest.raises(face_edit.FaceEditError) as e:
            await self._edit(monkeypatch, exc=httpx.ConnectError("refused"))
        assert e.value.status_code == 503 and "unreachable" in e.value.detail

    async def test_a_timeout_is_503_and_says_so(self, monkeypatch):
        with pytest.raises(face_edit.FaceEditError) as e:
            await self._edit(monkeypatch, exc=httpx.ReadTimeout("slow"))
        assert e.value.status_code == 503 and "did not answer" in e.value.detail

    async def test_busy_is_503_with_the_services_reason(self, monkeypatch):
        with pytest.raises(face_edit.FaceEditError) as e:
            await self._edit(monkeypatch, resp=_Resp(503, {"detail": "face-edit is busy: x"}))
        assert e.value.status_code == 503 and "busy" in e.value.detail

    async def test_no_face_is_422(self, monkeypatch):
        with pytest.raises(face_edit.FaceEditError) as e:
            await self._edit(monkeypatch, resp=_Resp(422, {"detail": "no face detected"}))
        assert e.value.status_code == 422 and "no face" in e.value.detail

    async def test_anything_else_is_502(self, monkeypatch):
        with pytest.raises(face_edit.FaceEditError) as e:
            await self._edit(monkeypatch, resp=_Resp(500, None, text="Traceback"))
        assert e.value.status_code == 502


# ------------------------------------------------------------------------ where to

class TestWhereAResultGoes:
    def test_a_repo_image_is_edited_into_its_own_folder(self):
        from app.routes.image_edit import result_key
        k = result_key("2026-09-01/sel_008.jpg", "smile")
        assert k.startswith("2026-09-01/sel_008_edit-smile_") and k.endswith(".png")

    def test_two_edits_never_collide(self):
        from app.routes.image_edit import result_key
        assert result_key("f/a.jpg", "smile") != result_key("f/a.jpg", "smile")

    def test_a_dataset_image_saved_as_new_goes_to_a_repo_date_folder(self):
        """datasets/ is not a repo folder (#464); an edit saved "as new image" belongs in the
        repo, where it can be seen."""
        from app.routes.image_edit import result_key
        k = result_key("datasets/abc/faces-1/003_x.jpg", "serious")
        assert not k.startswith("datasets/") and k.split("/")[0].count("-") == 2

    def test_a_dataset_target_goes_under_its_prefix(self):
        from app.models import Dataset
        from app.routes.image_edit import result_key
        ds = Dataset(id=uuid.uuid4(), name="d", prefix="datasets/xyz")
        k = result_key("2026-09-01/a.jpg", "look_up", ds)
        assert k.startswith("datasets/xyz/edits/a_edit-look_up_")


# ---------------------------------------------------------------------- over HTTP

class _FakeS3:
    def __init__(self):
        self.downloaded, self.uploaded = [], []

    def download_bytes(self, uri):
        self.downloaded.append(uri)
        return b"source bytes"

    def upload_bytes(self, data, key, bucket):
        self.uploaded.append((key, bucket, data))
        return f"s3://{bucket}/{key}"


@pytest.fixture
def wired(monkeypatch):
    """The app with fake S3 and a fake face-edit; records what reached each."""
    from app.routes import image_edit as mod

    fake_s3 = _FakeS3()
    calls = []
    state = {"error": None}

    async def fake_edit(image_bytes, params, preview=False):
        calls.append((image_bytes, params, preview))
        if state["error"]:
            raise state["error"]
        return {"image": b"edited png", "format": "jpeg" if preview else "png",
                "width": 1248, "height": 1824, "device": "cuda", "device_reason": "free"}

    monkeypatch.setattr(mod, "s3", fake_s3)
    monkeypatch.setattr(mod.face_edit, "edit", fake_edit)
    monkeypatch.setattr(settings, "s3_images_bucket", BUCKET)
    return fake_s3, calls, state


async def _http(db, method, url, **kw):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: _Usr()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await getattr(c, method)(url, **kw)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestOverHTTP:
    async def test_presets(self, db, wired):
        r = await _http(db, "get", "/images/edit/presets")
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "face" and len(d["presets"]) == 12
        assert {a["key"] for a in d["axes"]} >= {"rotate_yaw", "smile", "aaa", "blink"}

    async def test_save_to_the_repo_is_a_new_object_beside_the_original(self, db, wired):
        s3, calls, _ = wired
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "expression": {"aaa": 10}})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["source_uri"] == SRC and d["uri"] != SRC
        assert d["uri"].startswith(f"s3://{BUCKET}/2026-09-01/sel_008_edit-smile_")
        assert d["params"] == {"smile": 0.5, "aaa": 10} and d["mode"] == "face"
        assert calls[0][1] == {"smile": 0.5, "aaa": 10} and calls[0][2] is False
        [(key, _bucket, data)] = s3.uploaded
        assert data == b"edited png" and key != "2026-09-01/sel_008.jpg"

    async def test_save_to_an_unlocked_dataset_appends(self, db, wired):
        from app.models import Dataset
        assert wired  # fake S3 and service in place
        ds = await _ds(db)
        before = list(ds.images)
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": ds.images[0], "preset": "look_up",
                              "dataset_id": str(ds.id)})
        assert r.status_code == 200, r.text
        uri = r.json()["uri"]
        assert uri.startswith(f"s3://{BUCKET}/{ds.prefix}/edits/0_edit-look_up_")
        assert r.json()["dataset_id"] == str(ds.id)
        after = await db.get(Dataset, ds.id)
        await db.refresh(after)
        assert after.images == before + [uri], "appended, and nothing it held was replaced"

    async def test_a_locked_dataset_is_refused_before_anything_runs(self, db, wired):
        """409 like #356/#358 -- and no edit spent, no object written."""
        from app.models import Dataset
        from app.routes.datasets import lock_dataset
        s3, calls, _ = wired
        ds = await _ds(db)
        await lock_dataset(ds.id, None, _user=None, db=db)
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "dataset_id": str(ds.id)})
        assert r.status_code == 409 and "locked" in r.json()["detail"]
        assert calls == [] and s3.uploaded == [] and s3.downloaded == []
        assert (await db.get(Dataset, ds.id)).images == ds.images

    async def test_a_trained_dataset_is_refused_too(self, db, wired):
        s3, calls, _ = wired
        ds = await _ds(db)
        await _run(db, ds)
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "dataset_id": str(ds.id)})
        assert r.status_code == 409 and calls == [] and s3.uploaded == []

    async def test_an_unknown_dataset_is_404(self, db, wired):
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile",
                              "dataset_id": str(uuid.uuid4())})
        assert r.status_code == 404

    async def test_service_down_is_503_and_writes_nothing(self, db, wired):
        s3, _, state = wired
        ds = await _ds(db)
        state["error"] = face_edit.FaceEditError(503, "face-edit unreachable at http://x: boom")
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "dataset_id": str(ds.id)})
        assert r.status_code == 503 and "unreachable" in r.json()["detail"]
        assert s3.uploaded == []

    async def test_no_face_is_422(self, db, wired):
        _, _, state = wired
        state["error"] = face_edit.FaceEditError(422, "face-edit refused this image: no face")
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "preset": "smile"})
        assert r.status_code == 422 and "no face" in r.json()["detail"]

    async def test_a_preview_stores_nothing(self, db, wired):
        s3, calls, _ = wired
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "expression": {"rotate_yaw": -12}})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["image"].startswith("data:image/jpeg;base64,")
        assert d["params"] == {"rotate_yaw": -12} and (d["width"], d["height"]) == (1248, 1824)
        assert calls[0][2] is True and s3.uploaded == []

    @pytest.mark.parametrize("body,status", [
        ({"source_uri": SRC, "expression": {"rotate_yaw": 40}}, 422),       # out of range
        ({"source_uri": SRC, "expression": {"grin": 1}}, 422),              # unknown axis
        ({"source_uri": SRC, "preset": "grimace"}, 422),                    # unknown preset
        ({"source_uri": SRC}, 422),                                         # nothing to apply
        ({"source_uri": SRC, "mode": "full", "preset": "smile"}, 422),      # phase 2
        ({"source_uri": "s3://wanly-jobs/x.png", "preset": "smile"}, 400),  # wrong bucket
        ({"source_uri": "https://example.com/a.png", "preset": "smile"}, 400),
    ])
    async def test_validation_happens_before_anything_runs(self, db, wired, body, status):
        s3, calls, _ = wired
        for path in ("/images/edit", "/images/edit/preview"):
            r = await _http(db, "post", path, json=body)
            assert r.status_code == status, (path, r.text)
        assert calls == [] and s3.downloaded == [] and s3.uploaded == []

    async def test_a_missing_source_is_404(self, db, wired, monkeypatch):
        s3, calls, _ = wired

        def gone(uri):
            raise FileNotFoundError("NoSuchKey")

        monkeypatch.setattr(s3, "download_bytes", gone)
        r = await _http(db, "post", "/images/edit", json={"source_uri": SRC, "preset": "smile"})
        assert r.status_code == 404 and calls == []

    async def test_it_needs_a_login(self, db, wired):
        from httpx import ASGITransport, AsyncClient
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/images/edit/presets")
            p = await c.post("/images/edit", json={"source_uri": SRC, "preset": "smile"})
        assert r.status_code in (401, 403) and p.status_code in (401, 403)
