"""The Image Edit tool, phase 1: face mode (wanly-console#547).

What these pin down:
  * validation happens before anything is fetched or run -- ranges, unknown axes, unknown
    presets, "nothing to apply", a source outside the images bucket, a mode phase 1 lacks
  * a locked dataset is refused with 409 BEFORE the service is called or anything is written
  * the service being down, slow or busy is a 503 that says which, and writes nothing
  * a save is always a NEW object; the source is never overwritten
  * a preview stores nothing
  * a described change (#550) is forwarded as text, answered with the resolved numbers and the
    understood terms, and "no known terms" is a 422 that says so rather than "refused this image"
  * the face picker (#553): /images/edit/faces lists the service's boxes in the source's own
    pixels, and a box from it reaches the service unchanged on preview and save
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

    def test_a_prompt_alone_is_enough_and_sends_no_numbers(self):
        """The service reads the text; any `expression` at all would make it ignore the text."""
        assert face_edit.resolve(None, None, "big smile") == {}

    def test_a_prompt_with_numbers_keeps_the_numbers(self):
        assert face_edit.resolve("smile", None, "look left") == {"smile": 0.5}

    def test_the_nothing_message_mentions_describing(self):
        with pytest.raises(face_edit.FaceEditError) as e:
            face_edit.resolve(None, None, "")
        assert "describe" in e.value.detail

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


# ------------------------------------------------------------------ matched terms

class TestMatchedTerms:
    # The service's hits are the first \\b-piece of each lexicon regex (expression.py).
    CLOSE = "clos(e|es|ed|ing) (her |his |their )?eyes?"
    GRIN = "grin(s|ning)?"
    SMILE = "smil(e|es|ing)"
    LEFT = "look(s|ing)? (to (her |his |their )?)?left"

    def test_the_issues_example_reads_as_three_terms(self):
        """"big smile, eyes closed, look left" also hits the plain smile entry, whose value
        loses to the grin's -- one chip, not two."""
        src = f"prompt:{self.CLOSE},{self.GRIN},{self.SMILE},{self.LEFT}"
        assert face_edit.matched_terms(src) == ["eyes closed", "big smile", "look left"]

    @pytest.mark.parametrize("hit,label", [
        ("look(s|ing)? up", "look up"),
        ("turn(s|ed|ing)? (her |his |their )?head (to the )?left", "turn head left"),
        ("tilt(s|ed|ing)? (her |his |their )?head", "tilt head"),
        ("squint(s|ed|ing)?", "squint"),
        ("furrow(s|ed|ing)?", "frown"),
        ("chin up", "chin up"),
        ("purs(e|es|ed|ing)", "purse"),
    ])
    def test_regex_pieces_become_words(self, hit, label):
        assert face_edit.matched_terms(f"prompt:{hit}") == [label]

    def test_smile_alone_stays(self):
        assert face_edit.matched_terms(f"prompt:{self.SMILE}") == ["smile"]

    @pytest.mark.parametrize("src", [None, "", "explicit", "preset:smile"])
    def test_nothing_that_is_not_a_prompt(self, src):
        assert face_edit.matched_terms(src) == []


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

    async def test_a_prompt_is_sent_as_text_without_an_expression(self, monkeypatch):
        seen = []
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(200, _ok_body()), seen=seen)))
        await face_edit.edit(b"src", {}, prompt="big smile", preview=True)
        body = seen[0][1]
        assert body["prompt"] == "big smile" and "expression" not in body

    async def test_numbers_and_no_prompt_send_no_prompt(self, monkeypatch):
        seen = []
        await self._edit(monkeypatch, resp=_Resp(200, _ok_body()), seen=seen)
        assert "prompt" not in seen[0][1]

    async def test_no_known_terms_is_422_that_says_so(self, monkeypatch):
        """The service's NothingToApply (expression.py) is a 422 from app.py; it is about the
        words, not the image, so it must not read "refused this image"."""
        monkeypatch.setattr(face_edit, "httpx", _Httpx(_client(resp=_Resp(422, {
            "detail": "nothing to apply: send an `expression` object, or a prompt using a "
                      "known term. Recognised terms: smile, grin, laugh."}))))
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.edit(b"src", {}, prompt="dance a jig")
        d = e.value.detail
        assert e.value.status_code == 422
        assert d.startswith("nothing to apply: no known terms in 'dance a jig'")
        assert "Recognised terms: smile, grin, laugh." in d
        assert "refused this image" not in d and "`expression`" not in d

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
    state = {"error": None, "resolved": None, "prompts": []}

    async def fake_edit(image_bytes, params, prompt=None, preview=False, face_index=None,
                        face_box=None):
        calls.append((image_bytes, params, preview))
        state["prompts"].append(prompt)
        state.setdefault("faces_sent", []).append((face_index, face_box))
        if state["error"]:
            raise state["error"]
        out = {"image": b"edited png", "format": "jpeg" if preview else "png",
               "width": 1248, "height": 1824, "device": "cuda", "device_reason": "free"}
        if state["resolved"] is not None:
            out.update(state["resolved"])
        return out

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
        # "full" since console#569: the dialog is all Qwen. LivePortrait's presets and axes stay
        # listed for the face-mode endpoints that remain.
        assert d["mode"] == "full" and len(d["presets"]) == 12
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

    async def test_a_trained_dataset_takes_the_edit(self, db, wired):
        """#420: training no longer locks a set -- its runs recorded what they trained on."""
        s3, calls, _ = wired
        ds = await _ds(db)
        await _run(db, ds)
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "dataset_id": str(ds.id)})
        assert r.status_code == 200, r.text
        assert calls and s3.uploaded

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

    async def test_a_described_change_previews_and_says_what_it_understood(self, db, wired):
        s3, calls, state = wired
        exp = dict.fromkeys(face_edit.AXIS_KEYS, 0.0)
        exp.update(smile=1.3, blink=-20.0, pupil_x=-12.0)
        state["resolved"] = {"expression": exp, "source": (
            "prompt:clos(e|es|ed|ing) (her |his |their )?eyes?,grin(s|ning)?,smil(e|es|ing),"
            "look(s|ing)? (to (her |his |their )?)?left")}
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "prompt": "  big smile, eyes closed, look left "})
        assert r.status_code == 200, r.text
        d = r.json()
        assert state["prompts"] == ["big smile, eyes closed, look left"] and calls[0][1] == {}
        assert d["matched_terms"] == ["eyes closed", "big smile", "look left"]
        assert d["source"].startswith("prompt:")
        assert d["expression"] == exp, "all twelve axes, for the sliders"
        assert d["params"] == {"smile": 1.3, "blink": -20, "pupil_x": -12}
        assert s3.uploaded == []

    async def test_a_described_change_saves_its_numbers(self, db, wired):
        s3, _, state = wired
        exp = dict.fromkeys(face_edit.AXIS_KEYS, 0.0)
        exp.update(wink=15.0)
        state["resolved"] = {"expression": exp, "source": "prompt:wink(s|ed|ing)?"}
        r = await _http(db, "post", "/images/edit", json={"source_uri": SRC, "prompt": "wink"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["prompt"] == "wink" and d["params"] == {"wink": 15}
        assert d["matched_terms"] == ["wink"] and d["expression"]["wink"] == 15
        assert d["uri"].startswith(f"s3://{BUCKET}/2026-09-01/sel_008_edit-prompt_")
        assert len(s3.uploaded) == 1

    async def test_numbers_answer_with_all_twelve_axes_too(self, db, wired):
        """An older service returns no `expression`; the sent numbers are what it applied."""
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "preset": "look_left"})
        d = r.json()
        assert d["expression"]["pupil_x"] == -8 and d["expression"]["smile"] == 0
        assert len(d["expression"]) == 12 and d["matched_terms"] == []

    async def test_no_known_terms_is_422_and_writes_nothing(self, db, wired):
        s3, _, state = wired
        state["error"] = face_edit.FaceEditError(
            422, "nothing to apply: no known terms in 'dance'. Recognised terms: smile.")
        for path in ("/images/edit", "/images/edit/preview"):
            r = await _http(db, "post", path, json={"source_uri": SRC, "prompt": "dance"})
            assert r.status_code == 422 and "no known terms" in r.json()["detail"]
        assert s3.uploaded == []

    @pytest.mark.parametrize("body,status", [
        ({"source_uri": SRC, "expression": {"rotate_yaw": 40}}, 422),       # out of range
        ({"source_uri": SRC, "expression": {"grin": 1}}, 422),              # unknown axis
        ({"source_uri": SRC, "preset": "grimace"}, 422),                    # unknown preset
        ({"source_uri": SRC}, 422),                                         # nothing to apply
        ({"source_uri": SRC, "prompt": "   "}, 422),                        # blank description
        ({"source_uri": SRC, "prompt": "smile " * 100}, 422),               # too long
        ({"source_uri": SRC, "mode": "full", "preset": "grimace"}, 422),    # not an expression
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


# ---------------------------------------------------------------- the face picker (#553)

#: What the service's /faces says about a 1248x1824 two-person frame.
FACES = {"width": 1248, "height": 1824, "default_index": 1, "faces": [
    {"index": 0, "box": [101.5, 400.0, 351.5, 700.0], "width": 250.0},
    {"index": 1, "box": [700.0, 380.0, 940.0, 670.0], "width": 240.0},
]}


@pytest.mark.asyncio
class TestTheFacesClient:
    @pytest.fixture(autouse=True)
    def _url(self, monkeypatch):
        monkeypatch.setattr(settings, "face_edit_url", "http://edit.test:8085")

    async def test_it_asks_the_services_faces(self, monkeypatch):
        seen = []
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(200, FACES), seen=seen)))
        out = await face_edit.faces(b"src")
        assert seen[0][0] == "http://edit.test:8085/faces"
        assert base64.b64decode(seen[0][1]["image"]) == b"src"
        assert out == FACES

    async def test_a_service_too_old_for_faces_says_so(self, monkeypatch):
        """The 2070 not yet re-pinned: FastAPI's 404 for an unknown route."""
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(404, {"detail": "Not Found"}))))
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.faces(b"src")
        assert e.value.status_code == 502 and "too old" in e.value.detail

    async def test_a_nonsense_answer_is_502(self, monkeypatch):
        monkeypatch.setattr(face_edit, "httpx", _Httpx(_client(resp=_Resp(200, {"x": 1}))))
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.faces(b"src")
        assert e.value.status_code == 502

    async def test_busy_is_503(self, monkeypatch):
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(503, {"detail": "face-edit is busy"}))))
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.faces(b"src")
        assert e.value.status_code == 503

    async def test_an_edit_sends_the_choice_only_when_there_is_one(self, monkeypatch):
        seen = []
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(200, _ok_body()), seen=seen)))
        await face_edit.edit(b"src", {"smile": 0.5})
        monkeypatch.setattr(face_edit, "httpx",     # a fresh body: edit() decodes in place
                            _Httpx(_client(resp=_Resp(200, _ok_body()), seen=seen)))
        await face_edit.edit(b"src", {"smile": 0.5}, face_index=0,
                             face_box=[101.5, 400.0, 351.5, 700.0])
        assert "face_index" not in seen[0][1] and "face_box" not in seen[0][1]
        assert seen[1][1]["face_index"] == 0
        assert seen[1][1]["face_box"] == [101.5, 400.0, 351.5, 700.0]

    @pytest.mark.parametrize("detail", [
        "face_index 2 is out of range: 2 faces detected (0-1)",
        "face 0 cannot be edited on its own: it is too close to another face",
        "face_box [1, 2, 3, 4] matches none of the 2 faces detected in this image",
    ])
    async def test_the_services_face_refusals_are_422(self, monkeypatch, detail):
        monkeypatch.setattr(face_edit, "httpx",
                            _Httpx(_client(resp=_Resp(422, {"detail": detail}))))
        with pytest.raises(face_edit.FaceEditError) as e:
            await face_edit.edit(b"src", {"smile": 0.5}, face_index=2)
        assert e.value.status_code == 422 and detail in e.value.detail


@pytest.fixture
def faces_wired(wired, monkeypatch):
    from app.routes import image_edit as mod
    state = wired[2]
    state["faces"] = FACES
    seen = []

    async def fake_faces(image_bytes):
        seen.append(image_bytes)
        if isinstance(state["faces"], Exception):
            raise state["faces"]
        return state["faces"]

    monkeypatch.setattr(mod.face_edit, "faces", fake_faces)
    return (*wired, seen)


@pytest.mark.asyncio
class TestTheFacePickerOverHTTP:
    async def test_faces_are_the_services_boxes_in_source_pixels(self, db, faces_wired):
        """Nothing is resized on the way to the service, so its boxes ARE source pixels: the
        mapping is the identity, and must stay one (a downscale here would need its inverse on
        every box, both ways)."""
        s3, _, _, seen = faces_wired
        r = await _http(db, "post", "/images/edit/faces", json={"source_uri": SRC})
        assert r.status_code == 200, r.text
        d = r.json()
        assert seen == [b"source bytes"], "the source's bytes, unresized"
        assert (d["width"], d["height"]) == (1248, 1824) and d["default_index"] == 1
        assert [f["box"] for f in d["faces"]] == [f["box"] for f in FACES["faces"]]
        assert s3.downloaded == [SRC] and s3.uploaded == []

    async def test_a_box_from_faces_reaches_the_service_unchanged(self, db, faces_wired):
        """The round trip the console makes: /faces, then preview and save with that box."""
        s3, calls, state, _ = faces_wired
        box = (await _http(db, "post", "/images/edit/faces",
                           json={"source_uri": SRC})).json()["faces"][0]["box"]
        state["resolved"] = {"face_index": 0, "face_box": box}
        p = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "preset": "smile", "face_box": box})
        s = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "face_box": box})
        assert p.status_code == 200 and s.status_code == 200, (p.text, s.text)
        assert state["faces_sent"] == [(None, box), (None, box)]
        assert p.json()["face_box"] == box and p.json()["face_index"] == 0
        assert s.json()["face_box"] == box and s.json()["face_index"] == 0
        assert len(s3.uploaded) == 1

    async def test_an_index_is_forwarded_too(self, db, faces_wired):
        _, _, state, _ = faces_wired
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "preset": "smile", "face_index": 1})
        assert r.status_code == 200 and state["faces_sent"] == [(1, None)]

    async def test_no_choice_sends_none_and_echoes_none(self, db, faces_wired):
        _, _, state, _ = faces_wired
        r = await _http(db, "post", "/images/edit/preview",
                        json={"source_uri": SRC, "preset": "smile"})
        assert state["faces_sent"] == [(None, None)]
        assert r.json()["face_index"] is None and r.json()["face_box"] is None

    async def test_a_face_refusal_is_422_and_writes_nothing(self, db, faces_wired):
        s3, _, state, _ = faces_wired
        state["error"] = face_edit.FaceEditError(
            422, "face-edit refused this image: face_index 5 is out of range: 2 faces detected")
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "preset": "smile", "face_index": 5})
        assert r.status_code == 422 and "out of range" in r.json()["detail"]
        assert s3.uploaded == []

    async def test_faces_passes_service_errors_on(self, db, faces_wired):
        _, _, state, _ = faces_wired
        state["faces"] = face_edit.FaceEditError(502, "the face-edit service is too old")
        r = await _http(db, "post", "/images/edit/faces", json={"source_uri": SRC})
        assert r.status_code == 502 and "too old" in r.json()["detail"]

    async def test_faces_only_reads_the_images_bucket(self, db, faces_wired):
        s3, _, _, seen = faces_wired
        r = await _http(db, "post", "/images/edit/faces",
                        json={"source_uri": "s3://wanly-jobs/x.png"})
        assert r.status_code == 400 and seen == [] and s3.downloaded == []

    @pytest.mark.parametrize("extra", [
        {"face_index": -1},
        {"face_box": [10, 10, 5, 50]},          # x2 < x1
        {"face_box": [10, 10, 50]},             # three numbers
        {"face_box": [-1, 10, 50, 60]},
    ])
    async def test_a_malformed_choice_is_422_before_anything_runs(self, db, faces_wired, extra):
        s3, calls, _, _ = faces_wired
        for path in ("/images/edit", "/images/edit/preview"):
            r = await _http(db, "post", path, json={"source_uri": SRC, "preset": "smile", **extra})
            assert r.status_code == 422, (path, r.text)
        assert calls == [] and s3.downloaded == []

    async def test_faces_needs_a_login(self, db, faces_wired):
        from httpx import ASGITransport, AsyncClient
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/images/edit/faces", json={"source_uri": SRC})
        assert r.status_code in (401, 403)
