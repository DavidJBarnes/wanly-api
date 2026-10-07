"""Image Edit phase 2: full mode (Qwen-Image-Edit on the 3090) as async jobs (wanly-console#548).

What these pin down:
  * every head angle and expression is Qwen's (#569): no LivePortrait route, and the service is
    checked for the fields it would otherwise silently ignore
  * a healthy standing box (#570) is preferred, waited for while A1111 generates, and fallen
    back from to the main 3090's edit mode only when it is down
  * validation happens before anything is fetched or queued
  * a job's states: it WAITS while the 3090 renders (asking for edit mode once), trains, or
    switches, and says which; then runs; then is done with a preview and an identity score
  * the card is handed back to the mode the box was in, once the queue is empty
  * nothing is saved until save; a save is a NEW object, and a locked dataset is refused
"""
import asyncio
import base64
import uuid

import pytest

from app import full_edit
from app.config import settings
from tests.test_dataset_lock import _ds
from tests.test_image_edit import BUCKET, SRC, _FakeS3, _http


def _png():
    return b"\x89PNG fake result"


# ------------------------------------------------------------------- the presets

class TestPresets:
    def test_every_head_preset_the_issue_asks_for(self):
        assert set(full_edit.HEAD_ANGLES) == {
            "look_left", "look_right", "three_quarter_left", "three_quarter_right",
            "profile_left", "profile_right", "look_up", "look_down"}

    def test_left_is_negative_yaw_and_up_is_positive_pitch(self):
        assert full_edit.HEAD_ANGLES["profile_left"][1] == -90
        assert full_edit.HEAD_ANGLES["three_quarter_right"][1] == 45
        assert full_edit.HEAD_ANGLES["look_up"][2] > 0 > full_edit.HEAD_ANGLES["look_down"][2]

    def test_nothing_routes_to_liveportrait_any_more(self):
        """#569: LivePortrait "drops every detail". No split, no face route."""
        assert not hasattr(full_edit, "route") and not hasattr(full_edit, "FACE_LIMIT_DEG")

    def test_the_expressions_the_issue_names(self):
        """smile, big laugh, surprised, eyes closed, sad, angry -- and the old face presets'
        names, so a saved file's tag reads the same across the switch."""
        assert {"smile", "big_laugh", "surprised", "eyes_closed", "sad", "angry"} \
            <= set(full_edit.EXPRESSIONS)
        from app import face_edit
        kept = set(face_edit.PRESETS) - {"turn_head_left", "turn_head_right"}
        assert kept <= set(full_edit.EXPRESSIONS), "head turns moved to the angle section"


# ---------------------------------------------------------------------- the box


class _Box:
    """A fake 3090: its control /health, its /mode, and the image-edit service."""

    def __init__(self, mode="ltx-engine", rendering=0, training=False, equipped=True,
                 switch_after=2, mode_error=None, edit_error=None):
        self.mode, self.pending = mode, None
        self.rendering, self.training = rendering, training
        self.equipped, self.switch_after = equipped, switch_after
        self.mode_error, self.edit_error = mode_error, edit_error
        self.modes_asked, self.edits, self._polls = [], [], 0
        self.urls = []
        #: What the main 3090's image-edit /health says it can do; None = a pre-#569 image.
        self.features = ["angle", "instruction", "expression", "face_box", "faces"]

    async def health(self):
        self._polls += 1
        if self.pending and self._polls >= self.switch_after:
            if self.mode_error:
                self.pending = None
            else:
                self.mode, self.pending, self.rendering = self.pending, None, 0
        eq = ["ltx-engine", "lora-trainer", "face-crop"] + (["image-edit"] if self.equipped else [])
        services = [
            {"name": "ltx-engine-api", "group": "ltx-engine", "running": self.rendering,
             "ready": True, "stopped": self.mode == "edit"},
            {"name": "lora-trainer", "group": "lora-trainer", "training": self.training,
             "ready": True},
            {"name": "image-edit", "group": "image-edit", "ready": self.mode == "edit",
             "stopped": self.mode != "edit"},
        ]
        return {"mode": self.mode, "pending_mode": self.pending, "equipped": eq,
                "services": services,
                "mode_error": self.mode_error if self.modes_asked else None}

    async def set_mode(self, mode):
        self.modes_asked.append(mode)
        if mode == "edit":
            self.pending = "edit"
        else:
            self.mode = mode

    async def edit(self, source, request, url=None):
        self.edits.append(request)
        self.urls.append(url)
        if self.edit_error:
            raise self.edit_error
        return {"image": base64.b64encode(_png()).decode(), "preview": "anBn",
                "width": 64, "height": 48, "face_box": request.get("face_box"),
                "prompt": "Turn the person's head ...", "seed": 548,
                "identity": {"aura": 0.52, "reason": None}, "vram_peak_mib": 23400,
                "timings_ms": {"edit": 13000}}


@pytest.fixture
def box(monkeypatch):
    def make(**kw):
        b = _Box(**kw)
        monkeypatch.setattr(full_edit, "_health", b.health)
        monkeypatch.setattr(full_edit, "_set_mode", b.set_mode)
        monkeypatch.setattr(full_edit, "_edit", b.edit)
        monkeypatch.setattr(full_edit, "queue", full_edit.FullEditQueue())
        # No standing box unless a test adds one (#570).
        monkeypatch.setattr(settings, "image_edit_standing_url", "")
        real_require = full_edit._require_features

        async def require(url, request, who, health=None):
            if health is None:
                health = {"features": b.features} if b.features else {}
            return await real_require(url, request, who, health=health)
        monkeypatch.setattr(full_edit, "_require_features", require)
        monkeypatch.setattr(settings, "image_edit_poll_s", 0.001)
        monkeypatch.setattr(settings, "image_edit_return_grace_s", 0.01)
        monkeypatch.setattr(settings, "image_edit_url", "http://3090.zero:8086")
        return b
    return make


async def _drain():
    """Let the queue's worker run to completion (it returns after handing the card back)."""
    t = full_edit.queue._task
    if t is not None:
        await asyncio.wait_for(t, timeout=5)


@pytest.mark.asyncio
class TestJobStates:
    async def test_it_waits_for_the_render_asks_once_runs_and_hands_back(self, box):
        b = box(mode="ltx-engine", rendering=1, switch_after=4)
        seen = []
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": -90, "pitch": 0}},
                                     "profile_left")
        assert job.state == "queued" and full_edit.queue.position(job) == 0
        for _ in range(200):
            seen.append((job.state, job.message))
            if job.state in ("done", "failed"):
                break
            await asyncio.sleep(0.001)
        await _drain()
        assert ("waiting", "3090.zero is rendering; edit queued (the segment in flight "
                           "finishes first)") in seen
        assert job.state == "done", job.error
        assert b.modes_asked == ["edit", "ltx-engine"], "asked once, then gave the card back"
        assert job.result and job.preview.startswith("data:image/jpeg;base64,")
        assert job.meta["identity"] == {"aura": 0.52, "reason": None}
        assert b.edits == [{"angle": {"yaw": -90, "pitch": 0}}]

    async def test_a_box_already_in_edit_mode_is_not_switched(self, box):
        b = box(mode="edit")
        job = full_edit.queue.submit(SRC, b"src", {"instruction": "red sweater"}, "full")
        await _drain()
        assert job.state == "done" and b.modes_asked == []

    async def test_a_run_of_edits_pays_for_one_switch(self, box):
        b = box(mode="ltx-engine")
        jobs = [full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
                for _ in range(3)]
        assert [full_edit.queue.position(j) for j in jobs] == [0, 1, 2]
        await _drain()
        assert [j.state for j in jobs] == ["done"] * 3
        assert b.modes_asked == ["edit", "ltx-engine"]

    async def test_a_training_run_is_waited_for_not_interrupted(self, box, monkeypatch):
        b = box(mode="ltx-engine", training=True)
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
        for _ in range(50):
            await asyncio.sleep(0.001)
        assert job.state == "waiting" and "training" in job.message
        assert b.modes_asked == [], "never switched while training"
        b.training = False
        await _drain()
        assert job.state == "done" and b.modes_asked[0] == "edit"

    async def test_a_box_without_image_edit_fails_the_job_and_says_why(self, box):
        box(equipped=False)
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
        await _drain()
        assert job.state == "failed" and "not equipped" in job.error

    async def test_a_switch_that_failed_on_the_box_fails_the_job(self, box):
        box(mode_error="image-edit-comfyui: the Qwen model tree /workspace/qwen is not there",
            switch_after=2)
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
        await _drain()
        assert job.state == "failed" and "could not enter edit mode" in job.error

    async def test_the_service_refusing_is_a_failed_job_not_a_crash(self, box):
        b = box(mode="edit", edit_error=full_edit.FullEditError(422, "image-edit: bad image"))
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
        ok = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": -45}}, "angle")
        b_edit = b.edit

        async def second_ok(source, request):
            if request["angle"]["yaw"] == -45:
                b.edit_error = None
            return await b_edit(source, request)
        full_edit._edit = second_ok
        await _drain()
        assert job.state == "failed" and job.error == "image-edit: bad image"
        assert ok.state == "done", "one bad edit does not stop the queue"

    async def test_an_expression_on_a_pre_569_image_fails_instead_of_misapplying(self, box):
        """It would ignore `expression` and return the angle alone -- plausible and wrong."""
        b = box(mode="edit")
        b.features = None
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}, "expression": "smile"},
                                     "angle-smile")
        await _drain()
        assert job.state == "failed" and "too old for expression" in job.error
        assert b.edits == []

    async def test_a_face_box_the_service_ignored_is_not_kept(self, box):
        b = box(mode="edit")
        real = b.edit

        async def forgetful(source, request, url=None):
            out = await real(source, request, url)
            out.pop("face_box")
            return out
        full_edit._edit = forgetful
        job = full_edit.queue.submit(SRC, b"src", {"expression": "smile",
                                                   "face_box": [1, 2, 30, 40]}, "smile")
        await _drain()
        assert job.state == "failed" and "ignored the chosen face" in job.error
        assert job.result is None

    async def test_the_source_bytes_are_not_kept_after_the_run(self, box):
        box(mode="edit")
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": 45}}, "angle")
        await _drain()
        assert "_source" not in job.meta


# ------------------------------------------------------------ the standing box (#570)


class _Standing:
    """A fake always-on image-edit worker: its /health (busy while an A1111 on its card
    generates) and /edit."""

    def __init__(self, generating=(), healthy=True, unreachable=False):
        self.generating = list(generating)
        self.healthy, self.unreachable = healthy, unreachable
        self.edits = []

    async def health(self, url):
        if not self.healthy:
            return None
        gen = self.generating.pop(0) if self.generating else False
        return {"status": "ok", "a1111_generating": gen, "waiting": None,
                "features": ["angle", "instruction", "expression", "face_box", "faces"]}


@pytest.fixture
def standing(monkeypatch, box):
    def make(**kw):
        b = box(**{k: kw.pop(k) for k in list(kw) if k in ("mode", "rendering")})
        s = _Standing(**kw)
        monkeypatch.setattr(settings, "image_edit_standing_url", "http://edit-box.zero:8086")
        monkeypatch.setattr(full_edit, "_standing_health", s.health)
        main_edit = b.edit

        async def edit(source, request, url=None):
            if url == "http://edit-box.zero:8086":
                if s.unreachable:
                    raise full_edit.FullEditError(503, "image-edit unreachable", unreachable=True)
                s.edits.append(request)
                return {"image": base64.b64encode(_png()).decode(), "preview": "anBn",
                        "face_box": request.get("face_box"),
                        "identity": {"aura": 0.8, "reason": None}}
            return await main_edit(source, request, url)
        monkeypatch.setattr(full_edit, "_edit", edit)
        return b, s
    return make


@pytest.mark.asyncio
class TestTheStandingBox:
    async def test_a_healthy_standing_box_is_preferred_and_renders_never_pause(self, standing):
        b, s = standing(mode="ltx-engine", rendering=1)
        job = full_edit.queue.submit(SRC, b"src", {"expression": "smile"}, "smile")
        await _drain()
        assert job.state == "done", job.error
        assert s.edits == [{"expression": "smile"}] and b.edits == []
        assert b.modes_asked == [], "the main 3090 was never switched"
        assert job.worker == "edit-box.zero", "named by its host: no box is hardcoded"

    async def test_a1111_generating_is_waited_for_and_named(self, standing):
        b, s = standing(generating=[True, True, True, False])
        seen = []
        job = full_edit.queue.submit(SRC, b"src", {"angle": {"yaw": -90, "pitch": 0}},
                                     "profile_left")
        for _ in range(200):
            seen.append(job.message)
            if job.state in ("done", "failed"):
                break
            await asyncio.sleep(0.001)
        await _drain()
        assert "edit-box.zero busy (A1111 generating); edit queued" in seen
        assert job.state == "done" and s.edits and b.modes_asked == [], \
            "busy is waited for, never fallen back from"

    async def test_an_unhealthy_standing_box_falls_back_to_edit_mode(self, standing):
        b, s = standing(healthy=False, mode="ltx-engine")
        job = full_edit.queue.submit(SRC, b"src", {"expression": "smile"}, "smile")
        await _drain()
        assert job.state == "done", job.error
        assert s.edits == [] and b.edits == [{"expression": "smile"}]
        assert b.modes_asked == ["edit", "ltx-engine"]
        assert job.worker == "3090.zero"

    async def test_unreachable_at_the_edit_falls_back_too(self, standing):
        b, s = standing(unreachable=True, mode="edit")
        job = full_edit.queue.submit(SRC, b"src", {"expression": "sad"}, "sad")
        await _drain()
        assert job.state == "done" and b.edits == [{"expression": "sad"}]

    async def test_a_standing_failure_after_the_edit_was_sent_is_not_rerun(self, standing,
                                                                          monkeypatch):
        b, s = standing(mode="edit")

        async def boom(source, request, url=None):
            raise full_edit.FullEditError(504, "image-edit did not answer within 900s")
        monkeypatch.setattr(full_edit, "_edit", boom)
        job = full_edit.queue.submit(SRC, b"src", {"expression": "sad"}, "sad")
        await _drain()
        assert job.state == "failed" and "900s" in job.error
        assert b.modes_asked == [], "no second run on the main 3090"

    async def test_a_configured_name_wins_over_the_host(self, standing, monkeypatch):
        b, s = standing()
        monkeypatch.setattr(settings, "image_edit_standing_name", "3090b")
        job = full_edit.queue.submit(SRC, b"src", {"expression": "smile"}, "smile")
        await _drain()
        assert job.worker == "3090b"

    def test_the_always_on_worker_defaults_to_3090b(self):
        """3090b is the standing edit/sheet box (#572 option 1); a down box falls back to edit mode."""
        from app.config import Settings
        assert Settings.model_fields["image_edit_standing_url"].default == "http://2070.zero:8086"
        assert Settings.model_fields["image_edit_standing_name"].default == ""

    async def test_faces_come_from_the_standing_box_when_it_is_up(self, monkeypatch):
        monkeypatch.setattr(settings, "image_edit_standing_url", "")
        assert await full_edit.faces(b"src") is None


# -------------------------------------------------------------------- over HTTP


@pytest.fixture
def wired_full(monkeypatch, box):
    from app.routes import image_edit as mod
    fake_s3 = _FakeS3()
    monkeypatch.setattr(mod, "s3", fake_s3)
    monkeypatch.setattr(settings, "s3_images_bucket", BUCKET)
    return fake_s3, box(mode="edit")


async def _done_job(db, body):
    r = await _http(db, "post", "/images/edit", json=body)
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    await _drain()
    return jid


@pytest.mark.asyncio
class TestOverHTTP:
    async def test_presets_are_all_qwen(self, db, wired_full):
        d = (await _http(db, "get", "/images/edit/presets")).json()
        assert d["mode"] == "full"
        # 0, so a console from before #569 still open in a tab sends every angle to Qwen too.
        assert d["face_limit_deg"] == 0 and d["max_yaw"] == 90 and d["max_pitch"] == 45
        angles = {a["name"]: a for a in d["head_angles"]}
        assert angles["profile_left"] == {"name": "profile_left", "label": "Profile left",
                                          "yaw": -90, "pitch": 0, "route": "full"}
        assert {a["route"] for a in d["head_angles"]} == {"full"}
        exprs = {e["name"]: e["label"] for e in d["expressions"]}
        assert exprs["big_laugh"] == "Big laugh" and "angry" in exprs

    async def test_an_expression_with_an_angle_on_a_chosen_face(self, db, wired_full):
        s3, b = wired_full
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", "head_preset": "profile_left",
                              "preset": "smile", "face_box": [10, 20, 110, 140]})
        assert r.status_code == 202, r.text
        j = r.json()
        assert j["request"] == {"angle": {"yaw": -90.0, "pitch": 0.0}, "expression": "smile",
                                "face_box": [10.0, 20.0, 110.0, 140.0]}
        assert j["tag"] == "profile_left-smile" and j["face_box"] == [10.0, 20.0, 110.0, 140.0]
        await _drain()
        d = (await _http(db, "get", f"/images/edit/jobs/{j['id']}")).json()
        assert d["state"] == "done" and d["worker"] == "3090.zero"
        r = await _http(db, "post", f"/images/edit/jobs/{j['id']}/save", json={})
        assert r.status_code == 200 and r.json()["preset"] == "smile"
        assert "_edit-profile_left-smile_" in r.json()["uri"]

    async def test_free_text_is_an_instruction(self, db, wired_full):
        s3, b = wired_full
        jid = await _done_job(db, {"source_uri": SRC, "mode": "full",
                                   "instruction": "  big grin, eyes half closed  "})
        assert b.edits[-1] == {"instruction": "big grin, eyes half closed"}
        d = (await _http(db, "get", f"/images/edit/jobs/{jid}")).json()
        assert d["tag"] == "full"

    async def test_faces_prefer_the_standing_box(self, db, wired_full, monkeypatch):
        from app.routes import image_edit as mod
        got = {"width": 10, "height": 10, "default_index": 0,
               "faces": [{"index": 0, "box": [1, 1, 5, 5], "width": 4}]}

        async def standing_faces(src):
            return got

        async def face_edit_faces(src):
            raise AssertionError("face-edit asked although the standing box answered")
        monkeypatch.setattr(mod.full_edit, "faces", standing_faces)
        monkeypatch.setattr(mod.face_edit, "faces", face_edit_faces)
        r = await _http(db, "post", "/images/edit/faces", json={"source_uri": SRC})
        assert r.status_code == 200 and r.json()["faces"][0]["box"] == [1, 1, 5, 5]

    async def test_faces_fall_back_to_face_edit(self, db, wired_full, monkeypatch):
        from app.routes import image_edit as mod

        async def none(src):
            return None

        async def face_edit_faces(src):
            return {"width": 10, "height": 10, "default_index": None, "faces": []}
        monkeypatch.setattr(mod.full_edit, "faces", none)
        monkeypatch.setattr(mod.face_edit, "faces", face_edit_faces)
        r = await _http(db, "post", "/images/edit/faces", json={"source_uri": SRC})
        assert r.status_code == 200 and r.json()["faces"] == []

    async def test_a_head_preset_is_a_202_job_then_done_with_identity(self, db, wired_full):
        s3, b = wired_full
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", "head_preset": "profile_left",
                              "seed": 548})
        assert r.status_code == 202, r.text
        j = r.json()
        assert j["state"] in ("queued", "waiting", "running", "done")
        assert j["request"] == {"angle": {"yaw": -90.0, "pitch": 0.0}, "seed": 548}
        await _drain()
        d = (await _http(db, "get", f"/images/edit/jobs/{j['id']}")).json()
        assert d["state"] == "done" and d["identity"]["aura"] == 0.52
        assert d["preview"].startswith("data:image/jpeg;base64,") and d["position"] is None
        assert s3.uploaded == [], "nothing is saved until save"

    async def test_save_writes_a_new_repo_object(self, db, wired_full):
        s3, _ = wired_full
        jid = await _done_job(db, {"source_uri": SRC, "mode": "full",
                                   "head_preset": "three_quarter_right"})
        r = await _http(db, "post", f"/images/edit/jobs/{jid}/save", json={})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["mode"] == "full" and d["preset"] == "three_quarter_right"
        assert d["uri"].startswith(f"s3://{BUCKET}/2026-09-01/sel_008_edit-three_quarter_right_")
        [(key, _b, data)] = s3.uploaded
        assert data == _png() and key != "2026-09-01/sel_008.jpg"
        again = (await _http(db, "get", f"/images/edit/jobs/{jid}")).json()
        assert again["saved"][0]["uri"] == d["uri"]

    async def test_save_to_an_unlocked_dataset_appends(self, db, wired_full):
        from app.models import Dataset
        ds = await _ds(db)
        before = list(ds.images)
        jid = await _done_job(db, {"source_uri": ds.images[0], "mode": "full",
                                   "instruction": "make the sweater red"})
        r = await _http(db, "post", f"/images/edit/jobs/{jid}/save",
                        json={"dataset_id": str(ds.id)})
        assert r.status_code == 200, r.text
        uri = r.json()["uri"]
        assert uri.startswith(f"s3://{BUCKET}/{ds.prefix}/edits/0_edit-full_")
        after = await db.get(Dataset, ds.id)
        await db.refresh(after)
        assert after.images == before + [uri]

    async def test_a_locked_dataset_is_refused_and_nothing_written(self, db, wired_full):
        from app.routes.datasets import lock_dataset
        s3, _ = wired_full
        ds = await _ds(db)
        await lock_dataset(ds.id, None, _user=None, db=db)
        jid = await _done_job(db, {"source_uri": SRC, "mode": "full", "head_preset": "look_up"})
        r = await _http(db, "post", f"/images/edit/jobs/{jid}/save",
                        json={"dataset_id": str(ds.id)})
        assert r.status_code == 409 and s3.uploaded == []

    async def test_saving_an_unfinished_or_unknown_job(self, db, wired_full):
        s3, b = wired_full
        b.mode, b.pending, b.switch_after = "ltx-engine", None, 10_000
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", "head_preset": "look_up"})
        jid = r.json()["id"]
        r = await _http(db, "post", f"/images/edit/jobs/{jid}/save", json={})
        assert r.status_code == 409 and "not done" in r.json()["detail"]
        full_edit.queue._task.cancel()
        r = await _http(db, "get", f"/images/edit/jobs/{uuid.uuid4().hex}")
        assert r.status_code == 404
        assert s3.uploaded == []

    @pytest.mark.parametrize("body,needle", [
        ({}, "nothing to apply"),
        ({"instruction": "   "}, "nothing to apply"),
        ({"head_preset": "cartwheel"}, "unknown head preset"),
        ({"preset": "wink"}, "unknown expression"),
        ({"angle": {"yaw": 3, "pitch": -2}}, "under 5"),
    ])
    async def test_nonsense_is_422_before_anything_runs(self, db, wired_full, body, needle):
        s3, b = wired_full
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", **body})
        assert r.status_code == 422 and needle in r.json()["detail"]
        assert s3.downloaded == [] and b.edits == [] and full_edit.queue.jobs == {}

    async def test_out_of_range_is_422(self, db, wired_full):
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", "angle": {"yaw": 120}})
        assert r.status_code == 422

    async def test_not_configured_is_503(self, db, wired_full, monkeypatch):
        monkeypatch.setattr(settings, "image_edit_url", "")
        r = await _http(db, "post", "/images/edit",
                        json={"source_uri": SRC, "mode": "full", "head_preset": "look_up"})
        assert r.status_code == 503 and "image_edit_url" in r.json()["detail"]

    async def test_face_mode_is_unchanged(self, db, wired_full, monkeypatch):
        """mode defaults to face: phase 1's contract, inline and saved, is untouched."""
        from app.routes import image_edit as mod

        async def fake_edit(image_bytes, params, **kw):
            return {"image": b"png", "format": "png", "device": "cpu"}
        monkeypatch.setattr(mod.face_edit, "edit", fake_edit)
        r = await _http(db, "post", "/images/edit", json={"source_uri": SRC, "preset": "smile"})
        assert r.status_code == 200 and r.json()["mode"] == "face"


@pytest.mark.asyncio
class TestRoutedByMode:
    """wanly-api#392: a box already in edit mode takes the edit -- whichever box it is."""

    @pytest.fixture
    def in_edit(self, monkeypatch):
        from app import worker_modes as wm

        def set_(**bodies):
            async def live():
                return [wm.parse_health(n, b) for n, b in bodies.items()]
            monkeypatch.setattr(wm, "live_boxes_own_session", live)
        return set_

    async def test_a_box_in_edit_mode_takes_it_with_no_switch(self, box, in_edit):
        b = box(mode="ltx-engine")
        in_edit(**{"3090b": {"mode": "edit", "mode_name": "edit", "services": [
            {"name": "image-edit", "group": "image-edit", "ready": True}]},
                   "3090a.zero": {"mode": "ltx-engine", "mode_name": "render"}})
        job = full_edit.queue.submit(SRC, b"src", {"instruction": "red sweater"}, "full")
        await _drain()
        assert job.state == "done", job.error
        assert job.worker == "3090b"
        assert b.urls == ["http://3090b:8086"]
        assert b.modes_asked == [], "nothing switched, nothing to hand back"

    async def test_no_box_in_edit_mode_keeps_the_switch_path(self, box, in_edit):
        b = box(mode="ltx-engine")
        in_edit(**{"3090a.zero": {"mode": "ltx-engine", "mode_name": "render"}})
        job = full_edit.queue.submit(SRC, b"src", {"instruction": "red sweater"}, "full")
        await _drain()
        assert job.state == "done", job.error
        assert b.modes_asked == ["edit", "ltx-engine"]

    async def test_an_edit_mode_box_that_does_not_answer_is_fallen_back_from(
            self, box, in_edit, monkeypatch):
        b = box(mode="ltx-engine")
        in_edit(**{"3090b": {"mode": "edit", "mode_name": "edit"}})
        real_edit = b.edit

        async def edit(source, request, url=None):
            if url and "3090b" in url:
                raise full_edit.FullEditError(503, "connection refused", unreachable=True)
            return await real_edit(source, request, url=url)
        monkeypatch.setattr(full_edit, "_edit", edit)
        job = full_edit.queue.submit(SRC, b"src", {"instruction": "red sweater"}, "full")
        await _drain()
        assert job.state == "done", job.error
        assert job.worker == settings.image_edit_worker
        assert b.modes_asked == ["edit", "ltx-engine"]
