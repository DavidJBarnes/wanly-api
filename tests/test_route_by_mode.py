"""Pushed work goes to a box in the matching mode (wanly-api#392).

Motion captions and image edits are PUSHED to a URL. With two symmetric 3090s, each in one
mode at a time (wanly-gpu-docker#164), the URL is whichever box is in the mode right now --
read from the boxes, never configured per box, never a box in another mode.
"""
import pytest

from app import worker_modes as wm
from app.config import settings
from app.joycaption import NoGpuInMode
from app.routes import captions


def _health(mode_name=None, mode=None, pending_name=None, services=None, modes=None, **kw):
    body = {"mode": mode, "mode_name": mode_name, "pending_mode_name": pending_name,
            "modes": modes or ["render", "train", "motion", "edit"],
            "services": services if services is not None else [
                {"name": "image-description", "group": "image-description", "ready": True},
                {"name": "image-edit", "group": "image-edit", "ready": True}]}
    body.update(kw)
    return body


def _boxes(**bodies):
    return [wm.parse_health(name, body) for name, body in bodies.items()]


class TestReadingABox:
    def test_both_spellings_mean_the_same_mode(self):
        assert wm.canonical("ltx-engine") == "render"
        assert wm.canonical("caption") == "motion"
        assert wm.canonical("motion") == "motion"
        assert wm.canonical("nonsense") is None
        assert wm.canonical(None) is None

    def test_a_box_from_before_164_is_read_from_its_old_mode(self):
        b = wm.parse_health("3090a", {"mode": "caption", "pending_mode": "ltx-engine"})
        assert (b.mode, b.pending, b.reachable) == ("motion", "render", True)

    def test_the_four_mode_name_wins_over_the_old_one(self):
        b = wm.parse_health("3090a", {"mode": "train", "mode_name": "train", "modes": ["render",
                                      "train", "caption"]})
        assert b.mode == "train"
        assert b.modes == ["render", "train", "motion"]

    def test_no_answer_is_a_box_that_reports_nothing(self):
        b = wm.parse_health("3090a", None)
        assert b.reachable is False and b.mode is None


class TestChoosing:
    def test_the_box_in_the_mode_is_chosen(self):
        p = wm.choose("motion", _boxes(**{"3090a": _health("render"),
                                          "3090b": _health("motion")}))
        assert p.box == "3090b" and p.wait is None

    def test_two_boxes_in_the_mode_take_turns(self):
        boxes = _boxes(**{"3090a": _health("motion"), "3090b": _health("motion")})
        picked = {wm.choose("motion", boxes).box for _ in range(4)}
        assert picked == {"3090a", "3090b"}

    def test_no_box_in_the_mode_waits_and_says_where_everyone_is(self):
        p = wm.choose("motion", _boxes(**{"3090a": _health("render"),
                                          "3090b": _health("edit")}))
        assert p.box is None and p.reporting is True
        assert p.wait.startswith("no GPU in motion mode")
        assert "3090a: render" in p.wait and "3090b: edit" in p.wait

    def test_a_box_switching_away_is_not_used(self):
        p = wm.choose("motion", _boxes(**{"3090a": _health("motion", pending_name="render")}))
        assert p.box is None

    def test_a_box_switching_to_the_mode_is_the_reason(self):
        p = wm.choose("motion", _boxes(**{"3090a": _health("render", pending_name="motion")}))
        assert p.wait == "3090a is switching to motion mode"

    def test_a_box_in_the_mode_whose_service_is_down_is_not_used(self):
        p = wm.choose("motion", _boxes(**{"3090a": _health("motion", services=[
            {"name": "image-description", "group": "image-description", "ready": False}])}))
        assert p.box is None and "starting" in p.wait

    def test_nothing_reporting_means_the_old_way(self):
        p = wm.choose("motion", [wm.parse_health("3090a", None)])
        assert p.reporting is False and p.box is None and p.wait is None

    def test_the_url_takes_the_configured_port(self, monkeypatch):
        monkeypatch.setattr(settings, "motion_caption_url", "")
        monkeypatch.setattr(settings, "image_description_url", "http://3090a.zero:11435")
        monkeypatch.setattr(settings, "image_edit_url", "http://3090.zero:8086")
        assert wm.motion_url("3090b") == "http://3090b:11435"
        assert wm.edit_url("3090b") == "http://3090b:8086"


@pytest.fixture
def boxes(monkeypatch):
    """Set what the live boxes report: boxes({'3090a': body, ...})."""
    state = {}

    async def live(db):
        return [wm.parse_health(n, b) for n, b in state.items()]
    monkeypatch.setattr(wm, "live_boxes", live)
    monkeypatch.setattr(settings, "motion_caption_url", "")
    monkeypatch.setattr(settings, "image_description_url", "http://3090a.zero:11435")
    monkeypatch.setattr(settings, "image_description_fallback_url", "")

    def set_(new):
        state.clear()
        state.update(new)
    return set_


@pytest.mark.asyncio
class TestMotionCaptions:
    async def test_they_go_to_the_box_in_motion_mode(self, boxes):
        boxes({"3090a.zero": _health("render"), "3090b": _health("motion")})
        assert await captions._caption_base(None, True) == "http://3090b:11435"

    async def test_none_in_motion_mode_raises_the_wait_not_a_connection_failure(self, boxes):
        boxes({"3090a.zero": _health("render"), "3090b": _health("render")})
        with pytest.raises(NoGpuInMode) as e:
            await captions._caption_base(None, True)
        assert "no GPU in motion mode" in str(e.value)
        assert e.value.mode_wait is True

    async def test_claim_time_waits_the_same_way(self, boxes):
        boxes({"3090a.zero": _health("render")})
        with pytest.raises(NoGpuInMode):
            await captions._caption_base(None, False)

    async def test_a_configured_fallback_takes_them_when_nobody_is_in_motion_mode(
            self, boxes, monkeypatch):
        monkeypatch.setattr(settings, "image_description_fallback_url", "http://2070.zero:11434")
        boxes({"3090a.zero": _health("render")})
        assert await captions._caption_base(None, True) == "http://2070.zero:11434"

    async def test_a_box_flipping_mode_mid_wait_takes_the_next_ask(self, boxes):
        boxes({"3090a.zero": _health("render", pending_name="motion")})
        with pytest.raises(NoGpuInMode) as e:
            await captions._caption_base(None, True)
        assert "switching to motion" in str(e.value)
        boxes({"3090a.zero": _health("motion")})
        assert await captions._caption_base(None, True) == "http://3090a.zero:11435"

    async def test_with_nothing_reporting_the_configured_captioner_is_used(
            self, boxes, monkeypatch):
        boxes({"3090a.zero": None})

        async def not_busy(db):
            return None
        monkeypatch.setattr(captions, "busy_render_beside_the_captioner", not_busy)
        assert await captions._caption_base(None, True) == "http://3090a.zero:11435"


@pytest.mark.asyncio
class TestTheCache:
    async def test_a_box_is_asked_once_per_ttl_and_forget_asks_again(self, monkeypatch):
        calls = []

        async def fetch(name):
            calls.append(name)
            return _health("motion")
        monkeypatch.setattr(wm, "_fetch", fetch)
        await wm.box_state("3090b")
        await wm.box_state("3090B")
        assert calls == ["3090b"]
        wm.forget("3090b")
        await wm.box_state("3090b")
        assert calls == ["3090b", "3090b"]


@pytest.mark.asyncio
class TestTheSummary:
    """GET /worker-modes: every box's mode and what waits on each mode (console#589)."""

    async def _get(self, db):
        from httpx import ASGITransport, AsyncClient

        from app.auth import verify_api_key_or_bearer
        from app.database import get_db
        from app.main import app
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
        try:
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url="http://test") as c:
                r = await c.get("/worker-modes")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200, r.text
        return r.json()

    async def test_boxes_and_waits(self, db, monkeypatch):
        import uuid as _uuid

        from app import caption_hold
        from app.enums import JobStatus, SegmentStatus
        from app.models import Job, Segment, User, Worker

        for name in ("3090a.zero", "3090b"):
            db.add(Worker(friendly_name=name, hostname="h", ip_address="172.17.0.2",
                          status="online-idle"))
        db.add(Worker(friendly_name="gone", hostname="h", ip_address="x", status="offline"))
        user = User(username=str(_uuid.uuid4()), password_hash="x")
        db.add(user)
        await db.flush()
        job = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
                  status=JobStatus.PENDING)
        db.add(job)
        await db.flush()
        db.add(Segment(job_id=job.id, index=0, prompt="p", status=SegmentStatus.PENDING))
        await db.flush()

        bodies = {"3090a.zero": _health("render", mode="ltx-engine",
                                       gpu={"vram_used_mib": 23100, "vram_total_mib": 24576}),
                  "3090b": _health("edit", mode="edit")}

        async def fetch(name):
            return bodies.get(name)
        monkeypatch.setattr(wm, "_fetch", fetch)
        monkeypatch.setitem(caption_hold._mode_waits, "s3://b/x.png",
                            "no GPU in motion mode (3090a.zero: render, 3090b: edit)")

        out = await self._get(db)
        names = [b["friendly_name"] for b in out["boxes"]]
        assert names == ["3090a.zero", "3090b"], "offline boxes are left out"
        a = out["boxes"][0]
        assert a["mode_name"] == "render" and a["reachable"] is True
        assert a["gpu"]["vram_used_mib"] == 23100
        waits = {w["mode"]: w for w in out["waiting"]}
        assert waits["render"]["count"] >= 1 and waits["render"]["reason"] is None
        assert waits["motion"]["count"] == 1
        assert waits["motion"]["reason"].startswith("no GPU in motion mode")
        assert waits["edit"]["count"] == 0
        assert set(out["scene"]) >= {"depth", "up"}

    async def test_render_work_with_no_render_box_says_so(self, db, monkeypatch):
        import uuid as _uuid

        from app.enums import JobStatus, SegmentStatus
        from app.models import Job, Segment, User, Worker
        db.add(Worker(friendly_name="3090b", hostname="h", ip_address="x", status="online-idle"))
        user = User(username=str(_uuid.uuid4()), password_hash="x")
        db.add(user)
        await db.flush()
        job = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
                  status=JobStatus.PENDING)
        db.add(job)
        await db.flush()
        db.add(Segment(job_id=job.id, index=0, prompt="p", status=SegmentStatus.PENDING))
        await db.flush()

        async def fetch(name):
            return _health("motion", mode="caption")
        monkeypatch.setattr(wm, "_fetch", fetch)
        waits = {w["mode"]: w for w in (await self._get(db))["waiting"]}
        assert waits["render"]["reason"].startswith("no GPU in render mode")
