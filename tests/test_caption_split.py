"""Two captioners, one per half (wanly-console#572 phase 1).

The scene goes to JoyCaption on the scene-caption service; the motion paragraph to the shared
captioner that used to do both (image_description_*). Pinned here:

  * the motion captioner inherits every image_description_* value it does not set, so nothing
    changes until the new settings are set;
  * the scene service is never refused for a render, and when it is down the scene falls back
    to the shared captioner -- loudly, and without paying a connect timeout per caption;
  * the render refusal matches the MOTION captioner's host, never the scene service's.
"""
import logging
import uuid

import httpx
import pytest

from app import joycaption as jc
from app.config import Settings, settings
from app.joycaption import CaptionerBusy
from app.models import Worker
from app.routes import captions


@pytest.fixture
def unsplit(monkeypatch):
    monkeypatch.setattr(settings, "image_description_url", "http://3090a.zero:11435")
    monkeypatch.setattr(settings, "image_description_model", "qwen3-vl:32b")
    monkeypatch.setattr(settings, "image_description_keep_alive", "15m")
    monkeypatch.setattr(settings, "image_description_num_ctx", 4096)
    monkeypatch.setattr(settings, "image_description_timeout_s", 600)
    for k in ("motion_caption_url", "motion_caption_model", "motion_caption_keep_alive"):
        monkeypatch.setattr(settings, k, "")
    monkeypatch.setattr(settings, "motion_caption_num_ctx", 0)
    monkeypatch.setattr(settings, "motion_caption_timeout_s", 0)


@pytest.fixture
def split(unsplit, monkeypatch):
    monkeypatch.setattr(settings, "scene_caption_url", "http://3090b.zero:11436")
    monkeypatch.setattr(settings, "scene_caption_model", "joycaption:beta-one")
    monkeypatch.setattr(settings, "scene_caption_keep_alive", "-1")


class _Recorder:
    """httpx.AsyncClient stand-in: records each POST, answers from `answer(url)`."""
    def __init__(self, answer):
        self.answer = answer
        self.posts: list[tuple[str, dict]] = []

    def __call__(self, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self.posts.append((url, json))
        return self.answer(url)


class _Resp:
    def __init__(self, status=200, text="a woman on a pier"):
        self.status_code, self.text = status, text

    def json(self):
        return {"response": self.text}


def _down_for(host):
    def answer(url):
        if host in url:
            raise httpx.ConnectError("connection refused")
        return _Resp()
    return answer


class TestTheSettings:
    def test_the_defaults_point_the_scene_at_3090b_and_inherit_the_motion(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("JWT_SECRET", "x")
        s = Settings(_env_file=None)
        assert s.scene_caption_url == "http://3090b.zero:11436"
        assert s.scene_caption_model == "joycaption:beta-one"
        assert s.scene_caption_keep_alive == "-1"
        # Empty = inherit image_description_* (what EC2's env pins to 3090a's captioner).
        assert s.motion_caption_url == "" and s.motion_caption_model == ""

    def test_the_motion_captioner_inherits_every_unset_value(self, unsplit):
        cap = jc.motion_captioner()
        assert (cap.url, cap.model, cap.keep_alive, cap.num_ctx, cap.timeout_s) == (
            "http://3090a.zero:11435", "qwen3-vl:32b", "15m", 4096, 600)
        assert cap.beside_render is True

    def test_motion_settings_override_one_value_at_a_time(self, unsplit, monkeypatch):
        monkeypatch.setattr(settings, "motion_caption_url", "http://3090b.zero:11435/")
        monkeypatch.setattr(settings, "motion_caption_num_ctx", 8192)
        cap = jc.motion_captioner()
        assert cap.url == "http://3090b.zero:11435" and cap.num_ctx == 8192
        assert cap.model == "qwen3-vl:32b"

    def test_an_empty_scene_url_turns_the_split_off(self, unsplit):
        assert jc.scene_captioner() is None

    def test_the_scene_captioner_never_shares_a_render_card(self, split):
        cap = jc.scene_captioner()
        assert cap.url == "http://3090b.zero:11436" and cap.beside_render is False


class TestTheRequest:
    async def test_each_half_is_sent_its_own_model_and_keep_alive(self, split, monkeypatch):
        rec = _Recorder(lambda url: _Resp())
        monkeypatch.setattr(jc.httpx, "AsyncClient", rec)
        await jc.describe(b"img", "scene please", captioner=jc.scene_captioner())
        await jc.describe(b"img", "motion please")
        (u1, p1), (u2, p2) = rec.posts
        assert u1 == "http://3090b.zero:11436/api/generate"
        assert p1["model"] == "joycaption:beta-one" and p1["keep_alive"] == -1
        assert u2 == "http://3090a.zero:11435/api/generate"
        assert p2["model"] == "qwen3-vl:32b" and p2["keep_alive"] == "15m"


@pytest.mark.asyncio
class TestSceneFallback:
    async def test_the_scene_service_is_used_and_the_busy_box_is_never_asked(
            self, split, monkeypatch):
        rec = _Recorder(lambda url: _Resp())
        monkeypatch.setattr(jc.httpx, "AsyncClient", rec)

        async def refused():
            raise CaptionerBusy("3090a.zero is in render mode")
        text = await captions.describe_scene(b"img", "describe", refused)
        assert text == "a woman on a pier"
        assert [u for u, _ in rec.posts] == ["http://3090b.zero:11436/api/generate"]

    async def test_a_down_scene_service_falls_back_loudly_and_is_skipped_for_a_while(
            self, split, monkeypatch, caplog):
        rec = _Recorder(_down_for("3090b"))
        monkeypatch.setattr(jc.httpx, "AsyncClient", rec)
        asked = []

        async def fallback():
            asked.append(1)
            return "http://3090a.zero:11435"
        with caplog.at_level(logging.INFO):
            assert await captions.describe_scene(b"img", "d", fallback) == "a woman on a pier"
            assert await captions.describe_scene(b"img", "d", fallback) == "a woman on a pier"
        urls = [u for u, _ in rec.posts]
        # First caption: tried the scene service, then the fallback. Second: straight to the
        # fallback -- the service is marked down, no connect timeout paid again.
        assert urls == ["http://3090b.zero:11436/api/generate",
                        "http://3090a.zero:11435/api/generate",
                        "http://3090a.zero:11435/api/generate"]
        # The fallback asks the shared captioner for its own model, not JoyCaption.
        assert rec.posts[1][1]["model"] == "qwen3-vl:32b"
        assert len(asked) == 2
        assert "Scene captioner http://3090b.zero:11436 is DOWN" in caplog.text
        assert jc.scene_status()["up"] is False

    async def test_it_is_tried_again_once_the_down_window_passes(self, split, monkeypatch):
        rec = _Recorder(_down_for("3090b"))
        monkeypatch.setattr(jc.httpx, "AsyncClient", rec)

        async def fallback():
            return "http://3090a.zero:11435"
        await captions.describe_scene(b"img", "d", fallback)
        jc._scene_down["until"] = 0.0              # the window has passed
        rec.answer = lambda url: _Resp()
        await captions.describe_scene(b"img", "d", fallback)
        assert rec.posts[-1][0] == "http://3090b.zero:11436/api/generate"
        assert jc.scene_status()["up"] is True and jc._scene_down["why"] is None

    async def test_an_empty_caption_is_not_a_fallback(self, split, monkeypatch):
        """The service answered; a blank answer is the caller's to refuse, not a reason to
        ask a different model."""
        rec = _Recorder(lambda url: _Resp(text=""))
        monkeypatch.setattr(jc.httpx, "AsyncClient", rec)

        async def fallback():
            raise AssertionError("must not fall back on an empty caption")
        with pytest.raises(jc.CaptionError, match="empty"):
            await captions.describe_scene(b"img", "d", fallback)


class TestTheRefusalIsTheMotionCaptioners:
    def _w(self, name):
        return Worker(id=uuid.uuid4(), friendly_name=name, hostname="h", ip_address="1.2.3.4",
                      kind="render", kinds=["render"], status="online-busy")

    def test_the_motion_host_is_matched_and_the_scene_host_never_is(self, split, monkeypatch):
        monkeypatch.setattr(settings, "motion_caption_url", "http://3090a.zero:11435")
        a, b = self._w("3090a.zero"), self._w("3090b.zero")
        assert jc.captioner_host() == "3090a.zero"
        assert jc.render_worker_beside_the_captioner([b, a]) is a
        assert jc.render_worker_beside_the_captioner([b]) is None
