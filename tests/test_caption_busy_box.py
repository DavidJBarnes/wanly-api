"""An interactive caption is refused while the box it shares a card with is rendering.

wanly-gpu-docker#83 put image-description inside the GPU container on the 3090. A caption
loads a ~6 GB vision model; a 720p render holds ~23 of 24 GB. Until the local VRAM lease,
the API keeps them apart on the interactive path by refusing with the box's name -- not by
waiting, because the request sits behind a 60 s proxy timeout and a wait would read as a
dead captioner.
"""
import uuid

import pytest

from app.config import settings
from app.joycaption import captioner_host, render_worker_beside_the_captioner
from app.models import Worker


def _w(name, status="online-idle", kind="render", kinds=None, provides=None):
    return Worker(id=uuid.uuid4(), friendly_name=name, hostname="h", ip_address="1.2.3.4",
                  kind=kind, kinds=kinds, provides=provides, status=status)


@pytest.fixture(autouse=True)
def _url(monkeypatch):
    monkeypatch.setattr(settings, "image_description_url", "http://3090.zero:11434")


class TestWhichRowSharesTheCard:
    def test_the_host_comes_from_the_url(self):
        assert captioner_host() == "3090.zero"

    def test_the_row_that_says_it_provides_the_captioner_wins(self):
        """One row per box: the box itself reports what it runs, and that beats a name."""
        box = _w("3090.zero", kinds=["render", "trainer"],
                 provides=["ltx-engine", "lora-trainer", "image-description"])
        other = _w("3090.zero-old")
        assert render_worker_beside_the_captioner([other, box]) is box

    def test_a_row_named_for_the_url_host_is_the_fallback(self):
        """A daemon that does not report provides yet."""
        box = _w("3090.zero", provides=None)
        pod = _w("runpod-abc", provides=None)
        assert render_worker_beside_the_captioner([pod, box]) is box

    def test_a_captioner_only_box_is_never_a_collision(self):
        """The 2070-style deployment: a service row that cannot render has nothing to wait
        for, whatever it provides."""
        svc = _w("2070.zero/services", kind="service", kinds=["service"],
                 provides=["image-description"], status="online")
        assert render_worker_beside_the_captioner([svc]) is None

    def test_nothing_matches_means_nothing_to_refuse_on(self):
        assert render_worker_beside_the_captioner([_w("runpod-abc")]) is None


class TestTheRefusal:
    def test_the_interactive_path_refuses_by_name_and_claim_time_does_not(self):
        import inspect
        from app.routes import captions, segments
        src = inspect.getsource(captions.caption_image_bytes)
        assert "busy_render_beside_the_captioner(db) if interactive else None" in src
        assert "base = captioner_for(busy, interactive)" in src
        assert "raise CaptionerBusy" in src
        assert "describe(image, instruction, base_url=base)" in src
        # Claim-time <SCENE> resolution opts out: the worker was just handed the segment and
        # has not loaded the render; a failed caption there is non-fatal by design.
        assert "caption_image_bytes(db, image, interactive=False)" in inspect.getsource(segments)

    def test_it_is_a_caption_error_so_the_route_answers_503_with_the_message(self):
        from app.joycaption import CaptionError, CaptionerBusy
        assert issubclass(CaptionerBusy, CaptionError)

    async def test_a_busy_box_is_named_and_an_idle_one_is_not(self, db):
        from app.joycaption import busy_render_beside_the_captioner
        box = _w("3090.zero", status="online-busy", kinds=["render", "trainer"],
                 provides=["ltx-engine", "image-description"])
        db.add(box); await db.commit()
        assert await busy_render_beside_the_captioner(db) == "3090.zero"
        box.status = "online-idle"; await db.commit()
        assert await busy_render_beside_the_captioner(db) is None


class TestTheConfigIsNamedForTheCapability:
    def test_the_old_names_are_still_read_from_the_environment(self, monkeypatch):
        """JOYCAPTION_URL in an existing .env keeps working for one release."""
        from app.config import Settings
        monkeypatch.setenv("JOYCAPTION_URL", "http://old:11434")
        monkeypatch.setenv("DATABASE_URL", "sqlite://"); monkeypatch.setenv("JWT_SECRET", "x")
        assert Settings(_env_file=None).image_description_url == "http://old:11434"
        monkeypatch.setenv("IMAGE_DESCRIPTION_URL", "http://new:11434")
        assert Settings(_env_file=None).image_description_url == "http://new:11434"


class TestWhichCaptionerIsUsed:
    """The 2070's captioner is the fallback while the 3090 renders (the first night's
    "Request failed with status code 503" on every describe during a render)."""

    def test_idle_box_interactive_uses_the_primary(self, monkeypatch):
        from app.joycaption import captioner_for
        monkeypatch.setattr(settings, "image_description_fallback_url", "http://2070.zero:11434")
        assert captioner_for(None, True) == "http://3090.zero:11434"

    def test_busy_box_interactive_uses_the_fallback(self, monkeypatch):
        from app.joycaption import captioner_for
        monkeypatch.setattr(settings, "image_description_fallback_url", "http://2070.zero:11434")
        assert captioner_for("3090.zero", True) == "http://2070.zero:11434"

    def test_busy_box_with_no_fallback_refuses(self, monkeypatch):
        from app.joycaption import captioner_for
        monkeypatch.setattr(settings, "image_description_fallback_url", "")
        assert captioner_for("3090.zero", True) is None

    def test_claim_time_prefers_the_fallback(self, monkeypatch):
        """The claiming box is about to load a 23 GB render; a caption racing it timed out."""
        from app.joycaption import captioner_for
        monkeypatch.setattr(settings, "image_description_fallback_url", "http://2070.zero:11434")
        assert captioner_for(None, False) == "http://2070.zero:11434"
        monkeypatch.setattr(settings, "image_description_fallback_url", "")
        assert captioner_for(None, False) == "http://3090.zero:11434"

    def test_describe_honours_the_base_url(self):
        import inspect
        from app import joycaption
        src = inspect.getsource(joycaption.describe)
        assert 'base = (base_url or settings.image_description_url).rstrip("/")' in src
        assert 'url = f"{base}/api/generate"' in src
