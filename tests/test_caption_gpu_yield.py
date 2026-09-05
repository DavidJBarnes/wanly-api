"""A caption asks Automatic1111 for the GPU back, once, and only when that would help.

The 2070 hosts both. JoyCaption needs ~5970 MiB of an 8192 MiB card; A1111 sitting idle
with a checkpoint loaded holds ~1800. The load dies allocating the vision projector and
ollama reports it as `llama runner process has terminated: %!w(<nil>)` — a 500 that names
neither the GPU nor the memory, which is why this is worth a test rather than a comment.

Every case here is about NOT overreaching: one retry, never mid-generation, and no retry at
all when nothing was actually freed.
"""
import httpx
import pytest

from app.config import settings
from app.joycaption import CaptionError, describe

RUNNER_DIED = '{"error":"llama runner process has terminated: %!w(<nil>)"}'


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _Client:
    """Stands in for httpx.AsyncClient. One queue of responses, one log of calls."""

    calls: list = []
    responses: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _Client.calls.append(("POST", url))
        return _Client.responses.pop(0)

    async def get(self, url):
        _Client.calls.append(("GET", url))
        return _Client.responses.pop(0)


def _install(monkeypatch, *responses):
    _Client.calls = []
    _Client.responses = list(responses)
    monkeypatch.setattr(settings, "a1111_url", "http://2070.test:7860")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())


def _caption(text="a woman on a sofa"):
    return _Resp(payload={"response": text})


def _idle():
    return _Resp(payload={"state": {"job_count": 0}})


def _generating():
    return _Resp(payload={"state": {"job_count": 1}})


def _urls(method=None):
    return [u for m, u in _Client.calls if method is None or m == method]


class TestTheHappyPath:
    async def test_a_caption_that_works_never_touches_a1111(self, monkeypatch):
        """The fast path must stay free. A burst of captions coalesces inside the 5s
        keep_alive, and unloading A1111 before each one would thrash a checkpoint reload for
        nothing."""
        _install(monkeypatch, _caption())
        assert await describe(b"img", "describe this") == "a woman on a sofa"
        assert all("7860" not in u for u in _urls())


class TestYieldingOnFailure:
    async def test_a_500_frees_the_card_and_retries_once(self, monkeypatch):
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED), _idle(),
                 _Resp(), _caption())
        assert await describe(b"img", "d") == "a woman on a sofa"
        assert _urls() == [
            "http://2070.test:11434/api/generate",
            "http://2070.test:7860/sdapi/v1/progress",
            "http://2070.test:7860/sdapi/v1/unload-checkpoint",
            "http://2070.test:11434/api/generate",
        ]

    async def test_the_retry_is_not_a_loop(self, monkeypatch):
        """A second failure is a real failure. Freeing the card twice would just be two
        checkpoint reloads for one caption nobody is going to get."""
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED), _idle(),
                 _Resp(), _Resp(status=500, text=RUNNER_DIED))
        with pytest.raises(CaptionError, match="500"):
            await describe(b"img", "d")
        assert _urls().count("http://2070.test:11434/api/generate") == 2


class TestNotOverreaching:
    async def test_a_generating_a1111_is_left_alone(self, monkeypatch):
        """Unloading mid-generation takes down somebody's image to caption a frame. The
        caption has a fallback; the generation does not."""
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED), _generating())
        with pytest.raises(CaptionError, match="500"):
            await describe(b"img", "d")
        assert "http://2070.test:7860/sdapi/v1/unload-checkpoint" not in _urls()

    async def test_no_a1111_configured_means_no_retry(self, monkeypatch):
        """Anywhere the captioner does not share a card, there is nothing to ask — and a
        retry against an unchanged GPU would fail identically."""
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED))
        monkeypatch.setattr(settings, "a1111_url", "")
        with pytest.raises(CaptionError, match="500"):
            await describe(b"img", "d")
        assert _urls() == ["http://2070.test:11434/api/generate"]

    async def test_an_unreachable_a1111_is_not_an_error(self, monkeypatch):
        """The original 500 is what the caller needs to see, not a connection error to a
        service that has nothing to do with captioning."""
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED))
        monkeypatch.setattr(_Client, "get", _boom)
        with pytest.raises(CaptionError, match="runner process has terminated"):
            await describe(b"img", "d")

    async def test_a1111_refusing_to_unload_does_not_trigger_a_retry(self, monkeypatch):
        _install(monkeypatch, _Resp(status=500, text=RUNNER_DIED), _idle(),
                 _Resp(status=500))
        with pytest.raises(CaptionError, match="500"):
            await describe(b"img", "d")
        assert _urls().count("http://2070.test:11434/api/generate") == 1


async def _boom(self, url):
    raise httpx.ConnectError("connection refused")


@pytest.fixture(autouse=True)
def _point_at_a_test_captioner(monkeypatch):
    monkeypatch.setattr(settings, "joycaption_url", "http://2070.test:11434")
