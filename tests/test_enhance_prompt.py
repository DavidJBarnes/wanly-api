"""POST /images/enhance-prompt: the image and its scene description go to the Sulphur-2 prompt
enhancer, and the video prompt it writes is kept on the image (wanly-api#480)."""

from unittest.mock import patch

import pytest

from app import prompt_enhance
from app.models import ImageMeta

BUCKET = "wanly-images"
PATH = f"s3://{BUCKET}/2026-10-27/00001.png"


async def _client(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), app


async def _post(db, path=PATH):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.post("/images/enhance-prompt", params={"path": path})
    finally:
        app.dependency_overrides.clear()


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class _Fake:
    sent: list = []
    answer = None

    @staticmethod
    async def post(url, payload):
        _Fake.sent.append((url, payload))
        return _Fake.answer


@pytest.fixture
def fake_enhancer(monkeypatch):
    _Fake.sent = []
    _Fake.answer = _Resp(200, {"prompt": "  She turns slowly toward the camera.  ",
                               "ms": 1234, "cold": True})
    monkeypatch.setattr(prompt_enhance, "_post", _Fake.post)
    monkeypatch.setattr(prompt_enhance.settings, "prompt_enhance_url", "http://enhancer:11438")
    with patch("app.routes.images.download_bytes", return_value=b"PNGBYTES"):
        yield _Fake


class TestEnhancePrompt:
    @pytest.mark.asyncio
    async def test_sends_image_and_scene_and_keeps_the_prompt(self, db, fake_enhancer):
        db.add(ImageMeta(path=PATH, scene_description="a woman on a sofa"))
        await db.flush()
        resp = await _post(db)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["enhanced_prompt"] == "She turns slowly toward the camera."
        assert body["cold"] is True and body["ms"] == 1234 and body["words"] == 6

        url, sent = fake_enhancer.sent[0]
        assert url == "http://enhancer:11438/enhance"
        assert sent["text"] == "a woman on a sofa"
        assert sent["image_b64"] == "UE5HQllURVM="          # b"PNGBYTES"

        meta = await db.get(ImageMeta, PATH, populate_existing=True)
        assert meta.enhanced_prompt == "She turns slowly toward the camera."
        assert meta.enhanced_prompt_at is not None
        assert meta.scene_description == "a woman on a sofa", "the scene is untouched"

    @pytest.mark.asyncio
    async def test_scene_and_listings_carry_it(self, db, fake_enhancer):
        db.add(ImageMeta(path=PATH, scene_description="a woman on a sofa"))
        await db.flush()
        await _post(db)
        client, app = await _client(db)
        try:
            async with client as c:
                scene = (await c.get("/images/scene", params={"path": PATH})).json()
        finally:
            app.dependency_overrides.clear()
        assert scene["enhanced_prompt"] == "She turns slowly toward the camera."
        from app.routes.images import _meta_by_path
        row = (await _meta_by_path(db, [PATH]))[PATH]
        assert row["enhanced_prompt"] == "She turns slowly toward the camera."

    @pytest.mark.asyncio
    async def test_no_scene_description_is_409(self, db, fake_enhancer):
        resp = await _post(db)
        assert resp.status_code == 409
        assert "scene" in resp.json()["detail"]
        assert fake_enhancer.sent == []

    @pytest.mark.asyncio
    async def test_enhancer_failure_is_503_and_keeps_the_old_prompt(self, db, fake_enhancer):
        db.add(ImageMeta(path=PATH, scene_description="a sofa", enhanced_prompt="old words"))
        await db.flush()
        fake_enhancer.answer = _Resp(503, {"detail": "llama-server did not come up"})
        resp = await _post(db)
        assert resp.status_code == 503
        assert "did not come up" in resp.json()["detail"]
        meta = await db.get(ImageMeta, PATH, populate_existing=True)
        assert meta.enhanced_prompt == "old words"

    @pytest.mark.asyncio
    async def test_switched_off_is_503(self, db, fake_enhancer, monkeypatch):
        monkeypatch.setattr(prompt_enhance.settings, "prompt_enhance_url", "")
        db.add(ImageMeta(path=PATH, scene_description="a sofa"))
        await db.flush()
        assert (await _post(db)).status_code == 503


def test_a_row_holding_only_an_enhanced_prompt_is_not_empty():
    assert not ImageMeta(path=PATH, enhanced_prompt="She turns.").is_empty()
