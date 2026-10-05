"""GET /files/presigned (wanly-api#400): the presigned URL as JSON, for copy-as-curl.

The point is what it does NOT carry: the console's Download link has the user's JWT in its
query string, and that must never be what lands in a shell on 3090b."""
from httpx import ASGITransport, AsyncClient

from app.auth import verify_api_key_or_bearer
from app.main import app
from app.routes import files as mod


async def _get(path, monkeypatch, authed=True):
    monkeypatch.setattr(mod, "generate_presigned_url",
                        lambda uri, expires: f"https://s3.example/{uri[5:]}?X-Amz-Expires={expires}")
    if authed:
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.get("/files/presigned", params={"path": path})
    finally:
        app.dependency_overrides.clear()


async def test_it_returns_the_presigned_url_and_the_filename(monkeypatch):
    r = await _get("s3://ltx-loras/character/sdxl/Joana_sdxl_v2_final.safetensors", monkeypatch)
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "Joana_sdxl_v2_final.safetensors"
    assert body["expires_in"] == 21600
    assert body["url"].startswith("https://s3.example/ltx-loras/character/sdxl/")
    assert "token" not in body["url"]


async def test_a_non_s3_path_is_refused(monkeypatch):
    assert (await _get("/etc/passwd", monkeypatch)).status_code == 400


async def test_it_needs_auth(monkeypatch):
    """Header auth only: this is a fetch from the console, never a link with ?token=."""
    r = await _get("s3://ltx-loras/x.safetensors", monkeypatch, authed=False)
    assert r.status_code == 401
