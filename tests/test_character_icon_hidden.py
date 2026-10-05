"""Characters: an icon, hidden, and a character's own runs (wanly-api#404).

The icon is the one image that stands for a character everywhere it is picked; hidden takes
it out of the pickers without deleting it; the character card's versions table reads one
character's runs across both arches."""
import uuid

from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user, verify_api_key_or_bearer
from app.database import get_db
from app.enums import TrainingStatus
from app.main import app
from app.models import LtxCharacter, TrainingJob


async def _call(db, method, path, **kw):
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await getattr(c, method)(path, **kw)
    finally:
        app.dependency_overrides.clear()


async def _char(db, **kw):
    c = LtxCharacter(name=kw.pop("name", f"c{uuid.uuid4().hex[:8]}"), char_lora="none",
                     trigger="t", gender="woman", **kw)
    db.add(c)
    await db.flush()
    return c


async def test_a_new_character_is_offered_and_has_no_icon(db):
    r = await _call(db, "post", "/ltx/characters",
                    json={"name": f"n{uuid.uuid4().hex[:6]}", "trigger": "x", "gender": "woman"})
    assert r.status_code in (200, 201), r.text
    assert r.json()["hidden"] is False and r.json()["icon_uri"] is None


async def test_the_icon_is_set_and_cleared(db):
    c = await _char(db)
    r = await _call(db, "patch", f"/ltx/characters/{c.id}",
                    json={"icon_uri": "s3://wanly-images/Joana/a.jpg"})
    assert r.status_code == 200 and r.json()["icon_uri"] == "s3://wanly-images/Joana/a.jpg"
    r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"icon_uri": None})
    assert r.json()["icon_uri"] is None


async def test_the_icon_must_be_an_s3_uri(db):
    c = await _char(db)
    r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"icon_uri": "/etc/passwd"})
    assert r.status_code == 422


async def test_hide_and_unhide_and_null_leaves_it_alone(db):
    c = await _char(db)
    assert (await _call(db, "patch", f"/ltx/characters/{c.id}",
                        json={"hidden": True})).json()["hidden"] is True
    # null is "leave it", never "clear" -- the column has nothing to clear to.
    r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"hidden": None})
    assert r.status_code == 200 and r.json()["hidden"] is True
    assert (await _call(db, "patch", f"/ltx/characters/{c.id}",
                        json={"hidden": False})).json()["hidden"] is False


async def test_recipes_lists_hidden_characters_with_the_flag_and_icon(db):
    """Listed, not filtered: the Characters page and a job already using one need them.
    The pickers filter."""
    c = await _char(db, hidden=True, icon_uri="s3://wanly-images/x/i.jpg",
                    image_uri="s3://wanly-images/x/f.jpg")
    chars = {x["name"]: x for x in (await _call(db, "get", "/recipes")).json()["characters"]}
    assert chars[c.name]["hidden"] is True
    assert chars[c.name]["icon_uri"] == "s3://wanly-images/x/i.jpg"
    assert chars[c.name]["image_uri"] == "s3://wanly-images/x/f.jpg"


async def test_a_characters_runs_across_both_arches(db):
    name = f"k{uuid.uuid4().hex[:6]}"
    for v, arch in ((1, None), (1, "sdxl"), (2, None)):
        db.add(TrainingJob(id=uuid.uuid4(), character=name, trigger="t", version=v,
                           dataset_images=["s3://b/x.jpg"], config={"arch": arch} if arch else {},
                           status=TrainingStatus.COMPLETED))
    db.add(TrainingJob(id=uuid.uuid4(), character="someone-else", trigger="t", version=1,
                       dataset_images=["s3://b/x.jpg"], config={}, status=TrainingStatus.COMPLETED))
    await db.flush()
    rows = (await _call(db, "get", "/training", params={"character": name, "limit": 500})).json()
    assert sorted((r["version"], r["config"].get("arch") or "ltx") for r in rows) == [
        (1, "ltx"), (1, "sdxl"), (2, "ltx")]
