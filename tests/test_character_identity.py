"""Characters: a LoRA, a sheet, or both (wanly-api#379, epic wanly-console#581).

Phase 0 (wanly-gpu-docker#155) showed a 1536x1024 character sheet, conditioned into wanly's
own recipe graph, holding identity better than the character LoRA alone. So a character can
now be a LoRA, an identity reference (a sheet or a face close-up), or both. What this holds:

  * the registry: create/update accept the reference fields, refuse inconsistent ones, and a
    sheet-only character is stored with no LoRA and no trigger -- while every existing shape
    (a LoRA character, a registration ahead of training with "none") is untouched;
  * the database CHECKs behind that (a LoRA or a reference; a mode that names a reference);
  * <TRIGGER> for a character with no trigger: its description, or dropped -- and a prompt that
    is then empty is still refused (the #577 guards);
  * the claim: a presigned reference URL for the segment's character, honouring the job's
    `use_identity_ref`, and for a PAIR the first member's reference or none.

Run against a real database: the CHECK constraints and the claim query are the subject.
"""

import json
import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.auth import get_current_user, verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, LtxCharacter, Segment, User, Worker
from app.routes import segments as seg_routes

SHEET = "s3://wanly-images/chars/kelly_sheet.png"
FACE = "s3://wanly-images/chars/kelly_face.png"
WORKER_ID = uuid.UUID("95d5dffe-881b-46a6-bd5f-8930b9a66b75")


def _name(prefix="c"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _call(db, method, path, **kw):
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await getattr(client, method)(path, **kw)
    finally:
        app.dependency_overrides.clear()


async def _row(db, name) -> LtxCharacter:
    db.expire_all()
    return (await db.execute(select(LtxCharacter).where(LtxCharacter.name == name))
            ).scalar_one()


# ---------------------------------------------------------------------- the registry

@pytest.mark.asyncio
class TestCreate:
    async def test_a_sheet_only_character_has_no_lora_and_no_trigger(self, db):
        name = _name()
        r = await _call(db, "post", "/ltx/characters", json={
            "name": name, "sheet_uri": SHEET, "description": "a woman with auburn hair"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["char_lora"] is None
        assert body["trigger"] is None
        assert body["sheet_uri"] == SHEET and body["identity_mode"] == "sheet"
        assert body["description"] == "a woman with auburn hair"

    async def test_a_face_only_character_defaults_to_face_mode(self, db):
        r = await _call(db, "post", "/ltx/characters", json={"name": _name(),
                                                            "face_ref_uri": FACE})
        assert r.status_code == 201, r.text
        assert r.json()["identity_mode"] == "face"

    async def test_lora_and_sheet_is_both_and_keeps_its_trigger_default(self, db):
        name = _name()
        r = await _call(db, "post", "/ltx/characters", json={
            "name": name, "char_lora": "k3lly2026_v2", "trigger": "k3lly2026",
            "gender": "woman", "sheet_uri": SHEET, "face_ref_uri": FACE})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["char_lora"] == "k3lly2026_v2" and body["trigger"] == "k3lly2026"
        # Both references: the sheet renders unless told otherwise.
        assert body["identity_mode"] == "sheet"

    async def test_an_explicit_mode_is_kept(self, db):
        r = await _call(db, "post", "/ltx/characters", json={
            "name": _name(), "sheet_uri": SHEET, "face_ref_uri": FACE, "identity_mode": "face"})
        assert r.json()["identity_mode"] == "face"

    async def test_registration_ahead_of_training_is_unchanged(self, db):
        """No LoRA and no reference: the #352 registration, stored "none" with the name as
        its trigger, exactly as before."""
        name = _name()
        r = await _call(db, "post", "/ltx/characters", json={"name": name})
        assert r.status_code == 201, r.text
        assert r.json()["char_lora"] == "none" and r.json()["trigger"] == name
        assert r.json()["identity_mode"] is None

    @pytest.mark.parametrize("body,why", [
        ({"identity_mode": "sheet"}, "needs a sheet_uri"),
        ({"face_ref_uri": FACE, "identity_mode": "sheet"}, "needs a sheet_uri"),
        ({"sheet_uri": SHEET, "identity_mode": "face"}, "needs a face_ref_uri"),
    ])
    async def test_a_mode_must_name_a_reference_it_has(self, db, body, why):
        r = await _call(db, "post", "/ltx/characters", json={"name": _name(), **body})
        assert r.status_code == 422 and why in r.text

    @pytest.mark.parametrize("uri", ["https://example.com/x.png", "kelly.png", "s3://bucket"])
    async def test_a_reference_must_be_an_s3_uri(self, db, uri):
        """A presigned URL stored in the row would expire; a bare name is nothing."""
        r = await _call(db, "post", "/ltx/characters", json={"name": _name(), "sheet_uri": uri})
        assert r.status_code == 422

    async def test_the_mode_vocabulary_is_closed(self, db):
        r = await _call(db, "post", "/ltx/characters", json={
            "name": _name(), "sheet_uri": SHEET, "identity_mode": "both"})
        assert r.status_code == 422

    async def test_a_pair_carries_no_reference_of_its_own(self, db):
        a, b = _name("a"), _name("b")
        for n in (a, b):
            await _call(db, "post", "/ltx/characters",
                        json={"name": n, "trigger": n, "gender": "woman"})
        r = await _call(db, "post", "/ltx/characters", json={
            "name": _name("pair"), "kind": "pair", "members": [a, b], "sheet_uri": SHEET})
        assert r.status_code == 422 and "first member" in r.text


@pytest.mark.asyncio
class TestUpdate:
    async def _lora_character(self, db):
        c = LtxCharacter(name=_name(), char_lora="k3lly2026_v2", trigger="k3lly2026",
                         gender="woman", strength_stage_1=0.8, strength_stage_2=1.5,
                         trained_from=[{"dataset_id": None, "name": "x", "count": 1}])
        db.add(c)
        await db.flush()
        return c

    async def test_adding_a_sheet_to_a_lora_character_keeps_everything_else(self, db):
        """Existing characters keep their LoRAs: a sheet is ADDED beside them."""
        c = await self._lora_character(db)
        r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"sheet_uri": SHEET})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["char_lora"] == "k3lly2026_v2" and body["trigger"] == "k3lly2026"
        assert body["strength_stage_1"] == 0.8 and body["strength_stage_2"] == 1.5
        assert body["sheet_uri"] == SHEET and body["identity_mode"] == "sheet"

    async def test_removing_the_sheet_follows_the_mode(self, db):
        c = await self._lora_character(db)
        await _call(db, "patch", f"/ltx/characters/{c.id}",
                    json={"sheet_uri": SHEET, "face_ref_uri": FACE})
        r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"sheet_uri": None})
        assert r.status_code == 200, r.text
        assert r.json()["identity_mode"] == "face"
        r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"face_ref_uri": None})
        assert r.json()["identity_mode"] is None

    async def test_the_lora_can_go_while_a_sheet_remains(self, db):
        r = await _call(db, "post", "/ltx/characters", json={
            "name": _name(), "char_lora": "x_v1", "sheet_uri": SHEET})
        cid = r.json()["id"]
        r = await _call(db, "patch", f"/ltx/characters/{cid}", json={"char_lora": None})
        assert r.status_code == 200, r.text
        assert r.json()["char_lora"] is None

    async def test_a_character_cannot_end_up_with_neither(self, db):
        r = await _call(db, "post", "/ltx/characters", json={"name": _name(),
                                                            "sheet_uri": SHEET})
        cid = r.json()["id"]
        r = await _call(db, "patch", f"/ltx/characters/{cid}", json={"sheet_uri": None})
        assert r.status_code == 422 and "LoRA or a character sheet" in r.text
        # And the row is as it was.
        r = await _call(db, "get", "/ltx/characters")
        row = next(c for c in r.json() if c["id"] == cid)
        assert row["sheet_uri"] == SHEET

    async def test_an_inconsistent_mode_is_refused(self, db):
        c = await self._lora_character(db)
        r = await _call(db, "patch", f"/ltx/characters/{c.id}", json={"identity_mode": "face"})
        assert r.status_code == 422 and "needs a face_ref_uri" in r.text

    async def test_the_recipe_book_carries_the_fields(self, db):
        name = _name()
        await _call(db, "post", "/ltx/characters", json={"name": name, "sheet_uri": SHEET,
                                                         "description": "a woman"})
        r = await _call(db, "get", "/recipes")
        assert r.status_code == 200, r.text
        row = next(c for c in r.json()["characters"] if c["name"] == name)
        assert row["sheet_uri"] == SHEET and row["identity_mode"] == "sheet"
        assert row["char_lora"] is None and row["description"] == "a woman"


@pytest.mark.asyncio
class TestTheDatabaseHoldsIt:
    async def test_neither_lora_nor_reference_is_refused(self, db):
        db.add(LtxCharacter(name=_name(), char_lora=None, trigger=None))
        with pytest.raises(IntegrityError, match="ck_ltx_characters_lora_or_ref"):
            await db.flush()
        await db.rollback()

    async def test_a_mode_without_its_reference_is_refused(self, db):
        db.add(LtxCharacter(name=_name(), char_lora=None, sheet_uri=SHEET, identity_mode="face"))
        with pytest.raises(IntegrityError, match="ck_ltx_characters_identity_mode"):
            await db.flush()
        await db.rollback()

    async def test_a_legacy_none_row_still_passes(self, db):
        db.add(LtxCharacter(name=_name(), char_lora="none", trigger="t"))
        await db.flush()


# ---------------------------------------------------------------------- <TRIGGER>

async def _character(db, **kw) -> LtxCharacter:
    kw.setdefault("name", _name())
    c = LtxCharacter(**kw)
    db.add(c)
    await db.flush()
    return c


def _blob(name, **person):
    return {"recipe": "Turn", "character": name,
            "characters": [{"name": name, "trigger": person.get("trigger"),
                            "char_lora": person.get("char_lora"), "s1": None, "s2": None}]}


@pytest.mark.asyncio
class TestTrigger:
    async def test_a_sheet_only_character_fills_from_its_description(self, db):
        c = await _character(db, sheet_uri=SHEET, description="a woman with auburn hair")
        out = await seg_routes._resolve_trigger(db, "<TRIGGER>, she turns", _blob(c.name))
        assert out == "a woman with auburn hair, she turns"

    async def test_with_no_description_the_placeholder_is_dropped(self, db):
        c = await _character(db, sheet_uri=SHEET)
        out = await seg_routes._resolve_trigger(db, "<TRIGGER>, she turns", _blob(c.name))
        assert out == "she turns"

    async def test_a_lora_character_still_renders_its_trigger_phrase(self, db):
        c = await _character(db, char_lora="k_v1", trigger="k3lly", gender="woman",
                             sheet_uri=SHEET, description="ignored while there is a trigger")
        out = await seg_routes._resolve_trigger(db, "<TRIGGER>, she turns", _blob(c.name))
        assert out == "k3lly, woman, she turns"

    async def test_a_prompt_that_was_only_the_placeholder_is_refused(self, db):
        """Dropping <TRIGGER> must never let an empty prompt through (#577)."""
        c = await _character(db, sheet_uri=SHEET)
        with pytest.raises(HTTPException) as e:
            await seg_routes._refuse_empty_submit(db, "<TRIGGER>,", _blob(c.name))
        assert e.value.status_code == 422

    async def test_a_description_alone_is_not_a_prompt(self, db):
        """The description fills <TRIGGER>, so on its own it says nothing about the shot --
        exactly like a bare trigger phrase."""
        c = await _character(db, sheet_uri=SHEET, description="a woman with auburn hair")
        assert not await seg_routes._has_content(db, "a woman with auburn hair,", _blob(c.name))
        assert await seg_routes._has_content(db, "a woman with auburn hair, she turns",
                                             _blob(c.name))


# ---------------------------------------------------------------------- the claim

async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _queued(db, blob, *, use_identity_ref=None, reprocess_type=None) -> Segment:
    job = Job(user_id=(await _user(db)).id, name="j", width=832, height=1216, fps=24, seed=1,
              priority=-1000, status=JobStatus.PENDING, starting_image="s3://b/start.png",
              use_identity_ref=use_identity_ref)
    db.add(job)
    await db.flush()
    seg = Segment(job_id=job.id, index=0, prompt="she turns to the camera",
                  status=SegmentStatus.PENDING, ltx_recipe=blob, reprocess_type=reprocess_type)
    db.add(seg)
    await db.flush()
    return seg


@pytest.fixture
def presign(monkeypatch):
    calls = []

    def fake(uri, expires=21600):
        calls.append(uri)
        return f"https://signed.example/{uri.rsplit('/', 1)[-1]}?X-Amz-Signature=s"
    monkeypatch.setattr(seg_routes, "generate_presigned_url", fake)
    return calls


async def _claim(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment, Worker)):
                db.expire(obj)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/segments/next", params={
                "worker_id": str(WORKER_ID), "worker_name": "3090.zero", "kind": "gpu"})
        assert r.status_code == 200, r.text
        return r.json()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestTheClaim:
    async def test_a_sheet_character_hands_out_a_presigned_reference(self, db, presign):
        c = await _character(db, char_lora="k_v1", trigger="k", sheet_uri=SHEET,
                             identity_mode="sheet")
        seg = await _queued(db, _blob(c.name, char_lora="k_v1", trigger="k"))
        body = await _claim(db)
        assert body["id"] == str(seg.id)
        ref = body["identity_ref"]
        assert ref["mode"] == "sheet" and ref["uri"] == SHEET and ref["character"] == c.name
        assert ref["url"].startswith("https://signed.example/kelly_sheet.png")
        assert presign == [SHEET]

    async def test_face_mode_hands_out_the_face(self, db, presign):
        c = await _character(db, sheet_uri=SHEET, face_ref_uri=FACE, identity_mode="face")
        await _queued(db, _blob(c.name))
        ref = (await _claim(db))["identity_ref"]
        assert ref["mode"] == "face" and ref["uri"] == FACE

    async def test_the_job_toggle_turns_it_off(self, db, presign):
        c = await _character(db, char_lora="k_v1", trigger="k", sheet_uri=SHEET)
        await _queued(db, _blob(c.name), use_identity_ref=False)
        assert (await _claim(db))["identity_ref"] is None
        assert presign == []

    async def test_the_toggle_defaults_on(self, db, presign):
        c = await _character(db, sheet_uri=SHEET)
        await _queued(db, _blob(c.name), use_identity_ref=None)
        assert (await _claim(db))["identity_ref"]["mode"] == "sheet"

    async def test_a_character_without_a_reference_sends_none(self, db, presign):
        c = await _character(db, char_lora="k_v1", trigger="k")
        await _queued(db, _blob(c.name, char_lora="k_v1"))
        assert (await _claim(db))["identity_ref"] is None

    async def test_a_segment_with_no_recipe_sends_none(self, db, presign):
        await _queued(db, None)
        assert (await _claim(db))["identity_ref"] is None

    async def test_a_pair_renders_with_its_first_members_sheet(self, db, presign):
        """One reference per render: the pair's FIRST member's, as registered."""
        first = await _character(db, char_lora="a_v1", trigger="a", gender="woman",
                                 sheet_uri=SHEET)
        second = await _character(db, char_lora="b_v1", trigger="b", gender="man",
                                  sheet_uri="s3://wanly-images/chars/second.png")
        pair = await _character(db, char_lora="ab_v1", trigger="a, woman and b, man",
                                kind="pair", members=[first.name, second.name])
        await _queued(db, _blob(pair.name, char_lora="ab_v1"))
        ref = (await _claim(db))["identity_ref"]
        assert ref["uri"] == SHEET and ref["character"] == first.name

    async def test_a_pair_whose_first_member_has_none_sends_none(self, db, presign):
        """Not the second member's: conditioning both people on one face is wrong for one."""
        first = await _character(db, char_lora="a_v1", trigger="a", gender="woman")
        second = await _character(db, char_lora="b_v1", trigger="b", gender="man",
                                  sheet_uri=SHEET)
        pair = await _character(db, char_lora="ab_v1", trigger="a and b", kind="pair",
                                members=[first.name, second.name])
        await _queued(db, _blob(pair.name, char_lora="ab_v1"))
        assert (await _claim(db))["identity_ref"] is None

    async def test_a_presign_failure_renders_without_rather_than_stopping_the_queue(
            self, db, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("no credentials")
        monkeypatch.setattr(seg_routes, "generate_presigned_url", boom)
        c = await _character(db, sheet_uri=SHEET, description="a woman")
        seg = await _queued(db, _blob(c.name))
        body = await _claim(db)
        assert body["id"] == str(seg.id) and body["identity_ref"] is None

    async def test_a_reprocess_carrier_never_gets_one(self, db, presign):
        c = await _character(db, sheet_uri=SHEET)
        await _queued(db, _blob(c.name), reprocess_type="smashcut_concat")
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[verify_api_key] = lambda: None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/segments/next", params={
                    "worker_id": str(WORKER_ID), "kind": "hologram"})
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200, r.text
        assert r.json() is None or r.json()["identity_ref"] is None


# ---------------------------------------------------------------------- the job toggle

@pytest.mark.asyncio
class TestTheJobToggle:
    async def _create(self, db, **extra):
        user = await _user(db)
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: db
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                data = {"name": "j", "width": 832, "height": 1216, "fps": 24,
                        "first_segment": {"prompt": "she turns to the camera",
                                          "duration_seconds": 5}, **extra}
                r = await client.post("/jobs", data={"data": json.dumps(data)})
                assert r.status_code == 201, r.text
                jid = r.json()["id"]
                r2 = await client.patch(f"/jobs/{jid}", json={"use_identity_ref": False})
                assert r2.status_code == 200, r2.text
                return r.json(), r2.json()
        finally:
            app.dependency_overrides.clear()

    async def test_created_with_it_and_patched_off(self, db):
        created, patched = await self._create(db, use_identity_ref=True)
        assert created["use_identity_ref"] is True
        assert patched["use_identity_ref"] is False

    async def test_absent_is_the_default(self, db):
        created, _ = await self._create(db)
        assert created["use_identity_ref"] is None


# ---------------------------------------------------------------------- image moves

@pytest.mark.asyncio
async def test_a_moved_sheet_is_followed_by_its_character(db, monkeypatch):
    from app.config import settings
    from app.routes import images as image_routes
    monkeypatch.setattr(image_routes, "move_object", lambda *a, **k: None)
    bucket = settings.s3_images_bucket
    src = f"s3://{bucket}/2026-10-01/sheet-{uuid.uuid4().hex[:6]}.png"
    c = await _character(db, sheet_uri=src)
    r = await _call(db, "post", "/images/move",
                    json={"keys": [src.split(f"s3://{bucket}/", 1)[1]],
                          "target_folder": "characters"})
    assert r.status_code == 200, r.text
    row = await _row(db, c.name)
    assert row.sheet_uri == f"s3://{bucket}/characters/{src.rsplit('/', 1)[1]}"
