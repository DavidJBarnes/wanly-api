"""Building a character sheet in the console: async jobs on the image-edit queue (wanly-api#380,
epic wanly-console#582).

What these pin down:
  * the words: outfit required, hair/body/subject optional, body sent as its own field (the
    service makes it its own sentence), the pronoun from the character unless overridden
  * seeds: default 3, explicit ones kept in order, at most 6
  * the job runs on the SAME queue as the Edit dialog's edits: waits for the render, never for
    a training run's end by interrupting it, says why, and needs a service that advertises
    `turnaround`
  * each candidate is stored the moment it arrives; a later failure keeps the earlier ones; a
    restart reads the job back from its manifest, marked interrupted
  * compose: the chosen candidate's sheet is copied into the Image Repo, becomes the
    character's sheet (identity_mode 'sheet'), and the provenance row records face, words,
    prompt, seed and model; pairs and other characters' jobs are refused
"""
import asyncio
import base64
import json
import uuid

import pytest
from sqlalchemy import select

from app import full_edit, sheet_gen
from app.config import settings
from app.models import CharacterSheet, LtxCharacter
from tests.test_full_edit import _Box

FACE = "s3://wanly-images/2026-09-30/kelly_face.png"


class _Store:
    """An in-memory S3: upload_bytes / download_bytes by URI."""

    def __init__(self):
        self.objects: dict[str, bytes] = {FACE: b"face photo"}

    def upload_bytes(self, data, key, bucket):
        uri = f"s3://{bucket}/{key}"
        self.objects[uri] = data
        return uri

    def download_bytes(self, uri):
        if uri not in self.objects:
            raise KeyError(uri)
        return self.objects[uri]


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


@pytest.fixture
def rig(monkeypatch):
    """A fake 3090 (test_full_edit's _Box) whose image-edit answers /turnaround, and a fake S3."""
    def make(fail_on_seed=None, **kw):
        b = _Box(**kw)
        b.features = b.features + ["turnaround"]
        b.turnarounds = []
        store = _Store()
        monkeypatch.setattr(full_edit, "_health", b.health)
        monkeypatch.setattr(full_edit, "_set_mode", b.set_mode)
        monkeypatch.setattr(full_edit, "queue", full_edit.FullEditQueue())
        monkeypatch.setattr(settings, "image_edit_standing_url", "")
        monkeypatch.setattr(settings, "image_edit_url", "http://3090.zero:8086")
        monkeypatch.setattr(settings, "image_edit_poll_s", 0.001)
        monkeypatch.setattr(settings, "image_edit_return_grace_s", 0.01)
        real_require = full_edit._require_features

        async def require(url, request, who, health=None):
            return await real_require(url, request, who,
                                      health=health or {"features": b.features or []})
        monkeypatch.setattr(full_edit, "_require_features", require)

        async def post(url, path, body):
            assert path == "/turnaround"
            b.turnarounds.append({k: v for k, v in body.items() if k != "image"})
            b.urls.append(url)
            assert base64.b64decode(body["image"]) == b"face photo"
            if body["seed"] == fail_on_seed:
                raise full_edit.FullEditError(504, "image-edit did not answer within 900s")
            s = body["seed"]
            return {"candidate": _b64(b"turn%d" % s), "sheet": _b64(b"sheet%d" % s),
                    "sheet_preview": _b64(b"jpg%d" % s), "prompt": f"Create ... seed {s}",
                    "seed": s, "steps": 40, "cfg": 4.0,
                    "model": "Qwen-Image-Edit-2511 (Comfy-Org fp8mixed)",
                    "files": {"base": "qwen_image_edit_2511_fp8mixed.safetensors"},
                    "settings": "40 steps, cfg 4, euler/simple",
                    "face_panel": {"mode": "crop", "box": [1, 2, 3, 4], "note": None},
                    "identity": {"aura": 0.6, "reason": None},
                    "sheet_width": 1536, "sheet_height": 1024}
        monkeypatch.setattr(full_edit, "post_service", post)
        monkeypatch.setattr(sheet_gen, "s3", store)
        from app.routes import character_sheets, image_edit
        monkeypatch.setattr(character_sheets, "s3", store)
        monkeypatch.setattr(image_edit, "s3", store)
        monkeypatch.setattr(settings, "s3_images_bucket", "wanly-images")
        monkeypatch.setattr(settings, "s3_jobs_bucket", "wanly-jobs")
        return b, store
    return make


async def _drain():
    t = full_edit.queue._task
    if t is not None:
        await asyncio.wait_for(t, timeout=5)


class _Char:
    def __init__(self, gender="woman", kind="solo"):
        self.id, self.name, self.gender, self.kind = uuid.uuid4(), "Kelly", gender, kind


# ------------------------------------------------------------------------- the words


class TestTheRequest:
    def test_body_is_its_own_field_and_blanks_are_dropped(self):
        r = sheet_gen.service_request(" jeans ", "  ", "an athletic build", "female", None)
        assert r == {"outfit": "jeans", "body": "an athletic build", "gender": "female",
                     "score": True}

    def test_an_outfit_is_required(self):
        with pytest.raises(sheet_gen.SheetError) as e:
            sheet_gen.service_request("  ", None, None, "female", None)
        assert e.value.status_code == 422

    @pytest.mark.parametrize("registry,override,want", [
        ("woman", None, "female"), ("man", None, "male"), (None, None, "female"),
        ("person", None, "female"), ("woman", "male", "male")])
    def test_the_pronoun(self, registry, override, want):
        assert sheet_gen.gender_for(registry, override) == want

    def test_seeds(self):
        assert len(sheet_gen.pick_seeds()) == 3 == len(set(sheet_gen.pick_seeds()))
        assert sheet_gen.pick_seeds(seeds=[11, 22, 11, 33]) == [11, 22, 33]
        assert len(sheet_gen.pick_seeds(5)) == 5
        for bad in ({"count": 0}, {"count": 7}, {"seeds": list(range(7))}):
            with pytest.raises(sheet_gen.SheetError):
                sheet_gen.pick_seeds(**bad)

    def test_presets_complete_the_body_sentence(self):
        for _name, label, text in sheet_gen.BODY_PRESETS:
            assert label and text[0].islower() and not text.endswith(".")
        assert {"female", "male"} == set(sheet_gen.DEFAULTS)


# ---------------------------------------------------------------------------- the job


@pytest.mark.asyncio
class TestTheJob:
    async def test_it_waits_for_the_render_then_makes_each_candidate(self, rig):
        b, store = rig(mode="ltx-engine", rendering=1, switch_after=4)
        req = sheet_gen.service_request("jeans", "her hair", "a slim build", "female", None)
        job = await sheet_gen.submit(_Char(), FACE, b"face photo", req, [11, 22, 33])
        seen = []
        for _ in range(300):
            seen.append(job.message)
            if job.state in ("done", "failed"):
                break
            await asyncio.sleep(0.001)
        await _drain()
        assert job.state == "done", job.error
        assert any("rendering; edit queued" in m for m in seen), "says why it waits"
        assert b.modes_asked == ["edit", "ltx-engine"], "one switch, then the card back"
        assert [t["seed"] for t in b.turnarounds] == [11, 22, 33]
        assert b.turnarounds[0] == {**req, "seed": 11}
        cands = job.meta["candidates"]
        assert [c["seed"] for c in cands] == [11, 22, 33]
        assert store.objects[cands[1]["sheet_uri"]] == b"sheet22"
        assert cands[1]["candidate_uri"] == f"s3://wanly-jobs/sheet-jobs/{job.id}/s22_turnaround.png"
        assert cands[1]["preview_uri"].endswith("s22_sheet.jpg")
        assert cands[0]["face_panel"] == "crop" and cands[0]["identity"]["aura"] == 0.6
        manifest = json.loads(store.objects[f"s3://wanly-jobs/sheet-jobs/{job.id}/job.json"])
        assert manifest["state"] == "done" and len(manifest["candidates"]) == 3
        assert "_source" not in job.meta

    async def test_a_training_run_is_waited_for_not_interrupted(self, rig):
        b, _ = rig(mode="ltx-engine", training=True)
        job = await sheet_gen.submit(_Char(), FACE, b"face photo",
                                     {"outfit": "jeans", "gender": "female"}, [1])
        for _ in range(50):
            await asyncio.sleep(0.001)
        assert job.state == "waiting" and "training" in job.message
        assert b.modes_asked == []
        b.training = False
        await _drain()
        assert job.state == "done"

    async def test_an_image_without_turnaround_fails_and_says_repin(self, rig):
        b, _ = rig(mode="edit")
        b.features = ["angle", "instruction", "expression", "face_box"]
        job = await sheet_gen.submit(_Char(), FACE, b"face photo",
                                     {"outfit": "jeans", "gender": "female"}, [1])
        await _drain()
        assert job.state == "failed" and "too old for turnaround" in job.error
        assert b.turnarounds == []

    async def test_a_later_failure_keeps_the_earlier_candidates(self, rig):
        b, store = rig(mode="edit", fail_on_seed=33)
        job = await sheet_gen.submit(_Char(), FACE, b"face photo",
                                     {"outfit": "jeans", "gender": "female"}, [11, 22, 33])
        await _drain()
        assert job.state == "failed" and "900s" in job.error
        assert [c["seed"] for c in job.meta["candidates"]] == [11, 22]
        rec, _live = await sheet_gen.load(job.id)
        assert rec["state"] == "failed" and len(rec["candidates"]) == 2

    async def test_after_a_restart_the_job_is_read_back_and_marked_interrupted(self, rig,
                                                                              monkeypatch):
        b, store = rig(mode="edit")
        job = await sheet_gen.submit(_Char(), FACE, b"face photo",
                                     {"outfit": "jeans", "gender": "female"}, [11])
        # The process dies before the queue runs: a fresh queue knows nothing of it.
        full_edit.queue._task.cancel()
        monkeypatch.setattr(full_edit, "queue", full_edit.FullEditQueue())
        rec, live = await sheet_gen.load(job.id)
        assert live is None and rec["state"] == "failed" and "interrupted" in rec["error"]
        with pytest.raises(sheet_gen.SheetError):
            await sheet_gen.load("../../etc/passwd")


# ---------------------------------------------------------------------- over HTTP (db)


async def _http(db, method, url, **kw):
    from tests.test_character_identity import _call
    return await _call(db, method, url, **kw)


async def _character(db, **kw) -> LtxCharacter:
    c = LtxCharacter(name=f"k-{uuid.uuid4().hex[:8]}", char_lora="kelly.safetensors",
                     trigger="k3lly", gender="woman", **kw)
    db.add(c)
    await db.commit()
    return c


@pytest.mark.asyncio
class TestOverHTTP:
    async def test_presets(self, db, rig):
        rig()
        r = await _http(db, "get", "/ltx/characters/sheet/presets")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["default_count"] == 3 and any(p["name"] == "athletic" for p in d["body"])
        assert "she wears" in d["defaults"]["female"]["outfit"]

    async def test_generate_then_compose_sets_the_sheet_and_records_provenance(self, db, rig):
        b, store = rig(mode="edit")
        c = await _character(db)
        old = (c.sheet_uri, c.identity_mode)
        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/generate", json={
            "face_uri": FACE, "outfit": "a grey fleece and jeans", "hair": "her hair in a bun",
            "body": "an athletic build", "seeds": [11, 22, 33]})
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        assert r.json()["seeds"] == [11, 22, 33] and r.json()["character_id"] == str(c.id)
        await _drain()
        r = await _http(db, "get", f"/ltx/characters/sheet/jobs/{job_id}")
        assert r.status_code == 200 and r.json()["state"] == "done"
        assert len(r.json()["candidates"]) == 3
        assert b.turnarounds[0]["body"] == "an athletic build"
        assert b.turnarounds[0]["gender"] == "female"
        assert old == (None, None), "nothing is the character's before compose"

        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/compose",
                        json={"job_id": job_id, "seed": 22})
        assert r.status_code == 200, r.text
        body = r.json()
        uri = body["character"]["sheet_uri"]
        assert uri.startswith("s3://wanly-images/character-sheets/") and "_sheet_s22_" in uri
        assert store.objects[uri] == b"sheet22"
        assert body["character"]["identity_mode"] == "sheet"
        assert body["character"]["char_lora"] == "kelly.safetensors", "the LoRA is untouched"
        s = body["sheet"]
        assert (s["seed"], s["face_uri"], s["outfit"], s["hair"], s["body"]) == (
            22, FACE, "a grey fleece and jeans", "her hair in a bun", "an athletic build")
        assert s["model"].startswith("Qwen-Image-Edit-2511") and s["prompt"].endswith("seed 22")
        rows = (await db.execute(select(CharacterSheet).where(
            CharacterSheet.character_id == c.id))).scalars().all()
        assert len(rows) == 1 and rows[0].job_id == job_id

        r = await _http(db, "get", f"/ltx/characters/sheet/jobs/{job_id}")
        assert r.json()["saved"][0]["seed"] == 22
        r = await _http(db, "get", f"/ltx/characters/{c.id}/sheets")
        assert [x["sheet_uri"] for x in r.json()] == [uri]

    async def test_refusals(self, db, rig):
        rig(mode="edit")
        c = await _character(db)
        other = await _character(db)
        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/generate",
                        json={"face_uri": FACE, "outfit": " "})
        assert r.status_code == 422
        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/generate",
                        json={"face_uri": "s3://wanly-jobs/x.png", "outfit": "jeans"})
        assert r.status_code == 400, "only an images-bucket photo"
        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/generate",
                        json={"face_uri": FACE, "outfit": "jeans", "count": 1})
        job_id = r.json()["id"]
        await _drain()
        r = await _http(db, "post", f"/ltx/characters/{other.id}/sheet/compose",
                        json={"job_id": job_id, "seed": r.json()["seeds"][0]})
        assert r.status_code == 409, "another character's job"
        r = await _http(db, "post", f"/ltx/characters/{c.id}/sheet/compose",
                        json={"job_id": job_id, "seed": 999999999})
        assert r.status_code == 404
        r = await _http(db, "get", f"/ltx/characters/sheet/jobs/{uuid.uuid4().hex}")
        assert r.status_code == 404

    async def test_a_pair_is_refused(self, db, rig):
        rig(mode="edit")
        pair = LtxCharacter(name=f"p-{uuid.uuid4().hex[:8]}", char_lora="p.safetensors",
                            trigger="a and b", kind="pair", members=["a", "b"])
        db.add(pair)
        await db.commit()
        r = await _http(db, "post", f"/ltx/characters/{pair.id}/sheet/generate",
                        json={"face_uri": FACE, "outfit": "jeans"})
        assert r.status_code == 422 and "pair" in r.json()["detail"]
