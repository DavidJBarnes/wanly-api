"""Where a character LoRA's trigger and gender come from (wanly-console#596).

Picking a LoRA in the character editor fills Trigger and Gender from how that file trained,
and the Characters page flags rows that disagree -- several were typed wrong by hand. What
this holds:

  * the training run is found from the file name, and gives what publish writes: a solo's
    trigger and gender, a pair's joined phrase with no gender;
  * a LoRA trained elsewhere is read from its safetensors header with two RANGED reads, never
    the whole file, cached per etag;
  * the header keys kohya writes (ss_dataset_dirs, ss_datasets class_tokens, ss_tag_frequency)
    each yield "<trigger>, <gender>" -- and musubi's LTX-2 header, which writes none of them,
    yields an honest "none" rather than a guess;
  * the bulk check reports every mismatch, and a trained character's lock lets exactly the
    trained values through.
"""
import json
import struct
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app import lora_provenance as lp
from app.auth import get_current_user
from app.database import get_db
from app.enums import TrainingStatus
from app.main import app
from app.models import LtxCharacter, TrainingJob
from app.routes import ltx_recipes

#: The real __metadata__ of k3lly2026_v2.safetensors (CLI musubi-tuner, LTX-2), trimmed to
#: the keys that could have carried a caption. It carries none: this is the case to design for.
MUSUBI_LTX2 = {
    "modelspec.title": "k3lly2026_v2",
    "ss_output_name": "k3lly2026_v2",
    "ss_datasets": json.dumps([{
        "resolution": [1024, 1024], "caption_extension": ".txt", "caption_field": None,
        "batch_size_per_device": 1, "num_repeats": 10, "enable_bucket": True,
        "bucket_no_upscale": True, "separate_audio_buckets": False, "reference_frames": None,
        "image_directory": "data", "has_control": False, "cache_only": False}]),
    "ss_network_module": "networks.lora_ltx2",
    "ss_training_comment": "None",
    "sshs_model_hash": "f4640f97213a2247d6ec6f0c54510012a65b5673cac1d480de738467a7fb2f89",
}


def _uniq(p="x"):
    return f"{p}{uuid.uuid4().hex[:6]}"


def _job(**kw):
    base = dict(id=uuid.uuid4(), character=_uniq("c"), trigger="p@yton", version=1,
                dataset_images=["s3://i/1.jpg"], config={}, status=TrainingStatus.COMPLETED,
                created_at=datetime.now(timezone.utc))
    base.update(kw)
    return TrainingJob(**base)


# ---------------------------------------------------------------------- the file's metadata

class TestMetadata:
    def test_musubi_ltx2_header_names_no_trigger(self):
        trig, gender, why = lp.metadata_identity(MUSUBI_LTX2)
        assert (trig, gender) == (None, None)
        assert "ss_datasets present but name no trigger" in why

    def test_an_empty_header_says_so(self):
        assert lp.metadata_identity({})[2] == "no caption or dataset-dir keys in the header"

    def test_kohya_dataset_dirs(self):
        meta = {"ss_dataset_dirs": json.dumps({
            "10_ohwx woman": {"n_repeats": 10, "img_count": 30},
            "1_woman": {"n_repeats": 1, "img_count": 200}})}
        assert lp.metadata_identity(meta) == ("ohwx", "woman", "ss_dataset_dirs")

    def test_kohya_class_tokens(self):
        meta = {"ss_datasets": json.dumps([{"subsets": [
            {"class_tokens": "p@yton woman", "img_count": 20}]}])}
        assert lp.metadata_identity(meta)[:2] == ("p@yton", "woman")

    def test_tag_frequency_takes_the_most_common_prefix(self):
        meta = {"ss_tag_frequency": json.dumps({"10_data": {
            "k3lly2026": 40, "woman": 40, "smiling": 12, "man": 1, "outdoors": 9}})}
        assert lp.metadata_identity(meta) == ("k3lly2026", "woman", "ss_tag_frequency")

    def test_a_tag_in_few_captions_is_not_a_trigger(self):
        meta = {"ss_tag_frequency": json.dumps({"d": {"woman": 40, "smiling": 5}})}
        assert lp.metadata_identity(meta)[:2] == (None, None)


def _safetensors(meta: dict) -> bytes:
    header = json.dumps({"__metadata__": meta, "w": {"dtype": "F32", "shape": [1],
                                                     "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * 4


class _FakeS3:
    """Serves one object, and only by Range: a GET without one fails the test."""

    def __init__(self, blob: bytes):
        self.blob, self.ranges = blob, []

    def get_object(self, Bucket, Key, Range=None):
        assert Range, "the whole LoRA must never be downloaded"
        a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        self.ranges.append((a, b))
        body = self.blob[a:b + 1]
        return {"Body": type("B", (), {"read": lambda self_: body})()}


class TestHeaderRead:
    def test_two_ranged_reads_and_only_the_header(self, monkeypatch):
        meta = {"ss_dataset_dirs": json.dumps({"5_ohwx man": {"img_count": 3}})}
        blob = _safetensors(meta)
        fake = _FakeS3(blob)
        import app.s3 as s3mod
        monkeypatch.setattr(s3mod, "_client_for_bucket", lambda b: fake)
        assert lp.read_header_metadata("ltx-loras", "character/x.safetensors") == meta
        n = struct.unpack("<Q", blob[:8])[0]
        assert fake.ranges == [(0, 7), (8, 7 + n)]

    def test_garbage_is_no_metadata(self, monkeypatch):
        import app.s3 as s3mod
        monkeypatch.setattr(s3mod, "_client_for_bucket",
                            lambda b: _FakeS3(b"\xff" * 64))
        assert lp.read_header_metadata("ltx-loras", "k") is None

    async def test_cached_per_etag(self, monkeypatch):
        reads = []

        def fake_read(bucket, key):
            reads.append(key)
            return {"ss_dataset_dirs": json.dumps({"1_ohwx woman": {"img_count": 1}})}

        monkeypatch.setattr(lp, "read_header_metadata", fake_read)
        objs = {"x": {"key": "character/x.safetensors", "etag": "e1"}}
        p1 = await lp.provenance("x", [], objs)
        p2 = await lp.provenance("x.safetensors", [], objs)
        assert reads == ["character/x.safetensors"]
        assert (p1.source, p1.trigger, p1.gender) == ("lora_metadata", "ohwx", "woman")
        assert p2.public() == p1.public()
        objs["x"]["etag"] = "e2"  # retrained, republished under the same name
        await lp.provenance("x", [], objs)
        assert len(reads) == 2


# ---------------------------------------------------------------------- the training run

class TestRun:
    def test_the_final_checkpoint_finds_its_solo_run(self):
        job = _job(config={"mode": "solo", "gender": "woman", "lora_name": "Payton-Synthetic"},
                   checkpoints=["s3://ltx-loras/character/Payton-Synthetic_v1_final.safetensors"])
        p = lp.from_run("Payton-Synthetic_v1_final",
                        lp.find_run([job], "Payton-Synthetic_v1_final"))
        assert (p.trigger, p.gender, p.source, p.pair) == ("p@yton", "woman", "training_run", False)
        assert p.run_id == str(job.id) and p.run_version == 1

    def test_an_unpublished_epoch_is_matched_by_its_artifact_name(self):
        job = _job(config={"mode": "solo", "gender": "woman", "lora_name": "pay"}, version=2)
        assert lp.find_run([job], "pay_v2_e03") is job
        assert lp.find_run([job], "pay_v20_e03") is None

    def test_newest_run_wins(self):
        old = _job(config={"lora_name": "k", "gender": "man"},
                   created_at=datetime.now(timezone.utc) - timedelta(days=3))
        new = _job(config={"lora_name": "k", "gender": "woman"})
        assert lp.find_run([old, new], "k_v1_final") is new

    def test_a_pair_is_the_joined_phrase_with_no_gender(self):
        job = _job(trigger="d@vid", config={"mode": "pair", "gender": "man", "lora_name": "dk"},
                   identities=[{"kind": "identity", "trigger": "k3lly2026", "gender": "woman"},
                               {"kind": "composition", "trigger": None, "gender": None},
                               {"kind": "regularization", "gender": "woman"}])
        assert lp.run_identity(job) == ("d@vid, man and k3lly2026, woman", None, True)

    def test_no_recorded_gender_falls_back_to_the_captions(self):
        job = _job(trigger="p@y", config={"captions": ["p@y, woman, smiling", "p@y, woman"]})
        assert lp.run_identity(job) == ("p@y", "woman", False)


# ---------------------------------------------------------------------- routes

async def _call(db, method, path, **kw):
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await getattr(c, method)(path, **kw)
    finally:
        app.dependency_overrides.clear()


def _listing(monkeypatch, *names):
    async def fake():
        return [{"name": f"character/{n}.safetensors", "etag": "e", "size": 1,
                 "multipart": False} for n in names]
    monkeypatch.setattr(ltx_recipes, "_lora_listing", fake)


@pytest.mark.asyncio
class TestRoutes:
    async def test_provenance_from_a_run(self, db):
        stem = _uniq("Payton-Synthetic") + "_v1_final"
        job = _job(config={"mode": "solo", "gender": "woman"},
                   checkpoints=[f"s3://ltx-loras/character/{stem}.safetensors"])
        db.add(job)
        await db.flush()
        r = await _call(db, "get", f"/loras/{stem}.safetensors/provenance")
        assert r.status_code == 200
        body = r.json()
        assert (body["trigger"], body["gender"], body["source"]) == ("p@yton", "woman",
                                                                     "training_run")
        assert body["run_id"] == str(job.id)

    async def test_a_cli_lora_with_no_caption_keys_is_none(self, db, monkeypatch):
        name = _uniq("k3lly")
        _listing(monkeypatch, name)
        monkeypatch.setattr(lp, "read_header_metadata", lambda b, k: MUSUBI_LTX2)
        r = await _call(db, "get", f"/loras/{name}/provenance")
        assert r.json()["source"] == "none"
        assert r.json()["trigger"] is None

    async def test_provenance_check_reports_mismatches(self, db, monkeypatch):
        good, bad = _uniq("good"), _uniq("bad")
        db.add(_job(trigger="g@", config={"mode": "solo", "gender": "woman"},
                    checkpoints=[f"s3://ltx-loras/character/{good}_v1_final.safetensors"]))
        db.add(_job(trigger="p@yton", config={"mode": "solo", "gender": "woman"},
                    checkpoints=[f"s3://ltx-loras/character/{bad}_v1_final.safetensors"]))
        db.add(LtxCharacter(name=good, char_lora=f"{good}_v1_final", trigger="g@",
                            gender="woman"))
        db.add(LtxCharacter(name=bad, char_lora=f"{bad}_v1_final", trigger="payton",
                            gender="man"))
        db.add(LtxCharacter(name=_uniq("sheet"), char_lora=None, trigger=None))
        await db.flush()
        r = await _call(db, "get", "/ltx/characters/provenance-check")
        assert r.status_code == 200
        rows = {c["name"]: c for c in r.json()["characters"]}
        assert rows[good]["mismatches"] == []
        assert rows[bad]["mismatches"] == [
            {"field": "trigger", "stored": "payton", "trained": "p@yton"},
            {"field": "gender", "stored": "man", "trained": "woman"}]
        assert rows[bad]["provenance"]["source"] == "training_run"

    async def test_the_lock_lets_exactly_the_trained_values_through(self, db):
        name = _uniq("botched")
        db.add(_job(trigger="p@yton", config={"mode": "solo", "gender": "woman"},
                    checkpoints=[f"s3://ltx-loras/character/{name}_v1_final.safetensors"]))
        c = LtxCharacter(name=name, char_lora=f"{name}_v1_final", trigger="payton", gender="man",
                         trained_from=[{"dataset_id": None, "name": "x", "count": 1}])
        db.add(c)
        await db.flush()
        r = await _call(db, "patch", f"/ltx/characters/{c.id}",
                        json={"trigger": "something-else"})
        assert r.status_code == 409
        r = await _call(db, "patch", f"/ltx/characters/{c.id}",
                        json={"trigger": "p@yton", "gender": "woman"})
        assert r.status_code == 200, r.text
        db.expire_all()
        row = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == name))
               ).scalar_one()
        assert (row.trigger, row.gender) == ("p@yton", "woman")
