"""Video clips in a dataset (wanly-api#411): identity in motion for LTX character LoRAs.

The ffmpeg tests run the real binaries on synthetic clips: the normalization IS the feature
(a clip the trainer cannot window is worse than no clip), and mocking ffmpeg would test that
the mock agrees with itself.
"""
import shutil
import subprocess
import uuid

import pytest

from app import clips

def _has_x264() -> bool:
    """The API image (Debian ffmpeg) and CI (Ubuntu) have libx264; a Fedora laptop's
    patent-free build does not, and normalize cannot work there -- skipped, not failed."""
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True).stdout
    return b"libx264" in out


needs_ffmpeg = pytest.mark.skipif(not _has_x264(), reason="ffmpeg with libx264 not installed")


def _make_clip(tmp_path, seconds=3.0, fps=30, size="1920x1080", audio=True, suffix=".mp4"):
    out = tmp_path / f"src{suffix}"
    argv = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"testsrc=duration={seconds}:size={size}:rate={fps}"]
    if audio:
        argv += ["-f", "lavfi", "-i", f"sine=duration={seconds}", "-shortest"]
    argv += ["-pix_fmt", "yuv420p", str(out)]
    subprocess.run(argv, check=True)
    return out.read_bytes()


def _probe_file(path):
    import json
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                          "-of", "json", str(path)], capture_output=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _probe(data: bytes, tmp_path):
    p = tmp_path / "probe.mp4"
    p.write_bytes(data)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,nb_frames", "-of", "json", str(p)],
        capture_output=True, check=True).stdout
    import json
    return json.loads(out)["streams"]


@needs_ffmpeg
class TestNormalize:
    def test_a_phone_clip_becomes_the_trainers_clip(self, tmp_path):
        """30 fps 1080p with audio -> 25 fps, long edge 768, no audio track."""
        out = _run(clips.normalize(_make_clip(tmp_path), ".mp4"))
        streams = _probe(out, tmp_path)
        assert [s["codec_type"] for s in streams] == ["video"]
        v = streams[0]
        assert (v["width"], v["height"]) == (768, 432)
        assert v["r_frame_rate"] == "25/1"
        assert int(v["nb_frames"]) == 75

    def test_a_portrait_clip_keeps_its_orientation_and_even_sizes(self, tmp_path):
        out = _run(clips.normalize(_make_clip(tmp_path, size="1080x1920", audio=False), ".mp4"))
        v = _probe(out, tmp_path)[0]
        assert v["height"] == 768 and v["width"] % 2 == 0 and v["width"] < 768

    def test_a_small_clip_is_not_upscaled(self, tmp_path):
        out = _run(clips.normalize(_make_clip(tmp_path, size="640x360"), ".mp4"))
        v = _probe(out, tmp_path)[0]
        assert (v["width"], v["height"]) == (640, 360)

    def test_a_long_clip_is_cut_to_ten_seconds(self, tmp_path):
        out = _run(clips.normalize(_make_clip(tmp_path, seconds=12, size="320x240"), ".mp4"))
        assert int(_probe(out, tmp_path)[0]["nb_frames"]) == 250

    def test_a_mov_is_accepted(self, tmp_path):
        out = _run(clips.normalize(_make_clip(tmp_path, suffix=".mov", size="320x240"), ".mov"))
        assert _probe(out, tmp_path)[0]["r_frame_rate"] == "25/1"

    def test_a_clip_too_short_for_one_window_is_refused(self, tmp_path):
        """Under 49 frames the trainer cuts no window: it would count as an item and never
        train."""
        with pytest.raises(clips.ClipError, match="at least 2.0 s"):
            _run(clips.normalize(_make_clip(tmp_path, seconds=1.5, size="320x240"), ".mp4"))

    def test_not_a_video_is_refused(self, tmp_path):
        with pytest.raises(clips.ClipError):
            _run(clips.normalize(b"not a video", ".mp4"))


@needs_ffmpeg
class TestFrames:
    def test_evenly_spaced_stills(self, tmp_path):
        data = _run(clips.normalize(_make_clip(tmp_path, size="320x240"), ".mp4"))
        out = _run(clips.frames(data, 5))
        assert len(out) == 5 and all(b[:2] == b"\xff\xd8" for b in out)  # JPEG

    def test_the_contact_sheet_is_one_2x2_image(self, tmp_path):
        data = _run(clips.normalize(_make_clip(tmp_path, size="640x360"), ".mp4"))
        sheet = tmp_path / "sheet.jpg"
        sheet.write_bytes(_run(clips.contact_sheet(data)))
        v = _probe_file(sheet)
        assert (v["width"], v["height"]) == (1024, 576)  # two 512x288 frames by two


class TestScoring:
    def test_a_clip_scores_its_median_face_frame(self):
        """A head turn scores low against a frontal anchor without being somebody else; the
        median rides over it where a minimum or a mean would not."""
        assert clips.clip_score([0.7, 0.72, 0.2, None, 0.68]) == 0.69

    def test_a_clip_with_no_face_anywhere_is_none(self):
        assert clips.clip_score([None, None]) is None

    def test_the_floor_gives_every_window_its_own_frames(self):
        from app.training_plan import CLIP_WINDOWS
        assert clips.MIN_FRAMES - clips.WINDOW_FRAMES >= CLIP_WINDOWS - 1

    def test_a_clip_is_an_mp4(self):
        assert clips.is_clip("s3://b/x/A.MP4") and not clips.is_clip("s3://b/x/a.jpg")


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# ---------------------------------------------------------------------- routes and the plan

class _S3:
    def __init__(self, blobs=None):
        self.blobs = blobs or {}
        self.uploaded = []

    def download_bytes(self, uri):
        return self.blobs.get(uri, b"jpeg")

    def upload_bytes(self, data, key, bucket):
        self.uploaded.append(key)
        return f"s3://{bucket}/{key}"

    def head_object(self, uri):
        return {"Key": uri}


@pytest.mark.asyncio
class TestUpload:
    async def _upload(self, db, monkeypatch, files):
        from fastapi import UploadFile
        import io
        from app.models import Dataset
        from app.routes import datasets as mod
        fake = _S3()
        monkeypatch.setattr(mod, "s3", fake)
        ds = Dataset(name=f"u-{uuid.uuid4().hex[:6]}", images=[], prefix="datasets/u")
        db.add(ds)
        await db.flush()
        ups = [UploadFile(filename=n, file=io.BytesIO(b)) for n, b in files]
        return await mod.add_images(ds.id, ups, _user=None, db=db), fake

    async def test_a_clip_is_stored_normalized_as_mp4(self, db, monkeypatch):
        from app.routes import datasets as mod

        async def fake_normalize(data, suffix):
            return b"normalized"
        monkeypatch.setattr(mod.clips, "normalize", fake_normalize)
        out, fake = await self._upload(db, monkeypatch,
                                       [("me.MOV", b"raw"), ("a.jpg", b"jpg")])
        assert fake.uploaded == ["datasets/u/me.mp4", "datasets/u/a.jpg"]
        assert out.images == ["s3://wanly-images/datasets/u/me.mp4",
                              "s3://wanly-images/datasets/u/a.jpg"]

    async def test_one_bad_clip_refuses_the_whole_upload_by_name(self, db, monkeypatch):
        from fastapi import HTTPException
        from app.routes import datasets as mod

        async def fake_normalize(data, suffix):
            raise clips.ClipError("it is 1.2 s long")
        monkeypatch.setattr(mod.clips, "normalize", fake_normalize)
        with pytest.raises(HTTPException) as e:
            await self._upload(db, monkeypatch, [("a.jpg", b"jpg"), ("short.mp4", b"raw")])
        assert e.value.status_code == 422
        assert "short.mp4: it is 1.2 s long" in e.value.detail


@pytest.mark.asyncio
class TestScoreRoute:
    IMGS = ["s3://wanly-images/datasets/s/a.jpg", "s3://wanly-images/datasets/s/b.jpg",
            "s3://wanly-images/datasets/s/c.mp4"]

    async def _score(self, db, monkeypatch, anchor, clip_vecs):
        from app.config import settings
        from app.models import Dataset
        from app.routes import datasets as mod
        ds = Dataset(name=f"s-{uuid.uuid4().hex[:6]}", images=list(self.IMGS), prefix="p")
        db.add(ds)
        await db.flush()
        monkeypatch.setattr(mod, "s3", _S3())
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")

        async def fake_frames(data, n=clips.SCORE_FRAMES):
            return [b"f"] * len(clip_vecs)
        monkeypatch.setattr(mod.clips, "frames", fake_frames)
        sent = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                # a.jpg is the anchor [1,0]; b.jpg matches at 0.8; the clip's frames follow.
                return {"embeddings": [[1.0, 0.0], [0.8, 0.6], *clip_vecs]}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None):
                sent["n"] = len(json["images"])
                return _Resp()

        import httpx

        class _Shim:
            AsyncClient = _Client
            HTTPError = httpx.HTTPError
        monkeypatch.setattr(mod, "httpx", _Shim())
        return await mod.score_against_anchor(ds.id, anchor, _user=None, db=db), sent, ds

    async def test_a_clip_scores_the_median_of_its_frames(self, db, monkeypatch):
        vecs = [[0.7, 0.71], [0.72, 0.69], [0.2, 0.98], [], [0.68, 0.73]]
        out, sent, ds = await self._score(db, monkeypatch, self.IMGS[0], vecs)
        assert sent["n"] == 2 + 5          # two stills and five clip frames, one call
        by = {s.uri: s.cos for s in out.scores}
        assert by[self.IMGS[1]] == 0.8
        assert by[self.IMGS[2]] == 0.69
        assert ds.scores[self.IMGS[2]] == 0.69

    async def test_the_anchor_cannot_be_a_clip(self, db, monkeypatch):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as e:
            await self._score(db, monkeypatch, self.IMGS[2], [[1.0, 0.0]])
        assert e.value.status_code == 422 and "still image" in e.value.detail


@pytest.mark.asyncio
class TestTheCaptionLoop:
    async def test_a_clip_is_captioned_from_its_sheet_with_the_motion_prompt(self, db,
                                                                            monkeypatch):
        from app.joycaption import TRAINING_CAPTION, TRAINING_MOTION_CAPTION
        from app.models import Dataset
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        ds = Dataset(name=f"c-{uuid.uuid4().hex[:6]}", prefix="p",
                     images=["s3://b/p/a.jpg", "s3://b/p/m.mp4"])
        db.add(ds)
        await db.flush()
        monkeypatch.setattr(mod, "s3", _S3())

        async def sheet(data):
            return b"sheet"
        monkeypatch.setattr(mod.clips, "contact_sheet", sheet)
        asked = []

        async def fake_clip(db_, image, instruction):
            asked.append(("clip", image, instruction))
            return "medium shot, turns toward the camera and breaks into a smile"

        async def fake_still(db_, image, instruction=None, **k):
            asked.append(("still", image, instruction))
            return "close-up, smiling", instruction
        monkeypatch.setattr(cap_mod, "caption_clip_sheet", fake_clip)
        monkeypatch.setattr(cap_mod, "caption_image_bytes", fake_still)

        from contextlib import asynccontextmanager

        class _Q:
            @asynccontextmanager
            async def turn(self, *a, **k):
                yield
        import app.caption_queue as cq
        monkeypatch.setattr(cq, "queue", _Q())

        assert await mod.caption_dataset_images(db, ds.id, overwrite=False) == 2
        assert asked == [("still", b"jpeg", TRAINING_CAPTION),
                         ("clip", b"sheet", TRAINING_MOTION_CAPTION)]
        assert ds.captions["s3://b/p/m.mp4"].startswith("medium shot, turns")


def test_the_motion_prompt_asks_for_change_and_bans_identity():
    from app.joycaption import TRAINING_MOTION_CAPTION as p
    assert "verbs of change" in p and "no facial features" in p and "no names" in p


@pytest.mark.asyncio
class TestCrop:
    async def test_clips_are_never_cropped(self, db, monkeypatch):
        from fastapi import HTTPException
        from app.config import settings
        from app.models import Dataset
        from app.routes import datasets as mod
        ds = Dataset(name=f"k-{uuid.uuid4().hex[:6]}", prefix="p", images=["s3://b/p/m.mp4"])
        db.add(ds)
        await db.flush()
        monkeypatch.setattr(mod, "s3", _S3())
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")
        with pytest.raises(HTTPException) as e:
            await mod.crop_faces(ds.id, uris=None, _user=None, db=db)
        assert e.value.status_code == 422 and "clips" in e.value.detail


# --------------------------------------------------------------------------- the training plan

@pytest.mark.asyncio
class TestThePlan:
    async def _world_with_clips(self, db, n_clips=4):
        from tests.test_training_jobs import _world
        w = await _world(db)
        clip_uris = [f"s3://wanly-images/datasets/david/m{i}.mp4" for i in range(n_clips)]
        d = w["david"]
        d.images = list(d.images) + clip_uris
        d.captions = {**d.captions, **{u: f"turns and smiles {i}" for i, u in enumerate(clip_uris)}}
        d.scores = {**d.scores, **{u: 0.66 for u in clip_uris}}
        await db.flush()
        return w, clip_uris

    async def _plan(self, db, **body):
        from app.schemas.training import TrainingCreate
        from app.training_plan import plan_training
        return (await plan_training(db, TrainingCreate(**body))).public()

    async def test_stills_and_clips_train_as_two_groups(self, db):
        _, clip_uris = await self._world_with_clips(db)
        out = await self._plan(db, mode="solo", character="David", regularization=False,
                               steps=1200)
        assert out["ok"], out["problems"]
        groups = [(g["kind"], g["images"], g["num_repeats"], g["windows"]) for g in out["groups"]]
        assert groups == [("identity", 10, 10, 1), ("clip", 4, 5, 3)]
        clip = out["groups"][1]
        assert clip["sample_captions"][0] == "d@vid, man, turns and smiles 0"
        # 10 stills x 10 + 4 clips x 3 windows x 5 repeats
        assert out["samples_per_epoch"] == 100 + 60

    async def test_the_still_floor_ignores_clips(self, db):
        """Eight stills are the identity floor; clips do not make up a short set."""
        w, _ = await self._world_with_clips(db, n_clips=20)
        w["david"].images = w["david"].images[2:]   # 8 stills left... then drop one more
        w["david"].images = w["david"].images[1:]
        await db.flush()
        out = await self._plan(db, mode="solo", character="David", regularization=False)
        assert "too_few_images" in {p["code"] for p in out["problems"]}

    async def test_a_low_scoring_clip_blocks_like_a_still(self, db):
        w, clip_uris = await self._world_with_clips(db)
        w["david"].scores = {**w["david"].scores, clip_uris[0]: 0.1}
        await db.flush()
        out = await self._plan(db, mode="solo", character="David", regularization=False)
        assert "score_below_floor" in {p["code"] for p in out["problems"]}

    async def test_sdxl_leaves_clips_out_and_says_so(self, db):
        await self._world_with_clips(db)
        out = await self._plan(db, mode="solo", character="David", arch="sdxl", steps=960)
        assert out["ok"], out["problems"]
        assert [g["kind"] for g in out["groups"]] == ["identity"]
        assert out["groups"][0]["images"] == 10
        assert "sdxl_ignores_clips" in {w["code"] for w in out["warnings"]}

    async def test_regularization_is_sized_against_the_clip_samples_too(self, db):
        await self._world_with_clips(db)
        out = await self._plan(db, mode="solo", character="David", regularization=True)
        reg = [g for g in out["groups"] if g["kind"] == "regularization"][0]
        # 160 character samples at REG_RATIO 1.0 over the 30-image man pool -> 5 repeats
        assert reg["num_repeats"] == round(160 / 30)

    async def test_the_job_and_the_claim_carry_the_clip_group(self, db):
        from app.routes.training import _group_row
        from app.schemas.training import TrainingCreate
        from app.training_plan import plan_training
        await self._world_with_clips(db)
        plan = await plan_training(db, TrainingCreate(mode="solo", character="David",
                                                      regularization=False))
        row = _group_row(plan.groups[1])
        assert row["kind"] == "clip" and row["windows"] == 3 and len(row["images"]) == 4
        assert all(u.endswith(".mp4") for u in row["images"])


@pytest.mark.asyncio
class TestAddingByUri:
    async def test_a_repo_clip_cannot_skip_normalization(self, db):
        from fastapi import HTTPException
        from app.models import Dataset
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = Dataset(name=f"r-{uuid.uuid4().hex[:6]}", prefix="datasets/r", images=[])
        db.add(ds)
        await db.flush()
        with pytest.raises(HTTPException) as e:
            await update_dataset(ds.id, DatasetUpdate(images=["s3://wanly-images/outputs/x.mp4"]),
                                 _user=None, db=db)
        assert e.value.status_code == 422
        # A clip uploaded to a dataset (here, another set's) was normalized; it may move.
        out = await update_dataset(
            ds.id, DatasetUpdate(images=["s3://wanly-images/datasets/other/m.mp4"]),
            _user=None, db=db)
        assert out.images == ["s3://wanly-images/datasets/other/m.mp4"]
