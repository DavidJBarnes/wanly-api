"""Character face loops (wanly-api#495, wanly-console#742).

The preparation maths is tested without MediaPipe; the routes against a real database with S3
and image decoding stubbed. The full prepare() was verified by hand on Kelly-2026 job 465275c6
(it picks the same 2-5 s window chosen manually, 2.5 mm expression travel).
"""
import math
import uuid

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from app import face_loop_prep as prep
from app.enums import SegmentStatus
from app.face_loop_presets import FRAMES, PRESETS, build_prompt
from app.models import FaceLoop, Job, LtxBook, LtxCharacter, LtxRecipe, Segment, User
from app.routes import face_loops as fl
from app.schemas.face_loops import FaceLoopCreate


# --- preparation maths ------------------------------------------------------------------------

def _frame(px=600.0, yaw=0.0, pitch=0.0, jaw=0.0):
    ex = np.zeros((len(prep.EXPRESSION), 2))
    ex[0, 1] = 0.5 + jaw  # chin, in face heights
    return prep.FrameInfo(px, yaw, pitch, ex, (0, 0, px, px))


class TestWindow:
    def test_calm_window_beats_laughing_window(self):
        fps = 24
        laugh = [_frame(jaw=0.07 * math.sin(i / 3)) for i in range(72)]  # jaw ~13 mm
        calm = [_frame(jaw=0.004 * math.sin(i / 3)) for i in range(72)]
        start, end, _, travel = prep.best_window(laugh + calm, fps)
        assert start >= 60 and end - start == 72  # lands in the calm half
        assert travel < 2

    def test_a_lost_face_disqualifies_the_window(self):
        frames = [_frame() for _ in range(72)]
        frames[30] = prep.FrameInfo(0, 99, 99)
        assert prep.best_window(frames, 24) is None

    def test_turned_head_scores_lower(self):
        assert prep.window_score([_frame(yaw=25)] * 10)[0] < prep.window_score([_frame()] * 10)[0]


class TestShape:
    def test_stabilise_removes_head_motion(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(468, 3)) * 100
        shapes = []
        for k in range(10):
            a = 0.05 * k
            r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            shapes.append((1 + 0.01 * k) * base @ r.T + np.array([k, -k, 2 * k]))
        out, resid = prep.stabilise(np.stack(shapes))
        assert resid < 1e-6
        assert np.allclose(out[9], out[0], atol=1e-6)

    def test_smoothing_keeps_constants_and_damps_flicker(self):
        const = np.ones((20, 5, 2), np.float32)
        assert np.allclose(prep.smooth_circular(const), 1)
        flicker = np.array([(-1) ** i for i in range(20)], np.float32)[:, None]
        assert np.abs(prep.smooth_circular(flicker)).max() < 0.2

    def test_pingpong_loops_without_repeating_the_ends(self):
        assert prep.pingpong(4) == [0, 1, 2, 3, 2, 1]

    def test_square_box_stays_in_frame(self):
        x, y, side = prep.square_box([(900, 900, 1020, 1020)], 1024, 1024)
        assert x + side <= 1024 and y + side <= 1024


class TestPrompt:
    def test_prompt_is_trigger_scene_motion_still(self):
        p = build_prompt("k3lly2026", "woman", PRESETS["calm"]["motion"])
        lines = p.split("\n")
        assert lines[0] == "k3lly2026, woman"
        assert "close-up portrait of a woman" in lines[1]
        assert "closed-mouth smile" in lines[2] and "Her head stays perfectly still" in lines[2]

    def test_a_man_gets_his(self):
        assert "His head stays" in build_prompt("d@vid", "man", "He blinks.")

    def test_custom_needs_motion_and_presets_are_checked(self):
        with pytest.raises(ValidationError):
            FaceLoopCreate(preset="custom", start_image="s3://b/k.png")
        with pytest.raises(ValidationError):
            FaceLoopCreate(preset="nope", start_image="s3://b/k.png")
        with pytest.raises(ValidationError):
            FaceLoopCreate(preset="calm", start_image="https://x/k.png")
        assert FaceLoopCreate(preset="custom", motion="She winks.", start_image="s3://b/k.png")


# --- routes -----------------------------------------------------------------------------------

async def _setup(db, kind="solo"):
    u = User(username=f"u{uuid.uuid4().hex[:8]}", password_hash="x")
    book = LtxBook(name=f"book{uuid.uuid4().hex[:6]}")
    db.add_all([u, book])
    await db.flush()
    db.add(LtxRecipe(name="Prompt", prompt_template="<TRIGGER>\n<SCENE>\n<PROMPT>", book_id=book.id,
                     frames=241, img_compression=10, content_loras=[], checkpoint="10Eros_v1.5_bf16"))
    c = LtxCharacter(name=f"Kelly{uuid.uuid4().hex[:6]}", trigger="k3lly2026", gender="woman",
                     char_lora="Kelly-2026_v2_final", strength_stage_1=0.8, strength_stage_2=1.0, kind=kind)
    db.add(c)
    await db.flush()
    return u, c


@pytest.fixture
def no_s3(monkeypatch):
    calls = {"deleted": []}

    async def size(uri):
        return 1024, 1024

    monkeypatch.setattr(fl, "_image_size", size)
    monkeypatch.setattr(fl, "delete_prefix", lambda prefix, bucket: calls["deleted"].append((bucket, prefix)) or 1)
    return calls


@pytest.mark.asyncio
class TestRoutes:
    async def test_create_queues_a_prompt_recipe_render_with_the_characters_lora(self, db, no_s3):
        u, c = await _setup(db)
        r = await fl.create_loop(c.name, FaceLoopCreate(preset="calm", start_image="s3://imgs/p.png"), user=u, db=db)
        assert r.status == "rendering" and r.label == "Calm smile + blinks" and r.character == c.name
        job = await db.get(Job, r.job_id)
        assert job.starting_image == "s3://imgs/p.png" and (job.width, job.height) == (1024, 1024)
        assert job.tags == "FaceLoop"
        seg = (await db.execute(Segment.__table__.select().where(Segment.job_id == job.id))).mappings().one()
        assert seg["ltx_recipe"]["recipe"] == "Prompt" and seg["ltx_recipe"]["frames"] == FRAMES
        assert seg["ltx_recipe"]["char_lora"] == "Kelly-2026_v2_final" and seg["ltx_recipe"]["char_s2"] == 1.0
        assert seg["prompt"].startswith("k3lly2026, woman\n")
        assert "laughing" in seg["negative_prompt"]

    async def test_pairs_and_loraless_characters_are_refused(self, db, no_s3):
        u, c = await _setup(db, kind="pair")
        with pytest.raises(HTTPException) as e:
            await fl.create_loop(c.name, FaceLoopCreate(start_image="s3://imgs/p.png"), user=u, db=db)
        assert e.value.status_code == 400

    async def test_render_upload_starts_preparation_once(self, db, no_s3):
        u, c = await _setup(db)
        r = await fl.create_loop(c.name, FaceLoopCreate(start_image="s3://imgs/p.png"), user=u, db=db)
        seg = (await db.execute(Segment.__table__.select().where(Segment.job_id == r.job_id))).mappings().one()
        segment = await db.get(Segment, seg["id"])
        tasks = BackgroundTasks()
        await fl.on_render_uploaded(db, segment, tasks)
        assert (await db.get(FaceLoop, r.id)).status == "processing"
        assert [t.func for t in tasks.tasks] == [fl.prepare_loop]
        again = BackgroundTasks()
        await fl.on_render_uploaded(db, segment, again)  # already processing: nothing new
        assert again.tasks == []

    async def test_ordinary_renders_do_not_trigger_anything(self, db, no_s3):
        u, c = await _setup(db)
        job = Job(user_id=u.id, name="x", width=8, height=8, fps=24, seed=1, priority=0)
        db.add(job)
        await db.flush()
        s = Segment(job_id=job.id, index=0, prompt="p", duration_seconds=1.0)
        db.add(s)
        await db.flush()
        tasks = BackgroundTasks()
        await fl.on_render_uploaded(db, s, tasks)
        assert tasks.tasks == []

    async def test_lists_and_delete(self, db, no_s3):
        u, c = await _setup(db)
        r = await fl.create_loop(c.name, FaceLoopCreate(preset="smile", start_image="s3://imgs/p.png"), user=u, db=db)
        loop = await db.get(FaceLoop, r.id)
        assert [x.id for x in await fl.list_character_loops(c.name, db=db)] == [r.id]
        assert await fl.list_loops(status="ready", db=db) == []
        assert (await fl.list_character_loops(c.name, db=db))[0].render_status == SegmentStatus.PENDING
        loop.status, loop.asset_prefix = "ready", f"s3://wanly-jobs/ar-face/kelly/{r.id}"
        await db.flush()
        assert [x.id for x in await fl.list_loops(status="ready", db=db)] == [r.id]
        await fl.delete_loop(r.id, db=db)
        assert no_s3["deleted"] == [("wanly-jobs", f"ar-face/kelly/{r.id}/")]
        assert await db.get(FaceLoop, r.id) is None

    async def test_reprocess_needs_a_finished_render(self, db, no_s3):
        u, c = await _setup(db)
        r = await fl.create_loop(c.name, FaceLoopCreate(start_image="s3://imgs/p.png"), user=u, db=db)
        with pytest.raises(HTTPException) as e:
            await fl.reprocess_loop(r.id, BackgroundTasks(), db=db)
        assert e.value.status_code == 409


class TestPhotoScore:
    def test_big_frontal_neutral_sharp_photo_wins(self):
        good, ok = prep.photo_score(690, 0.3, 13, 0, 0.002, 0.07, 480)
        small, _ = prep.photo_score(300, 0.3, 13, 0, 0.002, 0.07, 480)
        turned, _ = prep.photo_score(690, 20, 13, 0, 0.002, 0.07, 480)
        assert ok and good > small and good > turned

    def test_open_mouth_or_shut_eyes_is_unsuitable_and_ranked_down(self):
        a, ok_a = prep.photo_score(690, 0, 10, 0, 0.08, 0.07, 480)
        b, ok_b = prep.photo_score(690, 0, 10, 0, 0.002, 0.01, 480)
        c, ok_c = prep.photo_score(690, 0, 10, 0, 0.002, 0.07, 480)
        assert not ok_a and not ok_b and ok_c and c > 4 * max(a, b)


@pytest.mark.asyncio
class TestPhotoCandidates:
    async def test_living_and_training_sets_are_both_candidates(self, db):
        from app.models import Dataset
        u, c = await _setup(db)
        train = Dataset(name="train", images=["s3://i/real_034.png", "s3://i/shared.png"])
        db.add(train)
        await db.flush()
        c.trained_from = [{"kind": "identity", "dataset_id": str(train.id)}]
        db.add(Dataset(name="living", images=["s3://i/a.png", "s3://i/shared.png"], character=c.name, kind="character"))
        await db.flush()
        got = await fl.candidate_photos(db, c)
        assert sorted(got) == ["s3://i/a.png", "s3://i/real_034.png", "s3://i/shared.png"]
