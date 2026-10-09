"""Character-LoRA training as claimable work (wanly-api#274).

The design decision under test is that training is PULL, like everything else here: the console
creates a row, a trainer claims it. That buys orphan reclaim and the heartbeat sweep for free,
and it only works if the claim gate and the reclaim rules are right — both of which have already
cost this project real incidents in their segment form.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind
from app.models import LtxCharacter, TrainingJob, Worker
from app.routes.training import ORPHANED_TRAINING_MINUTES, _publish_character, _reclaim_orphans
from app.schemas.training import TrainingCreate, TrainingProgress


def _run_sync(coro):
    import asyncio
    return asyncio.run(coro)


def _images(n=13):
    return [f"s3://wanly-images/2026-09-07/img{i}.jpg" for i in range(n)]


def _job(**kw):
    base = dict(id=uuid.uuid4(), character="pay", trigger="p@y", version=1,
                dataset_images=_images(), config={}, status=TrainingStatus.PENDING,
                created_at=datetime.now(timezone.utc))
    base.update(kw)
    return TrainingJob(**base)


def _worker(**kw):
    base = dict(id=uuid.uuid4(), friendly_name="3090/services", hostname="h",
                ip_address="1.2.3.4", kind=WorkerKind.TRAINER, status="online",
                last_heartbeat=datetime.now(timezone.utc))
    base.update(kw)
    return Worker(**base)


class _U:
    """The console user a route is called as. Only its id and name are read."""
    id = None
    username = "t"


def _set(prefix, n):
    return [f"s3://wanly-images/datasets/{prefix}/{i:03d}.jpg" for i in range(n)]


async def _world(db, *, floor_ok=True):
    """The registry and datasets of a correctly set-up world (#352), from which each
    guardrail test breaks exactly one thing.

    David (d@vid, man) and Kelly-2026 (k3lly2026, woman) are registered solo characters,
    each owning one anchored, scored, captioned character set; DavidKelly-2026 owns a
    composition set; there is a regularization pool per gender.
    """
    from app.models import Dataset

    def _ds(name, kind, owner, imgs, reg_class=None, scored=True):
        return Dataset(
            id=uuid.uuid4(), name=name, prefix=f"datasets/{name}", kind=kind,
            character=owner, reg_class=reg_class, images=imgs,
            anchor_uri=imgs[0] if kind == "character" else None,
            captions={u: f"medium shot, standing, look {i}" for i, u in enumerate(imgs)},
            scores=({u: (1.0 if i == 0 else 0.7) for i, u in enumerate(imgs)}
                    if scored and kind == "character" else {}),
            # Measured, every face comfortably big at training size (#432).
            faces=({u: {"width": 1024, "height": 1024, "face_px": 420.0, "faces": 1}
                    for u in imgs} if kind == "character" else {}))

    w = {
        "david_c": LtxCharacter(name="David", trigger="d@vid", gender="man", char_lora="none"),
        "kelly_c": LtxCharacter(name="Kelly-2026", trigger="k3lly2026", gender="woman",
                                char_lora="none"),
        "david": _ds("David", "character", "David", _set("david", 10)),
        "kelly": _ds("Kelly-2026", "character", "Kelly-2026", _set("kelly", 12)),
        "comp": _ds("DavidKelly-2026", "composition", "DavidKelly-2026", _set("comp", 9)),
        "reg_woman": _ds("Reg-woman", "regularization", None, _set("regw", 40), "woman"),
        "reg_man": _ds("Reg-man", "regularization", None, _set("regm", 30), "man"),
    }
    db.add_all(list(w.values()))
    await db.flush()
    return w


def _codes(plan) -> set[str]:
    return {p["code"] for p in (plan["problems"] if isinstance(plan, dict) else plan.problems)}


class TestTheRequest:
    """The request says WHO to train (#352). Triggers, genders, images and captions are the
    server's to derive, so the fields that used to carry them are refused, not ignored."""

    def test_a_solo_request_names_a_mode_and_a_character(self):
        t = TrainingCreate(mode="solo", character="David")
        assert t.mode == "solo" and t.members is None and t.datasets == {}

    def test_a_pair_request_names_its_members(self):
        t = TrainingCreate(mode="pair", character="DavidKelly-2026",
                           members=["David", "Kelly-2026"])
        assert t.members == ["David", "Kelly-2026"]

    def test_the_mode_is_required(self):
        with pytest.raises(ValueError):
            TrainingCreate(character="David")

    @pytest.mark.parametrize("legacy", [
        {"trigger": "d@vid"}, {"gender": "man"}, {"caption": "d@vid, man"},
        {"dataset_images": _images()}, {"dataset_id": str(uuid.uuid4())},
        {"identities": []}, {"second_character": "Me"},
    ])
    def test_the_legacy_shape_is_refused_with_a_sentence(self, legacy):
        """An old console still thinks it chooses the trigger. Silently ignoring what it
        typed would train under different words than it showed."""
        with pytest.raises(ValueError, match="no longer accepted"):
            TrainingCreate(mode="solo", character="David", **legacy)

    def test_a_character_cannot_contain_a_path(self):
        """It becomes a directory name and an output filename on the trainer."""
        for bad in ("../etc", "a/b", ".hidden", "two words"):
            with pytest.raises(ValueError):
                TrainingCreate(mode="solo", character=bad)


class TestTheClaimGate:
    """Only a trainer may claim. A render worker or a captioner picking up a 50-minute training
    run would be worse than it going unclaimed."""

    @pytest.mark.parametrize("kind", [WorkerKind.RENDER, WorkerKind.SERVICE])
    def test_only_a_trainer_kind_passes(self, kind):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "not worker_can(worker, WorkerKind.TRAINER)" in src  # reads kinds, not kind (094)
        assert kind != WorkerKind.TRAINER

    def test_the_gate_runs_before_the_reclaim_and_the_select(self):
        """Order matters: an ineligible worker must not even trigger a reclaim pass."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert src.index("WorkerKind.TRAINER") < src.index("_reclaim_orphans")

    def test_it_locks_with_skip_locked(self):
        """Two trainers polling at once must not be handed the same row."""
        import inspect
        from app.routes import training as mod
        assert "with_for_update(skip_locked=True)" in inspect.getsource(mod.claim_next_training_job)

    def test_next_is_routed_before_the_id_wildcard(self):
        """/training/{job_id} declared first would swallow /training/next and every poll would
        404 looking for a job called "next"."""
        from app.main import app
        paths = [r.path for r in app.routes if r.path.startswith("/training")]
        assert paths.index("/training/next") < paths.index("/training/{job_id}")


@pytest.mark.asyncio
class TestReclaim:
    async def test_a_claim_from_a_dead_worker_comes_back(self, db):
        w = _worker(last_heartbeat=datetime.now(timezone.utc) - timedelta(minutes=30))
        j = _job(status=TrainingStatus.RUNNING, worker_id=w.id, worker_name=w.friendly_name,
                 claimed_at=datetime.now(timezone.utc) - timedelta(minutes=40),
                 progress_log="[3/4] training")
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.PENDING
        assert j.worker_id is None and j.progress_log is None

    async def test_a_live_worker_making_progress_keeps_its_claim(self, db):
        """The rule that matters. A training run is ~50 minutes; stealing one from a worker that
        is actually training is how two GPUs once converged on the same segment."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.RUNNING, worker_id=w.id,
                 claimed_at=datetime.now(timezone.utc) - timedelta(hours=2),
                 progress_log="[3/4] step 800/1200")
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.RUNNING

    async def test_a_live_idle_worker_with_no_progress_loses_it(self, db):
        """The lost-claim-response case: the row was assigned before the answer was sent, and
        the answer never arrived. An empty progress log is the only trustworthy evidence."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.CLAIMED, worker_id=w.id, progress_log=None,
                 claimed_at=datetime.now(timezone.utc)
                 - timedelta(minutes=ORPHANED_TRAINING_MINUTES + 5))
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.PENDING

    async def test_a_recent_claim_is_left_alone(self, db):
        """Staging a dataset and caching latents is legitimately quiet for minutes."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.CLAIMED, worker_id=w.id, progress_log=None,
                 claimed_at=datetime.now(timezone.utc) - timedelta(minutes=2))
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.CLAIMED


@pytest.mark.asyncio
class TestPublishing:
    async def test_completing_creates_the_character(self, db):
        """GET /loras makes the FILE discoverable, but a pose fills <TRIGGER> from an
        LtxCharacter row — without one the LoRA exists and no recipe can reach it."""
        j = _job(output_lora_path="s3://ltx-loras/character/pay_v1_e05.safetensors")
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.char_lora == "pay_v1_e05"
        assert row.trigger == "p@y"

    async def test_retraining_repoints_the_existing_character(self, db):
        """The whole point of a v2. The strengths are left alone — they may be hand-tuned."""
        db.add(LtxCharacter(name="pay", char_lora="pay_v1_e05.safetensors", trigger="p@y",
                            strength_stage_1=0.9, strength_stage_2=1.4))
        await db.flush()
        j = _job(version=2, output_lora_path="s3://ltx-loras/character/pay_v2_e03.safetensors")
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.char_lora == "pay_v2_e03"
        assert row.strength_stage_1 == 0.9, "hand-tuned strengths were overwritten"

    async def test_the_gender_the_run_captioned_lands_on_the_row(self, db):
        """The caption was "p@y, woman"; the row must say so or <TRIGGER> renders the
        trigger without the word the identity was bound to (wanly-console#487)."""
        j = _job(config={"gender": "woman", "caption": "p@y, woman"},
                 output_lora_path="s3://ltx-loras/character/pay_v1_e05.safetensors")
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.gender == "woman"

    async def test_a_run_without_a_gender_leaves_a_hand_set_one_alone(self, db):
        db.add(LtxCharacter(name="pay", char_lora="pay_v1_e05", trigger="p@y", gender="woman"))
        await db.flush()
        j = _job(version=2, config={"caption": "p@y, close-up"},
                 output_lora_path="s3://ltx-loras/character/pay_v2_e03.safetensors")
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.gender == "woman"

    async def test_nothing_is_published_without_a_file(self, db):
        j = _job(output_lora_path=None)
        await _publish_character(db, j)
        from sqlalchemy import select
        assert (await db.execute(select(LtxCharacter))).scalars().first() is None


class TestReporting:
    def test_an_omitted_field_does_not_blank_a_stored_one(self):
        """The mistake the worker heartbeat had to fix twice: a partial report erasing good
        data written by a fuller one."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.update_training_job)
        assert "if value is not None:" in src

    def test_a_progress_report_need_not_carry_a_status(self):
        p = TrainingProgress(step=400)
        assert p.status is None and p.progress_log is None

    def test_terminal_states_are_the_three_that_stop_work(self):
        assert TRAINING_TERMINAL == {TrainingStatus.COMPLETED, TrainingStatus.FAILED,
                                     TrainingStatus.CANCELLED}


class TestTheClaimIsAtomicEnough:
    """A claim that mutates and then fails leaves a job owned by a worker that never heard
    about it — wanly-api#242 in a new place. It recovers, but only after the grace period,
    with the queue stopped meanwhile. Cheaper to not create it."""

    def test_presigning_happens_before_the_row_is_marked(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert src.index("generate_presigned_url") < src.index("job.status = TrainingStatus.CLAIMED")

    def test_a_presign_failure_says_the_job_is_still_queued(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "still queued" in src and "503" in src


class TestCharacterNaming:
    """`character` is the LtxCharacter name, `@` and all — not a filesystem-safe version.

    Verified against a real API: training `p@y` v3 repointed the existing `p@y` row from
    `pay_v2_e05` to `pay_v3_e04.safetensors`, one row. Passing `pay` instead created a second
    character row pointing at the same LoRA while every recipe kept using the old one.
    """

    def test_an_at_sign_is_allowed_in_the_character(self):
        t = TrainingCreate(mode="solo", character="p@y")
        assert t.character == "p@y"

    def test_the_filename_stem_is_decided_at_creation_not_at_upload(self):
        """It is a field on the request, defaulted and correctable, rather than a guess made
        silently when the file lands. See TestTheLoraFilename for why."""
        import inspect
        from app.routes import training as mod
        assert "lora_name" in inspect.getsource(mod.create_training_job)
        assert 'get("lora_name")' in inspect.getsource(mod._artifact_key)

    def test_the_trigger_never_feeds_the_filename(self):
        """They are different things. The trigger keeps whatever trained."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.upload_training_artifact)
        assert "job.trigger" not in src

    @pytest.mark.asyncio
    async def test_retraining_a_character_with_an_at_sign_repoints_one_row(self, db):
        from sqlalchemy import select
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05.safetensors", trigger="p@y"))
        await db.flush()
        j = _job(character="p@y", version=3,
                 output_lora_path="s3://ltx-loras/character/pay_v3_e04.safetensors")
        await _publish_character(db, j)
        await db.flush()
        rows = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "p@y"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].char_lora == "pay_v3_e04"


class TestTheLoraFilename:
    """The stem is ASKED, not derived.

    Stripping `p@y` gives `py`. The file this project has actually been rendering with is
    `pay_v2_e05.safetensors` — a human read `@` as `a`, and no rule produces that. `@`->`a` is a
    transliteration, and a table for it generalises badly: k3lly2026 keeps its digits, so
    `3`->`e` would be wrong.
    """

    def test_the_default_strips_and_never_invents_a_letter(self):
        from app.routes.training import _default_lora_name
        assert _default_lora_name("p@y") == "py"
        assert _default_lora_name("k3lly2026") == "k3lly2026"

    def test_a_name_of_only_unsafe_characters_still_yields_something(self):
        from app.routes.training import _default_lora_name
        assert _default_lora_name("@@@") == "lora"

    def test_an_explicit_name_is_accepted(self):
        t = TrainingCreate(mode="solo", character="p@y", lora_name="pay")
        assert t.lora_name == "pay"

    def test_an_unsafe_explicit_name_is_refused(self):
        """Otherwise the field just moves the problem."""
        for bad in ("p@y", "a/b", "with space"):
            with pytest.raises(ValueError):
                TrainingCreate(mode="solo", character="x", lora_name=bad)

    def test_the_upload_uses_the_stored_stem_not_a_fresh_guess(self):
        from app.routes.training import _artifact_key
        job = _job(character="p@y", version=3, config={"lora_name": "pay"})
        assert _artifact_key(job, 4, False) == "character/pay_v3_e04.safetensors"

    def test_both_upload_paths_name_the_file_the_same_way(self):
        """The CLI backfill and the trainer's direct PUT must land on identical keys, or a
        backfilled epoch sits beside the live one under a second name."""
        import inspect
        from app.routes import training as mod
        assert "_artifact_key(job, epoch, final)" in inspect.getsource(mod.upload_training_artifact)
        assert "_artifact_key(job, epoch, final)" in inspect.getsource(mod.presign_training_artifact)


class TestEveryCheckpointIsOffered:
    """Choosing between epochs is a judgement made by eye at a fixed seed. Loss does not rank
    them — a confident "later epochs overfit" call read off a loss curve was refuted outright on
    d0ggyff — so a console that only offers the final epoch has thrown the decision away."""

    def test_uploading_appends_rather_than_replaces(self):
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=["s3://ltx-loras/character/pay_v1_e01.safetensors"])
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e02.safetensors")
        assert job.checkpoints == ["s3://ltx-loras/character/pay_v1_e01.safetensors",
                                   "s3://ltx-loras/character/pay_v1_e02.safetensors"]

    def test_the_same_uri_is_not_recorded_twice(self):
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=["s3://ltx-loras/character/pay_v1_e01.safetensors"])
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e01.safetensors")
        assert len(job.checkpoints) == 1

    def test_output_lora_path_tracks_the_most_recent(self):
        """It is what the character row points at until someone picks differently."""
        from app.routes.training import _record_checkpoint
        job = _job()
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e01.safetensors")
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_final.safetensors")
        assert job.output_lora_path == "s3://ltx-loras/character/pay_v1_final.safetensors"

    def test_both_upload_paths_record_through_one_function(self):
        import inspect
        from app.routes import training as mod
        assert "_record_checkpoint(job, uri)" in inspect.getsource(mod.upload_training_artifact)
        assert "_record_checkpoint(job, uri)" in inspect.getsource(mod.commit_training_artifact)


class TestTheFinalCheckpointIsNamed:
    """The trainer writes the checkpoint at the end of the step count WITHOUT an epoch number.
    It is usually a partial epoch and it is the one with the most training in it; published
    as a bare `pay_v2.safetensors` it read as "the v2" and hid that four other candidates
    existed. The console labelled it with the whole filename."""

    def test_final_gets_its_own_tag(self):
        from app.routes.training import _artifact_key
        job = _job(character="p@y", version=2, config={"lora_name": "pay"})
        assert _artifact_key(job, None, True) == "character/pay_v2_final.safetensors"

    def test_an_epoch_is_zero_padded(self):
        from app.routes.training import _artifact_key
        assert _artifact_key(_job(config={"lora_name": "pay"}), 5, False).endswith("_e05.safetensors")

    def test_the_url_endpoint_insists_on_one_or_the_other(self):
        """A checkpoint with neither an epoch nor `final` would land on the bare name."""
        from fastapi import HTTPException
        from app.routes.training import presign_training_artifact

        class _DB:
            async def get(self, *_a):
                return _job()

        with pytest.raises(HTTPException) as e:
            _run_sync(presign_training_artifact(uuid.uuid4(), epoch=None, final=False, db=_DB()))
        assert e.value.status_code == 422


class TestACommitIsBelievedOnlyAfterLooking:
    """The trainer PUTs straight to S3 and then says so. Saying so must not be enough: a PUT
    that failed, was truncated, or went to somebody else's key would otherwise put a dead
    download button on the page -- exactly what this replaces."""

    def _job(self):
        return _job(character="p@y", version=2, config={"lora_name": "pay"})

    def test_a_uri_of_another_job_is_refused(self):
        from app.routes.training import _belongs_to
        job = self._job()
        assert not _belongs_to(job, "s3://ltx-loras/character/laura_v2_e01.safetensors")
        assert not _belongs_to(job, "s3://ltx-loras/character/pay_v20_e01.safetensors")
        assert not _belongs_to(job, "s3://ltx-loras/character/pay_v3_e01.safetensors")
        assert not _belongs_to(job, "s3://wanly-images/character/pay_v2_e01.safetensors")

    def test_its_own_names_are_accepted(self):
        from app.routes.training import _belongs_to
        job = self._job()
        assert _belongs_to(job, "s3://ltx-loras/character/pay_v2_e01.safetensors")
        assert _belongs_to(job, "s3://ltx-loras/character/pay_v2_final.safetensors")

    def test_a_put_that_did_not_land_records_nothing(self, monkeypatch):
        from fastapi import HTTPException
        from app.routes import training as mod
        job = self._job()
        monkeypatch.setattr(mod.s3, "head_object", lambda uri: None)

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(mod.commit_training_artifact(
                job.id, uri="s3://ltx-loras/character/pay_v2_e01.safetensors", db=_DB()))
        assert e.value.status_code == 409
        assert not job.checkpoints

    def test_a_truncated_object_records_nothing(self, monkeypatch):
        from fastapi import HTTPException
        from app.routes import training as mod
        job = self._job()
        monkeypatch.setattr(mod.s3, "head_object", lambda uri: {"Key": "k", "Size": 1234})

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(mod.commit_training_artifact(
                job.id, uri="s3://ltx-loras/character/pay_v2_e01.safetensors", db=_DB()))
        assert e.value.status_code == 409
        assert not job.checkpoints

    async def test_a_real_object_is_recorded(self, db, monkeypatch):
        from app.routes import training as mod
        job = self._job()
        db.add(job)
        await db.commit()
        monkeypatch.setattr(mod.s3, "head_object",
                            lambda uri: {"Key": "k", "Size": 650 * 1024 * 1024})
        # Present, big enough, AND a real safetensors header (the 2026-09-08 zero-filled
        # final passed the first two).
        monkeypatch.setattr(mod.s3, "safetensors_header_ok", lambda uri: True)

        out = await mod.commit_training_artifact(
            job.id, uri="s3://ltx-loras/character/pay_v2_final.safetensors", db=db)

        assert out.checkpoints == ["s3://ltx-loras/character/pay_v2_final.safetensors"]
        assert out.output_lora_path == "s3://ltx-loras/character/pay_v2_final.safetensors"

    async def test_a_headerless_object_is_refused_and_records_nothing(self, db, monkeypatch):
        """The zero-filled Me_v2_final of 2026-09-08: right size, no header."""
        import pytest
        from fastapi import HTTPException
        from app.routes import training as mod
        job = self._job()
        db.add(job)
        await db.commit()
        monkeypatch.setattr(mod.s3, "head_object",
                            lambda uri: {"Key": "k", "Size": 650 * 1024 * 1024})
        monkeypatch.setattr(mod.s3, "safetensors_header_ok", lambda uri: False)
        with pytest.raises(HTTPException) as e:
            await mod.commit_training_artifact(
                job.id, uri="s3://ltx-loras/character/pay_v2_final.safetensors", db=db)
        assert e.value.status_code == 422 and "no safetensors header" in e.value.detail
        assert not (job.checkpoints or [])


class TestAFinishedRunCanBeDeleted:
    """Four failed and cancelled p@y rows sat above the one that worked, forever."""

    def test_the_files_go_with_it_by_default(self):
        """A deleted run whose checkpoints linger in the library is what someone deleting a
        run does not expect (console#464)."""
        import inspect
        from app.routes.training import delete_training_job
        sig = inspect.signature(delete_training_job)
        assert sig.parameters["purge"].default is True
        assert "s3.delete_object" in inspect.getsource(delete_training_job)

    async def test_a_character_rendering_with_one_stops_the_purge(self, db, monkeypatch):
        """Deleting the file under a character breaks every recipe that names it."""
        from fastapi import HTTPException
        from app.routes import training as mod
        job = _job(status=TrainingStatus.COMPLETED,
                   checkpoints=["s3://ltx-loras/character/pay_v2_final.safetensors"])
        db.add(job)
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_final", trigger="p@y"))
        await db.commit()
        deleted = []
        monkeypatch.setattr(mod.s3, "delete_object", lambda uri: deleted.append(uri))

        with pytest.raises(HTTPException) as e:
            await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert e.value.status_code == 409
        assert "p@y" in e.value.detail
        assert deleted == []
        assert await db.get(TrainingJob, job.id) is not None

    async def test_a_file_that_cannot_be_deleted_keeps_the_row_and_says_why(self, db, monkeypatch):
        """It 500'd in production: the role had PutObject on character/* and not DeleteObject."""
        from fastapi import HTTPException
        from app.routes import training as mod
        job = _job(status=TrainingStatus.COMPLETED,
                   checkpoints=["s3://ltx-loras/character/pay_v2_final.safetensors"])
        db.add(job)
        await db.commit()

        def denied(uri):
            raise PermissionError("AccessDenied")
        monkeypatch.setattr(mod.s3, "delete_object", denied)
        with pytest.raises(HTTPException) as e:
            await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert e.value.status_code == 503 and "DeleteObject" in e.value.detail
        assert await db.get(TrainingJob, job.id) is not None

    async def test_otherwise_the_files_are_deleted(self, db, monkeypatch):
        from app.routes import training as mod
        uris = ["s3://ltx-loras/character/pay_v2_e01.safetensors",
                "s3://ltx-loras/character/pay_v2_final.safetensors"]
        job = _job(status=TrainingStatus.COMPLETED, checkpoints=uris)
        db.add(job)
        await db.commit()
        deleted = []
        monkeypatch.setattr(mod.s3, "delete_object", lambda uri: deleted.append(uri))

        await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert sorted(deleted) == sorted(uris)
        assert await db.get(TrainingJob, job.id) is None

    def test_a_live_job_is_refused(self):
        from fastapi import HTTPException
        from app.routes.training import delete_training_job
        job = _job(status=TrainingStatus.RUNNING)

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(delete_training_job(job.id, purge=True, _user=None, db=_DB()))
        assert e.value.status_code == 409

    async def test_a_terminal_job_goes(self, db):
        from app.routes.training import delete_training_job
        job = _job(status=TrainingStatus.FAILED)
        db.add(job)
        await db.commit()
        await delete_training_job(job.id, purge=False, _user=None, db=db)
        assert await db.get(TrainingJob, job.id) is None


@pytest.mark.asyncio
class TestOnlyDownloadableCheckpointsSurvive:
    """The console turns every `checkpoints` entry into a download button pointed at
    GET /files, which serves an S3 URI and nothing else. Early trainer builds recorded the
    container-local output path instead of uploading, so a completed job carried five entries
    that all 404. Re-uploading has to REPLACE that list, not extend it."""

    async def _upload(self, db, job, monkeypatch, epoch=None):
        import io
        from fastapi import UploadFile
        from app.routes import training as mod

        uploaded = {}

        def fake_upload(data, key, bucket):
            uploaded["key"] = key
            return f"s3://{bucket}/{key}"

        monkeypatch.setattr(mod.s3, "upload_bytes", fake_upload)
        # Over the 10 MB floor the route enforces against truncated uploads.
        payload = b"\0" * (11 * 1024 * 1024)
        f = UploadFile(filename="lora.safetensors", file=io.BytesIO(payload))
        return await mod.upload_training_artifact(job.id, lora=f, epoch=epoch, db=db)

    async def test_a_container_local_path_is_dropped(self, db, monkeypatch):
        job = _job(status=TrainingStatus.COMPLETED, config={"lora_name": "pay"},
                   checkpoints=["/loras/p@y/ltx23b-v2/output/p@y_v2-000003.comfy.safetensors"])
        db.add(job)
        await db.commit()

        out = await self._upload(db, job, monkeypatch, epoch=3)

        assert all(c.startswith("s3://") for c in out.checkpoints), out.checkpoints
        assert len(out.checkpoints) == 1

    async def test_earlier_s3_epochs_are_still_kept(self, db, monkeypatch):
        """Dropping the dead entries must not also drop the good ones — every epoch is
        offered because loss does not rank them."""
        prior = "s3://ltx-loras/character/pay_v2_e01.safetensors"
        job = _job(status=TrainingStatus.COMPLETED, config={"lora_name": "pay"},
                   checkpoints=[prior, "/loras/output/p@y_v2-000002.comfy.safetensors"])
        db.add(job)
        await db.commit()

        out = await self._upload(db, job, monkeypatch, epoch=2)

        assert prior in out.checkpoints
        assert len(out.checkpoints) == 2


@pytest.mark.asyncio
class TestCancellingActuallyCancels:
    """Nothing in the API reaches into the GPU box, so a cancelled job keeps reporting
    `running` until the trainer notices. Writing that report unconditionally undid the cancel
    within seconds: the button appeared to work and the run went to completion."""

    async def _patch(self, db, job, **fields):
        from app.routes.training import update_training_job
        return await update_training_job(job.id, TrainingProgress(**fields), db=db)

    async def test_a_progress_report_does_not_revive_a_cancelled_job(self, db):
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING, step=412)

        assert out.status == TrainingStatus.CANCELLED
        # The progress itself is still recorded — it is what the run was doing when it stopped.
        assert out.step == 412

    async def test_a_cancelled_job_is_not_completed_by_a_late_finish(self, db):
        """A trainer that finishes before it notices has not made the run wanted again."""
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.COMPLETED)

        assert out.status == TrainingStatus.CANCELLED

    async def test_the_reply_tells_the_trainer_it_was_cancelled(self, db):
        """This is how the trainer finds out — there is no second call."""
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING)

        assert out.status == TrainingStatus.CANCELLED

    async def test_an_ordinary_running_job_still_advances(self, db):
        job = _job(status=TrainingStatus.CLAIMED)
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING, step=7)

        assert out.status == TrainingStatus.RUNNING
        assert out.step == 7


class TestOperatorNotes:
    """A note is a human write on a machine-reported row (wanly-console#484).

    The reason it is its own route rather than a field on the trainer's PATCH: TrainingProgress
    writes only what it is handed, which is exactly the wrong contract for a note -- a report
    that simply does not mention it would look like a clear. Here the whole field is the unit,
    and an explicit blank is a deliberate clear rather than an accident of omission.
    """

    async def _set(self, db, job, notes):
        from app.routes.training import set_training_notes
        from app.schemas.training import TrainingNotes
        return await set_training_notes(job.id, TrainingNotes(notes=notes), _user=None, db=db)

    async def test_a_note_is_set_and_returned(self, db):
        job = _job()
        db.add(job)
        await db.commit()

        out = await self._set(db, job, "e03 was the one that looked right; e04 plastic")

        assert out.notes == "e03 was the one that looked right; e04 plastic"

    async def test_a_trainer_report_cannot_touch_the_note(self, db):
        """The guarantee the separate route exists for: the report channel omits what it
        does not have, and omission must never mean clear."""
        job = _job(notes="picked e03 by eye at seed 42")
        db.add(job)
        await db.commit()

        from app.routes.training import update_training_job
        out = await update_training_job(
            job.id, TrainingProgress(step=99, progress_log="step 99"), db=db)

        assert out.step == 99
        assert out.notes == "picked e03 by eye at seed 42"

    async def test_an_explicit_blank_clears_it(self, db):
        job = _job(notes="obsolete")
        db.add(job)
        await db.commit()

        out = await self._set(db, job, None)

        assert out.notes is None

    async def test_an_oversized_note_is_refused(self):
        """A localStorage spill-out can be half a DVD image; a note is a note."""
        import pydantic
        from app.schemas.training import TrainingNotes
        with pytest.raises(pydantic.ValidationError, match="at most 20000"):
            TrainingNotes(notes="x" * 20001)



class TestNothingGoesUpByDefault:
    """A 650 MB checkpoint takes ~18 minutes to leave the 3090 and "I often only want 1 or 2
    epochs". Since #413 nothing uploads unasked: each checkpoint is tried, then uploaded or
    deleted from the Training page."""

    def test_the_default_policy_is_none(self):
        req = TrainingCreate(mode="solo", character="p@y")
        assert req.publish == "none"

    def test_final_is_still_a_choice(self):
        assert TrainingCreate(mode="solo", character="p@y", publish="final").publish == "final"

    def test_all_is_the_other_choice_and_nothing_else_is(self):
        assert TrainingCreate(mode="solo", character="p@y", publish="all").publish == "all"
        with pytest.raises(ValueError):
            TrainingCreate(mode="solo", character="p@y", publish="some")

    def test_the_policy_reaches_the_job_config(self):
        import inspect
        from app.routes import training as mod
        assert '"publish": body.publish' in inspect.getsource(mod.create_training_job)

    async def test_a_publish_request_is_recorded_once(self, db):
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED,
                   epochs=[{"label": "e01", "step": 80, "loss": 0.7},
                           {"label": "final", "step": 100, "loss": 0.68}],
                   checkpoints=["s3://ltx-loras/character/pay_v1_final.safetensors"])
        db.add(job)
        await db.commit()
        out = await request_publish(job.id, label="e01", _user=None, db=db)
        out = await request_publish(job.id, label="e01", _user=None, db=db)
        assert out.publish_requests == ["e01"]

    async def test_a_checkpoint_the_run_never_wrote_is_refused(self, db):
        from fastapi import HTTPException
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED, epochs=[{"label": "final", "step": 100}])
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await request_publish(job.id, label="e07", _user=None, db=db)
        assert e.value.status_code == 404

    async def test_one_already_in_the_bucket_is_refused(self, db):
        from fastapi import HTTPException
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED, epochs=[{"label": "final", "step": 100}],
                   checkpoints=["s3://ltx-loras/character/pay_v1_final.safetensors"])
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await request_publish(job.id, label="final", _user=None, db=db)
        assert e.value.status_code == 409


class TestThePublishedCharacterStoresTheStem:
    async def test_no_extension_on_the_row(self, db):
        """Every other row stores `pay_v2_e05`; the console compares stems."""
        job = _job(character="Me", output_lora_path="s3://ltx-loras/character/david_v1_final.safetensors")
        await _publish_character(db, job)
        await db.commit()
        from sqlalchemy import select
        c = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == "Me"))).scalar_one()
        assert c.char_lora == "david_v1_final"


class TestTheLoraHasAFace:
    async def test_the_dataset_anchor_is_snapshotted_at_creation(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(
            TrainingCreate(mode="solo", character="David"), user=_U(), db=db)
        assert job.thumbnail_uri == w["david"].anchor_uri

    async def test_publishing_puts_the_face_on_the_character(self, db):
        job = _job(character="p@y", output_lora_path="s3://ltx-loras/character/pay_v3_final.safetensors",
                   thumbnail_uri="s3://wanly-images/datasets/x/anchor.jpg")
        await _publish_character(db, job)
        await db.commit()
        from sqlalchemy import select
        c = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == "p@y"))).scalar_one()
        assert c.image_uri == "s3://wanly-images/datasets/x/anchor.jpg"

    def test_a_progress_report_can_carry_the_curve_and_the_epochs(self):
        p = TrainingProgress(loss_log=[[10, 0.9], [20, 0.8]],
                             epochs=[{"label": "e01", "step": 80, "loss": 0.7}])
        assert p.loss_log[-1] == [20, 0.8]
        import inspect
        from app.routes import training as mod
        assert '"loss_log", "epochs"' in inspect.getsource(mod.update_training_job)


class TestTheJointRun:
    """A PRE-#352 joint run (wanly-api#102, #106): one LoRA trained on several groups at once.

    These jobs still exist, and retrying one must publish where it always did, so the old
    publish path is kept for any job without a `mode` -- that is what these pin down. A
    #352 pair publishes to its own row instead: see TestAPairPublishesToItsOwnRow.

    #102 was two identities; #106 added COMPOSITION groups -- frames containing BOTH
    characters, captioned with both triggers. That is the group that teaches the model the
    identities appear together; a run of solo sets alone produced a LoRA that held one face
    and dropped the other (wanly-gpu-docker#100/#102)."""

    def _imgs(self, prefix, n=13):
        return [f"s3://wanly-images/2026-09-11/{prefix}{i}.jpg" for i in range(n)]

    async def test_publishing_a_joint_run_records_both_triggers(self, db):
        """The character row carries ONE trigger; a LoRA trained on several caption pairs
        must announce them all in the one phrase that fills <TRIGGER>."""
        j = _job(
            output_lora_path="s3://ltx-loras/character/payme_v1_final.safetensors",
            config={"gender": "woman", "caption": "p@y, woman"},
            identities=[{"character": "Me", "trigger": "d@vid", "gender": "man",
                         "caption": "d@vid, man", "images": self._imgs("m"),
                         "num_repeats": 10}])
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.trigger == "p@y, woman and d@vid, man"

    async def test_a_composition_group_does_not_add_a_pair(self, db):
        """#106: the composition group's caption repeats the identity pairs. Including it
        in the phrase would say the same face twice."""
        j = _job(
            output_lora_path="s3://ltx-loras/character/payme_v1_final.safetensors",
            config={"gender": "woman"},
            identities=[
                {"character": "Me", "trigger": "d@vid", "gender": "man",
                 "caption": "d@vid, man", "images": self._imgs("m")},
                # composition: no trigger, caption names both
                {"character": None, "trigger": None, "gender": None,
                 "caption": "p@y, woman and d@vid, man", "images": self._imgs("c")},
            ])
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.trigger == "p@y, woman and d@vid, man", (
            "the composition group's pair leaked into the phrase")

    async def test_publishing_a_single_run_stays_single(self, db):
        """Every run before #102 is single-identity; its row's trigger is unchanged."""
        j = _job(output_lora_path="s3://ltx-loras/character/pay_v1_e05.safetensors")
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.trigger == "p@y"
        assert "&" not in row.trigger


class TestTheClaimDeliversEveryGroup:
    """#106: a claim must carry every group's URLs, not just group 0's. A joint run that
    delivered group 0 only would stage a single-identity dataset silently -- worse than a
    503, because it trains and reports success."""

    def test_the_presign_loop_covers_every_identity_group(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "for g in (job.identities or [])" in src
        assert '"download_urls": g_urls' in src

    def test_a_presign_failure_anywhere_aborts_the_whole_claim(self):
        """All groups presign inside the one try, so a failure leaves the row untouched
        rather than claiming a job whose second dataset never arrived."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        presign = src.index("group_payload = []")
        mutate = src.index("job.status = TrainingStatus.CLAIMED")
        assert presign < mutate

    def test_the_claim_response_carries_the_group_list(self):
        from app.schemas.training import TrainingClaimResponse
        fields = TrainingClaimResponse.model_fields
        assert "identities" in fields
        assert "second_download_urls" not in fields


class TestTheProvenance:
    """Which DATASETS trained a character's LoRA (migration 099). The names are snapshotted
    at creation and stamped on the character at publish, so a rename later does not rewrite
    what trained -- the question "where did this face come from" always has an answer."""

    def _imgs(self, prefix, n=13):
        return [f"s3://wanly-images/2026-09-11/{prefix}{i}.jpg" for i in range(n)]

    async def test_creation_records_the_dataset_per_group(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(
            TrainingCreate(mode="pair", character="DavidKelly-2026",
                           members=["David", "Kelly-2026"]), user=_U(), db=db)
        assert job.config["dataset"] == {"id": str(w["david"].id), "name": "David",
                                         "count": 10}
        assert [g["dataset"]["name"] for g in job.identities] == [
            "Kelly-2026", "DavidKelly-2026", "Reg-man", "Reg-woman"]

    async def test_publishing_stamps_every_group(self, db):
        j = _job(
            output_lora_path="s3://ltx-loras/character/payme_v1_final.safetensors",
            config={"gender": "woman", "caption": "p@y, woman",
                    "dataset": {"id": "ds-payton", "name": "Payton Synthetic", "count": 55}},
            identities=[{"character": "Me", "trigger": "d@vid", "gender": "man",
                         "caption": "d@vid, man", "images": self._imgs("m"),
                         "dataset": {"id": "ds-me", "name": "Me Synthetic", "count": 50}}])
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.trained_from == [
            {"dataset_id": "ds-payton", "name": "Payton Synthetic", "count": 55},
            {"dataset_id": "ds-me", "name": "Me Synthetic", "count": 50},
        ]

    async def test_a_run_without_recorded_names_counts_the_images(self, db):
        """Pre-099 jobs recorded no dataset names. The stamp still says how many images,
        with null names rather than a guess."""
        j = _job(
            output_lora_path="s3://ltx-loras/character/pay_v1_e05.safetensors",
            config={"gender": "woman"},
            identities=[{"character": "Me", "trigger": "d@vid",
                         "images": self._imgs("m"), "num_repeats": 10}])
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert [d["count"] for d in row.trained_from] == [13, 13]
        assert all(d["name"] is None for d in row.trained_from)

    async def test_a_retrain_replaces_the_provenance(self, db):
        db.add(LtxCharacter(name="pay", char_lora="pay_v1_e05", trigger="p@y",
                            trained_from=[{"id": "old", "name": "Old Set", "count": 13}]))
        await db.flush()
        j = _job(
            output_lora_path="s3://ltx-loras/character/pay_v2_e03.safetensors",
            config={"gender": "woman",
                    "dataset": {"id": "new", "name": "New Set", "count": 40}})
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.trained_from == [{"dataset_id": "new", "name": "New Set", "count": 40}]


@pytest.mark.asyncio
class TestTheClaimEndpoint:
    """GET /training/next driven over HTTP against a real row (#322).

    The rest of this file reads the source with inspect.getsource. That never builds the
    response, so it could not see the bug that took training down for two days: the route
    passed `identities` twice, once via the base dump and once as the resolved override, and
    every claim 500'd after the row had already been committed CLAIMED. Nothing was exercised
    end to end, so nothing caught it. These tests call the endpoint the way the poller does.
    """

    WORKER_ID = uuid.UUID("a1111111-1111-1111-1111-111111111111")

    async def _trainer(self, db):
        w = Worker(id=self.WORKER_ID, friendly_name="3090.zero", hostname="h",
                   ip_address="10.0.0.9", kind=WorkerKind.RENDER, kinds=["render", "trainer"],
                   status="online", last_heartbeat=datetime.now(timezone.utc),
                   gpu_stats={"gpu_name": "RTX 3090"})
        db.add(w)
        await db.flush()
        return w

    async def _claim(self, db):
        from httpx import ASGITransport, AsyncClient
        from app.auth import verify_api_key
        from app.database import get_db
        from app.main import app

        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[verify_api_key] = lambda: None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get(
                    "/training/next",
                    params={"worker_id": str(self.WORKER_ID), "worker_name": "3090.zero"},
                )
        finally:
            app.dependency_overrides.clear()

    @pytest.fixture(autouse=True)
    def _no_s3(self, monkeypatch):
        # Presigning needs credentials this laptop/CI does not have; the URL's content is not
        # what is under test, only that the response can be built at all and pairs 1:1.
        from app.routes import training as mod
        monkeypatch.setattr(mod.s3, "generate_presigned_url",
                            lambda uri, expires=21600: f"https://presigned/{uri.rsplit('/', 1)[-1]}")

    async def test_a_multi_identity_job_is_claimed_and_returns_its_groups(self, db):
        """The exact shape that 500'd: `identities` non-empty, so the route both dumped it from
        the base and passed the resolved override."""
        await self._trainer(db)
        db.add(_job(character="Karoline", trigger="k@roline",
                    identities=[{"character": "Me", "trigger": "d@vid", "gender": "woman",
                                 "caption": "a woman", "num_repeats": 10,
                                 "images": ["s3://wanly-images/g1.jpg"],
                                 "dataset": {"name": "Karoline set"}}]))
        await db.flush()

        resp = await self._claim(db)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body is not None
        assert len(body["download_urls"]) == len(body["dataset_images"])
        assert body["identities"][0]["download_urls"] == ["https://presigned/g1.jpg"]
        assert body["identities"][0]["dataset_name"] == "Karoline set"

    async def test_a_single_identity_job_claims_with_no_groups(self, db):
        """No extra groups must still produce a response — the collision happened even with
        identities absent, because the base dump carries the key regardless."""
        await self._trainer(db)
        db.add(_job())
        await db.flush()

        resp = await self._claim(db)

        assert resp.status_code == 200, resp.text
        assert resp.json()["identities"] is None


# ---------------------------------------------------------------------------------------
# Retry (api#342): same row, same version. Since #423 it trains EXACTLY the run's snapshot:
# datasets are living (#420), so re-reading one would retrain a different set than the run
# records. The purge that once killed snapshots no longer reaches trained files (#421).
# ---------------------------------------------------------------------------------------


def _combo_failed(**kw):
    """A failed joint run: group 0 from dataset A, an identity group from dataset B, both
    with provenance and a snapshot, plus the debris of a dead attempt."""
    from app.models import Dataset
    a = Dataset(id=uuid.uuid4(), name="David", images=_images(9), prefix="p-a")
    b = Dataset(id=uuid.uuid4(), name="Me", images=_images(10), prefix="p-b")
    job = _job(
        character="DavidMe", trigger="d@vid", version=1, status=TrainingStatus.FAILED,
        dataset_images=_images(9),
        config={"steps": 1200, "dataset": {"id": str(a.id), "name": "David", "count": 9}},
        identities=[{"character": "Me", "trigger": "m3", "gender": "woman",
                     "caption": "m3, woman", "images": _images(10), "num_repeats": None,
                     "dataset": {"id": str(b.id), "name": "Me", "count": 10}}],
        error_message="404 on a dataset image", worker_id=uuid.uuid4(), worker_name="3090",
        claimed_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        step=42, progress_log="died", loss_log=[[10, 0.9]],
        epochs=[{"label": "e01", "step": 80}], publish_requests=["final"],
        checkpoints=["s3://ltx-loras/character/davidme_v1_e01.safetensors"],
        output_lora_path="s3://ltx-loras/character/davidme_v1_final.safetensors",
    )
    return a, b, job


class TestARunCanBeRetried:
    async def test_only_a_failed_run_is_retryable(self, db):
        from fastapi import HTTPException
        from app.routes.training import retry_training_job
        for status in (TrainingStatus.COMPLETED, TrainingStatus.CANCELLED,
                       TrainingStatus.PENDING, TrainingStatus.RUNNING):
            job = _job(status=status)
            db.add(job)
            await db.commit()
            with pytest.raises(HTTPException) as e:
                await retry_training_job(job.id, _user=None, db=db)
            assert e.value.status_code == 400
            await db.delete(job)
            await db.commit()

    async def test_an_unknown_id_is_404(self, db):
        from fastapi import HTTPException
        from app.routes.training import retry_training_job
        with pytest.raises(HTTPException) as e:
            await retry_training_job(uuid.uuid4(), _user=None, db=db)
        assert e.value.status_code == 404

    async def test_a_failed_run_comes_back_pending_in_place(self, db):
        from app.routes.training import retry_training_job
        _, _, job = _combo_failed()
        db.add(job); await db.commit()
        job_id = job.id

        out = await retry_training_job(job_id, _user=None, db=db)

        assert out.id == job_id, "retry must reuse the row, not clone it"
        assert out.character == "DavidMe" and out.version == 1
        assert out.status == TrainingStatus.PENDING
        # every trace of the dead attempt is cleared...
        assert out.error_message is None and out.progress_log is None
        assert out.worker_id is None and out.worker_name is None
        assert out.claimed_at is None and out.completed_at is None
        assert out.step is None and out.loss_log is None and out.epochs is None
        assert out.checkpoints is None and out.output_lora_path is None
        assert out.publish_requests is None
        # ...the recipe and the identity are not.
        assert out.config["steps"] == 1200
        assert out.trigger == "d@vid"

    async def test_it_trains_the_snapshot_never_the_living_dataset(self, db):
        """#423: the sets changed after the run -- David gained a face, Me lost some -- and
        the retry still trains exactly what the run recorded, provenance and all."""
        from app.routes.training import retry_training_job
        a, b, job = _combo_failed()
        db.add(a); db.add(b); db.add(job); await db.commit()
        a.images = _images(12)
        b.images = [f"s3://wanly-images/2026-09-11/me{i}.jpg" for i in range(11)]
        await db.commit()

        out = await retry_training_job(job.id, _user=None, db=db)

        assert out.dataset_images == _images(9)
        assert out.identities[0]["images"] == _images(10)
        assert out.config["dataset"]["count"] == 9
        assert out.identities[0]["dataset"]["count"] == 10

    async def test_the_snapshot_trains_when_its_dataset_is_gone(self, db):
        """A deleted dataset is not a retry-blocker — the recorded images may still be fine.
        Keep them and keep the provenance as it was recorded."""
        from app.routes.training import retry_training_job
        a, b, job = _combo_failed()
        db.add(a); db.add(b); db.add(job); await db.commit()
        await db.delete(b); await db.commit()

        out = await retry_training_job(job.id, _user=None, db=db)

        assert out.identities[0]["images"] == _images(10), "the snapshot should stand"
        assert out.identities[0]["dataset"]["name"] == "Me"

    async def test_a_group_trained_from_an_ad_hoc_list_keeps_its_snapshot(self, db):
        """No dataset to consult, so the frozen URIs are all there is — exactly as created."""
        from app.routes.training import retry_training_job
        job = _job(character="adhoc", trigger="a", version=1, status=TrainingStatus.FAILED,
                   dataset_images=_images(9),
                   config={"steps": 1200, "dataset": {"id": None, "name": None, "count": 9}})
        db.add(job); await db.commit()

        out = await retry_training_job(job.id, _user=None, db=db)

        assert out.dataset_images == _images(9)
        assert out.config["dataset"] == {"id": None, "name": None, "count": 9}

    async def test_a_shrunken_dataset_does_not_matter(self, db):
        """The set is the subject's living set; the run trains what it recorded (#423)."""
        from app.routes.training import retry_training_job
        a, b, job = _combo_failed()
        db.add(a); db.add(b); db.add(job); await db.commit()
        a.images = _images(3)
        await db.commit()
        out = await retry_training_job(job.id, _user=None, db=db)
        assert out.status == TrainingStatus.PENDING and out.dataset_images == _images(9)

    async def test_a_live_twin_of_the_same_version_is_refused(self, db):
        """The partial unique index would reject the commit anyway; say it as a sentence."""
        from fastapi import HTTPException
        from app.routes.training import retry_training_job
        _, _, job = _combo_failed()
        db.add(job)
        db.add(_job(character="DavidMe", version=1, status=TrainingStatus.RUNNING))
        await db.commit()

        with pytest.raises(HTTPException) as e:
            await retry_training_job(job.id, _user=None, db=db)
        assert e.value.status_code == 409 and "already running" in e.value.detail

    async def test_the_human_note_survives_a_retry(self, db):
        """Notes are about the character, not one attempt; the attempt's own artifacts clear."""
        from app.routes.training import retry_training_job
        _, _, job = _combo_failed()
        job.notes = "rank 16 was too weak"
        db.add(job); await db.commit()

        out = await retry_training_job(job.id, _user=None, db=db)

        assert out.notes == "rank 16 was too weak"


# ---------------------------------------------------------------------------------------
# #352: the server derives the run. One class per guardrail, each breaking exactly one
# thing in an otherwise correct _world, so a failure names the rule that moved.
# ---------------------------------------------------------------------------------------


async def _preflight(db, **body):
    from app.training_plan import plan_training
    return (await plan_training(db, TrainingCreate(**body))).public()


SOLO = {"mode": "solo", "character": "David"}
PAIR = {"mode": "pair", "character": "DavidKelly-2026", "members": ["David", "Kelly-2026"]}


class TestTheWellFormedRunsPass:
    async def test_a_solo_run_is_clean(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO)
        assert out["ok"], out["problems"]
        assert [g["kind"] for g in out["groups"]] == ["identity", "regularization"]

    async def test_a_pair_run_is_clean(self, db):
        await _world(db)
        out = await _preflight(db, **PAIR)
        assert out["ok"], out["problems"]
        assert [(g["kind"], g["dataset_name"]) for g in out["groups"]] == [
            ("identity", "David"), ("identity", "Kelly-2026"),
            ("composition", "DavidKelly-2026"),
            ("regularization", "Reg-man"), ("regularization", "Reg-woman")]


class TestThePreflightOutput:
    """What the Train dialog renders. The sample captions are FINAL -- prefix included --
    because the preview exists to show what the trainer will actually write."""

    async def test_the_shape(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO, steps=2000)
        assert set(out) == {"ok", "problems", "warnings", "groups", "steps",
                            "samples_per_epoch", "passes_per_image", "base_checkpoint", "arch"}
        g = out["groups"][0]
        assert set(g) == {"kind", "character", "trigger", "gender", "dataset_id",
                          "dataset_name", "images", "num_repeats", "windows",
                          "sample_captions"}

    async def test_identity_captions_carry_the_registrys_trigger_and_gender(self, db):
        await _world(db)
        g = (await _preflight(db, **SOLO))["groups"][0]
        assert g["trigger"] == "d@vid" and g["gender"] == "man"
        assert g["sample_captions"][0] == "d@vid, man, medium shot, standing, look 0"
        assert len(g["sample_captions"]) == 5

    async def test_composition_captions_name_both_people_in_member_order(self, db):
        await _world(db)
        comp = (await _preflight(db, **PAIR))["groups"][2]
        assert comp["sample_captions"][0] == (
            "d@vid, man and k3lly2026, woman, medium shot, standing, look 0")
        assert comp["trigger"] is None

    async def test_regularization_captions_carry_only_the_class_word(self, db):
        await _world(db)
        reg = (await _preflight(db, **SOLO))["groups"][1]
        assert reg["sample_captions"][0] == "man, medium shot, standing, look 0"
        assert reg["trigger"] is None and reg["character"] is None

    async def test_regularization_repeats_match_the_character_samples(self, db):
        """reg_ratio 1.0: as many generic "man" samples per epoch as "d@vid, man" ones."""
        await _world(db)
        out = await _preflight(db, **SOLO)
        ident, reg = out["groups"]
        assert ident["images"] * ident["num_repeats"] == 100
        assert reg["images"] == 30 and reg["num_repeats"] == 3  # 90 ~= 100
        assert out["samples_per_epoch"] == 190

    async def test_a_pair_splits_regularization_between_the_genders(self, db):
        await _world(db)
        out = await _preflight(db, **PAIR)
        character = sum(g["images"] * g["num_repeats"] for g in out["groups"]
                        if g["kind"] != "regularization")
        assert character == (10 + 12 + 9) * 10
        man, woman = out["groups"][3:]
        assert man["num_repeats"] == round(character / 2 / 30)
        assert woman["num_repeats"] == round(character / 2 / 40)

    async def test_passes_and_base_checkpoint(self, db):
        from app.ltx_stack import LTX_STACK
        await _world(db)
        out = await _preflight(db, **SOLO, steps=1900)
        assert out["passes_per_image"] == 100.0  # 1900 / 190 epochs x 10 repeats
        assert out["base_checkpoint"] == LTX_STACK["checkpoint"] == "10Eros_v1.5_bf16"

    async def test_the_preflight_writes_nothing(self, db):
        from sqlalchemy import func, select
        await _world(db)
        await _preflight(db, **SOLO)
        assert (await db.execute(select(func.count(TrainingJob.id)))).scalar_one() == 0


class TestABlankCaptionIsTheBarePhrase:
    """#365: blank = "<trigger>, <gender>" (the standard); only props are typed."""

    async def test_a_blank_caption_trains_as_the_bare_phrase(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        caps = dict(w["david"].captions)
        blank = w["david"].images[3]
        caps.pop(blank)
        w["david"].captions = caps
        await db.flush()
        out = await _preflight(db, **SOLO)
        assert out["ok"], out["problems"]
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        i = job.dataset_images.index(blank)
        assert job.config["captions"][i] == "d@vid, man"

    async def test_a_typed_caption_is_still_appended(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        u = w["david"].images[0]
        w["david"].captions = {u: "wearing glasses"}
        await db.flush()
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        assert job.config["captions"][job.dataset_images.index(u)] == "d@vid, man, wearing glasses"
        assert set(job.config["captions"]) == {"d@vid, man, wearing glasses", "d@vid, man"}

    async def test_a_blank_regularization_caption_is_the_class_word(self, db):
        w = await _world(db)
        w["reg_man"].captions = {}
        await db.flush()
        out = await _preflight(db, **SOLO)
        assert out["ok"], out["problems"]
        reg = [g for g in out["groups"] if g["kind"] == "regularization"][0]
        assert set(reg["sample_captions"]) == {"man"}


class TestGuardTheDatasetBelongsToTheCharacter:
    async def test_kellys_set_cannot_train_david(self, db):
        """The plan's own verification case."""
        w = await _world(db)
        out = await _preflight(db, **SOLO, datasets={"David": str(w["kelly"].id)})
        assert "dataset_wrong_owner" in _codes(out)

    async def test_a_regularization_pool_is_not_a_character_set(self, db):
        w = await _world(db)
        out = await _preflight(db, **SOLO, datasets={"David": str(w["reg_man"].id)})
        assert "dataset_wrong_owner" in _codes(out)

    async def test_no_owned_set_blocks(self, db):
        w = await _world(db)
        w["david"].character = None
        w["david"].kind = None
        await db.flush()
        assert "dataset_missing" in _codes(await _preflight(db, **SOLO))

    async def test_two_owned_sets_must_be_chosen_between(self, db):
        from app.models import Dataset
        w = await _world(db)
        db.add(Dataset(name="David 2", prefix="x", kind="character", character="David",
                       images=_set("d2", 9)))
        await db.flush()
        assert "dataset_ambiguous" in _codes(await _preflight(db, **SOLO))
        out = await _preflight(db, **SOLO, datasets={"David": str(w["david"].id)})
        assert out["ok"], out["problems"]

    async def test_a_choice_for_someone_not_in_the_run_blocks(self, db):
        w = await _world(db)
        out = await _preflight(db, **SOLO, datasets={"Kelly-2026": str(w["kelly"].id)})
        assert "dataset_for_non_member" in _codes(out)


class TestGuardTheSetIsProvablyOnePerson:
    async def test_no_anchor_blocks(self, db):
        w = await _world(db)
        w["david"].anchor_uri = None
        await db.flush()
        assert "anchor_missing" in _codes(await _preflight(db, **SOLO))

    async def test_unscored_images_block(self, db):
        w = await _world(db)
        w["david"].scores = {}
        await db.flush()
        assert "scores_missing" in _codes(await _preflight(db, **SOLO))

    async def test_a_face_below_the_floor_blocks(self, db):
        from app.config import settings
        w = await _world(db)
        w["david"].scores = {**w["david"].scores, w["david"].images[5]: settings.face_cos_floor - 0.01}
        await db.flush()
        assert "score_below_floor" in _codes(await _preflight(db, **SOLO))

    async def test_no_face_at_all_blocks(self, db):
        w = await _world(db)
        w["david"].scores = {**w["david"].scores, w["david"].images[5]: None}
        await db.flush()
        assert "score_below_floor" in _codes(await _preflight(db, **SOLO))

    async def test_verified_real_photos_can_train_below_the_floor(self, db):
        # console#575: profiles and face-filling selfies of the REAL person score low or read as
        # "no face". The acknowledgement turns the block into a warning -- never silently.
        from app.config import settings
        w = await _world(db)
        imgs = w["david"].images
        w["david"].scores = {**w["david"].scores, imgs[5]: settings.face_cos_floor - 0.1,
                             imgs[6]: None}
        await db.flush()
        out = await _preflight(db, **SOLO, allow_low_scores=True)
        assert "score_below_floor" not in _codes(out)
        assert "score_below_floor_allowed" in {w["code"] for w in out["warnings"]}

    async def test_the_acknowledgement_does_not_excuse_missing_scores(self, db):
        # Only a LOW score is waivable; an unscored set has no evidence at all.
        w = await _world(db)
        w["david"].scores = {}
        await db.flush()
        assert "scores_missing" in _codes(await _preflight(db, **SOLO, allow_low_scores=True))


class TestGuardPairMembers:
    async def test_one_member_is_not_a_pair(self, db):
        await _world(db)
        out = await _preflight(db, mode="pair", character="DavidKelly-2026", members=["David"])
        assert "pair_members" in _codes(out)

    async def test_the_same_member_twice_is_not_a_pair(self, db):
        await _world(db)
        out = await _preflight(db, mode="pair", character="DavidKelly-2026",
                               members=["David", "David"])
        assert "pair_members" in _codes(out)

    async def test_an_unregistered_member_blocks(self, db):
        await _world(db)
        out = await _preflight(db, mode="pair", character="DavidX",
                               members=["David", "Nobody"])
        assert "member_unknown" in _codes(out)

    async def test_a_member_without_a_gender_blocks(self, db):
        w = await _world(db)
        w["kelly_c"].gender = None
        await db.flush()
        assert "trigger_missing" in _codes(await _preflight(db, **PAIR))

    async def test_a_registered_pair_supplies_its_own_members(self, db):
        await _world(db)
        db.add(LtxCharacter(name="DavidKelly-2026", kind="pair", char_lora="none",
                            members=["David", "Kelly-2026"], trigger="x"))
        await db.flush()
        out = await _preflight(db, mode="pair", character="DavidKelly-2026")
        assert out["ok"], out["problems"]
        out = await _preflight(db, mode="pair", character="DavidKelly-2026",
                               members=["Kelly-2026", "David"])
        assert "members_mismatch" in _codes(out)

    async def test_members_with_one_trigger_cannot_train_together(self, db):
        w = await _world(db)
        w["kelly_c"].trigger = "d@vid"
        await db.flush()
        assert "duplicate_trigger" in _codes(await _preflight(db, **PAIR))


class TestGuardTheCompositionSet:
    async def test_a_pair_without_one_blocks(self, db):
        w = await _world(db)
        await db.delete(w["comp"])
        await db.flush()
        assert "composition_missing" in _codes(await _preflight(db, **PAIR))

    async def test_it_can_be_waived_knowingly_with_a_warning(self, db):
        w = await _world(db)
        await db.delete(w["comp"])
        await db.flush()
        out = await _preflight(db, **PAIR, allow_no_composition=True)
        assert out["ok"], out["problems"]
        assert "no_composition" in {x["code"] for x in out["warnings"]}
        assert "composition" not in [g["kind"] for g in out["groups"]]

    async def test_another_pairs_composition_set_blocks(self, db):
        w = await _world(db)
        w["comp"].character = "DavidKelly-2000"
        await db.flush()
        out = await _preflight(db, **PAIR, composition_dataset_id=str(w["comp"].id))
        assert "dataset_wrong_owner" in _codes(out)

    async def test_a_solo_run_has_none(self, db):
        w = await _world(db)
        out = await _preflight(db, **SOLO, composition_dataset_id=str(w["comp"].id))
        assert "composition_in_solo" in _codes(out)


class TestGuardSoloAndPairNamesDoNotCollide:
    async def test_a_pair_cannot_be_named_after_a_person(self, db):
        """That is the bug: a joint run publishing over David's own row."""
        await _world(db)
        out = await _preflight(db, mode="pair", character="David",
                               members=["David", "Kelly-2026"])
        assert {"pair_name_is_solo", "pair_is_member"} <= _codes(out)

    async def test_a_pair_row_cannot_be_trained_solo(self, db):
        await _world(db)
        db.add(LtxCharacter(name="DavidKelly-2026", kind="pair", char_lora="none",
                            members=["David", "Kelly-2026"], trigger="x"))
        await db.flush()
        out = await _preflight(db, mode="solo", character="DavidKelly-2026")
        assert "solo_on_pair" in _codes(out)

    async def test_an_unregistered_solo_character_blocks(self, db):
        await _world(db)
        assert "unknown_character" in _codes(await _preflight(db, mode="solo", character="Zed"))


class TestGuardRegularization:
    async def test_no_pool_for_the_gender_blocks(self, db):
        w = await _world(db)
        await db.delete(w["reg_man"])
        await db.flush()
        out = await _preflight(db, **SOLO)
        assert "regularization_missing" in _codes(out)
        # Kelly is a woman; her pool is untouched, so a Kelly run is still fine.
        assert (await _preflight(db, mode="solo", character="Kelly-2026"))["ok"]

    async def test_the_largest_pool_is_used(self, db):
        from app.models import Dataset
        await _world(db)
        db.add(Dataset(name="Reg-man-big", prefix="x", kind="regularization",
                       reg_class="man", images=_set("big", 60),
                       captions={u: "a" for u in _set("big", 60)}))
        await db.flush()
        out = await _preflight(db, **SOLO)
        assert out["groups"][1]["dataset_name"] == "Reg-man-big"


class TestGuardTheLongStandingChecks:
    async def test_too_few_images(self, db):
        w = await _world(db)
        w["david"].images = w["david"].images[:5]
        await db.flush()
        assert "too_few_images" in _codes(await _preflight(db, **SOLO))

    async def test_one_set_in_two_groups(self, db):
        w = await _world(db)
        w["kelly"].character = "David"
        await db.flush()
        out = await _preflight(db, **PAIR, datasets={"David": str(w["david"].id),
                                                     "Kelly-2026": str(w["kelly"].id)})
        # Kelly's set is David's now, so it is refused for her before it can be reused.
        assert "dataset_wrong_owner" in _codes(out)

    async def test_a_live_run_of_the_same_version(self, db):
        await _world(db)
        db.add(_job(character="David", version=1, status=TrainingStatus.RUNNING))
        await db.flush()
        assert "version_taken" in _codes(await _preflight(db, **SOLO))


class TestTheWarnings:
    async def test_too_many_passes_warns_but_does_not_block(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO, steps=3000)
        assert out["ok"]
        assert "passes_high" in {x["code"] for x in out["warnings"]}

    async def test_a_modest_run_does_not_warn(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO, steps=150)
        assert out["warnings"] == []

    async def test_small_faces_warn_with_the_count(self, db):
        """#432: a warning, never a block -- Joana v3 trained, just slowly (#431)."""
        w = await _world(db)
        imgs = w["david"].images
        w["david"].faces = {**w["david"].faces,
                            imgs[1]: {"width": 1080, "height": 1440, "face_px": 180.0},
                            imgs[2]: {"width": 1080, "height": 1440, "face_px": 249.9}}
        await db.flush()
        out = await _preflight(db, **SOLO, steps=150)
        assert out["ok"]
        msg = {x["code"]: x["message"] for x in out["warnings"]}
        assert "2 of 10 images show the face under 250 px" in msg["small_faces"]

    async def test_a_photo_whose_crop_is_in_the_set_is_fixed_not_small(self, db):
        """Live on "Me" after a Fix: 12 originals with their crops in the set still warned
        "12 of 45" -- the photo never changes. Its crop removed, it counts again."""
        w = await _world(db)
        imgs = w["david"].images
        w["david"].faces = {**w["david"].faces,
                            imgs[1]: {"width": 1080, "height": 1440, "face_px": 180.0,
                                      "crop_uri": imgs[3]},
                            imgs[2]: {"width": 1080, "height": 1440, "face_px": 120.0,
                                      "crop_uri": "s3://wanly-images/gone.jpg"}}
        await db.flush()
        out = await _preflight(db, **SOLO, steps=150)
        msg = {x["code"]: x["message"] for x in out["warnings"]}
        assert "1 of 10 images show the face under 250 px" in msg["small_faces"]

    async def test_no_face_is_not_a_small_face(self, db):
        """No face is the anchor scores' problem, and already caught there."""
        w = await _world(db)
        w["david"].faces = {**w["david"].faces,
                            w["david"].images[1]: {"width": 900, "height": 900, "face_px": None}}
        await db.flush()
        out = await _preflight(db, **SOLO, steps=150)
        assert "small_faces" not in {x["code"] for x in out["warnings"]}

    async def test_unmeasured_images_are_said_not_passed_over(self, db):
        """Silence about them would read as "no small faces"."""
        w = await _world(db)
        w["david"].faces = {}
        await db.flush()
        out = await _preflight(db, **SOLO, steps=150)
        assert out["ok"]
        msg = {x["code"]: x["message"] for x in out["warnings"]}
        assert "not measured for 10 of 10" in msg["faces_unmeasured"]


class TestTheV1Recipe:
    """caption_mode=trigger_only + regularization=False: the recipe before #352, which beat
    per-image captions with 1:1 regularization on Kelly-2000 (v1 vs v2, 2026-09-27)."""

    async def test_trigger_only_captions_every_image_with_the_bare_phrase(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO, caption_mode="trigger_only")
        g0 = out["groups"][0]
        assert g0["kind"] == "identity"
        assert set(g0["sample_captions"]) == {g0["sample_captions"][0]}
        assert "," in g0["sample_captions"][0] and g0["sample_captions"][0].count(",") == 1

    async def test_trigger_only_needs_no_stored_captions_on_the_character_set(self, db):
        w = await _world(db)
        w["david"].captions = {}
        await db.flush()
        assert (await _preflight(db, **SOLO))["ok"]
        assert (await _preflight(db, **SOLO, caption_mode="trigger_only"))["ok"]

    async def test_a_blank_regularization_pool_trains_under_the_class_word(self, db):
        w = await _world(db)
        w["reg_man"].captions = {}
        await db.flush()
        out = await _preflight(db, **SOLO, caption_mode="trigger_only")
        assert out["ok"], out["problems"]

    async def test_no_regularization_adds_no_pool_and_warns(self, db):
        w = await _world(db)
        await db.delete(w["reg_man"])
        await db.flush()
        out = await _preflight(db, **SOLO, regularization=False)
        assert out["ok"], out["problems"]
        assert [g["kind"] for g in out["groups"]] == ["identity"]
        assert "no_regularization" in {x["code"] for x in out["warnings"]}

    async def test_the_v1_recipe_end_to_end_snapshots_its_choices(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        w["david"].captions = {}
        await db.flush()
        job = await create_training_job(
            TrainingCreate(**SOLO, caption_mode="trigger_only", regularization=False),
            user=_U(), db=db)
        assert job.config["captions"] == ["d@vid, man"] * len(job.dataset_images)
        assert job.config["caption_mode"] == "trigger_only"
        assert job.config["reg_ratio"] == 0
        assert job.identities is None

    async def test_a_v1_length_run_does_not_warn_about_passes(self, db):
        """~30 passes is the proven recipe, not memorising."""
        await _world(db)
        out = await _preflight(db, **SOLO, regularization=False, steps=300)
        assert out["passes_per_image"] == 30
        assert "passes_high" not in {x["code"] for x in out["warnings"]}


class TestTheBaseAndSeedAreTheRequests:
    """E0/E1 of the identity-drift plan: v1 retrained on dev with another seed, and v1's recipe on
    10Eros. Neither is expressible if the base and the seed are fixed."""

    async def test_default_base_is_the_render_stack_and_default_seed_is_42(self, db):
        from app.routes.training import create_training_job
        await _world(db)
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        assert job.config["base_checkpoint"] == "10Eros_v1.5_bf16"
        assert job.config["seed"] == 42

    async def test_a_named_base_and_seed_are_snapshotted(self, db):
        from app.routes.training import create_training_job
        await _world(db)
        job = await create_training_job(
            TrainingCreate(**SOLO, base_checkpoint="ltx-2.3-22b-dev.safetensors", seed=7),
            user=_U(), db=db)
        assert job.config["base_checkpoint"] == "ltx-2.3-22b-dev"
        assert job.config["seed"] == 7

    async def test_a_base_other_than_the_render_stack_warns(self, db):
        await _world(db)
        out = await _preflight(db, **SOLO, base_checkpoint="ltx-2.3-22b-dev")
        assert out["ok"]
        assert out["base_checkpoint"] == "ltx-2.3-22b-dev"
        assert "base_differs" in {w["code"] for w in out["warnings"]}

    async def test_a_path_is_not_a_checkpoint_name(self):
        import pytest
        from pydantic import ValidationError
        for bad in ("../etc/passwd", "a/b", ".hidden"):
            with pytest.raises(ValidationError):
                TrainingCreate(**SOLO, base_checkpoint=bad)


class TestCreateRefusesWhatThePreflightRefuses:
    async def test_a_problem_is_a_422_with_the_list(self, db):
        from fastapi import HTTPException
        from app.routes.training import create_training_job
        w = await _world(db)
        w["david"].images = w["david"].images[:5]
        await db.flush()
        with pytest.raises(HTTPException) as e:
            await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        assert e.value.status_code == 422
        assert "too_few_images" in {p["code"] for p in e.value.detail["problems"]}

    async def test_over_http_too(self, db):
        """The console reads detail.problems off the wire, so check the wire."""
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user
        from app.database import get_db
        from app.main import app
        w = await _world(db)
        w["david"].images = w["david"].images[:5]
        await db.flush()
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _U()
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                pre = await c.post("/training/preflight", json=SOLO)
                made = await c.post("/training", json=SOLO)
        finally:
            app.dependency_overrides.clear()
        assert pre.status_code == 200 and pre.json()["ok"] is False
        assert made.status_code == 422
        assert made.json()["detail"]["problems"] == pre.json()["problems"]


class TestTheCaptionsAreSnapshotted:
    """A run trains on the words that were previewed. Editing a caption, or re-registering
    the owner, after pressing Train must not change what a queued job does."""

    async def test_group_zero_and_every_group_carry_their_captions(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(TrainingCreate(**PAIR), user=_U(), db=db)
        assert job.trigger == "d@vid"
        assert job.dataset_images == w["david"].images
        assert job.config["captions"][0] == "d@vid, man, medium shot, standing, look 0"
        assert len(job.config["captions"]) == len(job.dataset_images)
        assert job.config["caption"] == job.config["captions"][0]
        assert job.config["mode"] == "pair"
        assert job.config["members"] == ["David", "Kelly-2026"]
        assert job.config["base_checkpoint"] == "10Eros_v1.5_bf16"
        assert job.config["num_repeats"] == 10
        for g in job.identities:
            assert len(g["captions"]) == len(g["images"])
            assert g["caption"] == g["captions"][0]
        assert [g["kind"] for g in job.identities] == [
            "identity", "composition", "regularization", "regularization"]
        assert [g["trigger"] for g in job.identities] == ["k3lly2026", None, None, None]

    async def test_editing_a_caption_afterwards_changes_nothing(self, db):
        from app.routes.training import create_training_job
        w = await _world(db)
        job = await create_training_job(TrainingCreate(**SOLO), user=_U(), db=db)
        before = list(job.config["captions"])
        w["david"].captions = {u: "EDITED" for u in w["david"].images}
        w["david_c"].trigger = "somebody-else"
        await db.commit()
        await db.refresh(job)
        assert job.config["captions"] == before


class TestTheClaimContract:
    """What wanly-gpu-docker's trainer consumes (#146 on that side): top-level `captions`
    for group 0 parallel to download_urls, config.base_checkpoint as a bare name, and per
    group `kind`, `captions`, `num_repeats`."""

    async def _claim_one(self, db):
        return await TestTheClaimEndpoint()._claim(db)

    @pytest.fixture(autouse=True)
    def _no_s3(self, monkeypatch):
        from app.routes import training as mod
        monkeypatch.setattr(mod.s3, "generate_presigned_url",
                            lambda uri, expires=21600: f"https://presigned/{uri.rsplit('/', 1)[-1]}")

    async def test_a_pair_claim(self, db):
        from app.routes.training import create_training_job
        await _world(db)
        await TestTheClaimEndpoint()._trainer(db)
        await create_training_job(TrainingCreate(**PAIR), user=_U(), db=db)
        resp = await self._claim_one(db)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["captions"]) == len(body["download_urls"]) == 10
        assert body["captions"][0].startswith("d@vid, man, ")
        assert body["config"]["base_checkpoint"] == "10Eros_v1.5_bf16"
        assert "/" not in body["config"]["base_checkpoint"]
        kinds = [g["kind"] for g in body["identities"]]
        assert kinds == ["identity", "composition", "regularization", "regularization"]
        for g in body["identities"]:
            assert len(g["captions"]) == len(g["download_urls"])
            assert all(c.strip() for c in g["captions"])
            assert isinstance(g["num_repeats"], int) and g["num_repeats"] >= 1
            assert g["caption"] == g["captions"][0]
        reg = body["identities"][2]
        assert reg["trigger"] is None and reg["gender"] == "man"
        assert reg["captions"][0].startswith("man, ")

    async def test_a_legacy_job_claims_with_no_captions(self, db):
        """A pre-#352 row sends null, never [] -- the trainer reads [] as a list to match."""
        await TestTheClaimEndpoint()._trainer(db)
        db.add(_job(identities=[{"character": "Me", "trigger": "d@vid", "gender": "man",
                                 "caption": "d@vid, man", "images": ["s3://b/x.jpg"]}]))
        await db.flush()
        body = (await self._claim_one(db)).json()
        assert body["captions"] is None
        assert body["identities"][0]["captions"] is None
        assert body["identities"][0]["kind"] == "identity"


class TestAPairPublishesToItsOwnRow:
    async def _run(self, db, body):
        from app.routes.training import create_training_job
        job = await create_training_job(TrainingCreate(**body), user=_U(), db=db)
        job.output_lora_path = "s3://ltx-loras/character/dk_v1_final.safetensors"
        await _publish_character(db, job)
        await db.flush()
        return job

    async def _row(self, db, name):
        from sqlalchemy import select
        return (await db.execute(select(LtxCharacter).where(LtxCharacter.name == name))
                ).scalar_one_or_none()

    async def test_the_pair_row_is_created_with_the_joined_phrase(self, db):
        await _world(db)
        await self._run(db, PAIR)
        row = await self._row(db, "DavidKelly-2026")
        assert row.kind == "pair"
        assert row.members == ["David", "Kelly-2026"]
        assert row.trigger == "d@vid, man and k3lly2026, woman"
        assert row.gender is None
        assert row.char_lora == "dk_v1_final"
        assert row.base_checkpoint == "10Eros_v1.5_bf16"
        assert [t["kind"] for t in row.trained_from] == [
            "identity", "identity", "composition", "regularization", "regularization"]

    async def test_no_member_row_is_touched(self, db):
        w = await _world(db)
        w["david_c"].char_lora = "david_v1_final"
        w["david_c"].strength_stage_2 = 1.3
        await db.flush()
        await self._run(db, PAIR)
        david = await self._row(db, "David")
        assert david.char_lora == "david_v1_final"
        assert david.trigger == "d@vid" and david.gender == "man"
        assert david.strength_stage_2 == 1.3
        assert (await self._row(db, "Kelly-2026")).char_lora == "none"

    async def test_an_existing_pair_keeps_its_strengths(self, db):
        await _world(db)
        db.add(LtxCharacter(name="DavidKelly-2026", kind="pair", char_lora="old",
                            members=["David", "Kelly-2026"], trigger="old phrase",
                            strength_stage_1=0.7, strength_stage_2=1.1))
        await db.flush()
        await self._run(db, PAIR)
        row = await self._row(db, "DavidKelly-2026")
        assert row.char_lora == "dk_v1_final"
        assert (row.strength_stage_1, row.strength_stage_2) == (0.7, 1.1)
        assert row.trigger == "d@vid, man and k3lly2026, woman"

    async def test_a_solo_publish_leaves_trigger_and_gender_alone(self, db):
        w = await _world(db)
        await self._run(db, SOLO)
        david = await self._row(db, "David")
        assert david.char_lora == "dk_v1_final"
        assert (david.trigger, david.gender, david.kind) == ("d@vid", "man", "solo")
        assert david.base_checkpoint == "10Eros_v1.5_bf16"
        assert david.image_uri == w["david"].anchor_uri


class TestARegisteredRunRetriesFromItsSnapshot:
    async def _failed(self, db, body=SOLO):
        from app.routes.training import create_training_job
        job = await create_training_job(TrainingCreate(**body), user=_U(), db=db)
        job.status = TrainingStatus.FAILED
        job.error_message = "404"
        await db.commit()
        return job

    async def test_live_captions_are_never_read(self, db):
        from app.routes.training import retry_training_job
        w = await _world(db)
        job = await self._failed(db)
        before = list(job.config["captions"])
        w["david"].captions = {u: "EDITED" for u in w["david"].images}
        await db.commit()
        out = await retry_training_job(job.id, _user=None, db=db)
        assert out.status == TrainingStatus.PENDING
        assert out.config["captions"] == before
        assert all("EDITED" not in c for g in out.identities for c in g["captions"])

    async def test_removed_images_still_train(self, db):
        """#423: an image removed from the living set since is still in the run's record."""
        from app.routes.training import retry_training_job
        w = await _world(db)
        job = await self._failed(db)
        images = list(job.dataset_images)
        dead = w["david"].images[2]
        w["david"].images = [u for u in w["david"].images if u != dead]
        await db.commit()
        out = await retry_training_job(job.id, _user=None, db=db)
        assert out.dataset_images == images and dead in out.dataset_images
        assert len(out.config["captions"]) == len(images)

    async def test_an_image_added_since_is_not_trained(self, db):
        from app.routes.training import retry_training_job
        w = await _world(db)
        job = await self._failed(db)
        w["david"].images = w["david"].images + ["s3://wanly-images/datasets/new.jpg"]
        await db.commit()
        out = await retry_training_job(job.id, _user=None, db=db)
        assert "s3://wanly-images/datasets/new.jpg" not in out.dataset_images

    async def test_the_run_record_matches_its_snapshot(self, db):
        """#422: creation writes one training_run_datasets row per group, and a retry
        leaves them true."""
        from sqlalchemy import select
        from app.models import TrainingRunDataset
        from app.routes.training import retry_training_job
        w = await _world(db)
        job = await self._failed(db)
        rows = (await db.execute(select(TrainingRunDataset).where(
            TrainingRunDataset.training_job_id == job.id)
            .order_by(TrainingRunDataset.group_index))).scalars().all()
        assert len(rows) == 1 + len(job.identities or [])
        assert rows[0].dataset_id == w["david"].id and rows[0].images == job.dataset_images
        assert rows[0].captions == job.config["captions"]
        out = await retry_training_job(job.id, _user=None, db=db)
        assert rows[0].images == out.dataset_images


SDXL = {**SOLO, "arch": "sdxl", "steps": 960}


class TestSDXLStartImageLoras:
    """#398: SDXL character LoRAs for the START IMAGES, trained by the same queue with the
    hand-made "aio" recipe. Never an LTX character: the engine cannot load one."""

    def test_absent_arch_is_ltx(self):
        assert TrainingCreate(**SOLO).arch == "ltx"

    async def test_the_plan_is_the_aio_shape(self, db):
        await _world(db)
        out = await _preflight(db, **SDXL)
        assert out["ok"], out["problems"]
        assert out["arch"] == "sdxl"
        assert out["base_checkpoint"] == "BigaspV2Lustify"
        # One group, 8 repeats, no regularization -- aio had none.
        assert [(g["kind"], g["num_repeats"]) for g in out["groups"]] == [("identity", 8)]
        # The trigger alone; the trainer's WD14 tags follow it.
        assert set(out["groups"][0]["sample_captions"]) == {"d@vid"}
        assert "sdxl_wd14" in {w["code"] for w in out["warnings"]}

    async def test_aios_passes_are_not_warned_about(self, db):
        """12 epochs x 8 repeats = 96 passes is the recipe, not a mistake."""
        await _world(db)
        out = await _preflight(db, **{**SDXL, "steps": 10 * 8 * 12})
        assert out["passes_per_image"] == 96
        assert "passes_high" not in {w["code"] for w in out["warnings"]}

    async def test_a_pair_trains_each_trigger_beside_its_class_tag(self, db):
        """#407: the LTX pair shape at aio's repeats. Every caption is the WD14 prefix the
        trainer tags after: each trigger bound to its booru class tag, both in the composition
        set. No regularization -- aio had none."""
        await _world(db)
        out = await _preflight(db, **{**PAIR, "arch": "sdxl", "steps": 960})
        assert out["ok"], out["problems"]
        assert [(g["kind"], g["character"], g["num_repeats"]) for g in out["groups"]] == [
            ("identity", "David", 8), ("identity", "Kelly-2026", 8),
            ("composition", "DavidKelly-2026", 8)]
        assert [set(g["sample_captions"]) for g in out["groups"]] == [
            {"d@vid, 1boy"}, {"k3lly2026, 1girl"}, {"d@vid, k3lly2026, 1boy, 1girl"}]

    async def test_an_sdxl_pair_still_needs_its_composition_set(self, db):
        w = await _world(db)
        w["comp"].kind = None
        await db.flush()
        assert "composition_missing" in _codes(
            await _preflight(db, **{**PAIR, "arch": "sdxl"}))

    async def test_an_ltx_pair_is_unchanged(self, db):
        """The SDXL prefix must not leak into LTX: the composition set keeps its sentence
        prefix and the long-standing 10 repeats."""
        await _world(db)
        out = await _preflight(db, **PAIR)
        comp = [g for g in out["groups"] if g["kind"] == "composition"][0]
        assert comp["num_repeats"] == 10
        assert all(c.startswith("d@vid, man and k3lly2026, woman")
                   for c in comp["sample_captions"])

    def test_the_class_tags(self):
        from app.training_plan import sdxl_pair_prefix
        assert sdxl_pair_prefix(["a", "b"], ["woman", "woman"]) == "a, b, 2girls"
        assert sdxl_pair_prefix(["a", "b"], ["man", "man"]) == "a, b, 2boys"
        assert sdxl_pair_prefix(["a", "b"], ["woman", "person"]) == "a, b, 1girl"
        assert sdxl_pair_prefix(["a"], ["person"]) == "a"

    async def test_the_set_must_still_be_one_person(self, db):
        """The anchor/score gates are about the dataset, not the model -- they still apply."""
        w = await _world(db)
        w["david"].anchor_uri = None
        await db.flush()
        assert "anchor_missing" in _codes(await _preflight(db, **SDXL))

    async def test_create_snapshots_the_sdxl_recipe(self, db):
        from app.routes.training import create_training_job
        await _world(db)
        job = await create_training_job(TrainingCreate(**SDXL), user=_U(), db=db)
        c = job.config
        assert c["arch"] == "sdxl"
        assert (c["network_dim"], c["network_alpha"], c["learning_rate"],
                c["text_encoder_lr"], c["num_repeats"]) == (128, 64, 8e-5, 2e-5, 8)
        assert "lora_target_preset" not in c
        assert c["base_checkpoint"] == "BigaspV2Lustify"
        assert job.identities is None

    def test_the_files_cannot_collide_with_an_ltx_lora(self):
        """Workers flatten the prefix and skip BOTH files of a name under two prefixes, so a
        shared basename would knock the LTX character out of every worker."""
        from app.routes.training import _artifact_key, _belongs_to
        ltx = _job(character="David", version=1, config={"lora_name": "David"})
        sdxl = _job(character="David", version=1, config={"lora_name": "David", "arch": "sdxl"})
        assert _artifact_key(sdxl, 3, False) == "character/sdxl/David_sdxl_v1_e03.safetensors"
        assert _artifact_key(sdxl, None, True) == "character/sdxl/David_sdxl_v1_final.safetensors"
        assert _artifact_key(sdxl, None, True).rsplit("/", 1)[1] != \
            _artifact_key(ltx, None, True).rsplit("/", 1)[1]
        assert _belongs_to(sdxl, "s3://ltx-loras/character/sdxl/David_sdxl_v1_e03.safetensors")
        assert not _belongs_to(sdxl, "s3://ltx-loras/character/David_v1_e03.safetensors")

    async def test_completing_never_touches_a_character_row(self, db):
        from sqlalchemy import select
        db.add(LtxCharacter(name="David", trigger="d@vid", gender="man", char_lora="david_v4"))
        await db.flush()
        j = _job(character="David", config={"arch": "sdxl", "mode": "solo"},
                 output_lora_path="s3://ltx-loras/character/sdxl/David_sdxl_v1_final.safetensors")
        await _publish_character(db, j)
        await db.flush()
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "David"))).scalar_one()
        assert row.char_lora == "david_v4"


class TestVersionsArePerArch:
    """#402: KimJule's SDXL v1 training must not block her LTX v1 -- different models, kept
    apart everywhere downstream. Same arch still collides, as it always has."""

    async def _live(self, db, arch):
        cfg = {"arch": arch} if arch else {}
        j = _job(character="David", version=1, status=TrainingStatus.RUNNING, config=cfg)
        db.add(j)
        await db.flush()
        return j

    async def test_a_live_sdxl_v1_leaves_ltx_v1_free(self, db):
        await _world(db)
        await self._live(db, "sdxl")
        assert "version_taken" not in _codes(await _preflight(db, **{**SOLO, "version": 1}))

    async def test_a_live_ltx_v1_leaves_sdxl_v1_free(self, db):
        await _world(db)
        await self._live(db, None)          # pre-SDXL row: no arch = LTX
        assert "version_taken" not in _codes(
            await _preflight(db, **{**SOLO, "arch": "sdxl", "version": 1}))

    async def test_same_arch_still_collides(self, db):
        await _world(db)
        await self._live(db, "sdxl")
        out = await _preflight(db, **{**SOLO, "arch": "sdxl", "version": 1})
        assert "version_taken" in _codes(out)
        assert any("SDXL v1" in p["message"] for p in out["problems"])

    async def test_the_database_allows_one_live_run_per_arch(self, db):
        """The unique index, not just the preflight: both arches live at v1 is fine, a second
        live LTX v1 is not -- including when one row predates config.arch."""
        from sqlalchemy.exc import IntegrityError
        await self._live(db, "sdxl")
        await self._live(db, None)
        with pytest.raises(IntegrityError):
            await self._live(db, "ltx")
