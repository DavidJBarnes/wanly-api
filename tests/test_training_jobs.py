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


class TestTheRequest:
    def test_a_dataset_of_s3_uris_is_required(self):
        """A local path would be meaningless to a trainer on another machine."""
        with pytest.raises(ValueError, match="s3://"):
            TrainingCreate(character="pay", trigger="p@y",
                           dataset_images=["/home/david/img.jpg"] * 13)

    def test_duplicates_are_refused(self):
        """A duplicate trains the same image twice under two sel_NNN names, silently
        reweighting the set toward it."""
        with pytest.raises(ValueError, match="duplicates"):
            TrainingCreate(character="pay", trigger="p@y",
                           dataset_images=["s3://b/a.jpg"] * 13)

    def test_too_few_images_is_refused_by_the_route(self):
        """The floor moved off the schema when dataset_id arrived: a schema minimum on
        dataset_images would reject every dataset_id request, which is the normal path now."""
        import inspect
        from app.routes import training as mod
        assert "at least {MIN_DATASET_IMAGES} are needed" in inspect.getsource(mod.create_training_job)

    def test_a_character_cannot_contain_a_path(self):
        """It becomes a directory name and an output filename on the trainer."""
        for bad in ("../etc", "a/b", ".hidden", "two words"):
            with pytest.raises(ValueError):
                TrainingCreate(character=bad, trigger="t", dataset_images=_images())

    def test_the_trigger_may_differ_from_the_character(self):
        """p@y REQUIRES this: the LoRA is served over HTTP and lands in JSON and URLs, so the
        filename cannot hold an @ — but the captions trained on one."""
        t = TrainingCreate(character="pay", trigger="p@y", dataset_images=_images())
        assert t.character == "pay" and t.trigger == "p@y"


class TestTheClaimGate:
    """Only a trainer may claim. A render worker or a captioner picking up a 50-minute training
    run would be worse than it going unclaimed."""

    @pytest.mark.parametrize("kind", [WorkerKind.RENDER, WorkerKind.SERVICE])
    def test_only_a_trainer_kind_passes(self, kind):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "worker.kind != WorkerKind.TRAINER" in src
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
        assert row.char_lora == "pay_v1_e05.safetensors"
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
        assert row.char_lora == "pay_v2_e03.safetensors"
        assert row.strength_stage_1 == 0.9, "hand-tuned strengths were overwritten"

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
        t = TrainingCreate(character="p@y", trigger="p@y", dataset_images=_images())
        assert t.character == "p@y"

    def test_the_filename_stem_is_decided_at_creation_not_at_upload(self):
        """It is a field on the request, defaulted and correctable, rather than a guess made
        silently when the file lands. See TestTheLoraFilename for why."""
        import inspect
        from app.routes import training as mod
        assert "lora_name" in inspect.getsource(mod.create_training_job)
        assert 'get("lora_name")' in inspect.getsource(mod.upload_training_artifact)

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
        assert rows[0].char_lora == "pay_v3_e04.safetensors"


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
        t = TrainingCreate(character="p@y", trigger="p@y", lora_name="pay",
                           dataset_images=_images())
        assert t.lora_name == "pay"

    def test_an_unsafe_explicit_name_is_refused(self):
        """Otherwise the field just moves the problem."""
        for bad in ("p@y", "a/b", "with space"):
            with pytest.raises(ValueError):
                TrainingCreate(character="x", trigger="x", lora_name=bad,
                               dataset_images=_images())

    def test_the_upload_uses_the_stored_stem_not_a_fresh_guess(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.upload_training_artifact)
        assert 'get("lora_name")' in src


class TestEveryCheckpointIsOffered:
    """Choosing between epochs is a judgement made by eye at a fixed seed. Loss does not rank
    them — a confident "later epochs overfit" call read off a loss curve was refuted outright on
    d0ggyff — so a console that only offers the final epoch has thrown the decision away."""

    def test_uploading_appends_rather_than_replaces(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.upload_training_artifact)
        assert "job.checkpoints = existing + [uri]" in src

    def test_the_same_uri_is_not_recorded_twice(self):
        import inspect
        from app.routes import training as mod
        assert "if uri not in existing:" in inspect.getsource(mod.upload_training_artifact)

    def test_output_lora_path_tracks_the_most_recent(self):
        """It is what the character row points at until someone picks differently."""
        import inspect
        from app.routes import training as mod
        assert "job.output_lora_path = uri" in inspect.getsource(mod.upload_training_artifact)


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
