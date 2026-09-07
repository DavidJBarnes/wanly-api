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

    def test_too_few_images_is_refused(self):
        with pytest.raises(ValueError):
            TrainingCreate(character="pay", trigger="p@y", dataset_images=_images(3))

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

    def test_the_filename_is_sanitised_at_upload_not_at_creation(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.upload_training_artifact)
        assert 'c.isalnum() or c in "._-"' in src
        # and the trigger is untouched by that
        assert "job.trigger" not in src.split("safe =")[1].split("key =")[0]

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
