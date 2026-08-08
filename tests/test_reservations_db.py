"""Reservation guarantees that need a real database (wanly-console#296).

These are the Tier 2 cases that #162 unblocked. They cannot be faked honestly: the correct
implementation of "at most one pod" is a conditional UPDATE that can only succeed once, and a
mock cannot demonstrate that it holds under a second pass.

This matters more here than elsewhere because a reservation spends money unattended. A double
launch is two pods billing in parallel with nobody watching.
"""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from app.models import GpuReservation, Worker
from app.reservations import ReservationStatus


async def _reservation(db, name="res-1", status=ReservationStatus.PENDING, **kw):
    r = GpuReservation(
        id=uuid.uuid4(),
        name=name,
        status=status,
        expires_at=kw.pop("expires_at", datetime.now(timezone.utc) + timedelta(minutes=30)),
        **kw,
    )
    db.add(r)
    await db.flush()
    return r


async def _claim(db, reservation_id) -> bool:
    """The poller's claim: a conditional update that can only succeed once."""
    result = await db.execute(
        update(GpuReservation)
        .where(
            GpuReservation.id == reservation_id,
            GpuReservation.status == ReservationStatus.PENDING,
            GpuReservation.pod_id.is_(None),
        )
        .values(status=ReservationStatus.LAUNCHED)
        .returning(GpuReservation.id)
    )
    return result.scalar_one_or_none() is not None


class TestAtMostOnePod:
    async def test_a_second_pass_cannot_claim_the_same_reservation(self, db):
        """Two overlapping passes — a slow launch, a restarted container — would otherwise each
        see a pending row and each launch a pod, billing in parallel."""
        r = await _reservation(db)
        assert await _claim(db, r.id) is True
        assert await _claim(db, r.id) is False, "the second pass must not launch a second pod"

    async def test_a_reservation_with_a_pod_cannot_be_reclaimed(self, db):
        r = await _reservation(db, status=ReservationStatus.PENDING, pod_id="pod-existing")
        assert await _claim(db, r.id) is False

    async def test_cancelling_before_the_claim_prevents_the_launch(self, db):
        r = await _reservation(db)
        await db.execute(
            update(GpuReservation)
            .where(GpuReservation.id == r.id)
            .values(status=ReservationStatus.CANCELLED)
        )
        assert await _claim(db, r.id) is False


class TestSurvivesRestart:
    async def test_a_pending_reservation_is_found_by_a_fresh_query(self, db):
        """The state lives in the row, not in the process. A deploy recreates the container;
        the reservation has to still be there afterwards or it dies invisibly."""
        await _reservation(db, name="survives")
        found = (
            await db.execute(
                select(GpuReservation).where(
                    GpuReservation.status == ReservationStatus.PENDING
                )
            )
        ).scalars().all()
        assert [r.name for r in found] == ["survives"]

    async def test_terminal_reservations_are_not_polled_again(self, db):
        for status in (
            ReservationStatus.EXPIRED,
            ReservationStatus.CANCELLED,
            ReservationStatus.FAILED,
            ReservationStatus.LAUNCHED,
        ):
            await _reservation(db, name=f"done-{status}", status=status)
        pending = (
            await db.execute(
                select(GpuReservation).where(
                    GpuReservation.status == ReservationStatus.PENDING
                )
            )
        ).scalars().all()
        assert pending == []


class TestReservedDrainPolicy:
    async def test_the_policy_reaches_the_worker_when_it_registers(self, db):
        """The pod exists minutes before the worker row does, and drain_after_jobs lives on the
        worker — so the policy has to be applied at registration or not at all."""
        from app.routes.workers import _apply_reserved_drain

        await _reservation(
            db, name="reserved-worker", status=ReservationStatus.LAUNCHED, drain_after_jobs=3
        )
        worker = Worker(
            friendly_name="reserved-worker", hostname="h", ip_address="127.0.0.1",
        )
        db.add(worker)
        await db.flush()

        await _apply_reserved_drain(db, worker)
        assert worker.drain_after_jobs == 3

    async def test_an_existing_countdown_is_not_restarted(self, db):
        """A container restart re-registers the worker. Overwriting the countdown each time
        would mean it never reaches zero and never drains."""
        from app.routes.workers import _apply_reserved_drain

        await _reservation(
            db, name="mid-drain", status=ReservationStatus.LAUNCHED, drain_after_jobs=3
        )
        worker = Worker(
            friendly_name="mid-drain", hostname="h", ip_address="127.0.0.1",
            drain_after_jobs=1,
        )
        db.add(worker)
        await db.flush()

        await _apply_reserved_drain(db, worker)
        assert worker.drain_after_jobs == 1, "the countdown must keep counting down"

    async def test_a_worker_with_no_reservation_is_untouched(self, db):
        from app.routes.workers import _apply_reserved_drain

        worker = Worker(friendly_name="3090.zero", hostname="h", ip_address="127.0.0.1")
        db.add(worker)
        await db.flush()
        await _apply_reserved_drain(db, worker)
        assert worker.drain_after_jobs is None
