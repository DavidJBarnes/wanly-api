"""Polls pending GPU reservations and launches a worker when capacity appears.

A dumb driver around reservations.decide(). Every judgement lives there, pure and tested as a
table; this file only does the I/O and the state transitions.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update

from app import runpod_client
from app.config import settings
from app.database import async_session
from app.models import GpuReservation
from app.reservations import Action, ReservationStatus, decide

logger = logging.getLogger(__name__)

# Availability is checked, not held — losing a race to another customer is expected and normal.
# A tighter loop would not win meaningfully more often and would hammer the API for it.
POLL_SECONDS = 45


async def reservation_monitor():
    import asyncio

    while True:
        await asyncio.sleep(POLL_SECONDS)
        try:
            await run_once()
        except Exception:
            logger.exception("Reservation monitor error")


async def run_once() -> None:
    """One pass over pending reservations."""
    async with async_session() as session:
        pending = (
            await session.execute(
                select(GpuReservation).where(GpuReservation.status == ReservationStatus.PENDING)
            )
        ).scalars().all()

        if not pending:
            return

        # One availability check for the whole pass. Every reservation targets the same GPU in
        # the same datacenter, because the network volume is region-locked — so asking once per
        # reservation would only multiply the requests.
        try:
            availability = await runpod_client.get_availability()
            available = bool(availability.get("available"))
            probe_error = None
        except Exception as e:  # noqa: BLE001 - classified by decide()
            available = False
            probe_error = e

        now = datetime.now(timezone.utc)
        for reservation in pending:
            decision = decide(
                status=reservation.status,
                expires_at=reservation.expires_at,
                pod_id=reservation.pod_id,
                available=available,
                now=now,
                last_error=probe_error,
            )
            await _apply(session, reservation, decision, now)

        await session.commit()


async def _apply(session, reservation: GpuReservation, decision, now) -> None:
    if decision.action is Action.WAIT:
        return

    if decision.action is Action.EXPIRE:
        reservation.status = ReservationStatus.EXPIRED
        logger.info("Reservation %s expired without capacity", reservation.name)
        return

    if decision.action is Action.ABORT:
        reservation.status = ReservationStatus.FAILED
        reservation.error = decision.reason[:500]
        logger.error("Reservation %s aborted: %s", reservation.name, decision.reason)
        return

    # LAUNCH. Claim the row first, with a conditional update that can only succeed once.
    #
    # Two passes overlapping — a slow launch, a restarted container, a second worker process —
    # would otherwise each see a pending row and each launch a pod, billing in parallel. The
    # database is the only place that can arbitrate this; an in-process lock cannot.
    claimed = await session.execute(
        update(GpuReservation)
        .where(
            GpuReservation.id == reservation.id,
            GpuReservation.status == ReservationStatus.PENDING,
            GpuReservation.pod_id.is_(None),
        )
        .values(status=ReservationStatus.LAUNCHED, updated_at=now)
        .returning(GpuReservation.id)
    )
    if claimed.scalar_one_or_none() is None:
        logger.info("Reservation %s already claimed by another pass", reservation.name)
        return

    reservation.attempts = (reservation.attempts or 0) + 1
    env = {
        "FRIENDLY_NAME": reservation.name,
        "QUEUE_URL": settings.runpod_worker_queue_url,
        "QUEUE_API_KEY": settings.api_key,
    }
    if settings.runpod_api_key:
        env["RUNPOD_API_KEY"] = settings.runpod_api_key

    try:
        pod = await runpod_client.launch_worker(reservation.name, env)
    except Exception as e:  # noqa: BLE001
        # Release the claim so the window can still be used — unless the error is fatal, which
        # decide() will catch on the next pass and turn into an abort.
        reservation.status = ReservationStatus.PENDING
        reservation.error = str(e)[:500]
        logger.warning("Reservation %s launch failed: %s", reservation.name, e)
        return

    reservation.pod_id = pod.get("id")
    reservation.error = None
    logger.info(
        "Reservation %s launched pod %s (attempt %d)",
        reservation.name, reservation.pod_id, reservation.attempts,
    )
