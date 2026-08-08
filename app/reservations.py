"""Deciding what to do with a GPU reservation.

A reservation is "keep trying to launch a worker until you get one or the window closes".
Availability swings hard — the 4090 in EU-RO-1 was available in 10/10 samples one morning and
0/7 that evening — so the gap between a pod freeing up and someone asking for it is exactly what
a poller closes and a person does not.

Everything that has to be *correct* lives in decide(), which is pure: no clock, no database, no
network. The poller is a dumb driver around it. That is deliberate — the interesting cases here
are temporal (expiry boundaries, a clock that jumps) and adversarial (a revoked key retried for
an hour), and none of them are worth testing through a loop and a database if they can be a row
in a table instead.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ReservationStatus(str, Enum):
    PENDING = "pending"
    LAUNCHED = "launched"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Action(str, Enum):
    LAUNCH = "launch"
    WAIT = "wait"
    EXPIRE = "expire"
    ABORT = "abort"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str = ""


def decide(
    *,
    status: str,
    expires_at: datetime,
    pod_id: str | None,
    available: bool,
    now: datetime,
    last_error: Exception | None = None,
) -> Decision:
    """What should happen to this reservation on this poll pass?

    `now` is a parameter and never read from the clock inside, so a 60 minute window, an exact
    boundary and a backwards clock jump are all testable in microseconds.
    """
    # Terminal states first. A cancelled reservation must not launch even if a GPU is sitting
    # there, and a launched one must never launch twice — that is the difference between one
    # pod and two, billing in parallel.
    if pod_id:
        return Decision(Action.WAIT, "already launched")
    if status in (
        ReservationStatus.CANCELLED,
        ReservationStatus.EXPIRED,
        ReservationStatus.FAILED,
        ReservationStatus.LAUNCHED,
    ):
        return Decision(Action.WAIT, f"terminal: {status}")

    # A fatal error means stop, not keep trying. Retrying a revoked key for 60 minutes wastes
    # the window and teaches nothing; the user learns their reservation "expired" when it in
    # fact never had a chance.
    if last_error is not None and not is_capacity_error(last_error):
        return Decision(Action.ABORT, f"{type(last_error).__name__}: {last_error}")

    # Expiry beats availability. Launching a pod at the instant the window closes gives the user
    # a worker they have stopped expecting and are not watching.
    if now >= expires_at:
        return Decision(Action.EXPIRE, "window closed")

    # Attempt regardless of what the price check said. It is advice, not a gate.
    #
    # `available` comes from gpuTypes.lowestPrice, which answers "is this GPU sold here" — a
    # different question from "will a host accept this pod", and measured 2026-08-08 it is wrong
    # in BOTH directions. Community 4090 reported available/"Low" through an hour in which every
    # create failed. Minutes after a 3090 pod placed on the first try, the 3090 reported no price
    # at all — and a live 3090 reservation then sat at attempts=0, never calling RunPod once,
    # because this branch believed it.
    #
    # Gating on it is worst precisely here. A reservation exists to keep trying; refusing to try
    # on an unreliable signal makes it do nothing for hours and then report "expired without
    # capacity", which reads as "RunPod had none" when the truth is "we never asked". The create
    # call is the only honest test, its failures are classified by is_capacity_error(), and the
    # window is what bounds the cost.
    return Decision(
        Action.LAUNCH,
        "capacity available" if available else "no price quoted; attempting anyway",
    )


# RunPod reports "no inventory" as prose in a 4xx, not as a distinct code, so the retry/abort
# split has to be made on the message. Erring toward ABORT would strand a reservation on a
# transient blip; erring toward WAIT burns the window on an unfixable error. These phrases are
# what RunPod actually returns for capacity.
_CAPACITY_PHRASES = (
    "no instances available",
    "no instances currently available",
    "no longer any instances available",
    "not enough free gpu",
    "out of capacity",
    "no capacity",
    # RunPod's other placement failure, and the one that is easiest to misread. It means a host
    # WAS matched and could not fit the pod, not that the fleet is empty -- measured 2026-08-08,
    # when community 4090 returned this on every on-demand create while the identical spec
    # succeeded as interruptible. Retryable for the same reason: it is about a moment's headroom
    # on one machine, and the next attempt draws a different machine. It reached the right
    # verdict by falling through to the unknown-error default before; naming it makes that
    # deliberate rather than lucky, and keeps it right if the default ever changes.
    "does not have the resources",
)


def is_capacity_error(error: Exception) -> bool:
    """Is this the kind of failure that trying again might fix?

    Unknown errors are treated as capacity errors — i.e. retryable — because the window bounds
    the damage either way, while wrongly aborting loses a reservation the user asked for. The
    errors we know are fatal are named explicitly.
    """
    text = str(error).lower()

    # Configuration and credential problems: another 40 attempts will not help.
    fatal_markers = (
        "401",
        "unauthorized",
        "rejected the api key",
        "runpod_api_key",
        "runpod_network_volume_id",
        "not configured",
        "invalid",
        "does not exist",
        "not found",
    )
    if any(marker in text for marker in fatal_markers):
        return False

    if any(phrase in text for phrase in _CAPACITY_PHRASES):
        return True

    # Unknown: retry within the window.
    return True
