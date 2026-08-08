"""The reservation decision table (wanly-console#296).

Everything that must be correct about a reservation is a case here: it is pure, so a 60 minute
window, an exact boundary and a clock that jumps backwards all run in microseconds.

The two that matter most are the error split. A reservation spends money unattended — that is
its purpose, and the drain-at-reservation option exists because it can fire at 11:47pm with
nobody watching. Retrying a revoked key for the full window wastes it silently; aborting on a
transient blip loses a reservation the user asked for.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.reservations import Action, ReservationStatus, decide, is_capacity_error

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(minutes=30)


def _decide(**overrides):
    kwargs = dict(
        status=ReservationStatus.PENDING,
        expires_at=LATER,
        pod_id=None,
        available=False,
        now=NOW,
        last_error=None,
    )
    kwargs.update(overrides)
    return decide(**kwargs)


class TestCoreCases:
    def test_capacity_inside_the_window_launches(self):
        assert _decide(available=True).action is Action.LAUNCH

    def test_no_price_quoted_still_attempts(self):
        # The price check is advice, not a gate. Measured 2026-08-08 it is wrong in BOTH
        # directions: community 4090 reported available through an hour of total failure, and
        # the 3090 reported NO PRICE minutes after a 3090 pod placed on the first try. A live
        # reservation then sat at attempts=0 and expired without ever calling RunPod. A
        # reservation that will not try is not a reservation.
        assert _decide(available=False).action is Action.LAUNCH

    def test_no_capacity_past_the_deadline_expires(self):
        assert _decide(available=False, now=LATER).action is Action.EXPIRE

    def test_capacity_past_the_deadline_still_expires(self):
        """Do not launch on the way out. The user has stopped expecting a worker and is not
        watching one they would then be billed for."""
        assert _decide(available=True, now=LATER).action is Action.EXPIRE

    def test_the_boundary_is_inclusive_of_expiry(self):
        assert _decide(available=True, now=LATER - timedelta(seconds=1)).action is Action.LAUNCH
        assert _decide(available=True, now=LATER).action is Action.EXPIRE


class TestNeverLaunchesTwice:
    def test_a_reservation_with_a_pod_never_launches_again(self):
        """The difference between one pod and two billing in parallel."""
        assert _decide(pod_id="pod123", available=True).action is not Action.LAUNCH

    @pytest.mark.parametrize("status", [
        ReservationStatus.CANCELLED,
        ReservationStatus.EXPIRED,
        ReservationStatus.FAILED,
        ReservationStatus.LAUNCHED,
    ])
    def test_terminal_states_never_launch(self, status):
        assert _decide(status=status, available=True).action is not Action.LAUNCH

    def test_cancelled_wins_over_available_capacity(self):
        assert _decide(status=ReservationStatus.CANCELLED, available=True).action is Action.WAIT


class TestErrorHandling:
    def test_a_fatal_error_aborts_rather_than_burning_the_window(self):
        d = _decide(last_error=RuntimeError("RunPod rejected the API key (401)."), available=True)
        assert d.action is Action.ABORT
        assert "401" in d.reason

    def test_a_capacity_error_keeps_the_reservation_alive(self):
        # The point is that it does not ABORT -- a transient capacity failure must not kill a
        # window the user is still inside. Since the price check no longer gates, "alive" now
        # means it tries again rather than idles.
        d = _decide(last_error=RuntimeError("no instances available"), available=False)
        assert d.action is not Action.ABORT
        assert d.action is Action.LAUNCH


class TestErrorClassification:
    @pytest.mark.parametrize("message", [
        "no instances available",
        "There are no longer any instances available with the requested specifications",
        "out of capacity",
    ])
    def test_capacity_messages_are_retryable(self, message):
        assert is_capacity_error(RuntimeError(message)) is True

    @pytest.mark.parametrize("message", [
        "RunPod rejected the API key (401).",
        "RunPod is not configured on this server (RUNPOD_API_KEY is unset).",
        "RUNPOD_NETWORK_VOLUME_ID is unset",
        "network volume does not exist",
    ])
    def test_configuration_and_credential_failures_are_fatal(self, message):
        assert is_capacity_error(RuntimeError(message)) is False

    def test_unknown_errors_are_retried_within_the_window(self):
        """The window bounds the damage; wrongly aborting loses a reservation outright."""
        assert is_capacity_error(RuntimeError("something nobody has seen before")) is True


class TestPlacementErrorsAreRetryable:
    """Both of RunPod's placement failures must keep a reservation alive.

    Measured 2026-08-08: community 4090 answered "this machine does not have the resources" on
    every on-demand create while a 3090 placed instantly. That message means a host WAS matched
    and could not fit the pod -- a different machine next attempt may well fit it. Aborting on it
    would kill a reservation that was about to succeed.

    Both previously reached the right verdict only by falling through to the unknown-error
    default. These pin it, so a future change to that default cannot silently invert them.
    """

    def test_fit_failure_is_capacity_not_fatal(self):
        assert is_capacity_error(
            Exception("create pod: This machine does not have the resources to deploy your pod")
        )

    def test_no_instances_currently_available_is_capacity(self):
        assert is_capacity_error(Exception("create pod: There are no instances currently available"))

    def test_a_revoked_key_is_still_fatal(self):
        # The guard that matters: these phrases must not have widened the retryable set so far
        # that a credential failure gets retried for the whole window.
        assert not is_capacity_error(Exception("RunPod rejected the API key (401)."))
