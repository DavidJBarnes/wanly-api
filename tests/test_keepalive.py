"""The API must outlive the worker's connection pool (#262).

Every worker logged RemoteProtocolError on /segments/next and /workers/*/heartbeat every few
minutes, on every pod. It was first read as CPU starvation on a community host at load
average 251 -- but the same rate appeared on a secure-cloud pod at load 2.11, which ruled
that out.

The cause was a mismatch: uvicorn's default keep-alive is 5s, while the daemon's httpx pool
holds connections for 30s. httpx handed out sockets uvicorn had already closed and found out
only after writing the request. The daemon's own comment (wanly-gpu-daemon#105) states that
its expiry must sit BELOW the far end's idle timeout; it was six times above it.

Nothing broke -- the daemon retries once on a fresh socket and no claim was ever lost. What
it cost was legibility: a genuine transport failure looks exactly like the routine one, so
the claim loop's log could not answer "is this worker healthy?".
"""
import pathlib
import re

DOCKERFILE = pathlib.Path(__file__).parent.parent / "Dockerfile"

# wanly-gpu-daemon/daemon/queue_client.py: httpx.Limits(keepalive_expiry=30.0).
# Restated rather than imported: it lives in another repo that is not checked out in CI.
DAEMON_POOL_EXPIRY_S = 30


def _keepalive_from_dockerfile() -> int | None:
    m = re.search(r"--timeout-keep-alive\s+(\d+)", DOCKERFILE.read_text())
    return int(m.group(1)) if m else None


def test_uvicorn_is_given_an_explicit_keepalive():
    """The default is 5s and it is wrong here, so the flag must be present at all.

    Asserted separately from the comparison below so that dropping the flag reports "you
    removed it" rather than a confusing None comparison.
    """
    assert _keepalive_from_dockerfile() is not None, (
        "Dockerfile does not pass --timeout-keep-alive; uvicorn would default to 5s, which "
        "is below the daemon's 30s pool expiry and reintroduces #262"
    )


def test_the_server_outlives_the_workers_connection_pool():
    """The RELATIONSHIP is the thing that was wrong, so pin that, not the number.

    A bare `== 65` would be re-edited by anyone who wanted a different value, without
    learning why it cannot go under 30.
    """
    keepalive = _keepalive_from_dockerfile()
    # `or 0` for the same reason as below: a missing flag should fail with this message
    # rather than a TypeError comparing None to an int.
    assert (keepalive or 0) > DAEMON_POOL_EXPIRY_S, (
        f"uvicorn --timeout-keep-alive={keepalive}s is not above the daemon's "
        f"{DAEMON_POOL_EXPIRY_S}s httpx pool expiry. Workers will be handed sockets this "
        f"server has already closed, and every claim poll can fail with RemoteProtocolError "
        f"before its retry (#262)."
    )


def test_there_is_real_margin_not_a_race():
    """31 would satisfy the rule and still have the two expiring together. The point of the
    margin is that neither end is deciding at the same moment."""
    keepalive = _keepalive_from_dockerfile()
    # `or 0` so a missing flag fails with this test's message rather than a TypeError from
    # comparing None -- the reason a reader reaches for is in the assertion, not a traceback.
    assert (keepalive or 0) >= DAEMON_POOL_EXPIRY_S * 2, (
        f"--timeout-keep-alive is {keepalive!r}; wanted at least "
        f"{DAEMON_POOL_EXPIRY_S * 2}s so the two ends are not expiring together"
    )
