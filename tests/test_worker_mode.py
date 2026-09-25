"""Flipping a box between rendering and captioning, from the console (wanly-gpu-docker#131).

A thin relay on purpose. The CONTAINER owns what a mode means -- which services, in what
order, and the GPU-sharing rules between them -- so this API forwards and does not decide.
Two definitions of a mode would be two definitions free to drift, and the one on the box is
the one that is true.
"""
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.models import Worker
from app.routes import workers as routes


class _Resp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._body


def _worker(name="3090.zero", status="online-idle"):
    w = Worker(friendly_name=name, hostname="c0ffee", ip_address="172.17.0.2", status=status)
    w.id = uuid.uuid4()
    return w


class TestWhereItSendsThem:
    def test_it_uses_the_friendly_name_not_the_ip(self):
        """The address on the row is what the daemon saw from INSIDE the container -- on the
        3090 that is 172.17.0.2, which this process cannot reach. The friendly name is the
        box's real hostname, and is already how this API reaches its other services."""
        url = routes._control_url(_worker())
        assert url.startswith("http://3090.zero:")
        assert "172.17.0.2" not in url

    def test_it_uses_the_configured_control_port(self):
        assert routes._control_url(_worker()).endswith(":8081")


class TestSettingTheMode:
    @pytest.mark.asyncio
    async def test_it_forwards_the_mode_to_the_box(self, monkeypatch):
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock(return_value=_Resp(
            body={"mode": "caption", "services": ["image-description"], "changed": True}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            out = await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert post.call_args[0][0] == "http://3090.zero:8081/mode"
        assert post.call_args[1]["json"] == {"mode": "caption"}
        assert out.mode == "caption"
        assert out.changed is True

    @pytest.mark.asyncio
    async def test_the_boxs_own_refusal_is_passed_through_verbatim(self):
        """"MODE=caption leaves nothing to run" is text a person can act on. Rewording it
        here, or flattening it to a generic 400, throws that away."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock(return_value=_Resp(
            status_code=400, body={"detail": "MODE=caption leaves nothing to run: ..."}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            with pytest.raises(HTTPException) as e:
                await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert e.value.status_code == 400
        assert "leaves nothing to run" in e.value.detail

    @pytest.mark.asyncio
    async def test_a_box_that_does_not_answer_is_502_not_500(self):
        """This API is fine; the box is not answering. A 500 would send someone reading
        these logs to the wrong machine."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock(side_effect=OSError("connection refused"))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            with pytest.raises(HTTPException) as e:
                await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert e.value.status_code == 502
        assert "3090.zero" in e.value.detail

    @pytest.mark.asyncio
    async def test_an_offline_worker_is_refused_without_a_call(self):
        w = _worker(status="offline")
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock()
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            with pytest.raises(HTTPException) as e:
                await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert e.value.status_code == 400
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_worker_is_404(self):
        db = AsyncMock()
        db.get.return_value = None
        with pytest.raises(HTTPException) as e:
            await routes.set_worker_mode(uuid.uuid4(), routes.WorkerMode(mode="caption"), db)
        assert e.value.status_code == 404


class TestReadingTheMode:
    @pytest.mark.asyncio
    async def test_it_reports_what_the_box_says(self):
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        get = AsyncMock(return_value=_Resp(body={
            "mode": "caption",
            "equipped": ["ltx-engine", "image-description"],
            "services": [{"name": "ltx-engine-api", "group": "ltx-engine", "stopped": True},
                         {"name": "image-description", "group": "image-description"}],
        }))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.get = get
            out = await routes.get_worker_mode(w.id, db)
        assert out.mode == "caption"
        assert out.equipped == ["ltx-engine", "image-description"]
        assert out.services == ["image-description"], "a stopped service was reported as live"

    @pytest.mark.asyncio
    async def test_a_degraded_box_still_reports_its_mode(self):
        """A container with a service deliberately off answers 503 and its body is still
        the truth. Reading only 200s would make caption mode unreadable."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        get = AsyncMock(return_value=_Resp(
            status_code=503, body={"mode": "caption", "equipped": ["ltx-engine"], "services": []}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.get = get
            out = await routes.get_worker_mode(w.id, db)
        assert out.mode == "caption"

    @pytest.mark.asyncio
    async def test_a_box_with_no_mode_reads_as_render(self):
        """An older container answers /health without the field. Absent means it is running
        everything, which is render mode."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        get = AsyncMock(return_value=_Resp(body={"services": []}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.get = get
            out = await routes.get_worker_mode(w.id, db)
        assert out.mode == "ltx-engine"


class TestASwitchThatIsStillRunning:
    """Stopping the render daemon lets the segment in flight FINISH -- by design, so nothing
    is destroyed -- and that is up to ~27 minutes. The box accepts and runs it behind the
    request; this API must relay that rather than wait, or a working switch is reported as a
    failure because the wait timed out."""

    @pytest.mark.asyncio
    async def test_pending_is_relayed_not_waited_on(self):
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock(return_value=_Resp(body={
            "mode": "ltx-engine", "pending": "caption",
            "services": ["image-description"], "changed": True}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            out = await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert out.pending_mode == "caption"
        assert out.mode == "ltx-engine", "it reported a mode the box is not in yet"

    @pytest.mark.asyncio
    async def test_the_read_carries_pending_and_the_last_error(self):
        """A switch fails after the request that asked for it was answered, so /health is
        the only place it can be reported."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        get = AsyncMock(return_value=_Resp(body={
            "mode": "ltx-engine", "pending_mode": "caption",
            "mode_error": "ComfyUI would not stop", "equipped": [], "services": []}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.get = get
            out = await routes.get_worker_mode(w.id, db)
        assert out.pending_mode == "caption"
        assert "would not stop" in out.mode_error

    @pytest.mark.asyncio
    async def test_it_does_not_wait_minutes_for_the_box(self):
        """The timeout is the bug this fixes: 120s against a switch that legitimately takes
        up to ~27 minutes reported failure on every busy box."""
        w = _worker()
        db = AsyncMock()
        db.get.return_value = w
        post = AsyncMock(return_value=_Resp(body={"mode": "ltx-engine", "pending": "caption"}))
        with patch("httpx.AsyncClient") as cl:
            cl.return_value.__aenter__.return_value.post = post
            await routes.set_worker_mode(w.id, routes.WorkerMode(mode="caption"), db)
        assert cl.call_args[1]["timeout"] <= 30
