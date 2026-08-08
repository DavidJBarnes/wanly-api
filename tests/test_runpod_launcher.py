"""Tests for the RunPod worker launcher (wanly-console#288).

The valuable cases here are the refusals, not the happy path. Launching spends money per hour
and the failure modes are all "it looked like it worked":

  - no capacity must read as no capacity, not as an opaque RunPod spec error
  - a missing key must say so rather than 500
  - the datacenter must stay pinned to the volume's region, because a pod in the wrong region
    silently runs without the models and re-downloads ~39GB instead of failing
"""

import httpx
import pytest

from app import runpod_client
from app.config import settings
from app.runpod_client import RunPodError


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _Client:
    """Stands in for httpx.AsyncClient, recording what was sent."""

    calls: list = []

    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _Client.calls.append(("POST", url, json, headers))
        return self._responses.pop(0)

    async def get(self, url, headers=None):
        _Client.calls.append(("GET", url, None, headers))
        return self._responses.pop(0)

    async def delete(self, url, headers=None):
        _Client.calls.append(("DELETE", url, None, headers))
        return self._responses.pop(0)


def _install(monkeypatch, *responses):
    _Client.calls = []
    monkeypatch.setattr(settings, "runpod_api_key", "rpa_test")
    monkeypatch.setattr(settings, "runpod_network_volume_id", "vol_test")
    monkeypatch.setattr(settings, "runpod_cloud_type", "SECURE")
    monkeypatch.setattr(settings, "runpod_datacenter_id", "EU-RO-1")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client(list(responses)))


def _availability_payload(price):
    return {"data": {"gpuTypes": [{"lowestPrice": {
        "uninterruptablePrice": price, "stockStatus": "Low" if price else None}}]}}


class TestAvailability:
    async def test_price_means_available(self, monkeypatch):
        _install(monkeypatch, _Resp(payload=_availability_payload(0.74)))
        out = await runpod_client.get_availability()
        assert out["available"] is True
        assert out["price_per_hr"] == 0.74

    async def test_no_price_means_no_capacity(self, monkeypatch):
        """RunPod reports 'no inventory' as a null price, not an error."""
        _install(monkeypatch, _Resp(payload=_availability_payload(None)))
        out = await runpod_client.get_availability()
        assert out["available"] is False

    async def test_rejected_key_says_so(self, monkeypatch):
        _install(monkeypatch, _Resp(status=401))
        with pytest.raises(RunPodError, match="401"):
            await runpod_client.get_availability()

    async def test_missing_key_is_a_clear_message_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(settings, "runpod_api_key", "")
        with pytest.raises(RunPodError, match="RUNPOD_API_KEY"):
            await runpod_client.get_availability()


class TestLaunch:
    async def test_pins_datacenter_volume_and_gpu(self, monkeypatch):
        """The volume is region-locked. A pod in the wrong datacenter cannot mount it and
        RunPod does not complain — it just re-downloads ~39GB and runs anyway."""
        _install(monkeypatch, _Resp(status=201, payload={"id": "pod1", "name": "w"}))
        await runpod_client.launch_worker("w", {"FRIENDLY_NAME": "w"})
        _, url, body, _ = _Client.calls[0]
        assert url.endswith("/pods")
        assert body["dataCenterIds"] == [settings.runpod_datacenter_id]
        assert body["networkVolumeId"] == "vol_test"
        assert body["gpuTypeIds"] == [settings.runpod_gpu_type_id]
        assert body["cloudType"] == "SECURE"

    async def test_community_sends_no_volume_or_datacenter(self, monkeypatch):
        """Community cloud supports neither. Sending them is rejected or silently ignored, and
        a datacenter pin only means anything when there is a volume to sit beside."""
        _install(monkeypatch, _Resp(status=201, payload={"id": "pod1"}))
        monkeypatch.setattr(settings, "runpod_cloud_type", "COMMUNITY")
        monkeypatch.setattr(settings, "runpod_network_volume_id", "")
        monkeypatch.setattr(settings, "runpod_datacenter_id", "")
        await runpod_client.launch_worker("w", {})
        _, _, body, _ = _Client.calls[0]
        assert body["cloudType"] == "COMMUNITY"
        assert "networkVolumeId" not in body
        assert "dataCenterIds" not in body
        # ...but it MUST still get a disk. Without a network volume this is the only thing that
        # makes volumeMountPath mean anything.
        assert body["volumeInGb"] == settings.runpod_volume_gb

    async def test_runpod_error_prose_is_passed_through(self, monkeypatch):
        """Their wording distinguishes no-stock from a bad spec; ours would not."""
        _install(monkeypatch, _Resp(status=400, payload={"error": "no instances available"}))
        with pytest.raises(RunPodError, match="no instances available"):
            await runpod_client.launch_worker("w", {})


class TestTerminate:
    async def test_terminate_calls_delete(self, monkeypatch):
        _install(monkeypatch, _Resp(status=204))
        await runpod_client.terminate_worker("pod1")
        method, url, _, _ = _Client.calls[0]
        assert method == "DELETE" and url.endswith("/pods/pod1")

    async def test_already_gone_is_success(self, monkeypatch):
        """404 means the pod is in the state the caller asked for."""
        _install(monkeypatch, _Resp(status=404))
        await runpod_client.terminate_worker("pod1")


class TestAvailabilityFollowsTheConfiguredCloud:
    """Asking the wrong cloud reports the wrong answer, and the two differ by 2x in price and
    wildly in stock: community 4090 was purchasable in 6/6 samples while secure showed 0/7
    earlier the same evening."""

    async def test_community_queries_community(self, monkeypatch):
        _install(monkeypatch, _Resp(payload=_availability_payload(0.34)))
        monkeypatch.setattr(settings, "runpod_cloud_type", "COMMUNITY")
        monkeypatch.setattr(settings, "runpod_datacenter_id", "")
        out = await runpod_client.get_availability()
        _, _, body, _ = _Client.calls[0]
        assert body["variables"]["secure"] is False
        assert out["price_per_hr"] == 0.34

    async def test_no_datacenter_is_omitted_not_sent_as_null(self, monkeypatch):
        """Passing dataCenterId: null narrows the query to datacenters with no id, which reports
        no capacity anywhere — indistinguishable from being genuinely out of stock."""
        _install(monkeypatch, _Resp(payload=_availability_payload(0.34)))
        monkeypatch.setattr(settings, "runpod_cloud_type", "COMMUNITY")
        monkeypatch.setattr(settings, "runpod_datacenter_id", "")
        await runpod_client.get_availability()
        _, _, body, _ = _Client.calls[0]
        assert "dc" not in body["variables"]
        assert "dataCenterId" not in body["query"]

    async def test_secure_with_a_datacenter_still_pins_it(self, monkeypatch):
        _install(monkeypatch, _Resp(payload=_availability_payload(0.74)))
        monkeypatch.setattr(settings, "runpod_cloud_type", "SECURE")
        monkeypatch.setattr(settings, "runpod_datacenter_id", "EU-RO-1")
        await runpod_client.get_availability()
        _, _, body, _ = _Client.calls[0]
        assert body["variables"]["secure"] is True
        assert body["variables"]["dc"] == "EU-RO-1"


class TestGpuSelection:
    """Choosing between GPUs (wanly-console#286 follow-up, 2026-08-08).

    Measured that day: community 4090 returned "this machine does not have the resources" on
    every on-demand create for over an hour, while community 3090 and the identical 4090 spec as
    interruptible both placed on the first try. So the fleet was not empty -- the 4090 hosts were
    all partially committed. Being able to ask for a 3090 instead is the difference between
    working and waiting.
    """

    def test_offers_the_configured_list(self, monkeypatch):
        monkeypatch.setattr(settings, "runpod_gpu_type_ids", "GPU A,GPU B")
        monkeypatch.setattr(settings, "runpod_gpu_type_id", "GPU A")
        assert runpod_client.selectable_gpus() == ["GPU A", "GPU B"]

    def test_default_is_always_offered_even_if_absent_from_the_list(self, monkeypatch):
        # Otherwise a typo in the list silently removes the GPU every launch defaults to, and
        # the dialog offers everything except the one it is about to use.
        monkeypatch.setattr(settings, "runpod_gpu_type_ids", "GPU B")
        monkeypatch.setattr(settings, "runpod_gpu_type_id", "GPU A")
        assert runpod_client.selectable_gpus() == ["GPU A", "GPU B"]

    def test_blank_entries_are_dropped(self, monkeypatch):
        monkeypatch.setattr(settings, "runpod_gpu_type_ids", "GPU A, ,GPU B,")
        monkeypatch.setattr(settings, "runpod_gpu_type_id", "GPU A")
        assert runpod_client.selectable_gpus() == ["GPU A", "GPU B"]

    @pytest.mark.asyncio
    async def test_launch_asks_for_the_requested_gpu(self, monkeypatch):
        _install(monkeypatch, _Resp(200, {"id": "pod1", "name": "w"}))
        await runpod_client.launch_worker("w", {}, "NVIDIA GeForce RTX 3090")
        spec = _Client.calls[0][2]
        assert spec["gpuTypeIds"] == ["NVIDIA GeForce RTX 3090"]

    @pytest.mark.asyncio
    async def test_launch_falls_back_to_the_configured_default(self, monkeypatch):
        _install(monkeypatch, _Resp(200, {"id": "pod1", "name": "w"}))
        monkeypatch.setattr(settings, "runpod_gpu_type_id", "NVIDIA GeForce RTX 4090")
        await runpod_client.launch_worker("w", {}, None)
        assert _Client.calls[0][2]["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]

    @pytest.mark.asyncio
    async def test_availability_asks_about_the_requested_gpu(self, monkeypatch):
        _install(monkeypatch, _Resp(200, _availability_payload(0.22)))
        result = await runpod_client.get_availability("NVIDIA GeForce RTX 3090")
        assert _Client.calls[0][2]["variables"]["gpu"] == "NVIDIA GeForce RTX 3090"
        assert result["gpu_type_id"] == "NVIDIA GeForce RTX 3090"


class TestPlacementFailureWording:
    """RunPod says two different things in the same 500 and the difference is the diagnosis.

    Passing its prose through unexplained is what sent a user into dozens of blind retries
    against a request that could not succeed.
    """

    def test_fit_failure_says_retry_or_switch_gpu(self):
        msg = runpod_client.explain_placement_failure(
            "create pod: This machine does not have the resources to deploy your pod."
        )
        assert "different GPU" in msg
        # Must NOT claim there is no stock -- that is the other error, and acting on it (waiting)
        # is exactly the wrong response to a fit failure.
        assert "no stock" not in msg.lower()

    def test_no_stock_says_waiting_will_not_help_yet(self):
        msg = runpod_client.explain_placement_failure(
            "create pod: There are no instances currently available"
        )
        assert "no stock" in msg.lower()
        assert "retrying will not help" in msg.lower()

    def test_unrecognised_errors_pass_through_untouched(self):
        assert runpod_client.explain_placement_failure("something else") == "something else"

    @pytest.mark.asyncio
    async def test_launch_failure_is_explained_not_just_relayed(self, monkeypatch):
        _install(monkeypatch, _Resp(
            500, {"error": "create pod: This machine does not have the resources to deploy your pod"}
        ))
        with pytest.raises(RunPodError) as e:
            await runpod_client.launch_worker("w", {})
        assert "different GPU" in str(e.value)


class TestPodHasSomewhereToPutTheModels:
    """A pod without a volume dies during staging, not at create time.

    Reported 2026-08-08 from a live community pod: "No space left on device (os error 28)".
    The spec set volumeMountPath=/workspace but never volumeInGb, so RunPod allocated NO volume
    and /workspace fell back onto the 30GB container disk. download_models.sh pulls ~37GB there
    (two 13.3GB experts, a 6.7GB text encoder, CLIP vision, VAE, LoRAs, FaceFusion). It cannot fit.

    It had always worked before because a network volume supplied the mount. Terminating that
    volume removed the storage and nothing replaced it. Create still succeeded, which is why
    probing placement did not catch it.
    """

    @pytest.mark.asyncio
    async def test_community_pod_gets_a_volume_big_enough_for_the_models(self, monkeypatch):
        _install(monkeypatch, _Resp(status=201, payload={"id": "pod1"}))
        monkeypatch.setattr(settings, "runpod_cloud_type", "COMMUNITY")
        monkeypatch.setattr(settings, "runpod_network_volume_id", "")
        await runpod_client.launch_worker("w", {})
        body = _Client.calls[0][2]
        assert body["volumeMountPath"] == settings.runpod_volume_mount_path
        # 37GB of models plus room to generate into. A mount path with no volume behind it is
        # the exact bug this guards.
        assert body["volumeInGb"] >= 45

    @pytest.mark.asyncio
    async def test_a_network_volume_replaces_the_pod_disk_rather_than_adding_one(self, monkeypatch):
        # Paying for both would be waste, and the models live on the network volume already.
        _install(monkeypatch, _Resp(status=201, payload={"id": "pod1"}))
        monkeypatch.setattr(settings, "runpod_network_volume_id", "vol_test")
        await runpod_client.launch_worker("w", {})
        body = _Client.calls[0][2]
        assert body["networkVolumeId"] == "vol_test"
        assert "volumeInGb" not in body

    @pytest.mark.asyncio
    async def test_zero_disables_it_for_the_network_volume_case(self, monkeypatch):
        _install(monkeypatch, _Resp(status=201, payload={"id": "pod1"}))
        monkeypatch.setattr(settings, "runpod_network_volume_id", "")
        monkeypatch.setattr(settings, "runpod_volume_gb", 0)
        await runpod_client.launch_worker("w", {})
        assert "volumeInGb" not in _Client.calls[0][2]
