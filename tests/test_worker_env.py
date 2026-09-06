"""The environment a launched pod gets (#260).

Model staging ran unauthenticated on every pod -- ~58 GB fetched anonymously, with HF's own
warning in the boot log. That works today (public repos, measured 14 MB/s) and stops working
in two ways: HF's anonymous limits are per-IP and pods share datacenter egress, and a gated
repo 401s outright rather than warning.

huggingface_hub reads HF_TOKEN from the environment itself, so nothing in the image or in
download_models.sh changes. The whole job is getting the variable into the pod -- and doing
that without leaving two different answers to "what does a worker need to know".
"""
import inspect

import pytest

from app import reservation_monitor, runpod_client
from app.config import settings
from app.routes import runpod as runpod_routes


@pytest.fixture
def clean_settings():
    """Settings is a singleton shared across the suite; restore whatever was there."""
    before = (settings.hf_token, settings.runpod_api_key, settings.api_key)
    yield settings
    (settings.hf_token, settings.runpod_api_key, settings.api_key) = before


def test_the_token_is_omitted_when_not_configured(clean_settings):
    """Not sent blank. An empty HF_TOKEN is worse than none: huggingface_hub would try to
    authenticate with it, turning a warning into a 401 partway through a 46 GB stage."""
    settings.hf_token = ""
    assert "HF_TOKEN" not in runpod_client.worker_env("pod-1")


def test_the_token_is_passed_when_configured(clean_settings):
    settings.hf_token = "hf_notarealtoken"
    assert runpod_client.worker_env("pod-1")["HF_TOKEN"] == "hf_notarealtoken"


def test_the_worker_still_gets_what_it_always_did(clean_settings):
    """The keys that existed before must survive the extraction — a pod without QUEUE_URL
    or QUEUE_API_KEY boots and claims nothing."""
    settings.api_key = "queue-key"
    settings.runpod_api_key = "rp-key"
    env = runpod_client.worker_env("pod-1")
    assert env["FRIENDLY_NAME"] == "pod-1"
    assert env["QUEUE_API_KEY"] == "queue-key"
    assert env["RUNPOD_API_KEY"] == "rp-key"
    assert env["QUEUE_URL"] == settings.runpod_worker_queue_url


def test_an_explicit_queue_url_overrides_the_configured_one(clean_settings):
    """The manual launch can point a pod at a different queue; the reservation path cannot
    and passes nothing."""
    assert runpod_client.worker_env("pod-1", "http://elsewhere:8001")["QUEUE_URL"] \
        == "http://elsewhere:8001"


def test_runpod_key_omitted_when_unset(clean_settings):
    settings.runpod_api_key = ""
    assert "RUNPOD_API_KEY" not in runpod_client.worker_env("pod-1")


@pytest.mark.parametrize("mod,label", [
    (runpod_routes.launch_runpod_worker, "the manual launch"),
    (reservation_monitor, "the reservation monitor"),
])
def test_both_launch_paths_use_the_shared_builder(mod, label):
    """The point of the extraction, asserted rather than trusted.

    Both paths built the same dict by hand. Adding HF_TOKEN to one and not the other is the
    edit this guards: a pod launched overnight would then differ from one launched by hand,
    which presents as "the token works when I click the button" — debugged against a pod
    that has already been terminated.
    """
    src = inspect.getsource(mod)
    assert "worker_env(" in src, f"{label} no longer uses the shared builder"
    assert '"QUEUE_API_KEY"' not in src, f"{label} has grown its own env dict again"
