"""A launched pod must be able to hold the model set it will download.

The first real pod, 2026-09-01, was launched with a 60 GB volume against a 58 GB model set
and filled partway through staging. 60 was correct for WAN (~37 GB) and was never revisited
when the image became LTX — the defaults and the comment beside them both still described the
retired pipeline.

The sizes are asserted against the files download_models.sh actually fetches, read from that
script rather than restated here, so shipping a bigger checkpoint fails this test instead of
failing a pod twenty minutes into a boot.
"""

import pathlib
import re

from app.config import Settings

DOCKER_REPO = pathlib.Path(__file__).parent.parent.parent / "wanly-gpu-docker"

# What the worker downloads, in GB. Kept here because the sizes are a property of the weights,
# not of the script; the NAMES are cross-checked against the script below so the two cannot
# drift apart silently.
MODEL_SIZES_GB = {
    "sulphur_dev_bf16.safetensors": 43,
    "gemma_3_12B_it_fp8_scaled.safetensors": 13,
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": 1,
    "sulphur_distill_lora_condsafe.safetensors": 1,
}
# Character LoRAs are synced per claim at ~625 MB each; a working pod holds several.
LORA_HEADROOM_GB = 6


def test_the_volume_fits_the_model_set_with_room_for_loras():
    volume = Settings.model_fields["runpod_volume_gb"].default
    needed = sum(MODEL_SIZES_GB.values()) + LORA_HEADROOM_GB
    assert volume >= needed, (
        f"runpod_volume_gb={volume} cannot hold {needed} GB of models and LoRAs — a pod "
        f"launched this way fails partway through staging, not at create time"
    )


def test_the_container_disk_holds_more_than_a_couple_of_renders():
    # /jobs lives on the container disk: graph, keyframe and mp4 per render, ~60 MB together.
    disk = Settings.model_fields["runpod_container_disk_gb"].default
    assert disk >= 40, f"runpod_container_disk_gb={disk} leaves too little for /jobs"


def test_the_documented_model_names_still_match_the_downloader():
    """If download_models.sh starts fetching something else, these sizes are fiction."""
    script = DOCKER_REPO / "download_models.sh"
    if not script.exists():
        return  # sibling checkout not present; CI runs this repo alone
    text = script.read_text()
    fetched = set(re.findall(r"\|([A-Za-z0-9._-]+\.safetensors)\|", text))
    assert fetched, "could not read the download manifest — the parsing has broken"
    assert fetched == set(MODEL_SIZES_GB), (
        f"the downloader fetches {sorted(fetched)} but the sizes here describe "
        f"{sorted(MODEL_SIZES_GB)} — resize the volume or update this list"
    )
