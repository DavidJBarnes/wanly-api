"""Where a checkpoint can be fetched from, so a worker never has to guess a URL.

console#423. A worker that lacks a base model downloads it rather than being routed around
(console#422). To do that it needs one thing this repo owns: the mapping from a checkpoint
NAME, as a pose stores it, to a place the file actually lives.

WHY THE API HOLDS THIS
    The daemon could carry its own copy, and then adding a checkpoint would mean redeploying
    every worker. Held here, a new base model is one entry and every worker learns about it
    on its next claim. The API already answers "which checkpoints exist" (/ltx/checkpoints);
    this answers "and where does one come from".

THIS MUST AGREE WITH download_models.sh
    wanly-gpu-docker's _WANTED manifest stages the DEFAULT checkpoint at boot. This catalogue
    covers the same files plus the ones a pod does not carry. If the two disagree about a
    repo or a filename, a cold pod stages one file and fetches a different one under the same
    name -- two different base models sharing a name is the worst possible outcome, because
    every render still succeeds.
"""

# name (as a pose stores it, no extension) -> where to get it
#
# Verified 2026-09-06 by byte count against the copies on the 3090, not by filename:
#   10Eros_v1.5_bf16   46,139,886,366   matches TenStrip/LTX2.3-10Eros
#   sulphur_dev_bf16   46,139,885,414   matches SulphurAI/Sulphur-2-base
# A same-named file of a different size is a different model, and that is the failure this
# catalogue exists to prevent -- so entries are added by verifying bytes, never by matching
# a name that looks right.
CHECKPOINT_SOURCES: dict[str, dict[str, str | int]] = {
    "10Eros_v1.5_bf16": {
        "repo": "TenStrip/LTX2.3-10Eros",
        "path": "10Eros_v1.5_bf16.safetensors",
        "size_bytes": 46_139_886_366,
    },
    "sulphur_dev_bf16": {
        "repo": "SulphurAI/Sulphur-2-base",
        "path": "sulphur_dev_bf16.safetensors",
        "size_bytes": 46_139_885_414,
    },
}

# Deliberately absent: ltx-2.3-22b-dev and ltx-2.3-22b-distilled-1.1. Both sit on the 3090 and
# no pose names either -- adding them would invite a pod to spend 20 minutes fetching 46 GB
# for a base model nothing renders on. An entry here is a statement that the file is worth a
# cold pod's time.


def canonical_name(name: str) -> str:
    """Strip the extension, matching how recipes store a checkpoint and how the daemon
    reports one. Both spellings reach this module, and they mean the same file."""
    n = (name or "").strip()
    return n[: -len(".safetensors")] if n.endswith(".safetensors") else n


def source_for(name: str) -> dict[str, str | int] | None:
    """Where to fetch `name`, or None if this checkpoint has no recorded source.

    None is a real answer, not an error: a checkpoint can exist on a box without this repo
    knowing where it came from, and the caller should route to a worker that holds it rather
    than invent a download.
    """
    return CHECKPOINT_SOURCES.get(canonical_name(name))
