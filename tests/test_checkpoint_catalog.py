"""Where a checkpoint comes from (console#423).

A worker that lacks a base model fetches it instead of being routed around (#422). The
mapping from a checkpoint NAME to a place the file lives is the one piece of that only this
repo can own -- held in the daemon it would mean redeploying every worker to add a model.

Everything here guards a silent failure. A wrong repo or filename does not error: the worker
downloads a DIFFERENT model under the expected name, ComfyUI loads it, the render succeeds,
and the result is not comparable to anything. That is why entries are verified by byte count
rather than by a name that looks right.
"""
import pathlib
import re

from app.checkpoint_sources import CHECKPOINT_SOURCES, canonical_name, source_for
from app.ltx_stack import LTX_STACK

DOCKER_REPO = pathlib.Path(__file__).parent.parent.parent / "wanly-gpu-docker"


def test_the_stack_default_has_a_source():
    """THE one that matters. Without this, a cold pod handed a default-checkpoint pose can
    neither render it nor fetch it, and simply claims nothing -- which looks like an empty
    queue rather than a missing entry."""
    assert source_for(LTX_STACK["checkpoint"]) is not None, (
        f"the stack default {LTX_STACK['checkpoint']!r} has no entry in CHECKPOINT_SOURCES, "
        f"so no worker can fetch it on demand"
    )


def test_every_entry_is_complete():
    """A half-filled entry is worse than a missing one: the fetcher would build a URL from
    None and 404 inside a claimed segment."""
    for name, src in CHECKPOINT_SOURCES.items():
        assert src.get("repo"), f"{name} has no repo"
        assert src.get("path"), f"{name} has no path"
        assert isinstance(src.get("size_bytes"), int) and src["size_bytes"] > 0, (
            f"{name} has no usable size_bytes — the fetcher cannot verify a truncated "
            f"download without it, and a partial safetensors fails only at load"
        )


def test_paths_carry_the_extension_and_keys_do_not():
    """Recipes store a bare stem; the file on disk has .safetensors. Mixing the two is how a
    lookup silently misses and a worker decides it cannot fetch something it can."""
    for name, src in CHECKPOINT_SOURCES.items():
        assert not name.endswith(".safetensors"), f"key {name!r} should be a bare stem"
        assert src["path"].endswith(".safetensors"), f"{name} path should name a real file"


def test_both_spellings_resolve_to_the_same_entry():
    """The daemon strips the extension before reporting and recipes store it stripped, but
    the engine names files. Both reach this module."""
    for name in CHECKPOINT_SOURCES:
        assert source_for(name) is source_for(f"{name}.safetensors")
    assert canonical_name("x.safetensors") == "x"
    assert canonical_name("  x  ") == "x"


def test_an_unknown_checkpoint_returns_none_rather_than_guessing():
    """None means "route to a worker that holds it". Inventing a URL from the name would
    produce a confident 404 twenty minutes into a boot."""
    assert source_for("ltx-2.3-22b-dev") is None
    assert source_for("") is None
    assert source_for("not_a_real_model") is None


def test_the_catalogue_agrees_with_what_a_cold_pod_stages():
    """The catalogue and download_models.sh must not disagree about the same name.

    If they do, a cold pod stages one file and later fetches a different one under the same
    name. Two different base models sharing a name is the worst case here, because every
    render still succeeds and nothing is comparable afterwards.

    Skips when the sibling checkout is absent, so it does not run in CI -- a known weakness
    (noted in api#262). It is still worth having locally, because the failure it catches is
    invisible at runtime.
    """
    script = DOCKER_REPO / "download_models.sh"
    if not script.exists():
        return
    text = script.read_text()
    rows = re.findall(r'^\s*"([^"]*diffusion_models\|[^"]*)"\s*$', text, re.MULTILINE)
    for row in rows:
        _dest, filename, repo, path = (row.split("|") + [""])[:4]
        stem = canonical_name(filename)
        src = CHECKPOINT_SOURCES.get(stem)
        if src is None:
            raise AssertionError(
                f"download_models.sh stages {stem!r} but the catalogue has no entry — a pod "
                f"that lost the file could not re-fetch it"
            )
        assert src["repo"] == repo, f"{stem}: catalogue says {src['repo']}, script says {repo}"
        assert src["path"] == (path or filename), (
            f"{stem}: catalogue path {src['path']!r} vs script {(path or filename)!r}"
        )
