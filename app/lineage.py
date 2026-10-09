"""Which image in a dataset was made from which (wanly-api#445).

`datasets.derived` maps a DERIVED image's URI to where it came from:

    {derived_uri: {"from": source_uri, "how": "crop" | "upscale" | "edit" | "fix_crop" |
                   "duplicate", "at": iso8601}}

Keyed by the derived image because that is the one that is in the set: the source may have
left it (a replace-crop removes the photographs; a fix upscales in place) and the dataset page
still says "from <source> (removed)". Before this, crops, upscales and edits sat among 100+
images with nothing tying them to their originals, and choosing between an original and its
crop meant finding both by eye. Only within one set: the same file in two sets has two
histories.
"""
from __future__ import annotations

from datetime import datetime, timezone

HOWS = ("crop", "upscale", "edit", "fix_crop", "duplicate")


def record(ds, derived_uri: str, source_uri: str, how: str, at: str | None = None) -> None:
    """Note that `derived_uri` was made from `source_uri`. Never points an image at itself.
    Reassigned, never mutated in place: JSONB does not see an in-place change."""
    if how not in HOWS:
        raise ValueError(f"unknown lineage {how!r}")
    if not derived_uri or not source_uri or derived_uri == source_uri:
        return
    derived = dict(ds.derived or {})
    derived[derived_uri] = {"from": source_uri, "how": how,
                            "at": at or datetime.now(timezone.utc).isoformat()}
    ds.derived = derived


def prune(ds, also_drop: set[str] | None = None) -> None:
    """Drop entries for derived images that are no longer in the set (or whose content was
    replaced). An entry whose SOURCE left the set stays: that is the "(removed)" note."""
    keep = set(ds.images or []) - (also_drop or set())
    derived = {u: e for u, e in (ds.derived or {}).items() if u in keep}
    if derived != (ds.derived or {}):
        ds.derived = derived
