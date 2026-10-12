"""A dataset's "Upscale" (wanly-api#460): a high-res COPY of chosen stills, beside each one.

David, 2026-10-12: "add Upscale alongside Crop faces ... perform a high res upscale and save a
copy of the image to the images dataset". Not "Fix small faces", which upscales tiny images
IN PLACE and only up to the trainer's 1024 ceiling. This is a manual action: the user picks the
images and 2x or 4x, and each copy is ADDED next to its source, which stays.

The face-crop service does the work (Real-ESRGAN general-x4v3/wdn on CPU, the setting that
keeps skin texture). `target_edge` (wanly-gpu-docker#214) lifts its 1024 cap. The service
echoes the field, so an older one, which would silently cap at 1024, is refused rather than
stored as a "4x" copy.

BOUNDED like the fix (#437): ONE image per service call, its bytes dropped before the next is
fetched, and the whole run under face_size.FIX_LOCK, so the 2 GB API box never holds more than
one image and its result. A 4096 px JPEG is a few MB.

ALL THE WORK FIRST, ONE WRITE AT THE END (as fix_small_faces_job): the copies are uploaded under
a fresh sub-folder with no row lock held, then applied under FOR UPDATE to the set as it is
THEN. A source removed meanwhile gets no copy.
"""
from __future__ import annotations

import asyncio
import base64
import uuid

from app import face_size, lineage
from app.config import settings

#: The largest long edge a copy is made at. The face-crop service's own limit
#: (wanly-gpu-docker#214): one native 4x pass reaches it from 1024 px.
MAX_EDGE = 4096
FACTORS = (2, 4)

#: The runs THIS PROCESS is doing, by dataset id, in DatasetFixStatus's shape. In memory, like
#: FIX_RUNS: a restart ends a run, and nothing is half-applied (the write is at the end).
RUNS: dict[uuid.UUID, dict] = {}

NOT_TARGETED = ("the face-crop service cannot upscale to a chosen size yet — it needs the "
                "worker code updated (wanly-gpu-docker#214)")


def target_edge(width: int, height: int, factor: int) -> int:
    """The long edge `factor`x of this image comes to, capped at MAX_EDGE."""
    return min(MAX_EDGE, factor * max(width, height))


def _progress(ds_id: uuid.UUID, **kw) -> None:
    RUNS.setdefault(ds_id, {}).update(kw)


async def _size(blob: bytes, entry: dict | None) -> tuple[int | None, int | None]:
    """The source's size: from its stored measurement, else a /measure of the bytes in hand."""
    if entry and entry.get("width") and entry.get("height"):
        return entry["width"], entry["height"]
    m = (await face_size.measure_blobs([blob]))[0]
    return m.get("width"), m.get("height")


async def run_upscale(ds_id: uuid.UUID, uris: list[str], factor: int, prefix: str,
                      faces: dict[str, dict]) -> dict:
    """Make the copies; write nothing to the database. Returns {made: {source: copy},
    declined: [source], faces: {copy: entry}, factor}."""
    tag = uuid.uuid4().hex[:6]
    made: dict[str, str] = {}
    declined: list[str] = []
    _progress(ds_id, stage="upscaling", done=0, total=len(uris))
    async with face_size.httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        for i, u in enumerate(uris):
            blob = await asyncio.to_thread(face_size.s3.download_bytes, u)
            w, h = await _size(blob, faces.get(u))
            target = target_edge(w, h, factor) if w and h else None
            if not target or target <= max(w, h):
                # Unreadable, or already at the ceiling: nothing to make.
                declined.append(u)
                del blob
                _progress(ds_id, done=i + 1)
                continue
            r = await client.post(face_size._url("/upscale"), json={
                "images": [base64.b64encode(blob).decode()], "target_edge": target})
            del blob
            if r.status_code == 404:
                raise face_size.FaceCropTooOld(NOT_TARGETED)
            r.raise_for_status()
            body = r.json()
            # THE ECHO CHECK, before anything is stored: an older service ignores target_edge
            # and caps at 1024 -- not the copy that was asked for.
            if body.get("target_edge") != target:
                raise face_size.FaceCropTooOld(NOT_TARGETED)
            res = (body.get("images") or [None])[0]
            del body
            if not (res and res.get("upscaled") and res.get("b64")):
                declined.append(u)   # below the service's MIN_FACTOR, or unreadable
            else:
                stem = u.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                made[u] = await asyncio.to_thread(
                    face_size.s3.upload_bytes, base64.b64decode(res["b64"]),
                    f"{prefix}/upscaled-x{factor}-{tag}/{len(made):03d}_{stem}.jpg",
                    settings.s3_images_bucket)
            del res
            _progress(ds_id, done=i + 1)

    _progress(ds_id, stage="measuring results")
    new_faces = await face_size.measure_uris(list(made.values()))
    return {"made": made, "declined": declined, "faces": new_faces, "factor": factor}


def apply_upscale(ds, result: dict) -> str:
    """Add each copy right after its source (the ORM row, locked by the caller) and return the
    note. A source no longer in the set gets nothing.

    The copy is the same photograph: it takes the source's caption and score. It is NOT marked
    `upscaled_from`, which means "replaced in place" (the fix's bookkeeping, and the console's
    "upscaled" link); its lineage says "upscale" from the source instead."""
    images = list(ds.images or [])
    made = {o: n for o, n in result["made"].items() if o in images and n not in images}
    captions = dict(ds.captions or {})
    scores = dict(ds.scores or {})
    faces = dict(ds.faces or {})
    out: list[str] = []
    for u in images:
        out.append(u)
        if u in made:
            n = made[u]
            out.append(n)
            if u in captions:
                captions[n] = captions[u]
            if u in scores:
                scores[n] = scores[u]
            if n in result["faces"]:
                faces[n] = result["faces"][n]
            lineage.record(ds, n, u, "upscale")
    # JSONB columns do not see in-place mutation: reassign every one.
    ds.images, ds.captions, ds.scores, ds.faces = out, captions, scores, faces
    f = result["factor"]
    note = f"Upscaled {len(made)} image(s) {f}x (high-res copies added beside the originals)."
    if result["declined"]:
        n = len(result["declined"])
        note += (f" {n} {'were' if n != 1 else 'was'} left as is (already at the "
                 f"{MAX_EDGE} px ceiling, or too close to it to be worth a copy).")
    ds.notes = f"{ds.notes}\n{note}".strip() if ds.notes else note
    return note
