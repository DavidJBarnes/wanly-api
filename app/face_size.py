"""Face size at training size, and "Fix small faces" (wanly-api#432).

WHY. Both still recipes train with `bucket_no_upscale`: a photo is scaled DOWN to the trainer's
1024^2 area and a small image is never enlarged. So the size a face is learned at is its height
after that, not in the photograph -- and on Joana v3, 29 of 48 faces were under 250 px by that
measure, and none was ever seen large. v4 added head-and-shoulders crops of those shots and
upscaled the tiny close-ups, and reached v3-e11 likeness in about half the steps (#431).

This module is the API side of making that routine instead of a session at a terminal:

  measure           each still's face height at training size, from the face-crop service's
                    /measure (wanly-gpu-docker#206), stored on the set as `faces`
  the warning       the training preflight says "N of M images show the face under 250 px"
  Fix small faces   head-and-shoulders crop + Real-ESRGAN upscale of each flagged photo,
                    ADDED beside it; and tiny whole images upscaled IN PLACE of themselves

THE ORIGINALS ARE NEVER TOUCHED IN S3. An upscaled image is a new object under a new key; the
set's list points at it instead. A run that trained on the original keeps meaning those bytes
(#421), and the original is still in the bucket if the upscale turns out worse.

AN OLDER FACE-CROP SERVICE IS REFUSED, NOT TRUSTED. It ignores `upscale` and returns plain
crops, and has no /measure at all. Storing either as if it had worked would put wrong numbers
on the badges and un-upscaled crops in the set, both silently. So /health's `features` is
checked before a fix starts, the crop response's `upscale` echo is checked before anything is
stored, and a 404 from /measure is reported as "needs the worker image", the same way the
head-and-shoulders framing is checked.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid

import httpx

from app import clips, s3
from app.config import settings

logger = logging.getLogger(__name__)

#: ONE FIX AT A TIME, process-wide (#437) -- the fix's counterpart of _MEASURE_LOCK (#434).
#: The API runs on a 2 GB box with one uvicorn process; two sets fixed at once is two sets of
#: downloads, crops and upscales in flight. A second fix WAITS its turn holding no DB
#: connection and no image bytes. Its own lock, not _MEASURE_LOCK: a fix measures before and
#: after its work, and one lock across both would deadlock it.
FIX_LOCK = asyncio.Lock()

#: How many images go to the face-crop service per call. Each call is a full base64 copy of
#: every image in it, and the service works through them one by one on CPU, so a whole set in
#: one request is a ~300 MB body against a 300 s read timeout. Measuring is ~1 s an image;
#: an upscaled crop is a few seconds more.
MEASURE_CHUNK = 16
FIX_CHUNK = 4

TOO_OLD = ("the face-crop service cannot {what} yet — it needs the worker image updated "
           "(wanly-gpu-docker#206)")


#: A composition fix needs framing="pair" (wanly-gpu-docker#208). Code-only on the worker:
#: the face-crop service picks it up on a restart, no new image needed.
NO_PAIR = ("the face-crop service cannot make two-person crops yet — it needs restarting on "
           "wanly-gpu-docker#208 or later")


class FaceCropTooOld(Exception):
    """The deployed face-crop service predates #206. The message says what it needs."""


def _url(path: str) -> str:
    return f"{settings.face_crop_url.rstrip('/')}{path}"


def is_small(entry: dict | None, pair: bool = False) -> bool:
    """Measured, a face was found, and it trains under the small-face line. No face is a
    different problem -- the anchor scores already catch it -- and not counted here.

    `pair` (a composition set, #436): judged by `pair_px`, the SMALLER of the two largest
    faces -- both people must be learnable, and the pair LoRA learns the smaller one small.
    A photo with fewer than two faces has none and is not counted: cropping can't make it a
    pair photo, and the fix reports it instead."""
    if not entry:
        return False
    px = entry.get("pair_px" if pair else "face_px")
    return px is not None and px < settings.small_face_px


def needs_measure(entry: dict | None, pair: bool = False) -> bool:
    """Unmeasured -- or, on a composition set, measured before #436 kept the second face: no
    `boxes` (entry_from always writes a list, empty for no face; pair_px can be a real None,
    "fewer than two faces", so it cannot be the marker). Re-measuring those is how an existing
    pair set gets the numbers it is now judged by, with no migration. The console's
    faceSizeSummary uses the same marker, so its "unmeasured" and this agree."""
    return entry is None or (pair and entry.get("boxes") is None)


def is_fixed(entry: dict | None, images) -> bool:
    """The crop "Fix small faces" made of this photo is still in the set. The photo itself never
    changes -- its face is small forever -- so without this a set that had just been fixed still
    warned "12 of 45 small" and offered the button again. The same rule plan_fix uses to skip
    it: its crop removed, it counts as small (and is offered) again."""
    return bool(entry and entry.get("crop_uri") and entry["crop_uri"] in images)


def is_small_image(entry: dict | None) -> bool:
    """Short side under the line: a close-up too small for a crop to help. Upscaled whole."""
    if not entry or not entry.get("width") or not entry.get("height"):
        return False
    return min(entry["width"], entry["height"]) < settings.small_image_short_side


def stills(images: list[str]) -> list[str]:
    """Measurement is of stills. A clip's face is measured by nobody yet (#411)."""
    return [u for u in images if not clips.is_clip(u)]


def entry_from(result: dict | None) -> dict:
    """One /measure result, as stored per URI. The LARGEST face is taken as the subject's --
    the same rule the crop's face 0 follows, and the same caveat: in a two-person photo the
    bigger face may be the other person's. `faces` says how many there were."""
    if not result:
        return {"width": None, "height": None, "face_px": None, "faces": 0, "pair_px": None,
                "boxes": []}
    faces = result.get("faces") or []
    top = faces[0] if faces else {}
    # THE PAIR (#436): the two largest faces' boxes (source pixels), and the smaller of their
    # training-size heights -- what a composition set is judged by. Kept for every set: it is
    # two numbers, and a set's kind is not this function's business.
    two = faces[:2]
    pair_px = (min(f.get("face_px_at_train") or 0.0 for f in two) if len(two) == 2 else None)
    return {
        "width": result.get("width"),
        "height": result.get("height"),
        "face_px": top.get("face_px_at_train"),
        "face_h": top.get("face_h"),
        "yaw": top.get("yaw"),
        "pitch": top.get("pitch"),
        "roll": top.get("roll"),
        "det_score": top.get("det_score"),
        "faces": len(faces),
        "pair_px": pair_px,
        "boxes": [f.get("box") for f in two if f.get("box")],
    }


async def features() -> set[str]:
    """What the deployed face-crop service says it can do. Empty for one that predates the
    list -- which is exactly the service that cannot measure or upscale."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(_url("/health"))
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise FaceCropTooOld(f"face-crop unreachable: {e}") from e
    return set(r.json().get("features") or [])


async def measure_blobs(blobs: list[bytes]) -> list[dict]:
    """Stored entries for each image's bytes, in order, via /measure in chunks."""
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        for i in range(0, len(blobs), MEASURE_CHUNK):
            body = {"images": [base64.b64encode(b).decode() for b in blobs[i:i + MEASURE_CHUNK]]}
            try:
                r = await client.post(_url("/measure"), json=body)
            except httpx.HTTPError as e:
                raise FaceCropTooOld(f"face-crop unreachable: {e}") from e
            # 404 is an older service with no /measure: say what it needs, not "not found".
            if r.status_code == 404:
                raise FaceCropTooOld(TOO_OLD.format(what="measure face size"))
            r.raise_for_status()
            out.extend(entry_from(res) for res in r.json()["results"])
    return out


#: ONE MEASURING JOB AT A TIME, process-wide (wanly-api#434). The Datasets page asks every card
#: to measure on sight; after the living-datasets backfill that was ~1,000 unmeasured stills
#: across a dozen cards at once, and each request pulled ALL of its set's bytes into memory
#: together. On the 2 GB API box that thrashed it into a hang (2026-10-08). Requests now queue
#: here; each takes its turn.
_MEASURE_LOCK = asyncio.Lock()


async def measure_uris(uris: list[str]) -> dict[str, dict]:
    """{uri: entry} for these stills.

    BOUNDED (wanly-api#434): downloaded and measured one MEASURE_CHUNK at a time, and each
    chunk's bytes are dropped before the next is fetched, so memory holds one chunk -- never a
    whole set -- and only one set measures at a time (_MEASURE_LOCK).

    An image that cannot be downloaded (deleted under the set: the 409 dialog's dead entry) is
    left out rather than failing the rest -- it simply stays unmeasured."""
    if not uris:
        return {}
    out: dict[str, dict] = {}
    async with _MEASURE_LOCK:
        for i in range(0, len(uris), MEASURE_CHUNK):
            chunk = uris[i:i + MEASURE_CHUNK]
            got = await asyncio.gather(
                *(asyncio.to_thread(s3.download_bytes, u) for u in chunk),
                return_exceptions=True)
            ok = [(u, b) for u, b in zip(chunk, got) if isinstance(b, (bytes, bytearray))]
            for u, b in zip(chunk, got):
                if not isinstance(b, (bytes, bytearray)):
                    logger.warning("face size: could not download %s (%s); not measured", u, b)
            del got
            if ok:
                entries = await measure_blobs([b for _, b in ok])
                out.update({u: e for (u, _), e in zip(ok, entries)})
            del ok
    return out


# ---------------------------------------------------------------------------------------
# Fix small faces
# ---------------------------------------------------------------------------------------

#: The fix runs THIS PROCESS is doing, by dataset id. In memory, like captioning's: a run is
#: a task in this process and a restart ends it. Nothing is written to the set until the end,
#: so an interrupted run leaves the set as it was -- and pressing the button again starts over.
FIX_RUNS: dict[uuid.UUID, dict] = {}


def plan_fix(images: list[str], faces: dict[str, dict],
             pair: bool = False) -> tuple[list[str], list[str]]:
    """(whole images to upscale, photographs to crop) from the measurements.

    A SMALL IMAGE IS UPSCALED WHOLE, not cropped: a 300 px close-up is already all face, and
    cropping it only makes a smaller image to enlarge. A BIG PHOTO WITH A SMALL FACE IS
    CROPPED: the face is there at full resolution, the trainer just shrinks the frame around
    it. The two are exclusive, smallest-image rule first.

    IDEMPOTENT. A photo whose crop from an earlier fix is still in the set is not cropped again
    -- the photo itself stays small-faced forever, so without this every press added another
    copy. An upscaled image is ~1024 px and never qualifies a second time.

    `pair` (a composition set, #436): the crops are TWO-PERSON crops, and "small" is the
    smaller face of the pair (is_small). A photo with fewer than two faces is never cropped --
    see single_face_photos -- though a tiny one is still upscaled whole, which keeps whoever
    is in it and changes nothing about what the caption claims.
    """
    present = set(images)
    upscale, crop = [], []
    for u in stills(images):
        e = faces.get(u)
        if not e:
            continue
        if is_small_image(e):
            upscale.append(u)
        elif is_small(e, pair) and not is_fixed(e, present):
            crop.append(u)
    return upscale, crop


def single_face_photos(images: list[str], faces: dict[str, dict]) -> list[str]:
    """On a composition set (#436): measured stills where the detector found fewer than two
    faces, and that are not tiny (those are upscaled whole). Not cropped -- a one-face crop
    under a two-person caption teaches the pair LoRA one face is both people (#430) -- but
    named in the fix's summary, because a pair set's photo with one face is worth a look."""
    out = []
    for u in stills(images):
        e = faces.get(u)
        if e and (e.get("faces") or 0) < 2 and not is_small_image(e):
            out.append(u)
    return out


def _progress(ds_id: uuid.UUID, **kw) -> None:
    FIX_RUNS.setdefault(ds_id, {}).update(kw)


async def _upscale_chunk(client, chunk: list[str], prefix: str, tag: str,
                         replaced: dict[str, str]) -> None:
    """Upscale one chunk of tiny whole images and upload each, recording it in `replaced`.
    Its own function, so one chunk's bytes are gone before the next is downloaded (#437) --
    what measure_uris does with `del` (#434)."""
    blobs = await asyncio.gather(*(asyncio.to_thread(s3.download_bytes, u) for u in chunk))
    r = await client.post(_url("/upscale"),
                          json={"images": [base64.b64encode(b).decode() for b in blobs]})
    if r.status_code == 404:
        raise FaceCropTooOld(TOO_OLD.format(what="upscale images"))
    r.raise_for_status()
    for u, res in zip(chunk, r.json()["images"]):
        if res and res.get("upscaled") and res.get("b64"):
            stem = u.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            replaced[u] = await asyncio.to_thread(
                s3.upload_bytes, base64.b64decode(res["b64"]),
                f"{prefix}/upscaled-{tag}/{len(replaced):03d}_{stem}.jpg",
                settings.s3_images_bucket)


async def _crop_chunk(client, chunk: list[str], prefix: str, tag: str, anchor_vec: list[float],
                      added: dict[str, str], crop_cos: dict[str, float],
                      framing: str = "head_shoulders",
                      unpaired: list[str] | None = None) -> None:
    """Head-and-shoulders (or, with framing="pair", two-person) crop + upscale of one chunk of
    photos, each crop uploaded and recorded in `added` (and scored into `crop_cos`). One chunk
    held, as above. A pair photo the service found fewer than two faces in goes to `unpaired`
    -- the measurement said two, the crop's detection disagreed; nothing is made of it."""
    blobs = await asyncio.gather(*(asyncio.to_thread(s3.download_bytes, u) for u in chunk))
    r = await client.post(_url("/crop"), json={
        "images": [base64.b64encode(b).decode() for b in blobs],
        "reference": [], "largest_only": True,
        "framing": framing, "upscale": True,
    })
    r.raise_for_status()
    result = r.json()
    # THE ECHO CHECK, before anything is stored: an older service ignores both fields
    # and sends plain face crops, which would join the set as if they were the fix.
    if result.get("framing") != framing or result.get("upscale") is not True:
        raise FaceCropTooOld(NO_PAIR if framing == "pair"
                             else TOO_OLD.format(what="crop with upscale"))
    if unpaired is not None:
        unpaired.extend(chunk[i] for i in result.get("no_face") or [])
    for f in result["faces"]:
        src = chunk[f["source_index"]]
        stem = src.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ext = {"jpeg": "jpg"}.get(str(f.get("format", "png")).lower(), "png")
        crop = await asyncio.to_thread(
            s3.upload_bytes, base64.b64decode(f["png_b64"]),
            f"{prefix}/{'pairs' if framing == 'pair' else 'portraits'}-{tag}/"
            f"{len(added):03d}_{stem}.{ext}",
            settings.s3_images_bucket)
        added[src] = crop
        emb = f.get("embedding") or []
        if anchor_vec and emb:
            crop_cos[crop] = round(sum(a * b for a, b in zip(emb, anchor_vec)), 4)


async def run_fix(ds_id: uuid.UUID, images: list[str], faces: dict[str, dict], prefix: str,
                  anchor_uri: str | None, pair: bool = False) -> dict:
    """Do the work -- every service call and upload -- and return what to apply to the set.
    Writes nothing to the database: the route applies the result under a row lock, against
    the set as it is THEN (an image removed meanwhile is not resurrected by this).

    At most FIX_CHUNK images' bytes are held at once (#437): each chunk is downloaded, sent,
    and its results uploaded inside its own function call. What survives a chunk is only the
    S3 keys and scores. The caller holds FIX_LOCK, so only one fix does this at a time."""
    upscale_uris, crop_uris = plan_fix(images, faces, pair)
    framing = "pair" if pair else "head_shoulders"
    unpaired: list[str] = []
    total = len(upscale_uris) + len(crop_uris)
    _progress(ds_id, stage="upscaling", done=0, total=total)
    # A batch of its own under the set's prefix, so a second fix can never overwrite the
    # first's files while a run records them; the index keeps two same-named sources apart.
    tag = uuid.uuid4().hex[:6]
    replaced: dict[str, str] = {}          # original -> its upscaled copy
    added: dict[str, str] = {}             # original -> its head-and-shoulders crop
    crop_cos: dict[str, float] = {}        # crop -> likeness to the anchor, if there is one
    done = 0

    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        # ---- tiny whole images, upscaled
        for i in range(0, len(upscale_uris), FIX_CHUNK):
            chunk = upscale_uris[i:i + FIX_CHUNK]
            await _upscale_chunk(client, chunk, prefix, tag, replaced)
            done += len(chunk)
            _progress(ds_id, done=done)

        # ---- big photos with small faces: head-and-shoulders, upscaled, added beside them
        _progress(ds_id, stage="cropping")
        anchor_vec: list[float] = []
        if anchor_uri and crop_uris:
            # One /embed of the anchor, so the new crops arrive scored and the set does not
            # fail the training preflight's "not scored" check until someone re-scores it.
            # The crop response already carries each crop's embedding.
            anchor_vec = await _embed_one(client, anchor_uri)
        for i in range(0, len(crop_uris), FIX_CHUNK):
            chunk = crop_uris[i:i + FIX_CHUNK]
            await _crop_chunk(client, chunk, prefix, tag, anchor_vec, added, crop_cos,
                              framing, unpaired if pair else None)
            done += len(chunk)
            _progress(ds_id, done=done)

    # ---- measure what was made, so the badges show the fix
    _progress(ds_id, stage="measuring results")
    new_faces = await measure_uris(list(replaced.values()) + list(added.values()))
    for orig, new in replaced.items():
        new_faces.setdefault(new, {})["upscaled_from"] = orig
    return {"replaced": replaced, "added": added, "crop_cos": crop_cos, "faces": new_faces,
            "pair": pair,
            # Pair sets: photos with fewer than two faces, not cropped. The measurement's say
            # plus any the crop's own detection refused.
            "single_face": (sorted(set(single_face_photos(images, faces)) | set(unpaired))
                            if pair else [])}


async def _embed_one(client, uri: str) -> list[float]:
    blob = await asyncio.to_thread(s3.download_bytes, uri)
    r = await client.post(_url("/embed"), json={"images": [base64.b64encode(blob).decode()]})
    r.raise_for_status()
    return (r.json().get("embeddings") or [[]])[0] or []


def apply_fix(ds, fix: dict) -> str:
    """Apply run_fix's result to the set (the ORM row, already locked by the caller) and
    return the note. Only for images still in the set: one removed while the fix ran stays
    removed, and its crop is not added."""
    replaced = {o: n for o, n in fix["replaced"].items() if o in ds.images}
    added = {o: c for o, c in fix["added"].items() if o in ds.images}
    faces = dict(ds.faces or {})
    # The upscaled copy IS the photograph: same caption, and the same face against the anchor
    # (a re-score moves it a little at most). The original's measurement goes with it.
    captions = dict(ds.captions or {})
    scores = dict(ds.scores or {})
    for orig, new in replaced.items():
        if orig in captions:
            captions[new] = captions.pop(orig)
        if orig in scores:
            scores[new] = scores.pop(orig)
        faces.pop(orig, None)
        if ds.anchor_uri == orig:
            ds.anchor_uri = new
    # A crop is a new image, scored against the anchor from its own embedding; no caption,
    # as with any crop -- the photograph's caption describes a different framing.
    for orig, crop in added.items():
        if crop in fix["crop_cos"]:
            scores[crop] = fix["crop_cos"][crop]
        faces[orig] = {**faces.get(orig, {}), "crop_uri": crop}
    for uri, entry in fix["faces"].items():
        if uri in replaced.values() or uri in added.values():
            faces[uri] = {**faces.get(uri, {}), **entry}
    # JSONB columns do not see in-place mutation: reassign every one.
    ds.images = [replaced.get(u, u) for u in ds.images] + list(added.values())
    ds.captions, ds.scores, ds.faces = captions, scores, faces
    what = "two-person" if fix.get("pair") else "head-and-shoulders"
    note = (f"Fixed small faces: {len(replaced)} small image(s) upscaled in place (originals "
            f"kept in S3), {len(added)} {what} crop(s) added.{single_face_note(fix)}")
    ds.notes = f"{ds.notes}\n{note}".strip() if ds.notes else note
    return note


def single_face_note(fix: dict) -> str:
    """The pair-set report (#436): photos not cropped because fewer than two faces were found.
    Empty on a character set, or when there were none."""
    n = len(fix.get("single_face") or [])
    if not n:
        return ""
    return (f" {n} photo(s) show fewer than two faces and were not cropped (a one-person crop "
            f"would train under the two-person caption).")
