"""Named, taggable training datasets (wanly-console#453).

WHAT THIS REPLACES. Before it, "these 27 images are p@y v2's training set" could not be written
down. The only groupings were an S3 folder prefix and a per-user favourites list, so every
training run meant re-selecting images by hand and a v2 meant doing it again from memory.

Uploads land in one S3 prefix per dataset, so a dataset is ALSO browsable as a folder in the
Image Repo -- nothing new to learn, and the images stay ordinary images. But the dataset is the
ordered LIST, not the folder listing: it survives an image being moved, and it fixes the order
the trainer stages them in, which is what the captions pair against.
"""
import asyncio
import base64
import logging
import uuid

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.models import Dataset, User
from app.schemas.datasets import (
    DatasetCreate, DatasetResponse, DatasetScore, DatasetScores, DatasetUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _prefix(name: str) -> str:
    """The S3 folder a dataset's uploads go into.

    Spaces become dashes: a prefix with spaces works but is miserable to type in a URL or read
    in a listing, and these become part of every image's key forever.
    """
    return "dataset-" + name.strip().lower().replace(" ", "-")


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    body: DatasetCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    dupe = (await db.execute(select(Dataset).where(Dataset.name == body.name))).scalar_one_or_none()
    if dupe:
        raise HTTPException(status_code=409, detail=f"a dataset called {body.name!r} already exists")
    ds = Dataset(user_id=user.id, name=body.name, tags=body.tags, notes=body.notes,
                 images=[], prefix=_prefix(body.name))
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    # The marker is what makes the prefix show up as a folder in the Image Repo before anything
    # has been uploaded into it. Without it an empty dataset is invisible there.
    await asyncio.to_thread(s3.upload_bytes, b"", f"{ds.prefix}/.folder",
                            settings.s3_images_bucket)
    return ds


@router.get("/datasets", response_model=list[DatasetResponse],
            dependencies=[Depends(verify_api_key_or_bearer)])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Dataset).order_by(Dataset.updated_at.desc()))).scalars().all()
    return list(rows)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def get_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: uuid.UUID,
    body: DatasetUpdate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if body.name is not None:
        ds.name = body.name
        # The prefix deliberately does NOT follow a rename. Renaming a dataset must not move
        # objects that other rows -- a finished training job's dataset_images -- point at.
    if body.tags is not None:
        ds.tags = body.tags
    if body.notes is not None:
        ds.notes = body.notes
    if body.images is not None:
        ds.images = body.images
        # An anchor that was just removed would silently score everything against nothing.
        if ds.anchor_uri and ds.anchor_uri not in body.images:
            ds.anchor_uri = None
    if body.anchor_uri is not None:
        # "" clears it. None means the field was not sent, which must not clear anything --
        # the same distinction every other field here makes.
        ds.anchor_uri = body.anchor_uri or None
    await db.commit()
    await db.refresh(ds)
    return ds


@router.post("/datasets/{dataset_id}/images", response_model=DatasetResponse)
async def add_images(
    dataset_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload images into a dataset. Many at once, because a dataset is 13-50 of them."""
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    added: list[str] = []
    for f in files:
        name = (f.filename or "image.jpg").rsplit("/", 1)[-1]
        ext = ("." + name.rsplit(".", 1)[1].lower()) if "." in name else ".jpg"
        if ext not in IMAGE_SUFFIXES:
            # Skipped rather than fatal: one stray file in a folder drag-and-drop should not
            # reject the other forty-nine.
            logger.info("dataset %s: skipping %s (not an image)", ds.name, name)
            continue
        data = await f.read()
        uri = await asyncio.to_thread(
            s3.upload_bytes, data, f"{ds.prefix}/{name}", settings.s3_images_bucket)
        if uri not in ds.images:
            added.append(uri)

    if added:
        # Reassigned rather than appended in place: JSONB columns do not see a mutation of the
        # existing list, so `ds.images.append(...)` writes nothing and the upload silently
        # vanishes on the next read.
        ds.images = list(ds.images) + added
        await db.commit()
        await db.refresh(ds)
    logger.info("dataset %s: added %d image(s), now %d", ds.name, len(added), len(ds.images))
    return ds


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: uuid.UUID,
    purge: bool = False,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete the dataset row. `purge=true` also deletes its images from S3.

    Not by default: a finished training job records the URIs it trained on, and destroying them
    turns a reproducible run into an unreproducible one.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if purge and ds.prefix:
        await asyncio.to_thread(s3.delete_prefix, settings.s3_images_bucket, ds.prefix + "/")
    await db.delete(ds)
    await db.commit()


@router.post("/datasets/{dataset_id}/crop", response_model=DatasetResponse)
async def crop_faces(
    dataset_id: uuid.UUID,
    reference_dataset_id: uuid.UUID | None = None,
    gate: bool = True,
    largest_only: bool = False,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn a dataset of photographs into a dataset of face crops.

    This is steps 2 and 3 of the documented pipeline -- crop, then gate -- which until now
    existed only as laptop scripts that ssh'd to the box with insightface. A dataset uploaded in
    the console had no way to become face crops, so the whole flow stopped at step one.

    Detection is easy; telling p@y from someone else in the same photo set is what hand-culling
    failed at twice, once into a set that had already been culled by eye. The answer to that is
    a NUMBER PUT IN FRONT OF THE PERSON CULLING -- see POST /datasets/{id}/score -- not a gate
    that deletes on a score nobody looked at. An automatic drop still happens when a known-good
    reference dataset was named, because there the number means something.

    Writes a NEW dataset rather than replacing this one. The photographs are the source of truth
    and a crop is derived; overwriting them would make the operation unrepeatable with different
    padding or a different reference.

    `largest_only` DEFAULTS FALSE: KEEP EVERY FACE.

    It was hardcoded True, which is right only for solo portraits. In a photo of two people
    "largest" is whoever stood closer to the camera, so the output silently interleaves two
    people -- and it throws away the other face, which cannot be recovered without cropping
    again. Keeping everything is the recoverable default: an unwanted crop is one click to
    remove, a missing one is a re-run.

    THE GATE IS OFF unless something meaningful to score against was given. Scoring a mixed set
    against its own mean is not a check -- the mean is a blend of everyone in it -- and dropping
    images on that basis destroys work while looking like diligence. Nominate an anchor
    (`POST /datasets/{id}/score`) and the number means something.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not settings.face_crop_url:
        raise HTTPException(
            status_code=503,
            detail="no face-crop service is configured (face_crop_url is empty)")
    if not ds.images:
        raise HTTPException(status_code=422, detail="this dataset has no images")

    ref_embeddings: list[list[float]] = []
    if reference_dataset_id:
        ref = await db.get(Dataset, reference_dataset_id)
        if not ref:
            raise HTTPException(status_code=404, detail="Reference dataset not found")
        ref_embeddings = await _embed_all(ref.images)

    # CONCURRENTLY. Fetched one at a time this was fourteen serial round trips to S3 before any
    # work started; they are independent and the wait is entirely network.
    blobs = await asyncio.gather(
        *(asyncio.to_thread(s3.download_bytes, u) for u in ds.images))
    payload = {
        "images": [base64.b64encode(b).decode() for b in blobs],
        "reference": ref_embeddings,
        "largest_only": largest_only,
    }
    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        try:
            r = await client.post(f"{settings.face_crop_url.rstrip('/')}/crop", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"face-crop unreachable: {e}") from e
    result = r.json()

    faces = result["faces"]
    floor = result.get("cos_floor", 0.4)

    # NOTHING IS DROPPED WITHOUT A REAL REFERENCE.
    #
    # This used to fall back to the crops' own mean. On a set that still contains two people
    # that mean is a blend of both: it separates neither, and whichever person happens to be in
    # the minority scores lower and gets deleted. That is work destroyed by something that looks
    # like diligence, and it is worse now that every face is kept by default.
    #
    # Culling is a person's job, informed by POST /datasets/{id}/score against an anchor they
    # picked. The gate here is for the case where a known-good set was named.
    gating = gate and bool(ref_embeddings)
    kept, dropped = [], []
    for f in faces:
        if gating and f.get("cos") is not None and f["cos"] < floor:
            dropped.append((ds.images[f["source_index"]], f["cos"]))
        else:
            kept.append(f)
    if not kept:
        raise HTTPException(
            status_code=422,
            detail=f"every crop scored below the {floor} same-person floor — "
                   f"either the reference is wrong or this is not one person")

    name = f"{ds.name} faces"
    if (await db.execute(select(Dataset).where(Dataset.name == name))).scalar_one_or_none():
        name = f"{name} {uuid.uuid4().hex[:4]}"
    out = Dataset(user_id=user.id, name=name, tags=ds.tags, images=[], prefix=_prefix(name),
                  notes=(f"Cropped from {ds.name}: {len(faces)} faces from {len(ds.images)} "
                         f"photos ({'largest only' if largest_only else 'every face'}), "
                         f"{len(result.get('no_face', []))} with none detected, "
                         f"{len(dropped)} below the {floor} floor"
                         + ("" if ref_embeddings else
                            " — nothing dropped, no reference was given. Pick an anchor and "
                            "score to see how alike these are.")))
    db.add(out)
    await db.flush()

    # THE EXTENSION FOLLOWS WHAT THE SERVICE ACTUALLY SENT. It returns JPEG now, capped at the
    # trainer's resolution ceiling, because full-resolution lossless PNG made an 80 MB response
    # that could not cross a home uplink inside the read timeout. `format` is absent on a
    # face-crop that predates that, and the old contract there was PNG -- so the two repos can
    # deploy in either order.
    ext = {"jpeg": "jpg"}.get(str(kept[0].get("format", "png")).lower(), "png")

    # Uploaded concurrently, for the same reason the fetch is: independent, network-bound, and
    # serial round trips are the whole cost.
    async def put(i: int, f: dict) -> str:
        src = ds.images[f["source_index"]].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        key = f"{out.prefix}/{i:03d}_{src}_f{f['face_index']}.{ext}"
        return await asyncio.to_thread(
            s3.upload_bytes, base64.b64decode(f["png_b64"]), key, settings.s3_images_bucket)

    uris = list(await asyncio.gather(*(put(i, f) for i, f in enumerate(kept))))
    out.images = uris
    await db.commit()
    await db.refresh(out)
    logger.info("cropped %s -> %s: %d kept, %d dropped below %.2f",
                ds.name, out.name, len(uris), len(dropped), floor)
    return out


@router.post("/datasets/{dataset_id}/score", response_model=DatasetScores)
async def score_against_anchor(
    dataset_id: uuid.UUID,
    anchor_uri: str | None = None,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Score every image in a set against ONE image in it, nominated as the anchor.

    THE QUESTION THIS ANSWERS IS THE ONLY ONE WORTH ASKING: is this the same person as that.

    The reference machinery that came first scored against a whole dataset's MEAN, and that does
    not survive contact with a real set. A mean over images that still contain two people is a
    blend of both -- it separates neither, and whichever person is in the minority scores lower
    for no reason but being outnumbered. With no reference at all it scored against the crops'
    own mean, which shows they resemble each other and nothing else. One picked face has neither
    problem.

    IT RETURNS NUMBERS; IT DELETES NOTHING. Hand-culling failed twice in this project, but the
    failure was culling with no information, not culling by hand -- an unlabelled thumbnail of a
    stranger at a bad angle looks like a bad photo of the right person. A score beside each
    image fixes that, and leaves the judgement where it belongs. Removing is a separate,
    reversible click.

    The anchor is remembered on the dataset, so re-scoring after a cull compares against the
    same face rather than a moving target.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not settings.face_crop_url:
        raise HTTPException(
            status_code=503,
            detail="no face-crop service is configured (face_crop_url is empty)")

    uri = anchor_uri or ds.anchor_uri
    if not uri:
        raise HTTPException(status_code=422, detail="pick an anchor image first")
    if uri not in ds.images:
        # Most likely it was removed since it was picked. Say which, rather than returning a
        # set of scores quietly measured against nothing.
        raise HTTPException(
            status_code=422,
            detail="the anchor is not in this dataset — it may have been removed; pick another")

    embeddings = await _embed_all(ds.images)
    anchor_vec = embeddings[ds.images.index(uri)]
    if not anchor_vec:
        raise HTTPException(
            status_code=422,
            detail="no face was detected in the anchor image — pick one that is a clear face")

    floor = settings.face_cos_floor
    scores = [
        DatasetScore(
            uri=u,
            # -2.0 is _cos's "one side had no embedding", which is not a low score but an
            # absent one. Surfaced as null so the console can say "no face" rather than
            # rendering it as the worst match in the set.
            cos=None if not e else round(_cos(e, anchor_vec), 4),
            is_anchor=(u == uri),
        )
        for u, e in zip(ds.images, embeddings)
    ]

    # Remembered only once it has been shown to work on this set.
    if ds.anchor_uri != uri:
        ds.anchor_uri = uri
        await db.commit()

    return DatasetScores(anchor_uri=uri, cos_floor=floor, scores=scores)


async def _embed_all(uris: list[str]) -> list[list[float]]:
    blobs = await asyncio.gather(*(asyncio.to_thread(s3.download_bytes, u) for u in uris))
    body = {"images": [base64.b64encode(b).decode() for b in blobs]}
    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        r = await client.post(f"{settings.face_crop_url.rstrip('/')}/embed", json=body)
        r.raise_for_status()
    return r.json()["embeddings"]


async def _mean_via_service(embeddings: list[list[float]]) -> list[float]:
    """The mean is arithmetic, not a model call — done here rather than round-tripping bytes."""
    usable = [e for e in embeddings if e]
    if not usable:
        return []
    n = len(usable[0])
    mean = [sum(e[i] for e in usable) / len(usable) for i in range(n)]
    norm = sum(x * x for x in mean) ** 0.5
    return [x / norm for x in mean] if norm else []


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -2.0
    return sum(x * y for x, y in zip(a, b))
