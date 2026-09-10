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


#: Every dataset's objects live under this top-level prefix, and the Image Repo does not list
#: it (see `list_folders`). Datasets used to land in a `dataset-<name>/` folder that showed up
#: in the repo beside the generation folders, which made it look as if the two were connected
#: (wanly-console#464). They are not: a dataset is training input, the repo is render input.
DATASETS_PREFIX = "datasets"


def _prefix(ds_id: uuid.UUID) -> str:
    """The S3 folder a dataset's uploads go into.

    Keyed by id, not by name, so a rename is just a rename -- the name was becoming part of
    every image's key forever, which is why renaming was never offered.
    """
    return f"{DATASETS_PREFIX}/{ds_id}"


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    body: DatasetCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    dupe = (await db.execute(select(Dataset).where(Dataset.name == body.name))).scalar_one_or_none()
    if dupe:
        raise HTTPException(status_code=409, detail=f"a dataset called {body.name!r} already exists")
    ds_id = uuid.uuid4()
    ds = Dataset(id=ds_id, user_id=user.id, name=body.name, tags=body.tags, notes=body.notes,
                 images=[], prefix=_prefix(ds_id))
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
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
    if body.name is not None and body.name != ds.name:
        dupe = (await db.execute(
            select(Dataset).where(Dataset.name == body.name, Dataset.id != ds.id)
        )).scalar_one_or_none()
        if dupe:
            raise HTTPException(status_code=409,
                                detail=f"a dataset called {body.name!r} already exists")
        ds.name = body.name
        # The prefix does not follow a rename: it is keyed by id, and moving objects that a
        # finished training job's dataset_images point at would break that job's record.
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
    largest_only: bool = False,
    uris: list[str] | None = None,
    save_as: bool = False,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Replace this dataset's images with the faces cropped out of them -- or, with `save_as`,
    keep the set and append the crops as new images instead.

    IN PLACE by default. It used to write a second dataset called "<name> faces" and leave this
    one as it was, on the argument that the photographs are the source of truth and a crop is
    derived. In use that meant every dataset came in pairs, the one you trained from was
    never the one you named, and the question "why did cropping make a new dataset?" was
    asked on the first try (wanly-console#464). The photographs are still in the bucket under
    the dataset's own prefix, so nothing is destroyed; the dataset simply IS the crops now,
    which is what anyone cropping wanted. `save_as=True` is the other half of that question:
    the set keeps its photographs and the crops join the end of it, so one run produces both.

    SELECTIVE. `uris` crops those and only those; absent means every image in the set. A
    25-image set that needed 5 faces re-cropped cropped all 25 to get them, and the 20 good
    crops came back different -- the output is not deterministic, so a re-run of the whole set
    silently replaced crops nobody asked to change. Selected URIs that are not in the set are
    ignored rather than fatal, the way a stale selection in a long-open dialog survives an
    image being removed in another tab.

    NO GATE, NO REFERENCE. Cropping used to take a reference dataset and drop faces that
    scored below the same-person floor against its mean. A mean over a set that still
    contains two people is a blend of both and separates neither, so the gate was already
    off unless a reference was named, and the dropdown for naming one was the single most
    confusing control on the page. Culling is done afterwards, against ONE anchor face the
    user picks, with the numbers on screen: POST /datasets/{id}/score.

    `largest_only` DEFAULTS FALSE: KEEP EVERY FACE. In a photo of two people "largest" is
    whoever stood closer to the camera. An unwanted crop is one click to remove; a missing
    one is a re-run.
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

    targets = ds.images if uris is None else [u for u in ds.images if u in set(uris)]
    if not targets:
        raise HTTPException(
            status_code=422,
            detail="none of the selected images are in this dataset — they may have been removed")

    # CONCURRENTLY. Fetched one at a time this was fourteen serial round trips to S3 before any
    # work started; they are independent and the wait is entirely network.
    blobs = await asyncio.gather(
        *(asyncio.to_thread(s3.download_bytes, u) for u in targets))
    payload = {
        "images": [base64.b64encode(b).decode() for b in blobs],
        "reference": [],
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
    if not faces:
        raise HTTPException(status_code=422, detail="no faces were detected in any image")

    # THE EXTENSION FOLLOWS WHAT THE SERVICE ACTUALLY SENT. It returns JPEG now, capped at the
    # trainer's resolution ceiling, because full-resolution lossless PNG made an 80 MB response
    # that could not cross a home uplink inside the read timeout. `format` is absent on a
    # face-crop that predates that, and the old contract there was PNG -- so the two repos can
    # deploy in either order.
    ext = {"jpeg": "jpg"}.get(str(faces[0].get("format", "png")).lower(), "png")
    # A crop batch gets its own sub-folder, so — in either mode — cropping twice cannot
    # overwrite the first batch's files while a training job still records them. save_as does
    # NOT mean overwrite the originals with the crop: a URI a finished training job's
    # dataset_images points at must keep meaning the photograph it was created with.
    batch = f"{ds.prefix}/faces-{uuid.uuid4().hex[:6]}"

    # Uploaded concurrently, for the same reason the fetch is: independent, network-bound, and
    # serial round trips are the whole cost.
    async def put(i: int, f: dict) -> str:
        src = targets[f["source_index"]].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        key = f"{batch}/{i:03d}_{src}_f{f['face_index']}.{ext}"
        return await asyncio.to_thread(
            s3.upload_bytes, base64.b64decode(f["png_b64"]), key, settings.s3_images_bucket)

    uris = list(await asyncio.gather(*(put(i, f) for i, f in enumerate(faces))))
    no_face = len(result.get("no_face", []))
    scope = "every image" if uris is None else f"{len(targets)} selected images"
    note = (f"Cropped {len(faces)} faces from {scope} "
            f"({'largest only' if largest_only else 'every face'})"
            + (f", {no_face} with none detected" if no_face else "")
            + ("; the set kept its photos and the crops joined it" if save_as else "") + ".")
    ds.notes = f"{ds.notes}\n{note}".strip() if ds.notes else note
    if save_as:
        # JSONB columns do not see an in-place mutation (see add_images); reassign.
        ds.images = list(ds.images) + uris
        # The anchor is still a photograph in the set; scoring against it still works.
    else:
        replaced = set(targets)
        kept_others = [u for u in ds.images if u not in replaced]
        ds.images = kept_others + uris
        # The anchor was one of the cropped photographs; the set is faces now — but an anchor
        # outside the selection is still what it was.
        if ds.anchor_uri in replaced:
            ds.anchor_uri = None
    await db.commit()
    await db.refresh(ds)
    logger.info("cropped %s (save_as=%s): %d faces from %d of %d photos",
                ds.name, save_as, len(uris), len(targets), len(ds.images) - (len(uris) if save_as else 0))
    return ds


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


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -2.0
    return sum(x * y for x, y in zip(a, b))
