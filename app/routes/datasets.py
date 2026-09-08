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
from app.schemas.datasets import DatasetCreate, DatasetResponse, DatasetUpdate

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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn a dataset of photographs into a dataset of face crops.

    This is steps 2 and 3 of the documented pipeline -- crop, then gate -- which until now
    existed only as laptop scripts that ssh'd to the box with insightface. A dataset uploaded in
    the console had no way to become face crops, so the whole flow stopped at step one.

    THE GATE IS ON BY DEFAULT and that is the important half. Detection is easy; telling p@y
    from someone else in the same photo set is what hand-culling failed at twice, once into a
    set that had already been culled by eye. Scored against the mean embedding of a reference
    dataset -- or against the crops' own mean when none is given, which proves internal
    consistency and nothing more, and is reported as such.

    Writes a NEW dataset rather than replacing this one. The photographs are the source of truth
    and a crop is derived; overwriting them would make the operation unrepeatable with different
    padding or a different reference.
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

    payload = {
        "images": [base64.b64encode(await asyncio.to_thread(s3.download_bytes, u)).decode()
                   for u in ds.images],
        "reference": ref_embeddings,
        "largest_only": True,
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
    # With no reference, score against the crops' own mean. It cannot tell you the set is the
    # right person -- only that it is internally consistent -- and the note says so.
    if gate and not ref_embeddings and faces:
        own = [f.get("embedding") or [] for f in faces]
        mean = await _mean_via_service(own)
        for f in faces:
            f["cos"] = _cos(f.get("embedding") or [], mean)

    kept, dropped = [], []
    for f in faces:
        if gate and f.get("cos") is not None and f["cos"] < floor:
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
                         f"photos, {len(result.get('no_face', []))} with none detected, "
                         f"{len(dropped)} below the {floor} floor"
                         + ("" if ref_embeddings else
                            " (scored against the crops' own mean — internal consistency only, "
                            "not proof of identity)")))
    db.add(out)
    await db.flush()

    uris = []
    for i, f in enumerate(kept):
        src = ds.images[f["source_index"]].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        key = f"{out.prefix}/{i:03d}_{src}_f{f['face_index']}.png"
        uris.append(await asyncio.to_thread(
            s3.upload_bytes, base64.b64decode(f["png_b64"]), key, settings.s3_images_bucket))
    out.images = uris
    await db.commit()
    await db.refresh(out)
    logger.info("cropped %s -> %s: %d kept, %d dropped below %.2f",
                ds.name, out.name, len(uris), len(dropped), floor)
    return out


async def _embed_all(uris: list[str]) -> list[list[float]]:
    body = {"images": [base64.b64encode(await asyncio.to_thread(s3.download_bytes, u)).decode()
                       for u in uris]}
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
