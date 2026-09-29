"""The Image Edit tool: micro-adjustments to an existing image (wanly-console#547).

    GET  /images/edit/presets   the preset buttons and slider ranges the editor draws
    POST /images/edit/preview   run an edit, return a small JPEG, store NOTHING
    POST /images/edit           run it again and save the result as a NEW image

Phase 1 is `mode: "face"` -- LivePortrait via the face-edit service, see app/face_edit.py.

WHAT TO APPLY is a preset, slider values, a described change (`prompt`, #550), or a mix. A
prompt is read by the service, not here, so both calls answer with what it resolved to
(`expression`, all twelve axes) and which terms it understood (`source`, `matched_terms`): the
editor moves its sliders there, and the save records numbers rather than a sentence.

PREVIEW AND SAVE ARE TWO CALLS, AND THE SAVE RE-RUNS THE EDIT. The dialog previews on every
slider release, and if each of those wrote to the bucket the repo would fill with the drafts of
every edit ever tried. The warp is deterministic -- no seed, no sampler; same input and numbers,
same output -- so re-running at save costs ~1 s and needs no draft store, and the saved bytes
are produced server-side from the source rather than accepted from the browser.

NEVER OVERWRITES. Every result is a new key with a random suffix. A URI a dataset or a finished
training job points at must keep meaning the picture it was created with (#356).

WHERE A RESULT GOES:
  * `dataset_id`   -> `<dataset prefix>/edits/…`, appended to that set's list. A locked set is
                      refused with 409 BEFORE the source is fetched or the service is called,
                      the same place the crop refuses (#356/#358).
  * otherwise      -> the Image Repo: the source's own folder when the source is a repo image,
                      so the edit lists beside its original; today's date folder when the source
                      is a dataset image, because datasets/ is not a repo folder (#464).
"""
import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app import face_edit, s3
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Dataset
from app.routes.datasets import DATASETS_PREFIX, _prefix, _refuse_if_locked
from app.schemas.image_edit import (
    EditAxis, EditPreset, EditPresets, ImageEditPreview, ImageEditPreviewRequest,
    ImageEditRequest, ImageEditResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])


def _source_key(uri: str) -> str:
    """The key of an images-bucket URI, or 400. Only the images bucket: that is where the repo
    and the datasets live, and the only place an edit's source can come from in the console."""
    bucket = settings.s3_images_bucket
    prefix = f"s3://{bucket}/"
    if not uri.startswith(prefix) or len(uri) <= len(prefix):
        raise HTTPException(status_code=400,
                            detail=f"source_uri must be an image in s3://{bucket}/")
    return uri[len(prefix):]


async def _fetch(uri: str) -> bytes:
    _source_key(uri)
    try:
        return await asyncio.to_thread(s3.download_bytes, uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"source image not found: {e}") from e


def _prompt(body) -> str | None:
    """The described change, or None. Whitespace-only is nothing, not a prompt of spaces."""
    return (body.prompt or "").strip() or None


def _params(body) -> dict[str, float]:
    expr = body.expression.model_dump() if body.expression else None
    try:
        return face_edit.resolve(body.preset, expr, _prompt(body))
    except face_edit.FaceEditError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


async def _run(source: bytes, params: dict[str, float], prompt: str | None,
               preview: bool) -> dict:
    try:
        return await face_edit.edit(source, params, prompt=prompt, preview=preview)
    except face_edit.FaceEditError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


def _applied(params: dict[str, float], out: dict) -> tuple[dict[str, float], dict[str, float]]:
    """(params, expression): what was applied, non-zero only and as all twelve axes.

    Taken from the service's answer when it gave one -- for a prompt it is the only place the
    numbers exist. A service too old to report them edited with exactly `params`.
    """
    got = out.get("expression")
    if isinstance(got, dict):
        full = {k: float(got.get(k) or 0) for k in face_edit.AXIS_KEYS}
    else:
        full = {k: float(params.get(k, 0)) for k in face_edit.AXIS_KEYS}
    return {k: v for k, v in full.items() if v}, full


def result_key(source_key: str, tag: str, dataset: Dataset | None = None) -> str:
    """Where a new edit is written. Always a fresh name -- see the module docstring.

    `sel_008.jpg` edited with "smile" becomes `sel_008_edit-smile_1a2b3c.png`: the original's
    name first so the two sort together, the edit in the middle so a folder of them reads, and
    a random suffix so two edits with the same preset never collide.
    """
    name = source_key.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    filename = f"{stem}_edit-{tag}_{uuid.uuid4().hex[:6]}.png"
    if dataset is not None:
        return f"{dataset.prefix or _prefix(dataset.id)}/edits/{filename}"
    folder = source_key.rsplit("/", 1)[0] if "/" in source_key else ""
    if not folder or folder == DATASETS_PREFIX or folder.startswith(f"{DATASETS_PREFIX}/"):
        folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{folder}/{filename}"


@router.get("/images/edit/presets", response_model=EditPresets,
            dependencies=[Depends(get_current_user)])
async def edit_presets():
    return EditPresets(
        mode="face",
        presets=[EditPreset(name=n, label=label, expression=exp)
                 for n, (label, exp) in face_edit.PRESETS.items()],
        axes=[EditAxis(**a) for a in face_edit.AXES],
    )


@router.post("/images/edit/preview", response_model=ImageEditPreview,
             dependencies=[Depends(get_current_user)])
async def preview_edit(body: ImageEditPreviewRequest):
    """The dialog's "after" pane. Stores nothing."""
    params = _params(body)
    source = await _fetch(body.source_uri)
    t0 = time.monotonic()
    out = await _run(source, params, _prompt(body), preview=True)
    fmt = out.get("format") or "jpeg"
    applied, full = _applied(params, out)
    return ImageEditPreview(
        image=f"data:image/{fmt};base64,{base64.b64encode(out['image']).decode()}",
        params=applied, expression=full, source=out.get("source"),
        matched_terms=face_edit.matched_terms(out.get("source")),
        width=out.get("width") or 0, height=out.get("height") or 0,
        device=out.get("device"), device_reason=out.get("device_reason"),
        elapsed_ms=round((time.monotonic() - t0) * 1000),
    )


@router.post("/images/edit", response_model=ImageEditResponse)
async def save_edit(
    body: ImageEditRequest,
    _user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit `source_uri` and save the result as a new image -- in the repo, or in a dataset."""
    params = _params(body)
    source_key = _source_key(body.source_uri)

    ds = None
    if body.dataset_id is not None:
        ds = await db.get(Dataset, body.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        # Before the fetch and the service call: a refused save must not have spent an edit,
        # and must not have written an object nobody will ever see.
        await _refuse_if_locked(db, ds)

    source = await _fetch(body.source_uri)
    t0 = time.monotonic()
    prompt = _prompt(body)
    out = await _run(source, params, prompt, preview=False)
    if (out.get("format") or "png") != "png":
        # A saved edit is lossless by contract; a service that answered otherwise is wrong,
        # and saving its JPEG under a .png name would be worse.
        raise HTTPException(status_code=502,
                            detail=f"face-edit returned {out.get('format')} for a save, not png")

    applied, full = _applied(params, out)
    src = out.get("source")
    tag = body.preset or ("prompt" if (src or "").startswith("prompt:") else "custom")
    key = result_key(source_key, tag, ds)
    uri = await asyncio.to_thread(s3.upload_bytes, out["image"], key, settings.s3_images_bucket)

    if ds is not None:
        # Re-checked after the edit: it takes seconds, and a lock or a training run can land in
        # between. The object is already written, but under a fresh key no set lists, so a
        # refusal here strands one unlisted file rather than changing a locked set.
        await db.refresh(ds)
        await _refuse_if_locked(db, ds)
        # Reassigned, not appended in place: JSONB does not see an in-place mutation.
        ds.images = list(ds.images) + [uri]
        await db.commit()

    logger.info("edited %s -> %s (%s%s%s) on %s", body.source_uri, uri, tag,
                f" {prompt!r} -> {applied}" if prompt else "",
                f", dataset {ds.name}" if ds is not None else "", out.get("device"))
    return ImageEditResponse(
        uri=uri, source_uri=body.source_uri, mode=body.mode, preset=body.preset, prompt=prompt,
        params=applied, expression=full, source=src,
        matched_terms=face_edit.matched_terms(src),
        dataset_id=ds.id if ds is not None else None, device=out.get("device"),
        elapsed_ms=round((time.monotonic() - t0) * 1000),
    )
