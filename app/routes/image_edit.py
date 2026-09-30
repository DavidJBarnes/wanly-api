"""The Image Edit tool: micro-adjustments to an existing image (wanly-console#547).

    GET  /images/edit/presets   the preset buttons and slider ranges the editor draws
    POST /images/edit/faces     the faces an edit can be pointed at, when there is more than one
    POST /images/edit/preview   run an edit, return a small JPEG, store NOTHING
    POST /images/edit           run it again and save the result as a NEW image

Phase 1 is `mode: "face"` -- LivePortrait via the face-edit service, see app/face_edit.py.

Phase 2 (#548) adds `mode: "full"` -- Qwen-Image-Edit on the 3090, see app/full_edit.py -- on the
same POST /images/edit, answered with a JOB (202) because the 3090 has to finish its render
segment and switch into edit mode first:

    GET  /images/edit/jobs/{id}        state, queue position, why it is waiting; when done a
                                       preview and the AuraFace identity score vs the source
    POST /images/edit/jobs/{id}/save   write the held result as a NEW image (repo or dataset)

A full-mode edit is any mix of a head angle (`head_preset`, or `angle` {yaw, pitch}), an
expression preset (`preset`, one of full_edit.EXPRESSIONS) and free text (`instruction`), with
an optional `face_box` naming one face of several (#569). Since #569 it is the ONLY thing the
Edit dialog sends: every head angle, every expression and the free-text box go to Qwen, and
LivePortrait's routes below stay only because removing them is a separate decision.

WHAT TO APPLY is a preset, slider values, a described change (`prompt`, #550), or a mix. A
prompt is read by the service, not here, so both calls answer with what it resolved to
(`expression`, all twelve axes) and which terms it understood (`source`, `matched_terms`): the
editor moves its sliders there, and the save records numbers rather than a sentence.

WHICH FACE (#553): the node edits the face nearest the horizontal centre unless the request
names another with `face_box` (from /images/edit/faces) or `face_index`. Both are passed through
untouched, in the source's own pixels -- the service gets the source bytes unresized -- and
the service answers with the face it edited.

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
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import face_edit, full_edit, s3
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Dataset
from app.routes.datasets import DATASETS_PREFIX, _prefix, _refuse_if_locked
from app.schemas.image_edit import (
    EditAxis, EditPreset, EditPresets, ExpressionPreset, HeadAngle, HeadAnglePreset, ImageEditFaces,
    ImageEditFacesRequest,
    ImageEditJob, ImageEditJobSave, ImageEditPreview, ImageEditPreviewRequest, ImageEditRequest,
    ImageEditResponse,
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
               preview: bool, body) -> dict:
    try:
        return await face_edit.edit(source, params, prompt=prompt, preview=preview,
                                    face_index=body.face_index, face_box=body.face_box)
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
    """What the Edit dialog draws. Since #569 everything in it runs on Qwen: `expressions` are
    the expression buttons, `head_angles` the angle buttons (all `route: "full"`), and
    `face_limit_deg` is 0 -- which also makes a console from before #569, still open in a tab,
    send every angle to Qwen rather than to LivePortrait. `presets`/`axes` are LivePortrait's,
    kept for the face-mode endpoints that remain."""
    return EditPresets(
        mode="full",
        presets=[EditPreset(name=n, label=label, expression=exp)
                 for n, (label, exp) in face_edit.PRESETS.items()],
        axes=[EditAxis(**a) for a in face_edit.AXES],
        head_angles=[HeadAnglePreset(name=n, label=label, yaw=y, pitch=p, route="full")
                     for n, (label, y, p) in full_edit.HEAD_ANGLES.items()],
        expressions=[ExpressionPreset(name=n, label=label)
                     for n, label in full_edit.EXPRESSIONS.items()],
        face_limit_deg=0, max_yaw=full_edit.MAX_YAW, max_pitch=full_edit.MAX_PITCH,
    )


@router.post("/images/edit/faces", response_model=ImageEditFaces,
             dependencies=[Depends(get_current_user)])
async def edit_faces(body: ImageEditFacesRequest):
    """The faces the editor can point an edit at, left to right, and the one it edits when
    told nothing. The console draws these over the "before" image when there are two or more;
    with one or none it draws nothing, and the editor is unchanged.

    Asked of the standing image-edit service first (#569/#570) -- the one that will do the
    edit -- and of face-edit when there is none. The boxes only have to say where a face is:
    the Qwen edit crops around whichever box it is sent."""
    source = await _fetch(body.source_uri)
    out = await full_edit.faces(source)
    if out is None:
        try:
            out = await face_edit.faces(source)
        except face_edit.FaceEditError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    return ImageEditFaces(
        width=out.get("width") or 0, height=out.get("height") or 0,
        faces=out["faces"], default_index=out.get("default_index"),
    )


@router.post("/images/edit/preview", response_model=ImageEditPreview,
             dependencies=[Depends(get_current_user)])
async def preview_edit(body: ImageEditPreviewRequest):
    """The dialog's "after" pane. Stores nothing."""
    params = _params(body)
    source = await _fetch(body.source_uri)
    t0 = time.monotonic()
    out = await _run(source, params, _prompt(body), preview=True, body=body)
    fmt = out.get("format") or "jpeg"
    applied, full = _applied(params, out)
    return ImageEditPreview(
        image=f"data:image/{fmt};base64,{base64.b64encode(out['image']).decode()}",
        params=applied, expression=full, source=out.get("source"),
        matched_terms=face_edit.matched_terms(out.get("source")),
        width=out.get("width") or 0, height=out.get("height") or 0,
        device=out.get("device"), device_reason=out.get("device_reason"),
        elapsed_ms=round((time.monotonic() - t0) * 1000),
        face_index=out.get("face_index"), face_box=out.get("face_box"),
    )


@router.post("/images/edit", response_model=ImageEditResponse)
async def save_edit(
    body: ImageEditRequest,
    _user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit `source_uri` and save the result as a new image -- in the repo, or in a dataset.

    Full mode (#548) returns 202 with a job instead; see the module docstring."""
    if body.mode == "full":
        return await _submit_full(body)
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
    out = await _run(source, params, prompt, preview=False, body=body)
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

    face = out.get("face_index")
    logger.info("edited %s -> %s (%s%s%s%s) on %s", body.source_uri, uri, tag,
                f" {prompt!r} -> {applied}" if prompt else "",
                f", face {face}" if face is not None else "",
                f", dataset {ds.name}" if ds is not None else "", out.get("device"))
    return ImageEditResponse(
        uri=uri, source_uri=body.source_uri, mode=body.mode, preset=body.preset, prompt=prompt,
        params=applied, expression=full, source=src,
        matched_terms=face_edit.matched_terms(src),
        dataset_id=ds.id if ds is not None else None, device=out.get("device"),
        elapsed_ms=round((time.monotonic() - t0) * 1000),
        face_index=out.get("face_index"), face_box=out.get("face_box"),
    )


# ------------------------------------------------------------------ full mode (#548)


def _full_request(body: ImageEditRequest) -> tuple[dict, str]:
    """(what the image-edit service is sent, the tag for the file name). 422 on nonsense,
    before anything is fetched or queued.

    Any mix of a head angle, an expression preset (`preset`) and an instruction (#569): the
    service writes one prompt from all three. The tag names what was asked for, most specific
    first -- `profile_left-smile`, `smile`, `angle`, `full` (free text) -- so a folder of
    edits reads."""
    instruction = (body.instruction or "").strip() or None
    angle = body.angle
    tags: list[str] = []
    if body.head_preset is not None:
        if body.head_preset not in full_edit.HEAD_ANGLES:
            raise HTTPException(
                422, f"unknown head preset {body.head_preset!r}; known: "
                     f"{', '.join(full_edit.HEAD_ANGLES)}")
        _label, yaw, pitch = full_edit.HEAD_ANGLES[body.head_preset]
        angle = angle or HeadAngle(yaw=yaw, pitch=pitch)
        tags.append(body.head_preset)
    if body.preset is not None and body.preset not in full_edit.EXPRESSIONS:
        raise HTTPException(
            422, f"unknown expression {body.preset!r}; known: "
                 f"{', '.join(full_edit.EXPRESSIONS)}")
    req: dict = {}
    if angle is not None:
        if max(abs(angle.yaw), abs(angle.pitch)) < 5:
            if not (body.preset or instruction):
                raise HTTPException(422, "nothing to apply: a head angle under 5° is not a "
                                         "change")
        else:
            req["angle"] = {"yaw": angle.yaw, "pitch": angle.pitch}
            if not tags:
                tags.append("angle")
    if body.preset is not None:
        req["expression"] = body.preset
        tags.append(body.preset)
    if instruction:
        req["instruction"] = instruction
        if not tags:
            tags.append("full")
    if not req:
        raise HTTPException(422, "nothing to apply: full mode needs a head angle, an "
                                 "expression or an instruction")
    if body.face_box is not None:
        # The chosen face (#553/#569), in the source's pixels: the service crops around it,
        # edits that alone and pastes it back.
        req["face_box"] = list(body.face_box)
    if body.seed is not None:
        req["seed"] = body.seed
    if body.denoise is not None:
        req["denoise"] = body.denoise
    return req, "-".join(tags)


async def _submit_full(body: ImageEditRequest) -> JSONResponse:
    if not settings.image_edit_url:
        raise HTTPException(503, "no image-edit service is configured (image_edit_url is empty)")
    req, tag = _full_request(body)
    source = await _fetch(body.source_uri)
    job = full_edit.queue.submit(body.source_uri, source, req, tag)
    return JSONResponse(status_code=202, content=_job_view(job).model_dump(mode="json"))


def _job_view(job: full_edit.Job) -> ImageEditJob:
    end = job.finished_at or time.time()
    m = job.meta
    return ImageEditJob(
        id=job.id, state=job.state, message=job.message, source_uri=job.source_uri,
        position=full_edit.queue.position(job), tag=job.tag, request=job.request,
        error=job.error, elapsed_s=round(end - job.created_at, 1),
        preview=job.preview if job.state == "done" else None,
        width=m.get("width"), height=m.get("height"), identity=m.get("identity"),
        prompt=m.get("prompt"), seed=m.get("seed"), saved=job.saved, worker=job.worker,
        face_box=job.request.get("face_box"),
    )


def _job_or_404(job_id: str) -> full_edit.Job:
    job = full_edit.queue.get(job_id)
    if job is None:
        raise HTTPException(404, "no such edit job (unsaved results expire, and do not survive "
                                 "an API restart)")
    return job


@router.get("/images/edit/jobs/{job_id}", response_model=ImageEditJob,
            dependencies=[Depends(get_current_user)])
async def edit_job(job_id: str):
    return _job_view(_job_or_404(job_id))


@router.post("/images/edit/jobs/{job_id}/save", response_model=ImageEditResponse)
async def save_edit_job(
    job_id: str,
    body: ImageEditJobSave,
    _user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Write a finished full-mode result as a NEW image. Saving the same job twice writes two
    objects (to two places, usually): the held bytes are not consumed."""
    job = _job_or_404(job_id)
    if job.state != "done" or job.result is None:
        raise HTTPException(409, f"the edit is {job.state}, not done"
                                 + (f": {job.error}" if job.error else ""))
    source_key = _source_key(job.source_uri)
    ds = None
    if body.dataset_id is not None:
        ds = await db.get(Dataset, body.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        await _refuse_if_locked(db, ds)
    key = result_key(source_key, job.tag, ds)
    uri = await asyncio.to_thread(s3.upload_bytes, job.result, key, settings.s3_images_bucket)
    if ds is not None:
        ds.images = list(ds.images) + [uri]
        await db.commit()
    job.saved.append({"uri": uri, "dataset_id": str(ds.id) if ds is not None else None})
    ident = (job.meta.get("identity") or {}).get("aura")
    logger.info("saved full edit %s of %s -> %s (%s, aura %s)%s", job.id, job.source_uri, uri,
                job.tag, ident, f", dataset {ds.name}" if ds is not None else "")
    return ImageEditResponse(
        uri=uri, source_uri=job.source_uri, mode="full",
        preset=job.request.get("expression") or (
            job.tag if job.tag in full_edit.HEAD_ANGLES else None),
        prompt=job.request.get("instruction"), params={}, expression={},
        source=job.meta.get("prompt"), dataset_id=ds.id if ds is not None else None,
        device="cuda", elapsed_ms=round(((job.finished_at or 0) - job.created_at) * 1000),
        face_box=job.request.get("face_box"),
    )
