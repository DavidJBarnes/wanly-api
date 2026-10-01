"""Build a character's sheet from ONE photo of her, as an asynchronous job (wanly-console#582,
#585).

    POST /ltx/characters/{id}/sheet/generate   {photo_uri, outfit, hair?, crop_padding?,
                                                count|seeds}
    GET  /ltx/characters/sheet/jobs/{job_id}   state, why it waits, the candidates so far
    POST /ltx/characters/{id}/sheet/compose    {job_id, seed} -> the sheet, saved and set

THE RECIPE lives in the image-edit service (wanly-gpu-docker, `POST /turnaround`), in its
one-photo form (console#585, loras/phase0-2026-10-01/character_sheet_one_input.json): ONE photo
of her -- full body or most of it, in the outfit -- is image 1 of the official
Qwen-Image-Edit-2511's front / side / back full-body turnaround (40 steps at CFG 4), so her build
carries into all three views; and the service composes the 1536x1024 sheet with a face panel
AUTO-CROPPED FROM THAT SAME PHOTO on the left. This module sends the photo, the words, the crop
padding and the seeds and keeps what comes back; the prompt wording is the service's.

NO BODY WORDS. #582 shipped a BODY field ("She has an athletic build."); Qwen ignored it, and a
body photo as image 2 too -- it keeps image 1 and little else. The build is controllable only
by the photo, so the field is gone. Sheets saved before #585 keep their `body` in the table.

ON THE IMAGE-EDIT QUEUE (app/full_edit.py). A candidate is minutes of the same ~20 GB model the
Edit dialog uses, so a sheet job joins that queue: the always-on image-edit worker when there is
one, else the 3090's edit mode -- which lets the segment in flight finish and NEVER interrupts a
training run -- and the job's message says why it waits, exactly as an edit's does.

ONE SEED PER CALL, KEPT AS IT ARRIVES. Each candidate is written to the jobs bucket the moment
the service returns it (the turnaround, the composed sheet, a JPEG preview), with the job's
manifest beside them (sheet-jobs/<id>/job.json). So the console shows candidate 1 while 2 is
drawing; a failure on seed 3 keeps 1 and 2; and an API restart -- which loses the in-memory
queue -- loses no picture: the status is read back from the manifest, marked interrupted.

NOTHING IS THE CHARACTER'S UNTIL COMPOSE. The candidates are drafts in the jobs bucket, not
the Image Repo. Approving one copies its composed sheet into the repo (character-sheets/), sets
the character's sheet_uri and identity_mode='sheet', and writes the provenance row
(models.CharacterSheet: the photo, words, prompt, seed, model, and photo_mode "one_photo" with
how the face panel was cropped from the photo). The previous sheet image, if any, stays in the
repo untouched.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import time
import uuid

from app import full_edit, s3
from app.config import settings

logger = logging.getLogger(__name__)

#: Pre-filled outfit and hair, in the recipe's shape: describe what the photo shows, so the
#: turnaround keeps it. Per pronoun, because the words are the prompt's.
DEFAULTS: dict[str, dict[str, str]] = {
    "female": {"outfit": "the same clothes and shoes that she wears in image 1",
               "hair": "her hair exactly as in image 1"},
    "male": {"outfit": "the same clothes and shoes that he wears in image 1",
             "hair": "his hair exactly as in image 1"},
}

#: The face panel's padding around the detected face, in photo pixels: the tested workflow's
#: CropByBBoxes value (the service's default too; sent explicitly so the job records it).
CROP_PADDING = 140
#: What a job and its saved sheets record as the way the sheet was built (console#585).
PHOTO_MODE = "one_photo"

DEFAULT_COUNT = 3
MAX_COUNT = 6
#: Where the jobs bucket keeps a job's drafts and manifest.
PREFIX = "sheet-jobs"
#: The Image Repo folder an approved sheet is saved into.
SHEETS_FOLDER = "character-sheets"

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class SheetError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def gender_for(character_gender: str | None, override: str | None = None) -> str:
    """The prompt's pronoun: an explicit choice, else the character's registry gender."""
    if override in ("female", "male"):
        return override
    return "male" if character_gender == "man" else "female"


def pick_seeds(count: int | None = None, seeds: list[int] | None = None) -> list[int]:
    """Explicit seeds (deduplicated, order kept), else `count` random ones."""
    if seeds:
        out = list(dict.fromkeys(int(s) for s in seeds))
        if len(out) > MAX_COUNT:
            raise SheetError(422, f"at most {MAX_COUNT} candidates per job")
        return out
    n = DEFAULT_COUNT if count is None else int(count)
    if not 1 <= n <= MAX_COUNT:
        raise SheetError(422, f"between 1 and {MAX_COUNT} candidates per job")
    rng = random.SystemRandom()
    out: list[int] = []
    while len(out) < n:
        s = rng.randrange(2**32)
        if s not in out:
            out.append(s)
    return out


def service_request(outfit: str, hair: str | None, gender: str, subject: str | None,
                    crop_padding: int | None = None) -> dict:
    """What /turnaround is sent besides the photo and the seed. 422 without an outfit."""
    outfit = (outfit or "").strip()
    if not outfit:
        raise SheetError(422, "an outfit is required: describe what she wears in the photo")
    req: dict = {"outfit": outfit, "gender": gender, "score": True,
                 "crop_padding": CROP_PADDING if crop_padding is None else int(crop_padding)}
    for k, v in (("hair", hair), ("subject", subject)):
        v = (v or "").strip()
        if v:
            req[k] = v
    return req


def _key(job_id: str, name: str) -> str:
    return f"{PREFIX}/{job_id}/{name}"


def _uri(job_id: str, name: str) -> str:
    return f"s3://{settings.s3_jobs_bucket}/{_key(job_id, name)}"


# ------------------------------------------------------------------------------ the job


async def submit(character, photo_uri: str, photo: bytes, request: dict,
                 seeds: list[int]) -> full_edit.Job:
    """Queue a sheet job for `character` on the image-edit queue (manifest first, so a status
    can always be read back)."""
    # `one_photo`: an image from before #585 would take the photo as a FACE photo -- the old
    # prompt and a full-height strip for a panel -- and say nothing. Refused instead: re-pin.
    job = full_edit.Job(id=uuid.uuid4().hex, source_uri=photo_uri, request=request, tag="sheet",
                        kind="sheet", work=_work, needs={"turnaround": True, "one_photo": True},
                        on_finish=_finished)
    job.meta.update(character_id=str(character.id), character_name=character.name,
                    seeds=list(seeds), candidates=[])
    await asyncio.to_thread(_write_manifest, job)
    full_edit.queue.submit(photo_uri, photo, request, "sheet", job=job)
    logger.info("sheet job %s for %s: %d seed(s) %s from %s (one photo, crop padding %s)",
                job.id, character.name, len(seeds), seeds, photo_uri,
                request.get("crop_padding"))
    return job


async def _work(job: full_edit.Job, url: str) -> dict:
    """Every seed not yet drawn, one call each; each candidate stored before the next."""
    b64 = base64.b64encode(job.meta["_source"]).decode()
    seeds = job.meta["seeds"]
    for seed in seeds:
        if any(c["seed"] == seed for c in job.meta["candidates"]):
            continue                        # drawn before a fallback or an A1111 wait
        n = len(job.meta["candidates"]) + 1
        job.message = f"candidate {n} of {len(seeds)} on {job.worker}"
        t0 = time.time()
        out = await full_edit.post_service(url, "/turnaround",
                                           {"image": b64, **job.request, "seed": seed})
        cand = await asyncio.to_thread(_store, job.id, seed, out)
        job.meta["candidates"].append(cand)
        await asyncio.to_thread(_write_manifest, job)
        logger.info("sheet job %s: candidate %d/%d (seed %s) in %.0fs, aura %s", job.id, n,
                    len(seeds), seed, time.time() - t0, (cand.get("identity") or {}).get("aura"))
    return {"candidates": job.meta["candidates"]}


def _store(job_id: str, seed: int, out: dict) -> dict:
    """The candidate's images (turnaround, sheet, their previews, the face panel's) into the
    jobs bucket; its record."""
    for k in ("candidate", "sheet"):
        if not out.get(k):
            raise full_edit.FullEditError(502, f"image-edit returned no {k} for seed {seed}")
    turn = s3.upload_bytes(base64.b64decode(out["candidate"]), _key(job_id, f"s{seed}_turnaround.png"),
                           settings.s3_jobs_bucket)
    sheet = s3.upload_bytes(base64.b64decode(out["sheet"]), _key(job_id, f"s{seed}_sheet.png"),
                            settings.s3_jobs_bucket)
    preview = None
    if out.get("sheet_preview"):
        preview = s3.upload_bytes(base64.b64decode(out["sheet_preview"]),
                                  _key(job_id, f"s{seed}_sheet.jpg"), settings.s3_jobs_bucket)
    panel = None
    if out.get("face_panel_preview"):
        panel = s3.upload_bytes(base64.b64decode(out["face_panel_preview"]),
                                _key(job_id, f"s{seed}_face_panel.jpg"), settings.s3_jobs_bucket)
    fp = out.get("face_panel") or {}
    return {
        "seed": seed, "candidate_uri": turn, "sheet_uri": sheet, "preview_uri": preview or sheet,
        "face_panel_preview_uri": panel,
        "face_panel_crop": {k: fp.get(k) for k in ("source", "box", "crop", "padding", "scale",
                                                   "detector", "det_size", "photo_size")}
        if fp else None,
        "prompt": out.get("prompt"), "model": out.get("model"), "files": out.get("files"),
        "settings": out.get("settings"), "steps": out.get("steps"), "cfg": out.get("cfg"),
        "face_panel": fp.get("mode"),
        "face_panel_note": fp.get("note"),
        "identity": out.get("identity"), "width": out.get("sheet_width"),
        "height": out.get("sheet_height"), "timings_ms": out.get("timings_ms"),
        "vram_peak_mib": out.get("vram_peak_mib"),
    }


async def _finished(job: full_edit.Job) -> None:
    await asyncio.to_thread(_write_manifest, job)


def _manifest(job: full_edit.Job) -> dict:
    m = job.meta
    return {
        "id": job.id, "kind": "sheet", "photo_mode": PHOTO_MODE,
        "character_id": m.get("character_id"), "character_name": m.get("character_name"),
        "photo_uri": job.source_uri,
        "request": job.request, "seeds": m.get("seeds", []),
        "candidates": m.get("candidates", []), "state": job.state, "message": job.message,
        "error": job.error, "worker": job.worker, "created_at": job.created_at,
        "started_at": job.started_at, "finished_at": job.finished_at,
        "saved": m.get("saved", []),
    }


def _write_manifest(job: full_edit.Job) -> None:
    s3.upload_bytes(json.dumps(_manifest(job)).encode(), _key(job.id, "job.json"),
                    settings.s3_jobs_bucket)


def _read_manifest(job_id: str) -> dict | None:
    try:
        return json.loads(s3.download_bytes(_uri(job_id, "job.json")))
    except Exception:                               # noqa: BLE001 -- absent is an answer
        return None


async def load(job_id: str) -> tuple[dict, full_edit.Job | None]:
    """(the job's record, the live job if this process holds it). 404 if neither exists.

    A manifest whose job is not live and never finished was cut off by an API restart: it is
    reported failed, with whatever candidates it had already kept."""
    if not _JOB_ID.match(job_id or ""):
        raise SheetError(404, "no such sheet job")
    job = full_edit.queue.get(job_id)
    if job is not None and job.kind == "sheet":
        return _manifest(job), job
    rec = await asyncio.to_thread(_read_manifest, job_id)
    if rec is None:
        raise SheetError(404, "no such sheet job")
    if rec.get("state") not in ("done", "failed"):
        rec["state"] = "failed"
        rec["error"] = rec["message"] = (
            "interrupted: the API restarted while this job ran; the candidates it had "
            "already made are kept")
    return rec, None


def photo_uri(rec: dict) -> str | None:
    """The photo a job was built from. Manifests from before #585 named it face_uri."""
    return rec.get("photo_uri") or rec.get("face_uri")


def view(rec: dict, job: full_edit.Job | None) -> dict:
    """The status the console polls."""
    end = rec.get("finished_at") or time.time()
    return {
        "id": rec["id"], "state": rec["state"], "message": rec.get("message") or rec["state"],
        "character_id": rec.get("character_id"), "character_name": rec.get("character_name"),
        "photo_uri": photo_uri(rec), "request": rec.get("request") or {},
        "seeds": rec.get("seeds") or [], "candidates": rec.get("candidates") or [],
        "position": full_edit.queue.position(job) if job is not None else None,
        "error": rec.get("error"), "worker": rec.get("worker"),
        "elapsed_s": round(end - (rec.get("created_at") or end), 1),
        "saved": rec.get("saved") or [],
    }


def candidate(rec: dict, seed: int) -> dict:
    for c in rec.get("candidates") or []:
        if int(c["seed"]) == int(seed):
            return c
    raise SheetError(404, f"no candidate with seed {seed} in this job")


def sheet_key(character_name: str, seed: int) -> str:
    """Where an approved sheet lands in the repo: a fresh name, never an overwrite."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", character_name).strip("-") or "character"
    return f"{SHEETS_FOLDER}/{slug}_sheet_s{seed}_{uuid.uuid4().hex[:6]}.png"


async def record_saved(job_id: str, job: full_edit.Job | None, rec: dict, entry: dict) -> None:
    """Note an approval on the job, so the console can say which candidate became the sheet."""
    if job is not None:
        job.meta.setdefault("saved", []).append(entry)
        await asyncio.to_thread(_write_manifest, job)
        return
    rec = dict(rec, saved=list(rec.get("saved") or []) + [entry])
    await asyncio.to_thread(s3.upload_bytes, json.dumps(rec).encode(), _key(job_id, "job.json"),
                            settings.s3_jobs_bucket)
