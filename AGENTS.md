# Wanly API

FastAPI backend for the Wanly video generation system.

## Purpose

- Job and segment management
- S3 file storage for videos/images
- Video stitching (ffmpeg)
- PostgreSQL database with Alembic migrations
- REST API for daemon workers and console frontend

## Key Models

- **Job**: Top-level video generation request
- **Segment**: Individual ~5s video segment (chained via last frame)
- **Video**: Final stitched output
- **Lora**: LoRA library entries
- **PromptPreset**: Saved prompt templates

## Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /jobs` | Create new job |
| `POST /segments` | Add segment to job |
| `POST /segments/{id}/claim` | Daemon claims segment |
| `POST /segments/{id}/caption/retry`, `POST /segments/{id}/caption/skip` | Caption hold (console#562): a segment whose `<SCENE>`/`<MOTION>` has no saved words for its known start image is created `awaiting_caption` (never claimed) and released to `pending` with the saved ImageMeta text once both halves exist (`<MOTION>` gates whenever motion captioning can run, i.e. `MOTION_CAPTION_ENABLED`, default true; with it off only `<SCENE>` gates). A prompt that is blank once the placeholders are removed is a 422 at submit, and one that resolves to blank or only the trigger phrase is never claimed -- it goes to `caption_failed` and render-without refuses it (console#577). One background waiter per image joins the image's caption ticket if one is in flight, else asks for one; failure, or the captioner refusing for longer than `caption_hold_timeout_s` (time in the queue does not count) -> `caption_failed`, and these two routes (retry / render without) are the way out. `GET /caption-holds` sums it up (jobs waiting, per-image needs and queue place); job list/detail carry `caption_hold_detail` and per-segment `caption_needs`/`caption_queue_*`. See `app/caption_hold.py` |
| `POST /images/scene/describe`, `GET /images/scene/status`, `GET /images/scene/tickets/{id}`, `GET /images/caption-queue` | Async describe (console#564): answers 202 with a caption ticket at once. **Per half since console#590**: body `halves` = `["scene"]`, `["motion"]` or both (default both, for old clients); one ticket per half in its own lane (`tickets` in the answer), single-flight per image AND half (a second describe, the old sync `POST /images/scene`, and a held job all get the same ticket). Saving one half never clears or overwrites the other; motion is grounded on the SAVED scene (waits for a scene in flight, asks for one if there is none, re-made if a scene redo lands mid-paragraph). Bulk-tag auto-describe makes the scene only. Held jobs ask for exactly the halves they still need, and their tickets carry `requested_by` [{job_id, name}]. Status takes `?half=` and carries `scene`/`motion`; `GET /images/scene` carries `scene_caption`/`motion_caption`. Status: `queued` + position, `running`, `done`, `failed` (+ `busy`). `caption-queue` lists every entry (with kind: describe/hold/dataset/try) and recently finished tickets, so one poll can mark every image on a page. Tickets are in-process: a restart drops queued describes (held jobs are re-swept). See `app/caption_tickets.py` |
| `PATCH /segments/{id}` | Update segment status |
| `POST /segments/{id}/upload` | Upload segment output |
| `POST /videos/{id}/stitch` | Stitch all segments |
| `POST /images/edit` (+ `/preview`, `/faces`, `GET /presets`) | Image Edit tool: face mode (LivePortrait, **no longer called by the console since console#569**; kept until the 2070's face-edit service goes with the card swap) via the face-edit service (`face_edit_url`); saves a NEW image, refuses locked datasets (console#547). Takes a preset, slider values and/or a `prompt` ("describe the change", console#550 -- read by the service's lexicon; answers with the resolved `expression` and `matched_terms`, 422 "nothing to apply" when no term is known). `/faces` lists the faces (source pixels, left to right, `default_index`) -- from an always-on image-edit worker when one is configured and up, else face-edit (console#569); preview/save take `face_box` or `face_index` to edit one that is not the centre-most (console#553) |
| `POST /images/edit` `mode:"full"`, `GET /images/edit/jobs/{id}`, `POST /images/edit/jobs/{id}/save` | Image Edit full mode (console#548): Qwen-Image-Edit on the 3090's image-edit service (`image_edit_url`), an `instruction` or a head angle (`head_preset` / `angle` {yaw, pitch}, image-space: yaw<0 = toward the image's left, pitch>0 = chin up). Returns 202 with a job; `app/full_edit.py` switches the box into edit mode through its control API (waits for the render segment, never interrupts training), runs the queue FIFO, and hands the card back after `image_edit_return_grace_s`. Results are held in memory (with an AuraFace score vs the source) until saved as a NEW image. `GET /images/edit/presets` carries `head_angles` and `expressions`. **Since console#569 every Edit-dialog edit is full mode**: any mix of an angle, an expression preset (`preset`, names in `full_edit.EXPRESSIONS`; the words live in the gpu-docker service) and `instruction`, plus `face_box` (the service crops around that face and pastes it back). A service too old for `expression`/`face_box` (its `/health` `features`) fails the job rather than silently ignoring them. An always-on image-edit worker (`image_edit_standing_url`, empty by default: none) is preferred while healthy and waited for while busy; `image_edit_worker`'s edit mode is the fallback. Which boxes carry image-edit is being re-planned (3090a/3090b as symmetric workers), so no box is named in code |
| `POST /ltx/characters/{id}/sheet/generate`, `GET /ltx/characters/sheet/jobs/{job_id}`, `POST /ltx/characters/{id}/sheet/compose`, `GET /ltx/characters/{id}/sheets`, `GET /ltx/characters/sheet/presets` | Build a character sheet (console#582, #585, `app/sheet_gen.py`): ONE Image Repo photo of her, face + body (`photo_uri`) -- the turnaround's image 1, so her build comes from it, and the face panel is auto-cropped from it -- + `outfit` (required; what she wears in the photo), `hair`, `crop_padding` (default 140 px), N seeds (default 3, max 6). No `body` field: Qwen ignored body words (#585). A job on the **image-edit queue** (`app/full_edit.py`, `kind="sheet"`): same standing-box/edit-mode placement, never interrupts training, the reason in `message`, and it needs `turnaround` and `one_photo` in the service's `/health` features. One `/turnaround` call per seed; each candidate (turnaround, composed 1536x1024 sheet, JPEG preview, face-panel JPEG) is written to `wanly-jobs/sheet-jobs/<id>/` with a `job.json` manifest as it arrives, so a status survives an API restart (marked interrupted). Compose copies the chosen candidate's sheet to `character-sheets/` in the Image Repo, sets `sheet_uri` + `identity_mode='sheet'`, and writes a `character_sheets` provenance row (migration 108: photo, words, prompt, seed, model, settings; migration 109 adds `photo_mode` 'one_photo' and `face_panel_crop`, the crop's provenance -- NULL on rows from before #585). Pairs are refused |
| `POST/PATCH /ltx/characters` (identity) | A character is a LoRA, an identity reference, or both (migration 107, console#581). `sheet_uri` (1536x1024 character sheet) / `face_ref_uri` are Image Repo s3:// URIs; `identity_mode` (`sheet`/`face`) defaults to the sheet. A reference with no LoRA stores `char_lora` NULL and needs no trigger: `<TRIGGER>` fills from `description`, or is dropped (the #577 empty-prompt guards still apply). The claim carries `identity_ref {url (presigned), mode, uri, character}` unless the job's `use_identity_ref` is false (NULL = on). A PAIR has no reference of its own and renders with its FIRST member's, or none. Moving the image in the repo moves the character's URI with it. **A DRAFT has neither** (console#592, migration 110 dropped `ck_ltx_characters_lora_or_ref`): a create with no LoRA, no reference and no trigger stores `char_lora`/`trigger` NULL so Build sheet can make its first sheet; a legacy "none" row with no reference is a draft too. Job create, add segment, render-without-caption (422) and the claim (segment FAILED, job FAILED) refuse a draft with "<name> has no LoRA or sheet yet: build a sheet or attach a LoRA." -- see `segments._draft_refusal` |
| `GET/PUT /settings`, `POST /images/scene/try` | Settings returns the default text of every caption style and the default motion template (`{scene}`, `{style}`, `{#scene}...{/scene}`; see `app/joycaption.py`), and refuses a template with an unknown placeholder (422). `/images/scene/try` runs the saved or unsaved prompts on one image through the caption queue and stores nothing (console#555) |

## Quality Enhancement Features

### Motion Keywords
Segments extract and propagate motion keywords (walking, running, standing, etc.) to improve continuity.

### Reference Frames
Segments track up to 3 previous output frames for multi-frame identity anchoring in PainterLongVideo.

## Database

- PostgreSQL with Alembic migrations
- Key tables: `jobs`, `segments`, `videos`, `loras`, `app_settings`
- Migrations in `alembic/versions/`

## Deployment

- Docker container on EC2
- GitHub Actions workflow: `.github/workflows/deploy.yml`
- Migrations run automatically during deploy

## Related Projects

- `wanly-gpu-daemon`: Worker daemon that runs ComfyUI
- `wanly-console`: React frontend
