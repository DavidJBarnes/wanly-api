"""Prepare a rendered face clip as an AR face loop (wanly-api#495, wanly-console#742).

Ported from the hand-run tools in wanly-console/tools/ar-face (score_clips.py,
make_face_loop.py), which produced the loops David judged "very good" on a mannequin.

Steps, all CPU (MediaPipe Face Landmarker), sized for the t3.small this runs on:
1. One landmark pass over the render, keeping only per-frame numbers (never the frames).
2. Pick the best WINDOW_S window: frontal, large, steady, and CALM. Jaw/mouth travel is
   penalised, since a moving jaw reads as the whole face moving on a static mannequin.
3. Cut a square crop around the face over that window, ping-pong it (loops with no jump),
   encode `loop.mp4` (H.264, 768²). Crops are held JPEG-compressed, not as raw frames.
4. Landmarks on the exact encoded frames: per-frame UVs, and per-frame 3D shape aligned to
   frame 0 on expression-free anchors (removes the clip's own head motion), then a circular
   temporal Gaussian on both (removes landmark flicker).
5. Write `face.json`, `face.bin`, `poster.jpg`.

The pure maths (window scoring, alignment, smoothing) has no MediaPipe dependency and is
unit-tested; `prepare()` imports MediaPipe lazily.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

WINDOW_S = 3.0
SIZE = 768  # source crops are ~850-900 px; 512 threw away detail David judged on
FACE_MM = 190.0  # forehead-to-chin, for reporting travel in rough millimetres
MODEL_PATH = os.environ.get("FACE_LANDMARKER_MODEL", "/app/models/face_landmarker.task")
HAIR_MODEL_PATH = os.environ.get("HAIR_SEGMENTER_MODEL", "/app/models/hair_segmenter.tflite")
# Full head (console#753): MediaPipe's multiclass selfie segmenter (background, hair, body-skin,
# face-skin, clothes, others) gives the whole head down to the collar.
HEAD_MODEL_PATH = os.environ.get("HEAD_SEGMENTER_MODEL", "/app/models/selfie_multiclass_256x256.tflite")
HEAD_CLASSES = (1, 2, 3)  # hair, body-skin (neck), face-skin; not clothes, background, others
# Hair (console#751): a wider square of the same render frames, cut out by MediaPipe's hair
# segmenter, worn on a head shell in AR. HAIR_SPAN = the square's side in face-box sides. On
# today's renders the square is clipped to the whole 1024² frame; 1024 keeps it at full
# resolution, since the hair modes draw the FACE from this video too (console#753).
HAIR_SIZE = 1024
HAIR_SPAN = 3.0

# Expression-free landmarks (eye corners, nose bridge, forehead, cheekbones, face sides):
# aligning on these removes head motion without flattening the expression.
ANCHORS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 1, 10, 151, 9, 8, 116, 345, 123, 352, 234, 454]
# Chin, mouth corners, lips: how much the expression moves the jaw/mouth.
EXPRESSION = [152, 61, 291, 13, 14]
SMOOTH_SIGMA = 1.2
# wanly-api#505: the headset's video player stalls a few frames when a file wraps back to its
# start, so each file holds REPEATS ping-pong cycles (the data stays one cycle; the player
# indexes frames modulo it). And a window whose ends are still turns round softly instead of
# bouncing: END_FRAMES at each end are scored for speed, END_MM_S is the speed that halves it.
REPEATS = 3
END_FRAMES = 4
END_MM_S = 10.0


@dataclass
class FrameInfo:
    face_px: float  # forehead-to-chin, px; 0 = no face
    yaw: float
    pitch: float
    expr: np.ndarray = field(default_factory=lambda: np.zeros((len(EXPRESSION), 2)))  # / face height
    box: tuple[float, float, float, float] = (0, 0, 0, 0)  # landmark bbox, px


def euler(m: np.ndarray) -> tuple[float, float]:
    """Yaw, pitch (degrees) from MediaPipe's facial transformation matrix."""
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def window_score(frames: list[FrameInfo]) -> tuple[float, float]:
    """(score, expression travel mm) for one window. 0 if any frame lost the face."""
    if any(f.face_px == 0 for f in frames):
        return 0.0, float("inf")
    px = np.array([f.face_px for f in frames])
    yaw = np.array([f.yaw for f in frames])
    pitch = np.array([f.pitch for f in frames])
    ex = np.stack([f.expr for f in frames])  # n x 5 x 2
    travel = float(np.linalg.norm(ex - ex.mean(0), axis=2).max() * FACE_MM)
    frontal = float(((np.abs(yaw) < 20) & (np.abs(pitch) < 20)).mean())
    score = px.mean() * frontal / (1 + yaw.std() / 5 + pitch.std() / 5) / (1 + travel / 4)
    return float(score) / (1 + end_speed(frames) / END_MM_S), travel


def end_speed(frames: list[FrameInfo], fps: float = 24.0) -> float:
    """Expression speed (rough mm/s) over the first and last END_FRAMES — the ping-pong turns
    round there, so motion at full speed reads as a bounce."""
    if len(frames) < 2 * END_FRAMES:
        return 0.0
    ex = np.stack([f.expr for f in frames]) * FACE_MM
    step = np.linalg.norm(np.diff(ex, axis=0), axis=2).max(axis=1)  # worst point, per frame
    return float(max(step[:END_FRAMES - 1].mean(), step[-(END_FRAMES - 1):].mean()) * fps)


def best_window(frames: list[FrameInfo], fps: float, window_s: float = WINDOW_S, step: int = 2):
    """(start frame, end frame exclusive, score, travel mm) of the best window, or None."""
    n = max(2, int(round(window_s * fps)))
    best = None
    for s in range(0, len(frames) - n + 1, step):
        score, travel = window_score(frames[s:s + n])
        if score > 0 and (best is None or score > best[2]):
            best = (s, s + n, score, travel)
    return best


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, R, t) taking src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(d0.T @ s0 / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    sc = float(np.trace(np.diag(sv) @ d) / (s0 ** 2).sum(axis=1).mean())
    return sc, r, mu_d - sc * r @ mu_s


def stabilise(shapes: np.ndarray) -> tuple[np.ndarray, float]:
    """Align every frame's 3D landmarks (frames x N x 3) to frame 0 on ANCHORS.

    Returns the aligned shapes, centred on frame 0's anchor centroid, and the mean anchor
    residual (px) — how much non-rigid motion the anchors still had.
    """
    ref = shapes[0]
    out = []
    for p in shapes:
        sc, r, t = umeyama(p[ANCHORS], ref[ANCHORS])
        out.append((sc * (r @ p.T)).T + t)
    out = np.stack(out)
    resid = float(np.mean([np.linalg.norm(q[ANCHORS] - ref[ANCHORS], axis=1).mean() for q in out]))
    return out - ref[ANCHORS].mean(0), resid


def stabilise_transforms(shapes: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """The per-frame (scale, R, t) stabilise() applies (frame -> frame 0, before centring)."""
    ref = shapes[0]
    return [umeyama(p[ANCHORS], ref[ANCHORS]) for p in shapes]


def hair_window(boxes: list[tuple[float, float, float, float]], w: int, h: int, span: float = HAIR_SPAN):
    """Square (x, y, side) for the hair: the face box's union, `span` times its side, raised so
    the crown is in, clipped to the frame (kept square)."""
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x1 - x0, y1 - y0) * span, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.1 * side
    hx = int(np.clip(cx - side / 2, 0, w - side))
    hy = int(np.clip(cy - side / 2, 0, h - side))
    return hx, hy, int(side)


def hair_uv_rows(sc: float, r: np.ndarray, t: np.ndarray, centre: np.ndarray, face_box: tuple[int, int, int],
                 hair_box: tuple[int, int, int], size: int) -> np.ndarray:
    """2x4 affine taking a STABILISED head point q (the face.bin space) to that frame's hair-image
    coordinates (u right, v DOWN, 0..1 of the hair square).

    stabilised q = sc R p + t - centre, so p = M q + b with M = Rᵀ/sc, b = M (centre - t); p is
    the frame's own landmark space (x = px right, y = -px down, of the size² face crop), which
    maps to render px as (sx + p.x k, sy - p.y k), k = side/size; then into the hair square.
    """
    sx, sy, side = face_box
    hx, hy, hside = hair_box
    m = r.T / sc
    b = m @ (centre - t)
    k = side / size / hside
    row_u = np.append(k * m[0], (sx - hx) / hside + k * b[0])
    row_v = np.append(-k * m[1], (sy - hy) / hside - k * b[1])
    return np.stack([row_u, row_v]).astype(np.float32)


def repeat_file(path: str, repeats: int = REPEATS) -> None:
    """Rewrite an encoded loop as `repeats` back-to-back copies (stream copy, no re-encode)."""
    if repeats <= 1:
        return
    tmp = path + ".one.mp4"
    os.replace(path, tmp)
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-stream_loop", str(repeats - 1), "-i", tmp,
                        "-c", "copy", "-movflags", "+faststart", path], check=True, timeout=600)
    finally:
        os.remove(tmp)


def smooth_circular(a: np.ndarray, sigma: float = SMOOTH_SIGMA, radius: int = 3) -> np.ndarray:
    """Gaussian along axis 0, wrapping — a ping-pong loop is circular."""
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return sum(w * np.roll(a, s, axis=0) for s, w in zip(range(-radius, radius + 1), k)).astype(np.float32)


def pingpong(n: int) -> list[int]:
    """Frame order f0..fN-1, fN-2..f1: loops back to f0 with no jump."""
    return list(range(n)) + list(range(n - 2, 0, -1))


def square_box(boxes: list[tuple[float, float, float, float]], w: int, h: int, margin: float = 1.25):
    """One square crop (x, y, side) covering every box, with margin, clipped to the frame."""
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = max(x1 - x0, y1 - y0) * margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    side = int(min(side, w - sx, h - sy))
    return sx, sy, side


def _codec_args() -> list[str]:
    """libx264 (the API image's Debian ffmpeg has it). FACE_LOOP_VCODEC overrides for a dev box
    whose ffmpeg lacks it (Fedora's: libopenh264), at a fixed generous bitrate."""
    codec = os.environ.get("FACE_LOOP_VCODEC", "libx264")
    return ["-c:v", codec, "-crf", "18"] if codec == "libx264" else ["-c:v", codec, "-b:v", "6M"]


@dataclass
class Prepared:
    out_dir: str
    window: tuple[float, float]  # seconds in the render
    frames: int
    fps: float
    stats: dict


def prepare(video_path: str, out_dir: str, size: int = SIZE) -> Prepared:
    """Run the whole preparation; files land in out_dir. Raises ValueError if no usable window."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    def detector(matrices: bool):
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.VIDEO,
            num_faces=1, output_facial_transformation_matrixes=matrices))

    def mp_image(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # 1. landmark pass, numbers only
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    infos: list[FrameInfo] = []
    det = detector(True)
    try:
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            r = det.detect_for_video(mp_image(fr), int(i * 1000 / fps))
            i += 1
            if not r.face_landmarks:
                infos.append(FrameInfo(0, 99, 99))
                continue
            p = r.face_landmarks[0]
            fh = math.hypot((p[152].x - p[10].x) * w, (p[152].y - p[10].y) * h)
            yaw, pitch = euler(r.facial_transformation_matrixes[0])
            ex = np.array([((p[k].x - p[1].x) * w / fh, (p[k].y - p[1].y) * h / fh) for k in EXPRESSION])
            xs, ys = [q.x * w for q in p], [q.y * h for q in p]
            infos.append(FrameInfo(fh, yaw, pitch, ex, (min(xs), min(ys), max(xs), max(ys))))
    finally:
        det.close()
        cap.release()

    # 2. window
    win = best_window(infos, fps)
    if win is None:
        raise ValueError("no stretch of the render keeps a detectable face for the whole window")
    s0, s1, _, travel = win
    sx, sy, side = square_box([f.box for f in infos[s0:s1]], w, h)
    hx, hy, hside = hair_window([f.box for f in infos[s0:s1]], w, h)

    # 3. crops (JPEG-compressed in memory), ping-pong, encode
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    crops: list[bytes] = []
    hair_crops: list[bytes] = []
    for _ in range(s1 - s0):
        ok, fr = cap.read()
        if not ok:
            break
        c = cv2.resize(fr[sy:sy + side, sx:sx + side], (size, size), interpolation=cv2.INTER_AREA)
        crops.append(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        hc = cv2.resize(fr[hy:hy + hside, hx:hx + hside], (HAIR_SIZE, HAIR_SIZE), interpolation=cv2.INTER_AREA)
        hair_crops.append(cv2.imencode(".jpg", hc, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    cap.release()
    order = pingpong(len(crops))
    os.makedirs(out_dir, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size}x{size}",
         "-r", str(fps), "-i", "-", *_codec_args(), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", os.path.join(out_dir, "loop.mp4")],
        stdin=subprocess.PIPE)
    for k in order:
        enc.stdin.write(cv2.imdecode(np.frombuffer(crops[k], np.uint8), cv2.IMREAD_COLOR).tobytes())
    enc.stdin.close()
    if enc.wait(timeout=600) != 0:
        raise RuntimeError("ffmpeg failed to encode the loop")
    repeat_file(os.path.join(out_dir, "loop.mp4"))

    # 4. landmarks on the encoded frames: UVs + 3D shape
    det = detector(False)
    uvs, shapes = [], []
    try:
        for i, k in enumerate(order):
            img = cv2.imdecode(np.frombuffer(crops[k], np.uint8), cv2.IMREAD_COLOR)
            r = det.detect_for_video(mp_image(img), int(i * 1000 / fps))
            if not r.face_landmarks:
                if not uvs:
                    raise ValueError("face lost in the first loop frame")
                uvs.append(uvs[-1])
                shapes.append(shapes[-1])
                continue
            p = r.face_landmarks[0][:468]
            uvs.append(np.array([(q.x, q.y) for q in p], np.float32))
            # image frame (x right, y down, z away) -> right-handed (x right, y up, z toward camera), px
            shapes.append(np.array([(q.x * size, -q.y * size, -q.z * size) for q in p], np.float32))
    finally:
        det.close()
    raw = np.stack(shapes)
    pos, resid = stabilise(raw)
    pos, uv = smooth_circular(pos), smooth_circular(np.stack(uvs))
    centre = raw[0][ANCHORS].mean(0)
    hair_rows = smooth_circular(np.stack([
        hair_uv_rows(sc, r, t, centre, (sx, sy, side), (hx, hy, hside), size)
        for sc, r, t in stabilise_transforms(raw)]))
    hair_pct, head_pct, crown_clip_pct = write_hair(hair_crops, order, fps, out_dir)

    # 5. files
    with open(os.path.join(out_dir, "face.bin"), "wb") as f:
        f.write(pos.astype(np.float32).tobytes())
        f.write(uv.astype(np.float32).tobytes())
    meta = {
        "fps": fps, "size": size, "frames": len(order), "pingpong": True, "repeats": REPEATS, "vertices": 468,
        "anchors": ANCHORS, "units": f"px of the {size} crop, centred on the anchor centroid",
        "layout": "float32: positions[frames][468][3] then uvs[frames][468][2]",
        "anchorResidualPx": round(resid, 3),
        # console#751: hair.mp4 is HAIR_SIZE wide, colour on top and the masks below (red = hair,
        # green = the whole head, console#753); hair.bin maps a stabilised head point to the hair
        # image per frame.
        "hair": {"video": "hair.mp4", "size": HAIR_SIZE, "layout": "float32: rows[frames][2][4]",
                 "masks": {"r": "hair", "g": "head"},
                 "uv": "u right, v down, 0..1 of the colour square",
                 # a face-crop image point (u, v down) is (a + c u, b + c v) in the hair square,
                 # so the player can draw the face from hair.mp4 too (one video, no sync)
                 "faceToHair": [round((sx - hx) / hside, 6), round((sy - hy) / hside, 6), round(side / hside, 6)]},
    }
    with open(os.path.join(out_dir, "hair.bin"), "wb") as f:
        f.write(hair_rows.astype(np.float32).tobytes())
    with open(os.path.join(out_dir, "face.json"), "w") as f:
        json.dump(meta, f, indent=1)
    cv2.imwrite(os.path.join(out_dir, "poster.jpg"), cv2.imdecode(np.frombuffer(crops[0], np.uint8), cv2.IMREAD_COLOR))

    face_px = float(np.mean([f.face_px for f in infos[s0:s1]]))
    stats = {
        "expression_travel_mm": round(travel, 1),
        "face_px": round(face_px),
        "anchor_residual_px": round(resid, 2),
        "frontal_pct": round(100 * float(np.mean([abs(f.yaw) < 20 and abs(f.pitch) < 20 for f in infos[s0:s1]]))),
        "crop_px": side,
        "hair_pct": hair_pct,
        "head_pct": head_pct,
        "crown_clip_pct": crown_clip_pct,
        "end_speed_mm_s": round(end_speed(infos[s0:s1], fps), 1),
    }
    return Prepared(out_dir, (s0 / fps, s1 / fps), len(order), fps, stats)


def head_mask(hair: np.ndarray, classes: list[np.ndarray]) -> np.ndarray:
    """The whole head (hair + face + neck, not clothes) from the multiclass segmenter's confidence
    masks, never smaller than the hair mask — the hair segmenter has the finer hair edges."""
    head = np.clip(sum(classes[k] for k in HEAD_CLASSES), 0, 1)
    return np.maximum(hair, head)


def smooth_masks(masks: list[np.ndarray], order: list[int], i: int) -> np.ndarray:
    """Loop frame i's mask, smoothed around the (circular) loop so it does not flicker, uint8."""
    import cv2

    n = len(order)
    radius, sigma = 3, SMOOTH_SIGMA
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    weights /= weights.sum()
    m = sum(wt * masks[order[(i + d) % n]].astype(np.float32) for d, wt in zip(range(-radius, radius + 1), weights))
    return cv2.GaussianBlur(m, (0, 0), 1.5).astype(np.uint8)


def crown_clip(head: np.ndarray) -> float:
    """Share of the square's top rows that is head (uint8 mask): hair running off the top edge,
    which the player can only fill by stretching the edge. 0 = the crown is in."""
    return float((head[:3] > 127).mean())


def write_hair(hair_crops: list[bytes], order: list[int], fps: float, out_dir: str) -> tuple[float, float, float]:
    """hair.mp4: each loop frame's hair square on top, its masks below — red = hair (console#751),
    green and blue = the whole head down to the collar (console#753). Old loops have a grey mask
    (all three = hair), so `.r` is the hair either way. Masks are segmenter confidence, smoothed
    around the loop. Returns the mean hair and head shares of the square and the crown clip
    (crown_clip), in percent. The frames are written REPEATS times (wanly-api#505)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    def segmenter(path):
        return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=path), output_confidence_masks=True,
            output_category_mask=False))

    def as_u8(m):
        return (np.clip(m, 0, 1) * 255).astype(np.uint8)

    hair_seg, head_seg = segmenter(HAIR_MODEL_PATH), segmenter(HEAD_MODEL_PATH)
    hairs: list[np.ndarray] = []
    heads: list[np.ndarray] = []
    try:
        for jpg in hair_crops:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            shape = img.shape[:2]
            hair = np.asarray(hair_seg.segment(im).confidence_masks[-1].numpy_view(), np.float32).reshape(shape)
            classes = [np.asarray(c.numpy_view(), np.float32).reshape(shape) for c in head_seg.segment(im).confidence_masks]
            hairs.append(as_u8(hair))
            heads.append(as_u8(head_mask(hair, classes)))
    finally:
        hair_seg.close()
        head_seg.close()
    s = HAIR_SIZE
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{s}x{2 * s}",
         "-r", str(fps), "-i", "-", *_codec_args(), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", os.path.join(out_dir, "hair.mp4")],
        stdin=subprocess.PIPE)
    hair_share, head_share, clip = [], [], []
    for i, k in enumerate(order):
        hm, dm = smooth_masks(hairs, order, i), smooth_masks(heads, order, i)
        hair_share.append(float((hm > 127).mean()))
        head_share.append(float((dm > 127).mean()))
        clip.append(crown_clip(dm))
        img = cv2.imdecode(np.frombuffer(hair_crops[k], np.uint8), cv2.IMREAD_COLOR)
        enc.stdin.write(np.vstack([img, np.dstack([dm, dm, hm])]).tobytes())  # BGR: head, head, hair
    enc.stdin.close()
    if enc.wait(timeout=600) != 0:
        raise RuntimeError("ffmpeg failed to encode the hair")
    repeat_file(os.path.join(out_dir, "hair.mp4"))
    return (round(100 * float(np.mean(hair_share)), 1), round(100 * float(np.mean(head_share)), 1),
            round(100 * float(np.mean(clip)), 1))


# --- Start-photo ranking (the Face loops tab's photo picker) ---------------------------------
#
# The loop is only as sharp and still as its start photo: LTX animates FROM it. The hand-run
# pick (console#742) ranked Kelly-2026's 83 training photos by face size, frontality, a closed
# mouth, open eyes and sharpness; real_034 (face 691 px, dead-on, neutral) made the loops David
# called "very good". Same criteria here, so the picker can put the good candidates first.

# Start photos are cropped to a face-centred square (crop_start) and rendered at START_SIZE, so
# the face ends up START_SIZE / START_FACE_SPAN px tall whatever the photo's resolution. A face
# smaller than that in the photo gets upscaled (soft); bigger gives no further benefit.
START_SIZE = 1024
assert START_SIZE % 64 == 0, "LTX two-stage renders need sides divisible by 64"
START_FACE_SPAN = 1.75  # crop side, in face heights (real_034: face 691 px in a 1024 frame)
START_FACE_PX = START_SIZE / START_FACE_SPAN
START_HEAD_ROOM = 0.57  # face centre this far down the crop (0.5 = centred)


def photo_score(face_px: float, yaw: float, pitch: float, roll: float, mouth: float, eyes: float, sharp: float) -> tuple[float, bool]:
    """(score, suitable). mouth/eyes are openings as a fraction of face height. Face size counts
    up to what the crop can use (START_FACE_PX); past that, sharpness and pose decide."""
    suitable = mouth < 0.03 and eyes > 0.035 and abs(yaw) < 12 and abs(pitch) < 18 and abs(roll) < 10
    usable = min(face_px, START_FACE_PX)
    score = (usable / 400) * max(0.0, 1 - abs(yaw) / 25) * max(0.0, 1 - abs(pitch) / 40) * min(1.0, sharp / 100)
    return score * (1.0 if suitable else 0.2), suitable


def rate_photo(image_bytes: bytes, detector) -> dict | None:
    """Rate one photo with an IMAGE-mode FaceLandmarker (with transformation matrices).
    None if it holds no detectable face."""
    import cv2
    import mediapipe as mp

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    r = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return None
    p = r.face_landmarks[0]

    def pt(i):
        return np.array([p[i].x * w, p[i].y * h])

    fh = float(np.linalg.norm(pt(10) - pt(152)))
    mouth = float(np.linalg.norm(pt(13) - pt(14)) / fh)
    eyes = float((np.linalg.norm(pt(159) - pt(145)) + np.linalg.norm(pt(386) - pt(374))) / 2 / fh)
    m = np.asarray(r.facial_transformation_matrixes[0])
    yaw, pitch = euler(m)
    roll = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    xs, ys = [q.x * w for q in p], [q.y * h for q in p]
    sub = img[max(0, int(min(ys))):int(max(ys)), max(0, int(min(xs))):int(max(xs))]
    sharp = float(cv2.Laplacian(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if sub.size else 0.0
    score, suitable = photo_score(fh, yaw, pitch, roll, mouth, eyes, sharp)
    return {"score": round(score, 3), "suitable": suitable, "face_px": round(fh), "yaw": round(yaw, 1),
            "pitch": round(pitch, 1), "mouth_closed": mouth < 0.03, "eyes_open": eyes > 0.035}


def photo_detector():
    from mediapipe.tasks.python import BaseOptions, vision

    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), num_faces=1, output_facial_transformation_matrixes=True))


def crop_start(image_bytes: bytes, detector) -> tuple[bytes, int, int]:
    """Crop a start photo to a face-centred square, START_FACE_SPAN face heights a side, resized to
    exactly START_SIZE² (PNG). The render size derives from the start image (~1 MP), so an uncropped
    4000x3000 phone photo rendered the face at ~460 px; cropped, any good photo gives ~585+.
    Raises ValueError if there is no face."""
    import cv2
    import mediapipe as mp

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("start image could not be decoded")
    h, w = img.shape[:2]
    r = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        raise ValueError("no face found in the start photo")
    p = r.face_landmarks[0]
    top, chin = np.array([p[10].x * w, p[10].y * h]), np.array([p[152].x * w, p[152].y * h])
    fh = float(np.linalg.norm(chin - top))
    c = (top + chin) / 2
    side = int(min(fh * START_FACE_SPAN, w, h))
    x0 = int(np.clip(c[0] - side / 2, 0, w - side))
    # 0.57, not 0.48 (console#751): the crop's top sits ~0.5 face heights above the forehead
    # landmark, so the whole head (crown included) is in the render for the hair.
    y0 = int(np.clip(c[1] - side * START_HEAD_ROOM, 0, h - side))
    crop = img[y0:y0 + side, x0:x0 + side]
    # Always exactly START_SIZE²: the render size is the start image's, and LTX's two-stage graph
    # (spatial upsampler) needs both sides divisible by 64. A smaller crop (831 px) was refused.
    if side != START_SIZE:
        interp = cv2.INTER_AREA if side > START_SIZE else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (START_SIZE, START_SIZE), interpolation=interp)
    ok, png = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("could not encode the cropped start photo")
    return png.tobytes(), crop.shape[1], crop.shape[0]
