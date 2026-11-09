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

# Expression-free landmarks (eye corners, nose bridge, forehead, cheekbones, face sides):
# aligning on these removes head motion without flattening the expression.
ANCHORS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 1, 10, 151, 9, 8, 116, 345, 123, 352, 234, 454]
# Chin, mouth corners, lips: how much the expression moves the jaw/mouth.
EXPRESSION = [152, 61, 291, 13, 14]
SMOOTH_SIGMA = 1.2


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
    return float(score), travel


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

    # 3. crops (JPEG-compressed in memory), ping-pong, encode
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    crops: list[bytes] = []
    for _ in range(s1 - s0):
        ok, fr = cap.read()
        if not ok:
            break
        c = cv2.resize(fr[sy:sy + side, sx:sx + side], (size, size), interpolation=cv2.INTER_AREA)
        crops.append(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
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
    pos, resid = stabilise(np.stack(shapes))
    pos, uv = smooth_circular(pos), smooth_circular(np.stack(uvs))

    # 5. files
    with open(os.path.join(out_dir, "face.bin"), "wb") as f:
        f.write(pos.astype(np.float32).tobytes())
        f.write(uv.astype(np.float32).tobytes())
    meta = {
        "fps": fps, "size": size, "frames": len(order), "pingpong": True, "vertices": 468,
        "anchors": ANCHORS, "units": f"px of the {size} crop, centred on the anchor centroid",
        "layout": "float32: positions[frames][468][3] then uvs[frames][468][2]",
        "anchorResidualPx": round(resid, 3),
    }
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
    }
    return Prepared(out_dir, (s0 / fps, s1 / fps), len(order), fps, stats)
