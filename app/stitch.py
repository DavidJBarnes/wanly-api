import asyncio
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.enums import JobStatus, SegmentStatus, VideoStatus
from app.models import Job, Segment, Video
from app.s3 import download_file, upload_file

logger = logging.getLogger(__name__)

FADE_DURATION = 1.0
FLASH_DURATION = 1.0
CROSS_DISSOLVE_FRAMES = 12


def _apply_fades(input_path: str, output_path: str, duration: float,
                 fade_in: bool, fade_out: bool) -> None:
    """Re-encode a segment with fade-in and/or fade-out filters."""
    vfilters = []
    if fade_in:
        vfilters.append(f"fade=t=in:st=0:d={FADE_DURATION}")
    if fade_out:
        fade_start = max(duration - FADE_DURATION, 0)
        vfilters.append(f"fade=t=out:st={fade_start}:d={FADE_DURATION}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", ",".join(vfilters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fade failed: {proc.stderr.decode()[-500:]}")


def _generate_black(output_path: str, duration: float, width: int, height: int, fps: int) -> None:
    """Generate a short black video clip."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg black gen failed: {proc.stderr.decode()[-500:]}")


def _apply_trim(input_path: str, output_path: str, fps: int,
                 trim_start_frames: int, trim_end_frames: int) -> float:
    """Re-encode a segment with frames trimmed from start/end. Returns new duration."""
    # Get actual duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True, timeout=60,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr.decode()[-500:]}")
    actual_duration = float(probe.stdout.decode().strip())

    start_time = trim_start_frames / fps
    end_trim_time = trim_end_frames / fps
    new_duration = actual_duration - start_time - end_trim_time
    if new_duration <= 0:
        raise ValueError(f"Trim removes entire video: start={start_time}s end_trim={end_trim_time}s duration={actual_duration}s")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(new_duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {proc.stderr.decode()[-500:]}")
    return new_duration


def _compute_fades(segments: list) -> list[tuple[bool, bool]]:
    """Return (fade_in, fade_out) for each segment based on transition settings.

    A segment's transition controls what happens *after* it (fade-out on itself,
    fade-in on the next segment). The last segment can also fade out (fade to black).
    """
    n = len(segments)
    result = [(False, False)] * n
    for i in range(n):
        fade_in = False
        fade_out = False
        if i > 0 and segments[i - 1].transition == "fade":
            fade_in = True
        if segments[i].transition == "fade":
            fade_out = True
        result[i] = (fade_in, fade_out)
    return result


def _apply_cross_dissolve(seg_a_path: str, seg_b_path: str, output_a_path: str, fps: int, num_frames: int) -> None:
    """Apply cross-dissolve blend between end of seg_a and start of seg_b.
    
    Args:
        seg_a_path: Path to segment A video
        seg_b_path: Path to segment B video  
        output_a_path: Output path for modified segment A (with end blend baked in)
        fps: Frames per second for accurate timing
        num_frames: Number of frames to blend (e.g., 12)
    """
    blend_duration = num_frames / fps
    
    cmd = [
        "ffmpeg", "-y",
        "-i", seg_a_path,
        "-i", seg_b_path,
        "-filter_complex", f"[0:v][1:v]blend=all_mode=normal:shortest=0:repeatlast=0:blend_start={int(num_frames)}[outv]",
        "-map", "[outv]",
        "-t", f"{blend_duration}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        output_a_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cross-dissolve failed: {proc.stderr.decode()[-500:]}")


def _extract_end_frames(video_path: str, output_path: str, num_frames: int) -> None:
    """Extract the last N frames from a video as a separate video file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"select='gte(n\\,{num_frames})',setpts=N/FRAME_RATE/TB",
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract end frames failed: {proc.stderr.decode()[-500:]}")


def _extract_start_frames(video_path: str, output_path: str, num_frames: int) -> None:
    """Extract the first N frames from a video as a separate video file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-frames:v", str(num_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract start frames failed: {proc.stderr.decode()[-500:]}")


async def stitch_video(video_id: UUID, job_id: UUID) -> None:
    """Background task: download segment videos, concat with ffmpeg, upload result."""
    async with async_session() as db:
        try:
            # Set job to finalizing
            job = await db.get(Job, job_id)
            video = await db.get(Video, video_id)
            if not job or not video:
                logger.error("stitch_video: job %s or video %s not found", job_id, video_id)
                return
            job.status = JobStatus.FINALIZING
            await db.commit()

            # Fetch completed segments ordered by index
            result = await db.execute(
                select(Segment)
                .where(Segment.job_id == job_id, Segment.status == SegmentStatus.COMPLETED)
                .order_by(Segment.index)
            )
            segments = result.scalars().all()

            if not segments:
                raise ValueError("No completed segments to stitch")

            missing = [s.index for s in segments if not s.output_path]
            if missing:
                raise ValueError(f"Segments missing output_path: {missing}")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Download each segment video
                local_files = []
                for seg in segments:
                    local_name = f"seg_{seg.index:03d}.mp4"
                    local_path = tmppath / local_name
                    await asyncio.to_thread(download_file, seg.output_path, str(local_path))
                    local_files.append(local_name)

                # Trim pass: apply trim to segments with non-zero trim values
                durations = [seg.duration_seconds for seg in segments]
                for i, seg in enumerate(segments):
                    if seg.trim_start_frames > 0 or seg.trim_end_frames > 0:
                        trimmed_name = f"seg_{seg.index:03d}_trimmed.mp4"
                        new_dur = await asyncio.to_thread(
                            _apply_trim,
                            str(tmppath / local_files[i]),
                            str(tmppath / trimmed_name),
                            job.fps,
                            seg.trim_start_frames,
                            seg.trim_end_frames,
                        )
                        local_files[i] = trimmed_name
                        durations[i] = new_dur

                # Apply transitions
                needs_fade = _compute_fades(segments)
                concat_names = []
                for i, (seg, local_name) in enumerate(zip(segments, local_files)):
                    fade_in, fade_out = needs_fade[i]
                    if fade_in or fade_out:
                        faded_name = f"seg_{seg.index:03d}_faded.mp4"
                        await asyncio.to_thread(
                            _apply_fades,
                            str(tmppath / local_name),
                            str(tmppath / faded_name),
                            durations[i],
                            fade_in,
                            fade_out,
                        )
                        concat_names.append(faded_name)
                    else:
                        concat_names.append(local_name)

                    # Insert black clip for "flash" transition
                    if seg.transition == "flash":
                        black_name = f"black_{seg.index:03d}.mp4"
                        await asyncio.to_thread(
                            _generate_black,
                            str(tmppath / black_name),
                            FLASH_DURATION,
                            job.width,
                            job.height,
                            job.fps,
                        )
                        concat_names.append(black_name)

                # Apply cross-dissolve transitions between consecutive segments
                if len(segments) > 1:
                    processed_files = list(concat_names)
                    for i in range(len(segments) - 1):
                        seg_a = segments[i]
                        seg_b = segments[i + 1]
                        if seg_a.transition == "dissolve" or (seg_a.transition is None and seg_b.transition is None):
                            dissolve_frames = CROSS_DISSOLVE_FRAMES
                            file_a = processed_files[i]
                            file_b = processed_files[i + 1]
                            dissolved_a = f"seg_{seg_a.index:03d}_dissolved.mp4"
                            dissolved_b = f"seg_{seg_b.index:03d}_dissolved.mp4"
                            end_blend = f"seg_{seg_a.index:03d}_end_blend.mp4"
                            start_blend = f"seg_{seg_b.index:03d}_start_blend.mp4"
                            await asyncio.to_thread(
                                _extract_end_frames,
                                str(tmppath / file_a),
                                str(tmppath / end_blend),
                                dissolve_frames,
                            )
                            await asyncio.to_thread(
                                _extract_start_frames,
                                str(tmppath / file_b),
                                str(tmppath / start_blend),
                                dissolve_frames,
                            )
                            await asyncio.to_thread(
                                _apply_cross_dissolve,
                                str(tmppath / end_blend),
                                str(tmppath / start_blend),
                                str(tmppath / dissolved_a),
                                job.fps,
                                dissolve_frames,
                            )
                            blend_duration = dissolve_frames / job.fps
                            trim_a = f"seg_{seg_a.index:03d}_dissolved_trim.mp4"
                            trim_b = f"seg_{seg_b.index:03d}_dissolved_trim.mp4"
                            subprocess.run(
                                [
                                    "ffmpeg", "-y",
                                    "-i", str(tmppath / dissolved_a),
                                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                                    "-t", str(blend_duration),
                                    str(tmppath / trim_a),
                                ],
                                capture_output=True, timeout=120,
                            )
                            subprocess.run(
                                [
                                    "ffmpeg", "-y",
                                    "-ss", str(blend_duration),
                                    "-i", str(tmppath / file_b),
                                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                                    str(tmppath / trim_b),
                                ],
                                capture_output=True, timeout=120,
                            )
                            processed_files[i] = trim_a
                            processed_files[i + 1] = trim_b
                    concat_names = processed_files

                # Write ffmpeg concat list
                concat_list = tmppath / "concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{name}'" for name in concat_names)
                )

                # Run ffmpeg concat (no re-encoding)
                output_path = tmppath / "final.mp4"
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(concat_list),
                        "-c", "copy",
                        str(output_path),
                    ],
                    capture_output=True,
                    timeout=300,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-500:]}")

                # Upload to S3
                s3_key = f"{job_id}/final.mp4"
                s3_uri = await asyncio.to_thread(
                    upload_file, str(output_path), s3_key, settings.s3_jobs_bucket
                )

            # Update video record
            total_duration = sum(durations)
            video.output_path = s3_uri
            video.duration_seconds = total_duration
            video.status = VideoStatus.COMPLETED
            video.completed_at = datetime.now(timezone.utc)
            job.status = JobStatus.FINALIZED
            await db.commit()
            logger.info("Stitch complete for job %s -> %s", job_id, s3_uri)

        except Exception as e:
            logger.exception("Stitch failed for job %s: %s", job_id, e)
            try:
                await db.rollback()
                video = await db.get(Video, video_id)
                job = await db.get(Job, job_id)
                if video:
                    video.status = VideoStatus.FAILED
                    video.error_message = str(e)[:2000]
                if job:
                    job.status = JobStatus.FINALIZED
                await db.commit()
            except Exception:
                logger.exception("Failed to record stitch error for job %s", job_id)
