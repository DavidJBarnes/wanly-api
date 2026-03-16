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


def _compute_fades(segments: list) -> list[tuple[bool, bool]]:
    """Return (fade_in, fade_out) for each segment based on transition settings.

    A segment's transition controls what happens *after* it (fade-out on itself,
    fade-in on the next segment). The last segment's transition is ignored.
    """
    n = len(segments)
    result = [(False, False)] * n
    for i in range(n):
        fade_in = False
        fade_out = False
        if i > 0 and segments[i - 1].transition == "fade":
            fade_in = True
        if i < n - 1 and segments[i].transition == "fade":
            fade_out = True
        result[i] = (fade_in, fade_out)
    return result


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

                # Apply fade transitions where needed
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
                            seg.duration_seconds,
                            fade_in,
                            fade_out,
                        )
                        concat_names.append(faded_name)
                    else:
                        concat_names.append(local_name)

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
            total_duration = sum(s.duration_seconds for s in segments)
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
