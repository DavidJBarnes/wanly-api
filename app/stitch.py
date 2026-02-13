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
from app.models import Job, Segment, Video
from app.s3 import download_file, upload_file

logger = logging.getLogger(__name__)


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
            job.status = "finalizing"
            await db.commit()

            # Fetch completed segments ordered by index
            result = await db.execute(
                select(Segment)
                .where(Segment.job_id == job_id, Segment.status == "completed")
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

                # Write ffmpeg concat list
                concat_list = tmppath / "concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{name}'" for name in local_files)
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
            video.status = "completed"
            video.completed_at = datetime.now(timezone.utc)
            job.status = "finalized"
            await db.commit()
            logger.info("Stitch complete for job %s -> %s", job_id, s3_uri)

        except Exception as e:
            logger.exception("Stitch failed for job %s: %s", job_id, e)
            try:
                await db.rollback()
                video = await db.get(Video, video_id)
                job = await db.get(Job, job_id)
                if video:
                    video.status = "failed"
                    video.error_message = str(e)[:2000]
                if job:
                    job.status = "finalized"
                await db.commit()
            except Exception:
                logger.exception("Failed to record stitch error for job %s", job_id)
