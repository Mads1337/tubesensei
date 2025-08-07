"""
Celery tasks for TubeSensei processing pipeline
"""
import logging
import asyncio
from typing import List, UUID, Dict, Any, Optional
from datetime import datetime, timedelta
from celery import Task
from uuid import uuid4

from app.celery_app import celery_app, update_job_status
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.processing_job import ProcessingJob, JobType, JobStatus, JobPriority
from app.models.processing_session import ProcessingSession
from app.models.channel import Channel
from app.models.video import Video
from app.services.video_discovery import VideoDiscovery
from app.services.transcript_processor import TranscriptProcessor
from app.services.channel_manager import ChannelManager
from app.workers.monitoring import TaskMonitor
from app.utils.exceptions import ValidationError, YouTubeAPIError
from app.integrations.transcript_errors import TranscriptError

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with success/failure callbacks"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task executes successfully"""
        try:
            logger.info(f"Task {task_id} completed successfully")
            update_job_status(task_id, "completed", result=retval)
            TaskMonitor.record_task_complete(self.name, retval.get('duration', 0))
        except Exception as e:
            logger.error(f"Error in success callback for task {task_id}: {e}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        try:
            logger.error(f"Task {task_id} failed: {exc}")
            update_job_status(task_id, "failed", error=str(exc))
            TaskMonitor.record_task_fail(self.name, str(exc))
        except Exception as e:
            logger.error(f"Error in failure callback for task {task_id}: {e}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is being retried"""
        try:
            logger.warning(f"Task {task_id} being retried: {exc}")
            update_job_status(task_id, "retrying", error=str(exc))
        except Exception as e:
            logger.error(f"Error in retry callback for task {task_id}: {e}")


@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.processing_tasks.discover_channel_videos_task"
)
def discover_channel_videos_task(
    self,
    channel_id: str,
    force_refresh: bool = False,
    max_videos: int = 500
) -> Dict[str, Any]:
    """
    Discover all videos from a channel.
    
    Args:
        channel_id: Database channel ID (string UUID)
        force_refresh: Force refresh even if recently fetched
        max_videos: Maximum number of videos to discover
    
    Returns:
        Dictionary with discovery results
    """
    start_time = datetime.utcnow()
    TaskMonitor.record_task_start(self.name)
    
    try:
        # Convert string UUID to UUID object
        channel_uuid = UUID(channel_id)
        
        async def _discover():
            async with AsyncSessionLocal() as session:
                discovery_service = VideoDiscovery()
                
                # Get channel
                channel = await session.get(Channel, channel_uuid)
                if not channel:
                    raise ValidationError(f"Channel not found: {channel_id}")
                
                logger.info(f"Discovering videos for channel: {channel.title}")
                
                # Discover videos
                videos = await discovery_service.discover_videos(
                    channel_id=channel_uuid,
                    db=session,
                    max_videos=max_videos,
                    force_refresh=force_refresh
                )
                
                # Queue transcript extraction for videos with captions
                transcript_jobs = []
                for video in videos:
                    if video.has_captions:
                        job_id = str(uuid4())
                        task = extract_transcript_task.delay(str(video.id))
                        transcript_jobs.append({
                            "video_id": str(video.id),
                            "task_id": task.id
                        })
                
                result = {
                    "channel_id": channel_id,
                    "channel_title": channel.title,
                    "videos_discovered": len(videos),
                    "videos_with_captions": len(transcript_jobs),
                    "transcript_jobs_queued": len(transcript_jobs),
                    "task_ids": [job["task_id"] for job in transcript_jobs],
                    "duration": (datetime.utcnow() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }
                
                logger.info(f"Channel discovery complete: {result}")
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_discover())
        loop.close()
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in channel discovery: {e}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API error in channel discovery: {e}")
        # Retry with exponential backoff for API errors
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
    except Exception as exc:
        logger.error(f"Unexpected error in channel discovery: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    name="app.workers.processing_tasks.extract_transcript_task"
)
def extract_transcript_task(
    self,
    video_id: str,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Extract transcript for a single video.
    
    Args:
        video_id: Database video ID (string UUID)
        force_refresh: Force refresh even if transcript exists
    
    Returns:
        Dictionary with extraction results
    """
    start_time = datetime.utcnow()
    TaskMonitor.record_task_start(self.name)
    
    try:
        # Convert string UUID to UUID object
        video_uuid = UUID(video_id)
        
        async def _extract():
            async with AsyncSessionLocal() as session:
                processor = TranscriptProcessor(session=session)
                
                # Get video
                video = await session.get(Video, video_uuid)
                if not video:
                    raise ValidationError(f"Video not found: {video_id}")
                
                logger.info(f"Extracting transcript for video: {video.title}")
                
                # Extract transcript
                transcript = await processor.extract_transcript(
                    video_id=video_uuid,
                    force_refresh=force_refresh,
                    clean_content=True,
                    calculate_metrics=True,
                    save_to_db=True
                )
                
                result = {
                    "video_id": video_id,
                    "video_title": video.title,
                    "success": transcript is not None,
                    "transcript_id": str(transcript.id) if transcript else None,
                    "language": transcript.language if transcript else None,
                    "word_count": len(transcript.content.split()) if transcript else 0,
                    "is_auto_generated": transcript.is_auto_generated if transcript else None,
                    "duration": (datetime.utcnow() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }
                
                logger.info(f"Transcript extraction complete: {result}")
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_extract())
        loop.close()
        
        return result
        
    except TranscriptError as e:
        # Don't retry for non-retryable transcript errors
        if "not available" in str(e).lower() or "disabled" in str(e).lower():
            logger.warning(f"Transcript not available for video {video_id}: {e}")
            return {
                "video_id": video_id,
                "success": False,
                "error": str(e),
                "retryable": False,
                "duration": (datetime.utcnow() - start_time).total_seconds()
            }
        else:
            # Retry for other transcript errors
            raise self.retry(exc=e, countdown=120 * (2 ** self.request.retries))
    except ValidationError as e:
        logger.error(f"Validation error in transcript extraction: {e}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in transcript extraction: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name="app.workers.processing_tasks.batch_process_transcripts_task"
)
def batch_process_transcripts_task(
    self,
    video_ids: List[str],
    session_id: str,
    concurrent_limit: int = None
) -> Dict[str, Any]:
    """
    Process multiple transcripts in batch using subtasks.
    
    Args:
        video_ids: List of database video IDs (string UUIDs)
        session_id: Processing session ID
        concurrent_limit: Maximum concurrent extractions
    
    Returns:
        Dictionary with batch processing results
    """
    start_time = datetime.utcnow()
    TaskMonitor.record_task_start(self.name)
    
    try:
        session_uuid = UUID(session_id)
        concurrent_limit = concurrent_limit or settings.TRANSCRIPT_BATCH_SIZE
        
        logger.info(f"Starting batch transcript processing: {len(video_ids)} videos")
        
        async def _process_batch():
            async with AsyncSessionLocal() as session:
                # Get or create processing session
                proc_session = await session.get(ProcessingSession, session_uuid)
                if not proc_session:
                    raise ValidationError(f"Processing session not found: {session_id}")
                
                # Update session status
                proc_session.status = "running"
                proc_session.started_at = start_time
                proc_session.total_items = len(video_ids)
                await session.commit()
                
                # Create subtasks for each video
                tasks = []
                for video_id in video_ids:
                    task = extract_transcript_task.delay(video_id)
                    tasks.append({
                        "video_id": video_id,
                        "task_id": task.id
                    })
                
                # Wait for all tasks to complete (with timeout)
                successful = 0
                failed = 0
                errors = []
                
                # Note: In a real implementation, you might use Celery's chord
                # or group functionality for better task coordination
                
                # For now, we'll return the task IDs and let monitoring handle tracking
                result = {
                    "session_id": session_id,
                    "total_videos": len(video_ids),
                    "subtasks_created": len(tasks),
                    "task_ids": [task["task_id"] for task in tasks],
                    "status": "processing",
                    "started_at": start_time.isoformat(),
                    "duration": (datetime.utcnow() - start_time).total_seconds()
                }
                
                logger.info(f"Batch processing setup complete: {result}")
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_process_batch())
        loop.close()
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in batch processing: {e}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in batch processing: {exc}")
        raise


@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    name="app.workers.processing_tasks.sync_channel_metadata_task"
)
def sync_channel_metadata_task(
    self,
    channel_id: str
) -> Dict[str, Any]:
    """
    Update channel metadata from YouTube.
    
    Args:
        channel_id: Database channel ID (string UUID)
    
    Returns:
        Dictionary with sync results
    """
    start_time = datetime.utcnow()
    TaskMonitor.record_task_start(self.name)
    
    try:
        channel_uuid = UUID(channel_id)
        
        async def _sync():
            async with AsyncSessionLocal() as session:
                channel_manager = ChannelManager()
                
                # Get channel
                channel = await session.get(Channel, channel_uuid)
                if not channel:
                    raise ValidationError(f"Channel not found: {channel_id}")
                
                logger.info(f"Syncing metadata for channel: {channel.title}")
                
                # Update channel metadata
                updated_channel = await channel_manager.update_channel_metadata(
                    channel_id=channel_uuid,
                    db=session
                )
                
                result = {
                    "channel_id": channel_id,
                    "channel_title": updated_channel.title,
                    "subscriber_count": updated_channel.subscriber_count,
                    "video_count": updated_channel.video_count,
                    "view_count": updated_channel.view_count,
                    "last_updated": updated_channel.updated_at.isoformat() if updated_channel.updated_at else None,
                    "duration": (datetime.utcnow() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }
                
                logger.info(f"Channel metadata sync complete: {result}")
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_sync())
        loop.close()
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in channel sync: {e}")
        raise
    except YouTubeAPIError as e:
        logger.error(f"YouTube API error in channel sync: {e}")
        # Retry with exponential backoff for API errors
        raise self.retry(exc=e, countdown=300 * (2 ** self.request.retries))
    except Exception as exc:
        logger.error(f"Unexpected error in channel sync: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=300 * (2 ** self.request.retries))


@celery_app.task(
    name="app.workers.processing_tasks.cleanup_old_jobs_task"
)
def cleanup_old_jobs_task() -> Dict[str, Any]:
    """
    Cleanup old completed/failed jobs.
    This is a periodic task run via celery-beat.
    
    Returns:
        Dictionary with cleanup results
    """
    start_time = datetime.utcnow()
    
    try:
        async def _cleanup():
            async with AsyncSessionLocal() as session:
                # Delete jobs older than 7 days
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                # Count jobs to be deleted
                from sqlalchemy import select, and_, func
                
                count_query = select(func.count(ProcessingJob.id)).where(
                    and_(
                        ProcessingJob.completed_at < cutoff_date,
                        ProcessingJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
                    )
                )
                result = await session.execute(count_query)
                job_count = result.scalar()
                
                # Delete old jobs
                delete_query = """
                    DELETE FROM processing_jobs 
                    WHERE completed_at < :cutoff_date 
                    AND status IN ('completed', 'failed', 'cancelled')
                """
                await session.execute(delete_query, {"cutoff_date": cutoff_date})
                await session.commit()
                
                result = {
                    "jobs_cleaned": job_count,
                    "cutoff_date": cutoff_date.isoformat(),
                    "duration": (datetime.utcnow() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }
                
                logger.info(f"Job cleanup complete: {result}")
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_cleanup())
        loop.close()
        
        return result
        
    except Exception as exc:
        logger.error(f"Error in job cleanup: {exc}")
        return {
            "jobs_cleaned": 0,
            "error": str(exc),
            "duration": (datetime.utcnow() - start_time).total_seconds()
        }


@celery_app.task(
    base=CallbackTask,
    bind=True,
    name="app.workers.processing_tasks.health_check_task"
)
def health_check_task(self) -> Dict[str, Any]:
    """
    Health check task for monitoring worker status.
    
    Returns:
        Dictionary with health status
    """
    start_time = datetime.utcnow()
    
    try:
        # Basic health checks
        async def _health_check():
            async with AsyncSessionLocal() as session:
                # Test database connection
                await session.execute("SELECT 1")
                
                # Get basic stats
                from sqlalchemy import select, func
                
                # Count pending jobs
                pending_count = await session.execute(
                    select(func.count(ProcessingJob.id)).where(
                        ProcessingJob.status == JobStatus.PENDING
                    )
                )
                pending_jobs = pending_count.scalar()
                
                # Count running jobs
                running_count = await session.execute(
                    select(func.count(ProcessingJob.id)).where(
                        ProcessingJob.status == JobStatus.RUNNING
                    )
                )
                running_jobs = running_count.scalar()
                
                result = {
                    "status": "healthy",
                    "database_connection": "ok",
                    "pending_jobs": pending_jobs,
                    "running_jobs": running_jobs,
                    "worker_id": self.request.hostname,
                    "timestamp": start_time.isoformat(),
                    "duration": (datetime.utcnow() - start_time).total_seconds()
                }
                
                return result
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_health_check())
        loop.close()
        
        return result
        
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "unhealthy",
            "error": str(exc),
            "worker_id": self.request.hostname,
            "timestamp": start_time.isoformat(),
            "duration": (datetime.utcnow() - start_time).total_seconds()
        }