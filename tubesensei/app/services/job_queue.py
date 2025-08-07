"""
Job Queue Service for TubeSensei
Manages Celery task queue operations and job lifecycle
"""
import logging
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from celery.result import AsyncResult

from app.celery_app import celery_app
from app.config import settings
from app.models.processing_job import (
    ProcessingJob, 
    JobType, 
    JobStatus, 
    JobPriority
)
from app.models.processing_session import ProcessingSession, SessionType, SessionStatus
from app.models.video import Video
from app.models.channel import Channel
from app.workers.processing_tasks import (
    discover_channel_videos_task,
    extract_transcript_task,
    batch_process_transcripts_task,
    sync_channel_metadata_task
)
from app.workers.monitoring import TaskMonitor
from app.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class JobQueueService:
    """
    Service for managing job queues and Celery task lifecycle.
    Provides high-level interface for queuing, monitoring, and managing processing jobs.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.celery = celery_app
        self.monitor = TaskMonitor()
    
    async def create_job(
        self,
        job_type: JobType,
        entity_type: str,
        entity_id: UUID,
        priority: JobPriority = JobPriority.NORMAL,
        session_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> ProcessingJob:
        """
        Create a new processing job record.
        
        Args:
            job_type: Type of job to create
            entity_type: Type of entity being processed
            entity_id: ID of the entity being processed
            priority: Job priority level
            session_id: Optional session ID
            metadata: Optional job metadata
            input_data: Optional input parameters
            max_retries: Maximum retry attempts
            
        Returns:
            Created ProcessingJob instance
        """
        job = ProcessingJob(
            job_type=job_type,
            entity_type=entity_type,
            entity_id=entity_id,
            priority=priority,
            session_id=session_id,
            metadata=metadata or {},
            input_data=input_data or {},
            max_retries=max_retries,
            status=JobStatus.PENDING
        )
        
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        
        logger.info(f"Created job {job.id} of type {job_type.value}")
        return job
    
    async def queue_video_discovery(
        self,
        channel_id: UUID,
        priority: JobPriority = JobPriority.NORMAL,
        force_refresh: bool = False,
        max_videos: int = 500,
        session_id: Optional[UUID] = None
    ) -> str:
        """
        Queue a channel video discovery task.
        
        Args:
            channel_id: Database channel ID
            priority: Job priority
            force_refresh: Force refresh even if recently fetched
            max_videos: Maximum videos to discover
            session_id: Optional processing session ID
            
        Returns:
            Celery task ID
        """
        # Validate channel exists
        channel = await self.db.get(Channel, channel_id)
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Create job record
        job = await self.create_job(
            job_type=JobType.VIDEO_DISCOVERY,
            entity_type="channel",
            entity_id=channel_id,
            priority=priority,
            session_id=session_id,
            input_data={
                "force_refresh": force_refresh,
                "max_videos": max_videos
            }
        )
        
        # Queue Celery task
        task = discover_channel_videos_task.delay(
            str(channel_id),
            force_refresh=force_refresh,
            max_videos=max_videos
        )
        
        # Update job with Celery task ID
        job.metadata["celery_task_id"] = task.id
        job.worker_id = task.id
        await self.db.commit()
        
        logger.info(f"Queued video discovery for channel {channel_id}, task ID: {task.id}")
        return task.id
    
    async def queue_transcript_extraction(
        self,
        video_id: UUID,
        priority: JobPriority = JobPriority.NORMAL,
        force_refresh: bool = False,
        session_id: Optional[UUID] = None
    ) -> str:
        """
        Queue a transcript extraction task.
        
        Args:
            video_id: Database video ID
            priority: Job priority
            force_refresh: Force refresh even if transcript exists
            session_id: Optional processing session ID
            
        Returns:
            Celery task ID
        """
        # Validate video exists
        video = await self.db.get(Video, video_id)
        if not video:
            raise ValidationError(f"Video not found: {video_id}")
        
        # Create job record
        job = await self.create_job(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            entity_type="video",
            entity_id=video_id,
            priority=priority,
            session_id=session_id,
            input_data={
                "force_refresh": force_refresh
            }
        )
        
        # Queue Celery task
        task = extract_transcript_task.delay(
            str(video_id),
            force_refresh=force_refresh
        )
        
        # Update job with Celery task ID
        job.metadata["celery_task_id"] = task.id
        job.worker_id = task.id
        await self.db.commit()
        
        logger.info(f"Queued transcript extraction for video {video_id}, task ID: {task.id}")
        return task.id
    
    async def queue_batch_processing(
        self,
        video_ids: List[UUID],
        session_id: UUID,
        concurrent_limit: Optional[int] = None
    ) -> str:
        """
        Queue a batch processing task for multiple videos.
        
        Args:
            video_ids: List of database video IDs
            session_id: Processing session ID
            concurrent_limit: Maximum concurrent extractions
            
        Returns:
            Celery task ID for the batch coordinator task
        """
        # Validate session exists
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            raise ValidationError(f"Processing session not found: {session_id}")
        
        # Validate videos exist
        video_count = await self.db.execute(
            select(func.count(Video.id)).where(Video.id.in_(video_ids))
        )
        actual_count = video_count.scalar()
        
        if actual_count != len(video_ids):
            raise ValidationError(f"Some videos not found. Expected {len(video_ids)}, found {actual_count}")
        
        # Create coordinator job
        job = await self.create_job(
            job_type=JobType.BULK_PROCESSING,
            entity_type="batch",
            entity_id=session_id,  # Use session ID as entity ID for batch jobs
            priority=JobPriority.NORMAL,
            session_id=session_id,
            input_data={
                "video_ids": [str(vid) for vid in video_ids],
                "concurrent_limit": concurrent_limit or settings.TRANSCRIPT_BATCH_SIZE
            }
        )
        
        # Queue batch processing task
        task = batch_process_transcripts_task.delay(
            [str(vid) for vid in video_ids],
            str(session_id),
            concurrent_limit
        )
        
        # Update job with Celery task ID
        job.metadata["celery_task_id"] = task.id
        job.worker_id = task.id
        await self.db.commit()
        
        # Update session
        session.total_jobs = len(video_ids)
        session.status = SessionStatus.RUNNING
        await self.db.commit()
        
        logger.info(f"Queued batch processing for {len(video_ids)} videos, task ID: {task.id}")
        return task.id
    
    async def queue_channel_metadata_sync(
        self,
        channel_id: UUID,
        priority: JobPriority = JobPriority.LOW,
        session_id: Optional[UUID] = None
    ) -> str:
        """
        Queue a channel metadata synchronization task.
        
        Args:
            channel_id: Database channel ID
            priority: Job priority
            session_id: Optional processing session ID
            
        Returns:
            Celery task ID
        """
        # Validate channel exists
        channel = await self.db.get(Channel, channel_id)
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Create job record
        job = await self.create_job(
            job_type=JobType.CHANNEL_DISCOVERY,
            entity_type="channel",
            entity_id=channel_id,
            priority=priority,
            session_id=session_id
        )
        
        # Queue Celery task
        task = sync_channel_metadata_task.delay(str(channel_id))
        
        # Update job with Celery task ID
        job.metadata["celery_task_id"] = task.id
        job.worker_id = task.id
        await self.db.commit()
        
        logger.info(f"Queued metadata sync for channel {channel_id}, task ID: {task.id}")
        return task.id
    
    async def get_job_status(
        self,
        job_id: UUID
    ) -> Optional[ProcessingJob]:
        """
        Get current status of a processing job.
        
        Args:
            job_id: Database job ID
            
        Returns:
            ProcessingJob with current status or None if not found
        """
        # Get job from database
        job = await self.db.get(ProcessingJob, job_id)
        if not job:
            return None
        
        # If job has Celery task ID, check Celery status
        celery_task_id = job.metadata.get("celery_task_id")
        if celery_task_id:
            try:
                result = AsyncResult(celery_task_id, app=self.celery)
                
                # Update job status based on Celery status
                if result.state == "PENDING":
                    job.status = JobStatus.PENDING
                elif result.state == "STARTED":
                    job.status = JobStatus.RUNNING
                    if not job.started_at:
                        job.started_at = datetime.utcnow()
                elif result.state == "SUCCESS":
                    job.status = JobStatus.COMPLETED
                    if not job.completed_at:
                        job.completed_at = datetime.utcnow()
                        job.output_data = result.result
                elif result.state == "FAILURE":
                    job.status = JobStatus.FAILED
                    if not job.completed_at:
                        job.completed_at = datetime.utcnow()
                        job.error_message = str(result.info)
                elif result.state == "RETRY":
                    job.status = JobStatus.RETRYING
                elif result.state == "REVOKED":
                    job.status = JobStatus.CANCELLED
                    if not job.completed_at:
                        job.completed_at = datetime.utcnow()
                
                await self.db.commit()
                
            except Exception as e:
                logger.error(f"Error checking Celery status for job {job_id}: {e}")
        
        return job
    
    async def cancel_job(
        self,
        job_id: UUID
    ) -> bool:
        """
        Cancel a processing job.
        
        Args:
            job_id: Database job ID
            
        Returns:
            True if successfully cancelled, False otherwise
        """
        # Get job from database
        job = await self.db.get(ProcessingJob, job_id)
        if not job:
            return False
        
        # Can only cancel pending or running jobs
        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RETRYING]:
            logger.warning(f"Cannot cancel job {job_id} with status {job.status.value}")
            return False
        
        # Revoke Celery task if it exists
        celery_task_id = job.metadata.get("celery_task_id")
        if celery_task_id:
            try:
                self.celery.control.revoke(celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {celery_task_id}")
            except Exception as e:
                logger.error(f"Error revoking Celery task {celery_task_id}: {e}")
        
        # Update job status
        job.cancel()
        await self.db.commit()
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def retry_failed_jobs(
        self,
        job_type: Optional[JobType] = None,
        max_age_hours: int = 24,
        limit: int = 100
    ) -> List[ProcessingJob]:
        """
        Retry failed jobs that can be retried.
        
        Args:
            job_type: Optional job type filter
            max_age_hours: Only retry jobs failed within this many hours
            limit: Maximum number of jobs to retry
            
        Returns:
            List of jobs that were requeued
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Build query for failed jobs that can be retried
        query = select(ProcessingJob).where(
            and_(
                ProcessingJob.status == JobStatus.FAILED,
                ProcessingJob.retry_count < ProcessingJob.max_retries,
                ProcessingJob.completed_at >= cutoff_time
            )
        ).limit(limit)
        
        if job_type:
            query = query.where(ProcessingJob.job_type == job_type)
        
        result = await self.db.execute(query)
        failed_jobs = result.scalars().all()
        
        requeued_jobs = []
        
        for job in failed_jobs:
            try:
                # Requeue based on job type
                if job.job_type == JobType.VIDEO_DISCOVERY:
                    task_id = await self.queue_video_discovery(
                        channel_id=job.entity_id,
                        priority=job.priority,
                        force_refresh=job.input_data.get("force_refresh", False),
                        max_videos=job.input_data.get("max_videos", 500),
                        session_id=job.session_id
                    )
                elif job.job_type == JobType.TRANSCRIPT_EXTRACTION:
                    task_id = await self.queue_transcript_extraction(
                        video_id=job.entity_id,
                        priority=job.priority,
                        force_refresh=job.input_data.get("force_refresh", False),
                        session_id=job.session_id
                    )
                elif job.job_type == JobType.CHANNEL_DISCOVERY:
                    task_id = await self.queue_channel_metadata_sync(
                        channel_id=job.entity_id,
                        priority=job.priority,
                        session_id=job.session_id
                    )
                else:
                    logger.warning(f"Cannot retry job type {job.job_type.value}")
                    continue
                
                # Mark original job as retried
                job.retry()
                requeued_jobs.append(job)
                
                logger.info(f"Requeued failed job {job.id} as new task {task_id}")
                
            except Exception as e:
                logger.error(f"Error requeuing job {job.id}: {e}")
        
        await self.db.commit()
        
        logger.info(f"Requeued {len(requeued_jobs)} failed jobs")
        return requeued_jobs
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        # Get job counts by status
        status_counts = {}
        for status in JobStatus:
            count = await self.db.execute(
                select(func.count(ProcessingJob.id)).where(
                    ProcessingJob.status == status
                )
            )
            status_counts[status.value] = count.scalar()
        
        # Get job counts by type
        type_counts = {}
        for job_type in JobType:
            count = await self.db.execute(
                select(func.count(ProcessingJob.id)).where(
                    ProcessingJob.job_type == job_type
                )
            )
            type_counts[job_type.value] = count.scalar()
        
        # Get recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_count = await self.db.execute(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.created_at >= recent_cutoff
            )
        )
        
        # Get average processing times by job type
        avg_times = {}
        for job_type in JobType:
            result = await self.db.execute(
                select(func.avg(ProcessingJob.execution_time_seconds)).where(
                    and_(
                        ProcessingJob.job_type == job_type,
                        ProcessingJob.execution_time_seconds.isnot(None)
                    )
                )
            )
            avg_time = result.scalar()
            avg_times[job_type.value] = avg_time
        
        # Get Celery queue statistics
        celery_stats = self.monitor.get_queue_stats()
        
        return {
            "job_counts_by_status": status_counts,
            "job_counts_by_type": type_counts,
            "recent_jobs_24h": recent_count.scalar(),
            "average_processing_times": avg_times,
            "celery_queue_stats": celery_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_jobs(
        self,
        days: int = 7,
        keep_failed: bool = True
    ) -> int:
        """
        Clean up old completed jobs.
        
        Args:
            days: Delete jobs older than this many days
            keep_failed: Whether to keep failed jobs for analysis
            
        Returns:
            Number of jobs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Build delete criteria
        criteria = [
            ProcessingJob.completed_at < cutoff_date
        ]
        
        if keep_failed:
            criteria.append(ProcessingJob.status != JobStatus.FAILED)
        
        # Count jobs to be deleted
        count_query = select(func.count(ProcessingJob.id)).where(and_(*criteria))
        result = await self.db.execute(count_query)
        job_count = result.scalar()
        
        # Delete jobs
        delete_query = select(ProcessingJob).where(and_(*criteria))
        result = await self.db.execute(delete_query)
        jobs_to_delete = result.scalars().all()
        
        for job in jobs_to_delete:
            await self.db.delete(job)
        
        await self.db.commit()
        
        logger.info(f"Cleaned up {job_count} old jobs")
        return job_count