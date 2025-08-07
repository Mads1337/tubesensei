"""
Processing Session Manager for TubeSensei
Manages long-running processing sessions and their lifecycle
"""
import logging
from typing import Dict, Any, List, Optional, NamedTuple
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from dataclasses import dataclass

from app.config import settings
from app.models.processing_session import (
    ProcessingSession, 
    SessionType, 
    SessionStatus
)
from app.models.processing_job import ProcessingJob, JobStatus, JobType
from app.models.video import Video
from app.models.channel import Channel
from app.services.job_queue import JobQueueService
from app.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class SessionReport:
    """Comprehensive session report"""
    session_id: UUID
    name: str
    session_type: SessionType
    status: SessionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    success_rate: float
    failure_rate: float
    progress_percent: float
    statistics: Dict[str, Any]
    error_message: Optional[str]
    job_details: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class ProcessingSessionManager:
    """
    Manages processing sessions - long-running operations that coordinate multiple jobs.
    Provides session lifecycle management, progress tracking, and reporting.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.job_queue_service = JobQueueService(db)
    
    async def create_session(
        self,
        session_name: str,
        session_type: SessionType,
        configuration: Dict[str, Any],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        is_resumable: bool = True
    ) -> ProcessingSession:
        """
        Create a new processing session.
        
        Args:
            session_name: Human-readable session name
            session_type: Type of processing session
            configuration: Session configuration parameters
            description: Optional session description
            created_by: Optional creator identifier
            is_resumable: Whether session can be resumed after pause/failure
            
        Returns:
            Created ProcessingSession instance
        """
        session = ProcessingSession(
            name=session_name,
            session_type=session_type,
            description=description,
            status=SessionStatus.INITIALIZED,
            configuration=configuration,
            created_by=created_by,
            is_resumable=is_resumable,
            statistics={
                "created_at": datetime.utcnow().isoformat(),
                "jobs_queued": 0,
                "videos_processed": 0,
                "transcripts_extracted": 0,
                "errors": []
            }
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        logger.info(f"Created processing session: {session.name} ({session.id})")
        return session
    
    async def create_bulk_transcript_session(
        self,
        session_name: str,
        video_ids: List[UUID],
        force_refresh: bool = False,
        concurrent_limit: Optional[int] = None,
        created_by: Optional[str] = None
    ) -> ProcessingSession:
        """
        Create a session for bulk transcript processing.
        
        Args:
            session_name: Session name
            video_ids: List of video IDs to process
            force_refresh: Force refresh existing transcripts
            concurrent_limit: Maximum concurrent extractions
            created_by: Optional creator identifier
            
        Returns:
            Created ProcessingSession with queued jobs
        """
        # Validate videos exist
        video_count = await self.db.execute(
            select(func.count(Video.id)).where(Video.id.in_(video_ids))
        )
        actual_count = video_count.scalar()
        
        if actual_count != len(video_ids):
            raise ValidationError(f"Some videos not found. Expected {len(video_ids)}, found {actual_count}")
        
        # Create session
        configuration = {
            "video_ids": [str(vid) for vid in video_ids],
            "force_refresh": force_refresh,
            "concurrent_limit": concurrent_limit or settings.TRANSCRIPT_BATCH_SIZE,
            "batch_size": settings.MAX_VIDEOS_PER_BATCH
        }
        
        session = await self.create_session(
            session_name=session_name,
            session_type=SessionType.BULK_PROCESSING,
            configuration=configuration,
            description=f"Bulk transcript processing for {len(video_ids)} videos",
            created_by=created_by
        )
        
        # Queue the batch processing job
        task_id = await self.job_queue_service.queue_batch_processing(
            video_ids=video_ids,
            session_id=session.id,
            concurrent_limit=concurrent_limit
        )
        
        # Update session statistics
        session.statistics.update({
            "main_task_id": task_id,
            "jobs_queued": 1
        })
        await self.db.commit()
        
        logger.info(f"Created bulk transcript session with {len(video_ids)} videos")
        return session
    
    async def create_channel_sync_session(
        self,
        session_name: str,
        channel_ids: List[UUID],
        sync_videos: bool = True,
        sync_metadata: bool = True,
        max_videos_per_channel: int = 500,
        created_by: Optional[str] = None
    ) -> ProcessingSession:
        """
        Create a session for channel synchronization.
        
        Args:
            session_name: Session name
            channel_ids: List of channel IDs to sync
            sync_videos: Whether to discover/sync videos
            sync_metadata: Whether to sync channel metadata
            max_videos_per_channel: Maximum videos per channel
            created_by: Optional creator identifier
            
        Returns:
            Created ProcessingSession with queued jobs
        """
        # Validate channels exist
        channel_count = await self.db.execute(
            select(func.count(Channel.id)).where(Channel.id.in_(channel_ids))
        )
        actual_count = channel_count.scalar()
        
        if actual_count != len(channel_ids):
            raise ValidationError(f"Some channels not found. Expected {len(channel_ids)}, found {actual_count}")
        
        # Create session
        configuration = {
            "channel_ids": [str(cid) for cid in channel_ids],
            "sync_videos": sync_videos,
            "sync_metadata": sync_metadata,
            "max_videos_per_channel": max_videos_per_channel
        }
        
        session = await self.create_session(
            session_name=session_name,
            session_type=SessionType.CHANNEL_SYNC,
            configuration=configuration,
            description=f"Channel synchronization for {len(channel_ids)} channels",
            created_by=created_by
        )
        
        # Queue jobs for each channel
        task_ids = []
        jobs_queued = 0
        
        for channel_id in channel_ids:
            if sync_videos:
                task_id = await self.job_queue_service.queue_video_discovery(
                    channel_id=channel_id,
                    force_refresh=True,
                    max_videos=max_videos_per_channel,
                    session_id=session.id
                )
                task_ids.append(task_id)
                jobs_queued += 1
            
            if sync_metadata:
                task_id = await self.job_queue_service.queue_channel_metadata_sync(
                    channel_id=channel_id,
                    session_id=session.id
                )
                task_ids.append(task_id)
                jobs_queued += 1
        
        # Update session
        session.total_jobs = jobs_queued
        session.statistics.update({
            "task_ids": task_ids,
            "jobs_queued": jobs_queued
        })
        await self.db.commit()
        
        logger.info(f"Created channel sync session with {jobs_queued} jobs for {len(channel_ids)} channels")
        return session
    
    async def start_session(self, session_id: UUID) -> ProcessingSession:
        """
        Start a processing session.
        
        Args:
            session_id: Session ID to start
            
        Returns:
            Updated ProcessingSession
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            raise ValidationError(f"Session not found: {session_id}")
        
        if session.status != SessionStatus.INITIALIZED:
            raise ValidationError(f"Session must be initialized to start (current: {session.status.value})")
        
        session.start()
        await self.db.commit()
        
        logger.info(f"Started session: {session.name}")
        return session
    
    async def pause_session(self, session_id: UUID) -> ProcessingSession:
        """
        Pause a running session.
        
        Args:
            session_id: Session ID to pause
            
        Returns:
            Updated ProcessingSession
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            raise ValidationError(f"Session not found: {session_id}")
        
        if not session.is_resumable:
            raise ValidationError("Session is not resumable")
        
        # Pause the session
        session.pause()
        
        # Get current progress data for checkpoint
        checkpoint_data = await self._create_checkpoint_data(session)
        session.save_checkpoint(checkpoint_data)
        
        await self.db.commit()
        
        logger.info(f"Paused session: {session.name}")
        return session
    
    async def resume_session(self, session_id: UUID) -> ProcessingSession:
        """
        Resume a paused session.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            Updated ProcessingSession
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            raise ValidationError(f"Session not found: {session_id}")
        
        if session.status != SessionStatus.PAUSED:
            raise ValidationError(f"Session must be paused to resume (current: {session.status.value})")
        
        # Resume the session
        session.resume()
        
        # TODO: Resume any paused jobs based on checkpoint data
        # This would involve requeuing failed or interrupted jobs
        
        await self.db.commit()
        
        logger.info(f"Resumed session: {session.name}")
        return session
    
    async def cancel_session(self, session_id: UUID, cancel_jobs: bool = True) -> ProcessingSession:
        """
        Cancel a session and optionally its associated jobs.
        
        Args:
            session_id: Session ID to cancel
            cancel_jobs: Whether to cancel associated jobs
            
        Returns:
            Updated ProcessingSession
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            raise ValidationError(f"Session not found: {session_id}")
        
        if session.is_complete:
            raise ValidationError(f"Session already complete: {session.status.value}")
        
        # Cancel associated jobs if requested
        if cancel_jobs:
            cancelled_count = await self._cancel_session_jobs(session_id)
            logger.info(f"Cancelled {cancelled_count} jobs for session {session_id}")
        
        # Cancel the session
        session.cancel()
        await self.db.commit()
        
        logger.info(f"Cancelled session: {session.name}")
        return session
    
    async def update_progress(
        self,
        session_id: UUID,
        processed: int,
        total: Optional[int] = None,
        additional_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Update session progress.
        
        Args:
            session_id: Session ID to update
            processed: Number of items processed
            total: Total items (updates session.total_jobs if provided)
            additional_stats: Additional statistics to merge
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            return
        
        if total is not None:
            session.total_jobs = total
        
        # Update job counts based on actual job statuses
        await self._update_job_counts(session)
        
        # Update additional statistics
        if additional_stats:
            session.statistics.update(additional_stats)
        
        # Update progress and estimated completion
        session.update_progress()
        self._estimate_completion_time(session)
        
        await self.db.commit()
    
    async def complete_session(
        self,
        session_id: UUID,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Mark session as completed and finalize metrics.
        
        Args:
            session_id: Session ID to complete
            final_metrics: Final processing metrics
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            return
        
        # Update final job counts
        await self._update_job_counts(session)
        
        # Update final statistics
        if final_metrics:
            session.statistics.update(final_metrics)
        
        # Add completion metrics
        session.statistics.update({
            "completed_at": datetime.utcnow().isoformat(),
            "final_success_rate": session.success_rate,
            "final_failure_rate": session.failure_rate
        })
        
        # Complete the session
        session.complete()
        await self.db.commit()
        
        logger.info(f"Completed session: {session.name} ({session.success_rate:.1f}% success rate)")
    
    async def fail_session(
        self,
        session_id: UUID,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """
        Mark session as failed.
        
        Args:
            session_id: Session ID to fail
            error_message: Error description
            error_details: Additional error details
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            return
        
        # Update job counts
        await self._update_job_counts(session)
        
        # Add error details to statistics
        if error_details:
            session.statistics.setdefault("errors", []).append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": error_message,
                "details": error_details
            })
        
        # Fail the session
        session.fail(error_message)
        await self.db.commit()
        
        logger.error(f"Failed session: {session.name} - {error_message}")
    
    async def get_session_report(
        self,
        session_id: UUID,
        include_job_details: bool = False
    ) -> Optional[SessionReport]:
        """
        Generate comprehensive session report.
        
        Args:
            session_id: Session ID
            include_job_details: Whether to include detailed job information
            
        Returns:
            SessionReport or None if session not found
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            return None
        
        # Get job details if requested
        job_details = []
        if include_job_details:
            job_query = select(ProcessingJob).where(
                ProcessingJob.session_id == session_id
            ).order_by(ProcessingJob.created_at)
            
            result = await self.db.execute(job_query)
            jobs = result.scalars().all()
            
            for job in jobs:
                job_details.append({
                    "id": str(job.id),
                    "type": job.job_type.value,
                    "status": job.status.value,
                    "entity_type": job.entity_type,
                    "entity_id": str(job.entity_id),
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "duration_seconds": job.execution_time_seconds,
                    "retry_count": job.retry_count,
                    "error_message": job.error_message,
                    "progress_percent": job.progress_percent
                })
        
        # Calculate performance metrics
        performance_metrics = await self._calculate_performance_metrics(session_id)
        
        return SessionReport(
            session_id=session.id,
            name=session.name,
            session_type=session.session_type,
            status=session.status,
            started_at=session.started_at,
            completed_at=session.completed_at,
            duration_seconds=session.duration_seconds,
            total_jobs=session.total_jobs,
            completed_jobs=session.completed_jobs,
            failed_jobs=session.failed_jobs,
            cancelled_jobs=session.cancelled_jobs,
            success_rate=session.success_rate,
            failure_rate=session.failure_rate,
            progress_percent=session.progress_percent,
            statistics=session.statistics,
            error_message=session.error_message,
            job_details=job_details,
            performance_metrics=performance_metrics
        )
    
    async def list_active_sessions(self) -> List[ProcessingSession]:
        """
        Get all active (running or paused) sessions.
        
        Returns:
            List of active ProcessingSession instances
        """
        query = select(ProcessingSession).where(
            ProcessingSession.status.in_([
                SessionStatus.RUNNING,
                SessionStatus.PAUSED
            ])
        ).order_by(ProcessingSession.started_at.desc())
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_session_statistics(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get current session statistics.
        
        Args:
            session_id: Session ID
            
        Returns:
            Statistics dictionary or None
        """
        session = await self.db.get(ProcessingSession, session_id)
        if not session:
            return None
        
        # Update job counts
        await self._update_job_counts(session)
        await self.db.commit()
        
        # Return comprehensive stats
        return {
            "session_info": {
                "id": str(session.id),
                "name": session.name,
                "type": session.session_type.value,
                "status": session.status.value,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "duration_seconds": session.duration_seconds
            },
            "progress": {
                "total_jobs": session.total_jobs,
                "completed_jobs": session.completed_jobs,
                "failed_jobs": session.failed_jobs,
                "cancelled_jobs": session.cancelled_jobs,
                "progress_percent": session.progress_percent,
                "success_rate": session.success_rate,
                "failure_rate": session.failure_rate
            },
            "configuration": session.configuration,
            "statistics": session.statistics,
            "estimated_completion": session.estimated_completion_at.isoformat() if session.estimated_completion_at else None
        }
    
    async def _update_job_counts(self, session: ProcessingSession):
        """Update job counts based on actual job statuses"""
        # Count jobs by status
        status_counts = await self.db.execute(
            select(
                ProcessingJob.status,
                func.count(ProcessingJob.id)
            ).where(
                ProcessingJob.session_id == session.id
            ).group_by(ProcessingJob.status)
        )
        
        # Reset counts
        session.completed_jobs = 0
        session.failed_jobs = 0
        session.cancelled_jobs = 0
        
        # Update counts
        for status, count in status_counts:
            if status == JobStatus.COMPLETED:
                session.completed_jobs = count
            elif status == JobStatus.FAILED:
                session.failed_jobs = count
            elif status == JobStatus.CANCELLED:
                session.cancelled_jobs = count
    
    async def _cancel_session_jobs(self, session_id: UUID) -> int:
        """Cancel all jobs associated with a session"""
        # Get all non-completed jobs
        query = select(ProcessingJob).where(
            and_(
                ProcessingJob.session_id == session_id,
                ~ProcessingJob.status.in_([
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED
                ])
            )
        )
        
        result = await self.db.execute(query)
        jobs = result.scalars().all()
        
        cancelled_count = 0
        for job in jobs:
            success = await self.job_queue_service.cancel_job(job.id)
            if success:
                cancelled_count += 1
        
        return cancelled_count
    
    async def _create_checkpoint_data(self, session: ProcessingSession) -> Dict[str, Any]:
        """Create checkpoint data for session pause/resume"""
        # Get current job statuses
        job_status_query = select(
            ProcessingJob.id,
            ProcessingJob.status,
            ProcessingJob.progress_percent
        ).where(ProcessingJob.session_id == session.id)
        
        result = await self.db.execute(job_status_query)
        job_statuses = {str(job_id): {"status": status.value, "progress": progress} 
                       for job_id, status, progress in result}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "job_statuses": job_statuses,
            "progress_percent": session.progress_percent,
            "total_jobs": session.total_jobs,
            "completed_jobs": session.completed_jobs,
            "failed_jobs": session.failed_jobs
        }
    
    def _estimate_completion_time(self, session: ProcessingSession):
        """Estimate session completion time based on progress"""
        if session.progress_percent <= 0 or not session.started_at:
            return
        
        elapsed = (datetime.utcnow() - session.started_at).total_seconds()
        if elapsed <= 0:
            return
        
        # Simple linear estimation
        estimated_total_time = elapsed * (100.0 / session.progress_percent)
        remaining_time = estimated_total_time - elapsed
        
        if remaining_time > 0:
            session.estimated_completion_at = datetime.utcnow() + timedelta(seconds=remaining_time)
    
    async def _calculate_performance_metrics(self, session_id: UUID) -> Dict[str, Any]:
        """Calculate detailed performance metrics for a session"""
        # Get job execution times
        execution_times_query = select(
            ProcessingJob.job_type,
            ProcessingJob.execution_time_seconds
        ).where(
            and_(
                ProcessingJob.session_id == session_id,
                ProcessingJob.execution_time_seconds.isnot(None)
            )
        )
        
        result = await self.db.execute(execution_times_query)
        execution_data = result.all()
        
        # Calculate metrics by job type
        metrics_by_type = {}
        all_times = []
        
        for job_type, execution_time in execution_data:
            type_key = job_type.value
            if type_key not in metrics_by_type:
                metrics_by_type[type_key] = []
            
            metrics_by_type[type_key].append(execution_time)
            all_times.append(execution_time)
        
        # Calculate summary statistics
        performance_metrics = {
            "total_execution_time": sum(all_times) if all_times else 0,
            "average_job_time": sum(all_times) / len(all_times) if all_times else 0,
            "fastest_job": min(all_times) if all_times else 0,
            "slowest_job": max(all_times) if all_times else 0,
            "job_count": len(all_times)
        }
        
        # Add metrics by job type
        for job_type, times in metrics_by_type.items():
            performance_metrics[f"{job_type}_avg_time"] = sum(times) / len(times)
            performance_metrics[f"{job_type}_count"] = len(times)
        
        return performance_metrics