from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSource
from app.models.processing_job import ProcessingJob, JobStatus, JobType
from app.models.processing_session import ProcessingSession, SessionStatus
from app.utils.crud import CRUDBase


class CRUDChannel(CRUDBase[Channel]):
    async def get_by_youtube_id(
        self,
        db: AsyncSession,
        youtube_channel_id: str
    ) -> Optional[Channel]:
        return await self.get_by_field(
            db,
            field_name="youtube_channel_id",
            field_value=youtube_channel_id
        )
    
    async def get_active_channels(
        self,
        db: AsyncSession,
        limit: int = 100
    ) -> List[Channel]:
        query = (
            select(Channel)
            .where(Channel.status == ChannelStatus.ACTIVE)
            .order_by(Channel.priority_level.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_channels_to_check(
        self,
        db: AsyncSession,
        limit: int = 10
    ) -> List[Channel]:
        query = (
            select(Channel)
            .where(
                and_(
                    Channel.status == ChannelStatus.ACTIVE,
                    or_(
                        Channel.last_checked_at.is_(None),
                        Channel.last_checked_at < datetime.utcnow()
                    )
                )
            )
            .order_by(Channel.last_checked_at.asc().nullsfirst())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class CRUDVideo(CRUDBase[Video]):
    async def get_by_youtube_id(
        self,
        db: AsyncSession,
        youtube_video_id: str
    ) -> Optional[Video]:
        return await self.get_by_field(
            db,
            field_name="youtube_video_id",
            field_value=youtube_video_id
        )
    
    async def get_videos_to_process(
        self,
        db: AsyncSession,
        limit: int = 100
    ) -> List[Video]:
        query = (
            select(Video)
            .where(
                Video.status.in_([VideoStatus.DISCOVERED, VideoStatus.QUEUED])
            )
            .order_by(Video.discovered_at.asc())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_valuable_videos(
        self,
        db: AsyncSession,
        min_score: float = 0.7,
        limit: int = 100
    ) -> List[Video]:
        query = (
            select(Video)
            .where(
                and_(
                    Video.is_valuable == True,
                    Video.valuable_score >= min_score
                )
            )
            .order_by(Video.valuable_score.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_channel_videos(
        self,
        db: AsyncSession,
        channel_id: UUID,
        limit: int = 100
    ) -> List[Video]:
        query = (
            select(Video)
            .where(Video.channel_id == channel_id)
            .order_by(Video.published_at.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class CRUDTranscript(CRUDBase[Transcript]):
    async def get_video_transcript(
        self,
        db: AsyncSession,
        video_id: UUID,
        source: Optional[TranscriptSource] = None
    ) -> Optional[Transcript]:
        query = select(Transcript).where(Transcript.video_id == video_id)
        
        if source:
            query = query.where(Transcript.source == source)
        
        query = query.order_by(Transcript.created_at.desc())
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_video_transcripts(
        self,
        db: AsyncSession,
        video_id: UUID
    ) -> List[Transcript]:
        query = (
            select(Transcript)
            .where(Transcript.video_id == video_id)
            .order_by(Transcript.created_at.desc())
        )
        result = await db.execute(query)
        return result.scalars().all()


class CRUDProcessingJob(CRUDBase[ProcessingJob]):
    async def get_pending_jobs(
        self,
        db: AsyncSession,
        job_type: Optional[JobType] = None,
        limit: int = 100
    ) -> List[ProcessingJob]:
        query = (
            select(ProcessingJob)
            .where(
                and_(
                    ProcessingJob.status == JobStatus.PENDING,
                    ProcessingJob.scheduled_at <= datetime.utcnow()
                )
            )
        )
        
        if job_type:
            query = query.where(ProcessingJob.job_type == job_type)
        
        query = query.order_by(
            ProcessingJob.priority.desc(),
            ProcessingJob.scheduled_at.asc()
        ).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_entity_jobs(
        self,
        db: AsyncSession,
        entity_type: str,
        entity_id: UUID,
        status: Optional[JobStatus] = None
    ) -> List[ProcessingJob]:
        query = (
            select(ProcessingJob)
            .where(
                and_(
                    ProcessingJob.entity_type == entity_type,
                    ProcessingJob.entity_id == entity_id
                )
            )
        )
        
        if status:
            query = query.where(ProcessingJob.status == status)
        
        query = query.order_by(ProcessingJob.created_at.desc())
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_failed_jobs_to_retry(
        self,
        db: AsyncSession,
        limit: int = 50
    ) -> List[ProcessingJob]:
        query = (
            select(ProcessingJob)
            .where(
                and_(
                    ProcessingJob.status == JobStatus.FAILED,
                    ProcessingJob.retry_count < ProcessingJob.max_retries,
                    or_(
                        ProcessingJob.retry_after.is_(None),
                        ProcessingJob.retry_after <= datetime.utcnow()
                    )
                )
            )
            .order_by(ProcessingJob.retry_after.asc().nullsfirst())
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()


class CRUDProcessingSession(CRUDBase[ProcessingSession]):
    async def get_active_sessions(
        self,
        db: AsyncSession
    ) -> List[ProcessingSession]:
        query = (
            select(ProcessingSession)
            .where(
                ProcessingSession.status.in_([
                    SessionStatus.RUNNING,
                    SessionStatus.PAUSED
                ])
            )
            .order_by(ProcessingSession.started_at.desc())
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_session_with_jobs(
        self,
        db: AsyncSession,
        session_id: UUID
    ) -> Optional[ProcessingSession]:
        from sqlalchemy.orm import selectinload
        
        query = (
            select(ProcessingSession)
            .options(selectinload(ProcessingSession.jobs))
            .where(ProcessingSession.id == session_id)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()


crud_channel = CRUDChannel(Channel)
crud_video = CRUDVideo(Video)
crud_transcript = CRUDTranscript(Transcript)
crud_processing_job = CRUDProcessingJob(ProcessingJob)
crud_processing_session = CRUDProcessingSession(ProcessingSession)