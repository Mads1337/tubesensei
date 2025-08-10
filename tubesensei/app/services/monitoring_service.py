from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from collections import defaultdict
import redis.asyncio as redis

from app.models.processing_job import ProcessingJob, JobStatus, JobType
from app.models.video import Video, VideoStatus
from app.models.idea import Idea
from app.models.channel import Channel
from app.core.config import settings


class MonitoringService:
    """Service for system monitoring and metrics"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
        except Exception as e:
            print(f"Failed to initialize Redis: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        # Check database connection
        try:
            await self.db.execute(select(func.now()))
            db_status = "healthy"
        except:
            db_status = "error"
        
        # Check Redis connection
        redis_status = "error"
        if self.redis_client:
            try:
                await self.redis_client.ping()
                redis_status = "healthy"
            except:
                redis_status = "error"
        
        # Get queue status
        queue_status = await self.get_queue_status()
        
        return {
            "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
            "database": db_status,
            "redis": redis_status,
            "queue": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        # Get stats from cache first
        cache_key = "processing_stats"
        cached = None
        
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    import json
                    return json.loads(cached)
            except:
                pass
        
        # Calculate stats
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_24h = now - timedelta(hours=24)
        
        # Job statistics
        job_stats = await self.db.execute(
            select(
                func.count(ProcessingJob.id).label("total_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.COMPLETED
                ).label("completed_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.FAILED
                ).label("failed_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.RUNNING
                ).label("running_jobs"),
                func.count(ProcessingJob.id).filter(
                    and_(
                        ProcessingJob.status == JobStatus.COMPLETED,
                        ProcessingJob.completed_at >= last_hour
                    )
                ).label("completed_last_hour"),
                func.count(ProcessingJob.id).filter(
                    and_(
                        ProcessingJob.status == JobStatus.COMPLETED,
                        ProcessingJob.completed_at >= last_24h
                    )
                ).label("completed_last_24h")
            )
        )
        
        stats = job_stats.one()._asdict()
        
        # Calculate success rate
        total_finished = stats["completed_jobs"] + stats["failed_jobs"]
        stats["success_rate"] = (
            (stats["completed_jobs"] / total_finished * 100) 
            if total_finished > 0 else 0
        )
        
        # Processing rate
        stats["hourly_rate"] = stats["completed_last_hour"]
        stats["daily_rate"] = stats["completed_last_24h"]
        
        # Video and idea counts
        video_count = await self.db.scalar(
            select(func.count(Video.id)).where(
                Video.status == VideoStatus.COMPLETED
            )
        )
        idea_count = await self.db.scalar(
            select(func.count(Idea.id))
        )
        
        stats["total_videos_processed"] = video_count
        stats["total_ideas_extracted"] = idea_count
        
        # Channel counts
        channel_count = await self.db.scalar(
            select(func.count(Channel.id))
        )
        stats["total_channels"] = channel_count
        
        # Cache for 30 seconds
        if self.redis_client:
            try:
                import json
                await self.redis_client.set(
                    cache_key, 
                    json.dumps(stats),
                    ex=30
                )
            except:
                pass
        
        return stats
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get job queue status"""
        # Get queue lengths by job type
        queue_stats = await self.db.execute(
            select(
                ProcessingJob.job_type,
                func.count(ProcessingJob.id).label("count")
            ).where(
                ProcessingJob.status == JobStatus.PENDING
            ).group_by(ProcessingJob.job_type)
        )
        
        queue_by_type = {
            row.job_type.value: row.count 
            for row in queue_stats
        }
        
        # Get total queue length
        total_queued = sum(queue_by_type.values())
        
        # Get average wait time
        avg_wait = await self.db.scalar(
            select(
                func.avg(
                    func.extract('epoch', ProcessingJob.started_at - ProcessingJob.created_at)
                )
            ).where(
                and_(
                    ProcessingJob.status == JobStatus.RUNNING,
                    ProcessingJob.started_at.isnot(None)
                )
            )
        )
        
        return {
            "total_queued": total_queued,
            "by_type": queue_by_type,
            "average_wait_seconds": avg_wait or 0
        }
    
    async def get_recent_jobs(
        self,
        limit: int = 10,
        job_type: Optional[JobType] = None
    ) -> List[Dict[str, Any]]:
        """Get recent processing jobs"""
        query = select(ProcessingJob).order_by(
            ProcessingJob.created_at.desc()
        ).limit(limit)
        
        if job_type:
            query = query.where(ProcessingJob.job_type == job_type)
        
        result = await self.db.execute(query)
        jobs = result.scalars().all()
        
        return [
            {
                "id": str(job.id),
                "type": job.job_type.value,
                "status": job.status.value,
                "entity_type": job.entity_type,
                "entity_id": str(job.entity_id),
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "duration": self._calculate_duration(job),
                "error": job.error_message,
                "progress": job.progress_percent
            }
            for job in jobs
        ]
    
    async def get_processing_timeline(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get processing timeline for charts"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)
        
        # Get completed jobs grouped by hour
        timeline = await self.db.execute(
            select(
                func.date_trunc('hour', ProcessingJob.completed_at).label("hour"),
                func.count(ProcessingJob.id).label("count"),
                ProcessingJob.job_type
            ).where(
                and_(
                    ProcessingJob.status == JobStatus.COMPLETED,
                    ProcessingJob.completed_at >= start_time
                )
            ).group_by(
                func.date_trunc('hour', ProcessingJob.completed_at),
                ProcessingJob.job_type
            )
        )
        
        # Organize by hour and type
        timeline_data = defaultdict(lambda: defaultdict(int))
        for row in timeline:
            hour = row.hour.isoformat()
            timeline_data[hour][row.job_type.value] = row.count
        
        # Create complete timeline with all hours
        hours_list = []
        current_hour = start_time.replace(minute=0, second=0, microsecond=0)
        while current_hour <= now:
            hours_list.append(current_hour.isoformat())
            current_hour += timedelta(hours=1)
        
        # Build datasets for each job type
        datasets = []
        for job_type in JobType:
            dataset = {
                "label": job_type.value.replace("_", " ").title(),
                "data": [
                    timeline_data[hour].get(job_type.value, 0)
                    for hour in hours_list
                ]
            }
            datasets.append(dataset)
        
        return {
            "labels": hours_list,
            "datasets": datasets
        }
    
    async def get_channel_performance(self) -> Dict[str, Any]:
        """Get channel performance metrics"""
        # Get channels with video and idea counts
        channel_stats = await self.db.execute(
            select(
                Channel.id,
                Channel.name,
                func.count(Video.id).label("video_count"),
                func.count(Idea.id).label("idea_count")
            ).outerjoin(
                Video, Channel.id == Video.channel_id
            ).outerjoin(
                Idea, Video.id == Idea.video_id
            ).group_by(
                Channel.id, Channel.name
            ).order_by(
                func.count(Idea.id).desc()
            ).limit(10)
        )
        
        channels = []
        for row in channel_stats:
            channels.append({
                "id": str(row.id),
                "name": row.name,
                "video_count": row.video_count,
                "idea_count": row.idea_count,
                "ideas_per_video": row.idea_count / row.video_count if row.video_count > 0 else 0
            })
        
        return {
            "channels": channels,
            "total": len(channels)
        }
    
    async def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        # Get recent failed jobs
        failed_jobs = await self.db.execute(
            select(
                ProcessingJob.job_type,
                ProcessingJob.error_message,
                func.count(ProcessingJob.id).label("count")
            ).where(
                and_(
                    ProcessingJob.status == JobStatus.FAILED,
                    ProcessingJob.completed_at >= datetime.utcnow() - timedelta(hours=24)
                )
            ).group_by(
                ProcessingJob.job_type,
                ProcessingJob.error_message
            ).order_by(
                func.count(ProcessingJob.id).desc()
            ).limit(10)
        )
        
        errors = []
        for row in failed_jobs:
            errors.append({
                "job_type": row.job_type.value,
                "error": row.error_message or "Unknown error",
                "count": row.count
            })
        
        return {
            "errors": errors,
            "total": sum(e["count"] for e in errors)
        }
    
    def _calculate_duration(self, job: ProcessingJob) -> Optional[float]:
        """Calculate job duration in seconds"""
        if job.started_at and job.completed_at:
            return (job.completed_at - job.started_at).total_seconds()
        elif job.started_at:
            return (datetime.utcnow() - job.started_at).total_seconds()
        return None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()