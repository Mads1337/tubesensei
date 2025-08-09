"""
TubeSensei FastAPI Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.config import settings
from app.database import get_db as get_session, init_db, close_db
from app.models.channel import Channel
from app.models.video import Video
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.services.channel_manager import ChannelManager
from app.services.video_discovery import VideoDiscovery
from app.workers.processing_tasks import discover_channel_videos_task, extract_transcript_task

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting TubeSensei API...")
    await init_db()
    yield
    logger.info("Shutting down TubeSensei API...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description="YouTube Transcript Processing System",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.API_VERSION}


@app.get("/health/database")
async def health_check_database(session: AsyncSession = Depends(get_session)):
    """Check database connectivity"""
    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")


@app.get("/health/redis")
async def health_check_redis():
    """Check Redis connectivity"""
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        raise HTTPException(status_code=503, detail="Redis connection failed")


@app.get("/health/workers")
async def health_check_workers():
    """Check Celery workers status"""
    try:
        from app.celery_app import celery_app
        stats = celery_app.control.inspect().stats()
        if stats:
            worker_count = len(stats)
            return {"status": "healthy", "workers": worker_count, "details": stats}
        else:
            return {"status": "unhealthy", "workers": 0, "message": "No workers available"}
    except Exception as e:
        logger.error(f"Worker health check failed: {e}")
        return {"status": "error", "message": str(e)}


# Channel endpoints
@app.post("/channels/")
async def create_channel(
    channel_url: str,
    session: AsyncSession = Depends(get_session)
):
    """Add a new channel for processing"""
    try:
        from app.utils.youtube_parser import YouTubeParser
        
        parser = YouTubeParser()
        parsed = parser.parse_url(channel_url)
        
        if parsed['type'] != 'channel':
            raise HTTPException(status_code=400, detail="Invalid channel URL")
        
        channel_manager = ChannelManager(session)
        channel = await channel_manager.get_or_create_channel(parsed['id'])
        
        # Queue discovery task
        task = discover_channel_videos_task.delay(str(channel.id))
        
        return {
            "channel_id": str(channel.id),
            "youtube_channel_id": channel.youtube_channel_id,
            "title": channel.title,
            "task_id": task.id,
            "message": "Channel added and discovery queued"
        }
    except Exception as e:
        logger.error(f"Failed to create channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/channels/")
async def list_channels(
    limit: int = 10,
    offset: int = 0,
    session: AsyncSession = Depends(get_session)
):
    """List all channels"""
    try:
        from sqlalchemy import select
        
        query = select(Channel).offset(offset).limit(limit)
        result = await session.execute(query)
        channels = result.scalars().all()
        
        return {
            "channels": [
                {
                    "id": str(c.id),
                    "youtube_channel_id": c.youtube_channel_id,
                    "title": c.title,
                    "status": c.status.value if c.status else None,
                    "video_count": c.video_count,
                    "last_synced": c.last_synced.isoformat() if c.last_synced else None
                }
                for c in channels
            ],
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video endpoints
@app.get("/videos/")
async def list_videos(
    channel_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session)
):
    """List videos, optionally filtered by channel"""
    try:
        from sqlalchemy import select
        
        query = select(Video)
        if channel_id:
            query = query.where(Video.channel_id == channel_id)
        query = query.offset(offset).limit(limit)
        
        result = await session.execute(query)
        videos = result.scalars().all()
        
        return {
            "videos": [
                {
                    "id": str(v.id),
                    "youtube_video_id": v.youtube_video_id,
                    "title": v.title,
                    "channel_id": str(v.channel_id),
                    "duration": v.duration,
                    "published_at": v.published_at.isoformat() if v.published_at else None,
                    "has_transcript": v.has_transcript
                }
                for v in videos
            ],
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/videos/{video_id}/transcript")
async def extract_video_transcript(
    video_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Queue transcript extraction for a video"""
    try:
        from sqlalchemy import select
        
        # Check if video exists
        query = select(Video).where(Video.id == video_id)
        result = await session.execute(query)
        video = result.scalar_one_or_none()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Queue extraction task
        task = extract_transcript_task.delay(video_id)
        
        return {
            "video_id": video_id,
            "task_id": task.id,
            "message": "Transcript extraction queued"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue transcript extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Job endpoints
@app.get("/jobs/")
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session)
):
    """List processing jobs"""
    try:
        from sqlalchemy import select
        
        query = select(ProcessingJob)
        if status:
            query = query.where(ProcessingJob.status == status)
        query = query.order_by(ProcessingJob.created_at.desc())
        query = query.offset(offset).limit(limit)
        
        result = await session.execute(query)
        jobs = result.scalars().all()
        
        return {
            "jobs": [
                {
                    "id": str(j.id),
                    "job_type": j.job_type.value if j.job_type else None,
                    "status": j.status.value if j.status else None,
                    "resource_id": j.resource_id,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                    "error_message": j.error_message
                }
                for j in jobs
            ],
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/status")
async def job_status_summary(session: AsyncSession = Depends(get_session)):
    """Get summary of job statuses"""
    try:
        from sqlalchemy import select, func
        
        query = select(
            ProcessingJob.status,
            func.count(ProcessingJob.id).label('count')
        ).group_by(ProcessingJob.status)
        
        result = await session.execute(query)
        status_counts = {row.status.value if row.status else 'unknown': row.count for row in result}
        
        return {
            "status_counts": status_counts,
            "total": sum(status_counts.values())
        }
    except Exception as e:
        logger.error(f"Failed to get job status summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API documentation
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.API_VERSION,
        "description": "TubeSensei YouTube Transcript Processing System",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "channels": "/channels/",
            "videos": "/videos/",
            "jobs": "/jobs/"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )