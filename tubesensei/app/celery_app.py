"""
Celery application configuration for TubeSensei
"""
from celery import Celery
from app.config import settings


def create_celery_app() -> Celery:
    """Create and configure Celery application"""
    celery_app = Celery(
        "tubesensei",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=["app.workers.processing_tasks"]
    )
    
    # Basic configuration
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
        task_acks_late=True,
        worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
        worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=60,  # 1 minute
        task_max_retries=3,
    )
    
    # Task routing to different queues
    celery_app.conf.task_routes = {
        "app.workers.processing_tasks.discover_channel_videos_task": {
            "queue": "discovery"
        },
        "app.workers.processing_tasks.extract_transcript_task": {
            "queue": "transcripts"
        },
        "app.workers.processing_tasks.batch_process_transcripts_task": {
            "queue": "batch"
        },
        "app.workers.processing_tasks.sync_channel_metadata_task": {
            "queue": "metadata"
        },
    }
    
    # Rate limits for different task types
    celery_app.conf.task_annotations = {
        "app.workers.processing_tasks.extract_transcript_task": {
            "rate_limit": "30/m"  # 30 per minute to respect API limits
        },
        "app.workers.processing_tasks.discover_channel_videos_task": {
            "rate_limit": "10/m"  # 10 per minute for discovery
        },
        "app.workers.processing_tasks.sync_channel_metadata_task": {
            "rate_limit": "20/m"  # 20 per minute for metadata sync
        },
    }
    
    # Result backend configuration
    celery_app.conf.update(
        result_expires=3600,  # Results expire after 1 hour
        result_backend_transport_options={
            "master_name": "mymaster",
            "visibility_timeout": 3600,
            "retry_policy": {
                "timeout": 5.0
            }
        }
    )
    
    # Beat schedule for periodic tasks (if using celery-beat)
    celery_app.conf.beat_schedule = {
        # Example: clean up old jobs every hour
        'cleanup-old-jobs': {
            'task': 'app.workers.processing_tasks.cleanup_old_jobs_task',
            'schedule': 3600.0,  # Every hour
        },
    }
    
    return celery_app


# Create the Celery app instance
celery_app = create_celery_app()


# Task status update helper
def update_job_status(task_id: str, status: str, result=None, error=None):
    """Update job status in database"""
    try:
        from app.database import AsyncSessionLocal
        from app.models.processing_job import ProcessingJob
        from sqlalchemy.orm import sessionmaker
        import asyncio
        
        async def _update():
            async with AsyncSessionLocal() as session:
                # Find job by Celery task ID
                job = await session.execute(
                    "SELECT * FROM processing_jobs WHERE celery_task_id = :task_id",
                    {"task_id": task_id}
                )
                job = job.fetchone()
                if job:
                    # Update job status
                    await session.execute(
                        "UPDATE processing_jobs SET status = :status, result = :result, error_message = :error WHERE celery_task_id = :task_id",
                        {
                            "status": status,
                            "result": result,
                            "error": error,
                            "task_id": task_id
                        }
                    )
                    await session.commit()
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_update())
        loop.close()
    except Exception as e:
        # Log error but don't fail the task
        import logging
        logging.error(f"Failed to update job status: {e}")