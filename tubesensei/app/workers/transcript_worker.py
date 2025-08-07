import asyncio
import logging
import signal
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import time

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db_context
from ..models.video import Video, VideoStatus
from ..models.processing_job import ProcessingJob, JobType, JobStatus
from ..services.transcript_processor import TranscriptProcessor
from ..repositories.transcript_repository import TranscriptRepository

logger = logging.getLogger(__name__)


class TranscriptWorker:
    """
    Background worker for processing transcript extraction jobs.
    Polls for pending jobs and processes them asynchronously.
    """
    
    def __init__(
        self,
        poll_interval: int = 10,
        batch_size: int = None,
        concurrent_limit: int = None,
        use_cache: bool = True
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size or settings.TRANSCRIPT_BATCH_SIZE
        self.concurrent_limit = concurrent_limit or settings.TRANSCRIPT_BATCH_SIZE
        self.use_cache = use_cache
        
        # Worker state
        self.is_running = False
        self.should_stop = False
        self.current_jobs: List[ProcessingJob] = []
        
        # Processing statistics
        self.stats = {
            "started_at": None,
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(
            f"Initialized TranscriptWorker (poll={poll_interval}s, "
            f"batch={self.batch_size}, concurrent={self.concurrent_limit})"
        )
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start the worker processing loop."""
        logger.info("Starting TranscriptWorker...")
        self.is_running = True
        self.should_stop = False
        self.stats["started_at"] = datetime.utcnow()
        
        async with TranscriptProcessor(use_cache=self.use_cache) as processor:
            while not self.should_stop:
                try:
                    # Process one batch
                    processed = await self._process_batch(processor)
                    
                    if processed == 0:
                        # No jobs found, wait before polling again
                        logger.debug(f"No jobs found, waiting {self.poll_interval}s...")
                        await asyncio.sleep(self.poll_interval)
                    else:
                        # Jobs processed, check immediately for more
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}")
                    self.stats["errors"].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e)
                    })
                    await asyncio.sleep(self.poll_interval)
        
        self.is_running = False
        logger.info("TranscriptWorker stopped")
    
    async def stop(self):
        """Stop the worker gracefully."""
        logger.info("Stopping TranscriptWorker...")
        self.should_stop = True
        
        # Wait for current jobs to complete
        if self.current_jobs:
            logger.info(f"Waiting for {len(self.current_jobs)} jobs to complete...")
            max_wait = 60  # Maximum 60 seconds wait
            start_time = time.time()
            
            while self.current_jobs and (time.time() - start_time) < max_wait:
                await asyncio.sleep(1)
            
            if self.current_jobs:
                logger.warning(f"Force stopping with {len(self.current_jobs)} jobs still running")
    
    async def _process_batch(self, processor: TranscriptProcessor) -> int:
        """
        Process a batch of transcript jobs.
        
        Args:
            processor: TranscriptProcessor instance
            
        Returns:
            Number of jobs processed
        """
        async with get_db_context() as session:
            # Get pending jobs
            jobs = await self._get_pending_jobs(session)
            
            if not jobs:
                return 0
            
            logger.info(f"Processing batch of {len(jobs)} transcript jobs")
            self.current_jobs = jobs
            
            # Mark jobs as processing
            for job in jobs:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.utcnow()
            await session.commit()
            
            # Process jobs concurrently
            semaphore = asyncio.Semaphore(self.concurrent_limit)
            
            async def process_job(job: ProcessingJob):
                async with semaphore:
                    await self._process_single_job(job, processor, session)
            
            tasks = [process_job(job) for job in jobs]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Commit final status updates
            await session.commit()
            
            self.current_jobs = []
            return len(jobs)
    
    async def _process_single_job(
        self,
        job: ProcessingJob,
        processor: TranscriptProcessor,
        session: AsyncSession
    ):
        """
        Process a single transcript extraction job.
        
        Args:
            job: ProcessingJob to process
            processor: TranscriptProcessor instance
            session: Database session
        """
        try:
            logger.info(f"Processing job {job.id} for video {job.video_id}")
            
            # Update job metadata
            job.metadata = job.metadata or {}
            job.metadata["worker_id"] = id(self)
            job.metadata["started_at"] = datetime.utcnow().isoformat()
            
            # Process transcript
            processor.session = session  # Use same session
            transcript = await processor.extract_transcript(
                video_id=job.video_id,
                force_refresh=job.metadata.get("force_refresh", False)
            )
            
            if transcript:
                # Success
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.result = {
                    "transcript_id": str(transcript.id),
                    "language": transcript.language_code,
                    "word_count": transcript.word_count,
                    "confidence_score": transcript.confidence_score
                }
                
                self.stats["successful"] += 1
                logger.info(f"Successfully completed job {job.id}")
            else:
                # Failed but handled
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = "Failed to extract transcript"
                job.retry_count += 1
                
                self.stats["failed"] += 1
                logger.warning(f"Job {job.id} failed")
                
        except Exception as e:
            # Unexpected error
            logger.error(f"Error processing job {job.id}: {e}")
            
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            job.retry_count += 1
            
            # Check if should retry
            if job.retry_count < settings.TRANSCRIPT_MAX_RETRIES:
                job.status = JobStatus.PENDING
                job.scheduled_for = datetime.utcnow() + timedelta(
                    minutes=5 * job.retry_count  # Exponential backoff
                )
                logger.info(f"Rescheduled job {job.id} for retry #{job.retry_count}")
            
            self.stats["failed"] += 1
            
        finally:
            self.stats["total_processed"] += 1
    
    async def _get_pending_jobs(self, session: AsyncSession) -> List[ProcessingJob]:
        """
        Get pending transcript extraction jobs from database.
        
        Args:
            session: Database session
            
        Returns:
            List of pending jobs
        """
        from sqlalchemy import select, and_, or_
        
        # Query for pending transcript jobs
        query = select(ProcessingJob).where(
            and_(
                ProcessingJob.job_type == JobType.TRANSCRIPT_EXTRACTION,
                or_(
                    ProcessingJob.status == JobStatus.PENDING,
                    and_(
                        ProcessingJob.status == JobStatus.PROCESSING,
                        ProcessingJob.started_at < datetime.utcnow() - timedelta(minutes=30)  # Stale jobs
                    )
                ),
                or_(
                    ProcessingJob.scheduled_for.is_(None),
                    ProcessingJob.scheduled_for <= datetime.utcnow()
                )
            )
        ).order_by(
            ProcessingJob.priority.desc(),
            ProcessingJob.created_at.asc()
        ).limit(self.batch_size)
        
        result = await session.execute(query)
        return result.scalars().all()
    
    async def create_jobs_for_new_videos(
        self,
        limit: int = 100,
        only_with_captions: bool = True
    ) -> int:
        """
        Create transcript extraction jobs for videos without transcripts.
        
        Args:
            limit: Maximum number of jobs to create
            only_with_captions: Only create jobs for videos with captions
            
        Returns:
            Number of jobs created
        """
        async with get_db_context() as session:
            repo = TranscriptRepository(session)
            
            # Get videos without transcripts
            videos = await repo.get_videos_without_transcripts(limit=limit)
            
            if only_with_captions:
                videos = [v for v in videos if v.has_captions]
            
            if not videos:
                logger.info("No videos found for transcript extraction")
                return 0
            
            # Create jobs
            jobs_created = 0
            for video in videos:
                # Check if job already exists
                existing_job = await session.execute(
                    select(ProcessingJob).where(
                        and_(
                            ProcessingJob.video_id == video.id,
                            ProcessingJob.job_type == JobType.TRANSCRIPT_EXTRACTION,
                            ProcessingJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
                        )
                    )
                )
                
                if existing_job.scalar_one_or_none():
                    continue
                
                # Create new job
                job = ProcessingJob(
                    video_id=video.id,
                    job_type=JobType.TRANSCRIPT_EXTRACTION,
                    status=JobStatus.PENDING,
                    priority=self._calculate_priority(video),
                    metadata={
                        "channel_id": str(video.channel_id),
                        "has_captions": video.has_captions,
                        "language": video.language,
                        "created_by": "TranscriptWorker"
                    }
                )
                
                session.add(job)
                jobs_created += 1
            
            await session.commit()
            logger.info(f"Created {jobs_created} transcript extraction jobs")
            return jobs_created
    
    def _calculate_priority(self, video: Video) -> int:
        """
        Calculate job priority based on video attributes.
        
        Args:
            video: Video object
            
        Returns:
            Priority score (0-100)
        """
        priority = 50  # Base priority
        
        # Boost for high view count
        if video.view_count:
            if video.view_count > 1000000:
                priority += 20
            elif video.view_count > 100000:
                priority += 10
            elif video.view_count > 10000:
                priority += 5
        
        # Boost for recent videos
        if video.published_at:
            days_old = (datetime.utcnow() - video.published_at).days
            if days_old < 7:
                priority += 15
            elif days_old < 30:
                priority += 10
            elif days_old < 90:
                priority += 5
        
        # Boost for valuable videos
        if video.is_valuable:
            priority += 10
        
        # Penalty for failed attempts
        if video.retry_count > 0:
            priority -= (video.retry_count * 5)
        
        return max(0, min(100, priority))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        runtime = None
        if self.stats["started_at"]:
            runtime = (datetime.utcnow() - self.stats["started_at"]).total_seconds()
        
        return {
            "is_running": self.is_running,
            "started_at": self.stats["started_at"].isoformat() if self.stats["started_at"] else None,
            "runtime_seconds": runtime,
            "total_processed": self.stats["total_processed"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "success_rate": (
                (self.stats["successful"] / self.stats["total_processed"] * 100)
                if self.stats["total_processed"] > 0 else 0
            ),
            "current_jobs": len(self.current_jobs),
            "recent_errors": self.stats["errors"][-10:]  # Last 10 errors
        }


async def run_worker():
    """Main entry point for running the transcript worker."""
    logger.info("Starting Transcript Worker Service...")
    
    worker = TranscriptWorker(
        poll_interval=10,
        batch_size=settings.TRANSCRIPT_BATCH_SIZE,
        concurrent_limit=settings.TRANSCRIPT_BATCH_SIZE,
        use_cache=True
    )
    
    try:
        # Create initial jobs if needed
        await worker.create_jobs_for_new_videos(limit=100)
        
        # Start processing
        await worker.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Worker crashed: {e}")
    finally:
        await worker.stop()
        
        # Print final statistics
        stats = worker.get_statistics()
        logger.info(f"Worker statistics: {stats}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the worker
    asyncio.run(run_worker())