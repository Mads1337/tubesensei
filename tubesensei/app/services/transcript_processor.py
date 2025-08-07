import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import time

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.transcript import Transcript, TranscriptSource
from ..models.video import Video, VideoStatus
from ..models.transcript_data import (
    TranscriptData,
    TranscriptMetrics,
    ProcessingResult,
    ProcessingError,
    TranscriptExtractionRequest,
    TranscriptBatchRequest
)
from ..integrations.transcript_api import TranscriptAPIClient
from ..integrations.transcript_errors import (
    TranscriptError,
    TranscriptNotAvailableError,
    TranscriptDisabledError,
    TranscriptVideoUnavailableError,
    is_retryable_error,
    get_retry_delay
)
from .transcript_cleaner import TranscriptCleaner
from .transcript_analyzer import TranscriptAnalyzer
from .transcript_cache import TranscriptCache
from ..repositories.transcript_repository import TranscriptRepository
from ..database import get_db_context

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """
    Main orchestration service for transcript processing.
    Coordinates extraction, cleaning, analysis, caching, and storage.
    """
    
    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        use_cache: bool = True
    ):
        self.session = session
        self.use_cache = use_cache
        
        # Initialize components
        self.api_client = TranscriptAPIClient()
        self.cleaner = TranscriptCleaner()
        self.analyzer = TranscriptAnalyzer()
        
        # Initialize cache if enabled
        self.cache = TranscriptCache() if use_cache else None
        
        # Processing statistics
        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "from_cache": 0
        }
        
        logger.info(f"Initialized TranscriptProcessor (cache={'enabled' if use_cache else 'disabled'})")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.cache:
            await self.cache.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.cache:
            await self.cache.disconnect()
    
    async def extract_transcript(
        self,
        video_id: UUID,
        force_refresh: bool = False,
        clean_content: bool = True,
        calculate_metrics: bool = True,
        save_to_db: bool = True
    ) -> Optional[Transcript]:
        """
        Extract and process transcript for a single video.
        
        Args:
            video_id: Video UUID
            force_refresh: Force refresh even if exists
            clean_content: Whether to clean transcript
            calculate_metrics: Whether to calculate metrics
            save_to_db: Whether to save to database
            
        Returns:
            Transcript object or None
        """
        start_time = time.time()
        
        try:
            # Get video from database
            if self.session:
                repo = TranscriptRepository(self.session)
                video = await self.session.get(Video, video_id)
            else:
                async with get_db_context() as session:
                    repo = TranscriptRepository(session)
                    video = await session.get(Video, video_id)
            
            if not video:
                logger.error(f"Video {video_id} not found")
                return None
            
            # Check if transcript already exists
            if not force_refresh:
                existing = await self._get_existing_transcript(video_id, repo)
                if existing:
                    logger.info(f"Transcript already exists for video {video_id}")
                    self._stats["skipped"] += 1
                    return existing
            
            # Check cache if enabled
            if self.use_cache and self.cache and not force_refresh:
                cached_data = await self._get_from_cache(video.youtube_video_id)
                if cached_data:
                    logger.info(f"Retrieved transcript from cache for video {video_id}")
                    self._stats["from_cache"] += 1
                    
                    # Process cached data
                    return await self._process_cached_transcript(
                        video,
                        cached_data,
                        repo,
                        clean_content,
                        calculate_metrics,
                        save_to_db
                    )
            
            # Extract from YouTube
            logger.info(f"Extracting transcript for video {video.youtube_video_id}")
            transcript_data = await self._extract_from_youtube(video.youtube_video_id)
            
            if not transcript_data:
                logger.warning(f"No transcript available for video {video_id}")
                await self._mark_video_failed(video, "No transcript available")
                self._stats["failed"] += 1
                return None
            
            # Process the transcript
            processed_transcript = await self._process_transcript(
                video,
                transcript_data,
                repo,
                clean_content,
                calculate_metrics,
                save_to_db
            )
            
            # Cache if enabled
            if self.use_cache and self.cache and processed_transcript:
                await self._cache_transcript(video.youtube_video_id, transcript_data)
            
            # Update video status
            await self._mark_video_completed(video)
            
            self._stats["successful"] += 1
            self._stats["total_processed"] += 1
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed transcript for video {video_id} in {processing_time:.2f}s")
            
            return processed_transcript
            
        except TranscriptError as e:
            logger.error(f"Transcript error for video {video_id}: {e}")
            await self._handle_extraction_error(video, e)
            self._stats["failed"] += 1
            self._stats["total_processed"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error processing transcript for video {video_id}: {e}")
            await self._mark_video_failed(video, str(e))
            self._stats["failed"] += 1
            self._stats["total_processed"] += 1
            return None
    
    async def batch_process_transcripts(
        self,
        video_ids: List[UUID],
        concurrent_limit: int = None,
        skip_existing: bool = True,
        retry_failed: bool = False
    ) -> ProcessingResult:
        """
        Process multiple video transcripts in batch.
        
        Args:
            video_ids: List of video UUIDs
            concurrent_limit: Max concurrent extractions
            skip_existing: Skip videos with existing transcripts
            retry_failed: Retry previously failed extractions
            
        Returns:
            ProcessingResult with statistics
        """
        concurrent_limit = concurrent_limit or settings.TRANSCRIPT_BATCH_SIZE
        start_time = time.time()
        
        result = ProcessingResult()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def process_video(video_id: UUID):
            async with semaphore:
                try:
                    transcript = await self.extract_transcript(
                        video_id,
                        force_refresh=not skip_existing
                    )
                    
                    if transcript:
                        result.successful += 1
                    else:
                        result.failed += 1
                    
                except Exception as e:
                    logger.error(f"Error in batch processing for video {video_id}: {e}")
                    result.add_error(video_id, type(e).__name__, str(e))
                
                finally:
                    result.total_processed += 1
        
        # Process videos concurrently
        tasks = [process_video(video_id) for video_id in video_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate processing time
        result.processing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Batch processing complete: {result.successful}/{result.total_processed} successful "
            f"in {result.processing_time_seconds:.2f}s"
        )
        
        return result
    
    async def reprocess_failed_transcripts(
        self,
        session_id: Optional[UUID] = None,
        max_videos: int = 100,
        max_retries: int = 3
    ) -> ProcessingResult:
        """
        Reprocess videos that failed transcript extraction.
        
        Args:
            session_id: Optional session ID filter
            max_videos: Maximum videos to process
            max_retries: Maximum retry attempts
            
        Returns:
            ProcessingResult with statistics
        """
        logger.info(f"Starting reprocessing of failed transcripts (max: {max_videos})")
        
        # Get failed videos
        if self.session:
            repo = TranscriptRepository(self.session)
            failed_videos = await repo.get_failed_extractions(
                limit=max_videos,
                max_retries=max_retries
            )
        else:
            async with get_db_context() as session:
                repo = TranscriptRepository(session)
                failed_videos = await repo.get_failed_extractions(
                    limit=max_videos,
                    max_retries=max_retries
                )
        
        if not failed_videos:
            logger.info("No failed transcripts to reprocess")
            return ProcessingResult()
        
        # Extract video IDs
        video_ids = [video.id for video in failed_videos]
        
        logger.info(f"Found {len(video_ids)} failed videos to reprocess")
        
        # Process with exponential backoff
        result = ProcessingResult()
        
        for i, video_id in enumerate(video_ids):
            # Add delay between retries
            if i > 0:
                delay = min(2 ** i, 30)  # Exponential backoff, max 30s
                await asyncio.sleep(delay)
            
            try:
                transcript = await self.extract_transcript(
                    video_id,
                    force_refresh=True
                )
                
                if transcript:
                    result.successful += 1
                else:
                    result.failed += 1
                    
            except Exception as e:
                logger.error(f"Error reprocessing video {video_id}: {e}")
                result.add_error(video_id, type(e).__name__, str(e))
            
            finally:
                result.total_processed += 1
        
        logger.info(
            f"Reprocessing complete: {result.successful}/{result.total_processed} successful"
        )
        
        return result
    
    async def update_transcript_quality(
        self,
        transcript_id: UUID
    ) -> Optional[Transcript]:
        """
        Recalculate and update transcript quality metrics.
        
        Args:
            transcript_id: Transcript UUID
            
        Returns:
            Updated Transcript or None
        """
        try:
            if self.session:
                repo = TranscriptRepository(self.session)
                transcript = await repo.get_by_id(transcript_id)
            else:
                async with get_db_context() as session:
                    repo = TranscriptRepository(session)
                    transcript = await repo.get_by_id(transcript_id)
            
            if not transcript:
                logger.error(f"Transcript {transcript_id} not found")
                return None
            
            # Clean content if not already done
            if not transcript.processed_content:
                cleaned_content = self.cleaner.clean_transcript(transcript.content)
                transcript.processed_content = cleaned_content
            else:
                cleaned_content = transcript.processed_content
            
            # Analyze transcript
            metrics = self.analyzer.analyze_transcript(
                cleaned_content,
                is_auto_generated=transcript.is_auto_generated
            )
            
            # Update transcript with new metrics
            updated = await repo.update_quality_metrics(transcript_id, metrics)
            
            logger.info(f"Updated quality metrics for transcript {transcript_id}")
            return updated
            
        except Exception as e:
            logger.error(f"Error updating transcript quality for {transcript_id}: {e}")
            return None
    
    async def _extract_from_youtube(
        self,
        youtube_video_id: str
    ) -> Optional[TranscriptData]:
        """Extract transcript from YouTube."""
        try:
            return await self.api_client.get_transcript_with_retry(youtube_video_id)
        except TranscriptError as e:
            logger.warning(f"Failed to extract transcript: {e}")
            return None
    
    async def _process_transcript(
        self,
        video: Video,
        transcript_data: TranscriptData,
        repo: TranscriptRepository,
        clean_content: bool,
        calculate_metrics: bool,
        save_to_db: bool
    ) -> Optional[Transcript]:
        """Process extracted transcript data."""
        # Clean content
        if clean_content:
            cleaned_content = self.cleaner.clean_transcript(transcript_data.content)
        else:
            cleaned_content = transcript_data.content
        
        # Analyze transcript
        metrics = None
        if calculate_metrics:
            metrics = self.analyzer.analyze_transcript(
                cleaned_content,
                is_auto_generated=transcript_data.is_auto_generated
            )
        
        # Save to database
        if save_to_db:
            # Determine source
            source = (
                TranscriptSource.YOUTUBE_MANUAL
                if not transcript_data.is_auto_generated
                else TranscriptSource.YOUTUBE_AUTO
            )
            
            # Prepare metadata
            metadata = {
                "extracted_at": datetime.utcnow().isoformat(),
                "has_timestamps": transcript_data.has_timestamps
            }
            
            if metrics:
                metadata.update(metrics.to_metadata())
            
            # Create transcript record
            transcript = await repo.create(
                video_id=video.id,
                content=transcript_data.content,
                source=source,
                language=transcript_data.language,
                language_code=transcript_data.language_code,
                is_auto_generated=transcript_data.is_auto_generated,
                segments=[seg.dict() for seg in transcript_data.segments] if transcript_data.segments else None,
                metadata=metadata,
                confidence_score=transcript_data.confidence_score
            )
            
            # Store cleaned content
            if clean_content and cleaned_content != transcript_data.content:
                transcript.processed_content = cleaned_content
                await repo.update(transcript.id, processed_content=cleaned_content)
            
            return transcript
        
        return None
    
    async def _process_cached_transcript(
        self,
        video: Video,
        cached_data: TranscriptData,
        repo: TranscriptRepository,
        clean_content: bool,
        calculate_metrics: bool,
        save_to_db: bool
    ) -> Optional[Transcript]:
        """Process transcript from cache."""
        return await self._process_transcript(
            video,
            cached_data,
            repo,
            clean_content,
            calculate_metrics,
            save_to_db
        )
    
    async def _get_existing_transcript(
        self,
        video_id: UUID,
        repo: TranscriptRepository
    ) -> Optional[Transcript]:
        """Check for existing transcript."""
        return await repo.get_by_video(video_id)
    
    async def _get_from_cache(
        self,
        youtube_video_id: str
    ) -> Optional[TranscriptData]:
        """Get transcript from cache."""
        if not self.cache:
            return None
        
        try:
            return await self.cache.get(youtube_video_id)
        except Exception as e:
            logger.error(f"Cache error: {e}")
            return None
    
    async def _cache_transcript(
        self,
        youtube_video_id: str,
        transcript_data: TranscriptData
    ):
        """Cache transcript data."""
        if not self.cache:
            return
        
        try:
            await self.cache.set(youtube_video_id, transcript_data)
        except Exception as e:
            logger.error(f"Failed to cache transcript: {e}")
    
    async def _mark_video_completed(self, video: Video):
        """Mark video as completed."""
        video.status = VideoStatus.COMPLETED
        video.processed_at = datetime.utcnow()
        
        if self.session:
            await self.session.commit()
    
    async def _mark_video_failed(self, video: Video, error_message: str):
        """Mark video as failed."""
        video.status = VideoStatus.FAILED
        video.error_message = error_message
        video.retry_count += 1
        video.updated_at = datetime.utcnow()
        
        if self.session:
            await self.session.commit()
    
    async def _handle_extraction_error(self, video: Video, error: TranscriptError):
        """Handle transcript extraction error."""
        # Check if error is retryable
        if is_retryable_error(error):
            if video.retry_count < settings.TRANSCRIPT_MAX_RETRIES:
                video.status = VideoStatus.QUEUED
                video.retry_count += 1
                logger.info(f"Queued video {video.id} for retry (attempt {video.retry_count})")
            else:
                video.status = VideoStatus.FAILED
                logger.warning(f"Max retries reached for video {video.id}")
        else:
            # Non-retryable error
            if isinstance(error, (TranscriptDisabledError, TranscriptNotAvailableError)):
                video.status = VideoStatus.SKIPPED
                video.has_captions = False
            else:
                video.status = VideoStatus.FAILED
        
        video.error_message = str(error)
        video.updated_at = datetime.utcnow()
        
        if self.session:
            await self.session.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._stats,
            "api_stats": self.api_client.get_statistics(),
            "cache_stats": self.cache.get_cache_stats() if self.cache else None
        }