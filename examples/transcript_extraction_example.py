#!/usr/bin/env python3
"""
Example script demonstrating transcript extraction and processing.

This script shows how to:
1. Extract transcripts from YouTube videos
2. Clean and analyze transcript content
3. Store transcripts in the database with caching
4. Process multiple videos in batch
"""

import asyncio
import logging
from datetime import datetime
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def extract_single_transcript():
    """Example: Extract transcript for a single video."""
    from app.integrations.transcript_api import TranscriptAPIClient
    from app.services.transcript_cleaner import TranscriptCleaner
    from app.services.transcript_analyzer import TranscriptAnalyzer
    
    # Initialize components
    api_client = TranscriptAPIClient()
    cleaner = TranscriptCleaner()
    analyzer = TranscriptAnalyzer()
    
    # Example YouTube video ID (replace with actual video ID)
    youtube_video_id = "dQw4w9WgXcQ"  # Example ID
    
    try:
        # Extract transcript
        logger.info(f"Extracting transcript for video: {youtube_video_id}")
        transcript_data = await api_client.get_transcript(youtube_video_id)
        
        if transcript_data:
            logger.info(f"Successfully extracted transcript:")
            logger.info(f"  - Language: {transcript_data.language_code}")
            logger.info(f"  - Word count: {transcript_data.word_count}")
            logger.info(f"  - Auto-generated: {transcript_data.is_auto_generated}")
            logger.info(f"  - Confidence: {transcript_data.confidence_score:.2f}")
            
            # Clean the transcript
            cleaned_content = cleaner.clean_transcript(transcript_data.content)
            logger.info(f"Cleaned transcript (removed {len(transcript_data.content) - len(cleaned_content)} chars)")
            
            # Analyze the transcript
            metrics = analyzer.analyze_transcript(
                cleaned_content,
                is_auto_generated=transcript_data.is_auto_generated
            )
            logger.info(f"Analysis results:")
            logger.info(f"  - Quality score: {metrics.quality_score:.2f}")
            logger.info(f"  - Sentences: {metrics.sentence_count}")
            logger.info(f"  - Unique words: {metrics.unique_words}")
            logger.info(f"  - Reading level: {metrics.reading_level}")
            
            # Print first 500 characters of cleaned transcript
            logger.info(f"Transcript preview:")
            logger.info(cleaned_content[:500] + "...")
            
        else:
            logger.warning("No transcript available for this video")
            
    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")


async def process_with_cache():
    """Example: Process transcript with caching."""
    from app.services.transcript_cache import TranscriptCache
    from app.integrations.transcript_api import TranscriptAPIClient
    
    # Initialize components
    cache = TranscriptCache()
    api_client = TranscriptAPIClient()
    
    youtube_video_id = "dQw4w9WgXcQ"  # Example ID
    
    try:
        # Connect to cache
        await cache.connect()
        
        # Check cache first
        logger.info(f"Checking cache for video: {youtube_video_id}")
        cached_data = await cache.get(youtube_video_id)
        
        if cached_data:
            logger.info("Found in cache! Using cached transcript")
            logger.info(f"  - Cached word count: {cached_data.word_count}")
        else:
            logger.info("Not in cache, extracting from YouTube...")
            
            # Extract from YouTube
            transcript_data = await api_client.get_transcript(youtube_video_id)
            
            if transcript_data:
                # Store in cache
                await cache.set(youtube_video_id, transcript_data)
                logger.info("Transcript cached successfully")
                
                # Get cache statistics
                stats = await cache.get_cache_stats()
                logger.info(f"Cache statistics:")
                logger.info(f"  - Total cached: {stats['total_cached']}")
                logger.info(f"  - Hit rate: {stats['hit_rate']}%")
                logger.info(f"  - Memory used: {stats['memory_used_mb']} MB")
        
    finally:
        await cache.disconnect()


async def batch_processing_example():
    """Example: Process multiple videos in batch."""
    from app.services.transcript_processor import TranscriptProcessor
    from app.database import get_db_context
    from app.models.video import Video
    
    # Initialize processor
    async with TranscriptProcessor(use_cache=True) as processor:
        
        # Example video IDs (would normally come from database)
        video_ids = [
            uuid4(),  # Replace with actual video IDs from your database
            uuid4(),
            uuid4()
        ]
        
        logger.info(f"Processing batch of {len(video_ids)} videos...")
        
        # Process in batch
        result = await processor.batch_process_transcripts(
            video_ids=video_ids,
            concurrent_limit=3,
            skip_existing=True
        )
        
        # Display results
        logger.info(f"Batch processing complete:")
        logger.info(f"  - Total processed: {result.total_processed}")
        logger.info(f"  - Successful: {result.successful}")
        logger.info(f"  - Failed: {result.failed}")
        logger.info(f"  - Success rate: {result.success_rate:.1f}%")
        logger.info(f"  - Processing time: {result.processing_time_seconds:.2f}s")
        
        if result.errors:
            logger.warning(f"Errors encountered:")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error.video_id}: {error.error_message}")


async def worker_example():
    """Example: Run transcript worker for background processing."""
    from app.workers.transcript_worker import TranscriptWorker
    
    # Create worker
    worker = TranscriptWorker(
        poll_interval=10,
        batch_size=5,
        concurrent_limit=3,
        use_cache=True
    )
    
    # Create jobs for new videos
    logger.info("Creating transcript extraction jobs...")
    jobs_created = await worker.create_jobs_for_new_videos(
        limit=10,
        only_with_captions=True
    )
    logger.info(f"Created {jobs_created} new jobs")
    
    # Get worker statistics
    stats = worker.get_statistics()
    logger.info(f"Worker statistics:")
    logger.info(f"  - Is running: {stats['is_running']}")
    logger.info(f"  - Total processed: {stats['total_processed']}")
    logger.info(f"  - Success rate: {stats['success_rate']:.1f}%")
    
    # Note: To actually run the worker, you would call:
    # await worker.start()
    # But this would run indefinitely, so we skip it in this example


async def analyze_existing_transcripts():
    """Example: Analyze quality of existing transcripts."""
    from app.database import get_db_context
    from app.repositories.transcript_repository import TranscriptRepository
    from app.services.transcript_processor import TranscriptProcessor
    
    async with get_db_context() as session:
        repo = TranscriptRepository(session)
        processor = TranscriptProcessor(session=session)
        
        # Get low quality transcripts
        logger.info("Finding low quality transcripts...")
        low_quality = await repo.get_low_quality_transcripts(
            quality_threshold=0.5,
            limit=10
        )
        
        logger.info(f"Found {len(low_quality)} low quality transcripts")
        
        # Update quality metrics for each
        for transcript in low_quality[:3]:  # Process first 3
            logger.info(f"Updating quality for transcript {transcript.id}")
            updated = await processor.update_transcript_quality(transcript.id)
            
            if updated:
                logger.info(f"  - New quality score: {updated.confidence_score/100:.2f}")
        
        # Get overall statistics
        stats = await repo.get_statistics()
        logger.info(f"Overall transcript statistics:")
        logger.info(f"  - Total transcripts: {stats['total_transcripts']}")
        logger.info(f"  - Auto-generated: {stats['auto_generated_count']}")
        logger.info(f"  - Manual: {stats['manual_count']}")
        logger.info(f"  - Average quality: {stats['average_quality_score']:.2f}" if stats['average_quality_score'] else "  - Average quality: N/A")
        logger.info(f"  - Total words: {stats['word_statistics']['total_words']:,}")


async def main():
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("TubeSensei Transcript Processing Examples")
    logger.info("=" * 60)
    
    # Example 1: Extract single transcript
    logger.info("\n1. Extracting single transcript...")
    await extract_single_transcript()
    
    # Example 2: Process with cache
    logger.info("\n2. Processing with cache...")
    await process_with_cache()
    
    # Example 3: Batch processing
    logger.info("\n3. Batch processing example...")
    # Note: This requires actual video IDs in database
    # await batch_processing_example()
    
    # Example 4: Worker example
    logger.info("\n4. Worker example...")
    # Note: This requires database setup
    # await worker_example()
    
    # Example 5: Analyze existing transcripts
    logger.info("\n5. Analyzing existing transcripts...")
    # Note: This requires existing transcripts in database
    # await analyze_existing_transcripts()
    
    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())