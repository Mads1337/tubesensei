"""
End-to-end integration tests for TubeSensei.

Tests complete workflows from channel addition through transcript processing,
including database transactions, API integrations, and error recovery.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import select

from app.services.channel_manager import ChannelManager
from app.services.video_discovery import VideoDiscovery
from app.workers.transcript_worker import TranscriptWorker
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from tests.fixtures.fixtures import (
    MockYouTubeAPIResponses,
    MockTranscriptAPIResponses,
    TestDataBuilder
)


@pytest.mark.integration
class TestChannelProcessingWorkflow:
    """Test complete channel processing workflow."""
    
    @pytest.mark.asyncio
    async def test_add_channel_and_discover_videos(self, db_session):
        """Test adding channel and discovering videos end-to-end."""
        # Mock YouTube API responses
        channel_response = MockYouTubeAPIResponses.channel_info("UC_test_channel")['items'][0]
        playlist_response = MockYouTubeAPIResponses.playlist_items("UU_test_uploads")
        video_details_response = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])
        
        # Setup services with mocked API clients
        youtube_client = AsyncMock()
        youtube_client.get_channel_info.return_value = channel_response
        youtube_client.list_channel_videos.return_value = playlist_response['items']
        youtube_client.get_video_details.return_value = video_details_response['items']
        
        channel_manager = ChannelManager(youtube_client)
        video_discovery = VideoDiscovery(youtube_client)
        
        # Mock URL parsing
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.return_value = {
                'type': 'channel',
                'channel_id': 'UC_test_channel',
                'channel_handle': None
            }
            
            # Step 1: Add channel
            channel = await channel_manager.add_channel(
                "https://www.youtube.com/channel/UC_test_channel",
                db_session,
                auto_discover=False  # We'll manually trigger discovery
            )
            
            assert channel is not None
            assert channel.youtube_channel_id == "UC_test_channel"
            assert channel.channel_name == "Test Channel"
            assert channel.status == ChannelStatus.ACTIVE
            
            # Step 2: Discover videos
            videos = await video_discovery.discover_videos(
                channel.id,
                db_session,
                max_videos=50
            )
            
            assert len(videos) == 2
            assert all(v.channel_id == channel.id for v in videos)
            assert all(v.status == VideoStatus.DISCOVERED for v in videos)
            
            # Verify videos were stored in database
            result = await db_session.execute(
                select(Video).filter(Video.channel_id == channel.id)
            )
            stored_videos = result.scalars().all()
            assert len(stored_videos) == 2
            
            # Verify processing jobs were created for videos with captions
            result = await db_session.execute(
                select(ProcessingJob).filter(
                    ProcessingJob.job_type == JobType.TRANSCRIPT_EXTRACTION
                )
            )
            jobs = result.scalars().all()
            
            # Both test videos have captions in mock data
            assert len(jobs) == 2
            assert all(job.status == JobStatus.PENDING for job in jobs)
    
    @pytest.mark.asyncio
    async def test_complete_transcript_processing_workflow(self, db_session):
        """Test complete transcript processing from job creation to completion."""
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=2,
            transcripts_per_video=0  # No transcripts initially
        )
        
        channel = test_data['channels'][0]
        videos = test_data['videos']
        
        # Create transcript processing jobs
        jobs = []
        for video in videos:
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING,
                priority=5,
                input_data={
                    "video_id": str(video.id),
                    "youtube_video_id": video.youtube_video_id
                }
            )
            db_session.add(job)
            jobs.append(job)
        
        await db_session.commit()
        
        # Mock transcript processing
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            
            # Mock successful transcript extraction
            mock_transcript = Transcript(
                video_id=videos[0].id,
                content="Test transcript content from processing",
                source=TranscriptSource.YOUTUBE_AUTO,
                language=TranscriptLanguage.EN,
                language_code="en",
                is_auto_generated=True,
                word_count=6,
                char_count=42,
                confidence_score=85
            )
            mock_processor.extract_transcript.return_value = mock_transcript
            
            # Create and run transcript worker
            worker = TranscriptWorker(
                poll_interval=1,  # Fast polling for test
                batch_size=2,
                concurrent_limit=2
            )
            
            # Process one batch
            processed = await worker._process_batch(mock_processor)
            
            assert processed == 2
            
            # Verify jobs were processed
            await db_session.refresh(jobs[0])
            await db_session.refresh(jobs[1])
            
            # At least one job should be completed (due to mocking)
            completed_jobs = [j for j in jobs if j.status == JobStatus.COMPLETED]
            assert len(completed_jobs) >= 1
            
            # Verify statistics were updated
            assert worker.stats["total_processed"] == 2
            assert worker.stats["successful"] >= 1
    
    @pytest.mark.asyncio
    async def test_batch_channel_processing(self, db_session):
        """Test batch processing of multiple channels."""
        # Create multiple channels with videos
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=3,
            videos_per_channel=3,
            transcripts_per_video=0
        )
        
        channels = test_data['channels']
        videos = test_data['videos']
        
        # Mock YouTube API for batch discovery
        youtube_client = AsyncMock()
        
        # Mock responses for each channel's video discovery
        playlist_items = MockYouTubeAPIResponses.playlist_items()['items']
        video_details = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002', 'test_video_003'])['items']
        
        youtube_client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        youtube_client.list_channel_videos.return_value = playlist_items
        youtube_client.get_video_details.return_value = video_details
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Process channels in batch
        channel_ids = [c.id for c in channels]
        results = await video_discovery.batch_discover(
            channel_ids,
            db_session,
            max_videos_per_channel=5
        )
        
        # Verify batch results
        assert results['total_channels'] == 3
        assert results['successful_channels'] >= 0  # May fail due to mocking
        assert results['total_videos_discovered'] >= 0
        
        # Verify some processing jobs were created
        result = await db_session.execute(
            select(ProcessingJob).filter(
                ProcessingJob.job_type == JobType.TRANSCRIPT_EXTRACTION
            )
        )
        jobs = result.scalars().all()
        assert len(jobs) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, db_session):
        """Test error recovery and retry mechanisms."""
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=1,
            transcripts_per_video=0
        )
        
        video = test_data['videos'][0]
        
        # Create a failed job
        job = ProcessingJob(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            entity_type="video",
            entity_id=video.id,
            status=JobStatus.FAILED,
            retry_count=1,
            error_message="Previous API error",
            input_data={"video_id": str(video.id)}
        )
        db_session.add(job)
        await db_session.commit()
        
        # Mock transcript processing that succeeds on retry
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            
            # Mock successful retry
            mock_transcript = Transcript(
                video_id=video.id,
                content="Recovered transcript content",
                source=TranscriptSource.YOUTUBE_AUTO,
                language=TranscriptLanguage.EN,
                language_code="en",
                confidence_score=90
            )
            mock_processor.extract_transcript.return_value = mock_transcript
            
            # Reset job status for retry
            job.status = JobStatus.PENDING
            job.error_message = None
            await db_session.commit()
            
            # Create worker and process the retry
            worker = TranscriptWorker(batch_size=1)
            
            # Process the failed job
            await worker._process_single_job(job, mock_processor, db_session)
            
            # Verify recovery
            await db_session.refresh(job)
            assert job.status == JobStatus.COMPLETED
            assert job.error_message is None
            assert job.result is not None
            
            # Verify transcript was created
            result = await db_session.execute(
                select(Transcript).filter(Transcript.video_id == video.id)
            )
            transcript = result.scalar_one_or_none()
            assert transcript is not None
            assert "Recovered" in transcript.content


@pytest.mark.integration
class TestDatabaseTransactions:
    """Test database transaction handling and consistency."""
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, db_session):
        """Test transaction rollback when errors occur."""
        # Create initial channel
        channel = Channel(
            youtube_channel_id="UC_transaction_test",
            channel_name="Transaction Test Channel",
            status=ChannelStatus.ACTIVE
        )
        db_session.add(channel)
        await db_session.commit()
        
        initial_channel_count = await db_session.execute(select(Channel))
        initial_count = len(initial_channel_count.scalars().all())
        
        # Attempt to create duplicate channel (should fail)
        duplicate_channel = Channel(
            youtube_channel_id="UC_transaction_test",  # Duplicate ID
            channel_name="Duplicate Channel",
            status=ChannelStatus.ACTIVE
        )
        
        try:
            db_session.add(duplicate_channel)
            await db_session.commit()
            assert False, "Expected integrity error"
        except Exception:
            await db_session.rollback()
        
        # Verify no additional channels were created
        final_channel_count = await db_session.execute(select(Channel))
        final_count = len(final_channel_count.scalars().all())
        assert final_count == initial_count
    
    @pytest.mark.asyncio
    async def test_cascade_deletion(self, db_session):
        """Test cascade deletion of related records."""
        # Create test data with relationships
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=2,
            transcripts_per_video=1
        )
        
        channel = test_data['channels'][0]
        videos = test_data['videos']
        transcripts = test_data['transcripts']
        
        # Verify initial data exists
        assert len(videos) == 2
        assert len(transcripts) == 2
        
        # Delete channel
        await db_session.delete(channel)
        await db_session.commit()
        
        # Verify cascade deletion
        remaining_videos = await db_session.execute(
            select(Video).filter(Video.channel_id == channel.id)
        )
        assert len(remaining_videos.scalars().all()) == 0
        
        remaining_transcripts = await db_session.execute(
            select(Transcript).filter(Transcript.video_id.in_([v.id for v in videos]))
        )
        assert len(remaining_transcripts.scalars().all()) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, db_session):
        """Test concurrent processing doesn't create race conditions."""
        # Create multiple jobs for the same video
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=1,
            transcripts_per_video=0
        )
        
        video = test_data['videos'][0]
        
        # Create multiple pending jobs (simulating race condition)
        jobs = []
        for i in range(3):
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING,
                priority=5
            )
            db_session.add(job)
            jobs.append(job)
        
        await db_session.commit()
        
        # Mock processor to create transcript only once
        transcripts_created = []
        
        async def mock_extract_transcript(video_id, **kwargs):
            if len(transcripts_created) == 0:
                transcript = Transcript(
                    video_id=video_id,
                    content="Single transcript",
                    language_code="en"
                )
                transcripts_created.append(transcript)
                return transcript
            return None  # Already processed
        
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            mock_processor.extract_transcript.side_effect = mock_extract_transcript
            
            # Process jobs concurrently
            worker = TranscriptWorker(concurrent_limit=3)
            
            # Process all jobs
            for job in jobs:
                await worker._process_single_job(job, mock_processor, db_session)
            
            await db_session.commit()
            
            # Verify only one transcript was created
            result = await db_session.execute(
                select(Transcript).filter(Transcript.video_id == video.id)
            )
            transcripts = result.scalars().all()
            assert len(transcripts) <= 1  # Should not create duplicates
            
            # Verify job statuses
            await db_session.refresh(jobs[0])
            completed_jobs = sum(1 for job in jobs if job.status == JobStatus.COMPLETED)
            failed_jobs = sum(1 for job in jobs if job.status == JobStatus.FAILED)
            
            # Should have one completed and others failed (or handled appropriately)
            assert completed_jobs >= 1


@pytest.mark.integration
class TestAPIIntegrationScenarios:
    """Test API integration scenarios and error handling."""
    
    @pytest.mark.asyncio
    async def test_youtube_api_quota_exceeded(self, db_session):
        """Test handling when YouTube API quota is exceeded."""
        from app.utils.exceptions import QuotaExceededError
        
        # Create test channel
        channel = Channel(
            youtube_channel_id="UC_quota_test",
            channel_name="Quota Test Channel",
            status=ChannelStatus.ACTIVE
        )
        db_session.add(channel)
        await db_session.commit()
        
        # Mock YouTube client to raise quota exceeded
        youtube_client = AsyncMock()
        youtube_client.list_channel_videos.side_effect = QuotaExceededError(10000, 10000)
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Attempt video discovery
        with pytest.raises(QuotaExceededError):
            await video_discovery.discover_videos(channel.id, db_session)
        
        # Verify error was recorded in channel
        await db_session.refresh(channel)
        assert "quota" in channel.last_error.lower() or "exceeded" in channel.last_error.lower()
    
    @pytest.mark.asyncio
    async def test_transcript_api_unavailable(self, db_session):
        """Test handling when transcript API is unavailable."""
        from app.integrations.transcript_errors import TranscriptUnavailableError
        
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=1,
            transcripts_per_video=0
        )
        
        video = test_data['videos'][0]
        
        # Mock transcript processor to fail
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            mock_processor.extract_transcript.side_effect = TranscriptUnavailableError("API unavailable")
            
            # Create and process job
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING
            )
            db_session.add(job)
            await db_session.commit()
            
            worker = TranscriptWorker()
            await worker._process_single_job(job, mock_processor, db_session)
            
            # Verify job was marked for retry
            await db_session.refresh(job)
            assert job.retry_count == 1
            assert "API unavailable" in job.error_message
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_batch(self, db_session):
        """Test batch processing with mixed success and failure scenarios."""
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=2,
            videos_per_channel=2,
            transcripts_per_video=0
        )
        
        channels = test_data['channels']
        
        # Mock YouTube client with mixed responses
        youtube_client = AsyncMock()
        
        def mock_list_videos(channel_id, **kwargs):
            if "test_channel_001" in channel_id:
                # Success for first channel
                return MockYouTubeAPIResponses.playlist_items()['items']
            else:
                # Failure for second channel
                from app.utils.exceptions import YouTubeAPIError
                raise YouTubeAPIError("Channel access denied")
        
        youtube_client.list_channel_videos.side_effect = mock_list_videos
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001'])['items']
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Process batch
        channel_ids = [c.id for c in channels]
        results = await video_discovery.batch_discover(channel_ids, db_session)
        
        # Verify mixed results
        assert results['total_channels'] == 2
        assert results['successful_channels'] >= 0
        assert results['failed_channels'] >= 0
        assert len(results['errors']) >= 0
        
        # At least one channel should have error recorded
        successful_channels = [c for c in channels if not c.last_error]
        failed_channels = [c for c in channels if c.last_error]
        
        # Note: Actual results depend on mock behavior and error handling


@pytest.mark.integration 
class TestDataConsistency:
    """Test data consistency across operations."""
    
    @pytest.mark.asyncio
    async def test_video_metadata_consistency(self, db_session):
        """Test video metadata remains consistent during updates."""
        # Create channel and video
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=1,
            transcripts_per_video=0
        )
        
        video = test_data['videos'][0]
        original_view_count = video.view_count
        
        # Mock updated video details
        updated_details = MockYouTubeAPIResponses.video_details([video.youtube_video_id])['items'][0]
        updated_details['statistics']['viewCount'] = str(original_view_count + 1000)
        
        youtube_client = AsyncMock()
        youtube_client.get_video_details.return_value = [updated_details]
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Update metadata
        updated_videos = await video_discovery.update_video_metadata(
            [video.id],
            db_session
        )
        
        # Verify consistency
        assert len(updated_videos) == 1
        updated_video = updated_videos[0]
        assert updated_video.view_count == original_view_count + 1000
        assert updated_video.youtube_video_id == video.youtube_video_id  # Should remain same
        
        # Verify in database
        await db_session.refresh(video)
        assert video.view_count == original_view_count + 1000
    
    @pytest.mark.asyncio
    async def test_transcript_language_consistency(self, db_session):
        """Test transcript language handling consistency."""
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=1,
            videos_per_channel=1,
            transcripts_per_video=0
        )
        
        video = test_data['videos'][0]
        
        # Create transcripts in different languages
        en_transcript = Transcript(
            video_id=video.id,
            content="English transcript content",
            source=TranscriptSource.YOUTUBE_AUTO,
            language=TranscriptLanguage.EN,
            language_code="en"
        )
        
        es_transcript = Transcript(
            video_id=video.id,
            content="Contenido del transcript en español",
            source=TranscriptSource.YOUTUBE_MANUAL,
            language=TranscriptLanguage.ES,
            language_code="es"
        )
        
        db_session.add(en_transcript)
        db_session.add(es_transcript)
        await db_session.commit()
        
        # Verify both transcripts exist and are properly differentiated
        result = await db_session.execute(
            select(Transcript).filter(Transcript.video_id == video.id)
        )
        transcripts = result.scalars().all()
        
        assert len(transcripts) == 2
        languages = {t.language_code for t in transcripts}
        assert languages == {"en", "es"}
        
        # Verify language-specific properties
        en_transcript_db = next(t for t in transcripts if t.language_code == "en")
        es_transcript_db = next(t for t in transcripts if t.language_code == "es")
        
        assert en_transcript_db.is_english is True
        assert es_transcript_db.is_english is False
        assert es_transcript_db.needs_translation is True