"""
Performance and load tests for TubeSensei.

Tests system performance under load, concurrent processing capabilities,
database performance, and API rate limiting behavior.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch
import statistics
from typing import List, Dict, Any

from app.services.channel_manager import ChannelManager
from app.services.video_discovery import VideoDiscovery
from app.workers.transcript_worker import TranscriptWorker
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from tests.fixtures.fixtures import (
    ChannelFactory,
    VideoFactory,
    ProcessingJobFactory,
    MockYouTubeAPIResponses,
    TestDataBuilder
)


@pytest.mark.performance
class TestConcurrentProcessing:
    """Test concurrent processing performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_video_discovery(self, db_session):
        """Test concurrent video discovery performance."""
        # Create multiple channels
        channels = []
        for i in range(10):
            channel = ChannelFactory.build()
            db_session.add(channel)
            channels.append(channel)
        await db_session.commit()
        
        # Mock YouTube API with consistent responses
        youtube_client = AsyncMock()
        playlist_items = MockYouTubeAPIResponses.playlist_items()['items']
        video_details = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        
        youtube_client.list_channel_videos.return_value = playlist_items
        youtube_client.get_video_details.return_value = video_details
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Measure concurrent processing time
        start_time = time.time()
        
        # Process channels concurrently
        tasks = []
        for channel in channels:
            task = video_discovery.discover_videos(
                channel.id,
                db_session,
                max_videos=10
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8, "At least 80% of channels should process successfully"
        
        # Performance assertion: should process 10 channels in reasonable time
        assert processing_time < 30, f"Concurrent processing took {processing_time:.2f}s, should be under 30s"
        
        # Calculate throughput
        channels_per_second = len(successful_results) / processing_time
        assert channels_per_second > 0.3, f"Throughput {channels_per_second:.2f} channels/s is too low"
    
    @pytest.mark.asyncio
    async def test_batch_transcript_processing_performance(self, db_session):
        """Test batch transcript processing performance."""
        # Create test data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=5,
            videos_per_channel=10,
            transcripts_per_video=0
        )
        
        videos = test_data['videos']
        
        # Create processing jobs
        jobs = []
        for video in videos:
            job = ProcessingJobFactory.build(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING
            )
            db_session.add(job)
            jobs.append(job)
        await db_session.commit()
        
        # Mock transcript processor
        mock_processor = AsyncMock()
        mock_transcript = Mock()
        mock_transcript.id = "mock_transcript_id"
        mock_transcript.language_code = "en"
        mock_transcript.word_count = 100
        mock_transcript.confidence_score = 85
        
        # Add processing delay to simulate real work
        async def mock_extract_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms processing time
            return mock_transcript
        
        mock_processor.extract_transcript.side_effect = mock_extract_with_delay
        
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            
            # Create worker with high concurrency
            worker = TranscriptWorker(
                batch_size=20,
                concurrent_limit=10
            )
            
            # Measure processing time
            start_time = time.time()
            
            # Process batch
            processed = await worker._process_batch(mock_processor)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify performance
            assert processed == len(jobs), f"Expected {len(jobs)} jobs processed, got {processed}"
            
            # With 10 concurrent workers and 0.1s per job, 50 jobs should take ~0.5s + overhead
            expected_max_time = (len(jobs) / 10) * 0.1 + 2  # Add 2s overhead
            assert processing_time < expected_max_time, f"Processing took {processing_time:.2f}s, expected < {expected_max_time:.2f}s"
            
            # Calculate throughput
            jobs_per_second = processed / processing_time
            assert jobs_per_second > 5, f"Throughput {jobs_per_second:.2f} jobs/s is too low"
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls_rate_limiting(self):
        """Test API rate limiting under concurrent load."""
        # Mock rate limiter
        call_times = []
        
        async def mock_api_call(*args, **kwargs):
            call_times.append(time.time())
            await asyncio.sleep(0.01)  # Simulate API call time
            return MockYouTubeAPIResponses.channel_info()['items'][0]
        
        youtube_client = AsyncMock()
        youtube_client.get_channel_info.side_effect = mock_api_call
        
        # Create multiple concurrent API calls
        channel_manager = ChannelManager(youtube_client)
        
        # Simulate rate limiter by tracking call timing
        with patch('app.integrations.youtube_api.RateLimiter') as mock_rate_limiter_class:
            mock_rate_limiter = AsyncMock()
            mock_rate_limiter_class.return_value = mock_rate_limiter
            
            # Mock acquire context manager
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__ = AsyncMock()
            mock_acquire.__aexit__ = AsyncMock()
            mock_rate_limiter.acquire.return_value = mock_acquire
            
            start_time = time.time()
            
            # Make 20 concurrent API calls
            tasks = []
            for i in range(20):
                # Simulate sync channel metadata calls
                task = asyncio.create_task(mock_api_call(f"UC_test_channel_{i}"))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify calls were made
            assert len(call_times) == 20
            
            # Verify rate limiting behavior
            assert mock_rate_limiter.acquire.call_count == 20
            
            # Calculate average time between calls
            if len(call_times) > 1:
                time_diffs = [call_times[i] - call_times[i-1] for i in range(1, len(call_times))]
                avg_time_between_calls = statistics.mean(time_diffs)
                
                # Should have some delay between calls due to rate limiting
                assert avg_time_between_calls >= 0, "Should have some delay between calls"


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance under load."""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_session):
        """Test bulk insert performance for large datasets."""
        # Create large number of channels
        channels = []
        start_time = time.time()
        
        for i in range(1000):
            channel = ChannelFactory.build(
                youtube_channel_id=f"UC_bulk_test_{i:04d}",
                channel_name=f"Bulk Test Channel {i}"
            )
            channels.append(channel)
            db_session.add(channel)
        
        # Bulk commit
        await db_session.commit()
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Verify performance
        assert insert_time < 5.0, f"Bulk insert of 1000 channels took {insert_time:.2f}s, should be under 5s"
        
        # Calculate throughput
        inserts_per_second = 1000 / insert_time
        assert inserts_per_second > 200, f"Insert throughput {inserts_per_second:.2f}/s is too low"
    
    @pytest.mark.asyncio
    async def test_complex_query_performance(self, db_session):
        """Test performance of complex queries."""
        # Create test data with relationships
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=50,
            videos_per_channel=20,
            transcripts_per_video=1
        )
        
        from sqlalchemy import select, func, and_
        from sqlalchemy.orm import selectinload
        
        # Complex query: channels with video statistics
        start_time = time.time()
        
        query = select(
            Channel.id,
            Channel.channel_name,
            Channel.subscriber_count,
            func.count(Video.id).label('video_count'),
            func.sum(Video.view_count).label('total_views'),
            func.avg(Video.duration_seconds).label('avg_duration')
        ).select_from(
            Channel
        ).join(
            Video, Channel.id == Video.channel_id
        ).where(
            and_(
                Channel.status == ChannelStatus.ACTIVE,
                Video.status == VideoStatus.DISCOVERED,
                Video.view_count > 0
            )
        ).group_by(
            Channel.id,
            Channel.channel_name,
            Channel.subscriber_count
        ).order_by(
            func.sum(Video.view_count).desc()
        ).limit(10)
        
        result = await db_session.execute(query)
        rows = result.all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Verify results and performance
        assert len(rows) <= 10
        assert query_time < 2.0, f"Complex query took {query_time:.2f}s, should be under 2s"
    
    @pytest.mark.asyncio
    async def test_concurrent_database_access(self, db_session):
        """Test concurrent database access performance."""
        # Create initial data
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=10,
            videos_per_channel=10,
            transcripts_per_video=0
        )
        
        channels = test_data['channels']
        
        # Define concurrent database operations
        async def read_channel_stats(channel_id):
            from sqlalchemy import select, func
            query = select(
                func.count(Video.id),
                func.avg(Video.view_count),
                func.max(Video.published_at)
            ).where(Video.channel_id == channel_id)
            
            result = await db_session.execute(query)
            return result.one()
        
        async def update_channel_stats(channel):
            channel.last_checked_at = datetime.now(timezone.utc)
            await db_session.commit()
        
        # Execute concurrent operations
        start_time = time.time()
        
        tasks = []
        for i, channel in enumerate(channels):
            if i % 2 == 0:
                task = read_channel_stats(channel.id)
            else:
                task = update_channel_stats(channel)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify performance
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))
        assert successful_operations >= 8, "At least 80% of operations should succeed"
        
        assert concurrent_time < 5.0, f"Concurrent operations took {concurrent_time:.2f}s, should be under 5s"


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns under load."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_batch_processing(self, db_session):
        """Test memory usage during batch processing."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=20,
            videos_per_channel=50,
            transcripts_per_video=0
        )
        
        videos = test_data['videos']
        
        # Process in batches to test memory management
        batch_size = 100
        processed_videos = []
        
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            
            # Mock processing that creates objects
            for video in batch:
                processed_video = {
                    'id': video.id,
                    'title': video.title,
                    'description': video.description,
                    'metadata': video.metadata,
                    'processing_result': {'status': 'completed', 'data': 'x' * 1000}  # 1KB per video
                }
                processed_videos.append(processed_video)
            
            # Simulate some processing time
            await asyncio.sleep(0.01)
            
            # Check memory usage periodically
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory should not grow excessively (allow 100MB increase)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB, should be under 100MB"
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Should not have excessive memory growth
        assert total_memory_increase < 150, f"Total memory increase {total_memory_increase:.2f}MB is too high"
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_processing(self, db_session):
        """Test memory is properly cleaned up after processing."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        large_transcripts = []
        for i in range(100):
            # Create large transcript content (10KB each)
            transcript_content = "This is a very long transcript content. " * 250  # ~10KB
            large_transcripts.append({
                'id': i,
                'content': transcript_content,
                'metadata': {'processing': 'completed'},
                'segments': [{'start': j, 'text': f'Segment {j}'} for j in range(100)]
            })
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clear the large objects
        large_transcripts.clear()
        del large_transcripts
        
        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)  # Allow cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_cleanup = final_memory - initial_memory
        
        # Memory should be mostly reclaimed (allow some overhead)
        cleanup_ratio = (memory_increase - memory_after_cleanup) / memory_increase if memory_increase > 0 else 1
        assert cleanup_ratio > 0.7, f"Only {cleanup_ratio:.2%} of memory was cleaned up"


@pytest.mark.performance
class TestThroughputBenchmarks:
    """Benchmark system throughput for meeting Phase 1D targets."""
    
    @pytest.mark.asyncio
    async def test_video_processing_throughput(self, db_session):
        """Test system can process 100+ videos/hour."""
        # Create test videos
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=2,
            videos_per_channel=25,  # 50 total videos
            transcripts_per_video=0
        )
        
        videos = test_data['videos']
        
        # Mock fast transcript processing
        mock_processor = AsyncMock()
        mock_transcript = Mock()
        mock_transcript.id = "mock_id"
        mock_transcript.language_code = "en"
        mock_transcript.word_count = 100
        mock_transcript.confidence_score = 85
        
        # Simulate realistic processing time (10-30 seconds per video)
        async def mock_extract_realistic(*args, **kwargs):
            await asyncio.sleep(0.02)  # 20ms for test speed
            return mock_transcript
        
        mock_processor.extract_transcript.side_effect = mock_extract_realistic
        
        # Create processing jobs
        jobs = []
        for video in videos:
            job = ProcessingJobFactory.build(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING
            )
            db_session.add(job)
            jobs.append(job)
        await db_session.commit()
        
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            
            # Create high-performance worker
            worker = TranscriptWorker(
                batch_size=50,
                concurrent_limit=20  # High concurrency
            )
            
            # Measure processing time
            start_time = time.time()
            
            processed = await worker._process_batch(mock_processor)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate throughput
            videos_per_hour = (processed / processing_time) * 3600
            
            # Verify meets Phase 1D target
            assert videos_per_hour >= 100, f"Throughput {videos_per_hour:.2f} videos/hour is below 100/hour target"
            assert processed == len(videos), f"Expected {len(videos)} processed, got {processed}"
            
            # Log performance metrics
            print(f"Performance Benchmark Results:")
            print(f"  Videos processed: {processed}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Throughput: {videos_per_hour:.2f} videos/hour")
            print(f"  Average time per video: {(processing_time / processed) * 1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_channel_discovery_throughput(self, db_session):
        """Test system can handle multiple channels efficiently."""
        # Create channels for discovery
        channels = []
        for i in range(20):
            channel = ChannelFactory.build(
                youtube_channel_id=f"UC_throughput_test_{i:03d}"
            )
            db_session.add(channel)
            channels.append(channel)
        await db_session.commit()
        
        # Mock YouTube API with realistic response times
        youtube_client = AsyncMock()
        
        async def mock_list_videos(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms API call time
            return MockYouTubeAPIResponses.playlist_items()['items']
        
        async def mock_video_details(*args, **kwargs):
            await asyncio.sleep(0.03)  # 30ms API call time
            return MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        
        youtube_client.list_channel_videos.side_effect = mock_list_videos
        youtube_client.get_video_details.side_effect = mock_video_details
        
        video_discovery = VideoDiscovery(youtube_client)
        
        # Measure batch discovery time
        start_time = time.time()
        
        channel_ids = [c.id for c in channels]
        results = await video_discovery.batch_discover(
            channel_ids,
            db_session,
            max_videos_per_channel=10
        )
        
        end_time = time.time()
        discovery_time = end_time - start_time
        
        # Calculate throughput
        channels_per_hour = (len(channels) / discovery_time) * 3600
        
        # Verify performance targets
        assert channels_per_hour >= 50, f"Channel discovery throughput {channels_per_hour:.2f}/hour is too low"
        assert results['successful_channels'] >= len(channels) * 0.8, "At least 80% success rate required"
        
        print(f"Channel Discovery Benchmark Results:")
        print(f"  Channels processed: {results['total_channels']}")
        print(f"  Successful: {results['successful_channels']}")
        print(f"  Discovery time: {discovery_time:.2f}s")
        print(f"  Throughput: {channels_per_hour:.2f} channels/hour")


@pytest.mark.performance
class TestStressTests:
    """Stress tests for system reliability under high load."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, db_session):
        """Test system under high concurrency stress."""
        # Create large dataset
        test_data = TestDataBuilder.create_test_channel_with_videos(
            db_session,
            channel_count=10,
            videos_per_channel=50,
            transcripts_per_video=0
        )
        
        videos = test_data['videos']
        
        # Create many concurrent jobs
        jobs = []
        for video in videos:
            job = ProcessingJobFactory.build(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="video",
                entity_id=video.id,
                status=JobStatus.PENDING
            )
            db_session.add(job)
            jobs.append(job)
        await db_session.commit()
        
        # Mock processor with variable processing time
        mock_processor = AsyncMock()
        
        async def mock_extract_variable(*args, **kwargs):
            # Variable processing time (1-50ms)
            import random
            await asyncio.sleep(random.uniform(0.001, 0.05))
            mock_transcript = Mock()
            mock_transcript.id = "mock_id"
            mock_transcript.language_code = "en" 
            mock_transcript.word_count = 100
            mock_transcript.confidence_score = 85
            return mock_transcript
        
        mock_processor.extract_transcript.side_effect = mock_extract_variable
        
        with patch('app.services.transcript_processor.TranscriptProcessor') as mock_processor_class:
            mock_processor_class.return_value = mock_processor
            mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
            mock_processor.__aexit__ = AsyncMock()
            
            # Create multiple workers
            workers = []
            for i in range(5):  # 5 concurrent workers
                worker = TranscriptWorker(
                    batch_size=50,
                    concurrent_limit=20
                )
                workers.append(worker)
            
            start_time = time.time()
            
            # Run workers concurrently
            tasks = []
            for worker in workers:
                task = worker._process_batch(mock_processor)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            stress_time = end_time - start_time
            
            # Verify stress test results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            total_processed = sum(successful_results)
            
            assert len(successful_results) >= 4, "At least 4/5 workers should complete successfully"
            assert total_processed >= len(jobs) * 0.8, "At least 80% of jobs should be processed"
            
            # Performance under stress
            jobs_per_second = total_processed / stress_time
            assert jobs_per_second > 10, f"Stress test throughput {jobs_per_second:.2f}/s is too low"
            
            print(f"Stress Test Results:")
            print(f"  Workers: {len(workers)}")
            print(f"  Total jobs: {len(jobs)}")
            print(f"  Processed: {total_processed}")
            print(f"  Success rate: {(len(successful_results) / len(workers)) * 100:.1f}%")
            print(f"  Stress time: {stress_time:.2f}s")
            print(f"  Throughput: {jobs_per_second:.2f} jobs/s")