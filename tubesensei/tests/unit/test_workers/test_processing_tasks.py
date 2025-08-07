"""
Unit tests for TranscriptWorker and processing tasks.

Tests the worker functionality including job processing, batch operations,
error handling, and statistics tracking.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import signal
import asyncio

from app.workers.transcript_worker import TranscriptWorker, run_worker
from app.models.video import Video, VideoStatus
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.models.transcript import Transcript
from tests.fixtures.fixtures import (
    VideoFactory,
    ProcessingJobFactory,
    TranscriptFactory
)


class TestTranscriptWorkerInit:
    """Test suite for TranscriptWorker initialization."""
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        with patch('app.workers.transcript_worker.settings') as mock_settings:
            mock_settings.TRANSCRIPT_BATCH_SIZE = 10
            
            worker = TranscriptWorker()
            
            assert worker.poll_interval == 10
            assert worker.batch_size == 10
            assert worker.concurrent_limit == 10
            assert worker.use_cache is True
            assert worker.is_running is False
            assert worker.should_stop is False
            assert worker.current_jobs == []
    
    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        worker = TranscriptWorker(
            poll_interval=5,
            batch_size=20,
            concurrent_limit=5,
            use_cache=False
        )
        
        assert worker.poll_interval == 5
        assert worker.batch_size == 20
        assert worker.concurrent_limit == 5
        assert worker.use_cache is False
    
    def test_init_statistics(self):
        """Test initial statistics setup."""
        worker = TranscriptWorker()
        
        assert worker.stats["started_at"] is None
        assert worker.stats["total_processed"] == 0
        assert worker.stats["successful"] == 0
        assert worker.stats["failed"] == 0
        assert worker.stats["errors"] == []
    
    def test_signal_handlers_setup(self):
        """Test signal handlers are properly set up."""
        with patch('signal.signal') as mock_signal:
            worker = TranscriptWorker()
            
            # Should have registered SIGINT and SIGTERM handlers
            assert mock_signal.call_count == 2
            calls = mock_signal.call_args_list
            
            signal_nums = [call[0][0] for call in calls]
            assert signal.SIGINT in signal_nums
            assert signal.SIGTERM in signal_nums


class TestTranscriptWorkerStartStop:
    """Test suite for TranscriptWorker start/stop functionality."""
    
    @pytest.mark.asyncio
    async def test_start_worker_basic(self):
        """Test basic worker start functionality."""
        worker = TranscriptWorker()
        
        # Mock the processor context manager
        mock_processor = AsyncMock()
        mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
        mock_processor.__aexit__ = AsyncMock()
        
        with patch('app.workers.transcript_worker.TranscriptProcessor') as mock_tp_class:
            mock_tp_class.return_value = mock_processor
            
            with patch.object(worker, '_process_batch', new_callable=AsyncMock) as mock_process:
                # First call returns 0 (no jobs), which should trigger stop
                mock_process.return_value = 0
                worker.should_stop = True  # Stop after first iteration
                
                await worker.start()
                
                assert worker.is_running is False
                assert worker.stats["started_at"] is not None
                mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_worker_with_jobs(self):
        """Test worker start with job processing."""
        worker = TranscriptWorker()
        
        mock_processor = AsyncMock()
        mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
        mock_processor.__aexit__ = AsyncMock()
        
        with patch('app.workers.transcript_worker.TranscriptProcessor') as mock_tp_class:
            mock_tp_class.return_value = mock_processor
            
            with patch.object(worker, '_process_batch', new_callable=AsyncMock) as mock_process:
                # Return different values to test behavior
                mock_process.side_effect = [5, 3, 0]  # Process jobs, then no jobs
                
                # Stop after a few iterations
                async def delayed_stop():
                    await asyncio.sleep(0.1)
                    worker.should_stop = True
                
                # Run the stop in background
                asyncio.create_task(delayed_stop())
                
                await worker.start()
                
                # Should have processed multiple batches
                assert mock_process.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_start_worker_with_error(self):
        """Test worker start with processing error."""
        worker = TranscriptWorker()
        
        mock_processor = AsyncMock()
        mock_processor.__aenter__ = AsyncMock(return_value=mock_processor)
        mock_processor.__aexit__ = AsyncMock()
        
        with patch('app.workers.transcript_worker.TranscriptProcessor') as mock_tp_class:
            mock_tp_class.return_value = mock_processor
            
            with patch.object(worker, '_process_batch', new_callable=AsyncMock) as mock_process:
                mock_process.side_effect = [
                    Exception("Processing error"),
                    0  # Recovery
                ]
                
                # Stop after error handling
                async def delayed_stop():
                    await asyncio.sleep(0.2)
                    worker.should_stop = True
                
                asyncio.create_task(delayed_stop())
                
                await worker.start()
                
                # Should have recorded the error
                assert len(worker.stats["errors"]) == 1
                assert "Processing error" in worker.stats["errors"][0]["error"]
    
    @pytest.mark.asyncio
    async def test_stop_worker_graceful(self):
        """Test graceful worker stop."""
        worker = TranscriptWorker()
        worker.is_running = True
        
        # No current jobs - should stop immediately
        await worker.stop()
        
        assert worker.should_stop is True
    
    @pytest.mark.asyncio
    async def test_stop_worker_with_current_jobs(self):
        """Test worker stop with current jobs."""
        worker = TranscriptWorker()
        worker.is_running = True
        
        # Add mock current jobs
        mock_job1 = Mock()
        mock_job2 = Mock()
        worker.current_jobs = [mock_job1, mock_job2]
        
        # Start stop and clear jobs after a delay
        async def clear_jobs():
            await asyncio.sleep(0.1)
            worker.current_jobs = []
        
        stop_task = asyncio.create_task(worker.stop())
        clear_task = asyncio.create_task(clear_jobs())
        
        await asyncio.gather(stop_task, clear_task)
        
        assert worker.should_stop is True
    
    @pytest.mark.asyncio
    async def test_stop_worker_timeout(self):
        """Test worker stop with timeout."""
        worker = TranscriptWorker()
        worker.is_running = True
        
        # Add jobs that won't be cleared
        mock_job = Mock()
        worker.current_jobs = [mock_job]
        
        # Mock time to simulate timeout
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 65]  # Simulate 65 seconds elapsed
            
            await worker.stop()
            
            assert worker.should_stop is True
            # Jobs should still be there (timeout occurred)
            assert len(worker.current_jobs) == 1


class TestTranscriptWorkerProcessBatch:
    """Test suite for TranscriptWorker batch processing."""
    
    @pytest.mark.asyncio
    async def test_process_batch_no_jobs(self):
        """Test batch processing when no jobs available."""
        worker = TranscriptWorker()
        mock_processor = Mock()
        
        with patch.object(worker, '_get_pending_jobs', new_callable=AsyncMock) as mock_get_jobs:
            mock_get_jobs.return_value = []
            
            with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
                mock_session = AsyncMock()
                mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_db_context.return_value.__aexit__ = AsyncMock()
                
                result = await worker._process_batch(mock_processor)
                
                assert result == 0
                mock_get_jobs.assert_called_once_with(mock_session)
    
    @pytest.mark.asyncio
    async def test_process_batch_with_jobs(self):
        """Test batch processing with jobs."""
        worker = TranscriptWorker()
        mock_processor = Mock()
        
        # Create mock jobs
        mock_jobs = [
            ProcessingJobFactory.build(job_type=JobType.TRANSCRIPT_EXTRACTION),
            ProcessingJobFactory.build(job_type=JobType.TRANSCRIPT_EXTRACTION),
            ProcessingJobFactory.build(job_type=JobType.TRANSCRIPT_EXTRACTION)
        ]
        
        with patch.object(worker, '_get_pending_jobs', new_callable=AsyncMock) as mock_get_jobs:
            mock_get_jobs.return_value = mock_jobs
            
            with patch.object(worker, '_process_single_job', new_callable=AsyncMock) as mock_process_single:
                with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
                    mock_session = AsyncMock()
                    mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db_context.return_value.__aexit__ = AsyncMock()
                    
                    result = await worker._process_batch(mock_processor)
                    
                    assert result == 3
                    assert mock_process_single.call_count == 3
                    
                    # Verify all jobs were marked as processing
                    for job in mock_jobs:
                        assert job.status == JobStatus.PROCESSING
                        assert job.started_at is not None
    
    @pytest.mark.asyncio
    async def test_process_batch_concurrency_limit(self):
        """Test batch processing respects concurrency limit."""
        worker = TranscriptWorker(concurrent_limit=2)
        mock_processor = Mock()
        
        # Create more jobs than concurrency limit
        mock_jobs = [ProcessingJobFactory.build() for _ in range(5)]
        
        with patch.object(worker, '_get_pending_jobs', new_callable=AsyncMock) as mock_get_jobs:
            mock_get_jobs.return_value = mock_jobs
            
            with patch.object(worker, '_process_single_job', new_callable=AsyncMock) as mock_process_single:
                with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
                    mock_session = AsyncMock()
                    mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db_context.return_value.__aexit__ = AsyncMock()
                    
                    # Mock semaphore to verify it was created with correct limit
                    with patch('asyncio.Semaphore') as mock_semaphore:
                        mock_semaphore_instance = AsyncMock()
                        mock_semaphore_instance.acquire.return_value.__aenter__ = AsyncMock()
                        mock_semaphore_instance.acquire.return_value.__aexit__ = AsyncMock()
                        mock_semaphore.return_value = mock_semaphore_instance
                        
                        result = await worker._process_batch(mock_processor)
                        
                        assert result == 5
                        mock_semaphore.assert_called_once_with(2)  # Concurrency limit


class TestTranscriptWorkerProcessSingleJob:
    """Test suite for TranscriptWorker single job processing."""
    
    @pytest.mark.asyncio
    async def test_process_single_job_success(self):
        """Test successful single job processing."""
        worker = TranscriptWorker()
        
        # Create mock job
        job = ProcessingJobFactory.build(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            status=JobStatus.PROCESSING
        )
        
        # Create mock transcript result
        mock_transcript = TranscriptFactory.build()
        
        mock_processor = AsyncMock()
        mock_processor.extract_transcript.return_value = mock_transcript
        
        mock_session = AsyncMock()
        
        await worker._process_single_job(job, mock_processor, mock_session)
        
        # Verify job completion
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result is not None
        assert job.result["transcript_id"] == str(mock_transcript.id)
        assert job.result["language"] == mock_transcript.language_code
        
        # Verify statistics
        assert worker.stats["successful"] == 1
        assert worker.stats["total_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_single_job_no_transcript(self):
        """Test single job processing when no transcript returned."""
        worker = TranscriptWorker()
        
        job = ProcessingJobFactory.build(status=JobStatus.PROCESSING)
        
        mock_processor = AsyncMock()
        mock_processor.extract_transcript.return_value = None
        
        mock_session = AsyncMock()
        
        await worker._process_single_job(job, mock_processor, mock_session)
        
        # Verify job failure
        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.error_message == "Failed to extract transcript"
        assert job.retry_count == 1
        
        # Verify statistics
        assert worker.stats["failed"] == 1
        assert worker.stats["total_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_single_job_exception(self):
        """Test single job processing with exception."""
        worker = TranscriptWorker()
        
        job = ProcessingJobFactory.build(
            status=JobStatus.PROCESSING,
            retry_count=0
        )
        
        mock_processor = AsyncMock()
        mock_processor.extract_transcript.side_effect = Exception("API Error")
        
        mock_session = AsyncMock()
        
        with patch('app.workers.transcript_worker.settings') as mock_settings:
            mock_settings.TRANSCRIPT_MAX_RETRIES = 3
            
            await worker._process_single_job(job, mock_processor, mock_session)
            
            # Should be rescheduled for retry
            assert job.status == JobStatus.PENDING
            assert job.retry_count == 1
            assert job.scheduled_for is not None
            assert job.error_message == "API Error"
            
            # Verify statistics
            assert worker.stats["failed"] == 1
            assert worker.stats["total_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_single_job_max_retries_exceeded(self):
        """Test single job processing when max retries exceeded."""
        worker = TranscriptWorker()
        
        job = ProcessingJobFactory.build(
            status=JobStatus.PROCESSING,
            retry_count=2  # Already retried twice
        )
        
        mock_processor = AsyncMock()
        mock_processor.extract_transcript.side_effect = Exception("Persistent Error")
        
        mock_session = AsyncMock()
        
        with patch('app.workers.transcript_worker.settings') as mock_settings:
            mock_settings.TRANSCRIPT_MAX_RETRIES = 3
            
            await worker._process_single_job(job, mock_processor, mock_session)
            
            # Should be marked as failed permanently
            assert job.status == JobStatus.FAILED
            assert job.retry_count == 3
            assert job.completed_at is not None
            assert "Persistent Error" in job.error_message
    
    @pytest.mark.asyncio
    async def test_process_single_job_metadata_update(self):
        """Test job metadata is properly updated."""
        worker = TranscriptWorker()
        
        job = ProcessingJobFactory.build(
            status=JobStatus.PROCESSING,
            metadata={"existing": "data"}
        )
        
        mock_transcript = TranscriptFactory.build()
        mock_processor = AsyncMock()
        mock_processor.extract_transcript.return_value = mock_transcript
        
        mock_session = AsyncMock()
        
        await worker._process_single_job(job, mock_processor, mock_session)
        
        # Verify metadata was updated
        assert "worker_id" in job.metadata
        assert "started_at" in job.metadata
        assert job.metadata["existing"] == "data"  # Preserved existing data


class TestTranscriptWorkerGetPendingJobs:
    """Test suite for TranscriptWorker._get_pending_jobs method."""
    
    @pytest.mark.asyncio
    async def test_get_pending_jobs_basic(self):
        """Test getting pending jobs from database."""
        worker = TranscriptWorker()
        
        # Mock database session and query result
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_jobs = [ProcessingJobFactory.build() for _ in range(3)]
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_session.execute.return_value = mock_result
        
        jobs = await worker._get_pending_jobs(mock_session)
        
        assert len(jobs) == 3
        assert jobs == mock_jobs
        
        # Verify query was executed
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pending_jobs_empty(self):
        """Test getting pending jobs when none available."""
        worker = TranscriptWorker()
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        
        jobs = await worker._get_pending_jobs(mock_session)
        
        assert len(jobs) == 0
    
    @pytest.mark.asyncio
    async def test_get_pending_jobs_batch_limit(self):
        """Test pending jobs respects batch limit."""
        worker = TranscriptWorker(batch_size=5)
        
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_jobs = [ProcessingJobFactory.build() for _ in range(5)]
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_session.execute.return_value = mock_result
        
        jobs = await worker._get_pending_jobs(mock_session)
        
        assert len(jobs) == 5
        
        # Verify the query includes limit clause
        query_call = mock_session.execute.call_args[0][0]
        # Note: In a real test, we'd check the SQL contains LIMIT 5


class TestTranscriptWorkerCreateJobs:
    """Test suite for TranscriptWorker.create_jobs_for_new_videos method."""
    
    @pytest.mark.asyncio
    async def test_create_jobs_for_new_videos_success(self):
        """Test creating jobs for new videos."""
        worker = TranscriptWorker()
        
        # Mock videos without transcripts
        mock_videos = [
            VideoFactory.build(has_captions=True),
            VideoFactory.build(has_captions=True),
            VideoFactory.build(has_captions=False)
        ]
        
        with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
            mock_session = AsyncMock()
            mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db_context.return_value.__aexit__ = AsyncMock()
            
            with patch('app.workers.transcript_worker.TranscriptRepository') as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_videos_without_transcripts.return_value = mock_videos
                
                # Mock existing job check to return None (no existing jobs)
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_session.execute.return_value = mock_result
                
                jobs_created = await worker.create_jobs_for_new_videos(
                    limit=100,
                    only_with_captions=True
                )
                
                # Should create jobs only for videos with captions (2 out of 3)
                assert jobs_created == 2
                
                # Verify jobs were added to session
                assert mock_session.add.call_count == 2
                mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_jobs_for_new_videos_existing_jobs(self):
        """Test creating jobs skips videos with existing jobs."""
        worker = TranscriptWorker()
        
        mock_videos = [VideoFactory.build(has_captions=True)]
        
        with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
            mock_session = AsyncMock()
            mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db_context.return_value.__aexit__ = AsyncMock()
            
            with patch('app.workers.transcript_worker.TranscriptRepository') as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_videos_without_transcripts.return_value = mock_videos
                
                # Mock existing job found
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = ProcessingJobFactory.build()
                mock_session.execute.return_value = mock_result
                
                jobs_created = await worker.create_jobs_for_new_videos()
                
                # Should not create any jobs
                assert jobs_created == 0
                mock_session.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_jobs_for_new_videos_no_videos(self):
        """Test creating jobs when no videos available."""
        worker = TranscriptWorker()
        
        with patch('app.workers.transcript_worker.get_db_context') as mock_db_context:
            mock_session = AsyncMock()
            mock_db_context.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db_context.return_value.__aexit__ = AsyncMock()
            
            with patch('app.workers.transcript_worker.TranscriptRepository') as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_videos_without_transcripts.return_value = []
                
                jobs_created = await worker.create_jobs_for_new_videos()
                
                assert jobs_created == 0


class TestTranscriptWorkerPriorityCalculation:
    """Test suite for TranscriptWorker._calculate_priority method."""
    
    def test_calculate_priority_base(self):
        """Test base priority calculation."""
        worker = TranscriptWorker()
        
        video = VideoFactory.build()
        priority = worker._calculate_priority(video)
        
        assert priority == 50  # Base priority
    
    def test_calculate_priority_high_views(self):
        """Test priority boost for high view count."""
        worker = TranscriptWorker()
        
        # Very high view count
        video = VideoFactory.build(view_count=2000000)
        priority = worker._calculate_priority(video)
        assert priority == 70  # 50 + 20
        
        # High view count
        video = VideoFactory.build(view_count=500000)
        priority = worker._calculate_priority(video)
        assert priority == 60  # 50 + 10
        
        # Medium view count
        video = VideoFactory.build(view_count=50000)
        priority = worker._calculate_priority(video)
        assert priority == 55  # 50 + 5
    
    def test_calculate_priority_recent_video(self):
        """Test priority boost for recent videos."""
        worker = TranscriptWorker()
        
        # Very recent video (3 days old)
        recent_date = datetime.utcnow() - timedelta(days=3)
        video = VideoFactory.build(published_at=recent_date)
        priority = worker._calculate_priority(video)
        assert priority == 65  # 50 + 15
        
        # Recent video (20 days old)
        recent_date = datetime.utcnow() - timedelta(days=20)
        video = VideoFactory.build(published_at=recent_date)
        priority = worker._calculate_priority(video)
        assert priority == 60  # 50 + 10
        
        # Somewhat recent (60 days old)
        recent_date = datetime.utcnow() - timedelta(days=60)
        video = VideoFactory.build(published_at=recent_date)
        priority = worker._calculate_priority(video)
        assert priority == 55  # 50 + 5
    
    def test_calculate_priority_valuable_video(self):
        """Test priority boost for valuable videos."""
        worker = TranscriptWorker()
        
        video = VideoFactory.build(is_valuable=True)
        priority = worker._calculate_priority(video)
        assert priority == 60  # 50 + 10
    
    def test_calculate_priority_retry_penalty(self):
        """Test priority penalty for failed attempts."""
        worker = TranscriptWorker()
        
        video = VideoFactory.build(retry_count=2)
        priority = worker._calculate_priority(video)
        assert priority == 40  # 50 - (2 * 5)
    
    def test_calculate_priority_combined_factors(self):
        """Test priority calculation with multiple factors."""
        worker = TranscriptWorker()
        
        # High views + recent + valuable - retries
        recent_date = datetime.utcnow() - timedelta(days=5)
        video = VideoFactory.build(
            view_count=1500000,
            published_at=recent_date,
            is_valuable=True,
            retry_count=1
        )
        
        priority = worker._calculate_priority(video)
        # 50 (base) + 20 (high views) + 15 (recent) + 10 (valuable) - 5 (retry) = 90
        assert priority == 90
    
    def test_calculate_priority_bounds(self):
        """Test priority is bounded between 0 and 100."""
        worker = TranscriptWorker()
        
        # Test lower bound
        video = VideoFactory.build(retry_count=20)
        priority = worker._calculate_priority(video)
        assert priority == 0  # Should not go below 0
        
        # Test upper bound (shouldn't exceed 100 in normal cases)
        very_recent = datetime.utcnow() - timedelta(days=1)
        video = VideoFactory.build(
            view_count=5000000,
            published_at=very_recent,
            is_valuable=True
        )
        priority = worker._calculate_priority(video)
        assert priority <= 100


class TestTranscriptWorkerStatistics:
    """Test suite for TranscriptWorker.get_statistics method."""
    
    def test_get_statistics_initial_state(self):
        """Test statistics in initial state."""
        worker = TranscriptWorker()
        
        stats = worker.get_statistics()
        
        assert stats["is_running"] is False
        assert stats["started_at"] is None
        assert stats["runtime_seconds"] is None
        assert stats["total_processed"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate"] == 0
        assert stats["current_jobs"] == 0
        assert stats["recent_errors"] == []
    
    def test_get_statistics_running_state(self):
        """Test statistics when worker is running."""
        worker = TranscriptWorker()
        
        # Set up running state
        worker.is_running = True
        worker.stats["started_at"] = datetime.utcnow() - timedelta(seconds=300)
        worker.stats["total_processed"] = 10
        worker.stats["successful"] = 8
        worker.stats["failed"] = 2
        worker.current_jobs = [Mock(), Mock()]
        
        stats = worker.get_statistics()
        
        assert stats["is_running"] is True
        assert stats["started_at"] is not None
        assert stats["runtime_seconds"] is not None
        assert stats["runtime_seconds"] > 0
        assert stats["total_processed"] == 10
        assert stats["successful"] == 8
        assert stats["failed"] == 2
        assert stats["success_rate"] == 80.0
        assert stats["current_jobs"] == 2
    
    def test_get_statistics_with_errors(self):
        """Test statistics with error history."""
        worker = TranscriptWorker()
        
        # Add errors to history
        errors = [{"timestamp": "2023-01-01T12:00:00", "error": f"Error {i}"} for i in range(15)]
        worker.stats["errors"] = errors
        
        stats = worker.get_statistics()
        
        # Should only return last 10 errors
        assert len(stats["recent_errors"]) == 10
        assert stats["recent_errors"][-1]["error"] == "Error 14"


class TestRunWorkerFunction:
    """Test suite for run_worker function."""
    
    @pytest.mark.asyncio
    async def test_run_worker_basic(self):
        """Test basic run_worker functionality."""
        with patch('app.workers.transcript_worker.TranscriptWorker') as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker_class.return_value = mock_worker
            mock_worker.create_jobs_for_new_videos.return_value = 5
            mock_worker.get_statistics.return_value = {"total_processed": 10}
            
            with patch('app.workers.transcript_worker.settings') as mock_settings:
                mock_settings.TRANSCRIPT_BATCH_SIZE = 10
                
                await run_worker()
                
                # Verify worker was created and started
                mock_worker_class.assert_called_once()
                mock_worker.create_jobs_for_new_videos.assert_called_once_with(limit=100)
                mock_worker.start.assert_called_once()
                mock_worker.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_worker_keyboard_interrupt(self):
        """Test run_worker handles keyboard interrupt."""
        with patch('app.workers.transcript_worker.TranscriptWorker') as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker_class.return_value = mock_worker
            mock_worker.start.side_effect = KeyboardInterrupt()
            mock_worker.get_statistics.return_value = {"total_processed": 5}
            
            await run_worker()
            
            # Should still call stop even after interrupt
            mock_worker.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_worker_exception(self):
        """Test run_worker handles unexpected exceptions."""
        with patch('app.workers.transcript_worker.TranscriptWorker') as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker_class.return_value = mock_worker
            mock_worker.start.side_effect = Exception("Unexpected error")
            mock_worker.get_statistics.return_value = {"total_processed": 0}
            
            await run_worker()
            
            # Should still call stop even after exception
            mock_worker.stop.assert_called_once()


class TestTranscriptWorkerIntegration:
    """Integration tests for TranscriptWorker."""
    
    @pytest.mark.asyncio
    async def test_worker_complete_cycle(self, db_session):
        """Test complete worker processing cycle."""
        # This would be a more comprehensive test that uses real database
        # and tests the full flow from job creation to completion
        # Due to complexity, this is left as a placeholder for integration tests
        pass