"""
Unit tests for ProcessingJob model.

Tests the ProcessingJob model including properties, methods, validations,
and edge cases.
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.processing_job import ProcessingJob, JobType, JobStatus, JobPriority
from tests.fixtures.fixtures import ProcessingJobFactory


class TestProcessingJobModel:
    """Test suite for ProcessingJob model."""
    
    @pytest.mark.asyncio
    async def test_processing_job_creation(self, db_session):
        """Test basic processing job creation."""
        job = ProcessingJobFactory.build()
        db_session.add(job)
        await db_session.commit()
        
        assert job.id is not None
        assert job.created_at is not None
        assert job.updated_at is not None
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.progress_percent == 0.0
    
    @pytest.mark.asyncio
    async def test_processing_job_required_fields(self, db_session):
        """Test that required fields cannot be null."""
        with pytest.raises(IntegrityError):
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                # Missing entity_type and entity_id
            )
            db_session.add(job)
            await db_session.commit()
    
    def test_processing_job_repr(self):
        """Test processing job string representation."""
        job = ProcessingJobFactory.build(
            id="550e8400-e29b-41d4-a716-446655440000",
            job_type=JobType.VIDEO_DISCOVERY,
            status=JobStatus.RUNNING
        )
        
        expected = "<ProcessingJob(id=550e8400-e29b-41d4-a716-446655440000, type=video_discovery, status=running)>"
        assert repr(job) == expected
    
    def test_is_complete_property(self):
        """Test is_complete property for different statuses."""
        # Complete statuses
        complete_statuses = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
        for status in complete_statuses:
            job = ProcessingJobFactory.build(status=status)
            assert job.is_complete is True
        
        # Non-complete statuses
        non_complete_statuses = [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RETRYING]
        for status in non_complete_statuses:
            job = ProcessingJobFactory.build(status=status)
            assert job.is_complete is False
    
    def test_is_running_property(self):
        """Test is_running property."""
        # Running job
        running_job = ProcessingJobFactory.build(status=JobStatus.RUNNING)
        assert running_job.is_running is True
        
        # Non-running jobs
        non_running_statuses = [
            JobStatus.PENDING, JobStatus.COMPLETED, 
            JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RETRYING
        ]
        
        for status in non_running_statuses:
            job = ProcessingJobFactory.build(status=status)
            assert job.is_running is False
    
    def test_can_retry_property(self):
        """Test can_retry property."""
        # Can retry: failed with low retry count
        job = ProcessingJobFactory.build(
            status=JobStatus.FAILED,
            retry_count=2,
            max_retries=3
        )
        assert job.can_retry is True
        
        # Cannot retry: failed with max retries
        job = ProcessingJobFactory.build(
            status=JobStatus.FAILED,
            retry_count=3,
            max_retries=3
        )
        assert job.can_retry is False
        
        # Cannot retry: not failed
        job = ProcessingJobFactory.build(
            status=JobStatus.COMPLETED,
            retry_count=0,
            max_retries=3
        )
        assert job.can_retry is False
    
    def test_duration_seconds_property(self):
        """Test duration_seconds property."""
        # Job with both start and end times
        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
        
        job = ProcessingJobFactory.build(
            started_at=start_time,
            completed_at=end_time
        )
        
        expected_duration = 330.0  # 5 minutes 30 seconds
        assert job.duration_seconds == expected_duration
        
        # Job without end time
        job = ProcessingJobFactory.build(
            started_at=start_time,
            completed_at=None
        )
        assert job.duration_seconds is None
        
        # Job without start time
        job = ProcessingJobFactory.build(
            started_at=None,
            completed_at=end_time
        )
        assert job.duration_seconds is None
        
        # Job without both times
        job = ProcessingJobFactory.build(
            started_at=None,
            completed_at=None
        )
        assert job.duration_seconds is None
    
    def test_start_method(self):
        """Test start method."""
        job = ProcessingJobFactory.build(status=JobStatus.PENDING)
        worker_id = "worker_001"
        
        job.start(worker_id=worker_id)
        
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert job.worker_id == worker_id
        assert job.progress_percent == 0.0
    
    def test_start_method_without_worker_id(self):
        """Test start method without worker ID."""
        job = ProcessingJobFactory.build(status=JobStatus.PENDING)
        
        job.start()
        
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert job.worker_id is None
        assert job.progress_percent == 0.0
    
    def test_complete_method(self):
        """Test complete method."""
        job = ProcessingJobFactory.build(
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        output_data = {"result": "success", "processed_items": 100}
        
        job.complete(output_data=output_data)
        
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.progress_percent == 100.0
        assert job.output_data == output_data
        assert job.execution_time_seconds is not None
        assert job.execution_time_seconds > 0
    
    def test_complete_method_without_output_data(self):
        """Test complete method without output data."""
        job = ProcessingJobFactory.build(
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        job.complete()
        
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.progress_percent == 100.0
        assert job.output_data is None
    
    def test_fail_method(self):
        """Test fail method."""
        job = ProcessingJobFactory.build(
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        error_message = "Processing failed due to network error"
        error_details = {"error_code": "NETWORK_TIMEOUT", "retryable": True}
        
        job.fail(error_message=error_message, error_details=error_details)
        
        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.error_message == error_message
        assert job.error_details == error_details
        assert job.execution_time_seconds is not None
        assert job.execution_time_seconds > 0
    
    def test_fail_method_without_error_details(self):
        """Test fail method without error details."""
        job = ProcessingJobFactory.build(
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        error_message = "Simple error message"
        
        job.fail(error_message=error_message)
        
        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.error_message == error_message
        assert job.error_details is None
    
    def test_retry_method(self):
        """Test retry method."""
        job = ProcessingJobFactory.build(
            status=JobStatus.FAILED,
            retry_count=1,
            max_retries=3,
            error_message="Previous error",
            error_details={"code": "ERROR"}
        )
        
        delay_seconds = 120
        
        job.retry(delay_seconds=delay_seconds)
        
        assert job.status == JobStatus.RETRYING
        assert job.retry_count == 2
        assert job.retry_after is not None
        assert job.error_message is None
        assert job.error_details is None
        
        # Check delay is approximately correct (within 1 second tolerance)
        expected_retry_time = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        time_diff = abs((job.retry_after - expected_retry_time).total_seconds())
        assert time_diff < 1
    
    def test_retry_method_default_delay(self):
        """Test retry method with default delay."""
        job = ProcessingJobFactory.build(
            status=JobStatus.FAILED,
            retry_count=0,
            max_retries=3
        )
        
        job.retry()
        
        assert job.status == JobStatus.RETRYING
        assert job.retry_count == 1
        assert job.retry_after is not None
        
        # Default delay is 60 seconds
        expected_retry_time = datetime.now(timezone.utc) + timedelta(seconds=60)
        time_diff = abs((job.retry_after - expected_retry_time).total_seconds())
        assert time_diff < 1
    
    def test_retry_method_cannot_retry(self):
        """Test retry method when job cannot be retried."""
        # Max retries reached
        job = ProcessingJobFactory.build(
            status=JobStatus.FAILED,
            retry_count=3,
            max_retries=3
        )
        
        with pytest.raises(ValueError, match="Job cannot be retried"):
            job.retry()
        
        # Not in failed status
        job = ProcessingJobFactory.build(
            status=JobStatus.COMPLETED,
            retry_count=0,
            max_retries=3
        )
        
        with pytest.raises(ValueError, match="Job cannot be retried"):
            job.retry()
    
    def test_cancel_method(self):
        """Test cancel method."""
        job = ProcessingJobFactory.build(
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        job.cancel()
        
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None
        assert job.execution_time_seconds is not None
        assert job.execution_time_seconds > 0
    
    def test_update_progress_method(self):
        """Test update_progress method."""
        job = ProcessingJobFactory.build(status=JobStatus.RUNNING)
        
        # Update progress with message
        job.update_progress(45.5, "Processing item 45 of 100")
        
        assert job.progress_percent == 45.5
        assert job.progress_message == "Processing item 45 of 100"
        
        # Update progress without message
        job.update_progress(75.0)
        
        assert job.progress_percent == 75.0
        assert job.progress_message == "Processing item 45 of 100"  # Previous message retained
    
    def test_update_progress_bounds(self):
        """Test update_progress method bounds checking."""
        job = ProcessingJobFactory.build(status=JobStatus.RUNNING)
        
        # Progress above 100% should be capped
        job.update_progress(150.0)
        assert job.progress_percent == 100.0
        
        # Negative progress should be set to 0
        job.update_progress(-10.0)
        assert job.progress_percent == 0.0
    
    @pytest.mark.asyncio
    async def test_processing_job_default_values(self, db_session):
        """Test default values are set correctly."""
        from uuid import uuid4
        
        job = ProcessingJob(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            entity_type="video",
            entity_id=uuid4()
        )
        db_session.add(job)
        await db_session.commit()
        
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.progress_percent == 0.0
        assert job.input_data == {}
        assert job.metadata == {}
        assert job.scheduled_at is not None
    
    @pytest.mark.asyncio
    async def test_processing_job_json_fields(self, db_session):
        """Test JSONB fields can store and retrieve complex data."""
        from uuid import uuid4
        
        input_data = {
            "channel_id": "UC_test_channel",
            "max_videos": 100,
            "filters": {
                "duration_min": 300,
                "language": "en"
            }
        }
        
        output_data = {
            "videos_processed": 95,
            "transcripts_extracted": 90,
            "errors": ["Video 5 unavailable", "Video 12 private"]
        }
        
        metadata = {
            "batch_id": "batch_001",
            "worker_version": "1.2.3",
            "performance_metrics": {
                "avg_processing_time": 45.2
            }
        }
        
        job = ProcessingJob(
            job_type=JobType.VIDEO_DISCOVERY,
            entity_type="channel",
            entity_id=uuid4(),
            input_data=input_data,
            output_data=output_data,
            metadata=metadata
        )
        db_session.add(job)
        await db_session.commit()
        
        # Refresh from database
        await db_session.refresh(job)
        
        assert job.input_data == input_data
        assert job.output_data == output_data
        assert job.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_job_type_enum(self, db_session):
        """Test different job type values."""
        from uuid import uuid4
        
        job_types = [
            JobType.CHANNEL_DISCOVERY,
            JobType.VIDEO_DISCOVERY,
            JobType.TRANSCRIPT_EXTRACTION,
            JobType.VALUABLE_DETECTION,
            JobType.IDEA_EXTRACTION,
            JobType.BULK_PROCESSING,
            JobType.CLEANUP
        ]
        
        entity_id = uuid4()
        
        for i, job_type in enumerate(job_types):
            job = ProcessingJob(
                job_type=job_type,
                entity_type="test",
                entity_id=entity_id
            )
            db_session.add(job)
        
        await db_session.commit()
        
        # Query jobs by type
        for job_type in job_types:
            result = await db_session.execute(
                select(ProcessingJob).filter(ProcessingJob.job_type == job_type)
            )
            jobs = result.scalars().all()
            assert len(jobs) == 1
            assert jobs[0].job_type == job_type
    
    @pytest.mark.asyncio
    async def test_job_priority_enum(self, db_session):
        """Test different job priority values."""
        from uuid import uuid4
        
        priorities = [
            JobPriority.LOW,
            JobPriority.NORMAL,
            JobPriority.HIGH,
            JobPriority.CRITICAL
        ]
        
        entity_id = uuid4()
        
        for i, priority in enumerate(priorities):
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                entity_type="test",
                entity_id=entity_id,
                priority=priority
            )
            db_session.add(job)
        
        await db_session.commit()
        
        # Query jobs by priority (ordered)
        result = await db_session.execute(
            select(ProcessingJob).order_by(ProcessingJob.priority.desc())
        )
        jobs = result.scalars().all()
        
        assert len(jobs) == 4
        assert jobs[0].priority == JobPriority.CRITICAL  # 20
        assert jobs[1].priority == JobPriority.HIGH      # 10
        assert jobs[2].priority == JobPriority.NORMAL    # 5
        assert jobs[3].priority == JobPriority.LOW       # 1
    
    def test_job_edge_cases(self):
        """Test edge cases and boundary values."""
        from uuid import uuid4
        
        # Very long entity type
        long_entity_type = "A" * 50
        job = ProcessingJobFactory.build(entity_type=long_entity_type)
        assert len(job.entity_type) == 50
        
        # Maximum retry count
        job = ProcessingJobFactory.build(
            retry_count=999,
            max_retries=1000
        )
        assert job.retry_count == 999
        assert job.max_retries == 1000
        
        # Zero retry limit
        job = ProcessingJobFactory.build(max_retries=0)
        assert job.max_retries == 0
        assert job.can_retry is False  # Even if failed
        
        # Very long worker ID
        long_worker_id = "worker_" + "A" * 90
        job = ProcessingJobFactory.build(worker_id=long_worker_id)
        assert len(job.worker_id) == 97  # "worker_" + 90 A's
    
    def test_job_timestamps(self):
        """Test timestamp handling."""
        now = datetime.now(timezone.utc)
        
        # Future scheduled time
        future_scheduled = now + timedelta(hours=1)
        job = ProcessingJobFactory.build(scheduled_at=future_scheduled)
        assert job.scheduled_at == future_scheduled
        
        # Past scheduled time
        past_scheduled = now - timedelta(hours=1)
        job = ProcessingJobFactory.build(scheduled_at=past_scheduled)
        assert job.scheduled_at == past_scheduled


class TestProcessingJobValidation:
    """Test suite for ProcessingJob model validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_job_type_enum(self, db_session):
        """Test that invalid job type enum values are rejected."""
        from uuid import uuid4
        
        with pytest.raises((ValueError, IntegrityError)):
            job = ProcessingJob(
                job_type="invalid_job_type",  # Invalid enum value
                entity_type="test",
                entity_id=uuid4()
            )
            db_session.add(job)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_invalid_status_enum(self, db_session):
        """Test that invalid status enum values are rejected."""
        from uuid import uuid4
        
        with pytest.raises((ValueError, IntegrityError)):
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                status="invalid_status",  # Invalid enum value
                entity_type="test",
                entity_id=uuid4()
            )
            db_session.add(job)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_invalid_priority_enum(self, db_session):
        """Test that invalid priority enum values are rejected."""
        from uuid import uuid4
        
        with pytest.raises((ValueError, IntegrityError)):
            job = ProcessingJob(
                job_type=JobType.TRANSCRIPT_EXTRACTION,
                priority="invalid_priority",  # Invalid enum value
                entity_type="test",
                entity_id=uuid4()
            )
            db_session.add(job)
            await db_session.commit()


class TestProcessingJobWorkflow:
    """Test suite for ProcessingJob workflow scenarios."""
    
    def test_complete_job_workflow(self):
        """Test complete job workflow from pending to completion."""
        from uuid import uuid4
        
        # Create pending job
        job = ProcessingJob(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            entity_type="video",
            entity_id=uuid4()
        )
        
        assert job.status == JobStatus.PENDING
        assert job.is_complete is False
        assert job.is_running is False
        
        # Start job
        job.start(worker_id="worker_001")
        
        assert job.status == JobStatus.RUNNING
        assert job.is_complete is False
        assert job.is_running is True
        
        # Update progress
        job.update_progress(50.0, "Processing...")
        assert job.progress_percent == 50.0
        
        # Complete job
        output_data = {"result": "success"}
        job.complete(output_data=output_data)
        
        assert job.status == JobStatus.COMPLETED
        assert job.is_complete is True
        assert job.is_running is False
        assert job.progress_percent == 100.0
        assert job.output_data == output_data
    
    def test_failed_job_retry_workflow(self):
        """Test job failure and retry workflow."""
        from uuid import uuid4
        
        # Create and start job
        job = ProcessingJob(
            job_type=JobType.TRANSCRIPT_EXTRACTION,
            entity_type="video",
            entity_id=uuid4(),
            max_retries=2
        )
        job.start(worker_id="worker_001")
        
        # Job fails
        job.fail("Network error occurred")
        
        assert job.status == JobStatus.FAILED
        assert job.can_retry is True
        assert job.retry_count == 0
        
        # Retry job
        job.retry(delay_seconds=30)
        
        assert job.status == JobStatus.RETRYING
        assert job.retry_count == 1
        assert job.can_retry is True  # Still can retry once more
        assert job.error_message is None  # Cleared on retry
        
        # Fail again
        job.fail("Another error")
        job.retry()
        
        assert job.retry_count == 2
        assert job.can_retry is False  # Max retries reached
        
        # Try to retry again (should fail)
        with pytest.raises(ValueError):
            job.retry()
    
    def test_job_cancellation_workflow(self):
        """Test job cancellation workflow."""
        from uuid import uuid4
        
        # Create and start job
        job = ProcessingJob(
            job_type=JobType.VIDEO_DISCOVERY,
            entity_type="channel",
            entity_id=uuid4()
        )
        job.start(worker_id="worker_001")
        
        # Cancel running job
        job.cancel()
        
        assert job.status == JobStatus.CANCELLED
        assert job.is_complete is True
        assert job.is_running is False
        assert job.completed_at is not None
    
    def test_multiple_progress_updates(self):
        """Test multiple progress updates during job execution."""
        from uuid import uuid4
        
        job = ProcessingJob(
            job_type=JobType.BULK_PROCESSING,
            entity_type="batch",
            entity_id=uuid4()
        )
        job.start()
        
        # Simulate progress updates
        progress_steps = [
            (10.0, "Initializing..."),
            (25.0, "Processing batch 1/4"),
            (50.0, "Processing batch 2/4"),
            (75.0, "Processing batch 3/4"),
            (90.0, "Processing batch 4/4"),
            (100.0, "Finalizing...")
        ]
        
        for percent, message in progress_steps:
            job.update_progress(percent, message)
            assert job.progress_percent == percent
            assert job.progress_message == message
        
        job.complete()
        assert job.progress_percent == 100.0
        assert job.status == JobStatus.COMPLETED