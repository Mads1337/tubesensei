import pytest
from datetime import datetime
from uuid import uuid4

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from app.models.processing_job import ProcessingJob, JobType, JobStatus, JobPriority
from app.models.processing_session import ProcessingSession, SessionType, SessionStatus


@pytest.mark.asyncio
async def test_channel_model_creation(db_session):
    channel_data = {
        "youtube_channel_id": "UC123456789",
        "channel_name": "Test Channel",
        "channel_handle": "@testchannel",
        "description": "A test channel",
        "subscriber_count": 1000,
        "video_count": 50,
        "status": ChannelStatus.ACTIVE,
        "priority_level": 5,
        "check_frequency_hours": 24,
        "auto_process": True,
    }
    
    channel = Channel(**channel_data)
    db_session.add(channel)
    await db_session.commit()
    await db_session.refresh(channel)
    
    assert channel.id is not None
    assert channel.youtube_channel_id == "UC123456789"
    assert channel.channel_name == "Test Channel"
    assert channel.status == ChannelStatus.ACTIVE
    assert channel.is_active is True
    assert channel.created_at is not None
    assert channel.updated_at is not None


@pytest.mark.asyncio
async def test_video_model_creation(db_session):
    channel = Channel(
        youtube_channel_id="UC123456789",
        channel_name="Test Channel",
        status=ChannelStatus.ACTIVE
    )
    db_session.add(channel)
    await db_session.commit()
    
    video_data = {
        "youtube_video_id": "dQw4w9WgXcQ",
        "channel_id": channel.id,
        "title": "Test Video",
        "description": "A test video description",
        "duration_seconds": 300,
        "view_count": 1000000,
        "published_at": datetime.utcnow(),
        "status": VideoStatus.DISCOVERED,
        "has_captions": True,
    }
    
    video = Video(**video_data)
    db_session.add(video)
    await db_session.commit()
    await db_session.refresh(video)
    
    assert video.id is not None
    assert video.youtube_video_id == "dQw4w9WgXcQ"
    assert video.channel_id == channel.id
    assert video.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert video.duration_formatted == "5:00"
    assert video.is_processed is False


@pytest.mark.asyncio
async def test_transcript_model_creation(db_session):
    channel = Channel(
        youtube_channel_id="UC123456789",
        channel_name="Test Channel",
        status=ChannelStatus.ACTIVE
    )
    db_session.add(channel)
    await db_session.commit()
    
    video = Video(
        youtube_video_id="dQw4w9WgXcQ",
        channel_id=channel.id,
        title="Test Video",
        published_at=datetime.utcnow(),
        status=VideoStatus.DISCOVERED
    )
    db_session.add(video)
    await db_session.commit()
    
    transcript_data = {
        "video_id": video.id,
        "content": "This is a test transcript content.",
        "source": TranscriptSource.YOUTUBE_AUTO,
        "language": TranscriptLanguage.EN,
        "language_code": "en",
        "is_auto_generated": True,
        "is_complete": True,
    }
    
    transcript = Transcript(**transcript_data)
    db_session.add(transcript)
    await db_session.commit()
    await db_session.refresh(transcript)
    
    assert transcript.id is not None
    assert transcript.video_id == video.id
    assert transcript.is_english is True
    assert transcript.needs_translation is False
    
    transcript.calculate_stats()
    assert transcript.word_count == 6
    assert transcript.char_count == 34


@pytest.mark.asyncio
async def test_processing_job_model_creation(db_session):
    job_data = {
        "job_type": JobType.VIDEO_DISCOVERY,
        "status": JobStatus.PENDING,
        "priority": JobPriority.NORMAL,
        "entity_type": "channel",
        "entity_id": uuid4(),
        "scheduled_at": datetime.utcnow(),
        "input_data": {"channel_id": "UC123456789"},
    }
    
    job = ProcessingJob(**job_data)
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    
    assert job.id is not None
    assert job.job_type == JobType.VIDEO_DISCOVERY
    assert job.status == JobStatus.PENDING
    assert job.is_complete is False
    assert job.is_running is False
    assert job.can_retry is False
    
    job.start("worker-1")
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None
    assert job.worker_id == "worker-1"
    
    job.complete({"videos_found": 10})
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None
    assert job.output_data == {"videos_found": 10}
    assert job.progress_percent == 100.0


@pytest.mark.asyncio
async def test_processing_session_model_creation(db_session):
    session_data = {
        "session_type": SessionType.BULK_PROCESSING,
        "status": SessionStatus.INITIALIZED,
        "name": "Test Processing Session",
        "description": "A test session for bulk processing",
        "total_jobs": 100,
        "configuration": {"batch_size": 10},
    }
    
    session = ProcessingSession(**session_data)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    
    assert session.id is not None
    assert session.session_type == SessionType.BULK_PROCESSING
    assert session.status == SessionStatus.INITIALIZED
    assert session.is_active is False
    assert session.is_complete is False
    assert session.success_rate == 0.0
    
    session.start()
    assert session.status == SessionStatus.RUNNING
    assert session.started_at is not None
    assert session.is_active is True
    
    session.completed_jobs = 80
    session.failed_jobs = 10
    session.cancelled_jobs = 5
    session.update_progress()
    
    assert session.progress_percent == 95.0
    assert session.success_rate == 80.0
    assert session.failure_rate == 10.0


@pytest.mark.asyncio
async def test_model_relationships(db_session):
    channel = Channel(
        youtube_channel_id="UC123456789",
        channel_name="Test Channel",
        status=ChannelStatus.ACTIVE
    )
    db_session.add(channel)
    await db_session.commit()
    
    video1 = Video(
        youtube_video_id="video1",
        channel_id=channel.id,
        title="Video 1",
        published_at=datetime.utcnow()
    )
    video2 = Video(
        youtube_video_id="video2",
        channel_id=channel.id,
        title="Video 2",
        published_at=datetime.utcnow()
    )
    db_session.add_all([video1, video2])
    await db_session.commit()
    
    transcript1 = Transcript(
        video_id=video1.id,
        content="Transcript 1",
        source=TranscriptSource.YOUTUBE_AUTO
    )
    transcript2 = Transcript(
        video_id=video1.id,
        content="Transcript 2",
        source=TranscriptSource.YOUTUBE_MANUAL
    )
    db_session.add_all([transcript1, transcript2])
    await db_session.commit()
    
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    
    query = select(Channel).options(
        selectinload(Channel.videos)
    ).where(Channel.id == channel.id)
    
    result = await db_session.execute(query)
    loaded_channel = result.scalar_one()
    
    videos = await loaded_channel.videos.all()
    assert len(videos) == 2
    
    query = select(Video).options(
        selectinload(Video.transcripts)
    ).where(Video.id == video1.id)
    
    result = await db_session.execute(query)
    loaded_video = result.scalar_one()
    
    transcripts = await loaded_video.transcripts.all()
    assert len(transcripts) == 2