import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSource
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.utils.crud_models import (
    crud_channel,
    crud_video,
    crud_transcript,
    crud_processing_job,
)


@pytest.mark.asyncio
async def test_crud_channel_operations(db_session):
    channel_data = {
        "youtube_channel_id": "UC123456789",
        "channel_name": "Test Channel",
        "status": ChannelStatus.ACTIVE,
        "priority_level": 5,
        "check_frequency_hours": 24,
    }
    
    channel = await crud_channel.create(db_session, obj_in=channel_data)
    assert channel.id is not None
    assert channel.youtube_channel_id == "UC123456789"
    
    fetched_channel = await crud_channel.get(db_session, channel.id)
    assert fetched_channel is not None
    assert fetched_channel.id == channel.id
    
    by_youtube_id = await crud_channel.get_by_youtube_id(
        db_session, "UC123456789"
    )
    assert by_youtube_id is not None
    assert by_youtube_id.id == channel.id
    
    update_data = {"channel_name": "Updated Channel Name"}
    updated_channel = await crud_channel.update(
        db_session, id=channel.id, obj_in=update_data
    )
    assert updated_channel.channel_name == "Updated Channel Name"
    
    channel_count = await crud_channel.count(db_session)
    assert channel_count == 1
    
    exists = await crud_channel.exists(db_session, id=channel.id)
    assert exists is True
    
    deleted = await crud_channel.delete(db_session, id=channel.id)
    assert deleted is True
    
    exists_after_delete = await crud_channel.exists(db_session, id=channel.id)
    assert exists_after_delete is False


@pytest.mark.asyncio
async def test_crud_channel_get_active(db_session):
    active_channels = []
    for i in range(3):
        channel = await crud_channel.create(
            db_session,
            obj_in={
                "youtube_channel_id": f"UC{i}",
                "channel_name": f"Channel {i}",
                "status": ChannelStatus.ACTIVE,
                "priority_level": i + 1,
            }
        )
        active_channels.append(channel)
    
    inactive_channel = await crud_channel.create(
        db_session,
        obj_in={
            "youtube_channel_id": "UCinactive",
            "channel_name": "Inactive Channel",
            "status": ChannelStatus.INACTIVE,
        }
    )
    
    active = await crud_channel.get_active_channels(db_session)
    assert len(active) == 3
    assert all(c.status == ChannelStatus.ACTIVE for c in active)
    assert active[0].priority_level > active[1].priority_level


@pytest.mark.asyncio
async def test_crud_video_operations(db_session):
    channel = await crud_channel.create(
        db_session,
        obj_in={
            "youtube_channel_id": "UC123",
            "channel_name": "Test Channel",
            "status": ChannelStatus.ACTIVE,
        }
    )
    
    video_data = {
        "youtube_video_id": "dQw4w9WgXcQ",
        "channel_id": channel.id,
        "title": "Test Video",
        "published_at": datetime.utcnow(),
        "status": VideoStatus.DISCOVERED,
    }
    
    video = await crud_video.create(db_session, obj_in=video_data)
    assert video.id is not None
    
    by_youtube_id = await crud_video.get_by_youtube_id(
        db_session, "dQw4w9WgXcQ"
    )
    assert by_youtube_id is not None
    assert by_youtube_id.id == video.id
    
    video.status = VideoStatus.QUEUED
    await db_session.commit()
    
    videos_to_process = await crud_video.get_videos_to_process(db_session)
    assert len(videos_to_process) == 1
    assert videos_to_process[0].id == video.id
    
    channel_videos = await crud_video.get_channel_videos(
        db_session, channel.id
    )
    assert len(channel_videos) == 1
    assert channel_videos[0].id == video.id


@pytest.mark.asyncio
async def test_crud_video_valuable(db_session):
    channel = await crud_channel.create(
        db_session,
        obj_in={
            "youtube_channel_id": "UC123",
            "channel_name": "Test Channel",
            "status": ChannelStatus.ACTIVE,
        }
    )
    
    valuable_videos = []
    for i in range(3):
        video = await crud_video.create(
            db_session,
            obj_in={
                "youtube_video_id": f"video{i}",
                "channel_id": channel.id,
                "title": f"Video {i}",
                "published_at": datetime.utcnow(),
                "is_valuable": True,
                "valuable_score": 0.8 + (i * 0.05),
            }
        )
        valuable_videos.append(video)
    
    not_valuable = await crud_video.create(
        db_session,
        obj_in={
            "youtube_video_id": "notvaluable",
            "channel_id": channel.id,
            "title": "Not Valuable",
            "published_at": datetime.utcnow(),
            "is_valuable": False,
            "valuable_score": 0.2,
        }
    )
    
    valuable = await crud_video.get_valuable_videos(db_session, min_score=0.7)
    assert len(valuable) == 3
    assert all(v.is_valuable for v in valuable)
    assert valuable[0].valuable_score > valuable[1].valuable_score


@pytest.mark.asyncio
async def test_crud_transcript_operations(db_session):
    channel = await crud_channel.create(
        db_session,
        obj_in={
            "youtube_channel_id": "UC123",
            "channel_name": "Test Channel",
            "status": ChannelStatus.ACTIVE,
        }
    )
    
    video = await crud_video.create(
        db_session,
        obj_in={
            "youtube_video_id": "video1",
            "channel_id": channel.id,
            "title": "Test Video",
            "published_at": datetime.utcnow(),
        }
    )
    
    transcript_data = {
        "video_id": video.id,
        "content": "Test transcript content",
        "source": TranscriptSource.YOUTUBE_AUTO,
    }
    
    transcript = await crud_transcript.create(db_session, obj_in=transcript_data)
    assert transcript.id is not None
    
    video_transcript = await crud_transcript.get_video_transcript(
        db_session, video.id
    )
    assert video_transcript is not None
    assert video_transcript.id == transcript.id
    
    manual_transcript = await crud_transcript.create(
        db_session,
        obj_in={
            "video_id": video.id,
            "content": "Manual transcript",
            "source": TranscriptSource.YOUTUBE_MANUAL,
        }
    )
    
    all_transcripts = await crud_transcript.get_video_transcripts(
        db_session, video.id
    )
    assert len(all_transcripts) == 2
    
    manual_only = await crud_transcript.get_video_transcript(
        db_session, video.id, source=TranscriptSource.YOUTUBE_MANUAL
    )
    assert manual_only is not None
    assert manual_only.id == manual_transcript.id


@pytest.mark.asyncio
async def test_crud_processing_job_operations(db_session):
    job_data = {
        "job_type": JobType.VIDEO_DISCOVERY,
        "status": JobStatus.PENDING,
        "entity_type": "channel",
        "entity_id": uuid4(),
        "scheduled_at": datetime.utcnow(),
    }
    
    job = await crud_processing_job.create(db_session, obj_in=job_data)
    assert job.id is not None
    
    pending_jobs = await crud_processing_job.get_pending_jobs(db_session)
    assert len(pending_jobs) == 1
    assert pending_jobs[0].id == job.id
    
    entity_jobs = await crud_processing_job.get_entity_jobs(
        db_session,
        entity_type="channel",
        entity_id=job.entity_id
    )
    assert len(entity_jobs) == 1
    assert entity_jobs[0].id == job.id
    
    job.status = JobStatus.FAILED
    job.retry_count = 1
    job.max_retries = 3
    await db_session.commit()
    
    failed_to_retry = await crud_processing_job.get_failed_jobs_to_retry(
        db_session
    )
    assert len(failed_to_retry) == 1
    assert failed_to_retry[0].id == job.id


@pytest.mark.asyncio
async def test_crud_bulk_operations(db_session):
    channels_data = [
        {
            "youtube_channel_id": f"UC{i}",
            "channel_name": f"Channel {i}",
            "status": ChannelStatus.ACTIVE,
        }
        for i in range(5)
    ]
    
    channels = await crud_channel.bulk_create(db_session, objs_in=channels_data)
    assert len(channels) == 5
    
    all_channels = await crud_channel.get_multi(db_session, limit=10)
    assert len(all_channels) == 5
    
    filtered = await crud_channel.get_multi(
        db_session,
        filters={"status": ChannelStatus.ACTIVE}
    )
    assert len(filtered) == 5