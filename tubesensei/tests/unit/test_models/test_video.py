"""
Unit tests for Video model.

Tests the Video model including properties, methods, validations,
and edge cases.
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from tests.fixtures.fixtures import VideoFactory, ChannelFactory


class TestVideoModel:
    """Test suite for Video model."""
    
    @pytest.mark.asyncio
    async def test_video_creation(self, db_session):
        """Test basic video creation."""
        # Create channel first
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.commit()
        
        assert video.id is not None
        assert video.created_at is not None
        assert video.updated_at is not None
        assert video.status == VideoStatus.DISCOVERED
        assert video.retry_count == 0
        assert video.has_captions is True
    
    @pytest.mark.asyncio
    async def test_video_unique_constraint(self, db_session):
        """Test that youtube_video_id must be unique."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video_id = "unique_video_test"
        
        # Create first video
        video1 = VideoFactory.build(
            youtube_video_id=video_id,
            channel_id=channel.id
        )
        db_session.add(video1)
        await db_session.commit()
        
        # Attempt to create second video with same ID
        video2 = VideoFactory.build(
            youtube_video_id=video_id,
            channel_id=channel.id
        )
        db_session.add(video2)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_video_required_fields(self, db_session):
        """Test that required fields cannot be null."""
        with pytest.raises(IntegrityError):
            video = Video(
                title="Test Video",
                published_at=datetime.now(timezone.utc)
                # Missing youtube_video_id and channel_id
            )
            db_session.add(video)
            await db_session.commit()
    
    def test_video_repr(self):
        """Test video string representation."""
        video = VideoFactory.build(
            id="550e8400-e29b-41d4-a716-446655440000",
            title="This is a very long video title that should be truncated for display",
            status=VideoStatus.COMPLETED
        )
        
        expected = "<Video(id=550e8400-e29b-41d4-a716-446655440000, title=This is a very long video title that should be tr..., status=completed)>"
        assert repr(video) == expected
    
    def test_youtube_url_property(self):
        """Test youtube_url property."""
        video = VideoFactory.build(youtube_video_id="dQw4w9WgXcQ")
        expected_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert video.youtube_url == expected_url
    
    def test_is_processed_property(self):
        """Test is_processed property for different statuses."""
        # Processed statuses
        processed_statuses = [VideoStatus.COMPLETED, VideoStatus.FAILED, VideoStatus.SKIPPED]
        for status in processed_statuses:
            video = VideoFactory.build(status=status)
            assert video.is_processed is True
        
        # Non-processed statuses
        non_processed_statuses = [VideoStatus.DISCOVERED, VideoStatus.QUEUED, VideoStatus.PROCESSING]
        for status in non_processed_statuses:
            video = VideoFactory.build(status=status)
            assert video.is_processed is False
    
    def test_can_retry_property(self):
        """Test can_retry property."""
        # Can retry: failed with low retry count
        video = VideoFactory.build(status=VideoStatus.FAILED, retry_count=2)
        assert video.can_retry is True
        
        # Cannot retry: failed with max retries
        video = VideoFactory.build(status=VideoStatus.FAILED, retry_count=3)
        assert video.can_retry is False
        
        # Cannot retry: not failed
        video = VideoFactory.build(status=VideoStatus.COMPLETED, retry_count=0)
        assert video.can_retry is False
    
    def test_duration_formatted_property(self):
        """Test duration_formatted property."""
        # No duration
        video = VideoFactory.build(duration_seconds=None)
        assert video.duration_formatted == "Unknown"
        
        # Short video (minutes:seconds)
        video = VideoFactory.build(duration_seconds=150)  # 2:30
        assert video.duration_formatted == "2:30"
        
        # Medium video (minutes:seconds)
        video = VideoFactory.build(duration_seconds=630)  # 10:30
        assert video.duration_formatted == "10:30"
        
        # Long video (hours:minutes:seconds)
        video = VideoFactory.build(duration_seconds=3661)  # 1:01:01
        assert video.duration_formatted == "1:01:01"
        
        # Very long video
        video = VideoFactory.build(duration_seconds=7323)  # 2:02:03
        assert video.duration_formatted == "2:02:03"
    
    def test_mark_as_valuable(self):
        """Test mark_as_valuable method."""
        video = VideoFactory.build()
        
        video.mark_as_valuable(0.85, "High engagement and educational content")
        
        assert video.is_valuable is True
        assert video.valuable_score == 0.85
        assert video.valuable_reason == "High engagement and educational content"
    
    def test_mark_as_not_valuable(self):
        """Test mark_as_not_valuable method."""
        video = VideoFactory.build()
        
        video.mark_as_not_valuable("Low quality audio and poor content structure")
        
        assert video.is_valuable is False
        assert video.valuable_score == 0.0
        assert video.valuable_reason == "Low quality audio and poor content structure"
    
    def test_update_stats_method(self):
        """Test update_stats method."""
        video = VideoFactory.build(
            view_count=1000,
            like_count=50,
            comment_count=10
        )
        
        # Update all stats
        video.update_stats(
            view_count=2000,
            like_count=100,
            comment_count=25
        )
        
        assert video.view_count == 2000
        assert video.like_count == 100
        assert video.comment_count == 25
    
    def test_update_stats_partial(self):
        """Test update_stats with partial updates."""
        video = VideoFactory.build(
            view_count=1000,
            like_count=50,
            comment_count=10
        )
        
        # Update only view count
        video.update_stats(view_count=2000)
        
        assert video.view_count == 2000
        assert video.like_count == 50  # Unchanged
        assert video.comment_count == 10  # Unchanged
    
    def test_update_stats_none_values(self):
        """Test update_stats with None values doesn't change existing values."""
        original_view_count = 1000
        original_like_count = 50
        original_comment_count = 10
        
        video = VideoFactory.build(
            view_count=original_view_count,
            like_count=original_like_count,
            comment_count=original_comment_count
        )
        
        # Update with None values
        video.update_stats(
            view_count=None,
            like_count=None,
            comment_count=None
        )
        
        assert video.view_count == original_view_count
        assert video.like_count == original_like_count
        assert video.comment_count == original_comment_count
    
    @pytest.mark.asyncio
    async def test_video_default_values(self, db_session):
        """Test default values are set correctly."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = Video(
            youtube_video_id="test_defaults",
            channel_id=channel.id,
            title="Test Video",
            published_at=datetime.now(timezone.utc)
        )
        db_session.add(video)
        await db_session.commit()
        
        assert video.status == VideoStatus.DISCOVERED
        assert video.retry_count == 0
        assert video.view_count == 0
        assert video.like_count == 0
        assert video.comment_count == 0
        assert video.has_captions is False
        assert video.tags == []
        assert video.caption_languages == []
        assert video.metadata == {}
        assert video.processing_metadata == {}
        assert video.discovered_at is not None
    
    @pytest.mark.asyncio
    async def test_video_json_fields(self, db_session):
        """Test JSONB fields can store and retrieve complex data."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        metadata = {
            "definition": "hd",
            "dimension": "2d",
            "caption_info": {
                "automatic": True,
                "languages": ["en", "en-US"]
            }
        }
        
        processing_metadata = {
            "transcript_fetched": True,
            "analysis_complete": False,
            "processing_steps": ["download", "analyze"]
        }
        
        tags = ["technology", "tutorial", "programming"]
        caption_languages = ["en", "en-US", "es"]
        
        video = Video(
            youtube_video_id="test_json",
            channel_id=channel.id,
            title="Test Video",
            published_at=datetime.now(timezone.utc),
            metadata=metadata,
            processing_metadata=processing_metadata,
            tags=tags,
            caption_languages=caption_languages
        )
        db_session.add(video)
        await db_session.commit()
        
        # Refresh from database
        await db_session.refresh(video)
        
        assert video.metadata == metadata
        assert video.processing_metadata == processing_metadata
        assert video.tags == tags
        assert video.caption_languages == caption_languages
    
    @pytest.mark.asyncio
    async def test_video_status_enum(self, db_session):
        """Test different video status values."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        statuses = [
            VideoStatus.DISCOVERED,
            VideoStatus.QUEUED,
            VideoStatus.PROCESSING,
            VideoStatus.COMPLETED,
            VideoStatus.FAILED,
            VideoStatus.SKIPPED
        ]
        
        for i, status in enumerate(statuses):
            video = VideoFactory.build(
                youtube_video_id=f"test_status_{i}",
                channel_id=channel.id,
                status=status
            )
            db_session.add(video)
        
        await db_session.commit()
        
        # Query videos by status
        for status in statuses:
            result = await db_session.execute(
                select(Video).filter(Video.status == status)
            )
            videos = result.scalars().all()
            assert len(videos) == 1
            assert videos[0].status == status
    
    def test_video_edge_cases(self):
        """Test edge cases and boundary values."""
        # Very long title
        long_title = "A" * 500
        video = VideoFactory.build(title=long_title)
        assert len(video.title) == 500
        
        # Very high view count
        video = VideoFactory.build(view_count=999999999)
        assert video.view_count == 999999999
        
        # Zero duration
        video = VideoFactory.build(duration_seconds=0)
        assert video.duration_formatted == "0:00"
        
        # Very short video (1 second)
        video = VideoFactory.build(duration_seconds=1)
        assert video.duration_formatted == "0:01"
        
        # Very long video (24 hours)
        video = VideoFactory.build(duration_seconds=86400)  # 24:00:00
        assert video.duration_formatted == "24:00:00"
    
    def test_video_timestamps(self):
        """Test timestamp fields."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=30)
        
        video = VideoFactory.build(
            published_at=past,
            discovered_at=now,
            processed_at=now + timedelta(minutes=5)
        )
        
        assert video.published_at < video.discovered_at
        assert video.discovered_at < video.processed_at
    
    def test_valuable_scoring(self):
        """Test valuable scoring edge cases."""
        video = VideoFactory.build()
        
        # Perfect score
        video.mark_as_valuable(1.0, "Perfect content")
        assert video.valuable_score == 1.0
        
        # Zero score but marked valuable
        video.mark_as_valuable(0.0, "Edge case")
        assert video.is_valuable is True
        assert video.valuable_score == 0.0
        
        # Negative score (should be handled by business logic)
        video.mark_as_valuable(-0.1, "Invalid score")
        assert video.valuable_score == -0.1
        
        # Very high score (should be handled by business logic)
        video.mark_as_valuable(1.5, "Invalid high score")
        assert video.valuable_score == 1.5


class TestVideoValidation:
    """Test suite for Video model validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_status_enum(self, db_session):
        """Test that invalid status enum values are rejected."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        with pytest.raises((ValueError, IntegrityError)):
            video = Video(
                youtube_video_id="test_invalid_status",
                channel_id=channel.id,
                title="Test Video",
                published_at=datetime.now(timezone.utc),
                status="invalid_status"  # Invalid enum value
            )
            db_session.add(video)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, db_session):
        """Test that channel_id must reference existing channel."""
        from uuid import uuid4
        
        with pytest.raises(IntegrityError):
            video = Video(
                youtube_video_id="test_invalid_channel",
                channel_id=uuid4(),  # Non-existent channel
                title="Test Video",
                published_at=datetime.now(timezone.utc)
            )
            db_session.add(video)
            await db_session.commit()
    
    def test_youtube_video_id_formats(self):
        """Test various YouTube video ID formats."""
        # Standard format (11 characters)
        video = VideoFactory.build(youtube_video_id="dQw4w9WgXcQ")
        assert len(video.youtube_video_id) == 11
        
        # Shorter ID (though rare)
        video = VideoFactory.build(youtube_video_id="abc123")
        assert video.youtube_video_id == "abc123"
        
        # Longer ID (custom test)
        video = VideoFactory.build(youtube_video_id="test_video_001")
        assert video.youtube_video_id == "test_video_001"


class TestVideoRelationships:
    """Test suite for Video model relationships."""
    
    @pytest.mark.asyncio
    async def test_video_channel_relationship(self, db_session):
        """Test that channel relationship works correctly."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        # Create video
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(video)
        assert video.channel is not None
        assert video.channel.id == channel.id
        assert video.channel.channel_name == channel.channel_name
    
    @pytest.mark.asyncio
    async def test_video_transcripts_relationship(self, db_session):
        """Test that transcripts relationship works correctly."""
        from app.models.transcript import Transcript
        
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        # Create transcripts for this video
        transcript1 = Transcript(
            video_id=video.id,
            content="First transcript content",
            language_code="en"
        )
        transcript2 = Transcript(
            video_id=video.id,
            content="Second transcript content",
            language_code="es"
        )
        
        db_session.add(transcript1)
        db_session.add(transcript2)
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(video)
        transcripts = await video.transcripts.all()  # Dynamic relationship
        assert len(transcripts) == 2
        assert transcript1 in transcripts
        assert transcript2 in transcripts
    
    @pytest.mark.asyncio
    async def test_video_cascade_delete(self, db_session):
        """Test that deleting video cascades to transcripts."""
        from app.models.transcript import Transcript
        
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        transcript = Transcript(
            video_id=video.id,
            content="Test transcript content",
            language_code="en"
        )
        db_session.add(transcript)
        await db_session.commit()
        
        # Delete video
        await db_session.delete(video)
        await db_session.commit()
        
        # Transcript should be deleted too (cascade)
        result = await db_session.execute(select(Transcript).filter(Transcript.id == transcript.id))
        deleted_transcript = result.scalar_one_or_none()
        assert deleted_transcript is None
    
    @pytest.mark.asyncio
    async def test_channel_delete_cascades_to_video(self, db_session):
        """Test that deleting channel cascades to video."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.commit()
        
        video_id = video.id
        
        # Delete channel
        await db_session.delete(channel)
        await db_session.commit()
        
        # Video should be deleted too (cascade)
        result = await db_session.execute(select(Video).filter(Video.id == video_id))
        deleted_video = result.scalar_one_or_none()
        assert deleted_video is None