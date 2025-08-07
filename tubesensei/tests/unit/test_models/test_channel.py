"""
Unit tests for Channel model.

Tests the Channel model including properties, methods, validations,
and edge cases.
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.channel import Channel, ChannelStatus
from tests.fixtures.fixtures import ChannelFactory


class TestChannelModel:
    """Test suite for Channel model."""
    
    @pytest.mark.asyncio
    async def test_channel_creation(self, db_session):
        """Test basic channel creation."""
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        assert channel.id is not None
        assert channel.created_at is not None
        assert channel.updated_at is not None
        assert channel.status == ChannelStatus.ACTIVE
        assert channel.auto_process is True
        assert channel.priority_level == 5
        assert channel.check_frequency_hours == 24
    
    @pytest.mark.asyncio
    async def test_channel_unique_constraint(self, db_session):
        """Test that youtube_channel_id must be unique."""
        channel_id = "UC_unique_test_channel"
        
        # Create first channel
        channel1 = ChannelFactory.build(youtube_channel_id=channel_id)
        db_session.add(channel1)
        await db_session.commit()
        
        # Attempt to create second channel with same ID
        channel2 = ChannelFactory.build(youtube_channel_id=channel_id)
        db_session.add(channel2)
        
        with pytest.raises(IntegrityError):
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_channel_required_fields(self, db_session):
        """Test that required fields cannot be null."""
        with pytest.raises(IntegrityError):
            channel = Channel(
                channel_name="Test Channel",
                # Missing youtube_channel_id
            )
            db_session.add(channel)
            await db_session.commit()
    
    def test_channel_repr(self):
        """Test channel string representation."""
        channel = ChannelFactory.build(
            id="550e8400-e29b-41d4-a716-446655440000",
            channel_name="Test Channel",
            channel_handle="@testchannel"
        )
        
        expected = "<Channel(id=550e8400-e29b-41d4-a716-446655440000, name=Test Channel, handle=@testchannel)>"
        assert repr(channel) == expected
    
    def test_is_active_property(self):
        """Test is_active property."""
        # Active channel
        active_channel = ChannelFactory.build(status=ChannelStatus.ACTIVE)
        assert active_channel.is_active is True
        
        # Paused channel
        paused_channel = ChannelFactory.build(status=ChannelStatus.PAUSED)
        assert paused_channel.is_active is False
        
        # Inactive channel
        inactive_channel = ChannelFactory.build(status=ChannelStatus.INACTIVE)
        assert inactive_channel.is_active is False
    
    def test_needs_check_property_no_last_check(self):
        """Test needs_check when never checked."""
        channel = ChannelFactory.build(
            status=ChannelStatus.ACTIVE,
            last_checked_at=None
        )
        assert channel.needs_check is True
    
    def test_needs_check_property_inactive_channel(self):
        """Test needs_check for inactive channel."""
        channel = ChannelFactory.build(
            status=ChannelStatus.INACTIVE,
            last_checked_at=None
        )
        assert channel.needs_check is False
    
    def test_needs_check_property_recent_check(self):
        """Test needs_check when recently checked."""
        # Channel checked 1 hour ago, frequency is 24 hours
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        channel = ChannelFactory.build(
            status=ChannelStatus.ACTIVE,
            last_checked_at=recent_time,
            check_frequency_hours=24
        )
        assert channel.needs_check is False
    
    def test_needs_check_property_old_check(self):
        """Test needs_check when check is overdue."""
        # Channel checked 25 hours ago, frequency is 24 hours
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        channel = ChannelFactory.build(
            status=ChannelStatus.ACTIVE,
            last_checked_at=old_time,
            check_frequency_hours=24
        )
        assert channel.needs_check is True
    
    def test_update_stats_method(self):
        """Test update_stats method."""
        channel = ChannelFactory.build(
            subscriber_count=1000,
            video_count=100,
            view_count=50000
        )
        
        # Update all stats
        channel.update_stats(
            subscriber_count=2000,
            video_count=150,
            view_count=75000
        )
        
        assert channel.subscriber_count == 2000
        assert channel.video_count == 150
        assert channel.view_count == 75000
        assert channel.last_checked_at is not None
    
    def test_update_stats_partial(self):
        """Test update_stats with partial updates."""
        channel = ChannelFactory.build(
            subscriber_count=1000,
            video_count=100,
            view_count=50000
        )
        
        # Update only subscriber count
        channel.update_stats(subscriber_count=2000)
        
        assert channel.subscriber_count == 2000
        assert channel.video_count == 100  # Unchanged
        assert channel.view_count == 50000  # Unchanged
    
    def test_update_stats_none_values(self):
        """Test update_stats with None values doesn't change existing values."""
        original_subscriber_count = 1000
        original_video_count = 100
        original_view_count = 50000
        
        channel = ChannelFactory.build(
            subscriber_count=original_subscriber_count,
            video_count=original_video_count,
            view_count=original_view_count
        )
        
        # Update with None values
        channel.update_stats(
            subscriber_count=None,
            video_count=None,
            view_count=None
        )
        
        assert channel.subscriber_count == original_subscriber_count
        assert channel.video_count == original_video_count
        assert channel.view_count == original_view_count
    
    @pytest.mark.asyncio
    async def test_channel_default_values(self, db_session):
        """Test default values are set correctly."""
        channel = Channel(
            youtube_channel_id="UC_test_defaults",
            channel_name="Test Channel"
        )
        db_session.add(channel)
        await db_session.commit()
        
        assert channel.status == ChannelStatus.ACTIVE
        assert channel.priority_level == 5
        assert channel.check_frequency_hours == 24
        assert channel.auto_process is True
        assert channel.subscriber_count == 0
        assert channel.video_count == 0
        assert channel.view_count == 0
        assert channel.metadata == {}
        assert channel.processing_config == {}
        assert channel.tags == []
    
    @pytest.mark.asyncio
    async def test_channel_json_fields(self, db_session):
        """Test JSONB fields can store and retrieve complex data."""
        metadata = {
            "keywords": ["tech", "tutorial"],
            "branding": {"color": "#ff0000"},
            "stats": {"engagement_rate": 5.2}
        }
        
        processing_config = {
            "auto_transcript": True,
            "languages": ["en", "es"],
            "quality_threshold": 0.8
        }
        
        tags = ["technology", "education", "tutorial"]
        
        channel = Channel(
            youtube_channel_id="UC_test_json",
            channel_name="Test Channel",
            metadata=metadata,
            processing_config=processing_config,
            tags=tags
        )
        db_session.add(channel)
        await db_session.commit()
        
        # Refresh from database
        await db_session.refresh(channel)
        
        assert channel.metadata == metadata
        assert channel.processing_config == processing_config
        assert channel.tags == tags
    
    @pytest.mark.asyncio
    async def test_channel_status_enum(self, db_session):
        """Test different channel status values."""
        statuses = [ChannelStatus.ACTIVE, ChannelStatus.PAUSED, ChannelStatus.INACTIVE]
        
        for status in statuses:
            channel = ChannelFactory.build(status=status)
            db_session.add(channel)
        
        await db_session.commit()
        
        # Query channels by status
        for status in statuses:
            result = await db_session.execute(
                select(Channel).filter(Channel.status == status)
            )
            channels = result.scalars().all()
            assert len(channels) == 1
            assert channels[0].status == status
    
    @pytest.mark.asyncio
    async def test_channel_priority_levels(self, db_session):
        """Test different priority levels."""
        priorities = [1, 5, 10, 20]
        
        for priority in priorities:
            channel = ChannelFactory.build(priority_level=priority)
            db_session.add(channel)
        
        await db_session.commit()
        
        # Query channels by priority
        result = await db_session.execute(
            select(Channel).order_by(Channel.priority_level.desc())
        )
        channels = result.scalars().all()
        
        assert len(channels) == 4
        assert channels[0].priority_level == 20
        assert channels[1].priority_level == 10
        assert channels[2].priority_level == 5
        assert channels[3].priority_level == 1
    
    def test_channel_edge_cases(self):
        """Test edge cases and boundary values."""
        # Very long channel name
        long_name = "A" * 255
        channel = ChannelFactory.build(channel_name=long_name)
        assert len(channel.channel_name) == 255
        
        # Very high subscriber count
        channel = ChannelFactory.build(subscriber_count=999999999)
        assert channel.subscriber_count == 999999999
        
        # Zero values
        channel = ChannelFactory.build(
            subscriber_count=0,
            video_count=0,
            view_count=0
        )
        assert channel.subscriber_count == 0
        assert channel.video_count == 0
        assert channel.view_count == 0
    
    def test_channel_timestamps(self):
        """Test timestamp fields."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=30)
        
        channel = ChannelFactory.build(
            published_at=past,
            last_checked_at=now,
            last_video_published_at=now - timedelta(days=1)
        )
        
        assert channel.published_at < channel.last_checked_at
        assert channel.last_video_published_at < channel.last_checked_at
        assert channel.last_video_published_at > channel.published_at


class TestChannelValidation:
    """Test suite for Channel model validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_status_enum(self, db_session):
        """Test that invalid status enum values are rejected."""
        # This test depends on database constraints
        # SQLAlchemy should prevent invalid enum values at the Python level
        with pytest.raises((ValueError, IntegrityError)):
            channel = Channel(
                youtube_channel_id="UC_test_invalid_status",
                channel_name="Test Channel",
                status="invalid_status"  # Invalid enum value
            )
            db_session.add(channel)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_negative_counts(self, db_session):
        """Test handling of negative count values."""
        # The model should allow negative values as they might be valid in some cases
        channel = Channel(
            youtube_channel_id="UC_test_negative",
            channel_name="Test Channel",
            subscriber_count=-1,
            video_count=-1,
            view_count=-1
        )
        db_session.add(channel)
        await db_session.commit()
        
        # Should be allowed (business logic should handle validation)
        assert channel.subscriber_count == -1
        assert channel.video_count == -1
        assert channel.view_count == -1
    
    def test_youtube_channel_id_format(self):
        """Test various YouTube channel ID formats."""
        # Standard format
        channel = ChannelFactory.build(youtube_channel_id="UC1234567890123456789012")
        assert len(channel.youtube_channel_id) == 24
        assert channel.youtube_channel_id.startswith("UC")
        
        # Custom URL format (though this would be stored differently)
        channel = ChannelFactory.build(youtube_channel_id="c/TestChannel")
        assert "TestChannel" in channel.youtube_channel_id
        
        # Legacy username format
        channel = ChannelFactory.build(youtube_channel_id="TestUser")
        assert channel.youtube_channel_id == "TestUser"


class TestChannelRelationships:
    """Test suite for Channel model relationships."""
    
    @pytest.mark.asyncio
    async def test_channel_videos_relationship(self, db_session):
        """Test that videos relationship works correctly."""
        from app.models.video import Video
        
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        # Create videos for this channel
        video1 = Video(
            youtube_video_id="test_video_001",
            channel_id=channel.id,
            title="Test Video 1",
            published_at=datetime.now(timezone.utc)
        )
        video2 = Video(
            youtube_video_id="test_video_002",
            channel_id=channel.id,
            title="Test Video 2",
            published_at=datetime.now(timezone.utc)
        )
        
        db_session.add(video1)
        db_session.add(video2)
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(channel)
        videos = await channel.videos.all()  # Dynamic relationship
        assert len(videos) == 2
        assert video1 in videos
        assert video2 in videos
    
    @pytest.mark.asyncio
    async def test_channel_cascade_delete(self, db_session):
        """Test that deleting channel cascades to videos."""
        from app.models.video import Video
        
        # Create channel with videos
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = Video(
            youtube_video_id="test_video_cascade",
            channel_id=channel.id,
            title="Test Video",
            published_at=datetime.now(timezone.utc)
        )
        db_session.add(video)
        await db_session.commit()
        
        # Delete channel
        await db_session.delete(channel)
        await db_session.commit()
        
        # Video should be deleted too (cascade)
        result = await db_session.execute(select(Video).filter(Video.id == video.id))
        deleted_video = result.scalar_one_or_none()
        assert deleted_video is None