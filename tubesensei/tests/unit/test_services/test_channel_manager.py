"""
Unit tests for ChannelManager service.

Tests the ChannelManager service including channel operations, metadata updates,
video discovery, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from sqlalchemy import select

from app.services.channel_manager import ChannelManager
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.integrations.youtube_api import YouTubeAPIClient
from app.utils.exceptions import (
    ChannelNotFoundError,
    InvalidURLError,
    ValidationError,
    YouTubeAPIError
)
from tests.fixtures.fixtures import (
    ChannelFactory,
    VideoFactory,
    MockYouTubeAPIResponses
)


class TestChannelManagerInit:
    """Test suite for ChannelManager initialization."""
    
    def test_init_with_youtube_client(self):
        """Test initialization with provided YouTube client."""
        youtube_client = Mock(spec=YouTubeAPIClient)
        manager = ChannelManager(youtube_client)
        assert manager.youtube_client == youtube_client
    
    def test_init_without_youtube_client(self):
        """Test initialization without YouTube client."""
        manager = ChannelManager()
        assert manager.youtube_client is None
    
    @pytest.mark.asyncio
    async def test_get_youtube_client_lazy_creation(self):
        """Test YouTube client lazy creation."""
        manager = ChannelManager()
        
        with patch('app.services.channel_manager.YouTubeAPIClient') as mock_client:
            mock_instance = Mock(spec=YouTubeAPIClient)
            mock_client.return_value = mock_instance
            
            client = await manager._get_youtube_client()
            
            assert client == mock_instance
            assert manager.youtube_client == mock_instance
            mock_client.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_youtube_client_reuse_existing(self):
        """Test YouTube client reuses existing instance."""
        existing_client = Mock(spec=YouTubeAPIClient)
        manager = ChannelManager(existing_client)
        
        client = await manager._get_youtube_client()
        assert client == existing_client


class TestChannelManagerAddChannel:
    """Test suite for ChannelManager.add_channel method."""
    
    @pytest.mark.asyncio
    async def test_add_channel_valid_url(self, db_session):
        """Test adding channel with valid URL."""
        # Mock YouTube API responses
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        
        manager = ChannelManager(youtube_client)
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.return_value = {
                'type': 'channel',
                'channel_id': 'UC_test_channel_id',
                'channel_handle': None
            }
            
            channel = await manager.add_channel(
                "https://www.youtube.com/channel/UC_test_channel_id",
                db_session
            )
            
            assert channel is not None
            assert channel.youtube_channel_id == "UC_test_channel_id"
            assert channel.channel_name == "Test Channel"
            assert channel.status == ChannelStatus.ACTIVE
            
            # Verify YouTube API was called
            youtube_client.get_channel_info.assert_called_once_with("UC_test_channel_id")
    
    @pytest.mark.asyncio
    async def test_add_channel_by_handle(self, db_session):
        """Test adding channel by handle."""
        # Mock YouTube API responses
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_by_handle.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        
        manager = ChannelManager(youtube_client)
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.return_value = {
                'type': 'channel',
                'channel_id': None,
                'channel_handle': 'testchannel'
            }
            
            channel = await manager.add_channel(
                "https://www.youtube.com/@testchannel",
                db_session
            )
            
            assert channel is not None
            assert channel.youtube_channel_id == "UC_test_channel_id"
            
            # Verify correct API method was called
            youtube_client.get_channel_by_handle.assert_called_once_with("testchannel")
    
    @pytest.mark.asyncio
    async def test_add_channel_existing_channel(self, db_session):
        """Test adding existing channel returns existing record."""
        # Create existing channel
        existing_channel = ChannelFactory.build()
        db_session.add(existing_channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        manager = ChannelManager(youtube_client)
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.return_value = {
                'type': 'channel',
                'channel_id': existing_channel.youtube_channel_id,
                'channel_handle': None
            }
            
            channel = await manager.add_channel(
                f"https://www.youtube.com/channel/{existing_channel.youtube_channel_id}",
                db_session
            )
            
            assert channel.id == existing_channel.id
            assert channel.youtube_channel_id == existing_channel.youtube_channel_id
            
            # Verify YouTube API was not called
            youtube_client.get_channel_info.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_add_channel_invalid_url(self, db_session):
        """Test adding channel with invalid URL."""
        manager = ChannelManager()
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.side_effect = InvalidURLError(
                "invalid_url", "Not a valid YouTube URL"
            )
            mock_parser.validate_channel_id.return_value = False
            
            with pytest.raises(InvalidURLError):
                await manager.add_channel("invalid_url", db_session)
    
    @pytest.mark.asyncio
    async def test_add_channel_not_found_on_youtube(self, db_session):
        """Test adding channel that doesn't exist on YouTube."""
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.side_effect = ChannelNotFoundError("nonexistent_id")
        
        manager = ChannelManager(youtube_client)
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.return_value = {
                'type': 'channel',
                'channel_id': 'nonexistent_id',
                'channel_handle': None
            }
            
            with pytest.raises(ChannelNotFoundError):
                await manager.add_channel(
                    "https://www.youtube.com/channel/nonexistent_id",
                    db_session
                )
    
    @pytest.mark.asyncio
    async def test_add_channel_with_auto_discovery(self, db_session):
        """Test adding channel with automatic video discovery."""
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        
        manager = ChannelManager(youtube_client)
        
        # Mock the video discovery queue method
        with patch.object(manager, '_queue_video_discovery', new_callable=AsyncMock) as mock_queue:
            with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
                mock_parser.parse_url.return_value = {
                    'type': 'channel',
                    'channel_id': 'UC_test_channel_id',
                    'channel_handle': None
                }
                
                channel = await manager.add_channel(
                    "https://www.youtube.com/channel/UC_test_channel_id",
                    db_session,
                    auto_discover=True
                )
                
                assert channel is not None
                mock_queue.assert_called_once_with(channel.id, db_session)
    
    @pytest.mark.asyncio
    async def test_add_channel_direct_channel_id(self, db_session):
        """Test adding channel using direct channel ID."""
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        
        manager = ChannelManager(youtube_client)
        
        with patch('app.services.channel_manager.YouTubeParser') as mock_parser:
            mock_parser.parse_url.side_effect = InvalidURLError("channel_id", "Not a URL")
            mock_parser.validate_channel_id.return_value = True
            
            channel = await manager.add_channel(
                "UC_test_channel_id",
                db_session
            )
            
            assert channel is not None
            assert channel.youtube_channel_id == "UC_test_channel_id"
            youtube_client.get_channel_info.assert_called_once_with("UC_test_channel_id")


class TestChannelManagerSyncMetadata:
    """Test suite for ChannelManager.sync_channel_metadata method."""
    
    @pytest.mark.asyncio
    async def test_sync_channel_metadata_success(self, db_session):
        """Test successful channel metadata sync."""
        # Create existing channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        # Mock updated channel info
        updated_info = MockYouTubeAPIResponses.channel_info()['items'][0]
        updated_info['snippet']['title'] = 'Updated Channel Name'
        updated_info['statistics']['subscriberCount'] = '200000'
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.return_value = updated_info
        
        manager = ChannelManager(youtube_client)
        
        updated_channel = await manager.sync_channel_metadata(channel.id, db_session)
        
        assert updated_channel.channel_name == 'Updated Channel Name'
        assert updated_channel.subscriber_count == 200000
        assert updated_channel.last_checked_at is not None
        assert updated_channel.status == ChannelStatus.ACTIVE
        
        youtube_client.get_channel_info.assert_called_once_with(channel.youtube_channel_id)
    
    @pytest.mark.asyncio
    async def test_sync_channel_metadata_channel_not_found(self, db_session):
        """Test sync when channel not found in database."""
        manager = ChannelManager()
        
        with pytest.raises(ValidationError, match="Channel not found"):
            await manager.sync_channel_metadata(uuid4(), db_session)
    
    @pytest.mark.asyncio
    async def test_sync_channel_metadata_youtube_not_found(self, db_session):
        """Test sync when channel not found on YouTube."""
        # Create existing channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.side_effect = ChannelNotFoundError(channel.youtube_channel_id)
        
        manager = ChannelManager(youtube_client)
        
        with pytest.raises(ChannelNotFoundError):
            await manager.sync_channel_metadata(channel.id, db_session)
        
        # Check channel was marked as inactive
        await db_session.refresh(channel)
        assert channel.status == ChannelStatus.INACTIVE


class TestChannelManagerDiscoverVideos:
    """Test suite for ChannelManager.discover_channel_videos method."""
    
    @pytest.mark.asyncio
    async def test_discover_videos_success(self, db_session):
        """Test successful video discovery."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        # Mock YouTube API responses
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        
        manager = ChannelManager(youtube_client)
        
        videos = await manager.discover_channel_videos(channel.id, db_session)
        
        assert len(videos) == 2
        assert all(isinstance(v, Video) for v in videos)
        assert all(v.channel_id == channel.id for v in videos)
        
        # Verify YouTube API calls
        youtube_client.list_channel_videos.assert_called_once()
        youtube_client.get_video_details.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_videos_channel_not_found(self, db_session):
        """Test video discovery when channel not found."""
        manager = ChannelManager()
        
        with pytest.raises(ValidationError, match="Channel not found"):
            await manager.discover_channel_videos(uuid4(), db_session)
    
    @pytest.mark.asyncio
    async def test_discover_videos_no_new_videos(self, db_session):
        """Test video discovery when no new videos found."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = []
        
        manager = ChannelManager(youtube_client)
        
        videos = await manager.discover_channel_videos(channel.id, db_session)
        
        assert len(videos) == 0
        youtube_client.get_video_details.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_discover_videos_existing_videos_skipped(self, db_session):
        """Test that existing videos are skipped during discovery."""
        # Create channel and existing video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        existing_video = VideoFactory.build(
            channel_id=channel.id,
            youtube_video_id="test_video_001"
        )
        db_session.add(existing_video)
        await db_session.commit()
        
        # Mock YouTube API responses
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        
        manager = ChannelManager(youtube_client)
        
        videos = await manager.discover_channel_videos(channel.id, db_session)
        
        # Only one new video should be created (test_video_002)
        assert len(videos) == 1
        assert videos[0].youtube_video_id == "test_video_002"
    
    @pytest.mark.asyncio
    async def test_discover_videos_with_fetch_all_false(self, db_session):
        """Test incremental video discovery."""
        # Create channel with last fetch time
        channel = ChannelFactory.build(
            last_video_fetch_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001'])['items']
        
        manager = ChannelManager(youtube_client)
        
        videos = await manager.discover_channel_videos(
            channel.id,
            db_session,
            fetch_all=False
        )
        
        # Verify published_after parameter was used
        youtube_client.list_channel_videos.assert_called_once()
        call_args = youtube_client.list_channel_videos.call_args
        assert 'published_after' in call_args[1]
    
    @pytest.mark.asyncio
    async def test_discover_videos_api_error_handling(self, db_session):
        """Test error handling during video discovery."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.side_effect = YouTubeAPIError("API Error")
        
        manager = ChannelManager(youtube_client)
        
        with pytest.raises(YouTubeAPIError):
            await manager.discover_channel_videos(channel.id, db_session)
        
        # Check error was recorded
        await db_session.refresh(channel)
        assert channel.last_error == "API Error"


class TestChannelManagerGetChannelStatus:
    """Test suite for ChannelManager.get_channel_status method."""
    
    @pytest.mark.asyncio
    async def test_get_channel_status_success(self, db_session):
        """Test successful channel status retrieval."""
        # Create channel with videos
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video1 = VideoFactory.build(channel_id=channel.id, duration_seconds=600, view_count=1000)
        video2 = VideoFactory.build(channel_id=channel.id, duration_seconds=300, view_count=500)
        db_session.add(video1)
        db_session.add(video2)
        await db_session.commit()
        
        manager = ChannelManager()
        
        status = await manager.get_channel_status(channel.id, db_session)
        
        assert status['channel_id'] == str(channel.id)
        assert status['youtube_channel_id'] == channel.youtube_channel_id
        assert status['stored_videos'] == 2
        assert status['total_duration_hours'] == 0.25  # 900 seconds = 0.25 hours
        assert status['total_views'] == 1500
        assert status['health'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_get_channel_status_channel_not_found(self, db_session):
        """Test channel status when channel not found."""
        manager = ChannelManager()
        
        with pytest.raises(ValidationError, match="Channel not found"):
            await manager.get_channel_status(uuid4(), db_session)
    
    @pytest.mark.asyncio
    async def test_calculate_channel_health_inactive(self):
        """Test channel health calculation for inactive channel."""
        manager = ChannelManager()
        channel = ChannelFactory.build(status=ChannelStatus.INACTIVE)
        
        health = manager._calculate_channel_health(channel, [], [])
        assert health == 'inactive'
    
    @pytest.mark.asyncio
    async def test_calculate_channel_health_error(self):
        """Test channel health calculation for channel with error."""
        manager = ChannelManager()
        channel = ChannelFactory.build(
            status=ChannelStatus.ACTIVE,
            last_error="Some error occurred"
        )
        
        health = manager._calculate_channel_health(channel, [], [])
        assert health == 'error'
    
    @pytest.mark.asyncio
    async def test_calculate_channel_health_processing(self):
        """Test channel health calculation for channel with pending jobs."""
        manager = ChannelManager()
        channel = ChannelFactory.build(status=ChannelStatus.ACTIVE)
        pending_job = Mock()
        
        health = manager._calculate_channel_health(channel, [], [pending_job])
        assert health == 'processing'
    
    @pytest.mark.asyncio
    async def test_calculate_channel_health_empty(self):
        """Test channel health calculation for empty channel."""
        manager = ChannelManager()
        channel = ChannelFactory.build(status=ChannelStatus.ACTIVE)
        
        health = manager._calculate_channel_health(channel, [], [])
        assert health == 'empty'
    
    @pytest.mark.asyncio
    async def test_calculate_channel_health_stale(self):
        """Test channel health calculation for stale channel."""
        manager = ChannelManager()
        channel = ChannelFactory.build(
            status=ChannelStatus.ACTIVE,
            last_checked_at=datetime.now(timezone.utc) - timedelta(days=10)
        )
        video = Mock()
        
        health = manager._calculate_channel_health(channel, [video], [])
        assert health == 'stale'


class TestChannelManagerListChannels:
    """Test suite for ChannelManager.list_channels method."""
    
    @pytest.mark.asyncio
    async def test_list_channels_all(self, db_session):
        """Test listing all channels."""
        # Create multiple channels
        channels = [
            ChannelFactory.build(subscriber_count=10000),
            ChannelFactory.build(subscriber_count=5000),
            ChannelFactory.build(subscriber_count=15000)
        ]
        
        for channel in channels:
            db_session.add(channel)
        await db_session.commit()
        
        manager = ChannelManager()
        
        result = await manager.list_channels(db_session)
        
        # Should be ordered by subscriber count desc
        assert len(result) == 3
        assert result[0].subscriber_count == 15000
        assert result[1].subscriber_count == 10000
        assert result[2].subscriber_count == 5000
    
    @pytest.mark.asyncio
    async def test_list_channels_with_status_filter(self, db_session):
        """Test listing channels with status filter."""
        # Create channels with different statuses
        active_channel = ChannelFactory.build(status=ChannelStatus.ACTIVE)
        paused_channel = ChannelFactory.build(status=ChannelStatus.PAUSED)
        
        db_session.add(active_channel)
        db_session.add(paused_channel)
        await db_session.commit()
        
        manager = ChannelManager()
        
        result = await manager.list_channels(db_session, status=ChannelStatus.ACTIVE)
        
        assert len(result) == 1
        assert result[0].status == ChannelStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_list_channels_with_pagination(self, db_session):
        """Test listing channels with pagination."""
        # Create multiple channels
        for i in range(5):
            channel = ChannelFactory.build()
            db_session.add(channel)
        await db_session.commit()
        
        manager = ChannelManager()
        
        # Get first 2 channels
        result1 = await manager.list_channels(db_session, limit=2, offset=0)
        assert len(result1) == 2
        
        # Get next 2 channels
        result2 = await manager.list_channels(db_session, limit=2, offset=2)
        assert len(result2) == 2
        
        # Ensure no overlap
        result1_ids = {c.id for c in result1}
        result2_ids = {c.id for c in result2}
        assert result1_ids.isdisjoint(result2_ids)


class TestChannelManagerDeleteChannel:
    """Test suite for ChannelManager.delete_channel method."""
    
    @pytest.mark.asyncio
    async def test_delete_channel_success(self, db_session):
        """Test successful channel deletion."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        manager = ChannelManager()
        
        result = await manager.delete_channel(channel.id, db_session)
        
        assert result is True
        
        # Verify channel was deleted
        result = await db_session.execute(select(Channel).filter(Channel.id == channel.id))
        deleted_channel = result.scalar_one_or_none()
        assert deleted_channel is None
    
    @pytest.mark.asyncio
    async def test_delete_channel_not_found(self, db_session):
        """Test deleting non-existent channel."""
        manager = ChannelManager()
        
        result = await manager.delete_channel(uuid4(), db_session)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_channel_with_cascade(self, db_session):
        """Test channel deletion with cascade option."""
        # Create channel with video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.commit()
        
        manager = ChannelManager()
        
        result = await manager.delete_channel(channel.id, db_session, cascade=True)
        
        assert result is True
        
        # Both channel and video should be deleted due to model cascade rules
        channel_result = await db_session.execute(select(Channel).filter(Channel.id == channel.id))
        assert channel_result.scalar_one_or_none() is None
        
        video_result = await db_session.execute(select(Video).filter(Video.id == video.id))
        assert video_result.scalar_one_or_none() is None


class TestChannelManagerQueueMethods:
    """Test suite for ChannelManager queueing methods."""
    
    @pytest.mark.asyncio
    async def test_queue_video_discovery(self, db_session):
        """Test queueing video discovery job."""
        manager = ChannelManager()
        channel_id = uuid4()
        
        await manager._queue_video_discovery(channel_id, db_session)
        
        # Verify job was created
        result = await db_session.execute(
            select(ProcessingJob).filter(
                ProcessingJob.channel_id == channel_id,
                ProcessingJob.job_type == JobType.VIDEO_DISCOVERY
            )
        )
        job = result.scalar_one_or_none()
        
        assert job is not None
        assert job.status == JobStatus.PENDING
        assert job.priority == 5
    
    @pytest.mark.asyncio
    async def test_queue_transcript_processing(self, db_session):
        """Test queueing transcript processing job."""
        manager = ChannelManager()
        video_id = uuid4()
        
        await manager._queue_transcript_processing(video_id, db_session)
        
        # Verify job was created
        result = await db_session.execute(
            select(ProcessingJob).filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == JobType.TRANSCRIPT_FETCH
            )
        )
        job = result.scalar_one_or_none()
        
        assert job is not None
        assert job.status == JobStatus.PENDING
        assert job.priority == 3