"""
Unit tests for VideoDiscovery service.

Tests the VideoDiscovery service including video discovery, filtering,
metadata updates, and search functionality.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from app.services.video_discovery import VideoDiscovery
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.processing_job import ProcessingJob, JobType
from app.models.filters import VideoFilters
from app.integrations.youtube_api import YouTubeAPIClient
from app.utils.exceptions import ValidationError, QuotaExceededError, YouTubeAPIError
from tests.fixtures.fixtures import (
    ChannelFactory,
    VideoFactory,
    MockYouTubeAPIResponses
)


class TestVideoDiscoveryInit:
    """Test suite for VideoDiscovery initialization."""
    
    def test_init_with_youtube_client(self):
        """Test initialization with provided YouTube client."""
        youtube_client = Mock(spec=YouTubeAPIClient)
        discovery = VideoDiscovery(youtube_client)
        assert discovery.youtube_client == youtube_client
    
    def test_init_without_youtube_client(self):
        """Test initialization without YouTube client."""
        discovery = VideoDiscovery()
        assert discovery.youtube_client is None
    
    @pytest.mark.asyncio
    async def test_get_youtube_client_lazy_creation(self):
        """Test YouTube client lazy creation."""
        discovery = VideoDiscovery()
        
        with patch('app.services.video_discovery.YouTubeAPIClient') as mock_client:
            mock_instance = Mock(spec=YouTubeAPIClient)
            mock_client.return_value = mock_instance
            
            client = await discovery._get_youtube_client()
            
            assert client == mock_instance
            assert discovery.youtube_client == mock_instance
            mock_client.assert_called_once()


class TestVideoDiscoveryDiscoverVideos:
    """Test suite for VideoDiscovery.discover_videos method."""
    
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
        
        discovery = VideoDiscovery(youtube_client)
        
        # Mock the video processing method
        processed_videos = [
            VideoFactory.build(channel_id=channel.id, youtube_video_id="test_video_001"),
            VideoFactory.build(channel_id=channel.id, youtube_video_id="test_video_002")
        ]
        
        with patch.object(discovery, '_process_and_store_videos', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = processed_videos
            
            videos = await discovery.discover_videos(channel.id, db_session)
            
            assert len(videos) == 2
            assert all(v.channel_id == channel.id for v in videos)
            
            # Verify YouTube API calls
            youtube_client.list_channel_videos.assert_called_once()
            youtube_client.get_video_details.assert_called_once()
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_videos_channel_not_found(self, db_session):
        """Test video discovery when channel not found."""
        discovery = VideoDiscovery()
        
        with pytest.raises(ValidationError, match="Channel not found"):
            await discovery.discover_videos(uuid4(), db_session)
    
    @pytest.mark.asyncio
    async def test_discover_videos_with_filters(self, db_session):
        """Test video discovery with filters applied."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        # Create filter
        filters = VideoFilters(
            min_duration_seconds=300,
            published_after=datetime.now(timezone.utc) - timedelta(days=30)
        )
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001'])['items']
        
        discovery = VideoDiscovery(youtube_client)
        
        with patch.object(discovery, '_process_and_store_videos', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = []
            
            await discovery.discover_videos(channel.id, db_session, filters=filters)
            
            # Verify filters were passed to YouTube API
            call_args = youtube_client.list_channel_videos.call_args
            assert 'published_after' in call_args[1]
            assert 'published_before' in call_args[1]
            
            # Verify filters were passed to processing
            mock_process.assert_called_once()
            _, filters_arg, _, _ = mock_process.call_args[0]
            assert filters_arg == filters
    
    @pytest.mark.asyncio
    async def test_discover_videos_no_videos_found(self, db_session):
        """Test video discovery when no videos found."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = []
        
        discovery = VideoDiscovery(youtube_client)
        
        videos = await discovery.discover_videos(channel.id, db_session)
        
        assert len(videos) == 0
        youtube_client.get_video_details.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_discover_videos_quota_exceeded(self, db_session):
        """Test video discovery with quota exceeded error."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.side_effect = QuotaExceededError(10000, 10000)
        
        discovery = VideoDiscovery(youtube_client)
        
        with pytest.raises(QuotaExceededError):
            await discovery.discover_videos(channel.id, db_session)
    
    @pytest.mark.asyncio
    async def test_discover_videos_api_error(self, db_session):
        """Test video discovery with API error."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.side_effect = YouTubeAPIError("API Error")
        
        discovery = VideoDiscovery(youtube_client)
        
        with pytest.raises(YouTubeAPIError):
            await discovery.discover_videos(channel.id, db_session)
        
        # Check error was recorded
        await db_session.refresh(channel)
        assert channel.last_error == "API Error"
    
    @pytest.mark.asyncio
    async def test_discover_videos_from_database(self, db_session):
        """Test discovering videos from database when not fetching from API."""
        # Create channel with recent fetch
        channel = ChannelFactory.build(
            last_video_fetch_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        db_session.add(channel)
        await db_session.flush()
        
        # Create existing videos
        video1 = VideoFactory.build(
            channel_id=channel.id,
            duration_seconds=600,
            view_count=5000,
            published_at=datetime.now(timezone.utc) - timedelta(days=5)
        )
        video2 = VideoFactory.build(
            channel_id=channel.id,
            duration_seconds=200,
            view_count=1000,
            published_at=datetime.now(timezone.utc) - timedelta(days=10)
        )
        db_session.add(video1)
        db_session.add(video2)
        await db_session.commit()
        
        discovery = VideoDiscovery()
        
        # Apply filters that should exclude video2
        filters = VideoFilters(min_duration_seconds=300)
        
        with patch.object(discovery, '_apply_advanced_filters') as mock_filter:
            mock_filter.side_effect = lambda v, f: v.duration_seconds >= 300
            
            videos = await discovery.discover_videos(
                channel.id,
                db_session,
                filters=filters,
                force_refresh=False
            )
            
            assert len(videos) == 1
            assert videos[0].id == video1.id
    
    @pytest.mark.asyncio
    async def test_discover_videos_force_refresh(self, db_session):
        """Test video discovery with force refresh."""
        # Create channel with recent fetch
        channel = ChannelFactory.build(
            last_video_fetch_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001'])['items']
        
        discovery = VideoDiscovery(youtube_client)
        
        with patch.object(discovery, '_process_and_store_videos', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = []
            
            await discovery.discover_videos(
                channel.id,
                db_session,
                force_refresh=True
            )
            
            # Should fetch from YouTube despite recent fetch
            youtube_client.list_channel_videos.assert_called_once()
            mock_process.assert_called_once()


class TestVideoDiscoveryBatchDiscover:
    """Test suite for VideoDiscovery.batch_discover method."""
    
    @pytest.mark.asyncio
    async def test_batch_discover_success(self, db_session):
        """Test successful batch video discovery."""
        # Create channels
        channel1 = ChannelFactory.build()
        channel2 = ChannelFactory.build()
        db_session.add(channel1)
        db_session.add(channel2)
        await db_session.commit()
        
        discovery = VideoDiscovery()
        
        # Mock successful discovery for both channels
        mock_videos = [
            [VideoFactory.build(channel_id=channel1.id)],
            [VideoFactory.build(channel_id=channel2.id), VideoFactory.build(channel_id=channel2.id)]
        ]
        
        with patch.object(discovery, 'discover_videos', new_callable=AsyncMock) as mock_discover:
            mock_discover.side_effect = mock_videos
            
            results = await discovery.batch_discover(
                [channel1.id, channel2.id],
                db_session
            )
            
            assert results['total_channels'] == 2
            assert results['successful_channels'] == 2
            assert results['failed_channels'] == 0
            assert results['total_videos_discovered'] == 3
            assert len(results['errors']) == 0
            
            # Check individual channel results
            assert results['channel_results'][str(channel1.id)]['status'] == 'success'
            assert results['channel_results'][str(channel1.id)]['videos_discovered'] == 1
            assert results['channel_results'][str(channel2.id)]['status'] == 'success'
            assert results['channel_results'][str(channel2.id)]['videos_discovered'] == 2
    
    @pytest.mark.asyncio
    async def test_batch_discover_partial_failure(self, db_session):
        """Test batch discovery with some failures."""
        # Create channels
        channel1 = ChannelFactory.build()
        channel2 = ChannelFactory.build()
        db_session.add(channel1)
        db_session.add(channel2)
        await db_session.commit()
        
        discovery = VideoDiscovery()
        
        # Mock one success, one failure
        with patch.object(discovery, 'discover_videos', new_callable=AsyncMock) as mock_discover:
            mock_discover.side_effect = [
                [VideoFactory.build(channel_id=channel1.id)],
                ValidationError("Channel error")
            ]
            
            results = await discovery.batch_discover(
                [channel1.id, channel2.id],
                db_session
            )
            
            assert results['total_channels'] == 2
            assert results['successful_channels'] == 1
            assert results['failed_channels'] == 1
            assert results['total_videos_discovered'] == 1
            assert len(results['errors']) == 1
            assert "Channel error" in results['errors'][0]
            
            # Check individual channel results
            assert results['channel_results'][str(channel1.id)]['status'] == 'success'
            assert results['channel_results'][str(channel2.id)]['status'] == 'failed'


class TestVideoDiscoveryUpdateMetadata:
    """Test suite for VideoDiscovery.update_video_metadata method."""
    
    @pytest.mark.asyncio
    async def test_update_video_metadata_success(self, db_session):
        """Test successful video metadata update."""
        # Create videos
        video1 = VideoFactory.build(
            youtube_video_id="test_video_001",
            view_count=1000,
            title="Old Title 1"
        )
        video2 = VideoFactory.build(
            youtube_video_id="test_video_002",
            view_count=2000,
            title="Old Title 2"
        )
        db_session.add(video1)
        db_session.add(video2)
        await db_session.commit()
        
        # Mock updated video details
        updated_details = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        updated_details[0]['snippet']['title'] = "Updated Title 1"
        updated_details[0]['statistics']['viewCount'] = '1500'
        updated_details[1]['snippet']['title'] = "Updated Title 2"
        updated_details[1]['statistics']['viewCount'] = '2500'
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_video_details.return_value = updated_details
        
        discovery = VideoDiscovery(youtube_client)
        
        updated_videos = await discovery.update_video_metadata(
            [video1.id, video2.id],
            db_session
        )
        
        assert len(updated_videos) == 2
        
        # Check updates
        updated_video1 = next(v for v in updated_videos if v.id == video1.id)
        updated_video2 = next(v for v in updated_videos if v.id == video2.id)
        
        assert updated_video1.title == "Updated Title 1"
        assert updated_video1.view_count == 1500
        assert updated_video2.title == "Updated Title 2"
        assert updated_video2.view_count == 2500
        
        youtube_client.get_video_details.assert_called_once_with(['test_video_001', 'test_video_002'])
    
    @pytest.mark.asyncio
    async def test_update_video_metadata_empty_list(self, db_session):
        """Test updating metadata with empty video list."""
        discovery = VideoDiscovery()
        
        result = await discovery.update_video_metadata([], db_session)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_update_video_metadata_api_error(self, db_session):
        """Test metadata update with API error."""
        # Create video
        video = VideoFactory.build()
        db_session.add(video)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_video_details.side_effect = YouTubeAPIError("API Error")
        
        discovery = VideoDiscovery(youtube_client)
        
        with pytest.raises(YouTubeAPIError):
            await discovery.update_video_metadata([video.id], db_session)


class TestVideoDiscoverySearchAndDiscover:
    """Test suite for VideoDiscovery.search_and_discover method."""
    
    @pytest.mark.asyncio
    async def test_search_and_discover_success(self, db_session):
        """Test successful video search and discovery."""
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.search_videos.return_value = [
            {'video_id': 'search_result_001', 'title': 'Search Result 1'},
            {'video_id': 'search_result_002', 'title': 'Search Result 2'}
        ]
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['search_result_001', 'search_result_002'])['items']
        
        discovery = VideoDiscovery(youtube_client)
        
        # Mock channel creation
        with patch.object(discovery, '_ensure_channel_exists', new_callable=AsyncMock) as mock_ensure:
            mock_channel = ChannelFactory.build()
            mock_ensure.return_value = mock_channel
            
            videos = await discovery.search_and_discover(
                "programming tutorial",
                db_session
            )
            
            assert len(videos) == 2
            assert all(isinstance(v, Video) for v in videos)
            assert all(v.channel_id == mock_channel.id for v in videos)
            
            youtube_client.search_videos.assert_called_once()
            youtube_client.get_video_details.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_and_discover_with_channel_filter(self, db_session):
        """Test search with specific channel filter."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.search_videos.return_value = []
        
        discovery = VideoDiscovery(youtube_client)
        
        await discovery.search_and_discover(
            "tutorial",
            db_session,
            channel_id=channel.id
        )
        
        # Verify channel ID was passed to search
        call_args = youtube_client.search_videos.call_args
        assert call_args[1]['channel_id'] == channel.youtube_channel_id
    
    @pytest.mark.asyncio
    async def test_search_and_discover_existing_videos_skipped(self, db_session):
        """Test that existing videos are skipped during search discovery."""
        # Create existing video
        existing_video = VideoFactory.build(youtube_video_id="search_result_001")
        db_session.add(existing_video)
        await db_session.commit()
        
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.search_videos.return_value = [
            {'video_id': 'search_result_001', 'title': 'Existing Video'},
            {'video_id': 'search_result_002', 'title': 'New Video'}
        ]
        youtube_client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['search_result_001', 'search_result_002'])['items']
        
        discovery = VideoDiscovery(youtube_client)
        
        with patch.object(discovery, '_ensure_channel_exists', new_callable=AsyncMock) as mock_ensure:
            mock_channel = ChannelFactory.build()
            mock_ensure.return_value = mock_channel
            
            videos = await discovery.search_and_discover(
                "tutorial",
                db_session
            )
            
            # Only the new video should be created
            assert len(videos) == 1
            assert videos[0].youtube_video_id == "search_result_002"


class TestVideoDiscoveryHelperMethods:
    """Test suite for VideoDiscovery helper methods."""
    
    @pytest.mark.asyncio
    async def test_process_and_store_videos(self, db_session):
        """Test processing and storing video details."""
        # Create channel
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.commit()
        
        # Mock video details
        video_details = MockYouTubeAPIResponses.video_details(['test_video_001', 'test_video_002'])['items']
        
        # Mock filters
        filters = Mock()
        filters.apply_to_video.return_value = True
        
        discovery = VideoDiscovery()
        
        # Mock transcript processing queue
        with patch.object(discovery, '_queue_transcript_processing', new_callable=AsyncMock) as mock_queue:
            videos = await discovery._process_and_store_videos(
                video_details,
                channel,
                db_session,
                filters
            )
            
            assert len(videos) == 2
            assert all(v.channel_id == channel.id for v in videos)
            assert all(v.has_captions for v in videos)  # Mock data has captions
            
            # Verify transcript processing was queued for videos with captions
            assert mock_queue.call_count == 2
    
    def test_apply_advanced_filters_title_contains(self):
        """Test advanced filtering by title contains."""
        discovery = VideoDiscovery()
        
        video = VideoFactory.build(title="Python Programming Tutorial")
        
        # Should match
        filters = VideoFilters(title_contains=["programming", "tutorial"])
        assert discovery._apply_advanced_filters(video, filters) is True
        
        # Should not match
        filters = VideoFilters(title_contains=["javascript", "react"])
        assert discovery._apply_advanced_filters(video, filters) is False
    
    def test_apply_advanced_filters_title_excludes(self):
        """Test advanced filtering by title excludes."""
        discovery = VideoDiscovery()
        
        video = VideoFactory.build(title="Python Programming Tutorial")
        
        # Should be excluded
        filters = VideoFilters(title_excludes=["programming"])
        assert discovery._apply_advanced_filters(video, filters) is False
        
        # Should not be excluded
        filters = VideoFilters(title_excludes=["javascript"])
        assert discovery._apply_advanced_filters(video, filters) is True
    
    def test_apply_advanced_filters_tags(self):
        """Test advanced filtering by tags."""
        discovery = VideoDiscovery()
        
        video = VideoFactory.build(tags=["python", "programming", "tutorial"])
        
        # Required tags - all must be present
        filters = VideoFilters(required_tags=["python", "programming"])
        assert discovery._apply_advanced_filters(video, filters) is True
        
        filters = VideoFilters(required_tags=["python", "javascript"])
        assert discovery._apply_advanced_filters(video, filters) is False
        
        # Any tags - at least one must be present
        filters = VideoFilters(any_tags=["javascript", "programming"])
        assert discovery._apply_advanced_filters(video, filters) is True
        
        filters = VideoFilters(any_tags=["javascript", "react"])
        assert discovery._apply_advanced_filters(video, filters) is False
        
        # Excluded tags - none should be present
        filters = VideoFilters(excluded_tags=["programming"])
        assert discovery._apply_advanced_filters(video, filters) is False
        
        filters = VideoFilters(excluded_tags=["javascript"])
        assert discovery._apply_advanced_filters(video, filters) is True
    
    def test_apply_advanced_filters_language(self):
        """Test advanced filtering by language."""
        discovery = VideoDiscovery()
        
        video = VideoFactory.build(language="en")
        
        # Single language filter
        filters = VideoFilters(language="en")
        assert discovery._apply_advanced_filters(video, filters) is True
        
        filters = VideoFilters(language="es")
        assert discovery._apply_advanced_filters(video, filters) is False
        
        # Multiple languages filter
        filters = VideoFilters(languages=["en", "es"])
        assert discovery._apply_advanced_filters(video, filters) is True
        
        filters = VideoFilters(languages=["fr", "de"])
        assert discovery._apply_advanced_filters(video, filters) is False
    
    def test_apply_advanced_filters_captions(self):
        """Test advanced filtering by captions requirement."""
        discovery = VideoDiscovery()
        
        video_with_captions = VideoFactory.build(has_captions=True)
        video_without_captions = VideoFactory.build(has_captions=False)
        
        filters = VideoFilters(require_captions=True)
        assert discovery._apply_advanced_filters(video_with_captions, filters) is True
        assert discovery._apply_advanced_filters(video_without_captions, filters) is False
        
        filters = VideoFilters(require_captions=False)
        assert discovery._apply_advanced_filters(video_with_captions, filters) is True
        assert discovery._apply_advanced_filters(video_without_captions, filters) is True
    
    def test_apply_advanced_filters_shorts_exclusion(self):
        """Test advanced filtering by shorts exclusion."""
        discovery = VideoDiscovery()
        
        short_video = VideoFactory.build(duration_seconds=45)
        regular_video = VideoFactory.build(duration_seconds=300)
        
        filters = VideoFilters(exclude_shorts=True)
        assert discovery._apply_advanced_filters(short_video, filters) is False
        assert discovery._apply_advanced_filters(regular_video, filters) is True
        
        filters = VideoFilters(exclude_shorts=False)
        assert discovery._apply_advanced_filters(short_video, filters) is True
        assert discovery._apply_advanced_filters(regular_video, filters) is True
    
    def test_apply_advanced_filters_live_content_exclusion(self):
        """Test advanced filtering by live content exclusion."""
        discovery = VideoDiscovery()
        
        live_video = VideoFactory.build(is_live_content=True)
        regular_video = VideoFactory.build(is_live_content=False)
        
        filters = VideoFilters(exclude_live=True)
        assert discovery._apply_advanced_filters(live_video, filters) is False
        assert discovery._apply_advanced_filters(regular_video, filters) is True
        
        filters = VideoFilters(exclude_live=False)
        assert discovery._apply_advanced_filters(live_video, filters) is True
        assert discovery._apply_advanced_filters(regular_video, filters) is True
    
    @pytest.mark.asyncio
    async def test_ensure_channel_exists_existing(self, db_session):
        """Test ensuring existing channel exists."""
        # Create existing channel
        existing_channel = ChannelFactory.build()
        db_session.add(existing_channel)
        await db_session.commit()
        
        discovery = VideoDiscovery()
        
        result = await discovery._ensure_channel_exists(
            existing_channel.youtube_channel_id,
            db_session
        )
        
        assert result.id == existing_channel.id
        assert result.youtube_channel_id == existing_channel.youtube_channel_id
    
    @pytest.mark.asyncio
    async def test_ensure_channel_exists_new(self, db_session):
        """Test ensuring new channel exists."""
        youtube_client = AsyncMock(spec=YouTubeAPIClient)
        youtube_client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
        
        discovery = VideoDiscovery(youtube_client)
        
        channel = await discovery._ensure_channel_exists(
            "UC_new_channel_id",
            db_session
        )
        
        assert channel is not None
        assert channel.youtube_channel_id == "UC_test_channel_id"
        assert channel.status == ChannelStatus.ACTIVE
        
        youtube_client.get_channel_info.assert_called_once_with("UC_new_channel_id")
    
    @pytest.mark.asyncio
    async def test_queue_transcript_processing(self, db_session):
        """Test queueing transcript processing job."""
        discovery = VideoDiscovery()
        
        video = VideoFactory.build()
        db_session.add(video)
        await db_session.commit()
        
        await discovery._queue_transcript_processing(video, db_session)
        await db_session.commit()
        
        # Verify job was created
        from sqlalchemy import select
        result = await db_session.execute(
            select(ProcessingJob).filter(
                ProcessingJob.video_id == video.id,
                ProcessingJob.job_type == JobType.TRANSCRIPT_FETCH
            )
        )
        job = result.scalar_one_or_none()
        
        assert job is not None
        assert job.status.name == "PENDING"
        assert job.priority == 3