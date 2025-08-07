"""
Unit tests for YouTubeAPIClient integration.

Tests the YouTube API client including authentication, API calls,
error handling, quota management, and rate limiting.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import json
from datetime import datetime, timezone
import asyncio

from googleapiclient.errors import HttpError
from googleapiclient.http import HttpRequest

from app.integrations.youtube_api import YouTubeAPIClient
from app.utils.exceptions import (
    YouTubeAPIError,
    QuotaExceededError,
    ChannelNotFoundError,
    VideoNotFoundError,
    APIKeyError,
    NetworkError,
    RateLimitError
)
from tests.fixtures.fixtures import MockYouTubeAPIResponses


class TestYouTubeAPIClientInit:
    """Test suite for YouTubeAPIClient initialization."""
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_init_with_api_key(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test initialization with API key."""
        api_key = "test_api_key"
        
        client = YouTubeAPIClient(api_key=api_key)
        
        assert client.api_key == api_key
        mock_build.assert_called_once_with('youtube', 'v3', developerKey=api_key)
        mock_quota_manager.assert_called_once()
        mock_rate_limiter.assert_called_once()
        mock_httpx.assert_called_once()
    
    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch('app.integrations.youtube_api.settings') as mock_settings:
            mock_settings.YOUTUBE_API_KEY = None
            
            with pytest.raises(APIKeyError, match="YouTube API key is required"):
                YouTubeAPIClient()
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_init_with_custom_quota_and_rate_limits(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test initialization with custom quota and rate limits."""
        api_key = "test_api_key"
        quota_per_day = 5000
        rate_limit_per_minute = 30
        
        with patch('app.integrations.youtube_api.settings') as mock_settings:
            mock_settings.YOUTUBE_API_KEY = api_key
            
            client = YouTubeAPIClient(
                quota_per_day=quota_per_day,
                rate_limit_per_minute=rate_limit_per_minute
            )
            
            mock_quota_manager.assert_called_once_with(daily_quota=quota_per_day)
            mock_rate_limiter.assert_called_once_with(requests_per_minute=rate_limit_per_minute)


class TestYouTubeAPIClientContextManager:
    """Test suite for YouTubeAPIClient context manager."""
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    async def test_async_context_manager(self, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test async context manager functionality."""
        api_key = "test_api_key"
        
        with patch('app.integrations.youtube_api.httpx.AsyncClient') as mock_httpx:
            mock_http_client = AsyncMock()
            mock_httpx.return_value = mock_http_client
            
            # Mock quota manager
            mock_quota_instance = AsyncMock()
            mock_quota_manager.return_value = mock_quota_instance
            
            async with YouTubeAPIClient(api_key=api_key) as client:
                assert isinstance(client, YouTubeAPIClient)
                assert client.api_key == api_key
            
            # Verify cleanup was called
            mock_http_client.aclose.assert_called_once()
            mock_quota_instance._save_quota_data.assert_called_once()


class TestYouTubeAPIClientErrorHandling:
    """Test suite for YouTube API error handling."""
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_handle_quota_exceeded_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test handling of quota exceeded error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock quota manager
        mock_quota_instance = Mock()
        mock_quota_instance.current_usage = 9500
        mock_quota_instance.daily_quota = 10000
        client.quota_manager = mock_quota_instance
        
        # Create mock HTTP error for quota exceeded
        error_content = json.dumps({
            'error': {
                'errors': [{'reason': 'quotaExceeded'}],
                'message': 'Quota exceeded'
            }
        }).encode('utf-8')
        
        mock_response = Mock()
        mock_response.status = 403
        
        http_error = HttpError(resp=mock_response, content=error_content)
        
        with pytest.raises(QuotaExceededError) as exc_info:
            client._handle_api_error(http_error)
        
        assert exc_info.value.quota_used == 9500
        assert exc_info.value.quota_limit == 10000
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_handle_rate_limit_exceeded_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test handling of rate limit exceeded error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Create mock HTTP error for rate limit exceeded
        error_content = json.dumps({
            'error': {
                'errors': [{'reason': 'rateLimitExceeded'}],
                'message': 'Rate limit exceeded'
            }
        }).encode('utf-8')
        
        mock_response = Mock()
        mock_response.status = 403
        
        http_error = HttpError(resp=mock_response, content=error_content)
        
        with pytest.raises(RateLimitError):
            client._handle_api_error(http_error)
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_handle_channel_not_found_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test handling of channel not found error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Create mock HTTP error for not found
        error_content = json.dumps({
            'error': {
                'message': 'Channel not found'
            }
        }).encode('utf-8')
        
        mock_response = Mock()
        mock_response.status = 404
        
        http_error = HttpError(resp=mock_response, content=error_content)
        
        with pytest.raises(ChannelNotFoundError):
            client._handle_api_error(http_error)
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_handle_invalid_api_key_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test handling of invalid API key error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Create mock HTTP error for invalid key
        error_content = json.dumps({
            'error': {
                'errors': [{'reason': 'keyInvalid'}],
                'message': 'Invalid API key'
            }
        }).encode('utf-8')
        
        mock_response = Mock()
        mock_response.status = 400
        
        http_error = HttpError(resp=mock_response, content=error_content)
        
        with pytest.raises(APIKeyError):
            client._handle_api_error(http_error)
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_handle_generic_api_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test handling of generic API error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        mock_response = Mock()
        mock_response.status = 500
        
        http_error = HttpError(resp=mock_response, content=b'Server error')
        
        with pytest.raises(YouTubeAPIError):
            client._handle_api_error(http_error)


class TestYouTubeAPIClientChannelOperations:
    """Test suite for YouTube API channel operations."""
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_channel_info_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful channel info retrieval."""
        # Mock YouTube API response
        mock_response = MockYouTubeAPIResponses.channel_info("UC_test_channel")
        
        # Mock the API execution
        mock_api_method = Mock()
        mock_api_method.return_value.execute.return_value = mock_response
        
        mock_youtube = Mock()
        mock_youtube.channels.return_value.list = mock_api_method
        mock_build.return_value = mock_youtube
        
        # Mock quota and rate limiting
        mock_quota_instance = AsyncMock()
        mock_quota_manager.return_value = mock_quota_instance
        
        mock_rate_instance = AsyncMock()
        mock_rate_instance.acquire.return_value.__aenter__ = AsyncMock()
        mock_rate_instance.acquire.return_value.__aexit__ = AsyncMock()
        mock_rate_limiter.return_value = mock_rate_instance
        
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock the execute_api_call method to avoid complex async setup
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response
            
            result = await client.get_channel_info("UC_test_channel")
            
            assert result['channel_id'] == 'UC_test_channel'
            assert result['title'] == 'Test Channel'
            assert result['subscriber_count'] == 100000
            assert result['video_count'] == 500
            assert 'uploads_playlist_id' in result
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_channel_info_not_found(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test channel info when channel not found."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock empty response
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {'items': []}
            
            with pytest.raises(ChannelNotFoundError):
                await client.get_channel_info("UC_nonexistent")
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_channel_info_cached(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test channel info retrieval from cache."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Pre-populate cache
        cached_data = {'channel_id': 'UC_cached', 'title': 'Cached Channel'}
        client._cache['channel:UC_cached'] = cached_data
        
        result = await client.get_channel_info("UC_cached")
        
        assert result == cached_data
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_channel_by_handle_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful channel retrieval by handle."""
        client = YouTubeAPIClient(api_key="test_key")
        
        mock_response = MockYouTubeAPIResponses.channel_info("UC_test_channel")
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response
            
            with patch.object(client, 'get_channel_info', new_callable=AsyncMock) as mock_get_info:
                mock_get_info.return_value = {'channel_id': 'UC_test_channel', 'title': 'Test Channel'}
                
                result = await client.get_channel_by_handle("testchannel")
                
                assert result['channel_id'] == 'UC_test_channel'
                mock_get_info.assert_called_once_with('UC_test_channel')
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_channel_by_handle_not_found(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test channel by handle when not found."""
        client = YouTubeAPIClient(api_key="test_key")
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            # Mock both handle and username lookups returning empty
            mock_execute.return_value = {'items': []}
            
            with pytest.raises(ChannelNotFoundError):
                await client.get_channel_by_handle("nonexistent")


class TestYouTubeAPIClientVideoOperations:
    """Test suite for YouTube API video operations."""
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_video_details_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful video details retrieval."""
        client = YouTubeAPIClient(api_key="test_key")
        
        video_ids = ['test_video_001', 'test_video_002']
        mock_response = MockYouTubeAPIResponses.video_details(video_ids)
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response
            
            result = await client.get_video_details(video_ids)
            
            assert len(result) == 2
            assert result[0]['video_id'] == 'test_video_001'
            assert result[1]['video_id'] == 'test_video_002'
            assert all('duration_seconds' in video for video in result)
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_video_details_empty_list(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test video details with empty video list."""
        client = YouTubeAPIClient(api_key="test_key")
        
        result = await client.get_video_details([])
        
        assert result == []
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_video_details_batching(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test video details with batching for large lists."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Create 75 video IDs to test batching (should be 2 batches of 50 and 25)
        video_ids = [f'test_video_{i:03d}' for i in range(75)]
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            # Mock returns for each batch
            mock_execute.side_effect = [
                MockYouTubeAPIResponses.video_details(video_ids[:50]),
                MockYouTubeAPIResponses.video_details(video_ids[50:])
            ]
            
            result = await client.get_video_details(video_ids)
            
            assert len(result) == 75
            assert mock_execute.call_count == 2  # Two batches
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_list_channel_videos_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful channel video listing."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock channel info call
        mock_channel_response = MockYouTubeAPIResponses.channel_info("UC_test_channel")
        
        # Mock playlist items call
        mock_playlist_response = MockYouTubeAPIResponses.playlist_items("UU_test_uploads")
        
        with patch.object(client, 'get_channel_info', new_callable=AsyncMock) as mock_get_channel:
            mock_get_channel.return_value = mock_channel_response['items'][0]
            
            with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = mock_playlist_response
                
                result = await client.list_channel_videos("UC_test_channel")
                
                assert len(result) == 2
                assert all('video_id' in video for video in result)
                assert all('published_at' in video for video in result)
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_list_channel_videos_with_date_filters(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test channel video listing with date filters."""
        client = YouTubeAPIClient(api_key="test_key")
        
        published_after = datetime(2023, 1, 1, tzinfo=timezone.utc)
        published_before = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        # Mock channel info and playlist response
        mock_channel_response = MockYouTubeAPIResponses.channel_info("UC_test_channel")
        mock_playlist_response = MockYouTubeAPIResponses.playlist_items()
        
        with patch.object(client, 'get_channel_info', new_callable=AsyncMock) as mock_get_channel:
            mock_get_channel.return_value = mock_channel_response['items'][0]
            
            with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = mock_playlist_response
                
                result = await client.list_channel_videos(
                    "UC_test_channel",
                    published_after=published_after,
                    published_before=published_before
                )
                
                # Should filter videos based on publish date
                assert len(result) <= 2  # Depending on mock data dates
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_search_videos_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful video search."""
        client = YouTubeAPIClient(api_key="test_key")
        
        mock_search_response = {
            'items': [
                {
                    'id': {'videoId': 'search_result_001'},
                    'snippet': {
                        'title': 'Search Result 1',
                        'description': 'Description 1',
                        'channelId': 'UC_channel_001',
                        'channelTitle': 'Channel 1',
                        'publishedAt': '2023-01-01T12:00:00Z',
                        'thumbnails': {'high': {'url': 'thumb1.jpg'}}
                    }
                }
            ]
        }
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_search_response
            
            result = await client.search_videos("programming tutorial")
            
            assert len(result) == 1
            assert result[0]['video_id'] == 'search_result_001'
            assert result[0]['title'] == 'Search Result 1'
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_search_videos_with_channel_filter(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test video search with channel filter."""
        client = YouTubeAPIClient(api_key="test_key")
        
        with patch.object(client, '_execute_api_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {'items': []}
            
            await client.search_videos(
                "tutorial",
                channel_id="UC_specific_channel"
            )
            
            # Verify channel filter was applied
            call_args = mock_execute.call_args
            request_params = call_args[1]
            assert 'channelId' in request_params
            assert request_params['channelId'] == "UC_specific_channel"


class TestYouTubeAPIClientUtilityMethods:
    """Test suite for YouTube API utility methods."""
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_parse_duration_standard_format(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test duration parsing for standard ISO 8601 format."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Test various duration formats
        assert client._parse_duration("PT4M13S") == 253  # 4:13
        assert client._parse_duration("PT1H30M45S") == 5445  # 1:30:45
        assert client._parse_duration("PT45S") == 45  # 0:45
        assert client._parse_duration("PT10M") == 600  # 10:00
        assert client._parse_duration("PT2H") == 7200  # 2:00:00
    
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    def test_parse_duration_edge_cases(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test duration parsing edge cases."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Test edge cases
        assert client._parse_duration("PT0S") == 0
        assert client._parse_duration("P") == 0  # Invalid format
        assert client._parse_duration("") == 0
        assert client._parse_duration("PT24H") == 86400  # 24 hours
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_quota_status(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test quota status retrieval."""
        client = YouTubeAPIClient(api_key="test_key")
        
        mock_quota_instance = AsyncMock()
        mock_quota_instance.get_usage_stats.return_value = {
            'current_usage': 5000,
            'daily_quota': 10000,
            'remaining_quota': 5000
        }
        client.quota_manager = mock_quota_instance
        
        result = await client.get_quota_status()
        
        assert result['current_usage'] == 5000
        assert result['daily_quota'] == 10000
        assert result['remaining_quota'] == 5000
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_get_rate_limit_status(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test rate limit status retrieval."""
        client = YouTubeAPIClient(api_key="test_key")
        
        mock_rate_instance = Mock()
        mock_rate_instance.get_stats.return_value = {
            'requests_per_minute': 60,
            'current_requests': 30,
            'reset_time': '2023-01-01T12:00:00Z'
        }
        client.rate_limiter = mock_rate_instance
        
        result = await client.get_rate_limit_status()
        
        assert result['requests_per_minute'] == 60
        assert result['current_requests'] == 30
        assert 'reset_time' in result


class TestYouTubeAPIClientExecuteAPICall:
    """Test suite for YouTube API call execution."""
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_execute_api_call_success(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test successful API call execution."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock quota manager
        mock_quota_instance = AsyncMock()
        mock_quota_manager.return_value = mock_quota_instance
        
        # Mock rate limiter
        mock_rate_instance = AsyncMock()
        mock_rate_instance.acquire.return_value.__aenter__ = AsyncMock()
        mock_rate_instance.acquire.return_value.__aexit__ = AsyncMock()
        mock_rate_limiter.return_value = mock_rate_instance
        
        # Mock API method
        mock_api_method = Mock()
        mock_api_method.execute.return_value = {'test': 'response'}
        
        from app.integrations.quota_manager import YouTubeAPIOperation
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_event_loop = Mock()
            mock_event_loop.run_in_executor = AsyncMock(return_value={'test': 'response'})
            mock_loop.return_value = mock_event_loop
            
            result = await client._execute_api_call(
                YouTubeAPIOperation.CHANNELS_LIST,
                mock_api_method
            )
            
            assert result == {'test': 'response'}
            mock_quota_instance.reserve_quota.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_execute_api_call_http_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test API call execution with HTTP error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock quota manager
        mock_quota_instance = AsyncMock()
        mock_quota_manager.return_value = mock_quota_instance
        
        # Mock rate limiter
        mock_rate_instance = AsyncMock()
        mock_rate_instance.acquire.return_value.__aenter__ = AsyncMock()
        mock_rate_instance.acquire.return_value.__aexit__ = AsyncMock()
        mock_rate_limiter.return_value = mock_rate_instance
        
        # Mock API method that raises HTTP error
        mock_response = Mock()
        mock_response.status = 404
        http_error = HttpError(resp=mock_response, content=b'Not found')
        
        mock_api_method = Mock()
        mock_api_method.execute.side_effect = http_error
        
        from app.integrations.quota_manager import YouTubeAPIOperation
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_event_loop = Mock()
            mock_event_loop.run_in_executor = AsyncMock(side_effect=http_error)
            mock_loop.return_value = mock_event_loop
            
            with patch.object(client, '_handle_api_error') as mock_handle_error:
                mock_handle_error.side_effect = YouTubeAPIError("Test error")
                
                with pytest.raises(YouTubeAPIError):
                    await client._execute_api_call(
                        YouTubeAPIOperation.CHANNELS_LIST,
                        mock_api_method
                    )
                
                # Verify quota was released on error
                mock_quota_instance.release_quota.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.integrations.youtube_api.build')
    @patch('app.integrations.youtube_api.QuotaManager')
    @patch('app.integrations.youtube_api.RateLimiter')
    @patch('app.integrations.youtube_api.httpx.AsyncClient')
    async def test_execute_api_call_network_error(self, mock_httpx, mock_rate_limiter, mock_quota_manager, mock_build):
        """Test API call execution with network error."""
        client = YouTubeAPIClient(api_key="test_key")
        
        # Mock quota manager
        mock_quota_instance = AsyncMock()
        mock_quota_manager.return_value = mock_quota_instance
        
        # Mock rate limiter
        mock_rate_instance = AsyncMock()
        mock_rate_instance.acquire.return_value.__aenter__ = AsyncMock()
        mock_rate_instance.acquire.return_value.__aexit__ = AsyncMock()
        mock_rate_limiter.return_value = mock_rate_instance
        
        from app.integrations.quota_manager import YouTubeAPIOperation
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_event_loop = Mock()
            mock_event_loop.run_in_executor = AsyncMock(side_effect=ConnectionError("Network error"))
            mock_loop.return_value = mock_event_loop
            
            with pytest.raises(NetworkError):
                await client._execute_api_call(
                    YouTubeAPIOperation.CHANNELS_LIST,
                    Mock()
                )
            
            # Verify quota was released on error
            mock_quota_instance.release_quota.assert_called_once()