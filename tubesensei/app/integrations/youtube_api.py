import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime
import httpx
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import DefaultCredentialsError
import json

from ..config import settings
from ..utils.exceptions import (
    YouTubeAPIError,
    QuotaExceededError,
    ChannelNotFoundError,
    VideoNotFoundError,
    APIKeyError,
    NetworkError,
    RateLimitError
)
from ..utils.rate_limiter import RateLimiter
from .quota_manager import QuotaManager, YouTubeAPIOperation

logger = logging.getLogger(__name__)


class YouTubeAPIClient:
    """
    Async-compatible YouTube API v3 client with quota management and rate limiting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        quota_per_day: Optional[int] = None,
        rate_limit_per_minute: Optional[int] = None
    ):
        # API configuration
        self.api_key = api_key or settings.YOUTUBE_API_KEY
        if not self.api_key:
            raise APIKeyError("YouTube API key is required")
        
        # Initialize YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
        # Initialize quota manager
        self.quota_manager = QuotaManager(
            daily_quota=quota_per_day or settings.YOUTUBE_QUOTA_PER_DAY
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_per_minute or settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        )
        
        # HTTP client for async operations
        self.http_client = httpx.AsyncClient(
            timeout=settings.YOUTUBE_REQUEST_TIMEOUT,
            headers={'User-Agent': 'TubeSensei/1.0'}
        )
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = settings.YOUTUBE_CACHE_TTL
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close HTTP client and save quota data"""
        await self.http_client.aclose()
        await self.quota_manager._save_quota_data()
    
    def _handle_api_error(self, error: HttpError) -> None:
        """
        Handle YouTube API errors and raise appropriate exceptions.
        
        Args:
            error: The HttpError from Google API client
            
        Raises:
            Appropriate YouTubeAPIError subclass
        """
        try:
            error_details = json.loads(error.content.decode('utf-8'))
            error_info = error_details.get('error', {})
            
            # Check for quota exceeded
            if error.resp.status == 403:
                for err in error_info.get('errors', []):
                    if err.get('reason') == 'quotaExceeded':
                        raise QuotaExceededError(
                            quota_used=self.quota_manager.current_usage,
                            quota_limit=self.quota_manager.daily_quota
                        )
                    elif err.get('reason') == 'rateLimitExceeded':
                        raise RateLimitError()
            
            # Check for not found errors
            elif error.resp.status == 404:
                message = error_info.get('message', 'Resource not found')
                if 'channel' in message.lower():
                    raise ChannelNotFoundError('unknown')
                elif 'video' in message.lower():
                    raise VideoNotFoundError('unknown')
            
            # Check for invalid API key
            elif error.resp.status == 400:
                for err in error_info.get('errors', []):
                    if err.get('reason') == 'keyInvalid':
                        raise APIKeyError("Invalid YouTube API key")
            
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Default error
        raise YouTubeAPIError(f"YouTube API error: {error}")
    
    async def _execute_api_call(
        self,
        operation: YouTubeAPIOperation,
        api_method,
        **kwargs
    ) -> Any:
        """
        Execute an API call with quota and rate limiting.
        
        Args:
            operation: The API operation being performed
            api_method: The API method to call
            **kwargs: Arguments for the API method
            
        Returns:
            API response
        """
        # Reserve quota
        await self.quota_manager.reserve_quota(operation)
        
        try:
            # Apply rate limiting
            async with self.rate_limiter.acquire():
                # Execute in thread pool since googleapiclient is sync
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: api_method(**kwargs).execute()
                )
                return response
                
        except HttpError as e:
            # Release quota on error
            await self.quota_manager.release_quota(operation)
            self._handle_api_error(e)
        except Exception as e:
            # Release quota on error
            await self.quota_manager.release_quota(operation)
            raise NetworkError(f"API call failed: {e}", original_error=e)
    
    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Channel information dictionary
        """
        # Check cache
        cache_key = f"channel:{channel_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = await self._execute_api_call(
                YouTubeAPIOperation.CHANNELS_LIST,
                self.youtube.channels().list,
                part='snippet,statistics,contentDetails,brandingSettings',
                id=channel_id
            )
            
            if not response.get('items'):
                raise ChannelNotFoundError(channel_id)
            
            channel_data = response['items'][0]
            
            # Process and structure the data
            result = {
                'channel_id': channel_data['id'],
                'title': channel_data['snippet']['title'],
                'description': channel_data['snippet'].get('description', ''),
                'custom_url': channel_data['snippet'].get('customUrl'),
                'published_at': channel_data['snippet']['publishedAt'],
                'country': channel_data['snippet'].get('country'),
                'thumbnails': channel_data['snippet']['thumbnails'],
                'view_count': int(channel_data['statistics'].get('viewCount', 0)),
                'subscriber_count': int(channel_data['statistics'].get('subscriberCount', 0)),
                'video_count': int(channel_data['statistics'].get('videoCount', 0)),
                'uploads_playlist_id': channel_data['contentDetails']['relatedPlaylists']['uploads'],
                'keywords': channel_data.get('brandingSettings', {}).get('channel', {}).get('keywords', ''),
                'raw_data': channel_data
            }
            
            # Cache the result
            self._cache[cache_key] = result
            
            return result
            
        except (ChannelNotFoundError, YouTubeAPIError):
            raise
        except Exception as e:
            raise YouTubeAPIError(f"Failed to get channel info: {e}")
    
    async def get_channel_by_handle(self, handle: str) -> Dict[str, Any]:
        """
        Get channel information by handle/username.
        
        Args:
            handle: YouTube channel handle (e.g., @username)
            
        Returns:
            Channel information dictionary
        """
        # Remove @ if present
        handle = handle.lstrip('@')
        
        try:
            # First try with forUsername (for legacy usernames)
            response = await self._execute_api_call(
                YouTubeAPIOperation.CHANNELS_LIST,
                self.youtube.channels().list,
                part='snippet,statistics,contentDetails',
                forUsername=handle
            )
            
            if not response.get('items'):
                # For modern @handles, we need to search first
                search_response = await self._execute_api_call(
                    YouTubeAPIOperation.SEARCH_LIST,
                    self.youtube.search().list,
                    part='snippet',
                    q=f"@{handle}",
                    type='channel',
                    maxResults=5  # Get more results to find exact match
                )
                
                print(f"DEBUG: Search response for @{handle}: {search_response}")
                
                if search_response.get('items'):
                    # Try to find exact handle match first
                    for item in search_response['items']:
                        channel_title = item['snippet']['title'].lower()
                        channel_desc = item['snippet'].get('description', '').lower()
                        
                        # Check if the handle appears in the title or description
                        if handle.lower() in channel_title or f"@{handle}".lower() in channel_desc:
                            channel_id = item['snippet']['channelId']
                            print(f"DEBUG: Found exact match for @{handle}: {item['snippet']['title']}")
                            return await self.get_channel_info(channel_id)
                    
                    # If no exact match, check the first result's custom URL
                    first_result = search_response['items'][0]
                    channel_id = first_result['snippet']['channelId']
                    
                    # Get full channel info to check custom URL
                    channel_info = await self.get_channel_info(channel_id)
                    custom_url = channel_info.get('custom_url', '').lower()
                    
                    print(f"DEBUG: Checking custom URL: {custom_url} vs @{handle}")
                    
                    # Only return if custom URL matches the handle we're looking for
                    if custom_url == f"@{handle.lower()}" or custom_url == handle.lower():
                        return channel_info
                    else:
                        # Custom URL doesn't match, this is the wrong channel
                        print(f"DEBUG: Custom URL mismatch. Expected @{handle}, got {custom_url}")
                        pass  # Continue to raise ChannelNotFoundError
            
            if not response.get('items'):
                raise ChannelNotFoundError(f"@{handle}")
            
            channel_data = response['items'][0]
            return await self.get_channel_info(channel_data['id'])
            
        except (ChannelNotFoundError, YouTubeAPIError):
            raise
        except Exception as e:
            raise YouTubeAPIError(f"Failed to get channel by handle: {e}")
    
    async def list_channel_videos(
        self,
        channel_id: str,
        max_results: int = 500,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        List videos from a channel's uploads.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch
            published_after: Only fetch videos published after this date
            published_before: Only fetch videos published before this date
            
        Returns:
            List of video information dictionaries
        """
        # Get channel info to get uploads playlist
        channel_info = await self.get_channel_info(channel_id)
        uploads_playlist_id = channel_info['uploads_playlist_id']
        
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            # Calculate batch size
            remaining = max_results - len(videos)
            batch_size = min(50, remaining)  # YouTube API max is 50 per page
            
            try:
                # Fetch playlist items
                request_params = {
                    'part': 'snippet,contentDetails',
                    'playlistId': uploads_playlist_id,
                    'maxResults': batch_size
                }
                
                if next_page_token:
                    request_params['pageToken'] = next_page_token
                
                response = await self._execute_api_call(
                    YouTubeAPIOperation.PLAYLIST_ITEMS_LIST,
                    self.youtube.playlistItems().list,
                    **request_params
                )
                
                # Process videos
                for item in response.get('items', []):
                    video_published = datetime.fromisoformat(
                        item['contentDetails']['videoPublishedAt'].replace('Z', '+00:00')
                    )
                    
                    # Apply date filters
                    if published_after and video_published < published_after:
                        continue
                    if published_before and video_published > published_before:
                        continue
                    
                    videos.append({
                        'video_id': item['contentDetails']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'published_at': item['contentDetails']['videoPublishedAt'],
                        'thumbnails': item['snippet']['thumbnails'],
                        'channel_id': item['snippet']['channelId'],
                        'playlist_item_id': item['id']
                    })
                
                # Check for more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching channel videos: {e}")
                break
        
        return videos[:max_results]
    
    async def get_video_details(
        self,
        video_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get detailed information for multiple videos.
        
        Args:
            video_ids: List of YouTube video IDs (max 50 per call)
            
        Returns:
            List of video information dictionaries
        """
        if not video_ids:
            return []
        
        all_videos = []
        
        # Process in batches of 50 (YouTube API limit)
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            try:
                response = await self._execute_api_call(
                    YouTubeAPIOperation.VIDEOS_LIST,
                    self.youtube.videos().list,
                    part='snippet,contentDetails,statistics,status',
                    id=','.join(batch_ids)
                )
                
                for item in response.get('items', []):
                    # Parse duration
                    duration = item['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration)
                    
                    video_data = {
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'channel_id': item['snippet']['channelId'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'duration_seconds': duration_seconds,
                        'duration_iso': duration,
                        'view_count': int(item['statistics'].get('viewCount', 0)),
                        'like_count': int(item['statistics'].get('likeCount', 0)),
                        'comment_count': int(item['statistics'].get('commentCount', 0)),
                        'tags': item['snippet'].get('tags', []),
                        'category_id': item['snippet'].get('categoryId'),
                        'language': item['snippet'].get('defaultLanguage'),
                        'has_captions': item['contentDetails'].get('caption') == 'true',
                        'is_live': item['snippet'].get('liveBroadcastContent') != 'none',
                        'privacy_status': item['status']['privacyStatus'],
                        'thumbnails': item['snippet']['thumbnails'],
                        'raw_data': item
                    }
                    
                    all_videos.append(video_data)
                    
            except Exception as e:
                logger.error(f"Error fetching video details for batch: {e}")
                continue
        
        return all_videos
    
    async def search_videos(
        self,
        query: str,
        channel_id: Optional[str] = None,
        max_results: int = 50,
        order: str = 'relevance',
        published_after: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for videos on YouTube.
        
        Args:
            query: Search query
            channel_id: Limit search to specific channel
            max_results: Maximum results to return
            order: Sort order (relevance, date, rating, viewCount, title)
            published_after: Only return videos published after this date
            
        Returns:
            List of video search results
        """
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            batch_size = min(50, max_results - len(videos))
            
            request_params = {
                'part': 'snippet',
                'q': query,
                'type': 'video',
                'maxResults': batch_size,
                'order': order
            }
            
            if channel_id:
                request_params['channelId'] = channel_id
            if published_after:
                request_params['publishedAfter'] = published_after.isoformat() + 'Z'
            if next_page_token:
                request_params['pageToken'] = next_page_token
            
            try:
                response = await self._execute_api_call(
                    YouTubeAPIOperation.SEARCH_LIST,
                    self.youtube.search().list,
                    **request_params
                )
                
                for item in response.get('items', []):
                    videos.append({
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'channel_id': item['snippet']['channelId'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'thumbnails': item['snippet']['thumbnails']
                    })
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching videos: {e}")
                break
        
        return videos[:max_results]

    async def get_related_videos(
        self,
        video_id: str,
        max_results: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Get related/similar videos for a given video.

        Uses the YouTube Search API with relatedToVideoId parameter.
        Note: This parameter is deprecated but may still work.
        Falls back to title-based search if relatedToVideoId fails.

        Args:
            video_id: YouTube video ID to find related videos for
            max_results: Maximum results to return (default 25)

        Returns:
            List of related video information dictionaries
        """
        videos = []

        try:
            # First try with relatedToVideoId (deprecated but may work)
            request_params = {
                'part': 'snippet',
                'relatedToVideoId': video_id,
                'type': 'video',
                'maxResults': min(50, max_results),
            }

            try:
                response = await self._execute_api_call(
                    YouTubeAPIOperation.SEARCH_LIST,
                    self.youtube.search().list,
                    **request_params
                )

                for item in response.get('items', []):
                    # Skip the source video itself
                    related_video_id = item['id'].get('videoId')
                    if related_video_id and related_video_id != video_id:
                        videos.append({
                            'video_id': related_video_id,
                            'title': item['snippet']['title'],
                            'description': item['snippet']['description'],
                            'channel_id': item['snippet']['channelId'],
                            'channel_title': item['snippet']['channelTitle'],
                            'published_at': item['snippet']['publishedAt'],
                            'thumbnails': item['snippet']['thumbnails'],
                            'discovery_method': 'related_to_video_id',
                        })

                if videos:
                    logger.debug(f"Found {len(videos)} related videos using relatedToVideoId")
                    return videos[:max_results]

            except Exception as e:
                logger.debug(f"relatedToVideoId failed (deprecated): {e}")
                # Fall through to title-based search

            # Fallback: Search using the source video's title
            # First get the source video's details
            video_details = await self.get_video_details([video_id])
            if not video_details:
                logger.warning(f"Could not get details for video {video_id}")
                return []

            source_video = video_details[0]
            source_title = source_video.get('title', '')

            if not source_title:
                return []

            # Extract key terms from title (remove common words)
            import re
            # Remove special characters and split
            words = re.findall(r'\b[a-zA-Z]{3,}\b', source_title.lower())
            # Remove very common words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'how', 'what', 'with', 'this', 'that', 'from', 'they', 'will', 'have', 'has', 'been', 'were'}
            key_terms = [w for w in words if w not in stop_words][:5]

            if not key_terms:
                return []

            search_query = ' '.join(key_terms)
            logger.debug(f"Searching for related videos with query: {search_query}")

            # Search for videos with similar title
            response = await self._execute_api_call(
                YouTubeAPIOperation.SEARCH_LIST,
                self.youtube.search().list,
                part='snippet',
                q=search_query,
                type='video',
                maxResults=min(50, max_results + 5),  # Get a few extra to filter
                order='relevance'
            )

            for item in response.get('items', []):
                related_video_id = item['id'].get('videoId')
                # Skip the source video itself
                if related_video_id and related_video_id != video_id:
                    videos.append({
                        'video_id': related_video_id,
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'channel_id': item['snippet']['channelId'],
                        'channel_title': item['snippet']['channelTitle'],
                        'published_at': item['snippet']['publishedAt'],
                        'thumbnails': item['snippet']['thumbnails'],
                        'discovery_method': 'title_search',
                    })

            logger.debug(f"Found {len(videos)} related videos using title search")
            return videos[:max_results]

        except Exception as e:
            logger.error(f"Error getting related videos for {video_id}: {e}")
            return []

    def _parse_duration(self, duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds.
        
        Args:
            duration: ISO 8601 duration string (e.g., PT4M13S)
            
        Returns:
            Duration in seconds
        """
        import re
        
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    async def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota usage status"""
        return await self.quota_manager.get_usage_stats()
    
    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return self.rate_limiter.get_stats()