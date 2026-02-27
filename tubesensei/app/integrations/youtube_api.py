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

        # Redis client for persistent caching across campaigns
        self._redis_client = None
        self._redis_initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close HTTP client, Redis client, and save quota data"""
        await self.http_client.aclose()
        if self._redis_client:
            await self._redis_client.close()
        await self.quota_manager._save_quota_data()

    async def _get_redis(self):
        """Get or create async Redis client for video caching."""
        if not self._redis_initialized:
            self._redis_initialized = True
            try:
                import redis.asyncio as aioredis
                self._redis_client = aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True
                )
                await self._redis_client.ping()
                logger.debug("YouTube API Redis cache connected")
            except Exception as e:
                logger.debug(f"Redis unavailable for YouTube cache, continuing without: {e}")
                self._redis_client = None
        return self._redis_client
    
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

        Lookup order (cheapest first):
        1. forHandle (1 unit) - modern @handles
        2. forUsername (1 unit) - legacy usernames
        3. search.list (100 units) - last resort fallback

        Args:
            handle: YouTube channel handle (e.g., @username)

        Returns:
            Channel information dictionary
        """
        # Remove @ if present
        handle = handle.lstrip('@')

        try:
            # 1. Try forHandle first (1 unit - works for modern @handles)
            try:
                response = await self._execute_api_call(
                    YouTubeAPIOperation.CHANNELS_LIST,
                    self.youtube.channels().list,
                    part='snippet,statistics,contentDetails',
                    forHandle=handle
                )
                if response.get('items'):
                    channel_data = response['items'][0]
                    return await self.get_channel_info(channel_data['id'])
            except Exception as e:
                logger.debug(f"forHandle lookup failed for @{handle}: {e}")

            # 2. Try forUsername (1 unit - for legacy usernames)
            response = await self._execute_api_call(
                YouTubeAPIOperation.CHANNELS_LIST,
                self.youtube.channels().list,
                part='snippet,statistics,contentDetails',
                forUsername=handle
            )

            if response.get('items'):
                channel_data = response['items'][0]
                return await self.get_channel_info(channel_data['id'])

            # 3. Last resort: search (100 units)
            logger.debug(f"Falling back to search.list for @{handle}")
            search_response = await self._execute_api_call(
                YouTubeAPIOperation.SEARCH_LIST,
                self.youtube.search().list,
                part='snippet',
                q=f"@{handle}",
                type='channel',
                maxResults=5
            )

            if search_response.get('items'):
                # Try to find exact handle match
                for item in search_response['items']:
                    channel_title = item['snippet']['title'].lower()
                    channel_desc = item['snippet'].get('description', '').lower()

                    if handle.lower() in channel_title or f"@{handle}".lower() in channel_desc:
                        channel_id = item['snippet']['channelId']
                        return await self.get_channel_info(channel_id)

                # Check first result's custom URL
                first_result = search_response['items'][0]
                channel_id = first_result['snippet']['channelId']
                channel_info = await self.get_channel_info(channel_id)
                custom_url = channel_info.get('custom_url', '').lower()

                if custom_url == f"@{handle.lower()}" or custom_url == handle.lower():
                    return channel_info

            raise ChannelNotFoundError(f"@{handle}")

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

        Checks Redis cache first, then fetches uncached videos from the API.
        Results are cached with 24-hour TTL to avoid repeated lookups.

        Args:
            video_ids: List of YouTube video IDs (max 50 per call)

        Returns:
            List of video information dictionaries
        """
        if not video_ids:
            return []

        all_videos = []
        ids_to_fetch = list(video_ids)

        # Check Redis cache for already-fetched video details
        redis = await self._get_redis()
        if redis:
            try:
                cache_keys = [f"yt:video:{vid}" for vid in video_ids]
                cached_values = await redis.mget(cache_keys)

                cached_ids = set()
                for vid, cached_json in zip(video_ids, cached_values):
                    if cached_json:
                        try:
                            video_data = json.loads(cached_json)
                            all_videos.append(video_data)
                            cached_ids.add(vid)
                        except json.JSONDecodeError:
                            pass

                ids_to_fetch = [vid for vid in video_ids if vid not in cached_ids]

                if cached_ids:
                    logger.debug(f"Redis cache hit for {len(cached_ids)} videos, fetching {len(ids_to_fetch)} from API")
            except Exception as e:
                logger.debug(f"Redis cache read failed, fetching all from API: {e}")
                ids_to_fetch = list(video_ids)

        if not ids_to_fetch:
            return all_videos

        # Process in batches of 50 (YouTube API limit)
        for i in range(0, len(ids_to_fetch), 50):
            batch_ids = ids_to_fetch[i:i+50]

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

                    # Cache in Redis with 24-hour TTL
                    if redis:
                        try:
                            # Cache without raw_data to save space
                            cache_data = {k: v for k, v in video_data.items() if k != 'raw_data'}
                            await redis.setex(
                                f"yt:video:{item['id']}",
                                86400,  # 24 hours
                                json.dumps(cache_data)
                            )
                        except Exception:
                            pass  # Non-critical, continue without caching

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
        published_after: Optional[datetime] = None,
        video_duration: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for videos on YouTube.

        Args:
            query: Search query
            channel_id: Limit search to specific channel
            max_results: Maximum results to return
            order: Sort order (relevance, date, rating, viewCount, title)
            published_after: Only return videos published after this date
            video_duration: Duration filter - 'short' (<4min), 'medium' (4-20min), 'long' (>20min)

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
            if video_duration:
                request_params['videoDuration'] = video_duration
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
                    
            except QuotaExceededError:
                # Re-raise quota errors - these are critical and should not be silently ignored
                raise
            except Exception as e:
                logger.error(f"Error searching videos: {e}")
                break

        return videos[:max_results]

    async def get_related_videos(
        self,
        video_id: str,
        max_results: int = 25,
        video_duration: Optional[str] = None,
        source_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get related/similar videos for a given video using title-based search.

        Args:
            video_id: YouTube video ID to find related videos for
            max_results: Maximum results to return (default 25)
            video_duration: Duration filter - 'short' (<4min), 'medium' (4-20min), 'long' (>20min)
            source_title: Title of the source video (avoids extra API call if provided)

        Returns:
            List of related video information dictionaries
        """
        import re
        videos = []

        try:
            # Get source video title if not provided by caller
            if not source_title:
                video_details = await self.get_video_details([video_id])
                if not video_details:
                    logger.warning(f"Could not get details for video {video_id}")
                    return []
                source_title = video_details[0].get('title', '')

            if not source_title:
                return []

            # Extract key terms from title (remove common words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', source_title.lower())
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'how', 'what', 'with', 'this', 'that', 'from', 'they', 'will', 'have', 'has', 'been', 'were'}
            key_terms = [w for w in words if w not in stop_words][:5]

            if not key_terms:
                return []

            search_query = ' '.join(key_terms)
            logger.debug(f"Searching for related videos with query: {search_query}")

            search_params = {
                'part': 'snippet',
                'q': search_query,
                'type': 'video',
                'maxResults': min(50, max_results + 5),
                'order': 'relevance'
            }
            if video_duration:
                search_params['videoDuration'] = video_duration

            response = await self._execute_api_call(
                YouTubeAPIOperation.SEARCH_LIST,
                self.youtube.search().list,
                **search_params
            )

            for item in response.get('items', []):
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