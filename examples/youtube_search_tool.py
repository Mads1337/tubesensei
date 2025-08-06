import os
import re
import time
import logging
from typing import Type, List, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class YouTubeSearchToolInput(BaseModel):
    """Input schema for YouTubeSearchTool."""
    search_query: str = Field(..., description="Search query to find YouTube videos")
    max_results: int = Field(default=10, description="Maximum number of results to return (1-50)")
    search_type: str = Field(default="video", description="Type of search: 'video', 'channel', or 'playlist'")


class YouTubeRelatedVideosToolInput(BaseModel):
    """Input schema for finding related videos."""
    video_url: str = Field(..., description="YouTube video URL to find related videos for")
    max_results: int = Field(default=10, description="Maximum number of related videos to return (1-50)")


class YouTubeSearchTool(BaseTool):
    """
    YouTube Search Tool using Google YouTube Data API v3.
    
    This tool allows you to search for YouTube videos by query and find related videos
    using alternative approaches since relatedToVideoId was deprecated.
    """
    
    name: str = "YouTube Search Tool"
    description: str = (
        "Search YouTube for videos using the YouTube Data API v3. Provide a search query "
        "and this tool will return a list of relevant YouTube videos with metadata including "
        "title, description, view count, duration, and channel information."
    )
    args_schema: Type[BaseModel] = YouTubeSearchToolInput
    
    def _run(
        self,
        search_query: str,
        max_results: int = 10,
        search_type: str = "video"
    ) -> str:
        """
        Search YouTube for videos.
        
        Args:
            search_query: The search query
            max_results: Maximum number of results (1-50)
            search_type: Type of search ('video', 'channel', 'playlist')
            
        Returns:
            Formatted string containing search results
        """
        if not search_query.strip():
            return "Error: Search query cannot be empty."
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not found. Please set your Google API key as an environment variable."
        
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            return f"Error: Failed to initialize YouTube API client: {str(e)}"
        
        # Validate max_results
        max_results = max(1, min(50, max_results))
        
        # Initialize cache if not exists (class-level cache)
        if not hasattr(YouTubeSearchTool, '_cache'):
            YouTubeSearchTool._cache = {}
            YouTubeSearchTool._cache_timeout = 3600  # 1 hour
        
        # Check cache first
        cache_key = f"search:{search_query}:{max_results}:{search_type}"
        if cache_key in YouTubeSearchTool._cache:
            cached_result, timestamp = YouTubeSearchTool._cache[cache_key]
            if time.time() - timestamp < YouTubeSearchTool._cache_timeout:
                logger.debug(f"Using cached search result for query: {search_query}")
                return cached_result
        
        try:
            # Perform search
            search_response = youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=max_results,
                type=search_type,
                order='relevance'
            ).execute()
            
            videos = []
            video_ids = []
            
            # Extract video information
            for search_result in search_response.get('items', []):
                if search_result['id']['kind'] == 'youtube#video':
                    video_id = search_result['id']['videoId']
                    video_ids.append(video_id)
                    
                    snippet = search_result['snippet']
                    videos.append({
                        'video_id': video_id,
                        'title': snippet.get('title', 'N/A'),
                        'description': snippet.get('description', '')[:200] + '...' if len(snippet.get('description', '')) > 200 else snippet.get('description', ''),
                        'channel_title': snippet.get('channelTitle', 'N/A'),
                        'channel_id': snippet.get('channelId', 'N/A'),
                        'published_at': snippet.get('publishedAt', 'N/A'),
                        'thumbnail_url': snippet.get('thumbnails', {}).get('medium', {}).get('url', 'N/A'),
                        'video_url': f"https://www.youtube.com/watch?v={video_id}"
                    })
            
            # Get additional video details (statistics, duration)
            if video_ids:
                try:
                    video_details = youtube.videos().list(
                        part='statistics,contentDetails',
                        id=','.join(video_ids)
                    ).execute()
                    
                    # Map additional details to videos
                    details_map = {item['id']: item for item in video_details.get('items', [])}
                    
                    for video in videos:
                        video_id = video['video_id']
                        if video_id in details_map:
                            details = details_map[video_id]
                            stats = details.get('statistics', {})
                            content_details = details.get('contentDetails', {})
                            
                            video['view_count'] = int(stats.get('viewCount', 0))
                            video['like_count'] = int(stats.get('likeCount', 0))
                            video['duration'] = content_details.get('duration', 'N/A')
                            video['duration_readable'] = self._parse_duration(content_details.get('duration', ''))
                
                except Exception as e:
                    logger.warning(f"Failed to get additional video details: {str(e)}")
            
            # Format results
            formatted_result = self._format_search_results(videos, search_query)
            
            # Cache the result
            YouTubeSearchTool._cache[cache_key] = (formatted_result, time.time())
            
            return formatted_result
            
        except HttpError as e:
            error_msg = f"YouTube API error: {e.resp.status} - {e.content.decode('utf-8') if e.content else 'Unknown error'}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error during YouTube search: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _parse_duration(self, duration: str) -> str:
        """Parse YouTube duration format (PT4M13S) to readable format."""
        if not duration:
            return "Unknown"
        
        # YouTube duration format: PT4M13S (4 minutes 13 seconds)
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return duration
        
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def _format_search_results(self, videos: List[Dict[str, Any]], query: str) -> str:
        """Format search results into a readable string."""
        if not videos:
            return f"No YouTube videos found for query: '{query}'"
        
        formatted = []
        formatted.append(f"YOUTUBE SEARCH RESULTS")
        formatted.append(f"Query: {query}")
        formatted.append(f"Found {len(videos)} videos")
        formatted.append("=" * 50)
        formatted.append("")
        
        for i, video in enumerate(videos, 1):
            formatted.append(f"{i}. {video['title']}")
            formatted.append(f"   Channel: {video['channel_title']}")
            formatted.append(f"   Duration: {video.get('duration_readable', 'Unknown')}")
            formatted.append(f"   Views: {video.get('view_count', 0):,}")
            formatted.append(f"   Published: {video['published_at']}")
            formatted.append(f"   URL: {video['video_url']}")
            if video['description']:
                formatted.append(f"   Description: {video['description']}")
            formatted.append("")
        
        return "\n".join(formatted)


class YouTubeRelatedVideosTool(BaseTool):
    """
    YouTube Related Videos Tool using alternative approaches.
    
    Since YouTube deprecated the relatedToVideoId parameter, this tool uses
    alternative methods to find related videos including keyword extraction,
    channel analysis, and category-based searches.
    """
    
    name: str = "YouTube Related Videos Tool"
    description: str = (
        "Find videos related to a given YouTube video URL using alternative approaches. "
        "This tool extracts keywords from the video metadata and performs intelligent "
        "searches to find similar content since YouTube deprecated direct related video queries."
    )
    args_schema: Type[BaseModel] = YouTubeRelatedVideosToolInput
    
    def _run(self, video_url: str, max_results: int = 10) -> str:
        """
        Find related videos for a given YouTube video URL.
        
        Args:
            video_url: The YouTube video URL
            max_results: Maximum number of related videos to return
            
        Returns:
            Formatted string containing related videos
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not found. Please set your Google API key as an environment variable."
        
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            return f"Error: Failed to initialize YouTube API client: {str(e)}"
        
        # Extract video ID from URL
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return "Error: Could not extract video ID from the provided URL"
        
        # Validate max_results
        max_results = max(1, min(50, max_results))
        
        # Initialize cache if not exists (class-level cache)
        if not hasattr(YouTubeRelatedVideosTool, '_cache'):
            YouTubeRelatedVideosTool._cache = {}
            YouTubeRelatedVideosTool._cache_timeout = 3600  # 1 hour
        
        # Check cache first
        cache_key = f"related:{video_id}:{max_results}"
        if cache_key in YouTubeRelatedVideosTool._cache:
            cached_result, timestamp = YouTubeRelatedVideosTool._cache[cache_key]
            if time.time() - timestamp < YouTubeRelatedVideosTool._cache_timeout:
                logger.debug(f"Using cached related videos result for video: {video_id}")
                return cached_result
        
        try:
            # Get video details
            video_response = youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()
            
            if not video_response['items']:
                return f"Error: Video not found for ID: {video_id}"
            
            video_data = video_response['items'][0]
            snippet = video_data['snippet']
            
            # Extract information for related video search
            title = snippet.get('title', '')
            description = snippet.get('description', '')
            channel_id = snippet.get('channelId', '')
            channel_title = snippet.get('channelTitle', '')
            category_id = snippet.get('categoryId', '')
            
            # Strategy 1: Search by keywords extracted from title
            title_keywords = self._extract_keywords(title)
            
            # Strategy 2: Search by channel (other videos from same creator)
            # Strategy 3: Search by category if available
            
            related_videos = []
            
            # Execute multiple search strategies
            strategies = [
                ("Keywords from Title", " ".join(title_keywords[:3])),  # Top 3 keywords
                ("Channel Content", f"channel:{channel_title}"),
                ("Similar Titles", f"{title_keywords[0] if title_keywords else title[:20]}")
            ]
            
            results_per_strategy = max(1, max_results // len(strategies))
            
            for strategy_name, search_query in strategies:
                if len(related_videos) >= max_results:
                    break
                
                try:
                    strategy_results = self._search_related_videos(
                        youtube,
                        search_query, 
                        results_per_strategy,
                        exclude_video_id=video_id
                    )
                    
                    # Add strategy info to results
                    for result in strategy_results:
                        result['strategy'] = strategy_name
                    
                    related_videos.extend(strategy_results)
                    
                except Exception as e:
                    logger.warning(f"Strategy '{strategy_name}' failed: {str(e)}")
                    continue
            
            # Remove duplicates and limit results
            seen_video_ids = set()
            unique_videos = []
            
            for video in related_videos:
                if video['video_id'] not in seen_video_ids and video['video_id'] != video_id:
                    seen_video_ids.add(video['video_id'])
                    unique_videos.append(video)
                    
                if len(unique_videos) >= max_results:
                    break
            
            # Format results
            formatted_result = self._format_related_videos_results(
                unique_videos, 
                video_url, 
                title,
                channel_title
            )
            
            # Cache the result
            YouTubeRelatedVideosTool._cache[cache_key] = (formatted_result, time.time())
            
            return formatted_result
            
        except HttpError as e:
            error_msg = f"YouTube API error: {e.resp.status} - {e.content.decode('utf-8') if e.content else 'Unknown error'}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error finding related videos: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from title or description."""
        # Simple keyword extraction - remove common words and split
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'how', 'what', 'when', 'where',
            'why', 'who', 'which'
        }
        
        # Split and clean words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _search_related_videos(
        self, 
        youtube,
        query: str, 
        max_results: int, 
        exclude_video_id: str = None
    ) -> List[Dict[str, Any]]:
        """Search for videos using a specific query."""
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=min(max_results * 2, 50),  # Get more to filter out excluded video
            type='video',
            order='relevance'
        ).execute()
        
        videos = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                video_id = search_result['id']['videoId']
                
                # Skip if this is the original video
                if video_id == exclude_video_id:
                    continue
                
                snippet = search_result['snippet']
                videos.append({
                    'video_id': video_id,
                    'title': snippet.get('title', 'N/A'),
                    'description': snippet.get('description', '')[:200] + '...' if len(snippet.get('description', '')) > 200 else snippet.get('description', ''),
                    'channel_title': snippet.get('channelTitle', 'N/A'),
                    'channel_id': snippet.get('channelId', 'N/A'),
                    'published_at': snippet.get('publishedAt', 'N/A'),
                    'video_url': f"https://www.youtube.com/watch?v={video_id}"
                })
                
                if len(videos) >= max_results:
                    break
        
        return videos
    
    def _format_related_videos_results(
        self, 
        videos: List[Dict[str, Any]], 
        original_url: str,
        original_title: str,
        original_channel: str
    ) -> str:
        """Format related videos results into a readable string."""
        if not videos:
            return f"No related videos found for: '{original_title}'"
        
        formatted = []
        formatted.append("YOUTUBE RELATED VIDEOS")
        formatted.append(f"Original Video: {original_title}")
        formatted.append(f"Original Channel: {original_channel}")
        formatted.append(f"Original URL: {original_url}")
        formatted.append(f"Found {len(videos)} related videos using alternative methods")
        formatted.append("=" * 50)
        formatted.append("")
        
        for i, video in enumerate(videos, 1):
            formatted.append(f"{i}. {video['title']}")
            formatted.append(f"   Channel: {video['channel_title']}")
            formatted.append(f"   Published: {video['published_at']}")
            formatted.append(f"   URL: {video['video_url']}")
            formatted.append(f"   Strategy: {video.get('strategy', 'Unknown')}")
            if video['description']:
                formatted.append(f"   Description: {video['description']}")
            formatted.append("")
        
        formatted.append("Note: Related videos found using alternative methods since YouTube")
        formatted.append("deprecated the relatedToVideoId parameter in August 2023.")
        
        return "\n".join(formatted)