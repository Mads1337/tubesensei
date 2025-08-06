import re
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs
import logging

from .exceptions import InvalidURLError

logger = logging.getLogger(__name__)


class YouTubeParser:
    """
    Parser for YouTube URLs to extract channel IDs, video IDs, and playlist IDs.
    Handles various YouTube URL formats including legacy and modern patterns.
    """
    
    # Regular expressions for different YouTube URL patterns
    CHANNEL_ID_PATTERN = re.compile(r'^UC[\w-]{22}$')
    VIDEO_ID_PATTERN = re.compile(r'^[\w-]{11}$')
    PLAYLIST_ID_PATTERN = re.compile(r'^(PL|UU|LL|RD|OL)[\w-]+$')
    
    # URL patterns for channels
    CHANNEL_PATTERNS = [
        # Direct channel ID: youtube.com/channel/UCxxxxx
        re.compile(r'youtube\.com/channel/(UC[\w-]{22})'),
        # Handle URLs: youtube.com/@username
        re.compile(r'youtube\.com/@([\w.-]+)'),
        # Custom URL: youtube.com/c/customname
        re.compile(r'youtube\.com/c/([\w-]+)'),
        # User URL: youtube.com/user/username
        re.compile(r'youtube\.com/user/([\w-]+)'),
        # Short URL: youtu.be/@username
        re.compile(r'youtu\.be/@([\w.-]+)'),
    ]
    
    # URL patterns for videos
    VIDEO_PATTERNS = [
        # Standard watch URL: youtube.com/watch?v=xxxxxxxxxxx
        re.compile(r'youtube\.com/watch\?.*v=([\w-]{11})'),
        # Short URL: youtu.be/xxxxxxxxxxx
        re.compile(r'youtu\.be/([\w-]{11})'),
        # Embed URL: youtube.com/embed/xxxxxxxxxxx
        re.compile(r'youtube\.com/embed/([\w-]{11})'),
        # Mobile URL: m.youtube.com/watch?v=xxxxxxxxxxx
        re.compile(r'm\.youtube\.com/watch\?.*v=([\w-]{11})'),
    ]
    
    # URL patterns for playlists
    PLAYLIST_PATTERNS = [
        # Playlist URL: youtube.com/playlist?list=PLxxxxx
        re.compile(r'youtube\.com/playlist\?.*list=([\w-]+)'),
        # Watch with playlist: youtube.com/watch?v=xxx&list=PLxxxxx
        re.compile(r'youtube\.com/watch\?.*list=([\w-]+)'),
    ]
    
    @classmethod
    def parse_url(cls, url: str) -> Dict[str, Optional[str]]:
        """
        Parse a YouTube URL and extract relevant IDs.
        
        Args:
            url: YouTube URL to parse
            
        Returns:
            Dictionary with extracted information:
            - type: 'channel', 'video', or 'playlist'
            - id: The extracted ID
            - channel_id: Channel ID (if available)
            - video_id: Video ID (if available)
            - playlist_id: Playlist ID (if available)
            - original_url: The original URL
            
        Raises:
            InvalidURLError: If the URL is not a valid YouTube URL
        """
        if not url:
            raise InvalidURLError(url, "Empty URL provided")
        
        # Clean and normalize URL
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        result = {
            'type': None,
            'id': None,
            'channel_id': None,
            'video_id': None,
            'playlist_id': None,
            'channel_handle': None,
            'original_url': url
        }
        
        # Try to parse as channel URL
        channel_info = cls.extract_channel_info(url)
        if channel_info:
            result['type'] = 'channel'
            if channel_info.get('channel_id'):
                result['channel_id'] = channel_info['channel_id']
                result['id'] = channel_info['channel_id']
            elif channel_info.get('handle'):
                result['channel_handle'] = channel_info['handle']
                result['id'] = channel_info['handle']
            return result
        
        # Try to parse as video URL
        video_id = cls.extract_video_id(url)
        if video_id:
            result['type'] = 'video'
            result['video_id'] = video_id
            result['id'] = video_id
            
            # Check if there's also a playlist
            playlist_id = cls.extract_playlist_id(url)
            if playlist_id:
                result['playlist_id'] = playlist_id
            
            return result
        
        # Try to parse as playlist URL
        playlist_id = cls.extract_playlist_id(url)
        if playlist_id:
            result['type'] = 'playlist'
            result['playlist_id'] = playlist_id
            result['id'] = playlist_id
            return result
        
        # If nothing matched, raise an error
        raise InvalidURLError(url, "URL does not match any known YouTube format")
    
    @classmethod
    def extract_channel_info(cls, url: str) -> Optional[Dict[str, str]]:
        """
        Extract channel information from URL.
        
        Returns:
            Dictionary with 'channel_id' or 'handle' key, or None
        """
        for pattern in cls.CHANNEL_PATTERNS:
            match = pattern.search(url)
            if match:
                identifier = match.group(1)
                # Check if it's a channel ID or a handle/username
                if cls.CHANNEL_ID_PATTERN.match(identifier):
                    return {'channel_id': identifier}
                else:
                    # It's a handle or custom name
                    return {'handle': identifier}
        
        return None
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """Extract video ID from URL"""
        for pattern in cls.VIDEO_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        
        # Try to extract from query parameters
        try:
            parsed = urlparse(url)
            if 'youtube' in parsed.netloc or 'youtu.be' in parsed.netloc:
                params = parse_qs(parsed.query)
                if 'v' in params and params['v']:
                    video_id = params['v'][0]
                    if cls.VIDEO_ID_PATTERN.match(video_id):
                        return video_id
        except Exception as e:
            logger.debug(f"Error parsing URL query parameters: {e}")
        
        return None
    
    @classmethod
    def extract_playlist_id(cls, url: str) -> Optional[str]:
        """Extract playlist ID from URL"""
        for pattern in cls.PLAYLIST_PATTERNS:
            match = pattern.search(url)
            if match:
                playlist_id = match.group(1)
                if cls.PLAYLIST_ID_PATTERN.match(playlist_id):
                    return playlist_id
        
        # Try to extract from query parameters
        try:
            parsed = urlparse(url)
            if 'youtube' in parsed.netloc:
                params = parse_qs(parsed.query)
                if 'list' in params and params['list']:
                    playlist_id = params['list'][0]
                    if cls.PLAYLIST_ID_PATTERN.match(playlist_id):
                        return playlist_id
        except Exception as e:
            logger.debug(f"Error parsing URL query parameters: {e}")
        
        return None
    
    @classmethod
    def validate_channel_id(cls, channel_id: str) -> bool:
        """Validate if a string is a valid YouTube channel ID"""
        return bool(cls.CHANNEL_ID_PATTERN.match(channel_id))
    
    @classmethod
    def validate_video_id(cls, video_id: str) -> bool:
        """Validate if a string is a valid YouTube video ID"""
        return bool(cls.VIDEO_ID_PATTERN.match(video_id))
    
    @classmethod
    def validate_playlist_id(cls, playlist_id: str) -> bool:
        """Validate if a string is a valid YouTube playlist ID"""
        return bool(cls.PLAYLIST_ID_PATTERN.match(playlist_id))
    
    @classmethod
    def build_channel_url(cls, channel_id: str) -> str:
        """Build a standard YouTube channel URL from channel ID"""
        if not cls.validate_channel_id(channel_id):
            raise ValueError(f"Invalid channel ID: {channel_id}")
        return f"https://www.youtube.com/channel/{channel_id}"
    
    @classmethod
    def build_video_url(cls, video_id: str) -> str:
        """Build a standard YouTube video URL from video ID"""
        if not cls.validate_video_id(video_id):
            raise ValueError(f"Invalid video ID: {video_id}")
        return f"https://www.youtube.com/watch?v={video_id}"
    
    @classmethod
    def build_playlist_url(cls, playlist_id: str) -> str:
        """Build a standard YouTube playlist URL from playlist ID"""
        if not cls.validate_playlist_id(playlist_id):
            raise ValueError(f"Invalid playlist ID: {playlist_id}")
        return f"https://www.youtube.com/playlist?list={playlist_id}"
    
    @classmethod
    def extract_video_timestamp(cls, url: str) -> Optional[int]:
        """
        Extract timestamp from video URL if present.
        
        Returns:
            Timestamp in seconds, or None if not present
        """
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            # Check for 't' parameter (e.g., t=120s or t=2m30s)
            if 't' in params and params['t']:
                time_str = params['t'][0]
                return cls._parse_time_string(time_str)
            
            # Check for fragment (e.g., #t=120)
            if parsed.fragment and parsed.fragment.startswith('t='):
                time_str = parsed.fragment[2:]
                return cls._parse_time_string(time_str)
                
        except Exception as e:
            logger.debug(f"Error extracting timestamp: {e}")
        
        return None
    
    @staticmethod
    def _parse_time_string(time_str: str) -> int:
        """
        Parse time string to seconds.
        Supports formats: 120, 120s, 2m30s, 1h30m45s
        """
        time_str = time_str.lower().strip()
        
        # Simple numeric format
        if time_str.isdigit():
            return int(time_str)
        
        # Remove 's' suffix if present
        if time_str.endswith('s'):
            time_str = time_str[:-1]
        
        # Parse complex format (e.g., 1h30m45s)
        total_seconds = 0
        parts = re.findall(r'(\d+)([hms]?)', time_str)
        
        for value, unit in parts:
            value = int(value)
            if unit == 'h':
                total_seconds += value * 3600
            elif unit == 'm':
                total_seconds += value * 60
            else:  # 's' or empty
                total_seconds += value
        
        return total_seconds