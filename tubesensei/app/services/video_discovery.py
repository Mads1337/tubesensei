import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from ..models.channel import Channel, ChannelStatus
from ..models.video import Video
from ..models.processing_job import ProcessingJob, JobType, JobStatus
from ..models.filters import VideoFilters, ProcessingFilters
from ..integrations.youtube_api import YouTubeAPIClient
from ..utils.exceptions import (
    ValidationError,
    YouTubeAPIError,
    QuotaExceededError
)

logger = logging.getLogger(__name__)


class VideoDiscovery:
    """
    Service for discovering and managing YouTube videos.
    Handles video discovery, filtering, and metadata updates.
    """
    
    def __init__(self, youtube_client: Optional[YouTubeAPIClient] = None):
        self.youtube_client = youtube_client
    
    async def _get_youtube_client(self) -> YouTubeAPIClient:
        """Get or create YouTube API client"""
        if not self.youtube_client:
            self.youtube_client = YouTubeAPIClient()
        return self.youtube_client
    
    async def discover_videos(
        self,
        channel_id: UUID,
        db: AsyncSession,
        filters: Optional[VideoFilters] = None,
        max_videos: int = 500,
        force_refresh: bool = False
    ) -> List[Video]:
        """
        Discover videos from a channel with optional filtering.
        
        Args:
            channel_id: Database channel ID
            db: Database session
            filters: Video filtering criteria
            max_videos: Maximum number of videos to discover
            force_refresh: Force re-fetch even if videos exist
            
        Returns:
            List of discovered and filtered Video objects
        """
        # Get channel from database
        result = await db.execute(
            select(Channel).filter(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Apply default filters if none provided
        if not filters:
            filters = VideoFilters()
        
        # Determine if we need to fetch from YouTube
        should_fetch = force_refresh or not channel.last_video_fetch_at
        
        if should_fetch:
            # Fetch from YouTube API
            youtube = await self._get_youtube_client()
            
            try:
                # Get video list from channel
                video_list = await youtube.list_channel_videos(
                    channel.youtube_channel_id,
                    max_results=max_videos * 2,  # Fetch extra to account for filtering
                    published_after=filters.published_after,
                    published_before=filters.published_before
                )
                
                if not video_list:
                    logger.info(f"No videos found for channel: {channel.title}")
                    return []
                
                # Get detailed information
                video_ids = [v['video_id'] for v in video_list]
                video_details = await youtube.get_video_details(video_ids)
                
                # Apply filters and store videos
                discovered_videos = await self._process_and_store_videos(
                    video_details,
                    channel,
                    db,
                    filters
                )
                
                # Update channel fetch timestamp
                channel.last_video_fetch_at = datetime.now(timezone.utc)
                await db.commit()
                
                return discovered_videos[:max_videos]
                
            except QuotaExceededError:
                logger.error(f"Quota exceeded while discovering videos for {channel.title}")
                raise
            except Exception as e:
                logger.error(f"Error discovering videos for {channel.title}: {e}")
                channel.last_error = str(e)
                await db.commit()
                raise
        
        else:
            # Load from database and apply filters
            query = select(Video).filter(Video.channel_id == channel_id)
            
            # Apply database-level filters
            if filters.min_duration_seconds:
                query = query.filter(Video.duration_seconds >= filters.min_duration_seconds)
            if filters.max_duration_seconds:
                query = query.filter(Video.duration_seconds <= filters.max_duration_seconds)
            if filters.min_views:
                query = query.filter(Video.view_count >= filters.min_views)
            if filters.max_views:
                query = query.filter(Video.view_count <= filters.max_views)
            if filters.published_after:
                query = query.filter(Video.published_at >= filters.published_after)
            if filters.published_before:
                query = query.filter(Video.published_at <= filters.published_before)
            
            # Order by published date descending
            query = query.order_by(Video.published_at.desc())
            query = query.limit(max_videos)
            
            result = await db.execute(query)
            videos = result.scalars().all()
            
            # Apply additional filters that can't be done at database level
            filtered_videos = []
            for video in videos:
                if self._apply_advanced_filters(video, filters):
                    filtered_videos.append(video)
            
            return filtered_videos
    
    async def batch_discover(
        self,
        channel_ids: List[UUID],
        db: AsyncSession,
        filters: Optional[VideoFilters] = None,
        max_videos_per_channel: int = 100
    ) -> Dict[str, Any]:
        """
        Discover videos from multiple channels.
        
        Args:
            channel_ids: List of database channel IDs
            db: Database session
            filters: Video filtering criteria
            max_videos_per_channel: Maximum videos per channel
            
        Returns:
            Summary of discovery results
        """
        results = {
            'total_channels': len(channel_ids),
            'successful_channels': 0,
            'failed_channels': 0,
            'total_videos_discovered': 0,
            'errors': [],
            'channel_results': {}
        }
        
        for channel_id in channel_ids:
            try:
                videos = await self.discover_videos(
                    channel_id,
                    db,
                    filters,
                    max_videos_per_channel
                )
                
                results['successful_channels'] += 1
                results['total_videos_discovered'] += len(videos)
                results['channel_results'][str(channel_id)] = {
                    'status': 'success',
                    'videos_discovered': len(videos)
                }
                
            except Exception as e:
                results['failed_channels'] += 1
                error_msg = f"Channel {channel_id}: {str(e)}"
                results['errors'].append(error_msg)
                results['channel_results'][str(channel_id)] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(error_msg)
        
        return results
    
    async def update_video_metadata(
        self,
        video_ids: List[UUID],
        db: AsyncSession
    ) -> List[Video]:
        """
        Update metadata for existing videos.
        
        Args:
            video_ids: List of database video IDs
            db: Database session
            
        Returns:
            List of updated Video objects
        """
        # Get videos from database
        result = await db.execute(
            select(Video).filter(Video.id.in_(video_ids))
        )
        videos = result.scalars().all()
        
        if not videos:
            return []
        
        # Group by YouTube video ID for batch fetching
        youtube_ids = [v.youtube_video_id for v in videos]
        video_map = {v.youtube_video_id: v for v in videos}
        
        # Fetch updated metadata from YouTube
        youtube = await self._get_youtube_client()
        
        try:
            video_details = await youtube.get_video_details(youtube_ids)
        except Exception as e:
            logger.error(f"Error fetching video metadata: {e}")
            raise
        
        # Update video objects
        updated_videos = []
        for details in video_details:
            video = video_map.get(details['video_id'])
            if video:
                # Update metadata
                video.title = details['title']
                video.description = details.get('description', '')
                video.view_count = details.get('view_count', 0)
                video.like_count = details.get('like_count', 0)
                video.comment_count = details.get('comment_count', 0)
                video.tags = details.get('tags', [])
                video.has_captions = details.get('has_captions', False)
                video.metadata = details.get('raw_data', {})
                video.last_updated_at = datetime.now(timezone.utc)
                
                updated_videos.append(video)
        
        await db.commit()
        
        logger.info(f"Updated metadata for {len(updated_videos)} videos")
        return updated_videos
    
    async def search_and_discover(
        self,
        query: str,
        db: AsyncSession,
        channel_id: Optional[UUID] = None,
        filters: Optional[VideoFilters] = None,
        max_results: int = 50
    ) -> List[Video]:
        """
        Search for videos and add them to the database.
        
        Args:
            query: Search query
            db: Database session
            channel_id: Limit search to specific channel
            filters: Video filtering criteria
            max_results: Maximum search results
            
        Returns:
            List of discovered Video objects
        """
        youtube = await self._get_youtube_client()
        
        # Get channel YouTube ID if provided
        youtube_channel_id = None
        if channel_id:
            result = await db.execute(
                select(Channel).filter(Channel.id == channel_id)
            )
            channel = result.scalar_one_or_none()
            if channel:
                youtube_channel_id = channel.youtube_channel_id
        
        # Search videos
        try:
            search_results = await youtube.search_videos(
                query=query,
                channel_id=youtube_channel_id,
                max_results=max_results * 2,  # Get extra for filtering
                published_after=filters.published_after if filters else None
            )
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            raise
        
        if not search_results:
            return []
        
        # Get detailed information
        video_ids = [v['video_id'] for v in search_results]
        video_details = await youtube.get_video_details(video_ids)
        
        # Process and store videos
        discovered_videos = []
        for details in video_details:
            # Apply filters
            if filters and not filters.apply_to_video(details):
                continue
            
            # Check if video already exists
            existing = await db.execute(
                select(Video).filter(
                    Video.youtube_video_id == details['video_id']
                )
            )
            if existing.scalar_one_or_none():
                continue
            
            # Find or create channel for the video
            video_channel = await self._ensure_channel_exists(
                details['channel_id'],
                db
            )
            
            # Create video record
            video = Video(
                youtube_video_id=details['video_id'],
                channel_id=video_channel.id,
                title=details['title'],
                description=details.get('description', ''),
                published_at=datetime.fromisoformat(
                    details['published_at'].replace('Z', '+00:00')
                ),
                duration_seconds=details['duration_seconds'],
                view_count=details.get('view_count', 0),
                like_count=details.get('like_count', 0),
                comment_count=details.get('comment_count', 0),
                thumbnail_url=details['thumbnails'].get('high', {}).get('url'),
                tags=details.get('tags', []),
                category_id=details.get('category_id'),
                language=details.get('language'),
                has_captions=details.get('has_captions', False),
                is_live_content=details.get('is_live', False),
                metadata=details.get('raw_data', {})
            )
            
            db.add(video)
            discovered_videos.append(video)
            
            if len(discovered_videos) >= max_results:
                break
        
        await db.commit()
        
        logger.info(f"Discovered {len(discovered_videos)} videos from search: {query}")
        return discovered_videos
    
    async def _process_and_store_videos(
        self,
        video_details: List[Dict[str, Any]],
        channel: Channel,
        db: AsyncSession,
        filters: VideoFilters
    ) -> List[Video]:
        """Process video details and store in database"""
        stored_videos = []
        
        for details in video_details:
            # Apply filters
            if not filters.apply_to_video(details):
                continue
            
            # Check if video already exists
            existing = await db.execute(
                select(Video).filter(
                    Video.youtube_video_id == details['video_id']
                )
            )
            if existing.scalar_one_or_none():
                continue
            
            # Create video record
            video = Video(
                youtube_video_id=details['video_id'],
                channel_id=channel.id,
                title=details['title'],
                description=details.get('description', ''),
                published_at=datetime.fromisoformat(
                    details['published_at'].replace('Z', '+00:00')
                ),
                duration_seconds=details['duration_seconds'],
                view_count=details.get('view_count', 0),
                like_count=details.get('like_count', 0),
                comment_count=details.get('comment_count', 0),
                thumbnail_url=details['thumbnails'].get('high', {}).get('url'),
                tags=details.get('tags', []),
                category_id=details.get('category_id'),
                language=details.get('language'),
                has_captions=details.get('has_captions', False),
                is_live_content=details.get('is_live', False),
                metadata=details.get('raw_data', {})
            )
            
            db.add(video)
            stored_videos.append(video)
            
            # Queue for transcript processing if it has captions
            if video.has_captions:
                await self._queue_transcript_processing(video, db)
        
        await db.commit()
        return stored_videos
    
    def _apply_advanced_filters(self, video: Video, filters: VideoFilters) -> bool:
        """Apply filters that couldn't be applied at database level"""
        # Title filters
        if filters.title_contains:
            title_lower = video.title.lower()
            if not any(term.lower() in title_lower for term in filters.title_contains):
                return False
        
        if filters.title_excludes:
            title_lower = video.title.lower()
            if any(term.lower() in title_lower for term in filters.title_excludes):
                return False
        
        # Description filters
        if filters.description_contains and video.description:
            desc_lower = video.description.lower()
            if not any(term.lower() in desc_lower for term in filters.description_contains):
                return False
        
        if filters.description_excludes and video.description:
            desc_lower = video.description.lower()
            if any(term.lower() in desc_lower for term in filters.description_excludes):
                return False
        
        # Tag filters
        if filters.required_tags and video.tags:
            if not all(tag in video.tags for tag in filters.required_tags):
                return False
        
        if filters.any_tags and video.tags:
            if not any(tag in video.tags for tag in filters.any_tags):
                return False
        
        if filters.excluded_tags and video.tags:
            if any(tag in video.tags for tag in filters.excluded_tags):
                return False
        
        # Language filter
        if filters.language and video.language != filters.language:
            return False
        
        if filters.languages and video.language not in filters.languages:
            return False
        
        # Caption requirement
        if filters.require_captions and not video.has_captions:
            return False
        
        # Shorts exclusion (videos <= 60 seconds)
        if filters.exclude_shorts and video.duration_seconds <= 60:
            return False
        
        # Live content exclusion
        if filters.exclude_live and video.is_live_content:
            return False
        
        return True
    
    async def _ensure_channel_exists(
        self,
        youtube_channel_id: str,
        db: AsyncSession
    ) -> Channel:
        """Ensure a channel exists in the database"""
        # Check if channel exists
        result = await db.execute(
            select(Channel).filter(
                Channel.youtube_channel_id == youtube_channel_id
            )
        )
        channel = result.scalar_one_or_none()
        
        if channel:
            return channel
        
        # Fetch channel info and create
        youtube = await self._get_youtube_client()
        channel_info = await youtube.get_channel_info(youtube_channel_id)
        
        channel = Channel(
            youtube_channel_id=channel_info['channel_id'],
            title=channel_info['title'],
            description=channel_info.get('description', ''),
            custom_url=channel_info.get('custom_url'),
            country=channel_info.get('country'),
            published_at=datetime.fromisoformat(
                channel_info['published_at'].replace('Z', '+00:00')
            ),
            subscriber_count=channel_info.get('subscriber_count', 0),
            view_count=channel_info.get('view_count', 0),
            video_count=channel_info.get('video_count', 0),
            uploads_playlist_id=channel_info.get('uploads_playlist_id'),
            status=ChannelStatus.ACTIVE,
            metadata=channel_info.get('raw_data', {})
        )
        
        db.add(channel)
        await db.commit()
        await db.refresh(channel)
        
        return channel
    
    async def _queue_transcript_processing(self, video: Video, db: AsyncSession):
        """Queue a video for transcript processing"""
        job = ProcessingJob(
            video_id=video.id,
            job_type=JobType.TRANSCRIPT_FETCH,
            status=JobStatus.PENDING,
            priority=3,
            parameters={'video_id': str(video.id)}
        )
        db.add(job)
        # Commit will be handled by caller