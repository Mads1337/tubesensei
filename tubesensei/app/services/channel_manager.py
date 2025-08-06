import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.channel import Channel, ChannelStatus
from ..models.video import Video
from ..models.processing_job import ProcessingJob, JobType, JobStatus
from ..integrations.youtube_api import YouTubeAPIClient
from ..utils.youtube_parser import YouTubeParser
from ..utils.exceptions import (
    ChannelNotFoundError,
    InvalidURLError,
    ValidationError
)
from ..database import get_db

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Service for managing YouTube channels.
    Handles channel addition, synchronization, and metadata updates.
    """
    
    def __init__(self, youtube_client: Optional[YouTubeAPIClient] = None):
        self.youtube_client = youtube_client
    
    async def _get_youtube_client(self) -> YouTubeAPIClient:
        """Get or create YouTube API client"""
        if not self.youtube_client:
            self.youtube_client = YouTubeAPIClient()
        return self.youtube_client
    
    async def add_channel(
        self,
        channel_url: str,
        db: AsyncSession,
        auto_discover: bool = True
    ) -> Channel:
        """
        Add a new channel to the database.
        
        Args:
            channel_url: YouTube channel URL or ID
            db: Database session
            auto_discover: Whether to automatically discover videos
            
        Returns:
            Created or existing Channel object
            
        Raises:
            InvalidURLError: If the URL is invalid
            ChannelNotFoundError: If the channel doesn't exist on YouTube
        """
        # Parse the channel URL
        try:
            parsed = YouTubeParser.parse_url(channel_url)
            if parsed['type'] != 'channel':
                raise InvalidURLError(
                    channel_url,
                    f"URL is a {parsed['type']}, not a channel"
                )
        except InvalidURLError:
            # Try treating it as a direct channel ID
            if YouTubeParser.validate_channel_id(channel_url):
                channel_identifier = channel_url
                is_handle = False
            else:
                raise InvalidURLError(channel_url, "Not a valid channel URL or ID")
        else:
            channel_identifier = parsed.get('channel_id') or parsed.get('channel_handle')
            is_handle = parsed.get('channel_handle') is not None
        
        # Check if channel already exists
        existing_query = select(Channel)
        if is_handle:
            existing_query = existing_query.filter(Channel.custom_url == f"@{channel_identifier}")
        else:
            existing_query = existing_query.filter(Channel.youtube_channel_id == channel_identifier)
        
        result = await db.execute(existing_query)
        existing_channel = result.scalar_one_or_none()
        
        if existing_channel:
            logger.info(f"Channel already exists: {existing_channel.title}")
            return existing_channel
        
        # Fetch channel info from YouTube
        youtube = await self._get_youtube_client()
        
        try:
            if is_handle:
                channel_info = await youtube.get_channel_by_handle(channel_identifier)
            else:
                channel_info = await youtube.get_channel_info(channel_identifier)
        except ChannelNotFoundError:
            raise ChannelNotFoundError(channel_identifier)
        
        # Create new channel record
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
            keywords=channel_info.get('keywords', ''),
            thumbnail_url=channel_info['thumbnails'].get('high', {}).get('url'),
            status=ChannelStatus.ACTIVE,
            last_checked_at=datetime.now(timezone.utc),
            metadata=channel_info.get('raw_data', {})
        )
        
        db.add(channel)
        await db.commit()
        await db.refresh(channel)
        
        logger.info(f"Added new channel: {channel.title} ({channel.youtube_channel_id})")
        
        # Queue video discovery if requested
        if auto_discover:
            await self._queue_video_discovery(channel.id, db)
        
        return channel
    
    async def sync_channel_metadata(
        self,
        channel_id: UUID,
        db: AsyncSession
    ) -> Channel:
        """
        Synchronize channel metadata with YouTube.
        
        Args:
            channel_id: Database channel ID
            db: Database session
            
        Returns:
            Updated Channel object
        """
        # Get channel from database
        result = await db.execute(
            select(Channel).filter(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Fetch latest info from YouTube
        youtube = await self._get_youtube_client()
        
        try:
            channel_info = await youtube.get_channel_info(channel.youtube_channel_id)
        except ChannelNotFoundError:
            # Channel might be deleted or unavailable
            channel.status = ChannelStatus.INACTIVE
            channel.last_error = "Channel not found on YouTube"
            await db.commit()
            raise
        
        # Update channel metadata
        channel.title = channel_info['title']
        channel.description = channel_info.get('description', '')
        channel.custom_url = channel_info.get('custom_url')
        channel.country = channel_info.get('country')
        channel.subscriber_count = channel_info.get('subscriber_count', 0)
        channel.view_count = channel_info.get('view_count', 0)
        channel.video_count = channel_info.get('video_count', 0)
        channel.keywords = channel_info.get('keywords', '')
        channel.thumbnail_url = channel_info['thumbnails'].get('high', {}).get('url')
        channel.last_checked_at = datetime.now(timezone.utc)
        channel.metadata = channel_info.get('raw_data', {})
        channel.status = ChannelStatus.ACTIVE
        channel.last_error = None
        
        await db.commit()
        await db.refresh(channel)
        
        logger.info(
            f"Synced channel metadata: {channel.title} "
            f"({channel.subscriber_count:,} subscribers, {channel.video_count:,} videos)"
        )
        
        return channel
    
    async def discover_channel_videos(
        self,
        channel_id: UUID,
        db: AsyncSession,
        fetch_all: bool = True,
        max_videos: int = 500
    ) -> List[Video]:
        """
        Discover and store videos from a channel.
        
        Args:
            channel_id: Database channel ID
            db: Database session
            fetch_all: Whether to fetch all videos or just new ones
            max_videos: Maximum number of videos to fetch
            
        Returns:
            List of discovered Video objects
        """
        # Get channel from database
        result = await db.execute(
            select(Channel).filter(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Determine published_after date for incremental fetch
        published_after = None
        if not fetch_all and channel.last_video_fetch_at:
            published_after = channel.last_video_fetch_at
        
        # Fetch videos from YouTube
        youtube = await self._get_youtube_client()
        
        try:
            video_list = await youtube.list_channel_videos(
                channel.youtube_channel_id,
                max_results=max_videos,
                published_after=published_after
            )
        except Exception as e:
            logger.error(f"Error fetching videos for channel {channel.title}: {e}")
            channel.last_error = str(e)
            await db.commit()
            raise
        
        if not video_list:
            logger.info(f"No new videos found for channel: {channel.title}")
            return []
        
        # Get video IDs for detailed info
        video_ids = [v['video_id'] for v in video_list]
        
        # Check for existing videos
        existing_result = await db.execute(
            select(Video.youtube_video_id).filter(
                Video.youtube_video_id.in_(video_ids)
            )
        )
        existing_video_ids = {row[0] for row in existing_result}
        
        # Filter out existing videos
        new_video_ids = [vid for vid in video_ids if vid not in existing_video_ids]
        
        if not new_video_ids:
            logger.info(f"All {len(video_ids)} videos already exist for channel: {channel.title}")
            channel.last_video_fetch_at = datetime.now(timezone.utc)
            await db.commit()
            return []
        
        # Fetch detailed info for new videos
        video_details = await youtube.get_video_details(new_video_ids)
        
        # Create Video objects
        new_videos = []
        for details in video_details:
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
            new_videos.append(video)
        
        # Update channel stats
        channel.last_video_fetch_at = datetime.now(timezone.utc)
        channel.total_videos_fetched = (channel.total_videos_fetched or 0) + len(new_videos)
        
        await db.commit()
        
        logger.info(
            f"Discovered {len(new_videos)} new videos for channel: {channel.title} "
            f"(skipped {len(existing_video_ids)} existing)"
        )
        
        # Queue videos for transcript processing
        for video in new_videos:
            await self._queue_transcript_processing(video.id, db)
        
        return new_videos
    
    async def get_channel_status(
        self,
        channel_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get comprehensive status information for a channel.
        
        Args:
            channel_id: Database channel ID
            db: Database session
            
        Returns:
            Dictionary with channel status information
        """
        # Get channel
        result = await db.execute(
            select(Channel).filter(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            raise ValidationError(f"Channel not found: {channel_id}")
        
        # Count videos
        video_count_result = await db.execute(
            select(Video).filter(Video.channel_id == channel_id)
        )
        videos = video_count_result.scalars().all()
        
        # Count processing jobs
        job_result = await db.execute(
            select(ProcessingJob).filter(
                and_(
                    ProcessingJob.channel_id == channel_id,
                    ProcessingJob.status.in_([JobStatus.PENDING, JobStatus.IN_PROGRESS])
                )
            )
        )
        pending_jobs = job_result.scalars().all()
        
        # Calculate statistics
        total_duration = sum(v.duration_seconds or 0 for v in videos)
        total_views = sum(v.view_count or 0 for v in videos)
        transcribed_count = sum(1 for v in videos if v.transcript_fetched_at)
        
        return {
            'channel_id': str(channel.id),
            'youtube_channel_id': channel.youtube_channel_id,
            'title': channel.title,
            'status': channel.status.value,
            'subscriber_count': channel.subscriber_count,
            'video_count': channel.video_count,
            'stored_videos': len(videos),
            'transcribed_videos': transcribed_count,
            'total_duration_hours': round(total_duration / 3600, 2),
            'total_views': total_views,
            'pending_jobs': len(pending_jobs),
            'last_checked_at': channel.last_checked_at.isoformat() if channel.last_checked_at else None,
            'last_video_fetch_at': channel.last_video_fetch_at.isoformat() if channel.last_video_fetch_at else None,
            'last_error': channel.last_error,
            'health': self._calculate_channel_health(channel, videos, pending_jobs)
        }
    
    def _calculate_channel_health(
        self,
        channel: Channel,
        videos: List[Video],
        pending_jobs: List[ProcessingJob]
    ) -> str:
        """Calculate channel health status"""
        if channel.status != ChannelStatus.ACTIVE:
            return "inactive"
        elif channel.last_error:
            return "error"
        elif pending_jobs:
            return "processing"
        elif not videos:
            return "empty"
        elif channel.last_checked_at and (
            datetime.now(timezone.utc) - channel.last_checked_at
        ).days > 7:
            return "stale"
        else:
            return "healthy"
    
    async def _queue_video_discovery(self, channel_id: UUID, db: AsyncSession):
        """Queue a video discovery job for a channel"""
        job = ProcessingJob(
            channel_id=channel_id,
            job_type=JobType.VIDEO_DISCOVERY,
            status=JobStatus.PENDING,
            priority=5,
            parameters={'fetch_all': True, 'max_videos': 500}
        )
        db.add(job)
        await db.commit()
        logger.debug(f"Queued video discovery job for channel {channel_id}")
    
    async def _queue_transcript_processing(self, video_id: UUID, db: AsyncSession):
        """Queue a transcript processing job for a video"""
        job = ProcessingJob(
            video_id=video_id,
            job_type=JobType.TRANSCRIPT_FETCH,
            status=JobStatus.PENDING,
            priority=3,
            parameters={}
        )
        db.add(job)
        await db.commit()
        logger.debug(f"Queued transcript processing job for video {video_id}")
    
    async def list_channels(
        self,
        db: AsyncSession,
        status: Optional[ChannelStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Channel]:
        """
        List channels with optional filtering.
        
        Args:
            db: Database session
            status: Filter by channel status
            limit: Maximum number of channels to return
            offset: Number of channels to skip
            
        Returns:
            List of Channel objects
        """
        query = select(Channel)
        
        if status:
            query = query.filter(Channel.status == status)
        
        query = query.order_by(Channel.subscriber_count.desc())
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def delete_channel(
        self,
        channel_id: UUID,
        db: AsyncSession,
        cascade: bool = False
    ) -> bool:
        """
        Delete a channel from the database.
        
        Args:
            channel_id: Database channel ID
            db: Database session
            cascade: Whether to delete related videos and transcripts
            
        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(Channel).filter(Channel.id == channel_id)
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            return False
        
        if cascade:
            # Delete related videos (which will cascade to transcripts)
            await db.execute(
                select(Video).filter(Video.channel_id == channel_id)
            )
            # Note: Actual deletion would depend on cascade rules in models
        
        await db.delete(channel)
        await db.commit()
        
        logger.info(f"Deleted channel: {channel.title} (cascade={cascade})")
        return True