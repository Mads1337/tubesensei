from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from datetime import datetime, timedelta
import asyncio

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.idea import Idea
from app.core.exceptions import NotFoundException, ValidationException
from app.integrations.youtube_api import YouTubeAPIClient


class ChannelService:
    """Service for channel operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.youtube = YouTubeAPIClient()
    
    async def list_channels(
        self,
        status: Optional[ChannelStatus] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List channels with filtering and pagination"""
        query = select(Channel)
        
        # Apply filters
        if status:
            query = query.where(Channel.status == status)
        
        if search:
            query = query.where(
                or_(
                    Channel.name.ilike(f"%{search}%"),
                    Channel.description.ilike(f"%{search}%")
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get paginated results
        query = query.order_by(Channel.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        channels = result.scalars().all()
        
        # Enrich with stats
        enriched = []
        for channel in channels:
            stats = await self._get_channel_stats(channel.id)
            channel_dict = channel.to_dict()
            channel_dict["stats"] = stats
            enriched.append(channel_dict)
        
        return {
            "items": enriched,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    async def get_channel(self, channel_id: str) -> Channel:
        """Get channel by ID"""
        channel = await self.db.get(Channel, channel_id)
        if not channel:
            raise NotFoundException("Channel", channel_id)
        return channel
    
    async def add_channel(self, data: Dict[str, Any]) -> Channel:
        """Add new channel for monitoring"""
        youtube_channel_id = data.get("youtube_channel_id")
        
        # Extract YouTube channel ID from URL if provided
        if youtube_channel_id and "youtube.com" in youtube_channel_id:
            # Extract channel ID from URL
            if "/channel/" in youtube_channel_id:
                youtube_channel_id = youtube_channel_id.split("/channel/")[1].split("/")[0].split("?")[0]
            elif "/@" in youtube_channel_id:
                # Handle @username format - need to resolve to channel ID
                handle = youtube_channel_id.split("/@")[1].split("/")[0].split("?")[0]
                youtube_channel_id = handle  # YouTube API will resolve this
        
        # In development mode, create mock channel info
        from app.core.config import get_settings
        settings = get_settings()
        
        # Force real YouTube API even in DEBUG mode when API key is present
        use_mock = settings.DEBUG and not settings.YOUTUBE_API_KEY
        
        if use_mock:
            # Mock channel info for development
            if "/@" in str(data.get("youtube_channel_id", "")):
                handle = str(data.get("youtube_channel_id", "")).split("/@")[1].split("/")[0].split("?")[0]
                channel_info = {
                    "id": f"UC{handle[:22]}mock",  # Mock channel ID
                    "snippet": {
                        "title": handle.replace("_", " ").title(),
                        "description": f"Mock channel for {handle}",
                        "customUrl": f"@{handle}",
                        "publishedAt": "2020-01-01T00:00:00Z",
                        "thumbnails": {"default": {"url": "https://via.placeholder.com/88x88"}}
                    },
                    "statistics": {
                        "viewCount": "1000",
                        "subscriberCount": "100",
                        "videoCount": "10"
                    }
                }
                youtube_channel_id = channel_info["id"]  # Use the mock ID
            elif "/channel/" in str(data.get("youtube_channel_id", "")):
                channel_id = str(data.get("youtube_channel_id", "")).split("/channel/")[1].split("/")[0].split("?")[0]
                channel_info = {
                    "id": channel_id,
                    "snippet": {
                        "title": f"Mock Channel {channel_id[:8]}",
                        "description": f"Mock channel for {channel_id}",
                        "publishedAt": "2020-01-01T00:00:00Z",
                        "thumbnails": {"default": {"url": "https://via.placeholder.com/88x88"}}
                    },
                    "statistics": {
                        "viewCount": "1000",
                        "subscriberCount": "100",
                        "videoCount": "10"
                    }
                }
                youtube_channel_id = channel_id
            else:
                # Direct channel ID
                channel_info = {
                    "id": youtube_channel_id,
                    "snippet": {
                        "title": f"Mock Channel {youtube_channel_id[:8]}",
                        "description": f"Mock channel for {youtube_channel_id}",
                        "publishedAt": "2020-01-01T00:00:00Z",
                        "thumbnails": {"default": {"url": "https://via.placeholder.com/88x88"}}
                    },
                    "statistics": {
                        "viewCount": "1000",
                        "subscriberCount": "100", 
                        "videoCount": "10"
                    }
                }
        else:
            # Validate YouTube channel exists in production
            print(f"DEBUG: Calling YouTube API for channel: {youtube_channel_id}")
            try:
                # Check if this is a handle (extracted from @username URLs) or channel ID
                if "/@" in str(data.get("youtube_channel_id", "")) or not youtube_channel_id.startswith("UC"):
                    # This is a handle, use get_channel_by_handle
                    print(f"DEBUG: Using get_channel_by_handle for: {youtube_channel_id}")
                    channel_info = await self.youtube.get_channel_by_handle(youtube_channel_id)
                else:
                    # This is a channel ID, use get_channel_info
                    print(f"DEBUG: Using get_channel_info for: {youtube_channel_id}")
                    channel_info = await self.youtube.get_channel_info(youtube_channel_id)
                
                print(f"DEBUG: YouTube API returned: {channel_info}")
                if not channel_info:
                    raise ValidationException({
                        "youtube_channel_id": "Invalid YouTube channel ID"
                    })
            except Exception as e:
                print(f"DEBUG: YouTube API error: {e}")
                raise
        
        # Check if already exists
        # Handle both mock and real API responses
        channel_id_key = "channel_id" if "channel_id" in channel_info else "id"
        existing = await self.db.execute(
            select(Channel).where(
                Channel.youtube_channel_id == channel_info[channel_id_key]
            )
        )
        if existing.scalar_one_or_none():
            raise ValidationException({
                "youtube_channel_id": "Channel already being monitored"
            })
        
        # Create channel - handle both mock and real API response formats
        if use_mock:
            # Mock format has nested snippet/statistics
            channel = Channel(
                name=channel_info["snippet"]["title"],
                youtube_channel_id=channel_info["id"],
                channel_url=f"https://youtube.com/channel/{channel_info['id']}",
                description=channel_info["snippet"].get("description"),
                subscriber_count=int(channel_info.get("statistics", {}).get("subscriberCount", 0)),
                video_count=int(channel_info.get("statistics", {}).get("videoCount", 0)),
                view_count=int(channel_info.get("statistics", {}).get("viewCount", 0)),
                thumbnail_url=channel_info["snippet"].get("thumbnails", {}).get("default", {}).get("url"),
                published_at=datetime.fromisoformat(channel_info["snippet"]["publishedAt"].replace("Z", "+00:00")),
                country=channel_info["snippet"].get("country"),
                custom_url=channel_info["snippet"].get("customUrl"),
                status=ChannelStatus.ACTIVE,
                processing_config=data.get("processing_config", {})
            )
        else:
            # Real API format has flattened structure
            channel = Channel(
                name=channel_info["title"],
                youtube_channel_id=channel_info["channel_id"],
                channel_url=f"https://youtube.com/channel/{channel_info['channel_id']}",
                description=channel_info.get("description"),
                subscriber_count=int(channel_info.get("subscriber_count", 0)),
                video_count=int(channel_info.get("video_count", 0)),
                view_count=int(channel_info.get("view_count", 0)),
                thumbnail_url=channel_info.get("thumbnails", {}).get("default", {}).get("url"),
                published_at=datetime.fromisoformat(channel_info["published_at"].replace("Z", "+00:00")),
                country=channel_info.get("country"),
                custom_url=channel_info.get("custom_url"),
                status=ChannelStatus.ACTIVE,
                processing_config=data.get("processing_config", {})
            )
        
        self.db.add(channel)
        await self.db.commit()
        await self.db.refresh(channel)
        
        # Queue initial video discovery
        await self._queue_channel_discovery(channel.id)
        
        return channel
    
    async def update_channel(
        self,
        channel_id: str,
        data: Dict[str, Any]
    ) -> Channel:
        """Update channel configuration"""
        channel = await self.get_channel(channel_id)
        
        # Update fields
        for field, value in data.items():
            if hasattr(channel, field) and field not in ["id", "created_at", "youtube_channel_id"]:
                setattr(channel, field, value)
        
        channel.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(channel)
        
        return channel
    
    async def sync_channel(self, channel_id: str) -> Dict[str, Any]:
        """Manually sync channel data"""
        channel = await self.get_channel(channel_id)
        
        # Check if we're in DEBUG mode
        from app.core.config import get_settings
        settings = get_settings()
        
        # Force real YouTube API even in DEBUG mode when API key is present
        use_mock = settings.DEBUG and not settings.YOUTUBE_API_KEY
        
        if use_mock:
            # Mock channel sync for development
            channel_info = {
                "id": channel.youtube_channel_id,
                "snippet": {
                    "title": channel.name,
                    "description": channel.description or f"Mock channel for {channel.name}",
                },
                "statistics": {
                    "viewCount": str(channel.view_count + 100),  # Simulate some growth
                    "subscriberCount": str(channel.subscriber_count + 10),
                    "videoCount": str(channel.video_count + 2)
                }
            }
        else:
            # Get latest channel info from YouTube API
            channel_info = await self.youtube.get_channel_info(
                channel.youtube_channel_id
            )
        
        if channel_info:
            channel.subscriber_count = int(channel_info.get("statistics", {}).get("subscriberCount", 0))
            channel.video_count = int(channel_info.get("statistics", {}).get("videoCount", 0))
            channel.view_count = int(channel_info.get("statistics", {}).get("viewCount", 0))
            channel.description = channel_info["snippet"].get("description")
            channel.last_checked_at = datetime.utcnow()
            
            await self.db.commit()
        
        # Queue video discovery
        job_id = await self._queue_channel_discovery(channel.id)
        
        return {
            "channel_id": str(channel.id),
            "updated": True,
            "job_id": job_id,
            "stats": channel_info.get("statistics") if channel_info else None
        }
    
    async def delete_channel(self, channel_id: str) -> bool:
        """Delete channel (soft delete)"""
        channel = await self.get_channel(channel_id)
        channel.status = ChannelStatus.INACTIVE
        channel.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def _get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get processing statistics for channel"""
        # Video stats
        video_stats = await self.db.execute(
            select(
                func.count(Video.id).label("total_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.COMPLETED
                ).label("processed_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.PROCESSING
                ).label("processing_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.FAILED
                ).label("failed_videos")
            ).where(Video.channel_id == channel_id)
        )
        
        stats = video_stats.one()._asdict()
        
        # Idea stats
        idea_count = await self.db.scalar(
            select(func.count(Idea.id)).select_from(Idea)
            .join(Video, Idea.video_id == Video.id)
            .where(Video.channel_id == channel_id)
        )
        
        stats["total_ideas"] = idea_count
        
        return stats
    
    async def _queue_channel_discovery(self, channel_id: str) -> str:
        """Queue channel for video discovery"""
        # Import here to avoid circular imports
        from app.celery_app import celery_app
        
        # Create a Celery task for channel discovery
        task = celery_app.send_task(
            "app.workers.processing_tasks.discover_channel_videos_task",
            args=[str(channel_id)]
        )
        return task.id