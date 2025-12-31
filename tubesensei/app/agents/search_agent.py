"""
Search Agent

Discovers videos by searching YouTube for the campaign topic.
This is typically the first agent to run, finding initial seed videos
that are then expanded by other agents.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.agents.base import BaseAgent, AgentResult, AgentContext, AgentEventType
from app.models.agent_run import AgentType
from app.models.video import Video, VideoStatus
from app.models.channel import Channel, ChannelStatus
from app.models.campaign_video import CampaignVideo, DiscoverySource
from app.models.campaign_channel import CampaignChannel
from app.integrations.youtube_api import YouTubeAPIClient

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Agent that searches YouTube for videos matching the campaign topic.

    Input:
        topic: str - The search query
        max_results: int - Maximum videos to discover (default: 50)

    Output:
        video_ids: List[UUID] - IDs of discovered videos
        channel_ids: List[UUID] - IDs of channels found
        new_videos_count: int - Number of new videos added
        new_channels_count: int - Number of new channels added
    """

    agent_type = AgentType.SEARCH

    def __init__(self, context: AgentContext, youtube_client: Optional[YouTubeAPIClient] = None):
        super().__init__(context)
        self.youtube_client = youtube_client

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Search YouTube for videos matching the topic.

        Args:
            input_data: Contains 'topic' and optional 'max_results'

        Returns:
            AgentResult with discovered video and channel IDs
        """
        topic = input_data.get("topic", self.context.campaign.topic)
        max_results = input_data.get("max_results", self.config.get("search_limit", 50))

        logger.info(f"SearchAgent: Searching for '{topic}' (max {max_results} results)")

        discovered_video_ids: List[UUID] = []
        discovered_channel_ids: List[UUID] = []
        new_videos = 0
        new_channels = 0

        try:
            # Initialize YouTube client if not provided
            youtube = self.youtube_client or YouTubeAPIClient()

            # Apply rate limiting
            async with self.context.youtube_rate_limiter.acquire():
                # Search YouTube
                search_results = await youtube.search_videos(
                    query=topic,
                    max_results=max_results,
                    order="relevance"
                )
                self.increment_api_calls()

            if not search_results:
                logger.info("SearchAgent: No search results found")
                return self._build_result(success=True, data={
                    "video_ids": [],
                    "channel_ids": [],
                    "new_videos_count": 0,
                    "new_channels_count": 0,
                })

            # Get video details for all search results
            video_ids_to_fetch = [r["video_id"] for r in search_results]

            async with self.context.youtube_rate_limiter.acquire():
                video_details = await youtube.get_video_details(video_ids_to_fetch)
                self.increment_api_calls()

            total_items = len(video_details)

            for idx, video_data in enumerate(video_details):
                # Check if we should stop
                if await self.check_should_stop():
                    logger.info("SearchAgent: Stopping due to limit reached or cancellation")
                    break

                self.update_progress(
                    ((idx + 1) / total_items) * 100,
                    current_item=video_data.get("title", "")[:50]
                )

                try:
                    # Process video and channel
                    video_id, channel_id, is_new_video, is_new_channel = await self._process_video(
                        video_data
                    )

                    if video_id:
                        discovered_video_ids.append(video_id)
                        self.increment_produced()
                        if is_new_video:
                            new_videos += 1

                    if channel_id and channel_id not in discovered_channel_ids:
                        discovered_channel_ids.append(channel_id)
                        if is_new_channel:
                            new_channels += 1

                except Exception as e:
                    error_msg = f"Error processing video {video_data.get('video_id')}: {e}"
                    logger.error(error_msg)
                    self.add_error(error_msg)

                self.increment_processed()

            # Commit all changes
            await self.db.commit()

            logger.info(
                f"SearchAgent: Discovered {len(discovered_video_ids)} videos "
                f"from {len(discovered_channel_ids)} channels "
                f"({new_videos} new videos, {new_channels} new channels)"
            )

            return self._build_result(success=True, data={
                "video_ids": [str(vid) for vid in discovered_video_ids],
                "channel_ids": [str(cid) for cid in discovered_channel_ids],
                "new_videos_count": new_videos,
                "new_channels_count": new_channels,
            })

        except Exception as e:
            logger.exception(f"SearchAgent failed: {e}")
            return self._build_result(success=False, data={
                "video_ids": [str(vid) for vid in discovered_video_ids],
                "channel_ids": [str(cid) for cid in discovered_channel_ids],
                "new_videos_count": new_videos,
                "new_channels_count": new_channels,
                "error": str(e),
            })

    async def _process_video(
        self, video_data: Dict[str, Any]
    ) -> tuple[Optional[UUID], Optional[UUID], bool, bool]:
        """
        Process a single video from search results.

        Creates or retrieves Video and Channel records,
        and creates CampaignVideo and CampaignChannel junction records.

        Returns:
            Tuple of (video_id, channel_id, is_new_video, is_new_channel)
        """
        youtube_video_id = video_data.get("video_id")
        youtube_channel_id = video_data.get("channel_id")

        if not youtube_video_id or not youtube_channel_id:
            return None, None, False, False

        is_new_video = False
        is_new_channel = False

        # Get or create channel
        channel = await self._get_or_create_channel(youtube_channel_id, video_data)
        if not channel:
            return None, None, False, False

        is_new_channel = channel.created_at == channel.updated_at  # Rough check

        # Get or create video
        video = await self._get_or_create_video(youtube_video_id, channel.id, video_data)
        if not video:
            return None, channel.id, False, is_new_channel

        is_new_video = video.created_at == video.updated_at  # Rough check

        # Create CampaignVideo junction (if not exists)
        await self._create_campaign_video(video.id, channel.id)

        # Create CampaignChannel junction (if not exists)
        await self._create_campaign_channel(channel.id, video.id)

        return video.id, channel.id, is_new_video, is_new_channel

    async def _get_or_create_channel(
        self, youtube_channel_id: str, video_data: Dict[str, Any]
    ) -> Optional[Channel]:
        """Get existing channel or create new one."""
        # Check if channel exists
        result = await self.db.execute(
            select(Channel).where(Channel.youtube_channel_id == youtube_channel_id)
        )
        channel = result.scalar_one_or_none()

        if channel:
            return channel

        # Create new channel with minimal data from video
        channel = Channel(
            youtube_channel_id=youtube_channel_id,
            name=video_data.get("channel_title", "Unknown Channel"),
            status=ChannelStatus.ACTIVE,
            channel_metadata={
                "discovered_via": "topic_campaign_search",
                "campaign_id": str(self.campaign_id),
            }
        )
        self.db.add(channel)
        await self.db.flush()

        return channel

    async def _get_or_create_video(
        self, youtube_video_id: str, channel_id: UUID, video_data: Dict[str, Any]
    ) -> Optional[Video]:
        """Get existing video or create new one."""
        # Check if video exists
        result = await self.db.execute(
            select(Video).where(Video.youtube_video_id == youtube_video_id)
        )
        video = result.scalar_one_or_none()

        if video:
            return video

        # Create new video
        video = Video(
            youtube_video_id=youtube_video_id,
            channel_id=channel_id,
            title=video_data.get("title", "Untitled"),
            description=video_data.get("description", ""),
            published_at=video_data.get("published_at"),
            duration_seconds=video_data.get("duration_seconds", 0),
            view_count=video_data.get("view_count", 0),
            like_count=video_data.get("like_count", 0),
            comment_count=video_data.get("comment_count", 0),
            tags=video_data.get("tags", []),
            category_id=video_data.get("category_id"),
            language=video_data.get("language"),
            has_captions=video_data.get("has_captions", False),
            caption_languages=video_data.get("caption_languages", []),
            status=VideoStatus.DISCOVERED,
            video_metadata={
                "discovered_via": "topic_campaign_search",
                "campaign_id": str(self.campaign_id),
            }
        )
        self.db.add(video)
        await self.db.flush()

        # Update campaign stats
        self.context.campaign.increment_discovered()

        return video

    async def _create_campaign_video(
        self, video_id: UUID, channel_id: UUID
    ) -> Optional[CampaignVideo]:
        """Create CampaignVideo junction record."""
        # Check if already exists
        result = await self.db.execute(
            select(CampaignVideo).where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.video_id == video_id
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        # Create new junction
        campaign_video = CampaignVideo(
            campaign_id=self.campaign_id,
            video_id=video_id,
            discovery_source=DiscoverySource.SEARCH,
            source_channel_id=channel_id,
            agent_run_id=self._agent_run.id if self._agent_run else None,
            discovery_depth=0,
            discovery_order=self._items_produced,
        )
        self.db.add(campaign_video)
        await self.db.flush()

        # Emit discovery event
        self._emit_event(
            AgentEventType.ITEM_DISCOVERED,
            data={"video_id": str(video_id), "source": "search"},
            message=f"Discovered video from search"
        )

        return campaign_video

    async def _create_campaign_channel(
        self, channel_id: UUID, source_video_id: UUID
    ) -> Optional[CampaignChannel]:
        """Create CampaignChannel junction record."""
        # Check if already exists
        result = await self.db.execute(
            select(CampaignChannel).where(
                CampaignChannel.campaign_id == self.campaign_id,
                CampaignChannel.channel_id == channel_id
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        # Create new junction
        campaign_channel = CampaignChannel(
            campaign_id=self.campaign_id,
            channel_id=channel_id,
            discovery_source=DiscoverySource.SEARCH,
            source_video_id=source_video_id,
            videos_limit=self.per_channel_limit,
        )
        self.db.add(campaign_channel)
        await self.db.flush()

        return campaign_channel
