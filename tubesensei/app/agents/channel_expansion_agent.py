"""
Channel Expansion Agent

Discovers additional videos from channels found during the campaign.
Takes a channel ID and fetches all its videos, respecting per-channel limits.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update

from app.agents.base import BaseAgent, AgentResult, AgentContext, AgentEventType
from app.models.agent_run import AgentType
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.models.campaign_video import CampaignVideo, DiscoverySource
from app.models.campaign_channel import CampaignChannel
from app.integrations.youtube_api import YouTubeAPIClient

logger = logging.getLogger(__name__)


class ChannelExpansionAgent(BaseAgent):
    """
    Agent that expands a channel by fetching all its videos.

    Input:
        channel_id: UUID - Internal channel ID to expand
        youtube_channel_id: str - YouTube channel ID (optional, will lookup if not provided)
        max_videos: int - Maximum videos to fetch (default: per_channel_limit from config)

    Output:
        video_ids: List[UUID] - IDs of discovered videos
        new_videos_count: int - Number of new videos added
        was_expanded: bool - Whether the channel was fully expanded
    """

    agent_type = AgentType.CHANNEL_EXPANSION

    def __init__(self, context: AgentContext, youtube_client: Optional[YouTubeAPIClient] = None):
        super().__init__(context)
        self.youtube_client = youtube_client

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Expand a channel by fetching its videos.

        Args:
            input_data: Contains 'channel_id' and optional 'max_videos'

        Returns:
            AgentResult with discovered video IDs
        """
        channel_id = input_data.get("channel_id")
        if isinstance(channel_id, str):
            channel_id = UUID(channel_id)

        youtube_channel_id = input_data.get("youtube_channel_id")
        max_videos = input_data.get("max_videos", self.per_channel_limit)

        logger.info(f"ChannelExpansionAgent: Expanding channel {channel_id} (max {max_videos} videos)")

        discovered_video_ids: List[UUID] = []
        new_videos = 0

        try:
            # Get channel from database
            result = await self.db.execute(
                select(Channel).where(Channel.id == channel_id)
            )
            channel = result.scalar_one_or_none()

            if not channel:
                error_msg = f"Channel {channel_id} not found"
                logger.error(error_msg)
                return self._build_result(success=False, data={"error": error_msg})

            youtube_channel_id = youtube_channel_id or channel.youtube_channel_id

            # Get campaign channel record
            cc_result = await self.db.execute(
                select(CampaignChannel).where(
                    CampaignChannel.campaign_id == self.campaign_id,
                    CampaignChannel.channel_id == channel_id
                )
            )
            campaign_channel = cc_result.scalar_one_or_none()

            if not campaign_channel:
                # Create campaign channel record if it doesn't exist
                campaign_channel = CampaignChannel(
                    campaign_id=self.campaign_id,
                    channel_id=channel_id,
                    discovery_source=DiscoverySource.CHANNEL_EXPANSION,
                    videos_limit=max_videos,
                )
                self.db.add(campaign_channel)
                await self.db.flush()

            # Check if already expanded or limit reached
            if campaign_channel.was_expanded:
                logger.info(f"Channel {channel_id} already expanded")
                return self._build_result(success=True, data={
                    "video_ids": [],
                    "new_videos_count": 0,
                    "was_expanded": True,
                    "skipped": True,
                })

            if campaign_channel.limit_reached:
                logger.info(f"Channel {channel_id} limit already reached")
                return self._build_result(success=True, data={
                    "video_ids": [],
                    "new_videos_count": 0,
                    "was_expanded": True,
                    "limit_reached": True,
                })

            # Calculate remaining slots
            remaining_slots = campaign_channel.remaining_video_slots

            # Initialize YouTube client
            youtube = self.youtube_client or YouTubeAPIClient()

            # Fetch videos from channel
            async with self.context.youtube_rate_limiter.acquire():
                videos_data = await youtube.list_channel_videos(
                    channel_id=youtube_channel_id,
                    max_results=min(remaining_slots * 3, 100)  # Fetch more to filter
                )
                self.increment_api_calls()

            if not videos_data:
                logger.info(f"No videos found for channel {channel_id}")
                campaign_channel.mark_expanded()
                await self.db.commit()
                return self._build_result(success=True, data={
                    "video_ids": [],
                    "new_videos_count": 0,
                    "was_expanded": True,
                })

            # Get full details for videos
            video_ids_to_fetch = [v["video_id"] for v in videos_data]

            async with self.context.youtube_rate_limiter.acquire():
                video_details = await youtube.get_video_details(video_ids_to_fetch[:50])  # Batch limit
                self.increment_api_calls()

            total_items = len(video_details)

            for idx, video_data in enumerate(video_details):
                # Check if we should stop
                if await self.check_should_stop():
                    logger.info("ChannelExpansionAgent: Stopping due to limit reached or cancellation")
                    break

                # Check per-channel limit
                if len(discovered_video_ids) >= remaining_slots:
                    logger.info(f"Channel {channel_id} per-channel limit reached")
                    campaign_channel.mark_limit_reached()
                    break

                self.update_progress(
                    ((idx + 1) / total_items) * 100,
                    current_item=video_data.get("title", "")[:50]
                )

                try:
                    video_id, is_new = await self._process_video(video_data, channel.id)

                    if video_id:
                        discovered_video_ids.append(video_id)
                        campaign_channel.increment_discovered()
                        self.increment_produced()
                        if is_new:
                            new_videos += 1

                except Exception as e:
                    error_msg = f"Error processing video {video_data.get('video_id')}: {e}"
                    logger.error(error_msg)
                    self.add_error(error_msg)

                self.increment_processed()

            # Mark channel as expanded
            campaign_channel.mark_expanded()

            # Update campaign stats
            self.context.campaign.increment_channels()

            # Commit all changes
            await self.db.commit()

            logger.info(
                f"ChannelExpansionAgent: Discovered {len(discovered_video_ids)} videos "
                f"from channel {channel_id} ({new_videos} new)"
            )

            return self._build_result(success=True, data={
                "video_ids": [str(vid) for vid in discovered_video_ids],
                "new_videos_count": new_videos,
                "was_expanded": True,
            })

        except Exception as e:
            logger.exception(f"ChannelExpansionAgent failed: {e}")

            # Record error on campaign channel
            if campaign_channel:
                campaign_channel.record_error(str(e))
                await self.db.commit()

            return self._build_result(success=False, data={
                "video_ids": [str(vid) for vid in discovered_video_ids],
                "new_videos_count": new_videos,
                "was_expanded": False,
                "error": str(e),
            })

    async def _process_video(
        self, video_data: Dict[str, Any], channel_id: UUID
    ) -> tuple[Optional[UUID], bool]:
        """
        Process a single video from channel.

        Returns:
            Tuple of (video_id, is_new)
        """
        youtube_video_id = video_data.get("video_id")
        if not youtube_video_id:
            return None, False

        is_new = False

        # Check if video exists
        result = await self.db.execute(
            select(Video).where(Video.youtube_video_id == youtube_video_id)
        )
        video = result.scalar_one_or_none()

        if not video:
            # Parse published_at datetime string
            published_at_str = video_data.get("published_at")
            published_at = None
            if published_at_str:
                try:
                    published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass

            # Create new video
            video = Video(
                youtube_video_id=youtube_video_id,
                channel_id=channel_id,
                title=video_data.get("title", "Untitled"),
                description=video_data.get("description", ""),
                published_at=published_at,
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
                    "discovered_via": "topic_campaign_channel_expansion",
                    "campaign_id": str(self.campaign_id),
                }
            )
            self.db.add(video)
            await self.db.flush()
            is_new = True

            # Update campaign stats
            self.context.campaign.increment_discovered()

        # Create CampaignVideo junction
        cv_result = await self.db.execute(
            select(CampaignVideo).where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.video_id == video.id
            )
        )
        existing_cv = cv_result.scalar_one_or_none()

        if not existing_cv:
            campaign_video = CampaignVideo(
                campaign_id=self.campaign_id,
                video_id=video.id,
                discovery_source=DiscoverySource.CHANNEL_EXPANSION,
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
                data={"video_id": str(video.id), "source": "channel_expansion"},
                message=f"Discovered video from channel expansion"
            )

        return video.id, is_new
