"""
Similar Videos Agent

Discovers related videos using YouTube's Related Videos API.
Takes videos that passed the topic filter and finds similar content.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select

from app.agents.base import BaseAgent, AgentResult, AgentContext, AgentEventType
from app.models.agent_run import AgentType
from app.models.video import Video, VideoStatus
from app.models.channel import Channel, ChannelStatus
from app.models.campaign_video import CampaignVideo, DiscoverySource
from app.models.campaign_channel import CampaignChannel
from app.integrations.youtube_api import YouTubeAPIClient

logger = logging.getLogger(__name__)


class SimilarVideosAgent(BaseAgent):
    """
    Agent that discovers similar/related videos via YouTube API.

    Takes relevant videos as input and finds similar content.
    Respects depth limits to prevent infinite expansion.

    Input:
        video_ids: List[UUID] - Video IDs to find similar videos for
        depth: int - Current recursion depth (default: 0)
        max_per_video: int - Maximum similar videos per source video (default: 10)

    Output:
        discovered_video_ids: List[UUID] - IDs of newly discovered videos
        discovered_channel_ids: List[UUID] - IDs of newly discovered channels
        new_videos_count: int - Number of new videos added
    """

    agent_type = AgentType.SIMILAR_VIDEOS

    def __init__(self, context: AgentContext, youtube_client: Optional[YouTubeAPIClient] = None):
        super().__init__(context)
        self.youtube_client = youtube_client

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Discover similar videos.

        Args:
            input_data: Contains 'video_ids', optional 'depth' and 'max_per_video'

        Returns:
            AgentResult with discovered video IDs
        """
        video_ids = input_data.get("video_ids", [])
        if video_ids and isinstance(video_ids[0], str):
            video_ids = [UUID(vid) for vid in video_ids]

        depth = input_data.get("depth", 0)
        max_depth = self.config.get("similar_videos_depth", 2)
        max_per_video = input_data.get("max_per_video", 10)

        logger.info(
            f"SimilarVideosAgent: Finding similar videos for {len(video_ids)} videos "
            f"(depth {depth}/{max_depth})"
        )

        discovered_video_ids: List[UUID] = []
        discovered_channel_ids: List[UUID] = []
        new_videos = 0

        # Check depth limit
        if depth >= max_depth:
            logger.info(f"SimilarVideosAgent: Max depth {max_depth} reached, skipping")
            return self._build_result(success=True, data={
                "discovered_video_ids": [],
                "discovered_channel_ids": [],
                "new_videos_count": 0,
                "skipped_depth_limit": True,
            })

        if not video_ids:
            return self._build_result(success=True, data={
                "discovered_video_ids": [],
                "discovered_channel_ids": [],
                "new_videos_count": 0,
            })

        try:
            # Initialize YouTube client
            youtube = self.youtube_client or YouTubeAPIClient()

            total_source_videos = len(video_ids)

            for idx, video_id in enumerate(video_ids):
                # Check if we should stop
                if await self.check_should_stop():
                    logger.info("SimilarVideosAgent: Stopping due to limit reached or cancellation")
                    break

                self.update_progress(
                    ((idx + 1) / total_source_videos) * 100,
                    current_item=f"Video {idx + 1}/{total_source_videos}"
                )

                try:
                    # Get the source video's YouTube ID
                    result = await self.db.execute(
                        select(Video).where(Video.id == video_id)
                    )
                    source_video = result.scalar_one_or_none()

                    if not source_video:
                        continue

                    youtube_video_id = source_video.youtube_video_id

                    # Get related videos
                    async with self.context.youtube_rate_limiter.acquire():
                        related_videos = await youtube.get_related_videos(
                            video_id=youtube_video_id,
                            max_results=max_per_video
                        )
                        self.increment_api_calls()

                    if not related_videos:
                        continue

                    # Get details for related videos
                    related_ids = [v["video_id"] for v in related_videos]

                    async with self.context.youtube_rate_limiter.acquire():
                        video_details = await youtube.get_video_details(related_ids[:50])
                        self.increment_api_calls()

                    # Process each related video
                    for video_data in video_details:
                        # Check campaign limits
                        if await self.check_should_stop():
                            break

                        vid_id, ch_id, is_new = await self._process_video(
                            video_data, source_video.id, depth + 1
                        )

                        if vid_id:
                            discovered_video_ids.append(vid_id)
                            self.increment_produced()
                            if is_new:
                                new_videos += 1

                        if ch_id and ch_id not in discovered_channel_ids:
                            discovered_channel_ids.append(ch_id)

                except Exception as e:
                    error_msg = f"Error processing similar videos for {video_id}: {e}"
                    logger.error(error_msg)
                    self.add_error(error_msg)

                self.increment_processed()

            # Commit all changes
            await self.db.commit()

            logger.info(
                f"SimilarVideosAgent: Discovered {len(discovered_video_ids)} videos "
                f"from {len(discovered_channel_ids)} channels ({new_videos} new)"
            )

            return self._build_result(success=True, data={
                "discovered_video_ids": [str(vid) for vid in discovered_video_ids],
                "discovered_channel_ids": [str(cid) for cid in discovered_channel_ids],
                "new_videos_count": new_videos,
                "depth": depth,
            })

        except Exception as e:
            logger.exception(f"SimilarVideosAgent failed: {e}")
            return self._build_result(success=False, data={
                "discovered_video_ids": [str(vid) for vid in discovered_video_ids],
                "discovered_channel_ids": [str(cid) for cid in discovered_channel_ids],
                "new_videos_count": new_videos,
                "error": str(e),
            })

    async def _process_video(
        self,
        video_data: Dict[str, Any],
        source_video_id: UUID,
        depth: int
    ) -> tuple[Optional[UUID], Optional[UUID], bool]:
        """
        Process a single related video.

        Returns:
            Tuple of (video_id, channel_id, is_new)
        """
        youtube_video_id = video_data.get("video_id")
        youtube_channel_id = video_data.get("channel_id")

        if not youtube_video_id or not youtube_channel_id:
            return None, None, False

        is_new = False

        # Check if video already in this campaign
        existing_cv = await self.db.execute(
            select(CampaignVideo).join(Video).where(
                CampaignVideo.campaign_id == self.campaign_id,
                Video.youtube_video_id == youtube_video_id
            )
        )
        if existing_cv.scalar_one_or_none():
            # Already discovered in this campaign
            return None, None, False

        # Get or create channel
        channel_result = await self.db.execute(
            select(Channel).where(Channel.youtube_channel_id == youtube_channel_id)
        )
        channel = channel_result.scalar_one_or_none()

        if not channel:
            channel = Channel(
                youtube_channel_id=youtube_channel_id,
                name=video_data.get("channel_title", "Unknown Channel"),
                status=ChannelStatus.ACTIVE,
                channel_metadata={
                    "discovered_via": "topic_campaign_similar_videos",
                    "campaign_id": str(self.campaign_id),
                }
            )
            self.db.add(channel)
            await self.db.flush()

        # Get or create video
        video_result = await self.db.execute(
            select(Video).where(Video.youtube_video_id == youtube_video_id)
        )
        video = video_result.scalar_one_or_none()

        if not video:
            # Parse published_at datetime string
            published_at_str = video_data.get("published_at")
            published_at = None
            if published_at_str:
                try:
                    published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass

            video = Video(
                youtube_video_id=youtube_video_id,
                channel_id=channel.id,
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
                    "discovered_via": "topic_campaign_similar_videos",
                    "campaign_id": str(self.campaign_id),
                    "source_video_id": str(source_video_id),
                }
            )
            self.db.add(video)
            await self.db.flush()
            is_new = True

            # Update campaign stats
            self.context.campaign.increment_discovered()

        # Create CampaignVideo junction
        campaign_video = CampaignVideo(
            campaign_id=self.campaign_id,
            video_id=video.id,
            discovery_source=DiscoverySource.SIMILAR_VIDEOS,
            source_video_id=source_video_id,
            source_channel_id=channel.id,
            agent_run_id=self._agent_run.id if self._agent_run else None,
            discovery_depth=depth,
            discovery_order=self._items_produced,
        )
        self.db.add(campaign_video)

        # Create CampaignChannel junction if not exists
        cc_result = await self.db.execute(
            select(CampaignChannel).where(
                CampaignChannel.campaign_id == self.campaign_id,
                CampaignChannel.channel_id == channel.id
            )
        )
        if not cc_result.scalar_one_or_none():
            campaign_channel = CampaignChannel(
                campaign_id=self.campaign_id,
                channel_id=channel.id,
                discovery_source=DiscoverySource.SIMILAR_VIDEOS,
                source_video_id=source_video_id,
                videos_limit=self.per_channel_limit,
            )
            self.db.add(campaign_channel)

        await self.db.flush()

        # Emit discovery event
        self._emit_event(
            AgentEventType.ITEM_DISCOVERED,
            data={
                "video_id": str(video.id),
                "source": "similar_videos",
                "depth": depth,
            },
            message=f"Discovered similar video (depth {depth})"
        )

        return video.id, channel.id, is_new
