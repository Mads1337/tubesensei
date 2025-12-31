"""
Topic Discovery Service

Main service for managing topic-based video discovery campaigns.
Provides high-level API for creating, running, and managing campaigns.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.models.campaign_video import CampaignVideo, DiscoverySource
from app.models.campaign_channel import CampaignChannel
from app.models.agent_run import AgentRun, AgentType
from app.models.video import Video
from app.models.channel import Channel
from app.agents.base import AgentContext, AgentEvent
from app.agents.coordinator import CoordinatorAgent
from app.utils.rate_limiter import RateLimiter
from app.integrations.youtube_api import YouTubeAPIClient
from app.ai.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class TopicDiscoveryService:
    """
    Service for managing topic-based video discovery campaigns.

    Provides methods for:
    - Creating and configuring campaigns
    - Starting, pausing, resuming, and cancelling campaigns
    - Querying campaign progress and results
    - Exporting discovered videos
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # Campaign CRUD operations

    async def create_campaign(
        self,
        name: str,
        topic: str,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> TopicCampaign:
        """
        Create a new topic discovery campaign.

        Args:
            name: User-friendly campaign name
            topic: The search topic (e.g., "how to make money with YouTube shorts")
            config: Campaign configuration dict
            description: Optional campaign description
            created_by: User ID who created the campaign

        Returns:
            The created TopicCampaign
        """
        # Default configuration
        default_config = {
            "total_video_limit": 3000,
            "per_channel_limit": 5,
            "search_limit": 50,
            "similar_videos_depth": 2,
            "filter_threshold": 0.7,
            "enabled_agents": ["search", "channel_expansion", "topic_filter", "similar_videos"],
        }

        if config:
            default_config.update(config)

        campaign = TopicCampaign(
            name=name,
            topic=topic,
            description=description,
            status=CampaignStatus.DRAFT,
            config=default_config,
            created_by=created_by,
        )

        self.db.add(campaign)
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Created topic campaign: {campaign.id} - {name}")
        return campaign

    async def get_campaign(self, campaign_id: UUID) -> Optional[TopicCampaign]:
        """Get a campaign by ID."""
        result = await self.db.execute(
            select(TopicCampaign).where(TopicCampaign.id == campaign_id)
        )
        return result.scalar_one_or_none()

    async def list_campaigns(
        self,
        status: Optional[CampaignStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TopicCampaign]:
        """List campaigns with optional filtering."""
        query = select(TopicCampaign).order_by(desc(TopicCampaign.created_at))

        if status:
            query = query.where(TopicCampaign.status == status)

        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update_campaign(
        self,
        campaign_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[TopicCampaign]:
        """Update campaign properties (only if in DRAFT status)."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            return None

        if campaign.status != CampaignStatus.DRAFT:
            raise ValueError("Can only update campaigns in DRAFT status")

        if name:
            campaign.name = name
        if description is not None:
            campaign.description = description
        if config:
            current_config = campaign.config or {}
            current_config.update(config)
            campaign.config = current_config

        await self.db.commit()
        await self.db.refresh(campaign)
        return campaign

    async def delete_campaign(self, campaign_id: UUID) -> bool:
        """Delete a campaign (cascade deletes related records)."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            return False

        if campaign.status == CampaignStatus.RUNNING:
            raise ValueError("Cannot delete a running campaign. Cancel it first.")

        await self.db.delete(campaign)
        await self.db.commit()

        logger.info(f"Deleted topic campaign: {campaign_id}")
        return True

    # Campaign lifecycle operations

    async def start_campaign(
        self,
        campaign_id: UUID,
        event_callback: Optional[callable] = None,
    ) -> TopicCampaign:
        """
        Start a campaign.

        This is a synchronous start - the actual discovery runs in background.
        Use run_campaign() for direct execution.

        Args:
            campaign_id: Campaign to start
            event_callback: Optional callback for progress events

        Returns:
            Updated campaign
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.can_start:
            raise ValueError(f"Campaign cannot be started from {campaign.status.value} status")

        campaign.start()
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Started topic campaign: {campaign_id}")
        return campaign

    async def run_campaign(
        self,
        campaign_id: UUID,
        youtube_client: Optional[YouTubeAPIClient] = None,
        llm_manager: Optional[LLMManager] = None,
        event_callback: Optional[callable] = None,
    ) -> TopicCampaign:
        """
        Run a campaign synchronously (blocking).

        Use this for direct execution. For background execution,
        use the Celery task instead.

        Args:
            campaign_id: Campaign to run
            youtube_client: Optional YouTube API client
            llm_manager: Optional LLM manager
            event_callback: Optional callback for progress events

        Returns:
            Updated campaign after completion
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Create agent context
        context = AgentContext(
            campaign_id=campaign_id,
            campaign=campaign,
            db=self.db,
            config=campaign.config or {},
            youtube_rate_limiter=RateLimiter(requests_per_minute=120),
            llm_rate_limiter=RateLimiter(requests_per_minute=60),
            event_callback=event_callback,
        )

        # Create and run coordinator
        coordinator = CoordinatorAgent(
            context=context,
            youtube_client=youtube_client,
            llm_manager=llm_manager,
        )

        result = await coordinator.execute({
            "topic": campaign.topic,
            "resume_from_checkpoint": campaign.status == CampaignStatus.PAUSED,
        })

        # Refresh campaign from DB
        await self.db.refresh(campaign)

        logger.info(
            f"Campaign {campaign_id} completed: "
            f"{campaign.total_videos_relevant} relevant videos found"
        )

        return campaign

    async def pause_campaign(self, campaign_id: UUID) -> TopicCampaign:
        """Pause a running campaign."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.can_pause:
            raise ValueError(f"Campaign cannot be paused from {campaign.status.value} status")

        campaign.pause()
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Paused topic campaign: {campaign_id}")
        return campaign

    async def resume_campaign(self, campaign_id: UUID) -> TopicCampaign:
        """Resume a paused campaign."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.can_resume:
            raise ValueError(f"Campaign cannot be resumed from {campaign.status.value} status")

        campaign.resume()
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Resumed topic campaign: {campaign_id}")
        return campaign

    async def cancel_campaign(self, campaign_id: UUID) -> TopicCampaign:
        """Cancel a running or paused campaign."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.can_cancel:
            raise ValueError(f"Campaign cannot be cancelled from {campaign.status.value} status")

        campaign.cancel()
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Cancelled topic campaign: {campaign_id}")
        return campaign

    # Progress and results

    async def get_progress(self, campaign_id: UUID) -> Dict[str, Any]:
        """Get real-time progress data for a campaign."""
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Get agent run counts
        agent_stats = await self._get_agent_stats(campaign_id)

        # Get video breakdown
        video_stats = await self._get_video_stats(campaign_id)

        return {
            "campaign_id": str(campaign_id),
            "status": campaign.status.value,
            "progress_percent": campaign.progress_percent,
            "videos_discovered": campaign.total_videos_discovered,
            "videos_relevant": campaign.total_videos_relevant,
            "videos_filtered": campaign.total_videos_filtered,
            "channels_explored": campaign.total_channels_explored,
            "transcripts_extracted": campaign.total_transcripts_extracted,
            "api_calls": campaign.api_calls_made,
            "llm_calls": campaign.llm_calls_made,
            "error_count": campaign.error_count,
            "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
            "estimated_completion": campaign.estimated_completion_at.isoformat() if campaign.estimated_completion_at else None,
            "duration_seconds": campaign.duration_seconds,
            "filter_acceptance_rate": campaign.filter_acceptance_rate,
            "agent_stats": agent_stats,
            "video_stats": video_stats,
        }

    async def _get_agent_stats(self, campaign_id: UUID) -> Dict[str, Any]:
        """Get statistics by agent type."""
        result = await self.db.execute(
            select(
                AgentRun.agent_type,
                func.count(AgentRun.id).label("total_runs"),
                func.sum(AgentRun.items_processed).label("items_processed"),
                func.sum(AgentRun.items_produced).label("items_produced"),
                func.sum(AgentRun.api_calls_made).label("api_calls"),
            )
            .where(AgentRun.campaign_id == campaign_id)
            .group_by(AgentRun.agent_type)
        )

        stats = {}
        for row in result.all():
            stats[row.agent_type.value] = {
                "total_runs": row.total_runs,
                "items_processed": row.items_processed or 0,
                "items_produced": row.items_produced or 0,
                "api_calls": row.api_calls or 0,
            }

        return stats

    async def _get_video_stats(self, campaign_id: UUID) -> Dict[str, Any]:
        """Get video statistics by discovery source."""
        result = await self.db.execute(
            select(
                CampaignVideo.discovery_source,
                func.count(CampaignVideo.id).label("total"),
                func.sum(func.cast(CampaignVideo.is_topic_relevant == True, Integer)).label("relevant"),
            )
            .where(CampaignVideo.campaign_id == campaign_id)
            .group_by(CampaignVideo.discovery_source)
        )

        # Import Integer for casting
        from sqlalchemy import Integer

        stats = {}
        for row in result.all():
            stats[row.discovery_source.value] = {
                "total": row.total,
                "relevant": row.relevant or 0,
            }

        return stats

    async def get_videos(
        self,
        campaign_id: UUID,
        relevant_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get videos discovered by a campaign."""
        query = (
            select(CampaignVideo, Video)
            .join(Video, CampaignVideo.video_id == Video.id)
            .where(CampaignVideo.campaign_id == campaign_id)
            .order_by(desc(CampaignVideo.discovered_at))
        )

        if relevant_only:
            query = query.where(CampaignVideo.is_topic_relevant == True)

        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        rows = result.all()

        videos = []
        for cv, video in rows:
            videos.append({
                **cv.get_summary(),
                "video": {
                    "id": str(video.id),
                    "youtube_id": video.youtube_video_id,
                    "title": video.title,
                    "description": (video.description or "")[:200],
                    "view_count": video.view_count,
                    "duration_seconds": video.duration_seconds,
                    "published_at": video.published_at.isoformat() if video.published_at else None,
                    "has_captions": video.has_captions,
                },
            })

        return videos

    async def get_channels(
        self,
        campaign_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get channels discovered by a campaign."""
        query = (
            select(CampaignChannel, Channel)
            .join(Channel, CampaignChannel.channel_id == Channel.id)
            .where(CampaignChannel.campaign_id == campaign_id)
            .order_by(desc(CampaignChannel.videos_relevant))
        )

        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        rows = result.all()

        channels = []
        for cc, channel in rows:
            channels.append({
                **cc.get_summary(),
                "channel": {
                    "id": str(channel.id),
                    "youtube_id": channel.youtube_channel_id,
                    "name": channel.name,
                    "subscriber_count": channel.subscriber_count,
                    "video_count": channel.video_count,
                },
            })

        return channels

    async def get_agent_runs(
        self,
        campaign_id: UUID,
        agent_type: Optional[AgentType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get agent execution history for a campaign."""
        query = (
            select(AgentRun)
            .where(AgentRun.campaign_id == campaign_id)
            .order_by(desc(AgentRun.started_at))
        )

        if agent_type:
            query = query.where(AgentRun.agent_type == agent_type)

        query = query.limit(limit)

        result = await self.db.execute(query)
        runs = result.scalars().all()

        return [run.get_summary() for run in runs]

    # Export

    async def export_results(
        self,
        campaign_id: UUID,
        format: str = "json",
        relevant_only: bool = True,
    ) -> bytes:
        """
        Export campaign results.

        Args:
            campaign_id: Campaign to export
            format: Export format ("json" or "csv")
            relevant_only: Only export relevant videos

        Returns:
            Exported data as bytes
        """
        import json
        import csv
        import io

        # Get campaign
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Get videos
        videos = await self.get_videos(
            campaign_id,
            relevant_only=relevant_only,
            limit=10000,  # Export all
        )

        if format == "json":
            export_data = {
                "campaign": campaign.get_summary(),
                "videos": videos,
                "exported_at": datetime.utcnow().isoformat(),
            }
            return json.dumps(export_data, indent=2).encode("utf-8")

        elif format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "video_id",
                "youtube_id",
                "title",
                "channel",
                "view_count",
                "duration_seconds",
                "relevance_score",
                "discovery_source",
                "discovered_at",
            ])

            # Rows
            for v in videos:
                writer.writerow([
                    v["video"]["id"],
                    v["video"]["youtube_id"],
                    v["video"]["title"],
                    v.get("channel_name", ""),
                    v["video"]["view_count"],
                    v["video"]["duration_seconds"],
                    v.get("relevance_score", ""),
                    v["discovery_source"],
                    v["discovered_at"],
                ])

            return output.getvalue().encode("utf-8")

        else:
            raise ValueError(f"Unsupported format: {format}")
