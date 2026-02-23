"""
Topic Discovery Service

Main service for managing topic-based video discovery campaigns.
Provides high-level API for creating, running, and managing campaigns.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, func, desc, Integer
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
from app.agents.transcription_agent import TranscriptionAgent
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

    async def count_campaigns(
        self,
        status: Optional[CampaignStatus] = None,
    ) -> int:
        """Count campaigns with optional status filtering."""
        query = select(func.count(TopicCampaign.id))

        if status:
            query = query.where(TopicCampaign.status == status)

        result = await self.db.execute(query)
        return result.scalar() or 0

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

    async def run_transcription(
        self,
        campaign_id: UUID,
        event_callback: Optional[callable] = None,
    ) -> TopicCampaign:
        """
        Run the transcription agent for a campaign synchronously.
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Set stage to transcription in metadata so we know what we are doing
        if not campaign.campaign_metadata:
            campaign.campaign_metadata = {}
        campaign.campaign_metadata["stage"] = "transcription"
        
        # Determine if we are starting or resuming
        # Taking over the status to RUNNING
        if campaign.status != CampaignStatus.RUNNING:
            campaign.status = CampaignStatus.RUNNING
            await self.db.commit()

        context = AgentContext(
            campaign_id=campaign_id,
            campaign=campaign,
            db=self.db,
            config=campaign.config or {},
            # These rate limiters might not be heavily used by transcription but needed for context init
            youtube_rate_limiter=RateLimiter(requests_per_minute=120),
            llm_rate_limiter=RateLimiter(requests_per_minute=60),
            event_callback=event_callback,
        )

        # Initialize agent
        agent = TranscriptionAgent(context)

        logger.info(f"Starting transcription for campaign {campaign_id}")
        
        result = await agent.execute({
            "batch_size": 5
        })

        # After finish, what status?
        # If successfully finished all transcriptions, we can stay COMPLETED
        # or go back to whatever state.
        # Typically we just mark campaign COMPLETED if everything is done.
        
        if result.success:
            campaign.status = CampaignStatus.COMPLETED
            await self.db.commit()

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
        from app.celery_app import celery_app

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.can_cancel:
            raise ValueError(f"Campaign cannot be cancelled from {campaign.status.value} status")

        # Terminate the Celery task if running
        if campaign.celery_task_id:
            try:
                celery_app.control.revoke(campaign.celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {campaign.celery_task_id} for campaign {campaign_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task: {e}")
            campaign.celery_task_id = None

        campaign.cancel()
        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Cancelled topic campaign: {campaign_id}")
        return campaign

    # Transcription stats

    async def get_transcription_stats(self, campaign_id: UUID) -> Dict[str, Any]:
        """
        Get detailed transcription statistics for a campaign.

        Args:
            campaign_id: Campaign to get stats for

        Returns:
            Dict with transcription progress stats including:
            - total_relevant: Total relevant videos to process
            - extracted: Videos with transcripts extracted
            - failed: Videos where extraction failed
            - pending: Videos awaiting extraction
            - recent_videos: Last 10 processed videos with details
        """
        from sqlalchemy import case

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Query for detailed transcription stats
        result = await self.db.execute(
            select(
                func.count(CampaignVideo.id).label('total_relevant'),
                func.count(case((CampaignVideo.transcript_extracted == True, 1))).label('extracted'),
                func.count(case((
                    (CampaignVideo.transcript_extracted == False) &
                    (CampaignVideo.transcript_extracted_at != None), 1
                ))).label('failed'),
                func.count(case((
                    (CampaignVideo.transcript_extracted == False) &
                    (CampaignVideo.transcript_extracted_at == None), 1
                ))).label('pending'),
            )
            .where(
                CampaignVideo.campaign_id == campaign_id,
                CampaignVideo.is_topic_relevant == True,
            )
        )
        row = result.first()

        # Get recently processed videos (last 10)
        recent_result = await self.db.execute(
            select(CampaignVideo, Video)
            .join(Video, CampaignVideo.video_id == Video.id)
            .where(
                CampaignVideo.campaign_id == campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted_at != None,
            )
            .order_by(desc(CampaignVideo.transcript_extracted_at))
            .limit(10)
        )

        recent_videos = []
        for cv, video in recent_result.all():
            # Get word count from transcript if available
            word_count = None
            if cv.transcript_extracted:
                from app.models.transcript import Transcript
                transcript_result = await self.db.execute(
                    select(Transcript)
                    .where(Transcript.video_id == video.id)
                    .order_by(Transcript.created_at.desc())
                    .limit(1)
                )
                transcript = transcript_result.scalar_one_or_none()
                if transcript:
                    word_count = transcript.word_count or (
                        len(transcript.content.split()) if transcript.content else None
                    )

            recent_videos.append({
                "video_id": str(video.id),
                "youtube_id": video.youtube_video_id,
                "title": video.title,
                "extracted_at": cv.transcript_extracted_at.isoformat() if cv.transcript_extracted_at else None,
                "success": cv.transcript_extracted,
                "word_count": word_count,
                "error": None if cv.transcript_extracted else "Extraction failed",
            })

        total_relevant = row.total_relevant if row else 0
        extracted = row.extracted if row else 0

        return {
            "campaign_id": str(campaign_id),
            "stage": campaign.campaign_metadata.get("stage") if campaign.campaign_metadata else None,
            "total_relevant": total_relevant,
            "extracted": extracted,
            "failed": row.failed if row else 0,
            "pending": row.pending if row else 0,
            "progress_percent": (extracted / total_relevant * 100) if total_relevant > 0 else 0,
            "recent_videos": recent_videos,
            "is_active": (
                campaign.status == CampaignStatus.RUNNING and
                campaign.campaign_metadata is not None and
                campaign.campaign_metadata.get("stage") == "transcription"
            ),
            "is_paused": campaign.status == CampaignStatus.PAUSED,
            "api_calls": campaign.api_calls_made or 0,
        }

    # Stale campaign detection and recovery

    async def get_stale_campaigns(self, stale_minutes: int = 10) -> List[TopicCampaign]:
        """
        Get campaigns that are RUNNING but haven't had a heartbeat in X minutes.

        Args:
            stale_minutes: Minutes without heartbeat to consider stale

        Returns:
            List of stale campaigns
        """
        from datetime import timedelta

        stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)

        # Get RUNNING campaigns with old or missing heartbeat
        result = await self.db.execute(
            select(TopicCampaign)
            .where(TopicCampaign.status == CampaignStatus.RUNNING)
            .where(
                (TopicCampaign.last_heartbeat_at == None) |
                (TopicCampaign.last_heartbeat_at < stale_threshold)
            )
            .order_by(desc(TopicCampaign.started_at))
        )

        return list(result.scalars().all())

    async def check_campaign_task_alive(self, campaign_id: UUID) -> bool:
        """
        Check if the Celery task for this campaign is still active.

        Args:
            campaign_id: Campaign to check

        Returns:
            True if task is still active, False otherwise
        """
        from app.celery_app import celery_app

        campaign = await self.get_campaign(campaign_id)
        if not campaign or not campaign.celery_task_id:
            return False

        try:
            result = celery_app.AsyncResult(campaign.celery_task_id)
            # Check if task is in an active state
            return result.status in ['PENDING', 'STARTED', 'RETRY']
        except Exception as e:
            logger.warning(f"Failed to check task status: {e}")
            return False

    async def recover_stale_campaign(
        self,
        campaign_id: UUID,
        action: str = "fail",
    ) -> TopicCampaign:
        """
        Recover a stale campaign by marking it failed or restarting it.

        Args:
            campaign_id: Campaign to recover
            action: "fail" to mark as failed, "restart" to reset and restart

        Returns:
            Updated campaign
        """
        from app.celery_app import celery_app

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.status != CampaignStatus.RUNNING:
            raise ValueError(f"Campaign is not in RUNNING status, current: {campaign.status.value}")

        # Check if it's actually stale
        if not campaign.is_stale:
            raise ValueError("Campaign is not stale - it has a recent heartbeat")

        # Terminate any zombie Celery task
        if campaign.celery_task_id:
            try:
                celery_app.control.revoke(campaign.celery_task_id, terminate=True)
                logger.info(f"Revoked zombie Celery task {campaign.celery_task_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task: {e}")
            campaign.celery_task_id = None

        if action == "fail":
            campaign.fail("Campaign stalled - worker task died unexpectedly")
            logger.info(f"Marked stale campaign as failed: {campaign_id}")
        elif action == "restart":
            # Reset to DRAFT so it can be started again
            campaign.status = CampaignStatus.DRAFT
            campaign.error_message = None
            campaign.error_count = 0
            campaign.last_heartbeat_at = None
            # Keep progress data intact for resume
            logger.info(f"Reset stale campaign for restart: {campaign_id}")
        else:
            raise ValueError(f"Invalid action: {action}. Use 'fail' or 'restart'")

        await self.db.commit()
        await self.db.refresh(campaign)

        return campaign

    async def force_stop_campaign(self, campaign_id: UUID) -> TopicCampaign:
        """
        Force stop a campaign regardless of current state.

        This immediately terminates any running Celery task and marks the campaign
        as failed. Use this for stuck campaigns that can't be cancelled normally.

        Args:
            campaign_id: Campaign to force stop

        Returns:
            Updated campaign marked as failed
        """
        from app.celery_app import celery_app

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Terminate the Celery task if present
        if campaign.celery_task_id:
            try:
                celery_app.control.revoke(campaign.celery_task_id, terminate=True)
                logger.info(f"Force-terminated Celery task {campaign.celery_task_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task: {e}")
            campaign.celery_task_id = None

        # Mark as failed regardless of current state
        campaign.status = CampaignStatus.FAILED
        campaign.error_message = "Manually force-stopped by user"
        campaign.completed_at = datetime.now(timezone.utc)
        if campaign.started_at:
            campaign.execution_time_seconds = (campaign.completed_at - campaign.started_at).total_seconds()

        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Force-stopped campaign: {campaign_id}")
        return campaign

    async def retry_campaign(self, campaign_id: UUID) -> TopicCampaign:
        """
        Retry a failed campaign.

        Resets error state and sets status back to DRAFT so it can be started again.

        Args:
            campaign_id: Campaign to retry

        Returns:
            Updated campaign ready to be started
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.status != CampaignStatus.FAILED:
            raise ValueError(f"Can only retry campaigns in FAILED status, current status: {campaign.status.value}")

        # Reset error state
        campaign.status = CampaignStatus.DRAFT
        campaign.error_message = None
        campaign.error_count = 0
        campaign.completed_at = None
        campaign.execution_time_seconds = None

        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Reset failed campaign for retry: {campaign_id}")
        return campaign

    async def retry_transcription(self, campaign_id: UUID) -> TopicCampaign:
        """
        Retry transcription for a failed campaign.

        This resets the error state while keeping the transcription stage,
        allowing the transcription process to be restarted.

        Args:
            campaign_id: Campaign to retry transcription for

        Returns:
            Updated campaign ready for transcription restart
        """
        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.status != CampaignStatus.FAILED:
            raise ValueError(f"Can only retry FAILED campaigns, current status: {campaign.status.value}")

        # Verify it's in transcription stage
        if not campaign.campaign_metadata or campaign.campaign_metadata.get('stage') != 'transcription':
            raise ValueError("Campaign is not in transcription stage. Use regular retry for discovery campaigns.")

        # Reset error state but keep transcription stage
        campaign.status = CampaignStatus.RUNNING
        campaign.error_message = None
        campaign.error_count = 0
        campaign.completed_at = None
        campaign.execution_time_seconds = None
        # Send initial heartbeat
        campaign.heartbeat()

        await self.db.commit()
        await self.db.refresh(campaign)

        logger.info(f"Reset failed transcription campaign for retry: {campaign_id}")
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
    ) -> List[AgentRun]:
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

        return runs

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
                "exported_at": datetime.now(timezone.utc).isoformat(),
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

    # Idea extraction

    async def run_idea_extraction(
        self,
        campaign_id: UUID,
        event_callback: Optional[callable] = None,
    ) -> TopicCampaign:
        """
        Run the idea extraction agent for a campaign synchronously.
        """
        from app.agents.idea_extraction_agent import IdeaExtractionAgent
        from app.ai.llm_manager import LLMManager as _LLMManager

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Set stage
        if not campaign.campaign_metadata:
            campaign.campaign_metadata = {}
        campaign.campaign_metadata["stage"] = "idea_extraction"

        if campaign.status != CampaignStatus.RUNNING:
            campaign.status = CampaignStatus.RUNNING
            await self.db.commit()

        context = AgentContext(
            campaign_id=campaign_id,
            campaign=campaign,
            db=self.db,
            config=campaign.config or {},
            youtube_rate_limiter=RateLimiter(requests_per_minute=120),
            llm_rate_limiter=RateLimiter(requests_per_minute=60),
            event_callback=event_callback,
        )

        llm_manager = _LLMManager()
        agent = IdeaExtractionAgent(context, llm_manager=llm_manager)

        logger.info(f"Starting idea extraction for campaign {campaign_id}")

        result = await agent.execute({"batch_size": 5})

        if result.success:
            campaign.status = CampaignStatus.COMPLETED
            await self.db.commit()

        return campaign

    async def get_ideas_stats(self, campaign_id: UUID) -> Dict[str, Any]:
        """
        Get idea extraction statistics for a campaign.

        Returns:
            Dict with ideas stats including totals by category, average confidence, etc.
        """
        from sqlalchemy import case
        from app.models.idea import Idea, IdeaStatus

        campaign = await self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Count videos with/without ideas
        ideas_stats_result = await self.db.execute(
            select(
                func.count(CampaignVideo.id).label('total_with_transcripts'),
                func.count(case((CampaignVideo.ideas_extracted == True, 1))).label('ideas_extracted'),
                func.count(case((
                    (CampaignVideo.transcript_extracted == True) &
                    (CampaignVideo.ideas_extracted == False), 1
                ))).label('pending'),
            )
            .where(
                CampaignVideo.campaign_id == campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == True,
            )
        )
        row = ideas_stats_result.first()

        total_with_transcripts = row.total_with_transcripts if row else 0
        ideas_extracted_count = row.ideas_extracted if row else 0
        pending = row.pending if row else 0

        # Get idea totals - join through Video → CampaignVideo
        ideas_count_result = await self.db.execute(
            select(func.count(Idea.id))
            .join(Video, Idea.video_id == Video.id)
            .join(CampaignVideo, CampaignVideo.video_id == Video.id)
            .where(CampaignVideo.campaign_id == campaign_id)
        )
        total_ideas = ideas_count_result.scalar() or 0

        # Get ideas by category
        category_result = await self.db.execute(
            select(Idea.category, func.count(Idea.id).label('count'))
            .join(Video, Idea.video_id == Video.id)
            .join(CampaignVideo, CampaignVideo.video_id == Video.id)
            .where(CampaignVideo.campaign_id == campaign_id)
            .group_by(Idea.category)
            .order_by(desc(func.count(Idea.id)))
        )
        categories = {r.category or "Uncategorized": r.count for r in category_result.all()}

        # Get average confidence
        avg_confidence_result = await self.db.execute(
            select(func.avg(Idea.confidence_score))
            .join(Video, Idea.video_id == Video.id)
            .join(CampaignVideo, CampaignVideo.video_id == Video.id)
            .where(CampaignVideo.campaign_id == campaign_id)
        )
        avg_confidence = avg_confidence_result.scalar() or 0

        # Get recent ideas (last 20)
        recent_result = await self.db.execute(
            select(Idea, Video)
            .join(Video, Idea.video_id == Video.id)
            .join(CampaignVideo, CampaignVideo.video_id == Video.id)
            .where(CampaignVideo.campaign_id == campaign_id)
            .order_by(desc(Idea.created_at))
            .limit(20)
        )

        recent_ideas = []
        for idea, video in recent_result.all():
            recent_ideas.append({
                "id": str(idea.id),
                "title": idea.title,
                "description": idea.description[:200] if idea.description else "",
                "category": idea.category or "Uncategorized",
                "confidence_score": idea.confidence_score,
                "status": idea.status.value,
                "video_id": str(video.id),
                "video_title": video.title,
                "youtube_video_id": video.youtube_video_id,
                "created_at": idea.created_at.isoformat() if idea.created_at else None,
            })

        progress_percent = (ideas_extracted_count / total_with_transcripts * 100) if total_with_transcripts > 0 else 0

        return {
            "campaign_id": str(campaign_id),
            "total_with_transcripts": total_with_transcripts,
            "videos_processed": ideas_extracted_count,
            "pending": pending,
            "total_ideas": total_ideas,
            "categories": categories,
            "avg_confidence": round(float(avg_confidence), 3),
            "progress_percent": progress_percent,
            "recent_ideas": recent_ideas,
            "is_active": (
                campaign.status == CampaignStatus.RUNNING and
                campaign.campaign_metadata is not None and
                campaign.campaign_metadata.get("stage") == "idea_extraction"
            ),
        }
