"""
Coordinator Agent

Orchestrates the entire topic discovery pipeline by managing sub-agents.
Controls the discovery loop, monitors limits, and handles pause/resume/cancel.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy import select, func

from app.agents.base import BaseAgent, AgentResult, AgentContext, AgentEventType
from app.agents.search_agent import SearchAgent
from app.agents.channel_expansion_agent import ChannelExpansionAgent
from app.agents.topic_filter_agent import TopicFilterAgent
from app.agents.similar_videos_agent import SimilarVideosAgent
from app.models.agent_run import AgentType
from app.models.campaign_video import CampaignVideo
from app.models.campaign_channel import CampaignChannel
from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.integrations.youtube_api import YouTubeAPIClient
from app.ai.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Main orchestrator for topic-based video discovery.

    Manages the discovery pipeline:
    1. Search Agent: Find initial videos
    2. Channel Expansion Agent: Get more videos from discovered channels
    3. Topic Filter Agent: AI-filter videos for relevance
    4. Similar Videos Agent: Find related videos
    5. Repeat until limits reached

    Input:
        topic: str - The topic to discover videos for (optional, uses campaign topic)
        resume_from_checkpoint: bool - Whether to resume from saved checkpoint

    Output:
        total_videos_discovered: int
        total_videos_relevant: int
        total_channels_explored: int
        iterations_completed: int
    """

    agent_type = AgentType.COORDINATOR

    def __init__(
        self,
        context: AgentContext,
        youtube_client: Optional[YouTubeAPIClient] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        super().__init__(context)
        self.youtube_client = youtube_client
        self.llm_manager = llm_manager

        # Sub-agent instances
        self._search_agent: Optional[SearchAgent] = None
        self._channel_expansion_agent: Optional[ChannelExpansionAgent] = None
        self._topic_filter_agent: Optional[TopicFilterAgent] = None
        self._similar_videos_agent: Optional[SimilarVideosAgent] = None

        # State tracking
        self._iteration = 0
        self._processed_channels: Set[UUID] = set()
        self._videos_pending_filter: List[UUID] = []

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the full discovery pipeline.

        Args:
            input_data: Contains optional 'topic' and 'resume_from_checkpoint'

        Returns:
            AgentResult with discovery statistics
        """
        topic = input_data.get("topic", self.context.campaign.topic)
        resume = input_data.get("resume_from_checkpoint", False)

        logger.info(f"CoordinatorAgent: Starting discovery for topic '{topic[:50]}...'")

        # Update campaign status
        campaign = self.context.campaign
        if campaign.status == CampaignStatus.DRAFT:
            campaign.start()
        elif resume and campaign.status == CampaignStatus.PAUSED:
            campaign.resume()
            # Restore state from checkpoint
            await self._restore_checkpoint()

        await self.db.flush()

        try:
            # Initialize sub-agents with shared context
            child_context = AgentContext(
                campaign_id=self.campaign_id,
                campaign=campaign,
                db=self.db,
                config=self.config,
                youtube_rate_limiter=self.context.youtube_rate_limiter,
                llm_rate_limiter=self.context.llm_rate_limiter,
                event_callback=self.context.event_callback,
                cancelled=self.context.cancelled,
                parent_run_id=self._agent_run.id if self._agent_run else None,
                shared_state=self.context.shared_state,
            )

            self._search_agent = SearchAgent(child_context, self.youtube_client)
            self._channel_expansion_agent = ChannelExpansionAgent(child_context, self.youtube_client)
            self._topic_filter_agent = TopicFilterAgent(child_context, self.llm_manager)
            self._similar_videos_agent = SimilarVideosAgent(child_context, self.youtube_client)

            # Run discovery loop
            await self._run_discovery_loop(topic)

            # Mark campaign complete if we've finished
            if not self.context.is_cancelled():
                campaign.complete()
            else:
                campaign.pause()
                await self._save_checkpoint()

            await self.db.commit()

            return self._build_result(success=True, data={
                "total_videos_discovered": campaign.total_videos_discovered,
                "total_videos_relevant": campaign.total_videos_relevant,
                "total_channels_explored": campaign.total_channels_explored,
                "iterations_completed": self._iteration,
            })

        except Exception as e:
            logger.exception(f"CoordinatorAgent failed: {e}")
            campaign.fail(str(e))
            await self.db.commit()

            return self._build_result(success=False, data={
                "total_videos_discovered": campaign.total_videos_discovered,
                "total_videos_relevant": campaign.total_videos_relevant,
                "total_channels_explored": campaign.total_channels_explored,
                "iterations_completed": self._iteration,
                "error": str(e),
            })

    async def _run_discovery_loop(self, topic: str) -> None:
        """
        Run the main discovery loop.

        Loop continues until:
        - Video limit is reached
        - No new videos are being discovered
        - Campaign is cancelled/paused
        """
        max_iterations = 50  # Safety limit
        no_progress_count = 0
        max_no_progress = 3  # Stop after 3 iterations with no progress

        while self._iteration < max_iterations:
            self._iteration += 1

            logger.info(f"CoordinatorAgent: Starting iteration {self._iteration}")

            # Check if we should stop
            if await self.check_should_stop():
                logger.info("CoordinatorAgent: Stopping - limit reached or cancelled")
                break

            iteration_start_relevant = self.context.campaign.total_videos_relevant

            # Step 1: Search (only on first iteration)
            if self._iteration == 1:
                await self._run_search(topic)

            # Step 2: Expand channels
            await self._run_channel_expansion()

            # Step 3: Filter pending videos
            await self._run_topic_filter(topic)

            # Step 4: Find similar videos for relevant ones
            await self._run_similar_videos()

            # Check progress
            new_relevant = self.context.campaign.total_videos_relevant - iteration_start_relevant

            if new_relevant == 0:
                no_progress_count += 1
                logger.info(f"CoordinatorAgent: No new relevant videos in iteration {self._iteration}")
                if no_progress_count >= max_no_progress:
                    logger.info(f"CoordinatorAgent: Stopping after {max_no_progress} iterations with no progress")
                    break
            else:
                no_progress_count = 0
                logger.info(f"CoordinatorAgent: Found {new_relevant} new relevant videos in iteration {self._iteration}")

            # Update progress
            self.update_progress(
                min((self.context.campaign.total_videos_relevant / self.total_video_limit) * 100, 99),
                current_item=f"Iteration {self._iteration}"
            )

            # Save checkpoint after each iteration
            await self._save_checkpoint()

    async def _run_search(self, topic: str) -> None:
        """Run the search agent."""
        if not self._search_agent:
            return

        logger.info("CoordinatorAgent: Running SearchAgent")

        result = await self._search_agent.execute({
            "topic": topic,
            "max_results": self.config.get("search_limit", 50),
        })

        if result.success:
            # Add discovered videos to pending filter list
            video_ids = [UUID(vid) for vid in result.data.get("video_ids", [])]
            self._videos_pending_filter.extend(video_ids)

            self.increment_api_calls(result.api_calls_made)

    async def _run_channel_expansion(self) -> None:
        """Run channel expansion for unexpanded channels."""
        if not self._channel_expansion_agent:
            return

        # Get unexpanded channels
        result = await self.db.execute(
            select(CampaignChannel).where(
                CampaignChannel.campaign_id == self.campaign_id,
                CampaignChannel.was_expanded == False,
                CampaignChannel.limit_reached == False,
            ).limit(10)  # Process 10 channels per iteration
        )
        channels_to_expand = result.scalars().all()

        if not channels_to_expand:
            logger.info("CoordinatorAgent: No channels to expand")
            return

        logger.info(f"CoordinatorAgent: Expanding {len(channels_to_expand)} channels")

        for campaign_channel in channels_to_expand:
            if await self.check_should_stop():
                break

            if campaign_channel.channel_id in self._processed_channels:
                continue

            result = await self._channel_expansion_agent.execute({
                "channel_id": str(campaign_channel.channel_id),
                "max_videos": campaign_channel.videos_limit,
            })

            if result.success:
                video_ids = [UUID(vid) for vid in result.data.get("video_ids", [])]
                self._videos_pending_filter.extend(video_ids)
                self._processed_channels.add(campaign_channel.channel_id)

                self.increment_api_calls(result.api_calls_made)

    async def _run_topic_filter(self, topic: str) -> None:
        """Run topic filter on pending videos."""
        if not self._topic_filter_agent:
            return

        if not self._videos_pending_filter:
            logger.info("CoordinatorAgent: No videos pending filter")
            return

        # Get videos that haven't been filtered yet
        result = await self.db.execute(
            select(CampaignVideo.video_id).where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == None,
            ).limit(100)  # Process 100 videos per iteration
        )
        pending_video_ids = [row[0] for row in result.all()]

        if not pending_video_ids:
            self._videos_pending_filter.clear()
            logger.info("CoordinatorAgent: No videos pending filter")
            return

        logger.info(f"CoordinatorAgent: Filtering {len(pending_video_ids)} videos")

        result = await self._topic_filter_agent.execute({
            "video_ids": [str(vid) for vid in pending_video_ids],
            "topic": topic,
            "batch_size": 10,
        })

        if result.success:
            self.increment_llm_calls(
                result.llm_calls_made,
                result.tokens_used,
                result.estimated_cost_usd
            )

            # Store relevant videos for similar videos search
            relevant_ids = [UUID(vid) for vid in result.data.get("relevant_ids", [])]
            self.context.shared_state["last_relevant_videos"] = relevant_ids

        # Clear pending list
        self._videos_pending_filter = [
            vid for vid in self._videos_pending_filter
            if vid not in pending_video_ids
        ]

    async def _run_similar_videos(self) -> None:
        """Run similar videos agent for recently found relevant videos."""
        if not self._similar_videos_agent:
            return

        # Get relevant videos from last filter run
        relevant_videos = self.context.shared_state.get("last_relevant_videos", [])

        if not relevant_videos:
            logger.info("CoordinatorAgent: No relevant videos for similar video search")
            return

        # Limit to a subset for similar videos search
        videos_for_similar = relevant_videos[:20]

        logger.info(f"CoordinatorAgent: Finding similar videos for {len(videos_for_similar)} videos")

        result = await self._similar_videos_agent.execute({
            "video_ids": [str(vid) for vid in videos_for_similar],
            "depth": 0,
            "max_per_video": 5,
        })

        if result.success:
            video_ids = [UUID(vid) for vid in result.data.get("discovered_video_ids", [])]
            self._videos_pending_filter.extend(video_ids)

            self.increment_api_calls(result.api_calls_made)

    async def _save_checkpoint(self) -> None:
        """Save current state to campaign checkpoint."""
        checkpoint = {
            "iteration": self._iteration,
            "processed_channels": [str(cid) for cid in self._processed_channels],
            "videos_pending_filter": [str(vid) for vid in self._videos_pending_filter],
            "saved_at": datetime.utcnow().isoformat(),
        }
        self.context.campaign.save_checkpoint(checkpoint)
        await self.db.flush()

    async def _restore_checkpoint(self) -> None:
        """Restore state from campaign checkpoint."""
        checkpoint = self.context.campaign.checkpoint_data
        if not checkpoint:
            return

        self._iteration = checkpoint.get("iteration", 0)
        self._processed_channels = {
            UUID(cid) for cid in checkpoint.get("processed_channels", [])
        }
        self._videos_pending_filter = [
            UUID(vid) for vid in checkpoint.get("videos_pending_filter", [])
        ]

        logger.info(
            f"CoordinatorAgent: Restored from checkpoint at iteration {self._iteration}"
        )
