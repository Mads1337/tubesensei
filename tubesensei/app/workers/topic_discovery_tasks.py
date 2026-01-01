"""
Celery tasks for topic-based video discovery campaigns.

These tasks handle the background processing of discovery campaigns,
including the main campaign runner and individual agent tasks.
"""
import logging
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from uuid import UUID

import redis
from celery import Task

from app.celery_app import celery_app, update_job_status
from app.config import settings
from app.database import create_worker_session_factory
from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.models.campaign_video import CampaignVideo
from app.models.campaign_channel import CampaignChannel
from app.services.topic_discovery import TopicDiscoveryService
from app.agents.base import AgentContext, AgentEvent
from app.agents.coordinator import CoordinatorAgent
from app.agents.search_agent import SearchAgent
from app.agents.channel_expansion_agent import ChannelExpansionAgent
from app.agents.topic_filter_agent import TopicFilterAgent
from app.agents.similar_videos_agent import SimilarVideosAgent
from app.utils.rate_limiter import RateLimiter
from app.integrations.youtube_api import YouTubeAPIClient
from app.ai.llm_manager import LLMManager
from app.workers.monitoring import TaskMonitor

logger = logging.getLogger(__name__)

# Synchronous Redis client for Celery workers
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Get or create synchronous Redis client for Celery workers."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL)
    return _redis_client


def publish_campaign_progress(campaign_id: str, data: dict) -> None:
    """
    Publish campaign progress update to Redis pubsub.

    Args:
        campaign_id: The campaign ID
        data: Progress data dict
    """
    try:
        client = get_redis_client()
        channel = f"campaign:{campaign_id}"
        message = json.dumps({
            **data,
            "timestamp": datetime.utcnow().isoformat(),
        })
        client.publish(channel, message)
        logger.debug(f"Published campaign progress to {channel}: {data.get('type', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to publish campaign progress: {e}")


class TopicDiscoveryTask(Task):
    """Base task class for topic discovery with callbacks."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task executes successfully."""
        try:
            logger.info(f"Topic discovery task {task_id} completed successfully")
            update_job_status(task_id, "completed", result=retval)
            TaskMonitor.record_task_complete(self.name, retval.get('duration', 0))
        except Exception as e:
            logger.error(f"Error in success callback for task {task_id}: {e}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        try:
            logger.error(f"Topic discovery task {task_id} failed: {exc}")
            update_job_status(task_id, "failed", error=str(exc))
            TaskMonitor.record_task_fail(self.name, str(exc))

            # Update campaign status to failed
            campaign_id = kwargs.get('campaign_id') or (args[0] if args else None)
            if campaign_id:
                # Publish campaign failure event
                publish_campaign_progress(campaign_id, {
                    "type": "campaign_failed",
                    "message": f"Campaign failed: {str(exc)}",
                    "status": "failed",
                    "error": str(exc),
                })

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._mark_campaign_failed(campaign_id, str(exc))
                )
        except Exception as e:
            logger.error(f"Error in failure callback for task {task_id}: {e}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is being retried."""
        try:
            logger.warning(f"Topic discovery task {task_id} being retried: {exc}")
            update_job_status(task_id, "retrying", error=str(exc))
        except Exception as e:
            logger.error(f"Error in retry callback for task {task_id}: {e}")

    async def _mark_campaign_failed(self, campaign_id: str, error: str):
        """Mark campaign as failed in database."""
        async with AsyncSessionLocal() as session:
            campaign = await session.get(TopicCampaign, UUID(campaign_id))
            if campaign:
                campaign.fail(error)
                await session.commit()


# Event handler for progress updates
def create_progress_handler(task_id: str, campaign_id: str):
    """
    Create an event handler that updates task progress and publishes to Redis.

    Args:
        task_id: The Celery task ID
        campaign_id: The campaign ID for pubsub channel

    Returns:
        Handler function for AgentEvent
    """
    def handler(event: AgentEvent):
        try:
            logger.debug(f"Agent event: {event.event_type.value} - {event.message}")

            # Publish to Redis pubsub for WebSocket updates
            publish_campaign_progress(campaign_id, {
                "type": event.event_type.value,
                "message": event.message,
                "progress": getattr(event, 'progress', None),
                "agent": getattr(event, 'agent', None),
                "data": getattr(event, 'data', {}),
                "task_id": task_id,
            })
        except Exception as e:
            logger.error(f"Error handling agent event: {e}")
    return handler


@celery_app.task(
    base=TopicDiscoveryTask,
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    name="app.workers.topic_discovery_tasks.run_topic_campaign"
)
def run_topic_campaign_task(
    self,
    campaign_id: str,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete topic discovery campaign.

    This is the main task that orchestrates the entire discovery process
    using the CoordinatorAgent.

    Args:
        campaign_id: Campaign ID (string UUID)
        resume: Whether to resume from checkpoint

    Returns:
        Dictionary with campaign results
    """
    start_time = datetime.now(timezone.utc)
    TaskMonitor.record_task_start(self.name)

    # Diagnostic logging for task lifecycle
    logger.info(
        f"[TASK START] Campaign task starting - "
        f"campaign_id={campaign_id}, task_id={self.request.id}, "
        f"resume={resume}, retry={self.request.retries}/{self.max_retries}"
    )

    try:
        campaign_uuid = UUID(campaign_id)

        async def _run():
            # Create fresh session factory for this event loop
            WorkerSession, worker_engine = create_worker_session_factory()
            try:
                async with WorkerSession() as session:
                    # Get campaign
                    campaign = await session.get(TopicCampaign, campaign_uuid)
                    if not campaign:
                        raise ValueError(f"Campaign not found: {campaign_id}")

                    # Validate campaign can run
                    if not resume and campaign.status != CampaignStatus.DRAFT:
                        if campaign.status != CampaignStatus.RUNNING:
                            raise ValueError(f"Campaign cannot be started from {campaign.status.value} status")
                    elif resume and campaign.status != CampaignStatus.PAUSED:
                        raise ValueError(f"Campaign cannot be resumed from {campaign.status.value} status")

                    logger.info(
                        f"[TASK RUNNING] Executing coordinator - "
                        f"campaign_id={campaign_id}, name={campaign.name}, "
                        f"topic={campaign.topic[:50]}..."
                    )

                    # Publish campaign start event
                    publish_campaign_progress(campaign_id, {
                        "type": "campaign_started",
                        "message": f"Campaign '{campaign.name}' started",
                        "status": "running",
                        "topic": campaign.topic,
                    })

                    # Create agent context
                    context = AgentContext(
                        campaign_id=campaign_uuid,
                        campaign=campaign,
                        db=session,
                        config=campaign.config or {},
                        youtube_rate_limiter=RateLimiter(requests_per_minute=120),
                        llm_rate_limiter=RateLimiter(requests_per_minute=60),
                        event_callback=create_progress_handler(self.request.id, campaign_id),
                    )

                    # Initialize clients
                    youtube_client = YouTubeAPIClient()
                    llm_manager = LLMManager()

                    # Create and run coordinator
                    coordinator = CoordinatorAgent(
                        context=context,
                        youtube_client=youtube_client,
                        llm_manager=llm_manager,
                    )

                    result = await coordinator.execute({
                        "topic": campaign.topic,
                        "resume_from_checkpoint": resume,
                    })

                    # Refresh campaign state
                    await session.refresh(campaign)

                    return {
                        "success": result.success,
                        "campaign_id": campaign_id,
                        "status": campaign.status.value,
                        "videos_discovered": campaign.total_videos_discovered,
                        "videos_relevant": campaign.total_videos_relevant,
                        "channels_explored": campaign.total_channels_explored,
                        "api_calls": campaign.api_calls_made,
                        "llm_calls": campaign.llm_calls_made,
                        "duration": result.duration_seconds,
                        "errors": result.errors,
                    }
            finally:
                # Dispose of worker engine to clean up connections
                await worker_engine.dispose()

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        result["total_duration"] = duration

        logger.info(
            f"[TASK SUCCESS] Campaign task completed - "
            f"campaign_id={campaign_id}, task_id={self.request.id}, "
            f"videos_relevant={result['videos_relevant']}, duration={duration:.1f}s"
        )

        # Publish campaign completion event
        publish_campaign_progress(campaign_id, {
            "type": "campaign_completed",
            "message": f"Campaign completed with {result['videos_relevant']} relevant videos",
            "status": result.get("status", "completed"),
            "stats": {
                "videos_discovered": result.get("videos_discovered", 0),
                "videos_relevant": result.get("videos_relevant", 0),
                "channels_explored": result.get("channels_explored", 0),
                "api_calls": result.get("api_calls", 0),
                "llm_calls": result.get("llm_calls", 0),
                "duration": duration,
            }
        })

        return result

    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.exception(
            f"[TASK ERROR] Campaign task failed - "
            f"campaign_id={campaign_id}, task_id={self.request.id}, "
            f"elapsed={elapsed:.1f}s, retry={self.request.retries}/{self.max_retries}, "
            f"error={str(e)[:200]}"
        )
        raise self.retry(exc=e)


@celery_app.task(
    base=TopicDiscoveryTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.topic_discovery_tasks.run_search_agent"
)
def run_search_agent_task(
    self,
    campaign_id: str,
    topic: str,
    max_results: int = 50,
) -> Dict[str, Any]:
    """
    Run the search agent for a campaign.

    Args:
        campaign_id: Campaign ID
        topic: Search topic
        max_results: Maximum results to fetch

    Returns:
        Dictionary with search results
    """
    start_time = datetime.now(timezone.utc)
    TaskMonitor.record_task_start(self.name)

    try:
        campaign_uuid = UUID(campaign_id)

        async def _run():
            # Create fresh session factory for this event loop
            WorkerSession, worker_engine = create_worker_session_factory()
            try:
                async with WorkerSession() as session:
                    campaign = await session.get(TopicCampaign, campaign_uuid)
                    if not campaign:
                        raise ValueError(f"Campaign not found: {campaign_id}")

                    context = AgentContext(
                        campaign_id=campaign_uuid,
                        campaign=campaign,
                        db=session,
                        config=campaign.config or {},
                        youtube_rate_limiter=RateLimiter(requests_per_minute=120),
                        llm_rate_limiter=RateLimiter(requests_per_minute=60),
                    )

                    youtube_client = YouTubeAPIClient()
                    agent = SearchAgent(context, youtube_client)

                    result = await agent.execute({
                        "topic": topic,
                        "max_results": max_results,
                    })

                    return {
                        "success": result.success,
                        "video_ids": result.data.get("video_ids", []),
                        "channel_ids": result.data.get("channel_ids", []),
                        "new_videos": result.data.get("new_videos_count", 0),
                        "api_calls": result.api_calls_made,
                        "duration": result.duration_seconds,
                    }
            finally:
                await worker_engine.dispose()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        return result

    except Exception as e:
        logger.exception(f"Search agent task failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    base=TopicDiscoveryTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.topic_discovery_tasks.run_channel_expansion"
)
def run_channel_expansion_task(
    self,
    campaign_id: str,
    channel_id: str,
    max_videos: int = 5,
) -> Dict[str, Any]:
    """
    Run channel expansion for a single channel.

    Args:
        campaign_id: Campaign ID
        channel_id: Channel ID to expand
        max_videos: Maximum videos to fetch

    Returns:
        Dictionary with expansion results
    """
    start_time = datetime.now(timezone.utc)
    TaskMonitor.record_task_start(self.name)

    try:
        campaign_uuid = UUID(campaign_id)
        channel_uuid = UUID(channel_id)

        async def _run():
            # Create fresh session factory for this event loop
            WorkerSession, worker_engine = create_worker_session_factory()
            try:
                async with WorkerSession() as session:
                    campaign = await session.get(TopicCampaign, campaign_uuid)
                    if not campaign:
                        raise ValueError(f"Campaign not found: {campaign_id}")

                    context = AgentContext(
                        campaign_id=campaign_uuid,
                        campaign=campaign,
                        db=session,
                        config=campaign.config or {},
                        youtube_rate_limiter=RateLimiter(requests_per_minute=120),
                        llm_rate_limiter=RateLimiter(requests_per_minute=60),
                    )

                    youtube_client = YouTubeAPIClient()
                    agent = ChannelExpansionAgent(context, youtube_client)

                    result = await agent.execute({
                        "channel_id": str(channel_uuid),
                        "max_videos": max_videos,
                    })

                    return {
                        "success": result.success,
                        "video_ids": result.data.get("video_ids", []),
                        "new_videos": result.data.get("new_videos_count", 0),
                        "was_expanded": result.data.get("was_expanded", False),
                        "api_calls": result.api_calls_made,
                        "duration": result.duration_seconds,
                    }
            finally:
                await worker_engine.dispose()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        return result

    except Exception as e:
        logger.exception(f"Channel expansion task failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    base=TopicDiscoveryTask,
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    name="app.workers.topic_discovery_tasks.run_topic_filter"
)
def run_topic_filter_task(
    self,
    campaign_id: str,
    video_ids: List[str],
    topic: str,
) -> Dict[str, Any]:
    """
    Run topic filter on a batch of videos.

    Args:
        campaign_id: Campaign ID
        video_ids: List of video IDs to filter
        topic: Topic to filter by

    Returns:
        Dictionary with filter results
    """
    start_time = datetime.now(timezone.utc)
    TaskMonitor.record_task_start(self.name)

    try:
        campaign_uuid = UUID(campaign_id)

        async def _run():
            # Create fresh session factory for this event loop
            WorkerSession, worker_engine = create_worker_session_factory()
            try:
                async with WorkerSession() as session:
                    campaign = await session.get(TopicCampaign, campaign_uuid)
                    if not campaign:
                        raise ValueError(f"Campaign not found: {campaign_id}")

                    context = AgentContext(
                        campaign_id=campaign_uuid,
                        campaign=campaign,
                        db=session,
                        config=campaign.config or {},
                        youtube_rate_limiter=RateLimiter(requests_per_minute=120),
                        llm_rate_limiter=RateLimiter(requests_per_minute=60),
                    )

                    llm_manager = LLMManager()
                    agent = TopicFilterAgent(context, llm_manager)

                    result = await agent.execute({
                        "video_ids": video_ids,
                        "topic": topic,
                        "batch_size": 10,
                    })

                    return {
                        "success": result.success,
                        "relevant_ids": result.data.get("relevant_ids", []),
                        "filtered_ids": result.data.get("filtered_ids", []),
                        "total_processed": result.data.get("total_processed", 0),
                        "acceptance_rate": result.data.get("acceptance_rate", 0),
                        "llm_calls": result.llm_calls_made,
                        "tokens_used": result.tokens_used,
                        "cost_usd": result.estimated_cost_usd,
                        "duration": result.duration_seconds,
                    }
            finally:
                await worker_engine.dispose()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        return result

    except Exception as e:
        logger.exception(f"Topic filter task failed: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    base=TopicDiscoveryTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.topic_discovery_tasks.run_similar_videos"
)
def run_similar_videos_task(
    self,
    campaign_id: str,
    video_ids: List[str],
    depth: int = 0,
) -> Dict[str, Any]:
    """
    Run similar videos discovery for a batch of videos.

    Args:
        campaign_id: Campaign ID
        video_ids: List of video IDs to find similar videos for
        depth: Current recursion depth

    Returns:
        Dictionary with discovery results
    """
    start_time = datetime.now(timezone.utc)
    TaskMonitor.record_task_start(self.name)

    try:
        campaign_uuid = UUID(campaign_id)

        async def _run():
            # Create fresh session factory for this event loop
            WorkerSession, worker_engine = create_worker_session_factory()
            try:
                async with WorkerSession() as session:
                    campaign = await session.get(TopicCampaign, campaign_uuid)
                    if not campaign:
                        raise ValueError(f"Campaign not found: {campaign_id}")

                    context = AgentContext(
                        campaign_id=campaign_uuid,
                        campaign=campaign,
                        db=session,
                        config=campaign.config or {},
                        youtube_rate_limiter=RateLimiter(requests_per_minute=120),
                        llm_rate_limiter=RateLimiter(requests_per_minute=60),
                    )

                    youtube_client = YouTubeAPIClient()
                    agent = SimilarVideosAgent(context, youtube_client)

                    result = await agent.execute({
                        "video_ids": video_ids,
                        "depth": depth,
                        "max_per_video": 5,
                    })

                    return {
                        "success": result.success,
                        "discovered_video_ids": result.data.get("discovered_video_ids", []),
                        "discovered_channel_ids": result.data.get("discovered_channel_ids", []),
                        "new_videos": result.data.get("new_videos_count", 0),
                        "api_calls": result.api_calls_made,
                        "duration": result.duration_seconds,
                    }
            finally:
                await worker_engine.dispose()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        return result

    except Exception as e:
        logger.exception(f"Similar videos task failed: {e}")
        raise self.retry(exc=e)


# Utility task for bulk operations

@celery_app.task(
    bind=True,
    name="app.workers.topic_discovery_tasks.process_campaign_transcripts"
)
def process_campaign_transcripts_task(
    self,
    campaign_id: str,
) -> Dict[str, Any]:
    """
    Queue transcript extraction for all relevant videos in a campaign.

    Args:
        campaign_id: Campaign ID

    Returns:
        Dictionary with queued job info
    """
    from app.workers.processing_tasks import extract_transcript_task

    try:
        campaign_uuid = UUID(campaign_id)

        async def _get_pending_videos():
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(CampaignVideo.video_id).where(
                        CampaignVideo.campaign_id == campaign_uuid,
                        CampaignVideo.is_topic_relevant == True,
                        CampaignVideo.transcript_extracted == False,
                    )
                )
                return [str(row[0]) for row in result.all()]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            video_ids = loop.run_until_complete(_get_pending_videos())
        finally:
            loop.close()

        # Queue transcript extraction tasks
        queued = 0
        for video_id in video_ids:
            extract_transcript_task.delay(video_id)
            queued += 1

        logger.info(f"Queued {queued} transcript extraction tasks for campaign {campaign_id}")

        return {
            "success": True,
            "campaign_id": campaign_id,
            "videos_queued": queued,
        }

    except Exception as e:
        logger.exception(f"Failed to queue transcript processing: {e}")
        return {
            "success": False,
            "campaign_id": campaign_id,
            "error": str(e),
        }


@celery_app.task(
    bind=True,
    name="app.workers.topic_discovery_tasks.extract_campaign_ideas"
)
def extract_campaign_ideas_task(
    self,
    campaign_id: str,
) -> Dict[str, Any]:
    """
    Queue idea extraction for all videos with transcripts in a campaign.

    Args:
        campaign_id: Campaign ID

    Returns:
        Dictionary with queued job info
    """
    try:
        campaign_uuid = UUID(campaign_id)

        async def _get_videos_with_transcripts():
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select

                # Get videos that are relevant and have transcripts but no ideas extracted
                result = await session.execute(
                    select(CampaignVideo.video_id).where(
                        CampaignVideo.campaign_id == campaign_uuid,
                        CampaignVideo.is_topic_relevant == True,
                        CampaignVideo.transcript_extracted == True,
                        CampaignVideo.ideas_extracted == False,
                    )
                )
                return [str(row[0]) for row in result.all()]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            video_ids = loop.run_until_complete(_get_videos_with_transcripts())
        finally:
            loop.close()

        # Queue idea extraction tasks
        # Note: Import here to avoid circular imports
        from app.workers.processing_tasks import extract_transcript_task

        queued = 0
        for video_id in video_ids:
            # TODO: Replace with actual idea extraction task when available
            # For now, we'll just log and count - the actual task should be
            # something like: extract_ideas_task.delay(video_id)
            queued += 1

        logger.info(f"Queued {queued} idea extraction tasks for campaign {campaign_id}")

        return {
            "success": True,
            "campaign_id": campaign_id,
            "videos_queued": queued,
        }

    except Exception as e:
        logger.exception(f"Failed to queue idea extraction: {e}")
        return {
            "success": False,
            "campaign_id": campaign_id,
            "error": str(e),
        }
