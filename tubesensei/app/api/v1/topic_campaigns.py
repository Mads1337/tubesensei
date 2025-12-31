"""
Topic Campaigns API Endpoints

REST API for managing topic-based video discovery campaigns.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.permissions import require_permission, Permission
from app.database import get_db
from app.models.topic_campaign import CampaignStatus as DBCampaignStatus
from app.models.agent_run import AgentType as DBAgentType
from app.services.topic_discovery import TopicDiscoveryService
from app.schemas.topic_campaign import (
    TopicCampaignCreate,
    TopicCampaignUpdate,
    TopicCampaignResponse,
    TopicCampaignListResponse,
    CampaignProgress,
    CampaignVideoListResponse,
    CampaignChannelListResponse,
    AgentRunListResponse,
    CampaignActionResponse,
    ExportResponse,
    CampaignStatus,
    AgentType,
)
from app.workers.topic_discovery_tasks import (
    run_topic_campaign_task,
    process_campaign_transcripts_task,
    extract_campaign_ideas_task,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/topic-campaigns", tags=["Topic Campaigns"])


# Dependency for service
async def get_topic_discovery_service(
    db: AsyncSession = Depends(get_db),
) -> TopicDiscoveryService:
    return TopicDiscoveryService(db)


# Campaign CRUD

@router.post("/", response_model=TopicCampaignResponse, status_code=201)
async def create_campaign(
    data: TopicCampaignCreate,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Create a new topic discovery campaign.

    The campaign will be created in DRAFT status. Use the /start endpoint to begin discovery.
    """
    campaign = await service.create_campaign(
        name=data.name,
        topic=data.topic,
        description=data.description,
        config=data.config.model_dump() if data.config else None,
    )
    return campaign


@router.get("/", response_model=TopicCampaignListResponse)
async def list_campaigns(
    status: Optional[CampaignStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """List all topic discovery campaigns."""
    db_status = DBCampaignStatus(status.value) if status else None
    campaigns = await service.list_campaigns(
        status=db_status,
        limit=limit,
        offset=offset,
    )

    # Get actual total count for proper pagination
    total = await service.count_campaigns(status=db_status)

    return TopicCampaignListResponse(
        items=campaigns,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(campaigns)) < total,
    )


@router.get("/{campaign_id}", response_model=TopicCampaignResponse)
async def get_campaign(
    campaign_id: UUID,
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get a specific campaign by ID."""
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.put("/{campaign_id}", response_model=TopicCampaignResponse)
async def update_campaign(
    campaign_id: UUID,
    data: TopicCampaignUpdate,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Update a campaign.

    Only campaigns in DRAFT status can be updated.
    """
    try:
        campaign = await service.update_campaign(
            campaign_id=campaign_id,
            name=data.name,
            description=data.description,
            config=data.config.model_dump() if data.config else None,
        )
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{campaign_id}", status_code=204)
async def delete_campaign(
    campaign_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_DELETE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Delete a campaign.

    Running campaigns cannot be deleted. Cancel them first.
    """
    try:
        success = await service.delete_campaign(campaign_id)
        if not success:
            raise HTTPException(status_code=404, detail="Campaign not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Campaign Control

@router.post("/{campaign_id}/start", response_model=CampaignActionResponse)
async def start_campaign(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Start a campaign.

    The campaign will begin discovering videos in the background.
    Use the /progress endpoint to monitor progress.
    """
    try:
        campaign = await service.start_campaign(campaign_id)

        # Queue the Celery task to run the campaign
        task = run_topic_campaign_task.delay(str(campaign_id), resume=False)

        return CampaignActionResponse(
            campaign_id=campaign_id,
            action="start",
            success=True,
            status=CampaignStatus(campaign.status.value),
            message="Campaign started successfully",
            task_id=task.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{campaign_id}/pause", response_model=CampaignActionResponse)
async def pause_campaign(
    campaign_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Pause a running campaign."""
    try:
        campaign = await service.pause_campaign(campaign_id)
        return CampaignActionResponse(
            campaign_id=campaign_id,
            action="pause",
            success=True,
            status=CampaignStatus(campaign.status.value),
            message="Campaign paused successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{campaign_id}/resume", response_model=CampaignActionResponse)
async def resume_campaign(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Resume a paused campaign."""
    try:
        campaign = await service.resume_campaign(campaign_id)

        # Queue the Celery task to resume the campaign
        task = run_topic_campaign_task.delay(str(campaign_id), resume=True)

        return CampaignActionResponse(
            campaign_id=campaign_id,
            action="resume",
            success=True,
            status=CampaignStatus(campaign.status.value),
            message="Campaign resumed successfully",
            task_id=task.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{campaign_id}/cancel", response_model=CampaignActionResponse)
async def cancel_campaign(
    campaign_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Cancel a running or paused campaign."""
    try:
        campaign = await service.cancel_campaign(campaign_id)
        return CampaignActionResponse(
            campaign_id=campaign_id,
            action="cancel",
            success=True,
            status=CampaignStatus(campaign.status.value),
            message="Campaign cancelled successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{campaign_id}/retry", response_model=CampaignActionResponse)
async def retry_campaign(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Retry a failed campaign.

    Resets the error state and immediately starts the campaign again.
    Only campaigns in FAILED status can be retried.
    """
    try:
        # Reset the failed campaign to DRAFT status
        campaign = await service.retry_campaign(campaign_id)

        # Start the campaign immediately
        campaign = await service.start_campaign(campaign_id)

        # Queue the Celery task to run the campaign
        task = run_topic_campaign_task.delay(str(campaign_id), resume=False)

        return CampaignActionResponse(
            campaign_id=campaign_id,
            action="retry",
            success=True,
            status=CampaignStatus(campaign.status.value),
            message="Campaign retried successfully and started",
            task_id=task.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Progress and Results

@router.get("/{campaign_id}/progress", response_model=CampaignProgress)
async def get_campaign_progress(
    campaign_id: UUID,
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get real-time progress data for a campaign."""
    try:
        progress = await service.get_progress(campaign_id)
        return progress
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{campaign_id}/videos")
async def get_campaign_videos(
    campaign_id: UUID,
    relevant_only: bool = Query(False, description="Only show relevant videos"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get videos discovered by a campaign."""
    videos = await service.get_videos(
        campaign_id=campaign_id,
        relevant_only=relevant_only,
        limit=limit,
        offset=offset,
    )
    return {
        "items": videos,
        "total": len(videos),
        "limit": limit,
        "offset": offset,
        "has_more": len(videos) == limit,
    }


@router.get("/{campaign_id}/channels")
async def get_campaign_channels(
    campaign_id: UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get channels discovered by a campaign."""
    channels = await service.get_channels(
        campaign_id=campaign_id,
        limit=limit,
        offset=offset,
    )
    return {
        "items": channels,
        "total": len(channels),
        "limit": limit,
        "offset": offset,
        "has_more": len(channels) == limit,
    }


@router.get("/{campaign_id}/agent-runs")
async def get_campaign_agent_runs(
    campaign_id: UUID,
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    limit: int = Query(50, ge=1, le=100),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get agent execution history for a campaign."""
    db_agent_type = DBAgentType(agent_type.value) if agent_type else None
    runs = await service.get_agent_runs(
        campaign_id=campaign_id,
        agent_type=db_agent_type,
        limit=limit,
    )
    return {
        "items": runs,
        "total": len(runs),
    }


# Export

@router.get("/{campaign_id}/export")
async def export_campaign_results(
    campaign_id: UUID,
    format: str = Query("json", regex="^(json|csv)$"),
    relevant_only: bool = Query(True),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Export campaign results.

    Returns the data directly as a downloadable file.
    """
    try:
        data = await service.export_results(
            campaign_id=campaign_id,
            format=format,
            relevant_only=relevant_only,
        )

        media_type = "application/json" if format == "json" else "text/csv"
        filename = f"campaign_{campaign_id}.{format}"

        return Response(
            content=data,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Bulk Operations

@router.post("/{campaign_id}/process-transcripts", response_model=CampaignActionResponse)
async def process_campaign_transcripts(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Queue transcript extraction for all relevant videos in the campaign.

    This uses the existing transcript extraction pipeline.
    """
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Queue the Celery task for bulk transcript processing
    task = process_campaign_transcripts_task.delay(str(campaign_id))

    return CampaignActionResponse(
        campaign_id=campaign_id,
        action="process_transcripts",
        success=True,
        status=CampaignStatus(campaign.status.value),
        message="Transcript processing queued",
        task_id=task.id,
    )


@router.post("/{campaign_id}/extract-ideas", response_model=CampaignActionResponse)
async def extract_campaign_ideas(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """
    Queue idea extraction for all videos with transcripts in the campaign.

    This uses the existing idea extraction pipeline.
    """
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Queue the Celery task for bulk idea extraction
    task = extract_campaign_ideas_task.delay(str(campaign_id))

    return CampaignActionResponse(
        campaign_id=campaign_id,
        action="extract_ideas",
        success=True,
        status=CampaignStatus(campaign.status.value),
        message="Idea extraction queued",
        task_id=task.id,
    )
