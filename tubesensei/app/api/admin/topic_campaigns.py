"""
Admin routes for Topic Campaign management.

Provides HTML templates and partial updates for the topic campaign dashboard.
"""
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

logger = logging.getLogger(__name__)
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.auth import get_current_user
from app.core.permissions import require_permission, Permission
from app.database import get_db
from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.models.campaign_video import CampaignVideo
from app.models.campaign_channel import CampaignChannel
from app.models.agent_run import AgentRun
from app.services.topic_discovery import TopicDiscoveryService

# Set up templates
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Add get_flashed_messages stub for Flask compatibility
def get_flashed_messages(with_categories=False):
    return []
templates.env.globals['get_flashed_messages'] = get_flashed_messages

router = APIRouter(prefix="/topic-campaigns", tags=["admin-topic-campaigns"])


async def get_topic_discovery_service(
    db: AsyncSession = Depends(get_db),
) -> TopicDiscoveryService:
    return TopicDiscoveryService(db)


@router.get("/", response_class=HTMLResponse)
async def list_campaigns(
    request: Request,
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all topic discovery campaigns."""
    service = TopicDiscoveryService(db)

    # Convert status string to enum if provided
    db_status = None
    if status:
        try:
            db_status = CampaignStatus(status)
        except ValueError:
            pass

    # Get campaigns
    offset = (page - 1) * per_page
    campaigns = await service.list_campaigns(
        status=db_status,
        limit=per_page,
        offset=offset,
    )

    # Get stats
    stats_query = select(
        func.count(TopicCampaign.id).filter(TopicCampaign.status == CampaignStatus.RUNNING).label('running'),
        func.count(TopicCampaign.id).filter(TopicCampaign.status == CampaignStatus.COMPLETED).label('completed'),
        func.sum(TopicCampaign.total_videos_discovered).label('total_videos'),
        func.sum(TopicCampaign.total_videos_relevant).label('relevant_videos'),
    )
    stats_result = await db.execute(stats_query)
    stats_row = stats_result.first()

    stats = {
        'running': stats_row.running if stats_row else 0,
        'completed': stats_row.completed if stats_row else 0,
        'total_videos': stats_row.total_videos if stats_row else 0,
        'relevant_videos': stats_row.relevant_videos if stats_row else 0,
    }

    # Count total for pagination
    count_query = select(func.count(TopicCampaign.id))
    if db_status:
        count_query = count_query.where(TopicCampaign.status == db_status)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    total_pages = (total + per_page - 1) // per_page

    return templates.TemplateResponse(
        "admin/topic_campaigns/list.html",
        {
            "request": request,
            "campaigns": campaigns,
            "stats": stats,
            "filters": {"search": search, "status": status},
            "page": page,
            "total_pages": total_pages,
            "total": total,
        }
    )


@router.get("/create", response_class=HTMLResponse)
async def create_campaign_form(
    request: Request,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
):
    """Show the create campaign form."""
    return templates.TemplateResponse(
        "admin/topic_campaigns/create.html",
        {"request": request, "user": user}
    )


@router.get("/{campaign_id}", response_class=HTMLResponse)
async def campaign_detail(
    request: Request,
    campaign_id: UUID,
    user = Depends(get_current_user),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Show campaign detail dashboard."""
    try:
        campaign = await service.get_campaign(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        return templates.TemplateResponse(
            "admin/topic_campaigns/detail.html",
            {
                "request": request,
                "campaign": campaign,
                "user": user,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading campaign {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading campaign")


@router.get("/{campaign_id}/videos", response_class=HTMLResponse)
async def campaign_videos_partial(
    request: Request,
    campaign_id: UUID,
    relevant_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user = Depends(get_current_user),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign videos as HTML partial."""
    try:
        videos = await service.get_videos(
            campaign_id=campaign_id,
            relevant_only=relevant_only,
            limit=limit,
            offset=offset,
        )

        has_more = len(videos) == limit

        return templates.TemplateResponse(
            "admin/topic_campaigns/partials/video_table.html",
            {
                "request": request,
                "videos": videos,
                "campaign_id": campaign_id,
                "relevant_only": relevant_only,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading campaign videos {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading campaign videos")


@router.get("/{campaign_id}/channels", response_class=HTMLResponse)
async def campaign_channels_partial(
    request: Request,
    campaign_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user = Depends(get_current_user),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign channels as HTML partial."""
    try:
        channels = await service.get_channels(
            campaign_id=campaign_id,
            limit=limit,
            offset=offset,
        )

        has_more = len(channels) == limit

        return templates.TemplateResponse(
            "admin/topic_campaigns/partials/channel_table.html",
            {
                "request": request,
                "channels": channels,
                "campaign_id": campaign_id,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading campaign channels {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading campaign channels")


@router.get("/{campaign_id}/agent-runs", response_class=HTMLResponse)
async def campaign_agent_runs_partial(
    request: Request,
    campaign_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    user = Depends(get_current_user),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign agent runs as HTML partial."""
    try:
        runs = await service.get_agent_runs(
            campaign_id=campaign_id,
            limit=limit,
        )

        return templates.TemplateResponse(
            "admin/topic_campaigns/partials/agent_runs.html",
            {
                "request": request,
                "runs": runs,
                "campaign_id": campaign_id,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading campaign agent runs {campaign_id}: {e}")
        raise HTTPException(status_code=500, detail="Error loading campaign agent runs")


@router.get("/{campaign_id}/edit", response_class=HTMLResponse)
async def edit_campaign_form(
    request: Request,
    campaign_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Show the edit campaign form. Only allows editing DRAFT or FAILED campaigns."""
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Only allow editing of DRAFT or FAILED campaigns
    if campaign.status not in [CampaignStatus.DRAFT, CampaignStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit campaign in {campaign.status.value} status. Only DRAFT or FAILED campaigns can be edited."
        )

    return templates.TemplateResponse(
        "admin/topic_campaigns/edit.html",
        {
            "request": request,
            "campaign": campaign,
            "user": user,
        }
    )


@router.patch("/{campaign_id}", response_class=HTMLResponse)
async def update_campaign(
    request: Request,
    campaign_id: UUID,
    name: str = Form(...),
    topic: str = Form(...),
    description: Optional[str] = Form(None),
    total_video_limit: int = Form(3000),
    per_channel_limit: int = Form(5),
    search_limit: int = Form(50),
    similar_videos_depth: int = Form(2),
    filter_threshold: float = Form(0.7),
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
    db: AsyncSession = Depends(get_db),
):
    """Update a campaign. Only allows updating DRAFT or FAILED campaigns."""
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Only allow editing of DRAFT or FAILED campaigns
    if campaign.status not in [CampaignStatus.DRAFT, CampaignStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit campaign in {campaign.status.value} status. Only DRAFT or FAILED campaigns can be edited."
        )

    # Build config dict
    config = {
        "total_video_limit": total_video_limit,
        "per_channel_limit": per_channel_limit,
        "search_limit": search_limit,
        "similar_videos_depth": similar_videos_depth,
        "filter_threshold": filter_threshold,
        # Preserve enabled_agents from existing config
        "enabled_agents": campaign.config.get("enabled_agents", ["search", "channel_expansion", "topic_filter", "similar_videos"]),
    }

    # Update campaign directly since service.update_campaign only allows DRAFT status
    campaign.name = name
    campaign.topic = topic
    campaign.description = description
    campaign.config = config

    # If campaign was FAILED, reset to DRAFT so it can be started again
    if campaign.status == CampaignStatus.FAILED:
        campaign.status = CampaignStatus.DRAFT
        campaign.error_message = None
        campaign.error_count = 0

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Updated topic campaign: {campaign_id}")

    # Return success response with HX-Redirect header
    response = Response(
        content="<div>Campaign updated successfully</div>",
        media_type="text/html",
    )
    response.headers["HX-Redirect"] = f"/admin/topic-campaigns/{campaign_id}"
    return response


@router.delete("/{campaign_id}", response_class=HTMLResponse)
async def delete_campaign(
    request: Request,
    campaign_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_DELETE)),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Delete a campaign. Cannot delete running campaigns."""
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Cannot delete running campaigns
    if campaign.status == CampaignStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running campaign. Cancel it first."
        )

    try:
        await service.delete_campaign(campaign_id)
        logger.info(f"Deleted topic campaign: {campaign_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Return HTML response with HX-Redirect header for HTMX
    response = Response(
        content="<div>Campaign deleted successfully</div>",
        media_type="text/html",
    )
    response.headers["HX-Redirect"] = "/admin/topic-campaigns"
    return response
