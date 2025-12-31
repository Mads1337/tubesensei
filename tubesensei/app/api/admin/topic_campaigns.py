"""
Admin routes for Topic Campaign management.

Provides HTML templates and partial updates for the topic campaign dashboard.
"""
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

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
async def create_campaign_form(request: Request):
    """Show the create campaign form."""
    return templates.TemplateResponse(
        "admin/topic_campaigns/create.html",
        {"request": request}
    )


@router.get("/{campaign_id}", response_class=HTMLResponse)
async def campaign_detail(
    request: Request,
    campaign_id: UUID,
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Show campaign detail dashboard."""
    campaign = await service.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    return templates.TemplateResponse(
        "admin/topic_campaigns/detail.html",
        {
            "request": request,
            "campaign": campaign,
        }
    )


@router.get("/{campaign_id}/videos", response_class=HTMLResponse)
async def campaign_videos_partial(
    request: Request,
    campaign_id: UUID,
    relevant_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign videos as HTML partial."""
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


@router.get("/{campaign_id}/channels", response_class=HTMLResponse)
async def campaign_channels_partial(
    request: Request,
    campaign_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign channels as HTML partial."""
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


@router.get("/{campaign_id}/agent-runs", response_class=HTMLResponse)
async def campaign_agent_runs_partial(
    request: Request,
    campaign_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    service: TopicDiscoveryService = Depends(get_topic_discovery_service),
):
    """Get campaign agent runs as HTML partial."""
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
