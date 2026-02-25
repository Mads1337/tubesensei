"""Admin Dashboard API router for TubeSensei."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy import select, func

from app.core.auth import get_current_user
from app.core.config import settings
from app.database import get_db
from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.models.video import Video
from app.models.channel import Channel
from app.models.idea import Idea
from .template_helpers import get_template_context

# Set up templates
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/dashboard", tags=["admin-dashboard"])


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    """Render admin dashboard with summary statistics."""
    total_campaigns = await db.scalar(select(func.count(TopicCampaign.id)))
    running_campaigns = await db.scalar(
        select(func.count(TopicCampaign.id)).where(
            TopicCampaign.status == CampaignStatus.RUNNING  # type: ignore[arg-type]
        )
    )
    completed_campaigns = await db.scalar(
        select(func.count(TopicCampaign.id)).where(
            TopicCampaign.status == CampaignStatus.COMPLETED  # type: ignore[arg-type]
        )
    )
    failed_campaigns = await db.scalar(
        select(func.count(TopicCampaign.id)).where(
            TopicCampaign.status == CampaignStatus.FAILED  # type: ignore[arg-type]
        )
    )
    draft_campaigns = await db.scalar(
        select(func.count(TopicCampaign.id)).where(
            TopicCampaign.status == CampaignStatus.DRAFT  # type: ignore[arg-type]
        )
    )
    total_videos = await db.scalar(select(func.count(Video.id)))
    total_channels = await db.scalar(select(func.count(Channel.id)))
    total_ideas = await db.scalar(select(func.count(Idea.id)))

    # Fetch 5 most recent campaigns ordered by created_at descending
    recent_campaigns_result = await db.execute(
        select(TopicCampaign)
        .order_by(TopicCampaign.created_at.desc())
        .limit(5)
    )
    recent_campaigns = recent_campaigns_result.scalars().all()

    # System health: check which API keys are configured
    system_health = {
        "youtube_api_key": bool(settings.YOUTUBE_API_KEY),
        "openai_api_key": bool(settings.OPENAI_API_KEY),
        "anthropic_api_key": bool(settings.ANTHROPIC_API_KEY),
    }

    stats = {
        "campaigns": {
            "total": total_campaigns or 0,
            "running": running_campaigns or 0,
            "completed": completed_campaigns or 0,
            "failed": failed_campaigns or 0,
            "draft": draft_campaigns or 0,
        },
        "videos": {"total": total_videos or 0},
        "channels": {"total": total_channels or 0},
        "ideas": {"total": total_ideas or 0},
        "recent_campaigns": recent_campaigns,
    }

    context = get_template_context(
        request, user=user, stats=stats, system_health=system_health
    )
    return templates.TemplateResponse("admin/dashboard/index.html", context)
