"""Admin Ideas API router module."""

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
from uuid import UUID
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import joinedload

from app.core.auth import get_current_user
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.video import Video
from app.models.channel import Channel
from app.services.idea_service import IdeaService
from app.core.exceptions import NotFoundException
from app.database import get_db
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/ideas", tags=["admin-ideas"])


@router.get("/", response_class=HTMLResponse)
async def ideas_page(
    request: Request,
    search: Optional[str] = Query(None, description="Search in title/description"),
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    channel_id: Optional[UUID] = Query(None, description="Filter by channel"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence score"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1),
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    """Render ideas management page with filtering and sorting"""

    limit = settings.admin.ADMIN_PAGINATION_DEFAULT
    offset = (page - 1) * limit

    # Build query
    query = select(Idea).options(joinedload(Idea.video).joinedload(Video.channel))
    count_query = select(func.count(Idea.id))

    # Apply filters
    filters = []

    if search:
        filters.append(
            or_(
                Idea.title.ilike(f"%{search}%"),
                Idea.description.ilike(f"%{search}%"),
            )
        )

    if status:
        try:
            idea_status = IdeaStatus(status)
            filters.append(Idea.status == idea_status)
        except ValueError:
            pass

    if category:
        filters.append(Idea.category == category)

    if channel_id:
        filters.append(Video.channel_id == channel_id)
        # Need to join Video for channel filter on count query too
        count_query = count_query.join(Video, Idea.video_id == Video.id)

    if priority:
        try:
            idea_priority = IdeaPriority(priority)
            filters.append(Idea.priority == idea_priority)
        except ValueError:
            pass

    if min_confidence is not None:
        filters.append(Idea.confidence_score >= min_confidence)

    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))

    # Get total count
    total = await db.scalar(count_query)

    # Apply sorting
    sort_map = {
        "created_at": Idea.created_at,
        "confidence_score": Idea.confidence_score,
        "title": Idea.title,
        "priority": Idea.priority,
        "status": Idea.status,
        "category": Idea.category,
    }
    sort_column = sort_map.get(sort_by, Idea.created_at)
    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    ideas = result.scalars().unique().all()

    # Convert to dicts
    idea_list = []
    for idea in ideas:
        video = idea.video
        channel = video.channel if video else None
        idea_list.append({
            "id": str(idea.id),
            "title": idea.title,
            "description": idea.description,
            "category": idea.category,
            "status": idea.status.value if idea.status else None,
            "priority": idea.priority.value if idea.priority else None,
            "confidence_score": idea.confidence_score,
            "confidence_percentage": idea.confidence_percentage,
            "tags": idea.tags or [],
            "technologies": idea.technologies or [],
            "created_at": idea.created_at,
            "video_id": str(video.id) if video else None,
            "video_title": video.title if video else "Unknown",
            "youtube_video_id": video.youtube_video_id if video else None,
            "channel_id": str(channel.id) if channel else None,
            "channel_name": channel.name if channel else "Unknown",
        })

    total_pages = (total + limit - 1) // limit if total else 1

    # Get stats for header
    stats_query = select(
        func.count(Idea.id).label("total"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.EXTRACTED).label("extracted"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.REVIEWED).label("reviewed"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.SELECTED).label("selected"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.REJECTED).label("rejected"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.IN_PROGRESS).label("in_progress"),
        func.count(Idea.id).filter(Idea.status == IdeaStatus.IMPLEMENTED).label("implemented"),
        func.avg(Idea.confidence_score).label("avg_confidence"),
    )
    stats_result = await db.execute(stats_query)
    stats = stats_result.one()._asdict()
    stats["avg_confidence"] = round(float(stats["avg_confidence"] or 0) * 100, 1)

    # Get distinct categories for filter dropdown
    categories_query = (
        select(Idea.category)
        .where(Idea.category.isnot(None))
        .distinct()
        .order_by(Idea.category)
    )
    categories_result = await db.execute(categories_query)
    categories = [row[0] for row in categories_result.all()]

    # Get channels for filter dropdown
    channels_query = (
        select(Channel.id, Channel.name)
        .join(Video, Video.channel_id == Channel.id)
        .join(Idea, Idea.video_id == Video.id)
        .distinct()
        .order_by(Channel.name)
    )
    channels_result = await db.execute(channels_query)
    channels = [{"id": str(row.id), "name": row.name} for row in channels_result.all()]

    # Check if this is an HTMX request
    is_htmx = request.headers.get("HX-Request") == "true"

    context = get_template_context(
        request,
        user=user,
        ideas=idea_list,
        total=total,
        page=page,
        total_pages=total_pages,
        stats=stats,
        categories=categories,
        channels=channels,
        filters={
            "search": search,
            "status": status,
            "category": category,
            "channel_id": str(channel_id) if channel_id else None,
            "priority": priority,
            "min_confidence": min_confidence,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
        idea_statuses=[s.value for s in IdeaStatus],
        idea_priorities=[p.value for p in IdeaPriority],
    )

    if is_htmx:
        return templates.TemplateResponse("admin/ideas/partials/ideas_table.html", context)

    return templates.TemplateResponse("admin/ideas/list.html", context)


@router.get("/{idea_id}", response_class=HTMLResponse)
async def idea_detail(
    request: Request,
    idea_id: UUID,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    """Render idea detail partial (for HTMX expansion)"""

    service = IdeaService(db)
    try:
        idea_context = await service.get_idea_context(idea_id)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Idea not found")

    context = get_template_context(
        request,
        user=user,
        idea=idea_context["idea"],
        video=idea_context["video"],
        channel=idea_context["channel"],
        transcript_excerpt=idea_context.get("transcript_excerpt"),
    )

    return templates.TemplateResponse("admin/ideas/partials/idea_detail.html", context)
