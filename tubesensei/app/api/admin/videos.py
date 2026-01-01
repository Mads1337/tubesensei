"""Admin Videos API router module."""

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional, List
from uuid import UUID
from datetime import datetime, date
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import joinedload

from app.core.auth import get_current_user
from app.models.video import Video, VideoStatus
from app.models.channel import Channel
from app.models.idea import Idea
from app.database import get_db
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/videos", tags=["admin-videos"])


@router.get("/", response_class=HTMLResponse)
async def videos_page(
    request: Request,
    search: Optional[str] = Query(None, description="Search in title/description"),
    status: Optional[str] = Query(None, description="Filter by status"),
    channel_id: Optional[UUID] = Query(None, description="Filter by channel"),
    valuable: Optional[str] = Query(None, description="Filter by valuable status: true, false, or unrated"),
    date_from: Optional[date] = Query(None, description="Filter by published date from"),
    date_to: Optional[date] = Query(None, description="Filter by published date to"),
    min_duration: Optional[int] = Query(None, ge=0, description="Minimum duration in minutes"),
    max_duration: Optional[int] = Query(None, ge=0, description="Maximum duration in minutes"),
    min_views: Optional[int] = Query(None, ge=0, description="Minimum view count"),
    max_views: Optional[int] = Query(None, ge=0, description="Maximum view count"),
    sort_by: str = Query("published_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render videos management page with filtering and sorting"""

    limit = settings.admin.ADMIN_PAGINATION_DEFAULT
    offset = (page - 1) * limit

    # Build query
    query = select(Video).options(joinedload(Video.channel))
    count_query = select(func.count(Video.id))

    # Apply filters
    filters = []

    if search:
        filters.append(
            or_(
                Video.title.ilike(f"%{search}%"),
                Video.description.ilike(f"%{search}%")
            )
        )

    if status:
        try:
            video_status = VideoStatus(status)
            filters.append(Video.status == video_status)
        except ValueError:
            pass

    if channel_id:
        filters.append(Video.channel_id == channel_id)

    if valuable == "true":
        filters.append(Video.is_valuable == True)
    elif valuable == "false":
        filters.append(Video.is_valuable == False)
    elif valuable == "unrated":
        filters.append(Video.is_valuable.is_(None))

    if date_from:
        filters.append(Video.published_at >= datetime.combine(date_from, datetime.min.time()))

    if date_to:
        filters.append(Video.published_at <= datetime.combine(date_to, datetime.max.time()))

    if min_duration is not None:
        filters.append(Video.duration_seconds >= min_duration * 60)

    if max_duration is not None:
        filters.append(Video.duration_seconds <= max_duration * 60)

    if min_views is not None:
        filters.append(Video.view_count >= min_views)

    if max_views is not None:
        filters.append(Video.view_count <= max_views)

    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))

    # Get total count
    total = await db.scalar(count_query)

    # Apply sorting
    sort_column = getattr(Video, sort_by, Video.published_at)
    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    videos = result.scalars().unique().all()

    # Convert to dicts and enrich with channel info
    video_list = []
    for video in videos:
        video_dict = {
            "id": str(video.id),
            "youtube_video_id": video.youtube_video_id,
            "title": video.title,
            "description": video.description,
            "thumbnail_url": video.thumbnail_url,
            "duration_seconds": video.duration_seconds,
            "duration_formatted": video.duration_formatted,
            "view_count": video.view_count,
            "like_count": video.like_count,
            "published_at": video.published_at,
            "status": video.status.value if video.status else None,
            "is_valuable": video.is_valuable,
            "valuable_score": video.valuable_score,
            "channel_id": str(video.channel_id) if video.channel_id else None,
            "channel_name": video.channel.name if video.channel else "Unknown",
            "youtube_url": video.youtube_url,
            "error_message": video.error_message,
        }
        video_list.append(video_dict)

    total_pages = (total + limit - 1) // limit if total else 1

    # Get stats for header
    stats_query = select(
        func.count(Video.id).label("total"),
        func.count(Video.id).filter(Video.status == VideoStatus.COMPLETED).label("completed"),
        func.count(Video.id).filter(Video.status == VideoStatus.PROCESSING).label("processing"),
        func.count(Video.id).filter(Video.status == VideoStatus.FAILED).label("failed"),
        func.count(Video.id).filter(Video.is_valuable == True).label("valuable"),
    )
    stats_result = await db.execute(stats_query)
    stats = stats_result.one()._asdict()

    # Get channels for filter dropdown
    channels_query = select(Channel.id, Channel.name).order_by(Channel.name)
    channels_result = await db.execute(channels_query)
    channels = [{"id": str(row.id), "name": row.name} for row in channels_result.all()]

    # Check if this is an HTMX request for just the table body
    is_htmx = request.headers.get("HX-Request") == "true"

    context = get_template_context(
        request,
        user=user,
        videos=video_list,
        total=total,
        page=page,
        total_pages=total_pages,
        stats=stats,
        channels=channels,
        filters={
            "search": search,
            "status": status,
            "channel_id": str(channel_id) if channel_id else None,
            "valuable": valuable,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "min_views": min_views,
            "max_views": max_views,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
        video_statuses=[s.value for s in VideoStatus],
    )

    if is_htmx:
        return templates.TemplateResponse("admin/videos/partials/video_table.html", context)

    return templates.TemplateResponse("admin/videos/list.html", context)


@router.get("/{video_id}", response_class=HTMLResponse)
async def video_detail(
    request: Request,
    video_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render video detail page"""

    query = select(Video).options(joinedload(Video.channel)).where(Video.id == video_id)
    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get ideas for this video
    ideas_query = select(Idea).where(Idea.video_id == video_id).order_by(Idea.created_at.desc())
    ideas_result = await db.execute(ideas_query)
    ideas = ideas_result.scalars().all()

    video_dict = {
        "id": str(video.id),
        "youtube_video_id": video.youtube_video_id,
        "title": video.title,
        "description": video.description,
        "thumbnail_url": video.thumbnail_url,
        "duration_seconds": video.duration_seconds,
        "duration_formatted": video.duration_formatted,
        "view_count": video.view_count,
        "like_count": video.like_count,
        "comment_count": video.comment_count,
        "published_at": video.published_at,
        "discovered_at": video.discovered_at,
        "processed_at": video.processed_at,
        "status": video.status.value if video.status else None,
        "is_valuable": video.is_valuable,
        "valuable_score": video.valuable_score,
        "valuable_reason": video.valuable_reason,
        "channel_id": str(video.channel_id) if video.channel_id else None,
        "channel_name": video.channel.name if video.channel else "Unknown",
        "youtube_url": video.youtube_url,
        "error_message": video.error_message,
        "has_captions": video.has_captions,
        "caption_languages": video.caption_languages,
        "language": video.language,
        "tags": video.tags,
        "category_id": video.category_id,
    }

    ideas_list = [
        {
            "id": str(idea.id),
            "title": idea.title,
            "description": idea.description,
            "created_at": idea.created_at,
        }
        for idea in ideas
    ]

    context = get_template_context(
        request,
        user=user,
        video=video_dict,
        ideas=ideas_list,
    )

    return templates.TemplateResponse("admin/videos/detail.html", context)
