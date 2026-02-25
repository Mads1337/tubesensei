"""
Ideas API Endpoints

REST API for managing extracted ideas.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import require_permission, Permission
from app.database import get_db
from app.models.idea import Idea, IdeaStatus as DBIdeaStatus, IdeaPriority as DBIdeaPriority
from app.models.video import Video
from app.models.channel import Channel
from app.schemas.idea import (
    IdeaCreate,
    IdeaUpdate,
    IdeaResponse,
    IdeaListResponse,
    IdeaBulkAction,
    IdeaBulkActionResponse,
    IdeaWithContext,
    IdeaStatus,
    IdeaPriority,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ideas", tags=["Ideas"])


def _build_video_context(video: Video) -> dict:
    pub_at = getattr(video, "published_at", None)
    return {
        "id": str(video.id),
        "youtube_video_id": video.youtube_video_id,
        "title": video.title,
        "thumbnail_url": video.thumbnail_url,
        "published_at": pub_at.isoformat() if pub_at is not None else None,
        "duration_seconds": video.duration_seconds,
        "view_count": video.view_count,
    }


def _build_channel_context(channel: Channel) -> dict:
    return {
        "id": str(channel.id),
        "youtube_channel_id": channel.youtube_channel_id,
        "name": channel.name,
        "channel_handle": channel.channel_handle,
        "thumbnail_url": channel.thumbnail_url,
    }


def _idea_to_with_context(idea: Idea) -> IdeaWithContext:
    video = idea.video
    channel = video.channel if video else None
    return IdeaWithContext(
        **IdeaResponse.model_validate(idea).model_dump(),
        video=_build_video_context(video) if video else {},
        channel=_build_channel_context(channel) if channel else {},
        transcript_excerpt=None,
    )


@router.get("/", response_model=IdeaListResponse)
async def list_ideas(
    status: Optional[IdeaStatus] = Query(None, description="Filter by status"),
    priority: Optional[IdeaPriority] = Query(None, description="Filter by priority"),
    category: Optional[str] = Query(None, description="Filter by category"),
    video_id: Optional[UUID] = Query(None, description="Filter by video ID"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence score"),
    max_confidence: Optional[float] = Query(None, ge=0, le=1, description="Maximum confidence score"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List ideas with optional filters and pagination."""
    conditions = []

    if status:
        conditions.append(Idea.status == DBIdeaStatus(status.value))
    if priority:
        conditions.append(Idea.priority == DBIdeaPriority(priority.value))
    if category:
        conditions.append(Idea.category == category)
    if video_id:
        conditions.append(Idea.video_id == video_id)
    if min_confidence is not None:
        conditions.append(Idea.confidence_score >= min_confidence)
    if max_confidence is not None:
        conditions.append(Idea.confidence_score <= max_confidence)
    if search:
        search_term = f"%{search}%"
        conditions.append(
            or_(
                Idea.title.ilike(search_term),
                Idea.description.ilike(search_term),
            )
        )

    total_query = select(func.count(Idea.id))
    if conditions:
        total_query = total_query.where(*conditions)
    total_result = await db.execute(total_query)
    total: int = total_result.scalar() or 0

    query = (
        select(Idea)
        .join(Video, Idea.video_id == Video.id)
        .join(Channel, Video.channel_id == Channel.id)
    )
    if conditions:
        query = query.where(*conditions)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    ideas = result.scalars().all()

    items = [_idea_to_with_context(idea) for idea in ideas]

    return IdeaListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(ideas)) < total,
    )


@router.get("/{idea_id}", response_model=IdeaWithContext)
async def get_idea(
    idea_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific idea by ID."""
    result = await db.execute(
        select(Idea)
        .join(Video, Idea.video_id == Video.id)
        .join(Channel, Video.channel_id == Channel.id)
        .where(Idea.id == idea_id)
    )
    idea = result.scalar_one_or_none()
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")
    return _idea_to_with_context(idea)


@router.post("/", response_model=IdeaResponse, status_code=201)
async def create_idea(
    data: IdeaCreate,
    user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Create a new idea."""
    # Verify the video exists
    video_result = await db.execute(select(Video).where(Video.id == data.video_id))
    video = video_result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    idea = Idea(
        video_id=data.video_id,
        title=data.title,
        description=data.description,
        category=data.category,
        priority=DBIdeaPriority(data.priority.value),
        confidence_score=data.confidence_score,
        complexity_score=data.complexity_score,
        market_size_estimate=data.market_size_estimate,
        target_audience=data.target_audience,
        implementation_time_estimate=data.implementation_time_estimate,
        tags=data.tags,
        technologies=data.technologies,
        competitive_advantage=data.competitive_advantage,
        potential_challenges=data.potential_challenges,
        monetization_strategies=data.monetization_strategies,
        source_timestamp=data.source_timestamp,
        source_context=data.source_context,
        extraction_metadata=data.extraction_metadata,
        status=DBIdeaStatus.EXTRACTED,
    )
    db.add(idea)
    await db.commit()
    await db.refresh(idea)
    return IdeaResponse.model_validate(idea)


@router.put("/{idea_id}", response_model=IdeaResponse)
async def update_idea(
    idea_id: UUID,
    data: IdeaUpdate,
    user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Update an idea's status, priority, category, review notes, tags, or technologies."""
    result = await db.execute(select(Idea).where(Idea.id == idea_id))
    idea = result.scalar_one_or_none()
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")

    if data.status is not None:
        new_status = DBIdeaStatus(data.status.value)
        if new_status == DBIdeaStatus.SELECTED:
            idea.select(user.id)
        elif new_status == DBIdeaStatus.REJECTED:
            idea.reject(user.id)
        elif new_status == DBIdeaStatus.REVIEWED:
            idea.mark_as_reviewed(user.id, data.review_notes)
        else:
            idea.status = new_status
    if data.priority is not None:
        idea.priority = DBIdeaPriority(data.priority.value)
    if data.category is not None:
        idea.category = data.category
    if data.review_notes is not None:
        idea.review_notes = data.review_notes
    if data.tags is not None:
        idea.tags = data.tags
    if data.technologies is not None:
        idea.technologies = data.technologies

    await db.commit()
    await db.refresh(idea)
    return IdeaResponse.model_validate(idea)


@router.delete("/{idea_id}", status_code=204)
async def delete_idea(
    idea_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_DELETE)),
    db: AsyncSession = Depends(get_db),
):
    """Delete an idea."""
    result = await db.execute(select(Idea).where(Idea.id == idea_id))
    idea = result.scalar_one_or_none()
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")

    await db.delete(idea)
    await db.commit()


@router.post("/bulk-action", response_model=IdeaBulkActionResponse)
async def bulk_action_ideas(
    data: IdeaBulkAction,
    user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Perform a bulk action (select, reject, review, update_category) on multiple ideas."""
    updated = 0
    errors = []

    for idea_id in data.idea_ids:
        try:
            result = await db.execute(select(Idea).where(Idea.id == idea_id))
            idea = result.scalar_one_or_none()
            if not idea:
                errors.append({"idea_id": str(idea_id), "error": "Idea not found"})
                continue

            if data.action == "select":
                idea.select(user.id)
            elif data.action == "reject":
                idea.reject(user.id)
            elif data.action == "review":
                idea.mark_as_reviewed(user.id)
            elif data.action == "update_category":
                if data.category is None:
                    errors.append({"idea_id": str(idea_id), "error": "category is required for update_category action"})
                    continue
                idea.category = data.category

            updated += 1
        except Exception as exc:
            errors.append({"idea_id": str(idea_id), "error": str(exc)})

    await db.commit()
    return IdeaBulkActionResponse(updated=updated, errors=errors)
