"""
Videos API Endpoints

REST API for managing videos.
"""
import logging
from typing import Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.permissions import require_permission, Permission
from app.database import get_db
from app.models.video import Video, VideoStatus as DBVideoStatus
from app.models.idea import Idea, IdeaStatus as DBIdeaStatus
from app.models.channel import Channel
from app.schemas.video import VideoResponse, VideoListResponse
from app.schemas.idea import IdeaWithContext, IdeaListResponse, IdeaResponse, IdeaStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["Videos"])


def _build_channel_context(channel: Channel) -> dict:
    return {
        "id": str(channel.id),
        "youtube_channel_id": channel.youtube_channel_id,
        "name": channel.name,
        "channel_handle": channel.channel_handle,
        "thumbnail_url": channel.thumbnail_url,
    }


def _build_video_context(video: Video) -> dict:
    return {
        "id": str(video.id),
        "youtube_video_id": video.youtube_video_id,
        "title": video.title,
        "thumbnail_url": video.thumbnail_url,
        "published_at": video.published_at.isoformat() if video.published_at else None,
        "duration_seconds": video.duration_seconds,
        "view_count": video.view_count,
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


@router.get("/", response_model=VideoListResponse)
async def list_videos(
    channel_id: Optional[UUID] = Query(None, description="Filter by channel ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    has_captions: Optional[bool] = Query(None, description="Filter by caption availability"),
    published_after: Optional[datetime] = Query(None, description="Filter videos published after this date"),
    published_before: Optional[datetime] = Query(None, description="Filter videos published before this date"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List videos with optional filters and pagination."""
    conditions = []

    if channel_id:
        conditions.append(Video.channel_id == channel_id)
    if status:
        try:
            db_status = DBVideoStatus(status)
            conditions.append(Video.status == db_status)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status value: {status}")
    if has_captions is not None:
        conditions.append(Video.has_captions == has_captions)
    if published_after:
        conditions.append(Video.published_at >= published_after)
    if published_before:
        conditions.append(Video.published_at <= published_before)

    total_query = select(func.count(Video.id))
    if conditions:
        total_query = total_query.where(*conditions)
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = select(Video)
    if conditions:
        query = query.where(*conditions)
    query = query.order_by(Video.published_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    videos = result.scalars().all()

    return VideoListResponse(
        items=[VideoResponse.model_validate(v) for v in videos],
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(videos)) < total,
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific video by ID."""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse.model_validate(video)


@router.delete("/{video_id}", status_code=204)
async def delete_video(
    video_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_DELETE)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a video and all its associated ideas and transcripts."""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    await db.delete(video)
    await db.commit()


@router.get("/{video_id}/ideas", response_model=IdeaListResponse)
async def list_video_ideas(
    video_id: UUID,
    status: Optional[IdeaStatus] = Query(None, description="Filter by idea status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List all ideas extracted from a specific video."""
    video_result = await db.execute(select(Video).where(Video.id == video_id))
    video = video_result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    conditions = [Idea.video_id == video_id]
    if status:
        conditions.append(Idea.status == DBIdeaStatus(status.value))

    total_query = select(func.count(Idea.id)).where(*conditions)
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = (
        select(Idea)
        .where(*conditions)
        .offset(offset)
        .limit(limit)
    )
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
