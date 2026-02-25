"""
Channels API Endpoints

REST API for managing channels.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.permissions import require_permission, Permission
from app.database import get_db
from app.models.channel import Channel, ChannelStatus as DBChannelStatus
from app.models.video import Video
from app.models.idea import Idea, IdeaStatus as DBIdeaStatus
from app.schemas.channel import (
    ChannelResponse,
    ChannelListResponse,
    ChannelUpdate,
    ChannelStatus,
)
from app.schemas.video import VideoResponse, VideoListResponse
from app.schemas.idea import IdeaWithContext, IdeaListResponse, IdeaResponse, IdeaStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/channels", tags=["Channels"])


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


@router.get("/", response_model=ChannelListResponse)
async def list_channels(
    status: Optional[ChannelStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List channels with optional status filter and pagination."""
    conditions = []

    if status:
        conditions.append(Channel.status == DBChannelStatus(status.value))

    total_query = select(func.count(Channel.id))
    if conditions:
        total_query = total_query.where(*conditions)
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = select(Channel)
    if conditions:
        query = query.where(*conditions)
    query = query.order_by(Channel.priority_level.desc(), Channel.name).offset(offset).limit(limit)

    result = await db.execute(query)
    channels = result.scalars().all()

    return ChannelListResponse(
        items=[ChannelResponse.model_validate(c) for c in channels],
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(channels)) < total,
    )


@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific channel by ID."""
    result = await db.execute(select(Channel).where(Channel.id == channel_id))
    channel = result.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return ChannelResponse.model_validate(channel)


@router.put("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: UUID,
    data: ChannelUpdate,
    user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Update a channel's status, priority, scheduling, or processing config."""
    result = await db.execute(select(Channel).where(Channel.id == channel_id))
    channel = result.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    if data.status is not None:
        channel.status = DBChannelStatus(data.status.value)
    if data.priority_level is not None:
        channel.priority_level = data.priority_level
    if data.check_frequency_hours is not None:
        channel.check_frequency_hours = data.check_frequency_hours
    if data.auto_process is not None:
        channel.auto_process = data.auto_process
    if data.processing_config is not None:
        channel.processing_config = data.processing_config
    if data.tags is not None:
        channel.tags = data.tags
    if data.notes is not None:
        channel.notes = data.notes

    await db.commit()
    await db.refresh(channel)
    return ChannelResponse.model_validate(channel)


@router.delete("/{channel_id}", status_code=204)
async def delete_channel(
    channel_id: UUID,
    user=Depends(require_permission(Permission.CHANNEL_DELETE)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a channel and all associated videos and ideas."""
    result = await db.execute(select(Channel).where(Channel.id == channel_id))
    channel = result.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    await db.delete(channel)
    await db.commit()


@router.get("/{channel_id}/videos", response_model=VideoListResponse)
async def list_channel_videos(
    channel_id: UUID,
    status: Optional[str] = Query(None, description="Filter by video status"),
    has_captions: Optional[bool] = Query(None, description="Filter by caption availability"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List all videos for a specific channel."""
    channel_result = await db.execute(select(Channel).where(Channel.id == channel_id))
    channel = channel_result.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    from app.models.video import VideoStatus as DBVideoStatus

    conditions = [Video.channel_id == channel_id]
    if status:
        try:
            conditions.append(Video.status == DBVideoStatus(status))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status value: {status}")
    if has_captions is not None:
        conditions.append(Video.has_captions == has_captions)

    total_query = select(func.count(Video.id)).where(*conditions)
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = (
        select(Video)
        .where(*conditions)
        .order_by(Video.published_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    videos = result.scalars().all()

    return VideoListResponse(
        items=[VideoResponse.model_validate(v) for v in videos],
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(videos)) < total,
    )


@router.get("/{channel_id}/ideas", response_model=IdeaListResponse)
async def list_channel_ideas(
    channel_id: UUID,
    status: Optional[IdeaStatus] = Query(None, description="Filter by idea status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List all ideas from all videos belonging to a specific channel."""
    channel_result = await db.execute(select(Channel).where(Channel.id == channel_id))
    channel = channel_result.scalar_one_or_none()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    conditions = [Video.channel_id == channel_id]
    if status:
        conditions.append(Idea.status == DBIdeaStatus(status.value))

    total_query = (
        select(func.count(Idea.id))
        .join(Video, Idea.video_id == Video.id)
        .where(*conditions)
    )
    total_result = await db.execute(total_query)
    total = total_result.scalar()

    query = (
        select(Idea)
        .join(Video, Idea.video_id == Video.id)
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
