from fastapi import APIRouter, Depends, Query, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, Dict
from uuid import UUID
import re

from app.core.auth import get_current_user
from app.core.permissions import require_permission, Permission
from app.services.channel_service import ChannelService
from app.core.exceptions import NotFoundException, ValidationException
from app.schemas.channel import (
    ChannelCreate, 
    ChannelUpdate, 
    ChannelResponse,
    ChannelListResponse,
    ChannelSyncResponse
)
from app.database import get_db
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

# Set up templates
from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/channels", tags=["admin-channels"])


@router.get("/", response_class=HTMLResponse)
async def channels_page(
    request: Request,
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render channels management page"""
    service = ChannelService(db)
    
    limit = settings.admin.ADMIN_PAGINATION_DEFAULT
    offset = (page - 1) * limit
    
    # Convert string status to enum if provided
    channel_status = None
    if status:
        from app.models.channel import ChannelStatus as ModelChannelStatus
        try:
            channel_status = ModelChannelStatus(status)
        except ValueError:
            pass
    
    result = await service.list_channels(
        status=channel_status,
        search=search,
        limit=limit,
        offset=offset
    )
    
    total_pages = (result["total"] + limit - 1) // limit
    
    context = get_template_context(
        request,
        user=user,
        channels=result["items"],
        total=result["total"],
        page=page,
        total_pages=total_pages,
        filters={
            "status": status,
            "search": search
        }
    )
    
    return templates.TemplateResponse("admin/channels/list.html", context)


@router.get("/validate-url")
async def validate_channel_url(
    url: str = Query(..., description="YouTube channel URL to validate"),
    user = Depends(get_current_user)
):
    """Validate YouTube channel URL for HTMX"""
    if not url:
        return HTMLResponse('<span class="text-red-500 text-sm">URL is required</span>')
    
    # Simple URL validation
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/channel/[A-Za-z0-9_-]+',
        r'https?://(?:www\.)?youtube\.com/@[A-Za-z0-9_.-]+',
        r'https?://(?:www\.)?youtube\.com/c/[A-Za-z0-9_.-]+',
        r'https?://(?:www\.)?youtube\.com/user/[A-Za-z0-9_.-]+',
    ]
    
    is_valid_format = any(re.match(pattern, url) for pattern in youtube_patterns)
    
    if not is_valid_format:
        return HTMLResponse('<span class="text-red-500 text-sm">Please enter a valid YouTube channel URL</span>')
    
    # In development, just validate format
    from app.core.config import get_settings
    settings = get_settings()
    if settings.DEBUG:
        return HTMLResponse('<span class="text-green-500 text-sm">✓ Valid YouTube URL format</span>')
    
    # In production, this could validate with YouTube API
    return HTMLResponse('<span class="text-green-500 text-sm">✓ URL validation passed</span>')


@router.get("/add", response_class=HTMLResponse)
async def add_channel_form(
    request: Request,
    user = Depends(require_permission(Permission.CHANNEL_WRITE))
):
    """Render add channel form"""
    context = get_template_context(
        request,
        user=user
    )
    return templates.TemplateResponse("admin/channels/add.html", context)


@router.post("/add")
async def add_channel(
    url: Optional[str] = Form(None, description="YouTube channel URL"),
    name: Optional[str] = Form(None, description="Channel name (optional)"),
    description: Optional[str] = Form(None, description="Channel description (optional)"),
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    db = Depends(get_db)
):
    """Add new channel"""
    service = ChannelService(db)
    
    # Validate required fields
    if not url or url.strip() == "":
        return HTMLResponse(
            '<div class="text-red-500 p-2 bg-red-50 rounded">YouTube URL is required</div>',
            status_code=400
        )
    
    try:
        # Map form data to ChannelCreate schema
        channel_data = {
            "youtube_channel_id": url,  # Map url to youtube_channel_id
            "processing_config": {}
        }
        
        channel = await service.add_channel(channel_data)
        
        # Return HTMX-friendly response (empty HTML with redirect header)
        return HTMLResponse(
            content="",
            status_code=201,
            headers={
                "HX-Trigger": "channelAdded",
                "HX-Redirect": "/admin/channels/"
            }
        )
    except ValidationException as e:
        error_msg = "Validation failed"
        if hasattr(e, 'details') and e.details and 'errors' in e.details:
            # Extract the first error message from details.errors
            errors = e.details['errors']
            if isinstance(errors, dict):
                first_error = next(iter(errors.values()))
                error_msg = str(first_error)
            else:
                error_msg = str(errors)
        else:
            # Fallback to the exception message
            error_msg = str(e)
        
        return HTMLResponse(
            f'<div class="text-red-500 p-2 bg-red-50 rounded">{error_msg}</div>',
            status_code=400
        )
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in add_channel: {e}")
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        error_msg = str(e)
        return HTMLResponse(
            f'<div class="text-red-500 p-2 bg-red-50 rounded">Error adding channel: {error_msg}</div>',
            status_code=500
        )


@router.get("/{channel_id}", response_class=HTMLResponse)
async def channel_detail(
    request: Request,
    channel_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render channel detail page"""
    service = ChannelService(db)
    
    try:
        channel = await service.get_channel(str(channel_id))
        stats = await service._get_channel_stats(str(channel_id))
        
        return templates.TemplateResponse(
            "admin/channels/detail.html",
            {
                "request": request,
                "user": user,
                "channel": channel,
                "stats": stats
            }
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")


@router.get("/{channel_id}/edit", response_class=HTMLResponse)
async def edit_channel_form(
    request: Request,
    channel_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    db = Depends(get_db)
):
    """Render edit channel form"""
    service = ChannelService(db)
    
    try:
        channel = await service.get_channel(str(channel_id))
        
        return templates.TemplateResponse(
            "admin/channels/edit.html",
            {
                "request": request,
                "user": user,
                "channel": channel
            }
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")


@router.patch("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: UUID,
    data: ChannelUpdate,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    db = Depends(get_db)
):
    """Update channel"""
    service = ChannelService(db)
    
    try:
        channel = await service.update_channel(str(channel_id), data.dict(exclude_unset=True))
        return ChannelResponse.from_orm(channel)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{channel_id}/sync", response_model=ChannelSyncResponse)
async def sync_channel(
    channel_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_WRITE)),
    db = Depends(get_db)
):
    """Manually sync channel"""
    service = ChannelService(db)
    
    try:
        result = await service.sync_channel(str(channel_id))
        return ChannelSyncResponse(**result)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{channel_id}")
async def delete_channel(
    channel_id: UUID,
    user = Depends(require_permission(Permission.CHANNEL_DELETE)),
    db = Depends(get_db)
):
    """Delete channel"""
    service = ChannelService(db)
    
    try:
        await service.delete_channel(str(channel_id))
        return {
            "success": True,
            "redirect": "/admin/channels"
        }
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoints for HTMX partial updates
@router.get("/{channel_id}/videos", response_class=HTMLResponse)
async def channel_videos_partial(
    request: Request,
    channel_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get channel videos as HTML partial."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.video import Video

    service = ChannelService(db)

    try:
        # Verify channel exists
        channel = await service.get_channel(str(channel_id))

        # Query videos for this channel
        stmt = (
            select(Video)
            .where(Video.channel_id == channel_id)
            .order_by(Video.published_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        videos = result.scalars().all()

        return templates.TemplateResponse(
            "admin/channels/partials/videos_list.html",
            {
                "request": request,
                "channel": channel,
                "videos": videos
            }
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")


@router.get("/{channel_id}/ideas", response_class=HTMLResponse)
async def channel_ideas_partial(
    request: Request,
    channel_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user = Depends(get_current_user),
    db = Depends(get_db),
):
    """Get channel ideas as HTML partial."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.video import Video
    from app.models.idea import Idea

    service = ChannelService(db)

    try:
        # Verify channel exists
        channel = await service.get_channel(str(channel_id))

        # Query ideas for videos belonging to this channel
        stmt = (
            select(Idea)
            .join(Video, Idea.video_id == Video.id)
            .where(Video.channel_id == channel_id)
            .order_by(Idea.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        ideas = result.scalars().all()

        return templates.TemplateResponse(
            "admin/channels/partials/ideas_list.html",
            {
                "request": request,
                "channel": channel,
                "ideas": ideas
            }
        )
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Channel not found")


# API endpoints for HTMX partial updates
@router.get("/partials/channel-card/{channel_id}", response_class=HTMLResponse)
async def channel_card_partial(
    request: Request,
    channel_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get channel card HTML partial for HTMX updates"""
    service = ChannelService(db)

    try:
        channel = await service.get_channel(str(channel_id))
        stats = await service._get_channel_stats(str(channel_id))
        channel_dict = channel.to_dict()
        channel_dict["stats"] = stats

        return templates.TemplateResponse(
            "admin/channels/partials/channel_card.html",
            {
                "request": request,
                "channel": channel_dict
            }
        )
    except NotFoundException:
        return HTMLResponse(content="", status_code=204)  # No content for deleted channels