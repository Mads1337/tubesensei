from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional, Dict
from uuid import UUID

from app.core.auth import require_auth
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

# Set up templates
templates = Jinja2Templates(directory=settings.admin.TEMPLATE_DIR)

router = APIRouter(prefix="/channels", tags=["admin-channels"])


@router.get("/", response_class=HTMLResponse)
async def channels_page(
    request: Request,
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    user = Depends(require_auth),
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
    
    return templates.TemplateResponse(
        "admin/channels/list.html",
        {
            "request": request,
            "user": user,
            "channels": result["items"],
            "total": result["total"],
            "page": page,
            "total_pages": total_pages,
            "filters": {
                "status": status,
                "search": search
            }
        }
    )


@router.get("/add", response_class=HTMLResponse)
async def add_channel_form(
    request: Request,
    user = Depends(require_permission(Permission.CHANNEL_CREATE))
):
    """Render add channel form"""
    return templates.TemplateResponse(
        "admin/channels/add.html",
        {"request": request, "user": user}
    )


@router.post("/add", response_model=Dict)
async def add_channel(
    data: ChannelCreate,
    user = Depends(require_permission(Permission.CHANNEL_CREATE)),
    db = Depends(get_db)
):
    """Add new channel"""
    service = ChannelService(db)
    
    try:
        channel = await service.add_channel(data.dict())
        
        return {
            "success": True,
            "channel": ChannelResponse.from_orm(channel),
            "redirect": f"/admin/channels/{channel.id}"
        }
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.errors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{channel_id}", response_class=HTMLResponse)
async def channel_detail(
    request: Request,
    channel_id: UUID,
    user = Depends(require_auth),
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
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
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
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
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
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
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
@router.get("/partials/channel-card/{channel_id}", response_class=HTMLResponse)
async def channel_card_partial(
    request: Request,
    channel_id: UUID,
    user = Depends(require_auth),
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