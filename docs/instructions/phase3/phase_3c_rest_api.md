# TubeSensei Phase 3C: REST API Development
## Week 9 - Days 1-3: API Implementation

### Version: 1.0
### Duration: 3 Days
### Dependencies: Phase 3A & 3B Complete (Admin Core & UI)

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Day 1: API Structure & Core Endpoints](#day-1-api-structure--core-endpoints)
3. [Day 2: Advanced Endpoints & Filtering](#day-2-advanced-endpoints--filtering)
4. [Day 3: Authentication & Rate Limiting](#day-3-authentication--rate-limiting)
5. [Implementation Checklist](#implementation-checklist)
6. [Testing Requirements](#testing-requirements)

---

## Phase Overview

### Objectives
Build a comprehensive REST API for external integrations with proper versioning, authentication, rate limiting, and documentation.

### Deliverables
- Versioned API structure (v1)
- Core resource endpoints (channels, videos, ideas)
- Advanced filtering and search
- API key authentication system
- Rate limiting implementation
- Webhook system for notifications
- OpenAPI/Swagger documentation

### API Design Principles
- RESTful design patterns
- Consistent response formats
- Proper HTTP status codes
- Pagination for list endpoints
- Filtering and sorting capabilities
- Clear error messages

---

## Day 1: API Structure & Core Endpoints

### 1.1 API Router Structure

```python
# app/api/v1/__init__.py
from fastapi import APIRouter
from app.api.v1 import (
    channels,
    videos,
    transcripts,
    ideas,
    jobs,
    webhooks,
    export
)

api_v1_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_v1_router.include_router(channels.router, prefix="/channels", tags=["channels"])
api_v1_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_v1_router.include_router(transcripts.router, prefix="/transcripts", tags=["transcripts"])
api_v1_router.include_router(ideas.router, prefix="/ideas", tags=["ideas"])
api_v1_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_v1_router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
api_v1_router.include_router(export.router, prefix="/export", tags=["export"])
```

### 1.2 Response Models

```python
# app/schemas/api_responses.py
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List, Optional, Any
from datetime import datetime
from uuid import UUID

T = TypeVar('T')

class PaginationMeta(BaseModel):
    """Pagination metadata"""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    page: int = Field(..., description="Current page number")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")

class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    success: bool = Field(True, description="Request success status")
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Optional message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response"""
    success: bool = Field(True)
    data: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchOperationResponse(BaseModel):
    """Response for batch operations"""
    success: bool
    processed: int = Field(..., description="Number of items processed")
    succeeded: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    errors: List[dict] = Field(default_factory=list, description="List of errors")
    results: Optional[List[dict]] = Field(None, description="Individual results")
```

### 1.3 Channel API Endpoints

```python
# app/api/v1/channels.py
from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Optional, List
from uuid import UUID

from app.schemas.channel import ChannelResponse, ChannelCreate, ChannelUpdate
from app.schemas.api_responses import APIResponse, PaginatedResponse, ErrorResponse
from app.services.channel_service import ChannelService
from app.core.database import get_db
from app.core.api_auth import require_api_key
from app.models.api_key import APIKey

router = APIRouter()

@router.get(
    "/",
    response_model=PaginatedResponse[ChannelResponse],
    summary="List all channels",
    description="Get a paginated list of monitored YouTube channels with optional filtering"
)
async def list_channels(
    status: Optional[str] = Query(None, description="Filter by channel status"),
    search: Optional[str] = Query(None, description="Search in channel name or description"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=1000, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    List all monitored channels with filtering and pagination.
    
    **Query Parameters:**
    - `status`: Filter by channel status (active, paused, inactive)
    - `search`: Search term for channel name or description
    - `sort_by`: Field to sort by (created_at, name, subscriber_count)
    - `sort_order`: Sort order (asc, desc)
    - `limit`: Number of items per page (max 1000)
    - `offset`: Pagination offset
    """
    service = ChannelService(db)
    
    result = await service.list_channels(
        status=status,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset
    )
    
    # Calculate pagination metadata
    total = result["total"]
    pages = (total + limit - 1) // limit
    page = (offset // limit) + 1
    
    return PaginatedResponse(
        data=result["items"],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            page=page,
            pages=pages,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    )

@router.get(
    "/{channel_id}",
    response_model=APIResponse[ChannelResponse],
    summary="Get channel by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Channel not found"}
    }
)
async def get_channel(
    channel_id: UUID,
    include_stats: bool = Query(False, description="Include processing statistics"),
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get detailed information about a specific channel.
    
    **Path Parameters:**
    - `channel_id`: UUID of the channel
    
    **Query Parameters:**
    - `include_stats`: Include processing statistics in response
    """
    service = ChannelService(db)
    
    try:
        channel = await service.get_channel(str(channel_id))
        channel_data = ChannelResponse.from_orm(channel)
        
        if include_stats:
            stats = await service._get_channel_stats(str(channel_id))
            channel_data = {**channel_data.dict(), "stats": stats}
        
        return APIResponse(data=channel_data)
        
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post(
    "/",
    response_model=APIResponse[ChannelResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Add new channel",
    description="Add a new YouTube channel for monitoring"
)
async def create_channel(
    channel_data: ChannelCreate,
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Add a new YouTube channel for monitoring.
    
    **Request Body:**
    - `youtube_channel_id`: YouTube channel ID or URL
    - `processing_config`: Optional processing configuration
    
    This endpoint will:
    1. Validate the YouTube channel exists
    2. Fetch channel metadata from YouTube
    3. Create the channel record
    4. Queue initial video discovery
    """
    service = ChannelService(db)
    
    try:
        channel = await service.add_channel(channel_data)
        return APIResponse(
            data=ChannelResponse.from_orm(channel),
            message="Channel added successfully and queued for processing"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.details
        )

@router.patch(
    "/{channel_id}",
    response_model=APIResponse[ChannelResponse],
    summary="Update channel"
)
async def update_channel(
    channel_id: UUID,
    updates: ChannelUpdate,
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """Update channel configuration and settings."""
    service = ChannelService(db)
    
    try:
        channel = await service.update_channel(str(channel_id), updates)
        return APIResponse(
            data=ChannelResponse.from_orm(channel),
            message="Channel updated successfully"
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.delete(
    "/{channel_id}",
    response_model=APIResponse[dict],
    summary="Delete channel"
)
async def delete_channel(
    channel_id: UUID,
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """Soft delete a channel (marks as inactive)."""
    service = ChannelService(db)
    
    try:
        await service.delete_channel(str(channel_id))
        return APIResponse(
            data={"channel_id": str(channel_id)},
            message="Channel deleted successfully"
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post(
    "/{channel_id}/sync",
    response_model=APIResponse[dict],
    summary="Sync channel"
)
async def sync_channel(
    channel_id: UUID,
    full_sync: bool = Query(False, description="Perform full channel sync"),
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Manually trigger channel synchronization.
    
    This will:
    1. Update channel metadata from YouTube
    2. Queue video discovery job
    3. Return job ID for tracking
    """
    service = ChannelService(db)
    
    try:
        result = await service.sync_channel(str(channel_id), full_sync=full_sync)
        return APIResponse(
            data=result,
            message="Channel sync initiated"
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.get(
    "/{channel_id}/videos",
    response_model=PaginatedResponse[dict],
    summary="Get channel videos"
)
async def get_channel_videos(
    channel_id: UUID,
    status: Optional[str] = Query(None, description="Filter by video status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key: APIKey = Depends(require_api_key),
    db = Depends(get_db)
):
    """Get all videos from a specific channel."""
    from app.services.video_service import VideoService
    
    service = VideoService(db)
    
    result = await service.list_videos(
        channel_id=str(channel_id),
        status=status,
        limit=limit,
        offset=offset
    )
    
    # Calculate pagination
    total = result["total"]
    pages = (total + limit - 1) // limit
    page = (offset // limit) + 1
    
    return PaginatedResponse(
        data=result["items"],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            page=page,
            pages=pages,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    )
```

### 1.4 Video API Endpoints

```python
# app/api/v1/videos.py
from fastapi import APIRouter, Depends, Query, HTTPException, status, Body
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from app.schemas.video import VideoResponse, VideoUpdate
from app.schemas.api_responses import APIResponse, PaginatedResponse, BatchOperationResponse
from app.services.video_service import VideoService
from app.core.api_auth import require_api_key

router = APIRouter()

@router.get(
    "/",
    response_model=PaginatedResponse[VideoResponse],
    summary="List all videos"
)
async def list_videos(
    channel_id: Optional[UUID] = Query(None, description="Filter by channel"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    min_duration: Optional[int] = Query(None, description="Minimum duration in seconds"),
    max_duration: Optional[int] = Query(None, description="Maximum duration in seconds"),
    published_after: Optional[datetime] = Query(None, description="Videos published after date"),
    published_before: Optional[datetime] = Query(None, description="Videos published before date"),
    search: Optional[str] = Query(None, description="Search in title or description"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    List all videos with advanced filtering options.
    
    **Filters:**
    - Channel, status, duration range
    - Publication date range
    - Full-text search in title/description
    """
    service = VideoService(db)
    
    result = await service.list_videos(
        channel_id=str(channel_id) if channel_id else None,
        status=status,
        min_duration=min_duration,
        max_duration=max_duration,
        published_after=published_after,
        published_before=published_before,
        search=search,
        limit=limit,
        offset=offset
    )
    
    total = result["total"]
    pages = (total + limit - 1) // limit
    page = (offset // limit) + 1
    
    return PaginatedResponse(
        data=result["items"],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            page=page,
            pages=pages,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    )

@router.get(
    "/{video_id}",
    response_model=APIResponse[VideoResponse],
    summary="Get video by ID"
)
async def get_video(
    video_id: UUID,
    include_transcript: bool = Query(False, description="Include transcript if available"),
    include_ideas: bool = Query(False, description="Include extracted ideas"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get detailed information about a specific video.
    
    **Options:**
    - `include_transcript`: Include full transcript text
    - `include_ideas`: Include all extracted ideas
    """
    service = VideoService(db)
    
    try:
        video = await service.get_video(
            str(video_id),
            include_transcript=include_transcript,
            include_ideas=include_ideas
        )
        return APIResponse(data=video)
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post(
    "/process",
    response_model=APIResponse[dict],
    summary="Submit video for processing"
)
async def process_video(
    video_url: str = Body(..., description="YouTube video URL"),
    priority: int = Body(0, ge=0, le=10, description="Processing priority"),
    extract_ideas: bool = Body(True, description="Extract ideas from video"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Submit a single video for processing.
    
    **Request Body:**
    - `video_url`: Full YouTube video URL
    - `priority`: Processing priority (0-10, higher = sooner)
    - `extract_ideas`: Whether to extract ideas after transcription
    """
    service = VideoService(db)
    
    try:
        result = await service.submit_for_processing(
            video_url=video_url,
            priority=priority,
            extract_ideas=extract_ideas
        )
        return APIResponse(
            data=result,
            message="Video queued for processing"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.post(
    "/batch-process",
    response_model=BatchOperationResponse,
    summary="Submit multiple videos for processing"
)
async def batch_process_videos(
    video_urls: List[str] = Body(..., description="List of YouTube video URLs"),
    priority: int = Body(0, ge=0, le=10),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Submit multiple videos for processing in batch.
    
    **Request Body:**
    - `video_urls`: List of YouTube video URLs (max 100)
    - `priority`: Processing priority for all videos
    """
    if len(video_urls) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 100 videos per batch"
        )
    
    service = VideoService(db)
    
    results = await service.batch_submit_for_processing(
        video_urls=video_urls,
        priority=priority
    )
    
    return BatchOperationResponse(
        success=True,
        processed=len(video_urls),
        succeeded=results["succeeded"],
        failed=results["failed"],
        errors=results["errors"],
        results=results["results"]
    )

@router.get(
    "/{video_id}/transcript",
    response_model=APIResponse[dict],
    summary="Get video transcript"
)
async def get_video_transcript(
    video_id: UUID,
    format: str = Query("text", regex="^(text|srt|json)$", description="Transcript format"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get transcript for a video.
    
    **Formats:**
    - `text`: Plain text transcript
    - `srt`: SubRip subtitle format
    - `json`: Structured JSON with timestamps
    """
    from app.services.transcript_service import TranscriptService
    
    service = TranscriptService(db)
    
    try:
        transcript = await service.get_transcript(
            str(video_id),
            format=format
        )
        return APIResponse(data=transcript)
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found for this video"
        )
```

### 1.5 Ideas API Endpoints

```python
# app/api/v1/ideas.py
from fastapi import APIRouter, Depends, Query, HTTPException, status, Body
from typing import Optional, List
from uuid import UUID

from app.schemas.idea import IdeaResponse, IdeaUpdate
from app.schemas.api_responses import APIResponse, PaginatedResponse, BatchOperationResponse
from app.services.idea_service import IdeaService
from app.core.api_auth import require_api_key

router = APIRouter()

@router.get(
    "/",
    response_model=PaginatedResponse[IdeaResponse],
    summary="List all ideas"
)
async def list_ideas(
    status: Optional[str] = Query(None, description="Filter by idea status"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score"),
    category: Optional[str] = Query(None, description="Filter by category"),
    channel_id: Optional[UUID] = Query(None, description="Filter by source channel"),
    video_id: Optional[UUID] = Query(None, description="Filter by source video"),
    search: Optional[str] = Query(None, description="Search in title or description"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("confidence_score", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    List all extracted ideas with comprehensive filtering.
    
    **Filters:**
    - Status (extracted, reviewed, selected, rejected)
    - Minimum confidence score (0.0 to 1.0)
    - Category and tags
    - Source channel or video
    - Full-text search
    """
    service = IdeaService(db)
    
    result = await service.list_ideas(
        status=status,
        min_confidence=min_confidence,
        category=category,
        channel_id=str(channel_id) if channel_id else None,
        video_id=str(video_id) if video_id else None,
        search=search,
        tags=tags,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset
    )
    
    total = result["total"]
    pages = (total + limit - 1) // limit
    page = (offset // limit) + 1
    
    return PaginatedResponse(
        data=result["items"],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            page=page,
            pages=pages,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    )

@router.get(
    "/categories",
    response_model=APIResponse[List[str]],
    summary="Get all idea categories"
)
async def get_categories(
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """Get list of all unique idea categories."""
    service = IdeaService(db)
    categories = await service.get_categories()
    return APIResponse(data=categories)

@router.get(
    "/statistics",
    response_model=APIResponse[dict],
    summary="Get idea statistics"
)
async def get_idea_statistics(
    channel_id: Optional[UUID] = Query(None, description="Filter by channel"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get statistical summary of ideas.
    
    **Returns:**
    - Total ideas by status
    - Average confidence scores
    - Ideas by category
    - Top performing channels
    """
    service = IdeaService(db)
    
    stats = await service.get_statistics(
        channel_id=str(channel_id) if channel_id else None,
        date_from=date_from,
        date_to=date_to
    )
    
    return APIResponse(data=stats)

@router.get(
    "/{idea_id}",
    response_model=APIResponse[IdeaResponse],
    summary="Get idea by ID"
)
async def get_idea(
    idea_id: UUID,
    include_context: bool = Query(False, description="Include source video and channel info"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get detailed information about a specific idea.
    
    **Options:**
    - `include_context`: Include source video, channel, and transcript excerpt
    """
    service = IdeaService(db)
    
    try:
        if include_context:
            idea_data = await service.get_idea_context(str(idea_id))
        else:
            idea = await service.get_idea(str(idea_id))
            idea_data = IdeaResponse.from_orm(idea)
        
        return APIResponse(data=idea_data)
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.patch(
    "/{idea_id}",
    response_model=APIResponse[IdeaResponse],
    summary="Update idea"
)
async def update_idea(
    idea_id: UUID,
    updates: IdeaUpdate,
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """Update idea properties (status, category, tags, etc.)."""
    service = IdeaService(db)
    
    try:
        idea = await service.update_idea(str(idea_id), updates)
        return APIResponse(
            data=IdeaResponse.from_orm(idea),
            message="Idea updated successfully"
        )
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post(
    "/batch-update",
    response_model=BatchOperationResponse,
    summary="Batch update ideas"
)
async def batch_update_ideas(
    idea_ids: List[UUID] = Body(..., description="List of idea IDs"),
    action: str = Body(..., regex="^(select|reject|review)$", description="Action to perform"),
    category: Optional[str] = Body(None, description="New category (for update_category action)"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Perform batch operations on multiple ideas.
    
    **Actions:**
    - `select`: Mark ideas as selected
    - `reject`: Mark ideas as rejected
    - `review`: Mark ideas as reviewed
    """
    service = IdeaService(db)
    
    result = await service.bulk_update(
        idea_ids=[str(id) for id in idea_ids],
        action=action,
        category=category
    )
    
    return BatchOperationResponse(
        success=True,
        processed=len(idea_ids),
        succeeded=result["updated"],
        failed=len(result["errors"]),
        errors=result["errors"]
    )
```

---

## Day 2: Advanced Endpoints & Filtering

### 2.1 Search and Filter Service

```python
# app/services/search_service.py
from typing import Dict, Any, List, Optional
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.models.video import Video
from app.models.idea import Idea
from app.models.channel import Channel
from app.models.transcript import Transcript

class SearchService:
    """Advanced search and filtering service"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def search_global(
        self,
        query: str,
        types: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Global search across all content types.
        
        Args:
            query: Search query string
            types: Content types to search (channels, videos, ideas)
            limit: Maximum results per type
            offset: Pagination offset
        """
        if not types:
            types = ["channels", "videos", "ideas"]
        
        results = {}
        
        if "channels" in types:
            results["channels"] = await self._search_channels(query, limit, offset)
        
        if "videos" in types:
            results["videos"] = await self._search_videos(query, limit, offset)
        
        if "ideas" in types:
            results["ideas"] = await self._search_ideas(query, limit, offset)
        
        return {
            "query": query,
            "types": types,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _search_channels(
        self,
        query: str,
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """Search channels by name or description"""
        # Use PostgreSQL full-text search
        search_query = select(Channel).where(
            or_(
                Channel.name.ilike(f"%{query}%"),
                Channel.description.ilike(f"%{query}%")
            )
        ).limit(limit).offset(offset)
        
        result = await self.db.execute(search_query)
        channels = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(Channel.id)).where(
            or_(
                Channel.name.ilike(f"%{query}%"),
                Channel.description.ilike(f"%{query}%")
            )
        )
        total = await self.db.scalar(count_query)
        
        return {
            "items": [self._serialize_channel(c) for c in channels],
            "total": total
        }
    
    async def _search_videos(
        self,
        query: str,
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """Search videos by title or description"""
        search_query = select(Video).where(
            or_(
                Video.title.ilike(f"%{query}%"),
                Video.description.ilike(f"%{query}%")
            )
        ).limit(limit).offset(offset)
        
        result = await self.db.execute(search_query)
        videos = result.scalars().all()
        
        count_query = select(func.count(Video.id)).where(
            or_(
                Video.title.ilike(f"%{query}%"),
                Video.description.ilike(f"%{query}%")
            )
        )
        total = await self.db.scalar(count_query)
        
        return {
            "items": [self._serialize_video(v) for v in videos],
            "total": total
        }
    
    async def _search_ideas(
        self,
        query: str,
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """Search ideas by title, description, or tags"""
        # Search in title, description, and tags array
        search_query = select(Idea).where(
            or_(
                Idea.title.ilike(f"%{query}%"),
                Idea.description.ilike(f"%{query}%"),
                text(f"'{query}' = ANY(tags)")  # PostgreSQL array search
            )
        ).limit(limit).offset(offset)
        
        result = await self.db.execute(search_query)
        ideas = result.scalars().all()
        
        count_query = select(func.count(Idea.id)).where(
            or_(
                Idea.title.ilike(f"%{query}%"),
                Idea.description.ilike(f"%{query}%"),
                text(f"'{query}' = ANY(tags)")
            )
        )
        total = await self.db.scalar(count_query)
        
        return {
            "items": [self._serialize_idea(i) for i in ideas],
            "total": total
        }
    
    async def advanced_idea_search(
        self,
        filters: Dict[str, Any]
    ) -> List[Idea]:
        """
        Advanced idea search with multiple filter criteria.
        
        Filters:
            - confidence_range: (min, max) tuple
            - complexity_range: (min, max) tuple
            - date_range: (start, end) datetime tuple
            - categories: List of categories
            - tags: List of required tags
            - exclude_tags: List of excluded tags
        """
        query = select(Idea)
        conditions = []
        
        # Confidence range
        if "confidence_range" in filters:
            min_conf, max_conf = filters["confidence_range"]
            conditions.append(
                and_(
                    Idea.confidence_score >= min_conf,
                    Idea.confidence_score <= max_conf
                )
            )
        
        # Complexity range
        if "complexity_range" in filters:
            min_comp, max_comp = filters["complexity_range"]
            conditions.append(
                and_(
                    Idea.complexity_score >= min_comp,
                    Idea.complexity_score <= max_comp
                )
            )
        
        # Date range
        if "date_range" in filters:
            start_date, end_date = filters["date_range"]
            conditions.append(
                and_(
                    Idea.created_at >= start_date,
                    Idea.created_at <= end_date
                )
            )
        
        # Categories
        if "categories" in filters:
            conditions.append(Idea.category.in_(filters["categories"]))
        
        # Tags (all must be present)
        if "tags" in filters:
            for tag in filters["tags"]:
                conditions.append(text(f"'{tag}' = ANY(tags)"))
        
        # Exclude tags (none should be present)
        if "exclude_tags" in filters:
            for tag in filters["exclude_tags"]:
                conditions.append(text(f"NOT ('{tag}' = ANY(tags))"))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    def _serialize_channel(self, channel: Channel) -> dict:
        """Serialize channel for search results"""
        return {
            "id": str(channel.id),
            "name": channel.name,
            "description": channel.description[:200] if channel.description else None,
            "subscriber_count": channel.subscriber_count,
            "type": "channel"
        }
    
    def _serialize_video(self, video: Video) -> dict:
        """Serialize video for search results"""
        return {
            "id": str(video.id),
            "title": video.title,
            "description": video.description[:200] if video.description else None,
            "duration": video.duration_seconds,
            "views": video.view_count,
            "type": "video"
        }
    
    def _serialize_idea(self, idea: Idea) -> dict:
        """Serialize idea for search results"""
        return {
            "id": str(idea.id),
            "title": idea.title,
            "description": idea.description[:200],
            "confidence": idea.confidence_score,
            "category": idea.category,
            "type": "idea"
        }
```

### 2.2 Search API Endpoints

```python
# app/api/v1/search.py
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List
from datetime import datetime

from app.schemas.api_responses import APIResponse
from app.services.search_service import SearchService
from app.core.api_auth import require_api_key
from app.core.database import get_db

router = APIRouter()

@router.get(
    "/",
    response_model=APIResponse[dict],
    summary="Global search",
    description="Search across all content types"
)
async def global_search(
    q: str = Query(..., min_length=2, description="Search query"),
    types: Optional[List[str]] = Query(None, description="Content types to search"),
    limit: int = Query(10, ge=1, le=100, description="Max results per type"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Perform a global search across channels, videos, and ideas.
    
    **Parameters:**
    - `q`: Search query (minimum 2 characters)
    - `types`: Content types to search (channels, videos, ideas)
    - `limit`: Maximum results per content type
    
    **Returns:**
    Search results grouped by content type with relevance scoring.
    """
    service = SearchService(db)
    
    results = await service.search_global(
        query=q,
        types=types,
        limit=limit
    )
    
    return APIResponse(data=results)

@router.post(
    "/advanced",
    response_model=APIResponse[dict],
    summary="Advanced search",
    description="Advanced search with complex filters"
)
async def advanced_search(
    filters: dict,
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Advanced search with multiple filter criteria.
    
    **Filter Options:**
    ```json
    {
        "content_type": "ideas",
        "confidence_range": [0.7, 1.0],
        "complexity_range": [1, 5],
        "date_range": ["2024-01-01", "2024-12-31"],
        "categories": ["Technology", "Business"],
        "tags": ["ai", "automation"],
        "exclude_tags": ["crypto"],
        "channels": ["channel_id_1", "channel_id_2"]
    }
    ```
    """
    service = SearchService(db)
    
    try:
        results = await service.advanced_idea_search(filters)
        
        return APIResponse(
            data={
                "filters": filters,
                "results": results,
                "count": len(results)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filter parameters: {str(e)}"
        )

@router.get(
    "/suggestions",
    response_model=APIResponse[List[str]],
    summary="Search suggestions",
    description="Get search suggestions based on partial query"
)
async def search_suggestions(
    q: str = Query(..., min_length=1, max_length=50),
    type: str = Query("all", description="Content type for suggestions"),
    limit: int = Query(10, ge=1, le=20),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get search suggestions for autocomplete.
    
    Returns top matching terms from:
    - Channel names
    - Video titles
    - Idea titles and categories
    - Popular tags
    """
    service = SearchService(db)
    
    suggestions = await service.get_suggestions(
        query=q,
        content_type=type,
        limit=limit
    )
    
    return APIResponse(data=suggestions)
```

### 2.3 Job Queue API

```python
# app/api/v1/jobs.py
from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from app.schemas.job import JobResponse, JobCreate
from app.schemas.api_responses import APIResponse, PaginatedResponse
from app.services.job_service import JobService
from app.core.api_auth import require_api_key
from app.core.database import get_db

router = APIRouter()

@router.get(
    "/",
    response_model=PaginatedResponse[JobResponse],
    summary="List processing jobs"
)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    entity_id: Optional[UUID] = Query(None, description="Filter by entity ID"),
    created_after: Optional[datetime] = Query(None),
    created_before: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    List processing jobs with filtering.
    
    **Filters:**
    - Status: queued, running, completed, failed, cancelled
    - Job type: channel_discovery, video_filtering, transcript_extraction, idea_extraction
    - Entity ID: Related channel, video, or idea ID
    - Date range: Filter by creation date
    """
    service = JobService(db)
    
    result = await service.list_jobs(
        status=status,
        job_type=job_type,
        entity_id=str(entity_id) if entity_id else None,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset
    )
    
    total = result["total"]
    pages = (total + limit - 1) // limit
    page = (offset // limit) + 1
    
    return PaginatedResponse(
        data=result["items"],
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            page=page,
            pages=pages,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    )

@router.get(
    "/{job_id}",
    response_model=APIResponse[JobResponse],
    summary="Get job details"
)
async def get_job(
    job_id: UUID,
    include_result: bool = Query(False, description="Include job result data"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get detailed information about a specific job.
    
    **Options:**
    - `include_result`: Include full job result data (can be large)
    """
    service = JobService(db)
    
    try:
        job = await service.get_job(
            str(job_id),
            include_result=include_result
        )
        return APIResponse(data=job)
    except NotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post(
    "/{job_id}/cancel",
    response_model=APIResponse[dict],
    summary="Cancel job"
)
async def cancel_job(
    job_id: UUID,
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Cancel a queued or running job.
    
    Note: Running jobs may not stop immediately.
    """
    service = JobService(db)
    
    try:
        result = await service.cancel_job(str(job_id))
        return APIResponse(
            data=result,
            message="Job cancellation requested"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post(
    "/{job_id}/retry",
    response_model=APIResponse[dict],
    summary="Retry failed job"
)
async def retry_job(
    job_id: UUID,
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Retry a failed job.
    
    Creates a new job with the same parameters.
    """
    service = JobService(db)
    
    try:
        new_job = await service.retry_job(str(job_id))
        return APIResponse(
            data={"new_job_id": str(new_job.id)},
            message="Job retry initiated"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.get(
    "/stats/summary",
    response_model=APIResponse[dict],
    summary="Get job statistics"
)
async def get_job_stats(
    time_range: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Get job processing statistics.
    
    **Time Ranges:**
    - 1h: Last hour
    - 24h: Last 24 hours
    - 7d: Last 7 days
    - 30d: Last 30 days
    """
    service = JobService(db)
    
    stats = await service.get_statistics(time_range)
    
    return APIResponse(data=stats)
```

---

## Day 3: Authentication & Rate Limiting

### 3.1 API Key Management

```python
# app/models/api_key.py
from sqlalchemy import Column, String, Boolean, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import secrets

from app.core.database import Base

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="api_keys")
    
    # Permissions and limits
    is_active = Column(Boolean, default=True, nullable=False)
    tier = Column(String, default="standard", nullable=False)  # standard, premium, unlimited
    rate_limit = Column(Integer, default=100)  # Requests per hour
    daily_limit = Column(Integer, default=10000)  # Requests per day
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime)
    
    @classmethod
    def generate_key(cls) -> str:
        """Generate a secure API key"""
        return f"ts_{secrets.token_urlsafe(32)}"
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True
```

### 3.2 API Authentication Service

```python
# app/core/api_auth.py
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import hashlib

from app.models.api_key import APIKey
from app.core.database import get_db
from app.core.cache import cache_manager

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class APIKeyAuth:
    """API Key authentication handler"""
    
    def __init__(self):
        self.cache = cache_manager
    
    async def __call__(
        self,
        request: Request,
        api_key_str: str = Security(api_key_header),
        db: AsyncSession = Depends(get_db)
    ) -> APIKey:
        """Validate API key and return key object"""
        if not api_key_str:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key required"
            )
        
        # Check cache first
        cache_key = f"api_key:{self._hash_key(api_key_str)}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            return APIKey(**cached)
        
        # Query database
        result = await db.execute(
            select(APIKey).where(APIKey.key == api_key_str)
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        
        if not api_key.is_valid():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key expired or inactive"
            )
        
        # Update usage
        api_key.last_used_at = datetime.utcnow()
        api_key.usage_count += 1
        await db.commit()
        
        # Cache for 5 minutes
        await self.cache.set(
            cache_key,
            {
                "id": str(api_key.id),
                "key": api_key.key,
                "tier": api_key.tier,
                "rate_limit": api_key.rate_limit,
                "daily_limit": api_key.daily_limit,
                "is_active": api_key.is_active
            },
            expire=300
        )
        
        # Store in request state for rate limiting
        request.state.api_key = api_key
        
        return api_key
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for caching"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

# Dependency
require_api_key = APIKeyAuth()

# Optional API key (for public endpoints with higher limits for authenticated users)
optional_api_key = APIKeyAuth()

async def get_optional_api_key(
    request: Request,
    api_key_str: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Optional[APIKey]:
    """Get API key if provided, None otherwise"""
    if not api_key_str:
        return None
    
    try:
        return await require_api_key(request, api_key_str, db)
    except HTTPException:
        return None
```

### 3.3 Rate Limiting Implementation

```python
# app/core/rate_limiter.py
from fastapi import Request, HTTPException, status
from typing import Optional, Callable
import time
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

from app.core.cache import cache_manager

class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self):
        self.cache = cache_manager
        self.local_cache = defaultdict(dict)  # Fallback local cache
    
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 3600
    ) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier (API key ID or IP)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if allowed, False if rate limited
        """
        bucket_key = f"rate_limit:{key}:{window_seconds}"
        
        try:
            # Try Redis first
            return await self._check_redis_limit(
                bucket_key,
                max_requests,
                window_seconds
            )
        except:
            # Fallback to local cache
            return self._check_local_limit(
                key,
                max_requests,
                window_seconds
            )
    
    async def _check_redis_limit(
        self,
        bucket_key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check rate limit using Redis"""
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Remove old entries
        await self.cache.redis.zremrangebyscore(
            bucket_key,
            0,
            window_start
        )
        
        # Count requests in current window
        request_count = await self.cache.redis.zcard(bucket_key)
        
        if request_count >= max_requests:
            return False
        
        # Add current request
        await self.cache.redis.zadd(
            bucket_key,
            {str(current_time): current_time}
        )
        
        # Set expiry
        await self.cache.redis.expire(bucket_key, window_seconds)
        
        return True
    
    def _check_local_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Fallback local rate limiting"""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        if key not in self.local_cache:
            self.local_cache[key] = []
        
        # Remove old entries
        self.local_cache[key] = [
            t for t in self.local_cache[key]
            if t > window_start
        ]
        
        if len(self.local_cache[key]) >= max_requests:
            return False
        
        self.local_cache[key].append(current_time)
        return True
    
    async def get_remaining_requests(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 3600
    ) -> int:
        """Get number of remaining requests in current window"""
        bucket_key = f"rate_limit:{key}:{window_seconds}"
        
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Count requests in current window
            request_count = await self.cache.redis.zcount(
                bucket_key,
                window_start,
                current_time
            )
            
            return max(0, max_requests - request_count)
        except:
            # Fallback for local cache
            if key in self.local_cache:
                current_time = time.time()
                window_start = current_time - window_seconds
                valid_requests = [
                    t for t in self.local_cache[key]
                    if t > window_start
                ]
                return max(0, max_requests - len(valid_requests))
            return max_requests

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(
    max_requests: int = 100,
    window: int = 3600,
    key_func: Optional[Callable] = None
):
    """
    Rate limiting decorator for FastAPI endpoints.
    
    Args:
        max_requests: Maximum requests per window
        window: Time window in seconds
        key_func: Function to extract rate limit key from request
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Determine rate limit key
            if key_func:
                limit_key = key_func(request)
            elif hasattr(request.state, "api_key"):
                limit_key = str(request.state.api_key.id)
            else:
                # Fall back to IP address
                limit_key = request.client.host
            
            # Check rate limit
            allowed = await rate_limiter.check_rate_limit(
                limit_key,
                max_requests,
                window
            )
            
            if not allowed:
                remaining = await rate_limiter.get_remaining_requests(
                    limit_key,
                    max_requests,
                    window
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(int(time.time()) + window)
                    }
                )
            
            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)
            
            if hasattr(response, "headers"):
                remaining = await rate_limiter.get_remaining_requests(
                    limit_key,
                    max_requests,
                    window
                )
                response.headers["X-RateLimit-Limit"] = str(max_requests)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)
            
            return response
        
        return wrapper
    return decorator
```

### 3.4 API Key Management Endpoints

```python
# app/api/v1/api_keys.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from datetime import datetime, timedelta

from app.schemas.api_key import APIKeyCreate, APIKeyResponse, APIKeyUpdate
from app.schemas.api_responses import APIResponse
from app.models.api_key import APIKey
from app.models.user import User
from app.core.auth import require_auth
from app.core.database import get_db

router = APIRouter()

@router.post(
    "/",
    response_model=APIResponse[dict],
    status_code=status.HTTP_201_CREATED,
    summary="Create API key"
)
async def create_api_key(
    key_data: APIKeyCreate,
    user: User = Depends(require_auth),
    db = Depends(get_db)
):
    """
    Create a new API key.
    
    **Request Body:**
    - `name`: API key name
    - `description`: Optional description
    - `tier`: Access tier (standard, premium)
    - `expires_in_days`: Optional expiration in days
    """
    # Generate new key
    api_key = APIKey(
        key=APIKey.generate_key(),
        name=key_data.name,
        description=key_data.description,
        user_id=user.id,
        tier=key_data.tier or "standard",
        rate_limit=100 if key_data.tier == "standard" else 1000,
        daily_limit=10000 if key_data.tier == "standard" else 100000
    )
    
    if key_data.expires_in_days:
        api_key.expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    return APIResponse(
        data={
            "id": str(api_key.id),
            "key": api_key.key,  # Only shown once
            "name": api_key.name,
            "tier": api_key.tier,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
        },
        message="API key created. Save this key securely - it won't be shown again."
    )

@router.get(
    "/",
    response_model=APIResponse[List[APIKeyResponse]],
    summary="List API keys"
)
async def list_api_keys(
    user: User = Depends(require_auth),
    db = Depends(get_db)
):
    """List all API keys for the authenticated user."""
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == user.id)
    )
    keys = result.scalars().all()
    
    # Don't return the actual key values
    key_list = []
    for key in keys:
        key_data = APIKeyResponse.from_orm(key).dict()
        key_data["key"] = key.key[:8] + "..." + key.key[-4:]  # Partial key
        key_list.append(key_data)
    
    return APIResponse(data=key_list)

@router.delete(
    "/{key_id}",
    response_model=APIResponse[dict],
    summary="Revoke API key"
)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_auth),
    db = Depends(get_db)
):
    """Revoke an API key."""
    api_key = await db.get(APIKey, key_id)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    if api_key.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to revoke this key"
        )
    
    api_key.is_active = False
    await db.commit()
    
    # Clear from cache
    cache_key = f"api_key:{hashlib.sha256(api_key.key.encode()).hexdigest()[:16]}"
    await cache_manager.redis.delete(cache_key)
    
    return APIResponse(
        data={"key_id": key_id},
        message="API key revoked successfully"
    )

@router.get(
    "/{key_id}/usage",
    response_model=APIResponse[dict],
    summary="Get API key usage"
)
async def get_api_key_usage(
    key_id: str,
    user: User = Depends(require_auth),
    db = Depends(get_db)
):
    """Get usage statistics for an API key."""
    api_key = await db.get(APIKey, key_id)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    if api_key.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this key"
        )
    
    # Get usage stats from monitoring service
    from app.services.monitoring_service import MonitoringService
    monitoring = MonitoringService(db)
    
    usage_stats = await monitoring.get_api_key_usage(key_id)
    
    return APIResponse(data=usage_stats)
```

---

## Implementation Checklist

### Day 1 Tasks
- [ ] Create API v1 router structure
- [ ] Define response models and schemas
- [ ] Implement Channel API endpoints
- [ ] Implement Video API endpoints
- [ ] Implement Ideas API endpoints
- [ ] Add pagination support
- [ ] Test core endpoints
- [ ] Document endpoint specifications

### Day 2 Tasks
- [ ] Create SearchService for advanced filtering
- [ ] Implement global search endpoint
- [ ] Add advanced search with complex filters
- [ ] Create search suggestions endpoint
- [ ] Implement Job Queue API
- [ ] Add job management endpoints
- [ ] Create statistics endpoints
- [ ] Test search and filtering

### Day 3 Tasks
- [ ] Create APIKey model and schema
- [ ] Implement API authentication service
- [ ] Build rate limiting system
- [ ] Add rate limiter middleware
- [ ] Create API key management endpoints
- [ ] Implement usage tracking
- [ ] Test authentication flow
- [ ] Test rate limiting

---

## Testing Requirements

### API Endpoint Tests

```python
# tests/api/test_channels_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_list_channels_requires_auth():
    """Test that API endpoints require authentication"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/channels")
        assert response.status_code == 403

@pytest.mark.asyncio
async def test_list_channels_with_api_key(api_key):
    """Test channel listing with valid API key"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/channels",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "pagination" in data

@pytest.mark.asyncio
async def test_pagination(api_key):
    """Test pagination parameters"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/channels?limit=10&offset=0",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["limit"] == 10
        assert data["pagination"]["offset"] == 0
```

### Rate Limiting Tests

```python
# tests/test_rate_limiting.py
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_rate_limiting(api_key_standard):
    """Test rate limiting enforcement"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make requests up to limit
        for i in range(100):
            response = await client.get(
                "/api/v1/ideas",
                headers={"X-API-Key": api_key_standard}
            )
            assert response.status_code == 200
            assert "X-RateLimit-Remaining" in response.headers
        
        # Next request should be rate limited
        response = await client.get(
            "/api/v1/ideas",
            headers={"X-API-Key": api_key_standard}
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
```

---

## Success Criteria

### Day 1 Completion
- All core resource endpoints implemented
- Pagination working correctly
- Consistent response formats
- Basic filtering functional
- API responds with proper status codes

### Day 2 Completion
- Global search operational
- Advanced filtering working
- Search suggestions functional
- Job queue API complete
- Statistics endpoints returning data

### Day 3 Completion
- API key authentication working
- Rate limiting enforced correctly
- API key management functional
- Usage tracking operational
- All endpoints properly secured

### Performance Metrics
- API response time < 200ms for standard queries
- Search response time < 500ms
- Rate limiting overhead < 10ms
- Pagination handling 10,000+ records efficiently
- Concurrent request handling up to 100 req/sec