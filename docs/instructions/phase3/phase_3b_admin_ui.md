# TubeSensei Phase 3B: Admin UI Components
## Week 8 - Days 3-5: User Interface Implementation

### Version: 1.0
### Duration: 3 Days  
### Dependencies: Phase 3A Complete (FastAPI Core & Authentication)

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Day 3: Channel Management UI](#day-3-channel-management-ui)
3. [Day 4: Processing Status Dashboard](#day-4-processing-status-dashboard)
4. [Day 5: Idea Review Interface](#day-5-idea-review-interface)
5. [Implementation Checklist](#implementation-checklist)
6. [Testing Requirements](#testing-requirements)

---

## Phase Overview

### Objectives
Build the complete admin interface with channel management, real-time processing status monitoring, and idea review capabilities using HTMX for interactivity.

### Deliverables
- Channel management interface with CRUD operations
- Real-time processing status dashboard
- WebSocket integration for live updates
- Idea review and selection interface
- Bulk operations and filtering
- Export selection UI

### Technical Stack
- **Frontend**: Jinja2 templates, HTMX, Alpine.js, Tailwind CSS
- **Real-time**: WebSockets for live updates
- **Charts**: Chart.js for visualizations
- **Tables**: DataTables or custom HTMX tables

---

## Day 3: Channel Management UI

### 3.1 Channel Service Layer

```python
# app/services/channel_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from datetime import datetime, timedelta
import asyncio

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse
from app.core.exceptions import NotFoundException, ValidationException
from app.integrations.youtube import YouTubeAPI

class ChannelService:
    """Service for channel operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.youtube = YouTubeAPI()
    
    async def list_channels(
        self,
        status: Optional[ChannelStatus] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List channels with filtering and pagination"""
        query = select(Channel)
        
        # Apply filters
        if status:
            query = query.where(Channel.status == status)
        
        if search:
            query = query.where(
                or_(
                    Channel.name.ilike(f"%{search}%"),
                    Channel.description.ilike(f"%{search}%")
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get paginated results
        query = query.order_by(Channel.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        channels = result.scalars().all()
        
        # Enrich with stats
        enriched = []
        for channel in channels:
            stats = await self._get_channel_stats(channel.id)
            enriched.append({
                **ChannelResponse.from_orm(channel).dict(),
                "stats": stats
            })
        
        return {
            "items": enriched,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    async def get_channel(self, channel_id: str) -> Channel:
        """Get channel by ID"""
        channel = await self.db.get(Channel, channel_id)
        if not channel:
            raise NotFoundException("Channel", channel_id)
        return channel
    
    async def add_channel(self, data: ChannelCreate) -> Channel:
        """Add new channel for monitoring"""
        # Validate YouTube channel exists
        channel_info = await self.youtube.get_channel_info(data.youtube_channel_id)
        if not channel_info:
            raise ValidationException({
                "youtube_channel_id": "Invalid YouTube channel ID"
            })
        
        # Check if already exists
        existing = await self.db.execute(
            select(Channel).where(
                Channel.youtube_channel_id == data.youtube_channel_id
            )
        )
        if existing.scalar_one_or_none():
            raise ValidationException({
                "youtube_channel_id": "Channel already being monitored"
            })
        
        # Create channel
        channel = Channel(
            name=channel_info["title"],
            youtube_channel_id=data.youtube_channel_id,
            channel_url=f"https://youtube.com/channel/{data.youtube_channel_id}",
            description=channel_info.get("description"),
            subscriber_count=channel_info.get("subscriberCount"),
            video_count=channel_info.get("videoCount"),
            thumbnail_url=channel_info.get("thumbnail"),
            status=ChannelStatus.ACTIVE,
            processing_config=data.processing_config or {}
        )
        
        self.db.add(channel)
        await self.db.commit()
        await self.db.refresh(channel)
        
        # Queue initial video discovery
        await self._queue_channel_discovery(channel.id)
        
        return channel
    
    async def update_channel(
        self,
        channel_id: str,
        data: ChannelUpdate
    ) -> Channel:
        """Update channel configuration"""
        channel = await self.get_channel(channel_id)
        
        # Update fields
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(channel, field, value)
        
        channel.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(channel)
        
        return channel
    
    async def sync_channel(self, channel_id: str) -> Dict[str, Any]:
        """Manually sync channel data"""
        channel = await self.get_channel(channel_id)
        
        # Get latest channel info
        channel_info = await self.youtube.get_channel_info(
            channel.youtube_channel_id
        )
        
        if channel_info:
            channel.subscriber_count = channel_info.get("subscriberCount")
            channel.video_count = channel_info.get("videoCount")
            channel.description = channel_info.get("description")
            channel.last_checked_at = datetime.utcnow()
            
            await self.db.commit()
        
        # Queue video discovery
        job_id = await self._queue_channel_discovery(channel.id)
        
        return {
            "channel_id": channel.id,
            "updated": True,
            "job_id": job_id,
            "stats": channel_info
        }
    
    async def delete_channel(self, channel_id: str) -> bool:
        """Delete channel (soft delete)"""
        channel = await self.get_channel(channel_id)
        channel.status = ChannelStatus.INACTIVE
        channel.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def _get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get processing statistics for channel"""
        # Video stats
        video_stats = await self.db.execute(
            select(
                func.count(Video.id).label("total_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.COMPLETED
                ).label("processed_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.PROCESSING
                ).label("processing_videos"),
                func.count(Video.id).filter(
                    Video.status == VideoStatus.FAILED
                ).label("failed_videos")
            ).where(Video.channel_id == channel_id)
        )
        
        stats = video_stats.one()._asdict()
        
        # Idea stats
        idea_count = await self.db.scalar(
            select(func.count()).select_from(
                select(Video.id).where(
                    Video.channel_id == channel_id
                ).subquery()
            )
        )
        
        stats["total_ideas"] = idea_count
        
        return stats
    
    async def _queue_channel_discovery(self, channel_id: str) -> str:
        """Queue channel for video discovery"""
        from app.tasks.discovery import discover_channel_videos
        
        task = discover_channel_videos.delay(str(channel_id))
        return task.id
```

### 3.2 Channel Management Router

```python
# app/api/admin/channels.py
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from typing import Optional

from app.core.auth import require_auth
from app.core.permissions import require_permission, Permission
from app.services.channel_service import ChannelService
from app.schemas.channel import ChannelCreate, ChannelUpdate
from app.core.database import get_db
from app.templates import templates

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
    
    limit = 20
    offset = (page - 1) * limit
    
    result = await service.list_channels(
        status=status,
        search=search,
        limit=limit,
        offset=offset
    )
    
    return templates.TemplateResponse(
        "admin/channels/list.html",
        {
            "request": request,
            "user": user,
            "channels": result["items"],
            "total": result["total"],
            "page": page,
            "total_pages": (result["total"] + limit - 1) // limit,
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

@router.post("/add")
async def add_channel(
    data: ChannelCreate,
    user = Depends(require_permission(Permission.CHANNEL_CREATE)),
    db = Depends(get_db)
):
    """Add new channel"""
    service = ChannelService(db)
    channel = await service.add_channel(data)
    
    return {
        "success": True,
        "channel": channel,
        "redirect": f"/admin/channels/{channel.id}"
    }

@router.get("/{channel_id}", response_class=HTMLResponse)
async def channel_detail(
    request: Request,
    channel_id: str,
    user = Depends(require_auth),
    db = Depends(get_db)
):
    """Render channel detail page"""
    service = ChannelService(db)
    channel = await service.get_channel(channel_id)
    stats = await service._get_channel_stats(channel_id)
    
    return templates.TemplateResponse(
        "admin/channels/detail.html",
        {
            "request": request,
            "user": user,
            "channel": channel,
            "stats": stats
        }
    )

@router.get("/{channel_id}/edit", response_class=HTMLResponse)
async def edit_channel_form(
    request: Request,
    channel_id: str,
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
    db = Depends(get_db)
):
    """Render edit channel form"""
    service = ChannelService(db)
    channel = await service.get_channel(channel_id)
    
    return templates.TemplateResponse(
        "admin/channels/edit.html",
        {
            "request": request,
            "user": user,
            "channel": channel
        }
    )

@router.patch("/{channel_id}")
async def update_channel(
    channel_id: str,
    data: ChannelUpdate,
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
    db = Depends(get_db)
):
    """Update channel"""
    service = ChannelService(db)
    channel = await service.update_channel(channel_id, data)
    
    return {
        "success": True,
        "channel": channel
    }

@router.post("/{channel_id}/sync")
async def sync_channel(
    channel_id: str,
    user = Depends(require_permission(Permission.CHANNEL_UPDATE)),
    db = Depends(get_db)
):
    """Manually sync channel"""
    service = ChannelService(db)
    result = await service.sync_channel(channel_id)
    
    return {
        "success": True,
        "result": result
    }

@router.delete("/{channel_id}")
async def delete_channel(
    channel_id: str,
    user = Depends(require_permission(Permission.CHANNEL_DELETE)),
    db = Depends(get_db)
):
    """Delete channel"""
    service = ChannelService(db)
    await service.delete_channel(channel_id)
    
    return {
        "success": True,
        "redirect": "/admin/channels"
    }
```

### 3.3 Channel Management Templates

```html
<!-- templates/admin/channels/list.html -->
{% extends "base.html" %}

{% block title %}Channels - TubeSensei Admin{% endblock %}

{% block content %}
<div class="channels-container">
    <!-- Header -->
    <div class="flex justify-between items-center mb-6">
        <h1 class="text-2xl font-bold text-gray-900">Monitored Channels</h1>
        <button 
            class="btn-primary"
            onclick="showAddChannelModal()"
        >
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M12 4v16m8-8H4" />
            </svg>
            Add Channel
        </button>
    </div>

    <!-- Filters -->
    <div class="bg-white p-4 rounded-lg shadow mb-6">
        <form class="flex gap-4" hx-get="/admin/channels" hx-target="#channels-grid">
            <div class="flex-1">
                <input 
                    type="text" 
                    name="search"
                    placeholder="Search channels..."
                    value="{{ filters.search or '' }}"
                    class="form-input w-full"
                    hx-get="/admin/channels"
                    hx-trigger="keyup changed delay:500ms"
                    hx-target="#channels-grid"
                >
            </div>
            <select 
                name="status" 
                class="form-select"
                hx-get="/admin/channels"
                hx-trigger="change"
                hx-target="#channels-grid"
            >
                <option value="">All Status</option>
                <option value="active" {% if filters.status == 'active' %}selected{% endif %}>
                    Active
                </option>
                <option value="paused" {% if filters.status == 'paused' %}selected{% endif %}>
                    Paused
                </option>
                <option value="inactive" {% if filters.status == 'inactive' %}selected{% endif %}>
                    Inactive
                </option>
            </select>
            <button type="submit" class="btn-secondary">Apply Filters</button>
        </form>
    </div>

    <!-- Channels Grid -->
    <div id="channels-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {% for channel in channels %}
        <div class="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow">
            <div class="p-6">
                <!-- Channel Header -->
                <div class="flex items-start justify-between mb-4">
                    <div class="flex items-center">
                        {% if channel.thumbnail_url %}
                        <img 
                            src="{{ channel.thumbnail_url }}" 
                            alt="{{ channel.name }}"
                            class="w-12 h-12 rounded-full mr-3"
                        >
                        {% endif %}
                        <div>
                            <h3 class="font-semibold text-gray-900">
                                <a href="/admin/channels/{{ channel.id }}" 
                                   class="hover:text-blue-600">
                                    {{ channel.name }}
                                </a>
                            </h3>
                            <span class="status-badge status-{{ channel.status }}">
                                {{ channel.status }}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Channel Stats -->
                <div class="grid grid-cols-2 gap-2 text-sm text-gray-600 mb-4">
                    <div>
                        <span class="font-medium">Subscribers:</span>
                        {{ channel.subscriber_count|number_format }}
                    </div>
                    <div>
                        <span class="font-medium">Videos:</span>
                        {{ channel.video_count|number_format }}
                    </div>
                    <div>
                        <span class="font-medium">Processed:</span>
                        {{ channel.stats.processed_videos }}/{{ channel.stats.total_videos }}
                    </div>
                    <div>
                        <span class="font-medium">Ideas:</span>
                        {{ channel.stats.total_ideas }}
                    </div>
                </div>

                <!-- Last Checked -->
                <div class="text-xs text-gray-500 mb-4">
                    Last checked: 
                    <span class="font-medium">
                        {{ channel.last_checked_at|timeago if channel.last_checked_at else 'Never' }}
                    </span>
                </div>

                <!-- Actions -->
                <div class="flex gap-2">
                    <button 
                        class="btn-sm btn-secondary flex-1"
                        hx-post="/admin/channels/{{ channel.id }}/sync"
                        hx-confirm="Sync channel now?"
                        hx-indicator="#sync-indicator-{{ channel.id }}"
                    >
                        <span id="sync-indicator-{{ channel.id }}" class="htmx-indicator">
                            <svg class="animate-spin h-4 w-4" fill="none">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" 
                                        stroke-width="4" opacity="0.25"></circle>
                                <path fill="currentColor" d="..." opacity="0.75"></path>
                            </svg>
                        </span>
                        Sync
                    </button>
                    <a 
                        href="/admin/channels/{{ channel.id }}/edit"
                        class="btn-sm btn-secondary flex-1 text-center"
                    >
                        Edit
                    </a>
                    <button 
                        class="btn-sm btn-danger"
                        hx-delete="/admin/channels/{{ channel.id }}"
                        hx-confirm="Are you sure you want to delete this channel?"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Pagination -->
    {% if total_pages > 1 %}
    <div class="mt-6">
        {% include "components/pagination.html" %}
    </div>
    {% endif %}
</div>

<!-- Add Channel Modal -->
<div id="add-channel-modal" class="modal hidden">
    <div class="modal-content">
        <h2 class="text-xl font-bold mb-4">Add New Channel</h2>
        <form 
            hx-post="/admin/channels/add"
            hx-target="#channels-grid"
            hx-swap="afterbegin"
        >
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">
                    YouTube Channel ID or URL
                </label>
                <input 
                    type="text" 
                    name="youtube_channel_id"
                    placeholder="UC... or https://youtube.com/channel/..."
                    class="form-input w-full"
                    required
                >
                <p class="text-xs text-gray-500 mt-1">
                    Enter the YouTube channel ID or full channel URL
                </p>
            </div>
            
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">
                    Processing Configuration (Optional)
                </label>
                <textarea 
                    name="processing_config"
                    class="form-textarea w-full"
                    rows="3"
                    placeholder='{"min_duration": 300, "max_age_days": 30}'
                ></textarea>
            </div>
            
            <div class="flex justify-end gap-2">
                <button 
                    type="button"
                    onclick="hideAddChannelModal()"
                    class="btn-secondary"
                >
                    Cancel
                </button>
                <button type="submit" class="btn-primary">
                    Add Channel
                </button>
            </div>
        </form>
    </div>
</div>

<script>
function showAddChannelModal() {
    document.getElementById('add-channel-modal').classList.remove('hidden');
}

function hideAddChannelModal() {
    document.getElementById('add-channel-modal').classList.add('hidden');
}
</script>
{% endblock %}
```

---

## Day 4: Processing Status Dashboard

### 4.1 Real-time Monitoring Service

```python
# app/services/monitoring_service.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from collections import defaultdict

from app.models.processing_job import ProcessingJob, JobStatus, JobType
from app.models.video import Video, VideoStatus
from app.models.idea import Idea
from app.core.cache import cache_manager

class MonitoringService:
    """Service for system monitoring and metrics"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.cache = cache_manager
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        # Check database connection
        try:
            await self.db.execute("SELECT 1")
            db_status = "healthy"
        except:
            db_status = "error"
        
        # Check Redis connection
        try:
            await self.cache.redis.ping()
            redis_status = "healthy"
        except:
            redis_status = "error"
        
        # Get queue status
        queue_status = await self.get_queue_status()
        
        return {
            "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
            "database": db_status,
            "redis": redis_status,
            "queue": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        # Get stats from cache first
        cache_key = "processing_stats"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Calculate stats
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_24h = now - timedelta(hours=24)
        
        # Job statistics
        job_stats = await self.db.execute(
            select(
                func.count(ProcessingJob.id).label("total_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.COMPLETED
                ).label("completed_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.FAILED
                ).label("failed_jobs"),
                func.count(ProcessingJob.id).filter(
                    ProcessingJob.status == JobStatus.RUNNING
                ).label("running_jobs"),
                func.count(ProcessingJob.id).filter(
                    and_(
                        ProcessingJob.status == JobStatus.COMPLETED,
                        ProcessingJob.completed_at >= last_hour
                    )
                ).label("completed_last_hour"),
                func.count(ProcessingJob.id).filter(
                    and_(
                        ProcessingJob.status == JobStatus.COMPLETED,
                        ProcessingJob.completed_at >= last_24h
                    )
                ).label("completed_last_24h")
            )
        )
        
        stats = job_stats.one()._asdict()
        
        # Calculate success rate
        total_finished = stats["completed_jobs"] + stats["failed_jobs"]
        stats["success_rate"] = (
            (stats["completed_jobs"] / total_finished * 100) 
            if total_finished > 0 else 0
        )
        
        # Processing rate
        stats["hourly_rate"] = stats["completed_last_hour"]
        stats["daily_rate"] = stats["completed_last_24h"]
        
        # Video and idea counts
        video_count = await self.db.scalar(
            select(func.count(Video.id)).where(
                Video.status == VideoStatus.COMPLETED
            )
        )
        idea_count = await self.db.scalar(
            select(func.count(Idea.id))
        )
        
        stats["total_videos_processed"] = video_count
        stats["total_ideas_extracted"] = idea_count
        
        # Cache for 30 seconds
        await self.cache.set(cache_key, stats, expire=30)
        
        return stats
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get job queue status"""
        # Get queue lengths by job type
        queue_stats = await self.db.execute(
            select(
                ProcessingJob.job_type,
                func.count(ProcessingJob.id).label("count")
            ).where(
                ProcessingJob.status == JobStatus.QUEUED
            ).group_by(ProcessingJob.job_type)
        )
        
        queue_by_type = {
            row.job_type: row.count 
            for row in queue_stats
        }
        
        # Get total queue length
        total_queued = sum(queue_by_type.values())
        
        # Get average wait time
        avg_wait = await self.db.scalar(
            select(
                func.avg(
                    func.extract('epoch', ProcessingJob.started_at - ProcessingJob.created_at)
                )
            ).where(
                and_(
                    ProcessingJob.status == JobStatus.RUNNING,
                    ProcessingJob.started_at.isnot(None)
                )
            )
        )
        
        return {
            "total_queued": total_queued,
            "by_type": queue_by_type,
            "average_wait_seconds": avg_wait or 0
        }
    
    async def get_recent_jobs(
        self,
        limit: int = 10,
        job_type: Optional[JobType] = None
    ) -> List[Dict[str, Any]]:
        """Get recent processing jobs"""
        query = select(ProcessingJob).order_by(
            ProcessingJob.created_at.desc()
        ).limit(limit)
        
        if job_type:
            query = query.where(ProcessingJob.job_type == job_type)
        
        result = await self.db.execute(query)
        jobs = result.scalars().all()
        
        return [
            {
                "id": str(job.id),
                "type": job.job_type,
                "status": job.status,
                "entity_type": job.entity_type,
                "entity_id": str(job.entity_id),
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "duration": self._calculate_duration(job),
                "error": job.error_message
            }
            for job in jobs
        ]
    
    async def get_processing_timeline(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get processing timeline for charts"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)
        
        # Get completed jobs grouped by hour
        timeline = await self.db.execute(
            select(
                func.date_trunc('hour', ProcessingJob.completed_at).label("hour"),
                func.count(ProcessingJob.id).label("count"),
                ProcessingJob.job_type
            ).where(
                and_(
                    ProcessingJob.status == JobStatus.COMPLETED,
                    ProcessingJob.completed_at >= start_time
                )
            ).group_by(
                func.date_trunc('hour', ProcessingJob.completed_at),
                ProcessingJob.job_type
            )
        )
        
        # Organize by hour and type
        timeline_data = defaultdict(lambda: defaultdict(int))
        for row in timeline:
            hour = row.hour.isoformat()
            timeline_data[hour][row.job_type] = row.count
        
        return {
            "labels": sorted(timeline_data.keys()),
            "datasets": [
                {
                    "label": job_type,
                    "data": [
                        timeline_data[hour].get(job_type, 0)
                        for hour in sorted(timeline_data.keys())
                    ]
                }
                for job_type in JobType
            ]
        }
    
    def _calculate_duration(self, job: ProcessingJob) -> Optional[float]:
        """Calculate job duration in seconds"""
        if job.started_at and job.completed_at:
            return (job.completed_at - job.started_at).total_seconds()
        elif job.started_at:
            return (datetime.utcnow() - job.started_at).total_seconds()
        return None
```

### 4.2 WebSocket Handler

```python
# app/api/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import asyncio
import json
from datetime import datetime

from app.core.auth import auth_handler
from app.services.monitoring_service import MonitoringService
from app.core.database import get_db

router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "dashboard": set(),
            "jobs": set(),
            "ideas": set()
        }
    
    async def connect(self, websocket: WebSocket, channel: str = "dashboard"):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[channel].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, channel: str = "dashboard"):
        """Remove connection"""
        self.active_connections[channel].discard(websocket)
    
    async def broadcast(self, message: dict, channel: str = "dashboard"):
        """Broadcast message to all connections in channel"""
        disconnected = set()
        
        for websocket in self.active_connections[channel]:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.active_connections[channel].discard(websocket)
    
    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_json(message)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard updates"""
    await manager.connect(websocket, "dashboard")
    
    try:
        async with get_db() as db:
            monitoring = MonitoringService(db)
            
            while True:
                # Send updates every 2 seconds
                try:
                    status_data = {
                        "type": "status_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "system_status": await monitoring.get_system_status(),
                            "processing_stats": await monitoring.get_processing_stats(),
                            "queue_status": await monitoring.get_queue_status(),
                            "recent_jobs": await monitoring.get_recent_jobs(5)
                        }
                    }
                    
                    await manager.send_personal(status_data, websocket)
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    error_message = {
                        "type": "error",
                        "message": str(e)
                    }
                    await manager.send_personal(error_message, websocket)
                    
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "dashboard")

@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket for specific job updates"""
    await manager.connect(websocket, f"job_{job_id}")
    
    try:
        while True:
            # Send job-specific updates
            # Implementation depends on job tracking system
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket, f"job_{job_id}")
```

### 4.3 Dashboard Template

```html
<!-- templates/admin/dashboard/index.html -->
{% extends "base.html" %}

{% block title %}Dashboard - TubeSensei Admin{% endblock %}

{% block content %}
<div class="dashboard-container">
    <!-- Header -->
    <div class="mb-6">
        <h1 class="text-2xl font-bold text-gray-900">Processing Dashboard</h1>
        <p class="text-gray-600">Real-time system monitoring and statistics</p>
    </div>

    <!-- System Status -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-sm text-gray-600">System Status</p>
                    <p class="text-2xl font-bold" id="system-status">
                        <span class="text-green-500">Healthy</span>
                    </p>
                </div>
                <div class="text-green-500">
                    <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" 
                              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" 
                              clip-rule="evenodd" />
                    </svg>
                </div>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow p-6">
            <div>
                <p class="text-sm text-gray-600">Active Jobs</p>
                <p class="text-2xl font-bold" id="active-jobs">0</p>
                <p class="text-xs text-gray-500" id="queue-length">0 queued</p>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow p-6">
            <div>
                <p class="text-sm text-gray-600">Processing Rate</p>
                <p class="text-2xl font-bold" id="processing-rate">0/hr</p>
                <p class="text-xs text-gray-500">Last hour</p>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow p-6">
            <div>
                <p class="text-sm text-gray-600">Success Rate</p>
                <p class="text-2xl font-bold" id="success-rate">0%</p>
                <p class="text-xs text-gray-500">All time</p>
            </div>
        </div>
    </div>

    <!-- Processing Timeline Chart -->
    <div class="bg-white rounded-lg shadow p-6 mb-6">
        <h2 class="text-lg font-semibold mb-4">Processing Timeline (24h)</h2>
        <canvas id="timeline-chart" height="80"></canvas>
    </div>

    <!-- Two Column Layout -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Recent Jobs -->
        <div class="bg-white rounded-lg shadow">
            <div class="p-6">
                <h2 class="text-lg font-semibold mb-4">Recent Jobs</h2>
                <div id="recent-jobs" class="space-y-2">
                    <!-- Populated via WebSocket -->
                </div>
            </div>
        </div>

        <!-- Queue Status -->
        <div class="bg-white rounded-lg shadow">
            <div class="p-6">
                <h2 class="text-lg font-semibold mb-4">Queue Status</h2>
                <canvas id="queue-chart" height="200"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/dashboard');

// Chart instances
let timelineChart = null;
let queueChart = null;

// Initialize charts
document.addEventListener('DOMContentLoaded', function() {
    // Timeline chart
    const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
    timelineChart = new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Queue chart
    const queueCtx = document.getElementById('queue-chart').getContext('2d');
    queueChart = new Chart(queueCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#3B82F6',
                    '#10B981',
                    '#F59E0B',
                    '#EF4444'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
});

// Handle WebSocket messages
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    if (message.type === 'status_update') {
        updateDashboard(message.data);
    }
};

function updateDashboard(data) {
    // Update system status
    const statusEl = document.getElementById('system-status');
    if (data.system_status.status === 'healthy') {
        statusEl.innerHTML = '<span class="text-green-500">Healthy</span>';
    } else {
        statusEl.innerHTML = '<span class="text-yellow-500">Degraded</span>';
    }
    
    // Update stats
    document.getElementById('active-jobs').textContent = 
        data.processing_stats.running_jobs || 0;
    document.getElementById('queue-length').textContent = 
        `${data.queue_status.total_queued || 0} queued`;
    document.getElementById('processing-rate').textContent = 
        `${data.processing_stats.hourly_rate || 0}/hr`;
    document.getElementById('success-rate').textContent = 
        `${Math.round(data.processing_stats.success_rate || 0)}%`;
    
    // Update recent jobs
    updateRecentJobs(data.recent_jobs);
    
    // Update queue chart
    updateQueueChart(data.queue_status);
}

function updateRecentJobs(jobs) {
    const container = document.getElementById('recent-jobs');
    container.innerHTML = jobs.map(job => `
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
            <div class="flex-1">
                <p class="text-sm font-medium">${job.type}</p>
                <p class="text-xs text-gray-500">
                    ${new Date(job.created_at).toLocaleTimeString()}
                </p>
            </div>
            <span class="status-badge status-${job.status.toLowerCase()}">
                ${job.status}
            </span>
        </div>
    `).join('');
}

function updateQueueChart(queueStatus) {
    if (queueChart && queueStatus.by_type) {
        queueChart.data.labels = Object.keys(queueStatus.by_type);
        queueChart.data.datasets[0].data = Object.values(queueStatus.by_type);
        queueChart.update();
    }
}

// Reconnect on disconnect
ws.onclose = function() {
    setTimeout(function() {
        location.reload();
    }, 5000);
};
</script>
{% endblock %}
```

---

## Day 5: Idea Review Interface

### 5.1 Idea Service

```python
# app/services/idea_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from datetime import datetime
import json

from app.models.idea import Idea, IdeaStatus
from app.models.video import Video
from app.models.channel import Channel
from app.schemas.idea import IdeaCreate, IdeaUpdate, IdeaResponse
from app.core.exceptions import NotFoundException

class IdeaService:
    """Service for idea operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_ideas(
        self,
        status: Optional[IdeaStatus] = None,
        min_confidence: float = 0.0,
        category: Optional[str] = None,
        channel_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List ideas with filtering"""
        query = select(Idea).join(Video)
        
        # Apply filters
        conditions = []
        
        if status:
            conditions.append(Idea.status == status)
        
        if min_confidence > 0:
            conditions.append(Idea.confidence_score >= min_confidence)
        
        if category:
            conditions.append(Idea.category == category)
        
        if channel_id:
            conditions.append(Video.channel_id == channel_id)
        
        if search:
            conditions.append(
                or_(
                    Idea.title.ilike(f"%{search}%"),
                    Idea.description.ilike(f"%{search}%")
                )
            )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get paginated results
        query = query.order_by(Idea.confidence_score.desc(), Idea.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        ideas = result.scalars().all()
        
        # Enrich with video and channel data
        enriched = []
        for idea in ideas:
            video = await self.db.get(Video, idea.video_id)
            channel = await self.db.get(Channel, video.channel_id)
            
            enriched.append({
                **IdeaResponse.from_orm(idea).dict(),
                "video": {
                    "id": str(video.id),
                    "title": video.title,
                    "url": video.video_url,
                    "thumbnail": video.thumbnail_url
                },
                "channel": {
                    "id": str(channel.id),
                    "name": channel.name
                }
            })
        
        return {
            "items": enriched,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    async def get_idea(self, idea_id: str) -> Idea:
        """Get idea by ID"""
        idea = await self.db.get(Idea, idea_id)
        if not idea:
            raise NotFoundException("Idea", idea_id)
        return idea
    
    async def update_idea(
        self,
        idea_id: str,
        data: IdeaUpdate
    ) -> Idea:
        """Update idea"""
        idea = await self.get_idea(idea_id)
        
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(idea, field, value)
        
        idea.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(idea)
        
        return idea
    
    async def bulk_update(
        self,
        idea_ids: List[str],
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform bulk update on ideas"""
        updated = 0
        errors = []
        
        for idea_id in idea_ids:
            try:
                idea = await self.get_idea(idea_id)
                
                if action == "select":
                    idea.status = IdeaStatus.SELECTED
                elif action == "reject":
                    idea.status = IdeaStatus.REJECTED
                elif action == "review":
                    idea.status = IdeaStatus.REVIEWED
                elif action == "update_category":
                    idea.category = kwargs.get("category")
                
                idea.updated_at = datetime.utcnow()
                updated += 1
                
            except Exception as e:
                errors.append({
                    "idea_id": idea_id,
                    "error": str(e)
                })
        
        await self.db.commit()
        
        return {
            "updated": updated,
            "errors": errors
        }
    
    async def get_categories(self) -> List[str]:
        """Get all unique categories"""
        result = await self.db.execute(
            select(Idea.category).distinct().where(
                Idea.category.isnot(None)
            )
        )
        return [row[0] for row in result]
    
    async def get_idea_context(self, idea_id: str) -> Dict[str, Any]:
        """Get full context for an idea"""
        idea = await self.get_idea(idea_id)
        video = await self.db.get(Video, idea.video_id)
        channel = await self.db.get(Channel, video.channel_id)
        
        # Get transcript excerpt if available
        from app.models.transcript import Transcript
        transcript = await self.db.execute(
            select(Transcript).where(Transcript.video_id == video.id)
        )
        transcript = transcript.scalar_one_or_none()
        
        return {
            "idea": IdeaResponse.from_orm(idea).dict(),
            "video": {
                "id": str(video.id),
                "title": video.title,
                "description": video.description,
                "url": video.video_url,
                "published_at": video.published_at.isoformat(),
                "duration": video.duration_seconds,
                "views": video.view_count
            },
            "channel": {
                "id": str(channel.id),
                "name": channel.name,
                "url": channel.channel_url
            },
            "transcript_excerpt": self._get_transcript_excerpt(
                transcript,
                idea.source_timestamp
            ) if transcript else None
        }
    
    def _get_transcript_excerpt(
        self,
        transcript,
        timestamp: Optional[int],
        context_seconds: int = 60
    ) -> Optional[str]:
        """Get transcript excerpt around timestamp"""
        if not transcript or not timestamp:
            return None
        
        # Simple implementation - would need proper timestamp parsing
        # in production
        words = transcript.content.split()
        start_word = max(0, timestamp - context_seconds) * 2  # Rough estimate
        end_word = min(len(words), (timestamp + context_seconds) * 2)
        
        return " ".join(words[start_word:end_word])
```

### 5.2 Idea Review Template

```html
<!-- templates/admin/ideas/review.html -->
{% extends "base.html" %}

{% block title %}Idea Review - TubeSensei Admin{% endblock %}

{% block content %}
<div class="ideas-review-container" x-data="ideaReview()">
    <!-- Header -->
    <div class="flex justify-between items-center mb-6">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Idea Review</h1>
            <p class="text-gray-600">
                {{ total }} ideas found 
                <span x-show="selectedCount > 0">
                    ({{ selectedCount }} selected)
                </span>
            </p>
        </div>
        <div class="flex gap-2">
            <button 
                @click="exportSelected()"
                x-show="selectedCount > 0"
                class="btn-primary"
            >
                Export Selected ({{ selectedCount }})
            </button>
        </div>
    </div>

    <!-- Filters -->
    <div class="bg-white p-4 rounded-lg shadow mb-6">
        <form class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <!-- Search -->
            <input 
                type="text"
                name="search"
                placeholder="Search ideas..."
                class="form-input"
                hx-get="/admin/ideas"
                hx-trigger="keyup changed delay:500ms"
                hx-target="#ideas-container"
            >

            <!-- Status Filter -->
            <select 
                name="status"
                class="form-select"
                hx-get="/admin/ideas"
                hx-trigger="change"
                hx-target="#ideas-container"
            >
                <option value="">All Status</option>
                <option value="extracted">New</option>
                <option value="reviewed">Reviewed</option>
                <option value="selected">Selected</option>
                <option value="rejected">Rejected</option>
            </select>

            <!-- Category Filter -->
            <select 
                name="category"
                class="form-select"
                hx-get="/admin/ideas"
                hx-trigger="change"
                hx-target="#ideas-container"
            >
                <option value="">All Categories</option>
                {% for category in categories %}
                <option value="{{ category }}">{{ category }}</option>
                {% endfor %}
            </select>

            <!-- Confidence Filter -->
            <div class="flex items-center gap-2">
                <label class="text-sm">Min Confidence:</label>
                <input 
                    type="range"
                    name="min_confidence"
                    min="0"
                    max="100"
                    value="50"
                    class="flex-1"
                    hx-get="/admin/ideas"
                    hx-trigger="change throttle:500ms"
                    hx-target="#ideas-container"
                >
                <span class="text-sm font-medium">50%</span>
            </div>
        </form>
    </div>

    <!-- Bulk Actions -->
    <div class="bg-white p-4 rounded-lg shadow mb-6" x-show="selectedCount > 0">
        <div class="flex gap-2">
            <button 
                @click="bulkAction('select')"
                class="btn-success"
            >
                Select All Checked
            </button>
            <button 
                @click="bulkAction('review')"
                class="btn-warning"
            >
                Mark as Reviewed
            </button>
            <button 
                @click="bulkAction('reject')"
                class="btn-danger"
            >
                Reject All Checked
            </button>
        </div>
    </div>

    <!-- Ideas Grid -->
    <div id="ideas-container">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {% for idea in ideas %}
            <div 
                class="bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
                data-idea-id="{{ idea.id }}"
            >
                <div class="p-6">
                    <!-- Checkbox and Header -->
                    <div class="flex items-start gap-3 mb-4">
                        <input 
                            type="checkbox"
                            class="idea-checkbox mt-1"
                            value="{{ idea.id }}"
                            @change="toggleSelection('{{ idea.id }}')"
                        >
                        <div class="flex-1">
                            <h3 class="text-lg font-semibold text-gray-900">
                                {{ idea.title }}
                            </h3>
                            <div class="flex items-center gap-2 mt-1">
                                <span class="confidence-badge" 
                                      style="background: linear-gradient(90deg, #10b981 {{ idea.confidence_score * 100 }}%, #e5e7eb {{ idea.confidence_score * 100 }}%);">
                                    {{ (idea.confidence_score * 100)|round }}% confidence
                                </span>
                                <span class="status-badge status-{{ idea.status }}">
                                    {{ idea.status }}
                                </span>
                            </div>
                        </div>
                    </div>

                    <!-- Description -->
                    <div class="mb-4">
                        <p class="text-gray-700">{{ idea.description|truncate(200) }}</p>
                    </div>

                    <!-- Metadata -->
                    <div class="grid grid-cols-2 gap-2 text-sm text-gray-600 mb-4">
                        <div>
                            <span class="font-medium">Category:</span>
                            {{ idea.category or 'Uncategorized' }}
                        </div>
                        <div>
                            <span class="font-medium">Complexity:</span>
                            {{ idea.complexity_score }}/10
                        </div>
                        <div>
                            <span class="font-medium">Market Size:</span>
                            {{ idea.market_size_estimate or 'Unknown' }}
                        </div>
                        <div>
                            <span class="font-medium">Source:</span>
                            <a href="{{ idea.video.url }}" 
                               target="_blank"
                               class="text-blue-600 hover:underline">
                                Video
                            </a>
                        </div>
                    </div>

                    <!-- Tags -->
                    {% if idea.tags %}
                    <div class="flex flex-wrap gap-1 mb-4">
                        {% for tag in idea.tags %}
                        <span class="px-2 py-1 bg-gray-100 text-xs rounded">
                            {{ tag }}
                        </span>
                        {% endfor %}
                    </div>
                    {% endif %}

                    <!-- Source Info -->
                    <div class="border-t pt-4 mb-4">
                        <div class="flex items-center gap-2 text-sm text-gray-600">
                            {% if idea.video.thumbnail %}
                            <img src="{{ idea.video.thumbnail }}" 
                                 class="w-16 h-9 object-cover rounded">
                            {% endif %}
                            <div>
                                <p class="font-medium">{{ idea.channel.name }}</p>
                                <p class="text-xs">{{ idea.video.title|truncate(50) }}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Actions -->
                    <div class="flex gap-2">
                        <button 
                            class="btn-sm btn-success flex-1"
                            hx-patch="/admin/ideas/{{ idea.id }}"
                            hx-vals='{"status": "selected"}'
                            hx-target="closest div[data-idea-id]"
                            hx-swap="outerHTML"
                        >
                            Select
                        </button>
                        <button 
                            class="btn-sm btn-secondary flex-1"
                            onclick="showIdeaDetail('{{ idea.id }}')"
                        >
                            View Details
                        </button>
                        <button 
                            class="btn-sm btn-danger"
                            hx-patch="/admin/ideas/{{ idea.id }}"
                            hx-vals='{"status": "rejected"}'
                            hx-target="closest div[data-idea-id]"
                            hx-swap="outerHTML"
                        >
                            Reject
                        </button>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Pagination -->
        {% if total_pages > 1 %}
        <div class="mt-6">
            {% include "components/pagination.html" %}
        </div>
        {% endif %}
    </div>
</div>

<!-- Idea Detail Modal -->
<div id="idea-detail-modal" class="modal hidden">
    <div class="modal-content modal-lg">
        <div id="idea-detail-content">
            <!-- Loaded via HTMX -->
        </div>
    </div>
</div>

<script>
function ideaReview() {
    return {
        selectedIds: new Set(),
        selectedCount: 0,
        
        toggleSelection(id) {
            if (this.selectedIds.has(id)) {
                this.selectedIds.delete(id);
            } else {
                this.selectedIds.add(id);
            }
            this.selectedCount = this.selectedIds.size;
        },
        
        async bulkAction(action) {
            const ids = Array.from(this.selectedIds);
            
            const response = await fetch('/admin/ideas/bulk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    idea_ids: ids
                })
            });
            
            if (response.ok) {
                // Refresh the list
                htmx.trigger('#ideas-container', 'refresh');
                this.selectedIds.clear();
                this.selectedCount = 0;
            }
        },
        
        async exportSelected() {
            const ids = Array.from(this.selectedIds);
            window.location.href = `/admin/ideas/export?ids=${ids.join(',')}`;
        }
    }
}

function showIdeaDetail(ideaId) {
    htmx.ajax('GET', `/admin/ideas/${ideaId}/detail`, {
        target: '#idea-detail-content',
        swap: 'innerHTML'
    }).then(() => {
        document.getElementById('idea-detail-modal').classList.remove('hidden');
    });
}
</script>
{% endblock %}
```

---

## Implementation Checklist

### Day 3 Tasks
- [ ] Create Channel model and schema
- [ ] Implement ChannelService with CRUD operations
- [ ] Create channel management router and endpoints
- [ ] Build channel list template with HTMX
- [ ] Implement add/edit channel forms
- [ ] Add channel sync functionality
- [ ] Create channel detail view
- [ ] Test channel management flow
- [ ] Add channel statistics display

### Day 4 Tasks
- [ ] Create MonitoringService for system metrics
- [ ] Implement WebSocket connection manager
- [ ] Build dashboard WebSocket endpoint
- [ ] Create real-time dashboard template
- [ ] Implement Chart.js visualizations
- [ ] Add processing timeline chart
- [ ] Create job queue status display
- [ ] Test real-time updates
- [ ] Add error handling for WebSocket

### Day 5 Tasks
- [ ] Create Idea model and schema
- [ ] Implement IdeaService with filtering
- [ ] Build idea review router and endpoints
- [ ] Create idea review template with filters
- [ ] Implement bulk selection and actions
- [ ] Add idea detail modal
- [ ] Create export selection UI
- [ ] Test idea review workflow
- [ ] Add pagination and search

---

## Testing Requirements

### UI Component Tests

```python
# tests/ui/test_channels.py
import pytest
from playwright.sync_api import Page

def test_channel_list_display(page: Page, authenticated_page):
    """Test channel list displays correctly"""
    page.goto("/admin/channels")
    
    # Check page elements
    assert page.title() == "Channels - TubeSensei Admin"
    assert page.locator("h1").text_content() == "Monitored Channels"
    
    # Check add button
    add_button = page.locator("button:has-text('Add Channel')")
    assert add_button.is_visible()

def test_add_channel_modal(page: Page, authenticated_page):
    """Test add channel modal functionality"""
    page.goto("/admin/channels")
    
    # Open modal
    page.click("button:has-text('Add Channel')")
    
    # Fill form
    page.fill("input[name='youtube_channel_id']", "UC123456")
    
    # Submit
    page.click("button:has-text('Add Channel')")
    
    # Check redirect or success message
    page.wait_for_selector(".success-message")
```

### WebSocket Tests

```python
# tests/test_websocket.py
import pytest
import asyncio
from fastapi.testclient import TestClient

def test_dashboard_websocket(client: TestClient):
    """Test dashboard WebSocket connection"""
    with client.websocket_connect("/ws/dashboard") as websocket:
        # Receive status update
        data = websocket.receive_json()
        
        assert data["type"] == "status_update"
        assert "system_status" in data["data"]
        assert "processing_stats" in data["data"]
```

---

## Success Criteria

### Day 3 Completion
- Channel CRUD operations functional
- Channel list with filtering and pagination
- Add/edit channel forms working
- Channel sync triggers job queue
- Channel statistics displayed

### Day 4 Completion
- Real-time dashboard operational
- WebSocket connections stable
- Charts updating with live data
- Job queue status visible
- System metrics accurate

### Day 5 Completion
- Idea review interface complete
- Filtering and search functional
- Bulk operations working
- Idea detail modal displays
- Export selection ready

### Quality Metrics
- UI responsive on all screen sizes
- Real-time updates < 3 second latency
- All HTMX interactions smooth
- No JavaScript errors in console
- Accessibility standards met