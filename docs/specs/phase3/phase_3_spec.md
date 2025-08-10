# TubeSensei Phase 3: User Interface and API
## Technical Implementation Specification

### Version: 1.0
### Phase Duration: 2 Weeks (Weeks 8-9)
### Dependencies: Phase 1 (Core Infrastructure) and Phase 2 (AI Integration) completed

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Week 8: Admin Interface](#week-8-admin-interface)
3. [Week 9: API and Integration](#week-9-api-and-integration)
4. [Technical Architecture](#technical-architecture)
5. [Implementation Details](#implementation-details)
6. [Testing Requirements](#testing-requirements)
7. [Success Criteria](#success-criteria)

---

## Phase Overview

### Objectives
Phase 3 focuses on creating the user-facing components that enable interaction with the TubeSensei platform. This includes a comprehensive admin interface for managing the system and a REST API for external integrations.

### Key Deliverables
- **FastAPI-based Admin Interface**: Web-based UI for system management
- **Channel Management UI**: Interface for adding, monitoring, and configuring channels
- **Processing Status Dashboard**: Real-time monitoring of job processing
- **Idea Review Interface**: Tools for reviewing and selecting extracted ideas
- **REST API**: External integration endpoints with authentication
- **Export Functionality**: Data export in multiple formats
- **API Documentation**: Comprehensive API documentation using OpenAPI

### Prerequisites
- Completed database schema and models (Phase 1)
- Working job queue system (Phase 1)
- YouTube API integration operational (Phase 1)
- AI filtering and idea extraction functional (Phase 2)
- LLM integration complete (Phase 2)

---

## Week 8: Admin Interface

### 8.1 FastAPI Application Setup

#### Core Application Structure
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import admin, channels, videos, ideas, jobs
from app.core.config import settings

app = FastAPI(
    title="TubeSensei Admin",
    version="1.0.0",
    description="YouTube Content Analysis Platform"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(channels.router, prefix="/api/channels", tags=["channels"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(ideas.router, prefix="/api/ideas", tags=["ideas"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
```

#### Authentication and Authorization
```python
# app/core/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

class AuthHandler:
    security = HTTPBearer()
    secret = settings.SECRET_KEY
    algorithm = "HS256"
    
    def encode_token(self, user_id: str, role: str):
        payload = {
            "exp": datetime.utcnow() + timedelta(days=7),
            "iat": datetime.utcnow(),
            "sub": user_id,
            "role": role
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def decode_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload["sub"], payload["role"]
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Depends(security)):
        return self.decode_token(auth.credentials)
```

### 8.2 Channel Management UI

#### Channel Management Endpoints
```python
# app/api/channels.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from app.schemas.channel import ChannelCreate, ChannelUpdate, ChannelResponse
from app.services.channel_service import ChannelService

router = APIRouter()

@router.get("/", response_model=List[ChannelResponse])
async def list_channels(
    status: Optional[str] = Query(None, enum=["active", "paused", "inactive"]),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    service: ChannelService = Depends()
):
    """List all monitored channels with filtering and pagination"""
    return await service.list_channels(status=status, limit=limit, offset=offset)

@router.post("/", response_model=ChannelResponse)
async def add_channel(
    channel: ChannelCreate,
    service: ChannelService = Depends()
):
    """Add a new YouTube channel for monitoring"""
    return await service.add_channel(channel)

@router.patch("/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: str,
    updates: ChannelUpdate,
    service: ChannelService = Depends()
):
    """Update channel configuration and settings"""
    return await service.update_channel(channel_id, updates)

@router.post("/{channel_id}/sync")
async def sync_channel(
    channel_id: str,
    service: ChannelService = Depends()
):
    """Trigger manual sync for a specific channel"""
    return await service.sync_channel(channel_id)

@router.get("/{channel_id}/stats")
async def get_channel_stats(
    channel_id: str,
    service: ChannelService = Depends()
):
    """Get processing statistics for a channel"""
    return await service.get_channel_stats(channel_id)
```

#### Frontend Components (using Jinja2 templates with HTMX)
```html
<!-- templates/channels/list.html -->
<div class="channels-container">
    <div class="channels-header">
        <h2>Monitored Channels</h2>
        <button class="btn-primary" hx-get="/admin/channels/add-form" 
                hx-target="#modal-container">
            Add Channel
        </button>
    </div>
    
    <div class="filters">
        <select name="status" hx-get="/admin/channels" 
                hx-target="#channels-list" hx-trigger="change">
            <option value="">All Status</option>
            <option value="active">Active</option>
            <option value="paused">Paused</option>
            <option value="inactive">Inactive</option>
        </select>
    </div>
    
    <div id="channels-list" class="channels-grid">
        {% for channel in channels %}
        <div class="channel-card">
            <img src="{{ channel.thumbnail_url }}" alt="{{ channel.name }}">
            <div class="channel-info">
                <h3>{{ channel.name }}</h3>
                <span class="status-badge status-{{ channel.status }}">
                    {{ channel.status }}
                </span>
                <div class="channel-stats">
                    <span>{{ channel.subscriber_count }} subscribers</span>
                    <span>{{ channel.video_count }} videos</span>
                </div>
                <div class="channel-actions">
                    <button hx-get="/admin/channels/{{ channel.id }}/edit"
                            hx-target="#modal-container">Edit</button>
                    <button hx-post="/admin/channels/{{ channel.id }}/sync"
                            hx-confirm="Sync channel now?">Sync</button>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
```

### 8.3 Processing Status Monitoring

#### Real-time Status Dashboard
```python
# app/api/dashboard.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.monitoring_service import MonitoringService
import asyncio
import json

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_data(service: MonitoringService = Depends()):
    """Get current dashboard metrics"""
    return {
        "system_status": await service.get_system_status(),
        "processing_stats": await service.get_processing_stats(),
        "recent_jobs": await service.get_recent_jobs(limit=10),
        "queue_status": await service.get_queue_status(),
        "error_summary": await service.get_error_summary()
    }

@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates"""
    await websocket.accept()
    monitoring = MonitoringService()
    
    try:
        while True:
            # Send updates every 2 seconds
            status_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_jobs": await monitoring.get_active_jobs_count(),
                "queue_length": await monitoring.get_queue_length(),
                "processing_rate": await monitoring.get_processing_rate(),
                "recent_completions": await monitoring.get_recent_completions(5)
            }
            await websocket.send_json(status_data)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
```

#### Processing Status UI
```html
<!-- templates/dashboard/status.html -->
<div class="dashboard-container">
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Active Jobs</h3>
            <div class="metric-value" id="active-jobs">0</div>
        </div>
        <div class="metric-card">
            <h3>Queue Length</h3>
            <div class="metric-value" id="queue-length">0</div>
        </div>
        <div class="metric-card">
            <h3>Processing Rate</h3>
            <div class="metric-value" id="processing-rate">0/hr</div>
        </div>
        <div class="metric-card">
            <h3>Success Rate</h3>
            <div class="metric-value" id="success-rate">0%</div>
        </div>
    </div>
    
    <div class="jobs-section">
        <h3>Recent Jobs</h3>
        <div id="recent-jobs-list">
            <!-- Dynamically populated via WebSocket -->
        </div>
    </div>
    
    <div class="queue-visualization">
        <h3>Processing Queue</h3>
        <canvas id="queue-chart"></canvas>
    </div>
</div>

<script>
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/api/ws/status');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateDashboard(data);
};

function updateDashboard(data) {
    document.getElementById('active-jobs').textContent = data.active_jobs;
    document.getElementById('queue-length').textContent = data.queue_length;
    document.getElementById('processing-rate').textContent = data.processing_rate + '/hr';
    updateRecentJobs(data.recent_completions);
}
</script>
```

### 8.4 Idea Review and Selection Interface

#### Idea Management Endpoints
```python
# app/api/ideas.py
from fastapi import APIRouter, Depends, Query, Body
from typing import List, Optional
from app.schemas.idea import IdeaResponse, IdeaUpdate, IdeaBulkAction
from app.services.idea_service import IdeaService

router = APIRouter()

@router.get("/", response_model=List[IdeaResponse])
async def list_ideas(
    status: Optional[str] = Query(None, enum=["extracted", "reviewed", "selected", "rejected"]),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    category: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0),
    service: IdeaService = Depends()
):
    """List extracted ideas with filtering and pagination"""
    return await service.list_ideas(
        status=status,
        min_confidence=min_confidence,
        category=category,
        limit=limit,
        offset=offset
    )

@router.patch("/{idea_id}")
async def update_idea(
    idea_id: str,
    updates: IdeaUpdate,
    service: IdeaService = Depends()
):
    """Update idea status or metadata"""
    return await service.update_idea(idea_id, updates)

@router.post("/bulk-action")
async def bulk_action(
    action: IdeaBulkAction,
    idea_ids: List[str] = Body(...),
    service: IdeaService = Depends()
):
    """Perform bulk actions on multiple ideas"""
    return await service.bulk_action(action, idea_ids)

@router.get("/{idea_id}/context")
async def get_idea_context(
    idea_id: str,
    service: IdeaService = Depends()
):
    """Get full context including video and transcript for an idea"""
    return await service.get_idea_context(idea_id)
```

#### Idea Review UI
```html
<!-- templates/ideas/review.html -->
<div class="ideas-review-container">
    <div class="review-filters">
        <select name="status" hx-get="/admin/ideas" hx-target="#ideas-list">
            <option value="extracted">New Ideas</option>
            <option value="reviewed">Reviewed</option>
            <option value="selected">Selected</option>
            <option value="rejected">Rejected</option>
        </select>
        
        <input type="range" name="min_confidence" min="0" max="100" 
               hx-get="/admin/ideas" hx-target="#ideas-list" 
               hx-trigger="change throttle:500ms">
        
        <select name="category" hx-get="/admin/ideas" hx-target="#ideas-list">
            <option value="">All Categories</option>
            {% for category in categories %}
            <option value="{{ category }}">{{ category }}</option>
            {% endfor %}
        </select>
    </div>
    
    <div class="bulk-actions">
        <button onclick="bulkAction('select')">Select All Checked</button>
        <button onclick="bulkAction('reject')">Reject All Checked</button>
        <button onclick="exportSelected()">Export Selected</button>
    </div>
    
    <div id="ideas-list" class="ideas-grid">
        {% for idea in ideas %}
        <div class="idea-card" data-idea-id="{{ idea.id }}">
            <input type="checkbox" class="idea-checkbox" value="{{ idea.id }}">
            
            <div class="idea-header">
                <h3>{{ idea.title }}</h3>
                <span class="confidence-badge">
                    {{ (idea.confidence_score * 100)|round }}% confidence
                </span>
            </div>
            
            <div class="idea-content">
                <p>{{ idea.description }}</p>
                
                <div class="idea-metadata">
                    <span class="category">{{ idea.category }}</span>
                    <span class="complexity">Complexity: {{ idea.complexity_score }}/10</span>
                    <span class="market-size">{{ idea.market_size_estimate }}</span>
                </div>
                
                <div class="idea-source">
                    <a href="#" onclick="showVideoContext('{{ idea.id }}')">
                        View Source Video
                    </a>
                </div>
            </div>
            
            <div class="idea-actions">
                <button class="btn-success" 
                        hx-patch="/api/ideas/{{ idea.id }}"
                        hx-vals='{"status": "selected"}'>
                    Select
                </button>
                <button class="btn-warning"
                        hx-patch="/api/ideas/{{ idea.id }}"
                        hx-vals='{"status": "reviewed"}'>
                    Review Later
                </button>
                <button class="btn-danger"
                        hx-patch="/api/ideas/{{ idea.id }}"
                        hx-vals='{"status": "rejected"}'>
                    Reject
                </button>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
```

---

## Week 9: API and Integration

### 9.1 REST API Development

#### API Structure and Versioning
```python
# app/api/v1/__init__.py
from fastapi import APIRouter
from app.api.v1 import channels, videos, transcripts, ideas, export

api_router = APIRouter()

api_router.include_router(channels.router, prefix="/channels", tags=["channels"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(transcripts.router, prefix="/transcripts", tags=["transcripts"])
api_router.include_router(ideas.router, prefix="/ideas", tags=["ideas"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
```

#### Public API Endpoints
```python
# app/api/v1/public.py
from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.auth import APIKeyAuth
from app.schemas.api import PaginatedResponse
from typing import Optional

router = APIRouter()

@router.get("/ideas", response_model=PaginatedResponse[IdeaResponse])
async def get_ideas(
    api_key: str = Depends(APIKeyAuth()),
    status: Optional[str] = Query(None),
    min_confidence: float = Query(0.5),
    limit: int = Query(100, le=1000),
    offset: int = Query(0)
):
    """
    Retrieve extracted ideas with filtering
    
    - **status**: Filter by idea status (extracted, reviewed, selected)
    - **min_confidence**: Minimum confidence score (0.0-1.0)
    - **limit**: Maximum number of results
    - **offset**: Pagination offset
    """
    # Implementation

@router.get("/videos/{video_id}/ideas")
async def get_video_ideas(
    video_id: str,
    api_key: str = Depends(APIKeyAuth())
):
    """Get all ideas extracted from a specific video"""
    # Implementation

@router.post("/videos/process")
async def process_video(
    video_url: str = Body(...),
    priority: int = Body(0),
    api_key: str = Depends(APIKeyAuth())
):
    """Submit a video for processing"""
    # Implementation
```

### 9.2 Export Functionality

#### Export Service Implementation
```python
# app/services/export_service.py
from typing import List, Dict, Any
import pandas as pd
import json
from io import BytesIO

class ExportService:
    
    async def export_ideas_json(
        self, 
        idea_ids: List[str], 
        include_context: bool = False
    ) -> Dict[str, Any]:
        """Export ideas in JSON format"""
        ideas = await self.get_ideas_by_ids(idea_ids)
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "total_ideas": len(ideas),
            "ideas": []
        }
        
        for idea in ideas:
            idea_data = {
                "id": idea.id,
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "confidence_score": idea.confidence_score,
                "complexity_score": idea.complexity_score,
                "market_size_estimate": idea.market_size_estimate,
                "tags": idea.tags
            }
            
            if include_context:
                video = await self.get_video(idea.video_id)
                idea_data["source"] = {
                    "video_title": video.title,
                    "video_url": video.video_url,
                    "channel_name": video.channel.name,
                    "published_at": video.published_at.isoformat()
                }
            
            export_data["ideas"].append(idea_data)
        
        return export_data
    
    async def export_ideas_csv(
        self, 
        idea_ids: List[str]
    ) -> BytesIO:
        """Export ideas in CSV format"""
        ideas = await self.get_ideas_by_ids(idea_ids)
        
        data = []
        for idea in ideas:
            data.append({
                "ID": idea.id,
                "Title": idea.title,
                "Description": idea.description,
                "Category": idea.category,
                "Confidence Score": idea.confidence_score,
                "Complexity Score": idea.complexity_score,
                "Market Size": idea.market_size_estimate,
                "Tags": ", ".join(idea.tags),
                "Status": idea.status,
                "Created At": idea.created_at.isoformat()
            })
        
        df = pd.DataFrame(data)
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
    
    async def export_for_ideahunter(
        self, 
        idea_ids: List[str]
    ) -> Dict[str, Any]:
        """Export in IdeaHunter-compatible format"""
        ideas = await self.get_ideas_by_ids(idea_ids)
        
        return {
            "source": "TubeSensei",
            "export_version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "ideas": [
                {
                    "external_id": idea.id,
                    "title": idea.title,
                    "description": idea.description,
                    "category": idea.category,
                    "metadata": {
                        "confidence": idea.confidence_score,
                        "complexity": idea.complexity_score,
                        "market_size": idea.market_size_estimate,
                        "source_url": (await self.get_video(idea.video_id)).video_url,
                        "extraction_date": idea.created_at.isoformat()
                    }
                }
                for idea in ideas
            ]
        }
```

#### Export API Endpoints
```python
# app/api/v1/export.py
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
from app.services.export_service import ExportService

router = APIRouter()

@router.post("/json")
async def export_json(
    idea_ids: List[str] = Body(...),
    include_context: bool = Query(False),
    service: ExportService = Depends()
):
    """Export selected ideas in JSON format"""
    data = await service.export_ideas_json(idea_ids, include_context)
    return JSONResponse(content=data)

@router.post("/csv")
async def export_csv(
    idea_ids: List[str] = Body(...),
    service: ExportService = Depends()
):
    """Export selected ideas in CSV format"""
    buffer = await service.export_ideas_csv(idea_ids)
    
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ideas_export.csv"}
    )

@router.post("/ideahunter")
async def export_for_ideahunter(
    idea_ids: List[str] = Body(...),
    service: ExportService = Depends()
):
    """Export ideas in IdeaHunter-compatible format"""
    data = await service.export_for_ideahunter(idea_ids)
    return JSONResponse(content=data)

@router.get("/formats")
async def get_export_formats():
    """Get available export formats"""
    return {
        "formats": [
            {
                "format": "json",
                "description": "JSON format with full idea details",
                "supports_context": True
            },
            {
                "format": "csv",
                "description": "CSV format for spreadsheet applications",
                "supports_context": False
            },
            {
                "format": "ideahunter",
                "description": "IdeaHunter-compatible JSON format",
                "supports_context": True
            }
        ]
    }
```

### 9.3 Authentication and Authorization

#### API Key Management
```python
# app/core/api_auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.models.api_key import APIKey
from app.database import get_db

api_key_header = APIKeyHeader(name="X-API-Key")

class APIKeyAuth:
    async def __call__(self, api_key: str = Security(api_key_header)):
        db = get_db()
        
        # Validate API key
        key = await db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or inactive API key"
            )
        
        # Update last used timestamp
        key.last_used_at = datetime.utcnow()
        key.usage_count += 1
        await db.commit()
        
        return key

# Rate limiting decorator
from functools import wraps
from app.core.rate_limiter import RateLimiter

rate_limiter = RateLimiter()

def rate_limit(max_calls: int, time_window: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            api_key = kwargs.get('api_key')
            if api_key:
                allowed = await rate_limiter.check_rate_limit(
                    api_key.id, 
                    max_calls, 
                    time_window
                )
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### 9.4 API Documentation

#### OpenAPI Documentation Configuration
```python
# app/core/openapi.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TubeSensei API",
        version="1.0.0",
        description="""
        ## TubeSensei API Documentation
        
        TubeSensei is a YouTube content analysis platform that extracts business ideas from video content.
        
        ### Authentication
        All API endpoints require authentication via API key. Include your API key in the `X-API-Key` header.
        
        ### Rate Limiting
        - Standard tier: 100 requests per hour
        - Premium tier: 1000 requests per hour
        
        ### Available Endpoints
        - **Channels**: Manage YouTube channels
        - **Videos**: Access video metadata and processing status
        - **Ideas**: Retrieve and filter extracted ideas
        - **Export**: Export data in various formats
        
        ### Response Formats
        All endpoints return JSON responses with consistent error handling.
        """,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Apply security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [
                {"APIKeyHeader": []}
            ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
```

#### Interactive API Documentation
```python
# app/main.py additions
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )
```

---

## Technical Architecture

### Frontend Architecture

#### Technology Stack
- **Server-Side Rendering**: Jinja2 templates with FastAPI
- **Interactive UI**: HTMX for dynamic updates without full page reloads
- **Real-time Updates**: WebSockets for live status monitoring
- **Styling**: Tailwind CSS for responsive design
- **Charts**: Chart.js for data visualization

#### Component Structure
```
templates/
├── base.html                 # Base template with common layout
├── dashboard/
│   ├── index.html           # Main dashboard
│   └── status.html          # Processing status
├── channels/
│   ├── list.html           # Channel listing
│   ├── add.html            # Add channel form
│   └── edit.html           # Edit channel form
├── ideas/
│   ├── review.html         # Idea review interface
│   ├── detail.html         # Idea detail view
│   └── export.html         # Export selection
└── components/
    ├── pagination.html      # Reusable pagination
    ├── filters.html        # Filter components
    └── modal.html          # Modal dialog template
```

### Backend Architecture

#### Service Layer Pattern
```python
# app/services/base.py
from typing import Generic, TypeVar, Type
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db

ModelType = TypeVar("ModelType")

class BaseService(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(self, id: str) -> ModelType:
        async with get_db() as db:
            return await db.get(self.model, id)
    
    async def list(self, limit: int = 100, offset: int = 0):
        async with get_db() as db:
            result = await db.execute(
                select(self.model)
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    async def create(self, **kwargs) -> ModelType:
        async with get_db() as db:
            instance = self.model(**kwargs)
            db.add(instance)
            await db.commit()
            await db.refresh(instance)
            return instance
    
    async def update(self, id: str, **kwargs) -> ModelType:
        async with get_db() as db:
            instance = await db.get(self.model, id)
            for key, value in kwargs.items():
                setattr(instance, key, value)
            await db.commit()
            await db.refresh(instance)
            return instance
```

#### Dependency Injection
```python
# app/dependencies.py
from fastapi import Depends
from typing import Annotated
from app.services.channel_service import ChannelService
from app.services.video_service import VideoService
from app.services.idea_service import IdeaService

def get_channel_service() -> ChannelService:
    return ChannelService()

def get_video_service() -> VideoService:
    return VideoService()

def get_idea_service() -> IdeaService:
    return IdeaService()

# Type aliases for cleaner code
ChannelServiceDep = Annotated[ChannelService, Depends(get_channel_service)]
VideoServiceDep = Annotated[VideoService, Depends(get_video_service)]
IdeaServiceDep = Annotated[IdeaService, Depends(get_idea_service)]
```

---

## Implementation Details

### Database Queries Optimization

#### Efficient Pagination
```python
# app/utils/pagination.py
from typing import Generic, TypeVar, List
from pydantic import BaseModel

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool

async def paginate(query, limit: int, offset: int) -> PaginatedResponse:
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)
    
    # Get paginated results
    results = await db.execute(
        query.limit(limit).offset(offset)
    )
    items = results.scalars().all()
    
    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total
    )
```

#### Caching Strategy
```python
# app/core/cache.py
from redis import asyncio as aioredis
from typing import Optional, Any
import json

class CacheManager:
    def __init__(self):
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        return json.loads(value) if value else None
    
    async def set(self, key: str, value: Any, expire: int = 300):
        await self.redis.set(
            key, 
            json.dumps(value), 
            ex=expire
        )
    
    async def invalidate(self, pattern: str):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

# Usage in services
class ChannelService:
    def __init__(self):
        self.cache = CacheManager()
    
    async def get_channel_stats(self, channel_id: str):
        cache_key = f"channel_stats:{channel_id}"
        
        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Calculate stats
        stats = await self._calculate_stats(channel_id)
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, stats, expire=300)
        
        return stats
```

### Error Handling

#### Global Exception Handler
```python
# app/core/exceptions.py
from fastapi import Request, status
from fastapi.responses import JSONResponse

class TubeSenseiException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class NotFoundException(TubeSenseiException):
    def __init__(self, resource: str):
        super().__init__(
            message=f"{resource} not found",
            status_code=404
        )

class ValidationException(TubeSenseiException):
    def __init__(self, errors: dict):
        super().__init__(
            message="Validation failed",
            status_code=422
        )
        self.errors = errors

async def exception_handler(request: Request, exc: TubeSenseiException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

# Register in main app
app.add_exception_handler(TubeSenseiException, exception_handler)
```

### Logging and Monitoring

#### Structured Logging
```python
# app/core/logging.py
import structlog
from typing import Any, Dict

def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()

# Usage in endpoints
@router.get("/channels/{channel_id}")
async def get_channel(channel_id: str):
    logger.info("fetching_channel", channel_id=channel_id)
    try:
        channel = await service.get_channel(channel_id)
        logger.info("channel_fetched", channel_id=channel_id)
        return channel
    except Exception as e:
        logger.error("channel_fetch_failed", 
                    channel_id=channel_id, 
                    error=str(e))
        raise
```

---

## Testing Requirements

### Unit Tests

#### API Endpoint Tests
```python
# tests/api/test_channels.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_list_channels():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/channels")
        assert response.status_code == 200
        assert "items" in response.json()

@pytest.mark.asyncio
async def test_add_channel():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/channels",
            json={
                "name": "Test Channel",
                "youtube_channel_id": "UC123456",
                "channel_url": "https://youtube.com/c/testchannel"
            }
        )
        assert response.status_code == 201
        assert response.json()["name"] == "Test Channel"

@pytest.mark.asyncio
async def test_channel_not_found():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/channels/nonexistent")
        assert response.status_code == 404
```

#### Service Layer Tests
```python
# tests/services/test_idea_service.py
import pytest
from app.services.idea_service import IdeaService
from app.models.idea import Idea

@pytest.mark.asyncio
async def test_filter_ideas_by_confidence():
    service = IdeaService()
    
    # Create test ideas
    ideas = await service.list_ideas(min_confidence=0.7)
    
    # Verify all ideas meet confidence threshold
    for idea in ideas:
        assert idea.confidence_score >= 0.7

@pytest.mark.asyncio
async def test_bulk_update_status():
    service = IdeaService()
    
    # Create test ideas
    idea_ids = ["id1", "id2", "id3"]
    
    # Perform bulk update
    result = await service.bulk_action("select", idea_ids)
    
    # Verify all ideas updated
    assert result["updated"] == 3
    
    # Verify status changed
    for idea_id in idea_ids:
        idea = await service.get(idea_id)
        assert idea.status == "selected"
```

### Integration Tests

#### End-to-End Workflow Tests
```python
# tests/integration/test_workflow.py
import pytest
from app.tests.factories import ChannelFactory, VideoFactory

@pytest.mark.asyncio
async def test_complete_processing_workflow():
    # 1. Add a channel
    channel = await ChannelFactory.create()
    
    # 2. Discover videos
    videos = await discover_channel_videos(channel.id)
    assert len(videos) > 0
    
    # 3. Filter videos
    filtered = await filter_videos_ai(videos)
    assert len(filtered) <= len(videos)
    
    # 4. Extract transcripts
    for video in filtered:
        transcript = await extract_transcript(video.id)
        assert transcript is not None
    
    # 5. Extract ideas
    ideas = await extract_ideas_from_videos(filtered)
    assert len(ideas) > 0
    
    # 6. Export ideas
    export_data = await export_ideas_json([i.id for i in ideas])
    assert len(export_data["ideas"]) == len(ideas)
```

### Performance Tests

#### Load Testing
```python
# tests/performance/test_load.py
import asyncio
import time
from httpx import AsyncClient

async def test_concurrent_requests():
    async with AsyncClient(base_url="http://localhost:8000") as client:
        tasks = []
        
        # Create 100 concurrent requests
        for _ in range(100):
            task = client.get("/api/ideas")
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
        
        # Verify performance
        duration = end_time - start_time
        assert duration < 10  # Should complete within 10 seconds
        
        avg_response_time = duration / 100
        assert avg_response_time < 0.5  # Average < 500ms
```

---

## Success Criteria

### Week 8 Deliverables
- [ ] FastAPI application structure complete
- [ ] Authentication and authorization implemented
- [ ] Channel management UI functional
- [ ] Processing status dashboard with real-time updates
- [ ] Idea review interface with filtering and bulk actions
- [ ] Basic admin interface navigation
- [ ] WebSocket connections for live updates

### Week 9 Deliverables
- [ ] REST API endpoints implemented
- [ ] API authentication with key management
- [ ] Rate limiting configured
- [ ] Export functionality for all formats (JSON, CSV, IdeaHunter)
- [ ] OpenAPI documentation generated
- [ ] Interactive API documentation (Swagger/ReDoc)
- [ ] API client examples and SDK stubs

### Performance Metrics
- **API Response Time**: <500ms for standard requests
- **WebSocket Latency**: <100ms for status updates
- **Export Performance**: Handle 1000+ ideas in <5 seconds
- **Concurrent Users**: Support 10+ simultaneous admin users
- **API Rate Limits**: Properly enforce without impacting legitimate usage

### Quality Standards
- **Code Coverage**: >85% test coverage for new code
- **Documentation**: Complete API documentation with examples
- **Error Handling**: Graceful error handling with meaningful messages
- **Security**: Proper authentication, authorization, and input validation
- **User Experience**: Intuitive UI with responsive design

### Integration Requirements
- **Database Integration**: Seamless interaction with Phase 1 models
- **Job Queue Integration**: Real-time job status monitoring
- **AI Service Integration**: Display AI processing results
- **Export Compatibility**: IdeaHunter-compatible export format

---

## Conclusion

Phase 3 completes the TubeSensei platform by providing comprehensive user interfaces and API access. The admin interface enables efficient management of channels, monitoring of processing status, and review of extracted ideas. The REST API provides external integration capabilities with proper authentication, rate limiting, and multiple export formats. This phase transforms the backend processing engine into a complete, production-ready platform for YouTube content analysis and idea extraction.