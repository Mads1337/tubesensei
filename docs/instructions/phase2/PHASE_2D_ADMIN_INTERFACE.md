# Phase 2D: Admin Interface & Review System (Week 4)

## Objectives
- Build FastAPI admin interface
- Create idea review dashboard
- Implement export functionality
- Add performance monitoring

## Implementation Steps

### Step 1: Create Admin API Routes

**File:** `tubesensei/app/api/admin_routes.py`
```python
"""
Admin API routes for content review and management.

This module provides:
- Dashboard endpoints
- Idea review and management
- Export functionality
- System monitoring
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import csv
import io
import json

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..models.channel import Channel
from ..models.video import Video, VideoStatus
from ..models.idea import Idea, IdeaStatus
from ..models.processing_job import ProcessingJob, JobStatus
from ..ai.video_filter import VideoFilter, FilteringFeedback
from ..ai.idea_extractor import IdeaExtractor
from ..services.export_service import ExportService
from ..auth import get_current_user
from ..schemas import (
    IdeaResponse,
    VideoResponse,
    ChannelResponse,
    DashboardStats,
    ExportRequest
)

router = APIRouter(prefix="/admin", tags=["admin"])

# Initialize services
video_filter = VideoFilter()
idea_extractor = IdeaExtractor()
export_service = ExportService()
feedback_system = FilteringFeedback()

@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Get dashboard statistics."""
    
    # Get counts
    total_channels = await session.scalar(
        select(func.count(Channel.id))
    )
    
    total_videos = await session.scalar(
        select(func.count(Video.id))
    )
    
    processed_videos = await session.scalar(
        select(func.count(Video.id)).where(
            Video.status.in_([VideoStatus.COMPLETED, VideoStatus.PROCESSING])
        )
    )
    
    total_ideas = await session.scalar(
        select(func.count(Idea.id))
    )
    
    reviewed_ideas = await session.scalar(
        select(func.count(Idea.id)).where(
            Idea.status.in_([IdeaStatus.REVIEWED, IdeaStatus.SELECTED])
        )
    )
    
    # Get recent activity
    recent_jobs = await session.scalar(
        select(func.count(ProcessingJob.id)).where(
            ProcessingJob.created_at >= datetime.utcnow() - timedelta(hours=24)
        )
    )
    
    # Calculate rates
    processing_rate = processed_videos / max(total_videos, 1)
    review_rate = reviewed_ideas / max(total_ideas, 1)
    
    # Get AI metrics
    ai_metrics = video_filter.get_metrics()
    
    return DashboardStats(
        total_channels=total_channels,
        total_videos=total_videos,
        processed_videos=processed_videos,
        total_ideas=total_ideas,
        reviewed_ideas=reviewed_ideas,
        recent_activity=recent_jobs,
        processing_rate=processing_rate,
        review_rate=review_rate,
        ai_filtering_accuracy=ai_metrics.get("average_confidence", 0),
        total_ai_cost=sum(idea_extractor.llm_manager.get_cost_report().values())
    )

@router.get("/ideas", response_model=List[IdeaResponse])
async def get_ideas(
    status: Optional[IdeaStatus] = None,
    category: Optional[str] = None,
    min_quality: Optional[float] = None,
    limit: int = Query(50, le=100),
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Get ideas with filtering and pagination."""
    
    query = select(Idea).order_by(Idea.confidence_score.desc())
    
    # Apply filters
    filters = []
    if status:
        filters.append(Idea.status == status)
    if category:
        filters.append(Idea.category == category)
    if min_quality:
        filters.append(Idea.confidence_score >= min_quality)
    
    if filters:
        query = query.where(and_(*filters))
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    result = await session.execute(query)
    ideas = result.scalars().all()
    
    return [IdeaResponse.from_orm(idea) for idea in ideas]

@router.patch("/ideas/{idea_id}/review")
async def review_idea(
    idea_id: str,
    status: IdeaStatus,
    notes: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Review and update idea status."""
    
    idea = await session.get(Idea, idea_id)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")
    
    # Update status
    idea.status = status
    idea.updated_at = datetime.utcnow()
    
    # Add review notes
    if notes:
        idea.metadata = idea.metadata or {}
        idea.metadata["review_notes"] = notes
        idea.metadata["reviewed_by"] = current_user.id
        idea.metadata["reviewed_at"] = datetime.utcnow().isoformat()
    
    await session.commit()
    
    return {"message": "Idea reviewed successfully", "idea_id": idea_id}

@router.post("/ideas/{idea_id}/feedback")
async def submit_idea_feedback(
    idea_id: str,
    is_valuable: bool,
    reason: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Submit feedback on AI idea extraction."""
    
    idea = await session.get(Idea, idea_id)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")
    
    # Get original AI decision
    ai_valuable = idea.confidence_score > 0.5
    
    # Record feedback
    await feedback_system.record_feedback(
        video_id=idea.video_id,
        ai_decision=ai_valuable,
        human_decision=is_valuable,
        reason=reason
    )
    
    # Update idea with feedback
    idea.metadata = idea.metadata or {}
    idea.metadata["human_feedback"] = {
        "is_valuable": is_valuable,
        "reason": reason,
        "feedback_by": current_user.id,
        "feedback_at": datetime.utcnow().isoformat()
    }
    
    await session.commit()
    
    # Get updated accuracy metrics
    accuracy = feedback_system.calculate_accuracy()
    
    return {
        "message": "Feedback recorded",
        "current_accuracy": accuracy
    }

@router.get("/videos/review-queue", response_model=List[VideoResponse])
async def get_review_queue(
    limit: int = Query(20, le=50),
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Get videos queued for review."""
    
    query = select(Video).where(
        Video.status == VideoStatus.QUEUED
    ).order_by(
        Video.ai_evaluation_score.desc(),
        Video.view_count.desc()
    ).limit(limit)
    
    result = await session.execute(query)
    videos = result.scalars().all()
    
    return [VideoResponse.from_orm(video) for video in videos]

@router.post("/videos/{video_id}/process")
async def process_video(
    video_id: str,
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Manually trigger video processing."""
    
    video = await session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Create processing job
    from ..workers.tasks import process_video_task
    
    task = process_video_task.delay(video_id)
    
    # Update video status
    video.status = VideoStatus.PROCESSING
    await session.commit()
    
    return {
        "message": "Video processing started",
        "task_id": task.id,
        "video_id": video_id
    }

@router.post("/export/ideas")
async def export_ideas(
    request: ExportRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Export ideas in various formats."""
    
    # Build query
    query = select(Idea)
    
    filters = []
    if request.status:
        filters.append(Idea.status == request.status)
    if request.category:
        filters.append(Idea.category == request.category)
    if request.min_quality:
        filters.append(Idea.confidence_score >= request.min_quality)
    if request.date_from:
        filters.append(Idea.created_at >= request.date_from)
    if request.date_to:
        filters.append(Idea.created_at <= request.date_to)
    
    if filters:
        query = query.where(and_(*filters))
    
    result = await session.execute(query)
    ideas = result.scalars().all()
    
    # Export based on format
    if request.format == "json":
        data = [
            {
                "id": str(idea.id),
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "confidence_score": idea.confidence_score,
                "complexity_score": idea.complexity_score,
                "market_size": idea.market_size_estimate,
                "status": idea.status.value,
                "tags": idea.tags,
                "video_title": idea.video.title if idea.video else None,
                "channel": idea.video.channel.name if idea.video and idea.video.channel else None,
                "created_at": idea.created_at.isoformat()
            }
            for idea in ideas
        ]
        
        return {
            "format": "json",
            "count": len(ideas),
            "data": data
        }
    
    elif request.format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id", "title", "description", "category",
                "confidence_score", "complexity_score",
                "market_size", "status", "tags",
                "video_title", "channel", "created_at"
            ]
        )
        
        writer.writeheader()
        for idea in ideas:
            writer.writerow({
                "id": str(idea.id),
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "confidence_score": idea.confidence_score,
                "complexity_score": idea.complexity_score,
                "market_size": idea.market_size_estimate,
                "status": idea.status.value,
                "tags": ", ".join(idea.tags) if idea.tags else "",
                "video_title": idea.video.title if idea.video else "",
                "channel": idea.video.channel.name if idea.video and idea.video.channel else "",
                "created_at": idea.created_at.isoformat()
            })
        
        return {
            "format": "csv",
            "count": len(ideas),
            "data": output.getvalue()
        }
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@router.get("/monitoring/ai-costs")
async def get_ai_costs(
    current_user = Depends(get_current_user)
):
    """Get AI processing cost breakdown."""
    
    costs = idea_extractor.llm_manager.get_cost_report()
    video_costs = video_filter.llm_manager.get_cost_report()
    
    # Merge costs
    all_costs = costs.copy()
    for model, cost in video_costs.items():
        all_costs[model] = all_costs.get(model, 0) + cost
    
    total = sum(all_costs.values())
    
    return {
        "total_cost": total,
        "breakdown_by_model": all_costs,
        "video_filtering_cost": sum(video_costs.values()),
        "idea_extraction_cost": sum(costs.values())
    }

@router.get("/monitoring/processing-stats")
async def get_processing_stats(
    session: AsyncSession = Depends(get_db_session),
    current_user = Depends(get_current_user)
):
    """Get processing performance statistics."""
    
    # Get job statistics
    total_jobs = await session.scalar(
        select(func.count(ProcessingJob.id))
    )
    
    completed_jobs = await session.scalar(
        select(func.count(ProcessingJob.id)).where(
            ProcessingJob.status == JobStatus.COMPLETED
        )
    )
    
    failed_jobs = await session.scalar(
        select(func.count(ProcessingJob.id)).where(
            ProcessingJob.status == JobStatus.FAILED
        )
    )
    
    # Get average processing times
    avg_time_result = await session.execute(
        select(
            ProcessingJob.job_type,
            func.avg(
                func.extract('epoch', ProcessingJob.completed_at - ProcessingJob.started_at)
            ).label('avg_seconds')
        ).where(
            ProcessingJob.status == JobStatus.COMPLETED,
            ProcessingJob.completed_at.isnot(None),
            ProcessingJob.started_at.isnot(None)
        ).group_by(ProcessingJob.job_type)
    )
    
    avg_times = {row.job_type: row.avg_seconds for row in avg_time_result}
    
    # Get AI filtering metrics
    ai_metrics = video_filter.get_metrics()
    feedback_accuracy = feedback_system.calculate_accuracy()
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "success_rate": completed_jobs / max(total_jobs, 1),
        "average_processing_times": avg_times,
        "ai_filtering_metrics": ai_metrics,
        "ai_accuracy": feedback_accuracy
    }
```

### Step 2: Create Frontend Templates

**File:** `tubesensei/app/templates/admin_dashboard.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TubeSensei Admin Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
</head>
<body class="bg-gray-100">
    <div x-data="dashboard()" x-init="init()">
        <!-- Header -->
        <header class="bg-white shadow">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between items-center py-6">
                    <h1 class="text-3xl font-bold text-gray-900">TubeSensei Dashboard</h1>
                    <div class="flex space-x-4">
                        <button @click="refreshData()" 
                                class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                            Refresh
                        </button>
                        <button @click="showExportModal = true" 
                                class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                            Export Ideas
                        </button>
                    </div>
                </div>
            </div>
        </header>

        <!-- Stats Grid -->
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <!-- Total Channels -->
                <div class="bg-white overflow-hidden shadow rounded-lg">
                    <div class="p-5">
                        <div class="flex items-center">
                            <div class="flex-shrink-0">
                                <svg class="h-6 w-6 text-gray-400" fill="none" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                          d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                                </svg>
                            </div>
                            <div class="ml-5 w-0 flex-1">
                                <dl>
                                    <dt class="text-sm font-medium text-gray-500 truncate">
                                        Total Channels
                                    </dt>
                                    <dd class="text-lg font-medium text-gray-900">
                                        <span x-text="stats.total_channels">0</span>
                                    </dd>
                                </dl>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Total Videos -->
                <div class="bg-white overflow-hidden shadow rounded-lg">
                    <div class="p-5">
                        <div class="flex items-center">
                            <div class="ml-5 w-0 flex-1">
                                <dl>
                                    <dt class="text-sm font-medium text-gray-500 truncate">
                                        Videos Processed
                                    </dt>
                                    <dd class="text-lg font-medium text-gray-900">
                                        <span x-text="stats.processed_videos">0</span> / 
                                        <span x-text="stats.total_videos">0</span>
                                    </dd>
                                </dl>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Ideas Extracted -->
                <div class="bg-white overflow-hidden shadow rounded-lg">
                    <div class="p-5">
                        <div class="flex items-center">
                            <div class="ml-5 w-0 flex-1">
                                <dl>
                                    <dt class="text-sm font-medium text-gray-500 truncate">
                                        Ideas Extracted
                                    </dt>
                                    <dd class="text-lg font-medium text-gray-900">
                                        <span x-text="stats.total_ideas">0</span>
                                    </dd>
                                </dl>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- AI Cost -->
                <div class="bg-white overflow-hidden shadow rounded-lg">
                    <div class="p-5">
                        <div class="flex items-center">
                            <div class="ml-5 w-0 flex-1">
                                <dl>
                                    <dt class="text-sm font-medium text-gray-500 truncate">
                                        Total AI Cost
                                    </dt>
                                    <dd class="text-lg font-medium text-gray-900">
                                        $<span x-text="stats.total_ai_cost?.toFixed(2)">0.00</span>
                                    </dd>
                                </dl>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Ideas Table -->
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
            <div class="bg-white shadow overflow-hidden sm:rounded-lg">
                <div class="px-4 py-5 sm:px-6">
                    <h3 class="text-lg leading-6 font-medium text-gray-900">
                        Recent Ideas
                    </h3>
                </div>
                <div class="border-t border-gray-200">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Title
                                </th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Category
                                </th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Quality
                                </th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Status
                                </th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Actions
                                </th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                            <template x-for="idea in ideas" :key="idea.id">
                                <tr>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <div class="text-sm text-gray-900" x-text="idea.title"></div>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800"
                                              x-text="idea.category">
                                        </span>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <div class="text-sm text-gray-900" x-text="(idea.confidence_score * 100).toFixed(0) + '%'"></div>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                                              :class="getStatusClass(idea.status)"
                                              x-text="idea.status">
                                        </span>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                        <button @click="reviewIdea(idea)" class="text-indigo-600 hover:text-indigo-900 mr-2">
                                            Review
                                        </button>
                                        <button @click="exportIdea(idea)" class="text-green-600 hover:text-green-900">
                                            Export
                                        </button>
                                    </td>
                                </tr>
                            </template>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Export Modal -->
        <div x-show="showExportModal" class="fixed z-10 inset-0 overflow-y-auto" style="display: none;">
            <div class="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                <div class="fixed inset-0 transition-opacity" aria-hidden="true">
                    <div class="absolute inset-0 bg-gray-500 opacity-75"></div>
                </div>
                <div class="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                    <div class="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                        <h3 class="text-lg leading-6 font-medium text-gray-900">
                            Export Ideas
                        </h3>
                        <div class="mt-4">
                            <label class="block text-sm font-medium text-gray-700">Format</label>
                            <select x-model="exportFormat" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                                <option value="json">JSON</option>
                                <option value="csv">CSV</option>
                            </select>
                        </div>
                        <div class="mt-4">
                            <label class="block text-sm font-medium text-gray-700">Status Filter</label>
                            <select x-model="exportStatus" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                                <option value="">All</option>
                                <option value="extracted">Extracted</option>
                                <option value="reviewed">Reviewed</option>
                                <option value="selected">Selected</option>
                            </select>
                        </div>
                    </div>
                    <div class="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                        <button @click="performExport()" type="button" class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm">
                            Export
                        </button>
                        <button @click="showExportModal = false" type="button" class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function dashboard() {
            return {
                stats: {},
                ideas: [],
                showExportModal: false,
                exportFormat: 'json',
                exportStatus: '',
                
                async init() {
                    await this.refreshData();
                },
                
                async refreshData() {
                    // Fetch stats
                    const statsResponse = await fetch('/admin/dashboard/stats');
                    this.stats = await statsResponse.json();
                    
                    // Fetch recent ideas
                    const ideasResponse = await fetch('/admin/ideas?limit=10');
                    this.ideas = await ideasResponse.json();
                },
                
                getStatusClass(status) {
                    const classes = {
                        'extracted': 'bg-yellow-100 text-yellow-800',
                        'reviewed': 'bg-blue-100 text-blue-800',
                        'selected': 'bg-green-100 text-green-800',
                        'rejected': 'bg-red-100 text-red-800'
                    };
                    return classes[status] || 'bg-gray-100 text-gray-800';
                },
                
                async reviewIdea(idea) {
                    const status = prompt('Set status (extracted/reviewed/selected/rejected):');
                    if (status) {
                        const response = await fetch(`/admin/ideas/${idea.id}/review`, {
                            method: 'PATCH',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({status: status})
                        });
                        
                        if (response.ok) {
                            await this.refreshData();
                        }
                    }
                },
                
                async performExport() {
                    const params = new URLSearchParams({
                        format: this.exportFormat,
                        status: this.exportStatus
                    });
                    
                    const response = await fetch('/admin/export/ideas?' + params, {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    
                    if (this.exportFormat === 'json') {
                        const blob = new Blob([JSON.stringify(data.data, null, 2)], {type: 'application/json'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'ideas.json';
                        a.click();
                    } else if (this.exportFormat === 'csv') {
                        const blob = new Blob([data.data], {type: 'text/csv'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'ideas.csv';
                        a.click();
                    }
                    
                    this.showExportModal = false;
                },
                
                async exportIdea(idea) {
                    const blob = new Blob([JSON.stringify(idea, null, 2)], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `idea_${idea.id}.json`;
                    a.click();
                }
            };
        }
    </script>
</body>
</html>
```

## Testing Phase 2D

**File:** `tubesensei/tests/test_api/test_admin_routes.py`
```python
"""Tests for admin API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from tubesensei.app.main import app

client = TestClient(app)

def test_dashboard_stats():
    """Test dashboard statistics endpoint."""
    with patch('tubesensei.app.api.admin_routes.get_current_user') as mock_auth:
        mock_auth.return_value = Mock(id="test-user")
        
        response = client.get("/admin/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_channels" in data
        assert "total_videos" in data
        assert "total_ideas" in data

def test_get_ideas():
    """Test ideas listing endpoint."""
    with patch('tubesensei.app.api.admin_routes.get_current_user') as mock_auth:
        mock_auth.return_value = Mock(id="test-user")
        
        response = client.get("/admin/ideas?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

def test_export_ideas():
    """Test ideas export endpoint."""
    with patch('tubesensei.app.api.admin_routes.get_current_user') as mock_auth:
        mock_auth.return_value = Mock(id="test-user")
        
        response = client.post("/admin/export/ideas", json={
            "format": "json",
            "status": "extracted"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "json"
        assert "data" in data
```

## Key Features

### 1. Dashboard Statistics
- Real-time metrics display
- Processing progress tracking
- AI cost monitoring
- Activity monitoring

### 2. Idea Management
- List ideas with filtering
- Review and status updates
- Feedback submission
- Quality assessment

### 3. Export Functionality
- JSON and CSV export formats
- Flexible filtering options
- Batch export capability
- Single idea export

### 4. Video Queue Management
- Review queue display
- Manual processing triggers
- Priority-based ordering
- Status tracking

### 5. Monitoring & Analytics
- AI cost breakdown
- Processing performance stats
- Success rate tracking
- Accuracy metrics

## API Endpoints

### Dashboard
- `GET /admin/dashboard/stats` - Get overall statistics

### Ideas
- `GET /admin/ideas` - List ideas with filtering
- `PATCH /admin/ideas/{id}/review` - Review an idea
- `POST /admin/ideas/{id}/feedback` - Submit feedback

### Videos
- `GET /admin/videos/review-queue` - Get review queue
- `POST /admin/videos/{id}/process` - Trigger processing

### Export
- `POST /admin/export/ideas` - Export ideas

### Monitoring
- `GET /admin/monitoring/ai-costs` - Get AI costs
- `GET /admin/monitoring/processing-stats` - Get processing stats

## Frontend Features

### Dashboard View
- Statistics cards
- Recent ideas table
- Quick actions
- Export modal

### Interactive Elements
- Alpine.js for reactivity
- TailwindCSS for styling
- Real-time updates
- Responsive design

## Configuration

Configure the admin interface via environment variables:

```bash
# Admin settings
ADMIN_ITEMS_PER_PAGE=50
ADMIN_MAX_EXPORT_ITEMS=1000
ADMIN_REFRESH_INTERVAL=30

# Authentication
ADMIN_AUTH_ENABLED=true
ADMIN_SESSION_TIMEOUT=3600
```

## Security Considerations

1. **Authentication Required**
   - All endpoints require authentication
   - User context tracked for actions

2. **Authorization**
   - Role-based access control
   - Action logging

3. **Rate Limiting**
   - Export endpoint rate limited
   - Processing triggers limited

4. **Data Protection**
   - Sensitive data masked in exports
   - Audit trail for reviews

## Validation Checklist

- [ ] Dashboard displays correctly
- [ ] Statistics update in real-time
- [ ] Ideas list with proper pagination
- [ ] Review interface functional
- [ ] Feedback submission works
- [ ] Export generates valid files
- [ ] JSON export format correct
- [ ] CSV export format correct
- [ ] Video queue displays properly
- [ ] Manual processing triggers work
- [ ] Cost monitoring accurate
- [ ] Performance metrics calculated correctly
- [ ] Authentication enforced
- [ ] Error handling robust
- [ ] UI responsive on mobile