"""Admin Monitoring API router module."""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from app.core.auth import get_current_user
from app.database import get_db
from app.services.monitoring_service import MonitoringService
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/monitoring", tags=["admin-monitoring"])


@router.get("/", response_class=HTMLResponse)
async def monitoring_page(
    request: Request,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render monitoring dashboard"""

    monitoring = MonitoringService(db)

    # Get all monitoring data
    system_status = await monitoring.get_system_status()
    processing_stats = await monitoring.get_processing_stats()
    queue_status = await monitoring.get_queue_status()
    recent_jobs = await monitoring.get_recent_jobs(limit=10)
    error_summary = await monitoring.get_error_summary()

    context = get_template_context(
        request,
        user=user,
        system_status=system_status,
        processing_stats=processing_stats,
        queue_status=queue_status,
        recent_jobs=recent_jobs,
        error_summary=error_summary,
    )

    return templates.TemplateResponse("admin/monitoring/index.html", context)


@router.get("/health", response_class=HTMLResponse)
async def health_partial(
    request: Request,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Health status partial for HTMX polling"""

    monitoring = MonitoringService(db)
    system_status = await monitoring.get_system_status()

    context = get_template_context(
        request,
        user=user,
        system_status=system_status,
    )

    return templates.TemplateResponse("admin/monitoring/partials/health_cards.html", context)


@router.get("/stats", response_class=HTMLResponse)
async def stats_partial(
    request: Request,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Processing stats partial for HTMX polling"""

    monitoring = MonitoringService(db)
    processing_stats = await monitoring.get_processing_stats()
    queue_status = await monitoring.get_queue_status()

    context = get_template_context(
        request,
        user=user,
        processing_stats=processing_stats,
        queue_status=queue_status,
    )

    return templates.TemplateResponse("admin/monitoring/partials/stats_cards.html", context)


@router.get("/errors", response_class=HTMLResponse)
async def errors_partial(
    request: Request,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Error summary partial for HTMX polling"""

    monitoring = MonitoringService(db)
    error_summary = await monitoring.get_error_summary()

    context = get_template_context(
        request,
        user=user,
        error_summary=error_summary,
    )

    return templates.TemplateResponse("admin/monitoring/partials/error_list.html", context)


@router.get("/timeline")
async def processing_timeline(
    hours: int = Query(24, ge=1, le=168),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get processing timeline data for charts"""

    monitoring = MonitoringService(db)
    timeline = await monitoring.get_processing_timeline(hours=hours)

    return timeline
