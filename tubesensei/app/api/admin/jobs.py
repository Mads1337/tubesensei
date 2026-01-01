"""Admin Jobs API router module."""

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
from uuid import UUID
from sqlalchemy import select, func, and_
from sqlalchemy.orm import joinedload

from app.core.auth import get_current_user
from app.models.processing_job import ProcessingJob, JobType, JobStatus, JobPriority
from app.database import get_db
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/jobs", tags=["admin-jobs"])


@router.get("/", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    page: int = Query(1, ge=1),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render jobs management page with filtering"""

    limit = settings.admin.ADMIN_PAGINATION_DEFAULT
    offset = (page - 1) * limit

    # Build query
    query = select(ProcessingJob).options(joinedload(ProcessingJob.session))
    count_query = select(func.count(ProcessingJob.id))

    # Apply filters
    filters = []

    if status:
        try:
            job_status = JobStatus(status)
            filters.append(ProcessingJob.status == job_status)
        except ValueError:
            pass

    if job_type:
        try:
            jtype = JobType(job_type)
            filters.append(ProcessingJob.job_type == jtype)
        except ValueError:
            pass

    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))

    # Get total count
    total = await db.scalar(count_query)

    # Apply sorting (newest first)
    query = query.order_by(ProcessingJob.created_at.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    jobs = result.scalars().unique().all()

    # Convert to dicts
    job_list = []
    for job in jobs:
        job_dict = {
            "id": str(job.id),
            "job_type": job.job_type.value if job.job_type else None,
            "status": job.status.value if job.status else None,
            "priority": job.priority.value if job.priority else None,
            "entity_type": job.entity_type,
            "entity_id": str(job.entity_id) if job.entity_id else None,
            "progress_percent": job.progress_percent,
            "progress_message": job.progress_message,
            "error_message": job.error_message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "created_at": job.created_at,
            "execution_time_seconds": job.execution_time_seconds,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "can_retry": job.can_retry,
            "is_running": job.is_running,
        }
        job_list.append(job_dict)

    total_pages = (total + limit - 1) // limit if total else 1

    # Get stats for header
    stats_query = select(
        func.count(ProcessingJob.id).label("total"),
        func.count(ProcessingJob.id).filter(ProcessingJob.status == JobStatus.PENDING).label("pending"),
        func.count(ProcessingJob.id).filter(ProcessingJob.status == JobStatus.RUNNING).label("running"),
        func.count(ProcessingJob.id).filter(ProcessingJob.status == JobStatus.COMPLETED).label("completed"),
        func.count(ProcessingJob.id).filter(ProcessingJob.status == JobStatus.FAILED).label("failed"),
    )
    stats_result = await db.execute(stats_query)
    stats = stats_result.one()._asdict()

    # Check if this is an HTMX request for just the table body
    is_htmx = request.headers.get("HX-Request") == "true"

    context = get_template_context(
        request,
        user=user,
        jobs=job_list,
        total=total,
        page=page,
        total_pages=total_pages,
        stats=stats,
        filters={
            "status": status,
            "job_type": job_type,
        },
        job_statuses=[s.value for s in JobStatus],
        job_types=[t.value for t in JobType],
    )

    if is_htmx:
        return templates.TemplateResponse("admin/jobs/partials/jobs_table.html", context)

    return templates.TemplateResponse("admin/jobs/list.html", context)


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_detail(
    request: Request,
    job_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render job detail page"""

    query = select(ProcessingJob).options(joinedload(ProcessingJob.session)).where(ProcessingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = {
        "id": str(job.id),
        "job_type": job.job_type.value if job.job_type else None,
        "status": job.status.value if job.status else None,
        "priority": job.priority.value if job.priority else None,
        "entity_type": job.entity_type,
        "entity_id": str(job.entity_id) if job.entity_id else None,
        "progress_percent": job.progress_percent,
        "progress_message": job.progress_message,
        "error_message": job.error_message,
        "error_details": job.error_details,
        "input_data": job.input_data,
        "output_data": job.output_data,
        "job_metadata": job.job_metadata,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "created_at": job.created_at,
        "scheduled_at": job.scheduled_at,
        "execution_time_seconds": job.execution_time_seconds,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries,
        "worker_id": job.worker_id,
        "can_retry": job.can_retry,
        "is_running": job.is_running,
        "is_complete": job.is_complete,
    }

    context = get_template_context(
        request,
        user=user,
        job=job_dict,
    )

    return templates.TemplateResponse("admin/jobs/detail.html", context)


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Retry a failed job"""

    query = select(ProcessingJob).where(ProcessingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.can_retry:
        raise HTTPException(status_code=400, detail="Job cannot be retried")

    job.retry(delay_seconds=0)
    job.status = JobStatus.PENDING
    await db.commit()

    return {"success": True, "message": "Job queued for retry"}


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Cancel a pending or running job"""

    query = select(ProcessingJob).where(ProcessingJob.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.is_complete:
        raise HTTPException(status_code=400, detail="Job already completed")

    job.cancel()
    await db.commit()

    return {"success": True, "message": "Job cancelled"}
