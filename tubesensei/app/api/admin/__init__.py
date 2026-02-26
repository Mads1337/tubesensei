"""Admin API router module."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from .dashboard import router as dashboard_router, dashboard_page
from .channels import router as channels_router
from .topic_campaigns import router as topic_campaigns_router
from .videos import router as videos_router
from .jobs import router as jobs_router
from .transcripts import router as transcripts_router
from .settings import router as settings_router
from .monitoring import router as monitoring_router
from .ideas import router as ideas_router
from .quick_analysis import router as quick_analysis_router
from .investigation_agents import router as investigation_agents_router
from .investigations import router as investigations_router
from .roadmap import router as roadmap_router
from app.core.auth import get_current_user
from app.database import get_db

# Create main admin router
router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/", response_class=HTMLResponse)
async def admin_root(
    request: Request,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    """Render admin dashboard directly at /admin/."""
    return await dashboard_page(request=request, user=user, db=db)

# Include sub-routers
router.include_router(dashboard_router)
router.include_router(channels_router)
router.include_router(topic_campaigns_router)
router.include_router(videos_router)
router.include_router(jobs_router)
router.include_router(transcripts_router)
router.include_router(settings_router)
router.include_router(monitoring_router)
router.include_router(ideas_router)
router.include_router(quick_analysis_router)
router.include_router(investigation_agents_router)
router.include_router(investigations_router)
router.include_router(roadmap_router)