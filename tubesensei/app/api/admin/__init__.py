"""Admin API router module."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# from .dashboard import router as dashboard_router  # Temporarily disabled
from .channels import router as channels_router
from .topic_campaigns import router as topic_campaigns_router
from .videos import router as videos_router
from .jobs import router as jobs_router
from .transcripts import router as transcripts_router
from .settings import router as settings_router
from .monitoring import router as monitoring_router
from .template_helpers import get_template_context

# Set up templates
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Create main admin router
router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Main admin dashboard - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

@router.get("/ideas", response_class=HTMLResponse)
async def admin_ideas(request: Request):
    """Ideas page - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

# Include sub-routers
# router.include_router(dashboard_router)  # Temporarily disabled
router.include_router(channels_router)
router.include_router(topic_campaigns_router)
router.include_router(videos_router)
router.include_router(jobs_router)
router.include_router(transcripts_router)
router.include_router(settings_router)
router.include_router(monitoring_router)