"""Admin API router module."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# from .dashboard import router as dashboard_router  # Temporarily disabled
from .channels import router as channels_router
from .template_helpers import get_template_context

# Set up templates
template_dir = Path(__file__).parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# Create main admin router
router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Main admin dashboard - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

@router.get("/videos", response_class=HTMLResponse)
async def admin_videos(request: Request):
    """Videos page - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

@router.get("/ideas", response_class=HTMLResponse)
async def admin_ideas(request: Request):
    """Ideas page - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

@router.get("/jobs", response_class=HTMLResponse)
async def admin_jobs(request: Request):
    """Jobs page - redirect to channels for now"""
    return RedirectResponse(url="/admin/channels/", status_code=302)

# Include sub-routers
# router.include_router(dashboard_router)  # Temporarily disabled
router.include_router(channels_router)