"""Admin Roadmap page — visual project roadmap."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.core.auth import get_current_user
from .template_helpers import get_template_context

template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/roadmap", tags=["admin-roadmap"])


@router.get("/", response_class=HTMLResponse)
async def roadmap_page(
    request: Request,
    user=Depends(get_current_user),
):
    """Render visual project roadmap."""
    context = get_template_context(request, user=user)
    return templates.TemplateResponse("admin/roadmap/index.html", context)
