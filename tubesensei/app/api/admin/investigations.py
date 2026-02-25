"""Admin routes for running investigations on ideas."""
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.database import get_db
from app.models.investigation_agent import InvestigationAgent
from app.services.investigation_runner import InvestigationRunner
from .template_helpers import get_template_context

logger = logging.getLogger(__name__)

template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/investigations", tags=["admin-investigations"])


@router.post("/run", response_class=HTMLResponse)
async def run_investigation(
    request: Request,
    agent_id: UUID = Form(...),
    idea_id: UUID = Form(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    runner = InvestigationRunner(db)
    run = await runner.run_investigation(agent_id=agent_id, idea_id=idea_id)

    # Return the full investigation results section for the idea
    runs = await runner.get_runs_for_idea(idea_id)

    # Get agents for dropdown
    agents_result = await db.execute(
        select(InvestigationAgent)
        .where(InvestigationAgent.is_active == True)
        .order_by(InvestigationAgent.name)
    )
    agents = agents_result.scalars().all()

    context = get_template_context(
        request,
        user=user,
        idea_id=str(idea_id),
        agents=agents,
        runs=runs,
        latest_run=run,
    )
    return templates.TemplateResponse(
        "admin/ideas/partials/investigation_section.html", context
    )


@router.get("/idea/{idea_id}", response_class=HTMLResponse)
async def get_idea_investigations(
    request: Request,
    idea_id: UUID,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    runner = InvestigationRunner(db)
    runs = await runner.get_runs_for_idea(idea_id)

    agents_result = await db.execute(
        select(InvestigationAgent)
        .where(InvestigationAgent.is_active == True)
        .order_by(InvestigationAgent.name)
    )
    agents = agents_result.scalars().all()

    context = get_template_context(
        request,
        user=user,
        idea_id=str(idea_id),
        agents=agents,
        runs=runs,
    )
    return templates.TemplateResponse(
        "admin/ideas/partials/investigation_section.html", context
    )
