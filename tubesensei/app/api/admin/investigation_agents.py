"""Admin Investigation Agents API router module."""

from pathlib import Path
from uuid import UUID
from typing import Optional
from fastapi import APIRouter, Depends, Request, Query, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.auth import get_current_user
from app.database import get_db
from app.models.investigation_agent import InvestigationAgent
from app.models.investigation_run import InvestigationRun
from app.services.investigation_seed import seed_investigation_agents
from .template_helpers import get_template_context

template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/investigation-agents", tags=["admin-investigation-agents"])


@router.get("/", response_class=HTMLResponse)
async def list_agents(
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all investigation agents with run counts."""
    # Get run counts per agent
    run_counts_query = select(
        InvestigationRun.agent_id,
        func.count(InvestigationRun.id).label("run_count")
    ).group_by(InvestigationRun.agent_id).subquery()

    # Get all agents with run count via left join
    agents_query = select(
        InvestigationAgent,
        func.coalesce(run_counts_query.c.run_count, 0).label("run_count")
    ).outerjoin(
        run_counts_query,
        InvestigationAgent.id == run_counts_query.c.agent_id
    ).order_by(InvestigationAgent.name)

    result = await db.execute(agents_query)
    rows = result.all()

    agents = []
    for agent, run_count in rows:
        agents.append({
            "id": str(agent.id),
            "name": agent.name,
            "description": agent.description,
            "system_prompt": agent.system_prompt,
            "user_prompt_template": agent.user_prompt_template,
            "config": agent.config or {},
            "is_active": agent.is_active,
            "created_at": agent.created_at,
            "run_count": run_count,
        })

    context = get_template_context(
        request,
        user=user,
        agents=agents,
    )

    return templates.TemplateResponse("admin/investigation_agents/list.html", context)


@router.get("/create", response_class=HTMLResponse)
async def create_agent_form(
    request: Request,
    user=Depends(get_current_user),
):
    """Show the create agent form."""
    context = get_template_context(request, user=user)
    return templates.TemplateResponse("admin/investigation_agents/create.html", context)


@router.post("/create")
async def create_agent(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    system_prompt: str = Form(...),
    user_prompt_template: str = Form(...),
    model_type: str = Form("balanced"),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2000),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Handle create agent form submission."""
    agent = InvestigationAgent(
        name=name,
        description=description or None,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        config={"model_type": model_type, "temperature": temperature, "max_tokens": max_tokens},
    )
    db.add(agent)
    await db.commit()
    return RedirectResponse(url="/admin/investigation-agents/", status_code=303)


@router.get("/{agent_id}/edit", response_class=HTMLResponse)
async def edit_agent_form(
    request: Request,
    agent_id: UUID,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Show the edit agent form pre-populated with agent values."""
    result = await db.execute(
        select(InvestigationAgent).where(InvestigationAgent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Investigation agent not found")

    agent_data = {
        "id": str(agent.id),
        "name": agent.name,
        "description": agent.description or "",
        "system_prompt": agent.system_prompt,
        "user_prompt_template": agent.user_prompt_template,
        "config": agent.config or {},
        "is_active": agent.is_active,
        "created_at": agent.created_at,
    }

    context = get_template_context(request, user=user, agent=agent_data)
    return templates.TemplateResponse("admin/investigation_agents/edit.html", context)


@router.post("/{agent_id}/edit")
async def update_agent(
    request: Request,
    agent_id: UUID,
    name: str = Form(...),
    description: str = Form(""),
    system_prompt: str = Form(...),
    user_prompt_template: str = Form(...),
    model_type: str = Form("balanced"),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2000),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Handle edit agent form submission."""
    result = await db.execute(
        select(InvestigationAgent).where(InvestigationAgent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Investigation agent not found")

    agent.name = name
    agent.description = description or None
    agent.system_prompt = system_prompt
    agent.user_prompt_template = user_prompt_template
    agent.config = {"model_type": model_type, "temperature": temperature, "max_tokens": max_tokens}

    await db.commit()
    return RedirectResponse(url="/admin/investigation-agents/", status_code=303)


@router.post("/{agent_id}/toggle", response_class=HTMLResponse)
async def toggle_agent(
    request: Request,
    agent_id: UUID,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Toggle agent is_active status. Returns updated badge HTML for HTMX swap."""
    result = await db.execute(
        select(InvestigationAgent).where(InvestigationAgent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Investigation agent not found")

    agent.is_active = not agent.is_active
    await db.commit()

    # Return just the updated status badge HTML for HTMX outerHTML swap
    if agent.is_active:
        badge_html = (
            f'<span id="status-{agent_id}" class="inline-flex items-center px-2 py-0.5 rounded-full '
            f'text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">Active</span>'
        )
    else:
        badge_html = (
            f'<span id="status-{agent_id}" class="inline-flex items-center px-2 py-0.5 rounded-full '
            f'text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-400">Inactive</span>'
        )

    return HTMLResponse(content=badge_html)


@router.post("/{agent_id}/delete")
async def delete_agent(
    request: Request,
    agent_id: UUID,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an investigation agent."""
    result = await db.execute(
        select(InvestigationAgent).where(InvestigationAgent.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Investigation agent not found")

    await db.delete(agent)
    await db.commit()
    return RedirectResponse(url="/admin/investigation-agents/", status_code=303)


@router.post("/seed")
async def seed_agents(
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Seed default investigation agents."""
    await seed_investigation_agents(db)
    return RedirectResponse(url="/admin/investigation-agents/", status_code=303)
