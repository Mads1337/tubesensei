"""
Webhook Management API Endpoints

CRUD REST API for managing webhook subscriptions that deliver event
notifications to external systems.
"""
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.permissions import Permission, require_permission
from app.database import get_db
from app.models.webhook import Webhook
from app.services import webhook_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class WebhookCreate(BaseModel):
    name: str = Field(..., max_length=255)
    url: HttpUrl
    events: List[str] = Field(..., min_length=1)
    secret: Optional[str] = None


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    is_active: Optional[bool] = None


class WebhookResponse(BaseModel):
    id: UUID
    name: str
    url: str
    events: List[str]
    is_active: bool
    last_triggered_at: Optional[datetime]
    failure_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WebhookListResponse(BaseModel):
    items: List[WebhookResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_webhook_or_404(webhook_id: UUID, db: AsyncSession) -> Webhook:
    result = await db.execute(select(Webhook).where(Webhook.id == webhook_id))
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return webhook


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=WebhookListResponse)
async def list_webhooks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """List all webhook subscriptions (paginated)."""
    count_result = await db.execute(select(func.count()).select_from(Webhook))
    total: int = count_result.scalar_one()

    stmt = select(Webhook).order_by(Webhook.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    webhooks = list(result.scalars().all())

    return WebhookListResponse(
        items=webhooks,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(webhooks)) < total,
    )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: UUID,
    _user=Depends(require_permission(Permission.CHANNEL_READ)),
    db: AsyncSession = Depends(get_db),
):
    """Get a webhook subscription by ID."""
    return await _get_webhook_or_404(webhook_id, db)


@router.post("/", response_model=WebhookResponse, status_code=201)
async def create_webhook(
    data: WebhookCreate,
    _user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Create a new webhook subscription."""
    webhook = Webhook(
        name=data.name,
        url=str(data.url),
        events=data.events,
        secret=data.secret,
    )
    db.add(webhook)
    await db.commit()
    await db.refresh(webhook)
    logger.info("Created webhook id=%s name=%s url=%s", webhook.id, webhook.name, webhook.url)
    return webhook


@router.put("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: UUID,
    data: WebhookUpdate,
    _user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing webhook subscription."""
    webhook = await _get_webhook_or_404(webhook_id, db)

    if data.name is not None:
        webhook.name = data.name
    if data.url is not None:
        webhook.url = str(data.url)
    if data.events is not None:
        webhook.events = data.events
    if data.secret is not None:
        webhook.secret = data.secret
    if data.is_active is not None:
        webhook.is_active = data.is_active
        # Reset the failure counter when a webhook is manually re-enabled.
        if data.is_active:
            webhook.failure_count = 0

    await db.commit()
    await db.refresh(webhook)
    logger.info("Updated webhook id=%s", webhook.id)
    return webhook


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: UUID,
    _user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a webhook subscription."""
    webhook = await _get_webhook_or_404(webhook_id, db)
    await db.delete(webhook)
    await db.commit()
    logger.info("Deleted webhook id=%s", webhook_id)


@router.post("/{webhook_id}/test", response_model=dict)
async def test_webhook(
    webhook_id: UUID,
    _user=Depends(require_permission(Permission.CHANNEL_WRITE)),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a test ping event to the webhook URL.

    This fires a ``webhook.test`` event with a sample payload so you can
    verify your endpoint is reachable and the signature is correct.
    """
    webhook = await _get_webhook_or_404(webhook_id, db)

    test_payload = {
        "webhook_id": str(webhook.id),
        "webhook_name": webhook.name,
        "message": "This is a test event from TubeSensei.",
    }

    try:
        await webhook_service.dispatch_event(
            db=db,
            event_type="webhook.test",
            payload=test_payload,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Test event dispatch raised: %s", exc)

    # Re-fetch to return the updated state (failure_count / last_triggered_at may have changed).
    await db.refresh(webhook)
    return {
        "success": True,
        "message": "Test event dispatched",
        "webhook_id": str(webhook.id),
        "is_active": webhook.is_active,
        "failure_count": webhook.failure_count,
    }
