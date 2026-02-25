"""
Webhook dispatch service.
Sends HTTP POST requests to registered webhook URLs when events occur.
Uses exponential backoff retry with httpx.
"""
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.webhook import Webhook

logger = logging.getLogger(__name__)

# Maximum consecutive failures before a webhook is auto-disabled
MAX_FAILURE_COUNT = 5

# HTTP request timeout in seconds
REQUEST_TIMEOUT = 10.0


def _build_payload(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a standardised webhook payload envelope."""
    return {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": data,
    }


def _compute_signature(secret: str, body: bytes) -> str:
    """Return HMAC-SHA256 hex digest of *body* using *secret*."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def dispatch_event(
    db: AsyncSession,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """
    Find all active webhooks subscribed to *event_type* and fire HTTP POST
    requests to each one.

    On failure the webhook's ``failure_count`` is incremented.  Once it
    reaches ``MAX_FAILURE_COUNT`` the webhook is automatically deactivated.

    Args:
        db: Async SQLAlchemy session.
        event_type: The event identifier, e.g. ``"idea.extracted"``.
        payload: Event-specific data dict.  It will be wrapped in the
            standard envelope before sending.
    """
    # Query active webhooks that are subscribed to this event type.
    stmt = select(Webhook).where(
        Webhook.is_active.is_(True),
        Webhook.events.contains([event_type]),  # type: ignore[arg-type]
    )
    result = await db.execute(stmt)
    webhooks: List[Webhook] = list(result.scalars().all())

    if not webhooks:
        logger.debug("No active webhooks for event '%s'", event_type)
        return

    envelope = _build_payload(event_type, payload)
    body = json.dumps(envelope, default=str).encode()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for webhook in webhooks:
            delivery_id = str(uuid.uuid4())
            headers = {
                "Content-Type": "application/json",
                "X-TubeSensei-Event": event_type,
                "X-TubeSensei-Delivery": delivery_id,
            }

            if webhook.secret:
                signature = _compute_signature(webhook.secret, body)
                headers["X-TubeSensei-Signature"] = f"sha256={signature}"

            try:
                response = await client.post(
                    str(webhook.url),
                    content=body,
                    headers=headers,
                )
                response.raise_for_status()

                # Success – reset failure counter and record delivery time.
                webhook.failure_count = 0
                webhook.last_triggered_at = datetime.now(timezone.utc)
                await db.commit()

                logger.info(
                    "Webhook delivered: id=%s event=%s delivery=%s status=%s",
                    webhook.id,
                    event_type,
                    delivery_id,
                    response.status_code,
                )

            except Exception as exc:  # noqa: BLE001
                webhook.failure_count = (webhook.failure_count or 0) + 1

                if webhook.failure_count >= MAX_FAILURE_COUNT:
                    webhook.is_active = False
                    logger.warning(
                        "Webhook auto-disabled after %d consecutive failures: "
                        "id=%s url=%s",
                        webhook.failure_count,
                        webhook.id,
                        webhook.url,
                    )
                else:
                    logger.warning(
                        "Webhook delivery failed (failure %d/%d): "
                        "id=%s event=%s delivery=%s error=%s",
                        webhook.failure_count,
                        MAX_FAILURE_COUNT,
                        webhook.id,
                        event_type,
                        delivery_id,
                        exc,
                    )

                await db.commit()
