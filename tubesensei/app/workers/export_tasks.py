"""
Celery tasks for scheduled idea exports.

Runs daily/weekly exports of ideas to files, with optional webhook delivery.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.celery_app import celery_app
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Directory where scheduled exports are saved
EXPORTS_DIR = Path(os.environ.get("EXPORTS_DIR", "/tmp/tubesensei_exports"))


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _do_scheduled_export(period: str) -> dict:
    """
    Core async export logic.

    Args:
        period: 'daily' or 'weekly'

    Returns:
        dict with export metadata (path, row_count, period)
    """
    from app.services.export_service import ExportService
    from app.models.idea import IdeaStatus

    now = datetime.now(tz=timezone.utc)
    if period == "weekly":
        date_from = now - timedelta(days=7)
        filename = f"ideas_weekly_{now.strftime('%Y%m%d')}.csv"
    else:
        date_from = now - timedelta(days=1)
        filename = f"ideas_daily_{now.strftime('%Y%m%d')}.csv"

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / filename

    async with AsyncSessionLocal() as db:
        service = ExportService(db)
        # Export all non-rejected ideas from the period
        csv_content = await service.export_as_csv(
            status=[
                IdeaStatus.EXTRACTED,
                IdeaStatus.REVIEWED,
                IdeaStatus.SELECTED,
                IdeaStatus.IN_PROGRESS,
                IdeaStatus.IMPLEMENTED,
            ],
            date_from=date_from,
        )

    export_path.write_text(csv_content, encoding="utf-8")
    row_count = max(0, csv_content.count("\n") - 1)  # subtract header row
    logger.info(
        "Scheduled %s export complete: %s (%d rows)", period, export_path, row_count
    )
    return {"path": str(export_path), "row_count": row_count, "period": period}


async def _dispatch_export_webhook(export_meta: dict) -> None:
    """Send a webhook notification after a scheduled export completes."""
    try:
        from app.services.webhook_service import dispatch_event

        async with AsyncSessionLocal() as db:
            await dispatch_event(
                db,
                "export.completed",
                {
                    "period": export_meta["period"],
                    "path": export_meta["path"],
                    "row_count": export_meta["row_count"],
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
    except Exception as exc:
        logger.warning("Failed to dispatch export webhook: %s", exc)


@celery_app.task(
    name="app.workers.export_tasks.scheduled_daily_export_task",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def scheduled_daily_export_task(self):
    """Export ideas created/updated in the last 24 hours to a CSV file."""
    try:
        meta = _run_async(_do_scheduled_export("daily"))
        _run_async(_dispatch_export_webhook(meta))
        return meta
    except Exception as exc:
        logger.error("Daily export failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


@celery_app.task(
    name="app.workers.export_tasks.scheduled_weekly_export_task",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def scheduled_weekly_export_task(self):
    """Export ideas created/updated in the last 7 days to a CSV file."""
    try:
        meta = _run_async(_do_scheduled_export("weekly"))
        _run_async(_dispatch_export_webhook(meta))
        return meta
    except Exception as exc:
        logger.error("Weekly export failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)
