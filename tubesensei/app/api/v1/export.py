"""
Export API Endpoints

Provides idea export in JSON or CSV format with optional filtering by
campaign, status, priority, confidence score, and date range.
"""
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import Permission, require_permission
from app.database import get_db
from app.models.idea import IdeaPriority, IdeaStatus
from app.services.export_service import ExportService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

async def get_export_service(db: AsyncSession = Depends(get_db)) -> ExportService:
    return ExportService(db)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ideas")
async def export_ideas(
    format: str = Query("json", pattern="^(json|csv)$", description="Output format: json or csv"),
    campaign_id: Optional[UUID] = Query(None, description="Filter ideas to videos in this campaign"),
    status: Optional[List[IdeaStatus]] = Query(None, description="Filter by one or more statuses"),
    priority: Optional[List[IdeaPriority]] = Query(None, description="Filter by one or more priorities"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score (0-1)"),
    date_from: Optional[datetime] = Query(None, description="Include ideas created on or after this datetime (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Include ideas created on or before this datetime (ISO 8601)"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of ideas to export"),
    _user=Depends(require_permission(Permission.CHANNEL_READ)),
    service: ExportService = Depends(get_export_service),
) -> Response:
    """
    Export ideas as a downloadable file.

    Supports JSON and CSV formats.  Use the query parameters to narrow the
    export to a specific campaign, status, priority, confidence range, or
    creation-date window.

    The export also increments the `export_count` and updates
    `last_exported_at` for every idea included in the file.
    """
    try:
        if format == "csv":
            content = await service.export_as_csv(
                campaign_id=campaign_id,
                status=status,
                priority=priority,
                min_confidence=min_confidence,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=ideas_export.csv"},
            )

        # Default: JSON
        content = await service.export_as_json(
            campaign_id=campaign_id,
            status=status,
            priority=priority,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=ideas_export.json"},
        )

    except Exception as exc:
        logger.exception("Failed to export ideas: %s", exc)
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")
