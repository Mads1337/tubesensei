"""
Export service for TubeSensei - exports ideas as JSON or CSV.
Supports filtering by campaign, status, priority, and date range.
"""
import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.campaign_video import CampaignVideo
from app.models.channel import Channel
from app.models.idea import Idea, IdeaPriority, IdeaStatus
from app.models.video import Video

logger = logging.getLogger(__name__)

# Fields included in every export record
EXPORT_FIELDS = [
    "id",
    "title",
    "description",
    "category",
    "status",
    "priority",
    "confidence_score",
    "complexity_score",
    "market_size_estimate",
    "target_audience",
    "implementation_time_estimate",
    "tags",
    "technologies",
    "competitive_advantage",
    "review_notes",
    "created_at",
    "video_id",
    "video_title",
    "video_url",
    "channel_id",
    "channel_name",
    "channel_handle",
]


def _idea_to_dict(idea: Idea, video: Video, channel: Channel) -> dict:
    """Convert an Idea (with related Video and Channel) to an export-ready dict."""
    return {
        "id": str(idea.id),
        "title": idea.title,
        "description": idea.description,
        "category": idea.category,
        "status": idea.status.value if idea.status else None,
        "priority": idea.priority.value if idea.priority else None,
        "confidence_score": idea.confidence_score,
        "complexity_score": idea.complexity_score,
        "market_size_estimate": idea.market_size_estimate,
        "target_audience": idea.target_audience,
        "implementation_time_estimate": idea.implementation_time_estimate,
        "tags": idea.tags or [],
        "technologies": idea.technologies or [],
        "competitive_advantage": idea.competitive_advantage,
        "review_notes": idea.review_notes,
        "created_at": idea.created_at.isoformat() if idea.created_at else None,
        "video_id": str(video.id) if video else None,
        "video_title": video.title if video else None,
        "video_url": video.youtube_url if video else None,
        "channel_id": str(channel.id) if channel else None,
        "channel_name": channel.name if channel else None,
        "channel_handle": channel.channel_handle if channel else None,
    }


class ExportService:
    """
    Service for exporting ideas with optional filters.

    Supports JSON and CSV output formats.  Increments export_count and sets
    last_exported_at on every idea that is included in the export.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _fetch_ideas(
        self,
        campaign_id: Optional[UUID] = None,
        status: Optional[list[IdeaStatus]] = None,
        priority: Optional[list[IdeaPriority]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[tuple[Idea, Video, Channel]]:
        """
        Query ideas from the database with the requested filters and return
        them as (Idea, Video, Channel) tuples.
        """
        # Build the base query joining Idea -> Video -> Channel
        stmt = (
            select(Idea, Video, Channel)
            .join(Video, Idea.video_id == Video.id)
            .join(Channel, Video.channel_id == Channel.id)
        )

        # Optionally filter by campaign via CampaignVideo junction table
        if campaign_id is not None:
            stmt = stmt.join(
                CampaignVideo,
                and_(
                    CampaignVideo.video_id == Video.id,
                    CampaignVideo.campaign_id == campaign_id,
                ),
            )

        # Build WHERE clauses
        conditions = []

        if status:
            conditions.append(Idea.status.in_(status))

        if priority:
            conditions.append(Idea.priority.in_(priority))

        if min_confidence is not None:
            conditions.append(Idea.confidence_score >= min_confidence)

        if date_from is not None:
            conditions.append(Idea.created_at >= date_from)

        if date_to is not None:
            conditions.append(Idea.created_at <= date_to)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(Idea.created_at.desc()).limit(limit)

        result = await self.db.execute(stmt)
        return result.all()

    async def _mark_exported(self, ideas: list[Idea]) -> None:
        """Increment export_count and set last_exported_at for each idea."""
        now = datetime.now(timezone.utc)
        for idea in ideas:
            idea.export_count = (idea.export_count or 0) + 1
            idea.last_exported_at = now
        await self.db.commit()

    async def export_as_json(
        self,
        campaign_id: Optional[UUID] = None,
        status: Optional[list[IdeaStatus]] = None,
        priority: Optional[list[IdeaPriority]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 1000,
    ) -> str:
        """
        Export ideas as a JSON string (list of objects).

        Returns the serialized JSON string.
        """
        rows = await self._fetch_ideas(
            campaign_id=campaign_id,
            status=status,
            priority=priority,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

        records = [_idea_to_dict(idea, video, channel) for idea, video, channel in rows]

        await self._mark_exported([idea for idea, _video, _channel in rows])

        logger.info("Exported %d ideas as JSON", len(records))
        return json.dumps(records, ensure_ascii=False, indent=2)

    async def export_as_csv(
        self,
        campaign_id: Optional[UUID] = None,
        status: Optional[list[IdeaStatus]] = None,
        priority: Optional[list[IdeaPriority]] = None,
        min_confidence: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 1000,
    ) -> str:
        """
        Export ideas as a CSV string.

        Lists are serialised to pipe-separated strings so they fit in a
        single CSV cell.  Returns the UTF-8 encoded CSV string.
        """
        rows = await self._fetch_ideas(
            campaign_id=campaign_id,
            status=status,
            priority=priority,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for idea, video, channel in rows:
            record = _idea_to_dict(idea, video, channel)
            # Flatten list fields to pipe-separated strings for CSV compatibility
            record["tags"] = "|".join(record["tags"]) if record["tags"] else ""
            record["technologies"] = "|".join(record["technologies"]) if record["technologies"] else ""
            writer.writerow(record)

        await self._mark_exported([idea for idea, _video, _channel in rows])

        logger.info("Exported %d ideas as CSV", len(rows))
        return output.getvalue()
