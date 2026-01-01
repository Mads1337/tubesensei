"""Admin Transcripts API router module."""

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing import Optional
from uuid import UUID
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import joinedload

from app.core.auth import get_current_user
from app.models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from app.models.video import Video
from app.models.channel import Channel
from app.database import get_db
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/transcripts", tags=["admin-transcripts"])


@router.get("/", response_class=HTMLResponse)
async def transcripts_page(
    request: Request,
    search: Optional[str] = Query(None, description="Search in transcript content"),
    language: Optional[str] = Query(None, description="Filter by language"),
    source: Optional[str] = Query(None, description="Filter by source"),
    channel_id: Optional[UUID] = Query(None, description="Filter by channel"),
    page: int = Query(1, ge=1),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render transcripts page with search and filtering"""

    limit = settings.admin.ADMIN_PAGINATION_DEFAULT
    offset = (page - 1) * limit

    # Build query
    query = select(Transcript).options(
        joinedload(Transcript.video).joinedload(Video.channel)
    )
    count_query = select(func.count(Transcript.id))

    # Apply filters
    filters = []

    if search:
        filters.append(Transcript.content.ilike(f"%{search}%"))

    if language:
        try:
            lang = TranscriptLanguage(language)
            filters.append(Transcript.language == lang)
        except ValueError:
            pass

    if source:
        try:
            src = TranscriptSource(source)
            filters.append(Transcript.source == src)
        except ValueError:
            pass

    if channel_id:
        # Need to join with video to filter by channel
        query = query.join(Video, Transcript.video_id == Video.id)
        count_query = count_query.join(Video, Transcript.video_id == Video.id)
        filters.append(Video.channel_id == channel_id)

    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))

    # Get total count
    total = await db.scalar(count_query)

    # Apply sorting (newest first)
    query = query.order_by(Transcript.created_at.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    transcripts = result.scalars().unique().all()

    # Convert to dicts
    transcript_list = []
    for t in transcripts:
        transcript_dict = {
            "id": str(t.id),
            "video_id": str(t.video_id) if t.video_id else None,
            "video_title": t.video.title if t.video else "Unknown",
            "video_thumbnail": t.video.thumbnail_url if t.video else None,
            "channel_name": t.video.channel.name if t.video and t.video.channel else "Unknown",
            "source": t.source.value if t.source else None,
            "language": t.language.value if t.language else None,
            "language_code": t.language_code,
            "is_auto_generated": t.is_auto_generated,
            "word_count": t.word_count,
            "char_count": t.char_count,
            "content_preview": t.content_preview,
            "created_at": t.created_at,
        }
        transcript_list.append(transcript_dict)

    total_pages = (total + limit - 1) // limit if total else 1

    # Get stats for header
    stats_query = select(
        func.count(Transcript.id).label("total"),
        func.count(Transcript.id).filter(Transcript.is_auto_generated == True).label("auto_generated"),
        func.count(Transcript.id).filter(Transcript.is_auto_generated == False).label("manual"),
        func.coalesce(func.sum(Transcript.word_count), 0).label("total_words"),
    )
    stats_result = await db.execute(stats_query)
    stats = stats_result.one()._asdict()

    # Get channels for filter dropdown
    channels_query = select(Channel.id, Channel.name).order_by(Channel.name)
    channels_result = await db.execute(channels_query)
    channels = [{"id": str(row.id), "name": row.name} for row in channels_result.all()]

    # Check if this is an HTMX request
    is_htmx = request.headers.get("HX-Request") == "true"

    context = get_template_context(
        request,
        user=user,
        transcripts=transcript_list,
        total=total,
        page=page,
        total_pages=total_pages,
        stats=stats,
        channels=channels,
        filters={
            "search": search,
            "language": language,
            "source": source,
            "channel_id": str(channel_id) if channel_id else None,
        },
        languages=[l.value for l in TranscriptLanguage],
        sources=[s.value for s in TranscriptSource],
    )

    if is_htmx:
        return templates.TemplateResponse("admin/transcripts/partials/transcript_list.html", context)

    return templates.TemplateResponse("admin/transcripts/list.html", context)


@router.get("/{transcript_id}", response_class=HTMLResponse)
async def transcript_detail(
    request: Request,
    transcript_id: UUID,
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Render full transcript view"""

    query = select(Transcript).options(
        joinedload(Transcript.video).joinedload(Video.channel)
    ).where(Transcript.id == transcript_id)
    result = await db.execute(query)
    transcript = result.scalar_one_or_none()

    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript_dict = {
        "id": str(transcript.id),
        "video_id": str(transcript.video_id) if transcript.video_id else None,
        "video_title": transcript.video.title if transcript.video else "Unknown",
        "video_thumbnail": transcript.video.thumbnail_url if transcript.video else None,
        "video_youtube_id": transcript.video.youtube_video_id if transcript.video else None,
        "channel_name": transcript.video.channel.name if transcript.video and transcript.video.channel else "Unknown",
        "channel_id": str(transcript.video.channel.id) if transcript.video and transcript.video.channel else None,
        "source": transcript.source.value if transcript.source else None,
        "language": transcript.language.value if transcript.language else None,
        "language_code": transcript.language_code,
        "is_auto_generated": transcript.is_auto_generated,
        "word_count": transcript.word_count,
        "char_count": transcript.char_count,
        "confidence_score": transcript.confidence_score,
        "content": transcript.content,
        "created_at": transcript.created_at,
        "updated_at": transcript.updated_at,
    }

    context = get_template_context(
        request,
        user=user,
        transcript=transcript_dict,
    )

    return templates.TemplateResponse("admin/transcripts/detail.html", context)


@router.get("/{transcript_id}/download")
async def download_transcript(
    transcript_id: UUID,
    format: str = Query("txt", description="Download format: txt or json"),
    user = Depends(get_current_user),
    db = Depends(get_db)
):
    """Download transcript as text or JSON"""

    query = select(Transcript).options(
        joinedload(Transcript.video)
    ).where(Transcript.id == transcript_id)
    result = await db.execute(query)
    transcript = result.scalar_one_or_none()

    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    video_title = transcript.video.title if transcript.video else "transcript"
    # Sanitize filename
    safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]

    if format == "json":
        import json
        content = json.dumps({
            "video_title": transcript.video.title if transcript.video else None,
            "video_id": transcript.video.youtube_video_id if transcript.video else None,
            "language": transcript.language_code,
            "source": transcript.source.value if transcript.source else None,
            "word_count": transcript.word_count,
            "content": transcript.content,
        }, indent=2)
        return PlainTextResponse(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.json"'}
        )
    else:
        return PlainTextResponse(
            content=transcript.content or "",
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.txt"'}
        )
