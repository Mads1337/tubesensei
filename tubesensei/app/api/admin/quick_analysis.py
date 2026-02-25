"""Admin Quick Analysis API router module.

Provides endpoints for on-demand video analysis without persisting results
to the database. Useful for quickly evaluating a YouTube video's content
and extracting business ideas in real time.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from starlette.responses import Response

from app.core.auth import get_current_user
from app.utils.youtube_parser import YouTubeParser
from app.utils.exceptions import InvalidURLError, APIKeyError, YouTubeAPIError
from app.integrations.youtube_api import YouTubeAPIClient
from app.integrations.transcript_api import TranscriptAPIClient
from app.integrations.transcript_errors import TranscriptError
from app.ai.llm_manager import LLMManager, ModelType
from app.ai.prompt_templates import PromptTemplates, PromptType
from app.ai.response_parser import ResponseParser
from .template_helpers import get_template_context

logger = logging.getLogger(__name__)

template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/quick-analysis", tags=["admin-quick-analysis"])


class VideoAnalysisRequest(BaseModel):
    """Request body for video analysis."""
    url: str


class ChannelAnalysisRequest(BaseModel):
    """Request body for channel analysis."""
    url: str
    max_videos: int = 50


@router.post("/video", response_class=HTMLResponse)
async def analyze_video(
    request: Request,
    body: VideoAnalysisRequest,
    user=Depends(get_current_user),
) -> Response:
    """
    Analyze a YouTube video URL: fetch metadata, transcript, and extract
    business ideas via LLM.  Returns an HTML partial for HTMX swap.
    """

    # ------------------------------------------------------------------
    # 1. Parse the YouTube URL
    # ------------------------------------------------------------------
    try:
        parsed = YouTubeParser.parse_url(body.url)
    except InvalidURLError as exc:
        return _error_partial(request, user, str(exc))

    if parsed.get("type") != "video" or not parsed.get("video_id"):
        return _error_partial(
            request, user,
            "The URL does not point to a YouTube video. Please provide a valid video URL."
        )

    video_id = parsed["video_id"] or ""

    # ------------------------------------------------------------------
    # 2. Fetch video metadata from the YouTube Data API
    # ------------------------------------------------------------------
    video_info = None
    try:
        yt_client = YouTubeAPIClient()
        details = await yt_client.get_video_details([video_id])
        if details:
            video_info = details[0]
    except APIKeyError:
        return _error_partial(
            request, user,
            "YouTube API key is not configured. Please add it in Settings."
        )
    except YouTubeAPIError as exc:
        return _error_partial(
            request, user,
            f"YouTube API error: {exc.message}"
        )
    except Exception as exc:
        logger.exception("Unexpected error fetching video details")
        return _error_partial(
            request, user,
            f"Could not fetch video details: {exc}"
        )

    if not video_info:
        return _error_partial(
            request, user,
            f"No video found for ID '{video_id}'. It may be private or deleted."
        )

    # ------------------------------------------------------------------
    # 3. Fetch the transcript
    # ------------------------------------------------------------------
    transcript_data = None
    transcript_error: Optional[str] = None
    try:
        transcript_client = TranscriptAPIClient()
        transcript_data = await transcript_client.get_transcript(video_id)
    except TranscriptError as exc:
        transcript_error = str(exc)
        logger.warning("Transcript unavailable for %s: %s", video_id, exc)
    except Exception as exc:
        transcript_error = str(exc)
        logger.exception("Unexpected error fetching transcript for %s", video_id)

    if not transcript_data or not transcript_data.content:
        # Return partial results with metadata only
        return _results_partial(
            request,
            user,
            video_info=video_info,
            video_id=video_id,
            ideas=[],
            transcript_warning=transcript_error or "No transcript available for this video.",
        )

    # ------------------------------------------------------------------
    # 4. Extract ideas via the LLM
    # ------------------------------------------------------------------
    ideas = []
    llm_error: Optional[str] = None
    try:
        duration_seconds = video_info.get("duration_seconds", 0) or 0
        duration_minutes = round(duration_seconds / 60, 1) if duration_seconds else 0

        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.IDEA_EXTRACTION,
            variables={
                "transcript": transcript_data.content,
                "title": video_info.get("title", ""),
                "channel_name": video_info.get("channel_title", ""),
                "duration_minutes": str(duration_minutes),
            },
        )

        llm = LLMManager()
        result = await llm.generate(
            prompt=user_prompt,
            model_type=ModelType.BALANCED,
            system_prompt=system_prompt or "",
            temperature=0.3,
            max_tokens=4000,
        )

        content = result.get("content", "")
        ideas = ResponseParser.parse_idea_extraction_response(content)
    except Exception as exc:
        llm_error = str(exc)
        logger.exception("LLM idea extraction failed for video %s", video_id)

    # ------------------------------------------------------------------
    # 5. Return the results partial
    # ------------------------------------------------------------------
    return _results_partial(
        request,
        user,
        video_info=video_info,
        video_id=video_id,
        ideas=ideas,
        llm_warning=llm_error,
    )


@router.post("/channel", response_class=HTMLResponse)
async def analyze_channel(
    request: Request,
    user=Depends(get_current_user),
) -> Response:
    """Placeholder for channel-level quick analysis (coming soon)."""
    return HTMLResponse(
        content=(
            '<div class="p-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 '
            'dark:border-yellow-700 rounded-lg text-center">'
            '<i class="fas fa-hard-hat text-yellow-500 text-2xl mb-2"></i>'
            '<p class="text-yellow-800 dark:text-yellow-300 font-medium">'
            "Channel analysis is coming soon.</p>"
            "</div>"
        )
    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _error_partial(request: Request, user: object, message: str) -> Response:
    """Return an error HTML partial for HTMX."""
    context = get_template_context(
        request,
        user=user,
        error=message,
    )
    return templates.TemplateResponse(
        "admin/quick_analysis/video_results.html", context
    )


def _results_partial(
    request: Request,
    user: object,
    *,
    video_info: dict,
    video_id: str,
    ideas: list,
    transcript_warning: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Response:
    """Return the results HTML partial for HTMX."""
    # Build a list of plain dicts so the template can access attributes uniformly.
    idea_dicts = []
    for idea in ideas:
        idea_dicts.append({
            "title": idea.title,
            "description": idea.description,
            "category": idea.category,
            "target_market": idea.target_market,
            "value_proposition": idea.value_proposition,
            "complexity_score": idea.complexity_score,
            "confidence": idea.confidence,
            "confidence_pct": round(idea.confidence * 100),
            "source_context": idea.source_context,
        })

    # Format duration for display
    duration_seconds = video_info.get("duration_seconds", 0) or 0
    hours, remainder = divmod(int(duration_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        duration_display = f"{hours}h {minutes}m {seconds}s"
    elif minutes:
        duration_display = f"{minutes}m {seconds}s"
    else:
        duration_display = f"{seconds}s"

    # Format view count for display
    view_count = video_info.get("view_count", 0) or 0
    try:
        view_count = int(view_count)
        if view_count >= 1_000_000:
            views_display = f"{view_count / 1_000_000:.1f}M"
        elif view_count >= 1_000:
            views_display = f"{view_count / 1_000:.1f}K"
        else:
            views_display = str(view_count)
    except (ValueError, TypeError):
        views_display = str(view_count)

    # Pick the best available thumbnail
    thumbnails = video_info.get("thumbnails", {})
    thumbnail_url = ""
    if isinstance(thumbnails, dict):
        for key in ("medium", "high", "default", "standard", "maxres"):
            thumb = thumbnails.get(key)
            if thumb and isinstance(thumb, dict) and thumb.get("url"):
                thumbnail_url = thumb["url"]
                break
            elif thumb and isinstance(thumb, str):
                thumbnail_url = thumb
                break

    context = get_template_context(
        request,
        user=user,
        video={
            "title": video_info.get("title", "Unknown Title"),
            "channel_title": video_info.get("channel_title", "Unknown Channel"),
            "video_id": video_id,
            "view_count": view_count,
            "views_display": views_display,
            "duration_display": duration_display,
            "published_at": video_info.get("published_at", ""),
            "thumbnail_url": thumbnail_url,
            "like_count": video_info.get("like_count", 0),
            "comment_count": video_info.get("comment_count", 0),
        },
        ideas=idea_dicts,
        idea_count=len(idea_dicts),
        transcript_warning=transcript_warning,
        llm_warning=llm_warning,
    )
    return templates.TemplateResponse(
        "admin/quick_analysis/video_results.html", context
    )
