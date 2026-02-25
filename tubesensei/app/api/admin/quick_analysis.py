"""Admin Quick Analysis API router module.

Provides endpoints for on-demand video and channel analysis without persisting
results to the database. Useful for quickly evaluating YouTube content and
extracting business ideas in real time.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

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

# In-memory store for channel analysis jobs (job_id -> state dict).
# This is intentionally simple; for production scale you'd use Redis or a DB.
_channel_jobs: Dict[str, Dict[str, Any]] = {}


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class VideoAnalysisRequest(BaseModel):
    """Request body for video analysis."""
    url: str


class ChannelAnalysisRequest(BaseModel):
    """Request body for channel analysis."""
    url: str
    max_videos: int = 50


# ===================================================================
# VIDEO ANALYSIS
# ===================================================================

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

    # 1. Parse the YouTube URL
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

    # 2. Fetch video metadata from the YouTube Data API
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
        return _error_partial(request, user, f"YouTube API error: {exc.message}")
    except Exception as exc:
        logger.exception("Unexpected error fetching video details")
        return _error_partial(request, user, f"Could not fetch video details: {exc}")

    if not video_info:
        return _error_partial(
            request, user,
            f"No video found for ID '{video_id}'. It may be private or deleted."
        )

    # 3. Fetch the transcript
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
        return _video_results_partial(
            request, user,
            video_info=video_info, video_id=video_id, ideas=[],
            transcript_warning=transcript_error or "No transcript available for this video.",
        )

    # 4. Extract ideas via the LLM
    ideas: list = []
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

    # 5. Return the results partial
    return _video_results_partial(
        request, user,
        video_info=video_info, video_id=video_id,
        ideas=ideas, llm_warning=llm_error,
    )


# ===================================================================
# CHANNEL ANALYSIS
# ===================================================================

@router.post("/channel", response_class=HTMLResponse)
async def analyze_channel(
    request: Request,
    body: ChannelAnalysisRequest,
    user=Depends(get_current_user),
) -> Response:
    """
    Start a channel analysis: resolve the channel URL, fetch channel info
    and video list, then kick off background idea extraction.
    Returns an HTML partial with channel info + progress indicator.
    """

    # 1. Parse the YouTube URL
    try:
        parsed = YouTubeParser.parse_url(body.url)
    except InvalidURLError as exc:
        return _error_partial(request, user, str(exc))

    if parsed.get("type") != "channel":
        return _error_partial(
            request, user,
            "The URL does not point to a YouTube channel. "
            "Please provide a valid channel URL (e.g. youtube.com/@handle)."
        )

    # 2. Resolve channel ID
    try:
        yt_client = YouTubeAPIClient()
    except APIKeyError:
        return _error_partial(
            request, user,
            "YouTube API key is not configured. Please add it in Settings."
        )

    channel_info: Optional[Dict[str, Any]] = None
    try:
        channel_id = parsed.get("channel_id")
        handle = parsed.get("channel_handle")

        if channel_id:
            channel_info = await yt_client.get_channel_info(channel_id)
        elif handle:
            channel_info = await yt_client.get_channel_by_handle(handle)
        else:
            return _error_partial(request, user, "Could not extract channel identifier from URL.")
    except YouTubeAPIError as exc:
        return _error_partial(request, user, f"YouTube API error: {exc.message}")
    except Exception as exc:
        logger.exception("Error resolving channel")
        return _error_partial(request, user, f"Could not resolve channel: {exc}")

    if not channel_info:
        return _error_partial(request, user, "Channel not found. It may be private or deleted.")

    # 3. Fetch channel videos
    max_videos = min(body.max_videos, 100)
    try:
        videos = await yt_client.list_channel_videos(
            channel_info["channel_id"], max_results=max_videos
        )
    except Exception as exc:
        logger.exception("Error listing channel videos")
        return _error_partial(request, user, f"Could not list channel videos: {exc}")

    if not videos:
        return _channel_results_partial(
            request, user,
            channel_info=channel_info,
            job_id=None,
            videos_found=0,
            warning="No videos found for this channel.",
        )

    # 4. Create a background job and start processing
    job_id = str(uuid.uuid4())
    _channel_jobs[job_id] = {
        "status": "running",
        "channel_info": channel_info,
        "total_videos": len(videos),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "total_ideas": 0,
        "video_results": [],
        "started_at": datetime.now().isoformat(),
        "error": None,
    }

    # Fire-and-forget background task
    asyncio.create_task(_run_channel_analysis(job_id, videos))

    # 5. Return initial response with progress polling
    return _channel_results_partial(
        request, user,
        channel_info=channel_info,
        job_id=job_id,
        videos_found=len(videos),
    )


@router.get("/channel/progress/{job_id}", response_class=HTMLResponse)
async def channel_progress(
    request: Request,
    job_id: str,
    user=Depends(get_current_user),
) -> Response:
    """Return updated progress HTML for a running channel analysis job."""
    job = _channel_jobs.get(job_id)
    if not job:
        return _error_partial(request, user, "Analysis job not found or has expired.")

    context = get_template_context(
        request,
        user=user,
        job=job,
        job_id=job_id,
    )
    return templates.TemplateResponse(
        "admin/quick_analysis/channel_progress.html", context
    )


# ===================================================================
# Background channel analysis task
# ===================================================================

async def _run_channel_analysis(job_id: str, videos: List[Dict[str, Any]]) -> None:
    """Process channel videos in the background, extracting ideas from each."""
    job = _channel_jobs.get(job_id)
    if not job:
        return

    transcript_client = TranscriptAPIClient()

    try:
        llm = LLMManager()
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"LLM initialization failed: {exc}"
        return

    for video in videos:
        if job["status"] == "cancelled":
            break

        vid = video.get("video_id", "")
        title = video.get("title", "Unknown")
        video_result: Dict[str, Any] = {
            "video_id": vid,
            "title": title,
            "status": "processing",
            "ideas": [],
            "error": None,
        }

        try:
            # Fetch transcript
            transcript_data = await transcript_client.get_transcript(vid)
            if not transcript_data or not transcript_data.content:
                video_result["status"] = "skipped"
                video_result["error"] = "No transcript available"
                job["failed"] += 1
                job["processed"] += 1
                job["video_results"].append(video_result)
                continue

            # Extract ideas
            duration_minutes = 0
            system_prompt, user_prompt = PromptTemplates.get_prompt(
                PromptType.IDEA_EXTRACTION,
                variables={
                    "transcript": transcript_data.content,
                    "title": title,
                    "channel_name": video.get("channel_title", ""),
                    "duration_minutes": str(duration_minutes),
                },
            )

            result = await llm.generate(
                prompt=user_prompt,
                model_type=ModelType.BALANCED,
                system_prompt=system_prompt or "",
                temperature=0.3,
                max_tokens=4000,
            )

            content = result.get("content", "")
            ideas = ResponseParser.parse_idea_extraction_response(content)

            idea_dicts = [
                {
                    "title": idea.title,
                    "description": idea.description,
                    "category": idea.category,
                    "confidence": idea.confidence,
                    "confidence_pct": round(idea.confidence * 100),
                }
                for idea in ideas
            ]

            video_result["status"] = "done"
            video_result["ideas"] = idea_dicts
            job["total_ideas"] += len(idea_dicts)
            job["succeeded"] += 1

        except Exception as exc:
            video_result["status"] = "failed"
            video_result["error"] = str(exc)[:200]
            job["failed"] += 1
            logger.warning("Channel analysis: video %s failed: %s", vid, exc)

        job["processed"] += 1
        job["video_results"].append(video_result)

    job["status"] = "completed" if job["status"] != "cancelled" else "cancelled"


# ===================================================================
# Template helpers
# ===================================================================

def _error_partial(request: Request, user: object, message: str) -> Response:
    """Return an error HTML partial for HTMX."""
    context = get_template_context(request, user=user, error=message)
    return templates.TemplateResponse(
        "admin/quick_analysis/video_results.html", context
    )


def _video_results_partial(
    request: Request,
    user: object,
    *,
    video_info: dict,
    video_id: str,
    ideas: list,
    transcript_warning: Optional[str] = None,
    llm_warning: Optional[str] = None,
) -> Response:
    """Return the video results HTML partial for HTMX."""
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


def _channel_results_partial(
    request: Request,
    user: object,
    *,
    channel_info: dict,
    job_id: Optional[str],
    videos_found: int = 0,
    warning: Optional[str] = None,
) -> Response:
    """Return the channel results HTML partial for HTMX."""
    # Format subscriber count
    subs = channel_info.get("subscriber_count", 0) or 0
    if subs >= 1_000_000:
        subs_display = f"{subs / 1_000_000:.1f}M"
    elif subs >= 1_000:
        subs_display = f"{subs / 1_000:.1f}K"
    else:
        subs_display = str(subs)

    # Pick thumbnail
    thumbnails = channel_info.get("thumbnails", {})
    thumbnail_url = ""
    if isinstance(thumbnails, dict):
        for key in ("medium", "high", "default"):
            thumb = thumbnails.get(key)
            if thumb and isinstance(thumb, dict) and thumb.get("url"):
                thumbnail_url = thumb["url"]
                break

    context = get_template_context(
        request,
        user=user,
        channel={
            "title": channel_info.get("title", "Unknown Channel"),
            "channel_id": channel_info.get("channel_id", ""),
            "description": (channel_info.get("description", "") or "")[:200],
            "subscriber_count": subs,
            "subs_display": subs_display,
            "video_count": channel_info.get("video_count", 0),
            "thumbnail_url": thumbnail_url,
        },
        job_id=job_id,
        videos_found=videos_found,
        warning=warning,
    )
    return templates.TemplateResponse(
        "admin/quick_analysis/channel_results.html", context
    )
