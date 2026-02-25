"""Admin Settings API router module."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.core.auth import get_current_user
from app.core.config import settings, runtime_overrides
from .template_helpers import get_template_context

template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/settings", tags=["admin-settings"])


def _mask_url(url: str) -> str:
    """Return *url* with the password portion replaced by asterisks."""
    if not url:
        return "(not set)"
    try:
        # Mask everything between "://" and "@" (credentials section).
        if "@" in url and "://" in url:
            scheme_end = url.index("://") + 3
            at_pos = url.rindex("@")
            credentials = url[scheme_end:at_pos]
            if ":" in credentials:
                user, _ = credentials.split(":", 1)
                masked = f"{url[:scheme_end]}{user}:***@{url[at_pos + 1:]}"
            else:
                masked = f"{url[:scheme_end]}***@{url[at_pos + 1:]}"
            return masked
    except Exception:
        pass
    return url


def _build_settings_data() -> dict:
    """Build settings data dict, applying runtime overrides on top of defaults."""
    # --- feature flags ---
    features = {
        "registration_enabled": runtime_overrides.get(
            "features", "registration_enabled",
            settings.FEATURES_ENABLE_REGISTRATION,
        ),
        "api_docs_enabled": runtime_overrides.get(
            "features", "api_docs_enabled",
            settings.FEATURES_ENABLE_API_DOCS,
        ),
        "metrics_enabled": runtime_overrides.get(
            "features", "metrics_enabled",
            settings.FEATURES_ENABLE_METRICS,
        ),
        "health_checks_enabled": runtime_overrides.get(
            "features", "health_checks_enabled",
            settings.FEATURES_ENABLE_HEALTH_CHECKS,
        ),
    }

    # --- topic discovery ---
    td = settings.topic_discovery
    topic_discovery = {
        "default_video_limit": runtime_overrides.get(
            "topic_discovery", "default_video_limit", td.DEFAULT_VIDEO_LIMIT),
        "default_channel_limit": runtime_overrides.get(
            "topic_discovery", "default_channel_limit", td.DEFAULT_CHANNEL_LIMIT),
        "search_limit": runtime_overrides.get(
            "topic_discovery", "search_limit", td.SEARCH_LIMIT),
        "similar_depth": runtime_overrides.get(
            "topic_discovery", "similar_depth", td.SIMILAR_DEPTH),
        "filter_threshold": runtime_overrides.get(
            "topic_discovery", "filter_threshold", td.FILTER_THRESHOLD),
        "filter_batch_size": td.FILTER_BATCH_SIZE,
        "filter_model": td.FILTER_MODEL,
        "rate_limit": td.RATE_LIMIT,
        "max_concurrent_agents": runtime_overrides.get(
            "topic_discovery", "max_concurrent_agents", td.MAX_CONCURRENT_AGENTS),
    }

    # --- logging ---
    logging_settings = {
        "level": runtime_overrides.get("logging", "level", settings.LOG_LEVEL),
        "format": settings.LOG_FORMAT,
        "rotation": settings.LOG_ROTATION,
        "retention": settings.LOG_RETENTION,
    }

    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT,
        },
        "database": {
            "url_masked": _mask_url(settings.DATABASE_URL),
            "pool_size": settings.DATABASE_POOL_SIZE,
            "pool_max_overflow": settings.DATABASE_POOL_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "echo": settings.DATABASE_ECHO,
        },
        "redis": {
            "url_masked": _mask_url(settings.REDIS_URL),
            "max_connections": settings.REDIS_MAX_CONNECTIONS,
            "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
            "connection_timeout": settings.REDIS_CONNECTION_TIMEOUT,
        },
        "security": {
            "session_expire_hours": settings.security.SESSION_EXPIRE_HOURS,
            "rate_limit_enabled": settings.security.RATE_LIMIT_ENABLED,
            "rate_limit_rpm": settings.security.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "login_attempts_max": settings.security.LOGIN_ATTEMPTS_MAX,
            "login_lockout_minutes": settings.security.LOGIN_LOCKOUT_MINUTES,
            "password_min_length": settings.security.PASSWORD_MIN_LENGTH,
        },
        "admin": {
            "pagination_default": settings.admin.ADMIN_PAGINATION_DEFAULT,
            "pagination_max": settings.admin.ADMIN_PAGINATION_MAX,
            "enable_docs": settings.admin.ADMIN_ENABLE_DOCS,
            "ui_theme": settings.admin.UI_THEME,
        },
        "topic_discovery": topic_discovery,
        "features": features,
        "api_keys": {
            "youtube_configured": bool(settings.YOUTUBE_API_KEY),
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
        },
        "logging": logging_settings,
    }


@router.get("/", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user=Depends(get_current_user),
):
    """Render settings page."""
    settings_data = _build_settings_data()
    context = get_template_context(
        request,
        user=user,
        settings_data=settings_data,
    )
    return templates.TemplateResponse("admin/settings/index.html", context)


@router.post("/", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    user=Depends(get_current_user),
    # Feature flags – unchecked checkboxes are absent from the POST body, so
    # default to False and let FastAPI coerce "true" -> True when present.
    registration_enabled: bool = Form(False),
    api_docs_enabled: bool = Form(False),
    metrics_enabled: bool = Form(False),
    health_checks_enabled: bool = Form(False),
    # Topic-discovery limits – None means "field was not submitted", fall back
    # to the current config value.
    default_video_limit: Optional[int] = Form(None),
    default_channel_limit: Optional[int] = Form(None),
    search_limit: Optional[int] = Form(None),
    similar_depth: Optional[int] = Form(None),
    filter_threshold: Optional[float] = Form(None),
    max_concurrent_agents: Optional[int] = Form(None),
    # Logging
    log_level: str = Form("INFO"),
):
    """Handle settings form submission, updating the runtime overrides."""
    td = settings.topic_discovery

    runtime_overrides.update("features", {
        "registration_enabled": registration_enabled,
        "api_docs_enabled": api_docs_enabled,
        "metrics_enabled": metrics_enabled,
        "health_checks_enabled": health_checks_enabled,
    })

    runtime_overrides.update("topic_discovery", {
        "default_video_limit": (
            default_video_limit if default_video_limit is not None
            else td.DEFAULT_VIDEO_LIMIT
        ),
        "default_channel_limit": (
            default_channel_limit if default_channel_limit is not None
            else td.DEFAULT_CHANNEL_LIMIT
        ),
        "search_limit": (
            search_limit if search_limit is not None
            else td.SEARCH_LIMIT
        ),
        "similar_depth": (
            similar_depth if similar_depth is not None
            else td.SIMILAR_DEPTH
        ),
        "filter_threshold": (
            filter_threshold if filter_threshold is not None
            else td.FILTER_THRESHOLD
        ),
        "max_concurrent_agents": (
            max_concurrent_agents if max_concurrent_agents is not None
            else td.MAX_CONCURRENT_AGENTS
        ),
    })

    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    effective_log_level = log_level.upper() if log_level.upper() in valid_log_levels else "INFO"

    runtime_overrides.update("logging", {"level": effective_log_level})

    # Apply the new logging level to the root logger immediately.
    numeric_level = getattr(logging, effective_log_level, logging.INFO)
    logging.getLogger().setLevel(numeric_level)

    settings_data = _build_settings_data()
    context = get_template_context(
        request,
        user=user,
        settings_data=settings_data,
        success_message="Settings saved successfully.",
    )
    return templates.TemplateResponse("admin/settings/index.html", context)
