"""Admin Settings API router module."""

import logging
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from app.core.auth import get_current_user
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/settings", tags=["admin-settings"])

# Module-level runtime overrides (persists for current session)
_runtime_overrides: dict = {}


def _build_settings_data() -> dict:
    """Build settings data dict, applying runtime overrides."""
    features = {
        "registration_enabled": settings.FEATURES_ENABLE_REGISTRATION,
        "api_docs_enabled": settings.FEATURES_ENABLE_API_DOCS,
        "metrics_enabled": settings.FEATURES_ENABLE_METRICS,
        "health_checks_enabled": settings.FEATURES_ENABLE_HEALTH_CHECKS,
    }
    topic_discovery = {
        "default_video_limit": settings.topic_discovery.DEFAULT_VIDEO_LIMIT,
        "default_channel_limit": settings.topic_discovery.DEFAULT_CHANNEL_LIMIT,
        "search_limit": settings.topic_discovery.SEARCH_LIMIT,
        "similar_depth": settings.topic_discovery.SIMILAR_DEPTH,
        "filter_threshold": settings.topic_discovery.FILTER_THRESHOLD,
        "filter_batch_size": settings.topic_discovery.FILTER_BATCH_SIZE,
        "filter_model": settings.topic_discovery.FILTER_MODEL,
        "rate_limit": settings.topic_discovery.RATE_LIMIT,
        "max_concurrent_agents": settings.topic_discovery.MAX_CONCURRENT_AGENTS,
    }
    logging_settings = {
        "level": settings.LOG_LEVEL,
        "format": settings.LOG_FORMAT,
        "rotation": settings.LOG_ROTATION,
        "retention": settings.LOG_RETENTION,
    }

    # Apply overrides
    if "features" in _runtime_overrides:
        features.update(_runtime_overrides["features"])
    if "topic_discovery" in _runtime_overrides:
        topic_discovery.update(_runtime_overrides["topic_discovery"])
    if "logging" in _runtime_overrides:
        logging_settings.update(_runtime_overrides["logging"])

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
            "pool_size": settings.DATABASE_POOL_SIZE,
            "pool_max_overflow": settings.DATABASE_POOL_MAX_OVERFLOW,
            "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
            "echo": settings.DATABASE_ECHO,
        },
        "redis": {
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
    user = Depends(get_current_user),
):
    """Render settings page - view of system configuration"""

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
    user = Depends(get_current_user),
    # Feature flags - unchecked checkboxes submit nothing, so default to False
    registration_enabled: bool = Form(False),
    api_docs_enabled: bool = Form(False),
    metrics_enabled: bool = Form(False),
    health_checks_enabled: bool = Form(False),
    # Topic discovery limits - default to None so empty/missing fields fall back to config
    default_video_limit: int | None = Form(None),
    default_channel_limit: int | None = Form(None),
    search_limit: int | None = Form(None),
    similar_depth: int | None = Form(None),
    filter_threshold: float | None = Form(None),
    max_concurrent_agents: int | None = Form(None),
    # Logging
    log_level: str = Form("INFO"),
):
    """Save settings - updates runtime configuration"""
    global _runtime_overrides

    # Fall back to current settings values when a field is missing/empty
    _runtime_overrides = {
        "features": {
            "registration_enabled": registration_enabled,
            "api_docs_enabled": api_docs_enabled,
            "metrics_enabled": metrics_enabled,
            "health_checks_enabled": health_checks_enabled,
        },
        "topic_discovery": {
            "default_video_limit": default_video_limit if default_video_limit is not None else settings.topic_discovery.DEFAULT_VIDEO_LIMIT,
            "default_channel_limit": default_channel_limit if default_channel_limit is not None else settings.topic_discovery.DEFAULT_CHANNEL_LIMIT,
            "search_limit": search_limit if search_limit is not None else settings.topic_discovery.SEARCH_LIMIT,
            "similar_depth": similar_depth if similar_depth is not None else settings.topic_discovery.SIMILAR_DEPTH,
            "filter_threshold": filter_threshold if filter_threshold is not None else settings.topic_discovery.FILTER_THRESHOLD,
            "max_concurrent_agents": max_concurrent_agents if max_concurrent_agents is not None else settings.topic_discovery.MAX_CONCURRENT_AGENTS,
        },
        "logging": {
            "level": log_level,
        },
    }

    # Apply logging level immediately
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)

    # Re-render settings page with success message
    settings_data = _build_settings_data()
    context = get_template_context(
        request,
        user=user,
        settings_data=settings_data,
        success_message="Settings saved successfully.",
    )
    return templates.TemplateResponse("admin/settings/index.html", context)
