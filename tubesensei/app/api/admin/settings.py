"""Admin Settings API router module."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.core.auth import get_current_user
from app.core.config import settings
from fastapi.templating import Jinja2Templates
from .template_helpers import get_template_context

from pathlib import Path
template_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter(prefix="/settings", tags=["admin-settings"])


@router.get("/", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user = Depends(get_current_user),
):
    """Render settings page - read only view of system configuration"""

    # Gather all settings for display
    settings_data = {
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
        "topic_discovery": {
            "default_video_limit": settings.topic_discovery.DEFAULT_VIDEO_LIMIT,
            "default_channel_limit": settings.topic_discovery.DEFAULT_CHANNEL_LIMIT,
            "search_limit": settings.topic_discovery.SEARCH_LIMIT,
            "similar_depth": settings.topic_discovery.SIMILAR_DEPTH,
            "filter_threshold": settings.topic_discovery.FILTER_THRESHOLD,
            "filter_batch_size": settings.topic_discovery.FILTER_BATCH_SIZE,
            "filter_model": settings.topic_discovery.FILTER_MODEL,
            "rate_limit": settings.topic_discovery.RATE_LIMIT,
            "max_concurrent_agents": settings.topic_discovery.MAX_CONCURRENT_AGENTS,
        },
        "features": {
            "registration_enabled": settings.FEATURES_ENABLE_REGISTRATION,
            "api_docs_enabled": settings.FEATURES_ENABLE_API_DOCS,
            "metrics_enabled": settings.FEATURES_ENABLE_METRICS,
            "health_checks_enabled": settings.FEATURES_ENABLE_HEALTH_CHECKS,
        },
        "api_keys": {
            "youtube_configured": bool(settings.YOUTUBE_API_KEY),
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
        },
        "logging": {
            "level": settings.LOG_LEVEL,
            "format": settings.LOG_FORMAT,
            "rotation": settings.LOG_ROTATION,
            "retention": settings.LOG_RETENTION,
        },
    }

    context = get_template_context(
        request,
        user=user,
        settings_data=settings_data,
    )

    return templates.TemplateResponse("admin/settings/index.html", context)
