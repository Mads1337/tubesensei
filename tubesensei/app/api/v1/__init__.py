"""
API v1 Router

Combines all v1 API endpoints into a single router.
"""
from fastapi import APIRouter

from app.api.v1.topic_campaigns import router as topic_campaigns_router
from app.api.v1.ideas import router as ideas_router
from app.api.v1.videos import router as videos_router
from app.api.v1.channels import router as channels_router
from app.api.v1.export import router as export_router
from app.api.v1.webhooks import router as webhooks_router

# Create main v1 router
router = APIRouter()

# Include sub-routers
router.include_router(topic_campaigns_router)
router.include_router(ideas_router)
router.include_router(videos_router)
router.include_router(channels_router)
router.include_router(export_router)
router.include_router(webhooks_router)

__all__ = ["router"]
