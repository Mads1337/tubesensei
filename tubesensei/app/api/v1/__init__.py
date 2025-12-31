"""
API v1 Router

Combines all v1 API endpoints into a single router.
"""
from fastapi import APIRouter

from app.api.v1.topic_campaigns import router as topic_campaigns_router

# Create main v1 router
router = APIRouter()

# Include sub-routers
router.include_router(topic_campaigns_router)

__all__ = ["router"]
