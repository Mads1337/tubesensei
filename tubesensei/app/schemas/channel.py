from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class ChannelStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    INACTIVE = "inactive"


class ChannelBase(BaseModel):
    """Base channel schema"""
    youtube_channel_id: str = Field(..., description="YouTube channel ID or URL")
    processing_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing configuration")


class ChannelCreate(ChannelBase):
    """Schema for creating a channel"""
    pass


class ChannelUpdate(BaseModel):
    """Schema for updating a channel"""
    status: Optional[ChannelStatus] = None
    priority_level: Optional[int] = Field(None, ge=1, le=10)
    check_frequency_hours: Optional[int] = Field(None, ge=1, le=168)
    auto_process: Optional[bool] = None
    processing_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class ChannelStats(BaseModel):
    """Channel statistics schema"""
    total_videos: int
    processed_videos: int
    processing_videos: int
    failed_videos: int
    total_ideas: int


class ChannelResponse(BaseModel):
    """Channel response schema"""
    id: UUID
    youtube_channel_id: str
    name: str
    channel_handle: Optional[str]
    channel_url: Optional[str]
    description: Optional[str]
    subscriber_count: Optional[int]
    video_count: Optional[int]
    view_count: Optional[int]
    thumbnail_url: Optional[str]
    status: ChannelStatus
    priority_level: int
    check_frequency_hours: int
    last_checked_at: Optional[datetime]
    last_video_published_at: Optional[datetime]
    auto_process: bool
    tags: List[str]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    stats: Optional[ChannelStats] = None
    
    class Config:
        from_attributes = True


class ChannelListResponse(BaseModel):
    """Channel list response schema"""
    items: List[ChannelResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class ChannelSyncResponse(BaseModel):
    """Channel sync response schema"""
    channel_id: str
    updated: bool
    job_id: Optional[str]
    stats: Optional[Dict[str, Any]]