from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class VideoResponse(BaseModel):
    id: UUID
    youtube_video_id: str
    channel_id: UUID
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    duration_seconds: Optional[int]
    view_count: Optional[int]
    like_count: Optional[int]
    published_at: datetime
    status: str
    is_valuable: Optional[bool]
    valuable_score: Optional[float]
    has_captions: bool
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    items: List[VideoResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
