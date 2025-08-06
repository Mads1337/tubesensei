from datetime import datetime
from typing import Optional, List, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class VideoType(str, Enum):
    """Types of YouTube videos"""
    REGULAR = "regular"
    SHORT = "short"
    LIVE = "live"
    PREMIERE = "premiere"


class ContentRating(str, Enum):
    """YouTube content rating categories"""
    GENERAL = "general"
    AGE_RESTRICTED = "age_restricted"
    
    
class VideoStatus(str, Enum):
    """Video availability status"""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"
    DELETED = "deleted"


class VideoFilters(BaseModel):
    """
    Comprehensive filtering criteria for YouTube videos.
    Used to filter videos during discovery and processing.
    """
    
    # Duration filters (in seconds)
    min_duration_seconds: Optional[int] = Field(
        default=60,
        ge=0,
        description="Minimum video duration in seconds"
    )
    max_duration_seconds: Optional[int] = Field(
        default=7200,  # 2 hours
        ge=0,
        description="Maximum video duration in seconds"
    )
    
    # View count filters
    min_views: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum view count"
    )
    max_views: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum view count"
    )
    
    # Date filters
    published_after: Optional[datetime] = Field(
        default=None,
        description="Only include videos published after this date"
    )
    published_before: Optional[datetime] = Field(
        default=None,
        description="Only include videos published before this date"
    )
    
    # Video type filters
    include_types: Optional[Set[VideoType]] = Field(
        default={VideoType.REGULAR},
        description="Types of videos to include"
    )
    exclude_shorts: bool = Field(
        default=True,
        description="Exclude YouTube Shorts"
    )
    exclude_live: bool = Field(
        default=False,
        description="Exclude live streams"
    )
    exclude_premieres: bool = Field(
        default=True,
        description="Exclude premieres"
    )
    
    # Language filters
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code (e.g., 'en' for English)"
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="List of acceptable language codes"
    )
    
    # Content filters
    exclude_age_restricted: bool = Field(
        default=True,
        description="Exclude age-restricted content"
    )
    require_captions: bool = Field(
        default=False,
        description="Only include videos with captions"
    )
    caption_languages: Optional[List[str]] = Field(
        default=None,
        description="Required caption languages"
    )
    
    # Engagement filters
    min_likes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of likes"
    )
    min_comments: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of comments"
    )
    min_like_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum like/dislike ratio"
    )
    
    # Title and description filters
    title_contains: Optional[List[str]] = Field(
        default=None,
        description="Video title must contain one of these strings"
    )
    title_excludes: Optional[List[str]] = Field(
        default=None,
        description="Video title must not contain any of these strings"
    )
    description_contains: Optional[List[str]] = Field(
        default=None,
        description="Video description must contain one of these strings"
    )
    description_excludes: Optional[List[str]] = Field(
        default=None,
        description="Video description must not contain any of these strings"
    )
    
    # Tag filters
    required_tags: Optional[List[str]] = Field(
        default=None,
        description="Video must have all of these tags"
    )
    any_tags: Optional[List[str]] = Field(
        default=None,
        description="Video must have at least one of these tags"
    )
    excluded_tags: Optional[List[str]] = Field(
        default=None,
        description="Video must not have any of these tags"
    )
    
    # Category filters
    category_ids: Optional[List[str]] = Field(
        default=None,
        description="YouTube category IDs to include"
    )
    excluded_category_ids: Optional[List[str]] = Field(
        default=None,
        description="YouTube category IDs to exclude"
    )
    
    @field_validator('max_duration_seconds')
    @classmethod
    def validate_duration_range(cls, v, info):
        """Ensure max duration is greater than min duration"""
        min_duration = info.data.get('min_duration_seconds')
        if v is not None and min_duration is not None and v < min_duration:
            raise ValueError('max_duration_seconds must be greater than min_duration_seconds')
        return v
    
    @field_validator('max_views')
    @classmethod
    def validate_view_range(cls, v, info):
        """Ensure max views is greater than min views"""
        min_views = info.data.get('min_views')
        if v is not None and min_views is not None and v < min_views:
            raise ValueError('max_views must be greater than min_views')
        return v
    
    @field_validator('published_before')
    @classmethod
    def validate_date_range(cls, v, info):
        """Ensure published_before is after published_after"""
        published_after = info.data.get('published_after')
        if v is not None and published_after is not None and v < published_after:
            raise ValueError('published_before must be after published_after')
        return v
    
    def is_short_duration(self, duration_seconds: int) -> bool:
        """Check if video duration indicates it's a YouTube Short"""
        return duration_seconds <= 60
    
    def apply_to_video(self, video_data: dict) -> bool:
        """
        Apply filters to a video data dictionary.
        
        Args:
            video_data: Dictionary containing video metadata
            
        Returns:
            True if video passes all filters, False otherwise
        """
        # Duration check
        duration = video_data.get('duration_seconds', 0)
        if self.min_duration_seconds and duration < self.min_duration_seconds:
            return False
        if self.max_duration_seconds and duration > self.max_duration_seconds:
            return False
        
        # Shorts check
        if self.exclude_shorts and self.is_short_duration(duration):
            return False
        
        # Views check
        views = video_data.get('view_count', 0)
        if self.min_views and views < self.min_views:
            return False
        if self.max_views and views > self.max_views:
            return False
        
        # Date check
        published_at = video_data.get('published_at')
        if published_at:
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            if self.published_after and published_at < self.published_after:
                return False
            if self.published_before and published_at > self.published_before:
                return False
        
        # Language check
        video_language = video_data.get('language')
        if self.language and video_language != self.language:
            return False
        if self.languages and video_language not in self.languages:
            return False
        
        # Title filters
        title = video_data.get('title', '').lower()
        if self.title_contains:
            if not any(term.lower() in title for term in self.title_contains):
                return False
        if self.title_excludes:
            if any(term.lower() in title for term in self.title_excludes):
                return False
        
        # All checks passed
        return True


class ChannelFilters(BaseModel):
    """Filtering criteria for YouTube channels"""
    
    min_subscribers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum subscriber count"
    )
    max_subscribers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum subscriber count"
    )
    
    min_video_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of videos"
    )
    max_video_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of videos"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Channel created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None,
        description="Channel created before this date"
    )
    
    country: Optional[str] = Field(
        default=None,
        description="Channel country code"
    )
    
    language: Optional[str] = Field(
        default=None,
        description="Channel primary language"
    )
    
    is_verified: Optional[bool] = Field(
        default=None,
        description="Filter by verification status"
    )
    
    
class ProcessingFilters(BaseModel):
    """Filters for processing job selection"""
    
    priority_min: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Minimum priority level"
    )
    
    status_in: Optional[List[str]] = Field(
        default=None,
        description="Include only these statuses"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Jobs created after this time"
    )
    
    max_retries: Optional[int] = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts allowed"
    )