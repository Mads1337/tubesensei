"""
Pydantic schemas for Topic Campaign API endpoints.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class CampaignStatus(str, Enum):
    """Campaign status enum for API."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DiscoverySource(str, Enum):
    """Discovery source enum for API."""
    SEARCH = "search"
    CHANNEL_EXPANSION = "channel_expansion"
    SIMILAR_VIDEOS = "similar_videos"


class AgentType(str, Enum):
    """Agent type enum for API."""
    COORDINATOR = "coordinator"
    SEARCH = "search"
    CHANNEL_EXPANSION = "channel_expansion"
    TOPIC_FILTER = "topic_filter"
    SIMILAR_VIDEOS = "similar_videos"


class AgentRunStatus(str, Enum):
    """Agent run status enum for API."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Campaign Configuration

class CampaignConfig(BaseModel):
    """Configuration for a topic discovery campaign."""
    total_video_limit: int = Field(
        default=3000,
        ge=10,
        le=50000,
        description="Maximum total videos to discover"
    )
    per_channel_limit: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum videos from any single channel"
    )
    search_limit: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum results from initial search"
    )
    similar_videos_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Recursion depth for similar videos discovery"
    )
    filter_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for topic filter"
    )
    enabled_agents: List[str] = Field(
        default=["search", "channel_expansion", "topic_filter", "similar_videos"],
        description="List of enabled discovery agents"
    )
    min_duration_seconds: int = Field(
        default=60,
        ge=0,
        le=86400,
        description="Minimum video duration in seconds (default: 60 = 1 minute)"
    )
    max_duration_seconds: int = Field(
        default=7200,
        ge=0,
        le=86400,
        description="Maximum video duration in seconds (default: 7200 = 2 hours)"
    )

    @validator('max_duration_seconds')
    def validate_duration_range(cls, v, values):
        """Ensure max duration is greater than or equal to min duration."""
        min_duration = values.get('min_duration_seconds')
        if min_duration is not None and v < min_duration:
            raise ValueError('max_duration_seconds must be >= min_duration_seconds')
        return v


# Campaign CRUD Schemas

class TopicCampaignCreate(BaseModel):
    """Schema for creating a new campaign."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="User-friendly campaign name"
    )
    topic: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The search topic (e.g., 'how to make money with YouTube shorts')"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional campaign description"
    )
    config: Optional[CampaignConfig] = Field(
        default_factory=CampaignConfig,
        description="Campaign configuration"
    )


class TopicCampaignUpdate(BaseModel):
    """Schema for updating a campaign (draft only)."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    config: Optional[CampaignConfig] = None


# Campaign Response Schemas

class CampaignStats(BaseModel):
    """Campaign statistics."""
    videos_discovered: int
    videos_relevant: int
    videos_filtered: int
    channels_explored: int
    transcripts_extracted: int
    api_calls: int
    llm_calls: int
    error_count: int


class TopicCampaignResponse(BaseModel):
    """Full campaign response schema."""
    id: UUID
    name: str
    topic: str
    description: Optional[str]
    status: CampaignStatus
    config: Dict[str, Any]
    progress_percent: float
    total_videos_discovered: int
    total_videos_relevant: int
    total_videos_filtered: int
    total_channels_explored: int
    total_transcripts_extracted: int
    api_calls_made: int
    llm_calls_made: int
    error_count: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    paused_at: Optional[datetime]
    estimated_completion_at: Optional[datetime]
    execution_time_seconds: Optional[float]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

    class Config:
        from_attributes = True


class TopicCampaignListResponse(BaseModel):
    """Campaign list response."""
    items: List[TopicCampaignResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class TopicCampaignSummary(BaseModel):
    """Lightweight campaign summary for lists."""
    id: UUID
    name: str
    topic: str
    status: CampaignStatus
    progress_percent: float
    videos_relevant: int
    started_at: Optional[datetime]
    created_at: datetime


# Progress Schemas

class AgentStats(BaseModel):
    """Statistics for a single agent type."""
    total_runs: int
    items_processed: int
    items_produced: int
    api_calls: int


class VideoStats(BaseModel):
    """Statistics by discovery source."""
    total: int
    relevant: int


class CampaignProgress(BaseModel):
    """Real-time campaign progress."""
    campaign_id: UUID
    status: CampaignStatus
    progress_percent: float
    videos_discovered: int
    videos_relevant: int
    videos_filtered: int
    channels_explored: int
    transcripts_extracted: int
    api_calls: int
    llm_calls: int
    error_count: int
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    duration_seconds: Optional[float]
    filter_acceptance_rate: float
    agent_stats: Dict[str, AgentStats]
    video_stats: Dict[str, VideoStats]


# Video Response Schemas

class VideoSummary(BaseModel):
    """Video summary for campaign results."""
    id: UUID
    youtube_id: str
    title: str
    description: str
    view_count: int
    duration_seconds: int
    published_at: Optional[datetime]
    has_captions: bool


class CampaignVideoResponse(BaseModel):
    """Campaign video junction response."""
    id: UUID
    campaign_id: UUID
    video_id: UUID
    discovery_source: DiscoverySource
    discovery_depth: int
    discovered_at: datetime
    is_topic_relevant: Optional[bool]
    relevance_score: Optional[float]
    filter_reasoning: Optional[str]
    topic_alignment: Optional[str]
    transcript_extracted: bool
    ideas_extracted: bool
    video: VideoSummary


class CampaignVideoListResponse(BaseModel):
    """Campaign videos list response."""
    items: List[CampaignVideoResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# Channel Response Schemas

class ChannelSummary(BaseModel):
    """Channel summary for campaign results."""
    id: UUID
    youtube_id: str
    name: str
    subscriber_count: Optional[int]
    video_count: Optional[int]


class CampaignChannelResponse(BaseModel):
    """Campaign channel junction response."""
    id: UUID
    campaign_id: UUID
    channel_id: UUID
    discovery_source: DiscoverySource
    discovered_at: datetime
    was_expanded: bool
    expanded_at: Optional[datetime]
    videos_discovered: int
    videos_relevant: int
    videos_filtered_out: int
    videos_limit: int
    limit_reached: bool
    relevance_rate: float
    channel: ChannelSummary


class CampaignChannelListResponse(BaseModel):
    """Campaign channels list response."""
    items: List[CampaignChannelResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# Agent Run Response Schemas

class AgentRunResponse(BaseModel):
    """Agent run response schema."""
    id: UUID
    campaign_id: UUID
    agent_type: AgentType
    status: AgentRunStatus
    progress_percent: float
    current_item: Optional[str]
    items_processed: int
    items_produced: int
    api_calls_made: int
    llm_calls_made: int
    tokens_used: Optional[int]
    estimated_cost_usd: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    has_errors: bool
    rate_limited: bool


class AgentRunListResponse(BaseModel):
    """Agent runs list response."""
    items: List[AgentRunResponse]
    total: int


# Action Response Schemas

class CampaignActionResponse(BaseModel):
    """Response for campaign actions (start, pause, resume, cancel)."""
    campaign_id: UUID
    action: str
    success: bool
    status: CampaignStatus
    message: str
    task_id: Optional[str] = None  # Celery task ID if applicable


class ExportResponse(BaseModel):
    """Export response schema."""
    campaign_id: UUID
    format: str
    video_count: int
    file_size_bytes: int
    download_url: Optional[str] = None
