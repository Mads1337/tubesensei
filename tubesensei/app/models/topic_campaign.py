"""
TopicCampaign model for topic-based video discovery campaigns.

A campaign represents a user-initiated topic search that discovers videos
across YouTube using multiple discovery agents (search, channel expansion,
similar videos) and filters them by AI for topic relevance.
"""
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Index, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from app.models.base import BaseModel


class CampaignStatus(enum.Enum):
    """Status of a topic discovery campaign."""
    DRAFT = "draft"           # Created but not started
    RUNNING = "running"       # Actively discovering videos
    PAUSED = "paused"         # Temporarily paused by user
    COMPLETED = "completed"   # Reached limits or exhausted discovery
    FAILED = "failed"         # Failed due to errors
    CANCELLED = "cancelled"   # Cancelled by user


class TopicCampaign(BaseModel):
    """
    Represents a topic-based video discovery campaign.

    A campaign crawls YouTube to find videos related to a specific topic,
    expands to related channels, filters videos by AI relevance, and
    discovers similar videos - all with configurable limits.

    Example topic: "how to make money creating YouTube shorts with AI"
    """
    __tablename__ = "topic_campaigns"

    # Core identification
    name = Column(
        String(255),
        nullable=False,
        comment="User-friendly campaign name"
    )

    topic = Column(
        String(1000),
        nullable=False,
        index=True,
        comment="The search topic/query for video discovery"
    )

    description = Column(
        String,
        nullable=True,
        comment="Optional campaign description"
    )

    # Status
    status = Column(
        SQLEnum(CampaignStatus),
        nullable=False,
        default=CampaignStatus.DRAFT,
        index=True
    )

    # Configuration (JSONB for flexibility)
    config = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="""Campaign configuration:
        {
            "total_video_limit": 3000,
            "per_channel_limit": 5,
            "search_limit": 50,
            "similar_videos_depth": 2,
            "filter_threshold": 0.7,
            "enabled_agents": ["search", "channel_expansion", "topic_filter", "similar_videos"]
        }"""
    )

    # Progress tracking
    total_videos_discovered = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total videos found across all sources"
    )

    total_videos_relevant = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Videos that passed AI topic filter"
    )

    total_videos_filtered = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Videos that were filtered out"
    )

    total_channels_explored = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Channels that were expanded for videos"
    )

    total_transcripts_extracted = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Transcripts successfully extracted"
    )

    # Timing
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the campaign started running"
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the campaign finished"
    )

    paused_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the campaign was last paused"
    )

    estimated_completion_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Estimated completion time based on current rate"
    )

    # Progress percentage
    progress_percent = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Overall progress percentage (0-100)"
    )

    # Error tracking
    error_message = Column(
        String,
        nullable=True,
        comment="Last error message if failed"
    )

    error_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of errors encountered"
    )

    # API usage tracking
    api_calls_made = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total YouTube API calls made"
    )

    llm_calls_made = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total LLM API calls made for filtering"
    )

    # Checkpoint for resume capability
    checkpoint_data = Column(
        JSONB,
        nullable=True,
        comment="State data for resuming paused campaigns"
    )

    last_checkpoint_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    # Execution time tracking
    execution_time_seconds = Column(
        Float,
        nullable=True,
        comment="Total execution time in seconds"
    )

    # Campaign metadata
    campaign_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional metadata storage"
    )

    # Statistics
    statistics = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Campaign statistics and metrics"
    )

    # Creator tracking
    created_by = Column(
        String(100),
        nullable=True,
        comment="User who created this campaign"
    )

    # Celery task tracking
    celery_task_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="ID of the currently running Celery task"
    )

    # Relationships
    videos = relationship(
        "CampaignVideo",
        back_populates="campaign",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    channels = relationship(
        "CampaignChannel",
        back_populates="campaign",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    agent_runs = relationship(
        "AgentRun",
        back_populates="campaign",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="AgentRun.started_at.desc()"
    )

    # Indexes
    __table_args__ = (
        Index("idx_campaign_status", "status"),
        Index("idx_campaign_topic", "topic"),
        Index("idx_campaign_started", "started_at"),
        Index("idx_campaign_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<TopicCampaign(id={self.id}, name={self.name}, status={self.status.value})>"

    # Configuration accessors
    @property
    def total_video_limit(self) -> int:
        """Maximum videos to discover for this campaign."""
        return self.config.get("total_video_limit", 3000)

    @property
    def per_channel_limit(self) -> int:
        """Maximum videos from any single channel."""
        return self.config.get("per_channel_limit", 5)

    @property
    def search_limit(self) -> int:
        """Maximum results from initial search."""
        return self.config.get("search_limit", 50)

    @property
    def similar_videos_depth(self) -> int:
        """Recursion depth for similar videos discovery."""
        return self.config.get("similar_videos_depth", 2)

    @property
    def filter_threshold(self) -> float:
        """Minimum relevance score for topic filter."""
        return self.config.get("filter_threshold", 0.7)

    @property
    def enabled_agents(self) -> List[str]:
        """List of enabled discovery agents."""
        return self.config.get("enabled_agents", [
            "search", "channel_expansion", "topic_filter", "similar_videos"
        ])

    # Status checks
    @property
    def is_active(self) -> bool:
        """Check if campaign is currently running or paused."""
        return self.status in [CampaignStatus.RUNNING, CampaignStatus.PAUSED]

    @property
    def is_complete(self) -> bool:
        """Check if campaign has finished (any terminal state)."""
        return self.status in [
            CampaignStatus.COMPLETED,
            CampaignStatus.FAILED,
            CampaignStatus.CANCELLED
        ]

    @property
    def can_start(self) -> bool:
        """Check if campaign can be started."""
        return self.status == CampaignStatus.DRAFT

    @property
    def can_pause(self) -> bool:
        """Check if campaign can be paused."""
        return self.status == CampaignStatus.RUNNING

    @property
    def can_resume(self) -> bool:
        """Check if campaign can be resumed."""
        return self.status == CampaignStatus.PAUSED

    @property
    def can_cancel(self) -> bool:
        """Check if campaign can be cancelled."""
        return self.status in [CampaignStatus.RUNNING, CampaignStatus.PAUSED]

    @property
    def has_reached_limit(self) -> bool:
        """Check if total video limit has been reached."""
        return self.total_videos_relevant >= self.total_video_limit

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate campaign duration in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.now(timezone.utc)
            return (end_time - self.started_at).total_seconds()
        return None

    @property
    def videos_per_minute(self) -> Optional[float]:
        """Calculate discovery rate."""
        duration = self.duration_seconds
        if duration and duration > 0 and self.total_videos_discovered > 0:
            return (self.total_videos_discovered / duration) * 60
        return None

    @property
    def filter_acceptance_rate(self) -> float:
        """Calculate what percentage of videos pass the filter."""
        total_filtered = self.total_videos_relevant + self.total_videos_filtered
        if total_filtered > 0:
            return (self.total_videos_relevant / total_filtered) * 100
        return 0.0

    # State transitions
    def start(self) -> None:
        """Start the campaign."""
        if not self.can_start:
            raise ValueError(f"Cannot start campaign in {self.status.value} status")
        self.status = CampaignStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def pause(self) -> None:
        """Pause the campaign."""
        if not self.can_pause:
            raise ValueError(f"Cannot pause campaign in {self.status.value} status")
        self.status = CampaignStatus.PAUSED
        self.paused_at = datetime.now(timezone.utc)

    def resume(self) -> None:
        """Resume the campaign."""
        if not self.can_resume:
            raise ValueError(f"Cannot resume campaign in {self.status.value} status")
        self.status = CampaignStatus.RUNNING
        self.paused_at = None

    def complete(self) -> None:
        """Mark campaign as completed."""
        self.status = CampaignStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percent = 100.0
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def fail(self, error_message: str) -> None:
        """Mark campaign as failed."""
        self.status = CampaignStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def cancel(self) -> None:
        """Cancel the campaign."""
        if not self.can_cancel:
            raise ValueError(f"Cannot cancel campaign in {self.status.value} status")
        self.status = CampaignStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    # Progress updates
    def update_progress(self) -> None:
        """Recalculate progress percentage based on limits."""
        if self.total_video_limit > 0:
            self.progress_percent = min(
                (self.total_videos_relevant / self.total_video_limit) * 100,
                100.0
            )

    def calculate_eta(self) -> Optional[datetime]:
        """Calculate estimated completion time based on current rate."""
        if not self.started_at or self.total_videos_relevant == 0:
            return None

        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        if elapsed <= 0:
            return None

        rate = self.total_videos_relevant / elapsed  # videos per second
        if rate <= 0:
            return None

        remaining = self.total_video_limit - self.total_videos_relevant
        if remaining <= 0:
            return None

        seconds_remaining = remaining / rate

        return datetime.now(timezone.utc) + timedelta(seconds=seconds_remaining)

    def update_eta(self) -> None:
        """Update the estimated_completion_at field."""
        self.estimated_completion_at = self.calculate_eta()

    def increment_discovered(self, count: int = 1) -> None:
        """Increment discovered videos count."""
        self.total_videos_discovered += count

    def increment_relevant(self, count: int = 1) -> None:
        """Increment relevant videos count and update progress."""
        self.total_videos_relevant += count
        self.update_progress()
        self.update_eta()

    def increment_filtered(self, count: int = 1) -> None:
        """Increment filtered out videos count."""
        self.total_videos_filtered += count

    def increment_channels(self, count: int = 1) -> None:
        """Increment explored channels count."""
        self.total_channels_explored += count

    def increment_transcripts(self, count: int = 1) -> None:
        """Increment extracted transcripts count."""
        self.total_transcripts_extracted += count

    def increment_api_calls(self, count: int = 1) -> None:
        """Increment API calls count."""
        self.api_calls_made += count

    def increment_llm_calls(self, count: int = 1) -> None:
        """Increment LLM calls count."""
        self.llm_calls_made += count

    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.error_message = error

    # Checkpoint management
    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        """Save checkpoint data for resume capability."""
        self.checkpoint_data = data
        self.last_checkpoint_at = datetime.now(timezone.utc)

    def clear_checkpoint(self) -> None:
        """Clear checkpoint data."""
        self.checkpoint_data = None
        self.last_checkpoint_at = None

    # Statistics
    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Update campaign statistics."""
        if not self.statistics:
            self.statistics = {}
        self.statistics.update(stats)

    def get_summary(self) -> Dict[str, Any]:
        """Get campaign summary for API responses."""
        # Calculate current ETA if running
        eta = self.estimated_completion_at or self.calculate_eta()
        return {
            "id": str(self.id),
            "name": self.name,
            "topic": self.topic,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "videos_discovered": self.total_videos_discovered,
            "videos_relevant": self.total_videos_relevant,
            "videos_filtered": self.total_videos_filtered,
            "channels_explored": self.total_channels_explored,
            "transcripts_extracted": self.total_transcripts_extracted,
            "filter_acceptance_rate": self.filter_acceptance_rate,
            "api_calls": self.api_calls_made,
            "llm_calls": self.llm_calls_made,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_completion_at": eta.isoformat() if eta else None,
            "error_count": self.error_count,
            "config": self.config,
        }
