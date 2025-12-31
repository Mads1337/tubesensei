"""
CampaignChannel model - Junction table linking campaigns to explored channels.

Tracks which channels were explored during a campaign, how many videos
were discovered from each, and enforces per-channel video limits.
"""
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Index, Float, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any
from datetime import datetime

from app.models.base import BaseModel
from app.models.campaign_video import DiscoverySource


class CampaignChannel(BaseModel):
    """
    Junction table linking TopicCampaign to Channel.

    Tracks how channels were discovered and expanded during a campaign,
    including statistics on videos discovered and per-channel limits.
    """
    __tablename__ = "campaign_channels"

    # Foreign keys
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("topic_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    channel_id = Column(
        UUID(as_uuid=True),
        ForeignKey("channels.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Discovery metadata
    discovery_source = Column(
        SQLEnum(DiscoverySource),
        nullable=False,
        comment="How this channel was discovered"
    )

    discovered_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="When the channel was first encountered"
    )

    # What led us to this channel
    source_video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
        comment="The video that led to discovering this channel"
    )

    # Expansion status
    was_expanded = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether we've fetched all videos from this channel"
    )

    expanded_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When channel expansion was completed"
    )

    # Statistics
    videos_discovered = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total videos discovered from this channel"
    )

    videos_relevant = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Videos that passed topic filter"
    )

    videos_filtered_out = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Videos that failed topic filter"
    )

    videos_pending_filter = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Videos awaiting AI filtering"
    )

    # Per-channel limit for this campaign
    videos_limit = Column(
        Integer,
        nullable=False,
        default=5,
        comment="Maximum videos to include from this channel"
    )

    # Whether we've hit the limit
    limit_reached = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether videos_limit has been reached"
    )

    # Priority for expansion
    expansion_priority = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Priority for channel expansion (higher = sooner)"
    )

    # Channel quality indicators (cached from main channel)
    channel_subscriber_count = Column(
        Integer,
        nullable=True,
        comment="Cached subscriber count at discovery time"
    )

    channel_video_count = Column(
        Integer,
        nullable=True,
        comment="Cached total video count at discovery time"
    )

    # Metadata
    expansion_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional expansion context and statistics"
    )

    # Error tracking
    expansion_error = Column(
        String,
        nullable=True,
        comment="Error message if expansion failed"
    )

    expansion_retries = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of expansion retry attempts"
    )

    # Relationships
    campaign = relationship(
        "TopicCampaign",
        back_populates="channels"
    )

    channel = relationship(
        "Channel",
        backref="campaign_links"
    )

    source_video = relationship(
        "Video",
        foreign_keys=[source_video_id]
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("campaign_id", "channel_id", name="uq_campaign_channel"),
        Index("idx_cc_campaign_expanded", "campaign_id", "was_expanded"),
        Index("idx_cc_campaign_priority", "campaign_id", "expansion_priority"),
        Index("idx_cc_discovered_at", "discovered_at"),
    )

    def __repr__(self) -> str:
        return f"<CampaignChannel(campaign={self.campaign_id}, channel={self.channel_id}, expanded={self.was_expanded})>"

    # Properties
    @property
    def is_pending_expansion(self) -> bool:
        """Check if channel needs to be expanded."""
        return not self.was_expanded and not self.limit_reached

    @property
    def can_add_more_videos(self) -> bool:
        """Check if more videos can be added from this channel."""
        return self.videos_relevant < self.videos_limit

    @property
    def remaining_video_slots(self) -> int:
        """How many more videos can be added from this channel."""
        return max(0, self.videos_limit - self.videos_relevant)

    @property
    def relevance_rate(self) -> float:
        """Calculate what percentage of videos passed the filter."""
        total = self.videos_relevant + self.videos_filtered_out
        if total > 0:
            return (self.videos_relevant / total) * 100
        return 0.0

    # Methods
    def mark_expanded(self) -> None:
        """Mark channel as fully expanded."""
        self.was_expanded = True
        self.expanded_at = datetime.utcnow()

    def mark_limit_reached(self) -> None:
        """Mark that the per-channel limit has been reached."""
        self.limit_reached = True

    def increment_discovered(self, count: int = 1) -> None:
        """Increment discovered videos count."""
        self.videos_discovered += count

    def increment_relevant(self, count: int = 1) -> None:
        """Increment relevant videos count and check limit."""
        self.videos_relevant += count
        if self.videos_relevant >= self.videos_limit:
            self.mark_limit_reached()

    def increment_filtered_out(self, count: int = 1) -> None:
        """Increment filtered out videos count."""
        self.videos_filtered_out += count

    def update_pending_filter(self, count: int) -> None:
        """Update pending filter count."""
        self.videos_pending_filter = count

    def record_error(self, error: str) -> None:
        """Record expansion error."""
        self.expansion_error = error
        self.expansion_retries += 1

    def clear_error(self) -> None:
        """Clear expansion error."""
        self.expansion_error = None

    def cache_channel_stats(self, subscriber_count: Optional[int], video_count: Optional[int]) -> None:
        """Cache channel statistics at discovery time."""
        self.channel_subscriber_count = subscriber_count
        self.channel_video_count = video_count

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for API responses."""
        return {
            "id": str(self.id),
            "campaign_id": str(self.campaign_id),
            "channel_id": str(self.channel_id),
            "discovery_source": self.discovery_source.value,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "was_expanded": self.was_expanded,
            "expanded_at": self.expanded_at.isoformat() if self.expanded_at else None,
            "videos_discovered": self.videos_discovered,
            "videos_relevant": self.videos_relevant,
            "videos_filtered_out": self.videos_filtered_out,
            "videos_limit": self.videos_limit,
            "limit_reached": self.limit_reached,
            "relevance_rate": self.relevance_rate,
            "channel_subscriber_count": self.channel_subscriber_count,
        }
