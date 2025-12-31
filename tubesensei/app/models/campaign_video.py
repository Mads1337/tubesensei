"""
CampaignVideo model - Junction table linking campaigns to discovered videos.

Tracks which videos were discovered by which campaign, how they were found,
and their AI-determined topic relevance.
"""
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Index, Float, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from app.models.base import BaseModel


class DiscoverySource(enum.Enum):
    """How a video was discovered."""
    SEARCH = "search"                       # Found via YouTube search
    CHANNEL_EXPANSION = "channel_expansion" # Found by expanding a channel
    SIMILAR_VIDEOS = "similar_videos"       # Found via related videos API


class CampaignVideo(BaseModel):
    """
    Junction table linking TopicCampaign to Video.

    Tracks discovery metadata and AI filtering results for each video
    discovered during a campaign.
    """
    __tablename__ = "campaign_videos"

    # Foreign keys
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("topic_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Discovery metadata
    discovery_source = Column(
        SQLEnum(DiscoverySource),
        nullable=False,
        index=True,
        comment="How this video was discovered"
    )

    discovered_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="When the video was discovered"
    )

    # Agent tracking
    agent_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
        comment="Which agent run discovered this video"
    )

    # Source tracking for chain of discovery
    source_video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
        comment="For similar_videos: the video that led to this one"
    )

    source_channel_id = Column(
        UUID(as_uuid=True),
        ForeignKey("channels.id", ondelete="SET NULL"),
        nullable=True,
        comment="For channel_expansion: the channel this came from"
    )

    # AI Filter results
    is_topic_relevant = Column(
        Boolean,
        nullable=True,
        index=True,
        comment="AI determination: is this video relevant to the topic?"
    )

    relevance_score = Column(
        Float,
        nullable=True,
        comment="AI confidence score (0.0 to 1.0)"
    )

    filter_reasoning = Column(
        String,
        nullable=True,
        comment="AI explanation for relevance decision"
    )

    matched_keywords = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Keywords from video that matched the topic"
    )

    topic_alignment = Column(
        String(50),
        nullable=True,
        comment="exact, related, tangential, or unrelated"
    )

    # Filter timing
    filtered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the AI filter was applied"
    )

    # Processing status
    transcript_extracted = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether transcript has been extracted"
    )

    transcript_extracted_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    ideas_extracted = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether ideas have been extracted"
    )

    ideas_extracted_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    # Discovery depth (for similar videos recursion)
    discovery_depth = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Recursion depth: 0 for search/channel, 1+ for similar videos"
    )

    # Order within campaign discovery
    discovery_order = Column(
        Integer,
        nullable=True,
        comment="Sequential order in which video was discovered"
    )

    # Metadata
    discovery_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional discovery context"
    )

    # Relationships
    campaign = relationship(
        "TopicCampaign",
        back_populates="videos"
    )

    video = relationship(
        "Video",
        foreign_keys=[video_id],
        backref="campaign_links"
    )

    source_video = relationship(
        "Video",
        foreign_keys=[source_video_id]
    )

    source_channel = relationship(
        "Channel",
        foreign_keys=[source_channel_id]
    )

    agent_run = relationship(
        "AgentRun",
        backref="discovered_videos"
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("campaign_id", "video_id", name="uq_campaign_video"),
        Index("idx_cv_campaign_relevant", "campaign_id", "is_topic_relevant"),
        Index("idx_cv_campaign_source", "campaign_id", "discovery_source"),
        Index("idx_cv_relevance_score", "relevance_score"),
        Index("idx_cv_discovered_at", "discovered_at"),
    )

    def __repr__(self) -> str:
        return f"<CampaignVideo(campaign={self.campaign_id}, video={self.video_id}, relevant={self.is_topic_relevant})>"

    # Properties
    @property
    def is_filtered(self) -> bool:
        """Check if the video has been through AI filtering."""
        return self.is_topic_relevant is not None

    @property
    def is_pending_filter(self) -> bool:
        """Check if video is waiting for AI filtering."""
        return self.is_topic_relevant is None

    @property
    def is_pending_transcript(self) -> bool:
        """Check if video needs transcript extraction."""
        return self.is_topic_relevant is True and not self.transcript_extracted

    @property
    def is_pending_ideas(self) -> bool:
        """Check if video needs idea extraction."""
        return self.transcript_extracted and not self.ideas_extracted

    @property
    def passes_threshold(self) -> bool:
        """Check if relevance score meets threshold (default 0.7)."""
        if self.relevance_score is None:
            return False
        return self.relevance_score >= 0.7

    # Methods
    def mark_relevant(
        self,
        score: float,
        reasoning: str,
        keywords: Optional[list] = None,
        alignment: Optional[str] = None
    ) -> None:
        """Mark video as topic-relevant after AI filtering."""
        self.is_topic_relevant = True
        self.relevance_score = score
        self.filter_reasoning = reasoning
        self.matched_keywords = keywords or []
        self.topic_alignment = alignment
        self.filtered_at = datetime.now(timezone.utc)

    def mark_irrelevant(
        self,
        score: float,
        reasoning: str,
        alignment: Optional[str] = None
    ) -> None:
        """Mark video as not topic-relevant after AI filtering."""
        self.is_topic_relevant = False
        self.relevance_score = score
        self.filter_reasoning = reasoning
        self.topic_alignment = alignment or "unrelated"
        self.filtered_at = datetime.now(timezone.utc)

    def mark_transcript_extracted(self) -> None:
        """Mark that transcript has been extracted."""
        self.transcript_extracted = True
        self.transcript_extracted_at = datetime.now(timezone.utc)

    def mark_ideas_extracted(self) -> None:
        """Mark that ideas have been extracted."""
        self.ideas_extracted = True
        self.ideas_extracted_at = datetime.now(timezone.utc)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for API responses."""
        return {
            "id": str(self.id),
            "campaign_id": str(self.campaign_id),
            "video_id": str(self.video_id),
            "discovery_source": self.discovery_source.value,
            "discovery_depth": self.discovery_depth,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "is_topic_relevant": self.is_topic_relevant,
            "relevance_score": self.relevance_score,
            "filter_reasoning": self.filter_reasoning,
            "topic_alignment": self.topic_alignment,
            "transcript_extracted": self.transcript_extracted,
            "ideas_extracted": self.ideas_extracted,
        }
