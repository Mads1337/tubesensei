from sqlalchemy import Column, String, Boolean, Integer, DateTime, ForeignKey, Enum as SQLEnum, Index, Float, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import enum
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.models.base import BaseModel


class IdeaStatus(enum.Enum):
    EXTRACTED = "extracted"
    REVIEWED = "reviewed"
    SELECTED = "selected"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"


class IdeaPriority(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Idea(BaseModel):
    __tablename__ = "ideas"
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    title = Column(
        String(500),
        nullable=False
    )
    
    description = Column(
        Text,
        nullable=False
    )
    
    category = Column(
        String(100),
        nullable=True,
        index=True
    )
    
    status = Column(
        SQLEnum(IdeaStatus),
        nullable=False,
        default=IdeaStatus.EXTRACTED,
        index=True
    )
    
    priority = Column(
        SQLEnum(IdeaPriority),
        nullable=False,
        default=IdeaPriority.MEDIUM,
        index=True
    )
    
    confidence_score = Column(
        Float,
        nullable=False,
        default=0.0
    )
    
    complexity_score = Column(
        Integer,
        nullable=True
    )
    
    market_size_estimate = Column(
        String(50),
        nullable=True
    )
    
    target_audience = Column(
        String(200),
        nullable=True
    )
    
    implementation_time_estimate = Column(
        String(50),
        nullable=True
    )
    
    source_timestamp = Column(
        Integer,
        nullable=True
    )
    
    source_context = Column(
        Text,
        nullable=True
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list
    )
    
    technologies = Column(
        ARRAY(String),
        nullable=False,
        default=list
    )
    
    competitive_advantage = Column(
        Text,
        nullable=True
    )
    
    potential_challenges = Column(
        JSONB,
        nullable=False,
        default=list
    )
    
    monetization_strategies = Column(
        JSONB,
        nullable=False,
        default=list
    )
    
    related_ideas = Column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list
    )
    
    extraction_metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    review_notes = Column(
        Text,
        nullable=True
    )
    
    reviewed_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    reviewed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    selected_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    selected_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    export_count = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    last_exported_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    video = relationship(
        "Video",
        back_populates="ideas",
        lazy="joined"
    )
    
    reviewer = relationship(
        "User",
        foreign_keys=[reviewed_by],
        lazy="joined"
    )
    
    selector = relationship(
        "User",
        foreign_keys=[selected_by],
        lazy="joined"
    )
    
    __table_args__ = (
        Index("idx_idea_status_confidence", "status", "confidence_score"),
        Index("idx_idea_video_status", "video_id", "status"),
        Index("idx_idea_category_status", "category", "status"),
        Index("idx_idea_priority_status", "priority", "status"),
        Index("idx_idea_reviewed", "reviewed_at", "reviewed_by"),
    )
    
    def __repr__(self) -> str:
        return f"<Idea(id={self.id}, title={self.title[:50]}..., status={self.status.value})>"
    
    @property
    def is_selected(self) -> bool:
        return self.status == IdeaStatus.SELECTED
    
    @property
    def is_reviewed(self) -> bool:
        return self.status in [IdeaStatus.REVIEWED, IdeaStatus.SELECTED, IdeaStatus.REJECTED]
    
    @property
    def confidence_percentage(self) -> float:
        return self.confidence_score * 100
    
    def mark_as_reviewed(self, user_id: UUID, notes: Optional[str] = None) -> None:
        self.status = IdeaStatus.REVIEWED
        self.reviewed_by = user_id
        self.reviewed_at = datetime.utcnow()
        if notes:
            self.review_notes = notes
    
    def select(self, user_id: UUID) -> None:
        self.status = IdeaStatus.SELECTED
        self.selected_by = user_id
        self.selected_at = datetime.utcnow()
        if not self.reviewed_by:
            self.reviewed_by = user_id
            self.reviewed_at = datetime.utcnow()
    
    def reject(self, user_id: UUID, reason: Optional[str] = None) -> None:
        self.status = IdeaStatus.REJECTED
        if not self.reviewed_by:
            self.reviewed_by = user_id
            self.reviewed_at = datetime.utcnow()
        if reason:
            self.review_notes = reason
    
    def mark_as_exported(self) -> None:
        self.export_count += 1
        self.last_exported_at = datetime.utcnow()