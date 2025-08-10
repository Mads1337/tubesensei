from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Enum as SQLEnum, Index, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import enum
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.models.base import BaseModel


class VideoStatus(enum.Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    FILTERED_OUT = "filtered_out"


class Video(BaseModel):
    __tablename__ = "videos"
    
    youtube_video_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    
    channel_id = Column(
        UUID(as_uuid=True),
        ForeignKey("channels.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    title = Column(
        String(500),
        nullable=False
    )
    
    description = Column(
        String,
        nullable=True
    )
    
    thumbnail_url = Column(
        String(500),
        nullable=True
    )
    
    duration_seconds = Column(
        Integer,
        nullable=True
    )
    
    view_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    like_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    comment_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    published_at = Column(
        DateTime(timezone=True),
        nullable=False
    )
    
    discovered_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    status = Column(
        SQLEnum(VideoStatus),
        nullable=False,
        default=VideoStatus.DISCOVERED,
        index=True
    )
    
    is_valuable = Column(
        Boolean,
        nullable=True
    )
    
    valuable_score = Column(
        Float,
        nullable=True
    )
    
    valuable_reason = Column(
        String,
        nullable=True
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list
    )
    
    category_id = Column(
        String(50),
        nullable=True
    )
    
    language = Column(
        String(10),
        nullable=True
    )
    
    has_captions = Column(
        Boolean,
        nullable=False,
        default=False
    )
    
    caption_languages = Column(
        ARRAY(String),
        nullable=False,
        default=list
    )
    
    video_metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    processing_metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    error_message = Column(
        String,
        nullable=True
    )
    
    retry_count = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    channel = relationship(
        "Channel",
        back_populates="videos",
        lazy="joined"
    )
    
    transcripts = relationship(
        "Transcript",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    ideas = relationship(
        "Idea",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    __table_args__ = (
        Index("idx_video_channel_status", "channel_id", "status"),
        Index("idx_video_status_discovered", "status", "discovered_at"),
        Index("idx_video_valuable", "is_valuable", "valuable_score"),
        Index("idx_video_published", "published_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Video(id={self.id}, title={self.title[:50]}..., status={self.status.value})>"
    
    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_video_id}"
    
    @property
    def is_processed(self) -> bool:
        return self.status in [VideoStatus.COMPLETED, VideoStatus.FAILED, VideoStatus.SKIPPED]
    
    @property
    def can_retry(self) -> bool:
        return self.status == VideoStatus.FAILED and self.retry_count < 3
    
    @property
    def duration_formatted(self) -> str:
        if not self.duration_seconds:
            return "Unknown"
        
        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        seconds = self.duration_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    
    def mark_as_valuable(self, score: float, reason: str) -> None:
        self.is_valuable = True
        self.valuable_score = score
        self.valuable_reason = reason
    
    def mark_as_not_valuable(self, reason: str) -> None:
        self.is_valuable = False
        self.valuable_score = 0.0
        self.valuable_reason = reason
    
    def update_stats(
        self,
        view_count: Optional[int] = None,
        like_count: Optional[int] = None,
        comment_count: Optional[int] = None
    ) -> None:
        if view_count is not None:
            self.view_count = view_count
        if like_count is not None:
            self.like_count = like_count
        if comment_count is not None:
            self.comment_count = comment_count