from sqlalchemy import Column, String, Boolean, Integer, DateTime, Enum as SQLEnum, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.models.base import BaseModel


class ChannelStatus(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    INACTIVE = "inactive"


class Channel(BaseModel):
    __tablename__ = "channels"
    
    youtube_channel_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    
    channel_name = Column(
        String(255),
        nullable=False
    )
    
    channel_handle = Column(
        String(255),
        nullable=True
    )
    
    description = Column(
        String,
        nullable=True
    )
    
    subscriber_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    video_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    view_count = Column(
        Integer,
        nullable=True,
        default=0
    )
    
    country = Column(
        String(10),
        nullable=True
    )
    
    custom_url = Column(
        String(500),
        nullable=True
    )
    
    published_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    thumbnail_url = Column(
        String(500),
        nullable=True
    )
    
    status = Column(
        SQLEnum(ChannelStatus),
        nullable=False,
        default=ChannelStatus.ACTIVE,
        index=True
    )
    
    priority_level = Column(
        Integer,
        nullable=False,
        default=5
    )
    
    check_frequency_hours = Column(
        Integer,
        nullable=False,
        default=24
    )
    
    last_checked_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    last_video_published_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    channel_metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    processing_config = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    auto_process = Column(
        Boolean,
        nullable=False,
        default=True
    )
    
    tags = Column(
        JSONB,
        nullable=False,
        default=list
    )
    
    notes = Column(
        String,
        nullable=True
    )
    
    videos = relationship(
        "Video",
        back_populates="channel",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    __table_args__ = (
        Index("idx_channel_status_priority", "status", "priority_level"),
        Index("idx_channel_last_checked", "last_checked_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Channel(id={self.id}, name={self.channel_name}, handle={self.channel_handle})>"
    
    @property
    def is_active(self) -> bool:
        return self.status == ChannelStatus.ACTIVE
    
    @property
    def needs_check(self) -> bool:
        if not self.is_active:
            return False
        if not self.last_checked_at:
            return True
        
        from datetime import timedelta
        hours_since_check = (datetime.now() - self.last_checked_at).total_seconds() / 3600
        return hours_since_check >= self.check_frequency_hours
    
    def update_stats(
        self,
        subscriber_count: Optional[int] = None,
        video_count: Optional[int] = None,
        view_count: Optional[int] = None
    ) -> None:
        if subscriber_count is not None:
            self.subscriber_count = subscriber_count
        if video_count is not None:
            self.video_count = video_count
        if view_count is not None:
            self.view_count = view_count
        self.last_checked_at = datetime.now()