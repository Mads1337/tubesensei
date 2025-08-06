from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Index, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any
from datetime import datetime

from app.models.base import BaseModel


class SessionType(enum.Enum):
    BULK_PROCESSING = "bulk_processing"
    CHANNEL_SYNC = "channel_sync"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    RECOVERY = "recovery"


class SessionStatus(enum.Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingSession(BaseModel):
    __tablename__ = "processing_sessions"
    
    session_type = Column(
        SQLEnum(SessionType),
        nullable=False,
        index=True
    )
    
    status = Column(
        SQLEnum(SessionStatus),
        nullable=False,
        default=SessionStatus.INITIALIZED,
        index=True
    )
    
    name = Column(
        String(255),
        nullable=False
    )
    
    description = Column(
        String,
        nullable=True
    )
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    total_jobs = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    completed_jobs = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    failed_jobs = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    cancelled_jobs = Column(
        Integer,
        nullable=False,
        default=0
    )
    
    progress_percent = Column(
        Float,
        nullable=False,
        default=0.0
    )
    
    estimated_completion_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    configuration = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    statistics = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    error_message = Column(
        String,
        nullable=True
    )
    
    is_resumable = Column(
        Boolean,
        nullable=False,
        default=True
    )
    
    checkpoint_data = Column(
        JSONB,
        nullable=True
    )
    
    last_checkpoint_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    execution_time_seconds = Column(
        Float,
        nullable=True
    )
    
    created_by = Column(
        String(100),
        nullable=True
    )
    
    jobs = relationship(
        "ProcessingJob",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    __table_args__ = (
        Index("idx_session_type_status", "session_type", "status"),
        Index("idx_session_started", "started_at"),
        Index("idx_session_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<ProcessingSession(id={self.id}, name={self.name}, status={self.status.value})>"
    
    @property
    def is_active(self) -> bool:
        return self.status in [SessionStatus.RUNNING, SessionStatus.PAUSED]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs / self.total_jobs) * 100
    
    @property
    def failure_rate(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return (self.failed_jobs / self.total_jobs) * 100
    
    def start(self) -> None:
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def pause(self) -> None:
        if self.status != SessionStatus.RUNNING:
            raise ValueError("Can only pause running sessions")
        self.status = SessionStatus.PAUSED
        self.save_checkpoint()
    
    def resume(self) -> None:
        if self.status != SessionStatus.PAUSED:
            raise ValueError("Can only resume paused sessions")
        self.status = SessionStatus.RUNNING
    
    def complete(self) -> None:
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percent = 100.0
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error_message: str) -> None:
        self.status = SessionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def cancel(self) -> None:
        self.status = SessionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self) -> None:
        if self.total_jobs > 0:
            processed = self.completed_jobs + self.failed_jobs + self.cancelled_jobs
            self.progress_percent = (processed / self.total_jobs) * 100
        else:
            self.progress_percent = 0.0
    
    def increment_job_count(self, status: str) -> None:
        if status == "completed":
            self.completed_jobs += 1
        elif status == "failed":
            self.failed_jobs += 1
        elif status == "cancelled":
            self.cancelled_jobs += 1
        self.update_progress()
    
    def save_checkpoint(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.last_checkpoint_at = datetime.utcnow()
        if data:
            self.checkpoint_data = data
    
    def update_statistics(self, stats: Dict[str, Any]) -> None:
        if not self.statistics:
            self.statistics = {}
        self.statistics.update(stats)